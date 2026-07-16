from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def intrinsics_dict_to_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def point_cloud_from_depth(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_clip_range: tuple[float, float] = (0.15, 2.0),
    max_points: int = 25000,
    rng_seed: int = 42,
) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    if depth.ndim != 2:
        raise ValueError(f"depth_image must be 2D, got {depth.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {intrinsics.shape}")

    height, width = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    px, py = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    z = depth.reshape(-1)
    near, far = depth_clip_range
    valid = np.isfinite(z) & (z > 0.0) & (z >= near) & (z <= far)

    x = (px.reshape(-1) - cx) * z / fx
    y = (py.reshape(-1) - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)[valid]

    if points.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
    return np.ascontiguousarray(points, dtype=np.float64)


def transform_points(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if transform is None:
        return pts
    tf = np.asarray(transform, dtype=np.float64)
    if tf.shape != (4, 4):
        raise ValueError(f"transform must be (4, 4), got {tf.shape}")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    return np.ascontiguousarray((tf @ pts_h.T).T[:, :3], dtype=np.float64)


@lru_cache(maxsize=16)
def _get_robot_segmenter(
    robot_cfg_path: str,
    urdf_path: str,
    device: str,
    distance_threshold: float,
    collision_sphere_buffer: float,
):
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.wrap.model.robot_segmenter import RobotSegmenter
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

    cfg_path = Path(robot_cfg_path)
    urdf = Path(urdf_path)
    robot_cfg = yaml.safe_load(cfg_path.read_text())["robot_cfg"]
    kin = robot_cfg["kinematics"]
    kin["use_usd_kinematics"] = False
    kin["usd_path"] = ""
    kin["isaac_usd_path"] = ""
    kin["urdf_path"] = str(urdf)
    kin["asset_root_path"] = str(urdf.parent)
    kin["collision_sphere_buffer"] = float(collision_sphere_buffer)
    tensor_args = TensorDeviceType(device=torch.device(device))
    robot = RobotConfig.from_dict(robot_cfg, tensor_args=tensor_args)
    rw_cfg = RobotWorldConfig.load_from_config(
        robot,
        None,
        tensor_args=tensor_args,
        collision_activation_distance=0.0,
    )
    segmenter = RobotSegmenter(
        RobotWorld(rw_cfg),
        distance_threshold=float(distance_threshold),
        use_cuda_graph=True,
        ops_dtype=torch.float16,
        depth_to_meter=0.001,
    )
    joint_names = tuple(kin["cspace"]["joint_names"])
    return segmenter, joint_names, tensor_args


def filter_depth_with_robot_mask(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    joint_position: np.ndarray,
    *,
    robot_cfg_path: str,
    urdf_path: str,
    device: str = "cuda:0",
    distance_threshold: float = 0.08,
    collision_sphere_buffer: float = 0.015,
    mask_dilation_pixels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from curobo.types.camera import CameraObservation
    from curobo.types.math import Pose
    from curobo.types.state import JointState

    segmenter, joint_names, tensor_args = _get_robot_segmenter(
        robot_cfg_path,
        urdf_path,
        device,
        float(distance_threshold),
        float(collision_sphere_buffer),
    )
    depth = np.asarray(depth_image, dtype=np.float32)
    intr = np.asarray(intrinsics, dtype=np.float32)
    pose = np.asarray(camera_pose, dtype=np.float32)
    q = np.asarray(joint_position, dtype=np.float32).reshape(1, -1)

    depth_mm = tensor_args.to_device(torch.from_numpy(depth).float()) * 1000.0
    if depth_mm.ndim == 2:
        depth_mm = depth_mm.unsqueeze(0)
    intr_t = tensor_args.to_device(torch.from_numpy(intr))
    pose_t = tensor_args.to_device(torch.from_numpy(pose))
    cam_obs = CameraObservation(
        depth_image=depth_mm,
        intrinsics=intr_t,
        pose=Pose.from_matrix(pose_t),
    )
    q_js = JointState(
        position=tensor_args.to_device(torch.from_numpy(q)),
        joint_names=list(joint_names),
    )
    if not segmenter.ready:
        segmenter.update_camera_projection(cam_obs)
    mask_t, filtered_t = segmenter.get_robot_mask_from_active_js(
        cam_obs,
        segmenter.robot_world.get_active_js(q_js),
    )
    mask = mask_t[0].detach().cpu().numpy()
    if int(mask_dilation_pixels) > 0:
        import cv2

        k = 2 * int(mask_dilation_pixels) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        filtered_depth = depth.copy()
        filtered_depth[mask] = 0.0
        return filtered_depth.astype(np.float32), mask
    return (
        filtered_t[0].detach().cpu().numpy().astype(np.float32) * 0.001,
        mask,
    )


def get_robot_spheres_world(
    joint_position: np.ndarray,
    *,
    robot_cfg_path: str,
    urdf_path: str,
    device: str = "cuda:0",
    collision_sphere_buffer: float = 0.015,
) -> np.ndarray:
    import torch
    from curobo.types.state import JointState

    segmenter, joint_names, tensor_args = _get_robot_segmenter(
        robot_cfg_path,
        urdf_path,
        device,
        0.08,
        float(collision_sphere_buffer),
    )
    q = np.asarray(joint_position, dtype=np.float32).reshape(1, -1)
    q_t = tensor_args.to_device(torch.from_numpy(q))
    js = segmenter.robot_world.get_active_js(
        JointState(
            position=q_t,
            joint_names=list(joint_names),
        )
    )
    spheres = segmenter.robot_world.get_kinematics(js.position).link_spheres_tensor
    return spheres.view(-1, 4).detach().cpu().numpy().astype(np.float32)


def filter_points_near_robot(
    points_world: np.ndarray,
    joint_position: np.ndarray,
    *,
    robot_cfg_path: str,
    urdf_path: str,
    device: str = "cuda:0",
    collision_sphere_buffer: float = 0.015,
    point_margin: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return points, np.zeros((0,), dtype=bool)
    spheres = get_robot_spheres_world(
        joint_position,
        robot_cfg_path=robot_cfg_path,
        urdf_path=urdf_path,
        device=device,
        collision_sphere_buffer=collision_sphere_buffer,
    )
    centers = spheres[:, :3]
    radii = spheres[:, 3] + float(point_margin)
    d = points[:, None, :] - centers[None, :, :]
    dist2 = np.sum(d * d, axis=-1)
    keep = np.all(dist2 > (radii[None, :] * radii[None, :]), axis=1)
    return points[keep], keep


def create_world_config_from_points(
    points_world: np.ndarray,
    *,
    marching_cubes_pitch: float = 0.04,
    scene_name: str = "zed_depth_scene",
):
    from curobo.geom.types import Mesh, WorldConfig

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    meshes = []
    if points.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                points,
                pitch=marching_cubes_pitch,
                name=scene_name,
            )
        )
    return WorldConfig(mesh=meshes)


def create_world_config_from_depth(
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    *,
    camera_pose: np.ndarray | None = None,
    depth_clip_range: tuple[float, float] = (0.15, 2.0),
    max_points: int = 25000,
    marching_cubes_pitch: float = 0.04,
    scene_name: str = "zed_depth_scene",
):
    from curobo.geom.types import Mesh, WorldConfig

    points_cam = point_cloud_from_depth(
        depth_image,
        intrinsics,
        depth_clip_range=depth_clip_range,
        max_points=max_points,
    )
    points_world = transform_points(points_cam, camera_pose)
    meshes = []
    if points_world.shape[0] > 0:
        meshes.append(
            Mesh.from_pointcloud(
                points_world,
                pitch=marching_cubes_pitch,
                name=scene_name,
            )
        )
    return WorldConfig(mesh=meshes)
