# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuRobo motion planner for the YAM bimanual station.

This module mirrors the public API of ``experimental.motion_planner.YamMotionPlanner``
well enough to drop into the scripted Move-To flow. It uses cuRobo's
``MotionGen`` as the primary planner and optionally validates the generated
trajectory against the existing MuJoCo collision checker for parity with the
current planner.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import os
from pathlib import Path
import sys
from typing import Any, Literal
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from experimental.curobo_depth_world import (
    create_world_config_from_points,
    filter_depth_with_robot_mask,
    point_cloud_from_depth,
    transform_points,
)
from robot.yam.kinematics import YamKinematics


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBOT_CFG = (
    _REPO_ROOT
    / "robot"
    / "models"
    / "station"
    / "curobo"
    / "yam_dual_isaacsim_physics_fixed_fingers.yml"
)
_SEGMENTER_ROBOT_CFG = (
    _REPO_ROOT
    / "robot"
    / "models"
    / "station"
    / "curobo"
    / "yam_dual_isaacsim_physics_fixed_fingers.yml"
)
_URDF_PATH = (
    _REPO_ROOT / "robot" / "models" / "station" / "station_physics_fixed_fingers.urdf"
)
_ASSET_ROOT = _URDF_PATH.parent
_WORLD_OBSTACLE_LINKS = ("gate_collision", "play_table")
_INTERPOLATION_DT = 1.0 / 30.0
_DEFAULT_POSITION_THRESHOLD_M = 0.005
_DEFAULT_ROTATION_THRESHOLD_RAD = 0.05
_DEFAULT_CSPACE_THRESHOLD_RAD = 0.05
_USE_CUDA_GRAPH_BY_DEFAULT = True
_ENABLE_GRAPH_SEARCH_BY_DEFAULT = True
_CUROBO_TORCH_COMPILE_DISABLE_DEFAULT = "1"
_DEFAULT_BATCH_PLANNER_CAPACITY = 16
_DEFAULT_SOLVER_SPEED = "fast"
_SOLVER_PRESET_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "slow": {
        "motion_gen": {
            "num_ik_seeds": 32,
            "num_graph_seeds": 12,
            "num_trajopt_seeds": 12,
            "trajopt_tsteps": 48,
            "ik_opt_iters": 256,
            "grad_trajopt_iters": 256,
        },
        "plan": {
            "enable_graph_attempt": 4,
            "max_attempts": 10,
            "timeout": 10.0,
            "time_dilation_factor_single": 0.5,
            "time_dilation_factor_batch": None,
        },
    },
    "fast": {
        "motion_gen": {
            "num_ik_seeds": 8,
            "num_graph_seeds": 1,
            "num_trajopt_seeds": 2,
            "trajopt_tsteps": 32,
            "ik_opt_iters": 96,
            "grad_trajopt_iters": 96,
        },
        "plan": {
            "enable_graph_attempt": 1,
            "max_attempts": 2,
            "timeout": 2.5,
            "time_dilation_factor_single": 0.5,
            "time_dilation_factor_batch": None,
        },
    },
}

warnings.filterwarnings(
    "ignore",
    message=r"The symbol `warp\.torch\.device_from_torch` will soon be removed from the public API\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Logical operators 'and' and 'or' are deprecated for non-scalar tensors; please use '&' or '\|' instead",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Enable tracemalloc to get the object allocation traceback",
    category=UserWarning,
)


def _ensure_local_curobo_on_syspath() -> None:
    candidate = _REPO_ROOT / "third_party" / "curobo" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _silence_curobo_import_logs() -> None:
    logging.getLogger("curobo").setLevel(logging.ERROR)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1).astype(
        np.float64
    )


def _pose_vec(position: np.ndarray, quat_wxyz: np.ndarray) -> list[float]:
    return [
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    ]


def _parse_vec(text: str | None, fallback: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _curobo_timing_info(result: Any) -> dict[str, Any]:
    return {
        "curobo_solve_time_ms": 1000.0 * float(getattr(result, "solve_time", 0.0)),
        "curobo_total_time_ms": 1000.0 * float(getattr(result, "total_time", 0.0)),
        "curobo_ik_time_ms": 1000.0 * float(getattr(result, "ik_time", 0.0)),
        "curobo_graph_time_ms": 1000.0 * float(getattr(result, "graph_time", 0.0)),
        "curobo_trajopt_time_ms": 1000.0 * float(getattr(result, "trajopt_time", 0.0)),
        "curobo_finetune_time_ms": 1000.0
        * float(getattr(result, "finetune_time", 0.0)),
        "curobo_attempts": int(getattr(result, "attempts", 0)),
        "curobo_trajopt_attempts": int(getattr(result, "trajopt_attempts", 0)),
        "curobo_used_graph": bool(getattr(result, "used_graph", False)),
    }


def _env_flag_enabled(name: str, *, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    try:
        return bool(1 - int(value))
    except Exception:
        return str(value).strip().lower() not in {"1", "true", "yes", "on"}


def _normalize_solver_speed(value: str | None) -> str:
    solver_speed = str(value or _DEFAULT_SOLVER_SPEED).strip().lower().replace("_", "-")
    if solver_speed in _SOLVER_PRESET_CONFIGS:
        return solver_speed
    warnings.warn(
        f"Unknown cuRobo solver_speed {value!r}; falling back to {_DEFAULT_SOLVER_SPEED!r}",
        RuntimeWarning,
        stacklevel=2,
    )
    return _DEFAULT_SOLVER_SPEED


def _as_numpy_metric_array(metric: Any, batch_size: int) -> np.ndarray:
    if metric is None:
        return np.full((batch_size,), np.nan, dtype=np.float64)
    if hasattr(metric, "detach"):
        metric = metric.detach().cpu().numpy()
    metric = np.asarray(metric, dtype=np.float64).reshape(-1)
    if metric.shape[0] == batch_size:
        return metric
    if metric.shape[0] == 1:
        return np.full((batch_size,), float(metric[0]), dtype=np.float64)
    raise ValueError(
        f"Unexpected cuRobo metric shape {metric.shape}; expected batch={batch_size}"
    )


class YamMotionPlannerCurobo:
    """cuRobo-backed motion planner with the same return format as YamMotionPlanner."""

    ROBOT_TYPE = "yam"

    def __init__(
        self,
        robot_cfg_path: str | Path | None = None,
        urdf_path: str | Path | None = None,
        validate_with_mujoco: bool = True,
        device: str = "cuda:0",
        collision_checking: bool = True,
        enable_finetune_trajopt: bool = False,
        enable_depth_collision: bool = False,
        solver_speed: str = _DEFAULT_SOLVER_SPEED,
        position_threshold: float = _DEFAULT_POSITION_THRESHOLD_M,
        rotation_threshold: float = _DEFAULT_ROTATION_THRESHOLD_RAD,
        cspace_threshold: float = _DEFAULT_CSPACE_THRESHOLD_RAD,
    ) -> None:
        self._robot_cfg_path = Path(robot_cfg_path or _ROBOT_CFG).resolve()
        self._urdf_path = Path(urdf_path or _URDF_PATH).resolve()
        self._asset_root = self._urdf_path.parent
        self._device = device
        self._collision_checking = bool(collision_checking)
        self._left_gripper = 1.0
        self._right_gripper = 1.0
        self._enable_finetune_trajopt = bool(enable_finetune_trajopt)
        self._enable_depth_collision = bool(enable_depth_collision)
        self._solver_speed = _normalize_solver_speed(solver_speed)
        self._solver_preset = deepcopy(_SOLVER_PRESET_CONFIGS[self._solver_speed])
        self._position_threshold = float(position_threshold)
        self._rotation_threshold = float(rotation_threshold)
        self._cspace_threshold = float(cspace_threshold)

        self._imports = self._import_curobo_dependencies()
        self._torch = self._imports["torch"]
        if not self._torch.cuda.is_available():
            raise RuntimeError(
                "cuRobo planner requires CUDA, but torch.cuda.is_available() is False"
            )

        self._TensorDeviceType = self._imports["TensorDeviceType"]
        self._Pose = self._imports["Pose"]
        self._JointState = self._imports["JointState"]
        self._WorldConfig = self._imports["WorldConfig"]
        self._Cuboid = self._imports["Cuboid"]
        self._Sphere = self._imports["Sphere"]
        self._CollisionCheckerType = self._imports["CollisionCheckerType"]
        self._MotionGen = self._imports["MotionGen"]
        self._MotionGenConfig = self._imports["MotionGenConfig"]
        self._MotionGenPlanConfig = self._imports["MotionGenPlanConfig"]
        self._MotionGenStatus = self._imports["MotionGenStatus"]

        self._tensor_args = self._TensorDeviceType(
            device=self._torch.device(self._device)
        )
        self._kin = YamKinematics()
        self._joint_names: list[str] | None = None
        self._joint_limit_lower: np.ndarray | None = None
        self._joint_limit_upper: np.ndarray | None = None
        self._static_world_cfg = None
        self._depth_world_cfg = None
        self._debug_world_cfg = None
        self._depth_scene_version = 0
        self._batch_planner_capacity = int(_DEFAULT_BATCH_PLANNER_CAPACITY)
        self._motion_gen = None
        self._ee_pose_plan_config = None
        self._setup_motion_gen()
        print(
            "[cuRoboPlanner] initialized "
            f"device={self._device} "
            f"solver_speed={self._solver_speed} "
            f"torch_compile={'on' if _env_flag_enabled('CUROBO_TORCH_COMPILE_DISABLE', default=_CUROBO_TORCH_COMPILE_DISABLE_DEFAULT) else 'off'} "
            f"cuda_graph={'on' if _USE_CUDA_GRAPH_BY_DEFAULT else 'off'} "
            f"graph_search={'on' if _ENABLE_GRAPH_SEARCH_BY_DEFAULT else 'off'} "
            f"batch_capacity={self._batch_planner_capacity} "
            f"finetune={'on' if self._enable_finetune_trajopt else 'off'} "
            f"depth_collision={'on' if self._enable_depth_collision else 'off'}"
        )

        self._validator = None
        if validate_with_mujoco and self._collision_checking:
            try:
                from experimental.motion_planner import (
                    YamMotionPlanner as MujocoPlanner,
                )

                self._validator = MujocoPlanner()
            except Exception as exc:  # pragma: no cover - best effort only
                print(
                    f"[cuRoboPlanner] MuJoCo validator unavailable, continuing without it: {exc}"
                )

    @staticmethod
    def _import_curobo_dependencies() -> dict[str, Any]:
        os.environ.setdefault(
            "CUROBO_TORCH_COMPILE_DISABLE",
            _CUROBO_TORCH_COMPILE_DISABLE_DEFAULT,
        )
        _silence_curobo_import_logs()
        _ensure_local_curobo_on_syspath()
        try:
            import torch
            from curobo.geom.sdf.world import CollisionCheckerType
            from curobo.geom.types import Cuboid, Sphere, WorldConfig
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.state import JointState
            from curobo.util.logger import setup_curobo_logger
            from curobo.wrap.reacher.motion_gen import (
                MotionGen,
                MotionGenConfig,
                MotionGenPlanConfig,
                MotionGenStatus,
            )

            setup_curobo_logger("error")
            return {
                "torch": torch,
                "CollisionCheckerType": CollisionCheckerType,
                "Cuboid": Cuboid,
                "Sphere": Sphere,
                "WorldConfig": WorldConfig,
                "TensorDeviceType": TensorDeviceType,
                "Pose": Pose,
                "JointState": JointState,
                "MotionGen": MotionGen,
                "MotionGenConfig": MotionGenConfig,
                "MotionGenPlanConfig": MotionGenPlanConfig,
                "MotionGenStatus": MotionGenStatus,
            }
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "Failed to import cuRobo. Install it into the same Python env used to run lecar "
                "or keep third_party/curobo available in this repo before selecting the cuRobo planner."
            ) from exc

    def _load_robot_cfg(self) -> dict[str, Any]:
        robot_cfg = deepcopy(
            yaml.safe_load(self._robot_cfg_path.read_text())["robot_cfg"]
        )
        kin = robot_cfg["kinematics"]
        kin["use_usd_kinematics"] = False
        kin["usd_path"] = ""
        kin["isaac_usd_path"] = ""
        kin["urdf_path"] = str(self._urdf_path)
        kin["asset_root_path"] = str(self._asset_root)
        self._joint_names = list(kin["cspace"]["joint_names"])
        return robot_cfg

    def _build_world_cfg(self):
        root = ET.parse(self._urdf_path).getroot()
        fixed_joint_to_child: dict[str, tuple[str, np.ndarray]] = {}
        for joint in root.findall("joint"):
            if joint.attrib.get("type") != "fixed":
                continue
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            if parent_el is None or child_el is None:
                continue
            origin = joint.find("origin")
            xyz = _parse_vec(
                origin.attrib.get("xyz") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_vec(
                origin.attrib.get("rpy") if origin is not None else None,
                (0.0, 0.0, 0.0),
            )
            fixed_joint_to_child[child_el.attrib["link"]] = (
                parent_el.attrib["link"],
                _transform_from_xyz_rpy(xyz, rpy),
            )

        world_links = set(_WORLD_OBSTACLE_LINKS)
        link_T_world: dict[str, np.ndarray] = {"base_link": np.eye(4, dtype=np.float64)}

        def _link_transform(link_name: str) -> np.ndarray:
            if link_name in link_T_world:
                return link_T_world[link_name]
            parent, parent_T_child = fixed_joint_to_child[link_name]
            link_T_world[link_name] = _link_transform(parent) @ parent_T_child
            return link_T_world[link_name]

        cuboids = []
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            if link_name not in world_links:
                continue
            T_world_link = _link_transform(link_name)
            for idx, collision in enumerate(link.findall("collision")):
                geometry = collision.find("geometry")
                if geometry is None:
                    continue
                box = geometry.find("box")
                if box is None:
                    continue
                origin = collision.find("origin")
                xyz = _parse_vec(
                    origin.attrib.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                )
                rpy = _parse_vec(
                    origin.attrib.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                )
                T_world_collision = T_world_link @ _transform_from_xyz_rpy(xyz, rpy)
                quat_xyzw = Rotation.from_matrix(T_world_collision[:3, :3]).as_quat()
                quat_wxyz = np.array(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                    dtype=np.float64,
                )
                dims = [float(v) for v in box.attrib["size"].split()]
                cuboids.append(
                    self._Cuboid(
                        name=f"{link_name}_{idx}",
                        pose=_pose_vec(T_world_collision[:3, 3], quat_wxyz),
                        dims=dims,
                    )
                )
        return self._WorldConfig(cuboid=cuboids)

    def _build_motion_gen(
        self, *, robot_cfg: dict[str, Any], world_cfg, warmup_batch: int | None
    ):
        motion_gen_preset = self._solver_preset["motion_gen"]
        if self._collision_checking:
            static_mesh_count = len(world_cfg.mesh) if world_cfg.mesh is not None else 0
            collision_kwargs = {
                "collision_checker_type": self._CollisionCheckerType.MESH,
                "self_collision_check": False,
                "self_collision_opt": False,
                "collision_cache": {"mesh": max(static_mesh_count + 256, 260)},
                "collision_activation_distance": 0.01,
            }
        else:
            collision_kwargs = {
                "collision_checker_type": None,
                "self_collision_check": False,
                "self_collision_opt": False,
                "collision_activation_distance": None,
            }
        motion_gen_cfg = self._MotionGenConfig.load_from_robot_config(
            deepcopy(robot_cfg),
            world_cfg,
            self._tensor_args,
            **collision_kwargs,
            use_cuda_graph=_USE_CUDA_GRAPH_BY_DEFAULT,
            interpolation_dt=_INTERPOLATION_DT,
            num_ik_seeds=int(motion_gen_preset["num_ik_seeds"]),
            num_graph_seeds=int(motion_gen_preset["num_graph_seeds"]),
            num_trajopt_seeds=int(motion_gen_preset["num_trajopt_seeds"]),
            position_threshold=self._position_threshold,
            rotation_threshold=self._rotation_threshold,
            cspace_threshold=self._cspace_threshold,
            trajopt_tsteps=int(motion_gen_preset["trajopt_tsteps"]),
            maximum_trajectory_dt=0.5,
            fixed_iters_trajopt=True,
            ik_opt_iters=int(motion_gen_preset["ik_opt_iters"]),
            grad_trajopt_iters=int(motion_gen_preset["grad_trajopt_iters"]),
        )
        motion_gen = self._MotionGen(motion_gen_cfg)
        warmup_kwargs = {
            "enable_graph": _ENABLE_GRAPH_SEARCH_BY_DEFAULT,
            "warmup_js_trajopt": False,
        }
        if warmup_batch is not None:
            warmup_kwargs["batch"] = int(warmup_batch)
        motion_gen.warmup(**warmup_kwargs)
        return motion_gen

    def _build_plan_config(self, *, batch_mode: bool = False):
        plan_preset = self._solver_preset["plan"]
        return self._MotionGenPlanConfig(
            enable_graph=_ENABLE_GRAPH_SEARCH_BY_DEFAULT,
            enable_graph_attempt=int(plan_preset["enable_graph_attempt"]),
            enable_finetune_trajopt=self._enable_finetune_trajopt,
            max_attempts=int(plan_preset["max_attempts"]),
            timeout=float(plan_preset["timeout"]),
            time_dilation_factor=(
                plan_preset["time_dilation_factor_batch"]
                if batch_mode
                else plan_preset["time_dilation_factor_single"]
            ),
            use_start_state_as_retract=True,
        )

    def _all_motion_gens(self) -> list[Any]:
        return [self._motion_gen] if self._motion_gen is not None else []

    def _setup_motion_gen(self) -> None:
        robot_cfg = self._load_robot_cfg()
        if self._collision_checking:
            world_cfg = self._build_world_cfg().get_mesh_world()
            self._static_world_cfg = world_cfg.clone()
        else:
            world_cfg = None
            self._static_world_cfg = None
        self._motion_gen = self._build_motion_gen(
            robot_cfg=robot_cfg,
            world_cfg=world_cfg.clone() if world_cfg is not None else None,
            warmup_batch=self._batch_planner_capacity,
        )
        joint_limits = (
            self._motion_gen.kinematics.kinematics_config.joint_limits.position.detach()
            .cpu()
            .numpy()
        )
        self._joint_limit_lower = np.asarray(joint_limits[0], dtype=np.float64)
        self._joint_limit_upper = np.asarray(joint_limits[1], dtype=np.float64)
        self._ee_pose_plan_config = self._build_plan_config(batch_mode=True)

    def set_finetune_enabled(self, enabled: bool) -> None:
        self._enable_finetune_trajopt = bool(enabled)
        if self._ee_pose_plan_config is not None:
            self._ee_pose_plan_config.enable_finetune_trajopt = (
                self._enable_finetune_trajopt
            )

    @property
    def finetune_enabled(self) -> bool:
        return bool(self._enable_finetune_trajopt)

    @property
    def depth_collision_enabled(self) -> bool:
        return bool(self._enable_depth_collision)

    def _compose_world_cfg(self):
        if self._static_world_cfg is None:
            raise RuntimeError("Static world config is not initialized")
        world = self._static_world_cfg.clone()
        if self._depth_world_cfg is not None:
            for obstacle in self._depth_world_cfg.mesh:
                world.add_obstacle(obstacle)
        if self._debug_world_cfg is not None and self._debug_world_cfg.mesh is not None:
            for obstacle in self._debug_world_cfg.mesh:
                world.add_obstacle(obstacle)
        return world

    def _refresh_world(self) -> None:
        motion_gens = self._all_motion_gens()
        if not motion_gens:
            raise RuntimeError("Motion generator is not initialized")
        world_cfg = self._compose_world_cfg()
        for motion_gen in motion_gens:
            motion_gen.update_world(world_cfg.clone())

    def set_depth_collision_scene(
        self,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
        *,
        camera_pose: np.ndarray | None = None,
        joint_position: np.ndarray | None = None,
        segment_robot: bool = True,
        segment_distance_threshold: float = 0.08,
        segment_collision_sphere_buffer: float = 0.015,
        segment_mask_dilation_pixels: int = 2,
        depth_clip_range: tuple[float, float] = (0.15, 2.0),
        max_points: int = 25000,
        marching_cubes_pitch: float = 0.04,
        scene_name: str = "zed_depth_scene",
    ) -> bool:
        if not self._enable_depth_collision:
            return False
        depth_for_world = np.asarray(depth_image, dtype=np.float32)
        if segment_robot and joint_position is not None and camera_pose is not None:
            try:
                depth_for_world, _ = filter_depth_with_robot_mask(
                    depth_for_world,
                    np.asarray(intrinsics, dtype=np.float64),
                    np.asarray(camera_pose, dtype=np.float64),
                    np.asarray(joint_position, dtype=np.float32),
                    robot_cfg_path=str(_SEGMENTER_ROBOT_CFG),
                    urdf_path=str(self._urdf_path),
                    device=self._device,
                    distance_threshold=float(segment_distance_threshold),
                    collision_sphere_buffer=float(segment_collision_sphere_buffer),
                    mask_dilation_pixels=int(segment_mask_dilation_pixels),
                )
            except Exception as exc:
                print(
                    f"[cuRoboPlanner] Robot segmentation failed, using raw depth: {exc}"
                )
        points_cam = point_cloud_from_depth(
            depth_for_world,
            np.asarray(intrinsics, dtype=np.float64),
            depth_clip_range=depth_clip_range,
            max_points=max_points,
        )
        points_world = transform_points(points_cam, camera_pose)
        self._depth_scene_version += 1
        versioned_scene_name = f"{scene_name}_{self._depth_scene_version}"
        self._depth_world_cfg = create_world_config_from_points(
            points_world,
            marching_cubes_pitch=marching_cubes_pitch,
            scene_name=versioned_scene_name,
        )
        mesh_count = (
            0
            if self._depth_world_cfg is None or self._depth_world_cfg.mesh is None
            else len(self._depth_world_cfg.mesh)
        )
        print(
            f"[cuRoboPlanner] depth collision scene updated: points={points_world.shape[0]} meshes={mesh_count} scene={versioned_scene_name}"
        )
        self._refresh_world()
        return True

    def clear_depth_collision_scene(self) -> bool:
        if self._depth_world_cfg is None:
            return False
        self._depth_world_cfg = None
        self._refresh_world()
        return True

    def set_debug_collision_ball(
        self,
        position: np.ndarray,
        radius: float,
        *,
        scene_name: str = "debug_collision_ball",
    ) -> bool:
        center = np.asarray(position, dtype=np.float64).reshape(3)
        r = float(radius)
        if r <= 0.0:
            return self.clear_debug_collision_ball()
        sphere = self._Sphere(
            name=scene_name,
            pose=[
                float(center[0]),
                float(center[1]),
                float(center[2]),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            radius=r,
        )
        self._debug_world_cfg = self._WorldConfig(sphere=[sphere]).get_mesh_world()
        self._refresh_world()
        return True

    def clear_debug_collision_ball(self) -> bool:
        if self._debug_world_cfg is None:
            return False
        self._debug_world_cfg = None
        self._refresh_world()
        return True

    def _clip_joint_positions(
        self,
        left_jp: np.ndarray,
        right_jp: np.ndarray,
        *,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._joint_limit_lower is not None
        assert self._joint_limit_upper is not None
        q = np.concatenate(
            [
                np.asarray(left_jp, dtype=np.float64).reshape(6),
                np.asarray(right_jp, dtype=np.float64).reshape(6),
            ]
        )
        q_clipped = np.clip(q, self._joint_limit_lower, self._joint_limit_upper)
        max_delta = float(np.max(np.abs(q_clipped - q)))
        if max_delta > 1e-6:
            print(
                f"[cuRoboPlanner] Clipped {label} to joint limits (max Δ={max_delta:.4f} rad)"
            )
        return q_clipped[:6], q_clipped[6:12]

    def _make_joint_state(self, left_jp: np.ndarray, right_jp: np.ndarray):
        assert self._joint_names is not None
        left = np.asarray(left_jp, dtype=np.float32).reshape(-1, 6)
        right = np.asarray(right_jp, dtype=np.float32).reshape(-1, 6)
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"Mismatched batch sizes for left/right joint states: {left.shape[0]} vs {right.shape[0]}"
            )
        q = np.concatenate(
            [left, right],
            axis=1,
        )
        return self._JointState.from_position(
            self._tensor_args.to_device(q),
            joint_names=self._joint_names,
        )

    def _joint_state_positions(self, plan, motion_gen=None) -> np.ndarray:
        motion_gen = motion_gen or self._motion_gen
        if motion_gen is None:
            raise RuntimeError("Motion generator is not initialized")
        plan = motion_gen.get_full_js(plan)
        if plan.joint_names is not None and self._joint_names is not None:
            plan = plan.get_ordered_joint_state(self._joint_names)
        positions = plan.position.detach().cpu().numpy()
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions[0]
        return np.asarray(positions, dtype=np.float64)

    def _result_positions(self, result, motion_gen=None) -> np.ndarray:
        return self._joint_state_positions(result.get_interpolated_plan(), motion_gen)

    def _make_pose(self, pos: np.ndarray, quat_xyzw: np.ndarray):
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        quat_xyzw = quat_xyzw / np.maximum(
            np.linalg.norm(quat_xyzw, axis=-1, keepdims=True),
            1e-12,
        )
        return self._Pose(
            position=self._tensor_args.to_device(np.asarray(pos, dtype=np.float32)),
            quaternion=self._tensor_args.to_device(
                _xyzw_to_wxyz(quat_xyzw).astype(np.float32)
            ),
        )

    @staticmethod
    def _infer_pose_batch_size(**targets: np.ndarray | None) -> int:
        batch_size: int | None = None
        for name, value in targets.items():
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0:
                raise ValueError(f"{name} must not be scalar")
            current_size = (
                1 if arr.ndim == 1 else int(arr.reshape(-1, arr.shape[-1]).shape[0])
            )
            if batch_size is None:
                batch_size = current_size
            elif current_size == 1:
                continue
            elif batch_size == 1:
                batch_size = current_size
            elif current_size != batch_size:
                raise ValueError(
                    f"Mismatched batch sizes across targets; {name} had batch={current_size}, expected {batch_size}"
                )
        return int(batch_size or 1)

    @staticmethod
    def _broadcast_batch_array(
        value: np.ndarray | None,
        *,
        fallback: np.ndarray,
        batch_size: int,
        item_dim: int,
        label: str,
    ) -> np.ndarray:
        if value is None:
            return np.tile(
                np.asarray(fallback, dtype=np.float64).reshape(1, item_dim),
                (batch_size, 1),
            )
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != item_dim:
                raise ValueError(
                    f"{label} must have shape ({item_dim},) or (B, {item_dim})"
                )
            return np.tile(arr.reshape(1, item_dim), (batch_size, 1))
        arr = arr.reshape(-1, item_dim)
        if arr.shape[0] == batch_size:
            return arr
        if arr.shape[0] == 1:
            return np.tile(arr, (batch_size, 1))
        raise ValueError(
            f"{label} batch size {arr.shape[0]} does not match expected {batch_size}"
        )

    @staticmethod
    def _pad_batch_array(
        value: np.ndarray,
        *,
        target_batch_size: int,
        pad_value: np.ndarray,
    ) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] > target_batch_size:
            raise ValueError(
                f"Cannot pad batch of size {arr.shape[0]} into target size {target_batch_size}"
            )
        if arr.shape[0] == target_batch_size:
            return arr
        pad = np.tile(
            np.asarray(pad_value, dtype=np.float64).reshape(1, arr.shape[1]),
            (target_batch_size - arr.shape[0], 1),
        )
        return np.concatenate([arr, pad], axis=0)

    @staticmethod
    def _merge_batch_plan_results(results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {
                "status": "Success",
                "status_detail": None,
                "success_mask": np.empty((0,), dtype=bool),
                "status_by_index": [],
                "status_detail_by_index": [],
                "position_error_m": np.empty((0,), dtype=np.float64),
                "rotation_error_deg": np.empty((0,), dtype=np.float64),
                "left_positions_by_index": [],
                "right_positions_by_index": [],
                "curobo_solve_time_ms": 0.0,
                "curobo_total_time_ms": 0.0,
                "curobo_ik_time_ms": 0.0,
                "curobo_graph_time_ms": 0.0,
                "curobo_trajopt_time_ms": 0.0,
                "curobo_finetune_time_ms": 0.0,
                "curobo_attempts": 0,
                "curobo_trajopt_attempts": 0,
                "curobo_used_graph": False,
            }
        merged_success = np.concatenate(
            [
                np.asarray(result["success_mask"], dtype=bool).reshape(-1)
                for result in results
            ],
            axis=0,
        )
        merged_status_detail = next(
            (
                detail
                for result in results
                for detail in [result.get("status_detail")]
                if detail not in {None, "None", "null", ""}
            ),
            None,
        )
        return {
            "status": (
                "Success"
                if np.all(merged_success)
                else (
                    "Partial_Success" if np.any(merged_success) else "Planning_Failed"
                )
            ),
            "status_detail": merged_status_detail,
            "success_mask": merged_success,
            "status_by_index": [
                status
                for result in results
                for status in list(result.get("status_by_index", []))
            ],
            "status_detail_by_index": [
                detail
                for result in results
                for detail in list(result.get("status_detail_by_index", []))
            ],
            "position_error_m": np.concatenate(
                [
                    np.asarray(
                        result.get("position_error_m", []), dtype=np.float64
                    ).reshape(-1)
                    for result in results
                ],
                axis=0,
            ),
            "rotation_error_deg": np.concatenate(
                [
                    np.asarray(
                        result.get("rotation_error_deg", []), dtype=np.float64
                    ).reshape(-1)
                    for result in results
                ],
                axis=0,
            ),
            "left_positions_by_index": [
                positions
                for result in results
                for positions in list(result.get("left_positions_by_index", []))
            ],
            "right_positions_by_index": [
                positions
                for result in results
                for positions in list(result.get("right_positions_by_index", []))
            ],
            "curobo_solve_time_ms": sum(
                float(result.get("curobo_solve_time_ms", 0.0)) for result in results
            ),
            "curobo_total_time_ms": sum(
                float(result.get("curobo_total_time_ms", 0.0)) for result in results
            ),
            "curobo_ik_time_ms": sum(
                float(result.get("curobo_ik_time_ms", 0.0)) for result in results
            ),
            "curobo_graph_time_ms": sum(
                float(result.get("curobo_graph_time_ms", 0.0)) for result in results
            ),
            "curobo_trajopt_time_ms": sum(
                float(result.get("curobo_trajopt_time_ms", 0.0)) for result in results
            ),
            "curobo_finetune_time_ms": sum(
                float(result.get("curobo_finetune_time_ms", 0.0)) for result in results
            ),
            "curobo_attempts": sum(
                int(result.get("curobo_attempts", 0)) for result in results
            ),
            "curobo_trajopt_attempts": sum(
                int(result.get("curobo_trajopt_attempts", 0)) for result in results
            ),
            "curobo_used_graph": any(
                bool(result.get("curobo_used_graph", False)) for result in results
            ),
        }

    def _validate_trajectory(
        self,
        side: str,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        fixed_left = left_positions
        fixed_right = right_positions
        fixed_candidate_name: str | None = None
        if side == "left":
            fixed_candidate_name = "inactive-right-fixed"
            fixed_right = np.tile(
                np.asarray(current_right_jp, dtype=np.float64), (len(left_positions), 1)
            )
        elif side == "right":
            fixed_candidate_name = "inactive-left-fixed"
            fixed_left = np.tile(
                np.asarray(current_left_jp, dtype=np.float64), (len(right_positions), 1)
            )

        if fixed_candidate_name is not None and self._validator is None:
            return fixed_left, fixed_right, None
        if self._validator is None:
            return left_positions, right_positions, None

        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        if fixed_candidate_name is not None:
            candidates.append((fixed_candidate_name, fixed_left, fixed_right))
            candidates.append(("raw", left_positions, right_positions))
        else:
            candidates.append(("raw", left_positions, right_positions))

        for name, cand_left, cand_right in candidates:
            collision_idx = None
            for i in range(len(cand_left)):
                if self._validator.check_collision(cand_left[i], cand_right[i]):
                    collision_idx = i
                    break
            if collision_idx is None:
                if name != "raw":
                    print(f"[cuRoboPlanner] Using post-validated variant: {name}")
                return cand_left, cand_right, None

        if fixed_candidate_name is not None:
            return (
                fixed_left,
                fixed_right,
                "Single-arm cuRobo trajectory required moving the inactive arm; fixed-arm variant was not collision-free",
            )

        return (
            left_positions,
            right_positions,
            "MuJoCo validation failed for the generated cuRobo trajectory",
        )

    def set_gripper_qpos(
        self,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> None:
        if left_gripper is not None:
            self._left_gripper = float(np.clip(left_gripper, 0.0, 1.0))
        if right_gripper is not None:
            self._right_gripper = float(np.clip(right_gripper, 0.0, 1.0))
        if self._validator is not None:
            self._validator.set_gripper_qpos(left_gripper, right_gripper)

    def check_collision(self, left_jp: np.ndarray, right_jp: np.ndarray) -> bool:
        if self._validator is None:
            raise RuntimeError(
                "MuJoCo collision validation is not available for the cuRobo planner"
            )
        return self._validator.check_collision(left_jp, right_jp)

    def check_collision_verbose(self, left_jp: np.ndarray, right_jp: np.ndarray):
        if self._validator is None:
            raise RuntimeError(
                "MuJoCo collision validation is not available for the cuRobo planner"
            )
        return self._validator.check_collision_verbose(left_jp, right_jp)

    def _plan_batch_to_pose_chunk(
        self,
        *,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        target_left_pos: np.ndarray,
        target_left_quat_xyzw: np.ndarray,
        target_right_pos: np.ndarray,
        target_right_quat_xyzw: np.ndarray,
        side: Literal["left", "right", "both"],
        validate_trajectory: bool = True,
    ) -> dict[str, Any]:
        if self._motion_gen is None or self._ee_pose_plan_config is None:
            raise RuntimeError("Pose motion generator is not initialized")
        actual_batch_size = int(
            np.asarray(target_left_pos, dtype=np.float64).reshape(-1, 3).shape[0]
        )
        if actual_batch_size <= 0:
            return self._merge_batch_plan_results([])
        if actual_batch_size > self._batch_planner_capacity:
            raise ValueError(
                f"Batch query size {actual_batch_size} exceeds CUDA-graphed batch capacity {self._batch_planner_capacity}"
            )

        fixed_batch_size = int(self._batch_planner_capacity)
        # Pad with a real query from this chunk instead of the current EE pose.
        # Using no-op current-pose padding can trigger cuRobo shape bugs for
        # partially filled fixed CUDA-graph batches, especially for bimanual queries.
        pad_left_pos = np.asarray(target_left_pos, dtype=np.float64).reshape(-1, 3)[-1]
        pad_left_quat = np.asarray(target_left_quat_xyzw, dtype=np.float64).reshape(
            -1, 4
        )[-1]
        pad_right_pos = np.asarray(target_right_pos, dtype=np.float64).reshape(-1, 3)[
            -1
        ]
        pad_right_quat = np.asarray(target_right_quat_xyzw, dtype=np.float64).reshape(
            -1, 4
        )[-1]
        padded_left_pos = self._pad_batch_array(
            target_left_pos,
            target_batch_size=fixed_batch_size,
            pad_value=pad_left_pos,
        )
        padded_left_quat = self._pad_batch_array(
            target_left_quat_xyzw,
            target_batch_size=fixed_batch_size,
            pad_value=pad_left_quat,
        )
        padded_right_pos = self._pad_batch_array(
            target_right_pos,
            target_batch_size=fixed_batch_size,
            pad_value=pad_right_pos,
        )
        padded_right_quat = self._pad_batch_array(
            target_right_quat_xyzw,
            target_batch_size=fixed_batch_size,
            pad_value=pad_right_quat,
        )

        start_state = self._make_joint_state(
            np.tile(
                np.asarray(current_left_jp, dtype=np.float64).reshape(1, 6),
                (fixed_batch_size, 1),
            ),
            np.tile(
                np.asarray(current_right_jp, dtype=np.float64).reshape(1, 6),
                (fixed_batch_size, 1),
            ),
        )
        goal_pose = self._make_pose(padded_left_pos, padded_left_quat)
        link_poses = {
            "right_grasp": self._make_pose(padded_right_pos, padded_right_quat)
        }

        plan_cfg = self._ee_pose_plan_config.clone()
        print(
            "[cuRoboPlanner] plan_batch_to_pose_chunk "
            f"solver_speed={self._solver_speed} "
            f"side={side} actual_batch={actual_batch_size} fixed_batch={fixed_batch_size} "
            f"pad_mode=repeat-last-query "
            f"validate_trajectory={'on' if validate_trajectory else 'off'} "
            f"time_dilation_factor={getattr(plan_cfg, 'time_dilation_factor', None)}"
        )
        try:
            result = self._motion_gen.plan_batch(
                start_state,
                goal_pose,
                plan_cfg,
                link_poses=link_poses,
            )
        except Exception as exc:
            raise RuntimeError(
                "cuRobo batch plan failed "
                f"(side={side}, actual_batch={actual_batch_size}, fixed_batch={fixed_batch_size}, "
                f"pad_mode=repeat-last-query, "
                f"validate_trajectory={validate_trajectory}, "
                f"time_dilation_factor={getattr(plan_cfg, 'time_dilation_factor', None)}): {exc}"
            ) from exc

        success_all = (
            np.asarray(result.success.detach().cpu().numpy(), dtype=bool).reshape(-1)
            if result.success is not None
            else np.zeros((fixed_batch_size,), dtype=bool)
        )
        if success_all.shape[0] != fixed_batch_size:
            raise ValueError(
                f"cuRobo returned batch size {success_all.shape[0]}, expected padded size {fixed_batch_size}"
            )
        success = success_all[:actual_batch_size].copy()

        status_detail = getattr(result.status, "value", str(result.status))
        status_detail = None if status_detail in {"None", "null"} else status_detail
        failed_status = (
            "IK_Failed"
            if result.status == self._MotionGenStatus.IK_FAIL
            else "Planning_Failed"
        )
        status_by_index = np.where(success, "Success", failed_status).tolist()
        status_detail_by_index: list[str | None] = [
            None if ok else status_detail for ok in success
        ]
        left_positions_by_index: list[np.ndarray | None] = [None] * actual_batch_size
        right_positions_by_index: list[np.ndarray | None] = [None] * actual_batch_size

        if np.any(success):
            interpolated_plan = getattr(result, "interpolated_plan", None)
            if interpolated_plan is None:
                for idx, ok in enumerate(success.tolist()):
                    if not ok:
                        continue
                    success[idx] = False
                    status_by_index[idx] = "Planning_Failed"
                    status_detail_by_index[idx] = (
                        "Batch planner did not return a trajectory"
                    )
            else:
                plan_batch = (
                    result.get_paths()
                    if getattr(result, "path_buffer_last_tstep", None) is not None
                    else [interpolated_plan[idx] for idx in range(result_batch_size)]
                )
                for idx, ok in enumerate(success.tolist()):
                    if not ok:
                        continue
                    positions = self._joint_state_positions(
                        plan_batch[idx], self._motion_gen
                    )
                    raw_left_positions = np.asarray(
                        positions[:, :6], dtype=np.float64
                    ).reshape(-1, 6)
                    raw_right_positions = np.asarray(
                        positions[:, 6:12], dtype=np.float64
                    ).reshape(-1, 6)
                    if validate_trajectory:
                        left_positions, right_positions, validation_error = (
                            self._validate_trajectory(
                                side=side,
                                left_positions=raw_left_positions,
                                right_positions=raw_right_positions,
                                current_left_jp=np.asarray(
                                    current_left_jp, dtype=np.float64
                                ),
                                current_right_jp=np.asarray(
                                    current_right_jp, dtype=np.float64
                                ),
                            )
                        )
                        if validation_error is not None:
                            success[idx] = False
                            status_by_index[idx] = "Planning_Failed"
                            status_detail_by_index[idx] = validation_error
                            continue
                    else:
                        left_positions = raw_left_positions
                        right_positions = raw_right_positions
                        if side == "left":
                            right_positions = np.tile(
                                np.asarray(current_right_jp, dtype=np.float64).reshape(
                                    1, 6
                                ),
                                (len(left_positions), 1),
                            )
                        elif side == "right":
                            left_positions = np.tile(
                                np.asarray(current_left_jp, dtype=np.float64).reshape(
                                    1, 6
                                ),
                                (len(right_positions), 1),
                            )
                    left_positions_by_index[idx] = left_positions
                    right_positions_by_index[idx] = right_positions

        timing_info = _curobo_timing_info(result)
        batch_position_error = _as_numpy_metric_array(
            getattr(result, "position_error", None),
            fixed_batch_size,
        )[:actual_batch_size]
        batch_rotation_error_rad = _as_numpy_metric_array(
            getattr(result, "rotation_error", None),
            fixed_batch_size,
        )[:actual_batch_size]
        return {
            "status": (
                "Success"
                if np.all(success)
                else ("Partial_Success" if np.any(success) else "Planning_Failed")
            ),
            "status_detail": status_detail,
            "success_mask": success,
            "status_by_index": status_by_index,
            "status_detail_by_index": status_detail_by_index,
            "position_error_m": batch_position_error,
            "rotation_error_deg": np.rad2deg(batch_rotation_error_rad),
            "left_positions_by_index": left_positions_by_index,
            "right_positions_by_index": right_positions_by_index,
            **timing_info,
        }

    def plan_to_pose(
        self,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        target_left_pos: np.ndarray | None = None,
        target_left_quat_xyzw: np.ndarray | None = None,
        target_right_pos: np.ndarray | None = None,
        target_right_quat_xyzw: np.ndarray | None = None,
        side: Literal["left", "right", "both"] = "both",
        max_iters: int = 5000,  # kept for API compatibility
        step_size: float = 0.15,  # kept for API compatibility
        max_joint_vel: float = 1.0,  # kept for API compatibility
        dt: float = _INTERPOLATION_DT,
        ik_error_threshold: float = 0.08,  # kept for API compatibility
        left_gripper: float | None = None,
        right_gripper: float | None = None,
        validate_trajectory: bool = True,
        verbose: bool = False,  # kept for API compatibility
        **_: Any,
    ) -> dict[str, Any]:
        del max_iters, step_size, max_joint_vel, dt, ik_error_threshold, verbose
        # Single-target EE planning is just the batched EE planner with batch size 1.
        batch_result = self.plan_batch_to_pose(
            current_left_jp=current_left_jp,
            current_right_jp=current_right_jp,
            target_left_pos=(
                None
                if target_left_pos is None
                else np.asarray(target_left_pos, dtype=np.float64).reshape(1, 3)
            ),
            target_left_quat_xyzw=(
                None
                if target_left_quat_xyzw is None
                else np.asarray(target_left_quat_xyzw, dtype=np.float64).reshape(1, 4)
            ),
            target_right_pos=(
                None
                if target_right_pos is None
                else np.asarray(target_right_pos, dtype=np.float64).reshape(1, 3)
            ),
            target_right_quat_xyzw=(
                None
                if target_right_quat_xyzw is None
                else np.asarray(target_right_quat_xyzw, dtype=np.float64).reshape(1, 4)
            ),
            side=side,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            validate_trajectory=validate_trajectory,
        )

        success_mask = np.asarray(
            batch_result.get("success_mask", [False]), dtype=bool
        ).reshape(-1)
        success = bool(success_mask[0]) if success_mask.shape[0] >= 1 else False
        status_by_index = list(batch_result.get("status_by_index", []))
        status_detail_by_index = list(batch_result.get("status_detail_by_index", []))
        left_positions_by_index = list(batch_result.get("left_positions_by_index", []))
        right_positions_by_index = list(
            batch_result.get("right_positions_by_index", [])
        )
        position_error_values = np.asarray(
            batch_result.get("position_error_m", [np.nan]), dtype=np.float64
        ).reshape(-1)
        rotation_error_values = np.asarray(
            batch_result.get("rotation_error_deg", [np.nan]), dtype=np.float64
        ).reshape(-1)
        status = (
            str(status_by_index[0])
            if status_by_index
            else (
                "Success"
                if success
                else str(batch_result.get("status", "Planning_Failed"))
            )
        )
        status_detail = (
            status_detail_by_index[0]
            if status_detail_by_index
            else batch_result.get("status_detail")
        )
        position_error_m = (
            float(position_error_values[0])
            if position_error_values.size
            else float("nan")
        )
        rotation_error_deg = (
            float(rotation_error_values[0])
            if rotation_error_values.size
            else float("nan")
        )

        if not success:
            return {
                "status": status,
                "status_detail": status_detail,
                "position": np.empty((0, 12), dtype=np.float64),
                "left_positions": np.empty((0, 6), dtype=np.float64),
                "right_positions": np.empty((0, 6), dtype=np.float64),
                "position_error_m": position_error_m,
                "rotation_error_deg": rotation_error_deg,
                **{
                    key: value
                    for key, value in batch_result.items()
                    if key.startswith("curobo_")
                },
            }

        left_positions = (
            np.asarray(left_positions_by_index[0], dtype=np.float64).reshape(-1, 6)
            if left_positions_by_index and left_positions_by_index[0] is not None
            else np.empty((0, 6), dtype=np.float64)
        )
        right_positions = (
            np.asarray(right_positions_by_index[0], dtype=np.float64).reshape(-1, 6)
            if right_positions_by_index and right_positions_by_index[0] is not None
            else np.empty((0, 6), dtype=np.float64)
        )
        positions = (
            np.concatenate([left_positions, right_positions], axis=1)
            if left_positions.shape[0] and right_positions.shape[0]
            else np.empty((0, 12), dtype=np.float64)
        )
        return {
            "status": status,
            "status_detail": status_detail,
            "position": positions,
            "left_positions": left_positions,
            "right_positions": right_positions,
            "position_error_m": position_error_m,
            "rotation_error_deg": rotation_error_deg,
            **{
                key: value
                for key, value in batch_result.items()
                if key.startswith("curobo_")
            },
        }

    def plan_batch_to_pose(
        self,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        target_left_pos: np.ndarray | None = None,
        target_left_quat_xyzw: np.ndarray | None = None,
        target_right_pos: np.ndarray | None = None,
        target_right_quat_xyzw: np.ndarray | None = None,
        side: Literal["left", "right", "both"] = "both",
        max_iters: int = 5000,  # kept for API compatibility
        step_size: float = 0.15,  # kept for API compatibility
        max_joint_vel: float = 1.0,  # kept for API compatibility
        dt: float = _INTERPOLATION_DT,
        ik_error_threshold: float = 0.08,  # kept for API compatibility
        left_gripper: float | None = None,
        right_gripper: float | None = None,
        validate_trajectory: bool = True,
        verbose: bool = False,  # kept for API compatibility
        **_: Any,
    ) -> dict[str, Any]:
        del max_iters, step_size, max_joint_vel, dt, ik_error_threshold, verbose
        self.set_gripper_qpos(left_gripper, right_gripper)
        current_left_jp, current_right_jp = self._clip_joint_positions(
            current_left_jp,
            current_right_jp,
            label="batch start state",
        )
        batch_size = self._infer_pose_batch_size(
            target_left_pos=target_left_pos,
            target_left_quat_xyzw=target_left_quat_xyzw,
            target_right_pos=target_right_pos,
            target_right_quat_xyzw=target_right_quat_xyzw,
        )
        if batch_size <= 0:
            return self._merge_batch_plan_results([])

        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kin.forward_kinematics(
            np.asarray(current_left_jp, dtype=np.float64),
            np.asarray(current_right_jp, dtype=np.float64),
        )
        tgt_l_pos = self._broadcast_batch_array(
            target_left_pos,
            fallback=cur_l_pos,
            batch_size=batch_size,
            item_dim=3,
            label="target_left_pos",
        )
        tgt_l_q = self._broadcast_batch_array(
            target_left_quat_xyzw,
            fallback=cur_l_q,
            batch_size=batch_size,
            item_dim=4,
            label="target_left_quat_xyzw",
        )
        tgt_r_pos = self._broadcast_batch_array(
            target_right_pos,
            fallback=cur_r_pos,
            batch_size=batch_size,
            item_dim=3,
            label="target_right_pos",
        )
        tgt_r_q = self._broadcast_batch_array(
            target_right_quat_xyzw,
            fallback=cur_r_q,
            batch_size=batch_size,
            item_dim=4,
            label="target_right_quat_xyzw",
        )
        chunk_results: list[dict[str, Any]] = []
        for start in range(0, batch_size, self._batch_planner_capacity):
            end = min(start + self._batch_planner_capacity, batch_size)
            chunk_results.append(
                self._plan_batch_to_pose_chunk(
                    current_left_jp=current_left_jp,
                    current_right_jp=current_right_jp,
                    target_left_pos=tgt_l_pos[start:end],
                    target_left_quat_xyzw=tgt_l_q[start:end],
                    target_right_pos=tgt_r_pos[start:end],
                    target_right_quat_xyzw=tgt_r_q[start:end],
                    side=side,
                    validate_trajectory=validate_trajectory,
                )
            )
        return self._merge_batch_plan_results(chunk_results)


YamMotionPlanner = YamMotionPlannerCurobo
