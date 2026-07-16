"""No-motion RGB-D gripper geometry analysis for YAM plate work.

This script does not move hardware.  It reads live RGB/depth/intrinsics/
extrinsics from the camera portal, segments the blue gripper material by RGB
color, computes simple keypoints/components, unprojects valid depth pixels into
world coordinates, and writes compact artifacts for plate-grasp planning.
"""

from __future__ import annotations

import base64
import inspect
import io
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from cap.saved_scripts.yam_runtime import append_stage_summary, current_run_dir, json_safe, write_json


TASK_NAME = "analyze_gripper_geometry_rgbd"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _tool(name: str):
    value = globals().get(name)
    return value if callable(value) else None


def _runtime_api_surface(cameras: list[str]) -> dict[str, Any]:
    """Record the actual injected run_script names and camera portal surface."""
    expected = [
        "detect_objects_oneshot",
        "freespace_move",
        "get_camera_image",
        "get_robot_state",
        "run_in_background",
        "set_gripper",
        "open_gripper",
        "close_gripper",
        "go_home",
    ]
    semantic_signatures = {
        "detect_objects_oneshot": "detect_objects_oneshot(query: str | list[str], camera='top', max_retries=3)",
        "freespace_move": "freespace_move(..., preview_only=True) for no-motion checks; trajectory_cache_key executes motion",
        "get_camera_image": "get_camera_image(camera: 'top' | 'left' | 'right') -> RGB array",
        "get_robot_state": "get_robot_state() -> arms/{left,right}/ee_pos, ee_quat, ee_rpy, joint_pos, gripper_pos",
        "run_in_background": "run_in_background(fn, *args, **kwargs) -> ToolHandle",
        "set_gripper": "set_gripper(side, normalized_position, ...) MOTION-CAPABLE",
        "open_gripper": "open_gripper(side) MOTION-CAPABLE",
        "close_gripper": "close_gripper(side) MOTION-CAPABLE",
        "go_home": "go_home() MOTION-CAPABLE",
    }
    tools: dict[str, Any] = {}
    for name in expected:
        value = globals().get(name)
        item: dict[str, Any] = {
            "injected": callable(value),
            "repr": repr(value)[:240] if value is not None else None,
            "semantic_signature": semantic_signatures.get(name),
        }
        if callable(value):
            try:
                item["python_signature"] = str(inspect.signature(value))
            except Exception as exc:
                item["python_signature_error"] = f"{type(exc).__name__}: {exc}"
        tools[name] = item

    portal_packet: dict[str, Any] = {
        "addr": os.environ.get("OPENFORGE_CAMERA_PORTAL_ADDR", "127.0.0.1:8300"),
        "available": False,
        "methods_checked": ["get_camera_image", "get_camera_depth", "get_camera_intrinsics", "get_camera_extrinsics"],
        "cameras": {},
    }
    try:
        import portal

        client = portal.Client(portal_packet["addr"])
        portal_packet["available"] = True
        for camera in cameras:
            cam_item: dict[str, Any] = {}
            for method in portal_packet["methods_checked"]:
                fn = getattr(client, method, None)
                if not callable(fn):
                    cam_item[method] = {"available": False}
                    continue
                try:
                    data = fn(camera).result()
                    shape = list(np.asarray(data).shape) if method in {"get_camera_image", "get_camera_depth"} else None
                    cam_item[method] = {
                        "available": True,
                        "type": type(data).__name__,
                        "shape": shape,
                        "sample": json_safe(data) if method in {"get_camera_intrinsics", "get_camera_extrinsics"} else None,
                    }
                except Exception as exc:
                    cam_item[method] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
            portal_packet["cameras"][camera] = cam_item
    except Exception as exc:
        portal_packet["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "schema": "openforge.runtime_api_surface.v1",
        "source": "actual run_script globals plus camera portal probe",
        "tools": tools,
        "camera_portal": portal_packet,
        "motion_capable_tools_not_called": ["set_gripper", "open_gripper", "close_gripper", "go_home"],
    }


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _static_top_camera_extrinsics() -> dict[str, Any]:
    from robot.models.station.paths import get_top_camera_frame, needs_optical_flip
    from robot.yam.kinematics import YamKinematics

    frame_name = os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame())
    kin = YamKinematics()
    T = kin.configuration.get_transform_frame_to_world(frame_name, "body")
    rot = T.rotation()
    rot_mat = rot.as_matrix() if hasattr(rot, "as_matrix") else rot.matrix()
    return {
        "position": [float(x) for x in T.translation().tolist()],
        "rotation": np.asarray(rot_mat, dtype=np.float64).reshape(3, 3).tolist(),
        "needs_optical_flip": needs_optical_flip("top"),
        "source": "station_model_static_top_camera",
    }


def _camera_to_world_from_extrinsics(extrinsics: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extrinsics["position"], dtype=np.float64).reshape(3)
    if extrinsics.get("needs_optical_flip", False):
        rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _world_to_camera_from_extrinsics(extrinsics: dict[str, Any]) -> np.ndarray:
    return np.linalg.inv(_camera_to_world_from_extrinsics(extrinsics))


def _capture_portal_rgbd(camera: str) -> dict[str, Any]:
    import portal

    addr = os.environ.get("OPENFORGE_CAMERA_PORTAL_ADDR", "127.0.0.1:8300").strip()
    client = portal.Client(addr)
    rgb = np.asarray(client.get_camera_image(camera).result())
    depth = np.asarray(client.get_camera_depth(camera).result(), dtype=np.float32)
    intrinsics = [float(x) for x in client.get_camera_intrinsics(camera).result()]
    try:
        extrinsics = client.get_camera_extrinsics(camera).result()
        extrinsics = dict(extrinsics)
        extrinsics.setdefault("source", f"portal:{addr}")
    except Exception as exc:
        if camera != "top":
            raise
        extrinsics = _static_top_camera_extrinsics()
        extrinsics["portal_extrinsics_error"] = f"{type(exc).__name__}: {exc}"
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError(f"invalid RGB shape for {camera}: {rgb.shape}")
    if depth.ndim != 2:
        raise RuntimeError(f"invalid depth shape for {camera}: {depth.shape}")
    if depth.size and float(np.nanmedian(depth)) > 10.0:
        depth = depth / 1000.0
    if len(intrinsics) != 4:
        raise RuntimeError(f"invalid intrinsics for {camera}: {intrinsics}")
    return {
        "camera": camera,
        "portal_addr": addr,
        "rgb": np.ascontiguousarray(rgb[:, :, :3]),
        "depth": np.ascontiguousarray(depth),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }


def _robot_arm_state(robot_state: dict[str, Any] | None, side: str) -> dict[str, Any] | None:
    arms = (robot_state or {}).get("arms") if isinstance(robot_state, dict) else None
    arm = (arms or {}).get(side)
    return arm if isinstance(arm, dict) else None


def _project_world_point(point_world: Any, intr: list[float], world_to_camera: np.ndarray) -> dict[str, Any] | None:
    xyz = _round_list(point_world, 8, 3)
    if len(xyz) != 3:
        return None
    p = np.asarray([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
    pc = world_to_camera @ p
    if pc[2] <= 1e-6:
        return {"world_m": xyz, "camera_xyz_m": _round_list(pc[:3], 5, 3), "valid": False}
    fx, fy, cx, cy = intr
    uv = [float(fx * pc[0] / pc[2] + cx), float(fy * pc[1] / pc[2] + cy)]
    return {
        "world_m": xyz,
        "camera_xyz_m": _round_list(pc[:3], 5, 3),
        "uv_px": _round_list(uv, 2, 2),
        "valid": True,
    }


def _project_robot_state(robot_state: dict[str, Any] | None, intr: list[float], extrinsics: dict[str, Any]) -> dict[str, Any]:
    w2c = _world_to_camera_from_extrinsics(extrinsics)
    projections: dict[str, Any] = {}
    for side in ("left", "right"):
        arm = _robot_arm_state(robot_state, side)
        if not arm:
            continue
        ee_pos = arm.get("ee_pos") or []
        projected = _project_world_point(ee_pos, intr, w2c)
        projections[side] = {
            "ee_pos_world_m": _round_list(ee_pos, 5, 3),
            "ee_rpy_deg": _round_list(arm.get("ee_rpy") or [], 4, 3),
            "ee_quat_xyzw": _round_list(arm.get("ee_quat") or arm.get("ee_quaternion") or [], 5, 4),
            "gripper_pos_normalized": arm.get("gripper_pos"),
            "projected_ee": projected,
        }
    return projections


def _save_rgb(path: Path, rgb: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)
    return str(path)


def _save_depth_preview(path: Path, depth: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    if not valid.any():
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [5, 95])
        scaled = np.clip((depth - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        preview = (255.0 * scaled).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)
    return str(path)


def _blue_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(
        [
            int(_env_float("OPENFORGE_GRIPPER_RGBD_H_MIN", 88)),
            int(_env_float("OPENFORGE_GRIPPER_RGBD_S_MIN", 45)),
            int(_env_float("OPENFORGE_GRIPPER_RGBD_V_MIN", 45)),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            int(_env_float("OPENFORGE_GRIPPER_RGBD_H_MAX", 128)),
            int(_env_float("OPENFORGE_GRIPPER_RGBD_S_MAX", 255)),
            int(_env_float("OPENFORGE_GRIPPER_RGBD_V_MAX", 255)),
        ],
        dtype=np.uint8,
    )
    mask = cv2.inRange(hsv, lower, upper) > 0
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    color_guard = (b >= r + int(_env_float("OPENFORGE_GRIPPER_RGBD_BLUE_MINUS_R_MIN", 25))) & (
        b >= g - int(_env_float("OPENFORGE_GRIPPER_RGBD_BLUE_MINUS_G_ALLOW", 35))
    )
    mask &= color_guard
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 0


def _scale_intrinsics(intr: list[float], depth_shape: tuple[int, int], rgb_shape: tuple[int, int, int]) -> list[float]:
    fx, fy, cx, cy = [float(x) for x in intr]
    dh, dw = depth_shape
    rh, rw = rgb_shape[:2]
    if dw > 0 and dh > 0 and (dw != rw or dh != rh):
        fx *= rw / dw
        cx *= rw / dw
        fy *= rh / dh
        cy *= rh / dh
    return [fx, fy, cx, cy]


def _pixel_depth(depth: np.ndarray, u: float, v: float) -> float | None:
    h, w = depth.shape
    x = int(round(u))
    y = int(round(v))
    xs = range(max(0, x - 2), min(w, x + 3))
    ys = range(max(0, y - 2), min(h, y + 3))
    vals = [float(depth[yy, xx]) for yy in ys for xx in xs if np.isfinite(depth[yy, xx]) and 0.05 < depth[yy, xx] < 3.0]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def _unproject_pixels(
    uv: np.ndarray,
    z: np.ndarray,
    intr: list[float],
    camera_to_world: np.ndarray,
) -> np.ndarray:
    fx, fy, cx, cy = intr
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_world = (camera_to_world @ pts_cam.T).T[:, :3]
    return pts_world


def _component_pca(points: np.ndarray) -> dict[str, Any] | None:
    if points.shape[0] < 6:
        return None
    center = np.median(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    proj = centered @ eigvecs
    lo = np.percentile(proj, 5, axis=0)
    hi = np.percentile(proj, 95, axis=0)
    extents = hi - lo
    return {
        "centroid_world_m": _round_list(center, 5, 3),
        "pca_axes_world": [_round_list(eigvecs[:, i], 5, 3) for i in range(3)],
        "pca_extents_m_p05_p95": _round_list(extents, 5, 3),
        "pca_eigenvalues": _round_list(eigvals, 8, 3),
    }


def _filtered_component_points(
    xs: np.ndarray,
    ys: np.ndarray,
    depth_values: np.ndarray,
    intr: list[float],
    c2w: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.isfinite(depth_values) & (depth_values > 0.05) & (depth_values < 3.0)
    stats: dict[str, Any] = {"valid_depth_pixels_raw": int(valid.sum())}
    if valid.sum() < 6:
        return np.empty((0, 3), dtype=np.float64), stats
    valid_depth = depth_values[valid].astype(np.float64)
    median_depth = float(np.median(valid_depth))
    mad = float(np.median(np.abs(valid_depth - median_depth)))
    window = max(_env_float("OPENFORGE_GRIPPER_RGBD_DEPTH_FILTER_MIN_WINDOW_M", 0.012), 3.0 * 1.4826 * mad)
    near = valid & (np.abs(depth_values.astype(np.float64) - median_depth) <= window)
    stats.update(
        {
            "median_depth_m": round(median_depth, 5),
            "mad_depth_m": round(mad, 5),
            "depth_filter_window_m": round(float(window), 5),
            "valid_depth_pixels_filtered": int(near.sum()),
        }
    )
    if near.sum() < 6:
        return np.empty((0, 3), dtype=np.float64), stats
    uv = np.stack([xs[near].astype(np.float64), ys[near].astype(np.float64)], axis=1)
    z = depth_values[near].astype(np.float64)
    return _unproject_pixels(uv, z, intr, c2w), stats


def _component_side_assignment(center_px: list[float], robot_projection: dict[str, Any]) -> dict[str, Any]:
    distances: dict[str, float] = {}
    for side, item in robot_projection.items():
        uv = ((item.get("projected_ee") or {}).get("uv_px") or [])
        if len(uv) == 2:
            distances[side] = float(np.linalg.norm(np.asarray(center_px, dtype=np.float64) - np.asarray(uv, dtype=np.float64)))
    if not distances:
        return {"assigned_side": None, "distance_to_projected_ee_px": None, "all_distances_px": {}}
    side = min(distances, key=distances.get)
    return {
        "assigned_side": side,
        "distance_to_projected_ee_px": round(float(distances[side]), 2),
        "all_distances_px": {k: round(v, 2) for k, v in distances.items()},
    }


def _component_world_relation(
    center_world: list[float] | None,
    assigned_side: str | None,
    robot_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if center_world is None or assigned_side not in {"left", "right"}:
        return None
    arm = _robot_arm_state(robot_state, assigned_side)
    if not arm:
        return None
    ee = _round_list(arm.get("ee_pos") or [], 8, 3)
    if len(ee) != 3:
        return None
    delta = [float(center_world[i]) - ee[i] for i in range(3)]
    return {
        "ee_to_component_center_world_m": _round_list(delta, 5, 3),
        "ee_to_component_center_norm_m": round(float(np.linalg.norm(np.asarray(delta))), 5),
        "xy_norm_m": round(float(np.linalg.norm(np.asarray(delta[:2]))), 5),
        "z_delta_m": round(float(delta[2]), 5),
    }


def _analyze_camera(camera: str, run_dir: Path, robot_state: dict[str, Any] | None) -> dict[str, Any]:
    snap = _capture_portal_rgbd(camera)
    rgb = snap["rgb"]
    depth = snap["depth"]
    intr = _scale_intrinsics(snap["intrinsics"], depth.shape, rgb.shape)
    c2w = _camera_to_world_from_extrinsics(snap["extrinsics"])
    robot_projection = _project_robot_state(robot_state, intr, snap["extrinsics"])
    h, w = rgb.shape[:2]
    camera_dir = run_dir / "observations" / f"rgbd_{camera}"
    rgb_path = _save_rgb(camera_dir / "rgb.png", rgb)
    depth_preview_path = _save_depth_preview(camera_dir / "depth_preview.png", depth)
    np.save(camera_dir / "depth_m.npy", depth)
    depth_valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
    depth_summary = {
        "valid_pixels": int(depth_valid.sum()),
        "total_pixels": int(depth.size),
        "valid_fraction": round(float(depth_valid.sum()) / float(max(1, depth.size)), 5),
    }
    if depth_valid.any():
        depth_summary.update(
            {
                "min_m": round(float(np.min(depth[depth_valid])), 5),
                "p05_m": round(float(np.percentile(depth[depth_valid], 5)), 5),
                "median_m": round(float(np.median(depth[depth_valid])), 5),
                "p95_m": round(float(np.percentile(depth[depth_valid], 95)), 5),
                "max_m": round(float(np.max(depth[depth_valid])), 5),
            }
        )

    mask = _blue_mask(rgb)
    mask_path = _save_depth_preview(camera_dir / "blue_mask.png", mask.astype(np.float32))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    min_area = int(_env_float("OPENFORGE_GRIPPER_RGBD_MIN_AREA_PX", 20))
    max_area = int(_env_float("OPENFORGE_GRIPPER_RGBD_MAX_AREA_PX", 30000))
    components: list[dict[str, Any]] = []
    overlay = rgb.copy()
    for label in range(1, num_labels):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area:
            continue
        comp_mask = labels == label
        ys, xs = np.nonzero(comp_mask)
        valid_depth = depth[ys, xs]
        points_world, depth_stats = _filtered_component_points(xs, ys, valid_depth, intr, c2w)
        contour_points = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(contour_points)
        box = cv2.boxPoints(rect).astype(np.int32)
        rw, rh = [float(v) for v in rect[1]]
        long_px = max(rw, rh)
        short_px = min(rw, rh)
        angle = float(rect[2])
        center_px = [float(centroids[label][0]), float(centroids[label][1])]
        # Endpoint keypoints along minAreaRect's longer side in image space.
        if rw >= rh:
            theta = np.deg2rad(angle)
        else:
            theta = np.deg2rad(angle + 90.0)
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        p1 = np.asarray(center_px) - direction * (long_px / 2.0)
        p2 = np.asarray(center_px) + direction * (long_px / 2.0)
        endpoint_world: list[list[float] | None] = []
        for point in (p1, p2):
            z = _pixel_depth(depth, float(point[0]), float(point[1]))
            if z is None:
                endpoint_world.append(None)
            else:
                endpoint_world.append(
                    _round_list(_unproject_pixels(point.reshape(1, 2), np.asarray([z]), intr, c2w)[0], 5, 3)
                )
        pca = _component_pca(points_world)
        centroid_world = (pca or {}).get("centroid_world_m")
        assignment = _component_side_assignment(center_px, robot_projection)
        relation = _component_world_relation(centroid_world, assignment.get("assigned_side"), robot_state)
        median_depth = depth_stats.get("median_depth_m")
        metric_from_depth = None
        if median_depth is not None:
            fx, fy, _, _ = intr
            metric_from_depth = {
                "long_axis_m_pinhole": round(float(long_px) * float(median_depth) / float(fx), 5),
                "short_axis_m_pinhole": round(float(short_px) * float(median_depth) / float(fy), 5),
                "scale_source": "component_median_depth_and_camera_intrinsics",
            }
        nearfield_self = camera in {"left", "right"} and median_depth is not None and float(median_depth) < _env_float(
            "OPENFORGE_GRIPPER_RGBD_WRIST_NEARFIELD_MAX_DEPTH_M", 0.14
        )
        top_near_projected_ee = camera == "top" and assignment.get("distance_to_projected_ee_px") is not None and float(
            assignment["distance_to_projected_ee_px"]
        ) <= _env_float("OPENFORGE_GRIPPER_RGBD_TOP_EE_NEAR_PX", 35.0)
        reliability = "reliable_visible_blue_fingertip" if (nearfield_self or top_near_projected_ee) else "diagnostic_or_possible_contamination"
        item = {
            "label": int(label),
            "area_px": int(area),
            "bbox_xywh_px": [x, y, bw, bh],
            "center_px": _round_list(center_px, 2, 2),
            "min_area_rect": {
                "center_px": _round_list(rect[0], 2, 2),
                "size_px": _round_list(rect[1], 2, 2),
                "long_axis_px": round(long_px, 2),
                "short_axis_px": round(short_px, 2),
                "angle_deg": round(angle, 2),
            },
            "major_axis_endpoints_px": [_round_list(p1, 2, 2), _round_list(p2, 2, 2)],
            "major_axis_endpoints_world_m": endpoint_world,
            "depth_stats": depth_stats,
            "valid_depth_pixels": int(depth_stats.get("valid_depth_pixels_filtered") or 0),
            "median_depth_m": median_depth,
            "metric_from_depth": metric_from_depth,
            "world_pca": pca,
            "assigned_side_from_projected_ee": assignment,
            "relation_to_assigned_ee": relation,
            "reliability": reliability,
            "same_side_or_contamination": (
                "same_side_nearfield" if nearfield_self else "same_side_projected_ee" if top_near_projected_ee else "possible_opposite_or_background"
            ),
            "image_side": "left_half" if center_px[0] < w / 2 else "right_half",
            "visual_shape": "elongated_blue_finger_patch" if short_px > 0 and long_px / short_px >= 1.8 else "compact_blue_patch",
        }
        components.append(item)
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
        cv2.circle(overlay, tuple(np.round(center_px).astype(int)), 4, (255, 0, 0), -1)
        cv2.line(overlay, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), (255, 255, 0), 2)
        cv2.putText(
            overlay,
            f"{label} {long_px:.0f}x{short_px:.0f}px",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    overlay_path = _save_rgb(camera_dir / "blue_components_overlay.png", overlay)
    components = sorted(components, key=lambda item: item["area_px"], reverse=True)
    packet = {
        "camera": camera,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_summary": depth_summary,
        "rgb_path": rgb_path,
        "depth_preview_path": depth_preview_path,
        "depth_npy_path": str(camera_dir / "depth_m.npy"),
        "blue_mask_path": mask_path,
        "overlay_path": overlay_path,
        "intrinsics_raw": _round_list(snap["intrinsics"], 5, 4),
        "intrinsics_scaled_to_rgb": _round_list(intr, 5, 4),
        "extrinsics": json_safe(snap["extrinsics"]),
        "robot_projection": robot_projection,
        "component_count": len(components),
        "components": components,
    }
    packet["path"] = write_json(camera_dir / "analysis.json", packet)
    return packet


def _summarize_components(camera_packets: list[dict[str, Any]]) -> dict[str, Any]:
    all_components: list[dict[str, Any]] = []
    for packet in camera_packets:
        for comp in packet.get("components") or []:
            item = dict(comp)
            item["camera"] = packet.get("camera")
            all_components.append(item)
    reliable_components = [c for c in all_components if c.get("reliability") == "reliable_visible_blue_fingertip"]
    long_axes_px = [float(c["min_area_rect"]["long_axis_px"]) for c in all_components]
    pca_extents = [
        (c.get("world_pca") or {}).get("pca_extents_m_p05_p95")
        for c in all_components
        if (c.get("world_pca") or {}).get("pca_extents_m_p05_p95")
    ]
    primary_extents = [float(ext[0]) for ext in pca_extents if ext]
    reliable_long_m = [
        float((c.get("metric_from_depth") or {}).get("long_axis_m_pinhole"))
        for c in reliable_components
        if (c.get("metric_from_depth") or {}).get("long_axis_m_pinhole") is not None
    ]
    reliable_short_m = [
        float((c.get("metric_from_depth") or {}).get("short_axis_m_pinhole"))
        for c in reliable_components
        if (c.get("metric_from_depth") or {}).get("short_axis_m_pinhole") is not None
    ]
    reliable_by_camera: dict[str, int] = {}
    reliable_by_side: dict[str, int] = {}
    for component in reliable_components:
        camera = str(component.get("camera") or "unknown")
        side = str((component.get("assigned_side_from_projected_ee") or {}).get("assigned_side") or "unknown")
        reliable_by_camera[camera] = reliable_by_camera.get(camera, 0) + 1
        reliable_by_side[side] = reliable_by_side.get(side, 0) + 1

    def _component_digest(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "camera": c.get("camera"),
            "assigned_side": (c.get("assigned_side_from_projected_ee") or {}).get("assigned_side"),
            "area_px": c.get("area_px"),
            "bbox_xywh_px": c.get("bbox_xywh_px"),
            "long_axis_px": c.get("min_area_rect", {}).get("long_axis_px"),
            "short_axis_px": c.get("min_area_rect", {}).get("short_axis_px"),
            "median_depth_m": c.get("median_depth_m"),
            "metric_from_depth_m": c.get("metric_from_depth"),
            "pca_extents_m_p05_p95": (c.get("world_pca") or {}).get("pca_extents_m_p05_p95"),
            "relation_to_assigned_ee": c.get("relation_to_assigned_ee"),
            "same_side_or_contamination": c.get("same_side_or_contamination"),
            "visual_shape": c.get("visual_shape"),
        }

    return {
        "component_total": len(all_components),
        "reliable_component_total": len(reliable_components),
        "reliable_by_camera": reliable_by_camera,
        "reliable_by_side": reliable_by_side,
        "long_axis_px_range": _round_list([min(long_axes_px), max(long_axes_px)], 2, 2) if long_axes_px else [],
        "world_primary_extent_m_range": _round_list([min(primary_extents), max(primary_extents)], 4, 2)
        if primary_extents
        else [],
        "reliable_visible_long_axis_m_range": _round_list([min(reliable_long_m), max(reliable_long_m)], 4, 2)
        if reliable_long_m
        else [],
        "reliable_visible_short_axis_m_range": _round_list([min(reliable_short_m), max(reliable_short_m)], 4, 2)
        if reliable_short_m
        else [],
        "reliable_components": sorted(
            [_component_digest(c) for c in reliable_components],
            key=lambda item: int(item.get("area_px") or 0),
            reverse=True,
        )[:20],
        "largest_components": sorted(
            [_component_digest(c) for c in all_components],
            key=lambda item: int(item.get("area_px") or 0),
            reverse=True,
        )[:12],
        "width_sweep_support": {
            "current_run": "single live no-motion RGB-D snapshot only",
            "existing_calibration_log": "logs/calibrate_gripper_geometry_top_camera_20260529T130455",
            "limitation": "that calibration recorded videos/detections but not raw RGB-D per width step, so command-to-gap metrics for 0.68/0.54/0.40 are not supported by depth data yet",
            "next_required_artifact": "save raw RGB/depth/intrinsics/extrinsics for each physical calibration stage",
        },
        "interpretation": [
            "This is RGB-D geometry from blue color segmentation, not SAM3.",
            "World extents are point-cloud extents of visible blue pixels only; occluded/non-blue black gripper body is not measured.",
            "Reliable components are same-side top-camera components near projected EE or near-field wrist-camera blue components; other blue components are diagnostic/possible contamination.",
            "Pinhole metric extents from component median depth are usually more stable than full point-cloud PCA at depth discontinuities.",
            "Use these keypoints to check fingertip visibility and height; use the XML/contact model for full invisible contact geometry.",
        ],
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result: dict[str, Any] = {
        "schema": "openforge.gripper_rgbd_geometry.v1",
        "success": False,
        "run_dir": str(run_dir),
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "why_stopped": "not_started",
        "camera_packets": [],
        "summary": None,
        "robot_state": None,
    }
    if _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        result["why_stopped"] = "OPENFORGE_ALLOW_PHYSICAL_MOTION is set; refusing because this is a no-motion RGB-D analysis"
        return result
    get_robot_state = _tool("get_robot_state")
    if callable(get_robot_state):
        try:
            result["robot_state"] = json_safe(get_robot_state())
        except Exception as exc:
            result["robot_state_error"] = f"{type(exc).__name__}: {exc}"
    cameras = [cam for cam in _csv("OPENFORGE_GRIPPER_RGBD_CAMERAS", ["top", "left", "right"]) if cam in {"top", "left", "right"}]
    if not cameras:
        cameras = ["top"]
    api_surface = _runtime_api_surface(cameras)
    result["runtime_api_surface"] = api_surface
    result["runtime_api_surface_path"] = write_json(run_dir / "runtime_api_surface.json", api_surface)
    packets: list[dict[str, Any]] = []
    errors: list[str] = []
    for camera in cameras:
        try:
            packets.append(_analyze_camera(camera, run_dir, result["robot_state"]))
        except Exception as exc:
            errors.append(f"{camera}: {type(exc).__name__}: {exc}")
    result["camera_packets"] = packets
    result["errors"] = errors
    result["summary"] = _summarize_components(packets)
    result["success"] = bool(packets and not errors)
    result["why_stopped"] = "completed RGB-D gripper geometry analysis" if result["success"] else "RGB-D analysis incomplete"
    append_stage_summary(
        run_dir,
        [
            "## RGB-D gripper geometry analysis",
            f"- cameras: {cameras}",
            f"- success: {result['success']}",
            f"- errors: {errors}",
            f"- runtime_api_surface_path: {result.get('runtime_api_surface_path')}",
            f"- component_total: {(result['summary'] or {}).get('component_total')}",
            f"- reliable_component_total: {(result['summary'] or {}).get('reliable_component_total')}",
            f"- reliable_visible_long_axis_m_range: {(result['summary'] or {}).get('reliable_visible_long_axis_m_range')}",
            f"- reliable_visible_short_axis_m_range: {(result['summary'] or {}).get('reliable_visible_short_axis_m_range')}",
            f"- reliable_components: {(result['summary'] or {}).get('reliable_components')}",
            f"- largest_components: {(result['summary'] or {}).get('largest_components')}",
            "- physical_motion_executed: False",
        ],
    )
    return result


try:
    TASK_RESULT.update(_run())
except Exception as exc:
    TASK_RESULT.update(
        {
            "schema": "openforge.gripper_rgbd_geometry.v1",
            "success": False,
            "physical_motion_executed": False,
            "movement_capable_calls": [],
            "why_stopped": f"{type(exc).__name__}: {exc}",
        }
    )
finally:
    run_dir = current_run_dir(TASK_NAME)
    TASK_RESULT.setdefault("run_dir", str(run_dir))
    write_json(Path(run_dir) / "gripper_geometry_result.json", TASK_RESULT)
    write_json(Path(run_dir) / "result.json", TASK_RESULT)
    write_json(Path(run_dir) / "task_result.json", TASK_RESULT)
    append_stage_summary(
        run_dir,
        [
            "## result",
            f"- success: {TASK_RESULT.get('success')}",
            f"- physical_motion_executed: {TASK_RESULT.get('physical_motion_executed')}",
            f"- movement_capable_calls: {TASK_RESULT.get('movement_capable_calls')}",
            f"- why_stopped: {TASK_RESULT.get('why_stopped')}",
        ],
    )
    print(f"[{TASK_NAME}] success={TASK_RESULT.get('success')} why={TASK_RESULT.get('why_stopped')}")
