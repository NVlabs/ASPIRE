"""Recorded no-motion geometry analysis for two dish/plate candidates on YAM.

This script does not move hardware. It captures fresh live detections and
RGB-D, groups duplicate prompt detections into dish candidates, estimates basic
metric geometry from top-camera depth/calibration, and writes artifacts that a
later rim-pinch planner can use. It refuses to run if the physical gate or
manual object XYZ overrides are present.
"""

from __future__ import annotations

import inspect
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    write_json,
)


TASK_NAME = "analyze_two_dishes_no_motion"
TASK_RESULT: dict[str, Any] = {}
DEFAULT_DISH_PROMPTS = [
    "dish",
    "plate",
    "dinner plate",
    "white plate",
    "white dish",
    "round white dish",
    "bowl",
    "white bowl",
    "shallow bowl",
    "saucer",
]


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_PLATE_XYZ",
        "OPENFORGE_DISH_XYZ",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_DISH_GEOMETRY_RECORDING_WAIT_S", 4.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    evidence: dict[str, Any] = {"required": True, "passed": bool(videos), "video_paths": videos}
    if not videos:
        evidence.update(
            {
                "problem": (
                    "no mp4 files found; live YAM probes must run with recording.enabled=true "
                    "and debug_ui.enabled=true"
                ),
                "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
                "blocker_before_physical": True,
            }
        )
    return evidence


def _runtime_api_surface(cameras: list[str]) -> dict[str, Any]:
    expected = [
        "detect_objects_oneshot",
        "get_camera_image",
        "get_robot_state",
        "run_in_background",
        "freespace_move",
        "set_gripper",
        "open_gripper",
        "close_gripper",
        "go_home",
    ]
    tools: dict[str, Any] = {}
    for name in expected:
        value = globals().get(name)
        item: dict[str, Any] = {
            "injected": callable(value),
            "repr": repr(value)[:240] if value is not None else None,
            "motion_capable": name in {"set_gripper", "open_gripper", "close_gripper", "go_home"},
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
        "schema": "openforge.two_dish_geometry.runtime_api_surface.v1",
        "source": "actual run_script globals plus optional camera portal probe",
        "tools": tools,
        "camera_portal": portal_packet,
        "motion_capable_tools_not_called": ["freespace_move", "set_gripper", "open_gripper", "close_gripper", "go_home"],
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


def _unproject_pixel(u: float, v: float, depth_m: float, intrinsics: list[float], cam_to_world: np.ndarray) -> list[float] | None:
    if not math.isfinite(depth_m) or depth_m <= 0:
        return None
    fx, fy, cx, cy = [float(x) for x in intrinsics[:4]]
    pc = np.asarray([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m, 1.0], dtype=np.float64)
    pw = cam_to_world @ pc
    return [float(pw[0]), float(pw[1]), float(pw[2])]


def _static_top_camera_extrinsics() -> dict[str, Any]:
    from robot.models.station.paths import get_top_camera_frame, needs_optical_flip
    from robot.yam.kinematics import YamKinematics

    frame_name = os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame())
    kin = YamKinematics()
    transform = kin.configuration.get_transform_frame_to_world(frame_name, "body")
    rotation = transform.rotation()
    rot_mat = rotation.as_matrix() if hasattr(rotation, "as_matrix") else rotation.matrix()
    return {
        "position": [float(x) for x in transform.translation().tolist()],
        "rotation": np.asarray(rot_mat, dtype=np.float64).reshape(3, 3).tolist(),
        "needs_optical_flip": needs_optical_flip("top"),
        "source": "station_model_static_top_camera",
    }


def _capture_portal_rgbd(camera: str) -> dict[str, Any]:
    import portal

    addr = os.environ.get("OPENFORGE_CAMERA_PORTAL_ADDR", "127.0.0.1:8300").strip()
    client = portal.Client(addr)
    rgb = np.asarray(client.get_camera_image(camera).result())
    depth = np.asarray(client.get_camera_depth(camera).result(), dtype=np.float32)
    intrinsics = [float(x) for x in client.get_camera_intrinsics(camera).result()]
    try:
        extrinsics = dict(client.get_camera_extrinsics(camera).result())
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


def _save_rgbd_artifacts(run_dir: Path, cameras: list[str]) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.two_dish_geometry.rgbd.v1",
        "captures": {},
        "errors": [],
    }
    for camera in cameras:
        try:
            capture = _capture_portal_rgbd(camera)
            out_dir = run_dir / "observations" / f"rgbd_{camera}"
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(capture["rgb"]).save(out_dir / "rgb.png")
            np.save(out_dir / "depth_m.npy", capture["depth"])
            valid = capture["depth"][np.isfinite(capture["depth"]) & (capture["depth"] > 0)]
            depth_summary: dict[str, Any] = {"valid_count": int(valid.size)}
            if valid.size:
                depth_summary.update(
                    {
                        "min_m": round(float(np.nanmin(valid)), 5),
                        "p05_m": round(float(np.nanpercentile(valid, 5)), 5),
                        "median_m": round(float(np.nanmedian(valid)), 5),
                        "p95_m": round(float(np.nanpercentile(valid, 95)), 5),
                        "max_m": round(float(np.nanmax(valid)), 5),
                    }
                )
                depth_clip = np.clip(capture["depth"], depth_summary["p05_m"], depth_summary["p95_m"])
                denom = max(1e-6, depth_summary["p95_m"] - depth_summary["p05_m"])
                preview = (255.0 * (depth_clip - depth_summary["p05_m"]) / denom).astype(np.uint8)
                Image.fromarray(preview).save(out_dir / "depth_preview.png")
            meta = {
                "camera": camera,
                "portal_addr": capture["portal_addr"],
                "rgb_shape": list(capture["rgb"].shape),
                "depth_shape": list(capture["depth"].shape),
                "intrinsics": capture["intrinsics"],
                "extrinsics": json_safe(capture["extrinsics"]),
                "depth_summary": depth_summary,
                "rgb_path": str(out_dir / "rgb.png"),
                "depth_path": str(out_dir / "depth_m.npy"),
                "depth_preview_path": str(out_dir / "depth_preview.png"),
            }
            write_json(out_dir / "metadata.json", meta)
            packet["captures"][camera] = {**meta, "_runtime_capture": capture}
        except Exception as exc:
            packet["errors"].append(f"{camera}: {type(exc).__name__}: {exc}")
    safe_packet = {
        "schema": packet["schema"],
        "captures": {camera: {k: v for k, v in data.items() if k != "_runtime_capture"} for camera, data in packet["captures"].items()},
        "errors": packet["errors"],
    }
    safe_packet["path"] = write_json(run_dir / "observations" / "rgbd_capture_summary.json", safe_packet)
    return packet


def _box_center_px(box: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return None
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def _box_size_px(box: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return None
    return [max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _detection_touches_image_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_TWO_DISH_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_TWO_DISH_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_TWO_DISH_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _xy_distance(a: Any, b: Any) -> float | None:
    aa = _round_list(a, digits=8, limit=3)
    bb = _round_list(b, digits=8, limit=3)
    if len(aa) < 2 or len(bb) < 2:
        return None
    return float(math.hypot(aa[0] - bb[0], aa[1] - bb[1]))


def _eligible_detections(packet: dict[str, Any]) -> list[dict[str, Any]]:
    min_score = _env_float("OPENFORGE_TWO_DISH_MIN_SCORE", 0.20)
    rows: list[dict[str, Any]] = []
    for det in packet.get("all_detections") or []:
        item = dict(det)
        camera = str(item.get("source_camera") or item.get("camera") or "")
        score = float(item.get("score") or 0.0)
        edge = _detection_touches_image_edge(item)
        item["touches_image_edge"] = edge
        item["box_center_px"] = _box_center_px(item.get("box_2d") or [])
        item["box_size_px"] = _box_size_px(item.get("box_2d") or [])
        item["eligible_for_geometry"] = bool(score >= min_score and camera == "top" and not edge)
        rows.append(item)
    return rows


def _cluster_top_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    threshold_m = _env_float("OPENFORGE_TWO_DISH_CLUSTER_XY_M", 0.075)
    threshold_px = _env_float("OPENFORGE_TWO_DISH_CLUSTER_CENTER_PX", 48.0)
    candidates = [det for det in detections if det.get("eligible_for_geometry")]
    prompt_priority = {prompt: idx for idx, prompt in enumerate(_csv("OPENFORGE_TWO_DISH_PROMPTS", DEFAULT_DISH_PROMPTS))}

    def _sort_key(det: dict[str, Any]) -> tuple[int, float]:
        return (prompt_priority.get(str(det.get("prompt")), 99), -float(det.get("score") or 0.0))

    clusters: list[dict[str, Any]] = []
    for det in sorted(candidates, key=_sort_key):
        xyz = det.get("position_3d") or []
        center_px = det.get("box_center_px") or []
        matched: dict[str, Any] | None = None
        for cluster in clusters:
            rep = cluster["representative"]
            xy = _xy_distance(xyz, rep.get("position_3d") or [])
            px = None
            rep_px = rep.get("box_center_px") or []
            if len(center_px) >= 2 and len(rep_px) >= 2:
                px = math.hypot(float(center_px[0]) - float(rep_px[0]), float(center_px[1]) - float(rep_px[1]))
            if (xy is not None and xy <= threshold_m) or (px is not None and px <= threshold_px):
                matched = cluster
                break
        if matched is None:
            clusters.append({"representative": det, "members": [det]})
        else:
            matched["members"].append(det)
            if _sort_key(det) < _sort_key(matched["representative"]):
                matched["representative"] = det

    for idx, cluster in enumerate(clusters):
        cluster["cluster_id"] = f"dish_{idx + 1}"
        cluster["member_count"] = len(cluster["members"])
        cluster["member_prompts"] = sorted({str(det.get("prompt")) for det in cluster["members"]})
    return clusters


def _safe_depth_values(depth: np.ndarray, box: Any, margin_frac: float = 0.0) -> np.ndarray:
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in list(box)[:4]]
    except Exception:
        return np.asarray([], dtype=np.float32)
    h, w = depth.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return np.asarray([], dtype=np.float32)
    dx = int(round((x1 - x0) * margin_frac))
    dy = int(round((y1 - y0) * margin_frac))
    x0 += dx
    x1 -= dx
    y0 += dy
    y1 -= dy
    if x1 <= x0 or y1 <= y0:
        return np.asarray([], dtype=np.float32)
    crop = depth[y0:y1, x0:x1]
    return crop[np.isfinite(crop) & (crop > 0)]


def _world_z_stats_for_box(depth: np.ndarray, intr: list[float], cam_to_world: np.ndarray, box: Any, mode: str) -> dict[str, Any]:
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in list(box)[:4]]
    except Exception:
        return {"available": False, "reason": "invalid box"}
    h, w = depth.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return {"available": False, "reason": "empty box"}
    crop = depth[y0:y1, x0:x1]
    if crop.size == 0:
        return {"available": False, "reason": "empty crop"}
    yy, xx = np.indices(crop.shape)
    if mode == "center":
        cx = crop.shape[1] / 2.0
        cy = crop.shape[0] / 2.0
        rx = max(2.0, crop.shape[1] * 0.22)
        ry = max(2.0, crop.shape[0] * 0.22)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    elif mode == "rim_band":
        nx = (xx + 0.5) / max(1.0, float(crop.shape[1]))
        ny = (yy + 0.5) / max(1.0, float(crop.shape[0]))
        edge_dist = np.minimum.reduce([nx, 1.0 - nx, ny, 1.0 - ny])
        mask = edge_dist <= 0.18
    else:
        mask = np.ones(crop.shape, dtype=bool)
    local_depths = crop[mask]
    local_ys = yy[mask] + y0
    local_xs = xx[mask] + x0
    valid = np.isfinite(local_depths) & (local_depths > 0)
    if int(valid.sum()) < 20:
        return {"available": False, "reason": f"too few valid depth pixels for {mode}", "valid_count": int(valid.sum())}
    # Downsample deterministically for speed.
    idx = np.flatnonzero(valid)
    if idx.size > 3500:
        idx = idx[:: max(1, idx.size // 3500)]
    zs: list[float] = []
    for i in idx:
        point = _unproject_pixel(float(local_xs[i]), float(local_ys[i]), float(local_depths[i]), intr, cam_to_world)
        if point is not None and math.isfinite(point[2]):
            zs.append(float(point[2]))
    if not zs:
        return {"available": False, "reason": f"no valid world z for {mode}"}
    arr = np.asarray(zs, dtype=np.float64)
    return {
        "available": True,
        "mode": mode,
        "sample_count": int(arr.size),
        "z_min_m": round(float(np.nanmin(arr)), 5),
        "z_p05_m": round(float(np.nanpercentile(arr, 5)), 5),
        "z_median_m": round(float(np.nanmedian(arr)), 5),
        "z_p95_m": round(float(np.nanpercentile(arr, 95)), 5),
        "z_max_m": round(float(np.nanmax(arr)), 5),
    }


def _metric_geometry_from_top(detection: dict[str, Any], rgbd_packet: dict[str, Any]) -> dict[str, Any]:
    capture_entry = (rgbd_packet.get("captures") or {}).get("top") or {}
    capture = capture_entry.get("_runtime_capture")
    if not capture:
        return {"available": False, "reason": "top RGB-D unavailable"}
    depth = capture["depth"]
    intr = capture["intrinsics"]
    extr = capture["extrinsics"]
    cam_to_world = _camera_to_world_from_extrinsics(extr)
    box = detection.get("box_2d") or []
    depths = _safe_depth_values(depth, box, margin_frac=0.05)
    if depths.size < 20:
        return {"available": False, "reason": "too few valid depth pixels in detection box", "valid_depth_count": int(depths.size)}
    median_depth = float(np.nanmedian(depths))
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return {"available": False, "reason": "invalid detection box"}
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    left = _unproject_pixel(x0, mid_y, median_depth, intr, cam_to_world)
    right = _unproject_pixel(x1, mid_y, median_depth, intr, cam_to_world)
    top = _unproject_pixel(mid_x, y0, median_depth, intr, cam_to_world)
    bottom = _unproject_pixel(mid_x, y1, median_depth, intr, cam_to_world)

    def _dist(a: list[float] | None, b: list[float] | None) -> float | None:
        if a is None or b is None:
            return None
        return float(math.dist(a, b))

    width_m = _dist(left, right)
    height_m = _dist(top, bottom)
    diameter_estimates = [value for value in (width_m, height_m) if value and math.isfinite(value)]
    radius_m = (sum(diameter_estimates) / len(diameter_estimates) / 2.0) if diameter_estimates else None
    center_stats = _world_z_stats_for_box(depth, intr, cam_to_world, box, "center")
    rim_stats = _world_z_stats_for_box(depth, intr, cam_to_world, box, "rim_band")
    rim_minus_center = None
    if center_stats.get("available") and rim_stats.get("available"):
        rim_minus_center = float(rim_stats["z_median_m"]) - float(center_stats["z_median_m"])
    return {
        "available": True,
        "source": "top RGB-D median-depth unprojection plus bbox zones",
        "uncertainty": (
            "bbox-based metric scale only; rim/center height uses all pixels in the box, "
            "so table pixels can bias thin-dish estimates"
        ),
        "median_depth_m": round(median_depth, 5),
        "valid_depth_count": int(depths.size),
        "bbox_metric_width_m": round(width_m, 5) if width_m is not None else None,
        "bbox_metric_height_m": round(height_m, 5) if height_m is not None else None,
        "estimated_radius_m": round(radius_m, 5) if radius_m is not None else None,
        "diameter_estimate_m": round(2.0 * radius_m, 5) if radius_m is not None else None,
        "bbox_midpoints_world_m": {
            "left": _round_list(left, limit=3) if left else None,
            "right": _round_list(right, limit=3) if right else None,
            "top": _round_list(top, limit=3) if top else None,
            "bottom": _round_list(bottom, limit=3) if bottom else None,
        },
        "center_patch_world_z": center_stats,
        "rim_band_world_z": rim_stats,
        "rim_minus_center_z_m": round(rim_minus_center, 5) if rim_minus_center is not None else None,
    }


def _robot_state_summary(robot_state: dict[str, Any] | None) -> dict[str, Any]:
    arms = (robot_state or {}).get("arms") if isinstance(robot_state, dict) else None
    summary: dict[str, Any] = {}
    for side in ("left", "right"):
        arm = (arms or {}).get(side) if isinstance(arms, dict) else None
        if not isinstance(arm, dict):
            continue
        summary[side] = {
            "ee_pos": _round_list(arm.get("ee_pos") or [], limit=3),
            "ee_rpy": _round_list(arm.get("ee_rpy") or [], limit=3),
            "gripper_pos": arm.get("gripper_pos"),
        }
    return summary


def _proximity_to_arms(detection: dict[str, Any], robot_state: dict[str, Any] | None) -> dict[str, Any]:
    xyz = detection.get("position_3d") or []
    arms = (robot_state or {}).get("arms") if isinstance(robot_state, dict) else None
    rows: dict[str, Any] = {}
    for side in ("left", "right"):
        arm = (arms or {}).get(side) if isinstance(arms, dict) else None
        ee = (arm or {}).get("ee_pos") if isinstance(arm, dict) else None
        if not ee or len(ee) < 3 or len(xyz) < 3:
            continue
        dx = float(ee[0]) - float(xyz[0])
        dy = float(ee[1]) - float(xyz[1])
        dz = float(ee[2]) - float(xyz[2])
        rows[side] = {
            "xy_distance_m": round(math.hypot(dx, dy), 5),
            "z_delta_m": round(dz, 5),
            "close_to_dish": bool(math.hypot(dx, dy) < 0.14 and abs(dz) < 0.12),
        }
    return rows


def _rim_option_geometry(center: list[float], radius_m: float, other_centers: list[list[float]]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    approach_clearance = _env_float("OPENFORGE_TWO_DISH_OTHER_DISH_CLEARANCE_M", 0.045)
    for angle_deg in [90.0, -90.0, 0.0, 180.0, 45.0, 135.0, -45.0, -135.0]:
        theta = math.radians(angle_deg)
        radial = [math.cos(theta), math.sin(theta)]
        rim_point = [center[0] + radial[0] * radius_m, center[1] + radial[1] * radius_m, center[2]]
        approach_start = [
            rim_point[0] + radial[0] * _env_float("OPENFORGE_TWO_DISH_APPROACH_STANDOFF_M", 0.09),
            rim_point[1] + radial[1] * _env_float("OPENFORGE_TWO_DISH_APPROACH_STANDOFF_M", 0.09),
            center[2],
        ]
        min_other = None
        for other in other_centers:
            dist = math.hypot(approach_start[0] - other[0], approach_start[1] - other[1])
            min_other = dist if min_other is None else min(min_other, dist)
        bounds = {
            "x_ok": 0.28 <= rim_point[0] <= 1.10 and 0.28 <= approach_start[0] <= 1.10,
            "y_ok": -0.66 <= rim_point[1] <= 0.66 and -0.66 <= approach_start[1] <= 0.66,
        }
        blocked_by_other = min_other is not None and min_other < radius_m + approach_clearance
        options.append(
            {
                "rim_angle_deg": angle_deg,
                "rim_point_m": _round_list(rim_point, limit=3),
                "approach_start_m": _round_list(approach_start, limit=3),
                "suggested_arm": "left" if rim_point[1] >= 0 else "right",
                "min_other_dish_distance_from_approach_m": round(min_other, 5) if min_other is not None else None,
                "workspace_bounds_ok": bool(bounds["x_ok"] and bounds["y_ok"]),
                "blocked_by_other_dish": bool(blocked_by_other),
            }
        )
    return options


def _ease_score(dish: dict[str, Any], all_centers: list[list[float]]) -> dict[str, Any]:
    detection = dish["selected_detection"]
    geom = dish.get("metric_geometry") or {}
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    if detection.get("source_camera") == "top":
        score += 1.0
        reasons.append("top-camera detection")
    if not detection.get("touches_image_edge"):
        score += 0.7
        reasons.append("not touching image edge")
    det_score = float(detection.get("score") or 0.0)
    score += min(1.0, max(0.0, det_score))
    radius = geom.get("estimated_radius_m")
    if isinstance(radius, (int, float)) and 0.05 <= float(radius) <= 0.14:
        score += 0.7
        reasons.append("radius estimate in useful dish range")
    else:
        warnings.append(f"radius estimate uncertain or outside expected range: {radius}")
    xyz = detection.get("position_3d") or []
    if len(xyz) >= 3:
        if 0.35 <= float(xyz[0]) <= 0.95 and -0.45 <= float(xyz[1]) <= 0.45 and 0.68 <= float(xyz[2]) <= 0.86:
            score += 0.7
            reasons.append("inside conservative plate workspace")
        other_distances = [
            math.hypot(float(xyz[0]) - other[0], float(xyz[1]) - other[1])
            for other in all_centers
            if math.hypot(float(xyz[0]) - other[0], float(xyz[1]) - other[1]) > 1e-5
        ]
        if other_distances:
            nearest = min(other_distances)
            radius_value = float(radius) if isinstance(radius, (int, float)) else 0.09
            clearance = nearest - 2.0 * radius_value
            dish["nearest_other_dish_xy_m"] = round(nearest, 5)
            dish["estimated_between_dish_edge_clearance_m"] = round(clearance, 5)
            if clearance > 0.04:
                score += 0.5
                reasons.append("separated from the other dish")
            else:
                warnings.append(f"close to other dish; estimated edge clearance {clearance:.3f} m")
    arm_proximity = dish.get("arm_proximity") or {}
    if any((item or {}).get("close_to_dish") for item in arm_proximity.values()):
        score -= 0.8
        warnings.append("an end effector is close to this dish; recovery/home may be needed before planning")
    return {"ease_score": round(score, 3), "reasons": reasons, "warnings": warnings}


def _build_dish_packets(
    clusters: list[dict[str, Any]],
    rgbd_packet: dict[str, Any],
    robot_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    dishes: list[dict[str, Any]] = []
    for idx, cluster in enumerate(clusters):
        selected = dict(cluster["representative"])
        metric = _metric_geometry_from_top(selected, rgbd_packet)
        radius = metric.get("estimated_radius_m")
        if not isinstance(radius, (int, float)) or float(radius) <= 0:
            radius = _env_float("OPENFORGE_TWO_DISH_DEFAULT_RADIUS_M", 0.09)
        packet = {
            "dish_id": f"dish_{idx + 1}",
            "selected_detection": selected,
            "member_count": cluster.get("member_count"),
            "member_prompts": cluster.get("member_prompts"),
            "members": cluster.get("members"),
            "metric_geometry": metric,
            "arm_proximity": _proximity_to_arms(selected, robot_state),
        }
        dishes.append(packet)
    centers = [
        [float(v) for v in (dish["selected_detection"].get("position_3d") or [])[:3]]
        for dish in dishes
        if len(dish["selected_detection"].get("position_3d") or []) >= 3
    ]
    for dish in dishes:
        center = [float(v) for v in (dish["selected_detection"].get("position_3d") or [])[:3]]
        radius = dish["metric_geometry"].get("estimated_radius_m")
        if not isinstance(radius, (int, float)) or float(radius) <= 0:
            radius = _env_float("OPENFORGE_TWO_DISH_DEFAULT_RADIUS_M", 0.09)
        other_centers = [other for other in centers if math.hypot(center[0] - other[0], center[1] - other[1]) > 1e-5]
        dish["rim_grasp_geometry_options"] = _rim_option_geometry(center, float(radius), other_centers) if len(center) >= 3 else []
        dish.update(_ease_score(dish, centers))
    return sorted(dishes, key=lambda item: float(item.get("ease_score") or 0.0), reverse=True)


def _write_top_overlay(run_dir: Path, rgbd_packet: dict[str, Any], dishes: list[dict[str, Any]], all_detections: list[dict[str, Any]]) -> str | None:
    capture = ((rgbd_packet.get("captures") or {}).get("top") or {}).get("_runtime_capture")
    if not capture:
        return None
    img = Image.fromarray(capture["rgb"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), "two-dish geometry: top detections", fill=(0, 255, 0))
    for det in all_detections:
        if det.get("source_camera") != "top":
            continue
        box = det.get("box_2d") or []
        if len(box) >= 4:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
            color = (255, 160, 40) if det.get("eligible_for_geometry") else (180, 80, 80)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            label = f"{det.get('prompt')} {float(det.get('score') or 0):.2f}"
            draw.text((x0, max(18, y0 - 14)), label, fill=color)
    for dish in dishes:
        det = dish["selected_detection"]
        box = det.get("box_2d") or []
        if len(box) >= 4:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 80), width=4)
            label = (
                f"{dish['dish_id']} ease={dish.get('ease_score')} "
                f"r={dish.get('metric_geometry', {}).get('estimated_radius_m')}"
            )
            draw.text((x0, min(img.height - 18, y1 + 4)), label, fill=(0, 255, 80))
    path = run_dir / "vis" / "two_dish_geometry_top_overlay.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.two_dish_geometry.result.v1",
        "task_name": TASK_NAME,
        "run_dir": str(run_dir),
        "success": False,
        "confidence": 0.0,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "why_stopped": None,
        "next_recommendation": None,
        "video_paths": [],
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        if _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
            result["why_stopped"] = "refusing: OPENFORGE_ALLOW_PHYSICAL_MOTION is set but this script is no-motion only"
            result["next_recommendation"] = "Unset the physical gate and run geometry analysis with live observation only."
            return result
        manual_vars = _manual_pose_env_vars_present()
        if manual_vars:
            result["why_stopped"] = f"refusing manual pose env vars for live dish geometry: {manual_vars}"
            result["next_recommendation"] = "Use fresh live detections only; do not provide hardcoded dish/plate XYZ."
            return result

        cameras = [cam for cam in _csv("OPENFORGE_TWO_DISH_CAMERAS", ["top", "left", "right"]) if cam in {"top", "left", "right"}]
        prompts = _csv("OPENFORGE_TWO_DISH_PROMPTS", DEFAULT_DISH_PROMPTS)
        result["runtime_api_surface"] = _runtime_api_surface(cameras)
        write_json(run_dir / "runtime_api_surface.json", result["runtime_api_surface"])

        scene = capture_scene(
            prompts=prompts,
            cameras=cameras,
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="two_dish_live_observe",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
            max_retries=max(1, _env_int("OPENFORGE_TWO_DISH_DETECT_RETRIES", 2)),
            save_overlays=True,
            motion_cameras=("top", "left", "right"),
        )
        result["pre_observation"] = {
            "packet_path": scene.get("packet_path"),
            "latest_path": scene.get("latest_path"),
            "best_detection": scene.get("best_detection"),
            "errors": scene.get("errors"),
        }
        robot_state = scene.get("robot_state")
        result["robot_state_summary"] = _robot_state_summary(robot_state)

        rgbd_packet = _save_rgbd_artifacts(run_dir, cameras)
        result["rgbd_capture_summary_path"] = str(run_dir / "observations" / "rgbd_capture_summary.json")
        result["rgbd_errors"] = rgbd_packet.get("errors", [])

        detections = _eligible_detections(scene)
        clusters = _cluster_top_detections(detections)
        dishes = _build_dish_packets(clusters, rgbd_packet, robot_state)
        selected = dishes[0] if dishes else None
        overlay_path = _write_top_overlay(run_dir, rgbd_packet, dishes, detections)
        geometry_packet = {
            "schema": "openforge.two_dish_geometry.plan_packet.v1",
            "source": "fresh live detections plus top RGB-D/calibration",
            "prompts": prompts,
            "cameras": cameras,
            "selection_policy": (
                "cluster top-camera non-edge detections across dish/plate prompts; "
                "side-camera detections are diagnostic unless explicitly promoted later"
            ),
            "blue_gripper_geometry_assumption": {
                "visible_blue_contact_length_m": [0.05, 0.06],
                "effective_contact_width_m": [0.016, 0.025],
                "source": ".agents/skills/yam-geometry/references/blue-gripper-geometry.md",
            },
            "all_detections": detections,
            "cluster_count": len(clusters),
            "dishes_ranked": dishes,
            "selected_easiest_dish": selected,
            "top_overlay_path": overlay_path,
            "not_planner_previewed": (
                "This script analyzes object geometry only. Run a fresh rim-pinch planner preview "
                "after selecting which dish to attempt."
            ),
        }
        geometry_packet["path"] = write_json(run_dir / "plans" / "two_dish_geometry_analysis.json", geometry_packet)
        result["geometry_packet_path"] = geometry_packet["path"]
        result["dish_count"] = len(dishes)
        result["selected_easiest_dish"] = selected
        result["candidate_plans"] = {
            "type": "geometry_only_rim_options",
            "path": geometry_packet["path"],
            "planner_preview_executed": False,
        }
        if selected:
            result["success"] = len(dishes) >= 2
            result["confidence"] = 0.65 if len(dishes) >= 2 else 0.35
            result["why_stopped"] = (
                f"fresh no-motion geometry found {len(dishes)} top-camera dish candidate(s); "
                f"best-ranked candidate is {selected.get('dish_id')}"
            )
            result["next_recommendation"] = (
                "Inspect plans/two_dish_geometry_analysis.json and the top overlay. "
                "If the selected dish is coherent, rerun the rim-pinch no-motion planner against fresh live detections "
                "or patch the planner to explicitly choose this ranked dish before any physical ticket."
            )
        else:
            result["why_stopped"] = "no coherent top-camera dish candidates found from fresh live detections"
            result["next_recommendation"] = (
                "Reposition the two dishes so they are visible to the top camera, then rerun this recorded no-motion analysis. "
                "Do not attempt physical pickup from side-camera edge detections."
            )
        append_stage_summary(
            run_dir,
            [
                "## two-dish geometry analysis",
                f"- prompts: {prompts}",
                f"- top eligible detections: {len([d for d in detections if d.get('eligible_for_geometry')])}",
                f"- clustered dish candidates: {len(dishes)}",
                f"- selected_easiest_dish: {(selected or {}).get('dish_id')}",
                f"- geometry_packet: {geometry_packet['path']}",
                f"- top_overlay: {overlay_path}",
                "- planner_preview_executed: false",
            ],
        )
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Fix the no-motion geometry analysis blocker and rerun with recording/debug evidence."
        return result
    finally:
        recording = _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        if not recording.get("passed"):
            result["success"] = False
            result["confidence"] = 0.0
            result["why_stopped"] = f"{result.get('why_stopped')}; recording evidence missing"
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- dish_count: {result.get('dish_count')}",
                f"- video_paths: {result.get('video_paths')}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- next_recommendation: {result.get('next_recommendation')}",
            ],
        )


TASK_RESULT.update(_run())
print(f"[{TASK_NAME}] success={TASK_RESULT.get('success')} why={TASK_RESULT.get('why_stopped')}")
