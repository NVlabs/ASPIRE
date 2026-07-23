# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded dry bottle-to-cup pour rehearsal for YAM.

This is a single-arm script: the left arm grasps/lifts the dry bottle using the
validated bottle side-grasp family, then previews a dry tilt above a stationary
cup.  It does not lift the cup and it never authorizes liquid by default.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    compare_detection_motion,
    current_run_dir,
    execute_grasp_lift_attempt,
    generate_side_grasp_candidates,
    json_safe,
    rank_motion_candidates,
    verify_lift,
    write_json,
)


TASK_NAME = "dry_bottle_to_cup_pour_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _optional_env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            pass
    return values or list(default)


def _optional_float_csv(name: str) -> list[float] | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            pass
    return values or None


def _offset_triples(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [list(item[:3]) for item in default]
    triples: list[list[float]] = []
    for chunk in raw.replace("|", ";").split(";"):
        parts = [part.strip() for part in chunk.replace(":", ",").split(",") if part.strip()]
        if len(parts) < 2:
            continue
        try:
            triples.append([float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) >= 3 else 0.0])
        except ValueError:
            continue
    return triples or [list(item[:3]) for item in default]


def _optional_vector3(name: str) -> list[float] | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    parts = [part.strip() for part in raw.replace(";", ",").replace(":", ",").split(",") if part.strip()]
    if len(parts) < 3:
        raise ValueError(f"{name} must have three comma-separated floats")
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except ValueError as exc:
        raise ValueError(f"{name} must have three comma-separated floats, got {raw!r}") from exc


def _env_vector3(name: str, default: list[float]) -> list[float]:
    value = _optional_vector3(name)
    if value is None:
        return list(default[:3])
    return value


def _round_list(values: Any, digits: int = 5, limit: int = 3) -> list[float]:
    try:
        return [round(float(value), digits) for value in list(values)[:limit]]
    except Exception:
        return []


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _find_env(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any | None:
    if value is None or depth > 6:
        return None
    seen = seen if seen is not None else set()
    ident = id(value)
    if ident in seen:
        return None
    seen.add(ident)

    env = getattr(value, "_env", None)
    if env is not None:
        return env

    env = _find_env(getattr(value, "__wrapped__", None), depth=depth + 1, seen=seen)
    if env is not None:
        return env

    for cell in getattr(value, "__closure__", ()) or ():
        try:
            cell_value = cell.cell_contents
        except ValueError:
            continue
        env = _find_env(cell_value, depth=depth + 1, seen=seen)
        if env is not None:
            return env
    return None


def _tool_env(name: str) -> Any | None:
    return _find_env(_tool(name))


def _pickup_compat_name(name: str) -> str | None:
    prefix = "OPENFORGE_BOTTLE_CUP_DRY_POUR_"
    if not name.startswith(prefix):
        return None
    return "OPENFORGE_GRASP_LIFT_" + name[len(prefix) :]


def _env_raw_pickup(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return raw
    compat = _pickup_compat_name(name)
    if compat is None:
        return None
    raw = os.environ.get(compat)
    if raw is not None and raw.strip():
        return raw
    return None


def _env_str_pickup(name: str, default: str) -> str:
    raw = _env_raw_pickup(name)
    return raw.strip() if raw is not None else default


def _env_float_pickup(name: str, default: float) -> float:
    raw = _env_raw_pickup(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_env_float_pickup(name: str) -> float | None:
    raw = _env_raw_pickup(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        compat = _pickup_compat_name(name)
        names = f"{name}/{compat}" if compat else name
        raise ValueError(f"{names} must be a float, got {raw!r}") from exc


def _float_csv_pickup(name: str, default: list[float]) -> list[float]:
    raw = _env_raw_pickup(name)
    if raw is None:
        return list(default)
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            pass
    return values or list(default)


def _truthy_env_pickup(name: str, default: bool = False) -> bool:
    raw = _env_raw_pickup(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_XYZ",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_XYZ",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_OBJECT_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_BOTTLE_XYZ",
        "OPENFORGE_CUP_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _xyz(detection_or_pose: dict[str, Any] | None) -> list[float] | None:
    if not detection_or_pose:
        return None
    xyz = detection_or_pose.get("position_3d") or detection_or_pose.get("position")
    if not xyz or len(xyz) < 3:
        return None
    try:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except (TypeError, ValueError):
        return None


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists():
            paths.append(str(path))
    return paths


def _capture_prompt_fallback(
    *,
    run_dir: Path,
    stage: str,
    role: str,
    prompts: list[str],
    cameras: list[str],
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    packets: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prompt_index, prompt in enumerate(prompts):
        packet = capture_scene(
            prompts=[prompt],
            cameras=cameras,
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage=f"{stage}_{role}_{prompt.replace(' ', '_')}",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
            max_retries=max(1, _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_DETECT_RETRIES", 2)),
            save_overlays=True,
            motion_cameras=("top",),
            image_only_cameras=tuple(camera for camera in cameras if camera != "top"),
        )
        packets.append(packet)
        detection = packet.get("best_detection")
        score = float((detection or {}).get("score") or 0.0)
        if detection is not None and score >= min_score:
            best = dict(detection)
            best["role"] = role
            best["prompt_index"] = prompt_index
            best["prompt_fallback_used"] = bool(prompt_index > 0)
            raw_source = best.get("source")
            live_source = "live_top_camera" if best.get("source_camera") == "top" else "live_perception"
            best["raw_detection_source"] = raw_source
            best["source"] = live_source
            best["detection_source"] = live_source
            break
    combined = {
        "stage": stage,
        "role": role,
        "source": "live_perception",
        "prompts_tried": prompts[: len(packets)],
        "packets": [
            {
                "packet_path": packet.get("packet_path"),
                "best_detection": packet.get("best_detection"),
                "errors": packet.get("errors"),
            }
            for packet in packets
        ],
        "best_detection": best,
    }
    combined["path"] = write_json(run_dir / "observations" / f"{stage}_{role}_prompt_fallback.json", combined)
    return combined, best


def _validate_detection(detection: dict[str, Any] | None, *, role: str) -> dict[str, Any]:
    prefix = f"OPENFORGE_BOTTLE_CUP_DRY_POUR_{role.upper()}"
    min_score = _env_float(f"{prefix}_MIN_DETECTION_SCORE", _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MIN_DETECTION_SCORE", 0.25))
    enforce_sanity = _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_ENFORCE_DETECTION_SANITY", True)
    result: dict[str, Any] = {
        "role": role,
        "passed": False,
        "enforced": enforce_sanity,
        "min_score": min_score,
        "problems": [],
        "warnings": [],
    }
    if detection is None:
        result["problems"].append("missing detection")
        return result
    score = float(detection.get("score") or 0.0)
    result["score"] = score
    if score < min_score:
        key = "problems" if enforce_sanity else "warnings"
        result[key].append(f"score {score:.3f} below minimum {min_score:.3f}")
    camera = str(detection.get("source_camera") or detection.get("camera") or "")
    result["source_camera"] = camera
    if _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_TOP_CAMERA", True) and camera != "top":
        key = "problems" if enforce_sanity else "warnings"
        result[key].append(f"source camera {camera!r} is not top")
    xyz = _xyz(detection)
    result["position_3d"] = _round_list(xyz or [])
    if xyz is None:
        result["problems"].append("missing 3D position")
    else:
        bounds = {
            "x": (
                _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_X_MIN_M", 0.40),
                _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_X_MAX_M", 0.95),
            ),
            "y": (
                _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_Y_MIN_M", -0.40),
                _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_Y_MAX_M", 0.40),
            ),
            "z": (
                _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_Z_MIN_M", 0.70),
                _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_Z_MAX_M", 1.20),
            ),
        }
        result["workspace_bounds_m"] = bounds
        for axis, value in zip(("x", "y", "z"), xyz):
            lo, hi = bounds[axis]
            if value < lo or value > hi:
                key = "problems" if enforce_sanity else "warnings"
                result[key].append(f"{axis}={value:.3f} outside [{lo:.3f}, {hi:.3f}]")
    result["passed"] = not result["problems"]
    return result


def _xy_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    ax = _xyz(a)
    bx = _xyz(b)
    if ax is None or bx is None:
        return None
    return math.hypot(ax[0] - bx[0], ax[1] - bx[1])


def _apply_bottle_pregrasp_clearance(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clearance = _optional_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_PREGRASP_CLEARANCE_M")
    xy_bias = _float_csv_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_PREGRASP_XY_BIAS_M", [0.0, 0.0])
    bias_x = float(xy_bias[0]) if len(xy_bias) >= 1 else 0.0
    bias_y = float(xy_bias[1]) if len(xy_bias) >= 2 else 0.0
    if clearance is None and abs(bias_x) < 1e-12 and abs(bias_y) < 1e-12:
        return candidates
    if clearance is not None and clearance < 0.0:
        raise ValueError("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_PREGRASP_CLEARANCE_M must be nonnegative")
    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        cand = dict(candidate)
        object_center = cand.get("object_center")
        approach = cand.get("approach_direction_world")
        grasp_pose = dict(cand.get("grasp_pose") or {})
        grasp_pos = grasp_pose.get("position")
        pre_pose = dict(cand.get("pregrasp_pose") or {})
        if not object_center or not approach or not grasp_pos:
            cand["pregrasp_policy"] = "unchanged_missing_geometry_for_clearance"
            adjusted.append(cand)
            continue
        center = [float(value) for value in list(object_center)[:3]]
        approach_vec = [float(value) for value in list(approach)[:3]]
        grasp = [float(value) for value in list(grasp_pos)[:3]]
        clearance_m = (
            float(clearance)
            if clearance is not None
            else float(cand.get("pregrasp_clearance_m") or cand.get("pregrasp_standoff_m") or 0.0)
        )
        pre = [
            center[0] - approach_vec[0] * clearance_m + bias_x,
            center[1] - approach_vec[1] * clearance_m + bias_y,
            grasp[2],
        ]
        old_pre_pose = cand.get("pregrasp_pose")
        if old_pre_pose is not None and "original_pregrasp_pose" not in cand:
            cand["original_pregrasp_pose"] = json_safe(old_pre_pose)
        pre_pose["position"] = pre
        cand["pregrasp_pose"] = pre_pose
        cand["pregrasp_policy"] = "object_center_clearance_decoupled_from_grasp_backoff"
        cand["pregrasp_clearance_m"] = float(clearance_m)
        cand["pregrasp_xy_bias_m"] = [bias_x, bias_y]
        cand["pregrasp_to_grasp_distance_m"] = float(math.dist(pre, grasp))
        cand["pregrasp_to_grasp_vector_m"] = [grasp[i] - pre[i] for i in range(3)]
        adjusted.append(cand)
    return adjusted


def _apply_fixed_bottle_grasp_z(
    candidates: list[dict[str, Any]],
    *,
    fixed_grasp_z_m: float,
    lift_z_m: float,
    env_name: str,
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        cand = dict(candidate)
        cand["fixed_grasp_z_env"] = env_name
        cand["fixed_grasp_z_m"] = float(fixed_grasp_z_m)
        cand["original_body_z_source"] = cand.get("body_z_source")
        cand["original_z_offset_m"] = cand.get("z_offset_m")

        for key in ("position", "object_center"):
            values = cand.get(key)
            if values and len(values) >= 3:
                updated = [float(value) for value in list(values)[:3]]
                cand[f"original_{key}"] = json_safe(values)
                updated[2] = float(fixed_grasp_z_m)
                cand[key] = _round_list(updated)

        for pose_key in ("pregrasp_pose", "grasp_pose"):
            pose = dict(cand.get(pose_key) or {})
            values = pose.get("position")
            if values and len(values) >= 3:
                updated = [float(value) for value in list(values)[:3]]
                pose[f"original_position"] = json_safe(values)
                updated[2] = float(fixed_grasp_z_m)
                pose["position"] = updated
                cand[pose_key] = pose

        lift_pose = dict(cand.get("lift_pose") or {})
        values = lift_pose.get("position")
        if values and len(values) >= 3:
            updated = [float(value) for value in list(values)[:3]]
            lift_pose["original_position"] = json_safe(values)
            updated[2] = float(fixed_grasp_z_m) + float(lift_z_m)
            lift_pose["position"] = updated
            cand["lift_pose"] = lift_pose

        cand["body_z_source"] = "fixed_grasp_z_m"
        cand["z_offset_m"] = 0.0
        adjusted.append(cand)
    return adjusted


def _build_bottle_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    requested_arm = _env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_ARM", "left").strip().lower()
    arm = requested_arm if requested_arm in {"left", "right"} else None
    lift_z_m = _env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_LIFT_Z_M", 0.30)
    candidates = generate_side_grasp_candidates(
        detection,
        object_kind="bottle",
        arm=arm,
        default_radius_m=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OBJECT_RADIUS_M", 0.04),
        default_half_height_m=0.0,
        body_fraction=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_BODY_GRASP_FRACTION", 0.45),
        pregrasp_standoff_m=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_PREGRASP_STANDOFF_M", 0.08),
        lift_z_m=lift_z_m,
        width_margin_m=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_GRIPPER_WIDTH_CLEARANCE_M", 0.015),
        include_topdown=False,
        yaw_angles_deg=_float_csv_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_YAWS_DEG", [0.0, 90.0, 70.0, 110.0, 180.0]),
        z_offsets_m=_float_csv_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_Z_OFFSETS_M", [0.0, 0.015, -0.015, 0.03]),
        center_z_offset_without_extents_m=_env_float_pickup(
            "OPENFORGE_BOTTLE_CUP_DRY_POUR_LIVE_DETECTION_Z_TO_BODY_OFFSET_M",
            -0.08,
        ),
        grasp_backoff_m=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_GRASP_BACKOFF_M", -0.02),
    )
    fixed_grasp_z = _optional_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_FIXED_GRASP_Z_M")
    if fixed_grasp_z is not None:
        candidates = _apply_fixed_bottle_grasp_z(
            candidates,
            fixed_grasp_z_m=fixed_grasp_z,
            lift_z_m=lift_z_m,
            env_name="OPENFORGE_BOTTLE_CUP_DRY_POUR_FIXED_GRASP_Z_M",
        )
    return _apply_bottle_pregrasp_clearance(candidates)


def _move_kwargs(side: str, pose: dict[str, Any], *, preview_only: bool = False, gripper: float | None = None) -> dict[str, Any]:
    prefix = "left" if side == "left" else "right"
    kwargs = {
        f"{prefix}_target_pos": [float(value) for value in pose["position"][:3]],
        f"{prefix}_target_rpy": [float(value) for value in pose["rpy"][:3]],
        "planner_backend": _env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": _env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SOLVER_SPEED", "slow"),
        "planning_speed": _env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNING_SPEED", 0.40),
        "preview_only": preview_only,
    }
    ik_error = _optional_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ERROR_THRESHOLD_M")
    ik_rot = _optional_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ROT_THRESHOLD_DEG")
    if ik_error is not None:
        kwargs["ik_error_threshold"] = ik_error
    if ik_rot is not None:
        kwargs["ik_rot_threshold_deg"] = ik_rot
    if gripper is not None:
        kwargs[f"{prefix}_gripper"] = float(gripper)
    return kwargs


def _preview_pose(
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    *,
    label: str,
    gripper: float | None = None,
) -> dict[str, Any]:
    try:
        preview = freespace_move(**_move_kwargs(side, pose, preview_only=True, gripper=gripper))
        cache_key = preview.get("trajectory_cache_key") if isinstance(preview, dict) else getattr(
            preview,
            "trajectory_cache_key",
            None,
        )
        return {
            "success": bool(cache_key),
            "label": label,
            "trajectory_cache_key": cache_key,
            "result": json_safe(preview),
        }
    except Exception as exc:
        return {"success": False, "label": label, "error": f"{type(exc).__name__}: {exc}"}


def _execute_pose(
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    *,
    label: str,
    gripper: float | None = None,
) -> dict[str, Any]:
    preview = _preview_pose(freespace_move, side, pose, label=label, gripper=gripper)
    if not preview.get("success"):
        return {"stage": label, "ok": False, "preview": preview, "why_stopped": "preview failed before execution"}
    executed = freespace_move(trajectory_cache_key=preview["trajectory_cache_key"])
    return {"stage": label, "ok": True, "preview": preview, "execute": json_safe(executed)}


def _tilt_rpy(base_rpy: list[float], pitch_delta: float, roll_delta: float = 0.0, yaw_delta: float = 0.0) -> list[float]:
    return [base_rpy[0] + roll_delta, base_rpy[1] + pitch_delta, base_rpy[2] + yaw_delta]


def _rotation_from_rpy_deg(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = [math.radians(float(value)) for value in rpy[:3]]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Match the usual freespace_move RPY convention: Rz(yaw) * Ry(pitch) * Rx(roll).
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rotation_from_quat_xyzw(quat_xyzw: Any) -> list[list[float]]:
    values = [float(value) for value in list(quat_xyzw)[:4]]
    if len(values) < 4:
        raise ValueError("quat_xyzw must contain four values")
    x, y, z, w = values
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quat_xyzw has near-zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _quat_angle_deg(a_xyzw: Any, b_xyzw: Any) -> float:
    a = np.asarray(list(a_xyzw)[:4], dtype=np.float64)
    b = np.asarray(list(b_xyzw)[:4], dtype=np.float64)
    if a.shape[0] < 4 or b.shape[0] < 4:
        return float("inf")
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return float("inf")
    dot = abs(float(np.dot(a / an, b / bn)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(float(mat[row][col]) * float(vec[col]) for col in range(3)) for row in range(3)]


def _mat_t_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(float(mat[row][col]) * float(vec[row]) for row in range(3)) for col in range(3)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _vec_scale(a: list[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in a[:3]]


def _vec_norm(a: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in a[:3]))


def _vec_unit(a: list[float]) -> list[float] | None:
    norm = _vec_norm(a)
    if norm < 1e-9:
        return None
    return [float(value) / norm for value in a[:3]]


def _unit_xy(vec: list[float]) -> list[float]:
    norm = math.hypot(float(vec[0]), float(vec[1]))
    if norm < 1e-9:
        return [0.0, 1.0]
    return [float(vec[0]) / norm, float(vec[1]) / norm]


def _yaw_deg_from_xy(vec: list[float]) -> float:
    unit = _unit_xy(vec)
    return math.degrees(math.atan2(unit[1], unit[0]))


def _coerce_vec3(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = list(value)
        if len(values) < 3:
            return None
        return [float(values[0]), float(values[1]), float(values[2])]
    except Exception:
        return None


def _estimate_mouth_local_offset(selected_pickup: dict[str, Any], lift_rpy: list[float]) -> dict[str, Any]:
    explicit = _optional_vector3("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_LOCAL_OFFSET_M")
    if explicit is not None:
        grasp_xyz = _xyz(selected_pickup.get("grasp_pose") or selected_pickup)
        mouth_world = None
        if grasp_xyz is not None:
            mouth_world = _vec_add(grasp_xyz, _mat_vec(_rotation_from_rpy_deg(lift_rpy), explicit))
        return {
            "mouth_local_offset_m": explicit,
            "source": "OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_LOCAL_OFFSET_M",
            "mouth_world_upright_m": mouth_world,
            "world_offset_upright_m": _mat_vec(_rotation_from_rpy_deg(lift_rpy), explicit),
            "bottle_geometry": {
                "source": "explicit_mouth_local_offset",
                "grasp_world_m": _round_list(grasp_xyz or []),
                "mouth_world_upright_m": _round_list(mouth_world or []),
                "mouth_local_offset_m": _round_list(explicit),
                "grasp_to_mouth_distance_m": round(_vec_norm(explicit), 5),
            },
        }

    grasp_xyz = _xyz(selected_pickup.get("grasp_pose") or selected_pickup)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    if grasp_xyz is None or source_xyz is None:
        fallback = _env_vector3("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_LOCAL_OFFSET_FALLBACK_M", [0.0, 0.0, 0.10])
        return {
            "mouth_local_offset_m": fallback,
            "source": "fallback_without_detection_or_grasp_xyz",
            "mouth_world_upright_m": None,
            "world_offset_upright_m": None,
        }

    mouth_world = _vec_add(source_xyz, _env_vector3("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_DETECTION_EXTRA_OFFSET_M", [0.0, 0.0, 0.0]))
    object_center = _coerce_vec3(selected_pickup.get("object_center"))
    configured_length = _optional_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_LENGTH_M")
    mouth_from_top_offset = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_FROM_TOP_OFFSET_M", 0.0)
    center_to_detection_m = None
    axis_world = None
    estimated_length = None
    if object_center is not None:
        center_to_detection = _vec_sub(source_xyz, object_center)
        center_to_detection_m = _vec_norm(center_to_detection)
        axis_world = _vec_unit(center_to_detection)
        if configured_length is not None and axis_world is not None:
            half_length = max(0.0, configured_length * 0.5 - mouth_from_top_offset)
            mouth_world = _vec_add(
                object_center,
                _vec_scale(axis_world, half_length),
            )
            mouth_world = _vec_add(
                mouth_world,
                _env_vector3("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_DETECTION_EXTRA_OFFSET_M", [0.0, 0.0, 0.0]),
            )
            estimated_length = configured_length
        elif center_to_detection_m is not None:
            estimated_length = center_to_detection_m * 2.0
    world_offset = _vec_sub(mouth_world, grasp_xyz)
    mouth_local = _mat_t_vec(_rotation_from_rpy_deg(lift_rpy), world_offset)
    grasp_to_mouth_m = _vec_norm(world_offset)
    grasp_fraction_from_mouth = None
    if estimated_length and estimated_length > 1e-6:
        grasp_fraction_from_mouth = grasp_to_mouth_m / estimated_length
    return {
        "mouth_local_offset_m": mouth_local,
        "source": "source_detection_minus_grasp_pose_transformed_to_lift_frame",
        "mouth_world_upright_m": mouth_world,
        "world_offset_upright_m": world_offset,
        "bottle_geometry": {
            "source": (
                "configured_length_from_body_center_axis"
                if configured_length is not None and object_center is not None and axis_world is not None
                else "live_detection_top_proxy_minus_grasp_pose"
            ),
            "detection_world_m": _round_list(source_xyz),
            "object_center_world_m": _round_list(object_center or []),
            "grasp_world_m": _round_list(grasp_xyz),
            "mouth_world_upright_m": _round_list(mouth_world),
            "axis_world_upright_m": _round_list(axis_world or []),
            "center_to_detection_m": None if center_to_detection_m is None else round(center_to_detection_m, 5),
            "estimated_bottle_length_m": None if estimated_length is None else round(estimated_length, 5),
            "configured_bottle_length_m": configured_length,
            "mouth_from_top_offset_m": round(mouth_from_top_offset, 5),
            "grasp_to_mouth_world_m": _round_list(world_offset),
            "grasp_to_mouth_distance_m": round(grasp_to_mouth_m, 5),
            "grasp_fraction_from_mouth_to_base": (
                None if grasp_fraction_from_mouth is None else round(grasp_fraction_from_mouth, 5)
            ),
            "mouth_local_offset_m": _round_list(mouth_local),
        },
    }


def _mouth_world_for_pose(pose: dict[str, Any], mouth_local_offset: list[float]) -> list[float]:
    pose_xyz = _xyz(pose)
    rpy = [float(value) for value in list(pose.get("rpy") or [])[:3]]
    if pose_xyz is None or len(rpy) < 3:
        raise RuntimeError("cannot estimate bottle mouth without pose xyz and rpy")
    return _vec_add(pose_xyz, _mat_vec(_rotation_from_rpy_deg(rpy), mouth_local_offset))


def _pose_solving_mouth_target(
    *,
    mouth_target: list[float],
    rpy: list[float],
    mouth_local_offset: list[float],
    min_tcp_z: float,
) -> dict[str, Any]:
    mouth_offset = _mat_vec(_rotation_from_rpy_deg(rpy), mouth_local_offset)
    tcp_xyz = _vec_sub(mouth_target, mouth_offset)
    z_adjustment = 0.0
    if tcp_xyz[2] < min_tcp_z:
        z_adjustment = min_tcp_z - tcp_xyz[2]
        tcp_xyz[2] = min_tcp_z
    pose = {"position": [tcp_xyz[0], tcp_xyz[1], tcp_xyz[2]], "rpy": list(rpy[:3])}
    estimated_mouth = _mouth_world_for_pose(pose, mouth_local_offset)
    mouth_error = _vec_sub(estimated_mouth, mouth_target)
    return {
        "pose": pose,
        "mouth_offset_world_m": mouth_offset,
        "estimated_mouth_m": estimated_mouth,
        "mouth_error_m": mouth_error,
        "mouth_error_norm_m": _vec_norm(mouth_error),
        "tcp_z_adjustment_m": z_adjustment,
    }


def _tilt_waypoint_fractions() -> list[float]:
    raw = _float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_WAYPOINT_FRACTIONS", [0.0, 0.5, 1.0])
    values = {0.0, 1.0}
    for value in raw:
        values.add(min(1.0, max(0.0, float(value))))
    return sorted(values)


def _tilt_pitch_profiles(lift_rpy: list[float]) -> list[dict[str, Any]]:
    base_pitch = float(lift_rpy[1])
    sequence = _optional_float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_PITCH_SEQUENCE_DEG")
    if sequence:
        pitches = [float(value) for value in sequence]
        if abs(pitches[0] - base_pitch) > 1e-3:
            pitches = [base_pitch, *pitches]
        if len(pitches) < 2:
            pitches = [base_pitch, base_pitch - 20.0, base_pitch]
        return [
            {
                "pitch_index": 0,
                "pitch_delta": pitches[-1] - base_pitch,
                "absolute_pitch_sequence_deg": pitches,
                "profile_source": "OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_PITCH_SEQUENCE_DEG",
                "profile_includes_return": abs(pitches[-1] - pitches[0]) <= 1e-3,
                "max_tilt_from_base_deg": max(abs(pitch - base_pitch) for pitch in pitches),
            }
        ]
    return [
        {
            "pitch_index": index,
            "pitch_delta": float(delta),
            "absolute_pitch_sequence_deg": None,
            "profile_source": "OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_PITCH_DELTAS_DEG",
            "profile_includes_return": False,
            "max_tilt_from_base_deg": abs(float(delta)),
        }
        for index, delta in enumerate(
            _float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_PITCH_DELTAS_DEG", [-20.0, -30.0, -40.0])
        )
    ]


def _densify_pitch_sequence(pitches: list[float]) -> tuple[list[float], bool]:
    max_step = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_PITCH_STEP_DEG", 25.0)
    if max_step <= 0.0 or len(pitches) < 2:
        return [float(value) for value in pitches], False
    dense = [float(pitches[0])]
    changed = False
    for target in [float(value) for value in pitches[1:]]:
        start = dense[-1]
        delta = target - start
        step_count = max(1, int(math.ceil(abs(delta) / max_step)))
        if step_count > 1:
            changed = True
        for step_index in range(1, step_count + 1):
            dense.append(start + delta * (step_index / step_count))
    return dense, changed


def _solve_mouth_locked_tilt_waypoints(
    *,
    requested_mouth_target: list[float],
    lift_rpy: list[float],
    pitch_delta: float,
    yaw_delta: float,
    mouth_local_offset: list[float],
    min_tcp_z: float,
    pitch_sequence_deg: list[float] | None = None,
) -> dict[str, Any]:
    # Pour geometry:
    #
    #   T(theta) = TCP/world position at a given wrist tilt.
    #   R(theta) = world-from-TCP rotation matrix for that wrist tilt.
    #   m_local  = vector from TCP/grasp frame to the bottle mouth.
    #   M(theta) = estimated bottle-mouth world position.
    #
    # The forward model is:
    #
    #   M(theta) = T(theta) + R(theta) * m_local
    #
    # For pouring, the controlled point should be the mouth, not the TCP.  So
    # for each tilt waypoint we solve the inverse:
    #
    #   T(theta) = M_target - R(theta) * m_local
    #
    # This lets the wrist rotate while the estimated mouth stays over the cup.
    # The grasp point matters because it defines m_local; if the grasp is lower,
    # higher, or offset from the bottle centerline, the computed TCP path changes.
    #
    # Safety adds one constraint: every solved TCP must stay above min_tcp_z.
    # Since T_z(theta) = M_target_z - (R(theta) * m_local)_z, the target mouth
    # height must satisfy:
    #
    #   M_target_z >= min_tcp_z + (R(theta) * m_local)_z
    #
    # for every waypoint.  Raising/clipping each TCP independently would move the
    # mouth to different heights at different tilt angles.  Instead, compute the
    # minimum feasible mouth height once, then solve every waypoint from that same
    # feasible target.  That keeps the mouth path internally consistent.
    orientation_models: list[dict[str, Any]] = []
    feasible_mouth_target = list(requested_mouth_target[:3])
    if pitch_sequence_deg:
        pitch_values = [float(value) for value in pitch_sequence_deg]
        denominator = max(1, len(pitch_values) - 1)
        profile_items = [
            {
                "index": index,
                "fraction": index / denominator,
                "pitch": pitch,
                "rpy": [float(lift_rpy[0]), pitch, float(lift_rpy[2]) + yaw_delta * (index / denominator)],
            }
            for index, pitch in enumerate(pitch_values)
        ]
    else:
        profile_items = [
            {
                "index": index,
                "fraction": fraction,
                "pitch": float(lift_rpy[1]) + pitch_delta * fraction,
                "rpy": _tilt_rpy(lift_rpy, pitch_delta * fraction, 0.0, yaw_delta * fraction),
            }
            for index, fraction in enumerate(_tilt_waypoint_fractions())
        ]
    for item in profile_items:
        waypoint_rpy = item["rpy"]
        mouth_offset_world = _mat_vec(_rotation_from_rpy_deg(waypoint_rpy), mouth_local_offset)
        required_mouth_z = min_tcp_z + mouth_offset_world[2]
        feasible_mouth_target[2] = max(feasible_mouth_target[2], required_mouth_z)
        orientation_models.append(
            {
                "label": f"tilt_waypoint_{int(item['index']):02d}",
                "fraction": float(item["fraction"]),
                "absolute_pitch_deg": float(item["pitch"]),
                "rpy": waypoint_rpy,
                "mouth_offset_world_m": mouth_offset_world,
                "required_mouth_z_for_tcp_clearance_m": required_mouth_z,
            }
        )

    waypoint_models = []
    max_tcp_z_adjustment = 0.0
    for item in orientation_models:
        model = _pose_solving_mouth_target(
            mouth_target=feasible_mouth_target,
            rpy=item["rpy"],
            mouth_local_offset=mouth_local_offset,
            min_tcp_z=min_tcp_z,
        )
        max_tcp_z_adjustment = max(max_tcp_z_adjustment, float(model["tcp_z_adjustment_m"]))
        waypoint_models.append(
            {
                "label": item["label"],
                "fraction": round(float(item["fraction"]), 5),
                "absolute_pitch_deg": round(float(item["absolute_pitch_deg"]), 5),
                "pose": model["pose"],
                "mouth_offset_world_m": _round_list(item["mouth_offset_world_m"]),
                "required_mouth_z_for_tcp_clearance_m": round(
                    float(item["required_mouth_z_for_tcp_clearance_m"]),
                    5,
                ),
                "estimated_mouth_m": _round_list(model["estimated_mouth_m"]),
                "mouth_error_m": _round_list(model["mouth_error_m"]),
                "mouth_error_norm_m": round(float(model["mouth_error_norm_m"]), 5),
                "tcp_z_adjustment_m": round(float(model["tcp_z_adjustment_m"]), 5),
            }
        )
    return {
        "requested_mouth_target_m": list(requested_mouth_target[:3]),
        "feasible_mouth_target_m": feasible_mouth_target,
        "mouth_target_z_adjustment_m": feasible_mouth_target[2] - requested_mouth_target[2],
        "max_tcp_z_adjustment_after_feasible_target_m": max_tcp_z_adjustment,
        "waypoints": waypoint_models,
        "clearance_constraints": [
            {
                "label": item["label"],
                "fraction": round(float(item["fraction"]), 5),
                "absolute_pitch_deg": round(float(item["absolute_pitch_deg"]), 5),
                "required_mouth_z_for_tcp_clearance_m": round(
                    float(item["required_mouth_z_for_tcp_clearance_m"]),
                    5,
                ),
            }
            for item in orientation_models
        ],
    }


def _solve_side_sweep_tilt_waypoints(
    *,
    final_mouth_target: list[float],
    lift_rpy: list[float],
    yaw_deg: float,
    body_side_xy: list[float],
    mouth_local_offset: list[float],
    min_tcp_z: float,
    pitch_sequence_deg: list[float],
) -> dict[str, Any]:
    # Safer pour geometry:
    #
    # The old "mouth locked" model solved every waypoint against the same cup
    # mouth target.  That is mathematically tidy but physically wrong: at the
    # upright 90 degree waypoint it places the vertical bottle body directly on
    # the cup line.  The bottle body and gripper have thickness, so that can hit
    # or drag the cup before any tilt happens.
    #
    # Instead, the mouth follows a side-sweep path:
    #
    #   progress = 0.0 at the upright side staging pose
    #   progress = 1.0 at the deepest pour tilt
    #   mouth_target(progress) =
    #       final_mouth_target
    #       + body_side_xy * side_standoff * (1 - progress)
    #       + z_lift * (1 - progress)
    #
    # `body_side_xy` points from the cup toward the bottle.  The wrist yaw is
    # chosen so that at the deepest tilt the vector from mouth -> grasp/TCP
    # points along this same body-side direction.  The body stays outside the
    # cup; the mouth moves toward the cup only as the bottle is already tilted.
    base_pitch = float(lift_rpy[1])
    requested_pitches = [float(value) for value in pitch_sequence_deg]
    pitches, densified = _densify_pitch_sequence(requested_pitches)
    if len(pitches) < 2:
        pitches = [base_pitch, base_pitch - 80.0, base_pitch]
    max_tilt = max(abs(pitch - base_pitch) for pitch in pitches) or 1.0
    side_standoff = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_SWEEP_STANDOFF_M", 0.18)
    side_z_lift = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_SWEEP_Z_LIFT_M", 0.08)
    body_side = _unit_xy(body_side_xy)
    waypoint_models = []
    for index, pitch in enumerate(pitches):
        progress = min(1.0, max(0.0, abs(pitch - base_pitch) / max_tilt))
        mouth_target = [
            float(final_mouth_target[0]) + body_side[0] * side_standoff * (1.0 - progress),
            float(final_mouth_target[1]) + body_side[1] * side_standoff * (1.0 - progress),
            float(final_mouth_target[2]) + side_z_lift * (1.0 - progress),
        ]
        rpy = [float(lift_rpy[0]), pitch, float(yaw_deg)]
        model = _pose_solving_mouth_target(
            mouth_target=mouth_target,
            rpy=rpy,
            mouth_local_offset=mouth_local_offset,
            min_tcp_z=min_tcp_z,
        )
        waypoint_models.append(
            {
                "label": f"tilt_waypoint_{index:02d}",
                "fraction": round(index / max(1, len(pitches) - 1), 5),
                "absolute_pitch_deg": round(pitch, 5),
                "progress_to_pour": round(progress, 5),
                "pose": model["pose"],
                "requested_mouth_target_m": _round_list(mouth_target),
                "estimated_mouth_m": _round_list(model["estimated_mouth_m"]),
                "mouth_error_m": _round_list(model["mouth_error_m"]),
                "mouth_error_norm_m": round(float(model["mouth_error_norm_m"]), 5),
                "mouth_offset_world_m": _round_list(model["mouth_offset_world_m"]),
                "tcp_z_adjustment_m": round(float(model["tcp_z_adjustment_m"]), 5),
            }
        )
    return {
        "final_mouth_target_m": list(final_mouth_target[:3]),
        "waypoints": waypoint_models,
        "requested_absolute_pitch_sequence_deg": [round(float(value), 5) for value in requested_pitches],
        "expanded_absolute_pitch_sequence_deg": [round(float(value), 5) for value in pitches],
        "pitch_sequence_densified": densified,
        "max_pitch_step_deg": _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_PITCH_STEP_DEG", 25.0),
        "body_side_xy": _round_list(body_side, limit=2),
        "side_sweep_standoff_m": side_standoff,
        "side_sweep_z_lift_m": side_z_lift,
        "yaw_deg": yaw_deg,
        "path_type": "side_sweep_keep_body_outside_cup",
        "max_tcp_z_adjustment_after_feasible_target_m": max(
            (float(item.get("tcp_z_adjustment_m") or 0.0) for item in waypoint_models),
            default=0.0,
        ),
    }


def _solve_fixed_pivot_tilt_waypoints(
    *,
    final_mouth_target: list[float],
    lift_rpy: list[float],
    yaw_deg: float,
    body_side_xy: list[float],
    mouth_local_offset: list[float],
    min_tcp_z: float,
    pitch_sequence_deg: list[float],
) -> dict[str, Any]:
    # Simple dry-pour geometry:
    #
    # Move once to a side pivot pose, then keep the TCP position fixed while the
    # hand pitch changes from upright toward the pour angle and back.  The pivot
    # is chosen so the bottle mouth lands over the final target at the deepest
    # tilt.  Unlike side-sweep, the mouth is not locked to a target at every
    # intermediate waypoint; the fixed TCP is the controlled point during tilt.
    base_pitch = float(lift_rpy[1])
    requested_pitches = [float(value) for value in pitch_sequence_deg]
    if len(requested_pitches) < 2:
        requested_pitches = [base_pitch, base_pitch - 80.0, base_pitch]
    pitches, densified = _densify_pitch_sequence(requested_pitches)
    deepest_pitch = max(pitches, key=lambda pitch: abs(float(pitch) - base_pitch))
    deepest_rpy = [float(lift_rpy[0]), float(deepest_pitch), float(yaw_deg)]
    deepest_model = _pose_solving_mouth_target(
        mouth_target=final_mouth_target,
        rpy=deepest_rpy,
        mouth_local_offset=mouth_local_offset,
        min_tcp_z=min_tcp_z,
    )
    pivot_xyz = [float(value) for value in deepest_model["pose"]["position"][:3]]
    waypoint_models = []
    max_tilt = max(abs(pitch - base_pitch) for pitch in pitches) or 1.0
    for index, pitch in enumerate(pitches):
        rpy = [float(lift_rpy[0]), float(pitch), float(yaw_deg)]
        pose = {"position": list(pivot_xyz), "rpy": rpy}
        estimated_mouth = _mouth_world_for_pose(pose, mouth_local_offset)
        mouth_error = _vec_sub(estimated_mouth, final_mouth_target)
        waypoint_models.append(
            {
                "label": f"tilt_waypoint_{index:02d}",
                "fraction": round(index / max(1, len(pitches) - 1), 5),
                "absolute_pitch_deg": round(float(pitch), 5),
                "progress_to_pour": round(
                    min(1.0, max(0.0, abs(float(pitch) - base_pitch) / max_tilt)),
                    5,
                ),
                "pose": pose,
                "requested_mouth_target_m": _round_list(final_mouth_target),
                "estimated_mouth_m": _round_list(estimated_mouth),
                "mouth_error_m": _round_list(mouth_error),
                "mouth_error_norm_m": round(_vec_norm(mouth_error), 5),
                "mouth_offset_world_m": _round_list(_mat_vec(_rotation_from_rpy_deg(rpy), mouth_local_offset)),
                "tcp_z_adjustment_m": 0.0,
            }
        )
    return {
        "final_mouth_target_m": list(final_mouth_target[:3]),
        "pivot_tcp_m": _round_list(pivot_xyz),
        "waypoints": waypoint_models,
        "requested_absolute_pitch_sequence_deg": [round(float(value), 5) for value in requested_pitches],
        "expanded_absolute_pitch_sequence_deg": [round(float(value), 5) for value in pitches],
        "pitch_sequence_densified": densified,
        "max_pitch_step_deg": _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_PITCH_STEP_DEG", 25.0),
        "body_side_xy": _round_list(_unit_xy(body_side_xy), limit=2),
        "yaw_deg": yaw_deg,
        "path_type": "fixed_pivot_tcp_orientation_tilt",
        "mouth_target_z_adjustment_m": float(deepest_model["estimated_mouth_m"][2]) - float(final_mouth_target[2]),
        "max_tcp_z_adjustment_after_feasible_target_m": float(deepest_model["tcp_z_adjustment_m"]),
    }


def _make_pour_candidates(selected_pickup: dict[str, Any], cup_detection: dict[str, Any]) -> list[dict[str, Any]]:
    cup_xyz = _xyz(cup_detection)
    lift_pose = selected_pickup.get("lift_pose") or {}
    lift_xyz = _xyz(lift_pose)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    lift_rpy = [float(value) for value in list(lift_pose.get("rpy") or selected_pickup.get("rpy") or [])[:3]]
    if cup_xyz is None or lift_xyz is None or len(lift_rpy) < 3:
        raise RuntimeError("cannot build pour candidates without cup xyz, lift pose, and lift rpy")

    source_for_direction = source_xyz or lift_xyz
    dx = source_for_direction[0] - cup_xyz[0]
    dy = source_for_direction[1] - cup_xyz[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        unit = [0.0, 1.0]
    else:
        unit = [dx / norm, dy / norm]
    cup_rim_z = cup_xyz[2] + _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_RIM_Z_OFFSET_M", 0.045)
    target_mode = os.environ.get("OPENFORGE_BOTTLE_CUP_DRY_POUR_TARGET_MODE", "mouth").strip().lower()
    if target_mode not in {"mouth", "tcp"}:
        target_mode = "mouth"
    standoffs = (
        _float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_TARGET_STANDOFFS_M", [0.0, 0.02, -0.02])
        if target_mode == "mouth"
        else _float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_TCP_STANDOFFS_M", [0.09, 0.07, 0.11])
    )
    pitch_profiles = _tilt_pitch_profiles(lift_rpy)
    yaw_deltas = _float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_YAW_DELTAS_DEG", [0.0])
    offsets = _offset_triples(
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_OFFSETS_M",
        [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, -0.02, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]],
    )
    clearance = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_CLEARANCE_ABOVE_CUP_M", 0.18)
    high_lift = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_APPROACH_Z_LIFT_M", 0.08)
    max_candidates = max(1, _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_POUR_CANDIDATES", 24))
    mouth_model = _estimate_mouth_local_offset(selected_pickup, lift_rpy)
    mouth_local_offset = list(mouth_model["mouth_local_offset_m"])
    mouth_path_mode = os.environ.get("OPENFORGE_BOTTLE_CUP_DRY_POUR_MOUTH_PATH_MODE", "side_sweep").strip().lower()
    if mouth_path_mode not in {"side_sweep", "locked", "fixed_pivot"}:
        mouth_path_mode = "side_sweep"
    align_tilt_yaw = _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_ALIGN_TILT_YAW_TO_CUP_BOTTLE_AXIS", True)
    min_tcp_z = cup_rim_z + _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MIN_TCP_CLEARANCE_ABOVE_CUP_M", 0.12)
    keep_tcp_above_lift = _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_KEEP_TCP_AT_OR_ABOVE_LIFT_Z", True)
    if keep_tcp_above_lift:
        min_tcp_z = max(lift_xyz[2], min_tcp_z)
    candidates: list[dict[str, Any]] = []
    for offset_index, offset in enumerate(offsets):
        target_xy = [cup_xyz[0] + float(offset[0]), cup_xyz[1] + float(offset[1])]
        for standoff_index, standoff in enumerate(standoffs):
            for pitch_profile in pitch_profiles:
                pitch_index = int(pitch_profile["pitch_index"])
                pitch_delta = float(pitch_profile["pitch_delta"])
                pitch_sequence = pitch_profile.get("absolute_pitch_sequence_deg")
                side_sweep_derived_pitch_sequence = False
                if mouth_path_mode in {"side_sweep", "fixed_pivot"} and not pitch_sequence:
                    pitch_sequence = [float(lift_rpy[1]) + pitch_delta * fraction for fraction in _tilt_waypoint_fractions()]
                    side_sweep_derived_pitch_sequence = True
                for yaw_index, yaw_delta in enumerate(yaw_deltas):
                    if len(candidates) >= max_candidates:
                        return candidates
                    pour_yaw_deg = (
                        _yaw_deg_from_xy(unit) + float(yaw_delta)
                        if target_mode == "mouth" and align_tilt_yaw
                        else float(lift_rpy[2]) + float(yaw_delta)
                    )
                    tilt_rpy = (
                        [float(lift_rpy[0]), float(pitch_sequence[-1]), pour_yaw_deg]
                        if pitch_sequence
                        else [*_tilt_rpy(lift_rpy, pitch_delta, 0.0, 0.0)[:2], pour_yaw_deg]
                    )
                    mouth_target = [
                        target_xy[0] + unit[0] * float(standoff),
                        target_xy[1] + unit[1] * float(standoff),
                        cup_rim_z + clearance + float(offset[2]),
                    ]
                    if target_mode == "mouth":
                        if _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_COMPENSATE_MOUTH_DURING_TILT", True):
                            requested_mouth_target = list(mouth_target[:3])
                            if mouth_path_mode == "fixed_pivot" and pitch_sequence:
                                mouth_path = _solve_fixed_pivot_tilt_waypoints(
                                    final_mouth_target=mouth_target,
                                    lift_rpy=lift_rpy,
                                    yaw_deg=pour_yaw_deg,
                                    body_side_xy=unit,
                                    mouth_local_offset=mouth_local_offset,
                                    min_tcp_z=min_tcp_z,
                                    pitch_sequence_deg=pitch_sequence,
                                )
                                mouth_target = list(mouth_path["final_mouth_target_m"])
                            elif mouth_path_mode == "side_sweep" and pitch_sequence:
                                mouth_path = _solve_side_sweep_tilt_waypoints(
                                    final_mouth_target=mouth_target,
                                    lift_rpy=lift_rpy,
                                    yaw_deg=pour_yaw_deg,
                                    body_side_xy=unit,
                                    mouth_local_offset=mouth_local_offset,
                                    min_tcp_z=min_tcp_z,
                                    pitch_sequence_deg=pitch_sequence,
                                )
                                mouth_target = list(mouth_path["final_mouth_target_m"])
                            else:
                                mouth_path = _solve_mouth_locked_tilt_waypoints(
                                    requested_mouth_target=mouth_target,
                                    lift_rpy=[float(lift_rpy[0]), float(lift_rpy[1]), pour_yaw_deg],
                                    pitch_delta=pitch_delta,
                                    yaw_delta=0.0,
                                    mouth_local_offset=mouth_local_offset,
                                    min_tcp_z=min_tcp_z,
                                    pitch_sequence_deg=pitch_sequence,
                                )
                                mouth_target = list(mouth_path["feasible_mouth_target_m"])
                            waypoint_models = list(mouth_path["waypoints"])
                            deepest_waypoint = max(
                                waypoint_models,
                                key=lambda item: abs(float(item.get("absolute_pitch_deg", lift_rpy[1])) - float(lift_rpy[1])),
                            )
                            upright_pose = waypoint_models[0]["pose"]
                            tilt_pose = deepest_waypoint["pose"]
                            estimated_mouth = list(deepest_waypoint["estimated_mouth_m"])
                            mouth_error = list(deepest_waypoint["mouth_error_m"])
                            mouth_target_z_adjustment = float(mouth_path.get("mouth_target_z_adjustment_m") or 0.0)
                            tcp_z_adjustment = float(mouth_path["max_tcp_z_adjustment_after_feasible_target_m"])
                        else:
                            requested_mouth_target = list(mouth_target[:3])
                            model = _pose_solving_mouth_target(
                                mouth_target=mouth_target,
                                rpy=tilt_rpy,
                                mouth_local_offset=mouth_local_offset,
                                min_tcp_z=min_tcp_z,
                            )
                            tcp_xyz = model["pose"]["position"]
                            upright_pose = {"position": [tcp_xyz[0], tcp_xyz[1], tcp_xyz[2]], "rpy": lift_rpy}
                            tilt_pose = model["pose"]
                            estimated_mouth = model["estimated_mouth_m"]
                            mouth_error = model["mouth_error_m"]
                            mouth_target_z_adjustment = 0.0
                            tcp_z_adjustment = float(model["tcp_z_adjustment_m"])
                            waypoint_models = []
                            mouth_path = None
                    else:
                        requested_mouth_target = list(mouth_target[:3])
                        tcp_xyz = [
                            target_xy[0] + unit[0] * float(standoff),
                            target_xy[1] + unit[1] * float(standoff),
                            max(lift_xyz[2], cup_rim_z + clearance + float(offset[2])),
                        ]
                        z_adjustment = 0.0
                        upright_pose = {"position": [tcp_xyz[0], tcp_xyz[1], tcp_xyz[2]], "rpy": lift_rpy}
                        tilt_pose = {
                            "position": [tcp_xyz[0], tcp_xyz[1], tcp_xyz[2]],
                            "rpy": tilt_rpy,
                        }
                        waypoint_models = []
                        estimated_mouth = _mouth_world_for_pose(tilt_pose, mouth_local_offset)
                        mouth_error = _vec_sub(estimated_mouth, mouth_target)
                        mouth_target_z_adjustment = 0.0
                        tcp_z_adjustment = 0.0
                    if target_mode == "mouth" and waypoint_models:
                        upright_waypoint = waypoint_models[0]
                        pivot_pose_role = (
                            "upright_fixed_tcp_pivot_beside_cup"
                            if mouth_path_mode == "fixed_pivot"
                            else "upright_side_stage_beside_cup"
                            if mouth_path_mode == "side_sweep"
                            else "upright_mouth_locked_target"
                        )
                        if mouth_path_mode == "fixed_pivot":
                            mouth_targeting_policy = (
                                "choose one TCP pivot from the deepest tilted mouth target; "
                                "keep TCP fixed while the mouth follows an arc through intermediate pitches"
                            )
                        elif mouth_path_mode == "side_sweep":
                            mouth_targeting_policy = (
                                "solve each TCP from an explicit mouth waypoint: side stage beside cup, "
                                "then sweep mouth toward the final cup target"
                            )
                        else:
                            mouth_targeting_policy = "solve each TCP from the same estimated mouth target at each tilt waypoint"
                    else:
                        upright_waypoint = None
                        pivot_pose_role = "upright_tcp_target"
                        mouth_targeting_policy = (
                            "solve_tcp_from_estimated_tilted_mouth_target"
                            if target_mode == "mouth"
                            else "legacy_tcp_standoff_from_cup"
                        )
                    high_base = [float(value) for value in list(upright_pose["position"])[:3]]
                    high_pose = {
                        "position": [high_base[0], high_base[1], high_base[2] + high_lift],
                        "rpy": list(upright_pose.get("rpy") or lift_rpy),
                    }
                    candidates.append(
                        {
                            "label": f"dry_pour_o{offset_index}_s{standoff_index}_p{pitch_index}_y{yaw_index}",
                            "arm": selected_pickup["arm"],
                            "strategy_type": "dry_bottle_tilt_over_stationary_cup",
                            "cup_detection": cup_detection,
                            "pickup_strategy_label": selected_pickup.get("label"),
                            "cup_target_ref_m": _round_list([target_xy[0], target_xy[1], cup_rim_z]),
                            "cup_relative_offset_m": _round_list(offset),
                            "target_mode": target_mode,
                            "tcp_standoff_m": None if target_mode == "mouth" else round(float(standoff), 5),
                            "mouth_target_standoff_m": round(float(standoff), 5) if target_mode == "mouth" else None,
                            "mouth_target_m": _round_list(mouth_target),
                            "estimated_mouth_at_tilt_m": _round_list(estimated_mouth),
                            "estimated_mouth_target_error_m": _round_list(mouth_error),
                            "estimated_mouth_target_error_norm_m": round(_vec_norm(mouth_error), 5),
                            "upright_estimated_mouth_m": (
                                _round_list(upright_waypoint.get("estimated_mouth_m") or [])
                                if isinstance(upright_waypoint, dict)
                                else None
                            ),
                            "upright_mouth_target_error_norm_m": (
                                round(float(upright_waypoint.get("mouth_error_norm_m")), 5)
                                if isinstance(upright_waypoint, dict)
                                and upright_waypoint.get("mouth_error_norm_m") is not None
                                else None
                            ),
                            "deepest_estimated_mouth_m": _round_list(deepest_waypoint.get("estimated_mouth_m") or [])
                            if waypoint_models
                            else None,
                            "deepest_mouth_target_error_norm_m": round(float(deepest_waypoint.get("mouth_error_norm_m")), 5)
                            if waypoint_models and deepest_waypoint.get("mouth_error_norm_m") is not None
                            else None,
                            "pivot_pose_role": pivot_pose_role,
                            "approach_direction_from_cup_to_bottle_xy": _round_list([unit[0], unit[1]], limit=2),
                            "requested_mouth_target_m": _round_list(requested_mouth_target),
                            "mouth_target_z_adjustment_m": round(float(mouth_target_z_adjustment), 5),
                            "tcp_z_adjustment_m": round(float(tcp_z_adjustment), 5),
                            "tilt_profile_source": pitch_profile["profile_source"],
                            "absolute_pitch_sequence_deg": (
                                [round(float(value), 5) for value in pitch_sequence]
                                if pitch_sequence
                                else None
                            ),
                            "expanded_absolute_pitch_sequence_deg": (
                                mouth_path.get("expanded_absolute_pitch_sequence_deg")
                                if isinstance(mouth_path, dict)
                                else None
                            ),
                            "pitch_sequence_densified": (
                                bool(mouth_path.get("pitch_sequence_densified"))
                                if isinstance(mouth_path, dict)
                                else False
                            ),
                            "side_sweep_derived_pitch_sequence": side_sweep_derived_pitch_sequence,
                            "tilt_profile_includes_return": bool(pitch_profile["profile_includes_return"]),
                            "max_tilt_from_base_deg": round(float(pitch_profile["max_tilt_from_base_deg"]), 5),
                            "deepest_tilt_waypoint": json_safe(deepest_waypoint) if waypoint_models else None,
                            "approach_high_pose": high_pose,
                            "upright_over_cup_pose": upright_pose,
                            "tilt_pose": tilt_pose,
                            "tilt_waypoints": waypoint_models,
                            "untilt_pose": upright_pose,
                            "return_lift_pose": lift_pose,
                            "geometry_model": {
                                "source": "fresh top-camera bottle and cup detections",
                                "cup_rim_z_m": round(cup_rim_z, 5),
                                "cup_rim_z_policy": "cup detection z plus configurable rim offset",
                                "mouth_clearance_above_cup_m": round(clearance, 5),
                                "base_lift_rpy": _round_list(lift_rpy),
                                "tilt_pitch_delta_deg": round(float(pitch_delta), 5),
                                "absolute_pitch_sequence_deg": (
                                    [round(float(value), 5) for value in pitch_sequence]
                                    if pitch_sequence
                                    else None
                                ),
                                "expanded_absolute_pitch_sequence_deg": (
                                    mouth_path.get("expanded_absolute_pitch_sequence_deg")
                                    if isinstance(mouth_path, dict)
                                    else None
                                ),
                                "pitch_sequence_densified": (
                                    bool(mouth_path.get("pitch_sequence_densified"))
                                    if isinstance(mouth_path, dict)
                                    else False
                                ),
                                "side_sweep_derived_pitch_sequence": side_sweep_derived_pitch_sequence,
                                "tilt_profile_source": pitch_profile["profile_source"],
                                "tilt_profile_includes_return": bool(pitch_profile["profile_includes_return"]),
                                "max_tilt_from_base_deg": round(float(pitch_profile["max_tilt_from_base_deg"]), 5),
                                "tilt_yaw_delta_deg": round(float(yaw_delta), 5),
                                "pour_yaw_deg": round(float(pour_yaw_deg), 5),
                                "align_tilt_yaw_to_cup_bottle_axis": align_tilt_yaw,
                                "body_side_xy": _round_list(unit, limit=2),
                                "mouth_model": json_safe(mouth_model),
                                "compensate_mouth_during_tilt": bool(target_mode == "mouth" and waypoint_models),
                                "tilt_waypoint_fractions": [
                                    item["fraction"] for item in waypoint_models
                                ],
                                "mouth_path_mode": mouth_path_mode,
                                "mouth_path_model": json_safe(mouth_path) if target_mode == "mouth" and waypoint_models else None,
                                "mouth_targeting_policy": mouth_targeting_policy,
                                "pivot_pose_role": pivot_pose_role,
                                "upright_estimated_mouth_m": (
                                    _round_list(upright_waypoint.get("estimated_mouth_m") or [])
                                    if isinstance(upright_waypoint, dict)
                                    else None
                                ),
                                "upright_mouth_target_error_norm_m": (
                                    round(float(upright_waypoint.get("mouth_error_norm_m")), 5)
                                    if isinstance(upright_waypoint, dict)
                                    and upright_waypoint.get("mouth_error_norm_m") is not None
                                    else None
                                ),
                                "deepest_estimated_mouth_m": _round_list(deepest_waypoint.get("estimated_mouth_m") or [])
                                if waypoint_models
                                else None,
                                "deepest_mouth_target_error_norm_m": round(float(deepest_waypoint.get("mouth_error_norm_m")), 5)
                                if waypoint_models and deepest_waypoint.get("mouth_error_norm_m") is not None
                                else None,
                                "min_tcp_z_m": round(min_tcp_z, 5),
                                "keep_tcp_at_or_above_lift_z": keep_tcp_above_lift,
                                "mouth_target_z_adjustment_m": round(float(mouth_target_z_adjustment), 5),
                                "tcp_z_adjustment_m": round(float(tcp_z_adjustment), 5),
                                "cup_is_stationary": True,
                                "liquid_used": False,
                                "mouth_model_status": (
                                    "approximate dry/no-liquid mouth targeting; real liquid trajectory is not authorized"
                                ),
                            },
                        }
                    )
    return candidates


def _simple_fixed_pour_candidate(
    selected_pickup: dict[str, Any],
    bottle_detection: dict[str, Any],
    cup_detection: dict[str, Any],
    *,
    reference_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bottle_xyz = (
        _xyz(bottle_detection)
        or _xyz(selected_pickup.get("source_detection") or {})
        or _xyz(selected_pickup.get("grasp_pose") or {})
    )
    cup_xyz = _xyz(cup_detection)
    lift_pose = selected_pickup.get("lift_pose") or {}
    lift_xyz = _xyz(lift_pose)
    lift_rpy = [float(value) for value in list(lift_pose.get("rpy") or selected_pickup.get("rpy") or [])[:3]]
    if bottle_xyz is None or lift_xyz is None or len(lift_rpy) < 3:
        raise RuntimeError("cannot build simple fixed pour without bottle xyz and selected lift pose")

    default_offset = [-0.07760, 0.05267, 0.23899]
    offset = _float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_TCP_OFFSET_FROM_BOTTLE_M", default_offset)
    while len(offset) < 3:
        offset.append(default_offset[len(offset)])

    min_z_above_lift = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_MIN_Z_ABOVE_LIFT_M", 0.006)
    tcp_xyz = [
        float(bottle_xyz[0]) + float(offset[0]),
        float(bottle_xyz[1]) + float(offset[1]),
        max(float(bottle_xyz[2]) + float(offset[2]), float(lift_xyz[2]) + min_z_above_lift),
    ]
    yaw_override = _optional_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_YAW_DEG")
    yaw_offset_deg = 0.0
    if yaw_override is not None:
        yaw_deg = float(yaw_override)
        yaw_source = "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_YAW_DEG"
    elif cup_xyz is not None:
        yaw_offset_deg = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_YAW_OFFSET_DEG", -2.0)
        yaw_deg = _yaw_deg_from_xy([float(bottle_xyz[0]) - float(cup_xyz[0]), float(bottle_xyz[1]) - float(cup_xyz[1])])
        yaw_deg += yaw_offset_deg
        yaw_source = "initial_bottle_minus_glass_axis_plus_offset"
    elif reference_candidate:
        yaw_deg = float((reference_candidate.get("upright_over_cup_pose") or {}).get("rpy", lift_rpy)[2])
        yaw_source = "reference_candidate_yaw"
    else:
        yaw_deg = float(lift_rpy[2])
        yaw_source = "pickup_lift_yaw"

    base_pitch = float(lift_rpy[1])
    pitch_sequence = (
        _optional_float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_TILT_PITCH_SEQUENCE_DEG")
        or _optional_float_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_PITCH_SEQUENCE_DEG")
        or [base_pitch, base_pitch - 15.0, base_pitch - 30.0, base_pitch - 45.0, base_pitch - 50.0]
    )
    pitch_sequence = [float(value) for value in pitch_sequence]
    if not pitch_sequence or abs(float(pitch_sequence[0]) - base_pitch) > 1e-3:
        pitch_sequence = [base_pitch, *pitch_sequence]

    approach_z_lift = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_APPROACH_Z_LIFT_M", 0.0)
    upright_pose = {"position": list(tcp_xyz), "rpy": [float(lift_rpy[0]), base_pitch, yaw_deg]}
    approach_high_pose = {
        "position": [tcp_xyz[0], tcp_xyz[1], tcp_xyz[2] + approach_z_lift],
        "rpy": list(upright_pose["rpy"]),
    }
    mouth_model = _estimate_mouth_local_offset(selected_pickup, lift_rpy)
    mouth_local_offset = list(mouth_model["mouth_local_offset_m"])
    tilt_waypoints: list[dict[str, Any]] = []
    for index, pitch in enumerate(pitch_sequence):
        pose = {"position": list(tcp_xyz), "rpy": [float(lift_rpy[0]), float(pitch), yaw_deg]}
        estimated_mouth = _mouth_world_for_pose(pose, mouth_local_offset)
        tilt_waypoints.append(
            {
                "label": f"simple_fixed_tilt_waypoint_{index:02d}",
                "fraction": round(index / max(1, len(pitch_sequence) - 1), 5),
                "absolute_pitch_deg": round(float(pitch), 5),
                "progress_to_pour": round(
                    min(1.0, max(0.0, abs(float(pitch) - base_pitch) / max(1e-6, max(abs(value - base_pitch) for value in pitch_sequence)))),
                    5,
                ),
                "pose": pose,
                "estimated_mouth_m": _round_list(estimated_mouth),
                "mouth_offset_world_m": _round_list(_mat_vec(_rotation_from_rpy_deg(pose["rpy"]), mouth_local_offset)),
            }
        )
    deepest_waypoint = max(
        tilt_waypoints,
        key=lambda item: abs(float(item.get("absolute_pitch_deg", base_pitch)) - base_pitch),
    )
    cup_rim_z = float(cup_xyz[2]) + _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_RIM_Z_OFFSET_M", 0.045) if cup_xyz else None
    return {
        "label": "simple_fixed_pour_offset_from_initial_bottle",
        "arm": selected_pickup["arm"],
        "strategy_type": "simple_fixed_tcp_offset_from_initial_bottle",
        "cup_detection": cup_detection,
        "pickup_strategy_label": selected_pickup.get("label"),
        "target_mode": "fixed_tcp_offset",
        "simple_fixed_tcp_offset_from_bottle_m": _round_list(offset[:3]),
        "simple_fixed_tcp_target_m": _round_list(tcp_xyz),
        "simple_fixed_tcp_target_policy": "initial bottle xyz plus fixed offset; no post-pickup live detections",
        "yaw_source": yaw_source,
        "simple_fixed_yaw_offset_deg": round(float(yaw_offset_deg), 5),
        "cup_target_ref_m": _round_list([cup_xyz[0], cup_xyz[1], cup_rim_z]) if cup_xyz and cup_rim_z is not None else None,
        "approach_direction_from_cup_to_bottle_xy": (
            _round_list(_unit_xy([float(bottle_xyz[0]) - float(cup_xyz[0]), float(bottle_xyz[1]) - float(cup_xyz[1])]), limit=2)
            if cup_xyz
            else None
        ),
        "absolute_pitch_sequence_deg": [round(float(value), 5) for value in pitch_sequence],
        "expanded_absolute_pitch_sequence_deg": [round(float(value), 5) for value in pitch_sequence],
        "pitch_sequence_densified": False,
        "tilt_profile_source": "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_TILT_PITCH_SEQUENCE_DEG",
        "tilt_profile_includes_return": abs(float(pitch_sequence[-1]) - float(pitch_sequence[0])) <= 1e-3,
        "max_tilt_from_base_deg": round(max(abs(float(value) - base_pitch) for value in pitch_sequence), 5),
        "deepest_tilt_waypoint": json_safe(deepest_waypoint),
        "approach_high_pose": approach_high_pose,
        "upright_over_cup_pose": upright_pose,
        "tilt_pose": deepest_waypoint["pose"],
        "tilt_waypoints": tilt_waypoints,
        "untilt_pose": upright_pose,
        "return_lift_pose": lift_pose,
        "geometry_model": {
            "source": "single initial top-camera bottle and glass detections",
            "bottle_xyz_m": _round_list(bottle_xyz),
            "cup_xyz_m": _round_list(cup_xyz) if cup_xyz else None,
            "fixed_tcp_offset_from_bottle_m": _round_list(offset[:3]),
            "min_z_above_lift_m": round(float(min_z_above_lift), 5),
            "approach_z_lift_m": round(float(approach_z_lift), 5),
            "base_lift_rpy": _round_list(lift_rpy),
            "yaw_deg": round(float(yaw_deg), 5),
            "yaw_source": yaw_source,
            "yaw_offset_deg": round(float(yaw_offset_deg), 5),
            "mouth_model": json_safe(mouth_model),
            "post_pickup_live_detection_used": False,
        },
    }


def _pour_motion_specs(pour_candidate: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if pour_candidate.get("target_mode") != "fixed_tcp_offset" and _truthy_env(
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_JOINT_TILT",
        False,
    ):
        # Direct-joint scout mode starts from the post-lift pose.  Do not insert
        # any cuRobo staging move between lift and the guarded direct joint arc.
        return []
    specs: list[tuple[str, dict[str, Any]]] = [
        ("pour_approach_high", pour_candidate["approach_high_pose"]),
    ]
    waypoints = list(pour_candidate.get("tilt_waypoints") or [])
    if waypoints:
        specs.append(("pour_upright_over_cup", waypoints[0]["pose"]))
        if pour_candidate.get("tilt_profile_includes_return"):
            deepest_index = max(
                range(len(waypoints)),
                key=lambda index: abs(
                    float(waypoints[index].get("absolute_pitch_deg", 0.0))
                    - float(waypoints[0].get("absolute_pitch_deg", 0.0))
                ),
            )
            for index, waypoint in enumerate(waypoints[1:], start=1):
                if index == deepest_index:
                    label = "dry_tilt_over_cup"
                elif index > deepest_index:
                    label = "dry_untilt_over_cup" if index == len(waypoints) - 1 else f"dry_untilt_over_cup_{index:02d}"
                else:
                    label = f"dry_tilt_over_cup_{index:02d}"
                specs.append((label, waypoint["pose"]))
        else:
            for index, waypoint in enumerate(waypoints[1:], start=1):
                label = "dry_tilt_over_cup" if index == len(waypoints) - 1 else f"dry_tilt_over_cup_{index:02d}"
                specs.append((label, waypoint["pose"]))
            for index, waypoint in enumerate(reversed(waypoints[1:-1]), start=1):
                specs.append((f"dry_untilt_over_cup_{index:02d}", waypoint["pose"]))
            specs.append(("dry_untilt_over_cup", waypoints[0]["pose"]))
    else:
        specs.extend(
            [
                ("pour_upright_over_cup", pour_candidate["upright_over_cup_pose"]),
                ("dry_tilt_over_cup", pour_candidate["tilt_pose"]),
                ("dry_untilt_over_cup", pour_candidate["untilt_pose"]),
            ]
        )
    specs.append(("return_to_lift_pose", pour_candidate["return_lift_pose"]))
    return specs


def _sequence_specs(
    selected_pickup: dict[str, Any],
    pour_candidate: dict[str, Any],
    *,
    return_to_source_after_pour: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    specs = [
        ("pickup_pregrasp", selected_pickup["pregrasp_pose"]),
        ("pickup_grasp", selected_pickup["grasp_pose"]),
        ("pickup_lift", selected_pickup["lift_pose"]),
        *_pour_motion_specs(pour_candidate),
    ]
    if return_to_source_after_pour:
        specs.extend(
            [
                ("return_to_source_grasp_pose", selected_pickup["grasp_pose"]),
                ("retreat_to_source_pregrasp_pose", selected_pickup["pregrasp_pose"]),
            ]
        )
    return specs


def _relaxed_preview_allowed(failed_stage: str | None, steps: list[dict[str, Any]]) -> bool:
    if not _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_ALLOW_PER_STEP_PHYSICAL_TILT_PREVIEW", False):
        return False
    if not failed_stage:
        return False
    if not (
        failed_stage.startswith("dry_tilt_over_cup")
        or failed_stage.startswith("dry_untilt_over_cup")
        or failed_stage == "return_to_lift_pose"
    ):
        return False
    completed = {str(step.get("stage")) for step in steps if step.get("ok")}
    required = {"pickup_pregrasp", "pickup_grasp", "pickup_lift", "pour_approach_high", "pour_upright_over_cup"}
    return required.issubset(completed)


def _preview_pour_sequence(
    run_dir: Path,
    selected_pickup: dict[str, Any],
    pour_candidates: list[dict[str, Any]],
    *,
    return_to_source_after_pour: bool = True,
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    attempts: list[dict[str, Any]] = []
    for candidate in pour_candidates:
        steps: list[dict[str, Any]] = []
        success = True
        for label, pose in _sequence_specs(
            selected_pickup,
            candidate,
            return_to_source_after_pour=return_to_source_after_pour,
        ):
            preview = _preview_pose(freespace_move, side, pose, label=label)
            step = {"stage": label, "pose": pose, "preview": preview, "ok": bool(preview.get("success"))}
            steps.append(step)
            if not step["ok"]:
                success = False
                break
        attempt = {
            "label": candidate.get("label"),
            "success": success,
            "failed_stage": next((step["stage"] for step in steps if not step.get("ok")), None),
            "pour_candidate": candidate,
            "steps": steps,
        }
        attempts.append(attempt)
        relaxed_accept = (not success) and _relaxed_preview_allowed(attempt.get("failed_stage"), steps)
        if success or relaxed_accept:
            packet = {
                "schema": "openforge.dry_bottle_to_cup_pour.sequence_preview.v1",
                "success": True,
                "full_offline_sequence_preview_success": bool(success),
                "relaxed_per_step_physical_preview_accept": bool(relaxed_accept),
                "relaxed_failed_stage": attempt.get("failed_stage") if relaxed_accept else None,
                "relaxed_reason": (
                    "offline no-motion preview starts every waypoint from the current home state; "
                    "physical execution will preview each remaining tilt waypoint immediately before execution "
                    "from the actual previous waypoint, with source-return recovery on failure"
                    if relaxed_accept
                    else None
                ),
                "side": side,
                "selected_pickup": selected_pickup,
                "selected_pour": candidate,
                "return_to_source_after_pour": bool(return_to_source_after_pour),
                "attempts": attempts,
            }
            packet["path"] = write_json(run_dir / "plans" / "dry_bottle_to_cup_pour_sequence_preview.json", packet)
            append_stage_summary(
                run_dir,
                [
                    "## dry bottle to cup pour sequence preview",
                    "- success: True",
                    f"- full_offline_sequence_preview_success: {success}",
                    f"- relaxed_per_step_physical_preview_accept: {relaxed_accept}",
                    f"- selected_pour: {candidate.get('label')}",
                    f"- relaxed_failed_stage: {attempt.get('failed_stage') if relaxed_accept else None}",
                    f"- upright_pose: {candidate.get('upright_over_cup_pose')}",
                    f"- tilt_pose: {candidate.get('tilt_pose')}",
                    f"- path: {packet['path']}",
                ],
            )
            return packet
    packet = {
        "schema": "openforge.dry_bottle_to_cup_pour.sequence_preview.v1",
        "success": False,
        "full_offline_sequence_preview_success": False,
        "relaxed_per_step_physical_preview_accept": False,
        "side": side,
        "selected_pickup": selected_pickup,
        "selected_pour": None,
        "return_to_source_after_pour": bool(return_to_source_after_pour),
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "plans" / "dry_bottle_to_cup_pour_sequence_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## dry bottle to cup pour sequence preview",
            "- success: False",
            f"- attempts: {[(item.get('label'), item.get('failed_stage')) for item in attempts]}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _make_open_gripper() -> Callable[[str], Any]:
    set_gripper = _tool("set_gripper")
    open_gripper = _tool("open_gripper")
    open_target = _env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_TARGET_POS", 1.0)
    open_vel = _env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_VEL_LIMIT", 0.50)
    open_torque = _env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_TORQUE_LIMIT", 0.30)

    def _open(side: str) -> Any:
        if callable(set_gripper):
            return set_gripper(side, open_target, vel_limit=open_vel, torque_limit=open_torque)
        if callable(open_gripper):
            return open_gripper(side)
        raise RuntimeError("no open gripper API is available")

    return _open


def _extract_gripper_pos(state: Any, side: str) -> float | None:
    safe = json_safe(state)
    if isinstance(safe, dict):
        arms = safe.get("arms")
        if isinstance(arms, dict):
            arm_state = arms.get(side)
            if isinstance(arm_state, dict):
                value = arm_state.get("gripper_pos")
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
        value = safe.get(f"{side}_gripper_pos")
        if value is None:
            value = safe.get(f"{side}_grip")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    arms_obj = getattr(state, "arms", None)
    if isinstance(arms_obj, dict):
        arm_obj = arms_obj.get(side)
        value = getattr(arm_obj, "gripper_pos", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _open_gripper_until(side: str, *, min_pos: float, timeout_s: float, poll_s: float, retry_s: float) -> dict[str, Any]:
    open_fn = _make_open_gripper()
    get_robot_state = _tool("get_robot_state")
    packet: dict[str, Any] = {
        "success": False,
        "side": side,
        "target_min_pos": float(min_pos),
        "timeout_s": float(timeout_s),
        "poll_s": float(poll_s),
        "retry_s": float(retry_s),
        "commands": [],
        "samples": [],
        "final_gripper_pos": None,
        "final_state": None,
        "why_stopped": None,
    }
    deadline = time.monotonic() + max(0.1, timeout_s)
    next_command_time = 0.0
    while time.monotonic() <= deadline:
        now = time.monotonic()
        if now >= next_command_time:
            packet["commands"].append(json_safe(open_fn(side)))
            next_command_time = now + max(0.1, retry_s)
        if callable(get_robot_state):
            state = get_robot_state()
            pos = _extract_gripper_pos(state, side)
            packet["final_state"] = json_safe(state)
            packet["final_gripper_pos"] = pos
            packet["samples"].append({"t_remaining_s": round(max(0.0, deadline - time.monotonic()), 3), "gripper_pos": pos})
            if pos is not None and pos >= min_pos:
                packet["success"] = True
                packet["why_stopped"] = "verified open from robot gripper encoder"
                return packet
        time.sleep(max(0.02, poll_s))
    packet["why_stopped"] = (
        f"timed out waiting for {side} gripper to open to >= {min_pos:.3f}; "
        f"final gripper_pos={packet['final_gripper_pos']}"
    )
    return packet


def _read_direct_arm_obs(env: Any, side: str) -> dict[str, Any]:
    if hasattr(env, "_arms") and side in getattr(env, "_arms"):
        return dict(env._arms[side].get_observations())
    if hasattr(env, "get_observations"):
        try:
            return dict(env.get_observations(side))
        except TypeError:
            obs = dict(env.get_observations())
            return {
                "joint_pos": obs[f"{side}_joint_pos"],
                "gripper_pos": obs[f"{side}_gripper_pos"],
            }
    raise RuntimeError("direct YAM env does not expose arm observations")


def _execute_direct_xy_align(
    run_dir: Path,
    side: str,
    *,
    post_lift_bottle: dict[str, Any] | None,
    cup_detection: dict[str, Any] | None,
) -> dict[str, Any]:
    # Direct XY alignment is a small joint-space correction after the bottle is
    # already lifted.  The visual heuristic computes how far the held bottle is
    # from a desired cup-relative XY lane; IK then shifts the active TCP by that
    # bounded XY delta while preserving current Z/orientation and both grippers.
    # The command is deliberately direct keypoints, not cuRobo planning, so the
    # artifact records FK/IK checks before motion is sent.
    packet: dict[str, Any] = {
        "schema": "openforge.dry_bottle_to_cup_pour.direct_xy_align_execute.v1",
        "success": False,
        "side": side,
        "enabled": True,
        "physical_motion_executed": False,
        "why_stopped": None,
        "post_lift_bottle_detection": json_safe(post_lift_bottle),
        "cup_detection": json_safe(cup_detection),
    }
    bottle_xyz = _xyz(post_lift_bottle or {})
    cup_xyz = _xyz(cup_detection or {})
    if bottle_xyz is None or cup_xyz is None:
        packet["why_stopped"] = "missing post-lift bottle or cup detection for direct XY alignment"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        return packet

    target_sep = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_TARGET_Y_SEPARATION_M", 0.210)
    max_xy_delta = max(0.0, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MAX_XY_DELTA_M", 0.050))
    min_xy_delta = max(0.0, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MIN_XY_DELTA_M", 0.003))
    desired_delta = [
        float(cup_xyz[0]) - float(bottle_xyz[0]),
        float(cup_xyz[1]) + target_sep - float(bottle_xyz[1]),
    ]
    desired_norm = math.hypot(desired_delta[0], desired_delta[1])
    if desired_norm > max_xy_delta and desired_norm > 1e-9:
        scale = max_xy_delta / desired_norm
        clipped_delta = [desired_delta[0] * scale, desired_delta[1] * scale]
    else:
        clipped_delta = list(desired_delta)
    clipped_norm = math.hypot(clipped_delta[0], clipped_delta[1])
    packet.update(
        {
            "post_lift_bottle_xyz_m": _round_list(bottle_xyz),
            "cup_xyz_m": _round_list(cup_xyz),
            "target_bottle_xy_m": _round_list([cup_xyz[0], cup_xyz[1] + target_sep, bottle_xyz[2]]),
            "target_y_separation_m": round(float(target_sep), 5),
            "desired_xy_delta_m": _round_list([desired_delta[0], desired_delta[1], 0.0]),
            "clipped_xy_delta_m": _round_list([clipped_delta[0], clipped_delta[1], 0.0]),
            "desired_xy_delta_norm_m": round(float(desired_norm), 5),
            "clipped_xy_delta_norm_m": round(float(clipped_norm), 5),
            "max_xy_delta_m": round(float(max_xy_delta), 5),
            "min_xy_delta_m": round(float(min_xy_delta), 5),
        }
    )
    if clipped_norm < min_xy_delta:
        packet["success"] = True
        packet["skipped"] = True
        packet["why_stopped"] = "alignment delta below minimum; no direct XY motion needed"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        append_stage_summary(
            run_dir,
            [
                "## direct XY align",
                "- success: True",
                "- skipped: True",
                f"- clipped_xy_delta_m: {packet['clipped_xy_delta_m']}",
                f"- path: {packet['path']}",
            ],
        )
        return packet

    env = _tool_env("freespace_move") or _tool_env("get_robot_state")
    if env is None:
        packet["why_stopped"] = "cannot access direct YAM env for direct XY alignment"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        return packet
    move_keypoints = getattr(env, "move_bimanual_joint_keypoints", None)
    if not callable(move_keypoints):
        packet["why_stopped"] = "direct YAM env does not expose move_bimanual_joint_keypoints"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        return packet

    try:
        from robot.yam.kinematics import YamKinematics
    except Exception as exc:
        packet["why_stopped"] = f"cannot load YamKinematics for direct XY alignment: {exc}"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        return packet

    left_obs = _read_direct_arm_obs(env, "left")
    right_obs = _read_direct_arm_obs(env, "right")
    left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
    right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
    left_gripper = float(np.asarray(left_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gripper = float(np.asarray(right_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])

    kin = YamKinematics()
    left_pos, left_quat, right_pos, right_quat = kin.forward_kinematics(left_base, right_base)
    left_target_pos = np.asarray(left_pos, dtype=np.float64).copy()
    right_target_pos = np.asarray(right_pos, dtype=np.float64).copy()
    if side == "left":
        left_target_pos[0] += clipped_delta[0]
        left_target_pos[1] += clipped_delta[1]
    elif side == "right":
        right_target_pos[0] += clipped_delta[0]
        right_target_pos[1] += clipped_delta[1]
    else:
        packet["why_stopped"] = f"unsupported side for direct XY alignment: {side!r}"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        return packet

    ik_max_iters = max(1, _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_IK_MAX_ITERS", 80))
    ik_err_threshold = max(1e-6, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_IK_INTERNAL_ERR_THRESHOLD", 1e-4))
    ik_left, ik_right = kin.inverse_kinematics(
        left_target_pos,
        np.asarray(left_quat, dtype=np.float64),
        right_target_pos,
        np.asarray(right_quat, dtype=np.float64),
        seeded=True,
        max_iters=ik_max_iters,
        err_threshold=ik_err_threshold,
    )
    end_left_pos, end_left_quat, end_right_pos, end_right_quat = kin.forward_kinematics(ik_left, ik_right)
    active_start = np.asarray(left_pos if side == "left" else right_pos, dtype=np.float64)
    active_target = np.asarray(left_target_pos if side == "left" else right_target_pos, dtype=np.float64)
    active_end = np.asarray(end_left_pos if side == "left" else end_right_pos, dtype=np.float64)
    active_start_quat = np.asarray(left_quat if side == "left" else right_quat, dtype=np.float64)
    active_end_quat = np.asarray(end_left_quat if side == "left" else end_right_quat, dtype=np.float64)
    other_start = np.asarray(right_pos if side == "left" else left_pos, dtype=np.float64)
    other_end = np.asarray(end_right_pos if side == "left" else end_left_pos, dtype=np.float64)
    left_joint_delta = np.asarray(ik_left, dtype=np.float64) - left_base
    right_joint_delta = np.asarray(ik_right, dtype=np.float64) - right_base
    max_joint_delta = max(float(np.max(np.abs(left_joint_delta))), float(np.max(np.abs(right_joint_delta))))
    active_pos_error = float(np.linalg.norm(active_end - active_target))
    active_xy_achieved = [float(active_end[0] - active_start[0]), float(active_end[1] - active_start[1])]
    active_z_delta = abs(float(active_end[2] - active_start[2]))
    other_tcp_motion = float(np.linalg.norm(other_end - other_start))
    active_rot_delta_deg = _quat_angle_deg(active_start_quat, active_end_quat)
    max_ik_error = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MAX_IK_ERROR_M", 0.015)
    max_joint_delta_allowed = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MAX_JOINT_DELTA_RAD", 0.45)
    max_other_motion = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MAX_OTHER_TCP_MOTION_M", 0.010)
    max_z_delta = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MAX_Z_DELTA_M", 0.010)
    max_rot_delta = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_MAX_ROT_DELTA_DEG", 5.0)
    gate = {
        "passed": (
            active_pos_error <= max_ik_error
            and max_joint_delta <= max_joint_delta_allowed
            and other_tcp_motion <= max_other_motion
            and active_z_delta <= max_z_delta
            and active_rot_delta_deg <= max_rot_delta
        ),
        "max_ik_error_m": max_ik_error,
        "max_joint_delta_rad": max_joint_delta_allowed,
        "max_other_tcp_motion_m": max_other_motion,
        "max_z_delta_m": max_z_delta,
        "max_rot_delta_deg": max_rot_delta,
        "observed_active_pos_error_m": round(active_pos_error, 5),
        "observed_max_joint_delta_rad": round(max_joint_delta, 5),
        "observed_other_tcp_motion_m": round(other_tcp_motion, 5),
        "observed_active_z_delta_m": round(active_z_delta, 5),
        "observed_active_rot_delta_deg": round(active_rot_delta_deg, 5),
        "reason": None,
    }
    if active_pos_error > max_ik_error:
        gate["reason"] = f"active TCP IK error {active_pos_error:.3f} m exceeds {max_ik_error:.3f} m"
    elif max_joint_delta > max_joint_delta_allowed:
        gate["reason"] = f"joint delta {max_joint_delta:.3f} rad exceeds {max_joint_delta_allowed:.3f} rad"
    elif other_tcp_motion > max_other_motion:
        gate["reason"] = f"other TCP moved {other_tcp_motion:.3f} m, above {max_other_motion:.3f} m"
    elif active_z_delta > max_z_delta:
        gate["reason"] = f"active TCP z changed {active_z_delta:.3f} m, above {max_z_delta:.3f} m"
    elif active_rot_delta_deg > max_rot_delta:
        gate["reason"] = f"active TCP rotation changed {active_rot_delta_deg:.2f} deg, above {max_rot_delta:.2f} deg"

    packet.update(
        {
            "gate": gate,
            "left_start_joint_pos": _round_list(left_base, limit=6),
            "right_start_joint_pos": _round_list(right_base, limit=6),
            "left_end_joint_pos": _round_list(ik_left, limit=6),
            "right_end_joint_pos": _round_list(ik_right, limit=6),
            "left_joint_delta_rad": _round_list(left_joint_delta, limit=6),
            "right_joint_delta_rad": _round_list(right_joint_delta, limit=6),
            "left_gripper_pos": round(left_gripper, 5),
            "right_gripper_pos": round(right_gripper, 5),
            "fk_start": {
                "left_tcp_m": _round_list(left_pos),
                "right_tcp_m": _round_list(right_pos),
            },
            "fk_target": {
                "left_tcp_m": _round_list(left_target_pos),
                "right_tcp_m": _round_list(right_target_pos),
            },
            "fk_end": {
                "left_tcp_m": _round_list(end_left_pos),
                "right_tcp_m": _round_list(end_right_pos),
            },
            "achieved_active_xy_delta_m": _round_list([active_xy_achieved[0], active_xy_achieved[1], 0.0]),
        }
    )
    if not gate["passed"]:
        packet["why_stopped"] = f"direct XY align gate failed: {gate['reason']}"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
        append_stage_summary(
            run_dir,
            [
                "## direct XY align",
                "- success: False",
                f"- why_stopped: {packet['why_stopped']}",
                f"- clipped_xy_delta_m: {packet['clipped_xy_delta_m']}",
                f"- path: {packet['path']}",
            ],
        )
        return packet

    duration_s = max(0.05, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_STEP_S", 0.80))
    result = move_keypoints(
        timestamps=[0.0, duration_s],
        left_joint_positions=[left_base.tolist(), np.asarray(ik_left, dtype=np.float64).tolist()],
        right_joint_positions=[right_base.tolist(), np.asarray(ik_right, dtype=np.float64).tolist()],
        left_gripper_positions=[left_gripper, left_gripper],
        right_gripper_positions=[right_gripper, right_gripper],
        playback_speed=_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_PLAYBACK_SPEED", 1.0),
        command_hz=_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_COMMAND_HZ", 60.0),
        start_interp_s=_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_START_INTERP_S", 0.25),
    )
    packet["physical_motion_executed"] = True
    packet["result"] = json_safe(result)
    packet["success"] = bool(result.get("success")) if isinstance(result, dict) else False
    packet["why_stopped"] = (
        "direct XY alignment executed"
        if packet["success"]
        else "direct XY alignment command returned unsuccessful result"
    )
    packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_execute.json", json_safe(packet))
    append_stage_summary(
        run_dir,
        [
            "## direct XY align",
            f"- success: {packet['success']}",
            f"- physical_motion_executed: {packet['physical_motion_executed']}",
            f"- desired_xy_delta_m: {packet['desired_xy_delta_m']}",
            f"- clipped_xy_delta_m: {packet['clipped_xy_delta_m']}",
            f"- achieved_active_xy_delta_m: {packet['achieved_active_xy_delta_m']}",
            f"- gate: {gate['passed']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _verify_direct_xy_align_visual(
    run_dir: Path,
    *,
    post_align_bottle: dict[str, Any] | None,
    post_align_cup: dict[str, Any] | None,
    fallback_cup_detection: dict[str, Any] | None,
    direct_align: dict[str, Any] | None,
) -> dict[str, Any]:
    # This gate verifies the object pose, not just the arm FK.  Direct XY align
    # can prove that the TCP moved as commanded, but the bottle may slip or the
    # TCP-to-bottle offset may differ after grasp.  Before rolling j5, require a
    # fresh top-camera bottle detection to be in the intended cup-relative lane:
    #
    #   target bottle x = cup x
    #   target bottle y = cup y + target_y_separation
    packet: dict[str, Any] = {
        "schema": "openforge.dry_bottle_to_cup_pour.direct_xy_align_visual_verify.v1",
        "success": False,
        "post_align_bottle_detection": json_safe(post_align_bottle),
        "post_align_cup_detection": json_safe(post_align_cup),
        "fallback_cup_detection": json_safe(fallback_cup_detection),
        "direct_align": json_safe(direct_align),
        "why_stopped": None,
    }
    bottle_xyz = _xyz(post_align_bottle or {})
    cup_xyz = _xyz(post_align_cup or {}) or _xyz(fallback_cup_detection or {})
    packet["cup_detection_source_for_gate"] = (
        "post_align" if _xyz(post_align_cup or {}) is not None else "fallback"
    )
    if bottle_xyz is None or cup_xyz is None:
        packet["why_stopped"] = "missing post-align bottle or cup detection for visual alignment verification"
        packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_visual_verify.json", json_safe(packet))
        return packet

    target_sep = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_TARGET_Y_SEPARATION_M", 0.210)
    max_x_error = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_VERIFY_MAX_X_ERROR_M", 0.035)
    max_y_sep_error = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_VERIFY_MAX_Y_SEPARATION_ERROR_M", 0.030)
    max_xy_error = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_VERIFY_MAX_XY_ERROR_M", 0.045)

    x_error = float(bottle_xyz[0]) - float(cup_xyz[0])
    y_separation = float(bottle_xyz[1]) - float(cup_xyz[1])
    y_sep_error = y_separation - target_sep
    xy_error = math.hypot(x_error, y_sep_error)
    gate = {
        "passed": (
            abs(x_error) <= max_x_error
            and abs(y_sep_error) <= max_y_sep_error
            and xy_error <= max_xy_error
        ),
        "max_x_error_m": max_x_error,
        "max_y_separation_error_m": max_y_sep_error,
        "max_xy_error_m": max_xy_error,
        "observed_x_error_m": round(x_error, 5),
        "observed_y_separation_m": round(y_separation, 5),
        "observed_y_separation_error_m": round(y_sep_error, 5),
        "observed_xy_error_m": round(xy_error, 5),
        "reason": None,
    }
    if abs(x_error) > max_x_error:
        gate["reason"] = f"post-align x error {x_error:.3f} m exceeds {max_x_error:.3f} m"
    elif abs(y_sep_error) > max_y_sep_error:
        gate["reason"] = (
            f"post-align y separation error {y_sep_error:.3f} m exceeds {max_y_sep_error:.3f} m"
        )
    elif xy_error > max_xy_error:
        gate["reason"] = f"post-align XY error {xy_error:.3f} m exceeds {max_xy_error:.3f} m"

    packet.update(
        {
            "post_align_bottle_xyz_m": _round_list(bottle_xyz),
            "cup_xyz_m": _round_list(cup_xyz),
            "target_bottle_xy_m": _round_list([cup_xyz[0], cup_xyz[1] + target_sep, bottle_xyz[2]]),
            "target_y_separation_m": round(float(target_sep), 5),
            "gate": gate,
            "success": bool(gate["passed"]),
            "why_stopped": (
                "post-align visual verification passed"
                if gate["passed"]
                else f"post-align visual verification failed: {gate['reason']}"
            ),
        }
    )
    packet["path"] = write_json(run_dir / "plans" / "direct_xy_align_visual_verify.json", json_safe(packet))
    append_stage_summary(
        run_dir,
        [
            "## direct XY align visual verify",
            f"- success: {packet['success']}",
            f"- cup_detection_source_for_gate: {packet['cup_detection_source_for_gate']}",
            f"- post_align_bottle_xyz_m: {packet['post_align_bottle_xyz_m']}",
            f"- target_bottle_xy_m: {packet['target_bottle_xy_m']}",
            f"- observed_x_error_m: {gate['observed_x_error_m']}",
            f"- observed_y_separation_error_m: {gate['observed_y_separation_error_m']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _fk_direct_joint_tilt_mouth_path(
    *,
    side: str,
    left_waypoints: list[list[float]],
    right_waypoints: list[list[float]],
    sequence: list[float],
    selected_pour: dict[str, Any],
) -> dict[str, Any]:
    # Direct-keypoint tilt bypasses cuRobo for the pour arc, so the math needs
    # its own guard.  The bottle mouth is modeled as a fixed point in the TCP
    # frame:
    #
    #   mouth_world(q) = FK_tcp_position(q) + FK_tcp_rotation(q) * mouth_local
    #
    # Target-mode direct tilt accepts a sequence only if this estimated mouth
    # path moves closer to the cup opening while staying above the rim. Scout
    # mode is deliberately weaker for first-axis tests: it only checks that a
    # small direct roll is bounded, returns near the start, and keeps the mouth
    # clear of the rim/cup. These gates do not make direct keypoints
    # collision-planned; they only reject obvious geometry mistakes before the
    # dry/no-liquid motion is attempted.
    try:
        from robot.yam.kinematics import YamKinematics
    except Exception as exc:
        raise RuntimeError(f"cannot load YamKinematics for direct tilt FK gate: {exc}") from exc

    mouth_model = ((selected_pour.get("geometry_model") or {}).get("mouth_model") or {})
    mouth_local = [float(value) for value in list(mouth_model.get("mouth_local_offset_m") or [])[:3]]
    if len(mouth_local) < 3:
        raise RuntimeError("selected pour candidate is missing mouth_local_offset_m for direct tilt FK gate")
    cup_ref = _round_list(selected_pour.get("cup_target_ref_m") or [], limit=3)
    if len(cup_ref) < 3:
        cup_xyz = _xyz(selected_pour.get("cup_detection") or {})
        if cup_xyz is None:
            raise RuntimeError("selected pour candidate is missing cup target for direct tilt FK gate")
        cup_ref = [cup_xyz[0], cup_xyz[1], cup_xyz[2] + _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_RIM_Z_OFFSET_M", 0.045)]

    kin = YamKinematics()
    rows: list[dict[str, Any]] = []
    for index, (left_joints, right_joints) in enumerate(zip(left_waypoints, right_waypoints)):
        left_pos, left_quat, right_pos, right_quat = kin.forward_kinematics(
            np.asarray(left_joints, dtype=np.float64),
            np.asarray(right_joints, dtype=np.float64),
        )
        tcp_pos = left_pos if side == "left" else right_pos
        tcp_quat = left_quat if side == "left" else right_quat
        mouth_world = _vec_add(
            [float(value) for value in tcp_pos[:3]],
            _mat_vec(_rotation_from_quat_xyzw(tcp_quat), mouth_local),
        )
        xy_error = math.hypot(float(mouth_world[0]) - float(cup_ref[0]), float(mouth_world[1]) - float(cup_ref[1]))
        rows.append(
            {
                "index": index,
                "joint_delta_rad": round(float(sequence[index]), 5) if index < len(sequence) else None,
                "tcp_world_m": _round_list(tcp_pos, limit=3),
                "mouth_world_m": _round_list(mouth_world, limit=3),
                "mouth_xy_error_to_cup_m": round(float(xy_error), 5),
                "mouth_z_above_cup_rim_m": round(float(mouth_world[2]) - float(cup_ref[2]), 5),
            }
        )

    start_error = float(rows[0]["mouth_xy_error_to_cup_m"]) if rows else float("inf")
    best = min(rows, key=lambda item: float(item["mouth_xy_error_to_cup_m"])) if rows else {}
    best_error = float(best.get("mouth_xy_error_to_cup_m", float("inf")))
    min_z_above_rim = min((float(item["mouth_z_above_cup_rim_m"]) for item in rows), default=float("-inf"))
    start_mouth = rows[0].get("mouth_world_m") if rows else None
    max_mouth_travel = 0.0
    if isinstance(start_mouth, list) and len(start_mouth) >= 3:
        for row in rows:
            mouth = row.get("mouth_world_m")
            if isinstance(mouth, list) and len(mouth) >= 3:
                max_mouth_travel = max(
                    max_mouth_travel,
                    _vec_norm(
                        [
                            float(mouth[0]) - float(start_mouth[0]),
                            float(mouth[1]) - float(start_mouth[1]),
                            float(mouth[2]) - float(start_mouth[2]),
                        ]
                    ),
                )
    min_xy_error = min((float(item["mouth_xy_error_to_cup_m"]) for item in rows), default=float("inf"))
    return {
        "schema": "openforge.dry_bottle_to_cup_pour.direct_joint_tilt_fk_mouth_path.v1",
        "cup_ref_m": _round_list(cup_ref, limit=3),
        "mouth_local_offset_m": _round_list(mouth_local, limit=3),
        "rows": rows,
        "start_mouth_xy_error_m": round(start_error, 5),
        "best_mouth_xy_error_m": round(best_error, 5),
        "best_index": best.get("index"),
        "best_joint_delta_rad": best.get("joint_delta_rad"),
        "min_mouth_z_above_cup_rim_m": round(float(min_z_above_rim), 5),
        "min_mouth_xy_error_to_cup_m": round(float(min_xy_error), 5),
        "max_mouth_travel_from_start_m": round(float(max_mouth_travel), 5),
    }


def _direct_joint_tilt_plan(
    run_dir: Path,
    env: Any,
    side: str,
    *,
    selected_pour: dict[str, Any] | None = None,
) -> dict[str, Any]:
    joint_index = _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_JOINT_INDEX", 3)
    if joint_index < 0 or joint_index >= 6:
        raise RuntimeError("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_JOINT_INDEX must be in [0,5]")
    sequence = _float_csv(
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_JOINT_SEQUENCE_RAD",
        [0.0, 0.35, 0.70, 1.05, 1.35, 1.45, 1.35, 1.05, 0.70, 0.35, 0.0],
    )
    if len(sequence) < 2:
        raise RuntimeError("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_JOINT_SEQUENCE_RAD needs at least 2 values")
    max_abs_delta = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_MAX_ABS_DELTA_RAD", 1.50)
    observed_max = max(abs(float(value)) for value in sequence)
    if observed_max > max_abs_delta + 1e-9:
        raise RuntimeError(
            "direct tilt joint sequence exceeds "
            f"OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_MAX_ABS_DELTA_RAD={max_abs_delta}"
        )

    per_step_s = max(0.05, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_STEP_S", 0.55))
    hold_s = max(0.0, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_HOLD_S", 0.75))
    timestamps = [index * per_step_s for index in range(len(sequence))]
    if hold_s > 1e-9:
        deepest_index = max(range(len(sequence)), key=lambda index: abs(float(sequence[index])))
        held_sequence = list(sequence)
        held_timestamps = list(timestamps)
        held_sequence.insert(deepest_index + 1, sequence[deepest_index])
        held_timestamps.insert(deepest_index + 1, timestamps[deepest_index] + hold_s)
        for index in range(deepest_index + 2, len(held_timestamps)):
            held_timestamps[index] += hold_s
        sequence = held_sequence
        timestamps = held_timestamps

    left_obs = _read_direct_arm_obs(env, "left")
    right_obs = _read_direct_arm_obs(env, "right")
    left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
    right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
    left_gripper = float(np.asarray(left_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    right_gripper = float(np.asarray(right_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])

    left_waypoints = []
    right_waypoints = []
    for delta in sequence:
        left_joints = left_base.copy()
        right_joints = right_base.copy()
        if side == "left":
            left_joints[joint_index] = left_base[joint_index] + float(delta)
        elif side == "right":
            right_joints[joint_index] = right_base[joint_index] + float(delta)
        else:
            raise RuntimeError(f"unsupported side for direct joint tilt: {side!r}")
        left_waypoints.append(left_joints.tolist())
        right_waypoints.append(right_joints.tolist())

    packet = {
        "schema": "openforge.dry_bottle_to_cup_pour.direct_joint_tilt_plan.v1",
        "side": side,
        "joint_index": joint_index,
        "joint_sequence_rad": [round(float(value), 5) for value in sequence],
        "max_abs_delta_rad": max_abs_delta,
        "step_s": per_step_s,
        "hold_s": hold_s,
        "timestamps_s": [round(float(value), 5) for value in timestamps],
        "command_hz": _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_COMMAND_HZ", 60.0),
        "playback_speed": _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_PLAYBACK_SPEED", 1.0),
        "start_interp_s": _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_START_INTERP_S", 0.25),
        "left_start_joint_pos": _round_list(left_base, limit=6),
        "right_start_joint_pos": _round_list(right_base, limit=6),
        "left_gripper_pos": round(left_gripper, 5),
        "right_gripper_pos": round(right_gripper, 5),
        "left_joint_positions": left_waypoints,
        "right_joint_positions": right_waypoints,
        "left_gripper_positions": [left_gripper] * len(sequence),
        "right_gripper_positions": [right_gripper] * len(sequence),
        "policy": (
            "direct joint-keypoint dry tilt; cuRobo is used before/after, "
            "but this arc is bounded by joint deltas and recorded because cuRobo could not solve the deep pitch"
        ),
    }
    if selected_pour is not None:
        fk_path = _fk_direct_joint_tilt_mouth_path(
            side=side,
            left_waypoints=left_waypoints,
            right_waypoints=right_waypoints,
            sequence=[float(value) for value in sequence],
            selected_pour=selected_pour,
        )
        geometry_mode = os.environ.get(
            "OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_GEOMETRY_MODE",
            "target",
        ).strip().lower()
        if geometry_mode not in {"target", "scout"}:
            geometry_mode = "target"
        max_best_xy = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_MAX_BEST_MOUTH_XY_ERROR_M", 0.12)
        min_improvement = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_MIN_XY_IMPROVEMENT_M", 0.04)
        min_z_above_rim = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_MIN_MOUTH_Z_ABOVE_RIM_M", 0.04)
        start_error = float(fk_path["start_mouth_xy_error_m"])
        best_error = float(fk_path["best_mouth_xy_error_m"])
        z_above = float(fk_path["min_mouth_z_above_cup_rim_m"])
        if geometry_mode == "scout":
            max_travel = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_SCOUT_MAX_MOUTH_TRAVEL_M", 0.08)
            min_xy_error = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_SCOUT_MIN_MOUTH_XY_ERROR_M", 0.03)
            return_tol = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_RETURN_TOLERANCE_RAD", 0.03)
            end_delta = abs(float(sequence[-1])) if sequence else float("inf")
            travel = float(fk_path.get("max_mouth_travel_from_start_m") or float("inf"))
            closest_xy = float(fk_path.get("min_mouth_xy_error_to_cup_m") or float("inf"))
            gate = {
                "mode": "scout",
                "passed": (
                    end_delta <= return_tol
                    and travel <= max_travel
                    and closest_xy >= min_xy_error
                    and z_above >= min_z_above_rim
                ),
                "max_mouth_travel_from_start_m": max_travel,
                "min_mouth_xy_error_to_cup_m": min_xy_error,
                "min_mouth_z_above_rim_m": min_z_above_rim,
                "return_tolerance_rad": return_tol,
                "observed_end_delta_rad": round(end_delta, 5),
                "observed_max_mouth_travel_from_start_m": round(travel, 5),
                "observed_min_mouth_xy_error_to_cup_m": round(closest_xy, 5),
                "reason": None,
            }
            if end_delta > return_tol:
                gate["reason"] = f"direct scout sequence does not return near start; end delta {end_delta:.3f} rad"
            elif travel > max_travel:
                gate["reason"] = f"direct scout mouth travel {travel:.3f} m exceeds {max_travel:.3f} m"
            elif closest_xy < min_xy_error:
                gate["reason"] = f"direct scout mouth comes within {closest_xy:.3f} m of cup, below {min_xy_error:.3f} m"
            elif z_above < min_z_above_rim:
                gate["reason"] = f"mouth z above rim {z_above:.3f} m is below {min_z_above_rim:.3f} m"
        else:
            gate = {
                "mode": "target",
                "passed": (
                    best_error <= max_best_xy
                    and (start_error - best_error) >= min_improvement
                    and z_above >= min_z_above_rim
                ),
                "max_best_mouth_xy_error_m": max_best_xy,
                "min_xy_improvement_m": min_improvement,
                "min_mouth_z_above_rim_m": min_z_above_rim,
                "xy_improvement_m": round(start_error - best_error, 5),
                "reason": None,
            }
            if best_error > max_best_xy:
                gate["reason"] = f"best mouth XY error {best_error:.3f} m exceeds {max_best_xy:.3f} m"
            elif (start_error - best_error) < min_improvement:
                gate["reason"] = (
                    f"mouth XY improvement {start_error - best_error:.3f} m is below {min_improvement:.3f} m"
                )
            elif z_above < min_z_above_rim:
                gate["reason"] = f"mouth z above rim {z_above:.3f} m is below {min_z_above_rim:.3f} m"
        packet["fk_mouth_path"] = fk_path
        packet["fk_geometry_gate"] = gate
        if _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_TILT_REQUIRE_FK_GATE", True) and not gate["passed"]:
            packet["path"] = write_json(run_dir / "plans" / "direct_joint_tilt_plan.json", json_safe(packet))
            raise RuntimeError(f"direct tilt FK geometry gate failed: {gate['reason']}")
    packet["path"] = write_json(run_dir / "plans" / "direct_joint_tilt_plan.json", json_safe(packet))
    return packet


def _execute_direct_joint_tilt(run_dir: Path, side: str, *, selected_pour: dict[str, Any]) -> dict[str, Any]:
    env = _tool_env("freespace_move") or _tool_env("get_robot_state")
    if env is None:
        raise RuntimeError("cannot access direct YAM env for direct joint tilt")
    move_keypoints = getattr(env, "move_bimanual_joint_keypoints", None)
    if not callable(move_keypoints):
        raise RuntimeError("direct YAM env does not expose move_bimanual_joint_keypoints")
    plan = _direct_joint_tilt_plan(run_dir, env, side, selected_pour=selected_pour)
    result = move_keypoints(
        timestamps=plan["timestamps_s"],
        left_joint_positions=plan["left_joint_positions"],
        right_joint_positions=plan["right_joint_positions"],
        left_gripper_positions=plan["left_gripper_positions"],
        right_gripper_positions=plan["right_gripper_positions"],
        playback_speed=plan["playback_speed"],
        command_hz=plan["command_hz"],
        start_interp_s=plan["start_interp_s"],
    )
    packet = {
        "schema": "openforge.dry_bottle_to_cup_pour.direct_joint_tilt_execute.v1",
        "success": bool(result.get("success")) if isinstance(result, dict) else False,
        "plan": plan,
        "result": json_safe(result),
    }
    packet["path"] = write_json(run_dir / "plans" / "direct_joint_tilt_execute.json", json_safe(packet))
    append_stage_summary(
        run_dir,
        [
            "## direct joint dry tilt",
            f"- success: {packet['success']}",
            f"- side: {side}",
            f"- joint_index: {plan['joint_index']}",
            f"- max_delta_rad: {max(abs(value) for value in plan['joint_sequence_rad'])}",
            f"- fk_geometry_gate: {(plan.get('fk_geometry_gate') or {}).get('passed')}",
            f"- fk_best_mouth_xy_error_m: {((plan.get('fk_mouth_path') or {}).get('best_mouth_xy_error_m'))}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _ticket_status(run_dir: Path) -> dict[str, Any]:
    required_true = [
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_PHYSICAL_TICKET",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_OPERATOR_PRESENT",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_ESTOP_REACHABLE",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_CLEAR",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_FOLLOWER_SERVERS_UP",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_RECORDING_DEBUG_CONFIRMED",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_DRY_EMPTY_OR_SEALED_BOTTLE",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_EMPTY_CUP",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_NO_LIQUID",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_NO_MANUAL_XYZ",
        "OPENFORGE_BOTTLE_CUP_DRY_POUR_NO_OTHER_MOTION_PRECHECK",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_BOTTLE_CUP_DRY_POUR_TICKET_SCOPE", "").strip()
    if scope != "dry_bottle_to_cup_pour":
        missing.append("OPENFORGE_BOTTLE_CUP_DRY_POUR_TICKET_SCOPE=dry_bottle_to_cup_pour")
    budget = _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_PHYSICAL_ATTEMPT_BUDGET", 0)
    max_budget = _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_ALLOWED_ATTEMPT_BUDGET", 20)
    if budget < 1 or budget > max_budget:
        missing.append(f"OPENFORGE_BOTTLE_CUP_DRY_POUR_PHYSICAL_ATTEMPT_BUDGET=1..{max_budget}")
    manual_env = _manual_pose_env_vars_present()
    if manual_env:
        missing.extend(manual_env)
    pgrep_status = _pgrep_competing_motion(run_dir)
    if pgrep_status["competitors"]:
        missing.append("no competing run_script.py/ffmpeg process")
    return {
        "passed": not missing,
        "missing_or_invalid": missing,
        "ticket_scope": scope,
        "attempt_budget": budget,
        "max_allowed_attempt_budget": max_budget,
        "pgrep": pgrep_status,
    }


def _pgrep_competing_motion(run_dir: Path) -> dict[str, Any]:
    try:
        import subprocess

        proc = subprocess.run(
            ["pgrep", "-af", "run_script.py|ffmpeg"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception as exc:
        return {"checked": False, "error": f"{type(exc).__name__}: {exc}", "competitors": []}
    current = str(run_dir)
    current_pids = {str(os.getpid()), str(os.getppid())}
    competitors: list[str] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        pid = line.split(maxsplit=1)[0]
        if pid in current_pids:
            continue
        if current in line:
            continue
        if "pgrep -af" in line:
            continue
        competitors.append(line)
    return {"checked": True, "ignored_pids": sorted(current_pids), "competitors": competitors}


def _recover_bottle_to_source_after_pour_failure(
    *,
    freespace_move: Callable[..., Any],
    side: str,
    selected_pickup: dict[str, Any],
    failed_label: str,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "success": False,
        "failed_label": failed_label,
        "movement_capable_calls": [],
        "steps": [],
        "release_open": None,
        "why_stopped": None,
    }
    for label, pose in (
        ("recovery_return_to_lift_pose", selected_pickup["lift_pose"]),
        ("recovery_return_to_source_grasp_pose", selected_pickup["grasp_pose"]),
    ):
        step = _execute_pose(freespace_move, side, pose, label=label)
        packet["steps"].append(step)
        packet["movement_capable_calls"].append(f"freespace_move:{label}")
        if not step.get("ok"):
            packet["why_stopped"] = f"{label} failed; keep bottle held and stop for operator intervention"
            return packet

    release = _open_gripper_until(
        side,
        min_pos=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_MIN_POS", 0.95),
        timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_TIMEOUT_S", 12.0),
        poll_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_POLL_S", 0.25),
        retry_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_RETRY_S", 1.5),
    )
    packet["release_open"] = release
    packet["movement_capable_calls"].append(f"open_gripper:{side}")
    if not release.get("success"):
        packet["why_stopped"] = str(release.get("why_stopped") or "recovery release open failed")
        return packet

    if _truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_RETREAT_TO_PREGRASP_AFTER_RELEASE", True):
        step = _execute_pose(
            freespace_move,
            side,
            selected_pickup["pregrasp_pose"],
            label="recovery_retreat_to_source_pregrasp_pose",
        )
        packet["steps"].append(step)
        packet["movement_capable_calls"].append("freespace_move:recovery_retreat_to_source_pregrasp_pose")
        if not step.get("ok"):
            packet["why_stopped"] = "recovery release succeeded, but retreat to source pregrasp failed"
            return packet

    if _truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_GO_HOME_AFTER_RELEASE", False):
        go_home = _tool("go_home")
        if callable(go_home):
            packet["go_home_after_recovery"] = json_safe(go_home())
            packet["movement_capable_calls"].append("go_home")
            if _truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME", True):
                post_home_open = _open_gripper_until(
                    side,
                    min_pos=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_MIN_POS", 0.90),
                    timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_TIMEOUT_S", 12.0),
                    poll_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_POLL_S", 0.25),
                    retry_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_RETRY_S", 1.5),
                )
                packet["open_after_home"] = post_home_open
                packet["movement_capable_calls"].append(f"open_gripper:{side}")
                if not post_home_open.get("success"):
                    packet["why_stopped"] = str(post_home_open.get("why_stopped") or "recovery open after home failed")
                    return packet

    packet["success"] = True
    packet["why_stopped"] = "returned bottle to source grasp pose, opened, and retreated after failed pour step"
    return packet


def _execute_physical_dry_pour(
    *,
    run_dir: Path,
    selected_pickup: dict[str, Any],
    selected_pour: dict[str, Any],
    bottle_detection: dict[str, Any],
    cup_detection: dict[str, Any],
    return_to_source_after_pour: bool = True,
    simple_fixed_pour: bool = False,
) -> dict[str, Any]:
    side = str(selected_pickup["arm"])
    freespace_move = _required_tool("freespace_move")
    attempt: dict[str, Any] = {
        "schema": "openforge.dry_bottle_to_cup_pour.physical_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "selected_pour": selected_pour,
        "simple_fixed_pour": bool(simple_fixed_pour),
        "steps": [],
        "why_stopped": None,
    }
    try:
        pickup = execute_grasp_lift_attempt(
            plan=selected_pickup,
            freespace_move=freespace_move,
            open_gripper=_make_open_gripper(),
            set_gripper=_required_tool("set_gripper"),
            get_robot_state=_required_tool("get_robot_state"),
            physical_gate=True,
            run_dir=run_dir,
            task_name=TASK_NAME,
            planning_speed=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNING_SPEED", 0.40),
            planner_backend=_env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNER_BACKEND", "rrtconnect"),
            solver_speed=_env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SOLVER_SPEED", "slow"),
            ik_error_threshold=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ERROR_THRESHOLD_M", 0.015),
            ik_rot_threshold_deg=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ROT_THRESHOLD_DEG", 8.0),
            gripper_target=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_TARGET_POS", 0.35),
            gripper_steps=_float_csv_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_STEPS_POS", [0.78, 0.58, 0.35]),
            gripper_vel_limit=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_VEL_LIMIT", 0.60),
            gripper_torque_limit=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_TORQUE_LIMIT", 0.80),
            gripper_min_contact_delta=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_CONTACT_MIN_DELTA_POS", 0.02),
            gripper_target_tolerance=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_TARGET_TOLERANCE_POS", 0.08),
            require_contact_for_lift=_truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_CLOSE_CONTACT", True),
            open_confirm_min=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_CONFIRM_POS", 0.95),
            open_confirm_timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_CONFIRM_TIMEOUT_S", 12.0),
            motion_gripper_pos=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_TARGET_POS", 1.0),
            preview_before_each_move=True,
            execute_lift=True,
        )
        attempt["physical_motion_executed"] = bool(pickup.get("physical_motion_executed"))
        attempt["movement_capable_calls"].extend(pickup.get("movement_capable_calls") or [])
        attempt["pickup_attempt"] = pickup
        if not pickup.get("success"):
            attempt["why_stopped"] = f"bottle pickup/lift failed before dry pour: {pickup.get('why_stopped')}"
            return attempt

        if simple_fixed_pour:
            post_lift_bottle = bottle_detection
            attempt["post_lift_observation"] = {
                "skipped": True,
                "reason": "simple fixed pour uses only the initial bottle/glass detections",
            }
            attempt["post_lift_bottle_detection"] = post_lift_bottle
            attempt["bottle_lift_verification"] = {
                "success": True,
                "skipped": True,
                "reason": "pickup tool success/contact is used; no post-lift perception gate in simple fixed pour",
            }
        else:
            post_lift_obs, post_lift_bottle = _capture_prompt_fallback(
                run_dir=run_dir,
                stage="post_lift_observe",
                role="bottle",
                prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_PROMPTS", ["bottle", "wine bottle"]),
                cameras=["top"],
                min_score=0.0,
            )
            attempt["post_lift_observation"] = post_lift_obs
            attempt["post_lift_bottle_detection"] = post_lift_bottle
            attempt["bottle_lift_verification"] = verify_lift(
                bottle_detection,
                post_lift_bottle,
                min_z_delta_m=_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MIN_Z_DELTA_M", 0.025),
                max_xy_delta_m=_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MAX_XY_DELTA_M", 0.08),
            )
            if _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_POST_LIFT_VERIFY", True) and not attempt[
                "bottle_lift_verification"
            ].get("success"):
                attempt["why_stopped"] = "post-lift observation did not verify a clean bottle lift; stop before dry tilt"
                return attempt

        if (not simple_fixed_pour) and _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_JOINT_TILT", False):
            if _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_DIRECT_ALIGN_ENABLE", False):
                try:
                    direct_align = _execute_direct_xy_align(
                        run_dir,
                        side,
                        post_lift_bottle=post_lift_bottle,
                        cup_detection=cup_detection,
                    )
                except Exception as exc:
                    direct_align = {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                attempt["direct_xy_align"] = direct_align
                if direct_align.get("physical_motion_executed"):
                    attempt["physical_motion_executed"] = True
                    attempt["movement_capable_calls"].append(f"direct_xy_align:{side}")
                if not direct_align.get("success"):
                    recovery = _recover_bottle_to_source_after_pour_failure(
                        freespace_move=freespace_move,
                        side=side,
                        selected_pickup=selected_pickup,
                        failed_label="direct_xy_align",
                    )
                    attempt["failure_recovery"] = recovery
                    attempt["movement_capable_calls"].extend(recovery.get("movement_capable_calls") or [])
                    attempt["why_stopped"] = (
                        "direct_xy_align failed after lift; "
                        f"source-return recovery success={bool(recovery.get('success'))}"
                    )
                    return attempt
                post_align_bottle_obs, post_align_bottle = _capture_prompt_fallback(
                    run_dir=run_dir,
                    stage="post_align_observe",
                    role="bottle",
                    prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_PROMPTS", ["bottle", "wine bottle"]),
                    cameras=["top"],
                    min_score=0.0,
                )
                post_align_cup_obs, post_align_cup = _capture_prompt_fallback(
                    run_dir=run_dir,
                    stage="post_align_observe",
                    role="cup",
                    prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_PROMPTS", ["cup", "glass cup", "mug"]),
                    cameras=["top"],
                    min_score=0.0,
                )
                post_align_verify = _verify_direct_xy_align_visual(
                    run_dir,
                    post_align_bottle=post_align_bottle,
                    post_align_cup=post_align_cup,
                    fallback_cup_detection=cup_detection,
                    direct_align=direct_align,
                )
                attempt["post_align_bottle_observation"] = post_align_bottle_obs
                attempt["post_align_cup_observation"] = post_align_cup_obs
                attempt["post_align_bottle_detection"] = post_align_bottle
                attempt["post_align_cup_detection"] = post_align_cup
                attempt["direct_xy_align_visual_verification"] = post_align_verify
                require_post_align_verify = _truthy_env(
                    "OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_POST_ALIGN_VERIFY",
                    True,
                )
                if require_post_align_verify and not post_align_verify.get("success"):
                    recovery = _recover_bottle_to_source_after_pour_failure(
                        freespace_move=freespace_move,
                        side=side,
                        selected_pickup=selected_pickup,
                        failed_label="direct_xy_align_visual_verify",
                    )
                    attempt["failure_recovery"] = recovery
                    attempt["movement_capable_calls"].extend(recovery.get("movement_capable_calls") or [])
                    attempt["why_stopped"] = (
                        "post-align visual verification failed before direct_joint_tilt; "
                        f"source-return recovery success={bool(recovery.get('success'))}"
                    )
                    return attempt
            try:
                direct_tilt = _execute_direct_joint_tilt(run_dir, side, selected_pour=selected_pour)
            except Exception as exc:
                direct_tilt = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            attempt["direct_joint_tilt"] = direct_tilt
            attempt["movement_capable_calls"].append(f"direct_joint_tilt:{side}")
            if not direct_tilt.get("success"):
                recovery = _recover_bottle_to_source_after_pour_failure(
                    freespace_move=freespace_move,
                    side=side,
                    selected_pickup=selected_pickup,
                    failed_label="direct_joint_tilt",
                )
                attempt["failure_recovery"] = recovery
                attempt["movement_capable_calls"].extend(recovery.get("movement_capable_calls") or [])
                attempt["why_stopped"] = (
                    "direct_joint_tilt failed after lift; "
                    f"source-return recovery success={bool(recovery.get('success'))}"
                )
                return attempt

        keep_held_after_pour = not return_to_source_after_pour
        post_pour_specs = list(_pour_motion_specs(selected_pour))
        if not keep_held_after_pour:
            post_pour_specs.append(("return_to_source_grasp_pose", selected_pickup["grasp_pose"]))

        for label, pose in post_pour_specs:
            step = _execute_pose(freespace_move, side, pose, label=label)
            attempt["steps"].append(step)
            attempt["physical_motion_executed"] = True
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            if not step.get("ok"):
                if simple_fixed_pour:
                    attempt["why_stopped"] = (
                        f"{label} preview/execute failed in simple fixed pour; "
                        "stopped without extra recovery motion because the bottle may still be held"
                    )
                    return attempt
                recovery = _recover_bottle_to_source_after_pour_failure(
                    freespace_move=freespace_move,
                    side=side,
                    selected_pickup=selected_pickup,
                    failed_label=label,
                )
                attempt["failure_recovery"] = recovery
                attempt["movement_capable_calls"].extend(recovery.get("movement_capable_calls") or [])
                attempt["why_stopped"] = (
                    f"{label} preview/execute failed; "
                    f"source-return recovery success={bool(recovery.get('success'))}"
                )
                return attempt
            if label == "dry_tilt_over_cup":
                hold_s = max(0.0, _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_HOLD_S", 0.75))
                if hold_s > 0.0:
                    time.sleep(hold_s)
                step["hold_s"] = hold_s

        if keep_held_after_pour:
            if simple_fixed_pour:
                attempt["held_after_pour"] = True
                attempt["post_pour_hold_observation"] = {
                    "skipped": True,
                    "reason": "simple fixed pour uses no live detections after the initial observation",
                }
                attempt["cup_motion_verification"] = {
                    "success": True,
                    "skipped": True,
                    "reason": "no post-pour perception gate in simple fixed pour",
                }
                attempt["success"] = True
                attempt["why_stopped"] = "simple fixed dry pour completed; bottle remains held for immediate rack placement"
                return attempt
            final_bottle_obs, final_bottle = _capture_prompt_fallback(
                run_dir=run_dir,
                stage="post_pour_hold_observe",
                role="bottle",
                prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_PROMPTS", ["bottle", "wine bottle"]),
                cameras=["top"],
                min_score=0.0,
            )
            final_cup_obs, final_cup = _capture_prompt_fallback(
                run_dir=run_dir,
                stage="post_pour_hold_observe",
                role="cup",
                prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_PROMPTS", ["cup", "plastic cup", "empty cup"]),
                cameras=["top"],
                min_score=0.0,
            )
            cup_motion = compare_detection_motion(cup_detection, final_cup)
            cup_max_xy_delta_m = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MAX_CUP_XY_DELTA_M", 0.04)
            cup_max_abs_z_delta_m = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MAX_CUP_ABS_Z_DELTA_M", 0.04)
            cup_motion_ok = bool(cup_motion.get("ok"))
            if cup_motion_ok:
                cup_motion_ok = (
                    float(cup_motion.get("xy_delta_m") or 0.0) <= cup_max_xy_delta_m
                    and abs(float(cup_motion.get("z_delta_m") or 0.0)) <= cup_max_abs_z_delta_m
                )
            attempt["held_after_pour"] = True
            attempt["post_pour_hold_bottle_observation"] = final_bottle_obs
            attempt["post_pour_hold_cup_observation"] = final_cup_obs
            attempt["post_pour_hold_bottle_detection"] = final_bottle
            attempt["post_pour_hold_cup_detection"] = final_cup
            attempt["post_pour_hold_motion"] = {"cup": cup_motion}
            attempt["post_pour_hold_thresholds"] = {
                "max_cup_xy_delta_m": cup_max_xy_delta_m,
                "max_cup_abs_z_delta_m": cup_max_abs_z_delta_m,
                "require_cup_verify": False,
            }
            attempt["cup_motion_verification"] = {
                "success": cup_motion_ok,
                "reason": (
                    "cup remained within post-pour motion threshold"
                    if cup_motion_ok
                    else "cup detection missing or moved beyond post-pour threshold"
                ),
            }
            attempt["success"] = True
            attempt["why_stopped"] = (
                "dry pour completed; bottle remains held for immediate rack placement"
            )
            return attempt

        release = _open_gripper_until(
            side,
            min_pos=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_MIN_POS", 0.95),
            timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_TIMEOUT_S", 12.0),
            poll_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_POLL_S", 0.25),
            retry_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_BEFORE_RETREAT_RETRY_S", 1.5),
        )
        attempt["release_open"] = release
        attempt["movement_capable_calls"].append(f"open_gripper:{side}")
        if not release.get("success"):
            attempt["why_stopped"] = str(release.get("why_stopped") or "release open failed")
            return attempt

        if _truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_RETREAT_TO_PREGRASP_AFTER_RELEASE", True):
            step = _execute_pose(freespace_move, side, selected_pickup["pregrasp_pose"], label="retreat_to_source_pregrasp_pose")
            attempt["steps"].append(step)
            attempt["movement_capable_calls"].append("freespace_move:retreat_to_source_pregrasp_pose")
            if not step.get("ok"):
                attempt["why_stopped"] = "retreat to source pregrasp failed after release"
                return attempt

        final_obs, final_bottle = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="post_release_observe",
            role="bottle",
            prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_PROMPTS", ["bottle", "wine bottle"]),
            cameras=["top"],
            min_score=0.0,
        )
        _, final_cup = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="post_release_observe",
            role="cup",
            prompts=_csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_PROMPTS", ["cup", "plastic cup", "empty cup"]),
            cameras=["top"],
            min_score=0.0,
        )
        attempt["post_release_observation"] = final_obs
        attempt["post_release_bottle_detection"] = final_bottle
        attempt["post_release_cup_detection"] = final_cup
        bottle_motion = compare_detection_motion(bottle_detection, final_bottle)
        cup_motion = compare_detection_motion(cup_detection, final_cup)
        cup_max_xy_delta_m = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MAX_CUP_XY_DELTA_M", 0.04)
        cup_max_abs_z_delta_m = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MAX_CUP_ABS_Z_DELTA_M", 0.04)
        cup_motion_ok = bool(cup_motion.get("ok"))
        if cup_motion_ok:
            cup_motion_ok = (
                float(cup_motion.get("xy_delta_m") or 0.0) <= cup_max_xy_delta_m
                and abs(float(cup_motion.get("z_delta_m") or 0.0)) <= cup_max_abs_z_delta_m
            )
        attempt["post_release_motion"] = {
            "bottle": bottle_motion,
            "cup": cup_motion,
        }
        attempt["post_release_thresholds"] = {
            "max_cup_xy_delta_m": cup_max_xy_delta_m,
            "max_cup_abs_z_delta_m": cup_max_abs_z_delta_m,
        }
        attempt["cup_motion_verification"] = {
            "success": cup_motion_ok,
            "reason": (
                "cup remained within post-release motion threshold"
                if cup_motion_ok
                else "cup moved too much for a liquid-safe dry-pour rehearsal"
            ),
        }
        attempt["success"] = bool(final_bottle is not None and final_cup is not None and cup_motion_ok)
        if attempt["success"]:
            attempt["why_stopped"] = "dry bottle-to-cup pour rehearsal completed; no liquid used"
        elif final_bottle is None or final_cup is None:
            attempt["why_stopped"] = "dry motions completed, but final object detections were inconclusive"
        else:
            attempt["why_stopped"] = "dry motions completed, but cup moved too much for liquid-safe geometry"
        if _truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_GO_HOME_AFTER_RELEASE", False):
            go_home = _tool("go_home")
            if callable(go_home):
                attempt["go_home_after_release"] = json_safe(go_home())
                attempt["movement_capable_calls"].append("go_home")
                if _truthy_env_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME", True):
                    post_home_open = _open_gripper_until(
                        side,
                        min_pos=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_MIN_POS", 0.90),
                        timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_TIMEOUT_S", 12.0),
                        poll_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_POLL_S", 0.25),
                        retry_s=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_AFTER_HOME_RETRY_S", 1.5),
                    )
                    attempt["open_after_home"] = post_home_open
                    attempt["movement_capable_calls"].append(f"open_gripper:{side}")
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
    finally:
        attempt["path"] = write_json(run_dir / "plans" / "physical_attempt_01.json", attempt)
        append_stage_summary(
            run_dir,
            [
                "## physical dry pour attempt 01",
                f"- success: {attempt['success']}",
                f"- physical_motion_executed: {attempt['physical_motion_executed']}",
                f"- why_stopped: {attempt['why_stopped']}",
            ],
        )
    return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.dry_bottle_to_cup_pour.result.v1",
        "success": False,
        "plan_success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "dry_run": True,
        "liquid_used": False,
        "cup_is_stationary": True,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False),
        "physical_ticket_status": None,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "attempt_count": 0,
        "bottle_detection": None,
        "cup_detection": None,
        "bottle_plan": None,
        "pour_candidates_path": None,
        "sequence_preview": None,
        "physical_attempt": None,
        "why_stopped": None,
        "next_recommendation": None,
        "video_paths": [],
    }


def _run(*, return_to_source_after_pour: bool = True, simple_fixed_pour: bool = False) -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    result["simple_fixed_pour"] = bool(simple_fixed_pour)
    try:
        manual_env = _manual_pose_env_vars_present()
        if manual_env:
            result["manual_pose_env_vars_present"] = manual_env
            result["why_stopped"] = "manual XYZ env vars are not allowed for this live bottle/cup dry-pour task"
            result["next_recommendation"] = "Unset manual target env vars and rerun with fresh top-camera detections."
            return result

        bottle_prompts = _csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_PROMPTS", ["bottle", "wine bottle"])
        cup_prompts = _csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_PROMPTS", ["cup", "plastic cup", "empty cup"])
        cameras = _csv("OPENFORGE_BOTTLE_CUP_DRY_POUR_CAMERAS", ["top"])
        min_score = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MIN_DETECTION_SCORE", 0.25)
        bottle_obs, bottle_det = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="pre_observe",
            role="bottle",
            prompts=bottle_prompts,
            cameras=cameras,
            min_score=min_score,
        )
        cup_obs, cup_det = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="pre_observe",
            role="cup",
            prompts=cup_prompts,
            cameras=cameras,
            min_score=min_score,
        )
        result["pre_observation"] = {"bottle": bottle_obs, "cup": cup_obs}
        result["bottle_detection"] = bottle_det
        result["cup_detection"] = cup_det
        detection_sanity = {
            "bottle": _validate_detection(bottle_det, role="bottle"),
            "cup": _validate_detection(cup_det, role="cup"),
        }
        result["detection_sanity"] = detection_sanity
        append_stage_summary(
            run_dir,
            [
                "## live detection diagnostics",
                f"- bottle: {detection_sanity['bottle']}",
                f"- cup: {detection_sanity['cup']}",
            ],
        )
        failed = [role for role, packet in detection_sanity.items() if not packet.get("passed")]
        if failed:
            result["why_stopped"] = "live top-camera detection sanity failed for " + ", ".join(failed)
            result["next_recommendation"] = "Inspect overlays and rerun no-motion after improving bottle/cup visibility."
            return result

        xy_dist = _xy_distance(bottle_det, cup_det)
        result["bottle_cup_xy_distance_m"] = xy_dist
        min_sep = _env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_MIN_BOTTLE_CUP_XY_SEPARATION_M", 0.06)
        if (
            _truthy_env("OPENFORGE_BOTTLE_CUP_DRY_POUR_ENFORCE_BOTTLE_CUP_SEPARATION", True)
            and xy_dist is not None
            and xy_dist < min_sep
        ):
            result["why_stopped"] = "live bottle/cup detections are too close to trust as distinct objects"
            result["next_recommendation"] = "Separate bottle and cup, then rerun no-motion."
            return result

        freespace_move = _required_tool("freespace_move")
        bottle_candidates = _build_bottle_candidates(bottle_det or {})
        result["candidate_counts"] = {"bottle": len(bottle_candidates)}
        result["candidate_artifacts"] = {
            "bottle_candidates": write_json(run_dir / "plans" / "bottle_side_grasp_candidates.json", bottle_candidates)
        }
        bottle_plan = rank_motion_candidates(
            candidates=bottle_candidates[: max(1, _env_int("OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_PREVIEW_CANDIDATES", 96))],
            freespace_move=freespace_move,
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="bottle_side_grasp_preview",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_BOTTLE_CUP_DRY_POUR_PREVIEW_TIMEOUT_S", 35.0),
            planner_backend=_env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNER_BACKEND", "rrtconnect"),
            solver_speed=_env_str_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_SOLVER_SPEED", "slow"),
            planning_speed=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNING_SPEED", 0.40),
            ik_error_threshold=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ERROR_THRESHOLD_M", 0.015),
            ik_rot_threshold_deg=_env_float_pickup("OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ROT_THRESHOLD_DEG", 8.0),
        )
        selected_pickup = bottle_plan.get("selected")
        result["bottle_plan"] = {
            "plan_packet": bottle_plan.get("plan_path"),
            "selected": selected_pickup,
            "preview_count": len(bottle_plan.get("candidates") or []),
        }
        if selected_pickup is None:
            result["why_stopped"] = "no left-arm bottle side-grasp candidate passed pickup/lift preview"
            result["next_recommendation"] = "Inspect bottle plan packet; tune bottle yaw/z/standoff in no-motion."
            return result

        pour_candidates = _make_pour_candidates(selected_pickup, cup_det or {})
        if simple_fixed_pour:
            pour_candidates = [
                _simple_fixed_pour_candidate(
                    selected_pickup,
                    bottle_det or {},
                    cup_det or {},
                    reference_candidate=pour_candidates[0] if pour_candidates else None,
                )
            ]
        result["pour_candidates_path"] = write_json(
            run_dir / "plans" / "dry_bottle_to_cup_pour_candidates.json",
            {
                "schema": "openforge.dry_bottle_to_cup_pour.candidates.v1",
                "selected_pickup": selected_pickup,
                "cup_detection": cup_det,
                "candidate_count": len(pour_candidates),
                "candidates": pour_candidates,
            },
        )
        sequence = _preview_pour_sequence(
            run_dir,
            selected_pickup,
            pour_candidates,
            return_to_source_after_pour=return_to_source_after_pour,
        )
        result["sequence_preview"] = sequence
        result["selected_strategy"] = {
            "bottle_pickup": selected_pickup,
            "dry_pour": sequence.get("selected_pour"),
            "sequence_preview_path": sequence.get("path"),
        }
        result["plan_success"] = bool(sequence.get("success"))
        ticket = _ticket_status(run_dir)
        result["physical_ticket_status"] = ticket

        if result["plan_success"] and result["physical_gate"] and ticket.get("passed"):
            attempt = _execute_physical_dry_pour(
                run_dir=run_dir,
                selected_pickup=selected_pickup,
                selected_pour=sequence["selected_pour"],
                bottle_detection=bottle_det or {},
                cup_detection=cup_det or {},
                return_to_source_after_pour=return_to_source_after_pour,
                simple_fixed_pour=simple_fixed_pour,
            )
            result["attempt_count"] = 1
            result["physical_attempt"] = attempt
            result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
            result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
            result["success"] = bool(attempt.get("success"))
            result["reward"] = 1.0 if result["success"] else 0.0
            result["why_stopped"] = attempt.get("why_stopped")
            result["next_recommendation"] = (
                (
                    "Continue directly to rack placement while the bottle is still held."
                    if not return_to_source_after_pour
                    else "Stop; dry bottle-to-cup pour rehearsal completed."
                )
                if result["success"]
                else "Inspect physical_attempt_01.json and videos before any retry."
            )
        elif result["plan_success"]:
            result["success"] = True
            result["reward"] = 1.0
            result["why_stopped"] = "no-motion live observe/plan/preview succeeded; no hardware motion executed"
            result["next_recommendation"] = (
                "For physical dry-pour motion, rerun with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and the "
                "OPENFORGE_BOTTLE_CUP_DRY_POUR_* dry ticket gates. No liquid is authorized by this script."
            )
        else:
            failed = [
                (item.get("label"), item.get("failed_stage"))
                for item in (sequence.get("attempts") or [])
                if not item.get("success")
            ]
            result["why_stopped"] = f"pickup preview succeeded, but dry pour sequence preview failed for {failed}"
            result["next_recommendation"] = "Inspect dry_bottle_to_cup_pour_sequence_preview.json and tune standoff/tilt/clearance."
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and run log before retrying no-motion."
        result["error"] = result["why_stopped"]
    finally:
        result["video_paths"] = _video_paths(Path(run_dir))
        write_json(Path(run_dir) / "result.json", result)
        write_json(Path(run_dir) / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- why_stopped: {result['why_stopped']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))
    return result


if not globals().get("_OPENFORGE_IMPORT_ONLY", False):
    TASK_RESULT.update(_run())
