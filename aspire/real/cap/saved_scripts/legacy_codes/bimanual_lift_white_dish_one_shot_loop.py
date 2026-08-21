# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-arm white-dish lift test for real YAM.

Fresh top-camera localization -> opposing rim waypoint preview -> optional
physical bimanual lift -> return/release. This script deliberately does not go
home; run a separate recovery/home helper after each physical test.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    read_gripper_width,
    verify_lift,
    write_json,
)


TASK_NAME = "bimanual_lift_white_dish_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
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


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _normalize_angle_deg(angle: float) -> float:
    value = float(angle)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _unit3(values: Any, fallback: tuple[float, float, float]) -> list[float]:
    try:
        raw = [float(v) for v in list(values)[:3]]
    except Exception:
        raw = list(fallback)
    if len(raw) < 3:
        raw = list(fallback)
    norm = math.sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2])
    if norm < 1e-8:
        raw = list(fallback)
        norm = math.sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2])
    return [raw[0] / norm, raw[1] / norm, raw[2] / norm]


def _display_rpy_from_axes(opening_axis: Any, approach_axis_hint: Any) -> list[float]:
    """Return display RPY whose local X is opening and local Z is approach."""
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation

        x_axis = np.asarray(_unit3(opening_axis, (1.0, 0.0, 0.0)), dtype=float)
        z_hint = np.asarray(_unit3(approach_axis_hint, (0.0, 0.0, -1.0)), dtype=float)
        z_axis = z_hint - x_axis * float(np.dot(z_hint, x_axis))
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-8:
            z_hint = np.asarray([0.0, 0.0, -1.0])
            z_axis = z_hint - x_axis * float(np.dot(z_hint, x_axis))
            z_norm = float(np.linalg.norm(z_axis))
        z_axis = z_axis / z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / float(np.linalg.norm(y_axis))
        matrix = np.column_stack([x_axis, y_axis, z_axis])
        euler_xyz = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
        roll, pitch, yaw = [float(euler_xyz[1]), float(-euler_xyz[0]), float(-euler_xyz[2] - 90.0)]
        return [_normalize_angle_deg(roll), _normalize_angle_deg(pitch), _normalize_angle_deg(yaw)]
    except Exception:
        return [0.0, -180.0, 90.0]


def _motion_common_kwargs(*, preview_only: bool) -> dict[str, Any]:
    return {
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_DISH_BIMANUAL_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_DISH_BIMANUAL_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_DISH_BIMANUAL_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_DISH_BIMANUAL_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_DISH_BIMANUAL_IK_ROT_THRESHOLD_DEG", 18.0),
        "ik_xyz_weight": _env_float("OPENFORGE_DISH_BIMANUAL_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_DISH_BIMANUAL_IK_RPY_WEIGHT", 0.20),
    }


def _waypoint_kwargs(waypoint: dict[str, Any], *, preview_only: bool) -> dict[str, Any]:
    kwargs = _motion_common_kwargs(preview_only=preview_only)
    for side in ("left", "right"):
        pose = waypoint.get(side)
        if pose:
            kwargs[f"{side}_target_pos"] = [float(v) for v in pose["position"][:3]]
            kwargs[f"{side}_target_rpy"] = [float(v) for v in pose["rpy"][:3]]
            kwargs[f"{side}_gripper"] = float(
                waypoint.get(f"{side}_gripper", _env_float("OPENFORGE_DISH_BIMANUAL_APPROACH_GRIPPER_POS", 0.68))
            )
    return kwargs


def _side_waypoint_kwargs(waypoint: dict[str, Any], side: str, *, preview_only: bool) -> dict[str, Any]:
    kwargs = _motion_common_kwargs(preview_only=preview_only)
    pose = waypoint.get(side)
    if not pose:
        raise RuntimeError(f"waypoint {waypoint.get('name')} has no {side} pose")
    kwargs[f"{side}_target_pos"] = [float(v) for v in pose["position"][:3]]
    kwargs[f"{side}_target_rpy"] = [float(v) for v in pose["rpy"][:3]]
    kwargs[f"{side}_gripper"] = float(
        waypoint.get(f"{side}_gripper", _env_float("OPENFORGE_DISH_BIMANUAL_APPROACH_GRIPPER_POS", 0.68))
    )
    return kwargs


def _raw_preview_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", False))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    success = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    return {
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "trajectory_cache_key": _field(result, "trajectory_cache_key", None),
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "success": success,
        "raw": json_safe(result),
    }


def _raw_execute_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", True))
    success = bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed)
    return {
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "success": success,
        "raw": json_safe(result),
    }


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_PLATE_XYZ",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
        "OPENFORGE_BOWL_RACK_MANUAL_XYZ",
        "OPENFORGE_DISH_BIMANUAL_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_path = Path("cap/saved_scripts/bimanual_lift_white_dish_one_shot_loop.py").resolve()
    snapshot = {
        "script_path": str(script_path),
        "task_name": TASK_NAME,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_DISH_BIMANUAL_")
            or name in {"OPENFORGE_ALLOW_PHYSICAL_MOTION", "YAM_STATION_CALIBRATED_XML"}
        },
        "git_head": None,
        "git_status_short": None,
    }
    try:
        snapshot["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        snapshot["git_status_short"] = subprocess.check_output(
            ["git", "status", "--short", "--", str(script_path.relative_to(Path.cwd()))],
            text=True,
        ).strip()
    except Exception as exc:
        snapshot["git_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["path"] = write_json(run_dir / "source_snapshot.json", snapshot)
    return snapshot


def _object_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_DISH_BIMANUAL_PROMPTS",
        ["white dish", "dish", "white plate", "plate", "dinner plate"],
    )


def _prompt_rank(detection: dict[str, Any]) -> int:
    text = f"{detection.get('prompt') or ''} {detection.get('label') or ''}".lower()
    if "white dish" in text:
        return 0
    if "dish" in text:
        return 1
    if "white plate" in text:
        return 2
    if "plate" in text:
        return 3
    return 4


def _touches_image_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_DISH_BIMANUAL_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_DISH_BIMANUAL_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_DISH_BIMANUAL_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _select_dish_detection(observation: dict[str, Any]) -> dict[str, Any] | None:
    min_score = _env_float("OPENFORGE_DISH_BIMANUAL_MIN_DETECTION_SCORE", 0.0)
    candidates = [
        det
        for det in list(observation.get("all_detections") or [])
        if det.get("position_3d")
        and len(det.get("position_3d") or []) >= 3
        and float(det.get("score") or 0.0) >= min_score
        and not _touches_image_edge(det)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda det: (_prompt_rank(det), -float(det.get("score") or 0.0)))[0]


def _radius_estimate(detection: dict[str, Any]) -> dict[str, Any]:
    half_extents = detection.get("half_extents") or []
    raw_radius: float | None = None
    source = "fallback_reference_radius"
    if len(half_extents) >= 2:
        values = [abs(float(half_extents[0])), abs(float(half_extents[1]))]
        candidate = max(values)
        if 0.04 <= candidate <= 0.18:
            raw_radius = candidate
            source = "detection_half_extents_xy"
    if raw_radius is None:
        box = detection.get("box_2d") or []
        if len(box) >= 4:
            long_axis = max(abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1])))
            ref_px = _env_float("OPENFORGE_DISH_BIMANUAL_REFERENCE_BOX_LONG_AXIS_PX", 56.0)
            ref_radius = _env_float("OPENFORGE_DISH_BIMANUAL_REFERENCE_RADIUS_M", 0.095)
            if long_axis > 1.0 and ref_px > 1.0:
                raw_radius = ref_radius * long_axis / ref_px
                source = "top_box_scaled_from_reference_radius"
    if raw_radius is None:
        raw_radius = _env_float("OPENFORGE_DISH_BIMANUAL_REFERENCE_RADIUS_M", 0.095)
    min_radius = _env_float("OPENFORGE_DISH_BIMANUAL_MIN_RADIUS_M", 0.075)
    max_radius = _env_float("OPENFORGE_DISH_BIMANUAL_MAX_RADIUS_M", 0.135)
    radius = max(min_radius, min(max_radius, float(raw_radius)))
    return {
        "radius_m": radius,
        "raw_radius_m": float(raw_radius),
        "clipped": abs(radius - float(raw_radius)) > 1e-6,
        "source": source,
        "half_extents": _round_list(half_extents, limit=3),
        "box_2d": json_safe(detection.get("box_2d") or []),
    }


def _pose(position: list[float], rpy: list[float], label: str) -> dict[str, Any]:
    return {"position": _round_list(position, limit=3), "rpy": _round_list(rpy, limit=3), "label": label}


def _build_candidate(
    detection: dict[str, Any],
    *,
    rim_pair_axis: str,
    assignment: str,
    rpy_mode: str,
    radius_bias: float,
    tcp_above_rim: float,
    index: int,
) -> dict[str, Any]:
    center = [float(v) for v in list(detection["position_3d"])[:3]]
    radius_packet = _radius_estimate(detection)
    radius = max(0.01, float(radius_packet["radius_m"]) + float(radius_bias))
    rim_inset = _env_float("OPENFORGE_DISH_BIMANUAL_RIM_INSET_M", 0.004)
    contact_radius = max(0.02, radius - rim_inset)
    rim_z = center[2] + _env_float("OPENFORGE_DISH_BIMANUAL_RIM_Z_BIAS_M", 0.020)
    pre_z = _env_float("OPENFORGE_DISH_BIMANUAL_PREGRASP_Z_LIFT_M", 0.080)
    low_z = _env_float("OPENFORGE_DISH_BIMANUAL_LOW_PREGRASP_Z_LIFT_M", 0.030)
    lift_z = _env_float("OPENFORGE_DISH_BIMANUAL_LIFT_Z_M", 0.080)
    if rim_pair_axis == "x":
        left_radial = [-1.0, 0.0, 0.0]
        right_radial = [1.0, 0.0, 0.0]
    else:
        rim_pair_axis = "y"
        left_radial = [0.0, 1.0, 0.0]
        right_radial = [0.0, -1.0, 0.0]
    if assignment == "swapped":
        left_radial, right_radial = right_radial, left_radial
    else:
        assignment = "natural"
    approach_axis = [0.0, 0.0, -1.0]
    left_rpy_axis = list(left_radial)
    right_rpy_axis = list(right_radial)
    if rpy_mode in {"left_flipped", "both_flipped"}:
        left_rpy_axis = [-left_rpy_axis[0], -left_rpy_axis[1], -left_rpy_axis[2]]
    if rpy_mode in {"right_flipped", "both_flipped"}:
        right_rpy_axis = [-right_rpy_axis[0], -right_rpy_axis[1], -right_rpy_axis[2]]
    if rpy_mode not in {"nominal", "left_flipped", "right_flipped", "both_flipped"}:
        rpy_mode = "nominal"
    left_rpy = _display_rpy_from_axes(left_rpy_axis, approach_axis)
    right_rpy = _display_rpy_from_axes(right_rpy_axis, approach_axis)
    left_contact = [
        center[0] + left_radial[0] * contact_radius,
        center[1] + left_radial[1] * contact_radius,
        rim_z,
    ]
    right_contact = [
        center[0] + right_radial[0] * contact_radius,
        center[1] + right_radial[1] * contact_radius,
        rim_z,
    ]
    grasp_z = rim_z + float(tcp_above_rim)
    left_grasp = [left_contact[0], left_contact[1], grasp_z]
    right_grasp = [right_contact[0], right_contact[1], grasp_z]
    left_high = [left_grasp[0], left_grasp[1], grasp_z + pre_z]
    right_high = [right_grasp[0], right_grasp[1], grasp_z + pre_z]
    left_low = [left_grasp[0], left_grasp[1], grasp_z + low_z]
    right_low = [right_grasp[0], right_grasp[1], grasp_z + low_z]
    left_lift = [left_grasp[0], left_grasp[1], grasp_z + lift_z]
    right_lift = [right_grasp[0], right_grasp[1], grasp_z + lift_z]
    separation = math.dist(left_grasp, right_grasp)
    open_width = _env_float("OPENFORGE_DISH_BIMANUAL_APPROACH_GRIPPER_POS", 0.68)
    hold_width = _env_float("OPENFORGE_DISH_BIMANUAL_HOLD_GRIPPER_POS", 0.0)
    waypoints = [
        {
            "name": "bimanual_high_pregrasp",
            "left": _pose(left_high, left_rpy, "left_high_pregrasp"),
            "right": _pose(right_high, right_rpy, "right_high_pregrasp"),
            "left_gripper": open_width,
            "right_gripper": open_width,
        },
        {
            "name": "bimanual_low_pregrasp",
            "left": _pose(left_low, left_rpy, "left_low_pregrasp"),
            "right": _pose(right_low, right_rpy, "right_low_pregrasp"),
            "left_gripper": open_width,
            "right_gripper": open_width,
        },
        {
            "name": "bimanual_grasp_pose",
            "left": _pose(left_grasp, left_rpy, "left_grasp_pose"),
            "right": _pose(right_grasp, right_rpy, "right_grasp_pose"),
            "left_gripper": open_width,
            "right_gripper": open_width,
        },
        {
            "name": "bimanual_lift",
            "left": _pose(left_lift, left_rpy, "left_lift"),
            "right": _pose(right_lift, right_rpy, "right_lift"),
            "left_gripper": hold_width,
            "right_gripper": hold_width,
        },
        {
            "name": "bimanual_return_to_grasp_height",
            "left": _pose(left_grasp, left_rpy, "left_return"),
            "right": _pose(right_grasp, right_rpy, "right_return"),
            "left_gripper": hold_width,
            "right_gripper": hold_width,
        },
        {
            "name": "bimanual_release_retreat",
            "left": _pose(left_high, left_rpy, "left_release_retreat"),
            "right": _pose(right_high, right_rpy, "right_release_retreat"),
            "left_gripper": open_width,
            "right_gripper": open_width,
        },
    ]
    target_radius = _env_float("OPENFORGE_DISH_BIMANUAL_REFERENCE_RADIUS_M", 0.095)
    grasp_frame = "front_back_x"
    pair_penalty = 0.08
    if rim_pair_axis == "y":
        grasp_frame = "left_right_y"
        pair_penalty = 0.0
    target_contact_radius = _env_float(
        "OPENFORGE_DISH_BIMANUAL_TARGET_CONTACT_RADIUS_M",
        0.080 if rim_pair_axis == "y" else target_radius,
    )
    assignment_penalty = 0.01 if assignment == "swapped" else 0.0
    rpy_penalty = {"nominal": 0.0, "left_flipped": 0.005, "right_flipped": 0.008, "both_flipped": 0.012}.get(
        rpy_mode, 0.02
    )
    return {
        "label": f"white_dish_bimanual_{rim_pair_axis}_{assignment}_{rpy_mode}_rim_{index}",
        "strategy_type": "bimanual_opposing_top_down_rim_straddle",
        "rim_pair_axis": rim_pair_axis,
        "grasp_frame": grasp_frame,
        "assignment": assignment,
        "rpy_mode": rpy_mode,
        "source_detection": detection,
        "center_m": _round_list(center, limit=3),
        "radius_estimate": radius_packet,
        "radius_bias_m": round(float(radius_bias), 5),
        "contact_radius_m": round(contact_radius, 5),
        "target_contact_radius_m": round(target_contact_radius, 5),
        "rim_z_m": round(rim_z, 5),
        "tcp_above_rim_m": round(float(tcp_above_rim), 5),
        "tcp_height_policy": "raised above prior 0.037 m single-arm grasp to reduce deep contact",
        "left_rim_contact_m": _round_list(left_contact, limit=3),
        "right_rim_contact_m": _round_list(right_contact, limit=3),
        "left_opening_axis_world": left_radial,
        "right_opening_axis_world": right_radial,
        "left_rpy_opening_axis_world": left_rpy_axis,
        "right_rpy_opening_axis_world": right_rpy_axis,
        "approach_axis_world": approach_axis,
        "gripper_separation_m": round(separation, 5),
        "min_gripper_separation_m": _env_float("OPENFORGE_DISH_BIMANUAL_MIN_GRIPPER_SEPARATION_M", 0.120),
        "waypoints": waypoints,
        "score": (
            1.0
            - pair_penalty
            - assignment_penalty
            - rpy_penalty
            - abs(float(tcp_above_rim) - 0.065) * 5.0
            - abs(contact_radius - target_contact_radius) * 2.0
        ),
    }


def _build_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    tcp_values = _float_csv("OPENFORGE_DISH_BIMANUAL_TCP_ABOVE_RIM_VALUES_M", [0.065, 0.058, 0.052])
    radius_biases = _float_csv("OPENFORGE_DISH_BIMANUAL_RADIUS_BIASES_M", [-0.045, -0.055, -0.032, -0.020, 0.0])
    pair_axes = [axis.strip().lower() for axis in _csv("OPENFORGE_DISH_BIMANUAL_RIM_PAIR_AXES", ["y", "x"])]
    assignments = [
        item.strip().lower()
        for item in _csv("OPENFORGE_DISH_BIMANUAL_ASSIGNMENTS", ["natural", "swapped"])
        if item.strip().lower() in {"natural", "swapped"}
    ]
    rpy_modes = [
        item.strip().lower()
        for item in _csv("OPENFORGE_DISH_BIMANUAL_RPY_MODES", ["nominal", "left_flipped", "right_flipped", "both_flipped"])
        if item.strip().lower() in {"nominal", "left_flipped", "right_flipped", "both_flipped"}
    ]
    candidates: list[dict[str, Any]] = []
    index = 0
    for tcp_above in tcp_values:
        for radius_bias in radius_biases:
            for pair_axis in pair_axes:
                if pair_axis not in {"x", "y"}:
                    continue
                for assignment in assignments or ["natural", "swapped"]:
                    for rpy_mode in rpy_modes or ["nominal", "left_flipped", "right_flipped", "both_flipped"]:
                        candidate = _build_candidate(
                            detection,
                            rim_pair_axis=pair_axis,
                            assignment=assignment,
                            rpy_mode=rpy_mode,
                            radius_bias=radius_bias,
                            tcp_above_rim=tcp_above,
                            index=index,
                        )
                        index += 1
                        if candidate["gripper_separation_m"] >= candidate["min_gripper_separation_m"]:
                            candidates.append(candidate)
                        if len(candidates) >= _env_int("OPENFORGE_DISH_BIMANUAL_MAX_CANDIDATES", 96):
                            return candidates
    return candidates


def _preview_candidate(
    *,
    freespace_move: Callable[..., Any],
    candidate: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.white_dish_bimanual.preview.v1",
        "candidate_label": candidate["label"],
        "success": False,
        "previews": [],
    }
    required_waypoints = list(candidate["waypoints"])
    if not _truthy_env("OPENFORGE_DISH_BIMANUAL_PREVIEW_RETURN_RELEASE", False):
        required_waypoints = required_waypoints[:4]
    for waypoint in required_waypoints:
        if _truthy_env("OPENFORGE_DISH_BIMANUAL_PREVIEW_SEQUENTIAL_APPROACH", True):
            side_summaries = []
            for side in ("left", "right"):
                kwargs = _side_waypoint_kwargs(waypoint, side, preview_only=True)
                try:
                    result = freespace_move(**kwargs)
                    summary = _raw_preview_summary(result)
                except Exception as exc:
                    summary = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                side_summaries.append({"side": side, "request": kwargs, "preview": summary})
                if not summary.get("success"):
                    break
            ok = all(bool(item["preview"].get("success")) for item in side_summaries)
            packet["previews"].append(
                {
                    "name": waypoint["name"],
                    "preview_policy": "sequential_side_preview",
                    "side_previews": side_summaries,
                    "preview": {"success": ok},
                }
            )
            if not ok:
                failed = next(item for item in side_summaries if not item["preview"].get("success"))
                packet["failed_waypoint"] = f"{waypoint['name']}:{failed['side']}"
                packet["why_stopped"] = (
                    failed["preview"].get("reason") or failed["preview"].get("error") or "side preview failed"
                )
                break
        else:
            kwargs = _waypoint_kwargs(waypoint, preview_only=True)
            try:
                result = freespace_move(**kwargs)
                summary = _raw_preview_summary(result)
            except Exception as exc:
                summary = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            packet["previews"].append({"name": waypoint["name"], "request": kwargs, "preview": summary})
            if not summary.get("success"):
                packet["failed_waypoint"] = waypoint["name"]
                packet["why_stopped"] = summary.get("reason") or summary.get("error") or "preview failed"
                break
    else:
        packet["success"] = True
        packet["why_stopped"] = "all required bimanual waypoints previewed"
    packet["path"] = write_json(run_dir / "plans" / f"{candidate['label']}_preview.json", packet)
    return packet


def _select_previewed_candidate(
    *,
    freespace_move: Callable[..., Any],
    candidates: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    previews = []
    selected: dict[str, Any] | None = None
    for candidate in sorted(candidates, key=lambda item: float(item.get("score") or 0.0), reverse=True):
        preview = _preview_candidate(freespace_move=freespace_move, candidate=candidate, run_dir=run_dir)
        candidate["preview"] = {"success": preview["success"], "path": preview["path"]}
        previews.append(preview)
        if preview["success"] and selected is None:
            selected = candidate
            if not _truthy_env("OPENFORGE_DISH_BIMANUAL_PREVIEW_ALL_CANDIDATES", False):
                break
    packet = {
        "schema": "openforge.white_dish_bimanual.selection.v1",
        "success": selected is not None,
        "candidate_count": len(candidates),
        "preview_count": len(previews),
        "selected_label": selected.get("label") if selected else None,
        "selected_candidate": selected,
        "previews": [{"candidate_label": item["candidate_label"], "success": item["success"], "path": item["path"]} for item in previews],
    }
    packet["path"] = write_json(run_dir / "plans" / "bimanual_lift_selection.json", packet)
    return packet


def _execute_waypoint(freespace_move: Callable[..., Any], waypoint: dict[str, Any]) -> dict[str, Any]:
    preview_kwargs = _waypoint_kwargs(waypoint, preview_only=True)
    preview = freespace_move(**preview_kwargs)
    preview_summary = _raw_preview_summary(preview)
    if not preview_summary.get("success") or not preview_summary.get("trajectory_cache_key"):
        raise RuntimeError(f"{waypoint['name']} preview failed before execution: {preview_summary}")
    executed = freespace_move(trajectory_cache_key=preview_summary["trajectory_cache_key"])
    execute_summary = _raw_execute_summary(executed)
    if not execute_summary.get("success"):
        raise RuntimeError(f"{waypoint['name']} execution failed: {execute_summary}")
    return {
        "stage": waypoint["name"],
        "request": preview_kwargs,
        "preview": preview_summary,
        "execute": execute_summary,
        "trajectory_cache_key": preview_summary["trajectory_cache_key"],
    }


def _execute_side_waypoint(freespace_move: Callable[..., Any], waypoint: dict[str, Any], side: str) -> dict[str, Any]:
    preview_kwargs = _side_waypoint_kwargs(waypoint, side, preview_only=True)
    preview = freespace_move(**preview_kwargs)
    preview_summary = _raw_preview_summary(preview)
    if not preview_summary.get("success") or not preview_summary.get("trajectory_cache_key"):
        raise RuntimeError(f"{waypoint['name']}:{side} preview failed before execution: {preview_summary}")
    executed = freespace_move(trajectory_cache_key=preview_summary["trajectory_cache_key"])
    execute_summary = _raw_execute_summary(executed)
    if not execute_summary.get("success"):
        raise RuntimeError(f"{waypoint['name']}:{side} execution failed: {execute_summary}")
    return {
        "stage": f"{waypoint['name']}:{side}",
        "request": preview_kwargs,
        "preview": preview_summary,
        "execute": execute_summary,
        "trajectory_cache_key": preview_summary["trajectory_cache_key"],
    }


def _set_gripper_pair(
    *,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    target: float,
    stage: str,
) -> dict[str, Any]:
    vel = _env_float("OPENFORGE_DISH_BIMANUAL_CLOSE_VEL_LIMIT", 1.0)
    torque = _env_float("OPENFORGE_DISH_BIMANUAL_CLOSE_TORQUE_LIMIT", 1.0)
    packet: dict[str, Any] = {"stage": stage, "target": float(target), "sides": {}}
    for side in ("left", "right"):
        before = read_gripper_width(get_robot_state, side)
        try:
            result = set_gripper(side, float(target), vel_limit=vel, torque_limit=torque)
            ok = True
            error = None
        except Exception as exc:
            result = None
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        after = read_gripper_width(get_robot_state, side)
        packet["sides"][side] = {
            "ok": ok,
            "error": error,
            "before": before,
            "after": after,
            "result": json_safe(result),
        }
    packet["success"] = all(bool(side_packet.get("ok")) for side_packet in packet["sides"].values())
    return packet


def _close_both_for_lift(
    *,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    run_dir: Path,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.white_dish_bimanual.close_both.v1",
        "success": False,
        "steps": [],
        "policy": "coordinated close ramp; commands are issued left/right at each target without long waits",
    }
    settle_s = _env_float("OPENFORGE_DISH_BIMANUAL_CLOSE_STEP_SETTLE_S", 0.05)
    for target in _float_csv("OPENFORGE_DISH_BIMANUAL_CLOSE_STEPS", [0.0]):
        step = _set_gripper_pair(set_gripper=set_gripper, get_robot_state=get_robot_state, target=target, stage=f"close_to_{target:g}")
        packet["steps"].append(step)
        if not step["success"]:
            packet["why_stopped"] = f"close step {target:g} failed"
            break
        time.sleep(max(0.0, settle_s))
    else:
        packet["success"] = True
        packet["why_stopped"] = "both grippers closed for lift"
    packet["final_widths"] = {side: read_gripper_width(get_robot_state, side) for side in ("left", "right")}
    packet["path"] = write_json(run_dir / "plans" / "bimanual_close_both_for_lift.json", packet)
    return packet


def _ticket_status() -> dict[str, Any]:
    required_true = [
        "OPENFORGE_DISH_BIMANUAL_PHYSICAL_TICKET",
        "OPENFORGE_DISH_BIMANUAL_OPERATOR_PRESENT",
        "OPENFORGE_DISH_BIMANUAL_ESTOP_REACHABLE",
        "OPENFORGE_DISH_BIMANUAL_WORKSPACE_CLEAR",
        "OPENFORGE_DISH_BIMANUAL_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_DISH_BIMANUAL_WHITE_DISH_VISIBLE",
        "OPENFORGE_DISH_BIMANUAL_NO_LIQUID",
        "OPENFORGE_DISH_BIMANUAL_NO_MANUAL_XYZ",
        "OPENFORGE_DISH_BIMANUAL_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_DISH_BIMANUAL_FOLLOWER_SERVERS_UP",
        "OPENFORGE_DISH_BIMANUAL_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_DISH_BIMANUAL_TICKET_SCOPE", "").strip()
    if scope != "white_dish_bimanual_lift":
        missing.append("OPENFORGE_DISH_BIMANUAL_TICKET_SCOPE=white_dish_bimanual_lift")
    budget = _env_int("OPENFORGE_DISH_BIMANUAL_PHYSICAL_ATTEMPT_BUDGET", 0)
    if budget != 1:
        missing.append("OPENFORGE_DISH_BIMANUAL_PHYSICAL_ATTEMPT_BUDGET=1")
    manual_env = _manual_pose_env_vars_present()
    if manual_env:
        missing.extend(manual_env)
    return {
        "passed": not missing,
        "missing_or_invalid": missing,
        "ticket_scope": scope,
        "attempt_budget": budget,
    }


def _execute_physical(run_dir: Path, selected: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    attempt: dict[str, Any] = {
        "schema": "openforge.white_dish_bimanual.physical_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "selected_candidate": selected,
        "why_stopped": None,
    }
    waypoints = list(selected["waypoints"])
    open_width = _env_float("OPENFORGE_DISH_BIMANUAL_APPROACH_GRIPPER_POS", 0.68)
    try:
        open_step = _set_gripper_pair(
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=open_width,
            stage="preshape_both_open",
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].extend(["set_gripper:left", "set_gripper:right"])
        attempt["steps"].append(open_step)
        if not open_step["success"]:
            attempt["why_stopped"] = "failed to preshape both grippers open"
            return attempt

        for waypoint in waypoints[:3]:
            for side in ("left", "right"):
                step = _execute_side_waypoint(freespace_move, waypoint, side)
                attempt["steps"].append(step)
                attempt["movement_capable_calls"].append(f"freespace_move:{waypoint['name']}:{side}")

        close = _close_both_for_lift(set_gripper=set_gripper, get_robot_state=get_robot_state, run_dir=run_dir)
        attempt["steps"].append(close)
        attempt["movement_capable_calls"].extend(["set_gripper:left", "set_gripper:right"])
        if not close["success"]:
            attempt["why_stopped"] = "bimanual close failed before lift"
            return attempt

        lift_step = _execute_waypoint(freespace_move, waypoints[3])
        attempt["steps"].append(lift_step)
        attempt["movement_capable_calls"].append("freespace_move:bimanual_lift")

        post_lift = capture_scene(
            prompts=_object_prompts(),
            cameras=_csv("OPENFORGE_DISH_BIMANUAL_POST_OBS_CAMERAS", ["top"]),
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=get_robot_state,
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="post_lift_observe",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DISH_BIMANUAL_OBS_TIMEOUT_S", 6.0),
            max_retries=_env_int("OPENFORGE_DISH_BIMANUAL_DETECT_RETRIES", 1),
            motion_cameras=tuple(_csv("OPENFORGE_DISH_BIMANUAL_MOTION_CAMERAS", ["top"])),
            image_only_cameras=tuple(_csv("OPENFORGE_DISH_BIMANUAL_IMAGE_ONLY_CAMERAS", [])),
        )
        post_det = _select_dish_detection(post_lift)
        attempt["post_lift_observation"] = post_lift
        attempt["post_lift_detection"] = post_det
        attempt["lift_verification"] = verify_lift(
            selected["source_detection"],
            post_det,
            min_z_delta_m=_env_float("OPENFORGE_DISH_BIMANUAL_VERIFY_LIFT_MIN_Z_DELTA_M", 0.020),
            max_xy_delta_m=_env_float("OPENFORGE_DISH_BIMANUAL_VERIFY_LIFT_MAX_XY_DELTA_M", 0.28),
        )

        if _truthy_env("OPENFORGE_DISH_BIMANUAL_RETURN_TO_TABLE", True):
            for waypoint in waypoints[4:5]:
                step = _execute_waypoint(freespace_move, waypoint)
                attempt["steps"].append(step)
                attempt["movement_capable_calls"].append(f"freespace_move:{waypoint['name']}")
            release = _set_gripper_pair(
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                target=open_width,
                stage="release_both_at_table",
            )
            attempt["steps"].append(release)
            attempt["movement_capable_calls"].extend(["set_gripper:left", "set_gripper:right"])
            for waypoint in waypoints[5:6]:
                for side in ("left", "right"):
                    step = _execute_side_waypoint(freespace_move, waypoint, side)
                    attempt["steps"].append(step)
                    attempt["movement_capable_calls"].append(f"freespace_move:{waypoint['name']}:{side}")

        attempt["success"] = bool((attempt.get("lift_verification") or {}).get("success"))
        attempt["why_stopped"] = (
            "bimanual lift verified; returned/released at source height"
            if attempt["success"]
            else "bimanual lift motion completed but perception verification was inconclusive or failed"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.white_dish_bimanual.result.v1",
        "success": False,
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "observation": None,
        "selected_strategy": None,
        "planner_result": None,
        "physical_attempt": None,
        "why_stopped": "not_started",
        "next_recommendation": None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _initial_result(run_dir)
    try:
        _source_snapshot(run_dir)
        if _manual_pose_env_vars_present():
            result["why_stopped"] = "manual XYZ/debug env vars are not allowed for this live dish task"
            result["manual_env_vars"] = _manual_pose_env_vars_present()
            return result

        obs = capture_scene(
            prompts=_object_prompts(),
            cameras=_csv("OPENFORGE_DISH_BIMANUAL_OBS_CAMERAS", ["top"]),
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="pre_observe",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DISH_BIMANUAL_OBS_TIMEOUT_S", 6.0),
            max_retries=_env_int("OPENFORGE_DISH_BIMANUAL_DETECT_RETRIES", 1),
            motion_cameras=tuple(_csv("OPENFORGE_DISH_BIMANUAL_MOTION_CAMERAS", ["top"])),
            image_only_cameras=tuple(_csv("OPENFORGE_DISH_BIMANUAL_IMAGE_ONLY_CAMERAS", [])),
        )
        result["observation"] = obs
        detection = _select_dish_detection(obs)
        if detection is None:
            result["why_stopped"] = "fresh top-camera observation did not produce a valid white dish detection"
            return result

        candidates = _build_candidates(detection)
        result["candidate_count"] = len(candidates)
        if not candidates:
            result["why_stopped"] = "no bimanual rim candidates generated after separation/radius gates"
            return result

        selection = _select_previewed_candidate(
            freespace_move=_required_tool("freespace_move"),
            candidates=candidates,
            run_dir=run_dir,
        )
        result["planner_result"] = selection
        result["plan_success"] = bool(selection.get("success"))
        selected = selection.get("selected_candidate")
        result["selected_strategy"] = selected.get("label") if selected else None
        if not result["plan_success"] or not selected:
            result["why_stopped"] = "no bimanual lift candidate passed preview"
            result["next_recommendation"] = "Inspect bimanual_lift_selection.json and preview failures."
            return result

        if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
            result["success"] = True
            result["why_stopped"] = "recorded no-motion bimanual white-dish lift preview succeeded"
            result["next_recommendation"] = "For physical test, provide the bimanual lift ticket and run physical once."
            return result

        ticket = _ticket_status()
        result["physical_ticket"] = ticket
        if not ticket["passed"]:
            result["why_stopped"] = "physical bimanual lift ticket is incomplete"
            result["next_recommendation"] = "Set the missing OPENFORGE_DISH_BIMANUAL_* ticket env vars and retry once."
            return result

        attempt = _execute_physical(run_dir, selected)
        result["physical_attempt"] = attempt
        result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
        result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
        result["success"] = bool(attempt.get("success"))
        result["why_stopped"] = attempt.get("why_stopped")
        result["next_recommendation"] = "Run recover_open_both_then_home.py after this physical test."
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json and stage_summary.md before retrying."
        return result
    finally:
        result["path"] = write_json(run_dir / "result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## bimanual white-dish lift result",
                f"- success: {result.get('success')}",
                f"- plan_success: {result.get('plan_success')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- selected_strategy: {result.get('selected_strategy')}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- result: {result.get('path')}",
            ],
        )


TASK_RESULT = _run()
print(json.dumps(json_safe(TASK_RESULT), indent=2))
