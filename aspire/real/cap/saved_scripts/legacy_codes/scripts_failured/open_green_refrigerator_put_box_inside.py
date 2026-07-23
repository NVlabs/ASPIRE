# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion first planner for opening the green refrigerator and placing a box.

This script intentionally has no physical execution branch yet.  It observes
fresh live detections, previews candidate handle-pull and box-place poses, and
refuses to run if OPENFORGE_ALLOW_PHYSICAL_MOTION is set.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    current_run_dir,
    generate_side_grasp_candidates,
    json_safe,
    rank_motion_candidates,
    write_json,
)


TASK_NAME = "open_green_refrigerator_put_box_inside"
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


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
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


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_REFRIGERATOR_XYZ",
        "OPENFORGE_REFRIGERATOR_HANDLE_XYZ",
        "OPENFORGE_REFRIGERATOR_DOOR_XYZ",
        "OPENFORGE_BOX_XYZ",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _calibrated_motion_cameras() -> set[str]:
    return {"top", "left", "right"}


def _capture_prompt_fallback(
    *,
    run_dir: Path,
    stage: str,
    object_role: str,
    prompts: list[str],
    cameras: list[str],
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    detect_objects_oneshot = _required_tool("detect_objects_oneshot")
    get_camera_image = _tool("get_camera_image")
    get_robot_state = _tool("get_robot_state")
    run_in_background = _tool("run_in_background")
    packets: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prompt_index, prompt in enumerate(prompts):
        packet = capture_scene(
            prompts=[prompt],
            cameras=cameras,
            detect_objects_oneshot=detect_objects_oneshot,
            get_camera_image=get_camera_image,
            get_robot_state=get_robot_state,
            run_in_background=run_in_background,
            run_dir=run_dir,
            stage=f"{stage}_{object_role}_{prompt.replace(' ', '_')}",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
            max_retries=max(1, _env_int("OPENFORGE_FRIDGE_DETECT_RETRIES", 2)),
            save_overlays=True,
            motion_cameras=tuple(_calibrated_motion_cameras()),
        )
        packets.append(packet)
        detection = packet.get("best_detection")
        score = float((detection or {}).get("score") or 0.0)
        camera = (detection or {}).get("source_camera")
        if detection is not None and score >= min_score and camera in _calibrated_motion_cameras():
            best = dict(detection)
            best["object_role"] = object_role
            best["prompt_index"] = prompt_index
            best["prompt_fallback_used"] = bool(prompt_index > 0)
            best["raw_detection_source"] = best.get("source")
            best["source"] = "live_top_camera" if camera == "top" else "live_calibrated_camera"
            best["detection_source"] = best["source"]
            break
    combined = {
        "stage": stage,
        "object_role": object_role,
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
    combined["path"] = write_json(run_dir / "observations" / f"{stage}_{object_role}_prompt_fallback.json", combined)
    return combined, best


def _workspace_bounds() -> dict[str, tuple[float, float]]:
    return {
        "x": (
            _env_float("OPENFORGE_FRIDGE_WORKSPACE_X_MIN_M", 0.35),
            _env_float("OPENFORGE_FRIDGE_WORKSPACE_X_MAX_M", 1.05),
        ),
        "y": (
            _env_float("OPENFORGE_FRIDGE_WORKSPACE_Y_MIN_M", -0.65),
            _env_float("OPENFORGE_FRIDGE_WORKSPACE_Y_MAX_M", 0.65),
        ),
        "z": (
            _env_float("OPENFORGE_FRIDGE_WORKSPACE_Z_MIN_M", 0.62),
            _env_float("OPENFORGE_FRIDGE_WORKSPACE_Z_MAX_M", 1.25),
        ),
    }


def _validate_detection(detection: dict[str, Any] | None, *, object_role: str) -> dict[str, Any]:
    role_prefix = f"OPENFORGE_FRIDGE_{object_role.upper()}"
    role_default_scores = {
        "refrigerator": 0.20,
        "door": 0.25,
        "handle": 0.20,
        "box": 0.35,
    }
    min_score = _env_float(
        f"{role_prefix}_MIN_SCORE",
        _env_float("OPENFORGE_FRIDGE_MIN_SCORE", role_default_scores.get(object_role, 0.35)),
    )
    bounds = _workspace_bounds()
    result: dict[str, Any] = {
        "object_role": object_role,
        "passed": False,
        "min_score": min_score,
        "workspace_bounds_m": bounds,
        "problems": [],
        "warnings": [],
    }
    if detection is None:
        result["problems"].append("missing detection")
        return result
    score = float(detection.get("score") or 0.0)
    result["score"] = score
    if score < min_score:
        result["problems"].append(f"score {score:.3f} below minimum {min_score:.3f}")
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        result["problems"].append("missing 3D position")
    else:
        rounded_xyz = _round_list(xyz, limit=3)
        result["position_3d"] = rounded_xyz
        for axis, value in zip(("x", "y", "z"), rounded_xyz):
            lo, hi = bounds[axis]
            if value < lo or value > hi:
                result["problems"].append(f"{axis}={value:.3f} outside [{lo:.3f}, {hi:.3f}]")
    box = detection.get("box_2d") or []
    margin_px = _env_int("OPENFORGE_FRIDGE_IMAGE_EDGE_MARGIN_PX", 6)
    if margin_px > 0 and len(box) >= 4:
        width_px = _env_int("OPENFORGE_FRIDGE_TOP_IMAGE_WIDTH_PX", 640)
        height_px = _env_int("OPENFORGE_FRIDGE_TOP_IMAGE_HEIGHT_PX", 480)
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        result["box_2d"] = _round_list([x0, y0, x1, y1], digits=1, limit=4)
        if x0 <= margin_px or y0 <= margin_px or x1 >= width_px - margin_px or y1 >= height_px - margin_px:
            camera = str(detection.get("source_camera") or detection.get("camera") or "")
            allow_side_edge = _truthy_env("OPENFORGE_FRIDGE_ALLOW_SIDE_CAMERA_EDGE_DETECTIONS", True)
            if allow_side_edge and camera in {"left", "right"} and object_role in {"refrigerator", "door", "handle"}:
                result["warnings"].append("2D detection box touches side-camera image edge")
            else:
                result["problems"].append("2D detection box touches image edge")
    result["passed"] = not result["problems"]
    return result


def _validate_handle_relationship(
    *,
    handle_det: dict[str, Any] | None,
    door_det: dict[str, Any] | None,
    refrigerator_body: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": False,
        "problems": [],
        "warnings": [],
        "policy": "handle must be near live door/body and not an opposite-side edge artifact",
    }
    if handle_det is None:
        result["problems"].append("missing handle detection")
        return result
    reference = door_det or refrigerator_body
    if reference is None:
        result["problems"].append("missing door/body reference for handle sanity")
        return result

    handle_xyz = [float(v) for v in handle_det["position_3d"][:3]]
    ref_xyz = [float(v) for v in reference["position_3d"][:3]]
    y_delta = abs(handle_xyz[1] - ref_xyz[1])
    xy_delta = math.hypot(handle_xyz[0] - ref_xyz[0], handle_xyz[1] - ref_xyz[1])
    z_delta = abs(handle_xyz[2] - ref_xyz[2])
    max_y_delta = _env_float("OPENFORGE_FRIDGE_HANDLE_MAX_DOOR_Y_DELTA_M", 0.18)
    max_xy_delta = _env_float("OPENFORGE_FRIDGE_HANDLE_MAX_DOOR_XY_DELTA_M", 0.28)
    max_z_delta = _env_float("OPENFORGE_FRIDGE_HANDLE_MAX_DOOR_Z_DELTA_M", 0.22)
    ref_score = float(reference.get("score") or 0.0)
    confident_score = _env_float("OPENFORGE_FRIDGE_HANDLE_REF_CONFIDENT_SCORE", 0.25)
    ref_confident = ref_score >= confident_score
    result.update(
        {
            "handle_position_3d": _round_list(handle_xyz, limit=3),
            "reference_position_3d": _round_list(ref_xyz, limit=3),
            "reference_object_role": reference.get("object_role") or reference.get("prompt"),
            "reference_prompt": reference.get("prompt"),
            "reference_score": ref_score,
            "reference_confident": ref_confident,
            "y_delta_m": round(y_delta, 5),
            "xy_delta_m": round(xy_delta, 5),
            "z_delta_m": round(z_delta, 5),
            "max_y_delta_m": max_y_delta,
            "max_xy_delta_m": max_xy_delta,
            "max_z_delta_m": max_z_delta,
        }
    )
    if y_delta > max_y_delta:
        result["problems"].append(f"handle_y differs from door/body y by {y_delta:.3f}m > {max_y_delta:.3f}m")
    if xy_delta > max_xy_delta:
        result["problems"].append(f"handle xy distance from door/body is {xy_delta:.3f}m > {max_xy_delta:.3f}m")
    if z_delta > max_z_delta:
        result["problems"].append(f"handle z differs from door/body by {z_delta:.3f}m > {max_z_delta:.3f}m")

    same_side_threshold = _env_float("OPENFORGE_FRIDGE_HANDLE_SIGNIFICANT_Y_M", 0.05)
    opposite_side = (
        abs(handle_xyz[1]) > same_side_threshold
        and abs(ref_xyz[1]) > same_side_threshold
        and handle_xyz[1] * ref_xyz[1] < 0.0
    )
    result["opposite_side_y_sign"] = opposite_side
    if ref_confident and opposite_side:
        result["problems"].append("handle and door/body are on opposite workspace y sides")

    handle_camera = handle_det.get("source_camera") or handle_det.get("camera")
    ref_camera = reference.get("source_camera") or reference.get("camera")
    result["handle_camera"] = handle_camera
    result["reference_camera"] = ref_camera
    if handle_camera and ref_camera and handle_camera != ref_camera:
        result["warnings"].append("handle and door/body came from different calibrated cameras")

    box = handle_det.get("box_2d") or []
    if len(box) >= 4 and (float(box[0]) <= 1.0 or float(box[2]) >= _env_int("OPENFORGE_FRIDGE_TOP_IMAGE_WIDTH_PX", 640) - 1.0):
        if ref_confident and opposite_side:
            result["problems"].append("handle detection is an opposite-side image-edge artifact")
        else:
            result["warnings"].append("handle detection touches image edge")
    result["passed"] = not result["problems"]
    return result


def _validate_body_relationship(
    *,
    refrigerator_det: dict[str, Any] | None,
    door_det: dict[str, Any] | None,
    refrigerator_passed: bool,
    door_passed: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": False,
        "use_door_fallback": False,
        "problems": [],
        "warnings": [],
        "policy": "refrigerator body prompt fallbacks must agree with the live door side before planning",
    }
    if not refrigerator_passed or refrigerator_det is None:
        if door_passed and door_det is not None:
            result["passed"] = True
            result["use_door_fallback"] = True
            result["reason"] = "refrigerator/body prompt missing or failed sanity; live door detection is used as body proxy"
        else:
            result["problems"].append("missing coherent refrigerator body and door detections")
        return result
    if not door_passed or door_det is None:
        result["passed"] = True
        result["reason"] = "refrigerator/body detection passed; no usable door detection for side consistency check"
        return result

    body_xyz = [float(v) for v in refrigerator_det["position_3d"][:3]]
    door_xyz = [float(v) for v in door_det["position_3d"][:3]]
    y_delta = abs(body_xyz[1] - door_xyz[1])
    xy_delta = math.hypot(body_xyz[0] - door_xyz[0], body_xyz[1] - door_xyz[1])
    z_delta = abs(body_xyz[2] - door_xyz[2])
    max_y_delta = _env_float("OPENFORGE_FRIDGE_BODY_MAX_DOOR_Y_DELTA_M", 0.18)
    max_xy_delta = _env_float("OPENFORGE_FRIDGE_BODY_MAX_DOOR_XY_DELTA_M", 0.35)
    max_z_delta = _env_float("OPENFORGE_FRIDGE_BODY_MAX_DOOR_Z_DELTA_M", 0.35)
    significant_y = _env_float("OPENFORGE_FRIDGE_BODY_SIGNIFICANT_Y_M", 0.05)
    opposite_side = (
        abs(body_xyz[1]) > significant_y
        and abs(door_xyz[1]) > significant_y
        and body_xyz[1] * door_xyz[1] < 0.0
    )
    body_camera = refrigerator_det.get("source_camera") or refrigerator_det.get("camera")
    door_camera = door_det.get("source_camera") or door_det.get("camera")
    result.update(
        {
            "body_position_3d": _round_list(body_xyz, limit=3),
            "door_position_3d": _round_list(door_xyz, limit=3),
            "body_prompt": refrigerator_det.get("prompt"),
            "door_prompt": door_det.get("prompt"),
            "body_camera": body_camera,
            "door_camera": door_camera,
            "y_delta_m": round(y_delta, 5),
            "xy_delta_m": round(xy_delta, 5),
            "z_delta_m": round(z_delta, 5),
            "max_y_delta_m": max_y_delta,
            "max_xy_delta_m": max_xy_delta,
            "max_z_delta_m": max_z_delta,
            "opposite_side_y_sign": opposite_side,
        }
    )
    if body_camera and door_camera and body_camera != door_camera:
        result["warnings"].append("refrigerator/body and door came from different calibrated cameras")
    if y_delta > max_y_delta:
        result["problems"].append(
            f"body_y differs from door_y by {y_delta:.3f}m > {max_y_delta:.3f}m"
        )
    if xy_delta > max_xy_delta:
        result["warnings"].append(f"body xy distance from door is {xy_delta:.3f}m > {max_xy_delta:.3f}m")
    if z_delta > max_z_delta:
        result["warnings"].append(f"body z differs from door by {z_delta:.3f}m > {max_z_delta:.3f}m")
    if opposite_side:
        result["problems"].append("refrigerator/body and door are on opposite workspace y sides")
    if result["problems"]:
        result["passed"] = True
        result["use_door_fallback"] = True
        result["reason"] = "body prompt likely selected an opposite-side/edge object; live door is safer for planning"
    else:
        result["passed"] = True
        result["reason"] = "refrigerator/body detection is consistent with live door"
    return result


def _select_refrigerator_body_for_planning(
    *,
    refrigerator_det: dict[str, Any] | None,
    door_det: dict[str, Any] | None,
    sanity: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    relationship = _validate_body_relationship(
        refrigerator_det=refrigerator_det,
        door_det=door_det,
        refrigerator_passed=bool((sanity.get("refrigerator") or {}).get("passed")),
        door_passed=bool((sanity.get("door") or {}).get("passed")),
    )
    if relationship.get("use_door_fallback"):
        if door_det is None:
            return None, None, relationship
        body = dict(door_det)
        body["object_role"] = "refrigerator_body_fallback_from_door"
        body["body_fallback_reason"] = relationship.get("reason")
        return body, "door_detection_fallback", relationship
    if relationship.get("passed") and refrigerator_det is not None:
        return refrigerator_det, "refrigerator_detection", relationship
    return None, None, relationship


def _unit_xy(vec: Any, fallback: list[float] | None = None) -> list[float]:
    values = list(vec)[:3] if vec is not None else []
    if len(values) < 3:
        values = list(fallback or [1.0, 0.0, 0.0])
    x = float(values[0])
    y = float(values[1])
    norm = math.hypot(x, y)
    if norm < 1e-6:
        if fallback is not None:
            return _unit_xy(fallback)
        return [1.0, 0.0, 0.0]
    return [x / norm, y / norm, 0.0]


def _rotate_xy(unit: list[float], deg: float) -> list[float]:
    rad = math.radians(float(deg))
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)
    return [
        float(unit[0]) * cos_t - float(unit[1]) * sin_t,
        float(unit[0]) * sin_t + float(unit[1]) * cos_t,
        0.0,
    ]


def _perp_xy(unit: list[float], sign: float = 1.0) -> list[float]:
    return [-float(unit[1]) * sign, float(unit[0]) * sign, 0.0]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _scale(vec: list[float], scale: float) -> list[float]:
    return [float(vec[i]) * float(scale) for i in range(3)]


def _display_rpy_from_rotation_matrix(rotation_matrix: Any) -> list[float]:
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation

        euler_xyz = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=float)).as_euler("xyz", degrees=True)
        display = [float(euler_xyz[1]), float(-euler_xyz[0]), float(-euler_xyz[2] - 90.0)]
        return [((value + 180.0) % 360.0) - 180.0 for value in display]
    except Exception:
        z_axis = _unit_xy([rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]])
        yaw = math.degrees(math.atan2(z_axis[1], z_axis[0]))
        return [0.0, 90.0, yaw]


def _side_grasp_display_rpy_from_approach(
    approach_dir: list[float],
    wrist_roll_deg: float,
) -> tuple[list[float], dict[str, list[float]]]:
    try:
        import numpy as np

        approach = np.asarray(approach_dir, dtype=float)
        norm = float(np.linalg.norm(approach))
        if norm < 1e-6:
            raise RuntimeError(f"invalid approach_dir: {approach_dir!r}")
        z_axis = approach / norm
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        x_axis = np.cross(world_up, z_axis)
        if float(np.linalg.norm(x_axis)) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / float(np.linalg.norm(y_axis))
        if abs(float(wrist_roll_deg)) > 1e-6:
            roll_rad = math.radians(float(wrist_roll_deg))
            cos_t = math.cos(roll_rad)
            sin_t = math.sin(roll_rad)
            x_base = x_axis.copy()
            y_base = y_axis.copy()
            x_axis = cos_t * x_base + sin_t * y_base
            y_axis = -sin_t * x_base + cos_t * y_base
            x_axis = x_axis / float(np.linalg.norm(x_axis))
            y_axis = y_axis / float(np.linalg.norm(y_axis))
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        return (
            _display_rpy_from_rotation_matrix(rotation_matrix),
            {
                "local_x_opening_axis": _round_list(x_axis, limit=3),
                "local_y_height_axis": _round_list(y_axis, limit=3),
                "local_z_approach_axis": _round_list(z_axis, limit=3),
            },
        )
    except Exception:
        approach = _unit_xy(approach_dir)
        yaw = math.degrees(math.atan2(approach[1], approach[0]))
        return (
            [0.0, 90.0, yaw],
            {
                "local_x_opening_axis": _round_list(_perp_xy(approach), limit=3),
                "local_y_height_axis": [0.0, 0.0, 1.0],
            "local_z_approach_axis": _round_list(approach, limit=3),
        },
    )


def _display_axes_for_side_rpy(rpy: list[float]) -> dict[str, list[float]]:
    roll_f = float(rpy[0])
    pitch_f = float(rpy[1])
    yaw_f = float(rpy[2])
    if abs(roll_f) < 1e-6 and abs(abs(pitch_f) - 90.0) < 1e-6:
        yaw_rad = math.radians(yaw_f)
        if pitch_f >= 0.0:
            x_axis = [-math.sin(yaw_rad), -math.cos(yaw_rad), 0.0]
            y_axis = [0.0, 0.0, -1.0]
        else:
            x_axis = [math.sin(yaw_rad), math.cos(yaw_rad), 0.0]
            y_axis = [0.0, 0.0, 1.0]
        z_axis = [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0]
        return {
            "local_x_opening_axis": _round_list(x_axis, limit=3),
            "local_y_height_axis": _round_list(y_axis, limit=3),
            "local_z_approach_axis": _round_list(z_axis, limit=3),
        }
    return {}


def _handle_rpy_options(approach: list[float], wrist_roll_deg: float) -> list[dict[str, Any]]:
    matrix_rpy, matrix_axes = _side_grasp_display_rpy_from_approach(approach, wrist_roll_deg)
    yaw = math.degrees(math.atan2(-float(approach[1]), float(approach[0])))
    options = [
        {
            "family": "display_pitch_plus",
            "rpy": [0.0, 90.0, yaw],
            "axes": _display_axes_for_side_rpy([0.0, 90.0, yaw]),
            "score_bonus": 0.03,
        },
        {
            "family": "display_pitch_minus",
            "rpy": [0.0, -90.0, yaw],
            "axes": _display_axes_for_side_rpy([0.0, -90.0, yaw]),
            "score_bonus": 0.015,
        },
        {
            "family": "drawer_matrix",
            "rpy": matrix_rpy,
            "axes": matrix_axes,
            "score_bonus": 0.0,
        },
    ]
    if abs(float(wrist_roll_deg)) > 1e-6:
        options.append(
            {
                "family": "drawer_matrix_roll",
                "rpy": matrix_rpy,
                "axes": matrix_axes,
                "score_bonus": -0.01,
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for option in options:
        rpy = [float(v) for v in option["rpy"][:3]]
        key = tuple(round(v, 3) for v in rpy)
        if key in seen:
            continue
        seen.add(key)
        option["rpy"] = rpy
        deduped.append(option)
    return deduped


def _state_arm_positions() -> dict[str, list[float]]:
    get_robot_state = _tool("get_robot_state")
    if callable(get_robot_state):
        try:
            state = get_robot_state()
            return {
                "left": _round_list(_field(state, "left_ee_pos", []), limit=3),
                "right": _round_list(_field(state, "right_ee_pos", []), limit=3),
            }
        except Exception:
            pass
    return {
        "left": [0.50, 0.31, 0.91],
        "right": [0.50, -0.31, 0.91],
    }


def _choose_arm(xyz: list[float], *, prefer: str | None = None) -> dict[str, Any]:
    arms = _state_arm_positions()
    distances = {
        side: math.sqrt(sum((float(xyz[i]) - float(pos[i])) ** 2 for i in range(3)))
        for side, pos in arms.items()
        if len(pos) >= 3
    }
    if prefer in {"left", "right"}:
        side = prefer
        reason = f"configured preference {prefer}; current distances left={distances.get('left'):.3f} right={distances.get('right'):.3f}"
    elif distances:
        side = "left" if distances.get("left", 999.0) <= distances.get("right", 999.0) else "right"
        reason = f"nearest current EE: left={distances.get('left'):.3f}m right={distances.get('right'):.3f}m"
    else:
        side = "left" if float(xyz[1]) >= 0 else "right"
        reason = "fallback from object y sign"
    return {"arm": side, "reason": reason, "distances_m": distances}


def _estimate_outward_axis(
    *,
    handle_det: dict[str, Any],
    door_det: dict[str, Any] | None,
    refrigerator_det: dict[str, Any],
) -> dict[str, Any]:
    handle_xyz = [float(v) for v in handle_det["position_3d"][:3]]
    reference = door_det or refrigerator_det
    reference_xyz = [float(v) for v in reference["position_3d"][:3]]
    delta = [handle_xyz[i] - reference_xyz[i] for i in range(3)]
    if math.hypot(delta[0], delta[1]) >= _env_float("OPENFORGE_FRIDGE_AXIS_MIN_HANDLE_OFFSET_M", 0.04):
        axis = _unit_xy(delta)
        return {
            "outward_axis_world": _round_list(axis, limit=3),
            "source": f"handle_minus_{reference.get('object_role', 'refrigerator')}_xy",
            "confidence": "tentative",
            "requires_review_before_physical": True,
            "reference_position_3d": _round_list(reference_xyz, limit=3),
        }
    fallback = [0.0, 1.0 if handle_xyz[1] >= 0.0 else -1.0, 0.0]
    return {
        "outward_axis_world": fallback,
        "source": "workspace_y_sign_fallback",
        "confidence": "low",
        "requires_review_before_physical": True,
        "reference_position_3d": _round_list(reference_xyz, limit=3),
    }


def _motion_common_kwargs(*, preview_only: bool) -> dict[str, Any]:
    return {
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_FRIDGE_PLANNER_BACKEND", "curobo"),
        "solver_speed": os.environ.get("OPENFORGE_FRIDGE_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_FRIDGE_PLANNING_SPEED", 0.25),
        "ik_error_threshold": _env_float("OPENFORGE_FRIDGE_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_FRIDGE_IK_ROT_THRESHOLD_DEG", 12.0),
        "ik_xyz_weight": _env_float("OPENFORGE_FRIDGE_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_FRIDGE_IK_RPY_WEIGHT", 0.25),
    }


def _raw_preview_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    reason = _field(result, "reason", _field(result, "error", None))
    executed = bool(_field(result, "executed", False))
    status_text = str(status) if status is not None else None
    success = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    return {
        "status": status_text,
        "reason": reason,
        "executed": executed,
        "trajectory_cache_key": _field(result, "trajectory_cache_key", None),
        "final_pos_error_m": _field(result, "final_pos_error_m", _field(result, "ik_error_m", None)),
        "final_rot_error_deg": _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None)),
        "success": success,
        "raw": json_safe(result),
    }


def _preview_pose(
    *,
    freespace_move: Callable[..., Any],
    arm: str,
    pose: dict[str, Any],
    label: str,
    gripper: float,
) -> dict[str, Any]:
    prefix = "left" if arm == "left" else "right"
    kwargs = _motion_common_kwargs(preview_only=True)
    kwargs[f"{prefix}_target_pos"] = [float(v) for v in pose["position"][:3]]
    kwargs[f"{prefix}_target_rpy"] = [float(v) for v in pose["rpy"][:3]]
    kwargs[f"{prefix}_gripper"] = float(gripper)
    call = call_with_timeout(
        label,
        freespace_move,
        _env_float("OPENFORGE_FRIDGE_PREVIEW_TIMEOUT_S", 10.0),
        run_in_background=_tool("run_in_background"),
        **kwargs,
    )
    summary = {
        "label": label,
        "arm": arm,
        "pose": pose,
        "request": kwargs,
        "call_ok": bool(call.get("ok")),
        "success": False,
    }
    if not call.get("ok"):
        summary["error"] = call.get("error")
        return summary
    preview = _raw_preview_summary(call.get("data"))
    summary["preview"] = preview
    summary["success"] = bool(preview.get("success"))
    if preview.get("executed"):
        summary["success"] = False
        summary["error"] = "preview_only call reported executed=True"
    return summary


def _build_door_open_candidates(
    *,
    handle_det: dict[str, Any],
    door_det: dict[str, Any] | None,
    refrigerator_det: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    handle_xyz = [float(v) for v in handle_det["position_3d"][:3]]
    axis_info = _estimate_outward_axis(handle_det=handle_det, door_det=door_det, refrigerator_det=refrigerator_det)
    outward_axis = [float(v) for v in axis_info["outward_axis_world"][:3]]
    approach_base = _scale(outward_axis, -1.0)
    arm_choice = _choose_arm(handle_xyz, prefer=os.environ.get("OPENFORGE_FRIDGE_HANDLE_ARM", "").strip() or None)
    preferred = arm_choice["arm"]
    alternate = "right" if preferred == "left" else "left"
    arms = [preferred]
    if _truthy_env("OPENFORGE_FRIDGE_TRY_ALTERNATE_HANDLE_ARM", True):
        arms.append(alternate)

    candidates: list[dict[str, Any]] = []
    approach_offsets = _float_csv(
        "OPENFORGE_FRIDGE_HANDLE_APPROACH_OFFSETS_DEG",
        [90.0, 0.0, -90.0, 45.0, -45.0, 15.0, -15.0, 30.0, -30.0],
    )
    wrist_rolls = _float_csv("OPENFORGE_FRIDGE_HANDLE_WRIST_ROLLS_DEG", [0.0, -20.0, 20.0])
    z_offsets = _float_csv("OPENFORGE_FRIDGE_HANDLE_Z_OFFSETS_M", [0.02, 0.0, -0.015, 0.04, 0.06, 0.09, 0.12])
    sweep_signs = _float_csv("OPENFORGE_FRIDGE_DOOR_SWEEP_SIGNS", [0.0, 1.0, -1.0])
    standoff = _env_float("OPENFORGE_FRIDGE_HANDLE_PREGRASP_STANDOFF_M", 0.12)
    contact_offset = _env_float("OPENFORGE_FRIDGE_HANDLE_CONTACT_OFFSET_M", 0.012)
    preferred_approach = _env_float("OPENFORGE_FRIDGE_HANDLE_PREFERRED_APPROACH_OFFSET_DEG", 90.0)
    preferred_z = _env_float("OPENFORGE_FRIDGE_HANDLE_PREFERRED_Z_OFFSET_M", 0.02)
    pull_distance = min(
        _env_float("OPENFORGE_FRIDGE_DOOR_PULL_DISTANCE_M", 0.035),
        _env_float("OPENFORGE_FRIDGE_DOOR_MAX_PREVIEW_TRAVEL_M", 0.05),
    )
    sweep_distance = _env_float("OPENFORGE_FRIDGE_DOOR_SWEEP_DISTANCE_M", 0.025)
    idx = 0
    for arm_rank, arm in enumerate(arms):
        for approach_offset in approach_offsets:
            approach = _unit_xy(_rotate_xy(approach_base, approach_offset), approach_base)
            for wrist_roll in wrist_rolls:
                for rpy_option in _handle_rpy_options(approach, wrist_roll):
                    rpy = rpy_option["rpy"]
                    axes = rpy_option["axes"]
                    for z_offset in z_offsets:
                        grasp = _add(handle_xyz, _scale(outward_axis, contact_offset))
                        grasp[2] += float(z_offset)
                        pregrasp = _add(grasp, _scale(approach, -standoff))
                        for sweep_sign in sweep_signs:
                            tangent = _perp_xy(outward_axis, sign=float(sweep_sign))
                            open_pose = _add(grasp, _add(_scale(outward_axis, pull_distance), _scale(tangent, sweep_distance)))
                            candidate = {
                                "label": f"door_handle_pull_{idx}",
                                "arm": arm,
                                "score": round(
                                    1.0
                                    + float(rpy_option.get("score_bonus") or 0.0)
                                    - 0.08 * arm_rank
                                    - 0.003 * abs(approach_offset - preferred_approach)
                                    - 0.002 * abs(wrist_roll)
                                    - 1.4 * abs(z_offset - preferred_z),
                                    4,
                                ),
                                "handle_detection": handle_det,
                                "door_detection": door_det,
                                "refrigerator_detection": refrigerator_det,
                                "axis_info": axis_info,
                                "rpy": _round_list(rpy, digits=3, limit=3),
                                "rpy_family": rpy_option["family"],
                                "gripper_local_axes_world": axes,
                                "pregrasp_pose": {"position": _round_list(pregrasp, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "handle_grasp_pose": {"position": _round_list(grasp, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "door_open_small_pose": {"position": _round_list(open_pose, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "release_retreat_pose": {"position": _round_list(pregrasp, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "approach_direction_world": _round_list(approach, limit=3),
                                "outward_axis_world": _round_list(outward_axis, limit=3),
                                "approach_offset_deg": float(approach_offset),
                                "wrist_roll_deg": float(wrist_roll),
                                "z_offset_m": float(z_offset),
                                "sweep_sign": float(sweep_sign),
                                "pull_distance_m": pull_distance,
                                "sweep_distance_m": sweep_distance,
                                "arm_choice": arm_choice,
                                "physical_status": "preview_only; physical handle contact requires a fresh refrigerator-specific ticket",
                            }
                            candidates.append(candidate)
                            idx += 1
    candidates.sort(key=lambda item: -float(item["score"]))
    max_candidates = max(1, _env_int("OPENFORGE_FRIDGE_HANDLE_MAX_CANDIDATES", 24))
    if not _truthy_env("OPENFORGE_FRIDGE_INTERLEAVE_HANDLE_ARMS", True):
        return axis_info, candidates[:max_candidates]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("arm")), []).append(candidate)
    arm_order = [arm for arm in arms if arm in grouped]
    for arm in sorted(grouped):
        if arm not in arm_order:
            arm_order.append(arm)
    interleaved: list[dict[str, Any]] = []
    while len(interleaved) < max_candidates and any(grouped.get(arm) for arm in arm_order):
        for arm in arm_order:
            group = grouped.get(arm) or []
            if group:
                interleaved.append(group.pop(0))
                if len(interleaved) >= max_candidates:
                    break
    return axis_info, interleaved


def _preview_door_candidates(
    *,
    run_dir: Path,
    candidates: list[dict[str, Any]],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    handle_gripper = _env_float("OPENFORGE_FRIDGE_HANDLE_PREVIEW_GRIPPER_POS", 0.55)
    for index, candidate in enumerate(candidates):
        previews: list[dict[str, Any]] = []
        sequence = (
            ("pregrasp_pose", "pregrasp", 1.0),
            ("handle_grasp_pose", "handle_grasp", handle_gripper),
            ("door_open_small_pose", "door_open_small", handle_gripper),
            ("release_retreat_pose", "release_retreat", 1.0),
        )
        for pose_key, label, gripper in sequence:
            preview = _preview_pose(
                freespace_move=freespace_move,
                arm=candidate["arm"],
                pose=candidate[pose_key],
                label=f"door_candidate_{index}_{label}",
                gripper=gripper,
            )
            previews.append(preview)
            if not preview.get("success"):
                break
        item = dict(candidate)
        item["candidate_index"] = index
        item["preview_success"] = len(previews) == len(sequence) and all(step.get("success") for step in previews)
        item["previews"] = previews
        ranked.append(item)
        if item["preview_success"] and selected is None:
            selected = item
            break
    packet = {
        "schema": "openforge.refrigerator_door_open_plan.v1",
        "stage": "plan_door_open",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "physical_motion_executed": False,
        "selected": selected,
        "candidates": ranked,
        "total_candidate_count": len(candidates),
    }
    packet["path"] = write_json(run_dir / "plans" / "door_open_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## plan door_open",
            f"- plan: {packet['path']}",
            f"- candidates_previewed: {len(ranked)}",
            f"- selected: {selected.get('label') if selected else None}",
        ],
    )
    return packet


def _build_box_candidates(box_det: dict[str, Any], preferred_arm: str | None) -> list[dict[str, Any]]:
    arm_order = []
    if preferred_arm in {"left", "right"}:
        arm_order.append(preferred_arm)
        arm_order.append("right" if preferred_arm == "left" else "left")
    else:
        arm_order = [_choose_arm([float(v) for v in box_det["position_3d"][:3]])["arm"]]
        arm_order.append("right" if arm_order[0] == "left" else "left")
    candidates: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(arm_order):
        yaws = [90.0, 70.0, 110.0, 50.0, 130.0, 0.0, 180.0] if arm == "left" else [-90.0, -70.0, -110.0, -50.0, -130.0, 180.0, 0.0]
        arm_candidates = generate_side_grasp_candidates(
            box_det,
            object_kind=f"box_{arm}",
            arm=arm,
            default_radius_m=_env_float("OPENFORGE_FRIDGE_BOX_RADIUS_M", 0.045),
            default_half_height_m=_env_float("OPENFORGE_FRIDGE_BOX_HALF_HEIGHT_M", 0.055),
            body_fraction=_env_float("OPENFORGE_FRIDGE_BOX_BODY_FRACTION", 0.50),
            pregrasp_standoff_m=_env_float("OPENFORGE_FRIDGE_BOX_PREGRASP_STANDOFF_M", 0.08),
            lift_z_m=_env_float("OPENFORGE_FRIDGE_BOX_LIFT_Z_M", 0.08),
            width_margin_m=_env_float("OPENFORGE_FRIDGE_BOX_WIDTH_MARGIN_M", 0.018),
            include_topdown=False,
            yaw_angles_deg=yaws,
            z_offsets_m=_float_csv("OPENFORGE_FRIDGE_BOX_Z_OFFSETS_M", [0.0, 0.015, -0.015, 0.03, -0.03]),
            center_z_offset_without_extents_m=_env_float("OPENFORGE_FRIDGE_BOX_Z_OFFSET_WITHOUT_EXTENTS_M", -0.015),
        )
        for candidate in arm_candidates:
            candidate["score"] = float(candidate.get("score") or 0.0) - 0.08 * arm_index
        candidates.extend(arm_candidates)
    candidates.sort(key=lambda item: -float(item.get("score") or 0.0))
    return candidates[: max(1, _env_int("OPENFORGE_FRIDGE_BOX_MAX_CANDIDATES", 96))]


def _guarded_preview_freespace(freespace_move: Callable[..., Any]) -> Callable[..., Any]:
    def _inner(**kwargs: Any) -> Any:
        result = freespace_move(**kwargs)
        summary = _raw_preview_summary(result)
        if bool(kwargs.get("preview_only", False)) and summary["executed"]:
            raise RuntimeError(f"preview unexpectedly executed: {summary}")
        if not summary["success"]:
            raise RuntimeError(f"planner preview failed: {summary}")
        return result

    return _inner


def _rank_box_candidates(
    *,
    run_dir: Path,
    candidates: list[dict[str, Any]],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    max_preview = max(1, _env_int("OPENFORGE_FRIDGE_BOX_MAX_PREVIEW_CANDIDATES", 80))
    return rank_motion_candidates(
        candidates=candidates[:max_preview],
        freespace_move=_guarded_preview_freespace(freespace_move),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="box_grasp_preview",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_FRIDGE_BOX_PREVIEW_TIMEOUT_S", 35.0),
        planner_backend=os.environ.get("OPENFORGE_FRIDGE_PLANNER_BACKEND", "curobo"),
        solver_speed=os.environ.get("OPENFORGE_FRIDGE_SOLVER_SPEED", "fast"),
        planning_speed=_env_float("OPENFORGE_FRIDGE_PLANNING_SPEED", 0.25),
        ik_error_threshold=_env_float("OPENFORGE_FRIDGE_IK_ERROR_THRESHOLD_M", 0.025),
        ik_rot_threshold_deg=_env_float("OPENFORGE_FRIDGE_IK_ROT_THRESHOLD_DEG", 12.0),
    )


def _build_place_inside_candidates(
    *,
    refrigerator_det: dict[str, Any],
    door_plan: dict[str, Any],
    box_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_door = door_plan.get("selected") or {}
    axis_info = selected_door.get("axis_info") or {}
    outward_axis = [float(v) for v in (axis_info.get("outward_axis_world") or [0.0, 1.0, 0.0])[:3]]
    outward_axis = _unit_xy(outward_axis)
    inward_axis = _scale(outward_axis, -1.0)
    lateral_axis = _perp_xy(outward_axis, sign=1.0)
    fridge_xyz = [float(v) for v in refrigerator_det["position_3d"][:3]]
    box_options: list[dict[str, Any]] = []
    selected_box = box_plan["selected"]
    if selected_box:
        box_options.append(selected_box)
    if _truthy_env("OPENFORGE_FRIDGE_PLACE_TRY_ALTERNATE_BOX_PLANS", True):
        seen_labels = {str(selected_box.get("label"))} if selected_box else set()
        for candidate in box_plan.get("candidates") or []:
            if not candidate.get("preview_success"):
                continue
            label = str(candidate.get("label"))
            if label in seen_labels:
                continue
            box_options.append(candidate)
            seen_labels.add(label)
            if len(box_options) >= max(1, _env_int("OPENFORGE_FRIDGE_PLACE_MAX_BOX_OPTIONS", 10)):
                break
    standoff = _env_float("OPENFORGE_FRIDGE_PLACE_PREPLACE_STANDOFF_M", 0.08)
    candidates: list[dict[str, Any]] = []
    idx = 0
    global_generate_max = max(1, _env_int("OPENFORGE_FRIDGE_PLACE_GENERATE_MAX_CANDIDATES", 240))
    per_box_generate_max = max(1, _env_int("OPENFORGE_FRIDGE_PLACE_CANDIDATES_PER_BOX_OPTION", 24))
    for box_rank, box_option in enumerate(box_options):
        box_generated = 0
        arm = box_option["arm"]
        base_rpy = [float(v) for v in box_option["lift_pose"]["rpy"][:3]]
        rpy_options = [base_rpy]
        if _truthy_env("OPENFORGE_FRIDGE_PLACE_TRY_RPY_SWEEP", True):
            place_yaws = _float_csv(
                "OPENFORGE_FRIDGE_PLACE_YAWS_DEG",
                [base_rpy[2], -90.0 if arm == "right" else 90.0, -70.0 if arm == "right" else 70.0, 0.0, 180.0],
            )
            seen_rpy = {tuple(round(v, 3) for v in base_rpy)}
            for yaw in place_yaws:
                for pitch in (90.0, -90.0):
                    rpy = [0.0, pitch, float(yaw)]
                    key = tuple(round(v, 3) for v in rpy)
                    if key not in seen_rpy:
                        rpy_options.append(rpy)
                        seen_rpy.add(key)
        for depth in _float_csv("OPENFORGE_FRIDGE_INTERIOR_DEPTH_OFFSETS_M", [0.04, 0.07, 0.10, 0.14]):
            for lateral in _float_csv("OPENFORGE_FRIDGE_INTERIOR_LATERAL_OFFSETS_M", [0.0, 0.03, -0.03]):
                for z_offset in _float_csv("OPENFORGE_FRIDGE_INTERIOR_Z_OFFSETS_M", [0.09, 0.12, 0.06, 0.03, 0.0, -0.03]):
                    for rpy_rank, rpy in enumerate(rpy_options):
                        if idx >= global_generate_max or box_generated >= per_box_generate_max:
                            break
                        place = _add(fridge_xyz, _add(_scale(inward_axis, depth), _scale(lateral_axis, lateral)))
                        place[2] += float(z_offset)
                        preplace = _add(place, _scale(outward_axis, standoff))
                        retreat = _add(preplace, [0.0, 0.0, _env_float("OPENFORGE_FRIDGE_PLACE_RETREAT_Z_M", 0.03)])
                        same_side_arm = "left" if preplace[1] >= 0.04 else "right" if preplace[1] <= -0.04 else arm
                        arm_side_penalty = 0.0 if arm == same_side_arm else 0.14
                        score = (
                            1.0
                            - 0.06 * box_rank
                            - arm_side_penalty
                            - 1.5 * abs(lateral)
                            - 0.8 * abs(float(z_offset) - 0.09)
                            - 0.015 * rpy_rank
                        )
                        candidates.append(
                            {
                                "label": f"place_inside_{idx}",
                                "arm": arm,
                                "score": round(score, 4),
                                "preplace_pose": {"position": _round_list(preplace, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "place_pose": {"position": _round_list(place, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "retreat_pose": {"position": _round_list(retreat, limit=3), "rpy": _round_list(rpy, digits=3, limit=3)},
                                "outward_axis_world": _round_list(outward_axis, limit=3),
                                "inward_axis_world": _round_list(inward_axis, limit=3),
                                "source_refrigerator_detection": refrigerator_det,
                                "source_box_plan_label": box_option.get("label"),
                                "source_box_plan_index": box_option.get("candidate_index"),
                                "source_box_candidate": box_option,
                                "depth_offset_m": float(depth),
                                "lateral_offset_m": float(lateral),
                                "z_offset_m": float(z_offset),
                                "rpy_rank": rpy_rank,
                                "box_option_rank": box_rank,
                                "same_side_arm_preference": same_side_arm,
                                "arm_side_penalty": arm_side_penalty,
                                "assumption": "inside target is derived from live refrigerator/door detection and tentative door outward axis; requires post-open interior observation before physical place",
                            }
                        )
                        idx += 1
                        box_generated += 1
                    if idx >= global_generate_max or box_generated >= per_box_generate_max:
                        break
                if idx >= global_generate_max or box_generated >= per_box_generate_max:
                    break
            if idx >= global_generate_max or box_generated >= per_box_generate_max:
                break
        if idx >= global_generate_max:
            break
    candidates.sort(key=lambda item: -float(item["score"]))
    return candidates[: max(1, _env_int("OPENFORGE_FRIDGE_PLACE_MAX_CANDIDATES", 48))]


def _preview_place_inside(
    *,
    run_dir: Path,
    candidates: list[dict[str, Any]],
    freespace_move: Callable[..., Any],
    box_plan: dict[str, Any],
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    gripper = float((box_plan.get("selected") or {}).get("width") or _env_float("OPENFORGE_FRIDGE_BOX_PREVIEW_HOLD_GRIPPER_POS", 0.08))
    for index, candidate in enumerate(candidates):
        previews: list[dict[str, Any]] = []
        for pose_key in ("preplace_pose", "place_pose", "retreat_pose"):
            preview = _preview_pose(
                freespace_move=freespace_move,
                arm=candidate["arm"],
                pose=candidate[pose_key],
                label=f"place_candidate_{index}_{pose_key}",
                gripper=gripper,
            )
            previews.append(preview)
            if not preview.get("success"):
                break
        item = dict(candidate)
        item["candidate_index"] = index
        item["preview_success"] = len(previews) == 3 and all(step.get("success") for step in previews)
        item["previews"] = previews
        ranked.append(item)
        if item["preview_success"] and selected is None:
            selected = item
            break
    packet = {
        "schema": "openforge.refrigerator_place_inside_plan.v1",
        "stage": "plan_place_inside",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "physical_motion_executed": False,
        "selected": selected,
        "candidates": ranked,
        "total_candidate_count": len(candidates),
        "box_hold_preview_gripper": gripper,
    }
    packet["path"] = write_json(run_dir / "plans" / "place_inside_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## plan place_inside",
            f"- plan: {packet['path']}",
            f"- candidates_previewed: {len(ranked)}",
            f"- selected: {selected.get('label') if selected else None}",
        ],
    )
    return packet


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists():
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    videos = _video_paths(run_dir)
    evidence: dict[str, Any] = {
        "required": True,
        "passed": bool(videos),
        "video_paths": videos,
    }
    if videos:
        return evidence
    ffmpeg_logs = sorted(str(path) for path in run_dir.glob("*.ffmpeg.log"))
    evidence["problem"] = (
        "no mp4 files were found in the run directory; recording/debug UI may be disabled, "
        "BundleSDF preview streams may be unavailable, or ffmpeg may have failed"
    )
    evidence["ffmpeg_logs"] = ffmpeg_logs
    evidence["blocker_before_physical"] = True
    return evidence


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.open_green_refrigerator_put_box_inside.result.v1",
        "success": False,
        "plan_success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "dry_run": True,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False),
        "observation_policy": "fresh live calibrated-camera detections only; manual XYZ env vars are rejected",
        "calibration_env": {
            "YAM_STATION_CALIBRATED_XML": os.environ.get("YAM_STATION_CALIBRATED_XML"),
            "CAP_TOP_CAMERA_BACKEND": os.environ.get("CAP_TOP_CAMERA_BACKEND"),
            "CAP_TOP_CAMERA_FRAME": os.environ.get("CAP_TOP_CAMERA_FRAME"),
            "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP": os.environ.get("CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP"),
        },
        "prompts": {},
        "pre_observation": {},
        "detections": {},
        "detection_sanity": {},
        "door_axis": None,
        "door_plan": None,
        "box_plan": None,
        "place_inside_plan": None,
        "selected_strategy": None,
        "why_stopped": None,
        "next_recommendation": None,
        "video_paths": [],
        "recording_evidence": None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        if result["physical_gate"]:
            result["why_stopped"] = "OPENFORGE_ALLOW_PHYSICAL_MOTION is set, but this first refrigerator script is no-motion-only"
            result["next_recommendation"] = "Unset OPENFORGE_ALLOW_PHYSICAL_MOTION and rerun observe/plan/preview."
            return result

        manual_env = _manual_pose_env_vars_present()
        if manual_env:
            result["manual_pose_env_vars_present"] = manual_env
            result["why_stopped"] = "manual XYZ env vars are not allowed for this live-detection refrigerator task"
            result["next_recommendation"] = "Unset manual pose env vars and rerun with calibrated live perception."
            return result

        cameras = _csv("OPENFORGE_FRIDGE_CAMERAS", ["top", "left", "right"])
        prompts = {
            "refrigerator": _csv(
                "OPENFORGE_FRIDGE_REFRIGERATOR_PROMPTS",
                [
                    "green refrigerator",
                    "green fridge",
                    "mini refrigerator",
                    "refrigerator",
                    "large green box",
                    "large green rectangular box",
                    "green cabinet",
                    "green door",
                ],
            ),
            "door": _csv(
                "OPENFORGE_FRIDGE_DOOR_PROMPTS",
                [
                    "refrigerator door",
                    "fridge door",
                    "green refrigerator door",
                    "large green rectangular door",
                    "green door",
                    "green cabinet",
                    "large green rectangular box",
                ],
            ),
            "handle": _csv(
                "OPENFORGE_FRIDGE_HANDLE_PROMPTS",
                [
                    "refrigerator handle",
                    "fridge handle",
                    "door handle",
                    "vertical handle",
                    "white vertical handle",
                    "green refrigerator handle",
                    "white handle",
                ],
            ),
            "box": _csv("OPENFORGE_FRIDGE_BOX_PROMPTS", ["box", "small box", "carton"]),
        }
        result["prompts"] = prompts

        obs: dict[str, Any] = {}
        detections: dict[str, Any] = {}
        for role in ("refrigerator", "door", "handle", "box"):
            role_default_scores = {
                "refrigerator": 0.20,
                "door": 0.25,
                "handle": 0.20,
                "box": 0.35,
            }
            role_obs, det = _capture_prompt_fallback(
                run_dir=run_dir,
                stage="pre_observe",
                object_role=role,
                prompts=prompts[role],
                cameras=cameras,
                min_score=_env_float(
                    f"OPENFORGE_FRIDGE_{role.upper()}_MIN_SCORE",
                    _env_float("OPENFORGE_FRIDGE_MIN_SCORE", role_default_scores.get(role, 0.35)),
                ),
            )
            obs[role] = role_obs
            detections[role] = det
        result["pre_observation"] = obs
        result["detections"] = detections

        sanity = {role: _validate_detection(det, object_role=role) for role, det in detections.items()}
        result["detection_sanity"] = sanity
        append_stage_summary(
            run_dir,
            [
                "## live detection sanity",
                f"- refrigerator: {sanity['refrigerator']}",
                f"- door: {sanity['door']}",
                f"- handle: {sanity['handle']}",
                f"- box: {sanity['box']}",
            ],
        )
        refrigerator_body, refrigerator_body_source, body_relationship = _select_refrigerator_body_for_planning(
            refrigerator_det=detections.get("refrigerator"),
            door_det=detections.get("door"),
            sanity=sanity,
        )
        result["refrigerator_body_relationship_sanity"] = body_relationship
        if refrigerator_body_source == "door_detection_fallback" and refrigerator_body is not None:
            result["refrigerator_body_fallback"] = {
                "source": refrigerator_body_source,
                "reason": body_relationship.get("reason"),
                "detection": refrigerator_body,
            }
        append_stage_summary(
            run_dir,
            [
                "## refrigerator body relationship sanity",
                f"- body_relationship: {body_relationship}",
                f"- selected_source: {refrigerator_body_source}",
            ],
        )
        result["refrigerator_body_for_planning"] = refrigerator_body
        result["refrigerator_body_source"] = refrigerator_body_source if refrigerator_body is not None else None

        required_missing = [role for role in ("handle", "box") if not sanity[role].get("passed")]
        if refrigerator_body is None:
            required_missing.append("refrigerator_or_door")
        if required_missing:
            result["why_stopped"] = "live detection sanity failed for " + ", ".join(required_missing)
            result["next_recommendation"] = "Inspect observation overlays; ensure green refrigerator, handle, and box are visible, then rerun no-motion."
            return result

        handle_relationship = _validate_handle_relationship(
            handle_det=detections.get("handle"),
            door_det=detections.get("door") if sanity["door"].get("passed") else None,
            refrigerator_body=refrigerator_body,
        )
        result["handle_relationship_sanity"] = handle_relationship
        append_stage_summary(
            run_dir,
            [
                "## handle relationship sanity",
                f"- handle_relationship: {handle_relationship}",
            ],
        )
        if not handle_relationship.get("passed"):
            result["why_stopped"] = "live handle detection failed door/body relationship sanity"
            result["next_recommendation"] = (
                "Inspect handle overlays; use door/vertical handle prompts or reposition so the live handle detection "
                "is on the same side as the refrigerator door/body before planning."
            )
            return result

        freespace_move = _required_tool("freespace_move")
        door_axis, door_candidates = _build_door_open_candidates(
            handle_det=detections["handle"],
            door_det=detections.get("door"),
            refrigerator_det=refrigerator_body,
        )
        result["door_axis"] = door_axis
        result["candidate_artifacts"] = {
            "door_open_candidates": write_json(run_dir / "plans" / "door_open_candidates.json", door_candidates),
        }
        door_plan = _preview_door_candidates(run_dir=run_dir, candidates=door_candidates, freespace_move=freespace_move)
        result["door_plan"] = {
            "path": door_plan.get("path"),
            "selected": door_plan.get("selected"),
            "previewed": len(door_plan.get("candidates") or []),
        }
        if door_plan.get("selected") is None:
            result["why_stopped"] = "no refrigerator handle open candidate passed no-motion preview"
            result["next_recommendation"] = "Inspect door_open_preview.json; tune handle prompt, approach offsets, or planner thresholds in no-motion."
            return result

        handle_arm = str(door_plan["selected"]["arm"])
        box_preferred_arm = "right" if handle_arm == "left" else "left"
        box_candidates = _build_box_candidates(detections["box"], preferred_arm=box_preferred_arm)
        result["candidate_artifacts"]["box_grasp_candidates"] = write_json(run_dir / "plans" / "box_grasp_candidates.json", box_candidates)
        box_plan = _rank_box_candidates(run_dir=run_dir, candidates=box_candidates, freespace_move=freespace_move)
        result["box_plan"] = {
            "path": box_plan.get("plan_path"),
            "selected": box_plan.get("selected"),
            "previewed": len(box_plan.get("candidates") or []),
        }
        if box_plan.get("selected") is None:
            result["why_stopped"] = "no box grasp/lift candidate passed no-motion preview"
            result["next_recommendation"] = "Inspect box_grasp_preview plan packet; tune box prompts, arm choice, yaw, or z offsets."
            return result

        place_candidates = _build_place_inside_candidates(
            refrigerator_det=refrigerator_body,
            door_plan=door_plan,
            box_plan=box_plan,
        )
        result["candidate_artifacts"]["place_inside_candidates"] = write_json(run_dir / "plans" / "place_inside_candidates.json", place_candidates)
        place_plan = _preview_place_inside(
            run_dir=run_dir,
            candidates=place_candidates,
            freespace_move=freespace_move,
            box_plan=box_plan,
        )
        result["place_inside_plan"] = {
            "path": place_plan.get("path"),
            "selected": place_plan.get("selected"),
            "previewed": len(place_plan.get("candidates") or []),
        }
        if place_plan.get("selected") is None:
            result["why_stopped"] = "door open and box grasp previews succeeded, but no inside-place pose passed preview"
            result["next_recommendation"] = "Inspect place_inside_preview.json; tune interior depth/z offsets. Physical place still requires post-open interior observation."
            return result

        result["selected_strategy"] = {
            "door_open": door_plan["selected"],
            "box_grasp": place_plan["selected"].get("source_box_candidate") or box_plan["selected"],
            "place_inside": place_plan["selected"],
            "caveat": "inside-place target is a pre-open geometric rehearsal; physical task must re-observe interior after door opening before placing.",
        }
        result["plan_success"] = True
        result["success"] = True
        result["reward"] = 1.0
        result["why_stopped"] = "no-motion live observe/plan/preview succeeded; no hardware motion executed"
        result["next_recommendation"] = "For physical work, request a bounded refrigerator-and-box ticket with staged observe/open/re-observe/pick/place/release/home permissions."
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and run log before retrying no-motion."
        result["error"] = result["why_stopped"]
    finally:
        recording_evidence = _recording_evidence(Path(run_dir))
        result["recording_evidence"] = recording_evidence
        result["video_paths"] = recording_evidence.get("video_paths", [])
        write_json(Path(run_dir) / "result.json", result)
        write_json(Path(run_dir) / "task_result.json", result)
        summary_lines = [
            "## recording evidence",
            f"- required: {recording_evidence['required']}",
            f"- passed: {recording_evidence['passed']}",
            f"- video_paths: {recording_evidence.get('video_paths')}",
        ]
        if not recording_evidence.get("passed"):
            summary_lines.append(f"- problem: {recording_evidence.get('problem')}")
            summary_lines.append("- blocker_before_physical: True")
        append_stage_summary(run_dir, summary_lines)
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


TASK_RESULT.update(_run())
