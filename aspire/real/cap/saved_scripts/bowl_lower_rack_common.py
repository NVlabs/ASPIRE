# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lower-rack bowl placement implementation for real YAM.

The script always starts with fresh live perception and a no-motion pickup
planner run. It then detects the lower rack target, previews a bounded
pickup/lift/front-entry sequence, and executes at most one physical attempt only
when the physical gate and ticket preconditions are present.

Object-specific entry scripts set their known-good pickup defaults, then import
and run this shared lower-rack implementation. The shared lower-rack path uses
cuRobo/planner motion for the large transport to a front-entry pose, then a
bounded direct IK keypoint adjustment for the final lower-shelf insertion and
horizontal retreat.

Rim-angle tuning note for white-bowl/lower-rack work:

- Trying another rim is reasonable, especially for lower-rack placement.
- Current white-bowl pickup has used:
  OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG=-90
- Planner rim angles are world-frame radial directions:
  0 deg    = +X rim
  90 deg   = +Y rim
  180 deg  = -X rim
  -90 deg  = -Y rim
- If the right-side rim in the top view is a bad grasp side, do not keep only
  -90. Prefer an ordered sweep such as:
  OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG=-90,0,180,90
  This tries the known pickup first, then lets no-motion preview choose another
  valid rim if the current side fails.
- If the desired grasp is the bottom part in the top image, first run a
  no-motion overlay and inspect the selected rim_angle_deg. Image-bottom is
  camera-frame language, but the planner uses world-frame angles; put the
  mapped world angle first only after checking the overlay.
- Do not blindly switch physical grasp side without the no-motion overlay.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    compare_detection_motion,
    current_run_dir,
    json_safe,
    read_gripper_width,
    staged_close_with_contact,
    verify_lift,
    write_json,
)


def _set_default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _configure_lower_rack_defaults() -> None:
    object_name = (
        os.environ.get("OPENFORGE_BOWL_LOWER_RACK_OBJECT_NAME")
        or os.environ.get("OPENFORGE_BOWL_RACK_OBJECT_NAME")
        or "orange bowl"
    ).strip()
    object_name = object_name or "orange bowl"
    object_slug = object_name.replace(" ", "_")
    visible_env = (
        os.environ.get("OPENFORGE_BOWL_LOWER_RACK_VISIBLE_ENV")
        or os.environ.get("OPENFORGE_BOWL_RACK_VISIBLE_ENV")
        or ("OPENFORGE_BOWL_RACK_WHITE_BOWL_VISIBLE" if object_name.lower().startswith("white") else "OPENFORGE_BOWL_RACK_ORANGE_BOWL_VISIBLE")
    )
    _set_default_env("OPENFORGE_BOWL_RACK_TASK_NAME", "bowl_lower_rack_common")
    _set_default_env("OPENFORGE_BOWL_RACK_OBJECT_NAME", object_name)
    _set_default_env("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", f"{object_slug}_lower_rack")
    _set_default_env("OPENFORGE_BOWL_RACK_SCRIPT_PATH", "cap/saved_scripts/bowl_lower_rack_common.py")
    _set_default_env("OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH", "cap/saved_scripts/bowl_lower_rack_common.py")
    _set_default_env("OPENFORGE_BOWL_RACK_VISIBLE_ENV", visible_env)
    _set_default_env("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "bowl_lower_rack")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", f"{object_name},bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS", f"{object_name},bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_POST_PROMPTS", f"{object_name},bowl,lower rack shelf,bottom shelf,lower shelf,rack,wooden rack,blue gripper fingers")
    _set_default_env("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", f"{object_name},bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_TARGET_PROMPTS", "lower rack shelf,bottom shelf,lower shelf,first rack level,wooden rack,rack,wooden shelf,shelf")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", "0")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", "0")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "top_down_bowl_rim")
    _set_default_env("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "top_down_bowl_rim_straddle")
    _set_default_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", "0")
    _set_default_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", "0")
    _set_default_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", "1")
    _set_default_env("OPENFORGE_PLATE_USE_TOP_BOX_RADIUS", "1")
    _set_default_env("OPENFORGE_PLATE_REFERENCE_BOX_LONG_AXIS_PX", "56.0")
    _set_default_env("OPENFORGE_PLATE_REFERENCE_RADIUS_M", "0.095")
    _set_default_env("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", "0.68")
    _set_default_env("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", "0.60")
    _set_default_env("OPENFORGE_BOWL_RACK_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.04")
    _set_default_env("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", "0.04")
    _set_default_env("OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS", "0.02")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_USE_MULTIVIEW_RIM_Z", "1")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_RIM_Z_BIAS_M", "0.080")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_RAISE_M", "0.035")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_XY_DISAGREEMENT_M", "0.080")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M", "-0.015")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_LIFT_Z_M", "0.150")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG", "-90")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_TCP_TO_CONTACT_OFFSETS_M", "0.000")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_FINGER_FORWARD_SIGNS", "-1.0")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.04")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TARGET_POS", "0.04")
    _set_default_env("OPENFORGE_DEBUG_OBS_CAMERAS", "top,left,right,bottom")
    _set_default_env("OPENFORGE_DEBUG_OBS_IMAGE_ONLY_CAMERAS", "bottom")
    _set_default_env("OPENFORGE_BOWL_RACK_IMAGE_ONLY_CAMERAS", "bottom")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_STANDOFF_M", "0.120")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_X_STANDOFF_M", "0.000")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_Z_LIFT_M", "0.080")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_ENTRY_Z_LIFT_M", "0.035")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_CLEARANCE_M", "0.010")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_OBJECT_REF_ABOVE_SUPPORT_M", "0.035")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_OFFSET_M", "0.000")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_RETREAT_STANDOFF_M", "0.120")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT", "0")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_XY_DELTA_M", "0.150")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_Z_DELTA_M", "0.050")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", "0.160")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_Z_DELTA_M", "0.020")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MIN_DELTA_M", "0.003")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_IK_ERROR_M", "0.015")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_JOINT_DELTA_RAD", "0.50")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_OTHER_TCP_MOTION_M", "0.010")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_ROT_DELTA_DEG", "5.0")
    _set_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_STEP_S", "0.80")


_configure_lower_rack_defaults()

TASK_NAME = os.environ.get("OPENFORGE_BOWL_RACK_TASK_NAME", "bowl_lower_rack_common").strip()
TASK_NAME = TASK_NAME or "bowl_lower_rack_common"
TASK_OBJECT_NAME = os.environ.get("OPENFORGE_BOWL_RACK_OBJECT_NAME", "orange bowl").strip() or "orange bowl"
TASK_ARTIFACT_PREFIX = (
    os.environ.get("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", TASK_OBJECT_NAME.replace(" ", "_")).strip()
    or TASK_OBJECT_NAME.replace(" ", "_")
)
TASK_TICKET_SCOPE = os.environ.get("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "bowl_lower_rack").strip()
TASK_TICKET_SCOPE = TASK_TICKET_SCOPE or "bowl_lower_rack"
TASK_VISIBLE_ENV = os.environ.get(
    "OPENFORGE_BOWL_RACK_VISIBLE_ENV",
    "OPENFORGE_BOWL_RACK_ORANGE_BOWL_VISIBLE",
).strip() or "OPENFORGE_BOWL_RACK_ORANGE_BOWL_VISIBLE"
TASK_SCRIPT_PATH = os.environ.get(
    "OPENFORGE_BOWL_RACK_SCRIPT_PATH",
    "cap/saved_scripts/bowl_lower_rack_common.py",
).strip() or "cap/saved_scripts/bowl_lower_rack_common.py"
TASK_IMPLEMENTATION_PATH = os.environ.get(
    "OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH",
    "cap/saved_scripts/bowl_lower_rack_common.py",
).strip() or "cap/saved_scripts/bowl_lower_rack_common.py"
PLANNER_PATH = Path(os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_PATH", "cap/saved_scripts/pick_plate_no_motion_plan.py"))
REQUIRED_PICKUP_STRATEGY_MODE = "top_down_bowl_rim"
REQUIRED_PICKUP_STRATEGY_TYPE = "top_down_bowl_rim_straddle"
ALLOWED_TOP_DOWN_MODES = {"top_down_bowl_rim", "bowl_top_down", "top_down_rim_straddle"}
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _pickup_strategy_mode() -> str:
    raw = os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "").strip()
    return raw or REQUIRED_PICKUP_STRATEGY_MODE


def _allowed_pickup_strategy_types() -> set[str]:
    raw = os.environ.get("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "").strip()
    if not raw:
        return {REQUIRED_PICKUP_STRATEGY_TYPE}
    return set(_csv("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", [REQUIRED_PICKUP_STRATEGY_TYPE]))


def _axis_gate_required_for_pickup() -> bool:
    return _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_AXIS_GATE", True)


def _open_front_rack_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT", False)


def _open_front_slide_enabled() -> bool:
    return _open_front_rack_enabled() and _truthy_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE", False)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_float_setting(name: str) -> tuple[bool, float | None]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return False, None
    value = raw.strip()
    if value.lower() in {"none", "null", "omit", "default"}:
        return True, None
    try:
        return True, float(value)
    except ValueError:
        return False, None


def _gripper_limit_kwargs() -> dict[str, float | None]:
    kwargs: dict[str, float | None] = {}
    vel_is_set, vel_limit = _env_optional_float_setting("OPENFORGE_BOWL_RACK_GRIPPER_VEL_LIMIT")
    torque_is_set, torque_limit = _env_optional_float_setting("OPENFORGE_BOWL_RACK_GRIPPER_TORQUE_LIMIT")
    if vel_is_set:
        kwargs["vel_limit"] = vel_limit
    if torque_is_set:
        kwargs["torque_limit"] = torque_limit
    return kwargs


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


def _csv_float(name: str, default: list[float]) -> list[float]:
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


def _default_pickup_prompts() -> str:
    return os.environ.get(
        "OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS",
        f"{TASK_OBJECT_NAME},bowl",
    )


def _default_post_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_BOWL_RACK_DEFAULT_POST_PROMPTS",
        [TASK_OBJECT_NAME, "bowl", "rack", "wooden rack", "blue gripper fingers"],
    )


def _object_detection_prompts() -> tuple[str, ...]:
    values: list[str] = []
    for prompt in _csv("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", [TASK_OBJECT_NAME, "bowl"]):
        lower = prompt.lower()
        if lower not in values:
            values.append(lower)
    return tuple(values)


def _offset_triples(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [list(item) for item in default]
    values: list[list[float]] = []
    for chunk in raw.replace("|", ";").split(";"):
        parts = [part.strip() for part in chunk.replace(":", ",").split(",") if part.strip()]
        if len(parts) not in {2, 3}:
            continue
        try:
            item = [float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) == 3 else 0.0]
        except ValueError:
            continue
        values.append(item)
    return values or [list(item) for item in default]


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


def _display_rpy_to_quat_xyzw(rpy: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(list(rpy)[:3], dtype=np.float64)
    # Saved-script display RPY convention matches ScriptedPolicy/Viser:
    # scipy xyz = [-pitch, roll, -yaw - 90].  Keep this conversion local so
    # post-lift wrist-roll FK can propagate the actual held orientation into
    # the later rack transport poses.
    return Rotation.from_euler("xyz", [-pitch, roll, -yaw - 90.0], degrees=True).as_quat()


def _quat_xyzw_to_display_rpy(quat_xyzw: Any) -> list[float]:
    from scipy.spatial.transform import Rotation

    euler_xyz = Rotation.from_quat(np.asarray(list(quat_xyzw)[:4], dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    roll = float(euler_xyz[1])
    pitch = float(-euler_xyz[0])
    yaw = float(-(euler_xyz[2] + 90.0))
    return [roll, pitch, yaw]


def _post_lift_j5_reorient_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_REORIENT", False)


def _post_lift_j5_joint_index() -> int:
    index = _env_int("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_JOINT_INDEX", 5)
    if index < 0 or index >= 6:
        raise RuntimeError("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_JOINT_INDEX must be in [0,5]")
    return index


def _post_lift_j5_delta_rad(delta_deg: float | None = None) -> float:
    if delta_deg is None:
        delta_deg = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_DEG", 90.0)
    max_abs_deg = abs(_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_MAX_ABS_DELTA_DEG", 100.0))
    if abs(delta_deg) > max_abs_deg + 1e-9:
        raise RuntimeError(
            "OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_DEG exceeds "
            f"OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_MAX_ABS_DELTA_DEG={max_abs_deg}"
        )
    return math.radians(delta_deg)


def _post_lift_j5_delta_candidates_deg() -> list[float]:
    default_deg = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_DEG", 90.0)
    raw = os.environ.get("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_CANDIDATES_DEG")
    values = _csv_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_CANDIDATES_DEG", [default_deg]) if raw else [default_deg]
    candidates: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(float(value), 5)
        if rounded in seen:
            continue
        _post_lift_j5_delta_rad(rounded)
        candidates.append(rounded)
        seen.add(rounded)
    return candidates or [default_deg]


def _post_lift_recompute_object_offset_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_RECOMPUTE_OBJECT_OFFSET", True)


def _post_lift_object_offset_axes() -> set[str]:
    raw = os.environ.get("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_OBJECT_OFFSET_AXES", "xy").strip().lower()
    if raw in {"", "none", "0", "false", "off"}:
        return set()
    axes = {axis.strip() for axis in raw.replace(",", "").replace(" ", "")}
    invalid = axes - {"x", "y", "z"}
    if invalid:
        raise RuntimeError(
            "OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_OBJECT_OFFSET_AXES may contain only x/y/z"
        )
    return axes


def _parse_xyz_axes_env(name: str, default: str) -> set[str]:
    raw = os.environ.get(name, default).strip().lower()
    if raw in {"", "none", "0", "false", "off"}:
        return set()
    axes = {axis.strip() for axis in raw.replace(",", "").replace(" ", "")}
    invalid = axes - {"x", "y", "z"}
    if invalid:
        raise RuntimeError(f"{name} may contain only x/y/z")
    return axes


def _split_direct_insert_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_SPLIT_Y_THEN_X", False)


def _split_direct_insert_final_axes() -> set[str]:
    return _parse_xyz_axes_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_FINAL_AXES", "x")


def _direct_place_down_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_Z", False)


def _combine_post_lift_orientation_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_COMBINE_POST_LIFT_ORIENTATION", False)


def _pre_insert_tilt_combined_with_post_lift() -> bool:
    return (
        _combine_post_lift_orientation_enabled()
        and _post_lift_j5_reorient_enabled()
        and _pre_insert_x_up_tilt_enabled()
    )


def _diagonal_retreat_after_release_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_AFTER_RELEASE", False)


def _post_lift_tangent_alignment_gate(selected_pickup: dict[str, Any], rotated_rpy: list[float]) -> dict[str, Any]:
    enabled = _truthy_env("OPENFORGE_BOWL_LOWER_RACK_REQUIRE_TANGENT_OBJECT_OFFSET", False)
    packet: dict[str, Any] = {"enabled": enabled, "passed": True, "why_stopped": None}
    try:
        offset_packet = _object_ref_offset_after_reorientation(selected_pickup, rotated_rpy)
        offset = np.asarray(offset_packet["rotated_world_object_ref_from_tcp_m"][:2], dtype=np.float64)
        norm = float(np.linalg.norm(offset))
        if norm <= 1e-8:
            raise RuntimeError("rotated TCP-to-object XY offset is too small to define a tangent direction")
        unit = offset / norm
        tangent_axis_name = os.environ.get("OPENFORGE_BOWL_LOWER_RACK_TANGENT_AXIS", "x").strip().lower()
        if tangent_axis_name in {"x", "shelf_x", "shelf_tangent"}:
            tangent_axis = np.asarray([1.0, 0.0], dtype=np.float64)
            cross_axis = np.asarray([0.0, 1.0], dtype=np.float64)
        elif tangent_axis_name in {"y", "shelf_y"}:
            tangent_axis = np.asarray([0.0, 1.0], dtype=np.float64)
            cross_axis = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            raise RuntimeError("OPENFORGE_BOWL_LOWER_RACK_TANGENT_AXIS must be x/shelf_tangent or y")
        tangent_alignment = abs(float(np.dot(unit, tangent_axis)))
        cross_alignment = abs(float(np.dot(unit, cross_axis)))
        min_tangent_alignment = _env_float("OPENFORGE_BOWL_LOWER_RACK_MIN_TANGENT_ALIGNMENT", 0.80)
        max_cross_alignment = _env_float("OPENFORGE_BOWL_LOWER_RACK_MAX_CROSS_ALIGNMENT", 0.60)
        passed = tangent_alignment >= min_tangent_alignment and cross_alignment <= max_cross_alignment
        packet.update(
            {
                "passed": bool(passed),
                "tangent_axis": tangent_axis_name,
                "rotated_object_ref_from_tcp_m": offset_packet["rotated_world_object_ref_from_tcp_m"],
                "object_ref_xy_unit": _round_list(unit, limit=2),
                "xy_offset_norm_m": round(norm, 5),
                "tangent_alignment": round(tangent_alignment, 5),
                "cross_alignment": round(cross_alignment, 5),
                "min_tangent_alignment": round(min_tangent_alignment, 5),
                "max_cross_alignment": round(max_cross_alignment, 5),
                "offset_model": json_safe(offset_packet),
                "policy": (
                    "Prefer the TCP-to-object-center line tangent to the shelf front, so the wrist "
                    "and gripper do not point into the shelf depth under the upper shelf."
                ),
            }
        )
        if enabled and not passed:
            packet["why_stopped"] = (
                "post-lift wrist orientation is not shelf-tangent enough: "
                f"tangent_alignment={tangent_alignment:.3f}, cross_alignment={cross_alignment:.3f}"
            )
    except Exception as exc:
        packet.update({"passed": not enabled, "why_stopped": f"{type(exc).__name__}: {exc}"})
    return packet


def _object_ref_offset_after_reorientation(
    selected_pickup: dict[str, Any], rotated_rpy: list[float]
) -> dict[str, Any]:
    from scipy.spatial.transform import Rotation

    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    grasp_xyz = np.asarray(list(grasp_pose.get("position") or [])[:3], dtype=np.float64)
    grasp_rpy = list(grasp_pose.get("rpy") or [])[:3]
    if source_xyz is None or grasp_xyz.shape[0] < 3 or len(grasp_rpy) < 3 or len(rotated_rpy) < 3:
        raise RuntimeError("cannot compute post-lift object offset without source xyz, grasp xyz/rpy, and rotated rpy")
    source = np.asarray(source_xyz[:3], dtype=np.float64)
    initial_world_offset = source - grasp_xyz
    max_offset = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_OBJECT_OFFSET_MAX_M", 0.200)
    offset_norm = float(np.linalg.norm(initial_world_offset))
    if offset_norm > max_offset:
        raise RuntimeError(
            f"object reference offset {offset_norm:.3f} m exceeds "
            f"OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_OBJECT_OFFSET_MAX_M={max_offset:.3f}"
        )
    initial_rot = Rotation.from_quat(_display_rpy_to_quat_xyzw(grasp_rpy))
    rotated_rot = Rotation.from_quat(_display_rpy_to_quat_xyzw(rotated_rpy))
    local_offset = initial_rot.inv().apply(initial_world_offset)
    rotated_world_offset = rotated_rot.apply(local_offset)
    return {
        "source_detection_xyz_m": _round_list(source, limit=3),
        "grasp_tcp_xyz_m": _round_list(grasp_xyz, limit=3),
        "initial_grasp_rpy": _round_list(grasp_rpy, limit=3),
        "rotated_rpy": _round_list(rotated_rpy, limit=3),
        "initial_world_object_ref_from_tcp_m": _round_list(initial_world_offset, limit=3),
        "local_object_ref_from_tcp_m": _round_list(local_offset, limit=3),
        "rotated_world_object_ref_from_tcp_m": _round_list(rotated_world_offset, limit=3),
        "offset_norm_m": round(offset_norm, 5),
        "max_offset_m": round(max_offset, 5),
    }


def _patch_place_candidate_after_reorientation(
    candidate: dict[str, Any],
    selected_pickup: dict[str, Any],
    rotated_rpy: list[float],
    *,
    source: str,
    reorientation: dict[str, Any],
) -> dict[str, Any]:
    if len(rotated_rpy) < 3:
        raise RuntimeError("cannot patch lower-rack place candidate without a 3-value rotated rpy")
    patched = copy.deepcopy(candidate)
    for key in ("front_high_pose", "front_entry_pose", "open_front_above_pose", "release_pose", "retreat_pose"):
        pose = patched.get(key)
        if isinstance(pose, dict):
            pose["rpy"] = [float(value) for value in rotated_rpy[:3]]
    geometry = dict(patched.get("geometry_model") or {})
    object_offset_packet: dict[str, Any] | None = None
    axes = _post_lift_object_offset_axes()
    if _post_lift_recompute_object_offset_enabled() and axes:
        object_offset_packet = _object_ref_offset_after_reorientation(selected_pickup, rotated_rpy)
        rotated_offset = np.asarray(
            object_offset_packet["rotated_world_object_ref_from_tcp_m"], dtype=np.float64
        )
        place_ref = patched.get("place_target_object_ref_m") or []
        release_pose = patched.get("release_pose") or {}
        release_position = list(release_pose.get("position") or [])
        if len(place_ref) < 3 or len(release_position) < 3:
            raise RuntimeError("cannot recompute lower-rack TCP target without object-ref and release positions")
        release_target = np.asarray(release_position[:3], dtype=np.float64)
        object_ref_target = np.asarray(place_ref[:3], dtype=np.float64)
        adjusted_release = release_target.copy()
        for index, axis in enumerate(("x", "y", "z")):
            if axis in axes:
                adjusted_release[index] = object_ref_target[index] - rotated_offset[index]

        release_tcp_clamp: dict[str, Any] | None = None
        x_min_set, x_min_value = _env_optional_float_setting("OPENFORGE_BOWL_LOWER_RACK_RELEASE_TCP_X_MIN_M")
        x_max_set, x_max_value = _env_optional_float_setting("OPENFORGE_BOWL_LOWER_RACK_RELEASE_TCP_X_MAX_M")
        original_adjusted_release = adjusted_release.copy()
        original_object_ref_target = object_ref_target.copy()
        if x_min_set and x_min_value is not None and adjusted_release[0] < float(x_min_value):
            adjusted_release[0] = float(x_min_value)
        if x_max_set and x_max_value is not None and adjusted_release[0] > float(x_max_value):
            adjusted_release[0] = float(x_max_value)
        if abs(float(adjusted_release[0] - original_adjusted_release[0])) > 1e-9:
            if "x" in axes:
                object_ref_target[0] = adjusted_release[0] + rotated_offset[0]
            release_tcp_clamp = {
                "env_min_m": x_min_value if x_min_set else None,
                "env_max_m": x_max_value if x_max_set else None,
                "original_release_tcp_m": _round_list(original_adjusted_release, limit=3),
                "clamped_release_tcp_m": _round_list(adjusted_release, limit=3),
                "original_object_ref_target_m": _round_list(original_object_ref_target, limit=3),
                "clamped_object_ref_target_m": _round_list(object_ref_target, limit=3),
                "policy": (
                    "Clamp the release TCP to the reachable shelf-insert band. "
                    "When object-offset recompute owns X, shift the object reference target "
                    "with the clamped TCP so logs match the actual placement intent."
                ),
            }

        front_x_standoff = float(geometry.get("front_x_standoff_m", 0.0) or 0.0)
        front_standoff = float(geometry.get("front_standoff_m", 0.0) or 0.0)
        retreat_standoff = float(geometry.get("retreat_standoff_m", front_standoff) or front_standoff)
        front_z_lift = float(geometry.get("front_z_lift_m", 0.0) or 0.0)
        entry_z_lift = float(geometry.get("entry_z_lift_m", 0.0) or 0.0)
        open_front_above_z_lift = float(geometry.get("open_front_above_z_lift_m", front_z_lift) or front_z_lift)
        front_side_sign = float(geometry.get("front_side_sign_y", 1.0) or 1.0)
        open_front_slide = bool(geometry.get("open_front_slide"))
        release_xy = adjusted_release[:2]
        front_xy = [release_xy[0] - front_x_standoff, release_xy[1] - front_side_sign * front_standoff]
        retreat_xy = [release_xy[0] - front_x_standoff, release_xy[1] - front_side_sign * retreat_standoff]
        patched["place_target_object_ref_m"] = _round_list(object_ref_target, digits=8, limit=3)
        patched["release_pose"]["position"] = _round_list(adjusted_release, digits=8, limit=3)
        patched["front_high_pose"]["position"] = [
            float(front_xy[0]),
            float(front_xy[1]),
            float(adjusted_release[2] + front_z_lift),
        ]
        patched["front_entry_pose"]["position"] = [
            float(front_xy[0]),
            float(front_xy[1]),
            float(adjusted_release[2] if open_front_slide else adjusted_release[2] + entry_z_lift),
        ]
        if isinstance(patched.get("open_front_above_pose"), dict):
            patched["open_front_above_pose"]["position"] = [
                float(release_xy[0]),
                float(release_xy[1]),
                float(adjusted_release[2] + open_front_above_z_lift),
            ]
        if open_front_slide:
            patched["retreat_pose"]["position"] = [
                float(retreat_xy[0]),
                float(retreat_xy[1]),
                float(adjusted_release[2]),
            ]
        elif bool(geometry.get("open_front_rack")):
            patched["retreat_pose"]["position"] = [
                float(release_xy[0]),
                float(release_xy[1]),
                float(adjusted_release[2] + open_front_above_z_lift),
            ]
        else:
            patched["retreat_pose"]["position"] = [
                float(retreat_xy[0]),
                float(retreat_xy[1]),
                float(adjusted_release[2]),
            ]
        object_offset_packet.update(
            {
                "enabled": True,
                "applied_axes": sorted(axes),
                "original_release_tcp_m": _round_list(release_target, limit=3),
                "adjusted_release_tcp_m": _round_list(adjusted_release, limit=3),
                "object_ref_target_m": _round_list(object_ref_target, limit=3),
                "release_tcp_clamp": release_tcp_clamp,
                "policy": (
                    "Keep the tuned lower-rack object reference target fixed, rotate the "
                    "TCP-to-object offset through the post-lift j5 FK orientation, then "
                    "solve the release TCP from object_ref_target - rotated_offset. "
                    "By default only XY is adjusted; Z remains the tuned shelf-clearance height."
                ),
            }
        )
    geometry["orientation_policy"] = "post_lift_direct_j5_reorientation"
    geometry["post_lift_orientation_source"] = source
    geometry["post_lift_rotated_rpy"] = _round_list(rotated_rpy, limit=3)
    geometry["post_lift_reorientation"] = json_safe(reorientation)
    geometry["post_lift_object_offset_recomputed"] = bool(object_offset_packet)
    if object_offset_packet is not None:
        geometry["post_lift_object_offset_model"] = json_safe(object_offset_packet)
        if object_offset_packet.get("release_tcp_clamp"):
            geometry["reachable_release_tcp_clamp"] = json_safe(object_offset_packet["release_tcp_clamp"])
    patched["geometry_model"] = geometry
    patched["post_lift_reorientation"] = json_safe(reorientation)
    if object_offset_packet is not None:
        patched["post_lift_object_offset_model"] = json_safe(object_offset_packet)
    return patched


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


def _xyz(value: Any) -> list[float] | None:
    if not value:
        return None
    raw = value.get("position_3d") or value.get("position") if isinstance(value, dict) else value
    if raw is None or len(raw) < 3:
        return None
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def _distance3(a: Any, b: Any) -> float | None:
    av = _xyz(a)
    bv = _xyz(b)
    if av is None or bv is None:
        return None
    return math.sqrt(sum((float(av[index]) - float(bv[index])) ** 2 for index in range(3)))


def _dot3(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        av = [float(value) for value in list(a)[:3]]
        bv = [float(value) for value in list(b)[:3]]
    except Exception:
        return None
    if len(av) < 3 or len(bv) < 3:
        return None
    return sum(av[index] * bv[index] for index in range(3))


def _quat_xyzw_axes_world(quat: Any) -> dict[str, Any]:
    try:
        x, y, z, w = [float(value) for value in list(quat)[:4]]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return {"error": "zero-length quaternion"}
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    return {
        "source": "robot_state_ee_quat_xyzw",
        "local_x_world": _round_list([matrix[row][0] for row in range(3)], limit=3),
        "local_y_world": _round_list([matrix[row][1] for row in range(3)], limit=3),
        "local_z_world": _round_list([matrix[row][2] for row in range(3)], limit=3),
    }


def _arm_state_from_observation(packet: dict[str, Any], side: str) -> dict[str, Any] | None:
    robot_state = packet.get("robot_state") or {}
    arms = robot_state.get("arms") if isinstance(robot_state, dict) else None
    arm = (arms or {}).get(side)
    return dict(arm) if isinstance(arm, dict) else None


def _pose_available(pose: Any) -> bool:
    return bool(
        isinstance(pose, dict)
        and len(pose.get("position") or []) >= 3
        and len(pose.get("rpy") or []) >= 3
    )


def _artifact_safe_label(value: Any) -> str:
    text = str(value or "candidate")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:80] or "candidate"


def _debug_observation_cameras() -> list[str]:
    raw = os.environ.get("OPENFORGE_BOWL_RACK_OBSERVATION_CAMERAS")
    if raw and raw.strip():
        cameras = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    else:
        cameras = _csv("OPENFORGE_DEBUG_OBS_CAMERAS", ["top", "left", "right", "bottom"])
    return cameras or ["top", "left", "right", "bottom"]


def _image_only_debug_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_BOWL_RACK_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_BOWL_RACK_RECORDING_WAIT_S", 5.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    packet: dict[str, Any] = {"required": True, "passed": bool(videos), "video_paths": videos}
    if not videos:
        packet.update(
            {
                "problem": "no mp4 files found; recording/debug evidence is required before physical motion",
                "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
                "blocker_before_physical": True,
            }
        )
    return packet


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_PLATE_XYZ",
        "OPENFORGE_PLATE_HALF_EXTENTS",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
        "OPENFORGE_BOWL_XYZ",
        "OPENFORGE_RACK_XYZ",
        "OPENFORGE_BOWL_RACK_OBJECT_XYZ",
        "OPENFORGE_BOWL_RACK_RACK_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _forbidden_pickup_strategy_env() -> list[str]:
    problems: list[str] = []
    if _truthy_env("OPENFORGE_BOWL_RACK_ALLOW_NON_TOP_DOWN_PICKUP_STRATEGY", False):
        return problems
    requested_mode = _pickup_strategy_mode().strip().lower()
    if requested_mode and requested_mode not in ALLOWED_TOP_DOWN_MODES:
        problems.append(
            "OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE must be one of "
            f"{sorted(ALLOWED_TOP_DOWN_MODES)} for this task; got {requested_mode!r}"
        )
    disallowed = sorted(item for item in _allowed_pickup_strategy_types() if item != REQUIRED_PICKUP_STRATEGY_TYPE)
    if disallowed:
        problems.append(
            "OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES may not admit non-top-down strategies; "
            f"disallowed={disallowed}"
        )
    return problems


def _pgrep_competing_motion(run_dir: Path | None = None) -> dict[str, Any]:
    try:
        output = subprocess.check_output(["pgrep", "-af", "run_script.py|ffmpeg"], text=True)
    except subprocess.CalledProcessError:
        output = ""
    self_pid = os.getpid()
    rows: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            pid = -1
        cmd = parts[1] if len(parts) > 1 else ""
        item = {"pid": pid, "cmd": cmd}
        rows.append(item)
        first_token = cmd.split(maxsplit=1)[0] if cmd else ""
        first_name = Path(first_token).name
        actual_ffmpeg = first_name == "ffmpeg" or first_token.endswith("/ffmpeg")
        actual_run_script = "run_script.py" in cmd and (
            first_name.startswith("python") or first_name in {"uv", "uvx"} or "/python" in first_token
        )
        own_recording = run_dir is not None and str(run_dir) in cmd and actual_ffmpeg
        if pid != self_pid and not own_recording and "debug_ui" not in cmd and (actual_ffmpeg or actual_run_script):
            competitors.append(item)
    return {"self_pid": self_pid, "rows": rows, "competitors": competitors}


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_path = Path(TASK_SCRIPT_PATH).resolve()
    implementation_path = Path(TASK_IMPLEMENTATION_PATH).resolve()
    snapshot = {
        "script_path": str(script_path),
        "implementation_path": str(implementation_path),
        "planner_path": str((Path.cwd() / PLANNER_PATH).resolve()),
        "task_name": TASK_NAME,
        "object_name": TASK_OBJECT_NAME,
        "artifact_prefix": TASK_ARTIFACT_PREFIX,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_BOWL_RACK_")
            or name.startswith("OPENFORGE_BOWL_LOWER_RACK_")
            or name.startswith("OPENFORGE_BOWL_TOP_DOWN_")
            or name.startswith("OPENFORGE_PLATE_")
            or name
            in {
                "OPENFORGE_ALLOW_PHYSICAL_MOTION",
                "OPENFORGE_DEBUG_OBS_TIMEOUT_S",
                "YAM_STATION_CALIBRATED_XML",
                "CAP_TOP_CAMERA_BACKEND",
                "CAP_TOP_CAMERA_FRAME",
                "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP",
            }
        },
    }
    try:
        snapshot["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        paths = [str(script_path.relative_to(Path.cwd()))]
        implementation_rel = str(implementation_path.relative_to(Path.cwd()))
        if implementation_rel not in paths:
            paths.append(implementation_rel)
        snapshot["git_status_short"] = subprocess.check_output(["git", "status", "--short", "--", *paths], text=True).strip()
    except Exception as exc:
        snapshot["git_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["path"] = write_json(run_dir / "source_snapshot.json", snapshot)
    return snapshot


def _with_env_overrides(overrides: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for name, value in overrides.items():
        previous[name] = os.environ.get(name)
        os.environ[name] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _run_fresh_pickup_planner() -> dict[str, Any]:
    previous_gate = os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
    overrides = {
        "OPENFORGE_PLATE_TASK_NAME": f"{TASK_NAME}_pickup_plan",
        "OPENFORGE_PLATE_OBJECT_NAME": TASK_OBJECT_NAME,
        "OPENFORGE_PLATE_PROMPTS": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", _default_pickup_prompts()),
        "OPENFORGE_PLATE_COLLECT_ALL_PROMPTS": os.environ.get(
            "OPENFORGE_BOWL_RACK_COLLECT_ALL_PICKUP_PROMPTS", "1"
        ),
        "OPENFORGE_PLATE_TARGET_Y_SIGN": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_TARGET_Y_SIGN", "negative"),
        "OPENFORGE_PLATE_STRATEGY_MODE": _pickup_strategy_mode(),
        "OPENFORGE_PLATE_ARM": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_ARM", "right"),
        "OPENFORGE_PLATE_MAX_CANDIDATES": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_MAX_CANDIDATES", "96"),
        "OPENFORGE_PLATE_PLANNER_BACKEND": os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_BACKEND", "rrtconnect"),
        "OPENFORGE_PLATE_SOLVER_SPEED": os.environ.get("OPENFORGE_BOWL_RACK_SOLVER_SPEED", "fast"),
        "OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M": os.environ.get("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", "0.025"),
        "OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG": os.environ.get("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", "16"),
        "OPENFORGE_PLATE_IK_XYZ_WEIGHT": os.environ.get("OPENFORGE_BOWL_RACK_IK_XYZ_WEIGHT", "1.0"),
        "OPENFORGE_PLATE_IK_RPY_WEIGHT": os.environ.get("OPENFORGE_BOWL_RACK_IK_RPY_WEIGHT", "0.3"),
        "OPENFORGE_PLATE_PREVIEW_MAX_POS_ERROR_M": os.environ.get(
            "OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", "0.018"
        ),
        "OPENFORGE_PLATE_PREVIEW_MAX_ROT_ERROR_DEG": os.environ.get(
            "OPENFORGE_BOWL_RACK_PREVIEW_MAX_ROT_ERROR_DEG", "10"
        ),
    }
    previous = _with_env_overrides(overrides)
    try:
        planner_path = Path.cwd() / PLANNER_PATH
        spec = importlib.util.spec_from_file_location(f"openforge_{TASK_ARTIFACT_PREFIX}_pickup_plan", planner_path)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"could not locate planner path {PLANNER_PATH}")
        tool_globals = {
            name: value
            for name, value in globals().items()
            if callable(value)
            and name
            in {
                "detect_objects_oneshot",
                "freespace_move",
                "get_camera_image",
                "get_robot_state",
                "run_in_background",
            }
        }
        namespace = runpy.run_path(spec.origin, init_globals=tool_globals, run_name=f"__openforge_{TASK_ARTIFACT_PREFIX}_plan__")
        return dict(namespace.get("TASK_RESULT") or {})
    finally:
        _restore_env(previous)
        if previous_gate is not None:
            os.environ["OPENFORGE_ALLOW_PHYSICAL_MOTION"] = previous_gate


def _validate_pickup_strategy(planner_result: dict[str, Any]) -> dict[str, Any]:
    selected = planner_result.get("selected_strategy") or {}
    source = selected.get("source_detection") or {}
    source_name = str(source.get("source") or source.get("detection_source") or "")
    allowed_strategy_types = _allowed_pickup_strategy_types()
    required = {
        "selected.label": selected.get("label"),
        "selected.arm": selected.get("arm"),
        "selected.strategy_type": selected.get("strategy_type"),
        "pregrasp_pose.position": (selected.get("pregrasp_pose") or {}).get("position"),
        "grasp_pose.position": (selected.get("grasp_pose") or {}).get("position"),
        "lift_pose.position": (selected.get("lift_pose") or {}).get("position"),
        "source_detection.position_3d": source.get("position_3d"),
    }
    missing = [name for name, value in required.items() if not value]
    strategy_type = selected.get("strategy_type")
    strategy_ok = strategy_type in allowed_strategy_types
    residual_gate = selected.get("preview_residual_gate") or {}
    residual_ok = bool(residual_gate.get("passed", True))
    axis_gate = selected.get("axis_gate") or {}
    axis_required = _axis_gate_required_for_pickup()
    axis_ok = (not axis_required) or (bool(axis_gate) and axis_gate.get("passed") is True)
    label_text = str(selected.get("label") or "")
    label_ok = not label_text.startswith("plate_rim_side")
    rim_edge = selected.get("selected_rim_edge") or {}
    rim_edge_ok = bool(
        rim_edge.get("gap_center_m")
        or rim_edge.get("contact_center_m")
        or rim_edge.get("edge_point_m")
        or rim_edge.get("pinch_tcp_center_m")
    )
    source_camera = str(source.get("source_camera") or source.get("camera") or "")
    top_camera_required = _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_TOP_CAMERA_PICKUP", True)
    camera_ok = (not top_camera_required) or source_camera == "top"
    manual_source = "manual" in source_name.lower()
    passed = bool(
        planner_result.get("success")
        and selected
        and not missing
        and strategy_ok
        and label_ok
        and rim_edge_ok
        and residual_ok
        and axis_ok
        and camera_ok
        and not manual_source
    )
    return {
        "passed": passed,
        "missing": missing,
        "strategy_ok": strategy_ok,
        "strategy_type": strategy_type,
        "label_ok": label_ok,
        "rim_edge_ok": rim_edge_ok,
        "residual_gate_passed": residual_ok,
        "axis_gate_passed": axis_ok,
        "axis_gate": axis_gate,
        "axis_gate_required": axis_required,
        "camera_ok": camera_ok,
        "source_camera": source_camera,
        "manual_source": manual_source,
        "selected_label": selected.get("label"),
        "arm": selected.get("arm"),
        "allowed_strategy_types": sorted(allowed_strategy_types),
        "source_detection": source,
        "selected_strategy": selected if passed else None,
    }


def _validated_pickup_retry_strategies(planner_result: dict[str, Any]) -> list[dict[str, Any]]:
    max_attempts = max(1, _env_int("OPENFORGE_BOWL_RACK_PICKUP_RETRY_ATTEMPTS", 1))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _consider(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        label = str(candidate.get("label") or "")
        key = label or json.dumps(json_safe(candidate.get("grasp_pose") or candidate.get("position")), sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        candidate_result = dict(planner_result)
        candidate_result["selected_strategy"] = candidate
        candidate_result["success"] = bool(planner_result.get("success"))
        validation = _validate_pickup_strategy(candidate_result)
        if validation.get("passed"):
            item = dict(validation["selected_strategy"])
            item["pickup_retry_validation"] = {
                key: value for key, value in validation.items() if key != "selected_strategy"
            }
            candidates.append(item)

    _consider(planner_result.get("selected_strategy"))
    plan_path = ((planner_result.get("planner_result") or {}).get("plan_path") or "").strip()
    if plan_path:
        try:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            for candidate in plan.get("candidates") or []:
                if len(candidates) >= max_attempts:
                    break
                if not bool(candidate.get("preview_success")):
                    continue
                _consider(candidate)
        except Exception as exc:
            append_stage_summary(
                current_run_dir(TASK_NAME),
                [
                    "## pickup retry candidate load",
                    f"- plan_path: {plan_path}",
                    f"- error: {type(exc).__name__}: {exc}",
                ],
            )
    return candidates[:max_attempts]


def _pickup_retry_candidate_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "label": item.get("label"),
            "tcp_above_rim_m": item.get("tcp_above_rim_m")
            or (item.get("rim_geometry_estimate") or {}).get("tcp_above_rim_m"),
            "grasp_z_m": ((item.get("grasp_pose") or {}).get("position") or [None, None, None])[2],
            "source_detection_xyz": (item.get("source_detection") or {}).get("position_3d"),
        }
        for index, item in enumerate(candidates)
    ]


def _relocalized_pickup_retry(run_dir: Path, retry_index: int) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.pickup_retry_relocalization.v1",
        "retry_index": retry_index,
        "enabled": _truthy_env("OPENFORGE_BOWL_RACK_RELOCALIZE_BETWEEN_PICKUP_RETRIES", True),
        "success": False,
        "why_stopped": None,
        "planner_result_path": None,
        "pickup_validation": None,
        "candidate_summary": [],
        "selected_candidate": None,
    }
    if not packet["enabled"]:
        packet["success"] = True
        packet["why_stopped"] = "pickup retry relocalization disabled"
        return packet

    planner_result = _run_fresh_pickup_planner()
    packet["planner_result_path"] = write_json(
        run_dir / "plans" / f"fresh_{TASK_ARTIFACT_PREFIX}_pickup_planner_result_retry_{retry_index}.json",
        planner_result,
    )
    validation = _validate_pickup_strategy(planner_result)
    packet["pickup_validation"] = validation
    if not validation.get("passed"):
        packet["why_stopped"] = "fresh pickup relocalization did not produce a valid configured pickup strategy"
        packet["path"] = write_json(
            run_dir / "plans" / f"pickup_retry_{retry_index}_relocalization.json",
            json_safe(packet),
        )
        append_stage_summary(
            run_dir,
            [
                "## pickup retry relocalization",
                f"- retry_index: {retry_index}",
                "- success: False",
                f"- why_stopped: {packet['why_stopped']}",
                f"- planner_result: {packet.get('planner_result_path')}",
                f"- path: {packet.get('path')}",
            ],
        )
        return packet

    candidates = _validated_pickup_retry_strategies(planner_result)
    packet["candidate_summary"] = _pickup_retry_candidate_summary(candidates)
    if not candidates:
        packet["why_stopped"] = "fresh pickup relocalization produced no preview-passing retry candidates"
    else:
        packet["selected_candidate"] = candidates[0]
        packet["success"] = True
        packet["why_stopped"] = "fresh bowl relocalization selected next pickup retry candidate"

    packet["path"] = write_json(
        run_dir / "plans" / f"pickup_retry_{retry_index}_relocalization.json",
        json_safe(packet),
    )
    append_stage_summary(
        run_dir,
        [
            "## pickup retry relocalization",
            f"- retry_index: {retry_index}",
            f"- success: {packet['success']}",
            f"- why_stopped: {packet['why_stopped']}",
            f"- candidates: {packet['candidate_summary']}",
            f"- planner_result: {packet.get('planner_result_path')}",
            f"- path: {packet.get('path')}",
        ],
    )
    return packet


def _pickup_retry_allowed(physical_attempt: dict[str, Any]) -> bool:
    reason = str(physical_attempt.get("why_stopped") or "").lower()
    if not reason:
        return False
    retry_markers = [
        "staged close did not produce contact evidence",
        "staged close did not produce plausible",
        "lift perception verification failed",
    ]
    return any(marker in reason for marker in retry_markers)


def _home_between_pickup_retries(run_dir: Path, retry_index: int) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.pickup_retry_home.v1",
        "retry_index": retry_index,
        "enabled": _truthy_env("OPENFORGE_BOWL_RACK_GO_HOME_BETWEEN_PICKUP_RETRIES", True),
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "why_stopped": None,
    }
    if not packet["enabled"]:
        packet["success"] = True
        packet["why_stopped"] = "go-home between pickup retries disabled"
        return packet
    go_home = _tool("go_home")
    if go_home is None:
        packet["why_stopped"] = "go_home tool unavailable; refusing pickup retry from stale pose"
        return packet
    try:
        packet["result"] = json_safe(go_home())
        packet["physical_motion_executed"] = True
        packet["movement_capable_calls"].append("go_home:pickup_retry")
        packet["success"] = True
        packet["why_stopped"] = "go_home completed before next pickup retry"
        open_packet = _open_grippers_after_home(context=f"pickup_retry_{retry_index}")
        packet["open_grippers_after_home"] = open_packet
        if open_packet.get("physical_motion_executed"):
            packet["movement_capable_calls"].extend(list(open_packet.get("movement_capable_calls") or []))
        if not open_packet.get("success"):
            packet["success"] = False
            packet["why_stopped"] = open_packet.get("why_stopped")
    except Exception as exc:
        packet["physical_motion_executed"] = True
        packet["why_stopped"] = f"go_home failed before pickup retry: {type(exc).__name__}: {exc}"
    packet["path"] = write_json(
        run_dir / "plans" / f"pickup_retry_{retry_index}_go_home.json",
        json_safe(packet),
    )
    append_stage_summary(
        run_dir,
        [
            "## pickup retry home",
            f"- retry_index: {retry_index}",
            f"- success: {packet['success']}",
            f"- why_stopped: {packet['why_stopped']}",
            f"- path: {packet.get('path')}",
        ],
    )
    return packet


def _open_grippers_after_home(*, context: str) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.open_grippers_after_home.v1",
        "context": context,
        "enabled": _truthy_env("OPENFORGE_BOWL_RACK_OPEN_GRIPPERS_AFTER_HOME", False),
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "sides": [],
        "steps": [],
        "why_stopped": None,
    }
    if not packet["enabled"]:
        packet["success"] = True
        packet["why_stopped"] = "open-grippers-after-home disabled"
        return packet
    sides = [side for side in _csv("OPENFORGE_BOWL_RACK_OPEN_GRIPPERS_AFTER_HOME_SIDES", ["left", "right"]) if side in {"left", "right"}]
    packet["sides"] = sides
    if not sides:
        packet["why_stopped"] = "no valid gripper sides configured for post-home open"
        return packet
    open_gripper = _tool("open_gripper")
    set_gripper = _tool("set_gripper")
    if open_gripper is None and set_gripper is None:
        packet["why_stopped"] = "neither open_gripper nor set_gripper tool is available"
        return packet
    target = _env_float("OPENFORGE_BOWL_RACK_POST_HOME_OPEN_GRIPPER_POS", 1.0)
    for side in sides:
        try:
            if open_gripper is not None:
                result = open_gripper(side)
                call = f"open_gripper:{side}:after_home"
                tool_name = "open_gripper"
            else:
                result = set_gripper(side, target, **_gripper_limit_kwargs())
                call = f"set_gripper:{side}:after_home"
                tool_name = "set_gripper"
            packet["physical_motion_executed"] = True
            packet["movement_capable_calls"].append(call)
            packet["steps"].append(
                {
                    "stage": f"open_{side}_gripper_after_home",
                    "side": side,
                    "tool": tool_name,
                    "target": target if tool_name == "set_gripper" else None,
                    "ok": True,
                    "result": json_safe(result),
                }
            )
        except Exception as exc:
            packet["physical_motion_executed"] = True
            packet["steps"].append(
                {
                    "stage": f"open_{side}_gripper_after_home",
                    "side": side,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    failed = [step for step in packet["steps"] if not bool(step.get("ok"))]
    packet["success"] = not failed
    packet["why_stopped"] = (
        "opened configured gripper(s) after home"
        if packet["success"]
        else f"failed to open {len(failed)} gripper(s) after home"
    )
    return packet


def _home_after_success(run_dir: Path) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.post_success_home.v1",
        "enabled": _truthy_env("OPENFORGE_BOWL_RACK_GO_HOME_AFTER_SUCCESS", False),
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "why_stopped": None,
    }
    if not packet["enabled"]:
        packet["success"] = True
        packet["why_stopped"] = "post-success go-home disabled"
        return packet
    go_home = _tool("go_home")
    if go_home is None:
        packet["why_stopped"] = "go_home tool unavailable after successful placement"
        return packet
    try:
        packet["result"] = json_safe(go_home())
        packet["physical_motion_executed"] = True
        packet["movement_capable_calls"].append("go_home:post_success")
        packet["success"] = True
        packet["why_stopped"] = "go_home completed after successful placement"
        open_packet = _open_grippers_after_home(context="post_success")
        packet["open_grippers_after_home"] = open_packet
        if open_packet.get("physical_motion_executed"):
            packet["movement_capable_calls"].extend(list(open_packet.get("movement_capable_calls") or []))
        if not open_packet.get("success"):
            packet["success"] = False
            packet["why_stopped"] = open_packet.get("why_stopped")
    except Exception as exc:
        packet["physical_motion_executed"] = True
        packet["why_stopped"] = f"go_home failed after successful placement: {type(exc).__name__}: {exc}"
    packet["path"] = write_json(
        run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_success_go_home.json",
        json_safe(packet),
    )
    append_stage_summary(
        run_dir,
        [
            "## post-success home",
            f"- enabled: {packet['enabled']}",
            f"- success: {packet['success']}",
            f"- why_stopped: {packet['why_stopped']}",
            f"- path: {packet.get('path')}",
        ],
    )
    return packet


def _home_after_failure(run_dir: Path, failure_reason: str | None) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.post_failure_home.v1",
        "enabled": _truthy_env("OPENFORGE_BOWL_RACK_GO_HOME_AFTER_FAILURE", False),
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "failure_reason": failure_reason,
        "why_stopped": None,
    }
    if not packet["enabled"]:
        packet["success"] = True
        packet["why_stopped"] = "post-failure go-home disabled"
        return packet
    go_home = _tool("go_home")
    if go_home is None:
        packet["why_stopped"] = "go_home tool unavailable after failed physical attempt"
        return packet
    try:
        packet["result"] = json_safe(go_home())
        packet["physical_motion_executed"] = True
        packet["movement_capable_calls"].append("go_home:post_failure")
        packet["success"] = True
        packet["why_stopped"] = "go_home completed after failed physical attempt"
        open_packet = _open_grippers_after_home(context="post_failure")
        packet["open_grippers_after_home"] = open_packet
        if open_packet.get("physical_motion_executed"):
            packet["movement_capable_calls"].extend(list(open_packet.get("movement_capable_calls") or []))
        if not open_packet.get("success"):
            packet["success"] = False
            packet["why_stopped"] = open_packet.get("why_stopped")
    except Exception as exc:
        packet["physical_motion_executed"] = True
        packet["why_stopped"] = f"go_home failed after failed physical attempt: {type(exc).__name__}: {exc}"
    packet["path"] = write_json(
        run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_failure_go_home.json",
        json_safe(packet),
    )
    append_stage_summary(
        run_dir,
        [
            "## post-failure home",
            f"- enabled: {packet['enabled']}",
            f"- success: {packet['success']}",
            f"- failure_reason: {failure_reason}",
            f"- why_stopped: {packet['why_stopped']}",
            f"- path: {packet.get('path')}",
        ],
    )
    return packet


def _detection_touches_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_BOWL_RACK_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_BOWL_RACK_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_BOWL_RACK_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _select_rack_detection(packet: dict[str, Any]) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    min_score = _env_float("OPENFORGE_BOWL_RACK_MIN_SCORE", 0.20)
    configured_prompt_order = {
        prompt.lower(): index
        for index, prompt in enumerate(_csv("OPENFORGE_BOWL_RACK_TARGET_PROMPTS", []))
    }
    default_prompt_order = {
        "lower rack shelf": 0,
        "bottom shelf": 1,
        "lower shelf": 2,
        "first rack level": 3,
        "wooden rack": 4,
        "rack": 5,
        "wooden shelf": 6,
        "shelf": 7,
        "box": 8,
    }
    fallback_rank = max(9, len(configured_prompt_order) + len(default_prompt_order) + 1)
    filtered: list[dict[str, Any]] = []
    for det in detections:
        xyz = _xyz(det)
        camera = str(det.get("source_camera") or det.get("camera") or "")
        score = float(det.get("score") or 0.0)
        if xyz is None or camera not in {"top", "left", "right"} or score < min_score:
            continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        filtered.append(det)
    if not filtered:
        return None
    target_y_sign = os.environ.get("OPENFORGE_BOWL_RACK_TARGET_Y_SIGN", "any").strip().lower()

    def _key(det: dict[str, Any]) -> tuple[int, int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "").lower()
        xyz = _xyz(det) or [0.0, 0.0, 0.0]
        prompt_rank = configured_prompt_order.get(prompt)
        if prompt_rank is None:
            prompt_rank = len(configured_prompt_order) + default_prompt_order.get(prompt, fallback_rank)
        y_rank = 0
        if target_y_sign in {"positive", "pos", "left", "left_side"}:
            y_rank = 0 if xyz[1] > 0.0 else 1
        elif target_y_sign in {"negative", "neg", "right", "right_side"}:
            y_rank = 0 if xyz[1] < 0.0 else 1
        return (
            0 if camera == "top" else 1,
            prompt_rank,
            1 if det.get("touches_image_edge") and camera != "top" else 0,
            y_rank,
            -float(det.get("score") or 0.0),
        )

    selected = sorted(filtered, key=_key)[0]
    selected["detection_source"] = "live_top_camera" if selected.get("source_camera") == "top" else "live_calibrated_camera"
    return selected


def _observe_rack(run_dir: Path) -> dict[str, Any]:
    prompts = _csv(
        "OPENFORGE_BOWL_RACK_TARGET_PROMPTS",
        ["lower rack shelf", "bottom shelf", "lower shelf", "first rack level", "wooden rack", "rack", "wooden shelf", "shelf"],
    )
    packet = capture_scene(
        prompts=prompts,
        cameras=_debug_observation_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="observe_rack_target",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=1,
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
        image_only_cameras=_image_only_debug_cameras(),
    )
    selected = _select_rack_detection(packet)
    packet["selected_rack_detection"] = selected
    packet["path"] = write_json(run_dir / "observations" / "observe_rack_target_selected.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## rack detection",
            f"- selected_rack_detection: {selected}",
            f"- packet: {packet.get('packet_path')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _pickup_bowl_radius_m(selected_pickup: dict[str, Any]) -> float:
    raw = selected_pickup.get("estimated_plate_radius_m")
    if raw is None:
        raw = (selected_pickup.get("radius_estimate") or {}).get("radius_m")
    try:
        radius = float(raw)
    except Exception:
        radius = _env_float("OPENFORGE_BOWL_TOP_DOWN_RADIUS_M", 0.076)
    return max(0.010, radius)


def _optional_pair_env(name: str) -> list[float] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = _csv_float(name, [])
    if len(values) != 2:
        raise RuntimeError(f"{name} must contain exactly two comma-separated values")
    return [float(values[0]), float(values[1])]


def _shelf_center_target(
    rack_xyz: list[float],
    *,
    selected_pickup: dict[str, Any],
    target_x_bias: float,
    target_y_bias: float,
) -> dict[str, Any]:
    mode = os.environ.get("OPENFORGE_BOWL_LOWER_RACK_TARGET_MODE", "shelf_center").strip().lower()
    if mode not in {"shelf_center", "bottom_shelf_center", "rack_detection_center"}:
        raise RuntimeError(
            "OPENFORGE_BOWL_LOWER_RACK_TARGET_MODE must be one of "
            "shelf_center,bottom_shelf_center,rack_detection_center"
        )
    center_override = _optional_pair_env("OPENFORGE_BOWL_LOWER_RACK_SHELF_CENTER_XY_M")
    target_xy = (
        [float(center_override[0]), float(center_override[1])]
        if center_override is not None
        else [float(rack_xyz[0]), float(rack_xyz[1])]
    )
    target_xy[0] += float(target_x_bias)
    target_xy[1] += float(target_y_bias)
    unclamped_xy = list(target_xy)
    bowl_radius = _pickup_bowl_radius_m(selected_pickup)
    margin = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_SHELF_EDGE_MARGIN_M", 0.010))
    x_bounds = _optional_pair_env("OPENFORGE_BOWL_LOWER_RACK_SHELF_X_BOUNDS_M")
    y_bounds = _optional_pair_env("OPENFORGE_BOWL_LOWER_RACK_SHELF_Y_BOUNDS_M")
    clamps: dict[str, Any] = {}
    for axis_index, axis_name, bounds in ((0, "x", x_bounds), (1, "y", y_bounds)):
        if bounds is None:
            continue
        lo, hi = sorted([float(bounds[0]), float(bounds[1])])
        inward_lo = lo + bowl_radius + margin
        inward_hi = hi - bowl_radius - margin
        if inward_lo > inward_hi:
            raise RuntimeError(
                f"configured shelf {axis_name}-bounds are too narrow for bowl radius "
                f"{bowl_radius:.3f} m plus margin {margin:.3f} m"
            )
        before = target_xy[axis_index]
        target_xy[axis_index] = min(inward_hi, max(inward_lo, before))
        clamps[axis_name] = {
            "bounds_m": [round(lo, 5), round(hi, 5)],
            "interior_bounds_m": [round(inward_lo, 5), round(inward_hi, 5)],
            "before_m": round(before, 5),
            "after_m": round(target_xy[axis_index], 5),
        }
    return {
        "mode": mode,
        "source": (
            "env:OPENFORGE_BOWL_LOWER_RACK_SHELF_CENTER_XY_M"
            if center_override is not None
            else "selected_rack_detection.position_3d"
        ),
        "rack_detection_center_xy_m": _round_list(rack_xyz[:2], limit=2),
        "target_bias_xy_m": [round(float(target_x_bias), 5), round(float(target_y_bias), 5)],
        "unclamped_target_xy_m": _round_list(unclamped_xy, limit=2),
        "target_xy_m": _round_list(target_xy, limit=2),
        "bowl_radius_m": round(bowl_radius, 5),
        "edge_margin_m": round(margin, 5),
        "clamps": clamps,
        "policy": (
            "The place target is the intended bowl-center XY on the lower shelf. "
            "Release TCP is solved separately from the measured TCP-to-bowl offset."
        ),
    }


def _make_place_candidates(selected_pickup: dict[str, Any], rack_detection: dict[str, Any]) -> list[dict[str, Any]]:
    rack_xyz = _xyz(rack_detection)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    grasp_xyz = grasp_pose.get("position") or []
    if rack_xyz is None or source_xyz is None or len(grasp_xyz) < 3:
        raise RuntimeError("cannot build place candidates without rack, source, and grasp poses")
    initial_object_ref_from_tcp = [
        float(source_xyz[0]) - float(grasp_xyz[0]),
        float(source_xyz[1]) - float(grasp_xyz[1]),
        float(source_xyz[2]) - float(grasp_xyz[2]),
    ]
    initial_object_ref_offset_norm = float(np.linalg.norm(np.asarray(initial_object_ref_from_tcp, dtype=np.float64)))
    max_release_object_offset = _env_float("OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_MAX_M", 0.200)
    if initial_object_ref_offset_norm > max_release_object_offset:
        raise RuntimeError(
            f"release object-reference offset {initial_object_ref_offset_norm:.3f} m exceeds "
            f"OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_MAX_M={max_release_object_offset:.3f}"
        )
    release_object_offset_axes = _parse_xyz_axes_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_AXES", "")
    tcp_above_source = float(grasp_xyz[2]) - float(source_xyz[2])
    if tcp_above_source < 0.05 or tcp_above_source > 0.25:
        tcp_above_source = _env_float("OPENFORGE_BOWL_RACK_DEFAULT_TCP_ABOVE_OBJECT_M", 0.135)
    release_z_from_bowl_height = _truthy_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_Z_FROM_BOWL_HEIGHT", False)
    bowl_height = _env_float("OPENFORGE_BOWL_TOP_DOWN_HEIGHT_M", 0.080)
    pickup_rim_model = selected_pickup.get("rim_geometry_estimate") or {}
    pickup_tcp_above_rim = selected_pickup.get("tcp_above_rim_m")
    if pickup_tcp_above_rim is None:
        pickup_tcp_above_rim = pickup_rim_model.get("tcp_above_rim_m")
    if pickup_tcp_above_rim is None:
        pickup_tcp_above_rim = _env_float("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M", -0.040)
    tcp_above_support_from_height = bowl_height + float(pickup_tcp_above_rim)
    tcp_above_support_from_height = max(
        _env_float("OPENFORGE_BOWL_LOWER_RACK_MIN_TCP_ABOVE_SUPPORT_M", 0.020),
        tcp_above_support_from_height,
    )
    object_ref_above_support = _env_float("OPENFORGE_BOWL_LOWER_RACK_OBJECT_REF_ABOVE_SUPPORT_M", 0.035)
    release_clearance = _env_float("OPENFORGE_BOWL_LOWER_RACK_RELEASE_CLEARANCE_M", 0.010)
    support_z_offset = _env_float("OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_OFFSET_M", 0.0)
    support_z_override_raw = os.environ.get("OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_M", "").strip()
    support_z_override = float(support_z_override_raw) if support_z_override_raw else None
    if support_z_override is not None and support_z_override < 0.3:
        raise RuntimeError(
            "OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_M is an absolute world-Z support plane. "
            f"Got {support_z_override:.3f} m, which is below the YAM table/world range; "
            "use a world Z near the top-camera table estimate (~0.760 m), or remove the override."
        )
    support_z_source = "env:OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_M" if support_z_override is not None else "rack_detection.z"
    front_standoff = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_FRONT_STANDOFF_M", 0.120))
    front_x_standoff = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_FRONT_X_STANDOFF_M", 0.0))
    retreat_standoff = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_RETREAT_STANDOFF_M", front_standoff))
    front_z_lift = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_FRONT_Z_LIFT_M", 0.080))
    entry_z_lift = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_ENTRY_Z_LIFT_M", 0.035))
    open_front_above_z_lift = max(
        0.0,
        _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_ABOVE_Z_LIFT_M", min(front_z_lift, entry_z_lift)),
    )
    target_x_bias = _env_float("OPENFORGE_BOWL_LOWER_RACK_TARGET_X_BIAS_M", 0.0)
    target_y_bias = _env_float("OPENFORGE_BOWL_LOWER_RACK_TARGET_Y_BIAS_M", 0.0)
    shelf_center_target = _shelf_center_target(
        rack_xyz,
        selected_pickup=selected_pickup,
        target_x_bias=target_x_bias,
        target_y_bias=target_y_bias,
    )
    shelf_target_xy = list(shelf_center_target["target_xy_m"])
    rpy = [float(v) for v in list(grasp_pose.get("rpy") or [])[:3]]
    if len(rpy) < 3:
        raise RuntimeError("selected pickup grasp pose has no rpy")
    pickup_arm = str(selected_pickup.get("arm") or "").strip().lower()
    if pickup_arm == "right":
        rack_side_sign = 1.0
        front_side_policy = "auto_from_pickup_arm:right"
    elif pickup_arm == "left":
        rack_side_sign = -1.0
        front_side_policy = "auto_from_pickup_arm:left"
    else:
        rack_side_sign = 1.0 if float(rack_xyz[1]) >= 0.0 else -1.0
        front_side_policy = "auto_from_rack_y"
    front_sign_raw = os.environ.get("OPENFORGE_BOWL_LOWER_RACK_FRONT_Y_SIGN", "auto").strip().lower()
    if front_sign_raw in {"positive", "pos", "+", "+1", "1"}:
        rack_side_sign = 1.0
        front_side_policy = "env_positive:front_y=place_y-standoff"
    elif front_sign_raw in {"negative", "neg", "-", "-1"}:
        rack_side_sign = -1.0
        front_side_policy = "env_negative:front_y=place_y+standoff"
    offsets = _offset_triples(
        "OPENFORGE_BOWL_RACK_PLACE_OFFSETS_M",
        [[0.0, 0.0, 0.0], [0.0, -0.025, 0.0], [0.0, 0.025, 0.0], [-0.025, 0.0, 0.0], [0.025, 0.0, 0.0]],
    )
    max_candidates = max(1, _env_int("OPENFORGE_BOWL_RACK_MAX_PLACE_CANDIDATES", 8))
    candidates: list[dict[str, Any]] = []
    for index, offset in enumerate(offsets[:max_candidates]):
        place_xy = [shelf_target_xy[0] + float(offset[0]), shelf_target_xy[1] + float(offset[1])]
        support_z_base = support_z_override if support_z_override is not None else float(rack_xyz[2])
        support_z = support_z_base + support_z_offset
        if release_z_from_bowl_height:
            release_z = support_z + tcp_above_support_from_height + release_clearance + float(offset[2])
            release_z_model = "support_z + bowl_height + tcp_above_rim + release_clearance"
        else:
            release_z = (
                support_z
                + tcp_above_source
                + object_ref_above_support
                + release_clearance
                + float(offset[2])
            )
            release_z_model = "support_z + tcp_above_source_detection + object_ref_above_support + release_clearance"
        object_ref_target = [
            float(place_xy[0]),
            float(place_xy[1]),
            support_z + object_ref_above_support + release_clearance + float(offset[2]),
        ]
        release_position = [float(place_xy[0]), float(place_xy[1]), float(release_z)]
        nominal_release_position = list(release_position)
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            if axis_name in release_object_offset_axes:
                release_position[axis_index] = object_ref_target[axis_index] - initial_object_ref_from_tcp[axis_index]
        open_front_slide = _open_front_slide_enabled()
        slide_release_z_lift = (
            max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_RELEASE_Z_LIFT_M", entry_z_lift))
            if open_front_slide
            else 0.0
        )
        if open_front_slide:
            release_position[2] += slide_release_z_lift
        release_xy = [release_position[0], release_position[1]]
        front_xy = [release_xy[0] - front_x_standoff, release_xy[1] - rack_side_sign * front_standoff]
        retreat_xy = [release_xy[0] - front_x_standoff, release_xy[1] - rack_side_sign * retreat_standoff]
        release_pose = {"position": release_position, "rpy": rpy}
        front_high_pose = {"position": [front_xy[0], front_xy[1], release_position[2] + front_z_lift], "rpy": rpy}
        front_entry_pose = {
            "position": [
                front_xy[0],
                front_xy[1],
                release_position[2] if open_front_slide else release_position[2] + entry_z_lift,
            ],
            "rpy": rpy,
        }
        open_front_above_pose = {
            "position": [release_xy[0], release_xy[1], release_position[2] + open_front_above_z_lift],
            "rpy": rpy,
        }
        if open_front_slide:
            retreat_pose = {"position": [retreat_xy[0], retreat_xy[1], release_position[2]], "rpy": rpy}
        elif _open_front_rack_enabled():
            retreat_pose = {
                "position": [release_xy[0], release_xy[1], release_position[2] + open_front_above_z_lift],
                "rpy": rpy,
            }
        else:
            retreat_pose = {"position": [retreat_xy[0], retreat_xy[1], release_position[2]], "rpy": rpy}
        candidates.append(
            {
                "label": f"lower_rack_place_{index}",
                "arm": selected_pickup["arm"],
                "strategy_type": f"{selected_pickup.get('strategy_type', 'bowl')}_release_on_lower_rack",
                "rack_detection": rack_detection,
                "pickup_strategy_label": selected_pickup.get("label"),
                "place_target_object_ref_m": [
                    round(object_ref_target[0], 5),
                    round(object_ref_target[1], 5),
                    round(object_ref_target[2], 5),
                ],
                "rack_surface_reference_m": _round_list(rack_xyz, limit=3),
                "lower_support_reference_m": _round_list([rack_xyz[0], rack_xyz[1], support_z], limit=3),
                "shelf_center_target": shelf_center_target,
                "relative_offset_m": _round_list(offset, limit=3),
                "front_entry_pose": front_entry_pose,
                "front_high_pose": front_high_pose,
                "open_front_above_pose": open_front_above_pose,
                "release_pose": release_pose,
                "retreat_pose": retreat_pose,
                "geometry_model": {
                    "source": "fresh pickup and rack detections",
                    "target_level": "lower rack / first shelf",
                    "tcp_above_source_detection_m": round(tcp_above_source, 5),
                    "release_z_model": release_z_model,
                    "release_z_from_bowl_height": release_z_from_bowl_height,
                    "bowl_height_m": round(bowl_height, 5),
                    "pickup_tcp_above_rim_m": round(float(pickup_tcp_above_rim), 5),
                    "tcp_above_support_from_height_m": round(tcp_above_support_from_height, 5),
                    "object_ref_above_support_m": round(object_ref_above_support, 5),
                    "release_clearance_m": round(release_clearance, 5),
                    "support_z_offset_m": round(support_z_offset, 5),
                    "support_z_source": support_z_source,
                    "support_z_override_m": round(support_z_override, 5) if support_z_override is not None else None,
                    "rack_detection_z_m": round(float(rack_xyz[2]), 5),
                    "release_object_offset_axes": sorted(release_object_offset_axes),
                    "initial_object_ref_from_tcp_m": _round_list(initial_object_ref_from_tcp, limit=3),
                    "initial_object_ref_offset_norm_m": round(initial_object_ref_offset_norm, 5),
                    "nominal_release_tcp_target_m": _round_list(nominal_release_position, limit=3),
                    "release_tcp_target_m": _round_list(release_position, limit=3),
                    "shelf_center_target": shelf_center_target,
                    "release_tcp_policy": (
                        "Treat place_target_object_ref_m as the bowl-center target; when "
                        "OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_AXES is set, solve "
                        "the release TCP from object_ref_target - measured pickup TCP-to-object offset."
                    ),
                    "front_standoff_m": round(front_standoff, 5),
                    "front_x_standoff_m": round(front_x_standoff, 5),
                    "target_x_bias_m": round(target_x_bias, 5),
                    "target_y_bias_m": round(target_y_bias, 5),
                    "front_z_lift_m": round(front_z_lift, 5),
                    "entry_z_lift_m": round(entry_z_lift, 5),
                    "open_front_above_z_lift_m": round(open_front_above_z_lift, 5),
                    "retreat_standoff_m": round(retreat_standoff, 5),
                    "open_front_rack": _open_front_rack_enabled(),
                    "open_front_slide": open_front_slide,
                    "open_front_slide_release_z_lift_m": round(slide_release_z_lift, 5),
                    "front_side_sign_y": round(rack_side_sign, 1),
                    "front_side_policy": front_side_policy,
                    "front_y_formula": "front_y = place_y - front_side_sign_y * front_standoff_m",
                    "front_x_formula": "front_x = place_x - front_x_standoff_m",
                    "policy": (
                        "planner transport to front-entry, direct IK insertion to lower-shelf "
                        "release, full open, then direct horizontal retreat out of rack"
                        if not open_front_slide
                        else "planner transport to open-side front-entry, Y-align at shelf mouth, X-slide into shelf "
                        "at lifted release height, full open, X-slide retreat, then Y-clear through the open side"
                    ),
                },
            }
        )
    return candidates


def _post_lift_j5_preview_context(
    run_dir: Path,
    selected_pickup: dict[str, Any],
    place_candidates: list[dict[str, Any]],
    *,
    delta_deg: float | None = None,
    artifact_suffix: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not _post_lift_j5_reorient_enabled():
        return place_candidates, None

    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.post_lift_j5_reorient_preview.v1",
        "success": False,
        "selected_pickup_label": selected_pickup.get("label"),
        "candidate_count": len(place_candidates),
        "why_stopped": None,
    }
    try:
        from robot.yam.kinematics import YamKinematics
        from scipy.spatial.transform import Rotation

        env = _tool_env("freespace_move") or _tool_env("get_robot_state")
        if env is None:
            raise RuntimeError("cannot access direct YAM env for post-lift j5 preview")
        side = str(selected_pickup.get("arm") or "").strip().lower()
        joint_index = _post_lift_j5_joint_index()
        delta_rad = _post_lift_j5_delta_rad(delta_deg)
        left_obs = _read_direct_arm_obs(env, "left")
        right_obs = _read_direct_arm_obs(env, "right")
        left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        kin = YamKinematics()
        left_pos, left_quat, right_pos, right_quat = kin.forward_kinematics(left_base, right_base)

        lift_pose = selected_pickup.get("lift_pose") or {}
        lift_xyz = np.asarray(list(lift_pose.get("position") or [])[:3], dtype=np.float64)
        lift_rpy = list(lift_pose.get("rpy") or [])[:3]
        if lift_xyz.shape[0] < 3 or len(lift_rpy) < 3:
            raise RuntimeError("selected pickup lift_pose is missing xyz/rpy for post-lift j5 preview")
        left_target_pos = np.asarray(left_pos, dtype=np.float64)
        right_target_pos = np.asarray(right_pos, dtype=np.float64)
        left_target_quat = np.asarray(left_quat, dtype=np.float64)
        right_target_quat = np.asarray(right_quat, dtype=np.float64)
        if side == "left":
            left_target_pos = lift_xyz
            left_target_quat = _display_rpy_to_quat_xyzw(lift_rpy)
        elif side == "right":
            right_target_pos = lift_xyz
            right_target_quat = _display_rpy_to_quat_xyzw(lift_rpy)
        else:
            raise RuntimeError(f"unsupported pickup side for post-lift j5 preview: {side!r}")

        lift_left, lift_right = kin.inverse_kinematics(
            left_target_pos,
            left_target_quat,
            right_target_pos,
            right_target_quat,
            seeded=True,
            max_iters=max(1, _env_int("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_PREVIEW_IK_MAX_ITERS", 120)),
            err_threshold=max(1e-6, _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_PREVIEW_IK_ERR_THRESHOLD", 1e-4)),
        )
        rotated_left = np.asarray(lift_left, dtype=np.float64).copy()
        rotated_right = np.asarray(lift_right, dtype=np.float64).copy()
        if side == "left":
            rotated_left[joint_index] += delta_rad
        else:
            rotated_right[joint_index] += delta_rad
        end_left_pos, end_left_quat, end_right_pos, end_right_quat = kin.forward_kinematics(rotated_left, rotated_right)
        active_start = np.asarray(left_target_pos if side == "left" else right_target_pos, dtype=np.float64)
        active_end = np.asarray(end_left_pos if side == "left" else end_right_pos, dtype=np.float64)
        active_quat = np.asarray(end_left_quat if side == "left" else end_right_quat, dtype=np.float64)
        rotated_rpy = _quat_xyzw_to_display_rpy(active_quat)
        tcp_translation = float(np.linalg.norm(active_end - active_start))
        max_tcp_translation = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_MAX_TCP_TRANSLATION_M", 0.120)
        tangent_gate = _post_lift_tangent_alignment_gate(selected_pickup, rotated_rpy)
        if tcp_translation > max_tcp_translation:
            raise RuntimeError(
                f"post-lift j5 preview TCP translation {tcp_translation:.3f} m exceeds "
                f"{max_tcp_translation:.3f} m"
            )
        packet.update(
            {
                "success": True,
                "side": side,
                "joint_index": joint_index,
                "delta_rad": round(delta_rad, 5),
                "delta_deg": round(math.degrees(delta_rad), 3),
                "rotated_rpy": _round_list(rotated_rpy, limit=3),
                "predicted_tcp_translation_m": round(tcp_translation, 5),
                "max_tcp_translation_m": round(max_tcp_translation, 5),
                "current_left_joint_pos": _round_list(left_base, limit=6),
                "current_right_joint_pos": _round_list(right_base, limit=6),
                "predicted_lift_left_joint_pos": _round_list(lift_left, limit=6),
                "predicted_lift_right_joint_pos": _round_list(lift_right, limit=6),
                "predicted_rotated_left_joint_pos": _round_list(rotated_left, limit=6),
                "predicted_rotated_right_joint_pos": _round_list(rotated_right, limit=6),
                "predicted_rotated_tcp_m": _round_list(active_end, limit=3),
                "tangent_alignment_gate": tangent_gate,
                "policy": (
                    "Preview simulates lift-pose IK, adds a bounded post-lift j5 delta, "
                    "then propagates the resulting FK orientation to rack transport poses."
                ),
            }
        )
        if tangent_gate.get("enabled") and not tangent_gate.get("passed"):
            packet["success"] = False
            packet["why_stopped"] = tangent_gate.get("why_stopped")
            return [], packet
        patched = [
            _patch_place_candidate_after_reorientation(
                candidate,
                selected_pickup,
                rotated_rpy,
                source="preview_predicted_post_lift_j5_fk",
                reorientation=packet,
            )
            for candidate in place_candidates
        ]
        return patched, packet
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
        return [], packet
    finally:
        packet["path"] = write_json(
            run_dir
            / "plans"
            / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_preview{artifact_suffix}.json",
            json_safe(packet),
        )
        append_stage_summary(
            run_dir,
            [
                "## post-lift j5 reorientation preview",
                f"- enabled: {_post_lift_j5_reorient_enabled()}",
                f"- success: {packet.get('success')}",
                f"- rotated_rpy: {packet.get('rotated_rpy')}",
                f"- why_stopped: {packet.get('why_stopped')}",
                f"- path: {packet.get('path')}",
            ],
        )


def _move_kwargs(
    side: str,
    pose: dict[str, Any],
    *,
    gripper: float | None = None,
    preview_only: bool = False,
    stage_label: str = "",
) -> dict[str, Any]:
    front_transport = str(stage_label) in {"lower_rack_front_high", "lower_rack_front_entry", "lower_rack_open_front_above"}
    ik_rot_threshold = _env_float("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", 16.0)
    ik_rpy_weight = _env_float("OPENFORGE_BOWL_RACK_IK_RPY_WEIGHT", 0.3)
    if front_transport:
        ik_rot_threshold = _env_float("OPENFORGE_BOWL_RACK_FRONT_IK_ROT_THRESHOLD_DEG", max(ik_rot_threshold, 30.0))
        ik_rpy_weight = _env_float("OPENFORGE_BOWL_RACK_FRONT_IK_RPY_WEIGHT", min(ik_rpy_weight, 0.10))
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_BOWL_RACK_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_BOWL_RACK_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": ik_rot_threshold,
        "ik_xyz_weight": _env_float("OPENFORGE_BOWL_RACK_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": ik_rpy_weight,
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _error_within_limit(value: Any, limit: float) -> bool:
    if value is None:
        return True
    try:
        return float(value) <= float(limit)
    except Exception:
        return False


def _stage_allows_axis_gated_rotation_residual(label: str) -> bool:
    if not _truthy_env("OPENFORGE_BOWL_RACK_ALLOW_AXIS_GATED_ROTATION_RESIDUAL", True):
        return False
    if not _axis_gate_required_for_pickup():
        return False
    label_text = str(label)
    return label_text.startswith("bowl_")


def _stage_tolerance(label: str, *, execution: bool = False) -> dict[str, float]:
    prefix = "OPENFORGE_BOWL_RACK_EXEC" if execution else "OPENFORGE_BOWL_RACK_PREVIEW"
    default_pos = _env_float(f"{prefix}_MAX_POS_ERROR_M", 0.018)
    default_rot = _env_float(f"{prefix}_MAX_ROT_ERROR_DEG", 10.0)
    loose_pos = max(default_pos, 0.025)
    loose_rot = max(default_rot, 16.0)
    label_text = str(label)
    if label_text in {"lower_rack_front_high", "lower_rack_front_entry", "lower_rack_open_front_above"}:
        return {
            "max_pos_error_m": loose_pos,
            "max_rot_error_deg": _env_float(f"{prefix}_FRONT_MAX_ROT_ERROR_DEG", max(loose_rot, 30.0)),
        }
    if "high" in label_text or "transport" in label_text or "retreat" in label_text:
        return {"max_pos_error_m": loose_pos, "max_rot_error_deg": loose_rot}
    return {"max_pos_error_m": default_pos, "max_rot_error_deg": default_rot}


def _preview_summary(result: Any, *, stage_label: str = "") -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", False))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    tolerance = _stage_tolerance(stage_label, execution=False)
    status_ok = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    axis_gated_rotation = _stage_allows_axis_gated_rotation_residual(stage_label)
    pos_ok = _error_within_limit(pos_error, tolerance["max_pos_error_m"])
    rot_ok = axis_gated_rotation or _error_within_limit(rot_error, tolerance["max_rot_error_deg"])
    tolerance_ok = pos_ok and rot_ok
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "trajectory_cache_key": _field(result, "trajectory_cache_key", None),
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": tolerance["max_pos_error_m"],
        "max_rot_error_deg": tolerance["max_rot_error_deg"],
        "tolerance_ok": tolerance_ok,
        "position_tolerance_ok": pos_ok,
        "rotation_tolerance_ok": rot_ok,
        "axis_gated_rotation_residual": axis_gated_rotation,
        "rotation_residual_policy": (
            "ignored for bowl stage because selected pickup already passed top-down axis_gate"
            if axis_gated_rotation
            else "raw motion-tool rotation residual must be within limit"
        ),
        "success": bool(status_ok and tolerance_ok),
        "raw": json_safe(result),
    }


def _execute_summary(result: Any, *, stage_label: str = "") -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", True))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    tolerance = _stage_tolerance(stage_label, execution=True)
    status_ok = bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed)
    axis_gated_rotation = _stage_allows_axis_gated_rotation_residual(stage_label)
    pos_ok = _error_within_limit(pos_error, tolerance["max_pos_error_m"])
    rot_ok = axis_gated_rotation or _error_within_limit(rot_error, tolerance["max_rot_error_deg"])
    tolerance_ok = pos_ok and rot_ok
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": tolerance["max_pos_error_m"],
        "max_rot_error_deg": tolerance["max_rot_error_deg"],
        "tolerance_ok": tolerance_ok,
        "position_tolerance_ok": pos_ok,
        "rotation_tolerance_ok": rot_ok,
        "axis_gated_rotation_residual": axis_gated_rotation,
        "rotation_residual_policy": (
            "ignored for bowl stage because selected pickup already passed top-down axis_gate"
            if axis_gated_rotation
            else "raw motion-tool rotation residual must be within limit"
        ),
        "success": bool(status_ok and tolerance_ok),
        "raw": json_safe(result),
    }


def _preview_pose(
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    *,
    label: str,
    gripper: float | None,
) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        freespace_move,
        _env_float("OPENFORGE_BOWL_RACK_PREVIEW_TIMEOUT_S", 20.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pose, gripper=gripper, preview_only=True, stage_label=label),
    )
    if not result.get("ok"):
        return {
            "stage_label": label,
            "success": False,
            "status": None,
            "reason": result.get("error"),
            "executed": False,
        }
    return _preview_summary(result.get("data"), stage_label=label)


def _preshape_gripper(selected: dict[str, Any]) -> float:
    override = os.environ.get("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS")
    if override not in {None, ""}:
        return float(override)
    preshape_plan = selected.get("preshape_plan") or {}
    return float(
        preshape_plan.get(
            "expected_gripper_open_before_close_normalized",
            _env_float("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", 0.68),
        )
    )


def _hold_gripper(selected: dict[str, Any]) -> float:
    close_plan = selected.get("staged_close_plan") or {}
    return _env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", float(close_plan.get("target_normalized_position", 0.40)))


def _use_high_approach(selected_pickup: dict[str, Any]) -> bool:
    default = str(selected_pickup.get("strategy_type") or "") != REQUIRED_PICKUP_STRATEGY_TYPE
    return _truthy_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", default)


def _pickup_retreat_pose(selected_pickup: dict[str, Any]) -> dict[str, Any]:
    if _use_high_approach(selected_pickup) and _pose_available(selected_pickup.get("approach_pose")):
        return selected_pickup["approach_pose"]
    if _pose_available(selected_pickup.get("low_pregrasp_pose")):
        return selected_pickup["low_pregrasp_pose"]
    return selected_pickup["pregrasp_pose"]


def _pickup_lift_specs(selected_pickup: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    pregrasp_open = _preshape_gripper(selected_pickup)
    hold = _hold_gripper(selected_pickup)
    specs: list[tuple[str, dict[str, Any], float]] = []
    if _use_high_approach(selected_pickup) and _pose_available(selected_pickup.get("approach_pose")):
        specs.append(("bowl_high_approach", selected_pickup["approach_pose"], pregrasp_open))
    if _pose_available(selected_pickup.get("low_pregrasp_pose")):
        specs.append(("bowl_low_pregrasp", selected_pickup["low_pregrasp_pose"], pregrasp_open))
    else:
        specs.append(("bowl_pregrasp", selected_pickup["pregrasp_pose"], pregrasp_open))
    specs.append(("bowl_grasp_pose", selected_pickup["grasp_pose"], pregrasp_open))
    specs.extend(_pickup_lift_step_specs(selected_pickup, hold))
    return specs


def _is_pickup_lift_label(label: str) -> bool:
    label_text = str(label)
    return label_text == "bowl_small_lift" or label_text.startswith("bowl_lift_step_")


def _pickup_lift_step_specs(selected_pickup: dict[str, Any], hold: float) -> list[tuple[str, dict[str, Any], float]]:
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    lift_pose = selected_pickup.get("lift_pose") or {}
    grasp_xyz = _xyz(grasp_pose)
    lift_xyz = _xyz(lift_pose)
    if grasp_xyz is None or lift_xyz is None:
        return [("bowl_small_lift", lift_pose, hold)]
    total_lift = max(0.0, float(lift_xyz[2]) - float(grasp_xyz[2]))
    step_m = max(0.001, _env_float("OPENFORGE_BOWL_TOP_DOWN_LIFT_STEP_M", total_lift or 0.001))
    max_steps = max(1, _env_int("OPENFORGE_BOWL_TOP_DOWN_LIFT_MAX_STEPS", 4))
    step_count = 1 if total_lift <= step_m else min(max_steps, int(math.ceil(total_lift / step_m)))
    if step_count <= 1:
        return [("bowl_small_lift", lift_pose, hold)]
    specs: list[tuple[str, dict[str, Any], float]] = []
    for index in range(step_count):
        fraction = float(index + 1) / float(step_count)
        pose = copy.deepcopy(lift_pose)
        pose["position"] = [
            float(grasp_xyz[0]) + (float(lift_xyz[0]) - float(grasp_xyz[0])) * fraction,
            float(grasp_xyz[1]) + (float(lift_xyz[1]) - float(grasp_xyz[1])) * fraction,
            float(grasp_xyz[2]) + (float(lift_xyz[2]) - float(grasp_xyz[2])) * fraction,
        ]
        label = "bowl_small_lift" if index == 0 else f"bowl_lift_step_{index + 1}"
        specs.append((label, pose, hold))
    return specs


def _sequence_specs(selected_pickup: dict[str, Any], place_candidate: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    hold = _hold_gripper(selected_pickup)
    specs = _pickup_lift_specs(selected_pickup)
    include_release_preview = _truthy_env("OPENFORGE_BOWL_LOWER_RACK_PREVIEW_RELEASE_WITH_PLANNER", False)
    if _split_direct_insert_enabled():
        return specs
    if _open_front_slide_enabled():
        specs.extend(
            [
                ("lower_rack_front_high", place_candidate["front_high_pose"], hold),
                ("lower_rack_front_entry", place_candidate["front_entry_pose"], hold),
            ]
        )
        if include_release_preview:
            specs.extend(
                [
                    ("lower_rack_release_pose_preview_only", place_candidate["release_pose"], hold),
                    ("lower_rack_retreat_preview_only", place_candidate["retreat_pose"], 1.0),
                ]
            )
        return specs
    if _open_front_rack_enabled():
        specs.append(("lower_rack_open_front_above", place_candidate["open_front_above_pose"], hold))
        if include_release_preview:
            specs.extend(
                [
                    ("lower_rack_release_pose_preview_only", place_candidate["release_pose"], hold),
                    ("lower_rack_retreat_preview_only", place_candidate["retreat_pose"], 1.0),
                ]
            )
        return specs
    specs.extend(
        [
            ("lower_rack_front_high", place_candidate["front_high_pose"], hold),
            ("lower_rack_front_entry", place_candidate["front_entry_pose"], hold),
        ]
    )
    if include_release_preview:
        specs.extend(
            [
                ("lower_rack_release_pose_preview_only", place_candidate["release_pose"], hold),
                ("lower_rack_retreat_preview_only", place_candidate["retreat_pose"], 1.0),
            ]
        )
    return specs


def _direct_delta_gate_step(
    *,
    stage: str,
    from_pose: dict[str, Any],
    to_pose: dict[str, Any],
    max_xy_delta_m: float,
    max_z_delta_m: float,
) -> dict[str, Any]:
    from_xyz = [float(v) for v in list(from_pose.get("position") or [])[:3]]
    to_xyz = [float(v) for v in list(to_pose.get("position") or [])[:3]]
    if len(from_xyz) < 3 or len(to_xyz) < 3:
        return {
            "stage": stage,
            "ok": False,
            "preview": {"success": False, "reason": "direct gate pose is missing xyz"},
            "from_pose": from_pose,
            "to_pose": to_pose,
        }
    delta = [to_xyz[index] - from_xyz[index] for index in range(3)]
    xy_delta = math.hypot(delta[0], delta[1])
    z_delta = abs(delta[2])
    ok = xy_delta <= max_xy_delta_m and z_delta <= max_z_delta_m
    reason = None
    if xy_delta > max_xy_delta_m:
        reason = f"direct XY delta {xy_delta:.3f} m exceeds {max_xy_delta_m:.3f} m"
    elif z_delta > max_z_delta_m:
        reason = f"direct Z delta {z_delta:.3f} m exceeds {max_z_delta_m:.3f} m"
    return {
        "stage": stage,
        "ok": ok,
        "from_pose": from_pose,
        "to_pose": to_pose,
        "preview": {
            "schema": "openforge.bowl_lower_rack.direct_delta_gate.v1",
            "success": ok,
            "reason": reason,
            "delta_m": _round_list(delta),
            "xy_delta_m": round(xy_delta, 5),
            "z_delta_m": round(z_delta, 5),
            "max_xy_delta_m": round(max_xy_delta_m, 5),
            "max_z_delta_m": round(max_z_delta_m, 5),
            "policy": "Physical release path uses bounded direct IK from front_entry to release, then direct retreat.",
        },
    }


def _open_front_slide_y_align_pose(place_candidate: dict[str, Any]) -> dict[str, Any]:
    front_entry = place_candidate["front_entry_pose"]
    release = place_candidate["release_pose"]
    front_xyz = _xyz(front_entry)
    release_xyz = _xyz(release)
    if front_xyz is None or release_xyz is None:
        raise RuntimeError("open-front slide Y-align requires front-entry and release xyz poses")
    pose = copy.deepcopy(release)
    pose["position"] = [front_xyz[0], release_xyz[1], front_xyz[2]]
    return pose


def _open_front_slide_x_retreat_pose(place_candidate: dict[str, Any]) -> dict[str, Any]:
    release = place_candidate["release_pose"]
    retreat = place_candidate["retreat_pose"]
    release_xyz = _xyz(release)
    retreat_xyz = _xyz(retreat)
    if release_xyz is None or retreat_xyz is None:
        raise RuntimeError("open-front slide X-retreat requires release and retreat xyz poses")
    pose = copy.deepcopy(retreat)
    pose["position"] = [retreat_xyz[0], release_xyz[1], release_xyz[2]]
    return pose


def _split_direct_place_down_pose(place_candidate: dict[str, Any]) -> dict[str, Any]:
    release = copy.deepcopy(place_candidate["release_pose"])
    release_xyz = _xyz(release)
    if release_xyz is None:
        raise RuntimeError("split direct place-down requires a release xyz pose")
    geometry = place_candidate.get("geometry_model") or {}
    slide_lift = 0.0
    if bool(geometry.get("open_front_slide")):
        slide_lift = max(0.0, float(geometry.get("open_front_slide_release_z_lift_m", 0.0) or 0.0))
    release["position"] = [release_xyz[0], release_xyz[1], release_xyz[2] - slide_lift]
    return release


def _split_direct_post_release_clear_pose(place_candidate: dict[str, Any]) -> dict[str, Any]:
    place_down = _split_direct_place_down_pose(place_candidate)
    place_xyz = _xyz(place_down)
    if place_xyz is None:
        raise RuntimeError("split direct post-release clear requires a place-down xyz pose")
    clear_lift = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_Z_LIFT_M", 0.070))
    pose = copy.deepcopy(place_down)
    pose["position"] = [place_xyz[0], place_xyz[1], place_xyz[2] + clear_lift]
    return pose


def _split_direct_post_release_x_retreat_pose(place_candidate: dict[str, Any]) -> dict[str, Any]:
    clear_pose = _split_direct_post_release_clear_pose(place_candidate)
    retreat = place_candidate["retreat_pose"]
    clear_xyz = _xyz(clear_pose)
    retreat_xyz = _xyz(retreat)
    if clear_xyz is None or retreat_xyz is None:
        raise RuntimeError("split direct X-retreat requires clear and retreat xyz poses")
    pose = copy.deepcopy(clear_pose)
    pose["position"] = [retreat_xyz[0], clear_xyz[1], clear_xyz[2]]
    return pose


def _split_direct_post_release_diagonal_retreat_pose(place_candidate: dict[str, Any]) -> dict[str, Any]:
    place_down = _split_direct_place_down_pose(place_candidate)
    retreat = place_candidate["retreat_pose"]
    place_xyz = _xyz(place_down)
    retreat_xyz = _xyz(retreat)
    if place_xyz is None or retreat_xyz is None:
        raise RuntimeError("split direct diagonal retreat requires place-down and retreat xyz poses")
    z_lift = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_Z_LIFT_M", 0.070))
    z_to_x_ratio = max(1e-6, _env_float("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_Z_TO_X_RATIO", 2.0))
    default_x_delta = z_lift / z_to_x_ratio
    x_delta = max(
        0.0,
        _env_float("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_X_DELTA_M", default_x_delta),
    )
    available_x_delta = abs(retreat_xyz[0] - place_xyz[0])
    x_delta = min(x_delta, available_x_delta)
    x_sign = 1.0 if retreat_xyz[0] >= place_xyz[0] else -1.0
    pose = copy.deepcopy(place_down)
    pose["position"] = [place_xyz[0] + x_sign * x_delta, place_xyz[1], place_xyz[2] + z_lift]
    pose["diagonal_retreat_model"] = {
        "z_lift_m": z_lift,
        "requested_z_to_x_ratio": z_to_x_ratio,
        "x_delta_m": x_delta,
        "x_sign": x_sign,
        "available_x_delta_m": available_x_delta,
    }
    return pose


def _pose_with_axis_delta_limit(
    from_pose: dict[str, Any],
    to_pose: dict[str, Any],
    *,
    axis: str,
    max_delta_m: float,
) -> dict[str, Any]:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    start = [float(v) for v in list(from_pose.get("position") or [])[:3]]
    target = [float(v) for v in list(to_pose.get("position") or [])[:3]]
    delta = target[axis_index] - start[axis_index]
    limited = max(-abs(max_delta_m), min(abs(max_delta_m), delta))
    pose = copy.deepcopy(to_pose)
    pose["position"] = target
    pose["position"][axis_index] = start[axis_index] + limited
    return pose


def _direct_preview_gate_steps(selected_pickup: dict[str, Any], place_candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if _split_direct_insert_enabled():
        start_pose = copy.deepcopy(selected_pickup.get("lift_pose") or place_candidate["front_entry_pose"])
        release = place_candidate["release_pose"]
        start_xyz = [float(v) for v in list(start_pose.get("position") or [])[:3]]
        release_xyz = [float(v) for v in list(release.get("position") or [])[:3]]
        y_align_max_m = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_Y_ALIGN_MAX_XY_DELTA_M", 0.150)
        y_align_pose = copy.deepcopy(release)
        y_align_pose["position"] = [start_xyz[0], release_xyz[1], start_xyz[2]]
        y_align_pose = _pose_with_axis_delta_limit(start_pose, y_align_pose, axis="y", max_delta_m=y_align_max_m)
        y_align_xyz = [float(v) for v in list(y_align_pose.get("position") or [])[:3]]
        x_insert_pose = copy.deepcopy(release)
        x_insert_pose["position"] = [release_xyz[0], y_align_xyz[1], start_xyz[2]]
        gate_steps = [
            _direct_delta_gate_step(
                stage="lower_rack_direct_y_align_gate",
                from_pose=start_pose,
                to_pose=y_align_pose,
                max_xy_delta_m=y_align_max_m,
                max_z_delta_m=0.010,
            ),
            _direct_delta_gate_step(
                stage="lower_rack_direct_x_insert_gate",
                from_pose=y_align_pose,
                to_pose=x_insert_pose,
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_X_MAX_XY_DELTA_M", 0.120),
                max_z_delta_m=0.010,
            ),
        ]
        retreat_from_pose = x_insert_pose
        if _direct_place_down_enabled():
            z_place_pose = _split_direct_place_down_pose(place_candidate)
            z_place_pose["position"] = [
                x_insert_pose["position"][0],
                x_insert_pose["position"][1],
                min(start_xyz[2], z_place_pose["position"][2]),
            ]
            if z_place_pose["position"][2] < x_insert_pose["position"][2]:
                gate_steps.append(
                    _direct_delta_gate_step(
                        stage="lower_rack_direct_z_place_down_gate",
                        from_pose=x_insert_pose,
                        to_pose=z_place_pose,
                        max_xy_delta_m=0.010,
                        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_MAX_Z_DELTA_M", 0.050),
                    )
                )
            retreat_from_pose = z_place_pose
        z_clear_pose = _split_direct_post_release_clear_pose(place_candidate)
        z_clear_pose["position"] = [
            retreat_from_pose["position"][0],
            retreat_from_pose["position"][1],
            z_clear_pose["position"][2],
        ]
        if _diagonal_retreat_after_release_enabled():
            diagonal_retreat_pose = _split_direct_post_release_diagonal_retreat_pose(place_candidate)
            diagonal_retreat_pose["position"] = [
                diagonal_retreat_pose["position"][0],
                retreat_from_pose["position"][1],
                diagonal_retreat_pose["position"][2],
            ]
            gate_steps.append(
                _direct_delta_gate_step(
                    stage="lower_rack_direct_diagonal_xz_retreat_after_release_gate",
                    from_pose=retreat_from_pose,
                    to_pose=diagonal_retreat_pose,
                    max_xy_delta_m=_env_float(
                        "OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_XY_DELTA_M",
                        _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", 0.160),
                    ),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_Z_DELTA_M", 0.100),
                )
            )
        else:
            gate_steps.append(
                _direct_delta_gate_step(
                    stage="lower_rack_direct_z_clear_after_release_gate",
                    from_pose=retreat_from_pose,
                    to_pose=z_clear_pose,
                    max_xy_delta_m=0.010,
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_Z_DELTA_M", 0.100),
                )
            )
            retreat_pose = _split_direct_post_release_x_retreat_pose(place_candidate)
            retreat_pose["position"] = [
                retreat_pose["position"][0],
                z_clear_pose["position"][1],
                z_clear_pose["position"][2],
            ]
            gate_steps.append(
                _direct_delta_gate_step(
                    stage="lower_rack_direct_x_retreat_after_clear_gate",
                    from_pose=z_clear_pose,
                    to_pose=retreat_pose,
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", 0.160),
                    max_z_delta_m=0.010,
                )
            )
        return gate_steps
    if _open_front_slide_enabled():
        y_align_pose = _open_front_slide_y_align_pose(place_candidate)
        x_retreat_pose = _open_front_slide_x_retreat_pose(place_candidate)
        release_pose = place_candidate["release_pose"]
        place_down_pose = _split_direct_place_down_pose(place_candidate) if _direct_place_down_enabled() else release_pose
        gate_steps = [
            _direct_delta_gate_step(
                stage="lower_rack_open_front_slide_y_align_gate",
                from_pose=place_candidate["front_entry_pose"],
                to_pose=y_align_pose,
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_ALIGN_MAX_XY_DELTA_M", 0.220),
                max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
            ),
            _direct_delta_gate_step(
                stage="lower_rack_open_front_slide_x_insert_gate",
                from_pose=y_align_pose,
                to_pose=release_pose,
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_INSERT_MAX_XY_DELTA_M", 0.220),
                max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
            ),
        ]
        if _direct_place_down_enabled():
            gate_steps.append(
                _direct_delta_gate_step(
                    stage="lower_rack_open_front_slide_z_place_down_gate",
                    from_pose=release_pose,
                    to_pose=place_down_pose,
                    max_xy_delta_m=0.010,
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_MAX_Z_DELTA_M", 0.050),
                )
            )
        if _truthy_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_VERTICAL_CLEAR_AFTER_RELEASE", False):
            z_clear_pose = copy.deepcopy(place_down_pose)
            place_down_xyz = _xyz(place_down_pose)
            if place_down_xyz is None:
                raise RuntimeError("open-front slide Z-clear gate requires a place-down xyz pose")
            z_clear_pose["position"] = [
                place_down_xyz[0],
                place_down_xyz[1],
                place_down_xyz[2] + max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_Z_LIFT_M", 0.070)),
            ]
            gate_steps.append(
                _direct_delta_gate_step(
                    stage="lower_rack_open_front_slide_z_clear_after_release_gate",
                    from_pose=place_down_pose,
                    to_pose=z_clear_pose,
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_XY_DRIFT_M", 0.035),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_Z_DELTA_M", 0.100),
                )
            )
            if not _truthy_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_SKIP_XY_RETREAT_AFTER_VERTICAL_CLEAR", False):
                x_clear_retreat_pose = copy.deepcopy(x_retreat_pose)
                x_clear_retreat_pose["position"] = [
                    x_retreat_pose["position"][0],
                    z_clear_pose["position"][1],
                    z_clear_pose["position"][2],
                ]
                gate_steps.append(
                    _direct_delta_gate_step(
                        stage="lower_rack_open_front_slide_x_retreat_after_z_clear_gate",
                        from_pose=z_clear_pose,
                        to_pose=x_clear_retreat_pose,
                        max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_MAX_XY_DELTA_M", 0.220),
                        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                    )
                )
                if _truthy_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_AFTER_VERTICAL_CLEAR", False):
                    y_clear_retreat_pose = copy.deepcopy(place_candidate["retreat_pose"])
                    y_clear_retreat_pose["position"] = [
                        x_clear_retreat_pose["position"][0],
                        y_clear_retreat_pose["position"][1],
                        x_clear_retreat_pose["position"][2],
                    ]
                    gate_steps.append(
                        _direct_delta_gate_step(
                            stage="lower_rack_open_front_slide_y_retreat_after_z_clear_gate",
                            from_pose=x_clear_retreat_pose,
                            to_pose=y_clear_retreat_pose,
                            max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_MAX_XY_DELTA_M", 0.220),
                            max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                        )
                    )
            return gate_steps
        gate_steps.extend(
            [
            _direct_delta_gate_step(
                stage="lower_rack_open_front_slide_x_retreat_gate",
                from_pose=place_down_pose,
                to_pose=x_retreat_pose,
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_MAX_XY_DELTA_M", 0.220),
                max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
            ),
            _direct_delta_gate_step(
                stage="lower_rack_open_front_slide_y_retreat_gate",
                from_pose=x_retreat_pose,
                to_pose=place_candidate["retreat_pose"],
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_MAX_XY_DELTA_M", 0.220),
                max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
            ),
            ]
        )
        return gate_steps
    if _open_front_rack_enabled():
        return [
            _direct_delta_gate_step(
                stage="lower_rack_open_front_place_down_gate",
                from_pose=place_candidate["open_front_above_pose"],
                to_pose=place_candidate["release_pose"],
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_DIRECT_MAX_XY_DELTA_M", 0.030),
                max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_MAX_Z_DELTA_M", 0.120),
            ),
            _direct_delta_gate_step(
                stage="lower_rack_open_front_vertical_retreat_gate",
                from_pose=place_candidate["release_pose"],
                to_pose=place_candidate["retreat_pose"],
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_XY_DELTA_M", 0.030),
                max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_Z_DELTA_M", 0.120),
            ),
        ]
    return [
        _direct_delta_gate_step(
            stage="lower_rack_direct_insert_gate",
            from_pose=place_candidate["front_entry_pose"],
            to_pose=place_candidate["release_pose"],
            max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_XY_DELTA_M", 0.150),
            max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_Z_DELTA_M", 0.050),
        ),
        _direct_delta_gate_step(
            stage="lower_rack_direct_retreat_gate",
            from_pose=place_candidate["release_pose"],
            to_pose=place_candidate["retreat_pose"],
            max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", 0.160),
            max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_Z_DELTA_M", 0.020),
        ),
    ]


def _split_physical_sequence(
    selected_pickup: dict[str, Any], place_candidate: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, Any], float]], list[tuple[str, dict[str, Any], float]]]:
    before_close: list[tuple[str, dict[str, Any], float]] = []
    after_close: list[tuple[str, dict[str, Any], float]] = []
    saw_lift = False
    for spec in _sequence_specs(selected_pickup, place_candidate):
        label = spec[0]
        if _is_pickup_lift_label(label):
            saw_lift = True
        if saw_lift:
            after_close.append(spec)
        else:
            before_close.append(spec)
    if not saw_lift:
        raise RuntimeError("physical sequence missing bowl_small_lift stage")
    return before_close, after_close


def _preview_place_sequence(
    run_dir: Path,
    selected_pickup: dict[str, Any],
    place_candidates: list[dict[str, Any]],
    *,
    artifact_suffix: str = "",
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    attempts: list[dict[str, Any]] = []
    for candidate in place_candidates:
        steps: list[dict[str, Any]] = []
        success = True
        for label, pose, gripper in _sequence_specs(selected_pickup, candidate):
            preview = _preview_pose(freespace_move, side, pose, label=label, gripper=gripper)
            step = {"stage": label, "pose": pose, "gripper": gripper, "preview": preview, "ok": bool(preview.get("success"))}
            steps.append(step)
            if not step["ok"]:
                success = False
                break
        if success and not _truthy_env("OPENFORGE_BOWL_LOWER_RACK_PREVIEW_RELEASE_WITH_PLANNER", False):
            for step in _direct_preview_gate_steps(selected_pickup, candidate):
                steps.append(step)
                if not step["ok"]:
                    success = False
                    break
        attempt = {
            "label": candidate.get("label"),
            "success": success,
            "failed_stage": next((step["stage"] for step in steps if not step.get("ok")), None),
            "place_candidate": candidate,
            "steps": steps,
        }
        attempts.append(attempt)
        if success:
            packet = {
                "schema": "openforge.bowl_rack.sequence_preview.v1",
                "success": True,
                "side": side,
                "selected_pickup": selected_pickup,
                "selected_place": candidate,
                "attempts": attempts,
            }
            packet["path"] = write_json(
                run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_to_rack_sequence_preview{artifact_suffix}.json",
                packet,
            )
            append_stage_summary(
                run_dir,
                [
                    f"## {TASK_OBJECT_NAME} to rack sequence preview",
                    "- success: True",
                    f"- selected_place: {candidate.get('label')}",
                    f"- release_pose: {candidate.get('release_pose')}",
                    f"- path: {packet['path']}",
                ],
            )
            return packet
    packet = {
        "schema": "openforge.bowl_rack.sequence_preview.v1",
        "success": False,
        "side": side,
        "selected_pickup": selected_pickup,
        "selected_place": None,
        "attempts": attempts,
    }
    packet["path"] = write_json(
        run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_to_rack_sequence_preview{artifact_suffix}.json",
        packet,
    )
    append_stage_summary(
        run_dir,
        [
            f"## {TASK_OBJECT_NAME} to rack sequence preview",
            "- success: False",
            f"- attempts: {[(item.get('label'), item.get('failed_stage')) for item in attempts]}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _preview_pickup_lift_sequence(run_dir: Path, selected_pickup: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    steps: list[dict[str, Any]] = []
    success = True
    for label, pose, gripper in _pickup_lift_specs(selected_pickup):
        preview = _preview_pose(freespace_move, side, pose, label=label, gripper=gripper)
        if not preview.get("success"):
            accepted = _accept_staged_bowl_transition_from_direct_preview(selected_pickup, label, pose, steps, preview)
            if accepted:
                preview = accepted
        step = {"stage": label, "pose": pose, "gripper": gripper, "preview": preview, "ok": bool(preview.get("success"))}
        steps.append(step)
        if not step["ok"]:
            success = False
            break
    packet = {
        "schema": "openforge.bowl_rack.pickup_lift_preview.v1",
        "success": success,
        "side": side,
        "selected_pickup": selected_pickup,
        "failed_stage": next((step["stage"] for step in steps if not step.get("ok")), None),
        "steps": steps,
    }
    packet["path"] = write_json(run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_pickup_lift_sequence_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            f"## {TASK_OBJECT_NAME} pickup/lift sequence preview",
            f"- success: {success}",
            f"- failed_stage: {packet['failed_stage']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _direct_preview_gate_check(selected_pickup: dict[str, Any], stage: str) -> dict[str, Any] | None:
    gate = selected_pickup.get("preview_residual_gate") or {}
    for check in gate.get("checks") or []:
        if check.get("stage") == stage and check.get("passed") is True:
            return check
    return None


def _pose_close(a: Any, b: Any, *, max_pos_m: float = 0.002, max_rpy_deg: float = 0.1) -> bool:
    if not (_pose_available(a) and _pose_available(b)):
        return False
    pos_error = _distance3(a, b)
    if pos_error is None or pos_error > max_pos_m:
        return False
    try:
        ar = [float(v) for v in a["rpy"][:3]]
        br = [float(v) for v in b["rpy"][:3]]
    except Exception:
        return False
    return all(abs(ar[index] - br[index]) <= max_rpy_deg for index in range(3))


def _accept_staged_bowl_transition_from_direct_preview(
    selected_pickup: dict[str, Any],
    label: str,
    pose: dict[str, Any],
    prior_steps: list[dict[str, Any]],
    failed_preview: dict[str, Any],
) -> dict[str, Any] | None:
    if not _truthy_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", True):
        return None
    stage_map = {
        "bowl_low_pregrasp": ("pregrasp_pose", {"bowl_high_approach"}, "pregrasp_pose"),
        "bowl_pregrasp": ("pregrasp_pose", {"bowl_high_approach"}, "pregrasp_pose"),
        "bowl_grasp_pose": ("grasp_pose", {"bowl_low_pregrasp", "bowl_pregrasp"}, "grasp_pose"),
        "bowl_small_lift": ("lift_pose", {"bowl_grasp_pose"}, "lift_pose"),
    }
    if label not in stage_map:
        return None
    direct_stage, required_prior_stages, pose_key = stage_map[label]
    prior_ok = any(step.get("stage") in required_prior_stages and step.get("ok") for step in prior_steps)
    if label in {"bowl_low_pregrasp", "bowl_pregrasp"} and not _use_high_approach(selected_pickup):
        return None
    if not prior_ok:
        return None
    direct_check = _direct_preview_gate_check(selected_pickup, direct_stage)
    if not direct_check:
        return None
    if not _pose_close(pose, selected_pickup.get(pose_key)):
        return None
    if label in {"bowl_low_pregrasp", "bowl_pregrasp"}:
        approach_xyz = _xyz(selected_pickup.get("approach_pose") or {})
        low_xyz = _xyz(pose)
        if not approach_xyz or not low_xyz or float(approach_xyz[2]) <= float(low_xyz[2]):
            return None
    accepted = dict(failed_preview)
    accepted.update(
        {
            "success": True,
            "status": "AcceptedDirectStagedPreview",
            "reason": (
                f"accepted {label} sequence preview from selected candidate's direct "
                f"{direct_stage} residual gate plus a reachable prior staged pose; "
                "physical execution will still re-preview from the actually reached "
                "previous pose before executing this stage"
            ),
            "accepted_direct_staged_preview": True,
            "direct_stage": direct_stage,
            "direct_gate_check": direct_check,
            "failed_sequence_preview": failed_preview,
        }
    )
    return accepted


def _ticket_status(run_dir: Path) -> dict[str, Any]:
    required_true = [
        "OPENFORGE_BOWL_RACK_PHYSICAL_TICKET",
        "OPENFORGE_BOWL_RACK_OPERATOR_PRESENT",
        "OPENFORGE_BOWL_RACK_ESTOP_REACHABLE",
        "OPENFORGE_BOWL_RACK_WORKSPACE_CLEAR",
        "OPENFORGE_BOWL_RACK_HANDS_TOOLS_CABLES_CLEAR",
        TASK_VISIBLE_ENV,
        "OPENFORGE_BOWL_RACK_RACK_CLEAR",
        "OPENFORGE_BOWL_RACK_NO_LIQUID",
        "OPENFORGE_BOWL_RACK_NO_MANUAL_XYZ",
        "OPENFORGE_BOWL_RACK_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_BOWL_RACK_FOLLOWER_SERVERS_UP",
        "OPENFORGE_BOWL_RACK_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_BOWL_RACK_TICKET_SCOPE", "").strip()
    if scope != TASK_TICKET_SCOPE:
        missing.append(f"OPENFORGE_BOWL_RACK_TICKET_SCOPE={TASK_TICKET_SCOPE}")
    budget = _env_int("OPENFORGE_BOWL_RACK_PHYSICAL_ATTEMPT_BUDGET", 0)
    max_budget = _env_int("OPENFORGE_BOWL_RACK_MAX_ALLOWED_ATTEMPT_BUDGET", 20)
    if budget < 1 or budget > max_budget:
        missing.append(f"OPENFORGE_BOWL_RACK_PHYSICAL_ATTEMPT_BUDGET=1..{max_budget}")
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


def _set_gripper_checked(
    *,
    side: str,
    target: float,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    stage: str,
    confirm_min: float | None = None,
    confirm_max: float | None = None,
    vel_limit: float | None = None,
    torque_limit: float | None = None,
) -> dict[str, Any]:
    before = read_gripper_width(get_robot_state, side)
    command_kwargs: dict[str, float] = {}
    if vel_limit is not None:
        command_kwargs["vel_limit"] = float(vel_limit)
    if torque_limit is not None:
        command_kwargs["torque_limit"] = float(torque_limit)
    result = set_gripper(side, target, **command_kwargs)
    time.sleep(_env_float("OPENFORGE_BOWL_RACK_GRIPPER_SETTLE_S", 0.25))
    after = read_gripper_width(get_robot_state, side)
    ok = True
    confirm_timeout_s = _env_float("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_TIMEOUT_S", 2.5)
    confirm_poll_s = max(0.02, _env_float("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_POLL_S", 0.1))
    if confirm_min is not None and after is not None:
        deadline = time.monotonic() + confirm_timeout_s
        while after < confirm_min and time.monotonic() < deadline:
            time.sleep(confirm_poll_s)
            after = read_gripper_width(get_robot_state, side)
        ok = after is None or after >= confirm_min
    if confirm_max is not None and after is not None:
        ok = ok and after <= confirm_max
    return {
        "stage": stage,
        "side": side,
        "target": target,
        "before": before,
        "after": after,
        "confirm_min": confirm_min,
        "confirm_max": confirm_max,
        "confirm_timeout_s": confirm_timeout_s,
        "confirm_poll_s": confirm_poll_s,
        "ok": ok,
        "result": json_safe(result),
    }


def _execute_pose(
    *,
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    label: str,
    gripper: float | None,
) -> dict[str, Any]:
    max_attempts = max(1, _env_int("OPENFORGE_BOWL_RACK_EXEC_STALE_CACHE_REPLAN_ATTEMPTS", 3))
    retry_sleep_s = max(0.0, _env_float("OPENFORGE_BOWL_RACK_EXEC_STALE_CACHE_REPLAN_SLEEP_S", 0.20))
    replan_attempts: list[dict[str, Any]] = []
    preview_packet: dict[str, Any] | None = None
    execute_packet: dict[str, Any] | None = None
    executed: Any = None
    for attempt_index in range(max_attempts):
        preview = freespace_move(**_move_kwargs(side, pose, gripper=gripper, preview_only=True, stage_label=label))
        preview_packet = _preview_summary(preview, stage_label=label)
        replan_attempts.append(
            {
                "attempt_index": attempt_index,
                "preview_success": bool(preview_packet.get("success")),
                "trajectory_cache_key": preview_packet.get("trajectory_cache_key"),
            }
        )
        if not preview_packet["success"] or not preview_packet.get("trajectory_cache_key"):
            raise RuntimeError(f"{label} preview failed before execution: {preview_packet}")
        try:
            executed = freespace_move(trajectory_cache_key=preview_packet["trajectory_cache_key"])
        except RuntimeError as exc:
            error_text = str(exc)
            replan_attempts[-1]["execute_error"] = error_text
            if "trajectory cache entry is stale" in error_text.lower() and attempt_index + 1 < max_attempts:
                if retry_sleep_s > 0.0:
                    time.sleep(retry_sleep_s)
                continue
            raise
        execute_packet = _execute_summary(executed, stage_label=label)
        replan_attempts[-1]["execute_success"] = bool(execute_packet.get("success"))
        replan_attempts[-1]["execute_status"] = execute_packet.get("status")
        stale_status = "stale" in str(execute_packet.get("status") or execute_packet.get("reason") or "").lower()
        if stale_status and attempt_index + 1 < max_attempts:
            replan_attempts[-1]["execute_error"] = "trajectory cache entry is stale"
            if retry_sleep_s > 0.0:
                time.sleep(retry_sleep_s)
            continue
        if not execute_packet["success"]:
            raise RuntimeError(f"{label} execution failed: {execute_packet}")
        break
    if preview_packet is None or execute_packet is None or executed is None:
        raise RuntimeError(f"{label} execution did not complete after {max_attempts} preview/execute attempts")
    return {
        "stage": label,
        "side": side,
        "pose": pose,
        "gripper": gripper,
        "preview": preview_packet,
        "execute": execute_packet,
        "trajectory_cache_key": preview_packet["trajectory_cache_key"],
        "replan_attempts": replan_attempts,
    }


def _capture_object_observation(run_dir: Path, stage: str) -> dict[str, Any]:
    prompts = _csv("OPENFORGE_BOWL_RACK_POST_PROMPTS", _default_post_prompts())
    packet = capture_scene(
        prompts=prompts,
        cameras=_debug_observation_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=1,
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
        image_only_cameras=_image_only_debug_cameras(),
    )
    packet["path"] = write_json(run_dir / "observations" / f"{_artifact_safe_label(stage)}.json", packet)
    return packet


def _capture_intermediate_observation(
    run_dir: Path,
    stage: str,
    *,
    env_name: str,
    force: bool = False,
) -> dict[str, Any]:
    enabled = force or (
        _truthy_env("OPENFORGE_BOWL_RACK_CAPTURE_INTERMEDIATE_OBSERVATIONS", True)
        and _truthy_env(env_name, True)
    )
    if enabled:
        return _capture_object_observation(run_dir, stage)
    packet = {
        "schema": "openforge.bowl_lower_rack.observation_skip.v1",
        "stage": stage,
        "skipped": True,
        "success": True,
        "reason": (
            "intermediate camera observation disabled; initial planning perception remains enabled"
        ),
    }
    append_stage_summary(
        run_dir,
        [
            f"## observation {stage}",
            "- skipped: True",
            f"- reason: {packet['reason']}",
        ],
    )
    return packet


def _select_object_detection(
    packet: dict[str, Any],
    preferred_prompts: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    if preferred_prompts is None:
        preferred_prompts = _object_detection_prompts()
    detections = [dict(det) for det in packet.get("all_detections") or []]
    prompt_rank = {prompt: index for index, prompt in enumerate(preferred_prompts)}
    filtered = []
    for det in detections:
        prompt = str(det.get("prompt") or "").lower()
        camera = str(det.get("source_camera") or det.get("camera") or "")
        if prompt not in prompt_rank or camera not in {"top", "left", "right"}:
            continue
        if float(det.get("score") or 0.0) < _env_float("OPENFORGE_BOWL_RACK_POST_MIN_SCORE", 0.20):
            continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        filtered.append(det)
    if not filtered:
        return None

    def _key(det: dict[str, Any]) -> tuple[int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "").lower()
        return (
            0 if camera == "top" else 1,
            1 if det.get("touches_image_edge") and camera != "top" else 0,
            prompt_rank.get(prompt, 9),
            -float(det.get("score") or 0.0),
        )

    return sorted(filtered, key=_key)[0]


def _xy_distance(a: Any, b: Any) -> float | None:
    av = _xyz(a)
    bv = _xyz(b)
    if av is None or bv is None:
        return None
    dx = av[0] - bv[0]
    dy = av[1] - bv[1]
    return math.sqrt(dx * dx + dy * dy)


def _select_gripper_detection(packet: dict[str, Any], side: str) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    side_camera = "right" if side == "right" else "left"
    prompts = {
        "blue gripper fingers": 0,
        f"{side} gripper": 1,
        "gripper fingers": 2,
        "robot gripper": 3,
    }
    filtered: list[dict[str, Any]] = []
    for det in detections:
        prompt = str(det.get("prompt") or "").lower()
        camera = str(det.get("source_camera") or det.get("camera") or "")
        if prompt not in prompts or camera not in {"top", "left", "right"}:
            continue
        if float(det.get("score") or 0.0) < _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_GRIPPER_MIN_SCORE", 0.18):
            continue
        xyz = _xyz(det)
        if xyz is not None and camera == "top":
            side_y_margin = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_GRIPPER_SIDE_Y_MARGIN_M", 0.08)
            if side == "right" and xyz[1] > side_y_margin:
                continue
            if side == "left" and xyz[1] < -side_y_margin:
                continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        filtered.append(det)
    if not filtered:
        return None

    def _key(det: dict[str, Any]) -> tuple[int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "").lower()
        return (
            0 if camera == side_camera else 1 if camera == "top" else 2,
            1 if det.get("touches_image_edge") and camera != side_camera else 0,
            prompts.get(prompt, 9),
            -float(det.get("score") or 0.0),
        )

    return sorted(filtered, key=_key)[0]


def _adjusted_contact_from_current_object(selected_pickup: dict[str, Any], current_object: dict[str, Any]) -> list[float] | None:
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    current_xyz = _xyz(current_object)
    rim = selected_pickup.get("selected_rim_edge") or {}
    planned_contact = _xyz(
        rim.get("contact_center_m")
        or rim.get("gap_center_m")
        or rim.get("edge_point_m")
        or rim.get("pinch_tcp_center_m")
        or (selected_pickup.get("grasp_pose") or {}).get("position")
        or selected_pickup.get("position")
        or {}
    )
    if source_xyz is None or current_xyz is None or planned_contact is None:
        return None
    # Keep the planned rim/contact offset but translate it to the current object
    # detection. This catches stale or occlusion-shifted object centers before close.
    return [
        planned_contact[0] + (current_xyz[0] - source_xyz[0]),
        planned_contact[1] + (current_xyz[1] - source_xyz[1]),
        planned_contact[2] + (current_xyz[2] - source_xyz[2]),
    ]


def _pre_close_visual_gate(run_dir: Path, selected_pickup: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    side = str(selected_pickup.get("arm") or "right")
    obj = _select_object_detection(observation)
    gripper = _select_gripper_detection(observation, side)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    rim = selected_pickup.get("selected_rim_edge") or {}
    planned_contact = _xyz(
        rim.get("contact_center_m")
        or rim.get("gap_center_m")
        or rim.get("edge_point_m")
        or rim.get("pinch_tcp_center_m")
        or (selected_pickup.get("grasp_pose") or {}).get("position")
        or selected_pickup.get("position")
        or {}
    )
    adjusted_contact = _adjusted_contact_from_current_object(selected_pickup, obj or {})
    object_xyz = _xyz(obj or {})
    gripper_xyz = _xyz(gripper or {})
    axis_gate = selected_pickup.get("axis_gate") or {}
    arm_state = _arm_state_from_observation(observation, side)
    ee_xyz = _xyz((arm_state or {}).get("ee_pos") or [])
    ee_quat = (arm_state or {}).get("ee_quat")
    ee_axes = _quat_xyzw_axes_world(ee_quat)

    max_object_xy_drift = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_XY_DRIFT_M", 0.055)
    max_object_z_jump = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_Z_JUMP_M", 0.060)
    max_gripper_xy_to_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_GRIPPER_XY_TO_CONTACT_M", 0.075)
    max_gripper_above_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_GRIPPER_ABOVE_CONTACT_M", 0.045)
    max_gripper_below_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_GRIPPER_BELOW_CONTACT_M", 0.035)
    max_ee_to_grasp = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_EE_TO_GRASP_M", 0.030)
    max_tcp_above_deviation = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_TCP_ABOVE_DEVIATION_M", 0.030)
    min_robot_axis_alignment = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MIN_ROBOT_AXIS_ALIGNMENT", 0.90)
    max_object_xy_to_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_XY_TO_CONTACT_M", 0.050)
    max_object_z_to_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_Z_TO_CONTACT_M", 0.040)
    max_contact_above_object_z = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_CONTACT_ABOVE_OBJECT_Z_M", 0.060)

    object_xy_drift = _xy_distance(source_xyz, object_xyz) if source_xyz and object_xyz else None
    object_z_delta = float(object_xyz[2]) - float(source_xyz[2]) if source_xyz and object_xyz else None
    object_depth_jump = object_z_delta is not None and abs(object_z_delta) > max_object_z_jump
    object_drift_ok = object_xy_drift is not None and object_xy_drift <= max_object_xy_drift
    object_xy_to_contact = _xy_distance(object_xyz, planned_contact) if object_xyz and planned_contact else None
    object_z_delta_to_contact = (
        float(object_xyz[2]) - float(planned_contact[2]) if object_xyz is not None and planned_contact is not None else None
    )
    contact_above_object_z = (
        float(planned_contact[2]) - float(object_xyz[2]) if object_xyz is not None and planned_contact is not None else None
    )
    object_contact_xy_ok = object_xy_to_contact is not None and object_xy_to_contact <= max_object_xy_to_contact
    object_contact_z_ok = object_z_delta_to_contact is not None and abs(object_z_delta_to_contact) <= max_object_z_to_contact
    contact_above_object_z_ok = contact_above_object_z is not None and contact_above_object_z <= max_contact_above_object_z
    object_contact_coherent = bool(obj and object_xyz and object_contact_xy_ok and object_contact_z_ok)
    object_center_coherent = bool(obj and object_xyz and object_drift_ok and not object_depth_jump)
    object_detection_coherent = bool(object_contact_coherent or object_center_coherent)
    object_reference_mode = "planned_contact_unverified_by_current_rim_detection"
    contact_for_gate = planned_contact
    if object_contact_coherent:
        object_reference_mode = "planned_contact_verified_by_current_rim_detection"
        contact_for_gate = planned_contact
    elif object_center_coherent:
        object_reference_mode = "planned_contact_with_current_center_drift_check"
        contact_for_gate = planned_contact
    elif not object_detection_coherent and _truthy_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", True):
        object_reference_mode = "planned_source_due_occluded_or_incoherent_detection"
        contact_for_gate = planned_contact

    gripper_xy_to_contact = _xy_distance(gripper_xyz, adjusted_contact) if gripper_xyz and adjusted_contact else None
    gripper_z_delta_to_contact = (
        float(gripper_xyz[2]) - float(adjusted_contact[2]) if gripper_xyz is not None and adjusted_contact is not None else None
    )
    grasp_pose = selected_pickup.get("grasp_pose") or selected_pickup
    grasp_xyz = _xyz(grasp_pose)
    ee_to_grasp = _distance3(ee_xyz, grasp_xyz) if ee_xyz and grasp_xyz else None
    ee_z_above_contact = (
        float(ee_xyz[2]) - float(contact_for_gate[2]) if ee_xyz is not None and contact_for_gate is not None else None
    )
    rim_model = selected_pickup.get("rim_geometry_estimate") or {}
    expected_tcp_above = rim_model.get("tcp_above_rim_m")
    if expected_tcp_above is None and grasp_xyz is not None and planned_contact is not None:
        expected_tcp_above = float(grasp_xyz[2]) - float(planned_contact[2])
    expected_opening_axis = axis_gate.get("expected_opening_axis_world") or selected_pickup.get("opening_axis_world")
    expected_approach_axis = axis_gate.get("expected_approach_axis_world") or selected_pickup.get("approach_direction_world")
    axis_required = _axis_gate_required_for_pickup()
    local_x_world = ee_axes.get("local_x_world")
    local_z_world = ee_axes.get("local_z_world")
    axis_reference_available = (
        expected_opening_axis is not None
        and expected_approach_axis is not None
        and local_x_world is not None
        and local_z_world is not None
    )
    robot_opening_alignment = (
        _dot3(local_x_world, expected_opening_axis) if axis_reference_available else None
    )
    robot_approach_alignment = (
        _dot3(local_z_world, expected_approach_axis) if axis_reference_available else None
    )
    robot_axis_ok = (
        (
            robot_opening_alignment is not None
            and robot_approach_alignment is not None
            and robot_opening_alignment >= min_robot_axis_alignment
            and robot_approach_alignment >= min_robot_axis_alignment
        )
        if axis_reference_available
        else not axis_required
    )
    robot_ee_to_grasp_ok = ee_to_grasp is not None and ee_to_grasp <= max_ee_to_grasp
    robot_ee_z_to_contact_ok = (
        ee_z_above_contact is not None
        and expected_tcp_above is not None
        and abs(ee_z_above_contact - float(expected_tcp_above)) <= max_tcp_above_deviation
    )
    robot_state_gate = {
        "available": bool(arm_state and ee_xyz and ee_quat),
        "passed": bool(robot_ee_to_grasp_ok and robot_ee_z_to_contact_ok and robot_axis_ok),
        "side": side,
        "ee_pos": _round_list(ee_xyz, limit=3) if ee_xyz else None,
        "ee_rpy": _round_list((arm_state or {}).get("ee_rpy") or [], limit=3),
        "ee_quat_xyzw": _round_list(ee_quat or [], limit=4),
        "ee_axes_world": ee_axes,
        "target_grasp_xyz": _round_list(grasp_xyz, limit=3) if grasp_xyz else None,
        "contact_center_used_m": _round_list(contact_for_gate, limit=3) if contact_for_gate else None,
        "contact_reference_mode": object_reference_mode,
        "ee_to_grasp_m": round(ee_to_grasp, 5) if ee_to_grasp is not None else None,
        "ee_z_above_contact_m": round(ee_z_above_contact, 5) if ee_z_above_contact is not None else None,
        "expected_tcp_above_rim_m": round(float(expected_tcp_above), 5) if expected_tcp_above is not None else None,
        "opening_axis_alignment": round(robot_opening_alignment, 5) if robot_opening_alignment is not None else None,
        "approach_axis_alignment": round(robot_approach_alignment, 5) if robot_approach_alignment is not None else None,
        "axis_reference_available": axis_reference_available,
        "expected_opening_axis_world": _round_list(expected_opening_axis, limit=3) if expected_opening_axis is not None else None,
        "expected_approach_axis_world": _round_list(expected_approach_axis, limit=3) if expected_approach_axis is not None else None,
        "checks": {
            "ee_to_grasp_ok": robot_ee_to_grasp_ok,
            "ee_z_to_contact_ok": robot_ee_z_to_contact_ok,
            "axis_ok": robot_axis_ok,
        },
        "limits": {
            "max_ee_to_grasp_m": max_ee_to_grasp,
            "max_tcp_above_deviation_m": max_tcp_above_deviation,
            "min_robot_axis_alignment": min_robot_axis_alignment,
        },
    }
    gripper_detection_ok = gripper_xy_to_contact is not None and gripper_xy_to_contact <= max_gripper_xy_to_contact
    gripper_z_ok = (
        gripper_z_delta_to_contact is not None
        and -max_gripper_below_contact <= gripper_z_delta_to_contact <= max_gripper_above_contact
    )
    require_sam_gripper = _truthy_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", False)
    visible_object_mismatch = bool(obj and object_xyz and not object_detection_coherent)
    allow_visible_object_mismatch = _truthy_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_VISIBLE_OBJECT_MISMATCH", False)
    object_reference_ok = bool(
        object_detection_coherent
        or (
            object_reference_mode != "current_detection"
            and (not visible_object_mismatch or allow_visible_object_mismatch)
        )
    )

    checks = {
        "object_detected": obj is not None and object_xyz is not None,
        "gripper_detected": gripper is not None and gripper_xyz is not None,
        "axis_gate_passed": (not axis_required) or (bool(axis_gate) and axis_gate.get("passed") is True),
        "object_xy_drift_ok": object_drift_ok,
        "object_depth_ok": not object_depth_jump if object_z_delta is not None else False,
        "object_xy_to_contact_ok": object_contact_xy_ok,
        "object_z_to_contact_ok": object_contact_z_ok,
        "contact_above_object_z_ok": contact_above_object_z_ok,
        "object_reference_ok": object_reference_ok,
        "robot_state_gate_ok": bool(robot_state_gate["passed"]),
        "gripper_xy_to_contact_ok": gripper_detection_ok,
        "gripper_z_to_contact_ok": gripper_z_ok,
    }
    required_checks = [
        checks["object_detected"],
        checks["object_reference_ok"],
        checks["robot_state_gate_ok"],
        checks["contact_above_object_z_ok"],
    ]
    if axis_required:
        required_checks.append(checks["axis_gate_passed"])
    if require_sam_gripper:
        required_checks.extend([checks["gripper_detected"], checks["gripper_xy_to_contact_ok"], checks["gripper_z_to_contact_ok"]])
    packet = {
        "schema": "openforge.bowl_rack.pre_close_visual_gate.v1",
        "passed": all(required_checks),
        "checks": checks,
        "required_checks": {
            "object_detected": checks["object_detected"],
            "axis_gate_passed": checks["axis_gate_passed"],
            "object_reference_ok": checks["object_reference_ok"],
            "robot_state_gate_ok": checks["robot_state_gate_ok"],
            "contact_above_object_z_ok": checks["contact_above_object_z_ok"],
            "axis_gate_required": axis_required,
            "sam_gripper_required": require_sam_gripper,
        },
        "selected_strategy_label": selected_pickup.get("label"),
        "selected_strategy_type": selected_pickup.get("strategy_type"),
        "side": side,
        "object_detection": obj,
        "gripper_detection": gripper,
        "source_object_xyz": source_xyz,
        "current_object_xyz": object_xyz,
        "planned_contact_center_m": _round_list(planned_contact, limit=3) if planned_contact else None,
        "adjusted_contact_center_m": _round_list(adjusted_contact, limit=3) if adjusted_contact else None,
        "contact_center_used_m": _round_list(contact_for_gate, limit=3) if contact_for_gate else None,
        "object_reference_mode": object_reference_mode,
        "object_detection_coherent": object_detection_coherent,
        "object_center_coherent": object_center_coherent,
        "object_contact_coherent": object_contact_coherent,
        "visible_object_mismatch": visible_object_mismatch,
        "allow_visible_object_mismatch": allow_visible_object_mismatch,
        "current_gripper_xyz": gripper_xyz,
        "object_xy_drift_m": round(object_xy_drift, 5) if object_xy_drift is not None else None,
        "object_z_delta_m": round(object_z_delta, 5) if object_z_delta is not None else None,
        "object_xy_to_contact_m": round(object_xy_to_contact, 5) if object_xy_to_contact is not None else None,
        "object_z_delta_to_contact_m": round(object_z_delta_to_contact, 5) if object_z_delta_to_contact is not None else None,
        "contact_above_object_z_m": round(contact_above_object_z, 5) if contact_above_object_z is not None else None,
        "gripper_xy_to_contact_m": round(gripper_xy_to_contact, 5) if gripper_xy_to_contact is not None else None,
        "gripper_z_delta_to_contact_m": round(gripper_z_delta_to_contact, 5) if gripper_z_delta_to_contact is not None else None,
        "limits": {
            "max_object_xy_drift_m": max_object_xy_drift,
            "max_object_z_jump_m": max_object_z_jump,
            "max_object_xy_to_contact_m": max_object_xy_to_contact,
            "max_object_z_to_contact_m": max_object_z_to_contact,
            "max_contact_above_object_z_m": max_contact_above_object_z,
            "max_gripper_xy_to_contact_m": max_gripper_xy_to_contact,
            "max_gripper_above_contact_m": max_gripper_above_contact,
            "max_gripper_below_contact_m": max_gripper_below_contact,
        },
        "axis_gate": axis_gate,
        "robot_state_gate": robot_state_gate,
        "axis_gate_required": axis_required,
    }
    packet["path"] = write_json(run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_pre_close_visual_gate.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## pre-close visual geometry gate",
            f"- passed: {packet['passed']}",
            f"- checks: {checks}",
            f"- required_checks: {packet['required_checks']}",
            f"- object_reference_mode: {packet['object_reference_mode']}",
            f"- object_xy_drift_m: {packet['object_xy_drift_m']}",
            f"- object_z_delta_m: {packet['object_z_delta_m']}",
            f"- object_xy_to_contact_m: {packet['object_xy_to_contact_m']}",
            f"- object_z_delta_to_contact_m: {packet['object_z_delta_to_contact_m']}",
            f"- contact_above_object_z_m: {packet['contact_above_object_z_m']}",
            f"- robot_state_gate: {robot_state_gate}",
            f"- gripper_xy_to_contact_m: {packet['gripper_xy_to_contact_m']}",
            f"- gripper_z_delta_to_contact_m: {packet['gripper_z_delta_to_contact_m']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _verify_place(
    *,
    selected_pickup: dict[str, Any],
    selected_place: dict[str, Any],
    post_observation: dict[str, Any],
) -> dict[str, Any]:
    source = selected_pickup.get("source_detection") or {}
    post = _select_object_detection(post_observation)
    rack = selected_place.get("rack_detection") or {}
    release_target = selected_place.get("place_target_object_ref_m")
    motion = compare_detection_motion(source, post)
    if post is None or not release_target:
        return {
            "success": False,
            "confidence": 0.2,
            "reason": f"missing post-place {TASK_OBJECT_NAME} detection or release target",
            "post_detection": post,
            "object_motion": motion,
        }
    post_xyz = _xyz(post)
    rack_xyz = _xyz(rack)
    lower_support = selected_place.get("lower_support_reference_m")
    if post_xyz is None:
        return {
            "success": False,
            "confidence": 0.2,
            "reason": "post-place detection has no 3D position",
            "post_detection": post,
            "object_motion": motion,
        }
    dx = post_xyz[0] - float(release_target[0])
    dy = post_xyz[1] - float(release_target[1])
    xy_to_target = (dx * dx + dy * dy) ** 0.5
    support_z = float(lower_support[2]) if isinstance(lower_support, list) and len(lower_support) >= 3 else (rack_xyz[2] if rack_xyz else -999.0)
    z_floor = support_z - _env_float("OPENFORGE_BOWL_RACK_VERIFY_Z_BELOW_RACK_TOL_M", 0.04)
    xy_ok = xy_to_target <= _env_float("OPENFORGE_BOWL_RACK_VERIFY_MAX_XY_TO_TARGET_M", 0.18)
    z_ok = post_xyz[2] >= z_floor
    success = bool(xy_ok and z_ok)
    return {
        "success": success,
        "confidence": 0.75 if success else 0.3,
        "reason": f"{TASK_OBJECT_NAME} detected near rack placement target" if success else "post detection not close enough to rack target",
        "post_detection": post,
        "object_motion": motion,
        "xy_to_target_m": round(xy_to_target, 5),
        "z_floor_m": round(z_floor, 5),
        "checks": {"xy_ok": xy_ok, "z_ok": z_ok},
    }


def _execute_pickup_lift_only(run_dir: Path, selected_pickup: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    side = str(selected_pickup["arm"])
    close_plan = selected_pickup.get("staged_close_plan") or {}
    attempt: dict[str, Any] = {
        "schema": "openforge.bowl_rack.pickup_lift_only_physical_attempt.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "steps": [],
        "gripper_after_close": None,
        "why_stopped": None,
    }
    try:
        gripper_limit_kwargs = _gripper_limit_kwargs()
        pregrasp_open = _preshape_gripper(selected_pickup)
        hold = _hold_gripper(selected_pickup)
        open_step = _set_gripper_checked(
            side=side,
            target=pregrasp_open,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="checked_preshape_open",
            confirm_min=_env_float("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(open_step)
        if not open_step["ok"]:
            attempt["why_stopped"] = "gripper did not confirm open before arm motion"
            return attempt

        pickup_specs = _pickup_lift_specs(selected_pickup)
        before_close_specs = [spec for spec in pickup_specs if not _is_pickup_lift_label(spec[0])]
        lift_specs = [spec for spec in pickup_specs if _is_pickup_lift_label(spec[0])]
        if not lift_specs:
            raise RuntimeError("pickup/lift sequence missing bowl_small_lift stage")

        def _open_and_retreat(stage_prefix: str) -> None:
            release = _set_gripper_checked(
                side=side,
                target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                stage=f"{stage_prefix}_open",
                confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
                **gripper_limit_kwargs,
            )
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
            attempt["steps"].append(release)
            retreat_pose = _pickup_retreat_pose(selected_pickup)
            retreat = _execute_pose(
                freespace_move=freespace_move,
                side=side,
                pose=retreat_pose,
                label=f"{stage_prefix}_retreat",
                gripper=1.0,
            )
            attempt["movement_capable_calls"].append(f"freespace_move:{stage_prefix}_retreat")
            attempt["steps"].append(retreat)

        for label, pose, gripper in before_close_specs:
            try:
                step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append(step)
            except Exception as exc:
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append({"stage": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                attempt["why_stopped"] = f"{label} motion failed before close; stop before close/lift"
                try:
                    _open_and_retreat("before_close_motion_failed")
                except Exception as retreat_exc:
                    attempt["steps"].append(
                        {
                            "stage": "before_close_motion_failed_retreat_exception",
                            "ok": False,
                            "error": f"{type(retreat_exc).__name__}: {retreat_exc}",
                        }
                    )
                return attempt

        attempt["pre_close_observation"] = _capture_object_observation(run_dir, "pre_close_grasp_pose_observe")
        pre_close_gate = _pre_close_visual_gate(run_dir, selected_pickup, attempt["pre_close_observation"])
        attempt["pre_close_visual_gate"] = pre_close_gate
        if _truthy_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", True) and not bool(pre_close_gate.get("passed")):
            attempt["why_stopped"] = "pre-close visual geometry gate failed; stop before close/lift"
            _open_and_retreat("pre_close_gate_failed")
            return attempt

        close = staged_close_with_contact(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=hold,
            steps=_csv_float("OPENFORGE_BOWL_RACK_CLOSE_STEPS", list(close_plan.get("steps") or [0.68, 0.56, 0.48, 0.40])),
            **gripper_limit_kwargs,
            min_contact_delta=_env_float(
                "OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS",
                float(close_plan.get("expected_contact_delta_min", 0.010)),
            ),
            target_tolerance=_env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_BOWL_RACK_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_BOWL_RACK_HOLD_MAX_POS", 0.95),
            confirm_timeout_s=_env_float(
                "OPENFORGE_BOWL_RACK_CLOSE_CONFIRM_TIMEOUT_S",
                _env_float("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_TIMEOUT_S", 0.0),
            ),
            state_poll_s=_env_float("OPENFORGE_BOWL_RACK_GRIPPER_STATE_POLL_S", 0.05),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage=f"{TASK_ARTIFACT_PREFIX}_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["gripper_after_close"] = close
        attempt["post_close_observation"] = _capture_object_observation(run_dir, "post_close_before_lift_observe")
        if _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True) and not bool(close.get("contact_detected")):
            attempt["why_stopped"] = "staged close did not produce contact evidence; stop before lift"
            _open_and_retreat("no_contact_after_close")
            return attempt
        if not bool(close.get("plausible_for_lift")):
            attempt["why_stopped"] = "staged close did not produce plausible gripper-state evidence; stop before lift"
            _open_and_retreat("implausible_hold_after_close")
            return attempt

        label, pose, gripper = lift_specs[0]
        lift_step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
        attempt["movement_capable_calls"].append(f"freespace_move:{label}")
        attempt["steps"].append(lift_step)

        post_lift = _capture_object_observation(run_dir, "post_lift_pickup_only_observe")
        post_lift_det = _select_object_detection(post_lift)
        attempt["post_lift_observation"] = post_lift
        attempt["lift_verification"] = verify_lift(
            selected_pickup.get("source_detection"),
            post_lift_det,
            min_z_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MIN_Z_DELTA_M", 0.015),
            max_xy_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MAX_XY_DELTA_M", 0.24),
        )
        attempt["verification_success"] = bool(attempt["lift_verification"].get("success"))
        attempt["success"] = bool(attempt["verification_success"])

        if _truthy_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", True):
            return_steps: list[dict[str, Any]] = []
            for return_label, return_pose, return_gripper in [
                ("return_to_grasp_pose", selected_pickup["grasp_pose"], hold),
                (
                    "retreat_after_table_release",
                    _pickup_retreat_pose(selected_pickup),
                    1.0,
                ),
            ]:
                if return_label == "retreat_after_table_release":
                    release = _set_gripper_checked(
                        side=side,
                        target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
                        set_gripper=set_gripper,
                        get_robot_state=get_robot_state,
                        stage="open_release_back_on_table",
                        confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
                        **gripper_limit_kwargs,
                    )
                    attempt["movement_capable_calls"].append(f"set_gripper:{side}")
                    return_steps.append(release)
                    attempt["steps"].append(release)
                step = _execute_pose(
                    freespace_move=freespace_move,
                    side=side,
                    pose=return_pose,
                    label=return_label,
                    gripper=return_gripper,
                )
                attempt["movement_capable_calls"].append(f"freespace_move:{return_label}")
                return_steps.append(step)
                attempt["steps"].append(step)
            attempt["return_to_table"] = {"enabled": True, "steps": return_steps}
            attempt["post_return_observation"] = _capture_object_observation(run_dir, "post_return_table_observe")

        attempt["why_stopped"] = (
            "pickup/lift verified; returned bowl to table"
            if attempt["success"] and attempt.get("return_to_table", {}).get("enabled")
            else "lift command completed; lift verification failed or inconclusive"
            if not attempt["success"]
            else "pickup/lift verified"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _execute_direct_lower_rack_pose_adjust(
    run_dir: Path,
    side: str,
    *,
    target_pose: dict[str, Any],
    stage: str,
    max_xy_delta_m: float,
    max_z_delta_m: float,
    target_axes: set[str] | None = None,
    step_s_env: str = "OPENFORGE_BOWL_LOWER_RACK_DIRECT_STEP_S",
    z_direction: str | None = None,
    max_axis_delta_m: float | None = None,
) -> dict[str, Any]:
    # Lower-rack insertion/retreat should be a small local correction from the
    # current front-entry/release pose, not another broad planner jump inside
    # the shelf.  We solve one seeded IK keypoint to the requested TCP position
    # while preserving the current TCP orientation and both gripper positions.
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.direct_pose_adjust.v1",
        "success": False,
        "stage": stage,
        "side": side,
        "physical_motion_executed": False,
        "target_pose": json_safe(target_pose),
        "target_axes": sorted(target_axes) if target_axes else ["x", "y", "z"],
        "step_s_env": step_s_env,
        "z_direction": z_direction,
        "max_axis_delta_m": max_axis_delta_m,
        "why_stopped": None,
    }
    target_xyz = target_pose.get("position") if isinstance(target_pose, dict) else None
    if not target_xyz or len(target_xyz) < 3:
        packet["why_stopped"] = "target pose has no xyz position"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    env = _tool_env("freespace_move") or _tool_env("get_robot_state")
    if env is None:
        packet["why_stopped"] = "cannot access direct YAM env for lower-rack pose adjust"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet
    move_keypoints = getattr(env, "move_bimanual_joint_keypoints", None)
    if not callable(move_keypoints):
        packet["why_stopped"] = "direct YAM env does not expose move_bimanual_joint_keypoints"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    try:
        from robot.yam.kinematics import YamKinematics
    except Exception as exc:
        packet["why_stopped"] = f"cannot load YamKinematics for lower-rack pose adjust: {exc}"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    try:
        left_obs = _read_direct_arm_obs(env, "left")
        right_obs = _read_direct_arm_obs(env, "right")
        left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        left_gripper = float(np.asarray(left_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        right_gripper = float(np.asarray(right_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    except Exception as exc:
        packet["why_stopped"] = f"cannot read direct arm observations for lower-rack pose adjust: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    kin = YamKinematics()
    try:
        left_pos, left_quat, right_pos, right_quat = kin.forward_kinematics(left_base, right_base)
    except Exception as exc:
        packet["why_stopped"] = f"forward kinematics failed for lower-rack pose adjust: {type(exc).__name__}: {exc}"
        packet["left_start_joint_pos"] = _round_list(left_base, limit=6)
        packet["right_start_joint_pos"] = _round_list(right_base, limit=6)
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet
    left_target_pos = np.asarray(left_pos, dtype=np.float64).copy()
    right_target_pos = np.asarray(right_pos, dtype=np.float64).copy()
    try:
        target = np.asarray([float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])], dtype=np.float64)
    except Exception as exc:
        packet["why_stopped"] = f"malformed target xyz for lower-rack pose adjust: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet
    transport_z_targeted = False
    transport_z_boost = 0.0
    if target_axes:
        invalid_axes = set(target_axes) - {"x", "y", "z"}
        if invalid_axes:
            packet["why_stopped"] = f"invalid target axes for lower-rack pose adjust: {sorted(invalid_axes)}"
            packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
            return packet
        axis_indices = {"x": 0, "y": 1, "z": 2}
        active_current = np.asarray(left_pos if side == "left" else right_pos, dtype=np.float64)
        axis_target = active_current.copy()
        for axis in target_axes:
            axis_target[axis_indices[axis]] = target[axis_indices[axis]]
        packet["raw_target_pose_xyz_m"] = _round_list(target)
        if max_axis_delta_m is not None:
            limit = abs(float(max_axis_delta_m))
            requested_axis_target = axis_target.copy()
            for axis in target_axes:
                idx = axis_indices[axis]
                delta = float(axis_target[idx] - active_current[idx])
                axis_target[idx] = active_current[idx] + max(-limit, min(limit, delta))
            packet["requested_axis_target_active_tcp_m"] = _round_list(requested_axis_target)
            packet["axis_delta_limit_m"] = round(limit, 5)
        stage_text = str(stage).lower()
        horizontal_transport = set(target_axes).issubset({"x", "y"}) and (
            "y_align" in stage_text or "x_insert" in stage_text
        )
        if horizontal_transport:
            transport_z_boost = max(
                0.0,
                _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_TRANSPORT_Z_BOOST_M", 0.0),
            )
            if "y_align" in stage_text:
                transport_z_boost = max(
                    0.0,
                    _env_float(
                        "OPENFORGE_BOWL_LOWER_RACK_DIRECT_Y_ALIGN_Z_BOOST_M",
                        transport_z_boost,
                    ),
                )
            elif "x_insert" in stage_text:
                transport_z_boost = max(
                    0.0,
                    _env_float(
                        "OPENFORGE_BOWL_LOWER_RACK_DIRECT_X_INSERT_Z_BOOST_M",
                        transport_z_boost,
                    ),
                )
            if transport_z_boost > 0.0:
                requested_transport_target = axis_target.copy()
                requested_transport_target[2] = active_current[2] + transport_z_boost
                axis_target[2] = requested_transport_target[2]
                target_axes = set(target_axes) | {"z"}
                transport_z_targeted = True
                max_z_delta_m = max(
                    max_z_delta_m,
                    _env_float(
                        "OPENFORGE_BOWL_LOWER_RACK_DIRECT_TRANSPORT_MAX_Z_DELTA_M",
                        transport_z_boost + 0.010,
                    ),
                )
                packet["target_axes"] = sorted(target_axes)
                packet["transport_z_targeted"] = True
                packet["transport_z_boost_m"] = round(transport_z_boost, 5)
                packet["requested_transport_target_active_tcp_m"] = _round_list(
                    requested_transport_target
                )
                packet["transport_z_policy"] = (
                    "Horizontal shelf transport targets a small upward TCP boost while preserving "
                    "the current gripper orientation, so IK residual does not let the dish sag into "
                    "the table or shelf lip during Y-align/X-insert."
                )
        if z_direction == "down_only" and "z" in target_axes:
            min_delta = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MIN_DELTA_M", 0.003))
            if target[2] >= active_current[2] - min_delta:
                packet.update(
                    {
                        "success": True,
                        "skipped": True,
                        "why_stopped": "down-only Z place skipped; current TCP is already at or below target place height",
                        "start_active_tcp_m": _round_list(active_current),
                        "target_active_tcp_m": _round_list(active_current),
                        "requested_target_active_tcp_m": _round_list(axis_target),
                        "command_delta_m": [0.0, 0.0, 0.0],
                        "left_start_joint_pos": _round_list(left_base, limit=6),
                        "right_start_joint_pos": _round_list(right_base, limit=6),
                        "left_gripper_pos": round(left_gripper, 5),
                        "right_gripper_pos": round(right_gripper, 5),
                        "fk_start": {"left_tcp_m": _round_list(left_pos), "right_tcp_m": _round_list(right_pos)},
                    }
                )
                packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
                return packet
        if z_direction == "up_only" and "z" in target_axes:
            min_delta = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MIN_DELTA_M", 0.003))
            if target[2] <= active_current[2] + min_delta:
                packet.update(
                    {
                        "success": True,
                        "skipped": True,
                        "why_stopped": "up-only Z clear skipped; current TCP is already at or above target clear height",
                        "start_active_tcp_m": _round_list(active_current),
                        "target_active_tcp_m": _round_list(active_current),
                        "requested_target_active_tcp_m": _round_list(axis_target),
                        "command_delta_m": [0.0, 0.0, 0.0],
                        "left_start_joint_pos": _round_list(left_base, limit=6),
                        "right_start_joint_pos": _round_list(right_base, limit=6),
                        "left_gripper_pos": round(left_gripper, 5),
                        "right_gripper_pos": round(right_gripper, 5),
                        "fk_start": {"left_tcp_m": _round_list(left_pos), "right_tcp_m": _round_list(right_pos)},
                    }
                )
                packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
                return packet
        target = axis_target
    if side == "left":
        left_target_pos[:] = target
    elif side == "right":
        right_target_pos[:] = target
    else:
        packet["why_stopped"] = f"unsupported side for lower-rack pose adjust: {side!r}"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    try:
        ik_left, ik_right = kin.inverse_kinematics(
            left_target_pos,
            np.asarray(left_quat, dtype=np.float64),
            right_target_pos,
            np.asarray(right_quat, dtype=np.float64),
            seeded=True,
            max_iters=max(1, _env_int("OPENFORGE_BOWL_LOWER_RACK_DIRECT_IK_MAX_ITERS", 80)),
            err_threshold=max(1e-6, _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_IK_INTERNAL_ERR_THRESHOLD", 1e-4)),
        )
        end_left_pos, end_left_quat, end_right_pos, end_right_quat = kin.forward_kinematics(ik_left, ik_right)
    except Exception as exc:
        packet["why_stopped"] = f"IK/FK solve failed for lower-rack pose adjust: {type(exc).__name__}: {exc}"
        packet["left_start_joint_pos"] = _round_list(left_base, limit=6)
        packet["right_start_joint_pos"] = _round_list(right_base, limit=6)
        packet["fk_start"] = {"left_tcp_m": _round_list(left_pos), "right_tcp_m": _round_list(right_pos)}
        packet["fk_target"] = {"left_tcp_m": _round_list(left_target_pos), "right_tcp_m": _round_list(right_target_pos)}
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    active_start = np.asarray(left_pos if side == "left" else right_pos, dtype=np.float64)
    active_target = np.asarray(left_target_pos if side == "left" else right_target_pos, dtype=np.float64)
    active_end = np.asarray(end_left_pos if side == "left" else end_right_pos, dtype=np.float64)
    active_start_quat = np.asarray(left_quat if side == "left" else right_quat, dtype=np.float64)
    active_end_quat = np.asarray(end_left_quat if side == "left" else end_right_quat, dtype=np.float64)
    other_start = np.asarray(right_pos if side == "left" else left_pos, dtype=np.float64)
    other_end = np.asarray(end_right_pos if side == "left" else end_left_pos, dtype=np.float64)
    active_delta = active_target - active_start
    active_xy_delta = float(np.linalg.norm(active_delta[:2]))
    active_z_delta_command = abs(float(active_delta[2]))
    active_xy_drift = float(np.linalg.norm((active_end - active_start)[:2]))
    left_joint_delta = np.asarray(ik_left, dtype=np.float64) - left_base
    right_joint_delta = np.asarray(ik_right, dtype=np.float64) - right_base
    max_joint_delta = max(float(np.max(np.abs(left_joint_delta))), float(np.max(np.abs(right_joint_delta))))
    active_pos_error = float(np.linalg.norm(active_end - active_target))
    active_z_drift = abs(float(active_end[2] - active_target[2]))
    other_tcp_motion = float(np.linalg.norm(other_end - other_start))
    active_rot_delta_deg = _quat_angle_deg(active_start_quat, active_end_quat)

    min_delta = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MIN_DELTA_M", 0.003))
    max_ik_error = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_IK_ERROR_M", 0.015)
    stage_text = str(stage).lower()
    if "z_clear_after_release" in stage_text:
        max_ik_error = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_IK_ERROR_M", max_ik_error)
    elif "y_align" in stage_text:
        max_ik_error = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_Y_ALIGN_MAX_IK_ERROR_M", max_ik_error)
    elif "x_insert" in stage_text:
        max_ik_error = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_X_INSERT_MAX_IK_ERROR_M", max_ik_error)
    elif "retreat" in stage_text:
        max_ik_error = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_IK_ERROR_M", max_ik_error)
    z_only_motion = bool(target_axes) and set(target_axes) == {"z"}
    max_z_only_xy_drift = _env_float(
        "OPENFORGE_BOWL_LOWER_RACK_DIRECT_Z_ONLY_MAX_ACTIVE_XY_DRIFT_M",
        max_ik_error,
    )
    if z_only_motion and "z_clear_after_release" in stage_text:
        max_z_only_xy_drift = _env_float(
            "OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_XY_DRIFT_M",
            max_z_only_xy_drift,
        )
    max_joint_delta_allowed = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_JOINT_DELTA_RAD", 0.50)
    max_other_motion = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_OTHER_TCP_MOTION_M", 0.010)
    max_rot_delta = _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_ROT_DELTA_DEG", 5.0)
    max_transport_z_error = _env_float(
        "OPENFORGE_BOWL_LOWER_RACK_DIRECT_TRANSPORT_MAX_Z_TARGET_ERROR_M",
        max_ik_error,
    )
    gate = {
        "passed": (
            active_pos_error <= max_ik_error
            and active_xy_delta <= max_xy_delta_m
            and active_z_delta_command <= max_z_delta_m
            and other_tcp_motion <= max_other_motion
            and active_rot_delta_deg <= max_rot_delta
            and (not z_only_motion or active_xy_drift <= max_z_only_xy_drift)
            and (not transport_z_targeted or active_z_drift <= max_transport_z_error)
        ),
        "max_ik_error_m": max_ik_error,
        "max_xy_delta_m": max_xy_delta_m,
        "max_z_delta_m": max_z_delta_m,
        "max_joint_delta_rad": max_joint_delta_allowed,
        "joint_delta_gate_enforced": False,
        "max_other_tcp_motion_m": max_other_motion,
        "max_rot_delta_deg": max_rot_delta,
        "observed_active_pos_error_m": round(active_pos_error, 5),
        "observed_command_xy_delta_m": round(active_xy_delta, 5),
        "observed_command_z_delta_m": round(active_z_delta_command, 5),
        "observed_active_xy_drift_m": round(active_xy_drift, 5),
        "z_only_xy_drift_gate_enforced": z_only_motion,
        "max_z_only_active_xy_drift_m": round(max_z_only_xy_drift, 5) if z_only_motion else None,
        "transport_z_targeted": transport_z_targeted,
        "transport_z_boost_m": round(transport_z_boost, 5) if transport_z_targeted else None,
        "max_transport_z_target_error_m": (
            round(max_transport_z_error, 5) if transport_z_targeted else None
        ),
        "observed_active_z_target_error_m": round(active_z_drift, 5),
        "observed_max_joint_delta_rad": round(max_joint_delta, 5),
        "observed_other_tcp_motion_m": round(other_tcp_motion, 5),
        "observed_active_rot_delta_deg": round(active_rot_delta_deg, 5),
        "reason": None,
    }
    if active_pos_error > max_ik_error:
        gate["reason"] = f"active TCP IK error {active_pos_error:.3f} m exceeds {max_ik_error:.3f} m"
    elif active_xy_delta > max_xy_delta_m:
        gate["reason"] = f"requested XY delta {active_xy_delta:.3f} m exceeds {max_xy_delta_m:.3f} m"
    elif active_z_delta_command > max_z_delta_m:
        gate["reason"] = f"requested Z delta {active_z_delta_command:.3f} m exceeds {max_z_delta_m:.3f} m"
    elif other_tcp_motion > max_other_motion:
        gate["reason"] = f"other TCP moved {other_tcp_motion:.3f} m, above {max_other_motion:.3f} m"
    elif active_rot_delta_deg > max_rot_delta:
        gate["reason"] = f"active TCP rotation changed {active_rot_delta_deg:.2f} deg, above {max_rot_delta:.2f} deg"
    elif z_only_motion and active_xy_drift > max_z_only_xy_drift:
        gate["reason"] = (
            f"z-only active TCP XY drift {active_xy_drift:.3f} m exceeds "
            f"{max_z_only_xy_drift:.3f} m"
        )
    elif transport_z_targeted and active_z_drift > max_transport_z_error:
        gate["reason"] = (
            f"transport active TCP Z target error {active_z_drift:.3f} m exceeds "
            f"{max_transport_z_error:.3f} m"
        )

    packet.update(
        {
            "gate": gate,
            "start_active_tcp_m": _round_list(active_start),
            "target_active_tcp_m": _round_list(active_target),
            "end_active_tcp_m": _round_list(active_end),
            "command_delta_m": _round_list(active_delta),
            "left_start_joint_pos": _round_list(left_base, limit=6),
            "right_start_joint_pos": _round_list(right_base, limit=6),
            "left_end_joint_pos": _round_list(ik_left, limit=6),
            "right_end_joint_pos": _round_list(ik_right, limit=6),
            "left_joint_delta_rad": _round_list(left_joint_delta, limit=6),
            "right_joint_delta_rad": _round_list(right_joint_delta, limit=6),
            "left_gripper_pos": round(left_gripper, 5),
            "right_gripper_pos": round(right_gripper, 5),
            "fk_start": {"left_tcp_m": _round_list(left_pos), "right_tcp_m": _round_list(right_pos)},
            "fk_end": {"left_tcp_m": _round_list(end_left_pos), "right_tcp_m": _round_list(end_right_pos)},
        }
    )
    if active_xy_delta < min_delta and active_z_delta_command < min_delta:
        packet["success"] = True
        packet["skipped"] = True
        packet["why_stopped"] = "direct pose adjustment below minimum; no direct motion needed"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet
    if not gate["passed"]:
        packet["why_stopped"] = f"direct lower-rack pose gate failed: {gate['reason']}"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet

    duration_s = max(0.05, _env_float(step_s_env, 0.80))
    packet["duration_s"] = round(duration_s, 5)
    packet["movement_command_invoked"] = True
    try:
        result = move_keypoints(
            timestamps=[0.0, duration_s],
            left_joint_positions=[left_base.tolist(), np.asarray(ik_left, dtype=np.float64).tolist()],
            right_joint_positions=[right_base.tolist(), np.asarray(ik_right, dtype=np.float64).tolist()],
            left_gripper_positions=[left_gripper, left_gripper],
            right_gripper_positions=[right_gripper, right_gripper],
            playback_speed=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLAYBACK_SPEED", 1.0),
            command_hz=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_COMMAND_HZ", 60.0),
            start_interp_s=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_START_INTERP_S", 0.25),
        )
    except Exception as exc:
        packet["physical_motion_executed"] = True
        packet["why_stopped"] = f"direct keypoint command raised for lower-rack pose adjust: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
        return packet
    packet["physical_motion_executed"] = True
    packet["result"] = json_safe(result)
    packet["success"] = bool(result.get("success")) if isinstance(result, dict) else False
    packet["why_stopped"] = (
        f"{stage} executed"
        if packet["success"]
        else f"{stage} command returned unsuccessful result"
    )
    packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(stage)}.json", json_safe(packet))
    append_stage_summary(
        run_dir,
        [
            f"## {stage}",
            f"- success: {packet['success']}",
            f"- physical_motion_executed: {packet['physical_motion_executed']}",
            f"- command_delta_m: {packet['command_delta_m']}",
            f"- gate: {gate['passed']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _current_active_tcp_pose_with_offsets(
    side: str,
    *,
    x_offset_m: float = 0.0,
    y_offset_m: float = 0.0,
    z_offset_m: float = 0.0,
) -> dict[str, Any]:
    env = _tool_env("freespace_move") or _tool_env("get_robot_state")
    if env is None:
        raise RuntimeError("cannot access direct YAM env for current active TCP pose")
    try:
        from robot.yam.kinematics import YamKinematics
    except Exception as exc:
        raise RuntimeError(f"cannot load YamKinematics for current active TCP pose: {exc}") from exc

    left_obs = _read_direct_arm_obs(env, "left")
    right_obs = _read_direct_arm_obs(env, "right")
    left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
    right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
    left_pos, left_quat, right_pos, right_quat = YamKinematics().forward_kinematics(left_base, right_base)
    if side == "left":
        active_pos = np.asarray(left_pos, dtype=np.float64)
        active_quat = np.asarray(left_quat, dtype=np.float64)
    elif side == "right":
        active_pos = np.asarray(right_pos, dtype=np.float64)
        active_quat = np.asarray(right_quat, dtype=np.float64)
    else:
        raise RuntimeError(f"unsupported side for current active TCP pose: {side!r}")
    return {
        "position": [
            float(active_pos[0]) + float(x_offset_m),
            float(active_pos[1]) + float(y_offset_m),
            float(active_pos[2]) + float(z_offset_m),
        ],
        "rpy": _quat_xyzw_to_display_rpy(active_quat),
    }


def _current_active_tcp_pose_with_z_offset(side: str, *, z_offset_m: float) -> dict[str, Any]:
    return _current_active_tcp_pose_with_offsets(side, z_offset_m=z_offset_m)


def _execute_post_lift_j5_reorientation(
    run_dir: Path,
    side: str,
    *,
    delta_deg: float | None = None,
    joint_index: int | None = None,
    selected_pickup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # This is intentionally a small joint-space wrist-roll scout, not a broad
    # planner move.  It keeps the lifted bowl at the current arm configuration,
    # changes only one wrist joint, then exports the resulting FK orientation
    # so later rack poses do not plan back to the old top-down pickup rpy.
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.post_lift_j5_reorient_execute.v1",
        "success": False,
        "side": side,
        "physical_motion_executed": False,
        "why_stopped": None,
    }
    if not _post_lift_j5_reorient_enabled():
        packet["success"] = True
        packet["skipped"] = True
        packet["why_stopped"] = "post-lift j5 reorientation disabled"
        return packet

    env = _tool_env("freespace_move") or _tool_env("get_robot_state")
    if env is None:
        packet["why_stopped"] = "cannot access direct YAM env for post-lift j5 reorientation"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet
    move_keypoints = getattr(env, "move_bimanual_joint_keypoints", None)
    if not callable(move_keypoints):
        packet["why_stopped"] = "direct YAM env does not expose move_bimanual_joint_keypoints"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    try:
        from robot.yam.kinematics import YamKinematics
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        packet["why_stopped"] = f"cannot load YamKinematics/Rotation for post-lift j5 reorientation: {exc}"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    try:
        joint_index = _post_lift_j5_joint_index() if joint_index is None else int(joint_index)
        if joint_index < 0 or joint_index >= 6:
            raise RuntimeError("post-lift j5 execution joint_index must be in [0,5]")
        delta_rad = _post_lift_j5_delta_rad(delta_deg)
        left_obs = _read_direct_arm_obs(env, "left")
        right_obs = _read_direct_arm_obs(env, "right")
        left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        left_gripper = float(np.asarray(left_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        right_gripper = float(np.asarray(right_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    except Exception as exc:
        packet["why_stopped"] = f"cannot read direct arm observations for post-lift j5: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    kin = YamKinematics()
    try:
        left_start_pos, left_start_quat, right_start_pos, right_start_quat = kin.forward_kinematics(left_base, right_base)
    except Exception as exc:
        packet["why_stopped"] = f"forward kinematics failed before post-lift j5: {type(exc).__name__}: {exc}"
        packet["left_start_joint_pos"] = _round_list(left_base, limit=6)
        packet["right_start_joint_pos"] = _round_list(right_base, limit=6)
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    left_target = left_base.copy()
    right_target = right_base.copy()
    if side == "left":
        left_target[joint_index] += delta_rad
    elif side == "right":
        right_target[joint_index] += delta_rad
    else:
        packet["why_stopped"] = f"unsupported side for post-lift j5 reorientation: {side!r}"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    try:
        left_end_pos, left_end_quat, right_end_pos, right_end_quat = kin.forward_kinematics(left_target, right_target)
    except Exception as exc:
        packet["why_stopped"] = f"forward kinematics failed after post-lift j5 target: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    combined_orientation = _combine_post_lift_orientation_enabled()
    combined_ops: list[dict[str, Any]] = []
    combined_target_quat: np.ndarray | None = None
    combined_target_rot_error = 0.0
    if combined_orientation:
        active_j5_quat = np.asarray(left_end_quat if side == "left" else right_end_quat, dtype=np.float64)
        target_rotation = Rotation.from_quat(active_j5_quat)

        if _truthy_env("OPENFORGE_BOWL_LOWER_RACK_UNDO_PICKUP_INWARD_TILT_AFTER_LIFT", False):
            pickup_tilt_model = ((selected_pickup or {}).get("rim_geometry_estimate") or {}).get(
                "inward_tilt_model"
            ) or (((selected_pickup or {}).get("selected_rim_edge") or {}).get("approach_side") or {}).get(
                "inward_tilt_model"
            ) or {}
            raw_axis = pickup_tilt_model.get("inward_axis_world")
            axis = np.asarray(raw_axis if raw_axis is not None else [], dtype=np.float64)
            applied_deg = pickup_tilt_model.get("applied_axis_rotation_deg")
            if applied_deg is None:
                applied_deg = -_env_float("OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG", 0.0)
            axis_norm = float(np.linalg.norm(axis)) if axis.size == 3 else 0.0
            if axis_norm > 1e-8 and abs(float(applied_deg)) > 1e-8:
                undo_sign = _env_float("OPENFORGE_BOWL_LOWER_RACK_PICKUP_INWARD_TILT_UNDO_SIGN", -1.0)
                undo_deg = undo_sign * float(applied_deg)
                target_rotation = Rotation.from_rotvec(axis / axis_norm * math.radians(undo_deg)) * target_rotation
                combined_ops.append(
                    {
                        "operation": "undo_pickup_inward_tilt",
                        "applied_pickup_axis_rotation_deg": round(float(applied_deg), 5),
                        "undo_sign": round(float(undo_sign), 5),
                        "undo_axis_rotation_deg": round(undo_deg, 5),
                        "axis_world": _round_list(axis / axis_norm, limit=5),
                    }
                )
            else:
                combined_ops.append(
                    {
                        "operation": "undo_pickup_inward_tilt",
                        "skipped": True,
                        "reason": "missing pickup inward tilt axis or zero applied tilt",
                    }
                )

        if _pre_insert_x_up_tilt_enabled():
            x_up_tilt_deg = _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT_DEG", 0.0)
            max_abs_deg = abs(_env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT_MAX_ABS_DEG", 20.0))
            if abs(x_up_tilt_deg) > max_abs_deg + 1e-9:
                packet["why_stopped"] = (
                    f"combined pre-insert X-up tilt {x_up_tilt_deg:.2f} deg exceeds max {max_abs_deg:.2f} deg"
                )
                packet["path"] = write_json(
                    run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
                    json_safe(packet),
                )
                return packet
            if abs(x_up_tilt_deg) > 1e-8:
                world_y_rotation_deg = -float(x_up_tilt_deg)
                target_rotation = Rotation.from_euler("y", world_y_rotation_deg, degrees=True) * target_rotation
                combined_ops.append(
                    {
                        "operation": "pre_insert_x_up_tilt",
                        "x_up_tilt_deg": round(x_up_tilt_deg, 5),
                        "world_y_rotation_deg": round(world_y_rotation_deg, 5),
                    }
                )

        combined_target_quat = target_rotation.as_quat()
        left_target_pos = np.asarray(left_end_pos, dtype=np.float64)
        right_target_pos = np.asarray(right_end_pos, dtype=np.float64)
        left_target_quat = np.asarray(left_end_quat, dtype=np.float64)
        right_target_quat = np.asarray(right_end_quat, dtype=np.float64)
        if side == "left":
            left_target_quat = combined_target_quat
        else:
            right_target_quat = combined_target_quat
        try:
            ik_left, ik_right = kin.inverse_kinematics(
                left_target_pos,
                left_target_quat,
                right_target_pos,
                right_target_quat,
                seeded=True,
                max_iters=max(1, _env_int("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_COMBINED_IK_MAX_ITERS", 140)),
                err_threshold=max(
                    1e-6,
                    _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_COMBINED_IK_ERR_THRESHOLD", 1e-4),
                ),
            )
            left_target = np.asarray(ik_left, dtype=np.float64)
            right_target = np.asarray(ik_right, dtype=np.float64)
            left_end_pos, left_end_quat, right_end_pos, right_end_quat = kin.forward_kinematics(
                left_target,
                right_target,
            )
        except Exception as exc:
            packet["why_stopped"] = f"IK/FK solve failed for combined post-lift orientation: {type(exc).__name__}: {exc}"
            packet["path"] = write_json(
                run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
                json_safe(packet),
            )
            return packet

    active_start = np.asarray(left_start_pos if side == "left" else right_start_pos, dtype=np.float64)
    active_end = np.asarray(left_end_pos if side == "left" else right_end_pos, dtype=np.float64)
    active_quat = np.asarray(left_end_quat if side == "left" else right_end_quat, dtype=np.float64)
    if combined_target_quat is not None:
        combined_target_rot_error = _quat_angle_deg(active_quat, combined_target_quat)
    other_start = np.asarray(right_start_pos if side == "left" else left_start_pos, dtype=np.float64)
    other_end = np.asarray(right_end_pos if side == "left" else left_end_pos, dtype=np.float64)
    tcp_translation = float(np.linalg.norm(active_end - active_start))
    other_tcp_motion = float(np.linalg.norm(other_end - other_start))
    max_tcp_translation = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_MAX_TCP_TRANSLATION_M", 0.120)
    max_other_motion = _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_MAX_OTHER_TCP_MOTION_M", 0.010)
    max_combined_rot_error = _env_float(
        "OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_COMBINED_MAX_ROT_ERROR_DEG",
        _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_MAX_ROT_ERROR_DEG", 4.0),
    )
    joint_delta = np.asarray(left_target if side == "left" else right_target, dtype=np.float64) - np.asarray(
        left_base if side == "left" else right_base, dtype=np.float64
    )
    rotated_rpy = _quat_xyzw_to_display_rpy(active_quat)
    tangent_gate = (
        _post_lift_tangent_alignment_gate(selected_pickup, rotated_rpy)
        if selected_pickup is not None
        else {"enabled": False, "passed": True, "why_stopped": None}
    )
    packet.update(
        {
            "joint_index": joint_index,
            "delta_rad": round(delta_rad, 5),
            "delta_deg": round(math.degrees(delta_rad), 3),
            "rotated_rpy": _round_list(rotated_rpy, limit=3),
            "active_tcp_translation_m": round(tcp_translation, 5),
            "other_tcp_motion_m": round(other_tcp_motion, 5),
            "max_tcp_translation_m": round(max_tcp_translation, 5),
            "max_other_tcp_motion_m": round(max_other_motion, 5),
            "combined_orientation": combined_orientation,
            "combined_orientation_ops": combined_ops,
            "combined_target_rot_error_deg": round(combined_target_rot_error, 5),
            "max_combined_target_rot_error_deg": round(max_combined_rot_error, 5),
            "left_start_joint_pos": _round_list(left_base, limit=6),
            "right_start_joint_pos": _round_list(right_base, limit=6),
            "left_target_joint_pos": _round_list(left_target, limit=6),
            "right_target_joint_pos": _round_list(right_target, limit=6),
            "active_joint_delta_rad": _round_list(joint_delta, limit=6),
            "left_gripper_pos": round(left_gripper, 5),
            "right_gripper_pos": round(right_gripper, 5),
            "fk_start": {"left_tcp_m": _round_list(left_start_pos), "right_tcp_m": _round_list(right_start_pos)},
            "fk_target": {"left_tcp_m": _round_list(left_end_pos), "right_tcp_m": _round_list(right_end_pos)},
            "tangent_alignment_gate": tangent_gate,
        }
    )
    if tcp_translation > max_tcp_translation:
        packet["why_stopped"] = (
            f"post-lift j5 TCP translation {tcp_translation:.3f} m exceeds {max_tcp_translation:.3f} m"
        )
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet
    if other_tcp_motion > max_other_motion:
        packet["why_stopped"] = (
            f"other TCP moved {other_tcp_motion:.3f} m, above {max_other_motion:.3f} m"
        )
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet
    if combined_target_quat is not None and combined_target_rot_error > max_combined_rot_error:
        packet["why_stopped"] = (
            f"combined post-lift orientation error {combined_target_rot_error:.2f} deg exceeds "
            f"{max_combined_rot_error:.2f} deg"
        )
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet
    if tangent_gate.get("enabled") and not tangent_gate.get("passed"):
        packet["why_stopped"] = tangent_gate.get("why_stopped")
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    duration_s = max(0.05, _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_STEP_S", 0.80))
    packet["movement_command_invoked"] = True
    try:
        result = move_keypoints(
            timestamps=[0.0, duration_s],
            left_joint_positions=[left_base.tolist(), left_target.tolist()],
            right_joint_positions=[right_base.tolist(), right_target.tolist()],
            left_gripper_positions=[left_gripper, left_gripper],
            right_gripper_positions=[right_gripper, right_gripper],
            playback_speed=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_PLAYBACK_SPEED", 1.0),
            command_hz=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_COMMAND_HZ", 60.0),
            start_interp_s=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_START_INTERP_S", 0.25),
        )
    except Exception as exc:
        packet["physical_motion_executed"] = True
        packet["why_stopped"] = f"post-lift j5 keypoint command raised: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
            json_safe(packet),
        )
        return packet

    packet["physical_motion_executed"] = True
    packet["result"] = json_safe(result)
    packet["success"] = bool(result.get("success")) if isinstance(result, dict) else False
    packet["why_stopped"] = (
        "post-lift j5 reorientation executed"
        if packet["success"]
        else "post-lift j5 command returned unsuccessful result"
    )
    packet["path"] = write_json(
        run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_post_lift_j5_reorient_execute.json",
        json_safe(packet),
    )
    append_stage_summary(
        run_dir,
        [
            "## post-lift j5 reorientation execute",
            f"- success: {packet['success']}",
            f"- physical_motion_executed: {packet['physical_motion_executed']}",
            f"- joint_index: {joint_index}",
            f"- delta_deg: {packet['delta_deg']}",
            f"- rotated_rpy: {packet.get('rotated_rpy')}",
            f"- active_tcp_translation_m: {packet.get('active_tcp_translation_m')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _pre_insert_x_up_tilt_enabled() -> bool:
    return _truthy_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT", False)


def _execute_pre_insert_x_up_tilt(run_dir: Path, side: str) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.pre_insert_x_up_tilt.v1",
        "success": False,
        "side": side,
        "physical_motion_executed": False,
        "why_stopped": None,
    }
    if not _pre_insert_x_up_tilt_enabled():
        packet["success"] = True
        packet["skipped"] = True
        packet["why_stopped"] = "pre-insert X-up tilt disabled"
        return packet

    x_up_tilt_deg = _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT_DEG", 0.0)
    max_abs_deg = abs(_env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT_MAX_ABS_DEG", 20.0))
    if abs(x_up_tilt_deg) > max_abs_deg + 1e-9:
        packet["why_stopped"] = (
            f"pre-insert X-up tilt {x_up_tilt_deg:.2f} deg exceeds max {max_abs_deg:.2f} deg"
        )
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet
    if abs(x_up_tilt_deg) <= 1e-6:
        packet["success"] = True
        packet["skipped"] = True
        packet["why_stopped"] = "pre-insert X-up tilt is zero"
        return packet

    env = _tool_env("freespace_move") or _tool_env("get_robot_state")
    if env is None:
        packet["why_stopped"] = "cannot access direct YAM env for pre-insert X-up tilt"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet
    move_keypoints = getattr(env, "move_bimanual_joint_keypoints", None)
    if not callable(move_keypoints):
        packet["why_stopped"] = "direct YAM env does not expose move_bimanual_joint_keypoints"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    try:
        from robot.yam.kinematics import YamKinematics
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        packet["why_stopped"] = f"cannot load kinematics/rotation helpers for pre-insert X-up tilt: {exc}"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    try:
        left_obs = _read_direct_arm_obs(env, "left")
        right_obs = _read_direct_arm_obs(env, "right")
        left_base = np.asarray(left_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        right_base = np.asarray(right_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        left_gripper = float(np.asarray(left_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        right_gripper = float(np.asarray(right_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
    except Exception as exc:
        packet["why_stopped"] = f"cannot read direct arm observations for pre-insert tilt: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    kin = YamKinematics()
    try:
        left_pos, left_quat, right_pos, right_quat = kin.forward_kinematics(left_base, right_base)
    except Exception as exc:
        packet["why_stopped"] = f"forward kinematics failed before pre-insert tilt: {type(exc).__name__}: {exc}"
        packet["left_start_joint_pos"] = _round_list(left_base, limit=6)
        packet["right_start_joint_pos"] = _round_list(right_base, limit=6)
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    left_target_pos = np.asarray(left_pos, dtype=np.float64)
    right_target_pos = np.asarray(right_pos, dtype=np.float64)
    left_target_quat = np.asarray(left_quat, dtype=np.float64)
    right_target_quat = np.asarray(right_quat, dtype=np.float64)
    active_start_pos = left_target_pos if side == "left" else right_target_pos
    active_start_quat = left_target_quat if side == "left" else right_target_quat

    # Positive X-up tilt means the shelf-insertion (+X) side of the held dish
    # rises.  In world coordinates that is a negative rotation about Y.
    world_y_rotation_deg = -float(x_up_tilt_deg)
    target_quat = (
        Rotation.from_euler("y", world_y_rotation_deg, degrees=True)
        * Rotation.from_quat(np.asarray(active_start_quat, dtype=np.float64))
    ).as_quat()
    if side == "left":
        left_target_quat = target_quat
    elif side == "right":
        right_target_quat = target_quat
    else:
        packet["why_stopped"] = f"unsupported side for pre-insert X-up tilt: {side!r}"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    try:
        ik_left, ik_right = kin.inverse_kinematics(
            left_target_pos,
            left_target_quat,
            right_target_pos,
            right_target_quat,
            seeded=True,
            max_iters=max(1, _env_int("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_IK_MAX_ITERS", 120)),
            err_threshold=max(1e-6, _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_IK_ERR_THRESHOLD", 1e-4)),
        )
        end_left_pos, end_left_quat, end_right_pos, end_right_quat = kin.forward_kinematics(ik_left, ik_right)
    except Exception as exc:
        packet["why_stopped"] = f"IK/FK solve failed for pre-insert X-up tilt: {type(exc).__name__}: {exc}"
        packet["left_start_joint_pos"] = _round_list(left_base, limit=6)
        packet["right_start_joint_pos"] = _round_list(right_base, limit=6)
        packet["fk_start"] = {"left_tcp_m": _round_list(left_pos), "right_tcp_m": _round_list(right_pos)}
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    active_end_pos = np.asarray(end_left_pos if side == "left" else end_right_pos, dtype=np.float64)
    active_end_quat = np.asarray(end_left_quat if side == "left" else end_right_quat, dtype=np.float64)
    other_start_pos = np.asarray(right_pos if side == "left" else left_pos, dtype=np.float64)
    other_end_pos = np.asarray(end_right_pos if side == "left" else end_left_pos, dtype=np.float64)
    left_joint_delta = np.asarray(ik_left, dtype=np.float64) - left_base
    right_joint_delta = np.asarray(ik_right, dtype=np.float64) - right_base
    active_tcp_translation = float(np.linalg.norm(active_end_pos - np.asarray(active_start_pos, dtype=np.float64)))
    other_tcp_motion = float(np.linalg.norm(other_end_pos - other_start_pos))
    target_rot_error = _quat_angle_deg(active_end_quat, target_quat)
    achieved_rot_delta = _quat_angle_deg(active_start_quat, active_end_quat)
    requested_rot_delta = _quat_angle_deg(active_start_quat, target_quat)
    max_tcp_translation = _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_MAX_TCP_TRANSLATION_M", 0.030)
    max_other_motion = _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_MAX_OTHER_TCP_MOTION_M", 0.010)
    max_target_rot_error = _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_MAX_ROT_ERROR_DEG", 4.0)
    gate = {
        "passed": (
            active_tcp_translation <= max_tcp_translation
            and other_tcp_motion <= max_other_motion
            and target_rot_error <= max_target_rot_error
        ),
        "x_up_tilt_deg": round(x_up_tilt_deg, 5),
        "world_y_rotation_deg": round(world_y_rotation_deg, 5),
        "max_tcp_translation_m": round(max_tcp_translation, 5),
        "max_other_tcp_motion_m": round(max_other_motion, 5),
        "max_rot_error_deg": round(max_target_rot_error, 5),
        "observed_active_tcp_translation_m": round(active_tcp_translation, 5),
        "observed_other_tcp_motion_m": round(other_tcp_motion, 5),
        "observed_target_rot_error_deg": round(target_rot_error, 5),
        "observed_requested_rot_delta_deg": round(requested_rot_delta, 5),
        "observed_achieved_rot_delta_deg": round(achieved_rot_delta, 5),
        "reason": None,
    }
    if active_tcp_translation > max_tcp_translation:
        gate["reason"] = (
            f"pre-insert tilt TCP translation {active_tcp_translation:.3f} m exceeds "
            f"{max_tcp_translation:.3f} m"
        )
    elif other_tcp_motion > max_other_motion:
        gate["reason"] = f"other TCP moved {other_tcp_motion:.3f} m, above {max_other_motion:.3f} m"
    elif target_rot_error > max_target_rot_error:
        gate["reason"] = (
            f"pre-insert tilt rotation error {target_rot_error:.2f} deg exceeds "
            f"{max_target_rot_error:.2f} deg"
        )

    packet.update(
        {
            "gate": gate,
            "x_up_tilt_deg": round(x_up_tilt_deg, 5),
            "world_y_rotation_deg": round(world_y_rotation_deg, 5),
            "start_rpy": _round_list(_quat_xyzw_to_display_rpy(active_start_quat), limit=3),
            "target_rpy": _round_list(_quat_xyzw_to_display_rpy(target_quat), limit=3),
            "end_rpy": _round_list(_quat_xyzw_to_display_rpy(active_end_quat), limit=3),
            "left_start_joint_pos": _round_list(left_base, limit=6),
            "right_start_joint_pos": _round_list(right_base, limit=6),
            "left_target_joint_pos": _round_list(ik_left, limit=6),
            "right_target_joint_pos": _round_list(ik_right, limit=6),
            "left_joint_delta_rad": _round_list(left_joint_delta, limit=6),
            "right_joint_delta_rad": _round_list(right_joint_delta, limit=6),
            "left_gripper_pos": round(left_gripper, 5),
            "right_gripper_pos": round(right_gripper, 5),
            "fk_start": {"left_tcp_m": _round_list(left_pos), "right_tcp_m": _round_list(right_pos)},
            "fk_end": {"left_tcp_m": _round_list(end_left_pos), "right_tcp_m": _round_list(end_right_pos)},
        }
    )
    if not gate["passed"]:
        packet["why_stopped"] = f"pre-insert X-up tilt gate failed: {gate['reason']}"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    duration_s = max(0.05, _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_STEP_S", 0.80))
    packet["duration_s"] = round(duration_s, 5)
    packet["movement_command_invoked"] = True
    try:
        result = move_keypoints(
            timestamps=[0.0, duration_s],
            left_joint_positions=[left_base.tolist(), np.asarray(ik_left, dtype=np.float64).tolist()],
            right_joint_positions=[right_base.tolist(), np.asarray(ik_right, dtype=np.float64).tolist()],
            left_gripper_positions=[left_gripper, left_gripper],
            right_gripper_positions=[right_gripper, right_gripper],
            playback_speed=_env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_PLAYBACK_SPEED", 0.60),
            command_hz=_env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_COMMAND_HZ", 60.0),
            start_interp_s=_env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_START_INTERP_S", 0.60),
        )
    except Exception as exc:
        packet["physical_motion_executed"] = True
        packet["why_stopped"] = f"pre-insert tilt keypoint command raised: {type(exc).__name__}: {exc}"
        packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
        return packet

    packet["physical_motion_executed"] = True
    packet["result"] = json_safe(result)
    packet["success"] = bool(result.get("success")) if isinstance(result, dict) else False
    packet["why_stopped"] = (
        "pre-insert X-up tilt executed"
        if packet["success"]
        else "pre-insert X-up tilt command returned unsuccessful result"
    )
    packet["path"] = write_json(run_dir / "plans" / "lower_rack_pre_insert_x_up_tilt.json", json_safe(packet))
    append_stage_summary(
        run_dir,
        [
            "## lower_rack_pre_insert_x_up_tilt",
            f"- success: {packet['success']}",
            f"- physical_motion_executed: {packet['physical_motion_executed']}",
            f"- x_up_tilt_deg: {packet.get('x_up_tilt_deg')}",
            f"- world_y_rotation_deg: {packet.get('world_y_rotation_deg')}",
            f"- end_rpy: {packet.get('end_rpy')}",
            f"- gate: {gate['passed']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _record_recovery_packet(run_dir: Path, attempt: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    attempt.setdefault("recovery_attempts", []).append(packet)
    packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(packet.get('label') or packet.get('phase') or 'recovery')}.json", json_safe(packet))
    append_stage_summary(
        run_dir,
        [
            f"## recovery {packet.get('label') or packet.get('phase')}",
            f"- success: {packet.get('success')}",
            f"- reason: {packet.get('reason') or packet.get('why_stopped')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _attempt_planner_recovery(
    run_dir: Path,
    attempt: dict[str, Any],
    *,
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any] | None,
    label: str,
    gripper: float | None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.planner_recovery.v1",
        "label": label,
        "success": False,
        "side": side,
        "pose": json_safe(pose),
        "gripper": gripper,
        "physical_motion_executed": False,
        "reason": None,
    }
    if not _pose_available(pose):
        packet["reason"] = "recovery pose is unavailable"
        return _record_recovery_packet(run_dir, attempt, packet)
    try:
        step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
        packet["success"] = True
        packet["physical_motion_executed"] = True
        packet["step"] = json_safe(step)
        packet["reason"] = "planner recovery executed"
        attempt["movement_capable_calls"].append(f"freespace_move:{label}")
        attempt["steps"].append(step)
    except Exception as exc:
        packet["reason"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": label, "ok": False, "recovery": True, "error": packet["reason"]})
    return _record_recovery_packet(run_dir, attempt, packet)


def _attempt_open_recovery(
    run_dir: Path,
    attempt: dict[str, Any],
    *,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    side: str,
    label: str,
    target: float = 1.0,
    confirm_min: float | None = None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.gripper_recovery.v1",
        "label": label,
        "success": False,
        "side": side,
        "target": target,
        "physical_motion_executed": False,
        "reason": None,
    }
    try:
        step = _set_gripper_checked(
            side=side,
            target=target,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage=label,
            confirm_min=confirm_min,
            **_gripper_limit_kwargs(),
        )
        packet["success"] = bool(step.get("ok"))
        packet["physical_motion_executed"] = True
        packet["step"] = json_safe(step)
        packet["reason"] = "recovery gripper command completed" if packet["success"] else "recovery gripper command did not confirm"
        attempt["movement_capable_calls"].append(f"set_gripper:{side}:{label}")
        attempt["steps"].append(step)
    except Exception as exc:
        packet["reason"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": label, "ok": False, "recovery": True, "error": packet["reason"]})
    return _record_recovery_packet(run_dir, attempt, packet)


def _attempt_pre_lift_recovery(
    run_dir: Path,
    attempt: dict[str, Any],
    *,
    freespace_move: Callable[..., Any],
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    side: str,
    selected_pickup: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.pre_lift_recovery.v1",
        "phase": phase,
        "success": False,
        "attempts": [],
        "reason": None,
    }
    open_packet = _attempt_open_recovery(
        run_dir,
        attempt,
        set_gripper=set_gripper,
        get_robot_state=get_robot_state,
        side=side,
        label=f"recovery_{phase}_open",
        target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
        confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
    )
    retreat_packet = _attempt_planner_recovery(
        run_dir,
        attempt,
        freespace_move=freespace_move,
        side=side,
        pose=_pickup_retreat_pose(selected_pickup),
        label=f"recovery_{phase}_pickup_retreat",
        gripper=_preshape_gripper(selected_pickup),
    )
    packet["attempts"] = [open_packet, retreat_packet]
    packet["success"] = bool(open_packet.get("success") and retreat_packet.get("success"))
    packet["reason"] = "pre-lift recovery completed" if packet["success"] else "pre-lift recovery incomplete"
    return _record_recovery_packet(run_dir, attempt, packet)


def _attempt_source_return_recovery(
    run_dir: Path,
    attempt: dict[str, Any],
    *,
    freespace_move: Callable[..., Any],
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    side: str,
    selected_pickup: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.source_return_recovery.v1",
        "phase": phase,
        "success": False,
        "attempts": [],
        "reason": None,
    }
    hold = _hold_gripper(selected_pickup)
    to_source = _attempt_planner_recovery(
        run_dir,
        attempt,
        freespace_move=freespace_move,
        side=side,
        pose=selected_pickup.get("grasp_pose"),
        label=f"recovery_{phase}_source_grasp_pose",
        gripper=hold,
    )
    open_packet = _attempt_open_recovery(
        run_dir,
        attempt,
        set_gripper=set_gripper,
        get_robot_state=get_robot_state,
        side=side,
        label=f"recovery_{phase}_open_at_source",
        target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
        confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
    )
    retreat = _attempt_planner_recovery(
        run_dir,
        attempt,
        freespace_move=freespace_move,
        side=side,
        pose=_pickup_retreat_pose(selected_pickup),
        label=f"recovery_{phase}_pickup_retreat",
        gripper=_preshape_gripper(selected_pickup),
    )
    packet["attempts"] = [to_source, open_packet, retreat]
    packet["success"] = bool(to_source.get("success") and open_packet.get("success") and retreat.get("success"))
    packet["reason"] = "source-return recovery completed" if packet["success"] else "source-return recovery incomplete"
    return _record_recovery_packet(run_dir, attempt, packet)


def _attempt_rack_front_recovery(
    run_dir: Path,
    attempt: dict[str, Any],
    *,
    freespace_move: Callable[..., Any],
    side: str,
    selected_place: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.bowl_lower_rack.rack_front_recovery.v1",
        "phase": phase,
        "success": False,
        "attempts": [],
        "reason": None,
    }
    direct = _execute_direct_lower_rack_pose_adjust(
        run_dir,
        side,
        target_pose=selected_place.get("retreat_pose") or selected_place.get("front_entry_pose") or {},
        stage=f"recovery_{phase}_direct_horizontal_retreat",
        max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_RECOVERY_DIRECT_MAX_XY_DELTA_M", 0.220),
        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_RECOVERY_DIRECT_MAX_Z_DELTA_M", 0.060),
    )
    if direct.get("physical_motion_executed"):
        attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:recovery_{phase}_direct_horizontal_retreat:{side}")
    attempt["steps"].append(direct)
    packet["attempts"].append(direct)

    if direct.get("success"):
        high = _attempt_planner_recovery(
            run_dir,
            attempt,
            freespace_move=freespace_move,
            side=side,
            pose=selected_place.get("front_high_pose"),
            label=f"recovery_{phase}_front_high",
            gripper=None,
        )
        packet["attempts"].append(high)
        packet["success"] = bool(high.get("success"))
    else:
        front_entry = _attempt_planner_recovery(
            run_dir,
            attempt,
            freespace_move=freespace_move,
            side=side,
            pose=selected_place.get("front_entry_pose"),
            label=f"recovery_{phase}_front_entry",
            gripper=None,
        )
        packet["attempts"].append(front_entry)
        if front_entry.get("success"):
            front_high = _attempt_planner_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                side=side,
                pose=selected_place.get("front_high_pose"),
                label=f"recovery_{phase}_front_high",
                gripper=None,
            )
            packet["attempts"].append(front_high)
            packet["success"] = bool(front_high.get("success"))
    packet["reason"] = "rack-front recovery completed" if packet["success"] else "rack-front recovery incomplete; operator may need to intervene"
    return _record_recovery_packet(run_dir, attempt, packet)


def _execute_physical(run_dir: Path, selected_pickup: dict[str, Any], selected_place: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    side = str(selected_pickup["arm"])
    close_plan = selected_pickup.get("staged_close_plan") or {}
    attempt: dict[str, Any] = {
        "schema": "openforge.bowl_rack.physical_attempt.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "selected_place": selected_place,
        "steps": [],
        "gripper_after_close": None,
        "why_stopped": None,
    }
    try:
        gripper_limit_kwargs = _gripper_limit_kwargs()
        pregrasp_open = _preshape_gripper(selected_pickup)
        hold = _hold_gripper(selected_pickup)
        open_step = _set_gripper_checked(
            side=side,
            target=pregrasp_open,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="checked_preshape_open",
            confirm_min=_env_float("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(open_step)
        if not open_step["ok"]:
            attempt["why_stopped"] = "gripper did not confirm open before arm motion"
            return attempt

        before_close_specs, after_close_specs = _split_physical_sequence(selected_pickup, selected_place)

        for label, pose, gripper in before_close_specs:
            try:
                step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append(step)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempt["why_stopped"] = f"{label} motion failed before close: {error}"
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append({"stage": label, "ok": False, "error": error})
                attempt["recovery"] = _attempt_pre_lift_recovery(
                    run_dir,
                    attempt,
                    freespace_move=freespace_move,
                    set_gripper=set_gripper,
                    get_robot_state=get_robot_state,
                    side=side,
                    selected_pickup=selected_pickup,
                    phase=f"{label}_motion_failed",
                )
                return attempt

        pre_close_gate_enabled = _truthy_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", True)
        attempt["pre_close_observation"] = _capture_intermediate_observation(
            run_dir,
            "pre_close_grasp_pose_observe",
            env_name="OPENFORGE_BOWL_RACK_CAPTURE_PRE_CLOSE_OBSERVATION",
            force=pre_close_gate_enabled,
        )
        if attempt["pre_close_observation"].get("skipped"):
            pre_close_gate = {
                "schema": "openforge.bowl_rack.pre_close_visual_gate.v1",
                "passed": True,
                "skipped": True,
                "reason": "pre-close visual gate disabled and intermediate observation skipped",
            }
        else:
            pre_close_gate = _pre_close_visual_gate(run_dir, selected_pickup, attempt["pre_close_observation"])
        attempt["pre_close_visual_gate"] = pre_close_gate
        if pre_close_gate_enabled and not bool(pre_close_gate.get("passed")):
            attempt["why_stopped"] = "pre-close visual geometry gate failed; stop before close/lift/place"
            attempt["recovery"] = _attempt_pre_lift_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                side=side,
                selected_pickup=selected_pickup,
                phase="pre_close_gate_failed",
            )
            return attempt
        close = staged_close_with_contact(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=hold,
            steps=_csv_float("OPENFORGE_BOWL_RACK_CLOSE_STEPS", list(close_plan.get("steps") or [0.68, 0.56, 0.48, 0.40])),
            **gripper_limit_kwargs,
            min_contact_delta=_env_float(
                "OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS",
                float(close_plan.get("expected_contact_delta_min", 0.010)),
            ),
            target_tolerance=_env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_BOWL_RACK_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_BOWL_RACK_HOLD_MAX_POS", 0.95),
            confirm_timeout_s=_env_float(
                "OPENFORGE_BOWL_RACK_CLOSE_CONFIRM_TIMEOUT_S",
                _env_float("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_TIMEOUT_S", 0.0),
            ),
            state_poll_s=_env_float("OPENFORGE_BOWL_RACK_GRIPPER_STATE_POLL_S", 0.05),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage=f"{TASK_ARTIFACT_PREFIX}_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["gripper_after_close"] = close
        attempt["post_close_observation"] = _capture_intermediate_observation(
            run_dir,
            "post_close_before_lift_observe",
            env_name="OPENFORGE_BOWL_RACK_CAPTURE_POST_CLOSE_OBSERVATION",
        )
        require_contact = _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True)
        if require_contact and not bool(close.get("contact_detected")):
            attempt["why_stopped"] = "staged close did not produce contact evidence; stop before lift/place"
            attempt["recovery"] = _attempt_pre_lift_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                side=side,
                selected_pickup=selected_pickup,
                phase="no_close_contact",
            )
            return attempt
        if not bool(close.get("plausible_for_lift")):
            attempt["why_stopped"] = "staged close did not produce plausible gripper-state evidence; stop before lift/place"
            attempt["recovery"] = _attempt_pre_lift_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                side=side,
                selected_pickup=selected_pickup,
                phase="implausible_hold",
            )
            return attempt

        lift_spec_count = 0
        for label, pose, gripper in after_close_specs:
            if not _is_pickup_lift_label(label):
                break
            step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)
            lift_spec_count += 1

        require_lift_perception = _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_LIFT_PERCEPTION_BEFORE_PLACE", False)
        post_lift = _capture_intermediate_observation(
            run_dir,
            "post_lift_before_rack_transport_observe",
            env_name="OPENFORGE_BOWL_RACK_CAPTURE_POST_LIFT_OBSERVATION",
            force=require_lift_perception,
        )
        attempt["post_lift_observation"] = post_lift
        if post_lift.get("skipped"):
            attempt["lift_verification"] = {
                "success": True,
                "skipped": True,
                "reason": "lift perception verification disabled before rack transport",
            }
        else:
            post_lift_det = _select_object_detection(post_lift)
            attempt["lift_verification"] = verify_lift(
                selected_pickup.get("source_detection"),
                post_lift_det,
                min_z_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MIN_Z_DELTA_M", 0.035),
                max_xy_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MAX_XY_DELTA_M", 0.24),
            )
        if require_lift_perception and not bool(attempt["lift_verification"].get("success")):
            attempt["why_stopped"] = "lift perception verification failed; stop before rack transport by configured policy"
            attempt["recovery"] = _attempt_source_return_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                side=side,
                selected_pickup=selected_pickup,
                phase="lift_verify_failed",
            )
            return attempt

        if _post_lift_j5_reorient_enabled():
            planned_reorient = selected_place.get("post_lift_reorientation") or {}
            planned_delta = planned_reorient.get("delta_deg")
            planned_joint_index = planned_reorient.get("joint_index")
            reorient = _execute_post_lift_j5_reorientation(
                run_dir,
                side,
                delta_deg=float(planned_delta) if planned_delta is not None else None,
                joint_index=int(planned_joint_index) if planned_joint_index is not None else None,
                selected_pickup=selected_pickup,
            )
            attempt["post_lift_j5_reorientation"] = reorient
            attempt["steps"].append(reorient)
            if reorient.get("physical_motion_executed"):
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:post_lift_j5_reorient:{side}")
            if not reorient.get("success"):
                attempt["why_stopped"] = f"post-lift j5 reorientation failed: {reorient.get('why_stopped')}"
                attempt["recovery"] = _attempt_source_return_recovery(
                    run_dir,
                    attempt,
                    freespace_move=freespace_move,
                    set_gripper=set_gripper,
                    get_robot_state=get_robot_state,
                    side=side,
                    selected_pickup=selected_pickup,
                    phase="post_lift_j5_reorient_failed",
                )
                return attempt
            rotated_rpy = list(reorient.get("rotated_rpy") or [])
            if len(rotated_rpy) < 3:
                attempt["why_stopped"] = "post-lift j5 reorientation did not produce rotated rpy"
                attempt["recovery"] = _attempt_source_return_recovery(
                    run_dir,
                    attempt,
                    freespace_move=freespace_move,
                    set_gripper=set_gripper,
                    get_robot_state=get_robot_state,
                    side=side,
                    selected_pickup=selected_pickup,
                    phase="post_lift_j5_missing_rpy",
                )
                return attempt
            selected_place = _patch_place_candidate_after_reorientation(
                selected_place,
                selected_pickup,
                rotated_rpy,
                source="physical_post_lift_j5_fk",
                reorientation=reorient,
            )
            attempt["selected_place"] = selected_place
            _, after_close_specs = _split_physical_sequence(selected_pickup, selected_place)
            after_close_specs = [spec for spec in after_close_specs if not _is_pickup_lift_label(spec[0])]
        else:
            after_close_specs = after_close_specs[lift_spec_count:]

        for label, pose, gripper in after_close_specs:
            try:
                step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append(step)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempt["why_stopped"] = f"{label} motion failed after lift: {error}"
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append({"stage": label, "ok": False, "error": error})
                attempt["recovery"] = _attempt_source_return_recovery(
                    run_dir,
                    attempt,
                    freespace_move=freespace_move,
                    set_gripper=set_gripper,
                    get_robot_state=get_robot_state,
                    side=side,
                    selected_pickup=selected_pickup,
                    phase=f"{label}_motion_failed",
                )
                return attempt

        if _split_direct_insert_enabled():
            direct_insert: dict[str, Any] = {
                "schema": "openforge.bowl_lower_rack.split_direct_insert.v1",
                "success": False,
                "stage": "lower_rack_split_direct_insert",
                "side": side,
                "policy": (
                    "After lift/J5, skip planner front_high/front_entry transport. Optionally tilt the held dish X-up, "
                    "optionally lift in Z, align Y once, insert in X, place down only in negative Z, then release."
                ),
                "steps": [],
                "final_axes": ["x"],
                "place_down_z_enabled": _direct_place_down_enabled(),
                "pre_insert_tilt_combined_with_post_lift": _pre_insert_tilt_combined_with_post_lift(),
            }
            if _pre_insert_tilt_combined_with_post_lift():
                pre_insert_tilt = {
                    "schema": "openforge.bowl_lower_rack.pre_insert_x_up_tilt.v1",
                    "success": True,
                    "skipped": True,
                    "stage": "lower_rack_pre_insert_x_up_tilt",
                    "side": side,
                    "physical_motion_executed": False,
                    "why_stopped": "pre-insert X-up tilt was combined into post-lift orientation move",
                }
            else:
                pre_insert_tilt = _execute_pre_insert_x_up_tilt(run_dir, side)
            if pre_insert_tilt:
                direct_insert["steps"].append(pre_insert_tilt)
                if pre_insert_tilt.get("physical_motion_executed"):
                    attempt["movement_capable_calls"].append(
                        f"move_bimanual_joint_keypoints:lower_rack_pre_insert_x_up_tilt:{side}"
                    )
                attempt["steps"].append(pre_insert_tilt)

            pre_insert_z_lift: dict[str, Any] = {}
            pre_insert_z_lift_m = max(
                0.0,
                _env_float("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_Z_LIFT_M", 0.0),
            )
            if pre_insert_tilt.get("success") and pre_insert_z_lift_m > 0.0:
                try:
                    pre_insert_z_lift_pose = _current_active_tcp_pose_with_z_offset(
                        side,
                        z_offset_m=pre_insert_z_lift_m,
                    )
                    pre_insert_z_lift = _execute_direct_lower_rack_pose_adjust(
                        run_dir,
                        side,
                        target_pose=pre_insert_z_lift_pose,
                        stage="lower_rack_direct_pre_insert_z_lift",
                        max_xy_delta_m=0.010,
                        max_z_delta_m=_env_float(
                            "OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_Z_LIFT_MAX_Z_DELTA_M",
                            pre_insert_z_lift_m + 0.020,
                        ),
                        target_axes={"z"},
                        step_s_env="OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_Z_LIFT_STEP_S",
                        z_direction="up_only",
                    )
                except Exception as exc:
                    pre_insert_z_lift = {
                        "schema": "openforge.bowl_lower_rack.pre_insert_z_lift.v1",
                        "success": False,
                        "stage": "lower_rack_direct_pre_insert_z_lift",
                        "side": side,
                        "physical_motion_executed": False,
                        "why_stopped": f"pre-insert Z lift failed before motion: {type(exc).__name__}: {exc}",
                    }
                    pre_insert_z_lift["path"] = write_json(
                        run_dir / "plans" / "lower_rack_direct_pre_insert_z_lift.json",
                        json_safe(pre_insert_z_lift),
                    )
                direct_insert["steps"].append(pre_insert_z_lift)
                if pre_insert_z_lift.get("physical_motion_executed"):
                    attempt["movement_capable_calls"].append(
                        f"move_bimanual_joint_keypoints:lower_rack_direct_pre_insert_z_lift:{side}"
                    )
                attempt["steps"].append(pre_insert_z_lift)

            y_align: dict[str, Any] = {}
            if pre_insert_tilt.get("success") and (
                pre_insert_z_lift_m <= 0.0 or pre_insert_z_lift.get("success")
            ):
                y_align = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=selected_place["release_pose"],
                    stage="lower_rack_direct_y_align",
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_Y_ALIGN_MAX_XY_DELTA_M", 0.150),
                    max_z_delta_m=0.010,
                    target_axes={"y"},
                    max_axis_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_Y_ALIGN_MAX_XY_DELTA_M", 0.150),
                )
                direct_insert["steps"].append(y_align)
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_direct_y_align:{side}")
                attempt["steps"].append(y_align)
            if y_align.get("success"):
                x_insert = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=selected_place["release_pose"],
                    stage="lower_rack_direct_x_insert",
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_X_MAX_XY_DELTA_M", 0.120),
                    max_z_delta_m=0.010,
                    target_axes={"x"},
                )
            else:
                x_insert = {}
            if x_insert:
                direct_insert["steps"].append(x_insert)
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_direct_x_insert:{side}")
                attempt["steps"].append(x_insert)
            if x_insert.get("success") and y_align.get("success"):
                if _direct_place_down_enabled():
                    place_down_pose = _split_direct_place_down_pose(selected_place)
                    z_place = _execute_direct_lower_rack_pose_adjust(
                        run_dir,
                        side,
                        target_pose=place_down_pose,
                        stage="lower_rack_direct_z_place_down",
                        max_xy_delta_m=0.010,
                        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_MAX_Z_DELTA_M", 0.050),
                        target_axes={"z"},
                        step_s_env="OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_STEP_S",
                        z_direction="down_only",
                    )
                    direct_insert["steps"].append(z_place)
                    direct_insert["place_down_pose"] = json_safe(place_down_pose)
                    attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_direct_z_place_down:{side}")
                    attempt["steps"].append(z_place)
                    direct_insert["success"] = bool(z_place.get("success"))
                    direct_insert["why_stopped"] = z_place.get("why_stopped")
                else:
                    direct_insert["success"] = True
                    direct_insert["why_stopped"] = y_align.get("why_stopped")
            else:
                failed = (
                    x_insert
                    if y_align.get("success")
                    else (
                        y_align
                        if y_align
                        else (
                            pre_insert_z_lift
                            if pre_insert_z_lift_m > 0.0 and pre_insert_tilt.get("success")
                            else pre_insert_tilt
                        )
                    )
                )
                direct_insert["why_stopped"] = failed.get("why_stopped")
            direct_insert["physical_motion_executed"] = any(bool(step.get("physical_motion_executed")) for step in direct_insert["steps"])
            direct_insert["path"] = write_json(
                run_dir / "plans" / "lower_rack_split_direct_insert.json",
                json_safe(direct_insert),
            )
        else:
            open_front_slide = _open_front_slide_enabled()
            if open_front_slide:
                y_align_pose = _open_front_slide_y_align_pose(selected_place)
                direct_insert = {
                    "schema": "openforge.bowl_lower_rack.open_front_slide_insert.v1",
                    "success": False,
                    "stage": "lower_rack_open_front_slide_insert",
                    "side": side,
                    "policy": (
                        "From the open-side front-entry pose, align in shelf Y first, then slide in along shelf X. "
                        "This avoids a diagonal insertion under the upper shelf."
                    ),
                    "steps": [],
                }
                y_align = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=y_align_pose,
                    stage="lower_rack_open_front_slide_y_align",
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_ALIGN_MAX_XY_DELTA_M", 0.220),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                    target_axes={"y"},
                    step_s_env="OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_ALIGN_STEP_S",
                )
                direct_insert["steps"].append(y_align)
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_y_align:{side}")
                attempt["steps"].append(y_align)
                if y_align.get("success"):
                    x_insert = _execute_direct_lower_rack_pose_adjust(
                        run_dir,
                        side,
                        target_pose=selected_place["release_pose"],
                        stage="lower_rack_open_front_slide_x_insert",
                        max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_INSERT_MAX_XY_DELTA_M", 0.220),
                        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                        target_axes={"x"},
                        step_s_env="OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_INSERT_STEP_S",
                    )
                else:
                    x_insert = {}
                if x_insert:
                    direct_insert["steps"].append(x_insert)
                    attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_x_insert:{side}")
                    attempt["steps"].append(x_insert)
                if y_align.get("success") and x_insert.get("success"):
                    if _direct_place_down_enabled():
                        place_down_pose = _split_direct_place_down_pose(selected_place)
                        z_place = _execute_direct_lower_rack_pose_adjust(
                            run_dir,
                            side,
                            target_pose=place_down_pose,
                            stage="lower_rack_open_front_slide_z_place_down",
                            max_xy_delta_m=0.010,
                            max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_MAX_Z_DELTA_M", 0.050),
                            target_axes={"z"},
                            step_s_env="OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_STEP_S",
                            z_direction="down_only",
                        )
                        direct_insert["steps"].append(z_place)
                        direct_insert["place_down_pose"] = json_safe(place_down_pose)
                        attempt["movement_capable_calls"].append(
                            f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_z_place_down:{side}"
                        )
                        attempt["steps"].append(z_place)
                        direct_insert["success"] = bool(z_place.get("success"))
                        direct_insert["why_stopped"] = (
                            "open-front slide Y-align, X-insert, then Z place-down executed"
                            if direct_insert["success"]
                            else z_place.get("why_stopped")
                        )
                    else:
                        direct_insert["success"] = True
                        direct_insert["why_stopped"] = "open-front slide Y-align then X-insert executed"
                else:
                    failed = x_insert if y_align.get("success") else y_align
                    direct_insert["success"] = False
                    direct_insert["why_stopped"] = failed.get("why_stopped")
                direct_insert["physical_motion_executed"] = any(
                    bool(step.get("physical_motion_executed")) for step in direct_insert["steps"]
                )
                direct_insert["path"] = write_json(
                    run_dir / "plans" / "lower_rack_open_front_slide_insert.json",
                    json_safe(direct_insert),
                )
            else:
                direct_target_axes = {"z"} if _open_front_rack_enabled() else None
                direct_insert = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=selected_place["release_pose"],
                    stage="lower_rack_direct_insert",
                    max_xy_delta_m=(
                        _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_DIRECT_MAX_XY_DELTA_M", 0.030)
                        if _open_front_rack_enabled()
                        else _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_XY_DELTA_M", 0.150)
                    ),
                    max_z_delta_m=(
                        _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_MAX_Z_DELTA_M", 0.120)
                        if _open_front_rack_enabled()
                        else _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_Z_DELTA_M", 0.050)
                    ),
                    target_axes=direct_target_axes,
                    step_s_env=(
                        "OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_STEP_S"
                        if _open_front_rack_enabled()
                        else "OPENFORGE_BOWL_LOWER_RACK_DIRECT_STEP_S"
                    ),
                )
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_direct_insert:{side}")
                attempt["steps"].append(direct_insert)
        attempt["direct_insert"] = direct_insert
        if not direct_insert.get("success"):
            attempt["why_stopped"] = f"lower-rack direct insertion failed: {direct_insert.get('why_stopped')}"
            attempt["recovery"] = _attempt_rack_front_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                side=side,
                selected_place=selected_place,
                phase="direct_insert_failed",
            )
            return attempt

        attempt["pre_release_observation"] = _capture_intermediate_observation(
            run_dir,
            "pre_release_lower_rack_observe",
            env_name="OPENFORGE_BOWL_RACK_CAPTURE_PRE_RELEASE_OBSERVATION",
        )

        release = _set_gripper_checked(
            side=side,
            target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="open_release_on_rack",
            confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(release)
        if not release["ok"]:
            attempt["why_stopped"] = "release gripper did not confirm open"
            attempt["recovery"] = _attempt_rack_front_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                side=side,
                selected_place=selected_place,
                phase="release_failed",
            )
            return attempt

        split_direct_insert = _split_direct_insert_enabled()
        open_front_slide = _open_front_slide_enabled()
        if split_direct_insert:
            clear_pose = _split_direct_post_release_clear_pose(selected_place)
            x_retreat_pose = _split_direct_post_release_x_retreat_pose(selected_place)
            diagonal_retreat_after_release = _diagonal_retreat_after_release_enabled()
            diagonal_retreat_pose = (
                _split_direct_post_release_diagonal_retreat_pose(selected_place)
                if diagonal_retreat_after_release
                else None
            )
            skip_x_retreat_after_clear = _truthy_env(
                "OPENFORGE_BOWL_LOWER_RACK_SKIP_X_RETREAT_AFTER_RELEASE_CLEAR",
                False,
            )
            direct_retreat = {
                "schema": "openforge.bowl_lower_rack.split_direct_clear_retreat.v1",
                "success": False,
                "stage": "lower_rack_split_direct_clear_retreat",
                "side": side,
                "policy": (
                    "After release, either move diagonally up/back in X-Z, or move up in Z "
                    "and then retreat in -X or skip straight to the post-success home sequence."
                ),
                "steps": [],
                "clear_pose": json_safe(clear_pose),
                "x_retreat_pose": json_safe(x_retreat_pose),
                "diagonal_retreat_pose": json_safe(diagonal_retreat_pose),
                "diagonal_retreat_after_release": diagonal_retreat_after_release,
                "skip_x_retreat_after_clear": skip_x_retreat_after_clear,
            }
            if diagonal_retreat_after_release:
                if diagonal_retreat_pose is None:
                    raise RuntimeError("diagonal retreat enabled without a target pose")
                diagonal_model = diagonal_retreat_pose.get("diagonal_retreat_model") or {}
                diagonal_x_offset = float(diagonal_model.get("x_sign", -1.0) or -1.0) * float(
                    diagonal_model.get("x_delta_m", 0.0) or 0.0
                )
                diagonal_z_offset = float(diagonal_model.get("z_lift_m", 0.0) or 0.0)
                live_diagonal_retreat_pose = _current_active_tcp_pose_with_offsets(
                    side,
                    x_offset_m=diagonal_x_offset,
                    z_offset_m=diagonal_z_offset,
                )
                live_diagonal_retreat_pose["diagonal_retreat_model"] = {
                    **diagonal_model,
                    "target_source": "live_active_tcp_at_release",
                    "commanded_x_offset_m": diagonal_x_offset,
                    "commanded_z_offset_m": diagonal_z_offset,
                }
                direct_retreat["live_diagonal_retreat_pose"] = json_safe(live_diagonal_retreat_pose)
                diagonal_retreat = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=live_diagonal_retreat_pose,
                    stage="lower_rack_direct_diagonal_xz_retreat_after_release",
                    max_xy_delta_m=_env_float(
                        "OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_XY_DELTA_M",
                        _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", 0.160),
                    ),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_Z_DELTA_M", 0.100),
                    target_axes={"x", "z"},
                    step_s_env="OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_STEP_S",
                    z_direction="up_only",
                )
                direct_retreat["steps"].append(diagonal_retreat)
                attempt["movement_capable_calls"].append(
                    f"move_bimanual_joint_keypoints:lower_rack_direct_diagonal_xz_retreat_after_release:{side}"
                )
                attempt["steps"].append(diagonal_retreat)
                direct_retreat["success"] = bool(diagonal_retreat.get("success"))
                direct_retreat["why_stopped"] = (
                    "split direct diagonal X-Z retreat executed"
                    if direct_retreat["success"]
                    else diagonal_retreat.get("why_stopped")
                )
            else:
                z_clear = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=clear_pose,
                    stage="lower_rack_direct_z_clear_after_release",
                    max_xy_delta_m=0.010,
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_Z_DELTA_M", 0.100),
                    target_axes={"z"},
                    step_s_env="OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_STEP_S",
                    z_direction="up_only",
                )
                direct_retreat["steps"].append(z_clear)
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_direct_z_clear_after_release:{side}")
                attempt["steps"].append(z_clear)
                if z_clear.get("success") and skip_x_retreat_after_clear:
                    x_retreat = {
                        "schema": "openforge.bowl_lower_rack.direct_pose_adjust.v1",
                        "success": True,
                        "skipped": True,
                        "stage": "lower_rack_direct_x_retreat_after_clear",
                        "side": side,
                        "physical_motion_executed": False,
                        "target_pose": json_safe(x_retreat_pose),
                        "target_axes": ["x"],
                        "why_stopped": (
                            "skipped configured X retreat after release clear; post-success home will clear the arm"
                        ),
                    }
                    x_retreat["path"] = write_json(
                        run_dir / "plans" / "lower_rack_direct_x_retreat_after_clear.json",
                        json_safe(x_retreat),
                    )
                elif z_clear.get("success"):
                    x_retreat = _execute_direct_lower_rack_pose_adjust(
                        run_dir,
                        side,
                        target_pose=x_retreat_pose,
                        stage="lower_rack_direct_x_retreat_after_clear",
                        max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", 0.160),
                        max_z_delta_m=0.010,
                        target_axes={"x"},
                        step_s_env="OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_STEP_S",
                    )
                else:
                    x_retreat = {}
                if x_retreat:
                    direct_retreat["steps"].append(x_retreat)
                    if x_retreat.get("physical_motion_executed"):
                        attempt["movement_capable_calls"].append(
                            f"move_bimanual_joint_keypoints:lower_rack_direct_x_retreat_after_clear:{side}"
                        )
                    attempt["steps"].append(x_retreat)
                direct_retreat["success"] = bool(z_clear.get("success")) and bool(x_retreat.get("success"))
                failed = x_retreat if z_clear.get("success") else z_clear
                direct_retreat["why_stopped"] = (
                    (
                        "split direct Z-clear executed; X-retreat skipped before post-success home"
                        if skip_x_retreat_after_clear
                        else "split direct Z-clear then X-retreat executed"
                    )
                    if direct_retreat["success"]
                    else failed.get("why_stopped")
                )
            direct_retreat["physical_motion_executed"] = any(
                bool(step.get("physical_motion_executed")) for step in direct_retreat["steps"]
            )
            direct_retreat["path"] = write_json(
                run_dir / "plans" / "lower_rack_split_direct_clear_retreat.json",
                json_safe(direct_retreat),
            )
        elif open_front_slide:
            x_retreat_pose = _open_front_slide_x_retreat_pose(selected_place)
            diagonal_retreat_after_release = _diagonal_retreat_after_release_enabled()
            diagonal_retreat_pose = (
                _split_direct_post_release_diagonal_retreat_pose(selected_place)
                if diagonal_retreat_after_release
                else None
            )
            direct_retreat = {
                "schema": "openforge.bowl_lower_rack.open_front_slide_retreat.v1",
                "success": False,
                "stage": "lower_rack_open_front_slide_retreat",
                "side": side,
                "policy": (
                    "After release, either move straight up in Z, move diagonally up/back "
                    "in X-Z, or slide out along X first, then clear in Y."
                ),
                "steps": [],
                "diagonal_retreat_after_release": diagonal_retreat_after_release,
                "diagonal_retreat_pose": json_safe(diagonal_retreat_pose),
            }
            vertical_clear_after_release = _truthy_env(
                "OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_VERTICAL_CLEAR_AFTER_RELEASE",
                False,
            )
            skip_xy_after_vertical_clear = _truthy_env(
                "OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_SKIP_XY_RETREAT_AFTER_VERTICAL_CLEAR",
                False,
            )
            y_retreat_after_vertical_clear = _truthy_env(
                "OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_AFTER_VERTICAL_CLEAR",
                False,
            )
            direct_retreat["vertical_clear_after_release"] = vertical_clear_after_release
            direct_retreat["skip_xy_after_vertical_clear"] = skip_xy_after_vertical_clear
            direct_retreat["y_retreat_after_vertical_clear"] = y_retreat_after_vertical_clear
            if vertical_clear_after_release:
                clear_lift = max(0.0, _env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_Z_LIFT_M", 0.070))
                live_z_clear_pose = _current_active_tcp_pose_with_offsets(side, z_offset_m=clear_lift)
                live_z_clear_pose["vertical_clear_model"] = {
                    "target_source": "live_active_tcp_at_release",
                    "commanded_z_offset_m": clear_lift,
                    "skip_xy_after_vertical_clear": skip_xy_after_vertical_clear,
                    "y_retreat_after_vertical_clear": y_retreat_after_vertical_clear,
                }
                direct_retreat["live_vertical_clear_pose"] = json_safe(live_z_clear_pose)
                z_clear = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=live_z_clear_pose,
                    stage="lower_rack_open_front_slide_z_clear_after_release",
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_XY_DRIFT_M", 0.035),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_Z_DELTA_M", 0.100),
                    target_axes={"z"},
                    step_s_env="OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_STEP_S",
                    z_direction="up_only",
                )
                direct_retreat["steps"].append(z_clear)
                attempt["movement_capable_calls"].append(
                    f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_z_clear_after_release:{side}"
                )
                attempt["steps"].append(z_clear)
                if skip_xy_after_vertical_clear:
                    direct_retreat["success"] = bool(z_clear.get("success"))
                    direct_retreat["why_stopped"] = (
                        "open-front slide Z-clear executed; XY retreat skipped before post-success home"
                        if direct_retreat["success"]
                        else z_clear.get("why_stopped")
                    )
                elif not z_clear.get("success"):
                    direct_retreat["success"] = False
                    direct_retreat["why_stopped"] = z_clear.get("why_stopped")
                else:
                    x_retreat = _execute_direct_lower_rack_pose_adjust(
                        run_dir,
                        side,
                        target_pose=x_retreat_pose,
                        stage="lower_rack_open_front_slide_x_retreat_after_z_clear",
                        max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_MAX_XY_DELTA_M", 0.220),
                        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                        target_axes={"x"},
                        step_s_env="OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_STEP_S",
                    )
                    direct_retreat["steps"].append(x_retreat)
                    attempt["movement_capable_calls"].append(
                        f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_x_retreat_after_z_clear:{side}"
                    )
                    attempt["steps"].append(x_retreat)
                    if x_retreat.get("success") and y_retreat_after_vertical_clear:
                        y_retreat = _execute_direct_lower_rack_pose_adjust(
                            run_dir,
                            side,
                            target_pose=selected_place["retreat_pose"],
                            stage="lower_rack_open_front_slide_y_retreat_after_z_clear",
                            max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_MAX_XY_DELTA_M", 0.220),
                            max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                            target_axes={"y"},
                            step_s_env="OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_STEP_S",
                        )
                    else:
                        y_retreat = {}
                    if y_retreat:
                        direct_retreat["steps"].append(y_retreat)
                        attempt["movement_capable_calls"].append(
                            f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_y_retreat_after_z_clear:{side}"
                        )
                        attempt["steps"].append(y_retreat)
                    direct_retreat["success"] = bool(x_retreat.get("success")) and (
                        bool(y_retreat.get("success")) if y_retreat_after_vertical_clear else True
                    )
                    failed = y_retreat if x_retreat.get("success") and y_retreat_after_vertical_clear else x_retreat
                    direct_retreat["why_stopped"] = (
                        (
                            "open-front slide Z-clear, X-retreat, then Y-clear executed"
                            if y_retreat_after_vertical_clear
                            else "open-front slide Z-clear then X-retreat executed"
                        )
                        if direct_retreat["success"]
                        else failed.get("why_stopped")
                    )

            if direct_retreat.get("success") or (direct_retreat.get("steps") and direct_retreat.get("why_stopped")):
                pass
            elif diagonal_retreat_after_release:
                if diagonal_retreat_pose is None:
                    raise RuntimeError("diagonal retreat enabled without a target pose")
                diagonal_model = diagonal_retreat_pose.get("diagonal_retreat_model") or {}
                diagonal_x_offset = float(diagonal_model.get("x_sign", -1.0) or -1.0) * float(
                    diagonal_model.get("x_delta_m", 0.0) or 0.0
                )
                diagonal_z_offset = float(diagonal_model.get("z_lift_m", 0.0) or 0.0)
                live_diagonal_retreat_pose = _current_active_tcp_pose_with_offsets(
                    side,
                    x_offset_m=diagonal_x_offset,
                    z_offset_m=diagonal_z_offset,
                )
                live_diagonal_retreat_pose["diagonal_retreat_model"] = {
                    **diagonal_model,
                    "target_source": "live_active_tcp_at_release",
                    "commanded_x_offset_m": diagonal_x_offset,
                    "commanded_z_offset_m": diagonal_z_offset,
                }
                direct_retreat["live_diagonal_retreat_pose"] = json_safe(live_diagonal_retreat_pose)
                diagonal_retreat = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=live_diagonal_retreat_pose,
                    stage="lower_rack_open_front_slide_diagonal_xz_retreat_after_release",
                    max_xy_delta_m=_env_float(
                        "OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_XY_DELTA_M",
                        _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_MAX_XY_DELTA_M", 0.220),
                    ),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_Z_DELTA_M", 0.100),
                    target_axes={"x", "z"},
                    step_s_env="OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_STEP_S",
                    z_direction="up_only",
                )
                direct_retreat["steps"].append(diagonal_retreat)
                attempt["movement_capable_calls"].append(
                    f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_diagonal_xz_retreat_after_release:{side}"
                )
                attempt["steps"].append(diagonal_retreat)
                direct_retreat["success"] = bool(diagonal_retreat.get("success"))
                direct_retreat["why_stopped"] = (
                    "open-front slide diagonal X-Z retreat executed"
                    if direct_retreat["success"]
                    else diagonal_retreat.get("why_stopped")
                )
            else:
                x_retreat = _execute_direct_lower_rack_pose_adjust(
                    run_dir,
                    side,
                    target_pose=x_retreat_pose,
                    stage="lower_rack_open_front_slide_x_retreat",
                    max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_MAX_XY_DELTA_M", 0.220),
                    max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                    target_axes={"x"},
                    step_s_env="OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_X_RETREAT_STEP_S",
                )
                direct_retreat["steps"].append(x_retreat)
                attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_x_retreat:{side}")
                attempt["steps"].append(x_retreat)
                if x_retreat.get("success"):
                    y_retreat = _execute_direct_lower_rack_pose_adjust(
                        run_dir,
                        side,
                        target_pose=selected_place["retreat_pose"],
                        stage="lower_rack_open_front_slide_y_retreat",
                        max_xy_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_MAX_XY_DELTA_M", 0.220),
                        max_z_delta_m=_env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_MAX_Z_DELTA_M", 0.010),
                        target_axes={"y"},
                        step_s_env="OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_Y_RETREAT_STEP_S",
                    )
                else:
                    y_retreat = {}
                if y_retreat:
                    direct_retreat["steps"].append(y_retreat)
                    attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_open_front_slide_y_retreat:{side}")
                    attempt["steps"].append(y_retreat)
                direct_retreat["success"] = bool(x_retreat.get("success")) and bool(y_retreat.get("success"))
                failed = y_retreat if x_retreat.get("success") else x_retreat
                direct_retreat["why_stopped"] = (
                    "open-front slide X-retreat then Y-clear executed"
                    if direct_retreat["success"]
                    else failed.get("why_stopped")
                )
            direct_retreat["physical_motion_executed"] = any(
                bool(step.get("physical_motion_executed")) for step in direct_retreat["steps"]
            )
            direct_retreat["path"] = write_json(
                run_dir / "plans" / "lower_rack_open_front_slide_retreat.json",
                json_safe(direct_retreat),
            )
        else:
            direct_retreat_axes = {"z"} if _open_front_rack_enabled() else {"x"}
            direct_retreat = _execute_direct_lower_rack_pose_adjust(
                run_dir,
                side,
                target_pose=selected_place["retreat_pose"],
                stage="lower_rack_direct_horizontal_retreat",
                max_xy_delta_m=(
                    _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_XY_DELTA_M", 0.030)
                    if _open_front_rack_enabled()
                    else _env_float("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", 0.160)
                ),
                max_z_delta_m=(
                    _env_float("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_Z_DELTA_M", 0.120)
                    if _open_front_rack_enabled()
                    else 0.010
                ),
                target_axes=direct_retreat_axes,
                step_s_env="OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_STEP_S",
            )
            attempt["movement_capable_calls"].append(f"move_bimanual_joint_keypoints:lower_rack_direct_retreat:{side}")
            attempt["steps"].append(direct_retreat)
        if not direct_retreat.get("success"):
            attempt["why_stopped"] = f"lower-rack direct retreat failed: {direct_retreat.get('why_stopped')}"
            attempt["recovery"] = _attempt_rack_front_recovery(
                run_dir,
                attempt,
                freespace_move=freespace_move,
                side=side,
                selected_place=selected_place,
                phase="direct_retreat_failed",
            )
            return attempt

        if _truthy_env("OPENFORGE_BOWL_RACK_CAPTURE_POST_PLACE_OBSERVATION", True):
            post_place = _capture_object_observation(run_dir, "post_place_on_rack_observe")
            attempt["post_observation"] = post_place
            verification = _verify_place(
                selected_pickup=selected_pickup,
                selected_place=selected_place,
                post_observation=post_place,
            )
            attempt["place_verification"] = verification
            attempt["verification_success"] = bool(verification.get("success"))
            attempt["success"] = bool(attempt["verification_success"])
            attempt["why_stopped"] = (
                f"{TASK_OBJECT_NAME} placement verified on/near rack"
                if attempt["success"]
                else f"place sequence completed but verification failed/inconclusive: {verification.get('reason')}"
            )
        else:
            attempt["post_observation"] = {
                "schema": "openforge.bowl_lower_rack.observation_skip.v1",
                "stage": "post_place_on_rack_observe",
                "skipped": True,
                "success": True,
                "reason": "post-place camera verification disabled to reduce action-to-action pause",
            }
            attempt["place_verification"] = {
                "success": True,
                "skipped": True,
                "reason": "release and retreat completed; post-place perception verification skipped",
            }
            attempt["verification_success"] = True
            attempt["success"] = True
            attempt["why_stopped"] = f"{TASK_OBJECT_NAME} place motion completed; post-place verification skipped"
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.bowl_rack.result.v1",
        "success": False,
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "observation_policy": "fresh live detections only; manual XYZ env vars are rejected",
        "selected_strategy": None,
        "selected_strategy_type": None,
        "planner_result": None,
        "pickup_validation": None,
        "pickup_lift_preview": None,
        "pickup_lift_attempt": None,
        "rack_observation": None,
        "place_candidates_path": None,
        "post_lift_j5_reorientation_preview": None,
        "sequence_preview": None,
        "physical_ticket_status": None,
        "physical_attempt": None,
        "source_snapshot": None,
        "recording_evidence": None,
        "video_paths": [],
        "why_stopped": None,
        "next_recommendation": None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        result["source_snapshot"] = _source_snapshot(run_dir)
        manual_vars = _manual_pose_env_vars_present()
        if manual_vars:
            result["why_stopped"] = "manual XYZ/debug env vars are not allowed for this live bowl/rack task"
            result["manual_pose_env_vars_present"] = manual_vars
            result["next_recommendation"] = "Clear manual pose env vars and rerun from fresh live perception."
            return result
        strategy_env_problems = _forbidden_pickup_strategy_env()
        if strategy_env_problems:
            result["why_stopped"] = f"refusing unsafe {TASK_OBJECT_NAME} pickup strategy configuration"
            result["pickup_strategy_env_problems"] = strategy_env_problems
            result["next_recommendation"] = (
                "Clear the pickup strategy env vars or explicitly opt in with "
                "OPENFORGE_BOWL_RACK_ALLOW_NON_TOP_DOWN_PICKUP_STRATEGY=1."
            )
            return result

        planner_result = _run_fresh_pickup_planner()
        result["planner_result"] = planner_result
        write_json(run_dir / "plans" / f"fresh_{TASK_ARTIFACT_PREFIX}_pickup_planner_result.json", planner_result)
        pickup_validation = _validate_pickup_strategy(planner_result)
        result["pickup_validation"] = pickup_validation
        append_stage_summary(
            run_dir,
            [
                f"## {TASK_OBJECT_NAME} pickup validation",
                f"- passed: {pickup_validation['passed']}",
                f"- selected_label: {pickup_validation.get('selected_label')}",
                f"- arm: {pickup_validation.get('arm')}",
                f"- source_detection: {pickup_validation.get('source_detection')}",
            ],
        )
        if not pickup_validation.get("passed"):
            result["why_stopped"] = f"fresh pickup planner did not produce a valid configured {TASK_OBJECT_NAME} pickup strategy"
            result["next_recommendation"] = "Inspect pickup planner artifacts; no physical motion executed."
            return result

        pickup_retry_strategies = _validated_pickup_retry_strategies(planner_result)
        result["pickup_retry_candidates"] = _pickup_retry_candidate_summary(pickup_retry_strategies)
        append_stage_summary(
            run_dir,
            [
                "## pickup retry candidates",
                f"- count: {len(pickup_retry_strategies)}",
                f"- candidates: {result['pickup_retry_candidates']}",
            ],
        )
        if not pickup_retry_strategies:
            result["why_stopped"] = "no preview-passing pickup retry candidates available"
            result["next_recommendation"] = "Inspect the pickup preview plan before physical motion."
            return result

        selected_pickup = pickup_retry_strategies[0]
        result["selected_strategy"] = selected_pickup.get("label")
        result["selected_strategy_type"] = selected_pickup.get("strategy_type")
        if _truthy_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", False):
            pickup_lift_preview = _preview_pickup_lift_sequence(run_dir, selected_pickup)
            result["pickup_lift_preview"] = pickup_lift_preview
            result["plan_success"] = bool(pickup_lift_preview.get("success"))
            if not pickup_lift_preview.get("success"):
                result["why_stopped"] = "pickup/lift no-motion sequence preview failed"
                result["next_recommendation"] = (
                    f"Inspect {TASK_ARTIFACT_PREFIX}_pickup_lift_sequence_preview.json before physical motion."
                )
                return result
            if not result["physical_gate"]:
                result["success"] = True
                result["why_stopped"] = "recorded no-motion pickup/lift validation succeeded; physical gate was not set"
                result["next_recommendation"] = (
                    f"For physical pickup/lift validation, rerun under the active {TASK_TICKET_SCOPE} ticket "
                    "with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and complete precondition env vars."
                )
                return result

            recording = _recording_evidence(run_dir)
            result["recording_evidence"] = recording
            if not recording.get("passed"):
                result["why_stopped"] = "recording evidence is missing; refusing physical motion"
                result["next_recommendation"] = "Fix recording/debug UI evidence and rerun no-motion validation."
                return result

            ticket_status = _ticket_status(run_dir)
            result["physical_ticket_status"] = ticket_status
            append_stage_summary(
                run_dir,
                [
                    "## physical ticket gate",
                    f"- passed: {ticket_status['passed']}",
                    f"- missing_or_invalid: {ticket_status['missing_or_invalid']}",
                ],
            )
            if not ticket_status.get("passed"):
                result["why_stopped"] = "physical ticket/precondition env acknowledgements are incomplete; refusing physical motion"
                result["next_recommendation"] = "Continue no-motion or rerun physical with complete ticket/precondition env vars."
                return result

            attempt = _execute_pickup_lift_only(run_dir, selected_pickup)
            result["pickup_lift_attempt"] = attempt
            result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
            result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
            result["success"] = bool(attempt.get("success"))
            result["why_stopped"] = attempt.get("why_stopped")
            result["next_recommendation"] = (
                "Review pickup/lift result, overlays, and videos before any rack-placement retry."
            )
            return result

        rack_observation = _observe_rack(run_dir)
        result["rack_observation"] = rack_observation
        rack_detection = rack_observation.get("selected_rack_detection")
        if not rack_detection:
            result["why_stopped"] = "fresh rack detection failed"
            result["next_recommendation"] = "Make the rack visible to top/side cameras and rerun recorded no-motion validation."
            return result

        recording_checked = False
        ticket_checked = False
        physical_retry_records: list[dict[str, Any]] = []
        all_movement_calls: list[str] = []
        for retry_index, retry_pickup in enumerate(pickup_retry_strategies):
            result["selected_strategy"] = retry_pickup.get("label")
            result["selected_strategy_type"] = retry_pickup.get("strategy_type")
            retry_record: dict[str, Any] = {
                "retry_index": retry_index,
                "pickup_label": retry_pickup.get("label"),
                "tcp_above_rim_m": retry_pickup.get("tcp_above_rim_m")
                or (retry_pickup.get("rim_geometry_estimate") or {}).get("tcp_above_rim_m"),
                "grasp_pose": retry_pickup.get("grasp_pose"),
                "place_candidates_path": None,
                "sequence_preview_path": None,
                "physical_attempt": None,
                "retry_allowed": False,
            }
            physical_retry_records.append(retry_record)

            base_place_candidates = _make_place_candidates(retry_pickup, rack_detection)
            j5_delta_candidates = _post_lift_j5_delta_candidates_deg() if _post_lift_j5_reorient_enabled() else [None]
            retry_record["post_lift_j5_delta_candidates_deg"] = j5_delta_candidates
            retry_record["sequence_candidate_attempts"] = []
            sequence: dict[str, Any] | None = None
            for j5_index, j5_delta in enumerate(j5_delta_candidates):
                suffix = (
                    f"_retry_{retry_index}"
                    if j5_delta is None
                    else f"_retry_{retry_index}_j5_{_artifact_safe_label(j5_delta)}"
                )
                place_candidates = copy.deepcopy(base_place_candidates)
                place_candidates, reorient_preview = _post_lift_j5_preview_context(
                    run_dir,
                    retry_pickup,
                    place_candidates,
                    delta_deg=j5_delta,
                    artifact_suffix=suffix,
                )
                candidate_record: dict[str, Any] = {
                    "j5_index": j5_index,
                    "j5_delta_deg": j5_delta,
                    "post_lift_j5_reorientation_preview": reorient_preview,
                }
                retry_record["sequence_candidate_attempts"].append(candidate_record)
                retry_record["post_lift_j5_reorientation_preview"] = reorient_preview
                result["post_lift_j5_reorientation_preview"] = reorient_preview
                if reorient_preview is not None and not bool(reorient_preview.get("success")):
                    candidate_record["why_stopped"] = (
                        f"post-lift j5 reorientation preview failed: {reorient_preview.get('why_stopped')}"
                    )
                    continue
                place_candidates_path = write_json(
                    run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_rack_place_candidates{suffix}.json",
                    {
                        "schema": "openforge.bowl_rack.place_candidates.v1",
                        "retry_index": retry_index,
                        "j5_index": j5_index,
                        "j5_delta_deg": j5_delta,
                        "selected_pickup": retry_pickup,
                        "rack_detection": rack_detection,
                        "candidate_count": len(place_candidates),
                        "candidates": place_candidates,
                    },
                )
                candidate_record["place_candidates_path"] = place_candidates_path
                retry_record["place_candidates_path"] = place_candidates_path
                result["place_candidates_path"] = place_candidates_path
                sequence = _preview_place_sequence(
                    run_dir,
                    retry_pickup,
                    place_candidates,
                    artifact_suffix=suffix,
                )
                candidate_record["sequence_preview_path"] = sequence.get("path")
                candidate_record["sequence_success"] = bool(sequence.get("success"))
                retry_record["sequence_preview_path"] = sequence.get("path")
                retry_record["sequence_success"] = bool(sequence.get("success"))
                result["sequence_preview"] = sequence
                result["plan_success"] = bool(sequence.get("success")) or bool(result.get("plan_success"))
                if sequence.get("success"):
                    retry_record["selected_j5_delta_deg"] = j5_delta
                    break

            if sequence is None or not sequence.get("success"):
                retry_record["why_stopped"] = "pickup-to-rack no-motion sequence preview failed"
                if _truthy_env("OPENFORGE_BOWL_RACK_ALLOW_PICKUP_RETRY_FOR_SEQUENCE_PREVIEW_FAILURE", False):
                    continue
                result["why_stopped"] = (
                    "pickup-to-rack no-motion sequence preview failed; refusing to change pickup depth "
                    "just to satisfy rack transport preview"
                )
                result["next_recommendation"] = "Fix the rack approach/release preview without selecting a deeper pickup grasp."
                result["physical_retry_attempts"] = physical_retry_records
                return result

            if not result["physical_gate"]:
                result["success"] = True
                result["why_stopped"] = "recorded no-motion validation succeeded; physical gate was not set"
                result["next_recommendation"] = (
                    f"For physical mode, rerun once under the active {TASK_TICKET_SCOPE} ticket "
                    "with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and complete precondition env vars."
                )
                result["physical_retry_attempts"] = physical_retry_records
                return result

            if not recording_checked:
                recording = _recording_evidence(run_dir)
                result["recording_evidence"] = recording
                recording_checked = True
                if not recording.get("passed"):
                    result["why_stopped"] = "recording evidence is missing; refusing physical motion"
                    result["next_recommendation"] = "Fix recording/debug UI evidence and rerun no-motion validation."
                    result["physical_retry_attempts"] = physical_retry_records
                    return result

            if not ticket_checked:
                ticket_status = _ticket_status(run_dir)
                result["physical_ticket_status"] = ticket_status
                append_stage_summary(
                    run_dir,
                    [
                        "## physical ticket gate",
                        f"- passed: {ticket_status['passed']}",
                        f"- missing_or_invalid: {ticket_status['missing_or_invalid']}",
                    ],
                )
                ticket_checked = True
                if not ticket_status.get("passed"):
                    result["why_stopped"] = "physical ticket/precondition env acknowledgements are incomplete; refusing physical motion"
                    result["next_recommendation"] = "Continue no-motion or rerun physical with complete ticket/precondition env vars."
                    result["physical_retry_attempts"] = physical_retry_records
                    return result

            attempt = _execute_physical(run_dir, retry_pickup, sequence["selected_place"])
            retry_record["physical_attempt"] = attempt
            retry_record["why_stopped"] = attempt.get("why_stopped")
            retry_record["success"] = bool(attempt.get("success"))
            retry_record["retry_allowed"] = _pickup_retry_allowed(attempt)
            all_movement_calls.extend(list(attempt.get("movement_capable_calls") or []))
            result["physical_attempt"] = attempt
            result["physical_motion_executed"] = bool(result.get("physical_motion_executed")) or bool(
                attempt.get("physical_motion_executed")
            )
            result["movement_capable_calls"] = all_movement_calls
            result["success"] = bool(attempt.get("success"))
            result["why_stopped"] = attempt.get("why_stopped")
            if result["success"]:
                home = _home_after_success(run_dir)
                attempt["post_success_home"] = home
                retry_record["post_success_home"] = home
                result["post_success_home"] = home
                if home.get("physical_motion_executed"):
                    all_movement_calls.extend(list(home.get("movement_capable_calls") or ["go_home:post_success"]))
                    result["physical_motion_executed"] = True
                    result["movement_capable_calls"] = all_movement_calls
                if not home.get("success"):
                    result["success"] = False
                    result["why_stopped"] = home.get("why_stopped")
                    result["next_recommendation"] = "Placement completed, but post-success home failed; inspect the robot state before rerun."
                    result["physical_retry_attempts"] = physical_retry_records
                    return result
                result["next_recommendation"] = "Review result.json, stage_summary.md, overlays, and videos."
                result["physical_retry_attempts"] = physical_retry_records
                return result
            if retry_record["retry_allowed"] and retry_index + 1 < len(pickup_retry_strategies):
                home = _home_between_pickup_retries(run_dir, retry_index + 1)
                retry_record["home_before_next_retry"] = home
                if home.get("physical_motion_executed"):
                    all_movement_calls.extend(list(home.get("movement_capable_calls") or ["go_home:pickup_retry"]))
                    result["movement_capable_calls"] = all_movement_calls
                if not home.get("success"):
                    result["why_stopped"] = home.get("why_stopped")
                    result["next_recommendation"] = "Inspect recovery/home state before another physical retry."
                    result["physical_retry_attempts"] = physical_retry_records
                    return result
                relocalization = _relocalized_pickup_retry(run_dir, retry_index + 1)
                retry_record["relocalization_before_next_retry"] = relocalization
                result.setdefault("pickup_retry_relocalizations", []).append(relocalization)
                if not relocalization.get("success"):
                    result["why_stopped"] = relocalization.get("why_stopped")
                    result["next_recommendation"] = "Inspect fresh bowl detection before another physical retry."
                    result["physical_retry_attempts"] = physical_retry_records
                    return result
                relocalized_pickup = relocalization.get("selected_candidate")
                if isinstance(relocalized_pickup, dict):
                    pickup_retry_strategies[retry_index + 1] = relocalized_pickup
                    result["pickup_retry_candidates"] = _pickup_retry_candidate_summary(pickup_retry_strategies)
                continue
            original_failure = result.get("why_stopped")
            home = _home_after_failure(run_dir, str(original_failure) if original_failure else None)
            attempt["post_failure_home"] = home
            retry_record["post_failure_home"] = home
            result["post_failure_home"] = home
            if home.get("physical_motion_executed"):
                all_movement_calls.extend(list(home.get("movement_capable_calls") or ["go_home:post_failure"]))
                result["physical_motion_executed"] = True
                result["movement_capable_calls"] = all_movement_calls
            if not home.get("success"):
                result["why_stopped"] = f"{original_failure}; post-failure home failed: {home.get('why_stopped')}"
                result["next_recommendation"] = "Inspect the robot state before another physical retry."
                result["physical_retry_attempts"] = physical_retry_records
                return result
            result["next_recommendation"] = (
                "Do not retry under this ticket without reviewing result.json, stage_summary.md, overlays, and videos."
            )
            result["physical_retry_attempts"] = physical_retry_records
            return result

        result["physical_retry_attempts"] = physical_retry_records
        result["why_stopped"] = "no pickup retry candidate produced a valid rack sequence or successful physical attempt"
        result["next_recommendation"] = "Inspect pickup retry records and sequence previews before physical motion."
        return result
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and run log before retrying."
        return result
    finally:
        recording = result.get("recording_evidence") or _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## bowl rack result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- recording_evidence: {result['recording_evidence']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))


def run_task() -> dict[str, Any]:
    TASK_RESULT.clear()
    TASK_RESULT.update(_run())
    return dict(TASK_RESULT)


if __name__ == "__main__":
    run_task()
