"""Pick the green apple and place it in the basket on real YAM.

This entrypoint is stage-gated for block-by-block robot debugging. By default
it observes only and does not move hardware.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from skill_library import grasp_geometry
from skill_library.debug_observation import capture_observation, current_run_dir, save_plan_packet
from skill_library.namespace import (
    detect_objects_oneshot,
    freespace_move,
    get_camera_image,
    get_robot_state,
    go_home,
    open_gripper,
    set_gripper,
)
from skill_library.pick_place import birdseye_pose

try:
    from skill_library.constants.planning import (  # type: ignore
        BATCH_SOLVER_SPEED,
        BATCH_TOP_K as DEFAULT_BATCH_TOP_K,
        BATCH_VALIDATE_TRAJECTORY,
        MOTION_PLANNER_BACKEND,
    )
except Exception:
    DEFAULT_BATCH_TOP_K = 16
    BATCH_SOLVER_SPEED = "fast"
    BATCH_VALIDATE_TRAJECTORY = False
    MOTION_PLANNER_BACKEND = "curobo"


OBJECT_NAME = os.environ.get("OPENFORGE_OBJECT_NAME", "green apple")
TARGET_NAME = os.environ.get("OPENFORGE_TARGET_NAME", "basket")
CAMERA = os.environ.get("OPENFORGE_CAMERA", "top")
OBJECT_CAMERA = os.environ.get("OPENFORGE_OBJECT_CAMERA", CAMERA)
TARGET_CAMERA = os.environ.get("OPENFORGE_TARGET_CAMERA", CAMERA)
GRASP_MODE = os.environ.get("OPENFORGE_GRASP_MODE", "topdown_geom").strip().lower()
TARGET_DROP_Z_OFFSET = float(os.environ.get("OPENFORGE_TARGET_DROP_Z_OFFSET", "0.14"))
PLACE_SEARCH_SOURCE_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_PLACE_SEARCH_SOURCE_RESULT_JSON", "")
PLACE_SEARCH_SOURCE_LOG_DIR = os.environ.get("OPENFORGE_APPLE_PLACE_SEARCH_SOURCE_LOG_DIR", "")
PLACE_X_OFFSETS_M = os.environ.get("OPENFORGE_APPLE_PLACE_X_OFFSETS_M", "0,-0.03,0.03")
PLACE_Y_OFFSETS_M = os.environ.get("OPENFORGE_APPLE_PLACE_Y_OFFSETS_M", "0,-0.03,0.03")
PLACE_Z_OFFSETS_M = os.environ.get("OPENFORGE_APPLE_PLACE_Z_OFFSETS_M", str(TARGET_DROP_Z_OFFSET))
PLACE_RPY_ROLLS_DEG = os.environ.get("OPENFORGE_APPLE_PLACE_RPY_ROLLS_DEG", "")
PLACE_RPY_PITCHES_DEG = os.environ.get("OPENFORGE_APPLE_PLACE_RPY_PITCHES_DEG", "")
PLACE_RPY_YAWS_DEG = os.environ.get("OPENFORGE_APPLE_PLACE_RPY_YAWS_DEG", "")
PLACE_RPY_YAW_OFFSETS_DEG = os.environ.get("OPENFORGE_APPLE_PLACE_RPY_YAW_OFFSETS_DEG", "0,-15,15")
PLACE_MAX_POSES = int(os.environ.get("OPENFORGE_APPLE_PLACE_MAX_POSES", "36"))
PLACE_MAX_CANDIDATES = int(os.environ.get("OPENFORGE_APPLE_PLACE_MAX_CANDIDATES", "8"))
PLACE_MAX_PREVIEW_CALLS = int(os.environ.get("OPENFORGE_APPLE_PLACE_MAX_PREVIEW_CALLS", "96"))
MAX_ATTEMPTS = int(os.environ.get("OPENFORGE_MAX_GRASP_ATTEMPTS", "2"))
MAX_GRASPS = int(os.environ.get("OPENFORGE_MAX_GRASPS", "32"))
TOP_GRASP_TRY = int(os.environ.get("OPENFORGE_TOP_GRASP_TRY", "24"))
BATCH_TOP_K = int(os.environ.get("OPENFORGE_BATCH_TOP_K", str(max(TOP_GRASP_TRY, DEFAULT_BATCH_TOP_K))))
TOPDOWN_GRASP_Z_OFFSET_M = float(os.environ.get("OPENFORGE_TOPDOWN_GRASP_Z_OFFSET_M", "0.0"))
TOPDOWN_GRASP_WIDTH_M = float(os.environ.get("OPENFORGE_TOPDOWN_GRASP_WIDTH_M", "0.08"))
TOPDOWN_GRASP_YAWS = os.environ.get("OPENFORGE_TOPDOWN_GRASP_YAWS", "")
TOPDOWN_GRASP_PITCHES = os.environ.get("OPENFORGE_TOPDOWN_GRASP_PITCHES", "180,165,150,135")
PREGRASP_Z_OFFSET_M = float(os.environ.get("OPENFORGE_APPLE_PREGRASP_Z_OFFSET_M", "0.08"))
LIFT_Z_M = float(os.environ.get("OPENFORGE_APPLE_LIFT_Z_M", "0.10"))
PREGRASP_Z_OFFSETS_M = os.environ.get("OPENFORGE_APPLE_PREGRASP_Z_OFFSETS_M", "0.04,0.06,0.08")
LIFT_Z_OFFSETS_M = os.environ.get("OPENFORGE_APPLE_LIFT_Z_OFFSETS_M", "0.04,0.06,0.08,0.10")
PLANNING_SPEED = float(os.environ.get("OPENFORGE_APPLE_PLANNING_SPEED", "0.45"))
IK_ERROR_THRESHOLD_M = float(os.environ.get("OPENFORGE_APPLE_IK_ERROR_THRESHOLD_M", "0.015"))
IK_ROT_THRESHOLD_DEG = float(os.environ.get("OPENFORGE_APPLE_IK_ROT_THRESHOLD_DEG", "5.0"))
IK_XYZ_WEIGHT = float(os.environ.get("OPENFORGE_APPLE_IK_XYZ_WEIGHT", "1.0"))
IK_RPY_WEIGHT = float(os.environ.get("OPENFORGE_APPLE_IK_RPY_WEIGHT", "0.3"))
SOLVER_SPEED = os.environ.get("OPENFORGE_APPLE_SOLVER_SPEED", BATCH_SOLVER_SPEED).strip().lower()
if SOLVER_SPEED not in {"fast", "slow"}:
    raise ValueError("OPENFORGE_APPLE_SOLVER_SPEED must be 'fast' or 'slow'")
VERIFY_BBOX_MARGIN_PX = float(os.environ.get("OPENFORGE_APPLE_VERIFY_BBOX_MARGIN_PX", "20"))
VERIFY_XY_RADIUS_M = float(os.environ.get("OPENFORGE_APPLE_VERIFY_XY_RADIUS_M", "0.18"))
RESUME_OBJECT_XY_RADIUS_M = float(os.environ.get("OPENFORGE_APPLE_RESUME_OBJECT_XY_RADIUS_M", "0.05"))
RESUME_OBJECT_Z_DELTA_M = float(os.environ.get("OPENFORGE_APPLE_RESUME_OBJECT_Z_DELTA_M", "0.08"))
RESUME_TARGET_XY_RADIUS_M = float(os.environ.get("OPENFORGE_APPLE_RESUME_TARGET_XY_RADIUS_M", "0.08"))
RESUME_RECENTER_ENABLE = os.environ.get("OPENFORGE_APPLE_RESUME_RECENTER_ENABLE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RESUME_RECENTER_MIN_XY_M = float(os.environ.get("OPENFORGE_APPLE_RESUME_RECENTER_MIN_XY_M", "0.012"))
RESUME_RECENTER_MAX_XY_M = float(
    os.environ.get("OPENFORGE_APPLE_RESUME_RECENTER_MAX_XY_M", str(RESUME_OBJECT_XY_RADIUS_M))
)
PLAN_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_PLAN_RESULT_JSON", "")
PLAN_LOG_DIR = os.environ.get("OPENFORGE_APPLE_PLAN_LOG_DIR", "")
PREGRASP_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_PREGRASP_RESULT_JSON", "")
PREGRASP_LOG_DIR = os.environ.get("OPENFORGE_APPLE_PREGRASP_LOG_DIR", "")
GRASP_CLOSE_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_GRASP_CLOSE_RESULT_JSON", "")
GRASP_CLOSE_LOG_DIR = os.environ.get("OPENFORGE_APPLE_GRASP_CLOSE_LOG_DIR", "")
POST_CLOSE_OBSERVE_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_POST_CLOSE_OBSERVE_RESULT_JSON", "")
POST_CLOSE_OBSERVE_LOG_DIR = os.environ.get("OPENFORGE_APPLE_POST_CLOSE_OBSERVE_LOG_DIR", "")
LIFT_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_LIFT_RESULT_JSON", "")
LIFT_LOG_DIR = os.environ.get("OPENFORGE_APPLE_LIFT_LOG_DIR", "")
POST_LIFT_OBSERVE_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_POST_LIFT_OBSERVE_RESULT_JSON", "")
POST_LIFT_OBSERVE_LOG_DIR = os.environ.get("OPENFORGE_APPLE_POST_LIFT_OBSERVE_LOG_DIR", "")
DEBUG_OBS_CAMERAS = os.environ.get("OPENFORGE_DEBUG_OBS_CAMERAS", "top")
DEBUG_OBS_TIMEOUT_S = float(os.environ.get("OPENFORGE_DEBUG_OBS_TIMEOUT_S", "8"))
DEBUG_OBS_CAPTURE_STATE = os.environ.get("OPENFORGE_DEBUG_OBS_CAPTURE_STATE", "0")
HOLD_OBJECT_XY_RADIUS_M = float(os.environ.get("OPENFORGE_APPLE_HOLD_OBJECT_XY_RADIUS_M", "0.035"))
HOLD_OBJECT_Z_DELTA_M = float(os.environ.get("OPENFORGE_APPLE_HOLD_OBJECT_Z_DELTA_M", "0.06"))
POST_LIFT_OBJECT_XY_RADIUS_M = float(os.environ.get("OPENFORGE_APPLE_POST_LIFT_OBJECT_XY_RADIUS_M", "0.04"))
POST_LIFT_OBJECT_Z_DELTA_M = float(os.environ.get("OPENFORGE_APPLE_POST_LIFT_OBJECT_Z_DELTA_M", "0.05"))
APPLE_CLOSE_TARGET_POS_RAW = os.environ.get("OPENFORGE_APPLE_CLOSE_TARGET_POS", "").strip()
APPLE_CLOSE_TARGET_WIDTH_M_RAW = os.environ.get("OPENFORGE_APPLE_CLOSE_TARGET_WIDTH_M", "").strip()
APPLE_GRIPPER_CLOSED_WIDTH_M_RAW = os.environ.get("OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M", "").strip()
APPLE_GRIPPER_OPEN_WIDTH_M_RAW = os.environ.get("OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M", "").strip()
APPLE_GRIPPER_CLOSED_POS_RAW = os.environ.get("OPENFORGE_APPLE_GRIPPER_CLOSED_POS", "").strip()
APPLE_GRIPPER_OPEN_POS_RAW = os.environ.get("OPENFORGE_APPLE_GRIPPER_OPEN_POS", "").strip()
APPLE_CLOSE_WIDTH_COMPRESSION_M = float(os.environ.get("OPENFORGE_APPLE_CLOSE_WIDTH_COMPRESSION_M", "0.005"))
APPLE_ONLINE_CLOSE_ENABLE = os.environ.get("OPENFORGE_APPLE_ONLINE_CLOSE_ENABLE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
APPLE_ONLINE_CLASS_WIDTH_PRIOR_M = float(os.environ.get("OPENFORGE_APPLE_ONLINE_CLASS_WIDTH_PRIOR_M", "0.070"))
APPLE_ONLINE_MIN_WIDTH_M = float(os.environ.get("OPENFORGE_APPLE_ONLINE_MIN_WIDTH_M", "0.055"))
APPLE_ONLINE_MAX_WIDTH_M = float(os.environ.get("OPENFORGE_APPLE_ONLINE_MAX_WIDTH_M", "0.085"))
APPLE_GRIPPER_OPEN_WIDTH_PRIOR_M = float(os.environ.get("OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_PRIOR_M", "0.080"))
APPLE_GRIPPER_CLOSED_WIDTH_PRIOR_M = float(os.environ.get("OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_PRIOR_M", "0.004"))
APPLE_CLOSE_MIN_TARGET_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_MIN_TARGET_POS", "0.08"))
APPLE_CLOSE_MIN_OBSERVED_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_MIN_OBSERVED_POS", "0.05"))
APPLE_CLOSE_MAX_OVERSHOOT_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_MAX_OVERSHOOT_POS", "0.06"))
APPLE_CLOSE_MAX_UNDERCLOSE_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_MAX_UNDERCLOSE_POS", "0.05"))
APPLE_CLOSE_MIN_SEAT_DELTA_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_MIN_SEAT_DELTA_POS", "0.012"))
APPLE_CLOSE_SCOUT_MIN_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SCOUT_MIN_POS", "0.35"))
APPLE_CLOSE_SCOUT_MARGIN_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SCOUT_MARGIN_POS", "0.12"))
APPLE_CLOSE_SCOUT_VEL_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SCOUT_VEL_LIMIT", "2.0"))
APPLE_CLOSE_SCOUT_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SCOUT_TORQUE_LIMIT", "0.15"))
APPLE_CLOSE_VEL_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_VEL_LIMIT", "1.0"))
APPLE_CLOSE_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_TORQUE_LIMIT", "0.12"))
APPLE_CLOSE_SEAT_RETRY_ENABLE = os.environ.get("OPENFORGE_APPLE_CLOSE_SEAT_RETRY_ENABLE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
APPLE_CLOSE_SEAT_RETRY_VEL_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SEAT_RETRY_VEL_LIMIT", "0.6"))
APPLE_CLOSE_SEAT_RETRY_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SEAT_RETRY_TORQUE_LIMIT", "0.2"))
APPLE_CLOSE_SEAT_RETRY_MAX_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SEAT_RETRY_MAX_TORQUE_LIMIT", "0.22"))
APPLE_CLOSE_SEAT_RETRY_TARGET_DELTA_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_SEAT_RETRY_TARGET_DELTA_POS", "0.0"))
APPLE_OPEN_TARGET_POS = float(os.environ.get("OPENFORGE_APPLE_OPEN_TARGET_POS", "1.0"))
APPLE_OPEN_VEL_LIMIT = float(os.environ.get("OPENFORGE_APPLE_OPEN_VEL_LIMIT", "1.0"))
APPLE_OPEN_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_APPLE_OPEN_TORQUE_LIMIT", "0.30"))
APPLE_OPEN_CONFIRM_POS = float(os.environ.get("OPENFORGE_APPLE_OPEN_CONFIRM_POS", "0.55"))
APPLE_OPEN_CONFIRM_TIMEOUT_S = float(os.environ.get("OPENFORGE_APPLE_OPEN_CONFIRM_TIMEOUT_S", "4.0"))
APPLE_OPEN_CONFIRM_POLL_S = float(os.environ.get("OPENFORGE_APPLE_OPEN_CONFIRM_POLL_S", "0.2"))
APPLE_GRIPPER_SETTLE_S = float(os.environ.get("OPENFORGE_APPLE_GRIPPER_SETTLE_S", "0.4"))

REQUESTED_STAGE = os.environ.get(
    "OPENFORGE_APPLE_BASKET_STAGE",
    os.environ.get("OPENFORGE_TASK_STAGE", "observe"),
).strip()

STAGE_ALIASES = {
    "observe": "observe",
    "observe_only": "observe",
    "observe-only": "observe",
    "plan": "plan",
    "plan_only": "plan",
    "plan-only": "plan",
    "resume_plan": "resume_plan",
    "resume-plan": "resume_plan",
    "resume": "resume_plan",
    "pregrasp": "pregrasp",
    "grasp_close": "grasp_close",
    "grasp-close": "grasp_close",
    "close": "grasp_close",
    "lift": "lift",
    "place": "place",
    "release": "place",
    "verify": "verify",
    "verification": "verify",
    "final": "verify",
    "final_observe": "verify",
    "final-observe": "verify",
    "all": "all",
}
PHYSICAL_STAGES = {"pregrasp", "grasp_close", "lift", "place"}

TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "pick_green_apple_to_basket",
    "object": OBJECT_NAME,
    "target": TARGET_NAME,
    "grasp_mode": GRASP_MODE,
    "camera": CAMERA,
    "object_camera": OBJECT_CAMERA,
    "target_camera": TARGET_CAMERA,
    "config": {
        "max_grasps": MAX_GRASPS,
        "top_grasp_try": TOP_GRASP_TRY,
        "batch_top_k": BATCH_TOP_K,
        "topdown_grasp_pitches": TOPDOWN_GRASP_PITCHES,
        "topdown_grasp_yaws": TOPDOWN_GRASP_YAWS,
        "target_drop_z_offset": TARGET_DROP_Z_OFFSET,
        "place_search": {
            "source_result_json": PLACE_SEARCH_SOURCE_RESULT_JSON,
            "source_log_dir": PLACE_SEARCH_SOURCE_LOG_DIR,
            "x_offsets_m": PLACE_X_OFFSETS_M,
            "y_offsets_m": PLACE_Y_OFFSETS_M,
            "z_offsets_m": PLACE_Z_OFFSETS_M,
            "rpy_rolls_deg": PLACE_RPY_ROLLS_DEG,
            "rpy_pitches_deg": PLACE_RPY_PITCHES_DEG,
            "rpy_yaws_deg": PLACE_RPY_YAWS_DEG,
            "rpy_yaw_offsets_deg": PLACE_RPY_YAW_OFFSETS_DEG,
            "max_poses": PLACE_MAX_POSES,
            "max_candidates": PLACE_MAX_CANDIDATES,
            "max_preview_calls": PLACE_MAX_PREVIEW_CALLS,
        },
        "motion_planner_backend": MOTION_PLANNER_BACKEND,
        "batch_solver_speed": SOLVER_SPEED,
        "batch_validate_trajectory": BATCH_VALIDATE_TRAJECTORY,
        "ik_xyz_weight": IK_XYZ_WEIGHT,
        "ik_rpy_weight": IK_RPY_WEIGHT,
        "debug_obs_cameras": DEBUG_OBS_CAMERAS,
        "debug_obs_timeout_s": DEBUG_OBS_TIMEOUT_S,
        "resume_recenter": {
            "enable": RESUME_RECENTER_ENABLE,
            "min_xy_m": RESUME_RECENTER_MIN_XY_M,
            "max_xy_m": RESUME_RECENTER_MAX_XY_M,
            "z_policy": "keep_cached_grasp_z",
        },
        "apple_close": {
            "target_pos_env_set": bool(APPLE_CLOSE_TARGET_POS_RAW),
            "target_width_env_set": bool(APPLE_CLOSE_TARGET_WIDTH_M_RAW),
            "closed_width_env_set": bool(APPLE_GRIPPER_CLOSED_WIDTH_M_RAW),
            "open_width_env_set": bool(APPLE_GRIPPER_OPEN_WIDTH_M_RAW),
            "online_close_enable": APPLE_ONLINE_CLOSE_ENABLE,
            "online_class_width_prior_m": APPLE_ONLINE_CLASS_WIDTH_PRIOR_M,
            "online_min_width_m": APPLE_ONLINE_MIN_WIDTH_M,
            "online_max_width_m": APPLE_ONLINE_MAX_WIDTH_M,
            "open_width_prior_m": APPLE_GRIPPER_OPEN_WIDTH_PRIOR_M,
            "closed_width_prior_m": APPLE_GRIPPER_CLOSED_WIDTH_PRIOR_M,
            "min_target_pos": APPLE_CLOSE_MIN_TARGET_POS,
            "min_observed_pos": APPLE_CLOSE_MIN_OBSERVED_POS,
            "max_underclose_pos": APPLE_CLOSE_MAX_UNDERCLOSE_POS,
            "min_seat_delta_pos": APPLE_CLOSE_MIN_SEAT_DELTA_POS,
            "scout_vel_limit": APPLE_CLOSE_SCOUT_VEL_LIMIT,
            "scout_torque_limit": APPLE_CLOSE_SCOUT_TORQUE_LIMIT,
            "close_vel_limit": APPLE_CLOSE_VEL_LIMIT,
            "close_torque_limit": APPLE_CLOSE_TORQUE_LIMIT,
            "seat_retry_enable": APPLE_CLOSE_SEAT_RETRY_ENABLE,
            "seat_retry_vel_limit": APPLE_CLOSE_SEAT_RETRY_VEL_LIMIT,
            "seat_retry_torque_limit": APPLE_CLOSE_SEAT_RETRY_TORQUE_LIMIT,
            "seat_retry_max_torque_limit": APPLE_CLOSE_SEAT_RETRY_MAX_TORQUE_LIMIT,
            "seat_retry_target_delta_pos": APPLE_CLOSE_SEAT_RETRY_TARGET_DELTA_POS,
            "hold_object_xy_radius_m": HOLD_OBJECT_XY_RADIUS_M,
            "hold_object_z_delta_m": HOLD_OBJECT_Z_DELTA_M,
            "post_lift_object_xy_radius_m": POST_LIFT_OBJECT_XY_RADIUS_M,
            "post_lift_object_z_delta_m": POST_LIFT_OBJECT_Z_DELTA_M,
            "open_target_pos": APPLE_OPEN_TARGET_POS,
            "open_vel_limit": APPLE_OPEN_VEL_LIMIT,
            "open_torque_limit": APPLE_OPEN_TORQUE_LIMIT,
            "open_confirm_pos": APPLE_OPEN_CONFIRM_POS,
            "open_confirm_timeout_s": APPLE_OPEN_CONFIRM_TIMEOUT_S,
            "open_confirm_poll_s": APPLE_OPEN_CONFIRM_POLL_S,
            "gripper_settle_s": APPLE_GRIPPER_SETTLE_S,
        },
    },
    "requested_stage": REQUESTED_STAGE,
    "implemented_stage": STAGE_ALIASES.get(REQUESTED_STAGE.lower(), "unsupported"),
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "observations": {},
    "plan": {},
    "physical_block": {},
    "verification": {},
    "resume_plan": {},
    "risk_notes": [],
    "why_stopped": "not_started",
}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(raw: str, name: str) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a float, got {raw!r}") from exc


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _debug_camera_portal_addr() -> str:
    return (
        os.environ.get("OPENFORGE_DEBUG_OBS_CAMERA_PORTAL", "").strip()
        or os.environ.get("OPENFORGE_CAMERA_PORTAL_ADDR", "").strip()
    )


def _debug_portal_camera(camera: str = "top") -> Any:
    import portal

    return portal.Client(_debug_camera_portal_addr()).get_camera_image(camera).result()


def _debug_observation_cameras() -> list[str]:
    cameras = [part.strip() for part in DEBUG_OBS_CAMERAS.replace(";", ",").split(",") if part.strip()]
    cameras = cameras or ["top"]
    unsupported = [camera for camera in cameras if camera not in {"top", "left", "right", "bottom"}]
    if unsupported:
        raise RuntimeError(f"Unsupported debug observation camera(s): {unsupported}. Use top, left, right, or bottom.")
    return cameras


def _capture_debug_observation(stage: str, *, capture_robot_state: bool | None = None) -> dict[str, Any]:
    if capture_robot_state is None:
        capture_robot_state = _truthy_env("OPENFORGE_DEBUG_OBS_CAPTURE_STATE")
    get_camera_fn = _debug_portal_camera if _debug_camera_portal_addr() else None
    try:
        return capture_observation(
            stage=stage,
            prompts=[OBJECT_NAME, TARGET_NAME],
            cameras=_debug_observation_cameras(),
            get_camera_fn=get_camera_fn,
            capture_robot_state=capture_robot_state,
            per_call_timeout_s=DEBUG_OBS_TIMEOUT_S,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        TASK_RESULT["risk_notes"].append(f"Debug observation capture failed for {stage}: {error}")
        return {
            "schema": "openforge.observation.v1",
            "stage": stage,
            "physical_motion_executed": False,
            "errors": [error],
        }


def _anygrasp_base_url() -> str:
    if os.environ.get("ANYGRASP_SERVICE_URL"):
        return os.environ["ANYGRASP_SERVICE_URL"].rstrip("/")
    host = os.environ.get("ANYGRASP_SERVER_HOST", "127.0.0.1")
    port = os.environ.get("ANYGRASP_SERVER_PORT", "8122")
    return f"http://{host}:{port}"


def _bundlesdf_base_url() -> str:
    if os.environ.get("BUNDLESDF_SERVICE_URL"):
        return os.environ["BUNDLESDF_SERVICE_URL"].rstrip("/")
    host = os.environ.get("BUNDLESDF_SERVER_HOST", "127.0.0.1")
    port = os.environ.get("BUNDLESDF_PORT", "8119")
    return f"http://{host}:{port}"


def _json_get(url: str, timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _uses_anygrasp(grasp_mode: str) -> bool:
    return grasp_mode in {"anygrasp", "top_anygrasp"}


def _round_list(values: Any, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in list(values)]


def _obj_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_detection(det_map: Any, prompt: str) -> Any | None:
    if isinstance(det_map, dict):
        dets = det_map.get(prompt) or []
    else:
        dets = det_map or []
    return dets[0] if dets else None


def _serialize_detection(
    det: Any,
    prompt: str,
    *,
    source: str = "detect_objects_oneshot",
    camera: str | None = None,
) -> dict[str, Any] | None:
    if det is None:
        return None
    position = _obj_value(det, "position_3d", _obj_value(det, "position"))
    if position is None:
        return None
    half_extents = _obj_value(det, "half_extents", [])
    if half_extents is None:
        half_extents = []
    out = {
        "label": _obj_value(det, "label", prompt),
        "prompt": prompt,
        "score": _obj_value(det, "score"),
        "box_2d": _obj_value(det, "box_2d", _obj_value(det, "bbox", [])),
        "position_3d": _round_list(position[:3], 5),
        "half_extents": _round_list(list(half_extents)[:3], 5) if len(half_extents) else [],
        "source": source,
    }
    if camera:
        out["camera"] = camera
    return out


def _debug_packet_detection(debug_packet: dict[str, Any], camera: str, prompt: str) -> dict[str, Any] | None:
    if not isinstance(debug_packet, dict):
        return None
    camera_packet = (debug_packet.get("cameras") or {}).get(camera)
    if not isinstance(camera_packet, dict):
        return None
    detections = camera_packet.get("detections") or {}
    dets = detections.get(prompt) or []
    if not dets:
        return None
    return _serialize_detection(dets[0], prompt, source=f"debug_observation:{camera}", camera=camera)


def _debug_packet_detection_any_safe_camera(
    debug_packet: dict[str, Any],
    prompt: str,
    preferred_camera: str,
) -> dict[str, Any] | None:
    cameras: list[str] = []
    for camera in [preferred_camera, "top", "right", "left"]:
        if camera and camera != "bottom" and camera not in cameras:
            cameras.append(camera)
    for camera in cameras:
        det = _debug_packet_detection(debug_packet, camera, prompt)
        if det is not None:
            return det
    return None


def _detect_one_with_debug_fallback(prompt: str, camera: str, debug_packet: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        det_map = detect_objects_oneshot(prompt, camera=camera)
        direct = _serialize_detection(_first_detection(det_map, prompt), prompt, camera=camera)
        if direct is not None:
            return direct, None
        fallback = _debug_packet_detection_any_safe_camera(debug_packet, prompt, camera)
        if fallback is not None:
            TASK_RESULT["risk_notes"].append(
                f"{prompt!r} direct detection on {camera!r} returned no object; using debug-observation fallback "
                f"from {fallback.get('source')}."
            )
            return fallback, None
        return None, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        fallback = _debug_packet_detection_any_safe_camera(debug_packet, prompt, camera)
        if fallback is not None:
            TASK_RESULT["risk_notes"].append(
                f"{prompt!r} direct detection on {camera!r} failed ({error}); using debug-observation fallback "
                f"from {fallback.get('source')}."
            )
            return fallback, error
        TASK_RESULT["risk_notes"].append(f"{prompt!r} direct detection on {camera!r} failed: {error}")
        return None, error


def _bbox_center(box: Any) -> list[float] | None:
    if not box or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
    return [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)]


def _center_inside_bbox(center: list[float] | None, box: Any, margin_px: float) -> bool | None:
    if center is None or not box or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
    return bool(x1 - margin_px <= center[0] <= x2 + margin_px and y1 - margin_px <= center[1] <= y2 + margin_px)


def _xy_distance_m(a_xyz: Any, b_xyz: Any) -> float | None:
    if not a_xyz or not b_xyz or len(a_xyz) < 2 or len(b_xyz) < 2:
        return None
    dx = float(a_xyz[0]) - float(b_xyz[0])
    dy = float(a_xyz[1]) - float(b_xyz[1])
    return round(float(np.hypot(dx, dy)), 4)


def _result_json_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "result.json"
    return path


def _read_result_details(raw_path: str) -> dict[str, Any]:
    path = _result_json_path(raw_path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object result JSON at {path}")
    details = data.get("details", data)
    if not isinstance(details, dict):
        raise RuntimeError(f"Expected details object in result JSON at {path}")
    details["_source_result_json"] = str(path)
    return details


def _log_dir_from_raw_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_dir():
        return path
    if path.name == "result.json":
        return path.parent
    return path.parent


def _observation_from_debug_packet(
    packet: dict[str, Any],
    *,
    object_camera: str,
    target_camera: str,
    source: str,
) -> dict[str, Any]:
    apple = _debug_packet_detection_any_safe_camera(packet, OBJECT_NAME, object_camera)
    basket = _debug_packet_detection_any_safe_camera(packet, TARGET_NAME, target_camera)
    return {
        "stage": packet.get("stage", "observe"),
        "physical_motion_executed": False,
        "camera": CAMERA,
        "object_camera": object_camera,
        "target_camera": target_camera,
        "object_detection": apple,
        "target_detection": basket,
        "debug_observation": packet,
        "source": source,
    }


def _read_observation_from_artifact(raw_path: str) -> dict[str, Any]:
    details: dict[str, Any] | None = None
    try:
        details = _read_result_details(raw_path)
    except Exception:
        details = None
    if isinstance(details, dict):
        observation = details.get("observations")
        if isinstance(observation, dict):
            observation = dict(observation)
            observation["source_result_json"] = details.get("_source_result_json")
            return observation

    log_dir = _log_dir_from_raw_path(raw_path)
    latest_packet = log_dir / "observations" / "latest.json"
    if not latest_packet.exists():
        raise RuntimeError(f"No observation result or observations/latest.json under {log_dir}")
    packet = json.loads(latest_packet.read_text())
    if not isinstance(packet, dict):
        raise RuntimeError(f"Observation packet is not an object: {latest_packet}")
    observation = _observation_from_debug_packet(
        packet,
        object_camera=OBJECT_CAMERA,
        target_camera=TARGET_CAMERA,
        source=f"debug_observation_packet:{latest_packet}",
    )
    observation["source_result_json"] = str(_result_json_path(raw_path))
    return observation


def _read_place_source_observation() -> dict[str, Any] | None:
    raw_path = PLACE_SEARCH_SOURCE_RESULT_JSON or PLACE_SEARCH_SOURCE_LOG_DIR
    if not raw_path:
        return None
    details = _read_result_details(raw_path)
    observation = details.get("observations")
    if not isinstance(observation, dict):
        raise RuntimeError(f"Place-search source has no observations object: {details.get('_source_result_json')}")
    if observation.get("object_detection") is None or observation.get("target_detection") is None:
        raise RuntimeError(f"Place-search source is missing apple or basket detection: {details.get('_source_result_json')}")
    sourced = dict(observation)
    sourced["source"] = "place_search_source_observe_result"
    sourced["source_result_json"] = details.get("_source_result_json")
    TASK_RESULT["observations"] = sourced
    return sourced


def _extract_plan_from_details(details: dict[str, Any]) -> dict[str, Any]:
    plan = details.get("plan")
    if not isinstance(plan, dict) or not plan.get("success"):
        block = details.get("physical_block", {})
        if isinstance(block, dict):
            plan = block.get("plan")
    if not isinstance(plan, dict) or not plan.get("success"):
        raise RuntimeError(f"Cached result did not contain a successful plan: {details.get('_source_result_json')}")
    cached = dict(plan)
    cached["source_result_json"] = details.get("_source_result_json")
    cached["source"] = "cached_result_json"
    return cached


def _override_plan_from_pregrasp(plan: dict[str, Any]) -> dict[str, Any]:
    raw_path = PREGRASP_RESULT_JSON or PREGRASP_LOG_DIR
    if not raw_path:
        return plan

    details = _read_result_details(raw_path)
    block = details.get("physical_block", {})
    if not isinstance(block, dict) or block.get("stage") != "pregrasp" or not block.get("success"):
        raise RuntimeError(f"Pregrasp result is not a successful pregrasp block: {details.get('_source_result_json')}")
    move = next(
        (
            item
            for item in list(block.get("moves", []) or [])
            if isinstance(item, dict) and item.get("label") == "move_pregrasp" and item.get("executed")
        ),
        None,
    )
    if move is None:
        raise RuntimeError(f"Pregrasp result has no executed move_pregrasp: {details.get('_source_result_json')}")

    pregrasp_pos = [float(value) for value in move["position"][:3]]
    grasp_rpy = [float(value) for value in move["rpy"][:3]]
    pregrasp_offset = float(plan.get("pregrasp_z_offset_m", PREGRASP_Z_OFFSET_M))
    lift_z_m = float(plan.get("lift_z_m", pregrasp_offset))
    grasp_pos = [pregrasp_pos[0], pregrasp_pos[1], pregrasp_pos[2] - pregrasp_offset]
    lift_pos = [pregrasp_pos[0], pregrasp_pos[1], grasp_pos[2] + lift_z_m]
    side = str(block.get("side") or plan.get("selected_grasp", {}).get("arm") or "")
    selected = dict(plan.get("selected_grasp", {}))
    selected.update({"arm": side, "position": _round_list(grasp_pos, 4), "rpy": _round_list(grasp_rpy, 3)})
    plan = dict(plan)
    plan.update(
        {
            "selected_grasp": selected,
            "pregrasp_pose": {"position": _round_list(pregrasp_pos, 4), "rpy": _round_list(grasp_rpy, 3)},
            "grasp_pose": {"position": _round_list(grasp_pos, 4), "rpy": _round_list(grasp_rpy, 3)},
            "lift_pose": {"position": _round_list(lift_pos, 4), "rpy": _round_list(grasp_rpy, 3)},
            "resume_from_pregrasp_result_json": details.get("_source_result_json"),
            "resume_pregrasp_z_offset_m": round(pregrasp_offset, 4),
        }
    )
    return plan


def _load_cached_plan() -> dict[str, Any] | None:
    raw_path = PLAN_RESULT_JSON or PLAN_LOG_DIR
    if not raw_path:
        return None
    plan = _extract_plan_from_details(_read_result_details(raw_path))
    return _override_plan_from_pregrasp(plan)


def _pose_position(plan: dict[str, Any], pose_name: str) -> list[float]:
    pose = plan.get(pose_name)
    if not isinstance(pose, dict) or not pose.get("position"):
        raise RuntimeError(f"Cached plan missing {pose_name}.position")
    return [float(value) for value in pose["position"][:3]]


def _pose_rpy(plan: dict[str, Any], pose_name: str) -> list[float]:
    pose = plan.get(pose_name)
    if not isinstance(pose, dict) or not pose.get("rpy"):
        raise RuntimeError(f"Cached plan missing {pose_name}.rpy")
    return [float(value) for value in pose["rpy"][:3]]


def _resume_coherence(observation: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    apple = observation.get("object_detection")
    basket = observation.get("target_detection")
    grasp_pos = _pose_position(plan, "grasp_pose")
    place_pos = _pose_position(plan, "place_pose")
    coherence: dict[str, Any] = {
        "object_xy_radius_m": RESUME_OBJECT_XY_RADIUS_M,
        "object_z_delta_m": RESUME_OBJECT_Z_DELTA_M,
        "target_xy_radius_m": RESUME_TARGET_XY_RADIUS_M,
        "success": False,
    }
    if apple is None or basket is None:
        coherence["reason"] = "apple_or_basket_detection_missing"
        return coherence

    object_xy_distance = _xy_distance_m(apple.get("position_3d"), grasp_pos)
    object_z_delta = round(abs(float(apple["position_3d"][2]) - grasp_pos[2]), 4)
    target_xy_distance = _xy_distance_m(basket.get("position_3d"), place_pos)
    object_xy_ok = object_xy_distance is not None and object_xy_distance <= RESUME_OBJECT_XY_RADIUS_M
    object_z_ok = object_z_delta <= RESUME_OBJECT_Z_DELTA_M
    target_xy_ok = target_xy_distance is not None and target_xy_distance <= RESUME_TARGET_XY_RADIUS_M
    coherence.update(
        {
            "object_xy_distance_to_grasp_m": object_xy_distance,
            "object_z_delta_to_grasp_m": object_z_delta,
            "target_xy_distance_to_place_m": target_xy_distance,
            "object_xy_ok": object_xy_ok,
            "object_z_ok": object_z_ok,
            "target_xy_ok": target_xy_ok,
            "success": bool(object_xy_ok and object_z_ok and target_xy_ok),
        }
    )
    coherence["reason"] = "split_camera_pose_matches_cached_plan" if coherence["success"] else "split_camera_pose_does_not_match_cached_plan"
    return coherence


def _resume_recenter_plan_from_observation(
    plan: dict[str, Any], observation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": RESUME_RECENTER_ENABLE,
        "applied": False,
        "min_xy_m": RESUME_RECENTER_MIN_XY_M,
        "max_xy_m": RESUME_RECENTER_MAX_XY_M,
        "z_policy": "keep_cached_grasp_z",
    }
    if not RESUME_RECENTER_ENABLE:
        info["reason"] = "disabled"
        return plan, info

    apple = observation.get("object_detection")
    if not isinstance(apple, dict):
        info["reason"] = "apple_detection_missing"
        return plan, info

    camera = _detection_camera(apple, str(observation.get("object_camera") or OBJECT_CAMERA))
    info["source_camera"] = camera
    if camera == "bottom":
        info["reason"] = "bottom_not_allowed_for_motion_pose_recenter"
        return plan, info

    object_pos = apple.get("position_3d")
    if not object_pos or len(object_pos) < 3:
        info["reason"] = "apple_position_3d_missing"
        return plan, info

    grasp_pos = _pose_position(plan, "grasp_pose")
    dx = float(object_pos[0]) - grasp_pos[0]
    dy = float(object_pos[1]) - grasp_pos[1]
    xy_distance = float((dx * dx + dy * dy) ** 0.5)
    info.update(
        {
            "object_position_3d": _round_list(object_pos[:3], 5),
            "old_grasp_position": _round_list(grasp_pos, 5),
            "xy_shift_m": round(xy_distance, 5),
            "xy_delta_m": [round(dx, 5), round(dy, 5)],
        }
    )
    if xy_distance < RESUME_RECENTER_MIN_XY_M:
        info["reason"] = "already_centered"
        return plan, info
    if xy_distance > RESUME_RECENTER_MAX_XY_M:
        info["reason"] = "shift_exceeds_recenter_guard"
        return plan, info

    recentered = dict(plan)
    new_x = float(object_pos[0])
    new_y = float(object_pos[1])
    for pose_name in ("grasp_pose", "pregrasp_pose", "lift_pose"):
        pose = dict(recentered.get(pose_name, {}))
        old_pos = _pose_position(recentered, pose_name)
        pose["position"] = _round_list([new_x, new_y, old_pos[2]], 4)
        recentered[pose_name] = pose

    selected = dict(recentered.get("selected_grasp", {}))
    selected["position"] = list(recentered["grasp_pose"]["position"])
    recentered["selected_grasp"] = selected
    info.update(
        {
            "applied": True,
            "reason": "fresh_calibrated_observation_xy_recenter",
            "new_grasp_position": recentered["grasp_pose"]["position"],
            "new_pregrasp_position": recentered["pregrasp_pose"]["position"],
            "new_lift_position": recentered["lift_pose"]["position"],
        }
    )
    recentered["resume_recenter"] = dict(info)
    return recentered, info


def _post_close_hold_check(plan: dict[str, Any]) -> dict[str, Any]:
    close_path = GRASP_CLOSE_RESULT_JSON or GRASP_CLOSE_LOG_DIR
    observe_path = POST_CLOSE_OBSERVE_RESULT_JSON or POST_CLOSE_OBSERVE_LOG_DIR
    if not close_path:
        raise RuntimeError(
            "Refusing lift: set OPENFORGE_APPLE_GRASP_CLOSE_LOG_DIR or "
            "OPENFORGE_APPLE_GRASP_CLOSE_RESULT_JSON to the inspected width-limited close artifact."
        )
    if not observe_path:
        raise RuntimeError(
            "Refusing lift: run and inspect a post-close no-motion observe, then set "
            "OPENFORGE_APPLE_POST_CLOSE_OBSERVE_LOG_DIR or "
            "OPENFORGE_APPLE_POST_CLOSE_OBSERVE_RESULT_JSON."
        )

    close_details = _read_result_details(close_path)
    block = close_details.get("physical_block", {})
    if not isinstance(block, dict) or block.get("stage") != "grasp_close" or not block.get("success"):
        raise RuntimeError(f"Refusing lift: close result is not a successful grasp_close block: {close_path}")
    movement_calls = list(block.get("movement_capable_calls", []) or [])
    if "close_gripper" in movement_calls:
        raise RuntimeError(f"Refusing lift: close result used raw close_gripper: {close_path}")
    if "set_gripper" not in movement_calls:
        raise RuntimeError(f"Refusing lift: close result did not record set_gripper: {close_path}")
    gripper = block.get("gripper", {})
    if not isinstance(gripper, dict):
        raise RuntimeError(f"Refusing lift: close result missing gripper summary: {close_path}")
    final_pos = gripper.get("final_observed_pos")
    if final_pos is None or float(final_pos) < APPLE_CLOSE_MIN_OBSERVED_POS:
        raise RuntimeError(f"Refusing lift: unsafe final gripper position after close: {final_pos}")
    target_pos = gripper.get("target_pos")
    if target_pos is not None and float(final_pos) > float(target_pos) + APPLE_CLOSE_MAX_UNDERCLOSE_POS:
        raise RuntimeError(
            "Refusing lift: width-limited close remained too open after seat step: "
            f"final={float(final_pos):.5f}, target={float(target_pos):.5f}, "
            f"max_underclose={APPLE_CLOSE_MAX_UNDERCLOSE_POS:.5f}."
        )

    observation = _read_observation_from_artifact(observe_path)
    apple = observation.get("object_detection")
    if not isinstance(apple, dict):
        raise RuntimeError(f"Refusing lift: post-close observation has no apple detection: {observe_path}")
    grasp_pos = _pose_position(plan, "grasp_pose")
    object_pos = apple.get("position_3d")
    object_xy_distance = _xy_distance_m(object_pos, grasp_pos)
    object_z_delta = None
    if object_pos and len(object_pos) >= 3:
        object_z_delta = round(abs(float(object_pos[2]) - grasp_pos[2]), 4)
    object_xy_ok = object_xy_distance is not None and object_xy_distance <= HOLD_OBJECT_XY_RADIUS_M
    object_z_ok = object_z_delta is not None and object_z_delta <= HOLD_OBJECT_Z_DELTA_M
    hold = {
        "success": bool(object_xy_ok and object_z_ok),
        "close_result_json": close_details.get("_source_result_json"),
        "post_close_observe_source": observation.get("source_result_json") or observation.get("source"),
        "object_detection": apple,
        "planned_grasp_position": _round_list(grasp_pos, 4),
        "object_xy_distance_to_grasp_m": object_xy_distance,
        "object_z_delta_to_grasp_m": object_z_delta,
        "object_xy_radius_m": HOLD_OBJECT_XY_RADIUS_M,
        "object_z_delta_m": HOLD_OBJECT_Z_DELTA_M,
        "final_gripper_observed_pos": round(float(final_pos), 5),
        "object_xy_ok": object_xy_ok,
        "object_z_ok": object_z_ok,
    }
    hold["reason"] = "post_close_object_near_grasp" if hold["success"] else "post_close_object_not_near_grasp"
    if not hold["success"]:
        raise RuntimeError(f"Refusing lift: post-close hold check failed: {hold}")
    return hold


def _post_lift_hold_check(plan: dict[str, Any]) -> dict[str, Any]:
    lift_path = LIFT_RESULT_JSON or LIFT_LOG_DIR
    observe_path = POST_LIFT_OBSERVE_RESULT_JSON or POST_LIFT_OBSERVE_LOG_DIR
    if not lift_path:
        raise RuntimeError(
            "Refusing place: set OPENFORGE_APPLE_LIFT_LOG_DIR or "
            "OPENFORGE_APPLE_LIFT_RESULT_JSON to the inspected lift block."
        )
    if not observe_path:
        raise RuntimeError(
            "Refusing place: run and inspect a post-lift no-motion observe, then set "
            "OPENFORGE_APPLE_POST_LIFT_OBSERVE_LOG_DIR or "
            "OPENFORGE_APPLE_POST_LIFT_OBSERVE_RESULT_JSON."
        )

    lift_details = _read_result_details(lift_path)
    block = lift_details.get("physical_block", {})
    if not isinstance(block, dict) or block.get("stage") != "lift" or not block.get("success"):
        raise RuntimeError(f"Refusing place: lift result is not a successful lift block: {lift_path}")

    observation = _read_observation_from_artifact(observe_path)
    apple = observation.get("object_detection")
    if not isinstance(apple, dict):
        raise RuntimeError(f"Refusing place: post-lift observation has no apple detection: {observe_path}")
    lift_pos = _pose_position(plan, "lift_pose")
    object_pos = apple.get("position_3d")
    object_xy_distance = _xy_distance_m(object_pos, lift_pos)
    object_z_delta = None
    if object_pos and len(object_pos) >= 3:
        object_z_delta = round(abs(float(object_pos[2]) - lift_pos[2]), 4)
    object_xy_ok = object_xy_distance is not None and object_xy_distance <= POST_LIFT_OBJECT_XY_RADIUS_M
    object_z_ok = object_z_delta is not None and object_z_delta <= POST_LIFT_OBJECT_Z_DELTA_M
    hold = {
        "success": bool(object_xy_ok and object_z_ok),
        "lift_result_json": lift_details.get("_source_result_json"),
        "post_lift_observe_source": observation.get("source_result_json") or observation.get("source"),
        "object_detection": apple,
        "planned_lift_position": _round_list(lift_pos, 4),
        "object_xy_distance_to_lift_m": object_xy_distance,
        "object_z_delta_to_lift_m": object_z_delta,
        "object_xy_radius_m": POST_LIFT_OBJECT_XY_RADIUS_M,
        "object_z_delta_m": POST_LIFT_OBJECT_Z_DELTA_M,
        "object_xy_ok": object_xy_ok,
        "object_z_ok": object_z_ok,
    }
    hold["reason"] = "post_lift_object_near_lift_pose" if hold["success"] else "post_lift_object_not_near_lift_pose"
    if not hold["success"]:
        raise RuntimeError(f"Refusing place: post-lift hold check failed: {hold}")
    return hold


def _gripper_pos(state: Any, side: str) -> float:
    return float(getattr(state, f"{side}_gripper_pos"))


def _current_gripper_pos(side: str) -> float:
    return _gripper_pos(get_robot_state(), side)


def _run_checked_open(side: str) -> dict[str, Any]:
    target = _clip01(APPLE_OPEN_TARGET_POS)
    before = _current_gripper_pos(side)
    result = set_gripper(
        side,
        target,
        vel_limit=APPLE_OPEN_VEL_LIMIT,
        torque_limit=APPLE_OPEN_TORQUE_LIMIT,
    )
    time.sleep(max(0.0, APPLE_GRIPPER_SETTLE_S))
    after = _current_gripper_pos(side)
    samples = [after]
    deadline = time.monotonic() + max(0.0, APPLE_OPEN_CONFIRM_TIMEOUT_S)
    while after < APPLE_OPEN_CONFIRM_POS and time.monotonic() < deadline:
        time.sleep(max(0.02, APPLE_OPEN_CONFIRM_POLL_S))
        after = _current_gripper_pos(side)
        samples.append(after)
    summary = {
        "mode": "checked_set_gripper_open",
        "side": side,
        "target_pos": round(target, 5),
        "observed_before_pos": round(before, 5),
        "observed_after_pos": round(after, 5),
        "confirm_pos": round(APPLE_OPEN_CONFIRM_POS, 5),
        "confirm_timeout_s": APPLE_OPEN_CONFIRM_TIMEOUT_S,
        "confirm_poll_s": APPLE_OPEN_CONFIRM_POLL_S,
        "vel_limit": APPLE_OPEN_VEL_LIMIT,
        "torque_limit": APPLE_OPEN_TORQUE_LIMIT,
        "samples": [round(sample, 5) for sample in samples],
        "result": result,
        "success": bool(after >= APPLE_OPEN_CONFIRM_POS),
    }
    if not summary["success"]:
        artifact_path = _save_gripper_open_artifact(summary)
        if artifact_path:
            summary["artifact_path"] = artifact_path
        raise RuntimeError(
            "Refusing to move with unconfirmed apple gripper open: "
            f"target={target:.5f}, observed_after={after:.5f}, "
            f"required>={APPLE_OPEN_CONFIRM_POS:.5f}."
            + (f" Open artifact: {artifact_path}" if artifact_path else "")
        )
    return summary


def _save_gripper_open_artifact(summary: dict[str, Any]) -> str | None:
    try:
        run_dir = current_run_dir()
        out_dir = run_dir / "gripper"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_apple_checked_open.json"
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        return str(path)
    except Exception as exc:
        TASK_RESULT["risk_notes"].append(f"Failed to save gripper open artifact: {type(exc).__name__}: {exc}")
        return None


def _require_gripper_open_for_descent(side: str, stage: str) -> dict[str, Any]:
    observed = _current_gripper_pos(side)
    check = {
        "stage": stage,
        "side": side,
        "observed_pos": round(observed, 5),
        "required_min_pos": round(APPLE_OPEN_CONFIRM_POS, 5),
        "success": bool(observed >= APPLE_OPEN_CONFIRM_POS),
    }
    if not check["success"]:
        raise RuntimeError(
            f"Refusing {stage}: {side} gripper is not confirmed open before descent "
            f"(observed={observed:.5f}, required>={APPLE_OPEN_CONFIRM_POS:.5f}). "
            "Run a checked pregrasp/open recovery before moving toward the apple."
        )
    return check


def _bbox_xyxy(box: Any) -> list[float] | None:
    if not box or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in list(box)[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _depth_bbox_width_estimate(camera: str, detection: dict[str, Any]) -> dict[str, Any]:
    box = _bbox_xyxy(detection.get("box_2d"))
    if box is None:
        return {"success": False, "camera": camera, "reason": "missing_or_invalid_bbox"}
    portal_addr = _debug_camera_portal_addr()
    if not portal_addr:
        return {"success": False, "camera": camera, "reason": "no_camera_portal_configured"}
    try:
        import portal

        client = portal.Client(portal_addr)
        depth = np.asarray(client.get_camera_depth(camera).result(), dtype=np.float32)
        intr = [float(x) for x in client.get_camera_intrinsics(camera).result()]
    except Exception as exc:
        return {"success": False, "camera": camera, "reason": f"{type(exc).__name__}: {exc}"}

    if depth.ndim != 2 or len(intr) != 4:
        return {"success": False, "camera": camera, "reason": "invalid_depth_or_intrinsics_shape"}
    fx, fy, _, _ = intr
    if fx <= 0.0 or fy <= 0.0:
        return {"success": False, "camera": camera, "reason": "invalid_intrinsics"}
    h, w = depth.shape
    x1, y1, x2, y2 = box
    xi1 = max(0, min(w - 1, int(np.floor(x1))))
    yi1 = max(0, min(h - 1, int(np.floor(y1))))
    xi2 = max(xi1 + 1, min(w, int(np.ceil(x2))))
    yi2 = max(yi1 + 1, min(h, int(np.ceil(y2))))
    crop = depth[yi1:yi2, xi1:xi2]
    valid = crop[np.isfinite(crop) & (crop > 0.05) & (crop < 3.0)]
    if valid.size < 8:
        return {
            "success": False,
            "camera": camera,
            "reason": f"not_enough_valid_depth_pixels:{int(valid.size)}",
            "bbox_xyxy": _round_list(box, 2),
        }
    median_depth_m = float(np.median(valid))
    width_px = max(1.0, x2 - x1)
    height_px = max(1.0, y2 - y1)
    width_x_m = width_px * median_depth_m / fx
    width_y_m = height_px * median_depth_m / fy
    return {
        "success": True,
        "camera": camera,
        "bbox_xyxy": _round_list(box, 2),
        "depth_shape": [int(h), int(w)],
        "valid_depth_pixels": int(valid.size),
        "median_depth_m": round(median_depth_m, 5),
        "intrinsics": [round(float(v), 4) for v in intr],
        "width_from_bbox_x_m": round(float(width_x_m), 5),
        "width_from_bbox_y_m": round(float(width_y_m), 5),
        "estimated_round_object_width_m": round(float(max(width_x_m, width_y_m)), 5),
    }


def _detection_camera(detection: dict[str, Any], default_camera: str) -> str:
    camera = str(detection.get("camera") or "").strip()
    if camera:
        return camera
    source = str(detection.get("source") or "")
    prefix = "debug_observation:"
    if source.startswith(prefix):
        parsed = source[len(prefix) :].strip()
        if parsed:
            return parsed
    return default_camera


def _online_close_estimate(observation: dict[str, Any], selected: dict[str, Any] | None) -> dict[str, Any]:
    selected = selected or {}
    selected_width_m = float(_obj_value(selected, "width", TOPDOWN_GRASP_WIDTH_M) or TOPDOWN_GRASP_WIDTH_M)
    apple = observation.get("object_detection") if isinstance(observation, dict) else None
    object_camera = str(observation.get("object_camera") or OBJECT_CAMERA) if isinstance(observation, dict) else OBJECT_CAMERA
    width_camera = object_camera
    width_sources: list[dict[str, Any]] = []
    observed_widths: list[float] = []

    if isinstance(apple, dict):
        width_camera = _detection_camera(apple, object_camera)
    if isinstance(apple, dict) and width_camera != "bottom":
        depth_estimate = _depth_bbox_width_estimate(width_camera, apple)
        width_sources.append({"type": "bbox_depth_intrinsics", **depth_estimate})
        if depth_estimate.get("success"):
            observed_widths.append(float(depth_estimate["estimated_round_object_width_m"]))
    elif width_camera == "bottom":
        width_sources.append(
            {
                "type": "bbox_depth_intrinsics",
                "success": False,
                "camera": width_camera,
                "reason": "bottom_not_allowed_for_motion_world_pose_or_close_width",
            }
        )

    class_prior = float(np.clip(APPLE_ONLINE_CLASS_WIDTH_PRIOR_M, APPLE_ONLINE_MIN_WIDTH_M, APPLE_ONLINE_MAX_WIDTH_M))
    planner_prior = float(np.clip(selected_width_m, APPLE_ONLINE_MIN_WIDTH_M, APPLE_ONLINE_MAX_WIDTH_M))
    estimated_width_m = max([class_prior, *observed_widths])
    estimated_width_m = float(np.clip(estimated_width_m, APPLE_ONLINE_MIN_WIDTH_M, APPLE_ONLINE_MAX_WIDTH_M))

    closed_width_m = _optional_float(APPLE_GRIPPER_CLOSED_WIDTH_M_RAW, "OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M")
    open_width_m = _optional_float(APPLE_GRIPPER_OPEN_WIDTH_M_RAW, "OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M")
    closed_width_m = APPLE_GRIPPER_CLOSED_WIDTH_PRIOR_M if closed_width_m is None else closed_width_m
    open_width_m = APPLE_GRIPPER_OPEN_WIDTH_PRIOR_M if open_width_m is None else open_width_m
    open_width_m = max(float(open_width_m), selected_width_m, estimated_width_m + 0.002)
    if open_width_m <= closed_width_m:
        open_width_m = closed_width_m + 0.001

    target_width_m = _optional_float(APPLE_CLOSE_TARGET_WIDTH_M_RAW, "OPENFORGE_APPLE_CLOSE_TARGET_WIDTH_M")
    target_width_m = estimated_width_m - APPLE_CLOSE_WIDTH_COMPRESSION_M if target_width_m is None else target_width_m
    target_width_m = float(np.clip(target_width_m, closed_width_m, open_width_m))
    ratio = (target_width_m - closed_width_m) / (open_width_m - closed_width_m)
    closed_pos = _optional_float(APPLE_GRIPPER_CLOSED_POS_RAW, "OPENFORGE_APPLE_GRIPPER_CLOSED_POS")
    open_pos = _optional_float(APPLE_GRIPPER_OPEN_POS_RAW, "OPENFORGE_APPLE_GRIPPER_OPEN_POS")
    closed_pos = 0.0 if closed_pos is None else closed_pos
    open_pos = 1.0 if open_pos is None else open_pos
    target_pos = _clip01(closed_pos + ratio * (open_pos - closed_pos))
    return {
        "mode": "online_observation_model_prior",
        "success": bool(APPLE_ONLINE_CLOSE_ENABLE and target_pos >= APPLE_CLOSE_MIN_TARGET_POS),
        "object_camera": object_camera,
        "width_camera": width_camera,
        "motion_pose_cameras_allowed": ["top", "left", "right"],
        "bottom_policy": "bottom artifacts may be saved but bottom position_3d is not used for motion or close width",
        "selected_planner_width_m": round(selected_width_m, 5),
        "class_width_prior_m": round(class_prior, 5),
        "planner_width_prior_m": round(planner_prior, 5),
        "observed_width_sources": width_sources,
        "estimated_apple_width_m": round(estimated_width_m, 5),
        "target_width_m": round(target_width_m, 5),
        "closed_width_m": round(float(closed_width_m), 5),
        "open_width_m": round(float(open_width_m), 5),
        "target_pos": round(float(target_pos), 5),
        "grasp_axis_evidence": {
            "source": "selected_topdown_grasp_rpy",
            "rpy": _round_list(selected.get("rpy", []), 3),
            "note": "apple treated as approximately round; close width uses conservative max observed/prior diameter",
        },
    }


def _resolve_width_limited_close(plan: dict[str, Any]) -> dict[str, Any]:
    explicit_target = _optional_float(APPLE_CLOSE_TARGET_POS_RAW, "OPENFORGE_APPLE_CLOSE_TARGET_POS")
    selected = plan.get("selected_grasp", {})
    selected_width_m = float(_obj_value(selected, "width", TOPDOWN_GRASP_WIDTH_M) or TOPDOWN_GRASP_WIDTH_M)

    if explicit_target is not None:
        target_pos = explicit_target
        target_width_m = _optional_float(
            APPLE_CLOSE_TARGET_WIDTH_M_RAW,
            "OPENFORGE_APPLE_CLOSE_TARGET_WIDTH_M",
        )
        source = "OPENFORGE_APPLE_CLOSE_TARGET_POS"
    else:
        closed_width_m = _optional_float(
            APPLE_GRIPPER_CLOSED_WIDTH_M_RAW,
            "OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M",
        )
        open_width_m = _optional_float(
            APPLE_GRIPPER_OPEN_WIDTH_M_RAW,
            "OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M",
        )
        if closed_width_m is not None and open_width_m is not None:
            if open_width_m <= closed_width_m:
                raise RuntimeError(
                    "OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M must be greater than "
                    "OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M."
                )
            closed_pos = _optional_float(APPLE_GRIPPER_CLOSED_POS_RAW, "OPENFORGE_APPLE_GRIPPER_CLOSED_POS")
            open_pos = _optional_float(APPLE_GRIPPER_OPEN_POS_RAW, "OPENFORGE_APPLE_GRIPPER_OPEN_POS")
            closed_pos = 0.0 if closed_pos is None else closed_pos
            open_pos = 1.0 if open_pos is None else open_pos
            target_width_env = _optional_float(
                APPLE_CLOSE_TARGET_WIDTH_M_RAW,
                "OPENFORGE_APPLE_CLOSE_TARGET_WIDTH_M",
            )
            target_width_m = target_width_env
            if target_width_m is None:
                target_width_m = selected_width_m - APPLE_CLOSE_WIDTH_COMPRESSION_M
            target_width_m = float(np.clip(target_width_m, closed_width_m, open_width_m))
            ratio = (target_width_m - closed_width_m) / (open_width_m - closed_width_m)
            target_pos = closed_pos + ratio * (open_pos - closed_pos)
            source = "width_calibration"
        else:
            if not APPLE_ONLINE_CLOSE_ENABLE:
                raise RuntimeError(
                    "Refusing physical apple grasp_close before moving to grasp: online close "
                    "estimation is disabled and no explicit close target/width calibration is configured."
                )
            online_estimate = plan.get("online_close_estimate")
            if not isinstance(online_estimate, dict):
                online_estimate = _online_close_estimate({"object_camera": OBJECT_CAMERA}, selected)
            if not online_estimate.get("success"):
                raise RuntimeError(f"Online apple close estimate is not usable: {online_estimate}")
            target_pos = float(online_estimate["target_pos"])
            target_width_m = float(online_estimate["target_width_m"])
            source = "online_observation_model_prior"

    target_pos = _clip01(target_pos)
    if target_pos < APPLE_CLOSE_MIN_TARGET_POS:
        raise RuntimeError(
            "Refusing physical apple grasp_close before moving to grasp: calibrated "
            f"target_pos={target_pos:.4f} is below OPENFORGE_APPLE_CLOSE_MIN_TARGET_POS="
            f"{APPLE_CLOSE_MIN_TARGET_POS:.4f}. Provide safer calibration before closing."
        )
    scout_pos = _clip01(max(APPLE_CLOSE_SCOUT_MIN_POS, target_pos + APPLE_CLOSE_SCOUT_MARGIN_POS))
    if scout_pos <= target_pos:
        scout_pos = _clip01(min(1.0, target_pos + 0.05))
    return {
        "mode": "staged_width_limited_set_gripper",
        "source": source,
        "selected_planner_width_m": round(selected_width_m, 5),
        "target_width_m": None if target_width_m is None else round(float(target_width_m), 5),
        "target_pos": round(target_pos, 5),
        "scout_pos": round(scout_pos, 5),
        "online_close_estimate": plan.get("online_close_estimate"),
        "min_target_pos": APPLE_CLOSE_MIN_TARGET_POS,
        "min_observed_pos": APPLE_CLOSE_MIN_OBSERVED_POS,
        "max_overshoot_pos": APPLE_CLOSE_MAX_OVERSHOOT_POS,
        "max_underclose_pos": APPLE_CLOSE_MAX_UNDERCLOSE_POS,
        "min_seat_delta_pos": APPLE_CLOSE_MIN_SEAT_DELTA_POS,
        "stop_conditions": [
            "observed gripper position near fully closed",
            "observed gripper position overshoots target beyond max_overshoot_pos",
            "final seat step remains too far open relative to target",
            "set_gripper tool failure",
            "object slip/drop or unclear post-block perception",
            "unexpected contact or gripper anomaly",
        ],
        "scout_vel_limit": APPLE_CLOSE_SCOUT_VEL_LIMIT,
        "scout_torque_limit": APPLE_CLOSE_SCOUT_TORQUE_LIMIT,
        "close_vel_limit": APPLE_CLOSE_VEL_LIMIT,
        "close_torque_limit": APPLE_CLOSE_TORQUE_LIMIT,
        "seat_retry_enable": APPLE_CLOSE_SEAT_RETRY_ENABLE,
        "seat_retry_vel_limit": APPLE_CLOSE_SEAT_RETRY_VEL_LIMIT,
        "seat_retry_torque_limit": min(APPLE_CLOSE_SEAT_RETRY_TORQUE_LIMIT, APPLE_CLOSE_SEAT_RETRY_MAX_TORQUE_LIMIT),
        "seat_retry_max_torque_limit": APPLE_CLOSE_SEAT_RETRY_MAX_TORQUE_LIMIT,
        "seat_retry_target_delta_pos": APPLE_CLOSE_SEAT_RETRY_TARGET_DELTA_POS,
    }


def _set_gripper_step(
    side: str,
    *,
    label: str,
    target_pos: float,
    vel_limit: float,
    torque_limit: float,
) -> dict[str, Any]:
    before = _gripper_pos(get_robot_state(), side)
    result = set_gripper(
        side,
        float(target_pos),
        vel_limit=float(vel_limit),
        torque_limit=float(torque_limit),
    )
    after = _gripper_pos(get_robot_state(), side)
    return {
        "label": label,
        "target_pos": round(float(target_pos), 5),
        "observed_before_pos": round(before, 5),
        "observed_after_pos": round(after, 5),
        "vel_limit": float(vel_limit),
        "torque_limit": float(torque_limit),
        "result": result,
    }


def _guard_gripper_observation(step: dict[str, Any], close_config: dict[str, Any]) -> None:
    observed = float(step["observed_after_pos"])
    target = float(step["target_pos"])
    min_observed = float(close_config["min_observed_pos"])
    max_overshoot = float(close_config["max_overshoot_pos"])
    if observed < min_observed:
        raise RuntimeError(
            f"Refusing to continue after {step['label']}: observed gripper pos "
            f"{observed:.5f} is near fully closed (< {min_observed:.5f})."
        )
    if observed < target - max_overshoot:
        raise RuntimeError(
            f"Refusing to continue after {step['label']}: observed gripper pos "
            f"{observed:.5f} is more closed than target {target:.5f} by over "
            f"{max_overshoot:.5f}."
        )


def _guard_final_seat_observation(step: dict[str, Any], close_config: dict[str, Any]) -> None:
    observed = float(step["observed_after_pos"])
    before = float(step["observed_before_pos"])
    target = float(step["target_pos"])
    max_underclose = float(close_config["max_underclose_pos"])
    min_seat_delta = float(close_config["min_seat_delta_pos"])
    if observed > target + max_underclose:
        raise RuntimeError(
            f"Refusing to treat close as successful after {step['label']}: observed gripper pos "
            f"{observed:.5f} remains too open for target {target:.5f} "
            f"(max_underclose={max_underclose:.5f})."
        )
    if target <= before - min_seat_delta and before - observed < min_seat_delta:
        raise RuntimeError(
            f"Refusing to treat close as successful after {step['label']}: measured seat delta "
            f"{before - observed:.5f} is below {min_seat_delta:.5f} even though target "
            f"{target:.5f} requested additional closure from {before:.5f}."
        )


def _run_width_limited_close(side: str, close_config: dict[str, Any]) -> dict[str, Any]:
    initial_pos = _gripper_pos(get_robot_state(), side)
    if initial_pos < APPLE_CLOSE_MIN_OBSERVED_POS:
        raise RuntimeError(
            f"Refusing physical apple grasp_close: {side} gripper already reads "
            f"{initial_pos:.5f}, below safe observed minimum {APPLE_CLOSE_MIN_OBSERVED_POS:.5f}."
        )
    steps: list[dict[str, Any]] = []

    def build_summary(success: bool, failure_reason: str | None = None) -> dict[str, Any]:
        summary = {
            **close_config,
            "success": success,
            "initial_observed_pos": round(initial_pos, 5),
            "steps": list(steps),
            "final_observed_pos": steps[-1]["observed_after_pos"] if steps else round(initial_pos, 5),
        }
        if failure_reason:
            summary["failure_reason"] = failure_reason
        return summary

    def raise_with_artifact(exc: RuntimeError) -> None:
        summary = build_summary(False, str(exc))
        artifact_path = _save_gripper_close_artifact(summary)
        if artifact_path:
            raise RuntimeError(f"{exc} Partial close artifact: {artifact_path}") from exc
        raise exc

    scout = _set_gripper_step(
        side,
        label="scout_close",
        target_pos=float(close_config["scout_pos"]),
        vel_limit=APPLE_CLOSE_SCOUT_VEL_LIMIT,
        torque_limit=APPLE_CLOSE_SCOUT_TORQUE_LIMIT,
    )
    steps.append(scout)
    try:
        _guard_gripper_observation(scout, close_config)
    except RuntimeError as exc:
        raise_with_artifact(exc)

    final = _set_gripper_step(
        side,
        label="seat_close",
        target_pos=float(close_config["target_pos"]),
        vel_limit=APPLE_CLOSE_VEL_LIMIT,
        torque_limit=APPLE_CLOSE_TORQUE_LIMIT,
    )
    steps.append(final)
    try:
        _guard_gripper_observation(final, close_config)
        _guard_final_seat_observation(final, close_config)
    except RuntimeError as exc:
        if not APPLE_CLOSE_SEAT_RETRY_ENABLE:
            raise_with_artifact(exc)
        retry_target = max(
            APPLE_CLOSE_MIN_TARGET_POS,
            float(close_config["target_pos"]) - max(0.0, APPLE_CLOSE_SEAT_RETRY_TARGET_DELTA_POS),
        )
        retry_config = dict(close_config)
        retry_config["target_pos"] = retry_target
        retry = _set_gripper_step(
            side,
            label="seat_retry",
            target_pos=retry_target,
            vel_limit=APPLE_CLOSE_SEAT_RETRY_VEL_LIMIT,
            torque_limit=min(APPLE_CLOSE_SEAT_RETRY_TORQUE_LIMIT, APPLE_CLOSE_SEAT_RETRY_MAX_TORQUE_LIMIT),
        )
        steps.append(retry)
        try:
            _guard_gripper_observation(retry, retry_config)
            _guard_final_seat_observation(retry, retry_config)
        except RuntimeError as retry_exc:
            raise_with_artifact(retry_exc)
    return build_summary(True)


def _save_gripper_close_artifact(summary: dict[str, Any]) -> str | None:
    try:
        run_dir = current_run_dir()
        out_dir = run_dir / "gripper"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_apple_width_limited_close.json"
        path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        return str(path)
    except Exception as exc:
        TASK_RESULT["risk_notes"].append(f"Failed to save gripper close artifact: {type(exc).__name__}: {exc}")
        return None


def _width_limited_close_readiness(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        close_config = _resolve_width_limited_close(plan)
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "required_evidence": (
                "Provide a usable online observation/model-prior apple width estimate, set "
                "OPENFORGE_APPLE_CLOSE_TARGET_POS to a validated nonzero normalized target, "
                "or set OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M and "
                "OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M for explicit width conversion."
            ),
        }
    return {"ready": True, "close_config": close_config}


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "status": _obj_value(result, "status"),
        "executed": bool(_obj_value(result, "executed", False)),
        "side": _obj_value(result, "side"),
        "trajectory_steps": int(_obj_value(result, "trajectory_steps", 0) or 0),
        "trajectory_cache_key": _obj_value(result, "trajectory_cache_key"),
        "final_pos_error_m": _obj_value(result, "final_pos_error_m"),
        "final_rot_error_deg": _obj_value(result, "final_rot_error_deg"),
        "reason": _obj_value(result, "reason", ""),
    }


def _grasp_to_dict(grasp: Any, index: int | None = None) -> dict[str, Any]:
    out = {
        "position": _round_list(_obj_value(grasp, "position", []), 4),
        "rpy": _round_list(_obj_value(grasp, "rpy", []), 3),
        "score": round(float(_obj_value(grasp, "score", 1.0) or 1.0), 4),
        "width": round(float(_obj_value(grasp, "width", TOPDOWN_GRASP_WIDTH_M)), 4),
        "trajectory_cache_key": _obj_value(grasp, "trajectory_cache_key"),
    }
    if index is not None:
        out["candidate_index"] = int(index)
    return out


def _parse_offsets(raw: str, fallback: list[float]) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    values.extend(fallback)
    cleaned: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(max(0.0, float(value)), 4)
        if rounded in seen:
            continue
        seen.add(rounded)
        cleaned.append(rounded)
    return cleaned or fallback


def _parse_float_candidates(raw: str, fallback: list[float], *, min_value: float | None = None) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        values = [float(value) for value in fallback]
    cleaned: list[float] = []
    seen: set[float] = set()
    for value in values:
        if min_value is not None:
            value = max(float(min_value), float(value))
        rounded = round(float(value), 4)
        if rounded in seen:
            continue
        seen.add(rounded)
        cleaned.append(rounded)
    return cleaned or [float(value) for value in fallback]


def _angle_delta_deg(value: float, base: float) -> float:
    return abs(((float(value) - float(base) + 180.0) % 360.0) - 180.0)


def _sample_topdown_from_observation(apple: dict[str, Any]) -> list[Any]:
    center = np.asarray(apple.get("position_3d"), dtype=float).reshape(-1)
    if center.shape[0] < 3:
        raise RuntimeError(f"Source apple detection has invalid position_3d={apple.get('position_3d')}")
    center = center[:3]
    yaws = _parse_float_candidates(TOPDOWN_GRASP_YAWS, [90.0, 0.0, -45.0, 45.0, -90.0, 135.0, -135.0, 180.0])
    pitches = _parse_float_candidates(TOPDOWN_GRASP_PITCHES, [180.0])
    z_offsets = _parse_float_candidates(
        os.environ.get("OPENFORGE_TOPDOWN_GRASP_Z_OFFSETS_M", ""),
        [TOPDOWN_GRASP_Z_OFFSET_M, TOPDOWN_GRASP_Z_OFFSET_M + 0.01, TOPDOWN_GRASP_Z_OFFSET_M - 0.01],
    )
    base_score = float(apple.get("score") or 1.0)
    base_score = max(0.05, min(1.0, base_score))
    make_grasp = getattr(grasp_geometry, "_as_grasp", None)
    grasps: list[Any] = []
    for z_index, z_offset in enumerate(z_offsets):
        for pitch_index, pitch in enumerate(pitches):
            for yaw_index, yaw in enumerate(yaws):
                grasp_pos = center.copy()
                grasp_pos[2] = float(grasp_pos[2] + z_offset)
                score = max(0.01, base_score - 0.02 * yaw_index - 0.04 * pitch_index - 0.05 * z_index)
                if callable(make_grasp):
                    grasp = make_grasp(grasp_pos, [0.0, float(pitch), float(yaw)], score=score, width=TOPDOWN_GRASP_WIDTH_M)
                else:
                    grasp = {
                        "position": [float(x) for x in grasp_pos],
                        "rpy": [0.0, float(pitch), float(yaw)],
                        "score": score,
                        "width": TOPDOWN_GRASP_WIDTH_M,
                        "trajectory_cache_key": None,
                    }
                grasps.append(grasp)
                if len(grasps) >= MAX_GRASPS:
                    return grasps
    return grasps


def _place_pose_candidates(side: str, basket_xyz: list[float]) -> list[dict[str, Any]]:
    _, base_rpy = birdseye_pose(side)
    x_offsets = _parse_float_candidates(PLACE_X_OFFSETS_M, [0.0])
    y_offsets = _parse_float_candidates(PLACE_Y_OFFSETS_M, [0.0])
    z_offsets = _parse_float_candidates(PLACE_Z_OFFSETS_M, [TARGET_DROP_Z_OFFSET], min_value=0.0)
    rolls = _parse_float_candidates(PLACE_RPY_ROLLS_DEG, [float(base_rpy[0])])
    pitches = _parse_float_candidates(
        PLACE_RPY_PITCHES_DEG,
        [float(base_rpy[1]), float(base_rpy[1]) - 10.0, float(base_rpy[1]) + 10.0],
    )
    if PLACE_RPY_YAWS_DEG.strip():
        yaws = _parse_float_candidates(PLACE_RPY_YAWS_DEG, [float(base_rpy[2])])
    else:
        yaw_offsets = _parse_float_candidates(PLACE_RPY_YAW_OFFSETS_DEG, [0.0])
        yaws = [round(float(base_rpy[2]) + offset, 4) for offset in yaw_offsets]

    scored: list[tuple[float, int, dict[str, Any]]] = []
    order = 0
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z_offset in z_offsets:
                position = [
                    float(basket_xyz[0]) + float(x_offset),
                    float(basket_xyz[1]) + float(y_offset),
                    float(basket_xyz[2]) + float(z_offset),
                ]
                for roll in rolls:
                    for pitch in pitches:
                        for yaw in yaws:
                            rpy = [float(roll), float(pitch), float(yaw)]
                            score = (
                                abs(float(x_offset))
                                + abs(float(y_offset))
                                + abs(float(z_offset) - TARGET_DROP_Z_OFFSET)
                                + 0.001 * sum(_angle_delta_deg(rpy[i], base_rpy[i]) for i in range(3))
                            )
                            scored.append(
                                (
                                    score,
                                    order,
                                    {
                                        "position": _round_list(position, 4),
                                        "rpy": _round_list(rpy, 3),
                                        "x_offset_m": round(float(x_offset), 4),
                                        "y_offset_m": round(float(y_offset), 4),
                                        "z_offset_m": round(float(z_offset), 4),
                                        "base_rpy": _round_list(base_rpy, 3),
                                    },
                                )
                            )
                            order += 1
    scored.sort(key=lambda item: (item[0], item[1]))
    return [spec for _, _, spec in scored[: max(1, PLACE_MAX_POSES)]]


def _preview_error_score(preview: dict[str, Any]) -> tuple[float, float]:
    pos_err = preview.get("final_pos_error_m")
    rot_err = preview.get("final_rot_error_deg")
    reason = str(preview.get("reason") or "")
    if pos_err is None:
        match = re.search(r"pos_err=([0-9.]+)\s*m", reason)
        if match:
            pos_err = float(match.group(1))
    if rot_err is None:
        match = re.search(r"rot_err=([0-9.]+)\s*deg", reason)
        if match:
            rot_err = float(match.group(1))
    return (
        float(pos_err) if pos_err is not None else 999.0,
        float(rot_err) if rot_err is not None else 999.0,
    )


def _search_place_pose(side: str, basket_xyz: list[float], max_calls: int) -> dict[str, Any]:
    pose_specs = _place_pose_candidates(side, basket_xyz)
    attempts: list[dict[str, Any]] = []
    selected_pose: dict[str, Any] | None = None
    selected_preview: dict[str, Any] | None = None
    best_failure: dict[str, Any] | None = None
    for index, spec in enumerate(pose_specs[: max(0, max_calls)], start=1):
        preview = _single_pose_preview(side, spec["position"], spec["rpy"], "preview_place")
        ok = preview.get("status") == "Success" and not preview.get("executed")
        attempt = {
            "index": index,
            "success": ok,
            "pose": spec,
            "preview": preview,
        }
        attempts.append(attempt)
        if ok:
            selected_pose = {"position": spec["position"], "rpy": spec["rpy"]}
            selected_preview = preview
            break
        if best_failure is None or _preview_error_score(preview) < _preview_error_score(best_failure["preview"]):
            best_failure = attempt
    return {
        "side": side,
        "success": selected_pose is not None,
        "candidate_count": len(pose_specs),
        "attempt_count": len(attempts),
        "max_calls": max_calls,
        "selected_pose": selected_pose,
        "selected_preview": selected_preview,
        "best_failure": best_failure,
        "attempts": attempts[:12],
        "truncated": len(attempts) < len(pose_specs) and selected_pose is None,
    }


def _check_bundlesdf_preview(bundlesdf_url: str, camera: str) -> None:
    preview_url = f"{bundlesdf_url}/preview/{camera}"
    try:
        with urllib.request.urlopen(preview_url, timeout=5.0) as response:
            chunk = response.read(256)
            content_type = response.headers.get("content-type", "")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"BundleSDF preview check failed at {preview_url}: {exc}") from exc
    if "multipart/x-mixed-replace" not in content_type or b"\xff\xd8" not in chunk:
        raise RuntimeError(
            f"BundleSDF preview {preview_url} did not return an MJPEG frame: "
            f"content_type={content_type!r}, first_bytes={chunk[:32]!r}"
        )
    print(f"[green_apple_to_basket] BundleSDF preview ok: {preview_url}")


def _no_motion_preflight(camera: str | None = None) -> None:
    camera = camera or CAMERA
    checks = TASK_RESULT.setdefault("no_motion_service_checks", {})
    if _uses_anygrasp(GRASP_MODE):
        anygrasp_url = _anygrasp_base_url()
        health = _json_get(f"{anygrasp_url}/health")
        checks["anygrasp_health"] = health
        image = np.asarray(get_camera_image(camera))
        if image.ndim != 3 or image.shape[0] < 100 or image.shape[1] < 100:
            raise RuntimeError(f"Camera {camera!r} did not return a usable RGB frame: {image.shape}")
        checks.setdefault("camera_frame_shapes", {})[camera] = list(image.shape)
        return

    bundlesdf_url = _bundlesdf_base_url()
    health = _json_get(f"{bundlesdf_url}/health")
    _check_bundlesdf_preview(bundlesdf_url, camera)
    checks["bundlesdf_health"] = health
    cameras = checks.setdefault("bundlesdf_preview_cameras", [])
    if camera not in cameras:
        cameras.append(camera)


def _require_physical_gate(stage: str) -> None:
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError(
            f"Refusing physical stage {stage!r}. A fresh physical-run ticket and "
            "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 are required."
        )


def _physical_preflight() -> None:
    _require_physical_gate(TASK_RESULT["implemented_stage"])
    if _uses_anygrasp(GRASP_MODE):
        anygrasp_url = _anygrasp_base_url()
        health = _json_get(f"{anygrasp_url}/health")
        if health.get("mock") is True or health.get("safe_for_robot_motion") is False:
            raise RuntimeError(f"Refusing mock or unsafe AnyGrasp for physical motion: {health}")
        TASK_RESULT["physical_block"]["anygrasp_health"] = health
    else:
        _no_motion_preflight(OBJECT_CAMERA)
        if TARGET_CAMERA != OBJECT_CAMERA:
            _no_motion_preflight(TARGET_CAMERA)


def _observe() -> dict[str, Any]:
    print("[green_apple_to_basket] Observe block: perception only; no robot motion.")
    _no_motion_preflight(OBJECT_CAMERA)
    if TARGET_CAMERA != OBJECT_CAMERA:
        _no_motion_preflight(TARGET_CAMERA)
    debug_packet = _capture_debug_observation("observe")

    apple, apple_error = _detect_one_with_debug_fallback(OBJECT_NAME, OBJECT_CAMERA, debug_packet)
    basket, basket_error = _detect_one_with_debug_fallback(TARGET_NAME, TARGET_CAMERA, debug_packet)
    observation = {
        "stage": "observe",
        "physical_motion_executed": False,
        "camera": CAMERA,
        "object_camera": OBJECT_CAMERA,
        "target_camera": TARGET_CAMERA,
        "object_detection": apple,
        "target_detection": basket,
        "detect_errors": {
            OBJECT_NAME: apple_error,
            TARGET_NAME: basket_error,
        },
        "debug_observation": debug_packet,
        "stop_condition": "stop after current perception snapshot and saved run artifacts",
    }
    TASK_RESULT["observations"] = observation
    if apple is None:
        TASK_RESULT["risk_notes"].append("Green apple was not localized; do not plan or move.")
    if basket is None:
        TASK_RESULT["risk_notes"].append("Basket was not localized; placement cannot be planned.")
    return observation


def _verify_goal() -> dict[str, Any]:
    print("[green_apple_to_basket] Final verification block: perception only; no robot motion.")
    observation = _observe()
    apple = observation.get("object_detection")
    basket = observation.get("target_detection")
    verification: dict[str, Any] = {
        "stage": "verify",
        "physical_motion_executed": False,
        "object_detection": apple,
        "target_detection": basket,
        "bbox_margin_px": VERIFY_BBOX_MARGIN_PX,
        "xy_radius_m": VERIFY_XY_RADIUS_M,
        "success": False,
        "reason": "not_checked",
    }
    if apple is None or basket is None:
        verification["reason"] = "apple_or_basket_detection_missing"
        return verification

    apple_center = _bbox_center(apple.get("box_2d"))
    basket_box = basket.get("box_2d")
    bbox_center_inside = _center_inside_bbox(apple_center, basket_box, VERIFY_BBOX_MARGIN_PX)
    xy_distance = _xy_distance_m(apple.get("position_3d"), basket.get("position_3d"))
    xy_close = None if xy_distance is None else bool(xy_distance <= VERIFY_XY_RADIUS_M)
    checks = [value for value in [bbox_center_inside, xy_close] if value is not None]
    verification.update(
        {
            "apple_bbox_center_px": apple_center,
            "basket_box_2d": basket_box,
            "bbox_center_inside_basket": bbox_center_inside,
            "apple_basket_xy_distance_m": xy_distance,
            "xy_close_to_basket": xy_close,
            "success": bool(checks) and all(checks),
        }
    )
    if verification["success"]:
        verification["reason"] = "apple_detection_is_inside_basket_region"
    else:
        verification["reason"] = "apple_detection_not_verified_inside_basket"
    return verification


def _choose_arm(object_xyz: list[float]) -> dict[str, Any]:
    state = get_robot_state()
    obj_pos = np.asarray(object_xyz, dtype=float)
    left_dist = float(np.linalg.norm(obj_pos - np.asarray(state.left_ee_pos, dtype=float)))
    right_dist = float(np.linalg.norm(obj_pos - np.asarray(state.right_ee_pos, dtype=float)))
    preferred = "left" if left_dist <= right_dist else "right"
    return {
        "preferred_side": preferred,
        "left_distance_m": round(left_dist, 4),
        "right_distance_m": round(right_dist, 4),
        "reason": "nearest current end-effector to observed apple",
    }


def _rank_side(side: str, grasps: list[Any]) -> dict[str, Any]:
    candidates = grasps[: min(TOP_GRASP_TRY, len(grasps))]
    if not candidates:
        return {"side": side, "success": False, "error": "no grasp candidates"}
    effective_batch_top_k = max(BATCH_TOP_K, len(candidates))
    result = freespace_move(
        grasp_candidates=candidates,
        batch_side=side,
        batch_top_k=effective_batch_top_k,
        solver_speed=SOLVER_SPEED,
        batch_validate_trajectory=BATCH_VALIDATE_TRAJECTORY,
        preview_only=True,
        planning_speed=PLANNING_SPEED,
        ik_error_threshold=IK_ERROR_THRESHOLD_M,
        ik_rot_threshold_deg=IK_ROT_THRESHOLD_DEG,
        ik_xyz_weight=IK_XYZ_WEIGHT,
        ik_rpy_weight=IK_RPY_WEIGHT,
        planner_backend=MOTION_PLANNER_BACKEND,
    )
    summary = _result_summary(result)
    summary.update(
        {
            "side": side,
            "configured_top_grasp_try": TOP_GRASP_TRY,
            "configured_batch_top_k": BATCH_TOP_K,
            "effective_batch_top_k": effective_batch_top_k,
            "input_candidate_count": int(_obj_value(result, "input_candidate_count", len(candidates)) or 0),
            "evaluated_candidate_count": int(_obj_value(result, "evaluated_candidate_count", 0) or 0),
            "truncated_input_count": int(_obj_value(result, "truncated_input_count", 0) or 0),
            "planning_mode": _obj_value(result, "planning_mode"),
        }
    )
    best = _obj_value(result, "best_candidate")
    feasible_candidates = []
    for row in list(_obj_value(result, "batch_candidates", []) or []):
        if _obj_value(row, "motion_plan_error", True) is not False:
            continue
        feasible = _grasp_to_dict(row, int(_obj_value(row, "rank", 0) or 0))
        feasible["arm"] = side
        feasible["source_index"] = _obj_value(row, "source_index")
        feasible["ik_error_m"] = _obj_value(row, "ik_error_m")
        feasible["ik_rot_error_deg"] = _obj_value(row, "ik_rot_error_deg")
        feasible["planner_status"] = _obj_value(row, "planner_status")
        feasible_candidates.append(feasible)
    summary["feasible_candidates"] = feasible_candidates[:8]
    if best is not None and _obj_value(best, "motion_plan_error", True) is False:
        selected = _grasp_to_dict(best, int(_obj_value(best, "rank", 0) or 0))
        selected["arm"] = side
        selected["trajectory_cache_key"] = _obj_value(best, "trajectory_cache_key")
        summary["selected_grasp"] = selected
        summary["success"] = True
    else:
        failures = []
        for candidate in list(_obj_value(result, "batch_candidates", []) or [])[:5]:
            failures.append(
                {
                    "rank": int(_obj_value(candidate, "rank", 0) or 0),
                    "reason": _obj_value(candidate, "motion_plan_reason"),
                    "planner_status": _obj_value(candidate, "planner_status"),
                }
            )
        summary["success"] = False
        summary["failures"] = failures
    return summary


def _candidate_preview_bundle(
    candidate: dict[str, Any],
    drop_pos: list[float],
    transport_rpy: list[float],
    pregrasp_z_offset_m: float,
    lift_z_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    side = str(candidate["arm"])
    grasp_pos = [float(x) for x in candidate["position"]]
    grasp_rpy = [float(x) for x in candidate["rpy"]]
    pregrasp_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + pregrasp_z_offset_m]
    lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + lift_z_m]
    previews = [
        _single_pose_preview(side, pregrasp_pos, grasp_rpy, "preview_pregrasp"),
        _single_pose_preview(side, grasp_pos, grasp_rpy, "preview_grasp"),
        _single_pose_preview(side, lift_pos, grasp_rpy, "preview_lift"),
        _single_pose_preview(side, drop_pos, transport_rpy, "preview_place"),
    ]
    poses = {
        "pregrasp_pose": {"position": _round_list(pregrasp_pos, 4), "rpy": _round_list(grasp_rpy, 3)},
        "grasp_pose": {"position": _round_list(grasp_pos, 4), "rpy": _round_list(grasp_rpy, 3)},
        "lift_pose": {"position": _round_list(lift_pos, 4), "rpy": _round_list(grasp_rpy, 3)},
        "place_pose": {"position": _round_list(drop_pos, 4), "rpy": _round_list(transport_rpy, 3)},
        "pregrasp_z_offset_m": round(float(pregrasp_z_offset_m), 4),
        "lift_z_m": round(float(lift_z_m), 4),
    }
    return poses, previews


def _single_pose_preview(side: str, position: list[float], rpy: list[float], label: str) -> dict[str, Any]:
    prefix = "left" if side == "left" else "right"
    try:
        result = freespace_move(
            **{
                f"{prefix}_target_pos": [float(x) for x in position],
                f"{prefix}_target_rpy": [float(x) for x in rpy],
                f"{prefix}_gripper": TOPDOWN_GRASP_WIDTH_M,
                "preview_only": True,
                "planning_speed": PLANNING_SPEED,
                "ik_error_threshold": IK_ERROR_THRESHOLD_M,
                "ik_rot_threshold_deg": IK_ROT_THRESHOLD_DEG,
                "ik_xyz_weight": IK_XYZ_WEIGHT,
                "ik_rpy_weight": IK_RPY_WEIGHT,
                "planner_backend": MOTION_PLANNER_BACKEND,
                "solver_speed": SOLVER_SPEED,
            }
        )
        summary = _result_summary(result)
    except Exception as exc:
        summary = {
            "status": "Error",
            "executed": False,
            "side": side,
            "trajectory_steps": 0,
            "trajectory_cache_key": None,
            "final_pos_error_m": None,
            "final_rot_error_deg": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    summary.update(
        {
            "label": label,
            "position": _round_list(position, 4),
            "rpy": _round_list(rpy, 3),
            "preview_only": True,
        }
    )
    return summary


def _plan() -> dict[str, Any]:
    observation = _read_place_source_observation() or _observe()
    apple = observation.get("object_detection")
    basket = observation.get("target_detection")
    if apple is None or basket is None:
        return {
            "stage": "plan",
            "success": False,
            "physical_motion_executed": False,
            "error": "apple_or_basket_detection_missing",
        }

    print("[green_apple_to_basket] Plan block: cuRobo previews only; no gripper close or motion.")
    if observation.get("source") == "place_search_source_observe_result":
        print(
            "[green_apple_to_basket] Using source observe result for apple/basket geometry: "
            f"{observation.get('source_result_json')}"
        )
        grasps = _sample_topdown_from_observation(apple)
    else:
        grasps = grasp_geometry.sample_topdown_geometric(
            OBJECT_NAME,
            camera=OBJECT_CAMERA,
            max_grasps=MAX_GRASPS,
            yaws=TOPDOWN_GRASP_YAWS,
            pitches=TOPDOWN_GRASP_PITCHES,
            z_offset_m=TOPDOWN_GRASP_Z_OFFSET_M,
            width=TOPDOWN_GRASP_WIDTH_M,
        )
    arm_choice = _choose_arm(apple["position_3d"])
    sides = [arm_choice["preferred_side"], "right" if arm_choice["preferred_side"] == "left" else "left"]
    side_results = [_rank_side(side, grasps) for side in sides]
    basket_xyz = [float(x) for x in basket["position_3d"][:3]]
    drop_pos = [basket_xyz[0], basket_xyz[1], basket_xyz[2] + TARGET_DROP_Z_OFFSET]
    candidate_pool: list[dict[str, Any]] = []
    seen_candidates: set[tuple[str, tuple[float, ...], tuple[float, ...]]] = set()
    for side_result in side_results:
        for candidate in list(side_result.get("feasible_candidates", []) or []):
            key = (
                str(candidate.get("arm")),
                tuple(float(x) for x in candidate.get("position", [])),
                tuple(float(x) for x in candidate.get("rpy", [])),
            )
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidate_pool.append(candidate)
    candidate_pool = candidate_pool[: max(1, PLACE_MAX_CANDIDATES)]

    selected = None
    selected_previews: list[dict[str, Any]] = []
    selected_poses: dict[str, Any] = {}
    candidate_preview_attempts = []
    place_search_by_side: dict[str, Any] = {}
    place_preview_calls_used = 0

    def _place_search_for_side(side: str) -> dict[str, Any]:
        nonlocal place_preview_calls_used
        if side not in place_search_by_side:
            remaining = max(0, PLACE_MAX_PREVIEW_CALLS - place_preview_calls_used)
            search = _search_place_pose(side, basket_xyz, remaining)
            place_preview_calls_used += int(search.get("attempt_count", 0) or 0)
            place_search_by_side[side] = search
        return place_search_by_side[side]

    for candidate in candidate_pool:
        side = str(candidate["arm"])
        place_search = _place_search_for_side(side)
        place_pose = place_search.get("selected_pose")
        if not isinstance(place_pose, dict):
            candidate_preview_attempts.append(
                {
                    "candidate": candidate,
                    "success": False,
                    "failure_stage": "preview_place",
                    "place_search_success": False,
                    "best_place_failure": place_search.get("best_failure"),
                }
            )
            continue
        for pregrasp_z in _parse_offsets(PREGRASP_Z_OFFSETS_M, [PREGRASP_Z_OFFSET_M]):
            for lift_z in _parse_offsets(LIFT_Z_OFFSETS_M, [LIFT_Z_M]):
                poses, previews = _candidate_preview_bundle(
                    candidate,
                    [float(x) for x in place_pose["position"][:3]],
                    [float(x) for x in place_pose["rpy"][:3]],
                    pregrasp_z,
                    lift_z,
                )
                preview_ok = all(
                    item.get("status") == "Success" and not item.get("executed")
                    for item in previews
                )
                candidate_preview_attempts.append(
                    {
                        "candidate": candidate,
                        "pregrasp_z_offset_m": round(float(pregrasp_z), 4),
                        "lift_z_m": round(float(lift_z), 4),
                        "place_pose": place_pose,
                        "success": preview_ok,
                        "previews": previews,
                    }
                )
                if preview_ok:
                    selected = candidate
                    selected_previews = previews
                    selected_poses = poses
                    break
            if selected is not None:
                break
        if selected is not None:
            break

    best_place_failure = None
    for search in place_search_by_side.values():
        failure = search.get("best_failure")
        if not failure:
            continue
        if best_place_failure is None or _preview_error_score(failure["preview"]) < _preview_error_score(best_place_failure["preview"]):
            best_place_failure = failure
    online_close_estimate = _online_close_estimate(observation, selected) if selected is not None else None
    close_readiness_plan = {
        "selected_grasp": selected,
        "online_close_estimate": online_close_estimate,
    }
    close_readiness = _width_limited_close_readiness(close_readiness_plan) if selected is not None else {
        "ready": False,
        "reason": "no selected grasp",
    }

    plan: dict[str, Any] = {
        "stage": "plan",
        "success": selected is not None,
        "physical_motion_executed": False,
        "grasp_mode": GRASP_MODE,
        "max_grasps": MAX_GRASPS,
        "top_grasp_try": TOP_GRASP_TRY,
        "batch_top_k": BATCH_TOP_K,
        "topdown_grasp_pitches": TOPDOWN_GRASP_PITCHES,
        "topdown_grasp_yaws": TOPDOWN_GRASP_YAWS,
        "object_camera": OBJECT_CAMERA,
        "target_camera": TARGET_CAMERA,
        "arm_choice": arm_choice,
        "candidate_count": len(grasps),
        "candidate_preview": [_grasp_to_dict(grasp, index) for index, grasp in enumerate(grasps[:5], start=1)],
        "side_results": side_results,
        "selected_grasp": selected,
        "online_close_estimate": online_close_estimate,
        "width_limited_close_readiness": close_readiness,
        "observation_source": observation.get("source", "fresh_plan_observe"),
        "observation_source_result_json": observation.get("source_result_json"),
        "guarded_preview_selection": {
            "success": selected is not None,
            "attempt_count": len(candidate_preview_attempts),
            "attempts": candidate_preview_attempts[:8],
        },
        "place_search": {
            "success": any(search.get("success") for search in place_search_by_side.values()),
            "preview_calls_used": place_preview_calls_used,
            "by_side": place_search_by_side,
            "best_failure": best_place_failure,
        },
        "basket_drop_pose": {
            "position": _round_list(drop_pos, 4),
            "z_offset_m": TARGET_DROP_Z_OFFSET,
            "source": "basket_position_3d_plus_offset",
        },
        "motion_previews": [],
        "stop_condition": "stop after ranking grasp/place feasibility; no arm move or gripper close",
    }
    if selected is not None:
        plan.update(selected_poses)
        plan["motion_previews"] = selected_previews
    else:
        plan["error"] = "no_candidate_passed_guarded_pregrasp_grasp_lift_place_previews"
    try:
        debug_packet = observation.get("debug_observation")
        packet_path = save_plan_packet(
            stage="plan",
            selected=selected,
            candidates=candidate_pool[:16],
            previews=selected_previews,
            observation=debug_packet if isinstance(debug_packet, dict) else None,
        )
        plan["debug_plan_packet"] = str(packet_path)
    except Exception as exc:
        plan["debug_plan_packet_error"] = f"{type(exc).__name__}: {exc}"
    return plan


def _resume_plan_check() -> dict[str, Any]:
    print("[green_apple_to_basket] Resume-plan block: cached geometry and previews only; no robot motion.")
    plan = _load_cached_plan()
    if plan is None:
        raise RuntimeError("Resume-plan check requires OPENFORGE_APPLE_PLAN_RESULT_JSON or OPENFORGE_APPLE_PLAN_LOG_DIR.")

    observation = _observe()
    plan, recenter = _resume_recenter_plan_from_observation(plan, observation)
    selected = plan.get("selected_grasp")
    if not isinstance(selected, dict) or not selected.get("arm"):
        raise RuntimeError("Cached plan missing selected_grasp.arm")
    side = str(selected["arm"])
    preview_specs = [
        ("grasp_pose", "preview_resume_grasp"),
        ("lift_pose", "preview_resume_lift"),
        ("place_pose", "preview_resume_place"),
    ]
    previews = [
        _single_pose_preview(side, _pose_position(plan, pose_name), _pose_rpy(plan, pose_name), label)
        for pose_name, label in preview_specs
    ]
    preview_ok = all(item.get("status") == "Success" and not item.get("executed") for item in previews)
    coherence = _resume_coherence(observation, plan)
    return {
        "stage": "resume_plan",
        "success": bool(preview_ok and coherence.get("success")),
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "plan": plan,
        "observation": observation,
        "coherence": coherence,
        "resume_recenter": recenter,
        "remaining_motion_previews": previews,
        "stop_condition": "stop after cached-plan coherence and no-motion previews; no arm move or gripper close",
    }


def _execute_move(side: str, position: list[float], rpy: list[float], label: str) -> dict[str, Any]:
    prefix = "left" if side == "left" else "right"
    result = freespace_move(
        **{
            f"{prefix}_target_pos": [float(x) for x in position],
            f"{prefix}_target_rpy": [float(x) for x in rpy],
            f"{prefix}_gripper": TOPDOWN_GRASP_WIDTH_M,
            "preview_only": False,
            "planning_speed": PLANNING_SPEED,
            "ik_error_threshold": IK_ERROR_THRESHOLD_M,
            "ik_rot_threshold_deg": IK_ROT_THRESHOLD_DEG,
            "ik_xyz_weight": IK_XYZ_WEIGHT,
            "ik_rpy_weight": IK_RPY_WEIGHT,
            "planner_backend": MOTION_PLANNER_BACKEND,
            "solver_speed": SOLVER_SPEED,
        }
    )
    summary = _result_summary(result)
    summary.update({"label": label, "position": _round_list(position, 4), "rpy": _round_list(rpy, 3)})
    if summary["status"] != "Success" or not summary["executed"]:
        raise RuntimeError(f"{label} failed: {summary}")
    return summary


def _run_physical_block(stage: str) -> dict[str, Any]:
    _physical_preflight()
    plan = _load_cached_plan() or _plan()
    if not plan.get("success"):
        raise RuntimeError(f"Refusing physical {stage}: plan did not pass: {plan.get('error')}")
    selected = plan["selected_grasp"]
    side = str(selected["arm"])
    block = {
        "stage": stage,
        "side": side,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "moves": [],
        "gripper": {},
        "plan": plan,
        "evidence_required_before_next_stage": "inspect result/run log/videos and run a no-motion observe/verification block if pose may have changed",
        "success": False,
    }
    close_config: dict[str, Any] | None = None
    if stage in {"pregrasp", "grasp_close"}:
        close_config = _resolve_width_limited_close(plan)
        block["gripper"]["close_config"] = close_config
    if stage == "pregrasp":
        block["movement_capable_calls"].append("set_gripper")
        try:
            block["gripper"]["open"] = _run_checked_open(side)
        except Exception as exc:
            block["physical_motion_executed"] = True
            block["success"] = False
            block["error"] = f"{type(exc).__name__}: {exc}"
            TASK_RESULT["physical_block"] = block
            TASK_RESULT["plan"] = dict(block.get("plan", {}))
            TASK_RESULT["physical_motion_executed"] = True
            TASK_RESULT["movement_capable_calls"] = list(block.get("movement_capable_calls", []))
            raise
        block["moves"].append(
            _execute_move(
                side,
                plan["pregrasp_pose"]["position"],
                plan["pregrasp_pose"]["rpy"],
                "move_pregrasp",
            )
        )
        block["movement_capable_calls"].append("freespace_move")
    elif stage == "grasp_close":
        assert close_config is not None
        block["gripper"]["pre_descent_open_check"] = _require_gripper_open_for_descent(side, stage)
        block["moves"].append(
            _execute_move(side, plan["grasp_pose"]["position"], plan["grasp_pose"]["rpy"], "move_grasp")
        )
        block["movement_capable_calls"].append("freespace_move")
        try:
            close_summary = _run_width_limited_close(side, close_config)
        except Exception as exc:
            block["physical_motion_executed"] = bool(block["movement_capable_calls"])
            block["success"] = False
            block["error"] = f"{type(exc).__name__}: {exc}"
            TASK_RESULT["physical_block"] = block
            TASK_RESULT["plan"] = dict(block.get("plan", {}))
            TASK_RESULT["physical_motion_executed"] = bool(block.get("physical_motion_executed"))
            TASK_RESULT["movement_capable_calls"] = list(block.get("movement_capable_calls", []))
            raise
        block["movement_capable_calls"].extend(["set_gripper"] * len(close_summary["steps"]))
        block["gripper"].update(close_summary)
        artifact_path = _save_gripper_close_artifact(close_summary)
        if artifact_path:
            block["gripper"]["artifact_path"] = artifact_path
    elif stage == "lift":
        block["hold_check"] = _post_close_hold_check(plan)
        block["moves"].append(
            _execute_move(side, plan["lift_pose"]["position"], plan["lift_pose"]["rpy"], "move_lift")
        )
        block["movement_capable_calls"].append("freespace_move")
    elif stage == "place":
        block["post_lift_hold_check"] = _post_lift_hold_check(plan)
        block["moves"].append(
            _execute_move(side, plan["place_pose"]["position"], plan["place_pose"]["rpy"], "move_place")
        )
        block["movement_capable_calls"].append("freespace_move")
        open_gripper(side)
        block["movement_capable_calls"].append("open_gripper")
        block["gripper"]["released"] = True
        if _truthy_env("OPENFORGE_GO_HOME_AFTER_PLACE"):
            go_home()
            block["movement_capable_calls"].append("go_home")
    else:
        raise RuntimeError(f"Unsupported physical block: {stage}")
    block["physical_motion_executed"] = bool(block["movement_capable_calls"])
    block["success"] = True
    return block


def _run_all() -> bool:
    raise RuntimeError(
        "Monolithic stage 'all' is disabled for real YAM apple-to-basket runs. "
        "Use block-by-block stages: observe, plan, pregrasp, grasp_close, lift, "
        "place, then final observe/verification. Command completion is not task "
        "success without post-action perception."
    )


def _main() -> None:
    stage = TASK_RESULT["implemented_stage"]
    print("[green_apple_to_basket] Starting staged task.")
    print(f"[green_apple_to_basket] stage={REQUESTED_STAGE!r} -> {stage!r}")
    print(f"[green_apple_to_basket] object={OBJECT_NAME!r}, target={TARGET_NAME!r}")
    print(
        f"[green_apple_to_basket] grasp_mode={GRASP_MODE!r}, camera={CAMERA!r}, "
        f"object_camera={OBJECT_CAMERA!r}, target_camera={TARGET_CAMERA!r}"
    )

    if _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION") and stage not in PHYSICAL_STAGES:
        TASK_RESULT["risk_notes"].append(
            "OPENFORGE_ALLOW_PHYSICAL_MOTION was set for a no-motion stage; no movement-capable calls were made."
        )

    try:
        if stage == "observe":
            observation = _observe()
            TASK_RESULT["success"] = observation.get("object_detection") is not None and observation.get("target_detection") is not None
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "Observe-only block completed; no arm move or gripper close was run."
        elif stage == "plan":
            plan = _plan()
            TASK_RESULT["plan"] = plan
            TASK_RESULT["success"] = bool(plan.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "Plan-only block completed; no physical motion or gripper close was run."
        elif stage == "resume_plan":
            resume = _resume_plan_check()
            TASK_RESULT["resume_plan"] = resume
            TASK_RESULT["plan"] = dict(resume.get("plan", {}))
            TASK_RESULT["success"] = bool(resume.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "Resume-plan block completed; cached plan and remaining previews were checked without physical motion."
        elif stage == "verify":
            verification = _verify_goal()
            TASK_RESULT["verification"] = verification
            TASK_RESULT["success"] = bool(verification.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "Final verification block completed; success requires apple-in-basket perception evidence."
        elif stage in {"pregrasp", "grasp_close", "lift", "place"}:
            block = _run_physical_block(stage)
            TASK_RESULT["physical_block"] = block
            TASK_RESULT["plan"] = dict(block.get("plan", {}))
            TASK_RESULT["physical_motion_executed"] = bool(block.get("physical_motion_executed"))
            TASK_RESULT["movement_capable_calls"] = list(block.get("movement_capable_calls", []))
            TASK_RESULT["success"] = bool(block.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = f"Physical block {stage!r} completed; re-observe before the next block."
        elif stage == "all":
            _run_all()
        else:
            TASK_RESULT["why_stopped"] = (
                "Unsupported stage. Use observe, plan, resume_plan, pregrasp, grasp_close, lift, place, or verify."
            )
    except Exception as exc:
        TASK_RESULT["success"] = False
        TASK_RESULT["reward"] = 0.0
        TASK_RESULT["why_stopped"] = f"Stage {stage!r} failed: {type(exc).__name__}: {exc}"
        TASK_RESULT["risk_notes"].append("Inspect result.json/run log before retrying or escalating.")
        print(f"[green_apple_to_basket] {TASK_RESULT['why_stopped']}")
        print(json.dumps(TASK_RESULT, indent=2, default=str))
        raise

    if stage in {"observe", "plan", "resume_plan", "verify"}:
        TASK_RESULT["physical_motion_executed"] = False
        TASK_RESULT["movement_capable_calls"] = []
    print(f"[green_apple_to_basket] {TASK_RESULT['why_stopped']}")
    print(json.dumps(TASK_RESULT, indent=2, default=str))


_main()
