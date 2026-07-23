# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-gated drawer handle pull-open / push-close planning for YAM.

This first implementation is intentionally no-motion-first.  It can observe,
select a handle detection, estimate a tentative pull axis, and preview candidate
poses.  Physical contact stages are guarded and currently refuse to execute
until a drawer-specific physical ticket and contact strategy are supplied.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from skill_library.debug_observation import (
    capture_observation,
    current_run_dir,
    save_plan_packet,
    write_stage_summary,
)
from skill_library.namespace import (
    detect_objects_oneshot,
    freespace_move,
    get_robot_state,
    set_gripper,
)


PROMPTS = [
    part.strip()
    for part in os.environ.get(
        "OPENFORGE_DRAWER_PROMPTS",
        os.environ.get("OPENFORGE_DEBUG_OBS_PROMPTS", "drawer handle,drawer,cabinet handle"),
    ).split(",")
    if part.strip()
]
CAMERAS = [
    part.strip()
    for part in os.environ.get(
        "OPENFORGE_DEBUG_OBS_CAMERAS",
        "top,left,right,bottom",
    ).split(",")
    if part.strip()
]
CALIBRATED_MOTION_CAMERAS = {"top", "left", "right"}
CAMERA_PORTAL = os.environ.get("OPENFORGE_DEBUG_OBS_CAMERA_PORTAL", "").strip()
DEBUG_OBS_TIMEOUT_S = float(os.environ.get("OPENFORGE_DEBUG_OBS_TIMEOUT_S", "8"))
CAPTURE_STATE = os.environ.get("OPENFORGE_DEBUG_OBS_CAPTURE_STATE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

PREGRASP_STANDOFF_M = float(os.environ.get("OPENFORGE_DRAWER_PREGRASP_STANDOFF_M", "0.08"))
HANDLE_CONTACT_OFFSET_M = float(os.environ.get("OPENFORGE_DRAWER_HANDLE_CONTACT_OFFSET_M", "0.015"))
PULL_DISTANCE_M = float(os.environ.get("OPENFORGE_DRAWER_PULL_DISTANCE_M", "0.03"))
PUSH_DISTANCE_M = float(os.environ.get("OPENFORGE_DRAWER_PUSH_DISTANCE_M", str(PULL_DISTANCE_M)))
MAX_INCREMENTAL_TRAVEL_M = float(os.environ.get("OPENFORGE_DRAWER_MAX_INCREMENTAL_TRAVEL_M", "0.03"))
HANDLE_MIN_SCORE = float(os.environ.get("OPENFORGE_DRAWER_HANDLE_MIN_SCORE", "0.45"))
PLAN_MAX_CANDIDATES = int(os.environ.get("OPENFORGE_DRAWER_PLAN_MAX_CANDIDATES", "8"))
APPROACH_OFFSETS_DEG = os.environ.get("OPENFORGE_DRAWER_APPROACH_OFFSETS_DEG", "0,-15,15,-30,30")
Z_OFFSETS_M = os.environ.get("OPENFORGE_DRAWER_Z_OFFSETS_M", "0,0.015,-0.015")
WRIST_ROLLS_DEG = os.environ.get("OPENFORGE_DRAWER_WRIST_ROLLS_DEG", "0,-20,20,-45,45")
PLANNING_SPEED = float(os.environ.get("OPENFORGE_DRAWER_PLANNING_SPEED", "0.25"))
IK_ERROR_THRESHOLD_M = float(os.environ.get("OPENFORGE_DRAWER_IK_ERROR_THRESHOLD_M", "0.015"))
IK_ROT_THRESHOLD_DEG = float(os.environ.get("OPENFORGE_DRAWER_IK_ROT_THRESHOLD_DEG", "8.0"))
IK_XYZ_WEIGHT = float(os.environ.get("OPENFORGE_DRAWER_IK_XYZ_WEIGHT", "1.0"))
IK_RPY_WEIGHT = float(os.environ.get("OPENFORGE_DRAWER_IK_RPY_WEIGHT", "0.25"))
PLANNER_BACKEND = os.environ.get("OPENFORGE_DRAWER_PLANNER_BACKEND", "curobo").strip()
SOLVER_SPEED = os.environ.get("OPENFORGE_DRAWER_SOLVER_SPEED", "fast").strip().lower()
PLAN_JSON = os.environ.get("OPENFORGE_DRAWER_PLAN_JSON", "").strip()
PLAN_LOG_DIR = os.environ.get("OPENFORGE_DRAWER_PLAN_LOG_DIR", "").strip()
GRIPPER_OPEN_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_OPEN_POS", "1.0"))
GRIPPER_SCOUT_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_SCOUT_POS", "0.65"))
GRIPPER_TARGET_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_TARGET_POS", "0.45"))
GRIPPER_MIN_SAFE_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_MIN_SAFE_POS", "0.12"))
GRIPPER_VEL_LIMIT = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_VEL_LIMIT", "0.6"))
GRIPPER_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_TORQUE_LIMIT", "0.12"))
GRIPPER_OPEN_VEL_LIMIT = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_OPEN_VEL_LIMIT", "1.0"))
GRIPPER_OPEN_TORQUE_LIMIT = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_OPEN_TORQUE_LIMIT", "0.30"))
GRIPPER_OPEN_CONFIRM_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_OPEN_CONFIRM_POS", "0.55"))
GRIPPER_OPEN_CONFIRM_TIMEOUT_S = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_OPEN_CONFIRM_TIMEOUT_S", "4.0"))
GRIPPER_OPEN_CONFIRM_POLL_S = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_OPEN_CONFIRM_POLL_S", "0.2"))
GRIPPER_SETTLE_S = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_SETTLE_S", "0.4"))
GRIPPER_CLOSE_CONFIRM_TIMEOUT_S = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_CLOSE_CONFIRM_TIMEOUT_S", "2.0"))
GRIPPER_CLOSE_CONFIRM_POLL_S = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_CLOSE_CONFIRM_POLL_S", "0.2"))
GRIPPER_CLOSE_MIN_DELTA_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_CLOSE_MIN_DELTA_POS", "0.05"))
GRIPPER_GRASP_MAX_OPEN_POS = float(os.environ.get("OPENFORGE_DRAWER_GRIPPER_GRASP_MAX_OPEN_POS", "0.90"))
RELEASE_LIFT_Z_CANDIDATES_M = os.environ.get("OPENFORGE_DRAWER_RELEASE_LIFT_Z_CANDIDATES_M", "0.03,0.06,0.09")
RELEASE_PULL_CLEARANCE_CANDIDATES_M = os.environ.get("OPENFORGE_DRAWER_RELEASE_PULL_CLEARANCE_CANDIDATES_M", "0.02,0.04")

REQUESTED_STAGE = os.environ.get(
    "OPENFORGE_DRAWER_STAGE",
    os.environ.get("OPENFORGE_TASK_STAGE", "observe"),
).strip()
STAGE_ALIASES = {
    "observe": "observe",
    "observe_only": "observe",
    "observe-only": "observe",
    "plan": "plan_handle_grasp",
    "plan_handle_grasp": "plan_handle_grasp",
    "plan-handle-grasp": "plan_handle_grasp",
    "preview": "plan_handle_grasp",
    "plan_release_retreat": "plan_release_retreat",
    "plan-release-retreat": "plan_release_retreat",
    "plan_retreat": "plan_release_retreat",
    "plan-retreat": "plan_release_retreat",
    "verify": "verify",
    "final_observe": "verify",
    "final-observe": "verify",
    "pregrasp": "pregrasp",
    "grasp_handle": "grasp_handle",
    "grasp-handle": "grasp_handle",
    "pull_open_small": "pull_open_small",
    "pull-open-small": "pull_open_small",
    "push_close_small": "push_close_small",
    "push-close-small": "push_close_small",
    "release": "release",
    "all": "all",
}
PHYSICAL_STAGES = {
    "pregrasp",
    "grasp_handle",
    "pull_open_small",
    "push_close_small",
    "release",
}

TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "drawer_pull_push",
    "requested_stage": REQUESTED_STAGE,
    "implemented_stage": STAGE_ALIASES.get(REQUESTED_STAGE, "unsupported"),
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "prompts": PROMPTS,
    "cameras": CAMERAS,
    "calibrated_motion_cameras": sorted(CALIBRATED_MOTION_CAMERAS),
    "config": {
        "pregrasp_standoff_m": PREGRASP_STANDOFF_M,
        "handle_contact_offset_m": HANDLE_CONTACT_OFFSET_M,
        "pull_distance_m": PULL_DISTANCE_M,
        "push_distance_m": PUSH_DISTANCE_M,
        "max_incremental_travel_m": MAX_INCREMENTAL_TRAVEL_M,
        "handle_min_score": HANDLE_MIN_SCORE,
        "plan_max_candidates": PLAN_MAX_CANDIDATES,
        "approach_offsets_deg": APPROACH_OFFSETS_DEG,
        "z_offsets_m": Z_OFFSETS_M,
        "wrist_rolls_deg": WRIST_ROLLS_DEG,
        "planning_speed": PLANNING_SPEED,
        "ik_error_threshold_m": IK_ERROR_THRESHOLD_M,
        "ik_rot_threshold_deg": IK_ROT_THRESHOLD_DEG,
        "planner_backend": PLANNER_BACKEND,
        "solver_speed": SOLVER_SPEED,
        "plan_json": PLAN_JSON or None,
        "plan_log_dir": PLAN_LOG_DIR or None,
        "gripper": {
            "open_pos": GRIPPER_OPEN_POS,
            "scout_pos": GRIPPER_SCOUT_POS,
            "target_pos": GRIPPER_TARGET_POS,
            "min_safe_pos": GRIPPER_MIN_SAFE_POS,
            "vel_limit": GRIPPER_VEL_LIMIT,
            "torque_limit": GRIPPER_TORQUE_LIMIT,
            "open_vel_limit": GRIPPER_OPEN_VEL_LIMIT,
            "open_torque_limit": GRIPPER_OPEN_TORQUE_LIMIT,
            "open_confirm_pos": GRIPPER_OPEN_CONFIRM_POS,
            "open_confirm_timeout_s": GRIPPER_OPEN_CONFIRM_TIMEOUT_S,
            "open_confirm_poll_s": GRIPPER_OPEN_CONFIRM_POLL_S,
            "close_confirm_timeout_s": GRIPPER_CLOSE_CONFIRM_TIMEOUT_S,
            "close_confirm_poll_s": GRIPPER_CLOSE_CONFIRM_POLL_S,
            "close_min_delta_pos": GRIPPER_CLOSE_MIN_DELTA_POS,
            "grasp_max_open_pos": GRIPPER_GRASP_MAX_OPEN_POS,
        },
        "release_lift_z_candidates_m": RELEASE_LIFT_Z_CANDIDATES_M,
        "release_pull_clearance_candidates_m": RELEASE_PULL_CLEARANCE_CANDIDATES_M,
    },
    "risk_notes": [
        "Drawer physical stages require a current drawer-specific ticket; no env gate alone is sufficient.",
        "Bottom camera is diagnostic only and is never used for robot/world motion poses.",
        "Drawer contact stages require incremental low-displacement moves, verified handle pose, pull axis, clearance, and travel-limit evidence.",
        "Drawer physical stages additionally require OPENFORGE_DRAWER_ENABLE_PHYSICAL_CONTACT=1.",
    ],
}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _round_list(values: Any, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def _portal_camera(camera: str = "top") -> Any:
    import portal

    return portal.Client(CAMERA_PORTAL).get_camera_image(camera).result()


def _capture(stage: str) -> dict[str, Any]:
    packet = capture_observation(
        stage=stage,
        prompts=PROMPTS,
        cameras=CAMERAS,
        detect_fn=detect_objects_oneshot,
        get_camera_fn=_portal_camera if CAMERA_PORTAL else None,
        get_robot_state_fn=get_robot_state if CAPTURE_STATE else None,
        capture_robot_state=CAPTURE_STATE,
        per_call_timeout_s=DEBUG_OBS_TIMEOUT_S,
    )
    TASK_RESULT.setdefault("artifacts", {})["observation_packet"] = packet.get("packet_path")
    TASK_RESULT.setdefault("artifacts", {})["stage_summary"] = str(current_run_dir() / "stage_summary.md")
    return packet


def _load_observation_source() -> dict[str, Any] | None:
    raw_path = os.environ.get("OPENFORGE_DRAWER_OBSERVATION_PACKET", "").strip()
    if not raw_path:
        raw_path = os.environ.get("OPENFORGE_DRAWER_OBSERVATION_JSON", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        path = current_run_dir() / raw_path
    data = json.loads(path.read_text(encoding="utf-8"))
    data["packet_path"] = str(path)
    return data


def _detections(packet: dict[str, Any], camera: str, prompt: str) -> list[dict[str, Any]]:
    cam = packet.get("cameras", {}).get(camera, {})
    dets = cam.get("detections", {}).get(prompt, [])
    return dets if isinstance(dets, list) else []


def _score(det: dict[str, Any]) -> float:
    try:
        return float(det.get("score") or 0.0)
    except Exception:
        return 0.0


def _xyz(det: dict[str, Any]) -> list[float] | None:
    value = det.get("position_3d")
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _best_detection(packet: dict[str, Any], prompts: list[str], cameras: list[str]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    prompt_rank = {prompt: idx for idx, prompt in enumerate(prompts)}
    for camera in cameras:
        if camera not in CALIBRATED_MOTION_CAMERAS:
            continue
        for prompt in prompts:
            for det in _detections(packet, camera, prompt):
                xyz = _xyz(det)
                if xyz is None:
                    continue
                candidates.append(
                    {
                        "camera": camera,
                        "prompt": prompt,
                        "label": det.get("label", prompt),
                        "score": _score(det),
                        "box_2d": det.get("box_2d"),
                        "position_3d": xyz,
                        "prompt_rank": prompt_rank.get(prompt, 99),
                    }
                )
    candidates.sort(key=lambda item: (item["prompt_rank"], -item["score"]))
    return candidates[0] if candidates else None


def _select_handle_and_drawer(packet: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    handle = _best_detection(packet, ["drawer handle", "cabinet handle"], CAMERAS)
    drawer = _best_detection(packet, ["drawer"], CAMERAS)
    if handle is None:
        notes.append("No calibrated-camera detection for 'drawer handle' or 'cabinet handle'.")
    elif handle["score"] < HANDLE_MIN_SCORE:
        notes.append(
            f"Best handle score {handle['score']:.3f} is below OPENFORGE_DRAWER_HANDLE_MIN_SCORE={HANDLE_MIN_SCORE:.3f}."
        )
    if drawer is None:
        notes.append("No calibrated-camera drawer-body detection; pull-axis confidence will be low.")
    return handle, drawer, notes


def _state_arm_positions() -> dict[str, list[float]]:
    try:
        state = get_robot_state()
        return {
            "left": [float(x) for x in state.left_ee_pos[:3]],
            "right": [float(x) for x in state.right_ee_pos[:3]],
        }
    except Exception as exc:
        TASK_RESULT["risk_notes"].append(f"get_robot_state unavailable for arm ranking: {type(exc).__name__}: {exc}")
        return {
            "left": [0.4975, 0.31, 0.914],
            "right": [0.4975, -0.31, 0.914],
        }


def _choose_arm(handle_xyz: list[float]) -> tuple[str, str]:
    arms = _state_arm_positions()
    handle = np.asarray(handle_xyz, dtype=np.float64)
    distances = {
        side: float(np.linalg.norm(handle - np.asarray(pos, dtype=np.float64)))
        for side, pos in arms.items()
    }
    side = "left" if distances["left"] <= distances["right"] else "right"
    return side, f"nearest current EE: left={distances['left']:.3f}m right={distances['right']:.3f}m"


def _unit_xy(vec: np.ndarray) -> np.ndarray | None:
    xy = np.asarray([float(vec[0]), float(vec[1]), 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(xy))
    if norm < 1e-6:
        return None
    return xy / norm


def _parse_axis_override() -> np.ndarray | None:
    raw = os.environ.get("OPENFORGE_DRAWER_PULL_AXIS_XY", "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("OPENFORGE_DRAWER_PULL_AXIS_XY must be 'x,y' in world frame")
    axis = _unit_xy(np.asarray([float(parts[0]), float(parts[1]), 0.0], dtype=np.float64))
    if axis is None:
        raise ValueError("OPENFORGE_DRAWER_PULL_AXIS_XY has near-zero length")
    return axis


def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values or list(default)


def _rotate_xy(unit: np.ndarray, deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)
    return np.asarray(
        [
            float(unit[0]) * cos_t - float(unit[1]) * sin_t,
            float(unit[0]) * sin_t + float(unit[1]) * cos_t,
            0.0,
        ],
        dtype=np.float64,
    )


def _estimate_pull_axis(handle: dict[str, Any], drawer: dict[str, Any] | None) -> dict[str, Any]:
    override = _parse_axis_override()
    if override is not None:
        return {
            "pull_axis_world": _round_list(override, 5),
            "source": "OPENFORGE_DRAWER_PULL_AXIS_XY",
            "confidence": "operator_configured",
            "requires_review_before_physical": True,
        }

    handle_xyz = np.asarray(handle["position_3d"], dtype=np.float64)
    if drawer is not None and drawer.get("position_3d"):
        drawer_xyz = np.asarray(drawer["position_3d"], dtype=np.float64)
        axis = _unit_xy(handle_xyz - drawer_xyz)
        if axis is not None:
            return {
                "pull_axis_world": _round_list(axis, 5),
                "source": "handle_minus_drawer_center_xy",
                "confidence": "tentative",
                "requires_review_before_physical": True,
                "drawer_center_world": _round_list(drawer_xyz),
            }

    fallback = np.asarray([0.0, 1.0 if handle_xyz[1] >= 0 else -1.0, 0.0], dtype=np.float64)
    return {
        "pull_axis_world": _round_list(fallback, 5),
        "source": "workspace_y_sign_fallback",
        "confidence": "low",
        "requires_review_before_physical": True,
    }


def _display_rpy_from_rotation_matrix(rotation_matrix: Any) -> list[float]:
    from scipy.spatial.transform import Rotation

    euler_xyz = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_euler(
        "xyz",
        degrees=True,
    )
    display = np.array([euler_xyz[1], -euler_xyz[0], -euler_xyz[2] - 90.0], dtype=np.float64)
    display = (display + 180.0) % 360.0 - 180.0
    return [float(x) for x in display]


def _side_grasp_display_rpy_from_approach(
    approach_dir: list[float],
    wrist_roll_deg: float,
) -> tuple[list[float], dict[str, list[float]]]:
    approach = np.asarray(approach_dir, dtype=np.float64)
    norm = float(np.linalg.norm(approach))
    if norm < 1e-6:
        raise RuntimeError(f"Invalid drawer approach direction: {approach_dir!r}")
    z_axis = approach / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(world_up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    if abs(float(wrist_roll_deg)) > 1e-6:
        roll_rad = math.radians(float(wrist_roll_deg))
        cos_t = math.cos(roll_rad)
        sin_t = math.sin(roll_rad)
        x_base = x_axis
        y_base = y_axis
        x_axis = cos_t * x_base + sin_t * y_base
        y_axis = -sin_t * x_base + cos_t * y_base
        x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = y_axis / float(np.linalg.norm(y_axis))
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return (
        _display_rpy_from_rotation_matrix(rotation_matrix),
        {
            "local_x_opening_axis": _round_list(x_axis, 5),
            "local_y_height_axis": _round_list(y_axis, 5),
            "local_z_approach_axis": _round_list(z_axis, 5),
        },
    )


def _candidate_poses(handle: dict[str, Any], drawer: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    handle_xyz = np.asarray(handle["position_3d"], dtype=np.float64)
    axis_info = _estimate_pull_axis(handle, drawer)
    pull_axis = np.asarray(axis_info["pull_axis_world"], dtype=np.float64)
    pull_axis[2] = 0.0
    pull_axis = pull_axis / float(np.linalg.norm(pull_axis))
    approach_dir = -pull_axis
    side, side_reason = _choose_arm(handle_xyz.tolist())
    alternate_side = "right" if side == "left" else "left"
    approach_offsets = _parse_float_list(APPROACH_OFFSETS_DEG, [0.0, -15.0, 15.0, -30.0, 30.0])
    wrist_rolls = _parse_float_list(WRIST_ROLLS_DEG, [0.0, -20.0, 20.0, -45.0, 45.0])
    z_offsets = _parse_float_list(Z_OFFSETS_M, [0.0, 0.015, -0.015])
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, float, float]] = set()
    for arm, arm_score in ((side, 1.0), (alternate_side, 0.78)):
        for approach_offset in approach_offsets:
            candidate_approach = _rotate_xy(approach_dir, approach_offset)
            candidate_approach = candidate_approach / float(np.linalg.norm(candidate_approach))
            approach_penalty = min(abs(float(approach_offset)) * 0.003, 0.18)
            for wrist_roll in wrist_rolls:
                rpy, axes = _side_grasp_display_rpy_from_approach(candidate_approach.tolist(), wrist_roll)
                for z_offset in z_offsets:
                    key = (
                        arm,
                        round(float(approach_offset), 3),
                        round(wrist_roll, 3),
                        round(z_offset, 4),
                        round(float(handle_xyz[2]), 4),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    grasp = handle_xyz + pull_axis * HANDLE_CONTACT_OFFSET_M
                    grasp[2] += z_offset
                    pregrasp = grasp - candidate_approach * PREGRASP_STANDOFF_M
                    pregrasp[2] += z_offset
                    pull_open = grasp + pull_axis * min(PULL_DISTANCE_M, MAX_INCREMENTAL_TRAVEL_M)
                    push_close = pull_open - pull_axis * min(PUSH_DISTANCE_M, MAX_INCREMENTAL_TRAVEL_M)
                    score = arm_score - approach_penalty - abs(wrist_roll) * 0.002 - abs(z_offset) * 2.0
                    candidates.append(
                        {
                            "arm": arm,
                            "score": round(float(score), 4),
                            "handle_detection": handle,
                            "rpy": _round_list(rpy, 3),
                            "gripper_local_axes_world": axes,
                            "pregrasp_pose": {
                                "position": _round_list(pregrasp),
                                "rpy": _round_list(rpy, 3),
                            },
                            "grasp_pose": {
                                "position": _round_list(grasp),
                                "rpy": _round_list(rpy, 3),
                            },
                            "pull_open_pose": {
                                "position": _round_list(pull_open),
                                "rpy": _round_list(rpy, 3),
                                "delta_m": round(float(min(PULL_DISTANCE_M, MAX_INCREMENTAL_TRAVEL_M)), 4),
                            },
                            "push_close_pose": {
                                "position": _round_list(push_close),
                                "rpy": _round_list(rpy, 3),
                                "delta_m": round(float(min(PUSH_DISTANCE_M, MAX_INCREMENTAL_TRAVEL_M)), 4),
                            },
                            "release_pose": {
                                "position": _round_list(pregrasp),
                                "rpy": _round_list(rpy, 3),
                            },
                            "approach_direction_world": _round_list(candidate_approach, 5),
                            "pull_axis_world": _round_list(pull_axis, 5),
                            "approach_offset_deg": round(float(approach_offset), 3),
                            "wrist_roll_deg": round(float(wrist_roll), 3),
                            "z_offset_m": round(float(z_offset), 4),
                            "side_reason": side_reason,
                            "contact_status": (
                                "preview_only; physical requires width-limited/contact-aware "
                                "grasp_handle and incremental pull/push ticket"
                            ),
                        }
                    )
    candidates.sort(key=lambda item: -float(item["score"]))
    return axis_info, candidates[: max(1, PLAN_MAX_CANDIDATES)]


def _preview_pose(label: str, candidate: dict[str, Any], pose_key: str) -> dict[str, Any]:
    side = candidate["arm"]
    pose = candidate[pose_key]
    kwargs: dict[str, Any] = {
        f"{side}_target_pos": [float(x) for x in pose["position"]],
        f"{side}_target_rpy": [float(x) for x in pose["rpy"]],
        f"{side}_gripper": 1.0,
        "preview_only": True,
        "planning_speed": PLANNING_SPEED,
        "ik_error_threshold": IK_ERROR_THRESHOLD_M,
        "ik_rot_threshold_deg": IK_ROT_THRESHOLD_DEG,
        "ik_xyz_weight": IK_XYZ_WEIGHT,
        "ik_rpy_weight": IK_RPY_WEIGHT,
        "planner_backend": PLANNER_BACKEND,
        "solver_speed": SOLVER_SPEED,
    }
    started = time.time()
    try:
        result = freespace_move(**kwargs)
        data = _json_safe(result)
        status = str(data.get("status", "")) if isinstance(data, dict) else ""
        executed = bool(data.get("executed")) if isinstance(data, dict) else False
        success = status == "Success" and not executed
        if executed:
            success = False
            TASK_RESULT["risk_notes"].append(f"Preview {label!r} reported executed=True; treating as unsafe failure.")
        return {
            "label": label,
            "pose_key": pose_key,
            "success": success,
            "duration_s": round(time.time() - started, 3),
            "request": _json_safe(kwargs),
            "result": data,
        }
    except Exception as exc:
        return {
            "label": label,
            "pose_key": pose_key,
            "success": False,
            "duration_s": round(time.time() - started, 3),
            "request": _json_safe(kwargs),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _preview_candidates(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    previews: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for index, candidate in enumerate(candidates, start=1):
        candidate = dict(candidate)
        candidate["candidate_index"] = index
        pre = _preview_pose(f"candidate_{index}_pregrasp", candidate, "pregrasp_pose")
        previews.append(pre)
        if not pre.get("success"):
            continue
        sequence: list[dict[str, Any]] = [pre]
        for label, pose_key in (
            (f"candidate_{index}_grasp_pose", "grasp_pose"),
            (f"candidate_{index}_pull_open_small", "pull_open_pose"),
            (f"candidate_{index}_push_close_small", "push_close_pose"),
            (f"candidate_{index}_release", "release_pose"),
        ):
            step = _preview_pose(label, candidate, pose_key)
            previews.append(step)
            sequence.append(step)
            if not step.get("success"):
                break
        if all(step.get("success") for step in sequence):
            selected = candidate
            selected["preview_sequence"] = [
                {
                    "label": step.get("label"),
                    "pose_key": step.get("pose_key"),
                    "status": (step.get("result") or {}).get("status") if isinstance(step.get("result"), dict) else None,
                    "final_pos_error_m": (step.get("result") or {}).get("final_pos_error_m")
                    if isinstance(step.get("result"), dict)
                    else None,
                    "final_rot_error_deg": (step.get("result") or {}).get("final_rot_error_deg")
                    if isinstance(step.get("result"), dict)
                    else None,
                }
                for step in sequence
            ]
            break
    return selected, previews


def _resolve_plan_json_path() -> Path:
    if PLAN_JSON:
        path = Path(PLAN_JSON)
        if not path.exists():
            path = current_run_dir() / PLAN_JSON
        if not path.exists():
            raise RuntimeError(f"OPENFORGE_DRAWER_PLAN_JSON does not exist: {PLAN_JSON}")
        return path
    if PLAN_LOG_DIR:
        root = Path(PLAN_LOG_DIR)
        plans = sorted(root.glob("plans/*/plan.json"), key=lambda p: p.stat().st_mtime)
        if plans:
            return plans[-1]
        candidate = root / "plan.json"
        if candidate.exists():
            return candidate
        raise RuntimeError(f"No drawer plan JSON found under OPENFORGE_DRAWER_PLAN_LOG_DIR={PLAN_LOG_DIR!r}")
    raise RuntimeError(
        "Physical drawer stages require OPENFORGE_DRAWER_PLAN_JSON or OPENFORGE_DRAWER_PLAN_LOG_DIR "
        "pointing to an inspected successful no-motion plan."
    )


def _load_selected_plan() -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = _resolve_plan_json_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected")
    if selected is None:
        selected = ((data.get("details") or {}).get("plan") or {}).get("selected")
    if not isinstance(selected, dict):
        raise RuntimeError(f"Drawer plan has no selected candidate: {path}")
    return data, selected, path


def _gripper_pos_from_state(side: str) -> float | None:
    try:
        state = get_robot_state()
        value = getattr(state, f"{side}_gripper_pos")
        return float(value)
    except Exception:
        return None


def _ee_pos_from_state(side: str) -> list[float]:
    state = get_robot_state()
    value = getattr(state, f"{side}_ee_pos")
    return [float(x) for x in value[:3]]


def _physical_motion_kwargs(side: str, pose: dict[str, Any], *, preview_only: bool = False) -> dict[str, Any]:
    return {
        f"{side}_target_pos": [float(x) for x in pose["position"]],
        f"{side}_target_rpy": [float(x) for x in pose["rpy"]],
        f"{side}_gripper": max(GRIPPER_TARGET_POS, GRIPPER_OPEN_POS),
        "preview_only": bool(preview_only),
        "planning_speed": PLANNING_SPEED,
        "ik_error_threshold": IK_ERROR_THRESHOLD_M,
        "ik_rot_threshold_deg": IK_ROT_THRESHOLD_DEG,
        "ik_xyz_weight": IK_XYZ_WEIGHT,
        "ik_rpy_weight": IK_RPY_WEIGHT,
        "planner_backend": PLANNER_BACKEND,
        "solver_speed": SOLVER_SPEED,
    }


def _execute_pose(side: str, pose: dict[str, Any], label: str) -> dict[str, Any]:
    result = freespace_move(**_physical_motion_kwargs(side, pose, preview_only=False))
    data = _json_safe(result)
    status = data.get("status") if isinstance(data, dict) else None
    executed = bool(data.get("executed")) if isinstance(data, dict) else False
    if status != "Success" or not executed:
        raise RuntimeError(f"{label} did not execute successfully: {data}")
    return data


def _set_gripper_checked(
    side: str,
    pos: float,
    *,
    label: str,
    vel_limit: float = GRIPPER_VEL_LIMIT,
    torque_limit: float = GRIPPER_TORQUE_LIMIT,
) -> dict[str, Any]:
    target = float(np.clip(float(pos), 0.0, 1.0))
    if target < GRIPPER_MIN_SAFE_POS:
        raise RuntimeError(
            f"Refusing {label}: target gripper pos {target:.3f} is below "
            f"OPENFORGE_DRAWER_GRIPPER_MIN_SAFE_POS={GRIPPER_MIN_SAFE_POS:.3f}."
        )
    before = _gripper_pos_from_state(side)
    result = set_gripper(side, target, vel_limit=vel_limit, torque_limit=torque_limit)
    TASK_RESULT["physical_motion_executed"] = True
    TASK_RESULT.setdefault("movement_capable_calls", []).append("set_gripper")
    time.sleep(max(0.0, GRIPPER_SETTLE_S))
    after = _gripper_pos_from_state(side)
    open_confirm_samples: list[float] = []
    close_confirm_samples: list[float] = []
    if target >= 0.95 and after is not None:
        open_confirm_samples.append(float(after))
        deadline = time.monotonic() + max(0.0, GRIPPER_OPEN_CONFIRM_TIMEOUT_S)
        while after < GRIPPER_OPEN_CONFIRM_POS and time.monotonic() < deadline:
            time.sleep(max(0.02, GRIPPER_OPEN_CONFIRM_POLL_S))
            after = _gripper_pos_from_state(side)
            if after is None:
                break
            open_confirm_samples.append(float(after))
    elif target < 0.95 and after is not None:
        close_confirm_samples.append(float(after))
        deadline = time.monotonic() + max(0.0, GRIPPER_CLOSE_CONFIRM_TIMEOUT_S)
        while after > target + 0.02 and time.monotonic() < deadline:
            time.sleep(max(0.02, GRIPPER_CLOSE_CONFIRM_POLL_S))
            after = _gripper_pos_from_state(side)
            if after is None:
                break
            close_confirm_samples.append(float(after))
    entry = {
        "label": label,
        "side": side,
        "target_pos": round(target, 4),
        "before_pos": round(float(before), 4) if before is not None else None,
        "after_pos": round(float(after), 4) if after is not None else None,
        "open_confirm_samples": [round(float(sample), 4) for sample in open_confirm_samples],
        "close_confirm_samples": [round(float(sample), 4) for sample in close_confirm_samples],
        "vel_limit": vel_limit,
        "torque_limit": torque_limit,
        "result": _json_safe(result),
    }
    TASK_RESULT.setdefault("gripper_commands", []).append(entry)
    if target >= 0.95 and after is not None and after < GRIPPER_OPEN_CONFIRM_POS:
        raise RuntimeError(
            f"Refusing to continue after {label}: commanded open target {target:.3f}, "
            f"but measured gripper pos {after:.3f} is below "
            f"OPENFORGE_DRAWER_GRIPPER_OPEN_CONFIRM_POS={GRIPPER_OPEN_CONFIRM_POS:.3f}."
        )
    if target < 0.95 and after is not None and after < GRIPPER_MIN_SAFE_POS:
        raise RuntimeError(
            f"Refusing to continue after {label}: measured gripper pos {after:.3f} "
            f"is below safe minimum {GRIPPER_MIN_SAFE_POS:.3f}."
        )
    return entry


def _validate_handle_close_steps(steps: list[dict[str, Any]]) -> None:
    if not steps:
        raise RuntimeError("Refusing drawer handle grasp: no gripper close steps were recorded.")
    start = steps[0].get("before_pos")
    final = steps[-1].get("after_pos")
    if start is None or final is None:
        raise RuntimeError("Refusing drawer handle grasp: measured gripper state is unavailable.")
    start_f = float(start)
    final_f = float(final)
    delta = start_f - final_f
    if final_f > GRIPPER_GRASP_MAX_OPEN_POS:
        raise RuntimeError(
            f"Refusing drawer pull: gripper remains too open after handle close "
            f"({final_f:.3f} > OPENFORGE_DRAWER_GRIPPER_GRASP_MAX_OPEN_POS={GRIPPER_GRASP_MAX_OPEN_POS:.3f})."
        )
    if delta < GRIPPER_CLOSE_MIN_DELTA_POS:
        raise RuntimeError(
            f"Refusing drawer pull: measured close delta {delta:.3f} is below "
            f"OPENFORGE_DRAWER_GRIPPER_CLOSE_MIN_DELTA_POS={GRIPPER_CLOSE_MIN_DELTA_POS:.3f}."
        )


def _bounded_grasp_handle(side: str) -> list[dict[str, Any]]:
    scout = max(GRIPPER_TARGET_POS, min(GRIPPER_OPEN_POS, GRIPPER_SCOUT_POS))
    steps = [
        _set_gripper_checked(
            side,
            scout,
            label="drawer_handle_scout_close",
            vel_limit=GRIPPER_VEL_LIMIT,
            torque_limit=GRIPPER_TORQUE_LIMIT,
        ),
        _set_gripper_checked(
            side,
            GRIPPER_TARGET_POS,
            label="drawer_handle_target_close",
            vel_limit=GRIPPER_VEL_LIMIT,
            torque_limit=GRIPPER_TORQUE_LIMIT,
        ),
    ]
    _validate_handle_close_steps(steps)
    return steps


def _observe(stage: str = "observe") -> dict[str, Any]:
    packet = _capture(stage)
    handle, drawer, notes = _select_handle_and_drawer(packet)
    summary = {
        "success": handle is not None and (handle.get("score") or 0.0) >= HANDLE_MIN_SCORE,
        "handle_detection": handle,
        "drawer_detection": drawer,
        "notes": notes,
        "packet_path": packet.get("packet_path"),
    }
    TASK_RESULT["observation"] = summary
    TASK_RESULT["risk_notes"].extend(notes)
    write_stage_summary(stage=stage, result=summary, log_dir=current_run_dir())
    return summary


def _plan_handle_grasp() -> dict[str, Any]:
    source = _load_observation_source()
    packet = source if source is not None else _capture("plan_observe")
    handle, drawer, notes = _select_handle_and_drawer(packet)
    if handle is None or (handle.get("score") or 0.0) < HANDLE_MIN_SCORE:
        plan = {
            "success": False,
            "physical_motion_executed": False,
            "error": "No calibrated high-confidence handle detection for planning.",
            "handle_detection": handle,
            "drawer_detection": drawer,
            "notes": notes,
        }
        save_plan_packet(stage="plan_handle_grasp", selected=None, candidates=[], previews=[], observation=packet)
        TASK_RESULT["risk_notes"].extend(notes)
        return plan

    axis_info, candidates = _candidate_poses(handle, drawer)
    selected, previews = _preview_candidates(candidates)
    plan = {
        "success": selected is not None,
        "physical_motion_executed": False,
        "source_observation_packet": packet.get("packet_path"),
        "handle_detection": handle,
        "drawer_detection": drawer,
        "pull_axis": axis_info,
        "candidate_count": len(candidates),
        "selected": selected,
        "preview_count": len(previews),
        "preview_failures": [
            {
                "label": preview.get("label"),
                "pose_key": preview.get("pose_key"),
                "status": (preview.get("result") or {}).get("status") if isinstance(preview.get("result"), dict) else None,
                "error": preview.get("error") or (preview.get("result") or {}).get("reason")
                if isinstance(preview.get("result"), dict)
                else preview.get("error"),
            }
            for preview in previews
            if not preview.get("success")
        ][:8],
        "notes": notes
        + [
            "Plan is preview-only and does not validate actual drawer contact, latch force, or travel stop.",
            "Physical pull/push requires a fresh drawer-specific ticket and block-by-block re-observe after contact.",
        ],
    }
    path = save_plan_packet(
        stage="plan_handle_grasp",
        selected=selected,
        candidates=candidates,
        previews=previews,
        observation=packet,
    )
    plan["plan_packet"] = str(path)
    return plan


def _plan_release_retreat() -> dict[str, Any]:
    _plan_packet, selected_plan, plan_path = _load_selected_plan()
    side = str(selected_plan["arm"]).strip().lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"Invalid selected drawer arm for retreat planning: {side!r}")
    current_pos = np.asarray(_ee_pos_from_state(side), dtype=np.float64)
    base_rpy = (
        (selected_plan.get("release_pose") or {}).get("rpy")
        or (selected_plan.get("grasp_pose") or {}).get("rpy")
        or selected_plan.get("rpy")
    )
    if not base_rpy:
        raise RuntimeError("Selected drawer plan has no RPY for release retreat planning.")
    pull_axis = np.asarray(selected_plan.get("pull_axis_world") or [0.0, 1.0, 0.0], dtype=np.float64)
    pull_axis[2] = 0.0
    if float(np.linalg.norm(pull_axis)) < 1e-6:
        pull_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    pull_axis = pull_axis / float(np.linalg.norm(pull_axis))

    candidates: list[dict[str, Any]] = []
    z_values = _parse_float_list(RELEASE_LIFT_Z_CANDIDATES_M, [0.03, 0.06, 0.09])
    clear_values = _parse_float_list(RELEASE_PULL_CLEARANCE_CANDIDATES_M, [0.02, 0.04])
    for z in z_values:
        pose = {
            "position": _round_list(current_pos + np.asarray([0.0, 0.0, z], dtype=np.float64)),
            "rpy": _round_list(base_rpy, 3),
        }
        candidates.append(
            {
                "arm": side,
                "candidate_index": len(candidates) + 1,
                "label": f"current_lift_{z:.3f}m",
                "release_pose": pose,
                "source_plan_json": str(plan_path),
                "current_pose": {"position": _round_list(current_pos), "rpy": _round_list(base_rpy, 3)},
                "retreat_kind": "vertical_lift_from_current",
            }
        )
    for z in z_values:
        for clear in clear_values:
            pose = {
                "position": _round_list(current_pos + pull_axis * clear + np.asarray([0.0, 0.0, z], dtype=np.float64)),
                "rpy": _round_list(base_rpy, 3),
            }
            candidates.append(
                {
                    "arm": side,
                    "candidate_index": len(candidates) + 1,
                    "label": f"current_pull_clear_{clear:.3f}m_lift_{z:.3f}m",
                    "release_pose": pose,
                    "pull_axis_world": _round_list(pull_axis, 5),
                    "source_plan_json": str(plan_path),
                    "current_pose": {"position": _round_list(current_pos), "rpy": _round_list(base_rpy, 3)},
                    "retreat_kind": "small_pull_axis_clearance_with_lift",
                }
            )
    if isinstance(selected_plan.get("release_pose"), dict):
        direct = dict(selected_plan)
        direct["candidate_index"] = len(candidates) + 1
        direct["label"] = "selected_plan_release_pose"
        direct["source_plan_json"] = str(plan_path)
        direct["retreat_kind"] = "selected_plan_release"
        candidates.append(direct)

    previews: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        preview = _preview_pose(f"release_retreat_{candidate['candidate_index']}_{candidate['label']}", candidate, "release_pose")
        previews.append(preview)
        if preview.get("success") and selected is None:
            selected = candidate
            selected["preview_sequence"] = [
                {
                    "label": preview.get("label"),
                    "pose_key": preview.get("pose_key"),
                    "status": (preview.get("result") or {}).get("status") if isinstance(preview.get("result"), dict) else None,
                    "final_pos_error_m": (preview.get("result") or {}).get("final_pos_error_m")
                    if isinstance(preview.get("result"), dict)
                    else None,
                    "final_rot_error_deg": (preview.get("result") or {}).get("final_rot_error_deg")
                    if isinstance(preview.get("result"), dict)
                    else None,
                }
            ]
            break

    path = save_plan_packet(
        stage="plan_release_retreat",
        selected=selected,
        candidates=candidates,
        previews=previews,
        observation=None,
    )
    plan = {
        "success": selected is not None,
        "physical_motion_executed": False,
        "source_plan_json": str(plan_path),
        "current_position": _round_list(current_pos),
        "candidate_count": len(candidates),
        "selected": selected,
        "preview_count": len(previews),
        "preview_failures": [
            {
                "label": preview.get("label"),
                "pose_key": preview.get("pose_key"),
                "status": (preview.get("result") or {}).get("status") if isinstance(preview.get("result"), dict) else None,
                "error": preview.get("error") or (preview.get("result") or {}).get("reason")
                if isinstance(preview.get("result"), dict)
                else preview.get("error"),
            }
            for preview in previews
            if not preview.get("success")
        ][:8],
        "plan_packet": str(path),
        "notes": [
            "Release-retreat plan is preview-only. Physical retreat still requires the active drawer ticket and gates.",
            "Use this plan only for release/safety retreat, not for drawer pull_open_small.",
        ],
    }
    TASK_RESULT["release_retreat_plan"] = plan
    return plan


def _guard_physical_stage(stage: str) -> None:
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError(
            f"Drawer physical stage {stage!r} refused: OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required "
            "and no drawer-specific physical ticket exists in this no-motion session."
        )
    if not _truthy_env("OPENFORGE_DRAWER_ENABLE_PHYSICAL_CONTACT"):
        raise RuntimeError(
            f"Drawer physical stage {stage!r} refused: contact motion also requires "
            "OPENFORGE_DRAWER_ENABLE_PHYSICAL_CONTACT=1 after a drawer-specific ticket, "
            "validated no-motion plan, and operator-reviewed pull axis."
        )


def _run_physical_block(stage: str) -> dict[str, Any]:
    _guard_physical_stage(stage)
    _plan_packet, selected, plan_path = _load_selected_plan()
    side = str(selected["arm"]).strip().lower()
    if side not in {"left", "right"}:
        raise RuntimeError(f"Invalid selected drawer arm: {side!r}")
    block: dict[str, Any] = {
        "stage": stage,
        "selected_plan_json": str(plan_path),
        "selected_candidate_index": selected.get("candidate_index"),
        "arm": side,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "plan": {
            "pull_axis_world": selected.get("pull_axis_world"),
            "approach_direction_world": selected.get("approach_direction_world"),
            "pregrasp_pose": selected.get("pregrasp_pose"),
            "grasp_pose": selected.get("grasp_pose"),
            "pull_open_pose": selected.get("pull_open_pose"),
            "push_close_pose": selected.get("push_close_pose"),
            "release_pose": selected.get("release_pose"),
        },
        "stop_after_block": "Re-observe and inspect artifacts before the next drawer contact block.",
    }
    if stage == "pregrasp":
        block["gripper_open"] = _set_gripper_checked(
            side,
            GRIPPER_OPEN_POS,
            label="drawer_pregrasp_open",
            vel_limit=GRIPPER_OPEN_VEL_LIMIT,
            torque_limit=GRIPPER_OPEN_TORQUE_LIMIT,
        )
        block["movement_capable_calls"].append("set_gripper")
        block["motion"] = _execute_pose(side, selected["pregrasp_pose"], "drawer pregrasp")
        block["movement_capable_calls"].append("freespace_move")
    elif stage == "grasp_handle":
        block["motion"] = _execute_pose(side, selected["grasp_pose"], "drawer grasp pose")
        block["movement_capable_calls"].append("freespace_move")
        block["gripper_close_steps"] = _bounded_grasp_handle(side)
        block["movement_capable_calls"].append("set_gripper")
    elif stage == "pull_open_small":
        block["motion"] = _execute_pose(side, selected["pull_open_pose"], "drawer pull_open_small")
        block["movement_capable_calls"].append("freespace_move")
    elif stage == "push_close_small":
        block["motion"] = _execute_pose(side, selected["push_close_pose"], "drawer push_close_small")
        block["movement_capable_calls"].append("freespace_move")
    elif stage == "release":
        block["gripper_release"] = _set_gripper_checked(
            side,
            GRIPPER_OPEN_POS,
            label="drawer_release_open",
            vel_limit=GRIPPER_OPEN_VEL_LIMIT,
            torque_limit=GRIPPER_OPEN_TORQUE_LIMIT,
        )
        block["movement_capable_calls"].append("set_gripper")
        block["motion"] = _execute_pose(side, selected["release_pose"], "drawer release retreat")
        block["movement_capable_calls"].append("freespace_move")
    else:
        raise RuntimeError(f"Unsupported drawer physical stage: {stage}")
    block["physical_motion_executed"] = bool(block["movement_capable_calls"])
    block["success"] = True
    write_stage_summary(stage=stage, result=block, log_dir=current_run_dir())
    return block


def _run_all() -> None:
    raise RuntimeError(
        "Monolithic stage 'all' is disabled. Use block-by-block drawer stages: observe, "
        "plan_handle_grasp, pregrasp, grasp_handle, pull_open_small, push_close_small, "
        "release, verify."
    )


def _main() -> None:
    stage = TASK_RESULT["implemented_stage"]
    print("[drawer_pull_push] Starting staged drawer task.")
    print(f"[drawer_pull_push] stage={REQUESTED_STAGE!r} -> {stage!r}")
    print(f"[drawer_pull_push] prompts={PROMPTS}")
    print(f"[drawer_pull_push] cameras={CAMERAS}")

    if _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION") and stage not in PHYSICAL_STAGES:
        TASK_RESULT["risk_notes"].append(
            "OPENFORGE_ALLOW_PHYSICAL_MOTION was set for a no-motion stage; no movement-capable calls were made."
        )

    try:
        if stage == "observe":
            observation = _observe("observe")
            TASK_RESULT["success"] = bool(observation.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "Observe-only drawer block completed; no arm or gripper command was run."
        elif stage == "plan_handle_grasp":
            plan = _plan_handle_grasp()
            TASK_RESULT["plan"] = plan
            TASK_RESULT["success"] = bool(plan.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = (
                "No-motion drawer plan/preview completed; contact stages remain guarded."
            )
        elif stage == "plan_release_retreat":
            plan = _plan_release_retreat()
            TASK_RESULT["plan"] = plan
            TASK_RESULT["success"] = bool(plan.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "No-motion drawer release-retreat preview completed."
        elif stage == "verify":
            observation = _observe("verify")
            TASK_RESULT["verification"] = observation
            TASK_RESULT["success"] = bool(observation.get("handle_detection") or observation.get("drawer_detection"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = "Verify observation completed; no physical motion was run."
        elif stage in PHYSICAL_STAGES:
            block = _run_physical_block(stage)
            TASK_RESULT["physical_block"] = block
            TASK_RESULT["physical_motion_executed"] = bool(block.get("physical_motion_executed"))
            TASK_RESULT["movement_capable_calls"] = list(block.get("movement_capable_calls", []))
            TASK_RESULT["success"] = bool(block.get("success"))
            TASK_RESULT["reward"] = 1.0 if TASK_RESULT["success"] else 0.0
            TASK_RESULT["why_stopped"] = f"Physical drawer block {stage!r} completed; re-observe before the next block."
        elif stage == "all":
            _run_all()
        else:
            TASK_RESULT["why_stopped"] = (
                "Unsupported stage. Use observe, plan_handle_grasp, plan_release_retreat, pregrasp, grasp_handle, "
                "pull_open_small, push_close_small, release, or verify."
            )
    except Exception as exc:
        TASK_RESULT["success"] = False
        TASK_RESULT["reward"] = 0.0
        TASK_RESULT["why_stopped"] = f"Stage {stage!r} failed: {type(exc).__name__}: {exc}"
        TASK_RESULT["risk_notes"].append("Inspect result.json/run log and artifacts before any retry.")
        print(f"[drawer_pull_push] {TASK_RESULT['why_stopped']}")
        print(json.dumps(_json_safe(TASK_RESULT), indent=2))
        raise

    if stage not in PHYSICAL_STAGES:
        TASK_RESULT["physical_motion_executed"] = False
        TASK_RESULT["movement_capable_calls"] = []
    print(f"[drawer_pull_push] {TASK_RESULT['why_stopped']}")
    print(json.dumps(_json_safe(TASK_RESULT), indent=2))


_main()
