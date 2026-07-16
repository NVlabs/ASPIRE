"""One-command internal feedback loop for a left-arm top-down can grasp/lift."""

from __future__ import annotations

import os
from typing import Any

from skill_library.simple_grasp_lift_loop import build_config, run_one_shot_loop


TASK_RESULT: dict[str, Any] = {}


def _set_default_can_topdown_env() -> None:
    """Default this can ticket to the LIBERO-transfer top-down cylinder policy."""
    defaults = {
        "OPENFORGE_CAN_PROMPTS": "can,sprite can,green can,soda can,aluminum can,tin can",
        "OPENFORGE_DEBUG_OBS_CAMERAS": "top",
        "OPENFORGE_GRASP_LIFT_POST_OBS_CAMERAS": "top,left,right",
        "OPENFORGE_DEBUG_OBS_TIMEOUT_S": "20",
        "OPENFORGE_GRASP_LIFT_ARM": "left",
        "OPENFORGE_GRASP_LIFT_ATTEMPT_BUDGET": "1",
        "OPENFORGE_GRASP_LIFT_TOPDOWN_ONLY": "1",
        "OPENFORGE_GRASP_LIFT_TOPDOWN_YAWS": "0",
        "OPENFORGE_GRASP_LIFT_TOPDOWN_PITCHES": "180",
        "OPENFORGE_GRASP_LIFT_TOPDOWN_Z_OFFSETS_M": "0.025",
        "OPENFORGE_GRASP_LIFT_PREGRASP_Z_M": "0.12",
        "OPENFORGE_GRASP_LIFT_BODY_GRASP_FRACTION": "0.70",
        "OPENFORGE_GRASP_LIFT_OBJECT_RADIUS_M": "0.033",
        "OPENFORGE_GRASP_LIFT_OBJECT_WIDTH_M": "0.066",
        "OPENFORGE_GRASP_LIFT_GRIPPER_WIDTH_CLEARANCE_M": "0.02",
        "OPENFORGE_GRASP_LIFT_LIFT_Z_M": "0.08",
        "OPENFORGE_GRASP_LIFT_REQUIRE_CLOSE_CONTACT": "1",
        "OPENFORGE_GRASP_LIFT_CLOSE_CONTACT_MIN_TARGET_DELTA_POS": "0.015",
        "OPENFORGE_GRASP_LIFT_CLOSE_MIN_DELTA_POS": "0.02",
        "OPENFORGE_GRASP_LIFT_CLOSE_WIDTH_COMPRESSION_M": "0.008",
        "OPENFORGE_GRASP_LIFT_CLOSE_TARGET_POS": "0.05",
        "OPENFORGE_GRASP_LIFT_CLOSE_TARGET_CAP_POS": "0.72",
        "OPENFORGE_GRASP_LIFT_CLOSE_SCOUT_POS": "0.05",
        "OPENFORGE_GRASP_LIFT_CLOSE_TORQUE_LIMIT": "0.80",
        "OPENFORGE_GRASP_LIFT_CLOSE_VEL_LIMIT": "0.80",
        "OPENFORGE_GRASP_LIFT_CLOSE_CONFIRM_TIMEOUT_S": "2.0",
        "OPENFORGE_GRASP_LIFT_OPEN_CONFIRM_POS": "0.75",
        "OPENFORGE_GRASP_LIFT_OPEN_CONFIRM_TIMEOUT_S": "10.0",
        "OPENFORGE_GRASP_LIFT_GRIPPER_MAX_PLAUSIBLE_POS": "0.98",
        "OPENFORGE_GRASP_LIFT_SUCCESS_ALLOW_MIXED_REAL_SOURCES": "1",
        "OPENFORGE_GRASP_LIFT_HOLD_S": "1.0",
        "OPENFORGE_GRASP_LIFT_MAX_CANDIDATES": "40",
        "OPENFORGE_GRASP_LIFT_MAX_PREVIEW_CANDIDATES": "40",
        "OPENFORGE_GRASP_LIFT_PLANNER_BACKEND": "rrtconnect",
        "OPENFORGE_GRASP_LIFT_PLANNING_SPEED": "0.30",
        "OPENFORGE_GRASP_LIFT_IK_ERROR_THRESHOLD_M": "0.025",
        "OPENFORGE_GRASP_LIFT_IK_ROT_THRESHOLD_DEG": "14.0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


_set_default_can_topdown_env()

TASK_RESULT.update(
    run_one_shot_loop(
        build_config(
            task_name="grasp_lift_can_one_shot_loop",
            object_kind="can",
            default_prompts=["can", "spirit can"],
            prompt_env="OPENFORGE_CAN_PROMPTS",
            manual_xyz_env="OPENFORGE_CAN_XYZ",
            manual_half_extents_env="OPENFORGE_CAN_HALF_EXTENTS",
        )
    )
)
