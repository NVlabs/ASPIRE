"""Orange-bowl lower-rack entrypoint for real YAM.

This file owns the orange bowl pickup defaults, then delegates the common
lower-rack sequence to ``bowl_lower_rack_common``:

pickup/lift -> rack front high -> rack front entry -> direct insert -> release
-> direct horizontal retreat.

The orange pickup defaults are intentionally explicit so they cannot be mixed
with white-bowl prompts or rim geometry by changing only the object name.
"""

from __future__ import annotations

import os
from typing import Any


def _force_env(name: str, value: str) -> None:
    os.environ[name] = value


def _default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _configure_orange_defaults() -> None:
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OBJECT_NAME", "orange bowl")
    _force_env("OPENFORGE_BOWL_RACK_TASK_NAME", "place_orange_bowl_on_lower_rack_one_shot_loop")
    _force_env("OPENFORGE_BOWL_RACK_OBJECT_NAME", "orange bowl")
    _force_env("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", "orange_bowl_lower_rack")
    _force_env("OPENFORGE_BOWL_RACK_SCRIPT_PATH", "cap/saved_scripts/place_orange_bowl_on_lower_rack_one_shot_loop.py")
    _force_env("OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH", "cap/saved_scripts/bowl_lower_rack_common.py")
    _force_env("OPENFORGE_BOWL_RACK_VISIBLE_ENV", "OPENFORGE_BOWL_RACK_ORANGE_BOWL_VISIBLE")
    _force_env("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "orange_bowl_lower_rack")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", "orange bowl,bowl")
    _force_env("OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS", "orange bowl,bowl")
    _force_env("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", "orange bowl,bowl")
    _force_env(
        "OPENFORGE_BOWL_RACK_POST_PROMPTS",
        "orange bowl,bowl,lower rack shelf,bottom shelf,lower shelf,rack,wooden rack,blue gripper fingers",
    )
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_ARM", "right")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "top_down_bowl_rim")
    _force_env("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "top_down_bowl_rim_straddle")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_MAX_CANDIDATES", "48")
    _force_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", "0")
    _force_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", "1")
    _force_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", "0")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", "1")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", "0")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_CONTACT_ABOVE_OBJECT_Z_M", "0.060")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", "0")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_LIFT_PERCEPTION_BEFORE_PLACE", "0")
    _force_env("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", "0.68")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", "0.60")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_VEL_LIMIT", "1")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_TORQUE_LIMIT", "1")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.04,0.01,0.00")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", "0.00")
    _force_env("OPENFORGE_BOWL_RACK_HOLD_MIN_POS", "0.000")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS", "0.02")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_SETTLE_S", "0.05")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_TIMEOUT_S", "1.2")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_POLL_S", "0.05")
    _force_env("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", "0.85")
    _force_env("OPENFORGE_BOWL_RACK_CAPTURE_INTERMEDIATE_OBSERVATIONS", "0")
    _force_env("OPENFORGE_BOWL_RACK_EXEC_STALE_CACHE_REPLAN_SLEEP_S", "0.05")
    _force_env("OPENFORGE_BOWL_RACK_GO_HOME_AFTER_SUCCESS", "1")
    _force_env("OPENFORGE_BOWL_RACK_GO_HOME_AFTER_FAILURE", "1")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_GRIPPERS_AFTER_HOME", "1")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_GRIPPERS_AFTER_HOME_SIDES", "left,right")
    _force_env("OPENFORGE_BOWL_RACK_PLACE_OFFSETS_M", "0.000,0.000,0.000")
    _force_env("OPENFORGE_BOWL_RACK_MAX_PLACE_CANDIDATES", "1")
    _force_env("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", "0.025")
    _force_env("OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", "0.050")
    _force_env("OPENFORGE_BOWL_RACK_EXEC_MAX_POS_ERROR_M", "0.025")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_Y_SIGN", "positive")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_X_STANDOFF_M", "0.080")
    _default_env("OPENFORGE_BOWL_RACK_FRONT_IK_ROT_THRESHOLD_DEG", "30.0")
    _default_env("OPENFORGE_BOWL_RACK_FRONT_IK_RPY_WEIGHT", "0.10")
    _default_env("OPENFORGE_BOWL_RACK_PREVIEW_FRONT_MAX_ROT_ERROR_DEG", "40.0")
    _default_env("OPENFORGE_BOWL_RACK_EXEC_FRONT_MAX_ROT_ERROR_DEG", "40.0")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_PREVIEW_RELEASE_WITH_PLANNER", "0")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_USE_MULTIVIEW_RIM_Z", "1")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_Z_MODE", "height_over_support")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_SUPPORT_Z_M", "0.760")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_HEIGHT_M", "0.080")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_Z_BIAS_M", "0.080")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_RAISE_M", "0.035")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_XY_DISAGREEMENT_M", "0.080")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M", "-0.015")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_CANDIDATES_M", "-0.015,-0.010,-0.020")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LOW_PREGRASP_Z_LIFT_M", "0.080")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LOW_PREGRASP_MIN_ABOVE_RIM_M", "0.045")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LIFT_Z_M", "0.100")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG", "-135,-90,-180,-45,135,90,0,45")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG", "10.0")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_INWARD_OFFSET_M", "0.035")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_XY_BIASES_M", "0.000")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_TO_CONTACT_OFFSETS_M", "0.000")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_FINGER_FORWARD_SIGNS", "-1.0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_RETRY_ATTEMPTS", "8")
    _force_env("OPENFORGE_BOWL_RACK_ALLOW_PICKUP_RETRY_FOR_SEQUENCE_PREVIEW_FAILURE", "1")
    _force_env("OPENFORGE_BOWL_RACK_GO_HOME_BETWEEN_PICKUP_RETRIES", "1")
    _force_env("OPENFORGE_BOWL_RACK_RELOCALIZE_BETWEEN_PICKUP_RETRIES", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_TARGET_MODE", "shelf_center")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_TARGET_X_BIAS_M", "-0.015")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_Z_FROM_BOWL_HEIGHT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_CLEARANCE_M", "0.035")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_AXES", "xy")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_MAX_M", "0.180")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_REORIENT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_DEG", "90.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_CANDIDATES_DEG", "90.0,60.0,45.0,30.0,-90.0,-60.0,-45.0,-30.0,0.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_OBJECT_OFFSET_AXES", "xy")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_MIN_TCP_ABOVE_SUPPORT_M", "0.020")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_ABOVE_Z_LIFT_M", "0.080")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_DIRECT_MAX_XY_DELTA_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_MAX_Z_DELTA_M", "0.120")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_XY_DELTA_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_Z_DELTA_M", "0.120")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_IK_ERROR_M", "0.008")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_Z_ONLY_MAX_ACTIVE_XY_DRIFT_M", "0.003")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_STEP_S", "0.40")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_STEP_S", "0.65")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_STEP_S", "0.65")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_STEP_S", "0.40")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLAYBACK_SPEED", "1.20")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_STEP_S", "0.35")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_PLAYBACK_SPEED", "1.20")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.04,0.01,0.00")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TARGET_POS", "0.00")


def _install_run_script_tools(common: Any) -> None:
    for name in (
        "detect_objects_oneshot",
        "freespace_move",
        "get_camera_image",
        "get_robot_state",
        "go_home",
        "open_gripper",
        "run_in_background",
        "set_gripper",
    ):
        value = globals().get(name)
        if callable(value):
            setattr(common, name, value)


_configure_orange_defaults()

from cap.saved_scripts import bowl_lower_rack_common as _common  # noqa: E402

_install_run_script_tools(_common)
TASK_RESULT = _common.run_task()


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)
