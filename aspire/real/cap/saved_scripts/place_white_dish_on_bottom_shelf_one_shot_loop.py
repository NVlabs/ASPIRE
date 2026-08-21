# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""White-dish bottom-shelf rack entrypoint for real YAM.

This script configures a fresh live pickup for an 8 inch wide, 2 inch deep
white dish with a top-down rim straddle, then delegates the lower-rack
placement sequence to
``bowl_lower_rack_common``:

pickup/lift -> wrist rotate -> direct Y align -> direct X insert -> direct
Z place-down -> release -> direct Z clear -> direct X retreat -> home.

The lower-rack implementation rejects manual XYZ inputs, runs a no-motion
planner preview first, and executes physical motion only when the physical gate
and dish/bottom-shelf ticket preconditions are present.
"""

from __future__ import annotations

import os
from typing import Any

from cap.saved_scripts.full_demo_speed import set_scaled_default_env


DISH_DIAMETER_M = 0.2032
DISH_RADIUS_M = DISH_DIAMETER_M / 2.0
DISH_DEPTH_M = 0.0508


def _force_env(name: str, value: str) -> None:
    os.environ[name] = value


def _default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _configure_white_dish_defaults() -> None:
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OBJECT_NAME", "white dish")
    _force_env("OPENFORGE_BOWL_RACK_TASK_NAME", "place_white_dish_on_bottom_shelf_one_shot_loop")
    _force_env("OPENFORGE_BOWL_RACK_OBJECT_NAME", "white dish")
    _force_env("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", "white_dish_bottom_shelf")
    _force_env(
        "OPENFORGE_BOWL_RACK_SCRIPT_PATH",
        "cap/saved_scripts/place_white_dish_on_bottom_shelf_one_shot_loop.py",
    )
    _force_env("OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH", "cap/saved_scripts/bowl_lower_rack_common.py")
    _force_env("OPENFORGE_BOWL_RACK_VISIBLE_ENV", "OPENFORGE_BOWL_RACK_WHITE_DISH_VISIBLE")
    _force_env("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "white_dish_bottom_shelf")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", "white dish,dish,white plate,plate,dinner plate")
    _force_env("OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS", "white dish,dish,white plate,plate,dinner plate")
    _force_env("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", "white dish,dish,white plate,plate,dinner plate")
    _force_env(
        "OPENFORGE_BOWL_RACK_POST_PROMPTS",
        "white dish,dish,white plate,plate,bottom shelf,lower shelf,wooden rack,rack,blue gripper fingers",
    )
    _force_env(
        "OPENFORGE_BOWL_RACK_TARGET_PROMPTS",
        "bottom shelf,lower shelf,lower rack shelf,first rack level,wooden rack,rack,wooden shelf,shelf",
    )
    _force_env("OPENFORGE_BOWL_RACK_TARGET_Y_SIGN", "negative")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_TARGET_Y_SIGN", "positive")

    _force_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_ARM", "left")
    _force_env("OPENFORGE_BOWL_RACK_COLLECT_ALL_PICKUP_PROMPTS", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "top_down_bowl_rim")
    _force_env("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "top_down_bowl_rim_straddle")
    _force_env("OPENFORGE_BOWL_RACK_ALLOW_NON_TOP_DOWN_PICKUP_STRATEGY", "0")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_AXIS_GATE", "1")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_MAX_CANDIDATES", "48")
    _force_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", "0")
    _force_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", "1")
    _force_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", "0")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", "1")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_VISIBLE_OBJECT_MISMATCH", "1")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", "0")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_CONTACT_ABOVE_OBJECT_Z_M", "0.045")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", "0")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_LIFT_PERCEPTION_BEFORE_PLACE", "0")

    _force_env("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", "0.68")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", "0.60")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_VEL_LIMIT", "1.0")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_TORQUE_LIMIT", "1.0")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_STEPS", "0.30,0.08,0.00")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", "0.00")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS", "0.02")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_TARGET_TOLERANCE_POS", "0.06")
    _force_env("OPENFORGE_BOWL_RACK_HOLD_MIN_POS", "0.000")
    _force_env("OPENFORGE_BOWL_RACK_HOLD_MAX_POS", "0.95")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_SETTLE_S", "0.02")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_TIMEOUT_S", "1.6")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_CONFIRM_TIMEOUT_S", "0.6")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_POLL_S", "0.03")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_STATE_POLL_S", "0.03")
    _force_env("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", "0.55")
    _force_env("OPENFORGE_BOWL_RACK_CAPTURE_INTERMEDIATE_OBSERVATIONS", "0")
    _force_env("OPENFORGE_BOWL_RACK_CAPTURE_POST_PLACE_OBSERVATION", "0")
    _force_env("OPENFORGE_BOWL_RACK_EXEC_STALE_CACHE_REPLAN_SLEEP_S", "0.05")
    _force_env("OPENFORGE_BOWL_RACK_GO_HOME_AFTER_SUCCESS", "1")
    _force_env("OPENFORGE_BOWL_RACK_GO_HOME_AFTER_FAILURE", "1")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_GRIPPERS_AFTER_HOME", "1")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_GRIPPERS_AFTER_HOME_SIDES", "left,right")

    _default_env("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", "0.025")
    _default_env("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", "16")
    _default_env("OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", "0.050")
    _default_env("OPENFORGE_BOWL_RACK_PREVIEW_MAX_ROT_ERROR_DEG", "12.0")
    _default_env("OPENFORGE_BOWL_RACK_EXEC_MAX_POS_ERROR_M", "0.025")
    _default_env("OPENFORGE_BOWL_RACK_EXEC_MAX_ROT_ERROR_DEG", "12.0")
    _default_env("OPENFORGE_BOWL_RACK_FRONT_IK_ROT_THRESHOLD_DEG", "30.0")
    _default_env("OPENFORGE_BOWL_RACK_FRONT_IK_RPY_WEIGHT", "0.10")
    _default_env("OPENFORGE_BOWL_RACK_PREVIEW_FRONT_MAX_ROT_ERROR_DEG", "40.0")
    _default_env("OPENFORGE_BOWL_RACK_EXEC_FRONT_MAX_ROT_ERROR_DEG", "40.0")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_PREVIEW_RELEASE_WITH_PLANNER", "0")
    set_scaled_default_env("OPENFORGE_BOWL_RACK_PLANNING_SPEED", 0.20)
    _force_env("OPENFORGE_BOWL_RACK_PREVIEW_TIMEOUT_S", "12")

    _force_env("OPENFORGE_PLATE_RADIUS_OVERRIDE_M", f"{DISH_RADIUS_M:.4f}")
    _force_env("OPENFORGE_PLATE_USE_TOP_BOX_RADIUS", "0")
    _force_env("OPENFORGE_PLATE_DEFAULT_RADIUS_M", f"{DISH_RADIUS_M:.4f}")
    _force_env("OPENFORGE_PLATE_REFERENCE_RADIUS_M", f"{DISH_RADIUS_M:.4f}")
    _force_env("OPENFORGE_PLATE_MIN_RADIUS_M", "0.090")
    _force_env("OPENFORGE_PLATE_MAX_RADIUS_M", "0.115")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RADIUS_M", f"{DISH_RADIUS_M:.4f}")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_USE_MULTIVIEW_RIM_Z", "1")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_Z_MODE", "height_over_support")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_SUPPORT_Z_M", "0.760")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_HEIGHT_M", f"{DISH_DEPTH_M:.4f}")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_Z_BIAS_M", f"{DISH_DEPTH_M:.4f}")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M", "-0.022")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_CANDIDATES_M", "-0.022,-0.028,-0.016")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LOW_PREGRASP_Z_LIFT_M", "0.065")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LOW_PREGRASP_MIN_ABOVE_RIM_M", "0.045")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LIFT_Z_M", "0.140")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LIFT_STEP_M", "0.075")
    _force_env("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MIN_Z_DELTA_M", "0.020")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG", "135,90,45,180,0,-135,-90,-45")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG", "4.0")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_INWARD_OFFSET_M", "-0.005")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_XY_BIASES_M", "0.000,-0.010,0.010")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_TO_CONTACT_OFFSETS_M", "0.000")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_FINGER_FORWARD_SIGNS", "1.0,-1.0")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_STEPS", "0.30,0.08,0.00")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TARGET_POS", "0.00")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_VEL_LIMIT", "1.0")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TORQUE_LIMIT", "1.0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_RETRY_ATTEMPTS", "8")
    _force_env("OPENFORGE_BOWL_RACK_ALLOW_PICKUP_RETRY_FOR_SEQUENCE_PREVIEW_FAILURE", "1")
    _force_env("OPENFORGE_BOWL_RACK_GO_HOME_BETWEEN_PICKUP_RETRIES", "1")
    _force_env("OPENFORGE_BOWL_RACK_RELOCALIZE_BETWEEN_PICKUP_RETRIES", "1")

    _default_env("OPENFORGE_BOWL_RACK_OBSERVATION_CAMERAS", "top")
    _default_env("OPENFORGE_DEBUG_OBS_CAMERAS", "top")
    _default_env("OPENFORGE_BOWL_RACK_IMAGE_ONLY_CAMERAS", "")
    _default_env("OPENFORGE_DEBUG_OBS_IMAGE_ONLY_CAMERAS", "")
    _default_env("OPENFORGE_DEBUG_OBS_TIMEOUT_S", "5")
    _default_env("OPENFORGE_PLATE_DETECT_RETRIES", "1")
    _default_env("OPENFORGE_BOWL_RACK_RECORDING_WAIT_S", "0.4")
    _default_env("OPENFORGE_PLATE_RECORDING_WAIT_S", "0.4")

    _force_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_Y_SIGN", "positive")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_TARGET_MODE", "shelf_center")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_TARGET_X_BIAS_M", "-0.040")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_TCP_X_MAX_M", "0.645")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_STANDOFF_M", "0.120")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_X_STANDOFF_M", "0.080")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_Z_LIFT_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_ENTRY_Z_LIFT_M", "0.000")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_SLIDE_RELEASE_Z_LIFT_M", "0.015")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_Z_FROM_BOWL_HEIGHT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_CLEARANCE_M", "0.025")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_AXES", "xy")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_RELEASE_OBJECT_OFFSET_MAX_M", "0.240")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_OBJECT_REF_ABOVE_SUPPORT_M", "0.020")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_MIN_TCP_ABOVE_SUPPORT_M", "0.020")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_M", "0.760")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_SUPPORT_Z_OFFSET_M", "0.000")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_RETREAT_STANDOFF_M", "0.140")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_XY_DELTA_M", "0.150")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_MAX_Z_DELTA_M", "0.055")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", "0.090")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_Z_DELTA_M", "0.020")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_MAX_IK_ERROR_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_REORIENT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_DEG", "-30.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_DELTA_CANDIDATES_DEG", "-30.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_COMBINE_POST_LIFT_ORIENTATION", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_UNDO_PICKUP_INWARD_TILT_AFTER_LIFT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PICKUP_INWARD_TILT_UNDO_SIGN", "1.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_COMBINED_MAX_ROT_ERROR_DEG", "4.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_REQUIRE_TANGENT_OBJECT_OFFSET", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_TANGENT_AXIS", "x")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_MIN_TANGENT_ALIGNMENT", "0.94")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_MAX_CROSS_ALIGNMENT", "0.36")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_OBJECT_OFFSET_AXES", "xy")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_ABOVE_Z_LIFT_M", "0.080")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_DIRECT_MAX_XY_DELTA_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_MAX_Z_DELTA_M", "0.120")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_XY_DELTA_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_RETREAT_MAX_Z_DELTA_M", "0.120")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_SPLIT_Y_THEN_X", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_Z", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_Y_ALIGN_MAX_XY_DELTA_M", "0.260")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_INSERT_X_MAX_XY_DELTA_M", "0.340")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_X_INSERT_MAX_IK_ERROR_M", "0.055")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_Z_LIFT_M", "0.000")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_Z_LIFT_MAX_Z_DELTA_M", "0.010")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_Z_LIFT_STEP_S", 0.90, kind="duration")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_X_UP_TILT_DEG", "20.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_MAX_TCP_TRANSLATION_M", "0.030")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_MAX_ROT_ERROR_DEG", "4.0")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_STEP_S", 0.90, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_PLAYBACK_SPEED", 0.60)
    _force_env("OPENFORGE_BOWL_LOWER_RACK_PRE_INSERT_TILT_START_INTERP_S", "0.60")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_MAX_Z_DELTA_M", "0.140")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_Z_LIFT_M", "0.100")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_Z_DELTA_M", "0.130")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_IK_ERROR_M", "0.055")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_MAX_XY_DRIFT_M", "0.035")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_AFTER_RELEASE", "1")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_Z_TO_X_RATIO", "4.0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_XY_DELTA_M", "0.060")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_MAX_Z_DELTA_M", "0.130")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_DIAGONAL_RETREAT_STEP_S", 0.90, kind="duration")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_SKIP_X_RETREAT_AFTER_RELEASE_CLEAR", "0")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_IK_ERROR_M", "0.045")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_Z_ONLY_MAX_ACTIVE_XY_DRIFT_M", "0.012")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_MAX_XY_DELTA_M", "0.090")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_STEP_S", 0.90, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_POST_RELEASE_CLEAR_STEP_S", 0.90, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_OPEN_FRONT_PLACE_DOWN_STEP_S", 1.00, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLACE_DOWN_STEP_S", 1.00, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_RETREAT_STEP_S", 0.90, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_PLAYBACK_SPEED", 0.60)
    _force_env("OPENFORGE_BOWL_LOWER_RACK_DIRECT_START_INTERP_S", "0.60")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_STEP_S", 0.90, kind="duration")
    set_scaled_default_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_PLAYBACK_SPEED", 0.60)
    _force_env("OPENFORGE_BOWL_LOWER_RACK_POST_LIFT_J5_START_INTERP_S", "0.60")
    _force_env("OPENFORGE_BOWL_RACK_PLACE_OFFSETS_M", "0.000,0.000,0.000")
    _force_env("OPENFORGE_BOWL_RACK_MAX_PLACE_CANDIDATES", "1")


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


_configure_white_dish_defaults()

from cap.saved_scripts import bowl_lower_rack_common as _common  # noqa: E402

_install_run_script_tools(_common)
TASK_RESULT = _common.run_task()


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)
