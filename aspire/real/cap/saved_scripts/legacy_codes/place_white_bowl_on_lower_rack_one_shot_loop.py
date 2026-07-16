"""White-bowl lower-rack entrypoint for real YAM.

This file owns the white bowl pickup defaults, then delegates the common
lower-rack sequence to ``bowl_lower_rack_common``:

pickup/lift -> rack front high -> rack front entry -> direct insert -> release
-> direct horizontal retreat.

The white pickup defaults are intentionally explicit so they cannot be mixed
with orange-bowl prompts or rim geometry by changing only the object name.
"""

from __future__ import annotations

import os
from typing import Any


def _force_env(name: str, value: str) -> None:
    os.environ[name] = value


def _default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _configure_white_defaults() -> None:
    _force_env("OPENFORGE_BOWL_LOWER_RACK_OBJECT_NAME", "white bowl")
    _force_env("OPENFORGE_BOWL_RACK_TASK_NAME", "place_white_bowl_on_lower_rack_one_shot_loop")
    _force_env("OPENFORGE_BOWL_RACK_OBJECT_NAME", "white bowl")
    _force_env("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", "white_bowl_lower_rack")
    _force_env("OPENFORGE_BOWL_RACK_SCRIPT_PATH", "cap/saved_scripts/place_white_bowl_on_lower_rack_one_shot_loop.py")
    _force_env("OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH", "cap/saved_scripts/bowl_lower_rack_common.py")
    _force_env("OPENFORGE_BOWL_RACK_VISIBLE_ENV", "OPENFORGE_BOWL_RACK_WHITE_BOWL_VISIBLE")
    _force_env("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "white_bowl_lower_rack")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", "white bowl,bowl")
    _force_env("OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS", "white bowl,bowl")
    _force_env("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", "white bowl,bowl")
    _force_env(
        "OPENFORGE_BOWL_RACK_POST_PROMPTS",
        "white bowl,bowl,lower rack shelf,bottom shelf,lower shelf,rack,wooden rack,blue gripper fingers",
    )
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", "0")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_ARM", "right")
    _force_env("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "top_down_bowl_rim")
    _force_env("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "top_down_bowl_rim_straddle")
    _default_env("OPENFORGE_BOWL_RACK_PICKUP_MAX_CANDIDATES", "4")
    _force_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", "0")
    _force_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", "1")
    _force_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", "1")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", "1")
    _force_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", "0")
    _default_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_CONTACT_ABOVE_OBJECT_Z_M", "0.060")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", "0")
    _force_env("OPENFORGE_BOWL_RACK_REQUIRE_LIFT_PERCEPTION_BEFORE_PLACE", "1")
    _force_env("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", "0.68")
    _force_env("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", "0.60")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_VEL_LIMIT", "1")
    _force_env("OPENFORGE_BOWL_RACK_GRIPPER_TORQUE_LIMIT", "1")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_STEPS", "0.68,0.18,0.04")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", "0.04")
    _force_env("OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS", "0.02")
    _force_env("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", "0.025")
    _force_env("OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", "0.025")
    _force_env("OPENFORGE_BOWL_RACK_EXEC_MAX_POS_ERROR_M", "0.025")
    _force_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_Y_SIGN", "positive")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_FRONT_X_STANDOFF_M", "0.080")
    _default_env("OPENFORGE_BOWL_RACK_FRONT_IK_ROT_THRESHOLD_DEG", "30.0")
    _default_env("OPENFORGE_BOWL_RACK_FRONT_IK_RPY_WEIGHT", "0.10")
    _default_env("OPENFORGE_BOWL_RACK_PREVIEW_FRONT_MAX_ROT_ERROR_DEG", "30.0")
    _default_env("OPENFORGE_BOWL_RACK_EXEC_FRONT_MAX_ROT_ERROR_DEG", "30.0")
    _default_env("OPENFORGE_BOWL_LOWER_RACK_PREVIEW_RELEASE_WITH_PLANNER", "0")
    _force_env("OPENFORGE_PLATE_USE_TOP_BOX_RADIUS", "1")
    _force_env("OPENFORGE_PLATE_REFERENCE_BOX_LONG_AXIS_PX", "56.0")
    _force_env("OPENFORGE_PLATE_REFERENCE_RADIUS_M", "0.095")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_USE_MULTIVIEW_RIM_Z", "1")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_RAISE_M", "0.035")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_XY_DISAGREEMENT_M", "0.080")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M", "-0.020")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_LIFT_Z_M", "0.150")
    _default_env("OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG", "-90")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_RIM_XY_BIASES_M", "-0.015")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_TCP_TO_CONTACT_OFFSETS_M", "0.000")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_FINGER_FORWARD_SIGNS", "-1.0")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.03")
    _force_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TARGET_POS", "0.03")


def _install_run_script_tools(common: Any) -> None:
    for name in (
        "detect_objects_oneshot",
        "freespace_move",
        "get_camera_image",
        "get_robot_state",
        "run_in_background",
        "set_gripper",
    ):
        value = globals().get(name)
        if callable(value):
            setattr(common, name, value)


_configure_white_defaults()

from cap.saved_scripts import bowl_lower_rack_common as _common  # noqa: E402

_install_run_script_tools(_common)
TASK_RESULT = _common.run_task()


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)
