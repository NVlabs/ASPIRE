"""One-shot green-apple-to-basket script for real YAM.

This is the compact operator path: run one script once, rather than manually
stepping through observe/plan/pregrasp/grasp/lift/place stages.
"""

from __future__ import annotations

import os

from skill_library.pick_place import pick_and_place


OBJECT_NAME = os.environ.get("OPENFORGE_ONE_SHOT_OBJECT", "green apple")
TARGET_NAME = os.environ.get("OPENFORGE_ONE_SHOT_TARGET", "basket")
GRASP_MODE = os.environ.get("OPENFORGE_ONE_SHOT_GRASP_MODE", "topdown_geom")
CAMERA = os.environ.get("OPENFORGE_ONE_SHOT_CAMERA", "top")
SIDE = os.environ.get("OPENFORGE_ONE_SHOT_SIDE", "").strip().lower() or None

TASK_RESULT = {
    "success": False,
    "reward": 0.0,
    "method": "one_shot_pick_place_apple_to_basket",
    "object": OBJECT_NAME,
    "target": TARGET_NAME,
    "grasp_mode": GRASP_MODE,
    "camera": CAMERA,
}


def get_task_info():
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return default if not raw else float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if not raw else int(raw)


if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
    raise RuntimeError(
        "Refusing one-shot physical motion. Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
        "only after the operator is present, E-stop is reachable, workspace is clear, "
        "and this exact one-shot task is authorized."
    )

print("[one_shot] Starting one-shot task.")
print(
    "[one_shot] "
    f"object={OBJECT_NAME!r}, target={TARGET_NAME!r}, "
    f"grasp_mode={GRASP_MODE!r}, camera={CAMERA!r}, side={SIDE or 'auto'!r}"
)

ok = pick_and_place(
    OBJECT_NAME,
    TARGET_NAME,
    grasp_mode=GRASP_MODE,
    camera=CAMERA,
    force_side=SIDE,
    max_attempts=_int_env("OPENFORGE_ONE_SHOT_MAX_ATTEMPTS", 1),
    max_grasps=_int_env("OPENFORGE_ONE_SHOT_MAX_GRASPS", 16),
    top_grasp_try=_int_env("OPENFORGE_ONE_SHOT_TOP_GRASP_TRY", 16),
    target_drop_z_offset=_float_env("OPENFORGE_ONE_SHOT_TARGET_DROP_Z_OFFSET", 0.14),
    topdown_width_m=_float_env("OPENFORGE_ONE_SHOT_TOPDOWN_WIDTH_M", 0.09),
    topdown_z_offset_m=_float_env("OPENFORGE_ONE_SHOT_TOPDOWN_Z_OFFSET_M", 0.0),
    planning_speed=_float_env("OPENFORGE_ONE_SHOT_PLANNING_SPEED", 1.5),
    ik_error_threshold=_float_env("OPENFORGE_ONE_SHOT_IK_ERROR_THRESHOLD_M", 0.01),
    place_aware_arm_selection=_truthy_env("OPENFORGE_ONE_SHOT_PLACE_AWARE")
    or os.environ.get("OPENFORGE_ONE_SHOT_PLACE_AWARE", "").strip() == "",
)

TASK_RESULT.update(
    {
        "success": bool(ok),
        "reward": 1.0 if ok else 0.0,
        "physical_motion_executed": True,
        "why_stopped": "one-shot task completed" if ok else "one-shot task returned failure",
    }
)

print(f"[one_shot] success={TASK_RESULT['success']}")
