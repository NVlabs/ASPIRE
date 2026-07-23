# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal real-YAM reset script.

Run only with the physical workspace clear and E-stop reachable. This script
uses the existing go_home tool; it does not start perception or grasping.
"""

import os

from skill_library.namespace import get_robot_state, go_home


TASK_RESULT = {
    "success": False,
    "reward": 0.0,
    "method": "reset_robot_pose_go_home",
}


def get_task_info():
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
    raise RuntimeError(
        "Refusing physical motion. Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
        "only after workspace is clear, hands are clear, and E-stop is reachable."
    )

print("[reset_robot_pose] Reading initial robot state...")
try:
    print(get_robot_state())
except Exception as exc:
    print(f"[reset_robot_pose] Initial state read failed: {exc}")

print("[reset_robot_pose] Calling go_home()...")
go_home()

print("[reset_robot_pose] Reading final robot state...")
try:
    print(get_robot_state())
except Exception as exc:
    print(f"[reset_robot_pose] Final state read failed: {exc}")

TASK_RESULT.update({"success": True, "reward": 1.0})

print("[reset_robot_pose] Done.")
