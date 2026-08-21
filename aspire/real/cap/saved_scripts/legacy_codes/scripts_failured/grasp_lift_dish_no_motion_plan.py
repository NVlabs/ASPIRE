# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dish/plate rim-grasp no-motion planner for YAM.

This entrypoint keeps the canonical plate planner reusable while giving the
dish task its own run directories and prompt ladder. It is observe/plan only;
the underlying planner refuses OPENFORGE_ALLOW_PHYSICAL_MOTION.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any


os.environ.setdefault("OPENFORGE_PLATE_TASK_NAME", "grasp_lift_dish_no_motion_plan")
os.environ.setdefault("OPENFORGE_PLATE_OBJECT_NAME", "dish/plate")
os.environ.setdefault("OPENFORGE_PLATE_PROMPTS", "dish,plate,dinner plate,white plate")
_entrypoint = Path.cwd() / "cap" / "saved_scripts" / "grasp_lift_dish_no_motion_plan.py"
os.environ.setdefault("OPENFORGE_PLATE_ENTRYPOINT", str(_entrypoint.resolve()))
os.environ.setdefault("OPENFORGE_PLATE_PLANNER_BACKEND", "rrtconnect")
os.environ.setdefault("OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M", "0.025")
os.environ.setdefault("OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG", "16")

_planner_path = Path.cwd() / "cap" / "saved_scripts" / "pick_plate_no_motion_plan.py"
_planner_globals = runpy.run_path(str(_planner_path), init_globals=dict(globals()))
TASK_RESULT: dict[str, Any] = dict(_planner_globals.get("TASK_RESULT") or {})


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)
