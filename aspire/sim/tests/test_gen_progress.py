# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for scripts/libero/gen_progress.py task discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "libero" / "gen_progress.py"

GOAL_SWAP_TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
]


def _load_gen_progress():
    spec = importlib.util.spec_from_file_location("gen_progress", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_suite_dirs(tmp_path: Path, suite: str, tasks: list[str], extra_dirs: list[str] | None = None):
    """Create fake suite output directories with task subdirs and optional metadata dirs."""
    suite_dir = tmp_path / "outputs" / "libero_fix_loop" / suite
    suite_dir.mkdir(parents=True)
    for task in tasks:
        (suite_dir / task).mkdir()
    for d in (extra_dirs or []):
        (suite_dir / d).mkdir()
    return suite_dir


def test_skill_promotions_excluded_from_tasks(tmp_path, monkeypatch):
    """skill_promotions directory must never appear as a task."""
    mod = _load_gen_progress()

    suite_dir = _make_suite_dirs(
        tmp_path, "libero_goal_swap", GOAL_SWAP_TASKS, extra_dirs=["skill_promotions"]
    )
    (suite_dir / "skill_promotions" / "before").mkdir(parents=True)

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "BUILD", tmp_path / "outputs" / "libero_fix_loop")
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "outputs" / "libero_baseline_image_diff_gemini")
    monkeypatch.setattr(mod, "FIXED", tmp_path / "outputs" / "libero_fix_loop_eval")
    out_file = tmp_path / "docs" / "progress" / "progress.md"
    monkeypatch.setattr(mod, "OUT_FILE", out_file)

    import sys
    monkeypatch.setattr(sys, "argv", ["gen_progress.py"])
    mod.main()

    output = out_file.read_text()

    assert "skill_promotions" not in output


def test_exactly_ten_goal_swap_tasks(tmp_path, monkeypatch):
    """With all ten goal_swap task dirs plus skill_promotions, exactly 10 tasks appear."""
    mod = _load_gen_progress()

    suite_dir = _make_suite_dirs(
        tmp_path, "libero_goal_swap", GOAL_SWAP_TASKS, extra_dirs=["skill_promotions"]
    )

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "BUILD", tmp_path / "outputs" / "libero_fix_loop")
    monkeypatch.setattr(mod, "BASELINE", tmp_path / "outputs" / "libero_baseline_image_diff_gemini")
    monkeypatch.setattr(mod, "FIXED", tmp_path / "outputs" / "libero_fix_loop_eval")
    out_file = tmp_path / "docs" / "progress" / "progress.md"
    monkeypatch.setattr(mod, "OUT_FILE", out_file)
    monkeypatch.setattr(mod, "SUITES", ["libero_goal_swap"])

    import sys
    monkeypatch.setattr(sys, "argv", ["gen_progress.py"])
    mod.main()

    output = out_file.read_text()

    task_rows = [
        line for line in output.splitlines()
        if line.startswith("| ") and "pending" in line
    ]
    assert len(task_rows) == 10, f"Expected 10 task rows, got {len(task_rows)}: {task_rows}"

    for task in GOAL_SWAP_TASKS:
        assert task in output or mod.shorten(task) in output, f"Missing task: {task}"
