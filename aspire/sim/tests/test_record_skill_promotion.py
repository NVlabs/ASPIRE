# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "libero" / "record_skill_promotion.py"


def load_script():
    spec = importlib.util.spec_from_file_location("record_skill_promotion", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_campaign(tmp_path: Path) -> Path:
    root = tmp_path / "sim"
    skills = root / ".claude" / "libero" / "skills"
    findings = (
        root
        / "outputs"
        / "libero_fix_loop"
        / "libero_goal_swap"
        / "example_task"
        / "findings.md"
    )
    skills.mkdir(parents=True)
    findings.parent.mkdir(parents=True)
    (skills / "grasp.md").write_text("# Grasp\n")
    (skills / "transport.md").write_text("# Transport\n")
    findings.write_text("# Findings\n")
    return root


def test_records_exact_per_task_patch_and_hashes(tmp_path: Path):
    promotion = load_script()
    root = make_campaign(tmp_path)

    begin = promotion.begin_promotion(
        root,
        suite="libero_goal_swap",
        task="example_task",
        timestamp=lambda: "2026-01-01T00:00:00+00:00",
    )
    (root / ".claude/libero/skills/grasp.md").write_text("# Grasp\n\nReusable pattern.\n")
    record = promotion.finish_promotion(
        root,
        suite="libero_goal_swap",
        task="example_task",
        timestamp=lambda: "2026-01-01T00:01:00+00:00",
    )

    assert begin["library_before_sha256"] != record["library_after_sha256"]
    assert record["changed_skill_files"] == [".claude/libero/skills/grasp.md"]
    assert "Reusable pattern." in (root / record["patch_path"]).read_text()
    ledger = root / "outputs/libero_fix_loop/libero_goal_swap/skill_promotions.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert entries == [record]
    assert promotion.verify_promotion(
        root, suite="libero_goal_swap", task="example_task"
    ) == record
    assert len(ledger.read_text().splitlines()) == 1


def test_requires_serial_promotions(tmp_path: Path):
    promotion = load_script()
    root = make_campaign(tmp_path)

    promotion.begin_promotion(root, suite="libero_goal_swap", task="example_task")
    with pytest.raises(ValueError, match="another promotion is unfinished"):
        promotion.begin_promotion(root, suite="libero_goal_swap", task="second_task")


def test_no_op_requires_and_records_reason(tmp_path: Path):
    promotion = load_script()
    root = make_campaign(tmp_path)

    promotion.begin_promotion(root, suite="libero_goal_swap", task="example_task")
    with pytest.raises(ValueError, match="pass --reason"):
        promotion.finish_promotion(
            root, suite="libero_goal_swap", task="example_task"
        )
    record = promotion.finish_promotion(
        root,
        suite="libero_goal_swap",
        task="example_task",
        reason="No generalizable Stage 1 finding.",
    )
    assert record["no_op"]
    assert record["changed_skill_files"] == []
    assert record["library_before_sha256"] == record["library_after_sha256"]
    assert record["reason"] == "No generalizable Stage 1 finding."


def test_verify_fails_before_promotion_finishes(tmp_path: Path):
    promotion = load_script()
    root = make_campaign(tmp_path)

    promotion.begin_promotion(root, suite="libero_goal_swap", task="example_task")
    with pytest.raises(ValueError, match="promotion is not complete"):
        promotion.verify_promotion(
            root, suite="libero_goal_swap", task="example_task"
        )
