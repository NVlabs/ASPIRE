"""The BEHAVIOR fix-loop skill folder and the shared scripts' benchmark awareness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude/skills/iterative-debugging/behavior"
SCRIPTS = ROOT / ".claude/skills/iterative-debugging/scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_behavior_skill_folder_is_complete_and_self_consistent() -> None:
    for name in ("SKILL.md", "subagent-prompt.md", "main-agent-prompt.md", "clean-task-slate.md"):
        assert (SKILL / name).is_file(), name
    for name in (
        "README.md",
        "search.md",
        "navigation.md",
        "mobile-grasp.md",
        "time-budget.md",
        "isaac-operations.md",
    ):
        assert (SKILL / "skills" / name).is_file(), name
    subagent = (SKILL / "subagent-prompt.md").read_text(encoding="utf-8")
    coordinator = (SKILL / "main-agent-prompt.md").read_text(encoding="utf-8")
    assert ".venv-behavior/bin/cap-harness" in subagent
    assert "OMNIGIBSON_GPU_ID" in subagent and "`seg_instance`" in subagent
    assert "odom" in subagent and "behavior.navigate_to_pose" in subagent
    assert "Non-privileged API audit: PASS/QUARANTINED" in subagent
    assert "never rewrite it to hide its provenance" in subagent
    assert "Non-privileged API audit: PASS" in coordinator
    assert "report the seed as invalid" in coordinator
    parent = (ROOT / ".claude/skills/iterative-debugging/SKILL.md").read_text(encoding="utf-8")
    assert "behavior/SKILL.md" in parent


def test_behavior_protocol_is_block_by_block_with_a_frozen_skill_library() -> None:
    """The BEHAVIOR campaign grows one policy per seed and freezes a library, not a program."""
    overview = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    subagent = (SKILL / "subagent-prompt.md").read_text(encoding="utf-8")
    coordinator = (SKILL / "main-agent-prompt.md").read_text(encoding="utf-8")
    slate = (SKILL / "clean-task-slate.md").read_text(encoding="utf-8")

    # The policy is grown one block at a time and the whole file is replayed every attempt.
    for text in (overview, subagent):
        assert "# Code block" in text
    assert "grown block by block" in subagent
    assert "result = report` is always the **last** line" in subagent
    assert "attempts/$A" in subagent and "attempt_%03d" in subagent

    # Seed partitions follow the ASPIRE behavior fix loop, not the parent's.
    for text in (overview, subagent, coordinator):
        assert "26-35" in text and "1-25" in text
    assert "51" not in overview.replace("512", "").replace("8115", "")

    # Stage 2 is agentic and append-only against a frozen library; policies are never frozen.
    assert "append only" in subagent and "append-only" in coordinator
    assert "skill-library-frozen" in subagent and "skill-library-frozen" in coordinator
    assert "frozen-manifest.sha256" in coordinator and "frozen-manifest.sha256" in subagent
    assert "sha256sum -c" in coordinator and "sha256sum -c" in slate
    assert "Policies are **not** frozen" in coordinator

    # The shipped examples are documentation, not campaign material.
    assert "examples/behavior/" in overview and "never reads them" in overview
    assert "must not enter a campaign" in coordinator

    # The parent's replay path is gone from this folder.
    for text in (overview, subagent, coordinator, slate):
        assert "run_validation.py" not in text
        assert "fix_code.py" not in text


def test_validation_script_environment_is_benchmark_aware(monkeypatch, tmp_path: Path) -> None:
    validation = _load("run_validation")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OMNIGIBSON_DATA_PATH", raising=False)
    monkeypatch.delenv("OMNIGIBSON_APPDATA_PATH", raising=False)
    monkeypatch.delenv("CAP_HARNESS_BEHAVIOR_DATA", raising=False)
    behavior = validation.benchmark_environment("behavior", 2, root=tmp_path)
    assert behavior["OMNIGIBSON_GPU_ID"] == "2" and behavior["CUDA_VISIBLE_DEVICES"] == "2"
    assert behavior["OMNIGIBSON_HEADLESS"] == "1"
    assert behavior["OMNIGIBSON_DATA_PATH"] == str(tmp_path / "behavior-data")
    assert behavior["OMNIGIBSON_APPDATA_PATH"] == str(tmp_path / ".behavior-appdata")
    assert "MUJOCO_GL" not in behavior
    mujoco = validation.benchmark_environment("libero-pro", 1, root=tmp_path)
    assert mujoco["MUJOCO_EGL_DEVICE_ID"] == "1" and mujoco["MUJOCO_GL"] == "egl"
    assert "OMNIGIBSON_GPU_ID" not in mujoco


def test_scoring_criteria_are_written_down_and_executable() -> None:
    """A fresh coordinator must not have to be told how to count a campaign.

    Exactly two numbers are reported, both per seed and both on the final policy:
    success (succeeded at least once) and navigation (reached the object). The first
    campaign reported several rates side by side and they were confused for one another,
    so this test also pins that the retired ones stay out of the documents.
    """
    folder = ROOT / ".claude" / "skills" / "iterative-debugging" / "behavior"

    def _flat(path: Path) -> str:
        # Markdown wraps prose freely, so phrase checks must not depend on line breaks.
        return " ".join(path.read_text(encoding="utf-8").split())

    overview = _flat(folder / "SKILL.md")
    coordinator = _flat(folder / "main-agent-prompt.md")
    subagent = _flat(folder / "subagent-prompt.md")
    script = folder / "score_campaign.py"

    # The unit of scoring is the final policy, stated in all three places.
    for text in (overview, coordinator, subagent):
        assert "final policy" in text.lower()
    assert "## Scoring" in coordinator

    # Success is one success on the final policy, regardless of reliability.
    assert "succeeded at least once" in coordinator
    assert "succeeded at least once" in subagent
    assert "succeeded at least once" in overview

    # Navigation is judged on the same final policy, and legs are not a navigation rate.
    assert "Navigation" in coordinator and "NAVIGATION" in subagent
    assert "navigate_to_pose" in coordinator and "repair ladder" in coordinator

    # Development attempts never enter scoring.
    for text in (overview, coordinator, subagent):
        assert "no grasp block" in text

    # The retired metrics are not offered anywhere an agent reads.
    for retired in ("episodes held", "final episode held", "middle row", "reliability number"):
        assert retired not in coordinator, f"retired metric still documented: {retired!r}"
    assert "holds over the episodes" not in subagent

    # Infrastructure losses are excluded and named.
    for text in (coordinator, subagent):
        assert "out-of-memory" in text

    # The script exists, finds the final policy by hashing the program, and prints only the two.
    assert script.is_file()
    body = script.read_text(encoding="utf-8")
    assert "source" in body and "program.py" in body and "sha256" in body
    assert "SUCCESS" in body and "NAVIGATION" in body
    for retired in ("EPISODES HELD", "FINAL EPISODE HELD", "navigate_to_pose legs"):
        assert retired not in body, f"script still prints retired metric: {retired!r}"
    assert "score_campaign.py" in overview
