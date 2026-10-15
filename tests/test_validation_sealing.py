"""Prove the runner's sealed-run verification reproduces the campaign guarantees.

This is the prerequisite for retiring robosuite/campaign.py: every provenance
check the campaign performed must be enforced by the shared engine's
_verify_sealed_run. Exercised on fabricated run trees, no simulator needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from cap_harness.validation.model import ValidationCase
from cap_harness.validation.runner import _verify_sealed_run

COMMIT = "0" * 40
ENV_SHA = "e" * 64
PROGRAM_SHA = "p" * 64
LOCK_SHA = "l" * 64


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _good_run(root: Path, *, max_steps: int) -> Path:
    run_dir = root / "run-1"
    _write(
        run_dir / "run.json",
        {
            "environment_sha256": ENV_SHA,
            "capture": {
                "profile": "balanced_evidence",
                "camera_width": 512,
                "camera_height": 512,
                "videos": True,
                "max_steps": max_steps,
            },
        },
    )
    _write(
        run_dir / "outcome.json",
        {
            "program_ok": True,
            "task_success": True,
            "protocol_success": True,
            "termination_reason": "task_succeeded",
            "finalization_errors": [],
        },
    )
    _write(
        run_dir / "source/provenance.json",
        {
            "sha256": PROGRAM_SHA,
            "harness_git": {"commit": COMMIT, "dirty": False},
            "dependency_lock_sha256": LOCK_SHA,
        },
    )
    videos_dir = run_dir / "media/videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for name in ("agentview.mp4", "robot0_eye_in_hand.mp4", "robot1_eye_in_hand.mp4"):
        (videos_dir / name).write_bytes(b"\x00\x01\x02")
    _write(
        run_dir / "media/video-index.json",
        {
            "fps": 20.0,
            "frames": {
                "agentview": 100,
                "robot0_eye_in_hand": 100,
                "robot1_eye_in_hand": 100,
            },
        },
    )
    return run_dir


def _case() -> ValidationCase:
    return ValidationCase(id="two-arm-lift", kind="program", seeds=(1,), max_steps=3000)


def _verify(run_dir: Path):
    return _verify_sealed_run(
        run_dir,
        _case(),
        program_sha256=PROGRAM_SHA,
        commit=COMMIT,
        dependency_lock_sha256=LOCK_SHA,
        environment_sha256=ENV_SHA,
    )


def test_valid_sealed_run_has_no_issues(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    assert _verify(run_dir) == []


def test_wrong_environment_hash_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    data = json.loads((run_dir / "run.json").read_text())
    data["environment_sha256"] = "different"
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("environment_sha256" in i for i in _verify(run_dir))


def test_bad_capture_profile_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    data = json.loads((run_dir / "run.json").read_text())
    data["capture"]["camera_width"] = 256
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("capture profile" in i for i in _verify(run_dir))


def test_non_success_termination_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    data = json.loads((run_dir / "outcome.json").read_text())
    data["termination_reason"] = "step_limit"
    (run_dir / "outcome.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("termination_reason" in i for i in _verify(run_dir))


def test_finalization_errors_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    data = json.loads((run_dir / "outcome.json").read_text())
    data["finalization_errors"] = ["boom"]
    (run_dir / "outcome.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("finalization errors" in i for i in _verify(run_dir))


def test_missing_required_video_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    (run_dir / "media/videos/robot1_eye_in_hand.mp4").unlink()
    assert any("required" in i for i in _verify(run_dir))


def test_empty_video_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    (run_dir / "media/videos/agentview.mp4").write_bytes(b"")
    assert any("empty" in i for i in _verify(run_dir))


def test_video_index_frames_mismatch_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    _write(run_dir / "media/video-index.json", {"frames": {"agentview": 100}})
    assert any("video-index frames" in i for i in _verify(run_dir))


def test_program_provenance_mismatch_is_flagged(tmp_path):
    run_dir = _good_run(tmp_path, max_steps=3000)
    _write(
        run_dir / "source/provenance.json",
        {
            "sha256": "wrong",
            "harness_git": {"commit": COMMIT, "dirty": True},
            "dependency_lock_sha256": LOCK_SHA,
        },
    )
    issues = _verify(run_dir)
    assert any("program provenance hash" in i for i in issues)
    assert any("clean sealed commit" in i for i in issues)


def test_capture_max_steps_must_match_case(tmp_path):
    # A run captured at a different max_steps than the case declares is rejected.
    run_dir = _good_run(tmp_path, max_steps=9999)
    assert any("capture profile" in i for i in _verify(run_dir))
