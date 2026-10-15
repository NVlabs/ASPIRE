from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cap_harness import cli


def test_parser_supports_doctor_manifest_validate_and_run():
    doctor = cli.parse_args(
        [
            "doctor",
            "--cpu-only",
            "--skip-services",
            "--reset",
            "--reset-suite",
            "libero_goal_task",
            "--reset-task-id",
            "2",
        ]
    )
    manifest = cli.parse_args(["manifest", "--output", "manifest.json"])
    validate = cli.parse_args(["validate", "--nightly", "--retry-failures"])
    run = cli.parse_args(
        [
            "run",
            "--suite",
            "libero_object_swap",
            "--task-id",
            "0",
            "--program",
            "policy.py",
        ]
    )

    assert doctor.command == "doctor"
    assert doctor.unit_mode and doctor.reset and doctor.reset_task_id == 2
    assert manifest.command == "manifest" and manifest.output == Path("manifest.json")
    assert validate.command == "validate"
    assert validate.nightly and validate.retry_failures
    assert run.command == "run" and run.output_root == Path("outputs")
    # Unset at parse time on purpose: the episode length is resolved from the
    # task's own stated horizon in run_program, not pinned to one number here.
    assert run.max_steps is None and run.program == Path("policy.py")
    assert (run.camera_width, run.camera_height) == (800, 512)
    assert not hasattr(run, "grasp_provider")
    assert not hasattr(run, "ik_provider")
    assert not hasattr(run, "trajectory_planner")
    assert not hasattr(run, "pose_planner")


def test_root_help_does_not_import_validation_or_libero(monkeypatch):
    monkeypatch.delitem(sys.modules, "cap_harness.validation", raising=False)
    monkeypatch.delitem(sys.modules, "cap_harness.libero", raising=False)

    with pytest.raises(SystemExit) as raised:
        cli.parse_args(["--help"])

    assert raised.value.code == 0
    assert "cap_harness.validation" not in sys.modules
    assert "cap_harness.libero" not in sys.modules


def test_main_dispatches_without_loading_production_runtime(monkeypatch):
    called = []

    def handler(args: argparse.Namespace) -> int:
        called.append((args.command, args.nightly))
        return 7

    monkeypatch.setattr(cli, "_validate_command", handler)

    assert cli.main(["validate", "--nightly"]) == 7
    assert called == [("validate", True)]


def test_validate_command_returns_failure_exit(monkeypatch, tmp_path):
    fake_validation = SimpleNamespace(
        validate_matrix=lambda *args, **kwargs: {
            "passed": 79,
            "expected": 80,
            "failed": 1,
            "remaining": 0,
            "success": False,
        }
    )
    monkeypatch.setitem(sys.modules, "cap_harness.validation", fake_validation)
    args = SimpleNamespace(
        output_dir=tmp_path,
        config=None,
        nightly=False,
        retry_failures=False,
    )

    assert cli._validate_command(args) == 1
