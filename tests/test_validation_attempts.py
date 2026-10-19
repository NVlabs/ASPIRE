"""Program validation must use the current invocation's evidence.

Controlled run commands write records; no simulator or provider is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cap_harness.validation.model import ValidationCase, ValidationPlan
from cap_harness.validation.runner import run_plan


@pytest.fixture
def plan(tmp_path):
    (tmp_path / "program.py").write_text("pass\n", encoding="utf-8")
    return ValidationPlan(
        schema_version=1,
        name="attempts",
        benchmark="behavior",
        cases=(
            ValidationCase(
                id="task",
                kind="program",
                seeds=(1,),
                suite="controlled",
                program="program.py",
                max_steps=10,
                require={"program_ok": True, "task_success": True},
            ),
        ),
    )


def _root(argv):
    return Path(argv[argv.index("--output-root") + 1])


def _write(root, *, name="run-current", success=True):
    directory = root / "0001" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "outcome.json").write_text(
        json.dumps({"program_ok": True, "task_success": success}), encoding="utf-8"
    )
    return directory


def _run(plan, tmp_path, command):
    return run_plan(plan, tmp_path / "output", repo_root=tmp_path, run_command=command)


@pytest.mark.parametrize("exit_code", [0, 139])
def test_current_success_survives_teardown_exit(plan, tmp_path, exit_code):
    # BEHAVIOR may exit 139 after finalizing a valid outcome (docs/run-artifacts.md).
    def command(argv, env):
        del env
        _write(_root(argv))
        return exit_code

    summary = _run(plan, tmp_path, command)
    assert summary["success"] is True
    assert summary["cases"][0]["seeds"][0]["detail"]["exit_code"] == exit_code


@pytest.mark.parametrize("exit_code", [0, 1])
def test_no_outcome_fails(plan, tmp_path, exit_code):
    summary = _run(plan, tmp_path, lambda *_: exit_code)
    assert summary["success"] is False


@pytest.mark.parametrize("exit_code", [0, 1, 139])
def test_previous_success_cannot_pass_an_empty_attempt(plan, tmp_path, exit_code):
    old = []

    def first(argv, env):
        del env
        old.append(_write(_root(argv)))
        return 0

    assert _run(plan, tmp_path, first)["success"] is True
    path = old[0] / "outcome.json"
    before = path.read_bytes()
    summary = _run(plan, tmp_path, lambda *_: exit_code)
    assert path.read_bytes() == before
    assert summary["success"] is False


@pytest.mark.parametrize("success", [False, True])
def test_current_outcome_wins_regardless_of_old_directory_sort_order(plan, tmp_path, success):
    old = _write(tmp_path / "output/task/runs/1", name="zz-prior")
    before = (old / "outcome.json").read_bytes()
    current = []

    def command(argv, env):
        del env
        current.append(_write(_root(argv), name="aa-current", success=success))
        return 0

    summary = _run(plan, tmp_path, command)
    assert (old / "outcome.json").read_bytes() == before
    assert summary["success"] is success
    assert Path(summary["cases"][0]["seeds"][0]["detail"]["run_dir"]) == current[0]


def test_late_completion_of_previous_attempt_is_not_current(plan, tmp_path):
    roots = []

    def first(argv, env):
        del env
        roots.append(_root(argv))
        return 0

    assert _run(plan, tmp_path, first)["success"] is False

    def second(argv, env):
        del argv, env
        # The earlier producer finishes during the second invocation. Merely
        # comparing outcome paths before and after the command would accept it.
        _write(roots[0])
        return 0

    assert _run(plan, tmp_path, second)["success"] is False
    assert (roots[0] / "0001/run-current/outcome.json").is_file()


def test_equal_fresh_outcomes_keep_distinct_evidence(plan, tmp_path):
    paths = []

    def command(argv, env):
        del env
        paths.append(_write(_root(argv), name=f"run-{len(paths)}"))
        return 0

    assert _run(plan, tmp_path, command)["success"] is True
    before = (paths[0] / "outcome.json").read_bytes()
    assert _run(plan, tmp_path, command)["success"] is True
    assert paths[0] != paths[1]
    assert (paths[0] / "outcome.json").read_bytes() == before
    assert (paths[1] / "outcome.json").read_bytes() == before
