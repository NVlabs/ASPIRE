"""The equivalence checker must never certify a comparison that did not happen."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/check_batch_equivalence.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_batch_equivalence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _outcome(success: bool, steps: int = 100) -> dict:
    return {
        "task_success": success,
        "program_ok": True,
        "termination_reason": "task_succeeded" if success else "program_completed",
        "steps_executed": steps,
    }


@pytest.mark.parametrize("produce_outcomes", [False, True])
def test_reruns_only_score_fresh_evidence(checker, tmp_path, monkeypatch, produce_outcomes) -> None:
    program = tmp_path / "program.py"
    program.write_text("result = True\n")
    output = tmp_path / "out"
    old_files = []
    for leg in ("unbatched-a", "unbatched-b", "batched"):
        old = output / leg / "0051" / "old-run"
        old.mkdir(parents=True)
        outcome = old / "outcome.json"
        outcome.write_text(json.dumps(_outcome(True)))
        old_files.append(outcome)

    monkeypatch.setattr(
        "sys.argv",
        [
            str(MODULE_PATH),
            "--suite",
            "libero_goal_swap",
            "--task-id",
            "0",
            "--program",
            str(program),
            "--gpu",
            "0",
            "--seeds",
            "51",
            "--output-root",
            str(output),
            "--keep",
        ],
    )
    invoked_roots = []

    def execute(command, **kwargs):
        result_root = Path(command[command.index("--output-root") + 1])
        invoked_roots.append(result_root)
        if produce_outcomes:
            run = result_root / "0051" / "new-run"
            run.mkdir(parents=True)
            (run / "outcome.json").write_text(json.dumps(_outcome(False)))
        return subprocess.CompletedProcess(command, 0 if produce_outcomes else 1)

    monkeypatch.setattr(checker.subprocess, "run", execute)
    verdicts = []
    for _ in range(2):
        assert checker.main() == (0 if produce_outcomes else 2)
        verdict = json.loads((output / "verdict.json").read_text())
        verdicts.append(verdict)
        assert verdict["conclusive"] is produce_outcomes
        assert verdict["equivalent"] is produce_outcomes
        assert verdict["seeds_that_never_ran"] == ([] if produce_outcomes else [51])
        assert all(root.parent == Path(verdict["results_root"]) for root in invoked_roots[-3:])
        assert (Path(verdict["results_root"]) / "verdict.json").is_file()
    assert verdicts[0]["results_root"] != verdicts[1]["results_root"]
    assert all(json.loads(path.read_text()) == _outcome(True) for path in old_files)


def test_agreement_is_reported_as_agreement(checker, capsys) -> None:
    left = {1: _outcome(True), 2: _outcome(False)}
    differing, missing = checker.compare("t", left, dict(left), [1, 2])

    assert differing == [] and missing == []


def test_a_changed_outcome_is_a_difference_not_a_missing_run(checker) -> None:
    left = {1: _outcome(True, steps=100)}
    right = {1: _outcome(True, steps=140)}
    differing, missing = checker.compare("t", left, right, [1])

    assert differing == [1]
    assert missing == []


def test_a_seed_that_never_ran_is_counted_apart_from_one_that_disagreed(checker) -> None:
    """Distinguish missing runs from runs whose outcomes disagree.

    The bug this guards: a run that produced nothing was folded in with runs
    that produced a different answer. When every seed failed to launch, both
    arms 'differed' on all of them, the counts matched, and the checker
    certified equivalence having executed zero episodes.
    """
    left = {1: _outcome(True), 2: None, 3: _outcome(True)}
    right = {1: _outcome(True), 2: None, 3: _outcome(False)}
    differing, missing = checker.compare("t", left, right, [1, 2, 3])

    assert missing == [2]
    assert differing == [3]


def test_a_total_failure_to_launch_yields_no_differences_to_compare(checker) -> None:
    nothing = dict.fromkeys((1, 2, 3))
    differing, missing = checker.compare("t", nothing, dict(nothing), [1, 2, 3])

    # Nothing disagreed, because nothing ran. The caller must gate on `missing`.
    assert differing == []
    assert missing == [1, 2, 3]


def test_the_script_defaults_the_config_path_libero_prompts_for(checker) -> None:
    """Set a noninteractive LIBERO configuration path by default.

    Unset, LIBERO reads stdin at import and a child with no terminal dies
    with a bare EOFError that names nothing about the real cause.
    """
    source = MODULE_PATH.read_text()
    assert 'env.setdefault("LIBERO_CONFIG_PATH"' in source
    assert "EOFError" in source  # the failure it explains, for whoever hits it


def test_an_inconclusive_run_is_never_reported_as_equivalent(checker) -> None:
    source = MODULE_PATH.read_text()
    assert "INCONCLUSIVE" in source
    assert '"conclusive": not missing' in source
    assert '"equivalent": (not missing) and len(changed) <= len(noise)' in source
