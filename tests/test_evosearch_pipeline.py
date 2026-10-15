from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".claude" / "skills" / "evosearch" / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_outcome(root: Path, benchmark: str, suite: str, seed: int, run: str, **fields) -> Path:
    run_dir = root / benchmark / suite / "00-task" / f"{seed:04d}" / run
    run_dir.mkdir(parents=True)
    (run_dir / "outcome.json").write_text(json.dumps(fields))
    return run_dir


def test_discover_candidates_normalizes_bare_files(tmp_path: Path):
    evaluator = load_script("evosearch_eval")
    (tmp_path / "candidate_A.py").write_text("result = {}\n")
    directory = tmp_path / "candidate_B"
    directory.mkdir()
    (directory / "code.py").write_text("result = {}\n")
    (tmp_path / "notes.py").write_text("not a candidate\n")

    candidates = evaluator.discover_candidates(tmp_path, None)
    assert sorted(candidates) == ["candidate_A", "candidate_B"]
    assert (tmp_path / "candidate_A" / "code.py").is_file()

    # Edits to the bare file must refresh the normalized copy.
    (tmp_path / "candidate_A.py").write_text("observation = get_observation()\nresult = {}\n")
    evaluator.discover_candidates(tmp_path, None)
    assert "observation" in (tmp_path / "candidate_A" / "code.py").read_text()

    subset = evaluator.discover_candidates(tmp_path, ["candidate_B"])
    assert list(subset) == ["candidate_B"]


def test_existing_outcome_prefers_newest_run(tmp_path: Path):
    evaluator = load_script("evosearch_eval")
    old = write_outcome(
        tmp_path, "libero-pro", "suite", 51, "run-a", program_ok=True, task_success=False
    )
    new = write_outcome(
        tmp_path, "libero-pro", "suite", 51, "run-b", program_ok=True, task_success=True
    )
    import os

    os.utime(old / "outcome.json", (1, 1))

    outcome = evaluator.existing_outcome(tmp_path, "libero-pro", "suite", 51)
    assert outcome is not None
    assert outcome["task_success"] is True
    assert outcome["_run_dir"] == str(new)
    assert evaluator.existing_outcome(tmp_path, "libero-pro", "suite", 52) is None


def stamp_run_identity(
    run_dir: Path,
    code: str,
    *,
    init_mode="seeded",
    max_steps=1000,
    camera_width=800,
    camera_height=512,
) -> None:
    source = run_dir / "source"
    source.mkdir()
    (source / "program.py").write_text(code)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "init_mode": init_mode,
                "capture": {
                    "max_steps": max_steps,
                    "camera_width": camera_width,
                    "camera_height": camera_height,
                },
            }
        )
    )


def test_existing_outcome_rejects_stale_code_and_settings(tmp_path: Path):
    evaluator = load_script("evosearch_eval")
    code_path = tmp_path / "code.py"
    code_path.write_text("result = {}\n")
    run_dir = write_outcome(
        tmp_path, "libero-pro", "suite", 51, "run-a", program_ok=True, task_success=True
    )
    stamp_run_identity(run_dir, "result = {}\n")

    identity = {
        "code_sha256": evaluator.sha256_file(code_path),
        "init_mode": "seeded",
        "max_steps": 1000,
        "camera_width": 800,
        "camera_height": 512,
    }
    assert evaluator.existing_outcome(tmp_path, "libero-pro", "suite", 51, identity) is not None

    # Different init mode → no match.
    assert (
        evaluator.existing_outcome(
            tmp_path, "libero-pro", "suite", 51, {**identity, "init_mode": "saved"}
        )
        is None
    )
    # Edited code → no match.
    code_path.write_text("observation = get_observation()\nresult = {}\n")
    stale = {**identity, "code_sha256": evaluator.sha256_file(code_path)}
    assert evaluator.existing_outcome(tmp_path, "libero-pro", "suite", 51, stale) is None
    # Runs recorded before run.json carried init_mode never match.
    legacy = write_outcome(
        tmp_path, "libero-pro", "suite", 52, "run-a", program_ok=True, task_success=True
    )
    (legacy / "source").mkdir()
    (legacy / "source" / "program.py").write_text("result = {}\n")
    (legacy / "run.json").write_text(json.dumps({"capture": {}}))
    assert evaluator.existing_outcome(tmp_path, "libero-pro", "suite", 52, identity) is None


def test_analyzer_extracts_prompts_from_args_kwargs_shape(tmp_path: Path):
    analyzer = load_script("analyze_evosearch_traces")
    run_dir = write_outcome(
        tmp_path,
        "libero-pro",
        "suite",
        51,
        "run-a",
        program_ok=True,
        task_success=False,
        termination_reason="program_completed",
    )
    call_dir = run_dir / "trace" / "calls" / "call-000003-segment-text"
    call_dir.mkdir(parents=True)
    (call_dir / "input.json").write_text(
        json.dumps({"args": ["agentview", "blue mug"], "kwargs": {}})
    )
    (call_dir / "output.json").write_text(
        json.dumps({"$type": "SegmentationSet", "ok": True, "segmentations": []})
    )
    trial = {
        "seed": "0051",
        "run_dir": run_dir,
        "outcome": json.loads((run_dir / "outcome.json").read_text()),
        "success": False,
    }
    signals = analyzer.extract_signals(trial)
    assert signals["prompts"] == ["blue mug"]
    assert signals["empty_prompts"] == ["blue mug"]


def test_trial_record_tracks_success_and_errors_separately(tmp_path: Path):
    evaluator = load_script("evosearch_eval")
    missing = evaluator.trial_record(51, None, 1)
    assert missing["error"] and not missing["task_success"]
    assert missing["termination_reason"] == "missing_artifact"

    # Crashed-but-completed still records task_success (a pass, like the
    # original pipeline) while flagging the crash in the errors count.
    crashed = evaluator.trial_record(
        51, {"program_ok": False, "task_success": True, "termination_reason": "program_error"}, 0
    )
    assert crashed["error"] is True and crashed["task_success"] is True

    passed = evaluator.trial_record(
        51, {"program_ok": True, "task_success": True, "termination_reason": "program_completed"}, 0
    )
    assert not passed["error"] and passed["task_success"]


def test_analyzer_loads_newest_trial_per_seed(tmp_path: Path):
    analyzer = load_script("analyze_evosearch_traces")
    eval_root = tmp_path / "eval"
    old = write_outcome(
        eval_root, "libero-pro", "suite", 51, "run-a", program_ok=True, task_success=False
    )
    write_outcome(eval_root, "libero-pro", "suite", 51, "run-b", program_ok=True, task_success=True)
    write_outcome(
        eval_root, "libero-pro", "suite", 52, "run-a", program_ok=False, task_success=False
    )
    import os

    os.utime(old / "outcome.json", (1, 1))

    trials = analyzer.load_trials(eval_root)
    assert [trial["seed"] for trial in trials] == ["0051", "0052"]
    assert trials[0]["success"] is True
    assert trials[1]["success"] is False


def test_analyzer_counts_failed_spans(tmp_path: Path):
    analyzer = load_script("analyze_evosearch_traces")
    run_dir = write_outcome(
        tmp_path,
        "libero-pro",
        "suite",
        51,
        "run-a",
        program_ok=True,
        task_success=False,
        termination_reason="program_completed",
    )
    trace = run_dir / "trace"
    trace.mkdir()
    events = [
        {"event": "span_end", "name": "solve_ik", "ok": False},
        {"event": "span_end", "name": "move_to_pose", "ok": True},
        {"event": "span_start", "name": "ignored"},
    ]
    (trace / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    trial = {
        "seed": "0051",
        "run_dir": run_dir,
        "outcome": json.loads((run_dir / "outcome.json").read_text()),
        "success": False,
    }
    signals = analyzer.extract_signals(trial)
    assert signals["failed_spans"]["solve_ik"] == 1
    assert signals["motion_calls"] == 2
