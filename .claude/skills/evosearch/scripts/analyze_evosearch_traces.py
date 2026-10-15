#!/usr/bin/env python3
"""Post-eval analysis for an Evolutionary Search iteration directory.

Reports outcome, span-failure, segmentation, and motion signals across success
vs failure trials for each candidate, from the recorded cap-harness artifacts:
``outcome.json``, ``trace/events.jsonl`` (span_end records), and
``trace/calls/*/{input,output}.json``. Task-agnostic: signals are derived from
span names present in the trace rather than assuming a fixed call structure.

Usage:
    python3 analyze_evosearch_traces.py --iter-dir outputs/evosearch/.../iter_01
    python3 analyze_evosearch_traces.py --iter-dir ... --candidate candidate_F
    python3 analyze_evosearch_traces.py --iter-dir ... --summary-only
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

MOTION_SPANS = {
    "move_to_pose",
    "move_to_joints",
    "plan_motion",
    "execute_trajectory",
    "solve_ik",
    "step",
    "go_home",
    "move_synchronized",
    "plan_synchronized_motion",
}
GRIPPER_SPANS = {"open_gripper", "close_gripper", "set_gripper"}
SEGMENT_SPANS = {"segment_text", "segment_points"}


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_trials(eval_root: Path) -> list[dict]:
    """Load the newest finalized run per seed from a candidate's eval root."""
    by_seed: dict[str, tuple[float, Path, dict]] = {}
    for outcome_path in eval_root.glob("*/*/*/*/*/outcome.json"):
        outcome = load_json(outcome_path)
        if outcome is None:
            continue
        seed = outcome_path.parent.parent.name
        stamp = outcome_path.stat().st_mtime
        if seed not in by_seed or stamp > by_seed[seed][0]:
            by_seed[seed] = (stamp, outcome_path.parent, outcome)
    trials = []
    for seed in sorted(by_seed):
        _, run_dir, outcome = by_seed[seed]
        trials.append(
            {
                "seed": seed,
                "run_dir": run_dir,
                "outcome": outcome,
                "success": bool(outcome.get("task_success")),
            }
        )
    return trials


def load_spans(run_dir: Path) -> list[dict]:
    events_path = run_dir / "trace" / "events.jsonl"
    spans = []
    if not events_path.is_file():
        return spans
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "span_end" or event.get("type") == "span_end":
            spans.append(event)
    return spans


def span_call_dirs(run_dir: Path, name: str) -> list[Path]:
    # Call dirs are named call-NNNNNN-<slug(name)>; slug maps '_' to '-'.
    return sorted((run_dir / "trace" / "calls").glob(f"call-*-{name.replace('_', '-')}*"))


def extract_signals(trial: dict) -> dict:
    run_dir = trial["run_dir"]
    outcome = trial["outcome"]
    spans = load_spans(run_dir)
    sig = {
        "seed": trial["seed"],
        "success": trial["success"],
        "termination_reason": outcome.get("termination_reason"),
        "steps_executed": outcome.get("steps_executed"),
        "wall_duration_s": outcome.get("wall_duration_s"),
        "error": outcome.get("error"),
        "failed_spans": Counter(),
        "motion_calls": 0,
        "gripper_calls": 0,
        "segment_calls": 0,
        "empty_prompts": [],
        "prompts": [],
    }
    for span in spans:
        name = str(span.get("name", ""))
        base = name.rsplit(".", maxsplit=1)[-1]
        if base in MOTION_SPANS:
            sig["motion_calls"] += 1
        if base in GRIPPER_SPANS:
            sig["gripper_calls"] += 1
        if base in SEGMENT_SPANS:
            sig["segment_calls"] += 1
        if span.get("ok") is False:
            sig["failed_spans"][base] += 1
    # Prompt-level segmentation detail from call payloads. input.json is
    # serialized as {"args": [...], "kwargs": {...}} (RunRecorder.span inputs);
    # segment_text(camera_name, text) puts the prompt in kwargs["text"] or as
    # the second positional argument.
    for call_dir in span_call_dirs(run_dir, "segment_text"):
        call_input = load_json(call_dir / "input.json") or {}
        call_output = load_json(call_dir / "output.json") or {}
        prompt = None
        if isinstance(call_input, dict):
            kwargs = call_input.get("kwargs")
            if isinstance(kwargs, dict):
                prompt = kwargs.get("text") or kwargs.get("text_prompt")
            if prompt is None:
                args_list = call_input.get("args")
                if isinstance(args_list, list):
                    strings = [value for value in args_list if isinstance(value, str)]
                    # args = [camera_name, text]; the prompt is the second string.
                    if len(strings) >= 2:
                        prompt = strings[1]
        if not isinstance(prompt, str):
            continue
        sig["prompts"].append(prompt)
        segmentations = None
        if isinstance(call_output, dict):
            segmentations = call_output.get("segmentations")
        if isinstance(segmentations, list) and not segmentations:
            sig["empty_prompts"].append(prompt)
        if call_output.get("ok") is False:
            sig["empty_prompts"].append(prompt)
    return sig


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(signals: list[dict], label: str) -> None:
    print(f"\n  {label} ({len(signals)} trials):")
    if not signals:
        print("    (none)")
        return
    terminations = Counter(sig["termination_reason"] for sig in signals)
    print(f"    termination_reason: {dict(terminations)}")
    steps = [sig["steps_executed"] for sig in signals if sig["steps_executed"] is not None]
    if steps:
        print(f"    steps_executed:  mean={mean(steps):.0f}  min={min(steps)}  max={max(steps)}")
    walls = [sig["wall_duration_s"] for sig in signals if sig["wall_duration_s"] is not None]
    if walls:
        # 30%+ faster than the group norm often means the arm hit an obstacle early.
        print(
            f"    wall_duration_s: mean={mean(walls):.0f}  min={min(walls):.0f}  max={max(walls):.0f}"
        )
    motion = [sig["motion_calls"] for sig in signals]
    print(f"    motion spans/trial: mean={mean(motion):.1f}  max={max(motion)}")
    failed = Counter()
    for sig in signals:
        failed.update(sig["failed_spans"])
    if failed:
        print(f"    *** FAILED SPANS: {dict(failed)} ***")
    empty = Counter(prompt for sig in signals for prompt in sig["empty_prompts"])
    if empty:
        print(f"    segment_text empty/failed prompts: {dict(empty)}")
    prompts = sorted({prompt for sig in signals for prompt in sig["prompts"]})
    if prompts:
        print(f"    prompts seen: {prompts}")
    errors = [sig for sig in signals if sig["error"]]
    if errors:
        print(f"    program errors: {len(errors)} (seeds {[sig['seed'] for sig in errors]})")


def analyze_candidate(candidate_dir: Path, verbose: bool = True) -> dict | None:
    eval_root = candidate_dir / "eval"
    if not eval_root.is_dir():
        print(f"  No eval/ directory in {candidate_dir}")
        return None
    trials = load_trials(eval_root)
    if not trials:
        print(f"  No finalized runs found in {eval_root}")
        return None
    signals = [extract_signals(trial) for trial in trials]
    successes = [sig for sig in signals if sig["success"]]
    failures = [sig for sig in signals if not sig["success"]]
    pass_rate = len(successes) / len(signals)
    print(f"\n{'=' * 60}")
    print(f"Candidate: {candidate_dir.name}  —  {pass_rate:.1%} ({len(successes)}/{len(signals)})")
    if verbose:
        summarize_group(successes, "SUCCESS")
        summarize_group(failures, "FAILURE")
    return {"candidate": candidate_dir.name, "pass_rate": pass_rate, "trials": len(signals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter-dir", required=True, type=Path)
    parser.add_argument("--candidate", help="analyze a single candidate")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    iter_dir = args.iter_dir.resolve()
    summary = (
        json.loads((iter_dir / "iter_summary.json").read_text())
        if (iter_dir / "iter_summary.json").is_file()
        else None
    )

    if args.summary_only:
        if summary is None:
            raise SystemExit(f"no iter_summary.json in {iter_dir} — run evosearch_eval.py first")
        print(f"\nLeaderboard ({iter_dir}):")
        for candidate in sorted(summary["candidates"], key=lambda c: -c["pass_rate"]):
            print(
                f"  {candidate['candidate']:<22} {candidate['pass_rate']:.1%}  "
                f"errors={candidate['errors']}"
            )
        print(f"  Winner: {summary['best_candidate']} ({summary['best_pass_rate']:.1%})")
        return

    if args.candidate:
        analyze_candidate(iter_dir / args.candidate)
        return
    for candidate_dir in sorted(iter_dir.glob("candidate_*")):
        if candidate_dir.is_dir():
            analyze_candidate(candidate_dir)


if __name__ == "__main__":
    main()
