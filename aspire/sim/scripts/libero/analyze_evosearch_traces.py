# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
analyze_evosearch_traces.py

Post-eval analysis for a Evolutionary Search iteration directory.
Reports gripper, IK, segmentation, and motion signals across success vs failure
trials for each candidate. Task-agnostic: signals are derived from function names
present in the trace rather than assuming a fixed call-index structure.

Usage:
    python3 scripts/libero/analyze_evosearch_traces.py --iter-dir outputs/claude_evosearch/.../iter_01
    python3 scripts/libero/analyze_evosearch_traces.py --iter-dir ... --candidate candidate_F
    python3 scripts/libero/analyze_evosearch_traces.py --iter-dir ... --summary-only
"""
import argparse
import json
from pathlib import Path
import numpy as np


def load_trials(eval_dir: Path):
    """Load all trials from a candidate's eval/ directory."""
    trials = []
    for trial_dir in sorted(eval_dir.iterdir()):
        trace_path = trial_dir / "trace.json"
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text())
        success = "reward_1.000" in trial_dir.name
        trials.append({"name": trial_dir.name, "trace": trace, "success": success})
    return trials


def extract_signals(trial):
    """Extract task-agnostic signals from a trial's trace."""
    trace = trial["trace"]
    sig = {"success": trial["success"], "name": trial["name"]}

    # --- Gripper signals ---
    close_steps = [s for s in trace if s.get("function") == "close_gripper"]
    open_steps  = [s for s in trace if s.get("function") == "open_gripper"]

    # All close_gripper widths (not just first — catches retries)
    sig["close_widths"] = [
        s["result"].get("gripper_width")
        for s in close_steps
        if isinstance(s.get("result"), dict)
    ]
    sig["grasp_width"] = sig["close_widths"][0] if sig["close_widths"] else None

    # Width at the last open_gripper (proxy for "was something held at release?")
    sig["release_width"] = (
        open_steps[-1]["result"].get("gripper_width")
        if open_steps and isinstance(open_steps[-1].get("result"), dict)
        else None
    )
    sig["n_close"] = len(close_steps)
    sig["n_open"]  = len(open_steps)

    # --- IK / motion signals ---
    solve_iks = [s for s in trace if s.get("function") == "solve_ik"]
    move_steps = [s for s in trace if s.get("function") == "move_to_joints"]

    # All IK target positions (not indexed — avoids assuming call order)
    sig["ik_positions"] = [
        s["args"]["position"]
        for s in solve_iks
        if isinstance(s.get("args"), dict) and "position" in s["args"]
    ]
    sig["n_ik"]   = len(solve_iks)
    sig["n_moves"] = len(move_steps)

    # IK failures: solve_ik called but returned None
    sig["n_ik_failures"] = sum(
        1 for s in solve_iks
        if s.get("result") is None or s.get("result", {}) == {}
    )

    # --- Segmentation signals ---
    sam3_steps = [s for s in trace if s.get("function") == "segment_sam3_text_prompt"]
    sig["n_sam3_calls"] = len(sam3_steps)
    sig["n_sam3_empty"] = sum(
        1 for s in sam3_steps
        if isinstance(s.get("result"), dict) and s["result"].get("num_masks", 1) == 0
    )
    sig["sam3_prompts"] = list({
        s["args"].get("text", s["args"].get("prompt", "?"))
        for s in sam3_steps
        if isinstance(s.get("args"), dict)
    })

    # --- Robot state displacement ---
    # Use robot_cartesian_pos from get_observation results to measure total movement
    obs_steps = [s for s in trace if s.get("function") == "get_observation"]
    positions = [
        s["result"].get("robot_cartesian_pos")
        for s in obs_steps
        if isinstance(s.get("result"), dict) and "robot_cartesian_pos" in s["result"]
    ]
    if len(positions) >= 2:
        first, last = np.array(positions[0]), np.array(positions[-1])
        sig["eef_displacement"] = float(np.linalg.norm(last[:3] - first[:3]))
    else:
        sig["eef_displacement"] = None

    # --- Blocking detection ---
    # For each get_observation, compare the last solve_ik target before it
    # against the actual robot_cartesian_pos. Discrepancy >3cm = arm was blocked.
    blocking_events = []
    last_ik_target = None
    for step in trace:
        fn = step.get("function")
        if fn == "solve_ik":
            args = step.get("args", {})
            result = step.get("result")
            # Only track IK calls that succeeded (non-None result)
            if isinstance(args, dict) and result is not None and "position" in args:
                last_ik_target = args["position"]
        elif fn == "get_observation" and last_ik_target is not None:
            result = step.get("result", {})
            if isinstance(result, dict):
                actual = result.get("robot_cartesian_pos")
                if actual is not None:
                    target = np.array(last_ik_target[:3])
                    actual_xyz = np.array(actual[:3])
                    error = float(np.linalg.norm(target - actual_xyz))
                    if error > 0.03:
                        blocking_events.append({
                            "ik_target": [round(v, 4) for v in last_ik_target[:3]],
                            "actual_pos": [round(v, 4) for v in actual[:3]],
                            "error_m": round(error, 4),
                        })
                    last_ik_target = None  # reset: paired with this observation
    sig["blocking_events"] = blocking_events
    sig["n_blocked"] = len(blocking_events)

    # --- Error / metadata ---
    sig["code_error"] = "sandboxrc_1" in trial["name"]
    sig["n_steps"]    = len(trace)

    return sig


def summarize_group(signals, label):
    print(f"\n  {label} ({len(signals)} trials):")
    if not signals:
        print("    (none)")
        return

    # Gripper
    gw = [s["grasp_width"] for s in signals if s["grasp_width"] is not None]
    if gw:
        print(f"    close_gripper width:  mean={np.mean(gw):.4f}  "
              f"min={np.min(gw):.4f}  max={np.max(gw):.4f}")

    rw = [s["release_width"] for s in signals if s["release_width"] is not None]
    if rw:
        print(f"    last open_gripper width:  mean={np.mean(rw):.4f}  "
              f"min={np.min(rw):.4f}  max={np.max(rw):.4f}")

    multi_close = sum(1 for s in signals if s["n_close"] > 1)
    if multi_close:
        print(f"    trials with >1 close_gripper: {multi_close}/{len(signals)}")

    # IK
    ik_fails = [s["n_ik_failures"] for s in signals]
    if any(f > 0 for f in ik_fails):
        total = sum(ik_fails)
        print(f"    IK failures: {total} across {sum(1 for f in ik_fails if f > 0)} trials")

    # Segmentation
    empty = sum(s["n_sam3_empty"] for s in signals)
    if empty:
        print(f"    SAM3 empty returns: {empty}")

    prompts = set(p for s in signals for p in s["sam3_prompts"])
    if prompts:
        print(f"    SAM3 prompts seen: {sorted(prompts)}")

    # Motion
    disps = [s["eef_displacement"] for s in signals if s["eef_displacement"] is not None]
    if disps:
        print(f"    EEF displacement (first→last obs):  "
              f"mean={np.mean(disps):.3f}  min={np.min(disps):.3f}  max={np.max(disps):.3f}")

    steps = [s["n_steps"] for s in signals]
    print(f"    trace steps:  mean={np.mean(steps):.0f}  "
          f"min={np.min(steps)}  max={np.max(steps)}")

    # Blocking detection
    blocked_trials = [s for s in signals if s["n_blocked"] > 0]
    if blocked_trials:
        total_blocks = sum(s["n_blocked"] for s in blocked_trials)
        print(f"    *** ARM BLOCKED: {total_blocks} events across {len(blocked_trials)}/{len(signals)} trials ***")
        # Show representative blocked positions (most common target)
        all_events = [e for s in blocked_trials for e in s["blocking_events"]]
        for ev in all_events[:3]:
            print(f"        target={ev['ik_target']}  actual={ev['actual_pos']}  "
                  f"error={ev['error_m']:.3f}m")
        if len(all_events) > 3:
            print(f"        ... ({len(all_events) - 3} more blocking events)")

    errors = sum(1 for s in signals if s["code_error"])
    if errors:
        print(f"    code_errors: {errors}")


def analyze_candidate(candidate_dir: Path, verbose: bool = True):
    eval_dir = candidate_dir / "eval"
    if not eval_dir.exists():
        print(f"  No eval/ directory in {candidate_dir}")
        return None

    trials = load_trials(eval_dir)
    if not trials:
        print(f"  No trials found in {eval_dir}")
        return None

    signals = [extract_signals(t) for t in trials]
    successes = [s for s in signals if s["success"]]
    failures  = [s for s in signals if not s["success"]]
    pass_rate = len(successes) / len(signals)

    print(f"\n{'='*60}")
    print(f"Candidate: {candidate_dir.name}  —  {pass_rate:.1%} ({len(successes)}/{len(signals)})")

    if verbose:
        summarize_group(successes, "SUCCESS")
        summarize_group(failures,  "FAILURE")

    return {"pass_rate": pass_rate, "n_trials": len(signals), "signals": signals}


def main():
    parser = argparse.ArgumentParser(description="Analyze Evolutionary Search eval traces")
    parser.add_argument("--iter-dir",     required=True,        help="Path to iter_NN directory")
    parser.add_argument("--candidate",    default=None,          help="Analyze one specific candidate")
    parser.add_argument("--summary-only", action="store_true",   help="Skip per-signal breakdown, leaderboard only")
    args = parser.parse_args()

    iter_dir = Path(args.iter_dir)
    assert iter_dir.exists(), f"Not found: {iter_dir}"

    # Leaderboard from iter_summary.json
    summary_path = iter_dir / "iter_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(f"\nLeaderboard ({iter_dir}):")
        for c in summary["candidates"]:
            print(f"  {c['candidate']:<20} {c['pass_rate']:.1%}  errors={c['errors']}")
        print(f"  Winner: {summary['best_candidate']} ({summary['best_pass_rate']:.1%})")

    if args.summary_only:
        return

    # Per-candidate signal breakdown
    if args.candidate:
        analyze_candidate(iter_dir / args.candidate)
    else:
        for cand_dir in sorted(iter_dir.iterdir()):
            if cand_dir.is_dir() and cand_dir.name.startswith("candidate_"):
                analyze_candidate(cand_dir)


if __name__ == "__main__":
    main()
