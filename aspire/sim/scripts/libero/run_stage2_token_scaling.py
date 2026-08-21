#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Stage 2 token-scaling eval for a single (suite, task) pair.

Given a completed Stage 1, selects code versions at 0/25/50/75/100% of
cumulative output tokens, then runs each on seeds 1-50.

Usage:
    python3 scripts/libero/run_stage2_token_scaling.py \
        --worktree /path/to/snapshot-N90/worktree \
        --snapshot snapshot-N90 \
        --suite libero_10_swap \
        --task "TASK_NAME" \
        --gpu 3 \
        [--jsonl /path/to/agent.jsonl]   # optional: generate token_usage.json first
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def compute_token_checkpoints(stage1_dir: Path) -> dict[str, Path]:
    """Map percentile labels to code version files.

    Uses even index spacing across code versions:
      0pct   = first code version
      25pct  = version at 25% of the way through
      50pct  = version at 50%
      75pct  = version at 75%
      100pct = last code version

    For N versions, P% maps to index round(P/100 * (N-1)).
    """
    code_versions_dir = stage1_dir / "code_versions"
    code_files = sorted(code_versions_dir.glob("iter_*.py"))
    if not code_files:
        code_files = sorted(code_versions_dir.glob("v*.py"),
                            key=lambda f: int(f.stem.lstrip("v")) if f.stem.lstrip("v").isdigit() else 0)
    if not code_files:
        raise FileNotFoundError(f"No code versions in {code_versions_dir}")

    n = len(code_files)
    result = {}
    for pct, label in [(0, "0pct"), (25, "25pct"), (50, "50pct"), (75, "75pct"), (100, "100pct")]:
        if n == 1:
            idx = 0
        else:
            idx = int(pct / 100 * (n - 1) + 0.5)
        result[label] = code_files[idx]
    return result


def run_seeds(worktree: str, suite: str, task: str, snapshot: str,
              gpu: int, code_path: Path, outdir: Path, pct_label: str):
    """Run seeds 1-50 with a given code file."""
    outdir.mkdir(parents=True, exist_ok=True)
    config = f"{worktree}/env_configs/libero/franka_libero_libero10_traced.yaml"
    python = f"{worktree}/.venv-libero/bin/python3"
    replay = f"{worktree}/scripts/libero/replay_trial.py"

    n_pass = 0
    n_total = 0
    for seed in range(1, 51):
        trial_padded = f"{seed:02d}"
        existing = list(outdir.glob(f"trial_{trial_padded}_*"))
        if existing:
            is_pass = any("taskcompleted_1" in str(d) for d in existing)
            n_total += 1
            if is_pass:
                n_pass += 1
            continue

        env = os.environ.copy()
        env.update({
            "MUJOCO_GL": "egl",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "PYTHONPATH": str(Path(worktree).resolve().parents[1]),
        })

        logfile = outdir / f"seed{seed}.log"
        try:
            with open(logfile, "w") as lf:
                subprocess.run(
                    [python, replay,
                     "--args.suite", suite,
                     "--args.task", task,
                     "--args.trial", str(seed),
                     "--args.replay-code", str(code_path),
                     "--args.config", config,
                     "--args.output-dir", str(outdir)],
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                    timeout=120, cwd=worktree,
                )
        except (subprocess.TimeoutExpired, Exception) as e:
            with open(logfile, "a") as lf:
                lf.write(f"\nERROR: {e}\n")

        log_text = logfile.read_text() if logfile.exists() else ""
        if "taskcompleted_1" in log_text:
            n_pass += 1
        n_total += 1

    result = {
        "snapshot": snapshot,
        "suite": suite,
        "task": task,
        "pct_label": pct_label,
        "code": code_path.name,
        "seeds": "1-50",
        "n_pass": n_pass,
        "n_total": n_total,
        "pass_rate": n_pass / n_total if n_total > 0 else 0.0,
    }
    (outdir / "stage2_result.json").write_text(json.dumps(result, indent=2))
    print(f"  {pct_label}: {n_pass}/{n_total} = {result['pass_rate']*100:.0f}% (code={code_path.name})")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--jsonl", help="Agent JSONL for token analysis")
    parser.add_argument("--pcts", default="0,25,50,75,100",
                        help="Comma-separated percentiles to run (default: 0,25,50,75,100)")
    args = parser.parse_args()

    worktree = args.worktree
    stage1_dir = Path(f"{worktree}/outputs/scaling_eval/{args.snapshot}/debug_eval/{args.suite}/{args.task}/stage1")

    if not stage1_dir.exists():
        print(f"ERROR: stage1 dir not found: {stage1_dir}")
        return 1

    if not (stage1_dir / "stage1_summary.json").exists():
        print(f"ERROR: stage1_summary.json not found in {stage1_dir}")
        return 1

    if args.jsonl and not (stage1_dir / "token_usage.json").exists():
        print(f"Generating token_usage.json from {args.jsonl}...")
        analyze_script = Path(worktree) / "scripts" / "analyze_stage1_tokens.py"
        if analyze_script.exists():
            subprocess.run(
                [f"{worktree}/.venv-libero/bin/python3", str(analyze_script),
                 "--jsonl", args.jsonl, "--stage1", str(stage1_dir)],
                check=False,
            )

    print(f"\n=== Stage 2 Token Scaling: {args.suite}/{args.task[:50]} ===")
    print(f"Snapshot: {args.snapshot}, GPU: {args.gpu}")

    checkpoints = compute_token_checkpoints(stage1_dir)
    print("\nToken checkpoints:")
    for label in ["0pct", "25pct", "50pct", "75pct", "100pct"]:
        print(f"  {label}: {checkpoints[label].name}")

    requested_pcts = [f"{p}pct" for p in args.pcts.split(",")]
    base_dir = stage1_dir.parent

    results = []
    for pct_label in requested_pcts:
        if pct_label not in checkpoints:
            print(f"SKIP: {pct_label} not in checkpoints")
            continue

        outdir = base_dir / f"stage2_{pct_label}"
        existing_result = outdir / "stage2_result.json"
        if existing_result.exists():
            r = json.loads(existing_result.read_text())
            print(f"  {pct_label}: ALREADY DONE ({r['n_pass']}/{r['n_total']} = {r['pass_rate']*100:.0f}%)")
            results.append(r)
            continue

        code_path = checkpoints[pct_label]
        r = run_seeds(worktree, args.suite, args.task, args.snapshot,
                      args.gpu, code_path, outdir, pct_label)
        results.append(r)

    print(f"\n=== Summary ===")
    for r in results:
        print(f"  {r['pct_label']}: {r['n_pass']}/{r['n_total']} = {r['pass_rate']*100:.0f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
