#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parallel seed execution for LIBERO-Long-Pro eval.

Phase 2 of the split eval pipeline: given pre-written code.py files,
run all seeds in parallel across GPUs.

Usage:
    python3 scripts/libero/eval_run_seeds.py \
        --worktree outputs/worktrees/snapshot-N50/aspire/sim \
        --snapshot snapshot-N50 \
        --gpus 3 4 5 6 7 \
        --seeds-per-gpu 3 \
        [--seeds 50] \
        [--suites libero_10_swap libero_10_task]

Each GPU runs --seeds-per-gpu concurrent replay_trial.py processes.
With 5 GPUs × 3 concurrent = 15 simultaneous seeds → ~3.3x speedup.
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def find_code(task_dir: Path) -> str | None:
    """Find code.py in task output dir — check top-level first, then trial dirs."""
    top = task_dir / "code.py"
    if top.exists():
        return str(top)
    for f in task_dir.rglob("code.py"):
        return str(f)
    return None


def get_completed_seeds(task_dir: Path) -> set[int]:
    """Get set of seed numbers that already have trial directories."""
    seeds = set()
    for t in task_dir.rglob("trial_*"):
        if t.is_dir():
            m = re.search(r"trial_(\d+)", t.name)
            if m:
                seeds.add(int(m.group(1)))
    return seeds


def run_seed(gpu, suite, task, seed, code_path, worktree, config, outdir, python):
    """Run a single seed. Returns (suite, task, seed, success, error)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PYTHONPATH"] = str(worktree.parents[1])

    cmd = [
        python,
        str(worktree / "scripts" / "libero" / "replay_trial.py"),
        "--args.suite", suite,
        "--args.task", task,
        "--args.trial", str(seed),
        "--args.replay-code", code_path,
        "--args.config", config,
        "--args.output-dir", outdir,
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600
        )
        output = result.stdout + result.stderr
        if "taskcompleted_1" in output:
            return (suite, task, seed, True, None)
        else:
            return (suite, task, seed, False, None)
    except subprocess.TimeoutExpired:
        return (suite, task, seed, False, "timeout")
    except Exception as e:
        return (suite, task, seed, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Parallel eval seed runner")
    parser.add_argument("--worktree", required=True, help="Path to snapshot workspace root")
    parser.add_argument("--snapshot", required=True, help="Snapshot name")
    parser.add_argument("--gpus", nargs="+", type=int, default=[3, 4, 5, 6, 7])
    parser.add_argument("--seeds-per-gpu", type=int, default=3,
                        help="Concurrent seeds per GPU")
    parser.add_argument("--seeds", type=int, default=50, help="Seeds per task")
    parser.add_argument("--suites", nargs="+",
                        default=["libero_10_swap", "libero_10_task"])
    parser.add_argument("--python", default=None,
                        help="Python path (default: worktree/.venv-libero/bin/python3)")
    args = parser.parse_args()

    worktree = Path(args.worktree).resolve()
    outdir = str(worktree / "outputs" / "scaling_eval" / args.snapshot / "one_shot")
    config = str(worktree / "env_configs" / "libero" / "franka_libero_libero10_traced.yaml")
    python = args.python or str(worktree / ".venv-libero" / "bin" / "python3")

    if not Path(python).exists():
        repo = Path(__file__).resolve().parents[2]
        python = str(repo / ".venv-libero" / "bin" / "python3")

    # Build job list: (gpu, suite, task, seed, code_path)
    jobs = []
    gpu_idx = 0
    tasks_found = 0
    tasks_missing_code = []

    for suite in args.suites:
        suite_dir = Path(outdir) / suite
        if not suite_dir.exists():
            print(f"WARNING: {suite_dir} does not exist, skipping")
            continue
        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            code = find_code(task_dir)
            if not code:
                tasks_missing_code.append(f"{suite}/{task}")
                continue
            tasks_found += 1
            completed = get_completed_seeds(task_dir)
            gpu = args.gpus[gpu_idx % len(args.gpus)]
            gpu_idx += 1
            for seed in range(1, args.seeds + 1):
                if seed in completed:
                    continue
                jobs.append((gpu, suite, task, seed, code))

    if tasks_missing_code:
        print(f"WARNING: {len(tasks_missing_code)} tasks missing code.py:")
        for t in tasks_missing_code:
            print(f"  {t}")

    print(f"Tasks: {tasks_found}, Jobs: {len(jobs)} seeds to run")
    print(f"GPUs: {args.gpus}, Concurrency: {args.seeds_per_gpu} per GPU")
    print(f"Total parallelism: {len(args.gpus) * args.seeds_per_gpu}")
    print(f"Estimated time: ~{len(jobs) * 1.5 / (len(args.gpus) * args.seeds_per_gpu):.0f} min")
    print()

    if not jobs:
        print("Nothing to do — all seeds complete.")
        return

    # Group jobs by GPU for load balancing
    gpu_jobs = {g: [] for g in args.gpus}
    for job in jobs:
        gpu_jobs[job[0]].append(job)

    for g in args.gpus:
        print(f"  GPU {g}: {len(gpu_jobs[g])} seeds")

    # Run with per-GPU thread pools
    total_concurrency = len(args.gpus) * args.seeds_per_gpu
    success_count = 0
    fail_count = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=total_concurrency) as pool:
        futures = {}
        for gpu, suite, task, seed, code in jobs:
            f = pool.submit(run_seed, gpu, suite, task, seed, code,
                            worktree, config, outdir, python)
            futures[f] = (suite, task, seed)

        for f in as_completed(futures):
            suite, task, seed, ok, err = f.result()
            done_count += 1
            if ok:
                success_count += 1
            else:
                fail_count += 1
            if done_count % 25 == 0 or done_count == len(jobs):
                print(f"  [{done_count}/{len(jobs)}] "
                      f"success={success_count} fail={fail_count}")

    print(f"\nDone: {success_count} success, {fail_count} fail, {done_count} total")


if __name__ == "__main__":
    main()
