#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate all candidate programs in an Evolutionary Search iteration directory.

Uses a persistent multiprocessing worker pool: each worker initializes the
LIBERO env once, then handles many trials via env.reset() + env.step(code).

Per-trial cost: ~8-15s (vs ~60-90s with subprocess-per-trial approach).
Speedup: ~5-10x. All candidates' trials run concurrently in the same pool.

OUTPUT STRUCTURE CONVENTION
----------------------------
Create a timestamped run dir before writing candidates:

    RUN_DIR=outputs/claude_evosearch/<suite>/<task>/$(date +%Y%m%d_%H%M%S)
    mkdir -p $RUN_DIR

    # Write candidates to $RUN_DIR/iter_00/candidate_*/code.py, then:
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv-libero/bin/python3 scripts/libero/evosearch_eval.py \\
        --iter-dir $RUN_DIR/iter_00 \\
        --suite libero_goal_swap \\
        --task "put_the_bowl_on_the_stove" \\
        --trials 50 \\
        --sim-gpus 4 5 6 7 --parallel-per-gpu 2

SPEED NOTES
-----------
Pool warmup: ~30s per worker (env init). With 8 workers: ~4min one-time cost.
Per trial: ~8-15s (env.reset + code execution + perception API calls).
8 workers, 8 candidates x 50 trials = 400 tasks: ~400/8 x 12s = ~10min total.
Use --trials 10 for quick screening (~2min), 50 for full validation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLAY = REPO_ROOT / "scripts" / "libero" / "replay_trial.py"
DEFAULT_CONFIG = REPO_ROOT / "env_configs" / "libero" / "franka_libero_traced.yaml"


# ──────────────────────────────────────────────────────────────
# LIBERO config helpers
# ──────────────────────────────────────────────────────────────

def _find_task_id(suite_name: str, task_name: str) -> int:
    """Resolve task name → integer task_id within the suite."""
    from libero import benchmark as libero_benchmark
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Suite '{suite_name}' not found. Available: {list(benchmark_dict.keys())}")
    task_suite = benchmark_dict[suite_name]()
    for task_id in range(task_suite.n_tasks):
        if task_suite.get_task(task_id).name == task_name:
            return task_id
    available = [task_suite.get_task(i).name for i in range(task_suite.n_tasks)]
    raise ValueError(
        f"Task '{task_name}' not found in '{suite_name}'. Available:\n"
        + "\n".join(f"  - {t}" for t in available)
    )


def _build_config(base_config_path: Path, suite_name: str, task_id: int) -> dict:
    """Load YAML config and inject suite_name + task_id into the low_level env."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    is_privileged = config.get("env", {}).get("cfg", {}).get("privileged", False)
    config["env"]["cfg"]["low_level"] = {
        "_target_": "aspire.sim.cap.envs.simulators.libero.FrankaLiberoEnv",
        "suite_name": suite_name,
        "task_id": task_id,
        "privileged": is_privileged,
        "max_steps": 4000,
        "seed": None,
        "enable_render": False,
        "viser_debug": False,
    }
    return config


# ──────────────────────────────────────────────────────────────
# Persistent worker pool
# ──────────────────────────────────────────────────────────────

_WORKER_ENV = None  # Module-level so spawned workers can access it


def _worker_init(config_json: str, gpu_queue) -> None:
    """Run once per worker at pool startup: assign GPU, init env."""
    global _WORKER_ENV

    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"

    repo_root = str(Path(__file__).resolve().parents[4])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    config = json.loads(config_json)
    from aspire.sim.cap.envs.configs.instantiate import instantiate
    _WORKER_ENV = instantiate(config["env"])


def _worker_run_trial(packed: tuple) -> dict:
    """Run one trial. Called in a persistent worker process.

    Returns a result dict with candidate, trial, reward, task_completed, etc.
    """
    candidate_name, code_str, trial_seed, eval_dir_str = packed
    env = _WORKER_ENV
    t0 = time.time()

    # Reset trace logger before episode (prevents accumulation across trials)
    if hasattr(env, "_apis"):
        for api in env._apis.values():
            if hasattr(api, "get_trace_logger"):
                api.get_trace_logger().reset()

    # Reset env with deterministic seed
    try:
        env.reset(seed=trial_seed)
    except TypeError:
        env.reset()

    # Execute candidate code (with timeout to prevent hanging workers)
    reward = 0.0
    sandbox_rc = 1
    task_completed = False
    stdout = ""
    stderr = ""
    TRIAL_TIMEOUT_S = 180  # 3 min max per trial; normal trials take 35-60s

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Trial exceeded {TRIAL_TIMEOUT_S}s timeout")

    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT_S)
    try:
        _, reward, terminated, truncated, info = env.step(code_str)
        reward = float(reward)
        sandbox_rc = int(info.get("sandbox_rc", 1))
        task_completed = bool(info.get("task_completed", False))
        stdout = str(info.get("stdout", ""))[:4000]
        stderr = str(info.get("stderr", ""))[:4000]
    except TimeoutError as exc:
        stderr = f"[TIMEOUT] {exc}"
        sandbox_rc = 1
    except Exception as exc:
        stderr = f"[EXCEPTION] {exc}"
        sandbox_rc = 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    elapsed = time.time() - t0

    # Write outputs to eval dir
    eval_dir = Path(eval_dir_str)
    trial_dir = (
        eval_dir
        / f"trial_{trial_seed:02d}_sandboxrc_{sandbox_rc}_"
          f"reward_{reward:.3f}_taskcompleted_{int(task_completed)}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "code.py").write_text(code_str)

    # Save trace.json (no numpy keyframes for bulk eval speed)
    if hasattr(env, "_apis"):
        for api in env._apis.values():
            if hasattr(api, "get_trace_logger"):
                tl = api.get_trace_logger()
                tl._buffered_keyframes.clear()
                tl._buffered_arrays.clear()
                tl.save(str(trial_dir))

    # summary.txt (mirrors replay_trial.py format for compatibility)
    summary_lines = [
        "-" * 100,
        f"Trial {trial_seed} — {candidate_name}",
        "Mode: persistent_worker",
        "",
        "Environment response:",
        f"  Sandbox failed: {sandbox_rc}",
        f"  Stdout: {stdout}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {task_completed}",
    ]
    (trial_dir / "summary.txt").write_text("\n".join(summary_lines))

    return {
        "candidate": candidate_name,
        "trial": trial_seed,
        "reward": reward,
        "task_completed": int(task_completed),
        "sandbox_rc": sandbox_rc,
        "elapsed_s": round(elapsed, 1),
    }


# ──────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────

def discover_candidates(iter_dir: Path) -> dict[str, Path]:
    """Return {candidate_name: code_path} dict, sorted by name."""
    found = {}
    for p in sorted(iter_dir.iterdir()):
        if p.is_dir() and p.name.startswith("candidate_") and (p / "code.py").exists():
            found[p.name] = p / "code.py"
    if not found:
        sys.exit(f"No candidate_*/code.py found in {iter_dir}")
    return found


# ──────────────────────────────────────────────────────────────
# Highlight video re-runner (uses replay_trial.py subprocess)
# ──────────────────────────────────────────────────────────────

def _replay_trial_with_video(
    *,
    candidate: str,
    code_path: Path,
    trial: int,
    label: str,
    videos_dir: Path,
    log_path: Path,
    suite: str,
    task: str,
    config_path: Path,
    gpu: int,
) -> None:
    """Re-run a single trial via replay_trial.py, saving full output (video + keyframes + trace).

    Output lands in videos_dir/trial_{trial:02d}_{label}/.
    """
    out_dir = videos_dir / f"trial_{trial:02d}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python3"),
        str(REPLAY),
        "--args.suite", suite,
        "--args.task", task,
        "--args.trial", str(trial),
        "--args.replay-code", str(code_path),
        "--args.config", str(config_path),
        "--args.output-dir", str(out_dir),
        # record_video=True and keyframe saving are both on by default in replay_trial.py
    ]
    with open(log_path, "w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    video_files = list(out_dir.rglob("video_*.mp4"))
    if video_files:
        print(f"  videos: [{candidate}] trial_{trial:02d}_{label}/ saved ({len(list(out_dir.rglob('*')))} files)")
    else:
        print(f"  videos: [{candidate}] no video produced for {label} trial {trial}")


def save_highlight_videos(
    *,
    all_summaries: list[dict],
    iter_dir: Path,
    suite: str,
    task: str,
    config_path: Path,
    sim_gpus: list[int],
    all_candidates: bool = False,
) -> None:
    """Re-run one success and one failure trial with full replay output for each candidate.

    Full output (video + keyframes + trace.json + summary.txt) saved to
    candidate_X/videos/trial_{N:02d}_{success|failure}/.

    Args:
        all_candidates: If True, generate videos for every candidate. If False
                        (default), only generate videos for the best candidate.
    """
    candidates_to_highlight = all_summaries if all_candidates else all_summaries[:1]
    results = {}

    for idx, cand_summary in enumerate(candidates_to_highlight):
        cand_name = cand_summary["candidate"]
        code_path = iter_dir / cand_name / "code.py"
        videos_dir = iter_dir / cand_name / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        gpu = sim_gpus[idx % len(sim_gpus)]

        success_trial = next(
            (r["trial"] for r in cand_summary["trial_results"] if r["task_completed"]), None
        )
        failure_trial = next(
            (r["trial"] for r in cand_summary["trial_results"] if not r["task_completed"]), None
        )

        cand_results = {}
        for label, trial in [("success", success_trial), ("failure", failure_trial)]:
            if trial is None:
                print(f"  videos: [{cand_name}] no {label} trial found")
                continue
            log_path = videos_dir / f"{label}.log"
            print(f"  videos: [{cand_name}] re-running {label} trial {trial} with video ...")
            _replay_trial_with_video(
                candidate=cand_name,
                code_path=code_path,
                trial=trial,
                label=label,
                videos_dir=videos_dir,
                log_path=log_path,
                suite=suite,
                task=task,
                config_path=config_path,
                gpu=gpu,
            )
            cand_results[label] = {"trial": trial}
        results[cand_name] = cand_results

    (iter_dir / "highlights.json").write_text(json.dumps(results, indent=2))


# ──────────────────────────────────────────────────────────────
# Deep debug: full replay with keyframes for rare successes
# ──────────────────────────────────────────────────────────────

RARE_SUCCESS_THRESHOLD = 0.10  # trigger when best pass rate ≤ this


def deep_debug_rare_successes(
    *,
    all_summaries: list[dict],
    iter_dir: Path,
    suite: str,
    task: str,
    config_path: Path,
    sim_gpus: list[int],
    threshold: float = RARE_SUCCESS_THRESHOLD,
) -> None:
    """If best pass rate ≤ threshold, replay all success trials preserving full output.

    Unlike standard highlight saving (video only), this keeps the complete replay
    output directory — trace.json, keyframes/, and video — so the agent can inspect
    step-by-step observations and update task_analysis.md before writing the next
    iteration's candidates.

    Output: candidate_X/videos/trial_NN_deep_debug/
    """
    best_rate = all_summaries[0]["pass_rate"]
    if best_rate > threshold:
        return

    print(f"\n  Rare success ({best_rate:.1%} ≤ {threshold:.0%}) — "
          f"running deep debug replay for all success trials ...")

    found_any = False
    for idx, cand_summary in enumerate(all_summaries):
        success_trials = [
            r["trial"] for r in cand_summary["trial_results"] if r["task_completed"]
        ]
        if not success_trials:
            continue

        found_any = True
        cand_name = cand_summary["candidate"]
        code_path = iter_dir / cand_name / "code.py"
        videos_dir = iter_dir / cand_name / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        gpu = sim_gpus[idx % len(sim_gpus)]

        for trial in success_trials:
            out_dir = videos_dir / f"trial_{trial:02d}_deep_debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = videos_dir / f"trial_{trial:02d}_deep_debug.log"

            print(f"  deep-debug: [{cand_name}] replaying success trial {trial} "
                  f"(full trace + keyframes) ...")
            env_vars = {
                **os.environ,
                "MUJOCO_GL": "egl",
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            }
            cmd = [
                str(REPO_ROOT / ".venv" / "bin" / "python3"),
                str(REPLAY),
                "--args.suite", suite,
                "--args.task", task,
                "--args.trial", str(trial),
                "--args.replay-code", str(code_path),
                "--args.config", str(config_path),
                "--args.output-dir", str(out_dir),
                # record_video=True is default; trace_logger always saves keyframes/
            ]
            with open(log_path, "w") as f:
                subprocess.run(cmd, env=env_vars, stdout=f, stderr=subprocess.STDOUT)
            print(f"  deep-debug: [{cand_name}] trial {trial} → {out_dir}")

    if found_any:
        print(
            f"\n  *** Rare success — inspect keyframes + trace before writing next candidates:\n"
            f"  ***   candidate_X/videos/trial_NN_deep_debug/\n"
            f"  ***     keyframes/  — per-step RGB/depth/mask/grasp arrays\n"
            f"  ***     trace.json  — API call log with positions and gripper widths\n"
            f"  ***     video_*.mp4 — full rollout video\n"
            f"  *** Update task_analysis.md with observed geometry from these results."
        )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Evolutionary Search iteration candidates with persistent worker pool."
    )
    parser.add_argument("--iter-dir", required=True, type=Path,
                        help="Path to iter_NN dir containing candidate_* subdirs.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--trials", type=int, default=50,
                        help="Trials per candidate. 10 = quick screen, 50 = full validation.")
    parser.add_argument("--sim-gpus", type=int, nargs="+", default=[4, 5, 6, 7],
                        help="GPU IDs for simulation workers (avoid 0=SAM3, 1=GraspNet).")
    parser.add_argument("--parallel-per-gpu", type=int, default=2,
                        help="Worker processes per GPU.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidates", nargs="*",
                        help="Evaluate only these candidates. Default: all.")
    parser.add_argument("--no-highlights", action="store_true",
                        help="Skip highlight video re-runs after eval.")
    parser.add_argument("--best-only-highlights", action="store_true",
                        help="Only generate videos for the best candidate (default: all candidates).")
    parser.add_argument("--deep-debug-threshold", type=float, default=0.0,
                        help="If best pass rate ≤ this, automatically replay all success trials "
                             "with full keyframes+trace. Default: 0 (disabled — use "
                             "replay_trial.py directly to choose specific trials to debug).")
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=None,
                        help="Explicit seed list instead of random sampling. Pass seeds from "
                             "a prior iter's iter_summary.json for clean cross-iteration "
                             "comparison. Final validation should use fresh random seeds.")
    args = parser.parse_args()

    iter_dir = args.iter_dir.resolve()
    if not iter_dir.exists():
        sys.exit(f"iter-dir not found: {iter_dir}")

    # Resolve task_id from task name
    print(f"  Resolving task_id for '{args.task}' in '{args.suite}'...")
    task_id = _find_task_id(args.suite, args.task)
    print(f"  task_id = {task_id}")

    # Build env config (inject suite + task_id)
    config = _build_config(args.config, args.suite, task_id)
    config_json = json.dumps(config, default=str)

    # Discover candidates
    all_candidates = discover_candidates(iter_dir)
    if args.candidates:
        all_candidates = {k: v for k, v in all_candidates.items() if k in args.candidates}
        if not all_candidates:
            sys.exit(f"None of the specified candidates found in {iter_dir}")

    total_workers = len(args.sim_gpus) * args.parallel_per_gpu
    total_tasks = len(all_candidates) * args.trials
    est_min = total_tasks / total_workers * 12 / 60  # ~12s per trial estimate

    print(f"\nEvolutionary Search Evaluation (persistent worker pool)")
    print(f"  iter_dir   : {iter_dir}")
    print(f"  suite      : {args.suite}")
    print(f"  task       : {args.task}  (task_id={task_id})")
    print(f"  trials     : {args.trials}")
    print(f"  sim_gpus   : {args.sim_gpus}  ({args.parallel_per_gpu}/GPU)")
    print(f"  workers    : {total_workers}  (warmup ~{total_workers * 30 // 60}min)")
    print(f"  candidates : {list(all_candidates.keys())}")
    print(f"  total tasks: {total_tasks}  (est. ~{est_min:.0f}min after warmup)")
    print()

    # Read all code files upfront (workers get code as strings, not paths)
    codes = {name: path.read_text() for name, path in all_candidates.items()}

    # Prepare eval dirs
    eval_dirs = {}
    for name, path in all_candidates.items():
        ed = path.parent / "eval"
        ed.mkdir(exist_ok=True)
        eval_dirs[name] = ed

    # Seed pool: fixed if --trial-seeds provided, otherwise random.
    # Use fixed seeds across iterations for clean comparison; use fresh random
    # seeds for final validation of the winning candidate.
    if args.trial_seeds:
        trial_seeds = sorted(args.trial_seeds[:args.trials])
        if len(trial_seeds) < args.trials:
            print(f"  WARNING: only {len(trial_seeds)} seeds provided, requested {args.trials}")
    else:
        trial_seeds = random.sample(range(1, 10_001), args.trials)
        trial_seeds.sort()
    print(f"  trial seeds: {trial_seeds[:8]}{'...' if len(trial_seeds) > 8 else ''}")

    # Build flat task list: all candidates × all trials, submitted together
    all_tasks = [
        (name, codes[name], seed, str(eval_dirs[name]))
        for name in all_candidates
        for seed in trial_seeds
    ]

    # GPU round-robin queue: worker i pops sim_gpus[i % len]
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    gpu_queue = manager.Queue()
    for i in range(total_workers):
        gpu_queue.put(args.sim_gpus[i % len(args.sim_gpus)])

    # Run all trials in parallel
    per_candidate: dict[str, list[dict]] = {name: [] for name in all_candidates}
    done = 0

    print(f"  Starting {total_workers} workers (env init ~30s each) ...")
    with ctx.Pool(
        processes=total_workers,
        initializer=_worker_init,
        initargs=(config_json, gpu_queue),
    ) as pool:
        for r in pool.imap_unordered(_worker_run_trial, all_tasks):
            per_candidate[r["candidate"]].append(r)
            done += 1
            status = "✓" if r["task_completed"] else "✗"
            n_cand_done = len(per_candidate[r["candidate"]])
            print(
                f"    {status} {r['candidate']} trial {r['trial']:02d}  "
                f"reward={r['reward']:.3f}  {r['elapsed_s']:.0f}s  "
                f"[{done}/{total_tasks}]",
                flush=True,
            )

    # Aggregate results and write per-candidate eval_results.json
    n_trials_run = len(trial_seeds)  # actual trials run (may differ from args.trials default)
    all_summaries = []
    for name in all_candidates:
        trial_results = per_candidate[name]
        pass_count = sum(r["task_completed"] for r in trial_results)
        pass_rate = pass_count / n_trials_run
        errors = sum(r["sandbox_rc"] != 0 for r in trial_results)
        summary = {
            "candidate": name,
            "code_path": str(all_candidates[name]),
            "trials": n_trials_run,
            "pass_count": pass_count,
            "pass_rate": round(pass_rate, 4),
            "mean_reward": round(sum(r["reward"] for r in trial_results) / n_trials_run, 4),
            "errors": errors,
            "trial_results": sorted(trial_results, key=lambda r: r["trial"]),
        }
        (all_candidates[name].parent / "eval_results.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(
            f"  [{name}] → pass_rate={pass_rate:.1%} ({pass_count}/{n_trials_run})  "
            f"errors={errors}"
        )
        all_summaries.append(summary)

    # Write combined iter_summary.json (ranked leaderboard)
    all_summaries.sort(key=lambda s: s["pass_rate"], reverse=True)
    iter_summary = {
        "suite": args.suite,
        "task": args.task,
        "trials_per_candidate": n_trials_run,
        "trial_seeds": trial_seeds,
        "candidates": all_summaries,
        "best_candidate": all_summaries[0]["candidate"],
        "best_pass_rate": all_summaries[0]["pass_rate"],
    }
    summary_path = iter_dir / "iter_summary.json"
    summary_path.write_text(json.dumps(iter_summary, indent=2))

    # Print leaderboard
    print(f"\n{'─'*55}")
    print(f"  RESULTS: {args.suite} / {args.task}")
    print(f"{'─'*55}")
    for s in all_summaries:
        bar = "█" * int(s["pass_rate"] * 20)
        print(f"  {s['candidate']:<20} {s['pass_rate']:>5.1%}  {bar}")
    print(f"{'─'*55}")
    print(f"  WINNER: {all_summaries[0]['candidate']}  ({all_summaries[0]['pass_rate']:.1%})")
    print(f"  Full results: {summary_path}")

    # Save highlight videos (all candidates by default)
    if not args.no_highlights:
        all_cands = not args.best_only_highlights
        if all_cands:
            print(f"\n  Saving highlight videos for ALL {len(all_summaries)} candidates ...")
        else:
            print(f"\n  Saving highlight videos (best only: {all_summaries[0]['candidate']}) ...")
        save_highlight_videos(
            all_summaries=all_summaries,
            iter_dir=iter_dir,
            suite=args.suite,
            task=args.task,
            config_path=args.config,
            sim_gpus=args.sim_gpus,
            all_candidates=all_cands,
        )

    # Deep debug: full keyframes+trace for rare successes
    if args.deep_debug_threshold > 0:
        deep_debug_rare_successes(
            all_summaries=all_summaries,
            iter_dir=iter_dir,
            suite=args.suite,
            task=args.task,
            config_path=args.config,
            sim_gpus=args.sim_gpus,
            threshold=args.deep_debug_threshold,
        )

    print(f"\nDone. Read iter_summary.json then diagnose before writing iter_N+1.")


if __name__ == "__main__":
    main()
