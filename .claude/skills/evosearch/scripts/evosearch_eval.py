#!/usr/bin/env python3
"""Evaluate every Evolutionary Search candidate in an iteration directory.

Runs the full candidate x seed matrix through ``cap-harness run`` (one fresh
process per episode), scheduling episodes over a pool of GPU slots
(``--sim-gpus`` x ``--parallel-per-gpu``). A seed is skipped on resume only
when an existing finalized run matches the current identity — same program
bytes (``source/program.py`` hash) and same ``init_mode``/``max_steps``/camera
settings in ``run.json`` — so editing a candidate or changing settings always
re-runs its seeds.

Layout (input):
    <iter-dir>/candidate_A/code.py     (or bare <iter-dir>/candidate_A.py,
                                        normalized into candidate_A/code.py)

Layout (output):
    <iter-dir>/candidate_X/eval/       cap-harness --output-root (immutable runs)
    <iter-dir>/candidate_X/logs/       one log per seed
    <iter-dir>/candidate_X/eval_results.json
    <iter-dir>/iter_summary.json       leaderboard across all candidates

A trial passes on ``task_success`` alone (matching the original pipeline,
where a crashed-but-completed trial still counted); crashed programs are
tracked separately in the ``errors`` count. There is no highlight re-run
stage: cap-harness already records videos, keyframes, and overlays for every
episode.
"""

# This is a standalone CLI script; progress/status is reported to stdout by design.

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading

DEFAULT_SEEDS = list(range(51, 66))
CANDIDATE_RE = re.compile(r"^candidate_[A-Za-z0-9]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter-dir", required=True, type=Path)
    parser.add_argument("--benchmark", default="libero-pro")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--trial-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--sim-gpus", type=int, nargs="+", required=True)
    parser.add_argument("--parallel-per-gpu", type=int, default=2)
    parser.add_argument("--candidates", nargs="*", help="subset of candidate names to evaluate")
    parser.add_argument("--init-mode", default="seeded", choices=("saved", "seeded"))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument(
        "--cap-harness",
        default="cap-harness",
        help="cap-harness executable (e.g. .venv-libero/bin/cap-harness)",
    )
    return parser.parse_args()


def discover_candidates(iter_dir: Path, subset: list[str] | None) -> dict[str, Path]:
    """Map candidate name -> code path, normalizing bare candidate_X.py files."""
    candidates: dict[str, Path] = {}
    for path in sorted(iter_dir.glob("candidate_*.py")):
        name = path.stem
        if not CANDIDATE_RE.match(name):
            continue
        directory = iter_dir / name
        directory.mkdir(exist_ok=True)
        target = directory / "code.py"
        source_text = path.read_text()
        # Refresh on content change so edits to the bare file are never
        # silently ignored in favor of a stale code.py.
        if not target.exists() or target.read_text() != source_text:
            target.write_text(source_text)
    for directory in sorted(iter_dir.iterdir()):
        if directory.is_dir() and CANDIDATE_RE.match(directory.name):
            code = directory / "code.py"
            if code.is_file():
                candidates[directory.name] = code
    if subset:
        missing = sorted(set(subset) - set(candidates))
        if missing:
            raise SystemExit(f"unknown candidates: {missing}")
        candidates = {name: candidates[name] for name in subset}
    if not candidates:
        raise SystemExit(f"no candidate_*/code.py found in {iter_dir}")
    return candidates


def run_matches_identity(run_dir: Path, identity: dict) -> bool:
    """True when a recorded run was produced by the same program and settings.

    Runs recorded before run.json carried ``init_mode`` never match, so stale
    pre-identity results are re-run rather than silently reused.
    """
    run_json = run_dir / "run.json"
    program = run_dir / "source" / "program.py"
    if not run_json.is_file() or not program.is_file():
        return False
    try:
        recorded = json.loads(run_json.read_text())
    except (OSError, ValueError):
        return False
    capture = recorded.get("capture", {})
    return (
        recorded.get("init_mode") == identity["init_mode"]
        and capture.get("max_steps") == identity["max_steps"]
        and capture.get("camera_width") == identity["camera_width"]
        and capture.get("camera_height") == identity["camera_height"]
        and sha256_file(program) == identity["code_sha256"]
    )


def existing_outcome(
    eval_root: Path, benchmark: str, suite: str, seed: int, identity: dict | None = None
) -> dict | None:
    """Return the newest finalized outcome for a seed matching the identity."""
    best: tuple[float, dict] | None = None
    for outcome_path in eval_root.glob(f"{benchmark}/{suite}/*/{seed:04d}/*/outcome.json"):
        if identity is not None and not run_matches_identity(outcome_path.parent, identity):
            continue
        try:
            outcome = json.loads(outcome_path.read_text())
        except (OSError, ValueError):
            continue
        stamp = outcome_path.stat().st_mtime
        if best is None or stamp > best[0]:
            outcome["_run_dir"] = str(outcome_path.parent)
            best = (stamp, outcome)
    return best[1] if best else None


def trial_record(seed: int, outcome: dict | None, exit_code: int | None) -> dict:
    if outcome is None:
        return {
            "seed": seed,
            "program_ok": False,
            "task_success": False,
            "error": True,
            "termination_reason": "missing_artifact",
            "process_exit_code": exit_code,
        }
    return {
        "seed": seed,
        "program_ok": bool(outcome.get("program_ok")),
        "task_success": bool(outcome.get("task_success")),
        "error": not bool(outcome.get("program_ok")),
        "termination_reason": outcome.get("termination_reason"),
        "cumulative_reward": outcome.get("cumulative_reward"),
        "steps_executed": outcome.get("steps_executed"),
        "run_dir": outcome.get("_run_dir"),
        "process_exit_code": exit_code,
    }


def main() -> int:
    args = parse_args()
    if args.init_mode != "saved" and args.benchmark != "libero-pro":
        raise SystemExit(
            "init_mode 'seeded' is only supported for the libero-pro benchmark; "
            "pass --init-mode saved for other benchmarks"
        )
    iter_dir = args.iter_dir.resolve()
    seeds = sorted(set(args.trial_seeds))
    candidates = discover_candidates(iter_dir, args.candidates)

    identities = {
        name: {
            "code_sha256": sha256_file(code),
            "init_mode": args.init_mode,
            "max_steps": args.max_steps,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
        }
        for name, code in candidates.items()
    }
    jobs: list[tuple[str, Path, int]] = []
    results: dict[str, dict[int, dict]] = {name: {} for name in candidates}
    for name, code in candidates.items():
        eval_root = iter_dir / name / "eval"
        for seed in seeds:
            outcome = existing_outcome(
                eval_root, args.benchmark, args.suite, seed, identities[name]
            )
            if outcome is not None:
                results[name][seed] = trial_record(seed, outcome, None)
                print(f"{name} seed {seed:02d}: SKIP matching existing run")
            else:
                jobs.append((name, code, seed))

    slots: queue.Queue[int] = queue.Queue()
    for gpu in args.sim_gpus:
        for _ in range(args.parallel_per_gpu):
            slots.put(gpu)

    lock = threading.Lock()

    def run_job(name: str, code: Path, seed: int) -> None:
        gpu = slots.get()
        try:
            candidate_dir = iter_dir / name
            logs = candidate_dir / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"seed_{seed:02d}.log"
            command = [
                args.cap_harness,
                "run",
                "--benchmark",
                args.benchmark,
                "--suite",
                args.suite,
                "--task-id",
                str(args.task_id),
                "--seed",
                str(seed),
                "--program",
                str(code),
                "--output-root",
                str(candidate_dir / "eval"),
                "--max-steps",
                str(args.max_steps),
                "--camera-width",
                str(args.camera_width),
                "--camera-height",
                str(args.camera_height),
                "--init-mode",
                args.init_mode,
            ]
            env = os.environ.copy()
            env.update(
                MUJOCO_GL="egl",
                PYOPENGL_PLATFORM="egl",
                EGL_PLATFORM="device",
                CUDA_VISIBLE_DEVICES=str(gpu),
                # EGL device enumeration ignores CUDA_VISIBLE_DEVICES, so without this MuJoCo
                # renders every episode on physical GPU 0 no matter which GPU was assigned.
                MUJOCO_EGL_DEVICE_ID=str(gpu),
                TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
            )
            with log_path.open("w") as log:
                process = subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            outcome = existing_outcome(
                candidate_dir / "eval", args.benchmark, args.suite, seed, identities[name]
            )
            record = trial_record(seed, outcome, process.returncode)
            with lock:
                results[name][seed] = record
                print(
                    f"{name} seed {seed:02d}: task_success={record['task_success']} "
                    f"program_ok={record['program_ok']} gpu={gpu}",
                    flush=True,
                )
        finally:
            slots.put(gpu)

    # Worker pool sized to the GPU slots; run_job pulls a GPU token per job.
    with ThreadPoolExecutor(max_workers=slots.qsize() or 1) as executor:
        for future in [executor.submit(run_job, *job) for job in jobs]:
            future.result()

    summary_candidates = []
    for name, code in candidates.items():
        rows = [results[name][seed] for seed in seeds if seed in results[name]]
        # A pass is task_success alone (matching the original pipeline); the
        # errors count tracks crashed programs separately.
        passes = sum(1 for row in rows if row["task_success"])
        errors = sum(1 for row in rows if row["error"])
        rewards = [
            row["cumulative_reward"]
            for row in rows
            if isinstance(row.get("cumulative_reward"), int | float)
        ]
        candidate_summary = {
            "candidate": name,
            "code_path": str(code),
            "trials": len(rows),
            "pass_count": passes,
            "pass_rate": round(passes / len(rows), 6) if rows else 0.0,
            "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else None,
            "errors": errors,
            "trial_results": rows,
        }
        summary_candidates.append(candidate_summary)
        (iter_dir / name / "eval_results.json").write_text(
            json.dumps(candidate_summary, indent=2) + "\n"
        )

    best = max(summary_candidates, key=lambda c: (c["pass_rate"], -c["errors"]))
    summary = {
        "benchmark": args.benchmark,
        "suite": args.suite,
        "task_id": args.task_id,
        "trial_seeds": seeds,
        "init_mode": args.init_mode,
        "candidates": summary_candidates,
        "best_candidate": best["candidate"],
        "best_pass_rate": best["pass_rate"],
    }
    (iter_dir / "iter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nLeaderboard ({iter_dir}):")
    for candidate in sorted(summary_candidates, key=lambda c: -c["pass_rate"]):
        print(
            f"  {candidate['candidate']:<22} {candidate['pass_rate']:.0%}  "
            f"errors={candidate['errors']}  passes={candidate['pass_count']}/{candidate['trials']}"
        )
    print(f"  Best: {summary['best_candidate']} ({summary['best_pass_rate']:.0%})")
    incomplete = any(
        len([seed for seed in seeds if seed in results[name]]) != len(seeds) for name in candidates
    )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
