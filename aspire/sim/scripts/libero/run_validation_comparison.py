#!/usr/bin/env python3
"""
Validation comparison: fix_code vs evosearch_code on seeds 66-80.

For each task with a completed Evolutionary Search stage2 and a fix_code baseline,
runs evosearch_eval.py with both codes as candidates (candidate_fix, candidate_evosearch)
on seeds 66-80 (15 seeds). Outputs a comparison table.

Usage:
    .venv/bin/python3 scripts/libero/run_validation_comparison.py [--gpus 4,7] [--parallel-per-gpu 2]
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ASPIRE = Path(__file__).resolve().parents[2]
EVOSEARCH_DIR = ASPIRE / "outputs/claude_evosearch"
FIX_DIR = ASPIRE / "outputs/libero_baseline_image_diff_gemini"
VAL_DIR = ASPIRE / "outputs/validation_comparison_seeds66_80"
PYTHON = str(ASPIRE / ".venv-libero/bin/python3")
EVAL_SEEDS = list(range(66, 81))  # 15 seeds
SUITES = ["libero_goal_swap", "libero_goal_task", "libero_spatial_swap", "libero_spatial_task"]


def resolve_under_repo(path: str, repo_root: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def find_evosearch_code(suite: str, task: str) -> Path | None:
    """Find the best Evolutionary Search code: prefer stage2/candidate_A/code.py, fall back to evosearch_best_code.py."""
    task_dir = EVOSEARCH_DIR / suite / task
    if not task_dir.exists():
        return None

    # Find most recent run with stage2/candidate_A/code.py
    run_dirs = sorted(task_dir.glob("2*"), reverse=True)
    for run_dir in run_dirs:
        stage2_code = run_dir / "stage2" / "candidate_A" / "code.py"
        if stage2_code.exists():
            return stage2_code

    # Fall back to evosearch_best_code.py
    best = task_dir / "evosearch_best_code.py"
    if best.exists():
        return best

    return None


def find_fix_code(suite: str, task: str) -> Path | None:
    """Find the fix_code baseline."""
    fix = FIX_DIR / suite / task / "fix_code.py"
    return fix if fix.exists() else None


def find_stage2_summary(suite: str, task: str) -> dict | None:
    """Find stage2 iter_summary.json for the task."""
    task_dir = EVOSEARCH_DIR / suite / task
    if not task_dir.exists():
        return None
    for run_dir in sorted(task_dir.glob("2*"), reverse=True):
        summary = run_dir / "stage2" / "iter_summary.json"
        if summary.exists():
            with open(summary) as f:
                return json.load(f)
    return None


def discover_tasks() -> list[tuple[str, str]]:
    """Find all (suite, task) pairs with both evosearch code and fix_code."""
    tasks = []
    for suite in SUITES:
        suite_dir = EVOSEARCH_DIR / suite
        if not suite_dir.exists():
            continue
        for task_dir in sorted(suite_dir.iterdir()):
            task = task_dir.name
            evosearch_code = find_evosearch_code(suite, task)
            fix_code = find_fix_code(suite, task)
            if evosearch_code and fix_code:
                tasks.append((suite, task))
    return tasks


def setup_val_dir(suite: str, task: str) -> Path:
    """Create validation iter-dir with both candidates."""
    val_task_dir = VAL_DIR / suite / task
    val_task_dir.mkdir(parents=True, exist_ok=True)

    evosearch_code = find_evosearch_code(suite, task)
    fix_code = find_fix_code(suite, task)

    # candidate_evosearch
    evosearch_cand = val_task_dir / "candidate_evosearch"
    evosearch_cand.mkdir(exist_ok=True)
    shutil.copy2(evosearch_code, evosearch_cand / "code.py")

    # candidate_fix
    fix_cand = val_task_dir / "candidate_fix"
    fix_cand.mkdir(exist_ok=True)
    shutil.copy2(fix_code, fix_cand / "code.py")

    return val_task_dir


def extract_rates(summary: dict) -> tuple[float, float, float]:
    """Return (evosearch_rate, fix_rate, delta) from evosearch_eval.py summary data."""
    rates = {c["candidate"]: c["pass_rate"] for c in summary.get("candidates", [])}
    evosearch_rate = rates.get("candidate_evosearch", 0.0)
    fix_rate = rates.get("candidate_fix", 0.0)
    return evosearch_rate, fix_rate, evosearch_rate - fix_rate


def run_eval(suite: str, task: str, gpu: int, parallel_per_gpu: int = 2) -> dict:
    """Run validation eval for one task. Returns result dict."""
    # Check if already done
    summary_path = VAL_DIR / suite / task / "iter_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        evosearch_rate, fix_rate, delta = extract_rates(data)
        print(f"  [SKIP] {suite}/{task} — already done")
        return {
            "suite": suite,
            "task": task,
            "evosearch": evosearch_rate,
            "fix": fix_rate,
            "delta": delta,
            "summary": data,
            "skipped": True,
        }

    val_dir = setup_val_dir(suite, task)

    cmd = [
        PYTHON, "scripts/libero/evosearch_eval.py",
        "--iter-dir", str(val_dir),
        "--suite", suite,
        "--task", task,
        "--trials", str(len(EVAL_SEEDS)),
        "--trial-seeds", *[str(s) for s in EVAL_SEEDS],
        "--sim-gpus", str(gpu),
        "--parallel-per-gpu", str(parallel_per_gpu),
        "--no-highlights",
    ]

    log_path = Path(f"/tmp/val_{suite}_{task}_gpu{gpu}.log")
    print(f"  [START] GPU{gpu}: {suite}/{task} → {log_path.name}")

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    with open(log_path, "w") as logf:
        result = subprocess.run(
            cmd, cwd=str(ASPIRE), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
            timeout=3600  # 1 hour max
        )

    if result.returncode != 0:
        print(f"  [ERROR] {suite}/{task} (GPU{gpu}) — exit code {result.returncode}")
        return {"suite": suite, "task": task, "error": f"exit {result.returncode}"}

    if not summary_path.exists():
        print(f"  [ERROR] {suite}/{task} — no iter_summary.json written")
        return {"suite": suite, "task": task, "error": "no iter_summary.json"}

    with open(summary_path) as f:
        data = json.load(f)

    evosearch_rate, fix_rate, delta = extract_rates(data)
    sign = "+" if delta >= 0 else ""
    print(f"  [DONE] {suite}/{task}: evosearch={evosearch_rate:.0%} fix={fix_rate:.0%} delta={sign}{delta:.0%}")

    return {"suite": suite, "task": task, "evosearch": evosearch_rate, "fix": fix_rate, "delta": delta, "summary": data}


def print_table(results: list[dict]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Suite':<22} {'Task':<45} {'Fix':>5} {'Evolutionary Search':>7} {'Delta':>7}")
    print("-" * 90)

    suite_totals = {}
    for r in sorted(results, key=lambda x: (x["suite"], x["task"])):
        suite = r["suite"]
        task = r["task"][:44]
        if "error" in r:
            print(f"{suite:<22} {task:<45} ERROR: {r['error']}")
            continue
        fix = r.get("fix", 0.0)
        evosearch = r.get("evosearch", 0.0)
        delta = r.get("delta", 0.0)
        sign = "+" if delta >= 0 else ""
        print(f"{suite:<22} {task:<45} {fix:>4.0%}  {evosearch:>6.0%}  {sign}{delta:>5.0%}")

        if suite not in suite_totals:
            suite_totals[suite] = {"fix": [], "evosearch": []}
        suite_totals[suite]["fix"].append(fix)
        suite_totals[suite]["evosearch"].append(evosearch)

    print("-" * 90)
    for suite in SUITES:
        if suite in suite_totals:
            d = suite_totals[suite]
            n = len(d["fix"])
            avg_fix = sum(d["fix"]) / n
            avg_evosearch = sum(d["evosearch"]) / n
            delta = avg_evosearch - avg_fix
            sign = "+" if delta >= 0 else ""
            print(f"{suite:<22} {'AVG (' + str(n) + ' tasks)':<45} {avg_fix:>4.0%}  {avg_evosearch:>6.0%}  {sign}{delta:>5.0%}")
    print("=" * 90)


def main():
    global ASPIRE, EVOSEARCH_DIR, FIX_DIR, VAL_DIR, PYTHON, SUITES

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(ASPIRE),
                        help="Repository root. Defaults to this script's parent repo.")
    parser.add_argument("--evosearch-dir", default="outputs/claude_evosearch",
                        help="Evolutionary Search output dir, relative to --repo-root unless absolute.")
    parser.add_argument("--fix-dir", default="outputs/libero_baseline_image_diff_gemini",
                        help="Fix-loop output dir, relative to --repo-root unless absolute.")
    parser.add_argument("--output-dir", default="outputs/validation_comparison_seeds66_80",
                        help="Validation output dir, relative to --repo-root unless absolute.")
    parser.add_argument("--python", default=None,
                        help="Python interpreter for evosearch_eval.py. Defaults to repo .venv-libero.")
    parser.add_argument("--suites", nargs="+", default=SUITES,
                        help="Suites to compare.")
    parser.add_argument("--gpus", default="4,7", help="Comma-separated GPU IDs")
    parser.add_argument("--parallel-per-gpu", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ASPIRE = Path(args.repo_root).expanduser().resolve()
    EVOSEARCH_DIR = resolve_under_repo(args.evosearch_dir, ASPIRE)
    FIX_DIR = resolve_under_repo(args.fix_dir, ASPIRE)
    VAL_DIR = resolve_under_repo(args.output_dir, ASPIRE)
    PYTHON = args.python or str(ASPIRE / ".venv-libero/bin/python3")
    SUITES = args.suites

    gpus = [int(g) for g in args.gpus.split(",")]
    tasks = discover_tasks()

    print(f"Found {len(tasks)} tasks with both evosearch + fix code")
    print(f"Repo: {ASPIRE}")
    print(f"Evolutionary Search dir: {EVOSEARCH_DIR}")
    print(f"Fix dir: {FIX_DIR}")
    print(f"GPUs: {gpus}, parallel-per-gpu: {args.parallel_per_gpu}")
    print(f"Seeds: {EVAL_SEEDS[0]}–{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} seeds)")
    print(f"Output: {VAL_DIR}")
    print()

    if args.dry_run:
        for suite, task in tasks:
            ec = find_evosearch_code(suite, task)
            fc = find_fix_code(suite, task)
            print(f"  {suite}/{task}")
            print(f"    evosearch: {ec}")
            print(f"    fix:    {fc}")
        return

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Round-robin GPU assignment
    task_gpu = [(tasks[i], gpus[i % len(gpus)]) for i in range(len(tasks))]

    # Run with thread pool (one thread per GPU, tasks serialized per GPU)
    # Group tasks by GPU
    gpu_tasks: dict[int, list] = {g: [] for g in gpus}
    for (suite, task), gpu in task_gpu:
        gpu_tasks[gpu].append((suite, task))

    results = []

    def run_gpu_tasks(gpu: int, task_list: list) -> list:
        gpu_results = []
        for suite, task in task_list:
            r = run_eval(suite, task, gpu, args.parallel_per_gpu)
            gpu_results.append(r)
        return gpu_results

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {executor.submit(run_gpu_tasks, gpu, task_list): gpu
                   for gpu, task_list in gpu_tasks.items()}
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                gpu_results = future.result()
                results.extend(gpu_results)
            except Exception as e:
                print(f"GPU {gpu} error: {e}")

    # Save results JSON
    results_path = VAL_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print_table(results)


if __name__ == "__main__":
    main()
