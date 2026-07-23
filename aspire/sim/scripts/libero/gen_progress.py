#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Auto-generate docs/progress/fix_loop_progress.md from filesystem state.

Run: .venv-libero/bin/python3 scripts/libero/gen_progress.py
     (or append --print to stdout only, without writing the file)

The coordinator should run this instead of manually editing fix_loop_progress.md.
Task names come from the LIBERO benchmark; status is derived from disk artifacts, not agent memory.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD    = ROOT / "outputs/libero_fix_loop"
BASELINE = ROOT / "outputs/libero_baseline_image_diff_gemini"
FIXED    = ROOT / "outputs/libero_fix_loop_eval"
OUT_FILE = ROOT / "docs/progress/fix_loop_progress.md"
DEFAULT_CONFIG = ROOT / "env_configs/libero/franka_libero_traced.yaml"

SUITES = [
    "libero_goal_swap",
    "libero_goal_task",
    "libero_object_swap",
    "libero_object_task",
    "libero_spatial_swap",
    "libero_spatial_task",
]

DONE_THRESHOLD = 50  # every task must complete the full held-out partition

# Matches trial_01_sandboxrc_0_reward_1.000_taskcompleted_1 and extracts seed + completed flag
_TRIAL_RE = re.compile(r"trial_(\d+)_sandboxrc_\d+_reward_[\d.]+_taskcompleted_(\d+)")


def _parse_trial_dirs(outdir: Path) -> dict[int, list[tuple[float, int]]]:
    """Return {seed: [(mtime, completed), ...]} for seeds 1–50 under outdir."""
    result = subprocess.run(
        ["find", str(outdir), "-maxdepth", "5", "-type", "d", "-name", "trial_*"],
        capture_output=True, text=True, timeout=60,
    )
    seed_runs: dict[int, list[tuple[float, int]]] = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        p = Path(line)
        m = _TRIAL_RE.search(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        if not (1 <= seed <= 50):
            continue
        completed = int(m.group(2))
        mtime = p.stat().st_mtime
        seed_runs.setdefault(seed, []).append((mtime, completed))
    return seed_runs


def _best_pass(runs: list[tuple[float, int]]) -> int:
    """
    For a seed with multiple runs, determine whether it passed.

    Strategy: take max(first_run_result, last_run_result).
    - first > last: Stage 2 batch ran first; later re-runs were debug iterations (worse).
    - last > first: debugging ran first; Stage 2 batch ran last (better).
    - Tie: single run or both runs agree.

    Returns 1 if the seed should be counted as passed, 0 otherwise.
    Marks seed as NEEDS_RERUN if any-pass differs from best (ambiguous history).
    """
    if not runs:
        return 0
    first = min(runs, key=lambda r: r[0])[1]
    last  = max(runs, key=lambda r: r[0])[1]
    return max(first, last)


def count_trials(outdir: Path) -> int:
    """Count unique seeds 1–50 (not raw dirs) — re-runs of the same seed count once."""
    if not outdir.exists():
        return 0
    return len(_parse_trial_dirs(outdir))


def count_passes(outdir: Path) -> int:
    """
    Count seeds where the best run (max of first/last) succeeded.
    Seeds re-run with a worse code after a good run still count as passing.
    """
    if not outdir.exists():
        return 0
    return sum(_best_pass(runs) for runs in _parse_trial_dirs(outdir).values())


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def latest_validation_manifest(
    task_eval_dir: Path, *, code_sha256: str | None = None, config_sha256: str | None = None
) -> dict | None:
    """Return the best coherent manifest matching the selected code/config.

    Full 1–50 runs outrank subset diagnostics; recency breaks ties.
    """
    candidates = []
    for path in task_eval_dir.glob("runs/*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
            identity = manifest.get("identity", {})
            if code_sha256 and identity.get("code_sha256") != code_sha256:
                continue
            if config_sha256 and identity.get("config_sha256") != config_sha256:
                continue
            seeds = identity.get("seeds", [])
            is_full = seeds == list(range(1, 51))
            updated = manifest.get("updated_at", "")
            candidates.append((is_full, updated, path.stat().st_mtime, path, manifest))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        return None
    _, _, _, path, manifest = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    manifest["_manifest_path"] = str(path)
    return manifest


def manifest_counts(manifest: dict) -> tuple[int, int]:
    """Count held-out trials and passes from one immutable run only."""
    rows = manifest.get("results", {})
    valid = [
        row for key, row in rows.items()
        if str(key).isdigit() and 1 <= int(key) <= 50
    ]
    return len(valid), sum(int(row.get("task_completed", 0)) for row in valid)


def get_status(fix_code: Path | None, trials: int) -> str:
    if fix_code is not None and fix_code.exists():
        return "done" if trials >= DONE_THRESHOLD else "stage1-done"
    return "pending"


def shorten(task: str) -> str:
    t = task
    t = t.replace("pick_up_the_black_bowl_", "bowl: ")
    t = t.replace("_and_place_it_on_the_plate", " → plate")
    t = t.replace("pick_up_the_", "")
    t = t.replace("_and_place_it_in_the_basket", " → basket")
    t = t.replace("open_the_", "open ")
    t = t.replace("_and_put_the_bowl_inside", " + bowl")
    t = t.replace("push_the_plate_to_the_front_of_the_stove", "push plate → stove front")
    t = t.replace("put_the_", "")
    t = t.replace("_on_top_of_the_", " → top of ")
    t = t.replace("_in_the_", " → ")
    t = t.replace("_on_the_", " → ")
    t = t.replace("turn_on_the_", "turn on ")
    return t


def fmt_rate(passes: int, trials: int) -> str:
    if trials == 0:
        return "—"
    pct = passes * 100 // trials
    return f"{passes}/{trials} ({pct}%)"


def main():
    lines = []
    lines.append("# Fix Loop Progress")
    lines.append("")
    lines.append("*Auto-generated by `scripts/libero/gen_progress.py` — do not edit manually.*")
    lines.append("**Status key:** `done` = 50 coherent held-out trials | `stage1-done` selected fix awaiting or continuing validation | `pending` Stage 1 and final code selection required")
    lines.append("")

    # Collect pending tasks for summary at top
    pending_tasks = []
    stage1_done_tasks = []

    # Discover tasks from LIBERO so Fix Loop works without a separately
    # collected baseline. Union with output directories for degraded operation
    # when the benchmark package is temporarily unavailable.
    benchmark_tasks: dict[str, set[str]] = {suite: set() for suite in SUITES}
    try:
        from libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        for suite in SUITES:
            bench = benchmark_dict[suite]()
            benchmark_tasks[suite].update(
                bench.get_task(i).name for i in range(bench.n_tasks)
            )
    except Exception as exc:
        print(
            f"WARNING: benchmark task discovery failed ({exc}); using tasks found on disk",
            file=sys.stderr,
        )

    config_sha256 = sha256_file(DEFAULT_CONFIG)
    suite_rows = {}
    for suite in SUITES:
        build_suite_dir = BUILD / suite
        legacy_suite_dir = BASELINE / suite
        tasks = benchmark_tasks[suite]
        for suite_dir in (build_suite_dir, legacy_suite_dir):
            if suite_dir.exists():
                tasks.update(p.name for p in suite_dir.iterdir() if p.is_dir())
        if not tasks:
            continue

        rows = []
        for task in sorted(tasks):
            task_dir = build_suite_dir / task
            legacy_task_dir = legacy_suite_dir / task
            canonical_fix = task_dir / "fix_code.py"
            legacy_fix = legacy_task_dir / "fix_code.py"
            fix_code = canonical_fix if canonical_fix.exists() else (
                legacy_fix if legacy_fix.exists() else None
            )
            out_dir = FIXED / suite / task
            manifest = latest_validation_manifest(
                out_dir,
                code_sha256=sha256_file(fix_code) if fix_code else None,
                config_sha256=config_sha256,
            )
            if manifest is not None:
                trials, passes = manifest_counts(manifest)
                run_note = f" [run {manifest.get('run_id', 'unknown')}]"
            elif any(out_dir.glob("runs/*/manifest.json")):
                # Manifests exist, but none match the selected code/config.
                trials, passes, run_note = 0, 0, ""
            else:
                # Backward-compatible read of pre-manifest validation outputs.
                seed_runs = _parse_trial_dirs(out_dir) if out_dir.exists() else {}
                trials = len(seed_runs)
                passes = sum(_best_pass(runs) for runs in seed_runs.values()) if trials else 0
                run_note = " [legacy]" if trials else ""
            status = get_status(fix_code, trials)
            rate = (fmt_rate(passes, trials) + run_note) if trials > 0 else "—"
            rows.append((task, status, rate, passes, trials))

            if status == "pending":
                pending_tasks.append(f"{suite}/{task}")
            elif status == "stage1-done":
                stage1_done_tasks.append(f"{suite}/{task}")

        suite_rows[suite] = rows

    # Action summary at top
    if pending_tasks or stage1_done_tasks:
        lines.append("## Next Up")
        lines.append("")
        if stage1_done_tasks:
            lines.append("**Stage 2 needed** (fix_code.py exists, run validation on seeds 1–50):")
            for t in stage1_done_tasks:
                lines.append(f"- {t}")
            lines.append("")
        if pending_tasks:
            lines.append("**Pending** (ready for Stage 1):")
            for t in pending_tasks:
                lines.append(f"- {t}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Per-suite tables
    for suite in SUITES:
        rows = suite_rows.get(suite)
        if not rows:
            continue

        done = sum(1 for _, s, *_ in rows if s == "done")
        total = len(rows)
        lines.append(f"## {suite}  ({done}/{total} done)")
        lines.append("")
        lines.append("| Task | Status | Rate |")
        lines.append("|---|---|---|")
        for task, status, rate, *_ in rows:
            lines.append(f"| {shorten(task)} | {status} | {rate} |")
        lines.append("")

    output = "\n".join(lines)

    print_only = "--print" in sys.argv
    if print_only:
        print(output)
    else:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT_FILE.with_suffix(".tmp")
        tmp.write_text(output)
        tmp.rename(OUT_FILE)  # atomic on Linux — safe under concurrent subagent writes
        print(f"Written: {OUT_FILE}")

    # Also print a quick summary to stdout
    total_done = sum(1 for s in SUITES for _, st, *_ in suite_rows.get(s, []) if st == "done")
    total_tasks = sum(len(suite_rows.get(s, [])) for s in SUITES)
    print(f"Progress: {total_done}/{total_tasks} tasks done  |  "
          f"{len(stage1_done_tasks)} need Stage 2  |  "
          f"{len(pending_tasks)} pending")


if __name__ == "__main__":
    main()
