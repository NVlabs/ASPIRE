#!/usr/bin/env python3
"""Plot cumulative seen-task SR for the LIBERO-90 scaling build.

The x-axis is curriculum progress in batches of 5 attempted tasks. The y-axis
is the average held-out evaluation success rate over tasks seen up to that
batch, using results from outputs/scaling_eval/libero_90_eval.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TaskResult:
    task: str
    success: int
    total: int
    missing: bool = False

    @property
    def rate_pct(self) -> float:
        return 100.0 * self.success / self.total if self.total else 0.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot LIBERO-90 cumulative seen-task average SR over chunks."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo_root / "outputs/scaling_eval/libero_90_eval",
    )
    parser.add_argument("--ordering", type=Path, default=repo_root / "ordering.txt")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "outputs/plots/libero90_seen_task_sr.png",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["zero", "skip", "fail"],
        default="zero",
        help=(
            "How to handle seen tasks with no eval directory. zero treats them as "
            "0/50, appropriate for blocked attempted tasks; skip excludes them; "
            "fail raises an error."
        ),
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["latest", "any_success", "strict"],
        default="latest",
        help="How to collapse duplicate trial dirs for the same seed.",
    )
    return parser.parse_args()


def read_ordering(path: Path) -> list[str]:
    tasks = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(tasks) != 90:
        raise ValueError(f"Expected 90 tasks in {path}, found {len(tasks)}")
    return tasks


def trial_id(path: Path) -> int:
    match = re.match(r"trial_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Could not parse trial id from {path.name}")
    return int(match.group(1))


def trial_success(path: Path) -> bool:
    return "taskcompleted_1" in path.name


def count_task_eval(
    eval_root: Path,
    task: str,
    duplicate_policy: str,
    missing_policy: str,
    expected_trials: int = 50,
) -> TaskResult | None:
    task_root = eval_root / task
    if not task_root.exists():
        if missing_policy == "fail":
            raise FileNotFoundError(f"Missing eval directory for seen task: {task_root}")
        if missing_policy == "skip":
            return None
        return TaskResult(task=task, success=0, total=expected_trials, missing=True)

    by_trial: dict[int, list[Path]] = defaultdict(list)
    for run_dir in task_root.glob("libero_90/*/*/run"):
        for trial_dir in run_dir.iterdir():
            if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
                by_trial[trial_id(trial_dir)].append(trial_dir)

    if not by_trial:
        if missing_policy == "fail":
            raise FileNotFoundError(f"No trial dirs found for seen task: {task_root}")
        if missing_policy == "skip":
            return None
        return TaskResult(task=task, success=0, total=expected_trials, missing=True)

    duplicate_trials = {tid: dirs for tid, dirs in by_trial.items() if len(dirs) > 1}
    if duplicate_trials and duplicate_policy == "strict":
        details = ", ".join(str(tid) for tid in sorted(duplicate_trials))
        raise ValueError(f"Duplicate trial dirs for {task}: {details}")

    success = 0
    for dirs in by_trial.values():
        if duplicate_policy == "any_success":
            success += int(any(trial_success(path) for path in dirs))
        else:
            latest = max(dirs, key=lambda path: path.stat().st_mtime)
            success += int(trial_success(latest))

    return TaskResult(task=task, success=success, total=len(by_trial))


def cumulative_seen_task_results(
    tasks: list[str],
    task_results: dict[str, TaskResult],
) -> tuple[list[int], list[float], list[float], list[int], list[int]]:
    x_tasks = []
    cumulative_sr = []
    chunk_sr = []
    cumulative_success = []
    cumulative_total = []

    seen: list[TaskResult] = []
    for chunk_idx in range(18):
        chunk_tasks = tasks[chunk_idx * 5 : (chunk_idx + 1) * 5]
        chunk_results = [task_results[task] for task in chunk_tasks if task in task_results]
        seen.extend(chunk_results)

        seen_success = sum(result.success for result in seen)
        seen_total = sum(result.total for result in seen)
        chunk_success = sum(result.success for result in chunk_results)
        chunk_total = sum(result.total for result in chunk_results)

        x_tasks.append((chunk_idx + 1) * 5)
        cumulative_sr.append(100.0 * seen_success / seen_total if seen_total else 0.0)
        chunk_sr.append(100.0 * chunk_success / chunk_total if chunk_total else 0.0)
        cumulative_success.append(seen_success)
        cumulative_total.append(seen_total)

    return x_tasks, cumulative_sr, chunk_sr, cumulative_success, cumulative_total


def plot(
    x_tasks: list[int],
    cumulative_sr: list[float],
    chunk_sr: list[float],
    out_path: Path,
    missing_count: int,
    missing_policy: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        x_tasks,
        cumulative_sr,
        marker="o",
        linewidth=2.3,
        markersize=5.5,
        color="#4C72B0",
        label="Cumulative seen-task average SR",
    )
    ax.scatter(
        x_tasks,
        chunk_sr,
        color="#C44E52",
        alpha=0.55,
        s=28,
        label="Per-batch SR (5 new tasks)",
    )

    ax.set_title("LIBERO-90 Training-Time Scaling: Seen-Task Average SR")
    ax.set_xlabel("Tasks attempted in scaling build")
    ax.set_ylabel("Average task success rate (%)")
    ax.set_xticks(x_tasks)
    ax.set_xlim(0, 92)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = f"Missing seen evals: {missing_count} ({missing_policy})"
    ax.text(
        0.99,
        0.02,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    args = parse_args()
    tasks = read_ordering(args.ordering)

    task_results: dict[str, TaskResult] = {}
    missing_count = 0
    for task in tasks:
        result = count_task_eval(
            eval_root=args.eval_root,
            task=task,
            duplicate_policy=args.duplicate_policy,
            missing_policy=args.missing_policy,
        )
        if result is None:
            continue
        task_results[task] = result
        missing_count += int(result.missing)

    x_tasks, cumulative_sr, chunk_sr, cumulative_success, cumulative_total = (
        cumulative_seen_task_results(tasks, task_results)
    )

    print("batch | tasks_attempted | cumulative_seen_SR | per_batch_SR")
    print("------|-----------------|--------------------|-------------")
    for batch, x_value, cum_sr, batch_sr, success, total in zip(
        range(1, 19),
        x_tasks,
        cumulative_sr,
        chunk_sr,
        cumulative_success,
        cumulative_total,
    ):
        print(
            f"{batch:>5} | {x_value:>15} | "
            f"{success}/{total} ({cum_sr:>5.1f}%) | {batch_sr:>5.1f}%"
        )

    plot(
        x_tasks=x_tasks,
        cumulative_sr=cumulative_sr,
        chunk_sr=chunk_sr,
        out_path=args.out,
        missing_count=missing_count,
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":
    main()
