#!/usr/bin/env python3
"""Plot average SR over all 90 LIBERO tasks during the scaling build.

Unlike the seen-task chart, this treats tasks that have not yet entered the
curriculum as 0 SR. This mostly visualizes coverage growth over the 18 batches.
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
    def rate(self) -> float:
        return self.success / self.total if self.total else 0.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Plot LIBERO-90 average SR with unseen tasks counted as zero."
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
        default=repo_root / "outputs/plots/libero90_all_task_zero_unseen_sr.png",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["zero", "skip", "fail"],
        default="zero",
        help="How to handle attempted tasks with no eval dir. Default zero treats blocked tasks as 0/50.",
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
            raise FileNotFoundError(f"Missing eval directory for attempted task: {task_root}")
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
            raise FileNotFoundError(f"No trial dirs found for attempted task: {task_root}")
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


def collect_attempted_results(
    tasks: list[str],
    eval_root: Path,
    duplicate_policy: str,
    missing_policy: str,
) -> tuple[dict[str, TaskResult], int]:
    results: dict[str, TaskResult] = {}
    missing_count = 0
    for task in tasks:
        result = count_task_eval(
            eval_root=eval_root,
            task=task,
            duplicate_policy=duplicate_policy,
            missing_policy=missing_policy,
        )
        if result is None:
            continue
        results[task] = result
        missing_count += int(result.missing)
    return results, missing_count


def all_task_curve(
    tasks: list[str],
    results: dict[str, TaskResult],
    expected_trials: int = 50,
) -> tuple[list[int], list[float], list[int], list[int]]:
    x_tasks = []
    avg_sr = []
    successes = []
    totals = []

    for chunk_idx in range(18):
        seen_tasks = set(tasks[: (chunk_idx + 1) * 5])
        total_success = 0
        total_trials = len(tasks) * expected_trials

        for task in tasks:
            if task not in seen_tasks:
                continue
            result = results.get(task)
            if result is None:
                continue
            total_success += result.success

        x_tasks.append((chunk_idx + 1) * 5)
        successes.append(total_success)
        totals.append(total_trials)
        avg_sr.append(100.0 * total_success / total_trials)

    return x_tasks, avg_sr, successes, totals


def plot(
    x_tasks: list[int],
    avg_sr: list[float],
    out_path: Path,
    missing_count: int,
    missing_policy: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        x_tasks,
        avg_sr,
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        color="#4C72B0",
        label="All-task average SR (unseen = 0)",
    )

    ax.annotate(
        f"{avg_sr[-1]:.1f}%",
        (x_tasks[-1], avg_sr[-1]),
        textcoords="offset points",
        xytext=(8, 0),
        va="center",
        fontsize=9,
        color="#4C72B0",
    )

    ax.set_title("LIBERO-90 Training-Time Scaling: All-Task Average SR")
    ax.set_xlabel("Tasks attempted in scaling build")
    ax.set_ylabel("Average SR over all 90 tasks (%)")
    ax.set_xticks(x_tasks)
    ax.set_xlim(0, 96)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")

    note = f"Unseen tasks count as 0 SR; missing seen evals: {missing_count} ({missing_policy})"
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
    results, missing_count = collect_attempted_results(
        tasks=tasks,
        eval_root=args.eval_root,
        duplicate_policy=args.duplicate_policy,
        missing_policy=args.missing_policy,
    )
    x_tasks, avg_sr, successes, totals = all_task_curve(tasks, results)

    print("batch | tasks_attempted | all_task_SR")
    print("------|-----------------|------------")
    for batch, x_value, sr, success, total in zip(
        range(1, 19), x_tasks, avg_sr, successes, totals
    ):
        print(f"{batch:>5} | {x_value:>15} | {success}/{total} ({sr:>5.1f}%)")

    plot(
        x_tasks=x_tasks,
        avg_sr=avg_sr,
        out_path=args.out,
        missing_count=missing_count,
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":
    main()
