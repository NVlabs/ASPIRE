#!/usr/bin/env python3
"""Plot cumulative solved-task fraction for the LIBERO-90 scaling build.

This is a capability-coverage view: after each batch of 5 attempted tasks, what
fraction of seen tasks have evaluation SR above fixed thresholds?
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
        description="Plot LIBERO-90 cumulative solved-task fraction over batches."
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
        default=repo_root / "outputs/plots/libero90_solved_fraction.png",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.8],
        help="Solved thresholds as fractions, e.g. 0.5 0.8.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["zero", "skip", "fail"],
        default="zero",
        help="How to handle seen tasks with no eval dir. Default zero treats blocked tasks as unsolved.",
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


def collect_task_results(args: argparse.Namespace, tasks: list[str]) -> dict[str, TaskResult]:
    results: dict[str, TaskResult] = {}
    for task in tasks:
        result = count_task_eval(
            eval_root=args.eval_root,
            task=task,
            duplicate_policy=args.duplicate_policy,
            missing_policy=args.missing_policy,
        )
        if result is not None:
            results[task] = result
    return results


def cumulative_solved_fraction(
    tasks: list[str],
    results: dict[str, TaskResult],
    thresholds: list[float],
) -> tuple[list[int], dict[float, list[float]], dict[float, list[int]], list[int]]:
    x_tasks = []
    fractions = {threshold: [] for threshold in thresholds}
    solved_counts = {threshold: [] for threshold in thresholds}
    seen_counts = []

    seen: list[TaskResult] = []
    for chunk_idx in range(18):
        chunk_tasks = tasks[chunk_idx * 5 : (chunk_idx + 1) * 5]
        seen.extend(results[task] for task in chunk_tasks if task in results)

        x_tasks.append((chunk_idx + 1) * 5)
        seen_counts.append(len(seen))
        for threshold in thresholds:
            solved = sum(result.rate >= threshold for result in seen)
            solved_counts[threshold].append(solved)
            fractions[threshold].append(100.0 * solved / len(seen) if seen else 0.0)

    return x_tasks, fractions, solved_counts, seen_counts


def plot(
    x_tasks: list[int],
    fractions: dict[float, list[float]],
    solved_counts: dict[float, list[int]],
    seen_counts: list[int],
    out_path: Path,
    missing_count: int,
    missing_policy: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]

    for idx, threshold in enumerate(sorted(fractions)):
        label = f"SR >= {threshold * 100:.0f}%"
        ax.plot(
            x_tasks,
            fractions[threshold],
            marker="o",
            linewidth=2.3,
            markersize=5.5,
            color=colors[idx % len(colors)],
            label=label,
        )

        final_solved = solved_counts[threshold][-1]
        final_seen = seen_counts[-1]
        ax.annotate(
            f"{final_solved}/{final_seen}",
            (x_tasks[-1], fractions[threshold][-1]),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=9,
            color=colors[idx % len(colors)],
        )

    ax.set_title("LIBERO-90 Training-Time Scaling: Solved-Task Coverage")
    ax.set_xlabel("Tasks attempted in scaling build")
    ax.set_ylabel("Seen tasks above SR threshold (%)")
    ax.set_xticks(x_tasks)
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")

    note = f"Missing seen evals: {missing_count} ({missing_policy})"
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        ha="left",
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
    results = collect_task_results(args, tasks)
    missing_count = sum(result.missing for result in results.values())

    x_tasks, fractions, solved_counts, seen_counts = cumulative_solved_fraction(
        tasks=tasks,
        results=results,
        thresholds=sorted(args.thresholds),
    )

    print("batch | tasks_attempted | seen | " + " | ".join(f"SR>={t:.0%}" for t in sorted(args.thresholds)))
    print("------|-----------------|------|" + "|".join("-" * 14 for _ in args.thresholds))
    for idx, x_value in enumerate(x_tasks):
        cells = []
        for threshold in sorted(args.thresholds):
            solved = solved_counts[threshold][idx]
            frac = fractions[threshold][idx]
            cells.append(f"{solved}/{seen_counts[idx]} ({frac:>5.1f}%)")
        print(f"{idx + 1:>5} | {x_value:>15} | {seen_counts[idx]:>4} | " + " | ".join(cells))

    plot(
        x_tasks=x_tasks,
        fractions=fractions,
        solved_counts=solved_counts,
        seen_counts=seen_counts,
        out_path=args.out,
        missing_count=missing_count,
        missing_policy=args.missing_policy,
    )


if __name__ == "__main__":
    main()
