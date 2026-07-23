#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot average task SR across checkpoint_eval token-usage checkpoints.

This summarizes the five LIBERO-Long-Pro debug_eval tasks for snapshot-N5 and
snapshot-N50. The 25% and 50% points come from
stage1/checkpoint_eval/{25pct,50pct}; the 100% point comes from Stage 2.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = [
    (
        "libero_10_swap",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    ),
    (
        "libero_10_swap",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    ),
    (
        "libero_10_swap",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    ),
    (
        "libero_10_task",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    ),
    (
        "libero_10_task",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    ),
]

BASELINE_SUMMARIES = {
    "N5": Path(
        "outputs/worktrees/snapshot-N5/outputs/scaling_eval/snapshot-N5/"
        "one_shot_apr26_unguarded/summary.json"
    ),
    "N50": Path(
        "outputs/worktrees/snapshot-N50/outputs/scaling_eval/snapshot-N50/"
        "one_shot/summary.json"
    ),
}


@dataclass(frozen=True)
class Count:
    success: int
    total: int

    @property
    def rate_pct(self) -> float:
        return 100.0 * self.success / self.total if self.total else 0.0

    def label(self) -> str:
        return f"{self.success}/{self.total} ({self.rate_pct:.1f}%)"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    # Baseline results default to this repo; override with --baseline-root if
    # they live in a separate checkout.
    default_baseline_root = repo_root

    parser = argparse.ArgumentParser(
        description="Plot average task SR over 25%, 50%, and full Stage2 checkpoints."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--baseline-root", type=Path, default=default_baseline_root)
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "outputs/plots/checkpoint_sr_pareto_n5_n50.png",
        help="Output plot path. A PDF with the same stem is also written.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["any_success", "latest", "strict"],
        default="any_success",
        help=(
            "How to collapse duplicate checkpoint_eval trial directories. "
            "any_success matches the summary table: a trial passes if any rerun "
            "directory for that trial is taskcompleted_1. latest uses newest mtime. "
            "strict fails when duplicates are present."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not draw dashed one-shot baseline reference lines.",
    )
    return parser.parse_args()


def trial_id(path: Path) -> int:
    match = re.match(r"trial_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse trial id from {path}")
    return int(match.group(1))


def trial_success(path: Path) -> bool:
    return "taskcompleted_1" in path.name


def checkpoint_count(task_root: Path, pct_name: str, duplicate_policy: str) -> Count:
    checkpoint_root = task_root / "stage1" / "checkpoint_eval" / pct_name
    by_trial: dict[int, list[Path]] = defaultdict(list)

    for run_dir in checkpoint_root.glob("*/*/*/run"):
        for trial_dir in run_dir.iterdir():
            if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
                by_trial[trial_id(trial_dir)].append(trial_dir)

    if not by_trial:
        raise FileNotFoundError(f"No trial dirs found under {checkpoint_root}")

    duplicate_trials = {tid: dirs for tid, dirs in by_trial.items() if len(dirs) > 1}
    if duplicate_trials and duplicate_policy == "strict":
        details = ", ".join(str(tid) for tid in sorted(duplicate_trials))
        raise ValueError(f"Duplicate trial dirs in {checkpoint_root}: {details}")

    successes = 0
    for dirs in by_trial.values():
        if duplicate_policy == "latest":
            selected = max(dirs, key=lambda p: p.stat().st_mtime)
            successes += int(trial_success(selected))
        else:
            successes += int(any(trial_success(path) for path in dirs))

    return Count(success=successes, total=len(by_trial))


def stage2_count(task_root: Path) -> Count:
    result_path = task_root / "stage2" / "stage2_result.json"
    result = json.loads(result_path.read_text())
    return Count(success=int(result["n_pass"]), total=int(result["n_total"]))


def baseline_counts(baseline_root: Path, snapshot: str) -> dict[tuple[str, str], Count]:
    summary_path = baseline_root / BASELINE_SUMMARIES[snapshot]
    summary = json.loads(summary_path.read_text())
    counts = {}

    for suite, task in TASKS:
        result = summary["results"][suite][task]
        counts[(suite, task)] = Count(
            success=int(result["success"]),
            total=int(result["total"]),
        )

    return counts


def task_root(repo_root: Path, snapshot: str, suite: str, task: str) -> Path:
    snapshot_name = f"snapshot-{snapshot}"
    return (
        repo_root
        / "outputs/worktrees"
        / snapshot_name
        / "outputs/scaling_eval"
        / snapshot_name
        / "debug_eval"
        / suite
        / task
    )


def aggregate(counts: list[Count]) -> Count:
    return Count(
        success=sum(count.success for count in counts),
        total=sum(count.total for count in counts),
    )


def collect_series(
    repo_root: Path,
    baseline_root: Path,
    duplicate_policy: str,
) -> tuple[dict[str, list[Count]], dict[str, Count]]:
    series: dict[str, list[Count]] = {}
    baselines: dict[str, Count] = {}

    for snapshot in ("N5", "N50"):
        per_task_25 = []
        per_task_50 = []
        per_task_100 = []

        for suite, task in TASKS:
            root = task_root(repo_root, snapshot, suite, task)
            per_task_25.append(checkpoint_count(root, "25pct", duplicate_policy))
            per_task_50.append(checkpoint_count(root, "50pct", duplicate_policy))
            per_task_100.append(stage2_count(root))

        series[snapshot] = [
            aggregate(per_task_25),
            aggregate(per_task_50),
            aggregate(per_task_100),
        ]
        baselines[snapshot] = aggregate(list(baseline_counts(baseline_root, snapshot).values()))

    return series, baselines


def print_summary(series: dict[str, list[Count]], baselines: dict[str, Count]) -> None:
    print("Average task SR:")
    print("snapshot | baseline        | 25%             | 50%             | 100% Stage2")
    print("---------|-----------------|-----------------|-----------------|----------------")
    for snapshot in ("N5", "N50"):
        q25, q50, full = series[snapshot]
        print(
            f"{snapshot:<8} | {baselines[snapshot].label():<15} | "
            f"{q25.label():<15} | {q50.label():<15} | {full.label()}"
        )


def plot(
    series: dict[str, list[Count]],
    baselines: dict[str, Count],
    out_path: Path,
    show_baseline: bool,
) -> None:
    checkpoints = [25, 50, 100]
    colors = {"N5": "#4C72B0", "N50": "#C44E52"}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for snapshot in ("N5", "N50"):
        y_values = [count.rate_pct for count in series[snapshot]]
        ax.plot(
            checkpoints,
            y_values,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=f"snapshot-{snapshot}",
            color=colors[snapshot],
        )

        for x_value, y_value in zip(checkpoints, y_values):
            ax.annotate(
                f"{y_value:.1f}%",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color=colors[snapshot],
            )

        if show_baseline:
            baseline_y = baselines[snapshot].rate_pct
            ax.axhline(
                baseline_y,
                linestyle="--",
                linewidth=1.4,
                color=colors[snapshot],
                alpha=0.45,
                label=f"snapshot-{snapshot} one-shot baseline ({baseline_y:.1f}%)",
            )

    ax.set_title("LIBERO-10 Debug Performance: Average Task SR vs Output Token Usage")
    ax.set_xlabel("% of total output tokens used")
    ax.set_ylabel("Average task success rate across 5 tasks (%)")
    ax.set_xlim(left=0, right=105)
    ax.set_xticks([0, *checkpoints], ["0%", *[f"{value}%" for value in checkpoints]])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    args = parse_args()
    series, baselines = collect_series(
        repo_root=args.repo_root,
        baseline_root=args.baseline_root,
        duplicate_policy=args.duplicate_policy,
    )
    print_summary(series, baselines)
    plot(
        series=series,
        baselines=baselines,
        out_path=args.out,
        show_baseline=not args.no_baseline,
    )


if __name__ == "__main__":
    main()
