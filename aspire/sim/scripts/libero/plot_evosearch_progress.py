# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot Evolutionary Search-Claude pipeline progress for a given run directory.
Produces a single line chart (no matrix) showing best and mean success rate per iteration.
Usage: python3 scripts/libero/plot_evosearch_progress.py --run-dir <path> [--output <path>]
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_iter_data(run_dir: Path):
    iters = []
    task_name = None
    for iter_dir in sorted(run_dir.glob("iter_*")):
        summary_file = iter_dir / "iter_summary.json"
        if not summary_file.exists():
            continue
        s = json.loads(summary_file.read_text())
        if task_name is None:
            task_name = s.get("task", run_dir.parent.name)
        iter_num = int(iter_dir.name.split("_")[1])
        # Recompute pass_rate from pass_count / actual trials run to correct old
        # normalization bug where pass_rate was divided by args.trials (default 50)
        # even when fewer seeds were evaluated.
        def actual_rate(c):
            n = len(c.get("trial_results", [])) or c.get("trials", 1)
            return c["pass_count"] / n if n > 0 else 0.0
        rates = [actual_rate(c) for c in s["candidates"]]
        best_rate = max(rates)
        mean_rate = sum(rates) / len(rates)
        iters.append({
            "iter": iter_num,
            "best_rate": best_rate * 100,
            "mean_rate": mean_rate * 100,
        })
    return task_name, iters


def plot_progress(run_dir: Path, output_path: Path, skip_iters: list[int] | None = None,
                  baseline: float | None = None):
    task_name, iters = load_iter_data(run_dir)
    if skip_iters:
        iters = [d for d in iters if d["iter"] not in skip_iters]
    if not iters:
        print("No iter_summary.json files found.")
        return

    iter_nums = [d["iter"] for d in iters]
    best_rates = [d["best_rate"] for d in iters]
    mean_rates = [d["mean_rate"] for d in iters]

    task_display = task_name.replace("_", " ").title() if task_name else run_dir.parent.name

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Claude-Evolutionary Search: Success Rate per Iteration\n{task_display}", fontsize=12)

    ax.plot(iter_nums, best_rates, "o-", color="#1f77b4", linewidth=2, markersize=6,
            label=f"{task_name} (best)")
    ax.plot(iter_nums, mean_rates, "o--", color="#1f77b4", linewidth=1.5, markersize=5,
            alpha=0.55, label=f"{task_name} (mean)")

    # Annotate final point
    if iter_nums:
        ax.annotate(f"{best_rates[-1]:.0f}%",
                    xy=(iter_nums[-1], best_rates[-1]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=9, color="#1f77b4")

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(min(iter_nums) - 0.3, max(iter_nums) + 0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    if baseline is not None:
        ax.axhline(y=baseline, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Baseline ({baseline:.0f}%)")

    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-iters", type=int, nargs="*", default=None,
                        help="Iteration numbers to exclude from the plot")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Horizontal baseline success rate (%%) to draw on the chart")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "progress.png"
    plot_progress(run_dir, output_path, skip_iters=args.skip_iters, baseline=args.baseline)
