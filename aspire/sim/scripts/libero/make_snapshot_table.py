#!/usr/bin/env python3
"""
Build per-task success rate table for all snapshots on LIBERO-10.
Rows = swap/task variants of 10 tasks (20 total) + suite averages + overall.
Columns = snapshots.
"""

import re
import csv
import math
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SNAPSHOTS = [
    ("N0",       ("snapshot-N0", "snapshot-N0-clean")),
    ("N5",       "snapshot-N5"),
    ("N10",      "snapshot-N10"),
    ("N15",      "snapshot-N15"),
    ("N20",      "snapshot-N20"),
    ("N25",      "snapshot-N25"),
    ("N30",      "snapshot-N30"),
    ("N35",      "snapshot-N35"),
    ("N40",      "snapshot-N40"),
    ("N45",      "snapshot-N45"),
    ("N50",      "snapshot-N50"),
    ("N55",      "snapshot-N55"),
    ("N60",      "snapshot-N60"),
    ("N65",      "snapshot-N65"),
    ("N70",      "snapshot-N70"),
    ("N75",      "snapshot-N75"),
    ("N80",      "snapshot-N80"),
    ("N85",      "snapshot-N85"),
    ("N90",      "snapshot-N90"),
]

TASK_SHORT = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it":                                                "K3_stove_moka",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it":                         "K4_bowl_drawer",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it":                                  "K6_mug_microwave",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove":                                                             "K8_both_mokas",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket":                       "LR1_soup_cheese",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket":                           "LR2_soup_sauce",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket":                              "LR2_cheese_butter",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": "LR5_mugs_plates",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate":  "LR6_mug_pudding",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy":                            "SS1_book_caddy",
}

SUITES = ["libero_10_swap", "libero_10_task"]
SUITE_SHORT = {"libero_10_swap": "swap", "libero_10_task": "task"}

TRIAL_RE = re.compile(r"trial_(\d+)_sandboxrc_\d+_reward_([\d.]+)_taskcompleted_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-task LIBERO-10 success tables across snapshot eval worktrees."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--worktrees", type=Path, default=None,
                        help="Defaults to <repo-root>/outputs/worktrees.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to <repo-root>/outputs.")
    return parser.parse_args()


def resolve_snapshot_one_shot(worktrees: Path, snapshot_spec: str | tuple[str, ...]) -> tuple[str, Path]:
    names = (snapshot_spec,) if isinstance(snapshot_spec, str) else snapshot_spec
    for name in names:
        scaling_root = worktrees / name / "outputs" / "scaling_eval"
        direct = scaling_root / name / "one_shot"
        if direct.exists():
            return name, direct
        if scaling_root.exists():
            matches = sorted(p / "one_shot" for p in scaling_root.iterdir() if (p / "one_shot").exists())
            if matches:
                return name, matches[0]
    name = names[0]
    return name, worktrees / name / "outputs" / "scaling_eval" / name / "one_shot"


def get_task_results(task_dir: Path) -> dict:
    results = {}
    run_dir = task_dir / "aws_anthropic_bedrock-claude-sonnet-4-6" / "run"
    if not run_dir.exists():
        return results
    for entry in run_dir.iterdir():
        m = TRIAL_RE.match(entry.name)
        if m:
            results[int(m.group(1))] = int(m.group(3))
    return results


def success_rate(results: dict) -> float:
    if not results:
        return float("nan")
    return sum(1 for v in results.values() if v == 1) / len(results)


def avg(vals):
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def pct(v: float) -> str:
    return f"{v*100:.0f}%" if not math.isnan(v) else "--"


def main():
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    worktrees = args.worktrees.expanduser().resolve() if args.worktrees else repo_root / "outputs/worktrees"
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_roots = [
        (label, *resolve_snapshot_one_shot(worktrees, snapshot_spec))
        for label, snapshot_spec in SNAPSHOTS
    ]

    # Discover task names from first available snapshot
    all_tasks = []
    for _, _, one_shot_root in snapshot_roots:
        base = one_shot_root / SUITES[0]
        if base.exists():
            all_tasks = sorted([d.name for d in base.iterdir() if d.is_dir()])
            break
    if not all_tasks:
        raise SystemExit(f"No task eval dirs found under {worktrees}")

    short_tasks = [TASK_SHORT.get(t, t) for t in all_tasks]
    snap_labels = [label for label, _, _ in snapshot_roots]

    # data[(suite_short, task_short)][snap_label] = rate
    data = {}
    for suite in SUITES:
        s = SUITE_SHORT[suite]
        for short in short_tasks:
            data[(s, short)] = {}

    suite_avgs  = {SUITE_SHORT[s]: {} for s in SUITES}  # suite_short → {snap_label: rate}
    overall_avg = {}

    for label, snap_dir, base_root in snapshot_roots:
        all_rates = []
        for suite in SUITES:
            s = SUITE_SHORT[suite]
            suite_rates = []
            for task, short in zip(all_tasks, short_tasks):
                task_dir = base_root / suite / task
                r = success_rate(get_task_results(task_dir)) if task_dir.exists() else float("nan")
                data[(s, short)][label] = r
                if not math.isnan(r):
                    suite_rates.append(r)
                    all_rates.append(r)
            suite_avgs[s][label] = avg(suite_rates)
        overall_avg[label] = avg(all_rates)
        print(f"  {label} ({snap_dir}): swap={pct(suite_avgs['swap'][label])}  task={pct(suite_avgs['task'][label])}  overall={pct(overall_avg[label])}")

    # Row order: swap tasks | task tasks | (summary: swap avg, task avg, overall)
    task_rows = []  # (label, {snap_label: rate})
    for suite in SUITES:
        s = SUITE_SHORT[suite]
        for short in short_tasks:
            task_rows.append((f"{s}/{short}", data[(s, short)]))

    summary_rows = [
        ("swap avg",   suite_avgs["swap"]),
        ("task avg",   suite_avgs["task"]),
        ("overall avg", overall_avg),
    ]

    all_labels = [r[0] for r in task_rows] + [r[0] for r in summary_rows]
    task_col_w = max(len(l) for l in all_labels) + 2
    # compact: 4-char numeric columns, no % in cells (noted in header)
    col_w = max(7, max(len(l) for l in snap_labels) + 1)

    def fmt(v: float) -> str:
        if math.isnan(v):
            return "--"
        return f"{v*100:.1f}%"

    # --- Fixed-width .txt ---
    txt_path = out_dir / "snapshot_pertask_success.txt"
    with open(txt_path, "w") as f:
        header = f"  {'Task':<{task_col_w}}" + "".join(f"{l:>{col_w}}" for l in snap_labels)
        sep    = f"  {'':-<{task_col_w}}" + "".join(f" {'':-<{col_w-1}}" for _ in snap_labels)

        f.write(header + "\n")
        f.write(sep + "\n")

        prev_suite = None
        for row_label, row_data in task_rows:
            cur_suite = row_label.split("/")[0]
            if prev_suite is not None and cur_suite != prev_suite:
                f.write(sep + "\n")
            prev_suite = cur_suite
            vals = "".join(f"{fmt(row_data.get(l, float('nan'))):>{col_w}}" for l in snap_labels)
            f.write(f"  {row_label:<{task_col_w}}{vals}\n")

        f.write(sep + "\n")
        for row_label, row_data in summary_rows:
            vals = "".join(f"{fmt(row_data.get(l, float('nan'))):>{col_w}}" for l in snap_labels)
            f.write(f"  {row_label:<{task_col_w}}{vals}\n")
        f.write(sep + "\n")

    print(f"\nFixed-width txt: {txt_path}")

    # --- CSV ---
    csv_path = out_dir / "snapshot_pertask_success.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task (%)"] + snap_labels)
        for row_label, row_data in task_rows:
            writer.writerow([row_label] + [fmt(row_data.get(l, float("nan"))) for l in snap_labels])
        writer.writerow([])
        for row_label, row_data in summary_rows:
            writer.writerow([row_label] + [fmt(row_data.get(l, float("nan"))) for l in snap_labels])
    print(f"CSV: {csv_path}")

    # --- LaTeX ---
    tex_path = out_dir / "snapshot_pertask_success.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l" + "r" * len(snap_labels) + "}\n\\hline\n")
        snap_headers = " & ".join(f"\\rotatebox{{60}}{{{l}}}" for l in snap_labels)
        f.write(f"Task (\\%) & {snap_headers} \\\\\n\\hline\n")
        prev_suite = None
        for row_label, row_data in task_rows:
            cur_suite = row_label.split("/")[0]
            if prev_suite is not None and cur_suite != prev_suite:
                f.write("\\hline\n")
            prev_suite = cur_suite
            vals = " & ".join(fmt(row_data.get(l, float("nan"))) for l in snap_labels)
            f.write(f"{row_label} & {vals} \\\\\n")
        f.write("\\hline\n")
        for row_label, row_data in summary_rows:
            vals = " & ".join(fmt(row_data.get(l, float("nan"))) for l in snap_labels)
            f.write(f"\\textbf{{{row_label}}} & {vals} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Per-task success rates (\\%) across LIBERO-10 snapshots (seeds 1--50).}\n")
        f.write("\\label{tab:snapshot_pertask}\n\\end{table}\n")
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
