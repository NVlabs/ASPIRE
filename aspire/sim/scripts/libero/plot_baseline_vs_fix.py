# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import subprocess
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "outputs/libero_baseline_image_diff_gemini"
FIXED    = ROOT / "outputs/libero_fix_loop_eval"

SUITES = [
    ("Goal Swap",     "libero_goal_swap"),
    ("Goal Task",     "libero_goal_task"),
    ("Object Swap",   "libero_object_swap"),
    ("Object Task",   "libero_object_task"),
    ("Spatial Swap",  "libero_spatial_swap"),
    ("Spatial Task",  "libero_spatial_task"),
]

TASK_LABELS = {
    "open_the_middle_drawer_of_the_cabinet":                                                "open_middle_drawer",
    "open_the_top_drawer_and_put_the_bowl_inside":                                          "open_top_drawer+bowl",
    "push_the_plate_to_the_front_of_the_stove":                                             "push_plate→stove",
    "put_the_bowl_on_the_plate":                                                             "bowl→plate",
    "put_the_bowl_on_the_stove":                                                             "bowl→stove",
    "put_the_bowl_on_top_of_the_cabinet":                                                   "bowl→cabinet",
    "put_the_cream_cheese_in_the_bowl":                                                     "cream_cheese→bowl",
    "put_the_wine_bottle_on_the_rack":                                                      "wine→rack",
    "put_the_wine_bottle_on_top_of_the_cabinet":                                            "wine→cabinet",
    "turn_on_the_stove":                                                                     "turn_on_stove",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket":                                 "alphabet_soup",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket":                                     "bbq_sauce",
    "pick_up_the_butter_and_place_it_in_the_basket":                                        "butter",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket":                             "choc_pudding",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket":                                  "cream_cheese",
    "pick_up_the_ketchup_and_place_it_in_the_basket":                                       "ketchup",
    "pick_up_the_milk_and_place_it_in_the_basket":                                          "milk",
    "pick_up_the_orange_juice_and_place_it_in_the_basket":                                  "orange_juice",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket":                                "salad_dressing",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket":                                  "tomato_sauce",
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":  "between_plate_ramekin",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":                  "table_center",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": "top_drawer",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":             "next_cookie_box",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":                  "next_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":                "next_ramekin",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":                  "on_cookie_box",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":                     "on_ramekin",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":                       "on_stove",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":              "on_wooden_cabinet",
}

DONE_THRESHOLD = 45


def find_trials(path: Path):
    if not path.exists():
        return []
    r = subprocess.run(
        ["find", str(path), "-maxdepth", "6", "-type", "d", "-name", "trial_*"],
        capture_output=True, text=True, timeout=60,
    )
    return [l for l in r.stdout.strip().split("\n") if l]


def baseline_rate(suite_dir: Path, task: str):
    """Pass rate from trial dir names (reward_1.000 = pass) for baseline seeds 51-65."""
    run_dir = suite_dir / task / "gcp_google_gemini-3.1-pro-preview" / "run"
    trials = find_trials(run_dir)
    if not trials:
        return 0.0, 0
    # One result per seed: take highest sandboxrc per trial number
    best = {}
    for t in trials:
        name = Path(t).name  # trial_52_sandboxrc_1_reward_0.000_...
        parts = name.split("_")
        try:
            seed = int(parts[1])
            src  = int(parts[3])
            passed = "reward_1" in name
            if seed not in best or src > best[seed][0]:
                best[seed] = (src, passed)
        except (IndexError, ValueError):
            continue
    total = len(best)
    passes = sum(1 for _, (_, p) in best.items() if p)
    return (passes / total if total else 0.0), total


def fix_rate(fixed_suite_dir: Path, task: str):
    """Pass rate per unique seed (1-50) from the fixed output dir.

    For seeds with multiple runs, count the seed as passed if ANY run passed
    (consistent with gen_progress.py max(first,last) logic).
    Only count seeds 1-50.
    """
    import re
    out_dir = fixed_suite_dir / task
    trials = find_trials(out_dir)
    # Count unique seeds 1-50
    unique_seeds = set()
    trial_re = re.compile(r"trial_(\d+)_sandboxrc")
    for t in trials:
        m = trial_re.search(Path(t).name)
        if m:
            seed = int(m.group(1))
            if 1 <= seed <= 50:
                unique_seeds.add(seed)
    total = len(unique_seeds)
    if total < DONE_THRESHOLD:
        return None, total  # pending / incomplete
    # Find summaries only for seeds 1-50
    passed_seeds = set()
    for t in trials:
        m = trial_re.search(Path(t).name)
        if not m:
            continue
        seed = int(m.group(1))
        if not (1 <= seed <= 50):
            continue
        summary = Path(t) / "summary.txt"
        if summary.exists():
            text = summary.read_text(errors="ignore")
            if "Task Completed: True" in text:
                passed_seeds.add(seed)
    return len(passed_seeds) / total, total


# --- Build data ---
suite_data = {}
for label, suite_key in SUITES:
    suite_dir   = BASELINE / suite_key
    fixed_dir   = FIXED    / suite_key
    tasks = sorted(suite_dir.iterdir()) if suite_dir.exists() else []
    rows = []
    for task_path in tasks:
        task = task_path.name
        short = TASK_LABELS.get(task, task[:18])
        base_frac, base_n = baseline_rate(suite_dir, task)
        fix_frac,  fix_n  = fix_rate(fixed_dir, task)
        rows.append((short, base_frac, fix_frac))
    suite_data[label] = rows

# --- Plot ---
BASE_COLOR    = "#6baed6"
FIX_COLOR     = "#fd8d3c"
PENDING_COLOR = "#cccccc"

layout = [
    ["Goal Swap",   "Object Swap",   "Spatial Swap"],
    ["Goal Task",   "Object Task",   "Spatial Task"],
]

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Baseline vs Fix Loop — Success Rate by Task", fontsize=16, fontweight='bold', y=1.01)

for row in range(2):
    for col in range(3):
        ax = axes[row, col]
        suite_name = layout[row][col]
        rows = suite_data.get(suite_name, [])
        if not rows:
            ax.set_visible(False)
            continue

        labels     = [r[0] for r in rows]
        base_vals  = [r[1] * 100 for r in rows]
        fix_fracs  = [r[2] for r in rows]
        n = len(labels)
        x = np.arange(n)
        w = 0.38

        ax.bar(x - w/2, base_vals, width=w, color=BASE_COLOR, alpha=0.9,
               edgecolor='white', linewidth=0.6)

        for i, frac in enumerate(fix_fracs):
            xi = x[i] + w/2
            if frac is None:
                ax.bar(xi, 95, width=w, color=PENDING_COLOR, alpha=0.25,
                       edgecolor='#aaaaaa', linewidth=0.6, linestyle='--')
                ax.text(xi, 2, "…", ha='center', va='bottom', fontsize=9, color='#999999')
            else:
                pct = frac * 100
                ax.bar(xi, pct, width=w, color=FIX_COLOR, alpha=0.9,
                       edgecolor='white', linewidth=0.6)
                if pct > 5:
                    ax.text(xi, pct + 1.5, f"{pct:.0f}%", ha='center', va='bottom',
                            fontsize=7, fontweight='bold', color='#333333')

        for i, v in enumerate(base_vals):
            if v > 0:
                ax.text(x[i] - w/2, v + 1.5, f"{v:.0f}%", ha='center', va='bottom',
                        fontsize=7, fontweight='bold', color='#1a6094')

        fix_known = [f * 100 for f in fix_fracs if f is not None]
        if fix_known:
            fix_avg = np.mean(fix_known)
            ax.axhline(fix_avg, color=FIX_COLOR, linewidth=1.5, linestyle='--', alpha=0.85,
                       label=f"Fix avg {fix_avg:.1f}% ({len(fix_known)}/10)")
        base_avg = np.mean(base_vals)
        ax.axhline(base_avg, color=BASE_COLOR, linewidth=1.5, linestyle='--', alpha=0.85,
                   label=f"Base avg {base_avg:.1f}%")

        ax.set_title(suite_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Success Rate (%)", fontsize=9)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.grid(True, alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.7)

legend_patches = [
    mpatches.Patch(color=BASE_COLOR,    alpha=0.9, label="Baseline (seeds 51–65, /15)"),
    mpatches.Patch(color=FIX_COLOR,     alpha=0.9, label="Fix loop (seeds 1–50, /50)"),
    mpatches.Patch(color=PENDING_COLOR, alpha=0.4, label="Pending / in-progress"),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout()
out = ROOT / "outputs/plots/baseline_vs_fix_chart.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
