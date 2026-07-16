import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Per-suite summary. fix_results = list of verified scores (done/blocked tasks only).
# Pending/unverified tasks are excluded from the fix average; n_total=10 for all suites.
suites = [
    {
        "name": "Goal\nSwap",
        "baseline": [0, 0, 0, 8, 5, 7, 1, 2, 10, 6],   # /15
        # done(7): 1,18,30,37,47,23,16 + blocked(1): 0  | pending: 2
        "fix": [1, 0, 18, 30, 37, 47, 23, 16],
        "n_verified": 8,
    },
    {
        "name": "Goal\nTask",
        "baseline": [0, 0, 2, 7, 0, 0, 5, 1, 6, 6],
        # done(5): 27,14,32,21,10 + blocked(2): 0,0  | pending: 3
        "fix": [27, 0, 0, 14, 32, 21, 10],
        "n_verified": 7,
    },
    {
        "name": "Object\nSwap",
        "baseline": [2, 2, 8, 3, 1, 6, 1, 4, 3, 4],
        # done(9): 50,50,50,50,50,46,50,50,50  | pending: 1 (bbq_sauce unverified)
        "fix": [50, 50, 50, 50, 50, 46, 50, 50, 50],
        "n_verified": 9,
    },
    {
        "name": "Object\nTask",
        "baseline": [6, 3, 4, 2, 6, 0, 0, 2, 9, 1],
        # done(7): 50,49,50,36,48,50,50  | pending: 3
        "fix": [50, 49, 50, 36, 48, 50, 50],
        "n_verified": 7,
    },
    {
        "name": "Spatial\nSwap",
        "baseline": [3, 4, 0, 1, 1, 0, 1, 0, 2, 0],
        "fix": [],
        "n_verified": 0,
    },
    {
        "name": "Spatial\nTask",
        "baseline": [3, 6, 0, 1, 2, 0, 1, 3, 1, 4],
        "fix": [],
        "n_verified": 0,
    },
]

N_TOTAL = 10
BASE_COLOR   = "#4C72B0"
FIX_COLOR    = "#DD8452"
PENDING_COLOR = "#cccccc"

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Fix Loop vs Baseline — Suite-Level Average", fontsize=14, fontweight='bold')

x = np.arange(len(suites))
w = 0.32

for i, s in enumerate(suites):
    base_avg = np.mean(s["baseline"]) / 15 * 100
    ax.bar(x[i] - w/2, base_avg, width=w, color=BASE_COLOR, alpha=0.9,
           edgecolor='white', linewidth=0.8)
    if base_avg > 1:
        ax.text(x[i] - w/2, base_avg + 1, f"{base_avg:.0f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=BASE_COLOR)

    if s["n_verified"] == 0:
        ax.bar(x[i] + w/2, 95, width=w, color=PENDING_COLOR, alpha=0.3,
               edgecolor='#aaaaaa', linewidth=0.8, linestyle='--')
        ax.text(x[i] + w/2, 4, "pending", ha='center', va='bottom',
                fontsize=8, color='#888888', style='italic')
    else:
        fix_avg = np.mean(s["fix"]) / 50 * 100
        ax.bar(x[i] + w/2, fix_avg, width=w, color=FIX_COLOR, alpha=0.9,
               edgecolor='white', linewidth=0.8)
        ax.text(x[i] + w/2, fix_avg + 1, f"{fix_avg:.0f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=FIX_COLOR)
        ax.text(x[i] + w/2, -8, f"{s['n_verified']}/{N_TOTAL}",
                ha='center', va='top', fontsize=7.5, color='#555555')

ax.set_xticks(x)
ax.set_xticklabels([s["name"] for s in suites], fontsize=10)
ax.set_ylim(-12, 108)
ax.set_ylabel("Average Success Rate (%)", fontsize=10)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True, alpha=0.3, linestyle=':')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color=BASE_COLOR, alpha=0.9, label="Baseline (seeds 51–65, /15)"),
    mpatches.Patch(color=FIX_COLOR,  alpha=0.9, label="Fix loop (seeds 1–50, /50)"),
    mpatches.Patch(color=PENDING_COLOR, alpha=0.4, label="Pending"),
]
ax.legend(handles=legend_patches, fontsize=9, loc='upper right', framealpha=0.8)
ax.text(0.5, -0.10, "Fix avg is over verified tasks only (n/10 shown below bar)",
        transform=ax.transAxes, ha='center', fontsize=8, color='#555555')

plt.tight_layout()
out = Path("outputs/plots/suite_summary_chart.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved to {out}")
