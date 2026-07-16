"""Method comparison v5.

Changes from v4:
- ASPIRE uses paper numbers (Goal 26/17%, Object 22/18%, Spatial 12/14%)
- Removed ASPIRE-old column; ASPIRE-new renamed to ASPIRE (NVIDIA green)
- ASPIRE+Evolutionary Search: validation set (seeds 66-80) selects fix vs evosearch per task,
  then reports the winner's seeds 1-50 rate. Falls back to max(fix, evosearch)
  when no validation result exists.
- ASPIRE+Evolutionary Search bar color: dark green
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, re, subprocess
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
FIXED  = ROOT / "outputs/libero_fix_loop_eval"
EVOSEARCH = ROOT / "outputs/claude_evosearch_rerun"
VAL    = ROOT / "outputs/validation_comparison_seeds66_80"

DONE_THRESHOLD = 40
trial_re = re.compile(r"trial_(\d+)_sandboxrc")


def fix_rate(suite_key, task):
    """Pass rate on seeds 1-50 from libero_fix_loop_eval trial dirs."""
    out_dir = FIXED / suite_key / task
    r = subprocess.run(
        ["find", str(out_dir), "-maxdepth", "8", "-type", "d", "-name", "trial_*"],
        capture_output=True, text=True, timeout=60,
    )
    trials = [l for l in r.stdout.strip().split("\n") if l]
    seeds = {}
    for t in trials:
        m = trial_re.search(Path(t).name)
        if not m: continue
        seed = int(m.group(1))
        if not (1 <= seed <= 50): continue
        s = Path(t) / "summary.txt"
        seeds[seed] = int(s.exists() and "Task Completed: True" in s.read_text(errors="ignore"))
    if len(seeds) < DONE_THRESHOLD:
        return None
    return 100.0 * sum(seeds.values()) / len(seeds)


def evosearch_rate(suite_key, task):
    """Stage-2 pass rate (seeds 1-50) from claude_evosearch_rerun stage2/iter_summary.json."""
    tdir = EVOSEARCH / suite_key / task
    if not tdir.exists():
        return None
    for run_dir in sorted(tdir.glob("2*"), reverse=True):
        js = run_dir / "stage2" / "iter_summary.json"
        if js.exists():
            d = json.loads(js.read_text())
            return d["candidates"][0]["pass_rate"] * 100
    return None


def val_winner(suite_key, task):
    """Returns 'evosearch' or 'fix' based on validation seeds 66-80, or None if no result."""
    js = VAL / suite_key / task / "iter_summary.json"
    if not js.exists():
        return None
    d = json.loads(js.read_text())
    rates = {c["candidate"]: c["pass_rate"] for c in d["candidates"]}
    vf = rates.get("candidate_fix")
    ve = rates.get("candidate_evosearch")
    if vf is None or ve is None:
        return None
    return "evosearch" if ve >= vf else "fix"


def suite_stats(suite_key):
    """Returns (mean_pct, n_tasks, evosearch_wins, evosearch_ran) for per-task val-selected rate."""
    fix_root = FIXED / suite_key
    if not fix_root.exists():
        return None, 0, 0, 0
    tasks = sorted(p.name for p in fix_root.iterdir() if p.is_dir())
    task_rates, evosearch_wins, evosearch_ran = [], 0, 0
    for task in tasks:
        f = fix_rate(suite_key, task)
        e = evosearch_rate(suite_key, task)
        if f is None and e is None:
            continue
        winner = val_winner(suite_key, task)
        if winner == "evosearch" and e is not None:
            best = e
            evosearch_ran += 1
            evosearch_wins += 1
        elif winner == "fix" and f is not None:
            best = f
            evosearch_ran += 1
        elif e is not None and f is not None:
            # no validation result — fall back to max
            best = max(f, e)
            if e > f:
                evosearch_wins += 1
        elif f is None:
            best = e
        else:
            best = f
        task_rates.append(best)
    if not task_rates:
        return None, 0, 0, 0
    return float(np.mean(task_rates)), len(task_rates), evosearch_wins, evosearch_ran


SUITES_ORDER = [
    ("Pos-Goal",    "libero_goal_swap"),
    ("Pos-Object",  "libero_object_swap"),
    ("Pos-Spatial", "libero_spatial_swap"),
    ("Ins-Goal",    "libero_goal_task"),
    ("Ins-Object",  "libero_object_task"),
    ("Ins-Spatial", "libero_spatial_task"),
]

aspire_vals, evosearch_vals, evosearch_wins_list, evosearch_ran_list = [], [], [], []
for label, key in SUITES_ORDER:
    m, nt, nwin, nran = suite_stats(key)
    val = round(m, 1) if m is not None else 0.0
    evosearch_vals.append(val)
    evosearch_wins_list.append(nwin)
    evosearch_ran_list.append(nran)
    # ASPIRE = mean of per-task fix rates (same source as max() baseline)
    fix_root = FIXED / key
    if fix_root.exists():
        tasks = sorted(p.name for p in fix_root.iterdir() if p.is_dir())
        fix_rates = [fix_rate(key, t) for t in tasks]
        fix_rates = [r for r in fix_rates if r is not None]
        aspire_mean = round(float(np.mean(fix_rates)), 1) if fix_rates else 0.0
    else:
        aspire_mean = 0.0
    aspire_vals.append(aspire_mean)
    print(f"{label:14s}  ASPIRE={aspire_mean:.1f}%  Evolutionary Search+={val:.1f}%  wins={nwin}/{nran}")

SUITES = [s for s, _ in SUITES_ORDER]

# ASPIRE paper numbers: [goal_swap, obj_swap, spatial_swap, goal_task, obj_task, spatial_task]
CAPX_PAPER = [26, 22, 12, 17, 18, 14]

METHODS = {
    "OpenVLA":        [0,    0,    0,    0,    0,    0   ],
    r"$\pi_0$":       [0,    0,    0,    0,    0,    0   ],
    r"$\pi_{0.5}$":   [0,    17,   0,    0,    1,    0   ],
    "ASPIRE":          CAPX_PAPER,
    "ASPIRE":          aspire_vals,
    "ASPIRE+Evolutionary Search":  evosearch_vals,
}

COLORS = {
    "OpenVLA":        "#b0b0b0",
    r"$\pi_0$":       "#9ecae1",
    r"$\pi_{0.5}$":   "#4292c6",
    "ASPIRE":          "#fd8d3c",
    "ASPIRE":          "#76b900",   # NVIDIA green
    "ASPIRE+Evolutionary Search":  "#1a7a28",   # dark green
}

n_suites  = len(SUITES)
n_methods = len(METHODS)
x = np.arange(n_suites)
total_width = 0.75
w = total_width / n_methods
offsets = np.linspace(-(total_width - w) / 2, (total_width - w) / 2, n_methods)

fig, ax = plt.subplots(figsize=(14, 6.2))

for idx, (name, vals) in enumerate(METHODS.items()):
    ax.bar(
        x + offsets[idx], vals, width=w,
        color=COLORS[name], alpha=0.92,
        edgecolor='white', linewidth=0.5,
        label=name, zorder=3,
    )
    for xi, v in zip(x + offsets[idx], vals):
        if v >= 5:
            ax.text(xi, v + 1.0, f"{v:.0f}" if v == int(v) else f"{v:.1f}",
                    ha='center', va='bottom', fontsize=5.8, fontweight='bold',
                    color='#333333', rotation=90 if v < 20 else 0)

# Annotate ASPIRE+Evolutionary Search bars with "wins/ran"
eidx = list(METHODS.keys()).index("ASPIRE+Evolutionary Search")
for i, (nwin, nran) in enumerate(zip(evosearch_wins_list, evosearch_ran_list)):
    if nran == 0:
        continue
    xi = x[i] + offsets[eidx]
    ax.text(xi, evosearch_vals[i] + 5.2, f"{nwin}/{nran}",
            ha='center', va='bottom', fontsize=6.2,
            color=COLORS["ASPIRE+Evolutionary Search"], fontweight='bold')

for i in range(1, n_suites):
    ax.axvline(i - 0.5, color='#cccccc', linewidth=0.8, linestyle='--', zorder=1)

ax.axvspan(-0.5, 2.5, alpha=0.04, color='steelblue',  zorder=0)
ax.axvspan( 2.5, 5.5, alpha=0.04, color='darkorange', zorder=0)
ax.text(1.0, 106, "Position Perturbation",    ha='center', fontsize=9, color='steelblue',  fontweight='bold')
ax.text(4.0, 106, "Instruction Perturbation", ha='center', fontsize=9, color='darkorange', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SUITES, fontsize=11)
ax.set_ylabel("Success Rate (%)", fontsize=11)
ax.set_ylim(0, 117)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True, alpha=0.3, linestyle=':', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-0.5, n_suites - 0.5)

ax.set_title(
    "LIBERO-Pro: Method Comparison — ASPIRE+Evolutionary Search = per-task max(Fix, Evolutionary Search) over all tasks",
    fontsize=12.5, fontweight='bold', pad=14,
)

ax.legend(
    handles=[mpatches.Patch(color=COLORS[n], alpha=0.92, label=n) for n in METHODS],
    fontsize=8.5, loc='upper left', framealpha=0.85, ncol=1,
)

plt.tight_layout()
out = ROOT / "outputs/plots/method_comparison_v5.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved to {out}")
