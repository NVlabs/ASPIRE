"""Two token-usage plots.

1. outputs/plots/token_usage_lines2.png  — daily totals, all models merged, 4 token-type lines
2. outputs/plots/subagent_token_trace.png — per-turn token trace for 3 representative Evolutionary Search subagents
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = Path.home() / ".claude/projects"

KNOWN_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"}

# ── 1. Load daily token totals (deduped, all models merged) ─────────────────

daily = defaultdict(lambda: defaultdict(int))
seen = set()

for f in sorted(PROJECTS_DIR.rglob("*.jsonl")):
    with open(f, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "assistant":
                continue
            msg = d.get("message", {})
            usage = msg.get("usage")
            if not usage or msg.get("model", "") not in KNOWN_MODELS:
                continue
            req_id = d.get("requestId", "")
            if req_id:
                if req_id in seen:
                    continue
                seen.add(req_id)
            day = d.get("timestamp", "")[:10]
            if not day:
                continue
            daily[day]["output"]      += usage.get("output_tokens", 0)
            daily[day]["input"]       += usage.get("input_tokens", 0)
            daily[day]["cache_write"] += usage.get("cache_creation_input_tokens", 0)
            daily[day]["cache_read"]  += usage.get("cache_read_input_tokens", 0)

days = sorted(daily.keys())
x = np.arange(len(days))
short_days = [d[5:] for d in days]

tok_output      = np.array([daily[d]["output"]      / 1e6 for d in days])
tok_input       = np.array([daily[d]["input"]        / 1e6 for d in days])
tok_cache_write = np.array([daily[d]["cache_write"]  / 1e6 for d in days])
tok_cache_read  = np.array([daily[d]["cache_read"]   / 1e9 for d in days])  # billions

# ── Plot 1: daily token lines ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
ax_r = ax.twinx()

kw = dict(marker="o", markersize=5, linewidth=2.0)
ax.plot(x, tok_output,      color="#e74c3c", label="Output (M)",       **kw)
ax.plot(x, tok_input,       color="#3498db", label="Input fresh (M)",  **kw)
ax.plot(x, tok_cache_write, color="#f39c12", label="Cache write (M)",  **kw)
ax_r.plot(x, tok_cache_read, color="#9b59b6", label="Cache read (B)",
          linestyle="--", **kw)

ax.set_ylabel("Tokens (M)", fontsize=11)
ax_r.set_ylabel("Cache Read (B tokens)", fontsize=11, color="#9b59b6")
ax_r.tick_params(axis="y", labelcolor="#9b59b6")
ax.yaxis.grid(True, alpha=0.25, linestyle=":", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(short_days, rotation=45, ha="right", fontsize=9)
ax.set_title("Claude Code — Daily Token Usage (all models + subagents)", fontsize=12, fontweight="bold")

lines_left  = ax.get_lines()
lines_right = ax_r.get_lines()
all_lines   = lines_left + lines_right
ax.legend(all_lines, [l.get_label() for l in all_lines],
          fontsize=9, loc="upper left", framealpha=0.85)

plt.tight_layout()
out1 = ROOT / "outputs/plots/token_usage_lines2.png"
out1.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")
plt.close()

# ── 2. Load 3 representative Evolutionary Search subagent traces ──────────────────────────

AGENTS = [
    {
        "path": PROJECTS_DIR / "-ASPIRE/8735de5d-2775-4614-adfd-bf666b806850/subagents/agent-a6bdbd637f1b18f92.jsonl",
        "label": "spatial_task / bowl_on_cookie_box",
        "color": "#e74c3c",
    },
    {
        "path": PROJECTS_DIR / "-ASPIRE/8735de5d-2775-4614-adfd-bf666b806850/subagents/agent-a1b49fb69af5fa253.jsonl",
        "label": "goal_swap / open_top_drawer",
        "color": "#2196f3",
    },
    {
        "path": PROJECTS_DIR / "-ASPIRE/8735de5d-2775-4614-adfd-bf666b806850/subagents/agent-ac69c87489dbeff49.jsonl",
        "label": "spatial_swap / bowl_between_ramekin",
        "color": "#43a047",
    },
]

def load_trace(path):
    """Return list of per-turn dicts with cumulative token totals."""
    turns = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "assistant":
                continue
            msg = d.get("message", {})
            usage = msg.get("usage")
            if not usage:
                continue
            turns.append({
                "output":      usage.get("output_tokens", 0),
                "input":       usage.get("input_tokens", 0),
                "cache_write": usage.get("cache_creation_input_tokens", 0),
                "cache_read":  usage.get("cache_read_input_tokens", 0),
            })
    return turns

fig2, axes2 = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
fig2.suptitle("Evolutionary Search Debug Subagent — Token Usage Per Turn", fontsize=13, fontweight="bold")

for ax2, agent in zip(axes2, AGENTS):
    turns = load_trace(agent["path"])
    n = len(turns)
    xs = np.arange(n)

    output      = np.array([t["output"]      / 1e3 for t in turns])
    cache_read  = np.array([t["cache_read"]  / 1e3 for t in turns])
    cache_write = np.array([t["cache_write"] / 1e3 for t in turns])

    ax2_r = ax2.twinx()
    ax2.fill_between(xs, cache_read,  alpha=0.18, color="#9b59b6", label="Cache read (k)")
    ax2.plot(xs, cache_read,  color="#9b59b6", linewidth=1.0, alpha=0.7)
    ax2.fill_between(xs, cache_write, alpha=0.30, color="#f39c12", label="Cache write (k)")
    ax2.plot(xs, cache_write, color="#f39c12", linewidth=1.0)
    ax2_r.plot(xs, output, color="#e74c3c", linewidth=1.2, label="Output (k)", alpha=0.9)

    ax2.set_ylabel("Tokens (k)", fontsize=9)
    ax2_r.set_ylabel("Output (k)", fontsize=9, color="#e74c3c")
    ax2_r.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_title(agent["label"], fontsize=10, fontweight="bold", color=agent["color"])
    ax2.yaxis.grid(True, alpha=0.2, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel(f"Turn (total {n})", fontsize=9)

    handles = [
        mpatches.Patch(color="#9b59b6", alpha=0.5, label="Cache read (k)"),
        mpatches.Patch(color="#f39c12", alpha=0.5, label="Cache write (k)"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=1.5, label="Output (k)"),
    ]
    ax2.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.8)

plt.tight_layout()
out2 = ROOT / "outputs/plots/subagent_token_trace.png"
out2.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved {out2}")
