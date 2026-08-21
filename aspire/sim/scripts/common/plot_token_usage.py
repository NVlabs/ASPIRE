# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot Claude Code token usage over time from ~/.claude/projects JSONL files.

Produces two charts:
  outputs/plots/token_usage.png       — stacked bar: daily cost by model
  outputs/plots/token_usage_lines.png — line chart: daily token counts by type
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = Path.home() / ".claude/projects"
OUT       = ROOT / "outputs/plots/token_usage.png"
OUT_LINES = ROOT / "outputs/plots/token_usage_lines.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Anthropic pricing (per million tokens, as of April 2026)
# Using approximate blended rates matching /cost output
PRICING = {
    # Effective rates back-calculated from /cost output ($5665 total, Apr 12-23)
    # Opus cache_read effective rate is ~$0.20/M (vs $1.50/M list — likely internal pricing)
    "claude-opus-4-6":             {"input": 15.0, "output": 75.0, "cache_write": 18.75, "cache_read": 0.20},
    "claude-sonnet-4-6":           {"input": 3.0,  "output": 15.0, "cache_write": 3.75,  "cache_read": 0.30},
    "claude-haiku-4-5-20251001":   {"input": 0.8,  "output": 4.0,  "cache_write": 1.0,   "cache_read": 0.08},
}

MODEL_COLORS = {
    "claude-opus-4-6":           "#9b59b6",   # purple
    "claude-sonnet-4-6":         "#2196f3",   # blue
    "claude-haiku-4-5-20251001": "#26a69a",   # teal
}
MODEL_LABELS = {
    "claude-opus-4-6":           "Opus 4.6",
    "claude-sonnet-4-6":         "Sonnet 4.6",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
}

# day -> model -> {input, output, cache_write, cache_read}
daily = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

n_files = 0
n_messages = 0
n_deduped = 0
seen_requests: set[str] = set()

for f in sorted(PROJECTS_DIR.rglob("*.jsonl")):
    n_files += 1
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
            if not usage:
                continue
            model = msg.get("model", "unknown")
            if model not in PRICING:
                continue
            ts = d.get("timestamp", "")
            if not ts:
                continue
            # Deduplicate: sidechain messages appear in both the parent JSONL
            # and the subagent's own JSONL file. requestId identifies unique API calls.
            req_id = d.get("requestId", "")
            if req_id:
                if req_id in seen_requests:
                    n_deduped += 1
                    continue
                seen_requests.add(req_id)
            day = ts[:10]
            u = daily[day][model]
            u["input"]       += usage.get("input_tokens", 0)
            u["output"]      += usage.get("output_tokens", 0)
            u["cache_write"] += usage.get("cache_creation_input_tokens", 0)
            u["cache_read"]  += usage.get("cache_read_input_tokens", 0)
            n_messages += 1

print(f"Parsed {n_files} files, {n_messages} unique messages ({n_deduped} deduplicated)")

days = sorted(daily.keys())
models = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

# Compute daily cost per model
def cost(model, u):
    p = PRICING[model]
    return (
        u["input"]       / 1e6 * p["input"] +
        u["output"]      / 1e6 * p["output"] +
        u["cache_write"] / 1e6 * p["cache_write"] +
        u["cache_read"]  / 1e6 * p["cache_read"]
    )

daily_costs = {m: [cost(m, daily[d][m]) for d in days] for m in models}
daily_output = {m: [daily[d][m]["output"] / 1e6 for d in days] for m in models}
daily_input  = {m: [(daily[d][m]["input"] + daily[d][m]["cache_write"]) / 1e6 for d in days] for m in models}
daily_cache  = {m: [daily[d][m]["cache_read"] / 1e6 for d in days] for m in models}

x = np.arange(len(days))
total_cost = sum(sum(v) for v in daily_costs.values())

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle(f"Claude Code Usage — Total Cost: ${total_cost:,.0f}", fontsize=14, fontweight="bold")

# Top: daily cost stacked bar
ax = axes[0]
bottom = np.zeros(len(days))
for m in models:
    vals = np.array(daily_costs[m])
    ax.bar(x, vals, bottom=bottom, color=MODEL_COLORS[m], alpha=0.88, label=MODEL_LABELS[m], zorder=3)
    bottom += vals
ax.set_ylabel("Cost (USD)", fontsize=11)
ax.yaxis.grid(True, alpha=0.3, linestyle=":", zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
# Annotate total per day
for i, b in enumerate(bottom):
    if b >= 50:
        ax.text(i, b + 5, f"${b:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333")

# Bottom: output tokens stacked bar (millions)
ax2 = axes[1]
bottom2 = np.zeros(len(days))
for m in models:
    vals = np.array(daily_output[m])
    ax2.bar(x, vals, bottom=bottom2, color=MODEL_COLORS[m], alpha=0.88, zorder=3)
    bottom2 += vals
ax2.set_ylabel("Output Tokens (M)", fontsize=11)
ax2.yaxis.grid(True, alpha=0.3, linestyle=":", zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# x-axis labels
short_days = [d[5:] for d in days]  # MM-DD
ax2.set_xticks(x)
ax2.set_xticklabels(short_days, rotation=45, ha="right", fontsize=9)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUT}")

# Print text summary
print(f"\n{'Date':<12} {'Opus':>8} {'Sonnet':>8} {'Haiku':>8} {'Total':>8}")
print("-" * 48)
for i, day in enumerate(days):
    costs = {m: daily_costs[m][i] for m in models}
    total = sum(costs.values())
    print(f"{day:<12} ${costs['claude-opus-4-6']:>6.0f}  ${costs['claude-sonnet-4-6']:>6.0f}  ${costs['claude-haiku-4-5-20251001']:>6.0f}  ${total:>6.0f}")
print("-" * 48)
totals = {m: sum(daily_costs[m]) for m in models}
print(f"{'TOTAL':<12} ${totals['claude-opus-4-6']:>6.0f}  ${totals['claude-sonnet-4-6']:>6.0f}  ${totals['claude-haiku-4-5-20251001']:>6.0f}  ${sum(totals.values()):>6.0f}")

# ── Line chart: token counts by type ────────────────────────────────────────

# Aggregate across models per day
def day_totals(key):
    return [sum(daily[d][m][key] for m in models) for d in days]

tok_output      = np.array(day_totals("output"))
tok_input       = np.array(day_totals("input"))
tok_cache_write = np.array(day_totals("cache_write"))
tok_cache_read  = np.array(day_totals("cache_read"))

# Also per-model output for the lower panel
model_output = {m: np.array([daily[d][m]["output"] for d in days]) for m in models}

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig2.suptitle("Claude Code Token Usage — All models + subagents included", fontsize=13, fontweight="bold")

# ── Top panel: all token types, dual-axis (cache_read dwarfs others) ────────
ax_top = axes2[0]
ax_top_r = ax_top.twinx()

line_kw = dict(marker="o", markersize=4, linewidth=1.8)
l1, = ax_top.plot(x, tok_output      / 1e6, color="#e74c3c", label="Output",       **line_kw)
l2, = ax_top.plot(x, tok_input       / 1e6, color="#3498db", label="Input (fresh)", **line_kw)
l3, = ax_top.plot(x, tok_cache_write / 1e6, color="#f39c12", label="Cache write",  **line_kw)
l4, = ax_top_r.plot(x, tok_cache_read / 1e9, color="#9b59b6", label="Cache read (B)",
                    linestyle="--", **line_kw)

ax_top.set_ylabel("Tokens (M)", fontsize=11)
ax_top_r.set_ylabel("Cache Read (B tokens)", fontsize=11, color="#9b59b6")
ax_top_r.tick_params(axis="y", labelcolor="#9b59b6")
ax_top.yaxis.grid(True, alpha=0.25, linestyle=":", zorder=0)
ax_top.set_axisbelow(True)
ax_top.spines["top"].set_visible(False)
lines = [l1, l2, l3, l4]
ax_top.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper left", framealpha=0.85)

# ── Bottom panel: output tokens broken by model ──────────────────────────────
ax_bot = axes2[1]
for m in models:
    ax_bot.plot(x, model_output[m] / 1e6, color=MODEL_COLORS[m],
                label=MODEL_LABELS[m], **line_kw)
ax_bot.set_ylabel("Output Tokens (M)", fontsize=11)
ax_bot.yaxis.grid(True, alpha=0.25, linestyle=":", zorder=0)
ax_bot.set_axisbelow(True)
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)
ax_bot.legend(fontsize=9, loc="upper left", framealpha=0.85)

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(short_days, rotation=45, ha="right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_LINES, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_LINES}")
