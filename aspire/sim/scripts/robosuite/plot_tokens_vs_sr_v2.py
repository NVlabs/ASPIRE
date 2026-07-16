#!/usr/bin/env python3
"""Plot per-task and cumulative tokens/iter vs success rate (iters 1–25 only).

Correctly accounts for multiple subagent dispatches per task by merging
usage events from ALL matching JSONLs (not just the largest).

Outputs:
  outputs/plots/per_task_iter_vs_sr.png        — 7 subplots: iter 0–25 vs SR
  outputs/plots/per_task_tokens_vs_sr.png      — 7 subplots: cumulative tokens vs SR
  outputs/plots/cumulative_iter_vs_sr.png      — averaged iter vs SR across 7 tasks
  outputs/plots/cumulative_tokens_vs_sr.png    — averaged tokens vs SR across 7 tasks
  outputs/plots/cumulative_tokens_norm_vs_sr.png — averaged normalized tokens vs SR

Usage:
    python3 scripts/robosuite/plot_tokens_vs_sr_v2.py [--project-dir PATH] [--print-table]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAX_ITER = 25

BASELINE_DIR = Path("outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel")

TASK_MAP = {
    "cube_lifting":     "cube_lifting_multimodel_aspire_traced",
    "cube_restack":     "cube_restack_multimodel_aspire_traced",
    "cube_stack":       "cube_stack_multimodel_aspire_traced",
    "nut_assembly":     "nut_assembly_multimodel_aspire_traced",
    "spill_wipe":       "spill_wipe_multimodel_aspire_traced",
    "two_arm_lift":     "two_arm_lift_multimodel_aspire_traced",
    "two_arm_handover": "two_arm_handover_multimodel_aspire_traced",
}


def find_project_dir() -> Path:
    cwd = Path(os.getcwd()).resolve()
    sanitized = str(cwd).replace("/", "-")
    candidate = Path.home() / ".claude" / "projects" / sanitized
    if candidate.exists():
        return candidate
    sanitized_alt = str(cwd).lstrip("/").replace("/", "-")
    candidate_alt = Path.home() / ".claude" / "projects" / sanitized_alt
    if candidate_alt.exists():
        return candidate_alt
    base = Path.home() / ".claude" / "projects"
    if base.exists():
        for d in base.iterdir():
            if d.name.endswith(sanitized_alt):
                return d
    return candidate


def local_tz() -> timezone:
    from datetime import timedelta
    local_offset = datetime.now(timezone.utc).astimezone().utcoffset()
    return timezone(local_offset)


def find_all_subagent_jsonls(project_dir: Path, config_stem: str, task_name: str) -> list[Path]:
    candidates = list(project_dir.rglob("subagents/*.jsonl"))
    dispatch_pattern = "TASK:"
    matches = []
    for jsonl_path in candidates:
        try:
            text = jsonl_path.read_text(errors="replace")
            if dispatch_pattern not in text or config_stem not in text:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "user":
                    msg_content = str(d.get("message", ""))
                    if "TASK:" in msg_content and task_name in msg_content and config_stem in msg_content:
                        # Verify it's the actual assignment, not template options
                        if re.search(rf"TASK:\s+{re.escape(task_name)}\b", msg_content):
                            matches.append(jsonl_path)
                    break
        except OSError:
            continue
    return matches


def extract_merged_usage_events(jsonl_paths: list[Path]) -> list[tuple[datetime, dict]]:
    events = []
    seen_request_ids: set[str] = set()
    for jsonl_path in jsonl_paths:
        try:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("type") != "assistant":
                        continue
                    req_id = d.get("requestId", "")
                    if req_id:
                        if req_id in seen_request_ids:
                            continue
                        seen_request_ids.add(req_id)
                    msg = d.get("message", {})
                    if isinstance(msg, dict) and "usage" in msg:
                        ts_str = d.get("timestamp", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            events.append((ts, msg["usage"]))
        except OSError:
            continue
    events.sort(key=lambda e: e[0])
    return events


def parse_iter_times(code_versions_dir: Path, tz: timezone) -> dict[int, datetime]:
    iter_times = {}
    for f in sorted(code_versions_dir.glob("iter_*.py")):
        m = re.match(r"iter_(\d+)_(\d{8}_\d{6})\.py", f.name)
        if m:
            it = int(m.group(1))
            if it > MAX_ITER:
                continue
            local_dt = datetime.strptime(m.group(2), "%Y%m%d_%H%M%S").replace(tzinfo=tz)
            if it not in iter_times:
                iter_times[it] = local_dt.astimezone(timezone.utc)
    return iter_times


def bucket_tokens_per_iter(events, iter_times):
    sorted_iters = sorted(iter_times.items())
    if not sorted_iters:
        return {}

    max_iter_ts = sorted_iters[-1][1]

    def get_bucket(event_ts):
        for it, code_ts in sorted_iters:
            if event_ts <= code_ts:
                return it
        # Event after last iter — attribute to last iter (within cap)
        return sorted_iters[-1][0]

    buckets = defaultdict(lambda: {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0})
    for ts, u in events:
        # Skip events that are clearly after iter 25's code was written
        # (they belong to iter 26+ from a later dispatch)
        if sorted_iters and ts > max_iter_ts:
            continue
        b = get_bucket(ts)
        if b > MAX_ITER:
            continue
        buckets[b]["input"] += u.get("input_tokens", 0)
        buckets[b]["output"] += u.get("output_tokens", 0)
        buckets[b]["cache_create"] += u.get("cache_creation_input_tokens", 0)
        buckets[b]["cache_read"] += u.get("cache_read_input_tokens", 0)
    return dict(buckets)


def load_results(code_versions_dir: Path) -> dict[int, dict]:
    results = {}
    for f in code_versions_dir.glob("iter_*_result.json"):
        try:
            r = json.loads(f.read_text())
            if r["iter"] <= MAX_ITER:
                results[r["iter"]] = r
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return results


def compute_task_data(task_name: str, config_stem: str, project_dir: Path, tz: timezone):
    task_dir = BASELINE_DIR / config_stem
    cv_dir = task_dir / "code_versions"

    if not cv_dir.exists():
        return None, f"no code_versions/ dir"

    results = load_results(cv_dir)
    if not results:
        return None, f"no result.json files"

    jsonl_paths = find_all_subagent_jsonls(project_dir, config_stem, task_name)
    if not jsonl_paths:
        return None, f"no subagent JSONL found"

    iter_times = parse_iter_times(cv_dir, tz)
    if not iter_times:
        return None, f"no timestamped code files"

    events = extract_merged_usage_events(jsonl_paths)
    if not events:
        return None, f"no usage events"

    token_buckets = bucket_tokens_per_iter(events, iter_times)

    all_iters = sorted(set(token_buckets.keys()) & set(results.keys()))
    if not all_iters:
        return None, f"no iterations with both tokens and results"

    cumulative_tokens = 0
    curve = []
    for it in all_iters:
        tb = token_buckets[it]
        total = tb["input"] + tb["output"] + tb["cache_create"] + tb["cache_read"]
        cumulative_tokens += total
        r = results[it]
        curve.append({
            "iter": it,
            "cumulative_tokens": cumulative_tokens,
            "tokens_this_iter": total,
            "pass_rate": r["pass_rate"],
            "n_pass": r["n_pass"],
            "n_total": r["n_total"],
        })

    return {
        "curve": curve,
        "n_dispatches": len(jsonl_paths),
        "total_tokens": cumulative_tokens,
    }, None


def print_table(all_data):
    for task_name, data in sorted(all_data.items()):
        curve = data["curve"]
        print(f"\n{'='*70}")
        print(f"Task: {task_name} ({data['n_dispatches']} dispatches, {data['total_tokens']:,} total tokens)")
        print(f"{'Iter':>5} {'Tokens (iter)':>14} {'Tokens (cum)':>14} {'SR':>8}")
        print(f"{'-'*5} {'-'*14} {'-'*14} {'-'*8}")
        for pt in curve:
            sr = f"{pt['n_pass']}/{pt['n_total']}"
            print(f"{pt['iter']:>5} {pt['tokens_this_iter']:>14,} {pt['cumulative_tokens']:>14,} {sr:>8}")


def plot_per_task_iter_vs_sr(all_data, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(all_data.keys())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, task_name in enumerate(tasks):
        ax = axes[idx]
        curve = all_data[task_name]["curve"]
        iters = [0] + [pt["iter"] for pt in curve]
        srs = [0.0] + [pt["pass_rate"] * 100 for pt in curve]
        ax.plot(iters, srs, marker="o", markersize=4, linewidth=1.5, color="C0")
        ax.set_xlim(-0.5, MAX_ITER + 0.5)
        ax.set_ylim(-5, 105)
        ax.set_title(task_name, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("SR (%)")
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(tasks) < len(axes):
        for i in range(len(tasks), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle("Per-Task: Iteration vs Success Rate (iters 0–25)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_per_task_tokens_vs_sr(all_data, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(all_data.keys())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, task_name in enumerate(tasks):
        ax = axes[idx]
        curve = all_data[task_name]["curve"]
        tokens = [0] + [pt["cumulative_tokens"] for pt in curve]
        srs = [0.0] + [pt["pass_rate"] * 100 for pt in curve]
        ax.plot(tokens, srs, marker="o", markersize=4, linewidth=1.5, color="C1")
        ax.set_ylim(-5, 105)
        ax.set_title(task_name, fontsize=11)
        ax.set_xlabel("Cumulative Tokens")
        ax.set_ylabel("SR (%)")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))

    if len(tasks) < len(axes):
        for i in range(len(tasks), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle("Per-Task: Cumulative Tokens vs Success Rate (iters 1–25)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_cumulative_iter_vs_sr(all_data, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters_range = np.arange(0, MAX_ITER + 1)
    task_srs = {}

    for task_name, data in all_data.items():
        curve = data["curve"]
        sr_by_iter = {0: 0.0}
        for pt in curve:
            sr_by_iter[pt["iter"]] = pt["pass_rate"] * 100

        srs = []
        last_sr = 0.0
        for i in iters_range:
            if i in sr_by_iter:
                last_sr = sr_by_iter[i]
            srs.append(last_sr)
        task_srs[task_name] = np.array(srs)

    all_srs = np.array(list(task_srs.values()))
    mean_sr = all_srs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for task_name, srs in sorted(task_srs.items()):
        ax.plot(iters_range, srs, linewidth=0.8, alpha=0.4, label=task_name)
    ax.plot(iters_range, mean_sr, linewidth=2.5, color="black", label="Mean (7 tasks)")
    ax.set_xlim(-0.5, MAX_ITER + 0.5)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Cumulative: Iteration vs Success Rate (avg across 7 tasks)", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_cumulative_tokens_vs_sr(all_data, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_tokens = max(d["total_tokens"] for d in all_data.values())
    grid = np.linspace(0, max_tokens, 200)

    task_srs = {}
    for task_name, data in all_data.items():
        curve = data["curve"]
        tok_points = [0] + [pt["cumulative_tokens"] for pt in curve]
        sr_points = [0.0] + [pt["pass_rate"] * 100 for pt in curve]
        last_sr = sr_points[-1]
        interp_sr = np.interp(grid, tok_points, sr_points, right=last_sr)
        task_srs[task_name] = interp_sr

    all_srs = np.array(list(task_srs.values()))
    mean_sr = all_srs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for task_name, srs in sorted(task_srs.items()):
        ax.plot(grid, srs, linewidth=0.8, alpha=0.4, label=task_name)
    ax.plot(grid, mean_sr, linewidth=2.5, color="black", label="Mean (7 tasks)")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Cumulative Tokens", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Cumulative: Tokens vs Success Rate (avg across 7 tasks)", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_cumulative_tokens_norm_vs_sr(all_data, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(0, 1, 200)

    task_srs = {}
    for task_name, data in all_data.items():
        curve = data["curve"]
        max_tok = data["total_tokens"]
        if max_tok == 0:
            continue
        tok_points = [0] + [pt["cumulative_tokens"] / max_tok for pt in curve]
        sr_points = [0.0] + [pt["pass_rate"] * 100 for pt in curve]
        last_sr = sr_points[-1]
        interp_sr = np.interp(grid, tok_points, sr_points, right=last_sr)
        task_srs[task_name] = interp_sr

    all_srs = np.array(list(task_srs.values()))
    mean_sr = all_srs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for task_name, srs in sorted(task_srs.items()):
        ax.plot(grid, srs, linewidth=0.8, alpha=0.4, label=task_name)
    ax.plot(grid, mean_sr, linewidth=2.5, color="black", label="Mean (7 tasks)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Normalized Tokens (0–1)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Cumulative: Normalized Tokens vs Success Rate (avg across 7 tasks)", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else find_project_dir()
    tz = local_tz()

    all_data = {}
    for task_name in TASK_MAP:
        config_stem = TASK_MAP[task_name]
        data, err = compute_task_data(task_name, config_stem, project_dir, tz)
        if err:
            print(f"SKIP {task_name}: {err}")
            continue
        all_data[task_name] = data
        print(f"OK   {task_name}: {len(data['curve'])} iters, {data['n_dispatches']} dispatches, {data['total_tokens']:,} tokens")

    if not all_data:
        print("\nNo data found.")
        return 1

    if args.print_table:
        print_table(all_data)

    out = Path("outputs/plots")
    out.mkdir(parents=True, exist_ok=True)
    plot_per_task_iter_vs_sr(all_data, out / "per_task_iter_vs_sr.png")
    plot_per_task_tokens_vs_sr(all_data, out / "per_task_tokens_vs_sr.png")
    plot_cumulative_iter_vs_sr(all_data, out / "cumulative_iter_vs_sr.png")
    plot_cumulative_tokens_vs_sr(all_data, out / "cumulative_tokens_vs_sr.png")
    plot_cumulative_tokens_norm_vs_sr(all_data, out / "cumulative_tokens_norm_vs_sr.png")

    print_table(all_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
