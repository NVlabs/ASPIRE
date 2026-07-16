#!/usr/bin/env python3
"""Plot cumulative tokens (x) vs debug success rate (y) per iteration for each task.

Finds each task's subagent JSONL by grepping for the config stem, computes
per-iteration token usage from the JSONL + code_versions timestamps, joins
with iter_*_result.json (debug seeds 101-125), and plots one curve per task.

Usage:
    python3 scripts/robosuite/plot_tokens_vs_debug_sr.py [--project-dir PATH] [--output PATH]

    --project-dir   Override ~/.claude/projects/<sanitized-cwd>
    --output        Output image path (default: outputs/plots/tokens_vs_debug_sr.png)
    --task TASK     Plot only one task (e.g. cube_lifting). Can repeat.
    --print-table   Print table to stdout instead of plotting
"""

import argparse
import json
import re
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path


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
    local_offset = datetime.now(timezone.utc).astimezone().utcoffset()
    return timezone(local_offset)


def find_subagent_jsonl(project_dir: Path, config_stem: str, task_name: str) -> Path | None:
    """Find the subagent JSONL for a fix loop task.
    Matches on the Task Assignment header pattern (e.g. 'TASK:       cube_lifting')
    which only appears in real dispatch prompts, not research agents.
    If multiple match (e.g. crash + redispatch), return the largest (most complete)."""
    candidates = list(project_dir.rglob("subagents/*.jsonl"))
    # Match the exact dispatch pattern from the subagent prompt template
    dispatch_pattern = f"TASK:"
    matches = []
    for jsonl_path in candidates:
        try:
            text = jsonl_path.read_text(errors="replace")
            # Must contain both the task assignment pattern AND the config stem
            # to distinguish from research agents that merely read the docs
            if dispatch_pattern in text and config_stem in text:
                # Extra check: the task name should appear near "TASK:" in the first user message
                found = False
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get("type") == "user":
                        msg_content = str(d.get("message", ""))
                        if f"TASK:" in msg_content and task_name in msg_content and config_stem in msg_content:
                            found = True
                        break
                if found:
                    matches.append(jsonl_path)
        except OSError:
            continue
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_size)


def parse_iter_times(code_versions_dir: Path, tz: timezone) -> dict[int, datetime]:
    """Return {iter_num: utc_datetime} from code version filenames."""
    iter_times = {}
    for f in sorted(code_versions_dir.glob("iter_*.py")):
        m = re.match(r"iter_(\d+)_(\d{8}_\d{6})\.py", f.name)
        if m:
            it = int(m.group(1))
            local_dt = datetime.strptime(m.group(2), "%Y%m%d_%H%M%S").replace(tzinfo=tz)
            if it not in iter_times:
                iter_times[it] = local_dt.astimezone(timezone.utc)
    return iter_times


def extract_usage_events(jsonl_path: Path) -> list[tuple[datetime, dict]]:
    """Extract (timestamp, usage_dict) from a JSONL file."""
    events = []
    seen = set()
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
                if req_id in seen:
                    continue
                seen.add(req_id)
            msg = d.get("message", {})
            if isinstance(msg, dict) and "usage" in msg:
                ts_str = d.get("timestamp", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    events.append((ts, msg["usage"]))
    return events


def bucket_tokens_per_iter(events, iter_times):
    """Bucket token usage events by iteration using code version timestamps."""
    sorted_iters = sorted(iter_times.items())
    if not sorted_iters:
        return {}

    def get_bucket(event_ts):
        for it, code_ts in sorted_iters:
            if event_ts <= code_ts:
                return it
        return sorted_iters[-1][0]

    buckets = defaultdict(lambda: {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0})
    for ts, u in events:
        b = get_bucket(ts)
        buckets[b]["input"] += u.get("input_tokens", 0)
        buckets[b]["output"] += u.get("output_tokens", 0)
        buckets[b]["cache_create"] += u.get("cache_creation_input_tokens", 0)
        buckets[b]["cache_read"] += u.get("cache_read_input_tokens", 0)
    return dict(buckets)


def load_results(code_versions_dir: Path) -> dict[int, dict]:
    """Load iter_*_result.json (debug) files, keyed by iter number."""
    results = {}
    for f in code_versions_dir.glob("iter_*_result.json"):
        if f.name.endswith("_eval_result.json"):
            continue
        try:
            r = json.loads(f.read_text())
            results[r["iter"]] = r
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return results


def compute_task_curve(task_name: str, config_stem: str, project_dir: Path, tz: timezone):
    """Compute (cumulative_tokens, pass_rate) curve for one task."""
    task_dir = BASELINE_DIR / config_stem
    cv_dir = task_dir / "code_versions"

    if not cv_dir.exists():
        return None, f"no code_versions/ dir"

    results = load_results(cv_dir)
    if not results:
        return None, f"no result.json files"

    jsonl_path = find_subagent_jsonl(project_dir, config_stem, task_name)
    if jsonl_path is None:
        return None, f"no subagent JSONL found (grep for {config_stem})"

    iter_times = parse_iter_times(cv_dir, tz)
    if not iter_times:
        return None, f"no timestamped code files"

    events = extract_usage_events(jsonl_path)
    if not events:
        return None, f"no usage events in {jsonl_path.name}"

    token_buckets = bucket_tokens_per_iter(events, iter_times)

    # Build curve: for each iteration that has both tokens and results
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

    return curve, None


def print_table(all_curves):
    for task_name, curve in sorted(all_curves.items()):
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'Iter':>5} {'Tokens (iter)':>14} {'Tokens (cum)':>14} {'SR':>8}")
        print(f"{'-'*5} {'-'*14} {'-'*14} {'-'*8}")
        for pt in curve:
            sr = f"{pt['n_pass']}/{pt['n_total']}"
            print(f"{pt['iter']:>5} {pt['tokens_this_iter']:>14,} {pt['cumulative_tokens']:>14,} {sr:>8}")


def plot_curves(all_curves, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    for task_name, curve in sorted(all_curves.items()):
        x = [pt["cumulative_tokens"] for pt in curve]
        y = [pt["pass_rate"] * 100 for pt in curve]
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=task_name)

    ax.set_xlabel("Cumulative Tokens", fontsize=12)
    ax.set_ylabel("Success Rate on Debug Seeds (%)", fontsize=12)
    ax.set_title("Robosuite Fix Loop: Tokens vs Success Rate", fontsize=14)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/plots/tokens_vs_debug_sr.png")
    parser.add_argument("--task", action="append", default=None,
                        help="Plot only these tasks (can repeat)")
    parser.add_argument("--print-table", action="store_true",
                        help="Print table to stdout instead of plotting")
    args = parser.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else find_project_dir()
    tz = local_tz()
    tasks = args.task if args.task else list(TASK_MAP.keys())

    all_curves = {}
    for task_name in tasks:
        config_stem = TASK_MAP.get(task_name)
        if not config_stem:
            print(f"SKIP {task_name}: unknown task")
            continue
        curve, err = compute_task_curve(task_name, config_stem, project_dir, tz)
        if err:
            print(f"SKIP {task_name}: {err}")
            continue
        all_curves[task_name] = curve
        print(f"OK   {task_name}: {len(curve)} iterations")

    if not all_curves:
        print("\nNo data to plot. Make sure subagent JSONL files exist in:")
        print(f"  {project_dir}")
        return 1

    if args.print_table:
        print_table(all_curves)
    else:
        plot_curves(all_curves, Path(args.output))
        print_table(all_curves)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
