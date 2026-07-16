#!/usr/bin/env python3
"""Analyze per-iteration token usage for a LIBERO-Long-Pro Stage 1 debug agent run.

Reads the agent JSONL transcript and code-version file timestamps to bucket
token usage by iteration, then writes stage1/token_usage.json.

Usage:
    python3 scripts/libero/analyze_stage1_tokens.py \\
        --jsonl  /path/to/agent_output.jsonl \\
        --stage1 /path/to/debug_eval/suite/task/stage1 \\
        [--tz-offset -7]   # local UTC offset of the machine that ran the agent (default: auto-detect)

Output: <stage1>/token_usage.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path


def local_tz(offset_hours: float | None) -> timezone:
    if offset_hours is not None:
        return timezone(timedelta(hours=offset_hours))
    # Auto-detect from system clock
    return timezone(timedelta(seconds=-time.timezone))


def parse_iter_times(code_versions_dir: Path, tz: timezone) -> dict[int, datetime]:
    """Return {iter_num: utc_datetime} from code version filenames."""
    iter_times = {}
    for f in sorted(code_versions_dir.glob("iter_*.py")):
        m = re.match(r"iter_(\d+)_(\d{8}_\d{6})\.py", f.name)
        if m:
            it = int(m.group(1))
            local_dt = datetime.strptime(m.group(2), "%Y%m%d_%H%M%S").replace(tzinfo=tz)
            iter_times[it] = local_dt.astimezone(timezone.utc)
    return iter_times


def get_utc_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def bucket_events(events, iter_times, promotion_start):
    """
    Bucket (timestamp, usage) events into iterations.

    Bucketing rule:
      - iter N  = events where event_ts <= iter_N_code_ts  (AND > iter_(N-1)_code_ts)
      - last iter seed run = after last code_ts, up to promotion_start
      - promotion = after promotion_start
    """
    sorted_iters = sorted(iter_times.items())
    last_iter = sorted_iters[-1][0] if sorted_iters else None

    def get_bucket(event_ts):
        if promotion_start and event_ts > promotion_start:
            return "promotion"
        for it, code_ts in sorted_iters:
            if event_ts <= code_ts:
                return it
        # After last code timestamp but before/at promotion_start
        return last_iter

    zero = lambda: {"input_tokens": 0, "output_tokens": 0,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                    "calls": 0}
    buckets = defaultdict(zero)

    for ts, u in events:
        b = get_bucket(ts)
        buckets[b]["input_tokens"]                += u.get("input_tokens", 0)
        buckets[b]["output_tokens"]               += u.get("output_tokens", 0)
        buckets[b]["cache_creation_input_tokens"] += u.get("cache_creation_input_tokens", 0)
        buckets[b]["cache_read_input_tokens"]     += u.get("cache_read_input_tokens", 0)
        buckets[b]["calls"]                       += 1

    return dict(buckets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl",   required=True, help="Agent JSONL output file")
    parser.add_argument("--stage1",  required=True, help="Path to stage1/ directory")
    parser.add_argument("--tz-offset", type=float, default=None,
                        help="Local UTC offset in hours (e.g. -7 for PDT). Default: auto-detect.")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1)
    jsonl_path = Path(args.jsonl)
    tz = local_tz(args.tz_offset)

    if not jsonl_path.exists():
        print(f"ERROR: JSONL file not found: {jsonl_path}")
        return 1
    if not stage1_dir.exists():
        print(f"ERROR: stage1 dir not found: {stage1_dir}")
        return 1

    code_versions_dir = stage1_dir / "code_versions"
    iter_times = parse_iter_times(code_versions_dir, tz)

    if not iter_times:
        print(f"ERROR: no code version files found in {code_versions_dir}")
        return 1

    last_iter = max(iter_times)
    last_result_json = stage1_dir / f"iter_{last_iter}" / "result.json"
    promotion_start = get_utc_mtime(last_result_json) if last_result_json.exists() else None

    # Extract assistant usage events from JSONL
    events = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "assistant":
                msg = d.get("message", {})
                if isinstance(msg, dict) and "usage" in msg:
                    ts_str = d.get("timestamp", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        events.append((ts, msg["usage"]))

    if not events:
        print("ERROR: no assistant usage events found in JSONL")
        return 1

    buckets = bucket_events(events, iter_times, promotion_start)

    # Build output — ordered: iter 1, 2, ..., N, promotion
    ordered_keys = sorted(k for k in buckets if k != "promotion")
    if "promotion" in buckets:
        ordered_keys.append("promotion")

    total = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "calls": 0}

    rows = []
    for b in ordered_keys:
        bk = buckets[b]
        code_ts = iter_times.get(b)
        row = {
            "phase": "promotion" if b == "promotion" else f"iter_{b}",
            "code_written_at_utc": code_ts.isoformat() if code_ts else None,
            "calls": bk["calls"],
            "input_tokens": bk["input_tokens"],
            "output_tokens": bk["output_tokens"],
            "cache_creation_input_tokens": bk["cache_creation_input_tokens"],
            "cache_read_input_tokens": bk["cache_read_input_tokens"],
        }
        rows.append(row)
        for k in total:
            total[k] += bk[k]

    output = {
        "stage1_dir": str(stage1_dir),
        "jsonl": str(jsonl_path),
        "tz_offset_hours": tz.utcoffset(None).total_seconds() / 3600,
        "iters_found": sorted(iter_times.keys()),
        "promotion_boundary_utc": promotion_start.isoformat() if promotion_start else None,
        "per_iter": rows,
        "total": total,
    }

    out_path = stage1_dir / "token_usage.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Written: {out_path}")

    # Print summary table
    print(f"\n{'Phase':<22} {'calls':>6} {'input':>8} {'output':>8} {'cache_new':>12} {'cache_hit':>12}")
    print("-" * 72)
    for row in rows:
        ts = f" [→{row['code_written_at_utc'][11:16]}Z]" if row["code_written_at_utc"] else ""
        phase = row["phase"] + ts
        print(f"{phase:<22} {row['calls']:>6} {row['input_tokens']:>8,} {row['output_tokens']:>8,} "
              f"{row['cache_creation_input_tokens']:>12,} {row['cache_read_input_tokens']:>12,}")
    print("-" * 72)
    print(f"{'TOTAL':<22} {total['calls']:>6} {total['input_tokens']:>8,} {total['output_tokens']:>8,} "
          f"{total['cache_creation_input_tokens']:>12,} {total['cache_read_input_tokens']:>12,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
