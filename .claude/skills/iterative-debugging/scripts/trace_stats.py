"""Campaign cost accounting, from the agent transcripts and the run's own manifests.

Answers "what did this campaign actually cost" in the three units that matter: wall-clock, tokens,
and simulator time. Wall-clock is reported two ways, and the distinction is the whole point.

**Span** is first-timestamp to last. **Active** removes gaps longer than `--stall-gap`, because a
gap that long is not compute — it is an agent waiting on a human to approve something, which looks
identical to thinking and is not. Reporting span as if it were compute overstated one campaign's
throughput by a factor of three before anyone noticed a single agent had sat idle for 14 h 18 m.

Choosing `--stall-gap` is a judgement call, so check it against your own data first: a 5-minute
threshold counted 120 legitimate `run-batch` waits as stalls and inflated the total to 43.5 h. The
gap histogram (`--histogram`) showed a clean break — many gaps of 5-15 min, then nothing until
27 min — which is what put the default at 15 minutes.

    uv run python .../trace_stats.py --histogram
    uv run python .../trace_stats.py --run-root outputs/runs/<campaign> --json out.json

NOTE: transcript parsing is coupled to the Claude Code JSONL layout. If no transcripts are found,
the validation-time and per-task sections still work — they read the campaign's own manifests.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import resolve_run_root

DEFAULT_TRANSCRIPT_GLOB = "/root/.claude/projects/*/*/subagents/*.jsonl"
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
TASK_RE = re.compile(r"(libero_[a-z0-9_]+)[/\\]task_(\d+)")
SUITE_RE = re.compile(r"SUITE:\s+(libero_[a-z0-9_]+).*?TASK_ID:\s+(\d+)", re.DOTALL)


def parse_ts(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def scan(path: Path, stall_gap: float) -> dict | None:
    """Span, active time, token usage, and which task an agent worked on."""
    stamps: list[dt.datetime] = []
    usage: collections.Counter = collections.Counter()
    task, kind = None, "other"
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("timestamp"):
            stamps.append(parse_ts(record["timestamp"]))
        message = record.get("message") or {}
        for field, value in (message.get("usage") or {}).items():
            if field in TOKEN_FIELDS:
                usage[field] += int(value or 0)
        if task is None:
            text = json.dumps(message.get("content"))[:4000]
            match = TASK_RE.search(text) or SUITE_RE.search(text)
            if match:
                task = f"{match.group(1)}/{match.group(2)}"
                kind = "verify" if "VERIFY_DIR" in text or "verifier" in text.lower() else "fix"
    if not stamps:
        return None
    stamps.sort()
    span = (stamps[-1] - stamps[0]).total_seconds()
    gaps = [(stamps[i] - stamps[i - 1]).total_seconds() for i in range(1, len(stamps))]
    stalled = sum(g for g in gaps if g > stall_gap)
    return {
        "file": path.name,
        "task": task,
        "kind": kind,
        "span": span,
        "active": span - stalled,
        "stalled": stalled,
        "gaps": gaps,
        **usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--transcripts", default=DEFAULT_TRANSCRIPT_GLOB)
    parser.add_argument(
        "--stall-gap",
        type=float,
        default=900.0,
        help="seconds; a gap longer than this is not compute (default 900)",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="print the gap distribution, to justify --stall-gap on YOUR data",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    run_root = resolve_run_root(args.run_root)
    agents = [
        a
        for a in (
            scan(p, args.stall_gap) for p in sorted(Path("/").glob(args.transcripts.lstrip("/")))
        )
        if a
    ]

    if args.histogram:
        buckets = collections.Counter()
        for agent in agents:
            for gap in agent["gaps"]:
                if gap < 60:
                    continue
                buckets[int(gap // 300) * 5] += 1
        print("=== GAP DISTRIBUTION (minutes) ===")
        print("Look for a clean break: below it are real waits, above it are stalls.")
        for low in sorted(buckets):
            print(f"  {low:>4}-{low + 5:<4} min  {'#' * min(buckets[low], 60)} {buckets[low]}")
        print()

    validation = {}
    for manifest in sorted(run_root.glob("*/*/*/validation/runs/*/manifest.json")):
        data = json.loads(manifest.read_text())
        identity = data["identity"]
        validation[f"{identity['suite']}/{identity['task_id']}"] = {
            "secs": (parse_ts(data["updated_at"]) - parse_ts(data["created_at"])).total_seconds(),
            "passes": data["passes"],
            "trials": data["trials"],
            "sha": (data.get("git_commit") or "")[:10],
        }

    print("=== AGENTS ===")
    print(f"transcripts: {len(agents)}")
    for kind in ("fix", "verify", "other"):
        chosen = [a for a in agents if a["kind"] == kind]
        if not chosen:
            continue
        print(
            f"  {kind:7s} n={len(chosen):3d}  active {sum(a['active'] for a in chosen) / 3600:8.2f} h"
            f"  stalled {sum(a['stalled'] for a in chosen) / 3600:6.2f} h"
            f"  out-tok {sum(a['output_tokens'] for a in chosen):>12,}"
        )

    totals: collections.Counter = collections.Counter()
    for agent in agents:
        for field in TOKEN_FIELDS:
            totals[field] += agent.get(field, 0)
    print("\n=== TOKENS ===")
    for field in TOKEN_FIELDS:
        print(f"  {field:32s} {totals[field]:>16,}")
    print(f"  {'TOTAL':32s} {sum(totals.values()):>16,}")

    print("\n=== WALL CLOCK ===")
    print(f"  agent span (sum)   {sum(a['span'] for a in agents) / 3600:8.2f} h")
    print(f"  agent ACTIVE (sum) {sum(a['active'] for a in agents) / 3600:8.2f} h")
    print(
        f"  stalled >{args.stall_gap / 60:.0f} min    "
        f"{sum(a['stalled'] for a in agents) / 3600:8.2f} h   <- humans, not compute"
    )
    if validation:
        times = sorted(v["secs"] for v in validation.values())
        print(
            f"  held-out validation{sum(times) / 3600:8.2f} h  ({len(validation)} runs, "
            f"median {times[len(times) // 2] / 60:.1f} min)"
        )
        scored = [v for v in validation.values() if v["trials"]]
        if scored:
            passes = sum(v["passes"] for v in scored)
            trials = sum(v["trials"] for v in scored)
            print(
                f"\n=== HELD-OUT ===\n  {passes}/{trials} ({passes / trials * 100:.1f}%) "
                f"over {len(scored)} tasks"
            )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "agents": [{k: v for k, v in a.items() if k != "gaps"} for a in agents],
                    "validation": validation,
                },
                indent=1,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
