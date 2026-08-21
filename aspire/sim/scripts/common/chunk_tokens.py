#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute token usage for a Claude Code session, optionally filtered to a time window.

Usage:
  # Print current ISO timestamp (record this before dispatching a chunk)
  python3 scripts/common/chunk_tokens.py --print-timestamp

  # Count all tokens in this project's sessions
  python3 scripts/common/chunk_tokens.py

  # Count only tokens after a given timestamp (i.e. for a single chunk)
  python3 scripts/common/chunk_tokens.py --since 2026-04-25T14:30:00.000Z

  # Count tokens between two timestamps
  python3 scripts/common/chunk_tokens.py --since 2026-04-25T14:30:00.000Z --until 2026-04-25T18:00:00.000Z
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def iso_to_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def find_project_dir() -> Path:
    cwd = Path(os.getcwd()).resolve()
    # Claude Code sanitizes CWD by replacing "/" with "-", keeping the leading "-"
    sanitized = str(cwd).replace("/", "-")
    candidate = Path.home() / ".claude" / "projects" / sanitized
    if candidate.exists():
        return candidate
    # fallback: try without leading dash (older convention)
    sanitized_alt = str(cwd).lstrip("/").replace("/", "-")
    candidate_alt = Path.home() / ".claude" / "projects" / sanitized_alt
    if candidate_alt.exists():
        return candidate_alt
    # fallback: search for matching suffix
    base = Path.home() / ".claude" / "projects"
    if base.exists():
        cwd_str = str(cwd)
        for d in base.iterdir():
            if d.name.endswith(sanitized_alt):
                return d
    return candidate


def collect_tokens(project_dir: Path, since: datetime | None, until: datetime | None):
    seen: set[str] = set()
    totals = dict(input=0, cache_create=0, cache_read=0, output=0)

    jsonl_files = list(project_dir.rglob("*.jsonl")) if project_dir.exists() else []
    if not jsonl_files:
        return totals, 0

    count = 0
    for f in jsonl_files:
        try:
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "assistant":
                continue

            # Timestamp filter
            ts_str = d.get("timestamp", "")
            if ts_str and (since or until):
                try:
                    ts = iso_to_dt(ts_str)
                    if since and ts < since:
                        continue
                    if until and ts > until:
                        continue
                except ValueError:
                    pass

            req_id = d.get("requestId", "")
            if req_id:
                if req_id in seen:
                    continue
                seen.add(req_id)

            usage = d.get("message", {}).get("usage", {})
            totals["input"] += usage.get("input_tokens", 0)
            totals["cache_create"] += usage.get("cache_creation_input_tokens", 0)
            totals["cache_read"] += usage.get("cache_read_input_tokens", 0)
            totals["output"] += usage.get("output_tokens", 0)
            count += 1

    return totals, count


def main():
    parser = argparse.ArgumentParser(description="Token usage for Claude Code project sessions")
    parser.add_argument("--print-timestamp", action="store_true",
                        help="Print current UTC ISO timestamp and exit (record before dispatching a chunk)")
    parser.add_argument("--since", type=str, default=None,
                        help="ISO timestamp: count only requests after this time")
    parser.add_argument("--until", type=str, default=None,
                        help="ISO timestamp: count only requests before this time")
    parser.add_argument("--project-dir", type=str, default=None,
                        help="Override ~/.claude/projects/<sanitized-cwd> path")
    args = parser.parse_args()

    if args.print_timestamp:
        print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        return

    since = iso_to_dt(args.since) if args.since else None
    until = iso_to_dt(args.until) if args.until else None

    project_dir = Path(args.project_dir) if args.project_dir else find_project_dir()
    totals, n_requests = collect_tokens(project_dir, since, until)

    total = totals["input"] + totals["cache_create"] + totals["cache_read"] + totals["output"]

    if args.since or args.until:
        window = f"since {args.since}" + (f" until {args.until}" if args.until else "")
        print(f"Tokens ({window}, {n_requests} requests):")
    else:
        print(f"Tokens (all time, {n_requests} requests):")

    print(f"  input:    {totals['input']:>12,}")
    print(f"  cache:    {totals['cache_create']:>12,}  (created)")
    print(f"  cache:    {totals['cache_read']:>12,}  (read)")
    print(f"  output:   {totals['output']:>12,}")
    print(f"  total:    {total:>12,}")


if __name__ == "__main__":
    main()
