#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute per-chunk and cumulative token usage for each snapshot commit,
then attach the results as git notes (non-destructive — does not change
commit SHAs or invalidate snapshot tags).

Usage:
  python3 scripts/libero/annotate_snapshot_tokens.py [--dry-run] [--project-dir PATH] [--force]

Options:
  --dry-run       Print notes without writing them to git
  --project-dir   Override Claude project JSONL directory
                  (default: auto-detected from current working directory)
  --experiment-start
                  ISO timestamp for the experiment start window
  --force         Overwrite existing notes (default: skip if note exists)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# Scaling-law experiment start: "Setup scaling-law pipeline" commit timestamp.
# Everything before this is prior Evolutionary Search / LIBERO-10 work and must be excluded.
DEFAULT_EXPERIMENT_START = "2026-04-25T13:30:17-07:00"


def find_project_dir() -> Path:
    cwd = Path.cwd().resolve()
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


def git(*args) -> str:
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def get_snapshot_tags() -> list[str]:
    """Return snapshot-N* tags sorted numerically by N."""
    tags = git("tag", "-l", "snapshot-N*").splitlines()
    def sort_key(t):
        try:
            return int(t.replace("snapshot-N", ""))
        except ValueError:
            return 0
    return sorted(tags, key=sort_key)


def get_commit_info(tag: str) -> tuple[str, str]:
    """Return (sha, iso_timestamp) for the commit a tag points to."""
    sha = git("rev-list", "-n", "1", tag)
    ts = git("log", "-1", "--format=%aI", sha)
    return sha, ts


def iso_to_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def collect_tokens(project_dir: Path, since: datetime | None, until: datetime | None) -> tuple[dict, int]:
    seen: set[str] = set()
    totals = dict(input=0, cache_create=0, cache_read=0, output=0)
    count = 0

    for f in project_dir.rglob("*.jsonl"):
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

    totals["total"] = totals["input"] + totals["cache_create"] + totals["cache_read"] + totals["output"]
    return totals, count


def fmt_tokens(totals: dict, n_requests: int) -> str:
    return (
        f"  requests:     {n_requests:>10,}\n"
        f"  input:        {totals['input']:>12,}\n"
        f"  cache_create: {totals['cache_create']:>12,}\n"
        f"  cache_read:   {totals['cache_read']:>12,}\n"
        f"  output:       {totals['output']:>12,}\n"
        f"  total:        {totals['total']:>12,}"
    )


def note_exists(sha: str) -> bool:
    result = subprocess.run(["git", "notes", "show", sha], capture_output=True, text=True)
    return result.returncode == 0


def add_note(sha: str, note_text: str, force: bool):
    cmd = ["git", "notes", "add", "-m", note_text]
    if force:
        cmd += ["--force"]
    cmd.append(sha)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: git notes failed: {result.stderr.strip()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-dir", type=str, default=None)
    parser.add_argument(
        "--experiment-start",
        type=str,
        default=DEFAULT_EXPERIMENT_START,
        help="ISO timestamp for the experiment start window",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing notes")
    args = parser.parse_args()

    project_dir = Path(args.project_dir) if args.project_dir else find_project_dir()
    if not project_dir.exists():
        print(f"ERROR: project dir not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    tags = get_snapshot_tags()
    if not tags:
        print("No snapshot-N* tags found.")
        sys.exit(0)

    print(f"Project dir: {project_dir}")
    print(f"Snapshots found: {', '.join(tags)}")
    print(f"Mode: {'dry-run' if args.dry_run else 'writing git notes'}\n")

    exp_start = iso_to_dt(args.experiment_start)
    prev_ts: datetime | None = None

    for tag in tags:
        sha, ts_str = get_commit_info(tag)
        curr_ts = iso_to_dt(ts_str)
        n = tag.replace("snapshot-N", "")

        cum_totals, cum_requests = collect_tokens(project_dir, since=exp_start, until=curr_ts)
        chunk_since = prev_ts if prev_ts is not None else exp_start
        chunk_totals, chunk_requests = collect_tokens(project_dir, since=chunk_since, until=curr_ts)

        note = (
            f"Token Usage — {tag}\n"
            f"Timestamp: {ts_str}\n"
            f"Experiment baseline: {args.experiment_start}\n"
            f"\n"
            f"Cumulative (experiment start → N={n}):\n"
            f"{fmt_tokens(cum_totals, cum_requests)}\n"
            f"\n"
            f"This chunk only:\n"
            f"{fmt_tokens(chunk_totals, chunk_requests)}"
        )

        print(f"{'='*60}")
        print(f"{tag} ({sha[:8]}) @ {ts_str}")
        print(note)
        print()

        if not args.dry_run:
            if note_exists(sha) and not args.force:
                print(f"  -> Note already exists, skipping (use --force to overwrite)\n")
            else:
                add_note(sha, note, force=args.force)
                print(f"  -> Note written\n")

        prev_ts = curr_ts

    print("Done.")
    if not args.dry_run:
        print("View with:  git log --show-notes --oneline")
        print("Single note: git notes show <sha>")


if __name__ == "__main__":
    main()
