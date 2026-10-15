"""Wake the coordinator when a live agent stops writing to its transcript.

An agent blocked on a permission prompt does not fail — it *waits*, and looks identical to one
that is thinking. This campaign lost **29.5 hours** that way, including single incidents of
14 h 18 m and 14 h 48 m, all of them an `rm` awaiting approval. The prompts now ban `rm`, but that
is the cause we know about; this catches the ones we do not.

Run it in the background and it exits as soon as something has been idle too long, which
re-invokes the coordinator with the diagnosis:

    uv run python .../watchdog.py --live-file /path/to/live_agents.txt &

`--live-file` holds one agent id per line and is **re-read every cycle**, so the coordinator just
rewrites it on each dispatch and each completion. An earlier version took a fixed list at launch,
so every agent that finished normally went idle and tripped a false alarm — three in one session.
A watchdog that cries wolf gets ignored, which is worse than not having one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

# Claude Code writes one JSONL transcript per subagent; its mtime is the liveness signal.
DEFAULT_SUBAGENT_GLOB = "/root/.claude/projects/*/*/subagents"
COMMAND_RE = re.compile(r'"command"\s*:\s*"((?:[^"\\]|\\.){0,300})')


def transcript_for(subagent_dirs: list[Path], agent_id: str) -> Path | None:
    for directory in subagent_dirs:
        candidate = directory / f"agent-{agent_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


def last_command(transcript: Path) -> str:
    """The most recent shell command in the transcript — usually the thing that is blocked."""
    try:
        tail = transcript.read_bytes()[-200_000:].decode("utf-8", "replace")
    except OSError:
        return "<unreadable>"
    matches = COMMAND_RE.findall(tail)
    return matches[-1][:300] if matches else "<no shell command in the recent transcript>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-file", type=Path, required=True, help="one agent id per line; re-read every cycle"
    )
    parser.add_argument(
        "--subagent-dir",
        type=Path,
        action="append",
        default=None,
        help="directory of agent-<id>.jsonl transcripts (repeatable)",
    )
    parser.add_argument(
        "--idle-limit",
        type=int,
        default=1500,
        help="seconds of silence before waking the coordinator (default 1500)",
    )
    parser.add_argument("--poll", type=int, default=120, help="seconds between checks")
    parser.add_argument(
        "--once", action="store_true", help="check a single time and exit; for testing"
    )
    args = parser.parse_args()

    dirs = args.subagent_dir or list(Path("/").glob(DEFAULT_SUBAGENT_GLOB.lstrip("/")))
    if not dirs:
        print("no subagent transcript directory found; pass --subagent-dir", file=sys.stderr)
        return 2

    while True:
        if not args.live_file.exists():
            print(f"live file vanished: {args.live_file}", file=sys.stderr)
            return 2
        now = time.time()
        for line in args.live_file.read_text().splitlines():
            agent = line.strip()
            if not agent or agent.startswith("#"):
                continue
            transcript = transcript_for(dirs, agent)
            if transcript is None:
                continue  # not started yet, or already reaped
            idle = now - transcript.stat().st_mtime
            if idle > args.idle_limit:
                print(f"STALL: agent {agent} idle {idle / 60:.0f} min")
                print(f"transcript: {transcript}")
                print(f"last command: {last_command(transcript)}")
                print(
                    "\nIf it is awaiting approval, the agent cannot proceed and cannot report it. "
                    "Decide whether to answer it, kill the agent and re-dispatch the task, or "
                    "salvage its work yourself."
                )
                return 0
        if args.once:
            print("no stalls")
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
