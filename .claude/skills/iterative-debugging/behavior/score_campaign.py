#!/usr/bin/env python3
"""Score a BEHAVIOR-1K block-by-block campaign from its run artifacts.

Exactly two numbers per task, both per seed:

  SUCCESS     the seed's final policy succeeded at least once
  NAVIGATION  the seed's final policy drove the robot to the object and got the arm
              working on it, in at least one episode

The unit of scoring is the FINAL POLICY, never the attempt. A worker grows one program
block by block and every attempt replays the whole program, so the early attempts are
development: their programs had no grasp block yet and could not have succeeded. They are
excluded entirely.

The final policy is found by hashing each run's ``source/program.py``. Runs sharing the
last distinct hash are the final policy's episodes: the winning attempt plus its
unchanged repeats. That is exact rather than inferred from attempt numbers.

Nothing else is reported as a rate. Reliability, last-episode outcome, per-leg
navigation success and attempts-to-first-hold are deliberately not printed: the first
campaign reported them side by side and they were confused for one another. The
artifacts keep every episode, so any of them can be recomputed if a specific question
ever needs it.

Usage:
    ./score_campaign.py <campaign-dir> [--stage stage2] [--per-seed]

<campaign-dir> holds one directory per task, each containing stage1/ and stage2/.
Directories whose name contains VOIDED are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

GRIPPER_CALLS = {"close-gripper", "set-gripper", "set-grippers"}
ARM_CALLS = {"move-to-pose", "execute-trajectory"}


def _call_names(run: Path) -> set[str]:
    """Tool names invoked in this run, from the trace's per-call directories."""
    calls = run / "trace" / "calls"
    if not calls.is_dir():
        return set()
    names = set()
    for entry in calls.iterdir():
        if entry.is_dir():
            parts = entry.name.split("-", 2)
            if len(parts) == 3:
                names.add(parts[2])
    return names


def episodes(seed: Path) -> list[dict]:
    """Every run of this seed in attempt order, tagged with its program hash."""
    found = []
    for attempt in sorted(seed.glob("attempts/attempt_*"), key=lambda p: p.name):
        if not attempt.is_dir():
            continue
        outcome_path = next(attempt.glob("**/outcome.json"), None)
        if outcome_path is None:
            continue  # launch crash or scene-load failure: not a policy failure
        run = outcome_path.parent
        program = run / "source" / "program.py"
        if not program.is_file():
            continue
        try:
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = _call_names(run)
        found.append(
            {
                "attempt": attempt.name,
                "program": hashlib.sha256(program.read_bytes()).hexdigest()[:12],
                "success": bool(outcome.get("task_success")),
                # Navigation delivered the robot and the arm worked on the object:
                # the jaws closed and a trajectory or pose move ran.
                "reached": bool(names & GRIPPER_CALLS) and bool(names & ARM_CALLS),
            }
        )
    return found


def score_task(task_dir: Path, stage: str) -> dict | None:
    seeds = [
        s
        for s in sorted((task_dir / stage).glob("seed_*"))
        if "VOIDED" not in s.name and s.is_dir()
    ]
    rows = []
    for seed in seeds:
        runs = episodes(seed)
        if not runs:
            continue
        final_hash = runs[-1]["program"]
        final = [r for r in runs if r["program"] == final_hash]
        rows.append(
            {
                "seed": seed.name.replace("seed_", ""),
                "attempts": len(runs),
                "success": any(r["success"] for r in final),
                "navigated": any(r["reached"] for r in final),
            }
        )
    if not rows:
        return None
    return {
        "task": task_dir.name,
        "seeds": len(rows),
        "success": sum(r["success"] for r in rows),
        "navigated": sum(r["navigated"] for r in rows),
        "failed": [r["seed"] for r in rows if not r["success"]],
        "unnavigated": [r["seed"] for r in rows if not r["navigated"]],
        "rows": rows,
    }


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "n/a"


def report(result: dict, per_seed: bool) -> None:
    n = result["seeds"]
    print(f"\n=== {result['task']} ===")
    print(f"  SUCCESS      {result['success']}/{n}  = {_pct(result['success'], n)}")
    print(f"  NAVIGATION   {result['navigated']}/{n}  = {_pct(result['navigated'], n)}")
    if result["failed"]:
        print(f"  seeds whose final policy never succeeded: {', '.join(result['failed'])}")
    if result["unnavigated"]:
        print(
            f"  seeds whose final policy never reached the object: {', '.join(result['unnavigated'])}"
        )
    if per_seed:
        print(f"  {'seed':>5} {'attempts':>9}  success  navigation")
        for row in result["rows"]:
            print(
                f"  {row['seed']:>5} {row['attempts']:>9}"
                f"  {'yes' if row['success'] else 'NO ':>7}"
                f"  {'yes' if row['navigated'] else 'NO '}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("campaign", type=Path, help="directory holding one folder per task")
    parser.add_argument("--stage", default="stage2", help="stage to score (default: stage2)")
    parser.add_argument("--per-seed", action="store_true", help="also print a row per seed")
    args = parser.parse_args()

    if not args.campaign.is_dir():
        print(f"error: {args.campaign} is not a directory", file=sys.stderr)
        return 2
    tasks = [d for d in sorted(args.campaign.iterdir()) if (d / args.stage).is_dir()]
    if not tasks:
        print(
            f"error: no task directory under {args.campaign} contains {args.stage}/",
            file=sys.stderr,
        )
        return 2

    for task in tasks:
        result = score_task(task, args.stage)
        if result is None:
            print(f"\n=== {task.name} — no scorable runs in {args.stage}/ ===")
            continue
        report(result, args.per_seed)

    print(
        "\nBoth numbers are per seed and judged on the final policy only. Development attempts\n"
        "are excluded. Infrastructure losses (launch crash, foreign job exhausting the GPU,\n"
        "service outage) are not policy failures: name them separately in the report."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
