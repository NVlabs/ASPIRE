#!/usr/bin/env python3
"""Initialize a new iterative-debugging campaign root.

Run once at campaign kickoff:

    python3 .claude/skills/iterative-debugging/scripts/init_run.py \
        --tasks /path/to/tasks.json [--name short-slug]

Creates ``outputs/runs/<YYYY-MM-DD_HH-MM-SS>[_<name>]/``, copies the task list
into it as ``tasks.json``, points ``outputs/runs/LATEST`` at it, and prints the
new RUN_ROOT. All other scripts default to LATEST, so after this every
gen_progress.py / run_validation.py call needs no path arguments.

tasks.json format (same as gen_progress.py):

    [
      {"benchmark": "libero-pro", "suite": "libero_goal_swap", "task_id": 0,
       "task_name": "open_the_middle_drawer_of_the_cabinet"},
      ...
    ]
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import DEFAULT_HELDOUT_COUNT, DEV_SEEDS, LATEST_LINK, RUNS_ROOT, UNSEEN_SEEDS


def validate_tasks(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"task list not found: {path}")
    text = path.read_text()
    tasks = json.loads(text)
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(f"task list must be a non-empty JSON list: {path}")
    for entry in tasks:
        for field in ("benchmark", "suite", "task_id"):
            if field not in entry:
                raise SystemExit(f"task entry missing '{field}': {entry}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True, help="campaign task list JSON")
    parser.add_argument("--name", default="", help="optional short slug appended to the stamp")
    parser.add_argument(
        "--heldout-count",
        type=int,
        default=DEFAULT_HELDOUT_COUNT,
        help="held-out partition is seeds 1..N (default 50; expensive campaigns use 20)",
    )
    parser.add_argument(
        "--dev-seeds",
        type=int,
        nargs="+",
        default=DEV_SEEDS,
        help="development seeds (default 51..65; use 101..125 for Robosuite)",
    )
    parser.add_argument(
        "--unseen-seeds",
        type=int,
        nargs="+",
        default=UNSEEN_SEEDS,
        help="unseen-check seeds, measured once and never debugged against "
        "(default 66..70; use 126..130 for Robosuite)",
    )
    args = parser.parse_args()
    if any(seed < 1 for seed in args.dev_seeds) or len(set(args.dev_seeds)) != len(args.dev_seeds):
        raise SystemExit("--dev-seeds must contain unique positive integers")
    if any(seed < 1 for seed in args.unseen_seeds) or len(set(args.unseen_seeds)) != len(
        args.unseen_seeds
    ):
        raise SystemExit("--unseen-seeds must contain unique positive integers")
    if set(args.dev_seeds) & set(args.unseen_seeds):
        raise SystemExit("--dev-seeds and --unseen-seeds must be disjoint")
    if args.heldout_count < 1 or args.heldout_count >= min(args.dev_seeds):
        raise SystemExit(f"--heldout-count must be in 1..{min(args.dev_seeds) - 1}")
    if args.heldout_count >= min(args.unseen_seeds):
        raise SystemExit(
            f"--heldout-count must be in 1..{min(args.unseen_seeds) - 1} "
            "(held-out and unseen-check seed bands must not overlap)"
        )

    tasks_text = validate_tasks(args.tasks)
    # .astimezone() attaches the local zone to a naive now(); the rendered fields are unchanged,
    # so campaign directory names keep their existing form while the value stops being ambiguous.
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    if args.name:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.name).strip("-")
        stamp = f"{stamp}_{slug}"

    run_root = RUNS_ROOT / stamp
    if run_root.exists():
        raise SystemExit(f"campaign root already exists: {run_root}")
    run_root.mkdir(parents=True)
    (run_root / "tasks.json").write_text(tasks_text)
    (run_root / "campaign.json").write_text(
        json.dumps(
            {
                "heldout_count": args.heldout_count,
                "dev_seeds": args.dev_seeds,
                "unseen_seeds": args.unseen_seeds,
            },
            indent=2,
        )
        + "\n"
    )

    temporary = LATEST_LINK.with_name("LATEST.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(stamp)
    temporary.replace(LATEST_LINK)

    print(f"RUN_ROOT={run_root}")


if __name__ == "__main__":
    main()
