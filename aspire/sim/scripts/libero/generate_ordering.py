#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate the reproducible random ordering of LIBERO-90 tasks for the scaling-law experiment.

Outputs: ordering.txt (one task name per line, 90 lines)

Usage:
    python3 scripts/libero/generate_ordering.py --seed 42 --output ordering.txt

Once committed, NEVER regenerate. The ordering defines the experiment.
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path


def get_libero_90_tasks(aspire_root: Path) -> list[str]:
    """Query ASPIRE env config for the 90 LIBERO-90 task names."""
    python_root = aspire_root.resolve().parents[1]
    py = aspire_root / ".venv/bin/python3"
    if not py.exists():
        raise FileNotFoundError(f".venv python not found at {py}. Run `uv venv + uv pip install -e .` first?")
    result = subprocess.run(
        [
            str(py),
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(python_root)!r}); "
                "from aspire.sim.cap.envs.configs.instantiate import get_task_names; "
                "print('\\n'.join(get_task_names('libero_90')))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(python_root)},
    )
    tasks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(tasks) != 90:
        raise RuntimeError(f"Expected 90 tasks, got {len(tasks)}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffle (default 42, locked once committed)")
    ap.add_argument("--aspire-root", type=Path, default=Path.cwd(), help="Path to aspire-scaling-law repo")
    ap.add_argument("--output", type=Path, default=Path("ordering.txt"), help="Output file")
    ap.add_argument("--force", action="store_true", help="Overwrite existing ordering.txt")
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        print(f"ERROR: {args.output} exists. Refusing to regenerate (pass --force to override).", file=sys.stderr)
        print("The ordering is part of the experimental specification and must NOT change.", file=sys.stderr)
        sys.exit(1)

    tasks = get_libero_90_tasks(args.aspire_root)
    print(f"Found {len(tasks)} LIBERO-90 tasks", file=sys.stderr)

    rng = random.Random(args.seed)
    shuffled = tasks.copy()
    rng.shuffle(shuffled)

    args.output.write_text("\n".join(shuffled) + "\n")
    print(f"Wrote {len(shuffled)} tasks to {args.output} (seed={args.seed})", file=sys.stderr)
    print(f"First 5 tasks (chunk 1): {shuffled[:5]}", file=sys.stderr)
    print(f"Last 5 tasks (chunk 18): {shuffled[-5:]}", file=sys.stderr)


if __name__ == "__main__":
    main()
