#!/usr/bin/env python3
r"""Decide whether ``run-batch`` may be trusted for a campaign, with evidence.

``cap-harness run-batch`` reuses a worker process across seeds. Every seed still
builds and closes its own environment there, but "still builds its own
environment" is a claim about code, and what a campaign needs is a measurement
on the machine it will run on.

So this runs the same seeds twice and compares the outcomes:

    A/A   the unbatched loop, twice          -- how much this task varies anyway
    A/B   the unbatched loop vs. run-batch   -- what batching changed

The A/A leg is the point. A LIBERO episode is not bit-identical run to run --
rendering and the perception services see to that -- so an A/B difference means
nothing until you know what a difference between two *identical* invocations
looks like. Batching is equivalent when A/B disagrees no more than A/A does.

    python3 scripts/check_batch_equivalence.py \\
      --suite libero_goal_swap --task-id 0 --program fix_code.py \\
      --gpu 3 --seeds 51-56 --workers 3 --cap-harness .venv-libero/bin/cap-harness

Costs three runs of every seed, so use a handful of seeds, not fifty.
"""

# A hand-run diagnostic: its output is the deliverable.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cap_harness.batch import parse_seed_arguments

#: What a seed is compared on. Wall clock and run ids are expected to differ;
#: these are the fields a campaign's conclusions actually rest on.
COMPARED = ("task_success", "program_ok", "termination_reason", "steps_executed")


def read_outcome(run_dir: Path) -> dict | None:
    outcome = run_dir / "outcome.json"
    if not outcome.is_file():
        return None
    try:
        return json.loads(outcome.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def newest_run(results_base: Path, seed: int) -> Path | None:
    candidates = [path for path in results_base.glob(f"{seed:04d}/*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def collect(results_base: Path, seeds: list[int]) -> dict[int, dict | None]:
    collected: dict[int, dict | None] = {}
    for seed in seeds:
        run_dir = newest_run(results_base, seed)
        outcome = read_outcome(run_dir) if run_dir else None
        collected[seed] = (
            {field: outcome.get(field) for field in COMPARED} if outcome is not None else None
        )
    return collected


def run_unbatched(
    args: argparse.Namespace, seeds: list[int], results_base: Path, env: dict
) -> None:
    logs = results_base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        command = [
            args.cap_harness, "run",
            "--benchmark", args.benchmark,
            "--suite", args.suite,
            "--task-id", str(args.task_id),
            "--seed", str(seed),
            "--program", str(args.program),
            "--output-root", str(results_base),
            "--max-steps", str(args.max_steps),
            "--camera-width", str(args.camera_width),
            "--camera-height", str(args.camera_height),
            "--init-mode", args.init_mode,
            "--flat-run-dir",
        ]  # fmt: skip
        print(f"  seed {seed:02d} ...", end="", flush=True)
        started = time.monotonic()
        with (logs / f"seed_{seed:02d}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        print(f" {time.monotonic() - started:.0f}s", flush=True)


def run_batched(args: argparse.Namespace, seeds: list[int], results_base: Path, env: dict) -> None:
    command = [
        args.cap_harness, "run-batch",
        "--benchmark", args.benchmark,
        "--suite", args.suite,
        "--task-id", str(args.task_id),
        "--seeds", *[str(seed) for seed in seeds],
        "--program", str(args.program),
        "--output-root", str(results_base),
        "--max-steps", str(args.max_steps),
        "--camera-width", str(args.camera_width),
        "--camera-height", str(args.camera_height),
        "--init-mode", args.init_mode,
        "--flat-run-dir",
        "--workers", str(args.workers),
        "--recycle-after", str(args.recycle_after),
        "--log-dir", str(results_base / "logs"),
    ]  # fmt: skip
    subprocess.run(command, env=env, check=False)


def compare(
    label: str,
    left: dict[int, dict | None],
    right: dict[int, dict | None],
    seeds: list[int],
) -> tuple[list[int], list[int]]:
    """Report per-seed disagreement, as (differing seeds, seeds that never ran).

    A seed that produced no run is counted apart from a seed that produced a
    different result. Folding the two together lets a total failure to execute
    read as agreement -- both arms "differ" on every seed, the counts match, and
    the comparison reports equivalence having measured nothing.
    """
    differing: list[int] = []
    missing: list[int] = []
    print(f"\n{label}")
    for seed in seeds:
        first, second = left[seed], right[seed]
        if first is None or second is None:
            missing.append(seed)
            print(
                f"  seed {seed:02d}: MISSING RUN "
                f"(left={first is not None} right={second is not None})"
            )
            continue
        deltas = [
            f"{field}: {first[field]!r} -> {second[field]!r}"
            for field in COMPARED
            if first[field] != second[field]
        ]
        if deltas:
            differing.append(seed)
            print(f"  seed {seed:02d}: {'; '.join(deltas)}")
        else:
            print(f"  seed {seed:02d}: same (task_success={first['task_success']})")
    print(
        f"  -> {len(differing)}/{len(seeds)} seed(s) differed"
        + (f", {len(missing)} never ran" if missing else "")
    )
    return differing, missing


def _cleanup(args: argparse.Namespace, roots: dict) -> None:
    """Drop the result trees, but only once nothing is left to inspect.

    This used to run before the verdict, so a divergence printed "compare the
    logs of seed(s) ..." about logs it had just deleted.
    """
    if args.keep:
        return
    for root in roots.values():
        shutil.rmtree(root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default="libero-pro")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", required=True, help="e.g. 51-56")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--recycle-after", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/batch-equivalence"))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument("--init-mode", default="seeded", choices=("saved", "seeded"))
    parser.add_argument("--cap-harness", default=".venv-libero/bin/cap-harness")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the three result trees (they hold videos; they are large)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_arguments(args.seeds)
    if not args.program.is_file():
        raise SystemExit(f"program does not exist: {args.program}")

    env = os.environ.copy()
    env.update(
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        EGL_PLATFORM="device",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MUJOCO_EGL_DEVICE_ID=str(args.gpu),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    # LIBERO prompts on stdin at import when this is unset, so a child with no
    # terminal dies with a bare `EOFError: EOF when reading a line` and nothing
    # naming the cause. Default it to what bootstrap_libero.sh creates.
    env.setdefault("LIBERO_CONFIG_PATH", str(Path.cwd() / ".libero"))
    # Each invocation owns fresh evidence, including after an inconclusive run
    # or a previous --keep. Failed children must not reuse an older outcome.
    args.output_root.mkdir(parents=True, exist_ok=True)
    invocation_root = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root))
    roots = {name: invocation_root / name for name in ("unbatched-a", "unbatched-b", "batched")}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    print(f"seeds: {seeds}\ngpu: {args.gpu}\nprogram: {args.program}")
    timings: dict[str, float] = {}

    print("\n[1/3] unbatched, pass A (one process per seed)")
    started = time.monotonic()
    run_unbatched(args, seeds, roots["unbatched-a"], env)
    timings["unbatched-a"] = time.monotonic() - started

    print("\n[2/3] unbatched, pass B (identical invocation, to measure run-to-run noise)")
    started = time.monotonic()
    run_unbatched(args, seeds, roots["unbatched-b"], env)
    timings["unbatched-b"] = time.monotonic() - started

    print(f"\n[3/3] batched ({args.workers} worker(s), recycling every {args.recycle_after})")
    started = time.monotonic()
    run_batched(args, seeds, roots["batched"], env)
    timings["batched"] = time.monotonic() - started

    outcomes = {name: collect(root, seeds) for name, root in roots.items()}
    noise, missing_aa = compare(
        "A/A  unbatched vs. unbatched (inherent run-to-run variation)",
        outcomes["unbatched-a"],
        outcomes["unbatched-b"],
        seeds,
    )
    changed, missing_ab = compare(
        "A/B  unbatched vs. batched (what batching changed)",
        outcomes["unbatched-a"],
        outcomes["batched"],
        seeds,
    )
    missing = sorted(set(missing_aa) | set(missing_ab))

    speedup = timings["unbatched-a"] / timings["batched"] if timings["batched"] else 0.0
    print(
        f"\nwall clock: unbatched {timings['unbatched-a']:.0f}s, "
        f"batched {timings['batched']:.0f}s on {args.workers} worker(s) -> {speedup:.1f}x"
    )

    verdict_path = args.output_root / "verdict.json"
    verdict = {
        "results_root": str(invocation_root),
        "seeds": seeds,
        "workers": args.workers,
        "recycle_after": args.recycle_after,
        "aa_differing_seeds": noise,
        "ab_differing_seeds": changed,
        "seeds_that_never_ran": missing,
        "conclusive": not missing,
        "equivalent": (not missing) and len(changed) <= len(noise),
        "timings_s": {name: round(value, 1) for name, value in timings.items()},
        "speedup": round(speedup, 2),
    }
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (invocation_root / "verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"verdict: {verdict_path}")

    if missing:
        # Refuse to grade a comparison that did not happen. Every arm "differs"
        # on a seed nobody ran, which would otherwise pass the equivalence test
        # by measuring nothing at all.
        print(
            f"\nVERDICT: INCONCLUSIVE — {len(missing)} of {len(seeds)} seed(s) produced no run "
            f"in at least one arm: {missing}. This says nothing about batching. Check a log "
            f"under {args.output_root} (a bare `EOFError: EOF when reading a line` means "
            "LIBERO_CONFIG_PATH was unset), fix the cause, and rerun. Result trees kept."
        )
        return 2

    if len(changed) > len(noise):
        print(
            "\nVERDICT: batching changed more than run-to-run noise does. Do NOT use "
            "--workers for this campaign; compare the logs of "
            f"seed(s) {changed}. Result trees kept under {args.output_root}."
        )
        return 1
    _cleanup(args, roots)
    if noise:
        print(
            f"\nVERDICT: batching is within this task's own noise ({len(noise)} seed(s) "
            f"disagreed between two identical unbatched passes). Batching is not the "
            "variable here -- but note that this task does not reproduce exactly."
        )
        return 0
    print("\nVERDICT: identical outcomes on every seed. Batching is safe for this task.")
    _cleanup(args, roots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
