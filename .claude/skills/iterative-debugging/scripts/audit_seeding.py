"""Audit that seeds are doing what a held-out split needs them to do.

A campaign's headline number is only meaningful if seeds 1-50 are 50 distinct layouts, reproducible
across processes, and disjoint from whatever the workers developed against. That is four separate
properties and none of them is implied by "the scene looks different each time".

This exists because the campaign it was written for had a real seeding bug — every seed resolving
to the same layout — which was fixed mid-run, invalidating every sweep taken before the fix. Run
all four checks before trusting a number, and again after any change to the adapter's reset path.

    uv run python .../audit_seeding.py vary    --suite libero_goal_swap --task-id 4
    uv run python .../audit_seeding.py repeat  --suite libero_goal_swap --task-id 4
    uv run python .../audit_seeding.py distinct --suite libero_goal_swap --task-id 4
    uv run python .../audit_seeding.py modes   --suite libero_goal_swap --task-id 4

- **vary**     — do objects actually move between seeds? Per-joint spread, so a task whose only
                 movable object is pinned shows up as such rather than as a passing check.
- **repeat**   — is seed -> layout stable within a process and reproducible in a fresh one? Run it
                 twice and diff; if the signatures differ, seeds are not a partition.
- **distinct** — are the held-out seeds N distinct layouts, and disjoint from the development set?
                 Collisions mean a "50-trial" score is fewer than 50 independent layouts.
- **modes**    — does `--init-mode seeded` differ from `saved`? `saved` replays a fixed bank of
                 recorded states and will happily look varied while ignoring the seed entirely.

Run under the LIBERO virtualenv, on a free GPU:

    CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 .venv-libero/bin/python .../audit_seeding.py ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

CAP_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("LIBERO_CONFIG_PATH", str(CAP_ROOT / ".libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from cap_harness.libero.adapter import LiberoAdapter  # noqa: E402


def free_joints(adapter) -> dict[str, tuple[float, ...]]:
    """Every free-joint xyz in the native sim, keyed by joint name.

    Free joints (`jnt_type == 0`) are the movable objects; everything else is fixture or robot.
    This reaches through the adapter deliberately — it is a diagnostic, not a program.
    """
    # Diagnostic-only simulator inspection; never exposed to task programs.
    inner = adapter._env
    inner = getattr(inner, "env", inner)
    sim = inner.sim
    out = {}
    for index in range(sim.model.njnt):
        if sim.model.jnt_type[index] != 0:
            continue
        addr = sim.model.jnt_qposadr[index]
        out[sim.model.joint_id2name(index)] = tuple(
            round(float(v), 5) for v in sim.data.qpos[addr : addr + 3]
        )
    return out


def signature(adapter) -> str:
    return "|".join(f"{name}:{xyz}" for name, xyz in sorted(free_joints(adapter).items()))


def check_vary(adapter, task: str, seeds: list[int]) -> None:
    rows = {}
    for seed in seeds:
        adapter.reset(task, seed)
        rows[seed] = free_joints(adapter)
    names = sorted(rows[seeds[0]])
    print(f"{'joint':<40} {'distinct':>9} {'spread (mm)':>12}")
    moving = 0
    for name in names:
        values = [rows[seed][name] for seed in seeds]
        distinct = len(set(values))
        spread = (
            max(max(abs(a[k] - b[k]) for k in range(3)) for a in values for b in values) * 1000.0
        )
        moving += distinct > 1
        print(f"{name:<40} {distinct:>9} {spread:>12.1f}")
    print(f"\n{moving} of {len(names)} free joints vary across {len(seeds)} seeds")
    if moving == 0:
        print("FAIL: nothing moves. Every seed is the same layout.")


def check_repeat(adapter, task: str, seeds: list[int]) -> None:
    first = {}
    for seed in seeds:
        adapter.reset(task, seed)
        first[seed] = signature(adapter)
    unstable = []
    for seed in seeds:  # same process, second visit
        adapter.reset(task, seed)
        if signature(adapter) != first[seed]:
            unstable.append(seed)
    for seed in seeds:
        print(f"seed {seed:>3}  {hash(first[seed]) & 0xFFFFFFFF:08x}")
    print(f"\nwithin this process: {'STABLE' if not unstable else f'UNSTABLE on {unstable}'}")
    print(
        "Now run this again in a FRESH process and diff the hashes — equal means the seed->layout"
        " map is reproducible, which is what makes seeds 1-50 a partition rather than a sample."
    )


def check_distinct(adapter, task: str, held_out: list[int], development: list[int]) -> None:
    held = {seed: (adapter.reset(task, seed), signature(adapter))[1] for seed in held_out}
    dev = {seed: (adapter.reset(task, seed), signature(adapter))[1] for seed in development}
    values = list(held.values())
    duplicated = sorted({seed for seed in held if values.count(held[seed]) > 1})
    overlap = sorted({seed for seed in held if held[seed] in set(dev.values())})
    print(f"distinct held-out layouts : {len(set(values))}/{len(held_out)}")
    print(f"duplicated held-out seeds : {duplicated or 'none'}")
    print(f"seeds shared with dev set : {overlap or 'none'}")
    if duplicated or overlap:
        print("FAIL: the held-out set is smaller than it looks, or it leaks into development.")


def check_modes(task: str, seeds: list[int]) -> None:
    signatures = {}
    for mode in ("seeded", "saved"):
        adapter = LiberoAdapter(init_mode=mode)
        signatures[mode] = [(adapter.reset(task, s), signature(adapter))[1] for s in seeds]
        distinct = len(set(signatures[mode]))
        print(f"init_mode={mode:<7} distinct layouts over {len(seeds)} seeds: {distinct}")
        adapter.close()
    if signatures["seeded"] == signatures["saved"]:
        print(
            "\nNOTE: the two modes agree on every seed here. That is possible, but it is also "
            "what you would see if the seed were being ignored — cross-check with `vary`."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("vary", "repeat", "distinct", "modes"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-mode", default="seeded")
    parser.add_argument(
        "--seeds", default="1,2,3,51,52,53", help="seeds for vary/repeat/modes (comma-separated)"
    )
    parser.add_argument("--held-out", default="1-50", help="held-out range for `distinct`")
    parser.add_argument("--development", default="51-70", help="development range for `distinct`")
    args = parser.parse_args()

    def expand(spec: str) -> list[int]:
        if "-" in spec and "," not in spec:
            low, high = spec.split("-")
            return list(range(int(low), int(high) + 1))
        return [int(s) for s in spec.split(",")]

    task = f"{args.suite}/{args.task_id}"
    seeds = expand(args.seeds)

    if args.check == "modes":
        check_modes(task, seeds)
        return 0

    adapter = LiberoAdapter(init_mode=args.init_mode)
    try:
        if args.check == "vary":
            check_vary(adapter, task, seeds)
        elif args.check == "repeat":
            check_repeat(adapter, task, seeds)
        else:
            check_distinct(adapter, task, expand(args.held_out), expand(args.development))
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
