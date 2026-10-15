"""Run one program over many seeds without paying for a process per seed.

The pipeline's debug and validation loops are shell ``for`` loops around
``cap-harness run``: fifteen or fifty interpreter starts, each re-importing the
simulator stack before it can step anything. This module keeps that cost per
*worker* instead of per *seed*, and lets several workers run at once.

What it deliberately does not share is the environment. Every seed still calls
:func:`cap_harness.run.run_program`, which builds its own registry, adapter,
simulator environment, and recorder exactly as a standalone ``cap-harness run``
does, and tears them down before the next seed starts. A worker reuses the
*imports*, never a live scene. Three further rules keep a seed's result
independent of what its worker ran before it:

* Workers are **spawned**, never forked, so no CUDA, EGL, or MuJoCo state is
  inherited from the parent or shared between workers.
* Each seed re-seeds the interpreter-global ``random`` and ``numpy.random``
  streams from its own seed, so a program that draws from them sees the same
  numbers whatever ran earlier in that worker.
* A worker is retired after ``recycle_after`` seeds. Repeatedly building and
  closing GL contexts in one process is the classic place for a driver-side
  leak, and a bounded worker lifetime bounds the exposure.

Only LIBERO-Pro and Robosuite support worker reuse. BEHAVIOR tears down its
Isaac runtime on close and must keep using one fresh process per episode.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import contextlib
import ctypes
from dataclasses import asdict, dataclass
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from typing import Any

PR_SET_PDEATHSIG = 1

#: Seeds run by this worker process so far. A fresh process starts at zero, so
#: a result carrying a high index is one that ran on a long-lived worker --
#: which is what a leak, if there is one, would correlate with.
_SEEDS_RUN = 0

DEFAULT_RECYCLE_AFTER = 10


@dataclass(frozen=True, slots=True)
class SeedResult:
    """One seed's outcome, as the batch parent recorded it."""

    seed: int
    ok: bool
    run_dir: str | None
    program_ok: bool | None
    task_success: bool | None
    termination_reason: str | None
    wall_s: float
    worker_pid: int
    worker_seed_index: int
    max_rss_mb: float
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_seed_arguments(values: Iterable[str]) -> list[int]:
    """Accept ``51 52 53`` and ``51-65`` alike, so a shell loop maps over directly."""
    seeds: list[int] = []
    for value in values:
        text = str(value).strip()
        start, separator, end = text.partition("-")
        if separator and start.strip().isdigit() and end.strip().isdigit():
            low, high = int(start), int(end)
            if high < low:
                raise ValueError(f"seed range must not run backwards: {text}")
            seeds.extend(range(low, high + 1))
            continue
        try:
            seeds.append(int(text))
        except ValueError as exc:
            raise ValueError(f"not a seed or seed range: {text!r}") from exc
    ordered = sorted(set(seeds))
    if not ordered:
        raise ValueError("no seeds requested")
    if ordered[0] < 0:
        raise ValueError("seeds must be non-negative")
    return ordered


#: What a worker actually calls. A dotted ``module:attribute`` string rather
#: than a function object because a spawned worker resolves it after import,
#: and because a test needs to substitute one without the simulator installed.
DEFAULT_RUNNER = "cap_harness.run:run_program"


def _resolve(target: str) -> Any:
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"runner must be 'module:attribute', got {target!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _max_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _die_with_parent() -> None:
    """Ask the kernel to kill this worker if the parent goes away.

    ``ProcessPoolExecutor`` shuts its workers down cleanly on every exit the parent survives --
    normal return, exception, KeyboardInterrupt -- because ``__exit__`` calls ``shutdown``. It
    cannot help when the parent is SIGKILLed or its terminal dies, and then a worker only notices
    on its next ``call_queue.get()``. A worker blocked inside MuJoCo or the EGL driver never gets
    there.

    That is not hypothetical: this campaign left four workers alive for **30 hours**, reparented to
    init, each holding ~1.4 GB and a GPU context, sleeping in ``futex_wait_queue_me`` with 64
    native threads apiece. Nothing in userspace reclaims those, which is why this asks the kernel.

    ``PR_SET_PDEATHSIG`` is Linux-only and fires on the death of the *thread* that spawned us, so
    it is exactly right for a pool whose workers are spawned from the main thread and useless
    anywhere else -- hence the platform guard rather than a hard requirement.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        return  # no prctl: fall back to the pre-existing behaviour rather than refusing to run
    # Close the race the flag cannot: if the parent died between spawn and here, the signal has
    # already been missed and this worker would inherit the leak it exists to prevent.
    if os.getppid() == 1:
        os._exit(1)


def _died(job: dict[str, Any], reason: str) -> dict[str, Any]:
    """A result for a seed whose worker died before it could report one."""
    return SeedResult(
        seed=int(job["seed"]),
        ok=False,
        run_dir=None,
        program_ok=None,
        task_success=None,
        termination_reason=None,
        wall_s=0.0,
        worker_pid=0,
        worker_seed_index=-1,
        max_rss_mb=0.0,
        error=reason,
    ).to_dict()


def _run_seed(job: dict[str, Any]) -> dict[str, Any]:
    """Run one seed in this worker. Never raises: a bad seed must not end the sweep."""
    global _SEEDS_RUN

    seed = int(job["seed"])
    seed_index, _SEEDS_RUN = _SEEDS_RUN, _SEEDS_RUN + 1
    log_path = Path(job["log_path"])
    started = time.monotonic()
    run_dir: str | None = None
    program_ok: bool | None = None
    task_success: bool | None = None
    termination_reason: str | None = None
    error: str | None = None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, _captured_output(log):
        try:
            # Import inside the worker: the parent process never loads the
            # simulator stack, and this is the cost being amortized.
            import random

            import numpy as np

            run_program = _resolve(job["runner"])

            # Make the seed, not the running order, decide any global draw.
            random.seed(seed)
            # Seed the simulator's legacy global RNG, not a separate generator.
            np.random.seed(seed)

            outcome = run_program(
                benchmark=job["benchmark"],
                suite=job["suite"],
                task_id=int(job["task_id"]),
                seed=seed,
                program_path=Path(job["program_path"]),
                output_root=Path(job["output_root"]),
                max_steps=int(job["max_steps"]),
                camera_width=int(job["camera_width"]),
                camera_height=int(job["camera_height"]),
                init_mode=job["init_mode"],
                flat_layout=bool(job["flat_layout"]),
                capture_videos=bool(job["capture_videos"]),
            )
            run_dir = str(outcome.run_dir)
            program_ok = bool(outcome.program_ok)
            task_success = outcome.task_success
            termination_reason = outcome.termination_reason
            # Worker protocol emits the run directory to its captured output.
            print(f"run: {run_dir}", flush=True)
        except (KeyboardInterrupt, SystemExit):
            # Not this seed's failure. Filing it as one made Ctrl-C on a
            # fifteen-seed sweep take fifteen Ctrl-Cs, each writing a bogus
            # recorded failure before the loop moved to the next seed.
            raise
        # Record worker failures; KeyboardInterrupt and SystemExit are re-raised above.
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    return SeedResult(
        seed=seed,
        ok=run_dir is not None,
        run_dir=run_dir,
        program_ok=program_ok,
        task_success=task_success,
        termination_reason=termination_reason,
        wall_s=round(time.monotonic() - started, 3),
        worker_pid=os.getpid(),
        worker_seed_index=seed_index,
        max_rss_mb=_max_rss_mb(),
        error=error,
    ).to_dict()


@contextlib.contextmanager
def _captured_output(stream: Any) -> Iterator[None]:
    """Point fds 1 and 2 at a log file, so the simulator's C-level output lands there too.

    ``contextlib.redirect_stdout`` rebinds a Python name; MuJoCo, EGL, and
    ffmpeg write to the file descriptor. Rebinding both is what makes a batched
    seed's log the same artifact its own process produced: the dup2 catches the
    simulator, and the redirect catches Python code that holds its own
    reference to ``sys.stdout``.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        stream.flush()
        for descriptor, original in zip((1, 2), saved, strict=False):
            os.dup2(original, descriptor)
            os.close(original)


def _task_horizon(benchmark: str, suite: str, task_id: int) -> int:
    """The episode length the task's own benchmark states, else the flat default."""
    from cap_harness.run import DEFAULT_MAX_STEPS

    if benchmark == "robosuite":
        from cap_harness.robosuite.registry import RobosuiteTaskRegistry

        stated = RobosuiteTaskRegistry().resolve(suite, task_id).horizon
    else:
        stated = None
    return int(stated) if stated else DEFAULT_MAX_STEPS


def run_seeds(
    *,
    benchmark: str,
    suite: str,
    task_id: int,
    program_path: Path,
    output_root: Path,
    seeds: Sequence[int],
    workers: int = 1,
    recycle_after: int = DEFAULT_RECYCLE_AFTER,
    max_steps: int | None = None,
    camera_width: int = 800,
    camera_height: int = 512,
    init_mode: str = "saved",
    flat_layout: bool = False,
    capture_videos: bool = True,
    log_dir: Path | None = None,
    results_jsonl: Path | None = None,
    on_result: Any = None,
    runner: str = DEFAULT_RUNNER,
) -> list[SeedResult]:
    """Run ``program_path`` on every seed, returning one record per seed.

    Results stream to ``results_jsonl`` as they land, so an interrupted sweep
    leaves the seeds it finished on disk rather than nothing at all.
    """
    if benchmark not in {"libero-pro", "robosuite"}:
        raise ValueError(
            f"run-batch does not accept {benchmark}: only libero-pro and robosuite "
            "support worker reuse. Use a fresh `cap-harness run` process for each episode."
        )
    if init_mode not in {"saved", "seeded"}:
        raise ValueError("init_mode must be 'saved' or 'seeded'")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if recycle_after < 0:
        raise ValueError("recycle_after must be non-negative")
    if not seeds:
        raise ValueError("no seeds requested")
    if not program_path.is_file():
        raise ValueError(f"program does not exist: {program_path}")

    logs = log_dir if log_dir is not None else output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if results_jsonl is not None:
        results_jsonl.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: this file is the record of *this* sweep. Appending left a
        # re-run of the same version with two rows per seed, and the documented
        # scoring command then reports "passed 12 of 30" for fifteen seeds.
        results_jsonl.write_text("")

    # One horizon for the whole sweep, resolved once so that every seed is scored
    # against the same budget and the manifest records which one. A sweep whose
    # seeds silently used different episode lengths would not be comparable.
    if max_steps is None:
        resolved_max_steps = _task_horizon(benchmark, suite, task_id)
    else:
        resolved_max_steps = int(max_steps)

    jobs = [
        {
            "seed": int(seed),
            "benchmark": benchmark,
            "suite": suite,
            "task_id": int(task_id),
            "program_path": str(program_path),
            "output_root": str(output_root),
            "max_steps": int(resolved_max_steps),
            "camera_width": int(camera_width),
            "camera_height": int(camera_height),
            "init_mode": init_mode,
            "flat_layout": bool(flat_layout),
            "capture_videos": bool(capture_videos),
            "log_path": str(logs / f"seed_{int(seed):02d}.log"),
            "runner": runner,
        }
        for seed in seeds
    ]

    collected: list[SeedResult] = []

    def record(payload: dict[str, Any]) -> None:
        result = SeedResult(**payload)
        collected.append(result)
        if results_jsonl is not None:
            with results_jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if on_result is not None:
            on_result(result)

    if workers == 1 and recycle_after == 0:
        # One worker that never retires is just this process doing the work --
        # no pool, no pickling, and a traceback the caller can actually see.
        for job in jobs:
            record(_run_seed(job))
        return collected

    # spawn: a forked child would inherit CUDA/EGL state from the parent, which
    # is exactly the cross-seed contamination this module exists to avoid.
    context = multiprocessing.get_context("spawn")
    # ...and spawn is also why the workers need _die_with_parent: they hold a
    # GPU context and ~1.4 GB each, and nothing else reclaims them.
    pool_size = min(workers, len(jobs))
    # Deliberately not multiprocessing.Pool. A worker that dies abruptly -- a
    # MuJoCo or EGL segfault, an OOM kill -- never returns a result, and Pool's
    # imap_unordered then blocks forever: the sweep holds its GPU and the
    # coordinator, told to go idle until notified, is never notified. That is
    # precisely the driver-side crash `recycle_after` exists to bound, and
    # since _run_seed catches everything else, it is the only failure that can
    # escape a worker at all. ProcessPoolExecutor reports it as
    # BrokenProcessPool instead. It has no max_tasks_per_child before 3.11, so
    # retirement comes from running each generation under fresh executors.
    # With recycling, give each worker its own queue of at most recycle_after
    # jobs. A shared queue lets a fast worker consume the whole generation
    # while another is still running its first seed, exceeding the limit.
    generation = pool_size * recycle_after if recycle_after else len(jobs)
    for start in range(0, len(jobs), generation):
        batch = jobs[start : start + generation]
        with contextlib.ExitStack() as stack:
            queues = (
                [batch[index::pool_size] for index in range(pool_size)]
                if recycle_after
                else [batch]
            )
            futures = {}
            for queue in queues:
                if not queue:
                    continue
                pool = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=1 if recycle_after else pool_size,
                        mp_context=context,
                        initializer=_die_with_parent,
                    )
                )
                for job in queue:
                    try:
                        futures[pool.submit(_run_seed, job)] = job
                    except BrokenProcessPool as exc:
                        record(_died(job, f"worker died: {type(exc).__name__}: {exc}"))
            for future in as_completed(futures):
                job = futures[future]
                try:
                    record(future.result())
                except BrokenProcessPool as exc:
                    # The worker died mid-seed. Record it and keep going; the
                    # next generation gets a fresh executor.
                    record(_died(job, f"worker died: {type(exc).__name__}: {exc}"))
    collected.sort(key=lambda result: result.seed)
    return collected


def summarize(results: Sequence[SeedResult]) -> dict[str, Any]:
    """Aggregate a sweep. ``passes`` counts benchmark success, not clean exits."""
    finished = [result for result in results if result.ok]
    passes = sum(1 for result in finished if result.task_success)
    return {
        "seeds": len(results),
        "runs": len(finished),
        "missing": [result.seed for result in results if not result.ok],
        "passes": passes,
        # Over every seed asked for, not just those that produced a run: a
        # sweep where five of fifteen crashed and ten passed is 10/15, not 1.0.
        "pass_rate": round(passes / len(results), 6) if results else 0.0,
        "program_errors": [result.seed for result in finished if result.program_ok is False],
        # Summed per-seed time, not the sweep's elapsed time: with several
        # workers the sweep finishes in a fraction of this.
        "seed_wall_s": round(sum(result.wall_s for result in results), 3),
        "results": [result.to_dict() for result in results],
    }


__all__ = [
    "DEFAULT_RECYCLE_AFTER",
    "DEFAULT_RUNNER",
    "SeedResult",
    "parse_seed_arguments",
    "run_seeds",
    "summarize",
]
