"""The batched seed runner must be indistinguishable from a process per seed."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from cap_harness import batch as batch_module
from cap_harness.batch import SeedResult, parse_seed_arguments, run_seeds, summarize
from cap_harness.cli import main

# A program that records what its own process and RNG looked like when it ran.
# Anything leaking between seeds -- a reused environment, an advanced global
# RNG, an inherited module -- shows up as a difference between two seeds' files.
PROBE_PROGRAM = """
result = {"ok": True}
"""


def _fake_run_program(**kwargs):
    """Stand in for the simulator: write the evidence a real run would leave."""
    import os
    import random

    import numpy as np

    seed = kwargs["seed"]
    run_dir = Path(kwargs["output_root"]) / f"{seed:04d}" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outcome.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "pid": os.getpid(),
                # Both are re-seeded per seed, so these must not depend on
                # which seeds this worker already ran.
                "random_draw": random.random(),
                # Exercise the legacy global RNG used by the simulators.
                "numpy_draw": float(np.random.random()),
                "env_id": id(object()),
            }
        )
    )
    print(f"stdout from seed {seed}")
    return type(
        "Outcome",
        (),
        {
            "run_dir": run_dir,
            "program_ok": True,
            "task_success": seed % 2 == 0,
            "termination_reason": "program_completed",
        },
    )()


@pytest.fixture
def program(tmp_path: Path) -> Path:
    path = tmp_path / "policy.py"
    path.write_text(PROBE_PROGRAM, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def stub_run_program(monkeypatch):
    """Patch the run entry point the worker imports, for the in-process path."""
    import cap_harness.run as run_module

    monkeypatch.setattr(run_module, "run_program", _fake_run_program)
    # The counter is per worker process; in-process sweeps share this one.
    monkeypatch.setattr(batch_module, "_SEEDS_RUN", 0)


def _sweep(program: Path, output_root: Path, **kwargs) -> list[SeedResult]:
    return run_seeds(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        program_path=program,
        output_root=output_root,
        init_mode="seeded",
        flat_layout=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (["51-65"], list(range(51, 66))),
        (["1", "2", "7-9"], [1, 2, 7, 8, 9]),
        (["5", "5", "4"], [4, 5]),
        (["3-3"], [3]),
    ],
)
def test_seed_arguments_accept_numbers_and_ranges(text, expected) -> None:
    assert parse_seed_arguments(text) == expected


@pytest.mark.parametrize("text", [["65-51"], ["abc"], ["1-x"], []])
def test_unusable_seed_arguments_are_refused(text) -> None:
    with pytest.raises(ValueError):
        parse_seed_arguments(text)


@pytest.mark.parametrize("benchmark", ["behavior", "yam_real", "yam_sim", "robocasa"])
def test_unsupported_worker_reuse_is_refused(program, tmp_path, benchmark) -> None:
    """Do not reuse Isaac runtimes or offer embodiments absent from this repo."""
    with pytest.raises(ValueError, match=benchmark):
        run_seeds(
            benchmark=benchmark,
            suite="bench",
            task_id=0,
            program_path=program,
            output_root=tmp_path / "out",
            seeds=[1, 2],
        )


def test_every_seed_runs_and_is_reported_in_seed_order(program, tmp_path) -> None:
    results = _sweep(program, tmp_path / "out", seeds=[51, 52, 53], recycle_after=0)

    assert [result.seed for result in results] == [51, 52, 53]
    assert all(result.ok and result.program_ok for result in results)
    assert [result.task_success for result in results] == [False, True, False]
    assert all(result.run_dir is not None for result in results)


def test_a_seed_draws_the_same_numbers_whatever_ran_before_it(program, tmp_path) -> None:
    """Running order must not reach a program.

    A worker is reused across seeds, so an un-reseeded global RNG would hand
    seed 53 a different stream depending on whether it ran first or third --
    a debugging loop that cannot be reproduced by rerunning the seed alone.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    _sweep(program, first, seeds=[51, 52, 53], recycle_after=0)
    _sweep(program, second, seeds=[53], recycle_after=0)

    swept = json.loads((first / "0053" / "run" / "outcome.json").read_text())
    alone = json.loads((second / "0053" / "run" / "outcome.json").read_text())
    assert swept["random_draw"] == alone["random_draw"]
    assert swept["numpy_draw"] == alone["numpy_draw"]


def test_each_seed_keeps_its_own_log_including_process_level_output(program, tmp_path) -> None:
    output_root = tmp_path / "out"
    _sweep(program, output_root, seeds=[51, 52], recycle_after=0)

    for seed in (51, 52):
        log = (output_root / "logs" / f"seed_{seed}.log").read_text()
        assert f"stdout from seed {seed}" in log
        assert f"run: {output_root}" in log
    # One seed's output never lands in another's log.
    assert "seed 52" not in (output_root / "logs" / "seed_51.log").read_text()


def test_stdout_is_restored_after_a_sweep(program, tmp_path, capfd) -> None:
    """The fd redirection must not outlive the seed that asked for it."""
    _sweep(program, tmp_path / "out", seeds=[51], recycle_after=0)
    print("back on the terminal")
    assert "back on the terminal" in capfd.readouterr().out


def test_a_failing_seed_is_recorded_and_the_sweep_continues(program, tmp_path, monkeypatch) -> None:
    """One bad seed must not cost the other forty-nine."""
    import cap_harness.run as run_module

    def explode(**kwargs):
        if kwargs["seed"] == 52:
            raise RuntimeError("simulator died")
        return _fake_run_program(**kwargs)

    monkeypatch.setattr(run_module, "run_program", explode)
    results = _sweep(program, tmp_path / "out", seeds=[51, 52, 53], recycle_after=0)

    assert [result.ok for result in results] == [True, False, True]
    failed = results[1]
    assert failed.run_dir is None
    assert "simulator died" in failed.error
    # The traceback lands in that seed's log, not on the parent's stdout.
    assert "simulator died" in (tmp_path / "out" / "logs" / "seed_52.log").read_text()


def test_results_stream_to_disk_as_they_land(program, tmp_path) -> None:
    """An interrupted sweep keeps the seeds it finished."""
    output_root = tmp_path / "out"
    results_jsonl = tmp_path / "results.jsonl"
    _sweep(
        program,
        output_root,
        seeds=[51, 52],
        recycle_after=0,
        results_jsonl=results_jsonl,
    )

    lines = [json.loads(line) for line in results_jsonl.read_text().splitlines() if line.strip()]
    assert sorted(line["seed"] for line in lines) == [51, 52]
    assert all(line["run_dir"] for line in lines)


def test_worker_seed_index_counts_seeds_within_a_worker(program, tmp_path) -> None:
    """The counter is how a leak across resets becomes visible in the results."""
    results = _sweep(program, tmp_path / "out", seeds=[51, 52, 53], recycle_after=0)
    assert [result.worker_seed_index for result in results] == [0, 1, 2]
    assert len({result.worker_pid for result in results}) == 1


def test_summary_counts_benchmark_success_not_clean_exits(tmp_path) -> None:
    results = [
        SeedResult(1, True, "a", True, True, "task_succeeded", 1.0, 1, 0, 100.0, None),
        SeedResult(2, True, "b", False, False, "program_error", 1.0, 1, 1, 100.0, None),
        SeedResult(3, False, None, None, None, None, 1.0, 1, 2, 100.0, "boom"),
    ]
    summary = summarize(results)

    assert summary["seeds"] == 3
    assert summary["runs"] == 2
    assert summary["missing"] == [3]
    assert summary["passes"] == 1
    # Over every seed asked for, not just those that produced a run. Dividing
    # by the survivors let a sweep where a third of the seeds crashed and every
    # survivor passed report a perfect rate.
    assert summary["pass_rate"] == pytest.approx(1 / 3)
    assert summary["program_errors"] == [2]


def test_cli_sweep_writes_a_summary_and_exits_nonzero_on_a_missing_run(
    program, tmp_path, monkeypatch
) -> None:
    import cap_harness.run as run_module

    def explode(**kwargs):
        if kwargs["seed"] == 52:
            raise RuntimeError("simulator died")
        return _fake_run_program(**kwargs)

    monkeypatch.setattr(run_module, "run_program", explode)
    output_root = tmp_path / "out"
    code = main(
        [
            "run-batch",
            "--suite",
            "libero_object_swap",
            "--task-id",
            "0",
            "--seeds",
            "51-53",
            "--program",
            str(program),
            "--output-root",
            str(output_root),
            "--recycle-after",
            "0",
            "--init-mode",
            "seeded",
            "--flat-run-dir",
        ]
    )

    assert code == 1
    summary = json.loads((output_root / "batch.json").read_text())
    assert summary["seeds"] == 3
    assert summary["runs"] == 2
    assert summary["missing"] == [52]
    assert summary["workers"] == 1
    assert "elapsed_s" in summary


def test_the_default_recycle_bound_is_documented_in_help() -> None:
    """The flag exists to bound a driver-side leak; a reader must be able to find it."""
    assert batch_module.DEFAULT_RECYCLE_AFTER > 0


# --- the worker-pool path: real spawning, real recycling ---------------------

FAKE_RUNNER = "batch_fake_runner:run_program"


def _pool_sweep(program: Path, output_root: Path, **kwargs) -> list[SeedResult]:
    return run_seeds(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        program_path=program,
        output_root=output_root,
        init_mode="seeded",
        flat_layout=True,
        runner=FAKE_RUNNER,
        **kwargs,
    )


def test_spawned_workers_run_every_seed_and_report_in_order(program, tmp_path) -> None:
    output_root = tmp_path / "out"
    results = _pool_sweep(program, output_root, seeds=[51, 52, 53, 54], workers=2, recycle_after=0)

    assert [result.seed for result in results] == [51, 52, 53, 54]
    assert all(result.ok for result in results)
    for seed in (51, 52, 53, 54):
        assert (output_root / f"{seed:04d}" / "run" / "outcome.json").is_file()
        assert f"stdout from seed {seed}" in (output_root / "logs" / f"seed_{seed}.log").read_text()


def test_work_actually_lands_on_more_than_one_process(program, tmp_path) -> None:
    results = _pool_sweep(program, tmp_path / "out", seeds=list(range(51, 59)), workers=2)
    assert len({result.worker_pid for result in results}) > 1


def test_a_spawned_worker_is_retired_after_its_recycle_bound(program, tmp_path) -> None:
    """Retire each worker when it reaches its recycle bound.

    The bound is the mitigation for a driver-side leak across resets; it has
    to actually retire a process, not just be accepted as a flag.
    """
    results = _pool_sweep(
        program, tmp_path / "out", seeds=list(range(51, 57)), workers=1, recycle_after=2
    )

    assert [result.worker_seed_index for result in results] == [0, 1] * 3
    assert len({result.worker_pid for result in results}) == 3


def test_a_seed_is_identical_whether_swept_or_run_alone(program, tmp_path) -> None:
    """Keep per-seed RNG draws identical in sweeps and standalone runs.

    Across real processes, not just in one: a batched seed and a lone seed
    must produce the same draws, or a fix verified in a sweep is not the fix
    that gets validated.
    """
    swept_root = tmp_path / "swept"
    alone_root = tmp_path / "alone"
    _pool_sweep(program, swept_root, seeds=[51, 52, 53], workers=1, recycle_after=0)
    _pool_sweep(program, alone_root, seeds=[53], workers=1, recycle_after=0)

    swept = json.loads((swept_root / "0053" / "run" / "outcome.json").read_text())
    alone = json.loads((alone_root / "0053" / "run" / "outcome.json").read_text())
    assert swept["random_draw"] == alone["random_draw"]
    assert swept["numpy_draw"] == alone["numpy_draw"]


def test_a_worker_that_dies_does_not_strand_the_sweep(program, tmp_path) -> None:
    """A seed whose runner cannot even be imported is reported, not hung on."""
    results = run_seeds(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        program_path=program,
        output_root=tmp_path / "out",
        seeds=[51, 52],
        workers=2,
        recycle_after=0,
        runner="cap_harness_no_such_module:run_program",
    )

    assert [result.ok for result in results] == [False, False]
    assert all("ModuleNotFoundError" in result.error for result in results)


def test_recycling_bounds_each_worker_with_uneven_jobs(program, tmp_path) -> None:
    results = run_seeds(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        program_path=program,
        output_root=tmp_path / "out",
        seeds=list(range(51, 59)),
        workers=2,
        recycle_after=2,
        max_steps=100,
        runner="batch_fake_runner:run_uneven_program",
    )
    assert [result.seed for result in results] == list(range(51, 59))
    assert all(result.ok for result in results)
    assert all(result.worker_seed_index < 2 for result in results)
    assert len({result.worker_pid for result in results}) == 4


def test_recycling_recovers_from_a_dead_worker_in_the_next_generation(program, tmp_path) -> None:
    results = run_seeds(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        program_path=program,
        output_root=tmp_path / "out",
        seeds=list(range(51, 59)),
        workers=2,
        recycle_after=2,
        max_steps=100,
        runner="batch_fake_runner:run_one_crashing_program",
    )
    assert [result.seed for result in results] == list(range(51, 59))
    assert not next(result for result in results if result.seed == 52).ok
    assert all(result.ok for result in results if result.seed >= 55)


def test_a_worker_dying_abruptly_does_not_hang_the_sweep(program, tmp_path) -> None:
    """Finish the sweep even when a worker dies abruptly.

    The failure `recycle_after` exists to bound, and the only one that can
    escape a worker at all, since `_run_seed` catches everything else.

    `multiprocessing.Pool.imap_unordered` never yields a result for a task
    whose worker died, so the sweep blocked forever: in the pipeline that holds
    a GPU while the coordinator waits for a notification that never arrives.
    """
    results = run_seeds(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        program_path=program,
        output_root=tmp_path / "out",
        seeds=[51, 52, 53],
        workers=2,
        recycle_after=0,
        runner="batch_suicidal_runner:run_program",
    )

    # Every requested seed is accounted for, including the one that took its
    # worker down -- rather than the call never returning.
    assert [result.seed for result in results] == [51, 52, 53]
    assert all(not result.ok for result in results)
    died = next(result for result in results if result.seed == 52)
    assert "worker died" in died.error or "process" in died.error.lower()


def test_a_worker_outliving_a_killed_parent_is_reclaimed_by_the_kernel(tmp_path) -> None:
    """A worker blocked in native code must not survive the parent being SIGKILLed.

    `ProcessPoolExecutor` shuts workers down on every exit the parent survives, and notices a dead
    parent only at the next `call_queue.get()`. A worker inside MuJoCo or the EGL driver never
    reaches that call, so a SIGKILLed parent used to strand it: one campaign left four alive for
    30 hours, each holding ~1.4 GB and a GPU context.

    The worker here sleeps inside libc, which is the same shape -- blocked outside Python's queue
    loop -- without needing a simulator. A zombie counts as reclaimed: it has already died and is
    waiting for init to reap it, holding no memory.
    """
    child = tmp_path / "parent.py"
    child.write_text(
        "import ctypes, multiprocessing, sys, time\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        f"sys.path.insert(0, {str(Path(batch_module.__file__).parents[2])!r})\n"
        "from cap_harness.batch import _die_with_parent\n"
        "def blocked(_):\n"
        "    ctypes.CDLL('libc.so.6').sleep(600)\n"
        "if __name__ == '__main__':\n"
        "    ctx = multiprocessing.get_context('spawn')\n"
        "    with ProcessPoolExecutor(2, mp_context=ctx, initializer=_die_with_parent) as pool:\n"
        "        pool.submit(blocked, 0)\n"
        "        pool.submit(blocked, 1)\n"
        "        time.sleep(600)\n"
    )
    parent = subprocess.Popen([sys.executable, str(child)], start_new_session=True)
    try:
        time.sleep(8)  # let both workers reach the libc sleep
        os.kill(parent.pid, signal.SIGKILL)
        time.sleep(3)
        found = subprocess.run(
            ["pgrep", "-g", str(parent.pid)], check=False, capture_output=True, text=True
        ).stdout.split()
        live = []
        for pid in found:
            if pid == str(parent.pid):
                continue
            try:
                if Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z":
                    live.append(pid)
            except (FileNotFoundError, IndexError):
                pass
        for pid in found:
            subprocess.run(["kill", "-9", pid], check=False, capture_output=True)
    finally:
        parent.poll()
    assert live == [], f"orphaned workers survived the parent: {live}"
