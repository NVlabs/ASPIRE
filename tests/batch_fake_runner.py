"""A ``run_program`` stand-in a spawned batch worker can import.

The worker-pool path is the reason :mod:`cap_harness.batch` exists, and a
``monkeypatch`` cannot reach a spawned child. This module is importable by
name, so a test can point ``run_seeds(runner=...)`` at it and exercise the real
pool: real spawning, real pickling, real worker recycling.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class _Outcome:
    run_dir: Path
    program_ok: bool
    task_success: bool | None
    termination_reason: str


def run_program(**kwargs: object) -> _Outcome:
    """Write the evidence a real run would leave, without a simulator."""
    seed = int(kwargs["seed"])  # type: ignore[arg-type]
    run_dir = Path(str(kwargs["output_root"])) / f"{seed:04d}" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outcome.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "pid": os.getpid(),
                "random_draw": random.random(),
                # Exercise the legacy global RNG used by the simulators.
                "numpy_draw": float(np.random.random()),
            }
        ),
        encoding="utf-8",
    )
    print(f"stdout from seed {seed}")
    return _Outcome(
        run_dir=run_dir,
        program_ok=True,
        task_success=seed % 2 == 0,
        termination_reason="program_completed",
    )


def run_uneven_program(**kwargs: object) -> _Outcome:
    """Leave one worker busy while the other can finish several seeds."""
    if int(kwargs["seed"]) == 51:
        time.sleep(2)
    return run_program(**kwargs)


def run_one_crashing_program(**kwargs: object) -> _Outcome:
    """Crash one seed while allowing later generations to finish normally."""
    if int(kwargs["seed"]) == 52:
        os._exit(1)
    return run_program(**kwargs)
