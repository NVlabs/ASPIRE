"""A runner whose worker dies abruptly, for the pool-death test.

`multiprocessing.Pool.imap_unordered` blocks forever on this; the test that
used to claim coverage of it raised an ordinary exception inside the worker
instead, which `_run_seed` catches, so no process ever died.
"""

from __future__ import annotations

import os


def run_program(**kwargs: object) -> object:
    seed = int(kwargs["seed"])  # type: ignore[arg-type]
    if seed == 52:
        os._exit(1)  # abrupt: no traceback, no result, no chance to report
    raise RuntimeError(f"seed {seed} declined to run")
