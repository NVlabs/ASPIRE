# Clean Task Slate: Evolutionary Search

Use this before rerunning a task or suite.

1. Confirm the Fix Loop `fix_code.py` baseline for the task is present
   (`outputs/runs/LATEST/<benchmark>/<suite>/task_<id>/fix_code.py`, or the legacy
   `outputs/iterative_debugging/<benchmark>/<suite>/task_<id>/fix_code.py`).
2. Decide whether previous Evolutionary Search candidates, summaries, and validation outputs
   are historical records or should be superseded (use the `_rerun` directory overrides in the
   subagent template for a clean rerun).
3. Check target GPU availability before launching a new candidate batch.
4. Verify perception services on ports 8114–8116 or let the coordinator preflight start them.
5. Keep Evolutionary Search debug seeds (51–65) separate from the final held-out eval
   (seeds 1–50), and keep `--init-mode seeded` on every run.
6. Do not promote candidate-specific tricks into shared skills unless they pass the coordinator
   review criteria.
