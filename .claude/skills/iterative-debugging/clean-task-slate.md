# Clean Task Slate: Iterative Debugging

Use this before rerunning a task or suite.

1. Confirm the benchmark, suite, task id, stage, and seeds being rerun and whether outputs should
   be archived. A task is fully self-contained in `$RUN_ROOT/<benchmark>/<suite>/task_<id>/` —
   archiving or deleting that one folder is a complete clean slate for the task.
2. Check for active `cap-harness run` or eval processes on the target GPU.
3. Regenerate `$RUN_ROOT/progress.md` with `scripts/gen_progress.py`.
4. Verify perception and planning services on ports 8114–8116 and 8118 (cuRobo).
5. Keep the three seed bands configured in `campaign.json` separate, and keep
   `--init-mode $INIT_MODE` (the benchmark's own, from the Benchmark Profiles table) on every
   run: `dev_seeds` are debugged against (LIBERO 51–65, Robosuite 101–125 by default),
   `unseen_seeds` are measured once at the end of Stage 1 and never tuned against (LIBERO 66–80,
   Robosuite 126–130 by default), and the held-out partition is reserved for the coordinator's
   Stage 2 (LIBERO 1–50, Robosuite 1–100 by default). A rerun that debugs against the unseen-check
   seeds has spent the only cheap evidence that a program generalizes; archive the task folder and
   start clean rather than carrying a used band forward.
6. Do not read external baseline code or outputs.
7. Subagents write task artifacts and findings; only the coordinator edits shared skills.
