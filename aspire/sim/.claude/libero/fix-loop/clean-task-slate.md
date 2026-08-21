# Clean Task Slate: Fix Loop

Use this before rerunning a task or suite.

1. Confirm the suite, task, stage, and seeds being rerun and whether outputs should be archived.
2. Check for active replay/eval processes on the target GPU.
3. Regenerate `docs/progress/fix_loop_progress.md`.
4. Verify perception servers on ports 8114–8116.
5. Keep development seeds 51–65 separate from held-out seeds 1–50.
6. Do not read external baseline code or outputs.
7. Subagents write task artifacts and findings; only the coordinator edits shared skills.
