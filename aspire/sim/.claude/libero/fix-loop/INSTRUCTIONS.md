# LIBERO Fix Loop Instructions

1. Complete LIBERO setup in `README.md`, including the dedicated environment, SAM3 authentication, and `~/.libero/config.yaml`.
2. Export `ASPIRE_ROOT`, `PYTHON_ROOT`, `MUJOCO_GL=egl`, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
3. Start perception servers and a persistent coordinator session.
4. Read `../CLAUDE.md`, [SKILL.md](SKILL.md), and [main-agent-prompt.md](main-agent-prompt.md).
5. Generate progress with `PYTHONPATH="$PYTHON_ROOT" .venv-libero/bin/python3 scripts/libero/gen_progress.py`.
6. Dispatch [subagent-prompt.md](subagent-prompt.md) once per `pending` task. Workers explore, generate initial code, and debug seeds 51–65 without reading external baseline outputs. The coordinator (not workers) runs the Stage 2 held-out evaluation (seeds 1–50) for each `stage1-done` task.
7. Continue until every task has all 50 held-out results, then promote validated findings and update docs.
