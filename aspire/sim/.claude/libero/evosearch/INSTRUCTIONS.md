# LIBERO Evolutionary Search Instructions

1. Complete the LIBERO continuation setup in the root `README.md`.
2. Confirm Fix Loop outputs exist for the target tasks.
3. Export `ASPIRE_ROOT`, `MUJOCO_GL=egl`, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
4. Start a persistent coordinator session, for example `tmux new -s aspire-libero`.
5. In the coordinator session, read `../CLAUDE.md`, this file, and [SKILL.md](SKILL.md).
6. Follow [main-agent-prompt.md](main-agent-prompt.md) to select tasks below threshold and dispatch Evolutionary Search subagents.
7. Fill [subagent-prompt.md](subagent-prompt.md) once per task and use [skills/](skills/) for Evolutionary Search-specific candidate guidance.
8. Run validation selection with `scripts/libero/run_validation_comparison.py`.
9. Generate comparison plots with `scripts/libero/plot_baseline_vs_fix_v3_evosearch.py`, `scripts/libero/plot_method_comparison_v5.py`, and `scripts/libero/plot_evosearch_progress.py` as needed.
