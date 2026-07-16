# LIBERO Library-Size Scaling Instructions

1. Complete the LIBERO continuation setup in the root `README.md`.
2. Confirm zero-shot transfer snapshots exist (`snapshot-N0`, `snapshot-N5`, ..., `snapshot-N90`).
3. Export `ASPIRE_ROOT`, `MUJOCO_GL=egl`, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
4. Start a persistent coordinator session, for example `tmux new -s aspire-libero`.
5. In the coordinator session, read `../CLAUDE.md`, this file, and [SKILL.md](SKILL.md).
6. Follow [main-agent-prompt.md](main-agent-prompt.md) to set up snapshot worktrees, dispatch codegen subagents, and run seed execution.
7. Fill [subagent-prompt.md](subagent-prompt.md) once per snapshot/suite/task codegen assignment.
8. Generate tables and plots with `scripts/libero/make_snapshot_table.py`, `scripts/libero/plot_checkpoint_sr_pareto.py`, and the related plotting helpers under `scripts/` as needed.
