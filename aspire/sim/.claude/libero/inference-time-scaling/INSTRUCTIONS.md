# LIBERO Inference-Time Scaling Instructions

1. Complete the LIBERO continuation setup in the root `README.md`.
2. Confirm target snapshot worktrees are available.
3. Export `ASPIRE_ROOT`, `MUJOCO_GL=egl`, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
4. Start a persistent coordinator session, for example `tmux new -s aspire-libero`.
5. In the coordinator session, read `../CLAUDE.md`, this file, and [SKILL.md](SKILL.md).
6. Follow [main-agent-prompt.md](main-agent-prompt.md) for Stage 1 debug and Stage 2 held-out eval.
7. Fill [subagent-prompt.md](subagent-prompt.md) once per debug assignment.
8. Follow [token-scaling-coordinator.md](token-scaling-coordinator.md) for token-budget checkpoint evals.
9. Generate Pareto plots with `scripts/libero/plot_checkpoint_sr_pareto.py`.
