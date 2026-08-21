# LIBERO Zero-Shot Transfer Instructions

1. Complete the LIBERO continuation setup in the root `README.md`.
2. Export `ASPIRE_ROOT`, `MUJOCO_GL=egl`, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
3. Start a persistent coordinator session, for example `tmux new -s aspire-libero`.
4. In the coordinator session, read `../CLAUDE.md`, this file, and [SKILL.md](SKILL.md).
5. Follow [main-agent-prompt.md](main-agent-prompt.md) to build the LIBERO-90 skill library in chunks.
6. Fill [subagent-prompt.md](subagent-prompt.md) once per LIBERO-90 task.
7. Commit and tag snapshots using [commit-convention.md](commit-convention.md).
8. Run transfer eval through [../library-size-scaling/INSTRUCTIONS.md](../library-size-scaling/INSTRUCTIONS.md).
