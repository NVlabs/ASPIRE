---
name: robosuite/evosearch/INSTRUCTIONS
description: End-to-end entrypoint for Robosuite Fix Loop + Evolutionary Search.
---

# Robosuite Evolutionary Search Instructions

1. Complete the base and Robosuite setup in [`../../../README.md`](../../../README.md).
2. Confirm the two tracked five-iteration Fix Loop baselines exist:
   `baselines/robosuite_nut_assembly_fix.py` (9/100 held out) and
   `baselines/robosuite_two_arm_lift_fix.py` (approximately 70/100 held out).
3. Export `ASPIRE_ROOT`, `PYTHON_ROOT`, `MUJOCO_GL=egl`, and
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
4. Start a persistent coordinator session, for example
   `tmux new -s aspire-robosuite-evo`.
5. Read [`../CLAUDE.md`](../CLAUDE.md), this file, and [SKILL.md](SKILL.md).
6. Before launching, follow the preflight gate in
   [main-agent-prompt.md](main-agent-prompt.md): report tasks, seed partitions,
   trial count/runtime, GPU mapping, credentials, services, and output paths,
   then wait for explicit user confirmation.
7. Follow [main-agent-prompt.md](main-agent-prompt.md) to dispatch one
   Evolutionary Search subagent per confirmed target task.
8. Fill [subagent-prompt.md](subagent-prompt.md) once per task. Stage 1 searches
   K=8 candidates on development seeds 101-125; Stage 2 evaluates only the
   selected program on held-out seeds 1-100.

Evolutionary Search reuses the existing Robosuite traced configs,
`scripts/robosuite/replay_trial_robosuite.py`, the two tracked five-iteration
programs, and the Fix Loop skill snapshot. It does not require a new simulator
runner or the 25 development-seed result directories from the Fix Loop run.
