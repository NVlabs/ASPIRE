---
name: libero-fix-loop
description: "Run the baseline-free LIBERO-Pro Fix Loop: inspect one initial observed scene, generate task-level code, debug failures on seeds 51–65 using traces and keyframes, validate on seeds 1–50, and promote reusable patterns."
---

# LIBERO Fix Loop

Use this skill for Experiment 1. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run Order

1. Follow [main-agent-prompt.md](main-agent-prompt.md) as coordinator.
2. Fill [subagent-prompt.md](subagent-prompt.md) once per task.
3. Each worker follows [skills/task-exploration.md](skills/task-exploration.md), generates its own initial code, then uses the original failure-by-failure debug loop.
4. Use [clean-task-slate.md](clean-task-slate.md) before reruns.
5. Promote held-out-supported patterns into [../skills/](../skills/).

No external baseline code or baseline output directory is used.
