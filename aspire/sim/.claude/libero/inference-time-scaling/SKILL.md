---
name: libero-inference-time-scaling
description: "Run the LIBERO inference-time scaling experiment: debug Long-Pro tasks at frozen library snapshots, evaluate token-budget checkpoints, and plot Pareto curves."
---

# LIBERO Inference-Time Scaling

Use this skill for Experiment 5. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run Order

1. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).
2. Set up snapshot worktrees for the target library sizes.
3. Follow [main-agent-prompt.md](main-agent-prompt.md) for Stage 1 debug + Stage 2 held-out eval.
4. Use [subagent-prompt.md](subagent-prompt.md) for each debug subagent.
5. Follow [token-scaling-coordinator.md](token-scaling-coordinator.md) to select token-budget checkpoints and run Stage 2 variants.
6. Use [clean-task-slate.md](clean-task-slate.md) before rerunning a snapshot or task.

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | End-to-end human/agent entrypoint |
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator recipe for debug+held-out eval |
| [subagent-prompt.md](subagent-prompt.md) | Per-task Stage 1 debug subagent prompt |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist before reruns |
| [token-scaling-coordinator.md](token-scaling-coordinator.md) | Token-budget checkpoint eval recipe |
| [../library-size-scaling/main-agent-prompt.md](../library-size-scaling/main-agent-prompt.md) | Related zero-shot snapshot eval flow |
