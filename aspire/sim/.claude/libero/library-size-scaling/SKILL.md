---
name: libero-library-size-scaling
description: "Run the LIBERO library-size scaling experiment: evaluate frozen snapshot skill libraries on LIBERO-Long-Pro and generate success-rate scaling tables/plots."
---

# LIBERO Library-Size Scaling

Use this skill for Experiment 4. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run Order

1. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).
2. Ensure the LIBERO-90 build has produced immutable `snapshot-N<size>` tags for the target library sizes.
3. Follow [main-agent-prompt.md](main-agent-prompt.md) to create snapshot worktrees, dispatch codegen subagents, and run seed execution.
4. Use [subagent-prompt.md](subagent-prompt.md) for Phase 1 code generation.
5. Use [clean-task-slate.md](clean-task-slate.md) before rerunning a snapshot or task.

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | End-to-end human/agent entrypoint |
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator recipe for frozen snapshot eval on LIBERO-Long-Pro |
| [subagent-prompt.md](subagent-prompt.md) | Preferred code-generation-only subagent prompt |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist before reruns |
| [legacy-seed-running-subagent-prompt.md](legacy-seed-running-subagent-prompt.md) | Legacy prompt where subagents also execute seeds; retained for reference |
| [../zeroshot-transfer/main-agent-prompt.md](../zeroshot-transfer/main-agent-prompt.md) | Build phase that creates the snapshots |
