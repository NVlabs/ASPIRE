---
name: libero-zeroshot-transfer
description: "Run the LIBERO-90 build and zero-shot transfer experiment: build a skill library on LIBERO-90, tag snapshots, then evaluate transfer to LIBERO-Long-Pro."
---

# LIBERO Zero-Shot Transfer

Use this skill for Experiment 3. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run Order

1. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).
2. Follow [main-agent-prompt.md](main-agent-prompt.md) to build the LIBERO-90 skill library in configured task batches.
3. Use [subagent-prompt.md](subagent-prompt.md) for each build subagent.
4. Commit and tag each chunk using [commit-convention.md](commit-convention.md).
5. For LIBERO-Long-Pro zero-shot evaluation, use [../library-size-scaling/main-agent-prompt.md](../library-size-scaling/main-agent-prompt.md).
6. Use [clean-task-slate.md](clean-task-slate.md) before rerunning a chunk or task.

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | End-to-end human/agent entrypoint |
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator runbook for the LIBERO-90 skill-library build |
| [subagent-prompt.md](subagent-prompt.md) | Per-task LIBERO-90 build subagent prompt |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist before reruns |
| [commit-convention.md](commit-convention.md) | Chunk commit and immutable snapshot tag convention |
| [libero-90-heldout-eval.md](libero-90-heldout-eval.md) | Same-domain LIBERO-90 held-out eval reference |
| [../skills/](../skills/) | Shared LIBERO robot skills grown during the build |
