---
name: libero-evosearch
description: "Run the LIBERO-Pro Fix Loop + Evolutionary Search experiment: iterative candidate search for low-performing tasks, validation selection, and final held-out evaluation."
---

# LIBERO Fix Loop + Evolutionary Search

Use this skill for Experiment 2. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run Order

1. Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).
2. Confirm the Fix Loop experiment has produced `fix_code.py` for the target tasks.
3. Follow [main-agent-prompt.md](main-agent-prompt.md) as the coordinator.
4. Fill [subagent-prompt.md](subagent-prompt.md) once per task below the success threshold and dispatch subagents on the available worker GPUs.
5. Read experiment-specific companion skills under [skills/](skills/) when generating candidates.
6. Use [clean-task-slate.md](clean-task-slate.md) before rerunning a task or suite.

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | End-to-end human/agent entrypoint |
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator runbook for Evolutionary Search task dispatch and result collection |
| [subagent-prompt.md](subagent-prompt.md) | Per-task Evolutionary Search iteration prompt |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist before reruns |
| [skills/evosearch-iteration.md](skills/evosearch-iteration.md) | Candidate/eval/keyframe iteration mechanics |
| [../api-reference.md](../api-reference.md) | LIBERO reduced API reference |
| [../skills/](../skills/) | Shared LIBERO robot skills |
