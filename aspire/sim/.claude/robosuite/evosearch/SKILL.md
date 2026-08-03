---
name: robosuite-evosearch
description: Run Robosuite Fix Loop + Evolutionary Search with K=8 candidate search on seeds 101-125 and selected-code evaluation on seeds 1-100.
---

# Robosuite Fix Loop + Evolutionary Search

Start with [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run Order

1. Confirm the traced config and tracked five-iteration baseline exist for each
   target.
2. Complete the coordinator preflight and wait for explicit launch approval.
3. Follow [main-agent-prompt.md](main-agent-prompt.md).
4. Fill [subagent-prompt.md](subagent-prompt.md) once per confirmed task.
5. Seed `candidate_A` verbatim from the task baseline file under `baselines/`.
6. Search on seeds 101-125 only, then evaluate the selected code on seeds 1-100.

## Reused Robosuite References

| File | Purpose |
|---|---|
| [../api-reference.md](../api-reference.md) | Allowed API, trace schema, task classes |
| [../fix-loop/skills/grasp.md](../fix-loop/skills/grasp.md) | Grasp patterns |
| [../fix-loop/skills/localize.md](../fix-loop/skills/localize.md) | Perception and prompt patterns |
| [../fix-loop/skills/manipulation.md](../fix-loop/skills/manipulation.md) | Contact-rich task patterns |
| [../fix-loop/skills/transport.md](../fix-loop/skills/transport.md) | Transport and bimanual patterns |
| [../fix-loop/clean-task-slate.md](../fix-loop/clean-task-slate.md) | Approval-gated reset of Fix Loop artifacts |

Never delete or replace prior Evolutionary Search outputs without listing the
exact targets and receiving explicit approval.
