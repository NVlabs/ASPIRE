# LIBERO Goal-Swap Fix Loop — Full Suite Quick Start

This runbook executes the complete ASPIRE Fix Loop for all ten `libero_goal_swap` tasks using the paper's per-task development and held-out protocol. It is a long-running reproduction, not a short smoke test.

## Fixed Scope

```text
SUITE=libero_goal_swap
DEVELOPMENT_SEEDS=51-65
HELD_OUT_SEEDS=1-50
```

All ten tasks in the suite:

| # | Task |
|---|------|
| 1 | `open_the_middle_drawer_of_the_cabinet` |
| 2 | `put_the_bowl_on_the_stove` |
| 3 | `put_the_wine_bottle_on_top_of_the_cabinet` |
| 4 | `open_the_top_drawer_and_put_the_bowl_inside` |
| 5 | `put_the_bowl_on_top_of_the_cabinet` |
| 6 | `push_the_plate_to_the_front_of_the_stove` |
| 7 | `put_the_cream_cheese_in_the_bowl` |
| 8 | `turn_on_the_stove` |
| 9 | `put_the_bowl_on_the_plate` |
| 10 | `put_the_wine_bottle_on_the_rack` |

Only this suite is in scope. The global progress report lists every LIBERO-Pro task; ignore all entries outside this allowlist.

The reference topology is:

| GPU | Owner |
| --- | ----- |
| 0 | SAM3 |
| 1 | GraspNet |
| 2 | PyRoKi |
| 3–7 | Fix Loop task slots (five concurrent) |

Do not silently remap or share these GPU roles. If the host cannot provide the reference topology, report the mismatch and stop for user direction.

## Required Reading

From the `aspire/sim` working root, read:

1. [`README.md`](../../../README.md) for installation and provider setup
2. [`../CLAUDE.md`](../CLAUDE.md) for the LIBERO constitution and API boundary
3. [`INSTRUCTIONS.md`](INSTRUCTIONS.md) and [`SKILL.md`](SKILL.md) for the Fix Loop protocol
4. [`main-agent-prompt.md`](main-agent-prompt.md) for coordinator responsibilities
5. [`subagent-prompt.md`](subagent-prompt.md) for Stage 1
6. [`clean-task-slate.md`](clean-task-slate.md) before any rerun

The tasks must be completed without reading external baseline code or baseline outputs. Development seeds 51–65 and held-out seeds 1–50 must remain strictly separated.

## Preflight and Confirmation Gate

Before installing dependencies, starting services, dispatching subagents, running replays, or launching validation, inspect the repository and host and report:

- repository commit and worktree state;
- Linux, NVIDIA driver, CUDA, and available GPU inventory (8 GPUs required);
- proposed GPU ownership matching the table above;
- required virtual environments and submodule state, including
  `contact_graspnet_pytorch@2d71da4e50a04aa353352d1cae99f20f7022145b`
  with the repository compatibility patch applied;
- SAM3 gated-model access and required inference credentials, without printing secrets;
- expected ports and current status for SAM3, GraspNet, and PyRoKi;
- development and held-out seed ranges;
- expected runtime, including that failed trials can take approximately 6–7 minutes each and the 50 held-out seeds run sequentially;
- exact Stage 1 and Stage 2 output paths;
- any existing outputs for this suite and the applicable clean-slate decision.

Then stop and wait for explicit user confirmation. Missing credentials, gated weights, services, environment support, or GPU resources are actionable blockers; do not bypass them or substitute a different task, suite, model, or seed range.

## Execution

After confirmation, follow the coordinator protocol in [`main-agent-prompt.md`](main-agent-prompt.md):

1. Complete the LIBERO setup and perception preflight from the required reading.
2. Regenerate the global progress report, but use it only to inspect this suite's ten tasks.
3. **Fill all five task GPUs (3–7) concurrently.** Assign one pending task to each free GPU and dispatch up to five Stage 1 subagents in a single message. Each GPU is retained by its task through Stage 1, coordinator skill promotion, and Stage 2 — then freed and refilled with the next pending task. Continue until all ten tasks reach `done`.
4. Follow the coordinator loop: on subagent completion, promote skills, start Stage 2 eval on the same GPU; on eval completion, free the GPU and assign the next pending task. Always dispatch all available work in one message — do not serialize tasks one at a time.

The coordinator must go idle after each dispatch round and wait for completion notifications. Do not poll or watch GPUs. See [`main-agent-prompt.md`](main-agent-prompt.md) for the full GPU ownership ledger, skill promotion, and Stage 2 validation protocol.

Never access `aspire/real`, real-robot services, cameras, follower processes, or physical hardware during this workflow. Do not push repository changes.

## Completion Report

Report:

- exact repository commit and suite;
- per-task: development successes, repaired seeds, blocked seeds;
- per-task: selected `fix_code.py` and `findings.md` paths;
- per-task: held-out manifest path, run ID, passes, trials, and pass rate;
- representative trace, keyframe, and video locations;
- promoted reusable skills per task;
- deviations, retries, missing artifacts, or unresolved blockers.

Completion means all ten tasks have a selected fix_code.py and one immutable manifest recording all 50 held-out seeds each. It does not require or imply a guaranteed success rate.
