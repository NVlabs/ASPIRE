# LIBERO-Pro Goal-Swap Quick Start

This runbook executes one complete ASPIRE Fix Loop task with the paper's per-task development and held-out protocol. It is a long-running reproduction, not a short smoke test.

## Fixed Scope

```text
SUITE=libero_goal_swap
TASK=put_the_wine_bottle_on_top_of_the_cabinet
TASK_GPU=3
DEVELOPMENT_SEEDS=51-65
HELD_OUT_SEEDS=1-50
```

Only this suite/task pair is in scope. The global progress report lists every LIBERO-Pro task; ignore all entries outside this allowlist. This rule overrides the instructions in `main-agent-prompt.md` to fill all free task GPUs and continue until every pending task is done.

The reference topology is:

| GPU | Owner |
| --- | ----- |
| 0 | SAM3 |
| 1 | GraspNet |
| 2 | PyRoKi |
| 3 | Fix Loop subagent and held-out validation |

Do not silently remap or share these GPU roles. If the host cannot provide the reference topology, report the mismatch and stop for user direction.

## Required Reading

From the `aspire/sim` working root, read:

1. [`README.md`](../../../README.md) for installation and provider setup
2. [`../CLAUDE.md`](../CLAUDE.md) for the LIBERO constitution and API boundary
3. [`INSTRUCTIONS.md`](INSTRUCTIONS.md) and [`SKILL.md`](SKILL.md) for the Fix Loop protocol
4. [`main-agent-prompt.md`](main-agent-prompt.md) for coordinator responsibilities
5. [`subagent-prompt.md`](subagent-prompt.md) for Stage 1
6. [`clean-task-slate.md`](clean-task-slate.md) before any rerun

The task must be completed without reading external baseline code or baseline outputs. Development seeds 51–65 and held-out seeds 1–50 must remain strictly separated.

## Preflight and Confirmation Gate

Before installing dependencies, starting services, dispatching a subagent, running a replay, or launching validation, inspect the repository and host and report:

- repository commit and worktree state;
- Linux, NVIDIA driver, CUDA, and available GPU inventory;
- proposed GPU ownership matching the table above;
- required virtual environments and submodule state;
- SAM3 gated-model access and required inference credentials, without printing secrets;
- expected ports and current status for SAM3, GraspNet, and PyRoKi;
- development and held-out seed ranges;
- expected runtime, including that failed trials can take approximately 6–7 minutes each and the 50 held-out seeds run sequentially;
- exact Stage 1 and Stage 2 output paths;
- any existing outputs for this suite/task and the applicable clean-slate decision.

Then stop and wait for explicit user confirmation. Missing credentials, gated weights, services, environment support, or GPU resources are actionable blockers; do not bypass them or substitute a different task, suite, model, or seed range.

## Execution

After confirmation:

1. Complete the LIBERO setup and perception preflight from the required reading.
2. Regenerate the global progress report, but use it only to inspect this fixed suite/task.
3. Claim GPU 3 for the task. Do not dispatch work on GPUs 4–7.
4. Fill `SUITE`, `TASK`, and `GPU=3` in `subagent-prompt.md` and dispatch exactly one Stage 1 subagent.
5. The subagent explores one observed scene, writes initial code, runs development seeds 51–65, diagnoses failures within the documented replay limits, selects one generalizable `fix_code.py`, and writes `findings.md`.
6. Confirm that the task reached `stage1-done`. Promote only supported generalizable findings according to `main-agent-prompt.md`.
7. Run the immutable held-out evaluation on GPU 3:

   ```bash
   SUITE=libero_goal_swap
   TASK=put_the_wine_bottle_on_top_of_the_cabinet
   GPU=3

   .venv-libero/bin/python3 scripts/libero/run_fix_loop_validation.py \
     --suite "$SUITE" --task "$TASK" --gpu "$GPU" \
     --fix-code "outputs/libero_fix_loop/$SUITE/$TASK/fix_code.py" \
     --output-dir outputs/libero_fix_loop_eval \
     --seeds $(seq 1 50) --resume
   ```

8. Regenerate progress and confirm that the matching immutable manifest contains all 50 held-out seeds. Resume only the same run identity if artifacts are missing.

Never access `aspire/real`, real-robot services, cameras, follower processes, or physical hardware during this workflow. Do not push repository changes.

## Completion Report

Report:

- exact repository commit and suite/task;
- development successes, repaired seeds, and blocked seeds;
- selected `fix_code.py` and `findings.md` paths;
- held-out manifest path and run ID;
- held-out passes, trials, and pass rate;
- representative trace, keyframe, and video locations;
- promoted reusable skills, if any;
- deviations, retries, missing artifacts, or unresolved blockers.

Completion means that the selected code and findings exist and one immutable manifest records all 50 held-out seeds. It does not require or imply a guaranteed success rate.
