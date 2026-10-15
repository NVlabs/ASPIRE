---
name: iterative-debugging-behavior
description: "Block-by-block fix loop on BEHAVIOR-1K (R1 Pro, OmniGibson on Isaac Sim). The policy is grown one code block at a time against what the robot actually did, and what carries across seeds is a skill library, not a program. Use for the two R1 Pro pickup tasks; Stage 1 learns on seeds 26-35, Stage 2 evaluates on 1-25 with a frozen library."
---

# Block-by-Block Fix Loop on BEHAVIOR-1K

The parent skill ([../SKILL.md](../SKILL.md)) writes one whole program per task, freezes it, and
replays that frozen program across the held-out seeds. **That protocol does not apply here.**
BEHAVIOR tasks are long-horizon: the robot starts in another room, cannot see the target, and
must search, drive, look down and grasp. A program authored up front is guessing at every stage
past the first.

So on this benchmark the policy is **grown**, not authored:

1. Write one `# Code block N` section.
2. Run the **whole** `policy.py` on the seed, from a fresh reset.
3. Read what actually happened: where the base ended up, what was localized, what the hand holds.
4. Write the next block against that observed state, and run the whole program again.
5. Repeat until the seed succeeds or the budget is spent.

The simulator resets on every attempt. Only `policy.py` persists and accumulates blocks.

What carries across seeds is the **skill library**, not any policy. Each seed's `policy.py` is
disposable. Stage 2 measures whether the frozen library is enough for a fresh agent to solve an
unseen seed, not whether one program generalizes.

**The library is shared by both tasks.** It lives at `$RUN_ROOT/behavior/skill-library/`, above the
suite directories, and a task's Stage 1 extends what the previous task left. This is a deliberate
departure from ASPIRE, which keeps one library per task. The reason is that most of what a campaign
learns here is about the harness, the robot and the simulator rather than about one object, and
re-discovering it per task wastes a bootstrap seed each time. It also asks a more interesting
question: does a library built on one task help on a different one.

It leaks nothing, because the partition is per task. Knowledge measured on the radio's seeds says
nothing about a soda-can instance, so the can's evaluation seeds stay unseen.

The cost is that **every entry must carry the task it was measured on**, and an entry from another
task is a hypothesis rather than a rule until this task's seeds confirm it.

| Difference from the parent | Consequence |
|---|---|
| Long horizon, target not visible at reset | Blocks are written against observed state, never predicted state |
| The deliverable is a library | Stage 1 promotes skills after every seed; policies are disposable |
| Stage 2 is agentic | A fresh agent per evaluation seed writes its own blocks against the frozen library |
| Frames: `odom` is fixed where the base stood at reset | Localized objects stay valid after the base drives |
| Episodes cost minutes, not seconds | Every attempt replays every earlier block; keep blocks lean |
| Isaac Sim, not MuJoCo | `OMNIGIBSON_*` variables, one Isaac process per GPU, exit 139 after a complete run is normal, judge by `outcome.json` |
| Planning runs in-process on cuRobo | `MotionStrategy()` already means cuRobo; PyRoKi and port 8118 are not involved |
| Two arms, a torso and a base | `behavior.*` extensions move base and torso; `RobotAction` moves arms |

## Setup

1. `scripts/bootstrap_behavior.sh --accept-dataset-tos` once per machine (see
   [docs/behavior.md](../../../../docs/behavior.md)); it prints the activation block.
2. Export the Benchmark Profile block from [subagent-prompt.md](subagent-prompt.md) verbatim in
   every shell that runs episodes. It sets `CAP_HARNESS`, `SERVICE_PORTS`, the `OMNIGIBSON_*`
   variables, `CUDA_VISIBLE_DEVICES` and `ulimit -c 0`. Run flags are spelled out in the run
   command rather than held in a variable, because zsh does not word-split one.
3. Bootstrap the two provider venvs once (`scripts/bootstrap_providers.sh --providers
   sam3,contact_graspnet --profile rtx4090`; profiles live in `configs/profiles/`), then start
   them: `scripts/supervise_services.sh --providers sam3,contact_graspnet --profile rtx4090`.
   On a single-GPU bench they share the card with Isaac (about 8 GB of VRAM together); leave
   PyRoKi and cuRobo services down, they are not used.
4. Read [docs/api-reference.md](../../../../docs/api-reference.md) (the `behavior.*` rows),
   [docs/behavior.md](../../../../docs/behavior.md) `## Semantics` (the `odom` paragraph),
   [docs/run-artifacts.md](../../../../docs/run-artifacts.md), and
   [main-agent-prompt.md](main-agent-prompt.md).

## Seeds

Seeds are task-instance ids; both tasks ship 300 training instances (0-299), so every partition
below is valid without any init-mode switch.

| Partition | Seeds | Rule |
|---|---|---|
| Stage 1, learning | 26-35 | Build the skill library here. Never inspect an evaluation seed. |
| Stage 2, evaluation | 1-25 | One fresh agent per seed, empty policy, frozen library, append-only. |
| Free diagnostics | 36-50 | One-off probes only; never reported. |

The shipped examples under `examples/behavior/` were verified on seeds 1-3, which are evaluation
seeds. They are API documentation, not campaign material: **a campaign never reads them**, and
the skill library starts empty.

## Campaign Layout

```
$RUN_ROOT/behavior/
├── skill-library/               # SHARED and cumulative across both tasks
└── <suite>/
├── campaign-state.md            # coordinator's ledger: seed, stage, attempts, outcome
├── skill-library-frozen/        # this task's read-only snapshot, taken at its freeze
├── frozen-manifest.sha256
├── stage1/seed_26 … seed_35/
│   ├── policy.py                # grows block by block
│   ├── attempts/attempt_001/ …  # one harness run directory per attempt
│   └── seed-summary.md
└── stage2/seed_01 … seed_25/    # same shape, frozen library, fresh agent each
```

## Run Order

1. Create the layout above for the suite you are running. If this is the **first** task of a
   campaign, seed the shared `skill-library/` with the topic files from [skills/](skills/) as
   templates only, headings without contents. If a previous task has already run, the shared
   library is what it left: use it as it stands, and do not reset it.
2. Stage 1: for each seed 26 to 35 in order, dispatch one worker with
   [subagent-prompt.md](subagent-prompt.md) at `STAGE: 1`. After each seed, fold what worked into
   `skill-library/` and record the seed in `campaign-state.md`.
3. Freeze after seed 35, following [main-agent-prompt.md](main-agent-prompt.md). Nothing in the
   library, the prompts, the config or the protocol changes after this point.
4. Stage 2: for each seed 1 to 25, dispatch one **fresh** worker at `STAGE: 2` with the frozen
   library. No information passes between evaluation seeds.
5. Score with `./score_campaign.py <campaign-dir>`. Do NOT hand-count attempts: the unit of
   scoring is the **final policy**, and the script finds it by hashing each run's
   `source/program.py`. Exactly two numbers are reported, both per seed: **success**, the seed's
   final policy succeeded at least once; and **navigation**, the seed's final policy reached the
   object. Reliability is not reported. Development attempts ran earlier programs with no grasp
   block and are excluded entirely. See **Scoring** in [main-agent-prompt.md](main-agent-prompt.md).
6. Use [clean-task-slate.md](clean-task-slate.md) before reruns.
7. Promote findings into this folder's [skills/](skills/) files only between campaigns, never
   during Stage 2.
