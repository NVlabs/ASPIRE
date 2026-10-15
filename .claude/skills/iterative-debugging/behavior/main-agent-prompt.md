# Block-by-Block Fix Loop on BEHAVIOR-1K — Coordinator Guide

Read [SKILL.md](SKILL.md) first. The parent coordinator guide
([../main-agent-prompt.md](../main-agent-prompt.md)) describes a campaign that freezes one program
per task and replays it; **that is not this protocol**. Here you run a per-seed agentic loop in
both stages and what you freeze is the skill library. The parent's provenance gate
(`Non-privileged API audit: PASS` before a seed counts; otherwise preserve the artifacts and
report the seed as invalid) still applies unchanged.

## Campaign shape

- Tasks: `turning_on_radio` and `picking_up_trash`, both `task_id 0`, benchmark `behavior`.
  Run one task to completion before starting the other; they do not share a library.
- Stage 1 learns on seeds 26-35, in order. Stage 2 evaluates on seeds 1-25. Seeds 36-50 are free
  diagnostics and are never reported. **Never inspect an evaluation seed during Stage 1.**
- `MAX_STEPS` default is 6000 (200 s of simulated time at 30 Hz); capture is 512x512 with three
  videos (`head`, `left_wrist`, `right_wrist`).
- Attempt budget: one attempt is one run of the whole policy, whether it appended a block or
  revised one, so the budget is not a block count. A pickup policy needs roughly five blocks
  (search, approach, re-acquire, grasp, lift) before it can succeed at all, and early seeds spend
  extra attempts on revisions while the library is still thin, so budget well above five. Start
  at fifteen per seed in Stage 1 and twelve in Stage 2 and adjust from what the first seeds
  actually use. At roughly two minutes an attempt, Stage 1 is a few hours and Stage 2 most of a
  day; plan for that and run it in the background.

## Layout

```bash
LIB="$RUN_ROOT/behavior/skill-library"          # shared by both tasks
CAMP="$RUN_ROOT/behavior/$SUITE"
mkdir -p "$LIB" "$CAMP/stage1" "$CAMP/stage2"
: > "$CAMP/campaign-state.md"
```

The library is **shared and cumulative**. On the first task of a campaign, seed it with the five
topic filenames from [skills/](skills/) and their headings only, empty of content, and in
particular nothing copied from `examples/behavior/`, which is API documentation verified on
evaluation seeds and must not enter a campaign. On every later task, take the library as the
previous task left it.

When you fold a Stage 1 worker's findings in, **record which task each entry was measured on**. An
entry carried over from another task is a hypothesis for this one: it may be a harness fact that
transfers unchanged, or an object-specific number that does not. Mark the difference, because a
worker cannot tell them apart from the text alone.

## Stage 1 (seeds 26-35)

For each seed in order:

1. Dispatch one worker with [subagent-prompt.md](subagent-prompt.md), `STAGE: 1`, that seed only.
2. When it returns, read its `seed-summary.md` and the skills it wrote.
3. Fold what generalizes into `skill-library/`, deduplicating against what is already there. A
   skill needs a trigger, a snippet that ran, and the seed and attempt it came from.
4. Append one line per seed to `campaign-state.md`: seed, attempts, outcome, skills added.

A seed that fails is still evidence. Record it and move on; do not spend the campaign on one seed.

**The first seed of a campaign is a bootstrap seed.** Its worker starts with an empty library and
pays for every harness fact the rest of the campaign will get for free. Measured on seed 26 of
`turning_on_radio`: fourteen attempts, six of them appends and eight revisions, and roughly three
spent on facts that were task-independent. Budget the first seed at least half again what you
give the others, and read its library contribution before dispatching the second.

## Freeze (after seed 35, before any evaluation seed)

```bash
cp -r "$RUN_ROOT/behavior/skill-library" "$CAMP/skill-library-frozen"
chmod -R a-w "$CAMP/skill-library-frozen"
( cd "$CAMP" && find skill-library-frozen -type f -print0 | sort -z \
    | xargs -0 sha256sum > frozen-manifest.sha256 )
```

The freeze takes a **snapshot** of the shared library for this task's Stage 2. The shared library
itself stays writable, so the next task's Stage 1 can extend it; each task keeps its own frozen
snapshot and manifest as the immutable record of what its evaluation actually measured.

From this point the library, this folder's prompts, the harness config, the model and the
protocol are all frozen. Policies are **not** frozen: every evaluation seed writes its own from
scratch. Verify before each Stage 2 dispatch:

```bash
( cd "$CAMP" && sha256sum -c frozen-manifest.sha256 ) || echo "FROZEN LIBRARY CHANGED - STOP"
```

## Stage 2 (seeds 1-25)

For each seed, dispatch a **fresh** worker with `STAGE: 2`. Each one gets an empty policy and the
frozen library, works append-only, and may not see any other seed. Never resume a Stage 2
worker's context for a different seed, and never pass one seed's summary, policy or attempts to
another. Record only seed, attempts and outcome in `campaign-state.md` as you go.

Do not read Stage 2 policies, summaries or videos until all twenty-five seeds are finished.
Reading them mid-stage is how the library gets edited to fit the evaluation set, which invalidates
the campaign.

The headline number is the fraction of the twenty-five seeds solved, with the attempt counts
beside it. Report failures honestly; a library that carries some seeds and not others is the
measurement. Count it exactly as the next section says.

## Scoring: two numbers, nothing else

**The unit of scoring is the FINAL POLICY, never the attempt.** A worker grows one program block by
block, and every attempt replays the whole program from a fresh reset. The early attempts are
development: their programs had no grasp block yet, so they could not have succeeded. They are
excluded from scoring entirely. A worker that authored six blocks is not worse than one that
authored three.

**Identify the final policy from the artifacts, not from the attempt numbers.** Hash each run's
`source/program.py`; runs sharing the last distinct hash are the final policy's episodes. That is
the winning attempt plus its unchanged repeats, and it is exact rather than inferred.
`./score_campaign.py <campaign-dir>` does this and prints the two numbers below.

**Report exactly two numbers per task. Both are per seed.**

| Question | A seed counts when | Denominator |
|---|---|---|
| **Success** | its final policy succeeded at least once | seeds |
| **Navigation** | its final policy drove the robot to the object and got the arm working on it | seeds |

**Success:** a seed is a success if its final policy succeeded at least once. How reliable it was
does not enter, and no reliability figure is reported. A policy that held once in three repeats and
a policy that held three times in three both count as one successful seed.

**Navigation:** judged on the same final policy. A seed's navigation is correct if, in at least one
episode of the final policy, a `close_gripper` or `set_gripper` call and a `move_to_pose` or
`execute_trajectory` call both occur. Do NOT report per-leg `behavior.navigate_to_pose` success:
individual legs fail routinely and the repair ladder recovers them inside the same episode, so that
figure says nothing about whether the seed navigated, and it has been mistaken for a navigation
rate before.

**Do not report anything else as a rate.** Not per-episode reliability, not the last episode
alone, not attempts-to-first-hold, not leg success. The first campaign reported several of these side by
side, and both the agents and the humans reading them confused one for another. The artifacts keep
every episode, so any of them can be recomputed later if a specific question needs it; the campaign
report carries the two numbers above and nothing more.

**Exclude infrastructure losses, and say how many there were.** An attempt that died to a launch
crash, a CUDA out-of-memory caused by another job, or a service outage is not a policy failure.
Name the cause and the count rather than silently dropping them.

## Services and GPUs

| Port | Service | Needed by |
|---|---|---|
| 8114 | SAM3 | every policy |
| 8115 | Contact-GraspNet | grasp candidates |

PyRoKi (8116) and the cuRobo service (8118) are not used: planning runs inside the simulator
process. Start only what is needed: `scripts/supervise_services.sh --providers sam3,contact_graspnet
--profile <profile>` (`rtx4090` on this bench; the script's default profile is `rtx5090`).

Isaac Sim needs about 12 GB of VRAM and 16 GB of RAM for these scenes; SAM3 and Contact-GraspNet
add about 8 GB of VRAM, which leaves little headroom on a 24 GB card. Nothing else may use the
GPU. Before every episode on a shared machine check `free -g` and `nvidia-smi`; if another Isaac
process is running, wait. **Single-GPU mode is the default for this benchmark**: strictly serial,
you run every `cap-harness run` yourself, workers analyse finished attempt directories.

Set in the coordinator shell and in every dispatch the Benchmark Profile block from
[subagent-prompt.md](subagent-prompt.md); use it verbatim rather than retyping the list.

## Reading a run

Judge every run by `outcome.json`. Kit can exit with status 139 after the attempt directory is
complete; that is a complete run. A run with no `outcome.json` and a short log is a crash during
launch or scene load: read the log tail for `[Error]`, check RAM, and retry once before treating
it as a policy failure. Never let a worker count a launch crash as a task failure.

## Stop and ask for approval if

- A worker would cross a seed boundary, or Stage 1 would touch seeds 1-25.
- `sha256sum -c frozen-manifest.sha256` fails.
- Two Isaac processes are alive at once.
- Required evidence is missing from an attempt directory.
- The protocol itself would have to change to make progress.
- You are tempted to edit the frozen library during Stage 2.

## Updating skills between campaigns

Promote durable patterns from `skill-library/` into this folder's [skills/](skills/) files, by
topic, with their provenance. Keep the parent's tabletop skill files untouched; a pattern that
also applies to fixed-base arms is cross-linked, not duplicated. Never do this while a Stage 2 is
in flight.
