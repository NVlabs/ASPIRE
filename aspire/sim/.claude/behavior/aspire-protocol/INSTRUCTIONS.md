---
name: behavior/aspire-protocol/INSTRUCTIONS
description: Source of truth for the canonical two-stage BEHAVIOR-1K ASPIRE experiment.
---

# Canonical BEHAVIOR-1K ASPIRE Experiment

This document is the source of truth for Soda Can and Radio ASPIRE campaigns.
It defines a **skill-learning stage** on development seeds followed by a
**fresh-agent adaptation stage** on held-out seeds.

The measured system is not one frozen policy. It is:

```text
frozen skill library + fixed agent/model/budget/protocol
  -> fresh per-seed policy construction and replay-based debugging
```

## 1. Invariants

These rules are mandatory:

- Development seeds are exactly 26-35. A Stage 1 agent must never inspect,
  reset, replay, or infer information from seeds 1-25.
- Held-out seeds are exactly 1-25 for both Soda Can and Radio, regardless of a
  YAML's default `resume_idx` or `trials` values.
- Stage 1 may share lessons across development seeds. Stage 2 may not share
  policy code, traces, observations, transcripts, summaries, or newly learned
  skills across held-out seeds.
- After Stage 1, freeze the skill library and the experimental contract. Do
  **not** freeze a Stage 1 policy for evaluation.
- Each held-out seed starts one fresh isolated Claude Code agent/context and
  one fresh empty policy. The agent may append blocks and replay repeatedly
  within that seed until success or its fixed budget is exhausted.
- A held-out seed gets one agent episode. Its internal replays are adaptation
  attempts, not replacement evaluation runs. Preserve every attempt and the
  final failure if it does not succeed.
- External Claude Code is the only model in the loop. Do not invoke normal
  ASPIRE batch generation or the built-in `REGENERATE`/`FINISH` loop.
- Use the traced config through replay mode for both stages. Replay mode ignores
  the YAML `multi_turn_prompt` and executes saved blocks directly with
  `env.step`.
- Run seeds sequentially with at most one Isaac Sim process on the node.
- Generated policy code may use only the public R1Pro APIs. It must not inspect
  simulator internals, BDDL predicates, object registries, or reward state.
- Experiment agents do not push, rewrite history, or delete prior outputs.
- No real-robot service, camera, follower, or physical motion is in scope.

Any deviation makes the affected seed protocol-invalid and must be reported.

## 2. Preflight Gate

The canonical request resolves the task but does not authorize immediate
execution. Before setup changes, service startup, agent dispatch, or trials,
the coordinator must report:

- repository path, exact commit SHA, branch, and dirty/clean state;
- selected task and traced config;
- host, GPU mapping, free capacity, and the one-simulator constraint;
- B1K environment and asset status, including whether this checkout has passed
  a clean oracle or replay smoke test;
- SAM3 and ContactGraspNet ports from the selected YAML;
- external Claude Code model/context configuration;
- fixed Stage 1 and per-evaluation-seed limits for wall time, replay attempts,
  and agent turns or tokens, as available on the chosen surface;
- exact seed partitions and total count: 10 development plus 25 held-out;
- fresh campaign output path and expected runtime;
- any unresolved setup or observability risk.

Wait for explicit user confirmation. Do not silently choose missing budgets or
change the task, model, seeds, or protocol. Cost estimation and token-price
analysis are outside this protocol unless the user separately requests them.

## 3. Task And Campaign Layout

Use the traced replay config selected by the request:

| Task | Config |
|---|---|
| Soda Can | `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml` |
| Radio | `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml` |

Create a new, non-existing campaign root:

```text
outputs/behavior/aspire-campaigns/<task>/<campaign-id>/
  manifest.md
  campaign-state.md
  skill-library-working/
  skill-library-frozen/
  frozen-manifest.sha256
  stage1/
    seed_26/
      policy.py
      attempts/attempt_001/...
      seed-summary.md
    ...
    seed_35/
  stage2/
    seed_01/
      policy.py
      attempts/attempt_001/...
      seed-summary.md
    ...
    seed_25/
  reports/
    final-report.md
```

Copy `.claude/behavior/skills/` into `skill-library-working/` at campaign
creation. Do not edit the repository templates during a run. Copy
`campaign-state-template.md` to `campaign-state.md`, fill the manifest, and let
only the coordinator update campaign state.

Every replay gets a new `attempt_NNN/` output path. Never reuse, rename,
overwrite, or delete an earlier attempt. A resuming coordinator continues at
the first incomplete state entry; it never reruns a terminal seed to improve
the score.

## 4. Replay-Only Inner Loop

For every development or evaluation seed, the owning agent starts `policy.py`
fresh when required by its stage and appends small executable sections marked
as code blocks:

```python
# Code block 0
# 5-20 lines of observation or setup code

# Code block 1
# the next evidence-driven action block
```

After each append, reset and replay the same seed:

```bash
OMNIGIBSON_GPU_ID=<gpu> uv run --no-sync --active \
  scripts/behavior/replay_trial_b1k.py \
  --config-path <traced-config> \
  --replay-code <seed-dir>/policy.py \
  --trial <seed> \
  --output-dir <seed-dir>/attempts/attempt_<NNN> \
  --record-video
```

The agent inspects only allowed evidence: `summary.txt`, `trace.json`,
`keyframes/`, videos, saved observations, and relevant perception-service logs.
It classifies the blocker, appends or minimally revises the next block, and
replays the same seed. It stops on task success or budget exhaustion.

Do not use `python -m aspire.sim.cap.envs.launch` for a protocol trial. Do not
pass a model endpoint or API key to replay mode. The config's built-in
multi-turn prompt remains dormant.

## 5. Stage 1: Skill Acquisition On Seeds 26-35

Run seeds 26 through 35 sequentially. A Stage 1 learning agent may carry its
reasoning, working policies, and validated lessons across these ten development
seeds. Keep each seed's policies and attempts in that seed's directory.

For each seed:

1. Follow `stage1-skill-acquisition-prompt.md` and the replay-only inner loop.
2. Preserve all evidence, including unsuccessful attempts.
3. Write `seed-summary.md` with outcome, attempt count, failure taxonomy,
   evidence paths, and candidate reusable lessons.
4. Promote a lesson to `skill-library-working/` only when the development
   evidence supports a reusable public-API strategy. Record the source seed and
   evidence path. Do not encode scene ground truth or seed-specific coordinates.
5. Mark the seed terminal in `campaign-state.md` before starting the next seed.

The Stage 1 deliverable is the learned skill library. Development policies are
debugging artifacts and must not become the evaluation starting policy.

## 6. Freeze Boundary After Seed 35

After all ten Stage 1 seeds are terminal:

1. Stop the Stage 1 agent and all simulator processes.
2. Copy `skill-library-working/` to a read-only
   `skill-library-frozen/` snapshot.
3. Record SHA-256 checksums for every frozen skill file in
   `frozen-manifest.sha256`.
4. In `manifest.md`, record the repository commit, selected config and its
   checksum, model/context, approved budgets, exact coordinator and seed-agent
   prompts, public API reference checksum, and frozen library checksum.
5. Verify the frozen snapshot matches its manifest before dispatching Stage 2.

“Freeze” refers to the skill library and experimental contract. There is no
frozen policy checkpoint or policy hash because evaluation intentionally starts
from a fresh policy on every held-out seed.

After this boundary, no one may edit the frozen library, prompts, model,
budgets, config, or protocol. If any changes are necessary, stop and create a
new campaign rather than silently continuing.

## 7. Stage 2: Fresh-Agent Adaptation On Seeds 1-25

Run seeds 1 through 25 sequentially. For every seed, launch a new Claude Code
process/session or fresh subagent context that has never seen another held-out
seed and receives no inherited experiment transcript. Never resume or reuse an
earlier held-out agent. If the available tooling cannot guarantee this context
isolation, stop before evaluation.

The agent's read allowlist is:

- the frozen experimental manifest and frozen skill library;
- suite/API/fix-loop documentation fixed by the recorded commit;
- the selected traced config;
- its own `stage2/seed_<NN>/` policy, attempts, and logs.

It must not read Stage 1 policies or raw traces, any other Stage 2 seed
directory, coordinator diagnostics about earlier held-out seeds, or mutable
repository/campaign skill files. Its prompt is the immutable
`stage2-evaluation-seed-prompt.md` with only task, seed, paths, GPU, and approved
budget placeholders filled in.

For each held-out seed:

1. Create a new seed directory and a fresh empty `policy.py`.
2. Start a fresh isolated agent with the fixed seed prompt.
3. Let that agent generate, replay, inspect, and debug block by block only on
   its assigned seed.
4. End the episode on success or budget exhaustion and terminate the agent.
5. Preserve all attempts and write a terminal `seed-summary.md`.
6. Have the coordinator record only status, outcome, attempt count, and paths;
   it must not relay lessons or policy details to the next seed.

Until all 25 episodes are terminal, the coordinator must not inspect detailed
seed summaries, policies, traces, observations, or videos. A seed agent returns
only its seed, terminal status/outcome, replay count, and summary path. The
coordinator may read and aggregate detailed summaries only after the last seed
ends; this prevents the coordinator itself from becoming a cross-seed channel.

Do not restart a failed held-out seed with a fresh reasoning context. A genuine
infrastructure failure before usable evidence may resume in the same seed
directory only if it is labeled, preserves all artifacts, and does not reset
the approved budget. A post-trial Isaac Sim `SIGSEGV` is only an infrastructure
warning if the terminal result, summary, trace, and video were already written;
otherwise the attempt is incomplete and must be reported as such.

## 8. Completion And Reporting

The campaign is complete only when seeds 1-25 each have one terminal agent
episode and the frozen manifest still verifies.

`reports/final-report.md` must include:

- repository/config/model/budget/frozen-library provenance;
- Stage 1 completion and learned-library summary;
- per-seed Stage 2 outcome, replay count, terminal reason, and artifact path;
- success count and rate over all 25 held-out seeds, with failures retained;
- failure taxonomy and protocol/infrastructure invalidities;
- confirmation that each held-out seed used a fresh policy and isolated agent;
- confirmation that no built-in ASPIRE LLM loop or real hardware was used.

Do not claim success from sandbox return code alone; use the task-completion
field/reward and saved evidence. Do not omit failed or invalid seeds from the
denominator. Do not calculate dollar cost unless separately requested.

## 9. Stop Conditions

Stop, preserve state, and ask the user before continuing if:

- the task, model, budget, seed range, or protocol would change;
- the campaign path already contains unrelated artifacts;
- an evaluation agent reads a forbidden seed or artifact;
- the frozen manifest changes or cannot be verified;
- normal batch launch or the built-in LLM loop is invoked accidentally;
- multiple Isaac Sim processes overlap on the node;
- required traces/videos are systematically missing;
- credentials would enter prompts, logs, YAML, Markdown, or generated code;
- execution would touch real hardware.

Record the deviation in `campaign-state.md`; never hide it by deleting or
replacing evidence.
