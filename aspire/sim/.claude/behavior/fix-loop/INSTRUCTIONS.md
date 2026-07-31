---
name: behavior/fix-loop/INSTRUCTIONS
description: Canonical two-stage BEHAVIOR-1K ASPIRE fix-loop protocol.
---

# BEHAVIOR-1K ASPIRE Fix Loop

This is the source of truth for Soda Can and Radio experiments. The measured
system is:

```text
frozen skill library + fixed model/prompts/budgets/protocol
  -> fresh per-seed policy construction and debugging
```

## Tasks

| Request | Traced replay config |
|---|---|
| Soda Can | `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml` |
| Radio | `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml` |

`pick_up_trash` is the historical filename for the blue soda-can task.

## Invariants

- Stage 1 uses only development seeds 26-35 to build the skill library. It must
  never inspect seeds 1-25.
- After seed 35, freeze the skill library, config, model, prompts, budgets, and
  protocol. Do **not** freeze a policy.
- Stage 2 evaluates seeds 1-25. Every seed gets a fresh non-resumed Claude Code
  context, fresh empty policy, and the frozen skill library.
- A Stage 2 agent may debug block by block within its seed. It cannot read
  another evaluation seed, update skills, or pass policy/transcript/lessons to
  the next agent.
- Use `scripts/behavior/replay_trial_b1k.py --replay-code` for every trial.
  External Claude Code is the only model in the loop; do not invoke the built-in
  ASPIRE `REGENERATE`/`FINISH` loop.
- Run seeds sequentially with one Isaac Sim process on the node. Preserve every
  attempt, including failures. Do not push or touch real hardware.

## Preflight

Before setup changes, services, agents, or trials, report and wait for approval:

- exact commit, task/config, host/GPU, environment and perception-server status;
- Claude model/context and fixed Stage 1/per-seed replay, time, and agent limits;
- seed split, fresh campaign path, expected runtime, and unresolved risks.

Do not invent missing model or budget choices. Cost analysis is out of scope
unless separately requested.

## Campaign Layout

Create a new path; never reuse an older campaign:

```text
outputs/behavior/aspire-campaigns/<task>/<campaign-id>/
  manifest.md
  campaign-state.md
  skill-library-working/
  skill-library-frozen/
  frozen-manifest.sha256
  stage1/seed_26/...seed_35/
  stage2/seed_01/...seed_25/
  reports/final-report.md
```

Copy `.claude/behavior/skills/` to `skill-library-working/`. Each seed directory
contains `policy.py`, `attempts/attempt_NNN/`, and `seed-summary.md`. Only the
coordinator updates `campaign-state.md`; resume at the first incomplete seed.

## Per-Seed Inner Loop

Build `policy.py` in small `# Code block N` sections. After each change, reset
and replay the same seed into a new attempt directory:

```bash
OMNIGIBSON_GPU_ID=<gpu> uv run --no-sync --active \
  scripts/behavior/replay_trial_b1k.py \
  --config-path <traced-config> \
  --replay-code <seed-dir>/policy.py \
  --trial <seed> \
  --output-dir <seed-dir>/attempts/attempt_<NNN> \
  --record-video
```

Inspect only allowed evidence: summary, trace, keyframes, video, saved
observations, and relevant perception logs. Diagnose the failure, append or
minimally revise the next block, and stop on success or budget exhaustion.
Generated code may use only public R1Pro APIs—never simulator internals, BDDL,
object registries, or reward state.

## Stage 1: Learn On Seeds 26-35

Run seeds 26-35 sequentially with `stage1-skill-acquisition-prompt.md`. The
learning agent may reuse reasoning and policies across these development seeds.
After each seed, preserve all attempts, write its summary, and add only
evidence-backed reusable public-API lessons to `skill-library-working/`. Record
the source seed and evidence path. Stage 1 produces skills, not an evaluation
policy.

## Freeze

After all ten development seeds are terminal:

1. Stop the Stage 1 agent and simulator.
2. Copy the working library to `skill-library-frozen/` and make it read-only.
3. SHA-256 the frozen skills, config, API reference, and prompt files.
4. Record commit, model/context, budgets, and hashes in `manifest.md`.

Any later change requires a new campaign.

## Stage 2: Evaluate On Seeds 1-25

For each seed in order:

1. Create a fresh seed directory and empty `policy.py`.
2. Launch a new non-resumed context with `stage2-evaluation-seed-prompt.md`.
3. Allow block-by-block replay and inspection only within that seed.
4. End on success or budget exhaustion; do not replace failed episodes.
5. Record only seed, outcome, replay count, and summary path in campaign state.

Until all 25 seeds finish, the coordinator must not inspect detailed Stage 2
policies, summaries, traces, observations, or videos. This prevents it from
becoming a cross-seed information channel. Aggregate details only afterward.

An infrastructure restart may continue the same episode only if it preserves
artifacts and budget. A post-trial Isaac Sim `SIGSEGV` is acceptable only when
the terminal result, summary, trace, and video were already saved.

## Report And Stop Conditions

Report provenance, learned skills, all 25 outcomes, success rate over 25,
replay counts, failures/invalidities, and confirmation of fresh policies,
isolated agents, replay-only execution, and no hardware use. Never drop failed
or invalid seeds from the denominator.

Stop and ask before continuing if a seed boundary is crossed, frozen hashes
change, multiple simulators overlap, required evidence is missing, the built-in
LLM loop starts, credentials enter artifacts, or the protocol must change.
