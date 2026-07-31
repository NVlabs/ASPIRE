---
name: behavior/aspire-protocol
description: Canonical two-stage BEHAVIOR-1K ASPIRE protocol for Soda Can and Radio: learn a skill library on seeds 26-35, freeze it, then run isolated fresh-agent adaptation on seeds 1-25.
---

# BEHAVIOR-1K ASPIRE Protocol Skill

Use this skill for either canonical request:

```text
Follow the protocol and run BEHAVIOR-1K Soda Can ASPIRE experiments.
```

```text
Follow the protocol and run BEHAVIOR-1K Radio ASPIRE experiments.
```

This is a complete campaign protocol, not a frozen-policy benchmark. Its inner
mechanic is the block-by-block replay loop in `../fix-loop/`, but this skill owns
the development/held-out split and all isolation rules.

## Read First

From `aspire/sim`, read in order:

1. `../../AGENTS.md`
2. `CLAUDE.md`
3. `.claude/README.md`
4. `.claude/behavior/CLAUDE.md`
5. `.claude/behavior/api-reference.md`
6. `.claude/behavior/skills/system-pipeline.md`
7. `.claude/behavior/aspire-protocol/INSTRUCTIONS.md`
8. `.claude/behavior/aspire-protocol/clean-task-slate.md`
9. `.claude/behavior/fix-loop/INSTRUCTIONS.md`

Then use `.claude/behavior/aspire-protocol/main-agent-prompt.md` as the
coordinator contract. Use the Stage 1 and Stage 2 prompt templates in that
directory without weakening their boundaries.

## Task Mapping

| Request | Traced replay config |
|---|---|
| Soda Can | `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml` |
| Radio | `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml` |

`pick_up_trash` is the historical filename for the blue soda-can task.

## Protocol Summary

1. Preflight and wait for explicit user confirmation.
2. Create a fresh campaign root and a working copy of the repository skill
   templates.
3. On development seeds 26-35, let the learning agent debug block by block and
   distill validated reusable lessons into the working skill library.
4. Freeze the skill library and experimental contract after seed 35.
5. On each held-out seed 1-25, launch a fresh isolated Claude Code context with
   a fresh empty policy. It may adapt block by block within that seed, using
   only the frozen library and its own artifacts.
6. Aggregate all 25 terminal outcomes without replacing failures.

The ASPIRE YAMLs contain a built-in `REGENERATE`/`FINISH` prompt, but this
protocol must not invoke that loop. Every trial uses
`scripts/behavior/replay_trial_b1k.py --replay-code`, which executes code
directly through `env.step` without an internal model call. External Claude
Code is the only model in the adaptation loop.

Do not remove the shared built-in multi-turn implementation from the codebase;
other experiments use it. Do not run experiments, install dependencies, start
services, or dispatch agents until preflight is approved.
