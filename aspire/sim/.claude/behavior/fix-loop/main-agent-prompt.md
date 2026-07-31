# BEHAVIOR-1K Fix-Loop Coordinator Prompt

You are coordinating a BEHAVIOR-1K R1Pro fix-loop experiment.
Your job is to build an interactive policy block by block, not to write a full
static policy in one pass.

This prompt covers one seed or an ad hoc shared-policy experiment. If the user
requested the canonical ASPIRE campaign, stop and use
`../aspire-protocol/main-agent-prompt.md` instead.

## Startup

1. Read `.claude/memory/MEMORY.md`.
2. Read `.claude/behavior/CLAUDE.md`.
3. Read `.claude/behavior/fix-loop/SKILL.md`.
4. Read sections 1-8 of `.claude/behavior/skills/system-pipeline.md`; use
   them for launch modes, trace artifacts, configs, and perception servers.
5. Read `.claude/behavior/skills/interactive-policy.md`.
6. Read the relevant task skills under `.claude/behavior/skills/`.

## Workflow

1. Select one supported radio or soda config from `env_configs/r1pro/`.
2. Select the policy file under `outputs/interactive/`.
3. Append 5-20 lines of code.
4. Replay the same seed with `scripts/behavior/replay_trial_b1k.py --replay-code`
   and the bare `--record-video` flag.
5. Inspect traced output and saved observations.
6. Classify failures as perception, search, navigation, grasping, sequencing, time
   budget, or setup.
7. Append the next block based on evidence.
8. For an ad hoc shared-policy experiment, evaluate first on traced debug seeds,
   then on the non-traced validation config. Record trial count, seed range,
   success count, and dominant failure modes. A canonical campaign must return
   to `aspire-protocol` instead.
9. Update skill notes only when a pattern is reusable across tasks.

## Constraints

- Do not use simulator internals, object registries, BDDL predicates, or reward
  state in generated policy code.
- Keep BEHAVIOR work in the B1K environment.
- Preserve outputs needed for videos, stdout/stderr, and observations.
- Always record video; use the bare `--record-video` flag.
- Do not run `uv sync`.
- Do not push from the fix-loop.
- If selecting an agent model, use the Opus 1M-context option from
  `.claude/behavior/CLAUDE.md`.
