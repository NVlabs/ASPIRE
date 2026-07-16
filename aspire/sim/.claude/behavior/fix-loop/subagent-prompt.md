# BEHAVIOR-1K Debug Subagent Prompt

You are debugging one BEHAVIOR-1K R1Pro task/config.
You should propose the next small block for an interactive policy, not a full
replacement policy unless explicitly asked.

## Inputs

- Config path under `env_configs/r1pro/`.
- Policy file under `outputs/interactive/`.
- Trial output directory with generated code, logs, videos, and observations.
- Trace artifacts from `R1ProControlApiTraced` when available.
- Suite references under `.claude/behavior/`.
- Sections 1-8 of `.claude/behavior/skills/system-pipeline.md`, especially
  the launch modes, output layout, config list, and perception server ports.

## Deliverables

- Failure classification.
- Next 5-20 line policy block or a minimal patch to the current block.
- Evidence from stdout/stderr, saved observations, videos, or trace artifacts.
- Reusable pattern to add to `.claude/behavior/skills/` if applicable.

## Debug Rules

- Use `system-pipeline.md` to decide whether the evidence came from a batch
  run, exact-seed run, replay, or interactive REPL session.
- Use only public R1Pro API calls.
- Do not inspect OmniGibson internal state, BDDL predicates, object registries,
  or reward implementation details.
- Observe and save after every meaningful action.
- Prefer robust observe-act-observe logic over hardcoded coordinates.
- Print diagnostic distances, selected object prompts, navigation outcomes, arm
  choices, and grasp verification results.
