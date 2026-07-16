---
name: behavior/fix-loop
description: BEHAVIOR-1K R1Pro block-by-block interactive policy fix-loop for radio and soda-can tasks.
---

# BEHAVIOR-1K Fix Loop Skill

Use this skill when building or debugging R1Pro BEHAVIOR interactive policies in
ASPIRE. The policy is written block by block, replayed on the same seed after
each append, and grown into one long observe-act-observe state machine.

## Read First

- `.claude/behavior/CLAUDE.md`
- `.claude/behavior/api-reference.md`
- `.claude/behavior/skills/system-pipeline.md`
- `.claude/behavior/skills/README.md`
- `.claude/behavior/fix-loop/INSTRUCTIONS.md`
- `.claude/behavior/skills/interactive-policy.md`
- `.claude/behavior/skills/search.md`
- `docs/behavior-tasks.md`

Read sections 1-8 of `system-pipeline.md` before changing launch commands or
interpreting traces. They cover the current ASPIRE/B1K scope, execution flow,
launch modes, API hierarchy, TraceLogger, output layout, config set, and
perception servers.

## Supported Task Set

- `env_configs/r1pro/r1pro_pick_up_radio.yaml`
- `env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml`
- `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml`
- `env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml`
- `env_configs/r1pro/r1pro_pick_up_trash.yaml`
- `env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml`
- `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml`
- `env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml`

The `r1pro_pick_up_trash*.yaml` files are the legacy filenames for soda-can
pickup.

## Debug Loop

1. Pick the target policy file:
   `outputs/interactive/fix_code_interactive_radio.py` for radio or
   `outputs/interactive/fix_code_interactive.py` for soda.
2. Append 5-20 lines of policy code.
3. Replay the same seed with `scripts/behavior/replay_trial_b1k.py --replay-code` and
   `--record-video True`.
4. Inspect `summary.txt`, `trace.json`, `keyframes/`, videos, VDM feedback, and
   saved observations.
5. Classify the current blocker and append the next block based on evidence.
6. Repeat until search, approach, grasp, verification, and fallbacks are present.
7. Validate with the non-traced config and report seed range, success count,
   and common failure modes.

## Policy Pattern

Build closed-loop state-machine code:

1. observe and save the start view;
2. search with prompt alternatives;
3. estimate object/table pose with retry guards;
4. navigate and verify actual movement;
5. re-observe before grasping;
6. try interleaved grasp attempts across both arms;
7. save post-grasp observations and report failure mode.

Use only public R1Pro API calls. Do not inspect OmniGibson internals, BDDL
predicates, simulator object registries, or privileged reward state.

Do not write the whole policy in one pass. Do not skip videos. Do not run
`uv sync` in the B1K virtual environment. If choosing the coding model for an
interactive-policy agent, follow `.claude/behavior/CLAUDE.md` and use the Opus
1M-context option rather than a smaller model.
