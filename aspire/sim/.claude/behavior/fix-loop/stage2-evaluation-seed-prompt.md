# Stage 2 Evaluation-Seed Prompt

You are a fresh isolated agent for held-out seed `<SEED>` of `<TASK>`. Use
`<CONFIG>`, frozen skills at `<FROZEN_SKILLS>`, private directory `<SEED_DIR>`,
GPU `<GPU>`, and fixed limits `<BUDGETS>`.

Follow all of `.claude/behavior/fix-loop/INSTRUCTIONS.md` and use only public
R1Pro APIs.

Verify the frozen hashes. Read only fixed suite/API docs, the frozen skills,
config, and your seed directory. Never read Stage 1 artifacts, another Stage 2
seed, coordinator diagnostics, or mutable skills.

Start with an empty `policy.py`. Build small `# Code block N` sections, replay
only your seed with `replay_trial_b1k.py --replay-code`, use a new
`attempt_NNN/` path each time, and inspect only your own evidence. Continue until
success or budget exhaustion.

This block-by-block debugging is the evaluation procedure, but this is the
seed's only agent episode. Do not request a replacement run or update skills.

Write `seed-summary.md`. Return only seed, terminal outcome, replay count, and
summary path—no policy details, diagnosis, or lessons—then end your context.
Do not invoke the built-in model loop, push, delete evidence, run another seed,
or touch hardware.
