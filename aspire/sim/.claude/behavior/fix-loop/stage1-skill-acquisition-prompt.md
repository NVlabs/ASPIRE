# Stage 1 Skill-Acquisition Prompt

You are the learning agent for `<TASK>` using `<CONFIG>` on development seeds
26-35. Your campaign root is `<CAMPAIGN_ROOT>`, GPU is `<GPU>`, and fixed limits
are `<BUDGETS>`.

Follow all of `.claude/behavior/fix-loop/INSTRUCTIONS.md` and use only public
R1Pro APIs.

Never inspect or run seeds 1-25. You may reuse reasoning and policies across
development seeds and update only `skill-library-working/`.

For each seed 26-35:

1. Build its policy in small `# Code block N` sections.
2. Replay only that seed with `replay_trial_b1k.py --replay-code`, using a new
   `attempt_NNN/` directory every time.
3. Inspect its trace, observations, videos, summary, and perception logs.
4. Debug until success or budget exhaustion; preserve every attempt.
5. Write `seed-summary.md`.

Promote only reusable, evidence-backed public-API lessons. Record their source
seed and evidence path; never encode privileged state or seed coordinates.

Return to the coordinator after seed 35 or a stop condition. Do not evaluate
held-out seeds, freeze the library yourself, invoke the built-in model loop,
push, delete evidence, or touch hardware.
