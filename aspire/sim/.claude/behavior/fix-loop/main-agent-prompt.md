# BEHAVIOR-1K Fix-Loop Coordinator Prompt

Coordinate one canonical Soda Can or Radio campaign. Read the suite guide,
`.claude/behavior/fix-loop/SKILL.md`, and all of
`.claude/behavior/fix-loop/INSTRUCTIONS.md`.

Before doing anything operational, report the exact commit, task/config,
host/GPU, environment/services, model/context, fixed budgets, seeds, campaign
path, runtime, and risks. Wait for user approval.

After approval:

1. Create a fresh campaign and working skill-library copy.
2. Run Stage 1 on seeds 26-35 with
   `.claude/behavior/fix-loop/stage1-skill-acquisition-prompt.md`.
3. Freeze and checksum the skills and experimental contract.
4. Run seeds 1-25 sequentially. Start a new non-resumed context for each seed
   with `.claude/behavior/fix-loop/stage2-evaluation-seed-prompt.md`.
5. Preserve all attempts and resume only from the first incomplete state.
6. Aggregate detailed held-out results only after all 25 episodes terminate.

During Stage 2, accept only seed, outcome, replay count, and summary path from
each agent. Do not inspect or relay its policy, traces, diagnoses, or lessons
before the final aggregation.

All trials must use `replay_trial_b1k.py --replay-code`. Never use normal batch
generation or the built-in `REGENERATE`/`FINISH` loop. Run one Isaac Sim process,
do not push, and do not touch real hardware.
