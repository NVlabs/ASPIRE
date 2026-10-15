# Shared BEHAVIOR-1K Robot Skills

Observation-only knowledge for the R1 Pro pickup tasks, separate from the parent skill's tabletop
files because a mobile robot in Isaac fails in different ways. Pipeline orchestration lives one
level up.

| Skill | Purpose |
|---|---|
| [search.md](search.md) | Finding the object: prompts, turning, torso tilt, when to give up |
| [navigation.md](navigation.md) | Standoff and approach poses, servo versus cuRobo, the odom frame |
| [mobile-grasp.md](mobile-grasp.md) | Grasp candidates on R1 Pro, arm choice, approach depth, the lift |
| [time-budget.md](time-budget.md) | What each call costs in steps and wall clock |
| [isaac-operations.md](isaac-operations.md) | Launch, crashes, memory, exit codes |

## How these files are used

These five files are the **topic template** for a campaign's skill library. A campaign copies
their filenames and headings into `$RUN_ROOT/behavior/<suite>/skill-library/` and starts with the
contents empty, so that nothing tuned outside the learning partition reaches an evaluation seed.
Stage 1 fills the library, the freeze locks it, and every Stage 2 seed reads it and nothing else.

A library entry is a **trigger**, a **snippet that actually ran**, and its **provenance**: task,
seed, attempt and date. An entry with no working snippet is a note, not a skill. Durable patterns
are promoted back into the files here between campaigns, never during one.

All snippets use only the documented public CaP program APIs (`docs/api-reference.md`): no
imports except `math`, no NumPy, no simulator internals, and a top-level `result` variable per
program.

Every entry carries provenance. Entries marked **unverified (ASPIRE prose)** were taken from the
ASPIRE reference runbooks and have not been reproduced through this harness; promote them to
verified only with a task, seeds, run path and date.
