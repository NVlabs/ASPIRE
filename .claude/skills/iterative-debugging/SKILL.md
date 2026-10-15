---
name: iterative-debugging
description: "Run the baseline-free CaP iterative debugging (fix loop) pipeline on a user-supplied list of benchmark tasks: inspect one initial observed scene per task, generate task-level code, debug failures on development seeds using recorded run artifacts, validate on the held-out partition, grade the skill library against what actually worked, and promote only those patterns a verifier could reproduce the fix from."
---

# Iterative Debugging (Fix Loop)

For BEHAVIOR-1K (R1 Pro on Isaac Sim) use the sibling folder
[behavior/SKILL.md](behavior/SKILL.md). It is a **different protocol**, not just different
prompts: those tasks are long-horizon, so each seed grows its own policy one code block at a
time, and the campaign freezes a skill library rather than a program. Everything below is for
the tabletop benchmarks (LIBERO-Pro, Robosuite).

Use this skill to run a coordinated debugging campaign over any list of CaP benchmark tasks.
No external baseline code or baseline output directory is used. Runs use
the benchmark's own `--init-mode` (LIBERO: `seeded`, so development seeds 51–65 and the
held-out partition are distinct scenes; Robosuite uses `saved`).

## Setup

1. Complete benchmark setup for the target embodiment (`docs/libero-pro.md`,
   or `docs/robosuite.md`), including the pinned environment (for LIBERO:
   `scripts/bootstrap_libero.sh` → `.venv-libero`) and SAM3 authentication.
2. Export the benchmark environment variables (for LIBERO: `LIBERO_CONFIG_PATH="$PWD/.libero"`,
   `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `EGL_PLATFORM=device`,
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`).
3. Start perception services (`scripts/supervise_services.sh`) and a persistent coordinator
   session.
4. Read `docs/api-reference.md`, `docs/run-artifacts.md`, and
   [main-agent-prompt.md](main-agent-prompt.md).

## Run Order

1. Take the task list from the user (any set of `(benchmark, suite, task_id)` cases), write it
   to a JSON file (format in [scripts/gen_progress.py](scripts/gen_progress.py)), and initialize
   the campaign root:
   `python3 .claude/skills/iterative-debugging/scripts/init_run.py --tasks <tasks.json>`.
   This creates `RUN_ROOT=outputs/runs/<stamp>/` (and points `outputs/runs/LATEST` at it) —
   every artifact of this campaign lives under it, one folder per task at
   `$RUN_ROOT/<benchmark>/<suite>/task_<id>/`. Generate progress with
   `python3 .claude/skills/iterative-debugging/scripts/gen_progress.py`.
   For Robosuite, append `--dev-seeds $(seq 101 125) --unseen-seeds $(seq 126 130)
   --heldout-count 100` to initialization. LIBERO defaults remain development 51–65,
   unseen-check 66–80, and held-out 1–50. Read the assigned development seeds, unseen-check
   seeds, and HELDOUT_COUNT from `campaign.json`, preserving those settings when resuming.
2. Follow [main-agent-prompt.md](main-agent-prompt.md) as coordinator. You are also the
   curriculum: choose each round's tasks from what the library can currently teach, and record
   the decision in `$RUN_ROOT/curriculum.md`. Dispatch [subagent-prompt.md](subagent-prompt.md)
   once per `pending` task. Workers explore
   ([skills/task-exploration.md](skills/task-exploration.md)), generate initial code, debug
   the assigned development seeds without reading external baseline outputs, and grade the
   skills they used in `skill_report.json`. Debugging is organized by failure mode rather than
   by seed, and every version is scored on all assigned development seeds. The selected program
   is then measured once on the assigned unseen-check seeds (LIBERO: 66–80; Robosuite: 126–130)
   before being handed over.
3. The coordinator (not workers) runs the Stage 2 held-out evaluation (seeds 1–$HELDOUT_COUNT) for
   each `stage1-done` task with [scripts/run_validation.py](scripts/run_validation.py).
4. When a task proposes skills, dispatch [verifier-prompt.md](verifier-prompt.md) on the same
   task. The verifier re-solves it from the skill library and the proposals alone — without the
   proposer's code — on the first few assigned development seeds (LIBERO: 51–55) within two
   attempts. **Only `verified` proposals are promoted into [skills/](skills/).**
5. Use [clean-task-slate.md](clean-task-slate.md) before reruns.
6. Continue until every task has all $HELDOUT_COUNT held-out results, then run the phase-3 rerun
   sweep once: tasks that scored below threshold under a library that has since changed are
   re-run against the final library, into `rerun_02/` beside the original attempt.
