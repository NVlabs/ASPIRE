---
name: evosearch
description: "Run the CaP Fix Loop + Evolutionary Search experiment on a user-supplied list of low-performing benchmark tasks: iterative K=8 candidate search on development seeds 51–65, validation selection, and final held-out evaluation on seeds 1–50."
---

# Fix Loop + Evolutionary Search

Use this skill for intensive debugging of difficult tasks — Evolutionary Search's
multi-candidate search finds strategies the single-pass fix loop misses. It builds on the
`iterative-debugging` skill: each task's fix loop `fix_code.py` (when present) is the baseline
to beat, seeded verbatim as `candidate_A`. LIBERO-Pro runs use `--init-mode seeded` so
development seeds 51–65 and held-out seeds 1–50 are distinct scenes.

## Setup

1. Complete benchmark setup for the target embodiment (`docs/libero-pro.md`,
   or `docs/robosuite.md`), including the pinned environment (for LIBERO:
   `scripts/bootstrap_libero.sh` → `.venv-libero`) and SAM3 authentication.
2. Export the benchmark environment variables (for LIBERO: `LIBERO_CONFIG_PATH="$PWD/.libero"`,
   `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `EGL_PLATFORM=device`,
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`).
3. Start perception services (`scripts/supervise_services.sh`) and a persistent coordinator
   session.
4. Confirm the Fix Loop experiment has produced `fix_code.py` (and a baseline held-out rate)
   for the target tasks when available.
5. Read `docs/api-reference.md`, `docs/run-artifacts.md`, and
   [main-agent-prompt.md](main-agent-prompt.md).

## Run Order

1. Take the task list from the user (tasks below the success threshold; any set of
   `(benchmark, suite, task_id)` cases) and follow
   [main-agent-prompt.md](main-agent-prompt.md) as the coordinator.
2. Fill [subagent-prompt.md](subagent-prompt.md) once per task and dispatch subagents on the
   available worker GPUs.
3. Subagents read the experiment-specific companion skills under [skills/](skills/) when
   generating candidates, iterate with [scripts/evosearch_eval.py](scripts/evosearch_eval.py),
   and run the Stage 2 held-out evaluation themselves with the shared
   `.claude/skills/iterative-debugging/scripts/run_validation.py` before returning.
4. Use [clean-task-slate.md](clean-task-slate.md) before rerunning a task or suite.

## Files

| File | Purpose |
|---|---|
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator runbook for task dispatch and result collection |
| [subagent-prompt.md](subagent-prompt.md) | Per-task Evolutionary Search iteration prompt |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist before reruns |
| [skills/evosearch-iteration.md](skills/evosearch-iteration.md) | Candidate/eval/keyframe iteration mechanics |
| [scripts/evosearch_eval.py](scripts/evosearch_eval.py) | Candidate × seed matrix evaluator |
| [scripts/analyze_evosearch_traces.py](scripts/analyze_evosearch_traces.py) | Post-eval per-candidate signal analysis |
| `docs/api-reference.md` | Public CaP program API reference |
| [../iterative-debugging/skills/](../iterative-debugging/skills/) | Shared robot skills (localize/grasp/transport/manipulation) |
