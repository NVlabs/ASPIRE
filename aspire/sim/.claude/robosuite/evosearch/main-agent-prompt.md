---
name: robosuite-evosearch-main-agent-prompt
description: Coordinator guide for K=8 Evolutionary Search on Robosuite Fix Loop programs. Stage 1 uses seeds 101-125; Stage 2 uses seeds 1-100 after selection.
---

# Evolutionary Search Multi-Task — Robosuite Coordinator Guide

> **What:** Improve existing Robosuite Fix Loop programs with K=8 candidate
> search on seeds 101-125. Each task subagent evaluates the selected program on
> seeds 1-100 only after Stage 1 stops.
>
> **Fix Loop seed programs:** `outputs/robosuite_fix_loop/<task>/fix_code.py` (produced by the Fix Loop experiment)
>
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)

## Mandatory Preflight and Confirmation Gate

Before starting services or dispatching subagents, report:

1. Hostname and the visible GPU inventory.
2. Explicit target tasks: `nut_assembly` and/or `two_arm_lift`. Never infer the
   task list from a threshold without showing the candidates and receiving
   confirmation.
3. Development partition: seeds 101-125 only.
4. Held-out partition: seeds 1-100, locked until a final candidate is selected.
5. Maximum scope per task: 8 candidates × 25 seeds × 5 iterations = 1,000
   Stage 1 trials, plus 100 Stage 2 trials.
6. Expected runtime derived from a measured replay on the active host; if no
   measurement exists, budget roughly 1-2 hours per task at ten parallel
   workers, plus diagnosis time.
7. GPU mapping for SAM3, GraspNet, PyRoKi, and each simulation subagent.
8. Required access: gated SAM3 weights, Contact-GraspNet source/checkpoints,
   and the coding-agent account. Provider keys must not enter generated-code
   worker environments.
9. Required services: SAM3 `:8114`, GraspNet `:8115`, PyRoKi `:8116`,
   and Molmo `:8122` (used as a baseline fallback).
10. Output roots:
    `outputs/robosuite_evosearch/` and
    `outputs/robosuite_evosearch_eval/`.

Wait for explicit user confirmation before launching.

## Initialization

Run from `$ASPIRE_ROOT`:

```bash
test -x .venv-robosuite/bin/python
.venv-robosuite/bin/python -c "import aspire, robosuite; print('robosuite env ok')"
test -f outputs/robosuite_fix_loop/nut_assembly/fix_code.py
test -f outputs/robosuite_fix_loop/two_arm_lift/fix_code.py

for p in 8114 8115 8116 8122; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```

For these services, `404` means the process is responding and `000` means it is
down. Start missing services from a persistent session using the documented
perception environment before dispatching.

## Directory Layout

```text
outputs/
  robosuite_evosearch/
    <task>/
      evosearch_best_code.py
      findings.md
      <run_id>/
        task_analysis.md
        iter_00/ ... iter_04/
          candidate_A/ ... candidate_H/
          iter_summary.json
  robosuite_evosearch_eval/
    <task>/
      trial_*_sandboxrc_*_reward_*_taskcompleted_*/
```

## Coordinator Loop

```text
confirmed task list -> assign one GPU per task -> dispatch available tasks -> idle
       ^                                                               |
       |                  completion frees GPU                          |
       +---------------------------------------------------------------+
```

Rules:

1. Coordinate and dispatch; do not perform task-specific debugging yourself.
2. One task owns one simulation GPU for its lifetime.
3. Do not open task `trace.json`, `code.py`, or `summary.txt`; completed
   `findings.md` is coordinator-readable.
4. Never dispatch a task whose Stage 2 already has all 100 unique held-out
   seeds.
5. Preserve prior runs. A rerun uses a new timestamped run directory or an
   explicitly approved alternate output root.

## Progress Check

```bash
for task in nut_assembly two_arm_lift; do
  best="outputs/robosuite_evosearch/$task/evosearch_best_code.py"
  eval_dir="outputs/robosuite_evosearch_eval/$task"
  seed_list=$(find "$eval_dir" -maxdepth 1 -type d -name 'trial_*' 2>/dev/null \
    | sed -nE 's#.*trial_([0-9]+)_.*#\1#p' | awk '{print $1 + 0}' \
    | sort -nu | paste -sd' ' -)
  seeds=$(wc -w <<<"$seed_list")
  if [ -f "$best" ]; then state=stage1-done; else state=pending; fi
  if [ "$seed_list" = "$(seq 1 100 | paste -sd' ' -)" ]; then state=done; fi
  echo "$task: $state, stage2=$seeds/100"
done
```

## Dispatch

Read [subagent-prompt.md](subagent-prompt.md), copy the fenced template
verbatim, and fill only:

- `TASK`
- `CONFIG`
- `FIX_CODE`
- `GPU`
- `BASELINE_RATE`
- `EVOSEARCH_DIR`
- `EVALDIR`

Use the highest-capability available coding model for cross-candidate trace
diagnosis. Dispatch only tasks confirmed in preflight.

## Task Reference

| Task | Config | Fix code | Held-out baseline | Notes |
|---|---|---|---|---|
| `nut_assembly` | `env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml` | `outputs/robosuite_fix_loop/nut_assembly/fix_code.py` | from Fix Loop seeds 1-100 | Single arm |
| `two_arm_lift` | `env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml` | `outputs/robosuite_fix_loop/two_arm_lift/fix_code.py` | from Fix Loop seeds 1-100 | Bimanual |

## Stopping Criteria

Stage 1 stops when either:

- the best candidate reaches 25/25 on seeds 101-125; or
- five complete K=8 iterations have run.

There is no early plateau stop. If an approach family stalls, use the remaining
budget for structurally different perception, grasp, manipulation, or motion
hypotheses. Stage 2 runs exactly once with the selected code.
