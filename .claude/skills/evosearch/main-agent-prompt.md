---
name: evosearch-main-agent-prompt
description: Coordinator guide for running Evolutionary Search-style iterative debugging on a list of low-performing CaP benchmark tasks. Each subagent iterates on seeds 51–65 and runs Stage 2 (seeds 1–50) itself; the coordinator just redispatches freed GPUs.
---

# Evolutionary Search Multi-Task — Coordinator Guide

> **What:** Run Evolutionary Search iterative debugging (K=8 candidates, seeds 51–65)
> autonomously on a list of tasks. Stage 2 (seeds 1–50) runs once per task after convergence,
> executed by the subagent before it returns.
> **Why:** Intensive debugging for difficult tasks — Evolutionary Search's multi-candidate
> search finds strategies the single-pass fix loop misses.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)

---

## Initialization: Verify Perception Services

```bash
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/openapi.json)"
done
```
200 = UP, 000 = DOWN. All three must be UP before dispatching. If any is down, start them with
`scripts/supervise_services.sh` in a persistent terminal.

---

## Directory Layout

```
outputs/
  evosearch/
    <benchmark>/<suite>/task_<id>/<run_id>/   ← Evolutionary Search iteration artifacts
      task_analysis.md
      iter_00/ iter_01/ ...
        candidate_A/code.py
        iter_summary.json
    <benchmark>/<suite>/task_<id>/
      evosearch_best_code.py                  ← best code from Stage 1 iterations (written by subagent)
      findings.md                             ← subagent summary + Stage 2 result

  evosearch_eval/
    <benchmark>/<suite>/task_<id>/runs/<run_id>/manifest.json   ← Stage 2 results
                                                                  (run_validation.py manifests)
```

---

## Task List

Take the task list from the user — anything below the success threshold in the fix loop
held-out results is a candidate (default threshold: <90%). Verify exact task ids before
dispatch, e.g. for LIBERO:

```bash
.venv-libero/bin/python3 -c "
from cap_harness.libero.registry import LiberoSuiteRegistry
for task in LiberoSuiteRegistry().enumerate_tasks(['<suite>']):
    print(f'  {task.task_id:2d}: {task.task_name}')
"
```

---

## The Loop

```
read progress → assign free worker GPUs → dispatch subagents → GO IDLE
                                                                    ↑
on notification: redispatch freed GPU to next pending task → GO IDLE ┘
```

One task = one subagent = one GPU. The subagent runs Evolutionary Search iterations
(seeds 51–65) AND Stage 2 (seeds 1–50) before returning. When the coordinator gets a completion
notification, the GPU is already free — just dispatch the next task.

---

## Coordinator Rules

1. **Dispatch subagents — never run iterations yourself.**
2. **Go idle after dispatching.** You will be notified when a subagent finishes.
3. **Keep every assigned worker GPU occupied.**
4. **On each notification: check which GPU freed up, dispatch next pending task to it.**
5. **NEVER re-dispatch a done task** — `done` = Stage 2 complete (a full 1–50 manifest in
   `outputs/evosearch_eval/`).
6. **Enforce the non-privileged provenance gate.** Dispatch the complete subagent template,
   including its forbidden-API and provenance rules. A returned task must explicitly report
   `Non-privileged API audit: PASS`. If the audit is missing or quarantined, preserve its
   artifacts, do not accept its Stage 2 results, and report the task as invalid.

---

## Workflow

### 1. Check progress

```bash
# Stage 1 complete (evosearch_best_code.py exists) + Stage 2 manifest counts:
for task_dir in outputs/evosearch/*/*/task_*/; do
  ref=${task_dir#outputs/evosearch/}
  if [ -f "$task_dir/evosearch_best_code.py" ]; then
    python3 - "$ref" << 'EOF'
import json, sys
from pathlib import Path
ref = sys.argv[1].rstrip("/")
best = (0, 0)
for manifest_path in Path("outputs/evosearch_eval", ref, "runs").glob("*/manifest.json"):
    manifest = json.loads(manifest_path.read_text())
    if (manifest["trials"], manifest["passes"]) > best:
        best = (manifest["trials"], manifest["passes"])
print(f"{ref}: stage1-done, stage2={best[1]}/{best[0]} of 50")
EOF
  fi
done
```

### 2. Check free GPUs

```bash
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```

(Adjust the GPU list to the set the user assigned; perception services own GPUs 0–2 by
default.)

### 3. Dispatch subagents

Use a high-capability model for Evolutionary Search's multi-candidate trace diagnosis and
cross-iteration refinement.

```python
Agent(
    description="Evosearch actor: <suite_short>/task_<id> GPU<N>",
    subagent_type="general-purpose",
    model="opus",
    prompt=<filled template from subagent-prompt.md>,
    run_in_background=True
)
```

Send all dispatches in one message (one per free GPU), then stop.

### 4. On each completion: redispatch

When a subagent notification arrives (it has already completed both Stage 1 and Stage 2):

1. Confirm the returned summary says `Non-privileged API audit: PASS`. If it is missing or says
   `QUARANTINED`, preserve the artifacts, reject any derived Stage 2 results, and report the task
   as invalid; stop handling that task.
2. Read `outputs/evosearch/$BENCHMARK/$SUITE/task_$TASK_ID/findings.md` — note Stage 1 and
   Stage 2 rates
3. Check free GPUs (§2) — the completed subagent's GPU is already free
4. Dispatch the next pending task to that GPU
5. Go idle

---

## Stopping Criteria Reference

Subagents stop Stage 1 when:
- Best candidate ≥ **80%** on seeds 51–65 → **solved**
- **5 iterations** completed → **max iterations**
- Best improvement < **5pp** for 2 consecutive iterations → **plateau**

If Stage 1 is BLOCKED: subagent skips Stage 2 and returns immediately.
