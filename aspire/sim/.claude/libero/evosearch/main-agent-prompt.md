---
name: libero-evosearch-main-agent-prompt
description: Coordinator guide for running Evolutionary Search-style iterative debugging on a list of low-performing LIBERO-Pro tasks. Each subagent iterates on seeds 51–65; coordinator runs Stage 2 (seeds 1–50) after convergence.
---

# Evolutionary Search Multi-Task — Coordinator Guide

> **What:** Run Evolutionary Search iterative debugging (K=8 candidates, seeds 51–65) autonomously on a list of tasks. Stage 2 (seeds 1–50) runs once per task after convergence, coordinated from here.
> **Why:** Intensive debugging for difficult tasks — Evolutionary Search's multi-candidate search finds strategies the single-pass actor misses.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)

---

## Initialization: Verify Perception Servers

```bash
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```
404 = UP, 000 = DOWN. All three must be UP before dispatching.

---

## Directory Layout

```
outputs/
  claude_evosearch/
    <suite>/<task>/<run_id>/       ← Evolutionary Search iteration artifacts (existing convention)
      task_analysis.md
      iter_00/ iter_01/ ...
        candidate_A/code.py
        iter_summary.json
    <suite>/<task>/
      evosearch_best_code.py          ← best code from Stage 1 iterations (written by subagent)
      findings.md                  ← subagent summary + Stage 2 result

  aspire_evosearch_eval/
    <suite>/<task>/trial_*_sandboxrc_*_reward_*_taskcompleted_*/  ← Stage 2 results
```

---

## Task List — All <80% Tasks (3 suites, in dispatch order)

Run `gen_progress.py` to verify exact task names before dispatch.

> **Dispatch order:** Goal Task → Goal Swap → Spatial Swap → (Spatial Task, later)
> **Threshold:** anything <80% in the ASPIRE actor run/rerun is a candidate.

---

## The Loop

```
read progress → assign free GPUs (3–7) → dispatch subagents → GO IDLE
                                                                    ↑
on notification: redispatch freed GPU to next pending task → GO IDLE ┘
```

One task = one subagent = one GPU. The subagent runs Evolutionary Search iterations (seeds 51–65) AND Stage 2 (seeds 1–50) before returning. When the coordinator gets a completion notification, the GPU is already free — just dispatch the next task.

---

## Coordinator Rules

1. **Dispatch subagents — never run iterations yourself.**
2. **Go idle after dispatching.** You will be notified when a subagent finishes.
3. **Keep all 5 GPUs (3–7) occupied.**
4. **On each notification: check which GPU freed up, dispatch next pending task to it.**
5. **NEVER re-dispatch a done task** — `done` = Stage 2 complete (≥45 seeds in `aspire_evosearch_eval/`).

---

## Workflow

### 1. Check progress

```bash
# Stage 1 complete (evosearch_best_code.py exists):
for suite in libero_goal_swap libero_goal_task libero_spatial_swap libero_spatial_task; do
  for task_dir in outputs/claude_evosearch/$suite/*/; do
    task=$(basename "$task_dir")
    if [ -f "$task_dir/evosearch_best_code.py" ]; then
      seeds=$(find outputs/aspire_evosearch_eval/$suite/$task -name "trial_*" -type d 2>/dev/null | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
      echo "$suite/$task: stage1-done, stage2=$seeds/50"
    fi
  done
done
```

### 2. Check free GPUs

```bash
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```

### 3. Dispatch subagents

Use a high-capability model for Evolutionary Search's multi-candidate trace diagnosis and cross-iteration refinement.

```python
Agent(
    description="Evolutionary Search actor: <suite_short>/<task_short> GPU<N>",
    subagent_type="general-purpose",
    model="opus",
    prompt=<filled template from subagent-prompt.md>,
    run_in_background=True
)
```

Send all dispatches in one message (up to 5, one per GPU), then stop.

### 4. On each completion: redispatch

When a subagent notification arrives (it has already completed both Stage 1 and Stage 2):

1. Read `outputs/claude_evosearch/$SUITE/$TASK/findings.md` — note Stage 1 and Stage 2 rates
2. Check free GPUs (§2) — the completed subagent's GPU is already free
3. Dispatch the next pending task to that GPU
4. Go idle

---

## Stopping Criteria Reference

Subagents stop Stage 1 when:
- Best candidate ≥ **80%** on seeds 51–65 → **solved**
- **5 iterations** completed → **max iterations**
- Best improvement < **5pp** for 2 consecutive iterations → **plateau**

If Stage 1 is BLOCKED: subagent skips Stage 2 and returns immediately.
