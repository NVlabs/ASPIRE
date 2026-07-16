---
name: libero-library-size-snapshot-eval
description: Evaluate frozen LIBERO skill-library snapshots on LIBERO-Long-Pro using split code-generation and seed-execution phases.
---

# Eval Strategy — LIBERO-Long-Pro

## Mode Index

| Mode | When to use | Skill file | Subagent template |
|---|---|---|---|
| **One-Shot (split)** — PREFERRED | Full measurement, no debug | This file (below) | [subagent-prompt.md](subagent-prompt.md) |
| **One-Shot (legacy)** | Same, but subagent runs seeds too | This file (below) | [legacy-seed-running-subagent-prompt.md](legacy-seed-running-subagent-prompt.md) |
| **Debug+Eval** (Stage1 debug + Stage2 held-out) | Inference-time scaling | [../inference-time-scaling/main-agent-prompt.md](../inference-time-scaling/main-agent-prompt.md) | [../inference-time-scaling/subagent-prompt.md](../inference-time-scaling/subagent-prompt.md) |

### One-Shot Split Architecture (preferred)

Subagents frequently stall during long seed-execution loops (600s watchdog kills them).
The split approach separates code generation from execution:

1. **Phase 1 — Code generation (subagents):** Dispatch 20 subagents using [subagent-prompt.md](subagent-prompt.md). Each reads the frozen skill library, probes the scene, writes ONE program, saves to `$OUTDIR/$SUITE/$TASK/code.py`, and returns immediately. ~10-20 min per subagent.
2. **Phase 2 — Seed execution (bash):** Coordinator collects code paths and runs all 1000 seeds via `scripts/libero/resume_eval_gpu.sh` with `ASPIRE_ROOT_OVERRIDE`. Pure bash, no watchdog, no stalls. ~2 hours across GPUs 3-7.

Code is saved to: `outputs/scaling_eval/<SNAPSHOT>/one_shot/<SUITE>/<TASK>/code.py`
`replay_trial.py` also copies code into each trial dir as `code.py` for provenance.

---

# One-Shot Mode — LIBERO-Long-Pro (Zero-Shot)

> **When:** After all 18 build chunks are complete and tagged (`snapshot-N5` … `snapshot-N90`). Can also run on partial snapshots for interim measurement.
> **What:** For each snapshot, dispatch one subagent per task per suite from a frozen skill library worktree. Subagents write one program and execute it across seeds 1–50. No debug loop.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)
> **Output:** `outputs/scaling_eval/<snapshot>/one_shot/<suite>/<task>/trial_*/`

---

## ⚠️ Critical Warnings

**Suite name collision:** `libero_10_swap` and `libero_10_task` share identical task names. Always include the full suite in paths and logs.

**`_task` suite language remapping:** The BDDL filename is misleading — always use `env.handle.task_language` for the actual instruction.

**Frozen library:** Eval subagents read skills from the snapshot worktree. They cannot modify skills.

**Seeds 1–50 are the eval set.** Build subagents used seeds 51–80. Do not cross-contaminate.

---

## LIBERO-Long-Pro Task List

10 tasks, shared across both suites (`libero_10_swap` and `libero_10_task`):

| # | Task name |
|---|---|
| 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` |
| 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` |
| 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` |
| 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` |
| 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` |
| 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` |
| 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` |
| 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` |
| 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` |
| 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` |

---

## Coordinator Recipe

### Step 0 — Preflight

```bash
need_servers=false
for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health 2>/dev/null || echo 000)
  echo "port $p: $code"
  [[ "$code" == "000" ]] && need_servers=true
done
$need_servers && bash scripts/common/start_perception_servers.sh

# 2. Tag exists
git rev-parse refs/tags/$SNAPSHOT >/dev/null 2>&1 && echo "tag OK" || echo "ERROR: tag missing"

# 3. Free GPUs
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]' || echo 0)
  echo "GPU $gpu: $( [ "$procs" -eq 0 ] && echo FREE || echo "BUSY ($procs procs)" )"
done
```

All three must pass before dispatching.

### Step 1 — Set up worktree

```bash
SNAPSHOT="snapshot-N50"   # set to the snapshot being evaluated
ASPIRE_ROOT="$PWD"
WORKTREE="$ASPIRE_ROOT/outputs/worktrees/$SNAPSHOT"

bash scripts/libero/eval_setup_worktree.sh --snapshot "$SNAPSHOT"
# creates $WORKTREE if not already present; idempotent
```

### Step 2 — Record eval start timestamp

```bash
EVAL_START_TS=$(python3 scripts/common/chunk_tokens.py --print-timestamp)
echo "Eval start: $EVAL_START_TS"
```

### Step 3a — Phase 1: Dispatch 20 code-generation subagents

Each subagent writes ONE program and saves to `$OUTDIR/$SUITE/$TASK/code.py`, then returns.
No seed execution — that's Phase 2.

```python
import itertools

SNAPSHOT = "snapshot-N50"
ASPIRE_ROOT = "$ASPIRE_ROOT"
WORKTREE = f"{ASPIRE_ROOT}/outputs/worktrees/{SNAPSHOT}"
SUITES = ["libero_10_swap", "libero_10_task"]
TASKS = [
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

# 1. Generate prompt files (extract template lines 21-404 from codegen prompt)
# Template: .claude/libero/library-size-scaling/subagent-prompt.md
# Extract content between outer ``` markers and do substitutions per task:
gpu_cycle = itertools.cycle([3, 4, 5, 6, 7])

for suite in SUITES:
    for task in TASKS:
        gpu = next(gpu_cycle)
        taskshort = task[:35].lower().replace(" ", "_")
        # Read template, substitute placeholders, save to /tmp/
        # Then dispatch:
        Agent(
            description=f"Codegen {SNAPSHOT} {suite[-4:]}/{task[:28]}",
            subagent_type="general-purpose",
            model="opus",
            prompt=f"Read /tmp/eval_{SNAPSHOT}_prompt_{idx}.txt and follow every instruction exactly. "
                   f"Perception servers running on 8114/8115/8116. .venv-libero/bin/python3 works. "
                   f"~/.libero/config.yaml exists. Do NOT view PNG images with Read tool. "
                   f"Write code and RETURN — do NOT run seeds.",
            run_in_background=True,
        )
```

Wait for all 20 to return. Each saves code to:
`outputs/scaling_eval/<SNAPSHOT>/one_shot/<SUITE>/<TASK>/code.py`

### Step 3b — Phase 2: Execute all seeds via bash

After all 20 code files exist, run seeds with `resume_eval_gpu.sh`:

```python
# Verify all 20 code files exist
OUTDIR = f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/one_shot"
missing = []
for suite in SUITES:
    for task in TASKS:
        code = f"{OUTDIR}/{suite}/{task}/code.py"
        if not os.path.exists(code):
            missing.append(f"{suite}/{task}")
if missing:
    print(f"MISSING CODE for {len(missing)} tasks: {missing}")
    # Re-dispatch failed subagents or handle manually
else:
    print("All 20 code files present — launching seed execution")
```

```bash
# Generate per-GPU resume scripts (balance by 50 seeds each = 200 seeds/GPU)
# Then run:
export ASPIRE_ROOT_OVERRIDE="$WORKTREE"
# ... per-GPU scripts calling resume_eval_gpu.sh for each task ...
# See scripts/libero/resume_eval_gpu.sh — it skips already-completed seeds automatically.
```

This runs ~1000 seeds across GPUs 3-7 in ~2 hours with no stall risk.

**One-command alternative:** `scripts/libero/eval_run_seeds.py` runs all of Phase 2 (every task, every GPU) in a single command and auto-skips completed seeds:
```bash
$WORKTREE/.venv-libero/bin/python3 scripts/libero/eval_run_seeds.py \
  --worktree "$WORKTREE" --snapshot "$SNAPSHOT" --gpus 3 4 5 6 7 --seeds-per-gpu 3
```

### Step 4 — Collect results

After all 20 subagents return (or after a 3-hour wall-clock timeout — mark stragglers as 0/50):

```bash
SNAPSHOT="snapshot-N50"
OUTDIR="outputs/scaling_eval/$SNAPSHOT/one_shot"

echo "=== Results: $SNAPSHOT ==="
for suite in libero_10_swap libero_10_task; do
  echo "--- $suite ---"
  for task_dir in "$OUTDIR/$suite"/*/; do
    task=$(basename "$task_dir")
    total=$(find "$task_dir" -maxdepth 1 -type d -name "trial_*" 2>/dev/null | wc -l)
    success=$(find "$task_dir" -maxdepth 1 -type d -name "*taskcompleted_1*" 2>/dev/null | wc -l)
    echo "  $task: $success/$total"
  done
done
```

### Step 5 — Write summary.json

```python
import json, os
from pathlib import Path

SNAPSHOT = "snapshot-N50"
OUTDIR = Path(f"outputs/scaling_eval/{SNAPSHOT}/one_shot")
SUITES = ["libero_10_swap", "libero_10_task"]

results = {}
total_success, total_trials = 0, 0

for suite in SUITES:
    results[suite] = {}
    suite_dir = OUTDIR / suite
    if not suite_dir.exists():
        continue
    for task_dir in sorted(suite_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        trials = list(task_dir.glob("trial_*"))
        success = sum(1 for t in trials if "taskcompleted_1" in t.name)
        results[suite][task] = {"success": success, "total": len(trials)}
        total_success += success
        total_trials += len(trials)

summary = {
    "snapshot": SNAPSHOT,
    "mode": "one_shot",
    "seeds": "1-50",
    "suites": SUITES,
    "results": results,
    "aggregate": {
        "success": total_success,
        "total": total_trials,
        "rate": round(total_success / total_trials, 4) if total_trials > 0 else 0,
    },
}

out = OUTDIR / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"Written: {out}")
print(f"Overall: {total_success}/{total_trials} = {summary['aggregate']['rate']*100:.1f}%")
```

### Step 6 — Annotate token usage

```bash
python3 scripts/libero/annotate_snapshot_tokens.py --force
# Updates the git note on the snapshot commit to include eval tokens
# (cumulative now includes both build and eval sessions)
```

---

## Long-Horizon Code Rules (for eval subagents)

1. `goto_home_joint_position()` **before every subtask** — prevents joint limit issues.
2. `get_observation()` **after every home reset** — arm movement shifts the camera view.
3. **Best-effort per subtask** — don't assume subtask 1 succeeded; attempt subtask 2 regardless.
4. **Always close** (drawer/microwave) even if the prior place step failed.
5. **Write for seeds 1–50** — no hardcoded coordinates; SAM3 prompts must generalize.
