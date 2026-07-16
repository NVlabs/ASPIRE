---
name: libero-inference-time-token-scaling-coordinator
description: Evaluate LIBERO-Long-Pro debug+eval at 0/25/50/75/100% of inference-time output tokens across all 20 (suite, task) pairs. One agent runs one snapshot end-to-end. Measures success rate as a function of debug compute.
---

# LIBERO-Long-Pro Token Scaling Eval Pipeline

> **Purpose:** Measure how success rate scales with inference-time debug compute (output tokens).
> **Scope:** One agent handles ONE snapshot end-to-end (all 20 task pairs).
> **Input:** A single snapshot tag (e.g., `snapshot-N0`).
> **Output:** Per-task success rates at 0%, 25%, 50%, 75%, 100% of total debug output tokens.

---

## Overview

1. **Stage 1 — Debug loop** (uses [subagent-prompt.md](subagent-prompt.md))
   - Dispatch 5 debug subagents at a time (one per GPU 3-7), 4 batches to cover all 20 pairs
   - Each debugs on seeds 51-65, up to 20 iterations, saving every code version
   - Produces `stage1/code_versions/iter_N_*.py` + `stage1_summary.json`
   - Skip any pair that already has `stage1_summary.json`

2. **Token analysis** — Compute cumulative output tokens per iteration
   - Run `scripts/libero/analyze_stage1_tokens.py` on each agent JSONL transcript
   - Pick code versions at 0%, 25%, 50%, 75%, 100% of cumulative output tokens

3. **Stage 2 — Held-out eval** (seeds 1-50)
   - Run seeds 1-50 on each of the 5 selected code versions per task
   - Output: `stage2_0pct/`, `stage2_25pct/`, `stage2_50pct/`, `stage2_75pct/`, `stage2_100pct/`

---

## Key Directories

```
ASPIRE_ROOT = $ASPIRE_ROOT
  (symlink → $ASPIRE_ROOT)

Skills:           $ASPIRE_ROOT/.claude/libero/inference-time-scaling/
Worktrees:        $ASPIRE_ROOT/outputs/worktrees/snapshot-N*/
Snapshot tags:    snapshot-N0, snapshot-N5, ..., snapshot-N90

Per-task output layout (inside worktree):
  outputs/scaling_eval/<snapshot>/debug_eval/<suite>/<task>/
    stage1/
      code_versions/          # every iter's code: iter_N_TIMESTAMP.py
      stage1_summary.json     # best iter, all iter results
      token_usage.json        # per-iter token breakdown
      task_code.py            # promoted best code
    stage2_0pct/              # eval at 0% output tokens (first code version)
      <suite>/<task>/<model>/run/trial_*/
      stage2_result.json
    stage2_25pct/
    stage2_50pct/
    stage2_75pct/
    stage2_100pct/            # eval at 100% (final best code)
```

---

## Full Task List — 20 (suite, task) pairs

10 tasks × 2 suites (`libero_10_swap`, `libero_10_task`):

| # | Task |
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

Each task runs on BOTH suites. Skip any (suite, task) pair that already has `stage1_summary.json`.

---

## Step-by-Step Recipe

### Step 0 — Preflight

```bash
SNAPSHOT="snapshot-N0"   # <-- set your snapshot
ASPIRE_ROOT="$ASPIRE_ROOT"

need_servers=false
for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health 2>/dev/null || echo 000)
  echo "port $p: $code"
  [[ "$code" == "000" ]] && need_servers=true
done
$need_servers && bash scripts/common/start_perception_servers.sh

# 2. Tag exists
git -C "$ASPIRE_ROOT" rev-parse refs/tags/$SNAPSHOT >/dev/null 2>&1 && echo "tag OK" || echo "ERROR: tag missing"

# 3. Free GPUs
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '\n' | grep -c '[0-9]' || echo 0)
  echo "GPU $gpu: $( [ "$procs" -eq 0 ] && echo FREE || echo BUSY )"
done
```

### Step 1 — Set up worktree

```bash
bash "$ASPIRE_ROOT/scripts/libero/eval_setup_worktree.sh" --snapshot "$SNAPSHOT"
WORKTREE="$ASPIRE_ROOT/outputs/worktrees/$SNAPSHOT"
```

### Step 2 — Dispatch Stage 1 debug subagents (4 batches of 5)

20 pairs, 5 GPUs -> 4 batches. Use the [subagent-prompt.md](subagent-prompt.md) template.

```python
import itertools

SNAPSHOT = "snapshot-N0"
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

# Build all 20 pairs, filter out already-done ones
ALL_PAIRS = [(suite, task) for suite in SUITES for task in TASKS]
pending = []
for suite, task in ALL_PAIRS:
    summary = f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}/stage1/stage1_summary.json"
    if not os.path.exists(summary):
        pending.append((suite, task))
    else:
        print(f"SKIP (done): {suite[-4:]}/{task[:40]}")

# Dispatch in batches of 5
GPUS = [3, 4, 5, 6, 7]
for batch_start in range(0, len(pending), 5):
    batch = pending[batch_start:batch_start+5]
    gpu_cycle = iter(GPUS)
    for suite, task in batch:
        gpu = next(gpu_cycle)
        # Fill in subagent prompt template and dispatch with run_in_background=True
        # ... (see subagent-prompt.md)
    # Wait for batch to complete before dispatching next batch
```

**Dispatch 5 at a time, wait for completion, then next batch.** Each batch takes ~1-2 hours.

### Step 3 — Token analysis

After ALL Stage 1 agents complete, run token analysis on each agent's JSONL transcript:

```bash
ASPIRE_ROOT="$ASPIRE_ROOT"
SNAPSHOT="snapshot-N0"

# For each (suite, task) pair:
python3 "$ASPIRE_ROOT/scripts/libero/analyze_stage1_tokens.py" \
  --jsonl <AGENT_JSONL_PATH> \
  --stage1 "$ASPIRE_ROOT/outputs/worktrees/$SNAPSHOT/outputs/scaling_eval/$SNAPSHOT/debug_eval/<SUITE>/<TASK>/stage1"
```

**Agent JSONL files** are symlinked at:
```
/tmp/claude/<session-id>/tasks/<agent-id>.output
→ ~/.claude/projects/.../subagents/agent-<id>.jsonl
```

**⚠️ Timezone warning:** If subagents name code versions `v0.py, v1.py` (not `iter_N_TIMESTAMP.py`), rename them first using file mtimes, then pass `--tz-offset 0` to the analyze script (the renamed filenames have UTC timestamps, but the script defaults to local time). Verify the output has multiple `per_iter` entries — a single bucket means the timezone was wrong.

### Step 4 — Pick code versions at token percentages

For each task, compute cumulative output tokens from `token_usage.json` and pick code versions at 0%, 25%, 50%, 75%, 100%.

- **0%** = first code version (before any debug feedback)
- **25/50/75%** = code version where cumulative output tokens is closest to that percentage of total
- **100%** = final/promoted code version

### Step 5 — Run Stage 2 evals (seeds 1-50)

For each task × pct level, run seeds 1-50. Use the eval script pattern:

```bash
# ALL PATHS MUST BE ABSOLUTE
WORKTREE="$ASPIRE_ROOT/outputs/worktrees/$SNAPSHOT"

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONPATH="$(cd "$WORKTREE/../.." && pwd)" \
"$WORKTREE/.venv-libero/bin/python3" "$WORKTREE/scripts/libero/replay_trial.py" \
    --args.suite "$SUITE" --args.task "$TASK" --args.trial $seed \
    --args.replay-code "$CODE" \
    --args.config "$WORKTREE/env_configs/libero/franka_libero_libero10_traced.yaml" \
    --args.output-dir "$OUTDIR"
```

**Parallel workers:** `xargs -P 4` across seeds.

**Trial nesting:** Output nests as `$OUTDIR/<suite>/<task>/<model>/run/trial_NN_*`. Use `find "$OUTDIR" -type d -name "trial_*"` (no maxdepth limit).

**Batching Stage 2:** 20 pairs × 5 pct levels = 100 evals per snapshot. With 5 GPUs, assign 4 pairs per GPU (each runs 5 pct levels sequentially). Each 50-seed eval takes ~30-45 min, so ~2.5-3.5 hours per GPU.

### Step 6 — Collect results

```python
import json
from pathlib import Path

SNAPSHOT = "snapshot-N0"
ASPIRE_ROOT = "$ASPIRE_ROOT"
WORKTREE = f"{ASPIRE_ROOT}/outputs/worktrees/{SNAPSHOT}"
SUITES = ["libero_10_swap", "libero_10_task"]
TASKS = [...]  # all 10

for suite in SUITES:
    for task in TASKS:
        for pct in ["0pct", "25pct", "50pct", "75pct", "100pct"]:
            rf = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}/stage2_{pct}/stage2_result.json")
            if rf.exists():
                d = json.loads(rf.read_text())
                print(f"{suite[-4:]}/{task[:30]}: {pct} = {d['n_pass']}/{d['n_total']} = {d['pass_rate']*100:.0f}%")
```

---

## Tips & Gotchas

1. **Absolute paths everywhere.** Every path to replay_trial.py, config, code, output dir must be absolute. #1 source of bugs.

2. **Trial directories are deeply nested** (5 levels). Never use `find -maxdepth 2` — use unlimited depth.

3. **Code version naming varies.** Subagents may write `iter_1_20260503_215837.py` or `v0.py`. If using `v*.py`, rename to `iter_N_TIMESTAMP.py` using file mtimes before running token analysis, and pass `--tz-offset 0`.

4. **Seeds 1-50 are held-out.** Stage 1 debug uses seeds 51-65 only.

5. **`libero_10_task` suite language remapping.** The BDDL task name is misleading — always use `env.handle.task_language` for the actual instruction.

6. **Skip completed work.** Check for existing `stage1_summary.json` before dispatching Stage 1, and existing `stage2_result.json` (with `n_total > 0`) before running Stage 2.

7. **SAM3 contention.** 5 GPUs × 4 parallel workers = 20 concurrent SAM3 requests. Servers handle it but it's slower.

8. **GPU assignment.** 5 GPUs (3-7), one task per GPU. Batch 5 at a time for Stage 1, then 5 at a time for Stage 2 (each running through all 5 pct levels).

---

## Related Skills

- [main-agent-prompt.md](main-agent-prompt.md) — Stage 1 debug+eval coordinator recipe (original 5-task version)
- [subagent-prompt.md](subagent-prompt.md) — Subagent prompt template for Stage 1 debug
- `SKILL.md` — Eval strategy overview, full task list, long-horizon code rules
- [../analysis/token-calculate.md](../analysis/token-calculate.md) — Token usage analysis patterns
- `scripts/libero/analyze_stage1_tokens.py` — Per-iter token bucketing
