---
name: libero-inference-time-debug-eval-coordinator
description: Two-stage eval for LIBERO-Long-Pro with debug loop on seeds 51-65, then held-out eval on seeds 1-50. Runs on 5 selected tasks per snapshot. Analogous to the LIBERO-Pro Stage1+Stage2 Evolutionary Search pipeline.
---

# LIBERO-Long-Pro Debug+Eval Pipeline (Stage 1 Debug → Stage 2 Held-Out)

> **When:** After `snapshot-NX` is tagged. Run on 5 selected (suite, task) pairs — 3 from `libero_10_swap`, 2 from `libero_10_task`.
> **Stage 1:** Subagent debugs on seeds 51–65 (15 seeds), up to 5 iterations, targeting ≥80%.
> **Stage 2:** Coordinator triggers held-out eval on seeds 1–50 with the best Stage 1 code.
> **Seeds 1–50 are NEVER touched during Stage 1.** Stage 2 is run once per task, once per snapshot.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)

---

## ⚠️ Critical Warnings

**Seeds 1–50 lockout during debug.** Stage 1 subagents use ONLY seeds 51–65. Running Stage 2 more than once invalidates the measurement.

**Stage 2 = DONE.** Once `stage2_result.json` exists for a task, that task is permanently closed — regardless of the pass rate. Do NOT re-run Stage 2.

**Frozen library.** Subagents read `.claude/libero/skills/` from the snapshot worktree. They CANNOT modify skills.

**`_task` suite language remapping.** Always use `env.handle.task_language` — never trust the BDDL filename for `libero_10_task`.

**Suite name collision.** `libero_10_swap` and `libero_10_task` share identical task names. Always include full suite in all paths.

---

## Output Layout

```
outputs/scaling_eval/<snapshot>/debug_eval/<suite>/<task>/
  stage1/
    code_versions/
      iter_1_20260427_143022.py   # timestamped at write time
      iter_2_20260427_151847.py
      ...
    iter_1/
      seed_outputs/trial_*/       # replay output dirs for seeds 51-65
      result.json                 # {pass_rate, n_pass, n_total, seeds, code}
      run.log
      seed51.log ... seed65.log
    iter_2/
      ...
    task_code.py                  # copy of best iter's code from code_versions/
    stage1_summary.json           # {best_iter, best_pass_rate, iters_run, promoted, all_iters}
    token_usage.json              # per-iter token counts written by coordinator after stage1 completes
  stage2/
    trial_*/                      # replay output dirs for seeds 1-50
    stage2_result.json            # {pass_rate, n_pass, n_total, seeds: "1-50"}
```

---

## Task Selection

5 selected (suite, task) pairs for debug+eval:

| Suite | Task |
|---|---|
| `libero_10_swap` | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` |
| `libero_10_swap` | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` |
| `libero_10_swap` | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` |
| `libero_10_task` | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` |
| `libero_10_task` | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` |

---

## Coordinator Recipe

### Step 0 — Preflight

```bash
SNAPSHOT="snapshot-N25"   # set to the snapshot being evaluated
ASPIRE_ROOT="$PWD"

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

### Step 1 — Set up worktree

```bash
bash scripts/libero/eval_setup_worktree.sh --snapshot "$SNAPSHOT"
WORKTREE="$ASPIRE_ROOT/outputs/worktrees/$SNAPSHOT"
```

### Step 2 — Check what's already done

Before dispatching, check which tasks already have `stage2_result.json` (DONE) or `stage1_summary.json` (Stage 1 complete, Stage 2 pending):

```bash
SNAPSHOT="snapshot-N25"
WORKTREE="$PWD/outputs/worktrees/$SNAPSHOT"

# Selected (suite, task) pairs — NOT a cross-product
declare -A SUITE_TASKS
SUITE_TASKS=(
  ["libero_10_swap/0"]="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  ["libero_10_swap/1"]="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  ["libero_10_swap/2"]="LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  ["libero_10_task/0"]="LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  ["libero_10_task/1"]="LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
)

for key in "${!SUITE_TASKS[@]}"; do
  suite="${key%%/*}"
  task="${SUITE_TASKS[$key]}"
  base="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$suite/$task"
  if [ -f "$base/stage2/stage2_result.json" ]; then
    rate=$(python3 -c "import json; d=json.load(open('$base/stage2/stage2_result.json')); print(f\"{d['pass_rate']*100:.0f}%\")")
    echo "  DONE (stage2=$rate): $suite/$task"
  elif [ -f "$base/stage1/stage1_summary.json" ]; then
    rate=$(python3 -c "import json; d=json.load(open('$base/stage1/stage1_summary.json')); print(f\"{d['best_pass_rate']*100:.0f}%\")")
    echo "  STAGE2_PENDING (stage1=$rate): $suite/$task"
  elif [ -d "$base/stage1" ]; then
    echo "  IN_PROGRESS: $suite/$task"
  else
    echo "  PENDING: $suite/$task"
  fi
done
```

### Step 3 — Dispatch Stage 1 subagents

Dispatch one subagent per pending (suite, task) pair. Use `run_in_background=True`.
GPUs 3–7, one pair per GPU (5 total).

```python
import itertools

SNAPSHOT = "snapshot-N25"
ASPIRE_ROOT = "$ASPIRE_ROOT"
WORKTREE = f"{ASPIRE_ROOT}/outputs/worktrees/{SNAPSHOT}"

# Selected (suite, task) pairs — NOT a cross-product
SUITE_TASK_PAIRS = [
    ("libero_10_swap",  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"),
    ("libero_10_swap",  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
    ("libero_10_swap",  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"),
    ("libero_10_task",  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"),
    ("libero_10_task",  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
]

prompt_template = open(f"{ASPIRE_ROOT}/.claude/libero/inference-time-scaling/subagent-prompt.md").read()

gpu_cycle = itertools.cycle([3, 4, 5, 6, 7])

import os
for suite, task in SUITE_TASK_PAIRS:
    summary = f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}/stage1/stage1_summary.json"
    if os.path.exists(summary):
        print(f"SKIP (stage1 done): {suite}/{task[:40]}")
        continue

    gpu = next(gpu_cycle)
    taskshort = task.split("_", 2)[2][:30].lower().replace(" ", "_")

    prompt = (prompt_template
        .replace("<SNAPSHOT>", SNAPSHOT)
        .replace("<SUITE>", suite)
        .replace("<TASK>", task)
        .replace("<GPU>", str(gpu))
        .replace("<TASKSHORT>", taskshort)
        .replace("<ASPIRE_ROOT_SNAPSHOT>", WORKTREE))

    Agent(
        description=f"Debug {SNAPSHOT} {suite[-4:]}/{task[:28]}",
        subagent_type="general-purpose",
        model="opus",
        prompt=prompt,
        run_in_background=True,
    )
    print(f"Dispatched GPU{gpu}: {suite}/{task[:50]}")
```

Then **go idle**. You will be notified as each subagent completes.

### Step 4 — On subagent completion: analyze tokens, check, trigger Stage 2

When a subagent returns, the completion notification includes the `output-file` path (the agent JSONL).
Run token analysis first, then trigger Stage 2.

```python
import json, subprocess
from pathlib import Path

SNAPSHOT = "snapshot-N25"
ASPIRE_ROOT = "$ASPIRE_ROOT"
WORKTREE = f"{ASPIRE_ROOT}/outputs/worktrees/{SNAPSHOT}"

def analyze_tokens(suite, task, agent_jsonl_path):
    """Write stage1/token_usage.json from the agent JSONL output."""
    stage1_dir = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}/stage1")
    if not Path(agent_jsonl_path).exists():
        print(f"WARN: agent JSONL not found at {agent_jsonl_path} — skipping token analysis")
        return
    result = subprocess.run(
        ["python3", f"{ASPIRE_ROOT}/scripts/libero/analyze_stage1_tokens.py",
         "--jsonl", agent_jsonl_path,
         "--stage1", str(stage1_dir)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARN: token analysis failed: {result.stderr[:200]}")

def trigger_stage2(suite, task, gpu, agent_jsonl_path=None):
    """Analyze tokens, check stage1 done, then run stage2."""
    base = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}")
    s1_summary = base / "stage1" / "stage1_summary.json"
    s2_result  = base / "stage2" / "stage2_result.json"

    # Token analysis — run regardless of stage2 status
    if agent_jsonl_path:
        analyze_tokens(suite, task, agent_jsonl_path)

    if s2_result.exists():
        print(f"Stage2 already done: {suite}/{task[:40]}")
        return

    if not s1_summary.exists():
        print(f"ERROR: no stage1_summary.json for {suite}/{task[:40]}")
        return

    s1 = json.loads(s1_summary.read_text())
    print(f"Stage1 done: {suite}/{task[:40]} — best={s1['best_pass_rate']*100:.0f}% ({s1['iters_run']} iters)")

    task_code = base / "stage1" / "task_code.py"
    if not task_code.exists():
        print(f"ERROR: no task_code.py for {suite}/{task[:40]}")
        return

    # Dispatch stage2 as a background agent
    taskshort = task.split("_", 2)[2][:30].lower().replace(" ", "_")
    Agent(

        description=f"Stage2 {SNAPSHOT} {suite[-4:]}/{task[:28]}",
        subagent_type="general-purpose",
        model="sonnet",  # stage2 is just execution, no reasoning needed
        prompt=f"""
Run Stage 2 eval for LIBERO-Long-Pro task.

SNAPSHOT: {SNAPSHOT}
SUITE: {suite}
TASK: {task}
GPU: {gpu}
TASKSHORT: {taskshort}
WORKTREE: {WORKTREE}
CODE: {task_code}
STAGE2_DIR: {base}/stage2

## Your job

Run the existing task code on seeds 1–50 and write stage2_result.json. No debugging, no code changes.

```bash
WORKTREE="{WORKTREE}"
SUITE="{suite}"
TASK="{task}"
SNAPSHOT="{SNAPSHOT}"
GPU={gpu}
OUTDIR="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$SUITE/$TASK/stage2"
CODE="{task_code}"
LOG="/tmp/stage2_{taskshort}.log"

mkdir -p "$OUTDIR"
echo "Stage2 start: $SUITE/$TASK seeds 1-50" | tee "$LOG"

for seed in $(seq 1 50); do
    trial_padded=$(printf "%02d" $seed)
    if find "$OUTDIR" -type d -name "trial_${{trial_padded}}_*" 2>/dev/null | grep -q .; then
        echo "Seed $seed: skip" | tee -a "$LOG"; continue
    fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \\
    PYTHONPATH="$(cd "$WORKTREE/../.." && pwd)" \\
    "$WORKTREE/.venv-libero/bin/python3" scripts/libero/replay_trial.py \\
        --args.suite "$SUITE" --args.task "$TASK" --args.trial $seed \\
        --args.replay-code "$CODE" \\
        --args.config env_configs/libero/franka_libero_libero10_traced.yaml \\
        --args.output-dir "$OUTDIR" > "/tmp/stage2_{taskshort}_seed${{seed}}.log" 2>&1 || true
    result=$(grep -oE "taskcompleted_[01]" "/tmp/stage2_{taskshort}_seed${{seed}}.log" | tail -1 || echo "ERROR")
    echo "Seed $seed: $result" | tee -a "$LOG"
done
```

Then count results and write stage2_result.json:

```python
import json
from pathlib import Path

outdir = Path("{base}/stage2")
trials = list(outdir.glob("trial_*"))
n_pass = sum(1 for t in trials if "taskcompleted_1" in t.name)
n_total = len(trials)
result = {{
    "snapshot": "{SNAPSHOT}",
    "suite": "{suite}",
    "task": "{task}",
    "seeds": "1-50",
    "n_pass": n_pass,
    "n_total": n_total,
    "pass_rate": n_pass / n_total if n_total > 0 else 0.0,
}}
(outdir / "stage2_result.json").write_text(json.dumps(result, indent=2))
print(f"Stage2 done: {{n_pass}}/{{n_total}} = {{result['pass_rate']*100:.0f}}%")
```

Return: SUITE/TASK, n_pass/n_total, pass_rate.
""",
        run_in_background=True,
    )
    print(f"Stage2 dispatched GPU{gpu}: {suite}/{task[:50]}")
```

### Step 5 — Collect final results

After all Stage 2 agents return:

```python
import json
from pathlib import Path

SNAPSHOT = "snapshot-N25"
WORKTREE = f"$ASPIRE_ROOT/outputs/worktrees/{SNAPSHOT}"
SUITE_TASK_PAIRS = [
    ("libero_10_swap",  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"),
    ("libero_10_swap",  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
    ("libero_10_swap",  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"),
    ("libero_10_task",  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"),
    ("libero_10_task",  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
]

print(f"{'Suite':<18} {'Task':<60} {'S1 best':>8} {'S2 rate':>8}")
print("-" * 96)

for suite, task in SUITE_TASK_PAIRS:
    base = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}")
    s1_f = base / "stage1" / "stage1_summary.json"
    s2_f = base / "stage2" / "stage2_result.json"
    s1_str = f"{json.loads(s1_f.read_text())['best_pass_rate']*100:.0f}%" if s1_f.exists() else "—"
    s2_str = f"{json.loads(s2_f.read_text())['pass_rate']*100:.0f}%" if s2_f.exists() else "—"
    print(f"{suite:<18} {task[:59]:<60} {s1_str:>8} {s2_str:>8}")

# Write aggregate summary.json
results = {"libero_10_swap": {}, "libero_10_task": {}}
for suite, task in SUITE_TASK_PAIRS:
    s2_f = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{suite}/{task}/stage2/stage2_result.json")
    if s2_f.exists():
        d = json.loads(s2_f.read_text())
        results[suite][task] = {"success": d["n_pass"], "total": d["n_total"]}

out = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "snapshot": SNAPSHOT,
    "mode": "debug_eval",
    "seeds_debug": "51-65",
    "seeds_eval": "1-50",
    "suite_task_pairs": [[s, t] for s, t in SUITE_TASK_PAIRS],
    "results": results,
}, indent=2))
print(f"\nSummary written: {out}")
```
