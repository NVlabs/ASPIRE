---
name: libero-evosearch-subagent-prompt
description: Self-contained prompt template for Evolutionary Search multi-task actor subagents. Runs Evolutionary Search iterations on seeds 51–65, then runs Stage 2 (seeds 1–50) before returning. Coordinator just redispatches the freed GPU.
---

# Evolutionary Search Multi-Task Actor Subagent Prompt Template

Copy the block below, fill in the variables, and pass it as `prompt` to a background subagent.

> **Model:** Use a high-capability model. Evolutionary Search's multi-candidate reasoning needs trace diagnosis across 8 candidates and cross-iteration hypothesis refinement.

---

```
## Task Assignment

SUITE:      <suite>
TASK:       <full_task_name>
GPU:        <3|4|5|6|7>
TASKSHORT:  <short_unique_name_for_logs>
BASELINE_RATE: <e.g. "16% (50 seeds)">   # from actor pipeline — what we're trying to beat
EVOSEARCH_DIR: outputs/claude_evosearch          # change to outputs/claude_evosearch_rerun for a clean rerun
EVALDIR:    outputs/aspire_evosearch_eval      # change to outputs/aspire_evosearch_eval_rerun for a clean rerun

Working directory: $REPO_ROOT

---

## ⛔ EVAL SET LOCKOUT — DURING ITERATIONS ONLY

**Seeds 1–50 are the held-out evaluation set. DO NOT run them during Evolutionary Search iterations.**

- All Evolutionary Search iteration evaluations use seeds **51–65 only** as the debug set.
- Never run the eval seeds 1–50 during iterations, it contaminates the results.
- **Stage 2 (seeds 1–50): YOU run this after iterations converge and the final code is chosen, before returning.**

Violation invalidates the benchmark.

---

## What You Are

You are a Evolutionary Search debugging subagent. Your job:
1. Run Evolutionary Search-style iterative debugging on seeds 51–65 until convergence or plateau
2. Save the best code as `$EVOSEARCH_DIR/$SUITE/$TASK/evosearch_best_code.py`
3. **Run Stage 2: eval the best code on seeds 1–50**
4. Return a structured findings report with both Stage 1 and Stage 2 results

You have full tool access (Bash, Read, Write, Edit, Glob, Grep).
**.venv-libero/bin/python3** — use the LIBERO experiment env for every replay/eval command.

---

## Context

**Baseline:** The actor pipeline achieved $BASELINE_RATE on this task. Your target: exceed that significantly, ideally ≥80% on seeds 51–65.

**Skill library** from LIBERO-Pro (goal/object/spatial). Read before writing any candidates.

**Evolutionary Search eval config:**
```
env_configs/libero/franka_libero_traced.yaml
```

**Servers:** 404 = UP on 8114 (SAM3), 8115 (GraspNet), 8116 (PyRoKi). Check before running:
```bash
for p in 8114 8115 8116; do echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"; done
```

---

## ⛔ FORBIDDEN APIs

```
sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
inner.parsed_problem, inner._eval_predicate, inner.obj_body_id,
env.handle.env (unwrapping), sim.model.*, sim.data.qpos, sim.forward(),
env._step_once(), reading .bddl/.xml/.urdf asset files
```

---

## Stage 1: Evolutionary Search Iterations on Seeds 51–65

### Step 0 — Check if evosearch_best_code.py already exists

```bash
ls $EVOSEARCH_DIR/$SUITE/$TASK/evosearch_best_code.py 2>/dev/null && echo EXISTS || echo MISSING
```

If EXISTS → skip to **What to Return**.

---

### Step 0b — BDDL remapping check (MANDATORY)

**The task name might not match the actual goal.** Always check and print `env.handle.task_language`. If it differs from `$TASK`, **task_language is ground truth** — base all strategy on it. Record the actual language in `task_analysis.md` before writing any candidates.

---

### Step 1 — Read skills and existing baseline

```bash
cat .claude/libero/skills/grasp.md
cat .claude/libero/skills/localize.md
cat .claude/libero/skills/transport.md
cat .claude/libero/skills/manipulation.md
```

**Also read the existing fix_code.py if it exists** — this is your baseline to beat and **must be seeded as candidate_A**:
```bash
cat $EVOSEARCH_DIR/$SUITE/$TASK/evosearch_best_code.py 2>/dev/null || \
cat outputs/libero_baseline_image_diff_gemini/$SUITE/$TASK/fix_code.py 2>/dev/null || \
echo "No baseline fix_code.py found"
```

**If a baseline code is found:** Record its path in `task_analysis.md`. Seed it as candidate_A verbatim — do not modify it.

Read companion skills for task-specific patterns:
```bash
cat .claude/libero/evosearch/skills/push-contact-tasks.md 2>/dev/null    # if push task
cat .claude/libero/evosearch/skills/wrist-rotation-blocking.md 2>/dev/null # if arm blocked
cat .claude/libero/evosearch/skills/motion-efficiency.md 2>/dev/null
```

---

### Step 2 — Create run directory and scene snapshot

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$EVOSEARCH_DIR/$SUITE/$TASK/$RUN_ID
mkdir -p $RUN_DIR
```

**Scene snapshot** (REQUIRED before writing any candidates):
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
SNAPSHOT_DIR=$RUN_DIR \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite $SUITE --args.task "$TASK" --args.trial 51 \
  --args.replay-code scripts/libero/scene_snapshot.py \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.output-dir /tmp/evosearch_snap_$TASKSHORT \
  2>&1 | grep -E "(snapshot|SAM3|\[OK\]|\[WEAK\]|\[FAIL\]|Saved)"
```

Read both images with the Read tool:
- `$RUN_DIR/scene_snapshot.jpg` — wide scene view
- `$RUN_DIR/scene_snapshot_wrist.jpg` — close-up

Then populate `$RUN_DIR/task_analysis.md` (§1–5 from the images):
```bash
cat > $RUN_DIR/task_analysis.md << 'EOF'
# Task Analysis: <task_name>

## 1. Object Shape
...

## 2. Grasp/Approach Strategy
...

## 3. Goal Geometry
...

## 4. Placement/Movement Strategy
...

## 5. Hypotheses for iter_00
- Baseline code achieves $BASELINE_RATE — what is it doing wrong?
- Blocked approach directions visible in snapshot:
- Uncertain geometry assumptions:

## 6. Iteration Log
EOF
```

---

### Step 3 — Write K=8 candidates for iter_00

**Always seed candidate_A from the existing fix_code.py baseline** (if it exists) — this gives a concrete performance floor and shows where the baseline fails.

Each candidate must test a distinct hypothesis. No two candidates should fail at the same stage for the same reason.

```bash
mkdir -p $RUN_DIR/iter_00
# Write candidate_A.py through candidate_H.py
# Each has a docstring: Hypothesis / Differs from prior / Expected failure if wrong
```

See `.claude/libero/evosearch/skills/evosearch-iteration.md` §8 for the K=8 diversity rule and template.

---

### Step 4 — Iteration eval (seeds 51–65)

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv-libero/bin/python3 scripts/libero/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_00 \
    --suite $SUITE --task "$TASK" \
    --trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 \
    --sim-gpus $GPU --parallel-per-gpu 2 \
    --no-highlights \
    2>&1 | tee $RUN_DIR/iter_00/eval.log
```

**Set `run_in_background=True`** on this Bash call. Results (15 trials per candidate) are saved under `$RUN_DIR/iter_00/candidate_X/eval/` and aggregated in `iter_summary.json`.

Read leaderboard after completion:
```bash
.venv/bin/python3 -c "
import json; from pathlib import Path
s = json.loads(Path('$RUN_DIR/iter_00/iter_summary.json').read_text())
for c in sorted(s['candidates'], key=lambda x: -x['pass_rate']):
    print(f\"{c['candidate']:<22} {c['pass_rate']:.0%}  errors={c['errors']}  passes={c['pass_count']}/{c['trials']}\")
print(f'Best: {s[\"best_candidate\"]} ({s[\"best_pass_rate\"]:.0%})')
"
```

Analyze traces:
```bash
.venv/bin/python3 scripts/libero/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_00 --summary-only
```

---

### Step 6 — Iterate

**Stopping criteria — stop Stage 1 when ANY of these is true:**
- Best candidate ≥ **80%** on seeds 51–65 → **SOLVED**
- **5 iterations** completed

**When progress stalls:** Do NOT stop early just because recent iterations showed small gains. Use the remaining iteration budget to explore structurally new approaches — different grasp strategies, different localization methods, different motion primitives. A plateau on the current approach family means the current family is wrong, not that the task is unsolvable.

**Generalization rule — think eval seeds, not debug seeds:**

The 15 debug seeds are a small, potentially unrepresentative sample. The code will be evaluated on 50 unseen seeds — write for those, not for the 15. Prefer strategies that work for mechanistic reasons (correct object identity, robust localization, physically sound grasp geometry) over strategies that score well by exploiting patterns specific to seeds 51–65. Avoid hard-coded thresholds, image-region masks, or XY offsets that were derived by fitting to observed debug-seed failures — these rarely transfer.

**Per iteration:**
1. Update `task_analysis.md` §1–5 with any new geometry from keyframes
2. Append `### iter_NN` to §6 (leaderboard table + eliminated hypotheses + open questions)
3. Write K=8 new candidates seeded from top-3 survivors of the current iteration
4. Eval all 8 candidates on seeds 51–65 (same command as Step 4, adjusted iter dir)
5. Read `analyze_evosearch_traces.py` output — watch for ARM BLOCKING, gripper width signals

```bash
# Iter N+1 setup
mkdir -p $RUN_DIR/iter_NN
# Write candidates, then eval as in Step 4
```

Use the same `--trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65` across all iterations (cross-iteration comparison is fair since seeds are fixed).

---

### Step 7 — Save best code, run Stage 2, write findings

**7a — Save best code:**
```bash
BEST_CANDIDATE="candidate_X"   # from final iter leaderboard
BEST_CODE="$RUN_DIR/iter_NN/candidate_X/code.py"

# Sanity check: Evolutionary Search best must beat fix_code.py (candidate_A = fix_code verbatim) on seeds 51–65
# If not, fall back to fix_code.py so Stage 2 uses the stronger baseline.
.venv/bin/python3 << 'PYEOF'
import json, sys
from pathlib import Path

rdir = Path("$RUN_DIR")
baseline, best_rate = 0.0, 0.0
for f in sorted(rdir.rglob("iter_summary.json")):
    if "stage2" in str(f):
        continue
    for c in json.loads(f.read_text()).get("candidates", []):
        best_rate = max(best_rate, c["pass_rate"])
        if c["candidate"] == "candidate_A":
            baseline = max(baseline, c["pass_rate"])

print(f"Best Evolutionary Search:          {best_rate:.0%}  (seeds 51–65)")
print(f"Baseline fix_code.py: {baseline:.0%}  (candidate_A verbatim)")
sys.exit(0 if best_rate > baseline else 1)
PYEOF

if [ $? -ne 0 ]; then
    echo "⚠ Evolutionary Search did not beat fix_code.py — using fix_code.py as evosearch_best_code.py"
    BEST_CODE="outputs/libero_baseline_image_diff_gemini/$SUITE/$TASK/fix_code.py"
    if [ ! -f "$BEST_CODE" ]; then
        echo "ERROR: fix_code.py not found at $BEST_CODE — marking BLOCKED"
        touch $EVOSEARCH_DIR/$SUITE/$TASK/BLOCKED
        exit 0
    fi
fi

cp "$BEST_CODE" $EVOSEARCH_DIR/$SUITE/$TASK/evosearch_best_code.py
mkdir -p outputs/working_codes
cp "$BEST_CODE" "outputs/working_codes/${SUITE}_${TASK}_evosearch.py"
```

If all approaches failed → mark BLOCKED and stop:
```bash
touch $EVOSEARCH_DIR/$SUITE/$TASK/BLOCKED
```
(skip Stage 2 if BLOCKED)

---

**7b — Stage 2: eval best code on seeds 1–50**

Launch Stage 2 as a detached `nohup` process, then poll completion with small periodic Bash calls. This avoids both timeout risk (nohup survives session interruptions) and context bloat (each poll is ~5 lines).

Step 1 — Write and launch the script:
```bash
cat > /tmp/evosearch_stage2_${TASKSHORT}.sh << 'SCRIPT'
#!/bin/bash
cd $REPO_ROOT
SUITE=FILL_SUITE
TASK=FILL_TASK
GPU=FILL_GPU
BEST_CODE="FILL_EVOSEARCH_DIR/$SUITE/$TASK/evosearch_best_code.py"
OUTDIR="FILL_EVALDIR"
LOG="/tmp/evosearch_s2_FILL_TASKSHORT_progress.log"

# Sanity checks — fail loudly rather than writing to wrong location
if [[ ! -f "$BEST_CODE" ]]; then
    echo "ERROR: BEST_CODE not found: $BEST_CODE" | tee -a "$LOG"; exit 1
fi
if [[ -z "$OUTDIR" ]] || [[ "$OUTDIR" == *"FILL_"* ]]; then
    echo "ERROR: OUTDIR not resolved: '$OUTDIR'" | tee -a "$LOG"; exit 1
fi
echo "Stage 2 start: OUTDIR=$OUTDIR  BEST_CODE=$BEST_CODE" | tee -a "$LOG"

for trial in $(seq 1 50); do
    trial_padded=$(printf "%02d" $trial)
    if ls "${OUTDIR}/${SUITE}/${TASK}"/*/run/trial_${trial_padded}_* 2>/dev/null | grep -q .; then
        echo "Trial ${trial}: skip" | tee -a "$LOG"; continue
    fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite "$SUITE" --args.task "$TASK" --args.trial $trial \
        --args.replay-code "$BEST_CODE" \
        --args.config env_configs/libero/franka_libero_traced.yaml \
        --args.output-dir "$OUTDIR" > "/tmp/evosearch_s2_FILL_TASKSHORT_${trial}.log" 2>&1 || true
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "/tmp/evosearch_s2_FILL_TASKSHORT_${trial}.log" | tail -1 | sed 's/reward_//')
    echo "Trial $trial: ${reward:-ERROR}" | tee -a "$LOG"
done
echo "STAGE2_DONE" | tee -a "$LOG"
SCRIPT
# Bake all values into the script — use | as sed delimiter for paths to avoid / conflicts
sed -i \
    "s/FILL_SUITE/$SUITE/g; s/FILL_GPU/$GPU/g; s/FILL_TASKSHORT/$TASKSHORT/g; s|FILL_TASK|$TASK|g; \
     s|FILL_EVOSEARCH_DIR|$EVOSEARCH_DIR|g; s|FILL_EVALDIR|$EVALDIR|g" \
    /tmp/evosearch_stage2_${TASKSHORT}.sh
# Verify the script looks correct BEFORE launching
echo "=== Stage 2 script sanity check ==="
grep -E "^(BEST_CODE|OUTDIR)=" /tmp/evosearch_stage2_${TASKSHORT}.sh
grep "FILL_" /tmp/evosearch_stage2_${TASKSHORT}.sh && echo "WARNING: unresolved FILL_ placeholders!" || echo "OK: no unresolved placeholders"
chmod +x /tmp/evosearch_stage2_${TASKSHORT}.sh
nohup bash /tmp/evosearch_stage2_${TASKSHORT}.sh > /tmp/evosearch_s2_${TASKSHORT}_nohup.log 2>&1 &
echo $! > /tmp/evosearch_s2_${TASKSHORT}.pid
echo "Stage 2 launched, PID=$(cat /tmp/evosearch_s2_${TASKSHORT}.pid)"
```

Step 2 — Poll every ~5 min with a short Bash call until `STAGE2_DONE` appears:
```bash
LOG="/tmp/evosearch_s2_${TASKSHORT}_progress.log"
done_flag=$(grep -c "STAGE2_DONE" "$LOG" 2>/dev/null || echo 0)
count=$(grep -c "Trial" "$LOG" 2>/dev/null || echo 0)
echo "Stage 2: $count/50 trials logged, done=$done_flag"
tail -3 "$LOG" 2>/dev/null
```

Repeat this poll call until `done=1`. Do not run other Bash commands between polls — just wait and re-poll.

Step 3 — Read final results:
```bash
EVAL_DIR="$EVALDIR/$SUITE/$TASK"
s2_trials=$(find "$EVAL_DIR" -maxdepth 5 -type d -name "trial_*" 2>/dev/null | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
s2_success=$(find "$EVAL_DIR" -maxdepth 5 -type d -name "*taskcompleted_1*" 2>/dev/null | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
echo "Stage 2: $s2_success/$s2_trials"
```

---

**7c — Write findings.md:**
```bash
cat > $EVOSEARCH_DIR/$SUITE/$TASK/findings.md << 'EOF'
## Task: $SUITE / $TASK
## Baseline rate (actor pipeline): $BASELINE_RATE
## Best Evolutionary Search rate (seeds 51–65): <N/15> (<pct>%)
## Stage 2 (seeds 1–50): <N/50> (<pct>%)

### Evolutionary Search Run
- Run dir: $RUN_DIR
- Iterations completed: <N>
- Stopping reason: <solved|max_iterations>

### What Fixed It vs Baseline
- <key strategy change>

### SAM3 Prompts That Worked
| Object | Prompts | Notes |
|---|---|---|

### Failure Modes Eliminated
- <approach that failed + why>

### Generalizable Patterns
- <anything worth adding to skill library>

### Skill Library Updates Made
- <if any skills were updated during this run>
EOF
```

---

## What to Return

```
SUITE: <suite>
TASK: <task>
GPU: <N>
Baseline rate: <from actor pipeline>

Stage 1 (seeds 51–65):
  Best candidate: <candidate_X> at iter_NN
  Best pass rate: <N>/15 (<pct>%)
  Stopping reason: <solved|max_iterations|BLOCKED>
  Iterations run: <N>
  Run dir: $EVOSEARCH_DIR/$SUITE/$TASK/<run_id>/

Stage 2 (seeds 1–50): <N>/50 (<pct>%)

Key findings (3 bullets):
  - <strategy that worked vs baseline failure mode>
  - <SAM3 prompts / grasp parameters>
  - <any generalizable pattern for skill library>
```
```
