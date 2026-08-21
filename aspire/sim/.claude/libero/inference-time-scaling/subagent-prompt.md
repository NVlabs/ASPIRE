---
name: libero-inference-time-debug-subagent-prompt
description: Self-contained prompt template for LIBERO-Long-Pro Stage 1 debug subagents. Given a frozen strategy library snapshot, debug on seeds 51–65 for up to 20 iterations targeting ≥90%, then write task_code.py + stage1_summary.json. Stage 2 (seeds 1–50) is run by the coordinator after completion. Copy, fill in SUITE/TASK/GPU/TASKSHORT/SNAPSHOT/ASPIRE_ROOT_SNAPSHOT, pass to Agent tool.
---

# LIBERO-Long-Pro Debug Subagent Prompt Template (Stage 1 Debug)

Copy the block below, fill in all `<PLACEHOLDER>` fields, pass as `prompt` to `Agent(subagent_type="general-purpose", model="opus", run_in_background=True)`.

> **Model:** Use `model="opus"` — iterative diagnosis and code revision require full reasoning.

> **Key properties:**
> - This is **Stage 1 debug only**. Seeds 51–65. Up to 20 iterations. **NEVER touch seeds 1–50.**
> - Library is **frozen** — read `.claude/libero/skills/` from snapshot worktree, **NEVER modify it**.
> - Your output is `stage1/task_code.py` + `stage1/stage1_summary.json`. Coordinator handles Stage 2.
> - Stop early when: pass rate ≥90% **OR** no improvement for 3 consecutive iters **OR** 20 iterations exhausted.

---

```
## Task Assignment

SNAPSHOT:           <SNAPSHOT>
SUITE:              <SUITE>
TASK:               <TASK>
GPU:                <GPU>
TASKSHORT:          <TASKSHORT>
SEED_START:         51
SEED_END:           65

Working directory: <ASPIRE_ROOT_SNAPSHOT>  (absolute path to worktree checked out at SNAPSHOT tag)

---

## ⚠️ STEP 0 — WORKTREE VERIFICATION AND ISOLATION (DO THIS BEFORE ANYTHING ELSE)

```bash
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
EXPECTED_SNAPSHOT="<SNAPSHOT>"

cd "$WORKTREE" || { echo "FATAL: Cannot cd to $WORKTREE — aborting"; exit 1; }
echo "PWD: $PWD"

# 1. Verify git tag — must match exactly
TAG=$(git describe --tags --exact-match 2>/dev/null || echo "NOTAG")
echo "Git tag: $TAG"
if [[ "$TAG" != "$EXPECTED_SNAPSHOT" ]]; then
  echo "FATAL: wrong worktree — expected tag $EXPECTED_SNAPSHOT, got $TAG — aborting"
  exit 1
fi
echo "✓ Worktree verified: $WORKTREE at $EXPECTED_SNAPSHOT"

# 2. Confirm skill library is unmodified
SKILL_DIRTY=$(git diff --name-only -- .claude/libero/skills/ 2>/dev/null)
if [[ -n "$SKILL_DIRTY" ]]; then
  echo "FATAL: skill library has uncommitted changes — aborting"
  echo "$SKILL_DIRTY"
  exit 1
fi
echo "✓ Skill library clean"

# 3. Set up output dir
STAGE1_DIR="$WORKTREE/outputs/scaling_eval/$EXPECTED_SNAPSHOT/debug_eval/<SUITE>/<TASK>/stage1"
CODE_VERSIONS_DIR="$STAGE1_DIR/code_versions"
mkdir -p "$STAGE1_DIR" "$CODE_VERSIONS_DIR"
echo "Stage1 dir: $STAGE1_DIR"
echo "Code versions dir: $CODE_VERSIONS_DIR"

# 4. Check if stage1 already complete — exit early if so
SUMMARY="$STAGE1_DIR/stage1_summary.json"
if [ -f "$SUMMARY" ]; then
  echo "Stage1 already done — exiting"
  cat "$SUMMARY"
  exit 0
fi

# 5. Print isolation summary — verify no forbidden paths are accessible
echo ""
echo "=== ISOLATION CHECK ==="
echo "WORKTREE (only allowed root): $WORKTREE"
echo "Skills source: $WORKTREE/.claude/libero/skills/  (READ ONLY)"
echo ""
echo "Forbidden paths that must NOT be read:"
for forbidden in \
    "$REPO_ROOT" \
    "$ASPIRE_ROOT/outputs/scaling_build" \
    "$WORKTREE/../../../outputs/scaling_build"; do
  [ -d "$forbidden" ] && echo "  EXISTS (do not read): $forbidden" || echo "  not present: $forbidden"
done
echo "✓ Isolation check complete — read ONLY from $WORKTREE/.claude/libero/skills/"
```

**If the tag check fails or skill library is dirty, STOP immediately.**

Use `$WORKTREE` as an absolute path in ALL subsequent commands.
Use the worktree repo root on `PYTHONPATH`, e.g. `PYTHONPATH="$(cd "$WORKTREE/../.." && pwd)"`, in every python call — no other path.
Use `$WORKTREE/.venv-libero/bin/python3` — never system python or any other venv.
Skills directory: `$WORKTREE/.claude/libero/skills/` — READ ONLY, never write.

---

## What You Are

Stage 1 debug subagent for ASPIRE/LIBERO-Long-Pro. Your job:
1. Verify worktree and isolation (Step 0 above — mandatory)
2. Read the frozen strategy library (read-only)
3. Understand the task
4. Write and iteratively debug a solution on seeds 51–65 (up to 5 iterations)
5. Save every code version to `stage1/code_versions/`
6. Promote the best code to `stage1/task_code.py` + write `stage1/stage1_summary.json`
7. **STOP.** The coordinator handles Stage 2 (seeds 1–50).

**NEVER touch seeds 1–50.** They are the held-out eval set.
**NEVER write to `.claude/libero/skills/`.** The skill library is frozen for this snapshot.
**NEVER read from outside `$WORKTREE`** — no other repos, no other worktrees, no other venvs.

---

## Context

**What is LIBERO-Long-Pro?**
LIBERO-Long-Pro (`libero_10_swap` / `libero_10_task`) is a long-horizon robotic manipulation benchmark.
Each task requires completing **2–3 sequential subtasks** in one episode.

**Perturbation types:**
- `libero_10_swap`: object positions randomized per seed — SAM3 handles naturally
- ⚠️ `libero_10_task`: **language goal is remapped — the BDDL filename is MISLEADING.** ALWAYS use `env.handle.task_language` for the actual instruction.

⚠️ **Suite name collision:** `libero_10_swap` and `libero_10_task` share **identical task names**. SUITE field is authoritative. Always include full SUITE in logs and output paths.

**Config for replay:**
```
env_configs/libero/franka_libero_libero10_traced.yaml
```

**Perception servers must be running** (404 = UP on 8114/8115/8116).

---

## ⛔ FORBIDDEN APIs

**Using these invalidates benchmark results — they don't transfer to real robots:**
```
sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
inner.parsed_problem, inner._eval_predicate, inner.obj_body_id,
env.handle.env (unwrapping), sim.model.*, sim.data.qpos, sim.forward(),
env._step_once(), reading .bddl/.xml/.urdf asset files for geometry
```

**Also forbidden: `settle_physics()` in any form.**
`settle_physics()` advances 720 extra physics steps via gripper toggling. In `_swap` suites, this moves objects away from their perturbed init_state positions — objects drift from where the perturbation placed them. Never call it.

## ⛔ FORBIDDEN SOURCES — BENCHMARK CONTAMINATION

**Your ONLY allowed source of task strategy is `.claude/libero/skills/` within `$WORKTREE`. All of the following are absolutely forbidden:**

```
# Other repos / ASPIRE versions
$REPO_ROOT/              — DO NOT READ (LIBERO-Pro Evolutionary Search pipeline, different experiment)
$ASPIRE_ROOT/ — DO NOT USE except via $WORKTREE

# Task codes and build outputs — contaminate the scaling-law measurement
outputs/scaling_build/ (repo root)           — DO NOT READ (LIBERO-90 task codes, seeds 51–80)
$WORKTREE/outputs/scaling_eval/<other tasks> — DO NOT READ (other tasks' eval results; you may write to your own task's path)
Any task_code.py or findings.md anywhere    — DO NOT READ
outputs/worktrees/snapshot-N*/              — DO NOT READ (other snapshot worktrees)

# Runtime contamination
/tmp/ code files from other tasks/agents    — DO NOT READ
Any PYTHONPATH other than $WORKTREE        — DO NOT USE
Any venv other than $WORKTREE/.venv        — DO NOT USE
Any .claude/libero/skills/ outside $WORKTREE     — DO NOT READ
```

**Self-check before writing code:** If the insight came from outside `$WORKTREE/.claude/libero/skills/`, discard it. Write only from the skill library patterns you read.

## ✅ ALLOWED APIs

```
get_observation()                          → RGB, depth, intrinsics, pose_mat, robot state
segment_sam3_text_prompt(rgb, text)        → SAM3 masks [{mask, box, score, label}]
segment_sam3_point_prompt(rgb, points)     → SAM3 masks by point
mask_to_world_points(mask, depth, K, T)   → (N,3) world-frame point cloud
plan_grasp(depth, intrinsics, mask)        → (grasp_poses, grasp_scores) in camera frame
select_top_down_grasp(poses, scores, E)   → best grasp 4×4 world-frame matrix
decompose_transform(T)                    → (position, quaternion_wxyz)
solve_ik(position, quaternion_wxyz)       → joint angles (7,) or None
move_to_joints(joints)                    → blocking motor control
open_gripper() / close_gripper()
goto_pose(pos, quat, z_approach=None)     → convenience motion
goto_home_joint_position()               → move to robot home config (safe reset)
get_oriented_bounding_box_from_3d_points(pts) → {center, extent, R}
point_prompt_molmo(image, text)           → pixel (x, y) for named object
env.handle.task_language                  → ACTUAL task instruction (use this, not task name for _task suites)
numpy, scipy
```

---

## Step 1 — Read the frozen skill library (MANDATORY, READ ONLY)

```bash
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
# READ ONLY — do not write, edit, or create any file under .claude/libero/skills/
cat "$WORKTREE/.claude/libero/skills/grasp.md"
cat "$WORKTREE/.claude/libero/skills/localize.md"
cat "$WORKTREE/.claude/libero/skills/transport.md"
cat "$WORKTREE/.claude/libero/skills/manipulation.md"
```

These skills were accumulated during the scaling-law build phase (tagged at `<SNAPSHOT>`). Use every pattern, helper function, prompt registry entry, and parameter you find. This is what we are measuring — whether the library helps you write better code.

**Do NOT:**
- Write or update any skill file
- Create new skill files
- Add entries to any SKILL.md
- Use skills from any other repo or path

---

## Step 2 — Understand the task

Get the actual task instruction and scene:

```bash
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
cd "$WORKTREE"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<GPU> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$(cd "$WORKTREE/../.." && pwd)" \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite <SUITE> --args.task "<TASK>" --args.trial 51 \
  --args.interactive \
  --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
  --args.output-dir /tmp/debug_repl_<TASKSHORT> 2>/dev/null << 'EOF'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
task_lang = env.handle.task_language
print(f"TASK_LANGUAGE: {task_lang}", flush=True)
obs = get_observation()
plt.imsave("/tmp/debug_scene_<TASKSHORT>.png", obs["agentview"]["images"]["rgb"])
print("Scene saved", flush=True)
# Also print obs keys so you know the exact API
print(f"obs keys: {list(obs.keys())}", flush=True)
for prompt in ["<describe objects from task name>", "basket", "drawer", "stove"]:
    masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], prompt)
    print(f"{prompt}: {len(masks)} masks" + (f", top={masks[0]['score']:.3f}" if masks else ""), flush=True)
EOF
```

View `/tmp/debug_scene_<TASKSHORT>.png`.

### ⛔ MANDATORY GATE — do not proceed until you have done this

Write down the following before writing any code:

```
ACTUAL TASK_LANGUAGE: <paste exact value from probe output>
SUITE: <SUITE>
BDDL task name: <TASK>
Match? <yes/no — for libero_10_task these will differ>
```

For `libero_10_task`: the BDDL task name is a misleading stub. Your code MUST target the objects named in `task_language`, not the BDDL name. If the task_language says "red mug", your SAM3 prompts must say "red mug". Do not code against the BDDL name.

Also note the exact obs keys printed above. The robot state key is `obs["robot_cartesian_pos"]` (shape (8,): x,y,z, qw,qx,qy,qz, gripper). There is no `obs["robot"]` key.

Now decompose `task_language` into subtasks and plan your approach.

---

## Step 3 — Debug loop (up to 20 iterations)

**Iteration structure:**
- Write/revise code → save to `code_versions/` → run on seeds 51–65 → count pass rate → check stop conditions → diagnose failures → revise

**Stop conditions (check after every iter, before starting the next):**
1. **Success:** pass rate ≥ 90% → promote immediately
2. **Plateau:** best pass rate has not improved over the last 3 consecutive completed iters → stop, promote best so far
3. **Hard limit:** 20 iterations exhausted → stop, promote best so far

The plateau check is mechanical — run it in 3c after writing result.json.

### Long-Horizon Code Rules (CRITICAL)

1. **Call `goto_home_joint_position()` BEFORE every subtask** — prevents joint limit accumulation between consecutive operations.
2. **Re-call `get_observation()` after each home reset** — arm movement shifts the camera view; stale observations cause wrong grasps.
3. **Best-effort per subtask** — don't assume subtask 1 succeeded; attempt subtask 2 regardless.
4. **For "AND close it" tasks (drawer, microwave):** ALWAYS do the close step even if prior steps failed. Closing is often the final success predicate.
5. **Write for seeds 51–65.** No hardcoded coordinates. SAM3 prompts must generalize across position variations.
6. **NEVER call `settle_physics()`** — breaks `_swap` suite init states.

### 3a — Write/revise code and save to code_versions/

```bash
ITER=1   # increment each iteration: 1, 2, 3, ..., up to 20
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
SUITE="<SUITE>"
TASK="<TASK>"
SNAPSHOT="<SNAPSHOT>"
TASKSHORT="<TASKSHORT>"
STAGE1_DIR="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$SUITE/$TASK/stage1"
CODE_VERSIONS_DIR="$STAGE1_DIR/code_versions"
mkdir -p "$STAGE1_DIR/iter_$ITER" "$CODE_VERSIONS_DIR"

# Hard limit guard — never exceed 20 iterations regardless of context
if [[ $ITER -gt 20 ]]; then
  echo "Hard limit: 20 iterations reached — skipping to Step 4 (promotion)"
  exit 0
fi

# Code lives in code_versions/ as the canonical version for each iter, timestamped
TS=$(date +%Y%m%d_%H%M%S)
CODE="$CODE_VERSIONS_DIR/iter_${ITER}_${TS}.py"
```

Write the task code and save to `$CODE`. Use patterns from the skill library you read in Step 1.

Key API reminders (from MANDATORY GATE above):
- Robot state: `obs["robot"]["eef_pose"]` (4×4 matrix), `obs["robot_cartesian_pos"]` (shape (8,): x,y,z,qw,qx,qy,qz,gripper)
- `select_top_down_grasp(poses, scores, cam_to_world)` — third arg is `obs["agentview"]["pose_mat"]`, NOT robot pose
- SAM3 prompts must match `task_language` objects exactly (especially for `libero_10_task`)
- `goto_home_joint_position()` before every subtask; `get_observation()` after every home reset
- Best-effort per subtask — attempt subtask 2 even if subtask 1 failed
- **Never call `settle_physics()`**

### 3b — Run on seeds 51–65

```bash
ITER=1   # current iteration
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
SUITE="<SUITE>"
TASK="<TASK>"
SNAPSHOT="<SNAPSHOT>"
GPU=<GPU>
TASKSHORT="<TASKSHORT>"
STAGE1_DIR="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$SUITE/$TASK/stage1"
CODE_VERSIONS_DIR="$STAGE1_DIR/code_versions"
CODE=$(ls "$CODE_VERSIONS_DIR"/iter_${ITER}_*.py 2>/dev/null | head -1)
if [[ -z "$CODE" ]]; then echo "FATAL: no code file found for iter $ITER in $CODE_VERSIONS_DIR" >&2; exit 1; fi
LOG="$STAGE1_DIR/iter_$ITER/run.log"
SEED_OUTDIR="$STAGE1_DIR/iter_$ITER/seed_outputs"
mkdir -p "$SEED_OUTDIR" "$STAGE1_DIR/iter_$ITER"

echo "Iter $ITER: $SUITE/$TASK seeds 51-65" | tee "$LOG"
echo "Code: $CODE" | tee -a "$LOG"

# ── Smoke test: run seeds 51–53 first ─────────────────────────────────────────
# If all 3 crash (Traceback in log, no taskcompleted), STOP and fix before spending
# the full 15-seed budget on broken code.
echo "=== SMOKE TEST: seeds 51-53 ===" | tee -a "$LOG"
smoke_crash=0
for seed in 51 52 53; do
    trial_padded=$(printf "%02d" $seed)
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH="$(cd "$WORKTREE/../.." && pwd)" \
    "$WORKTREE/.venv-libero/bin/python3" scripts/libero/replay_trial.py \
        --args.suite "$SUITE" --args.task "$TASK" --args.trial $seed \
        --args.replay-code "$CODE" \
        --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
        --args.output-dir "$SEED_OUTDIR" > "$STAGE1_DIR/iter_$ITER/seed${seed}.log" 2>&1 || true
    result=$(grep -oE "taskcompleted_[01]" "$STAGE1_DIR/iter_$ITER/seed${seed}.log" | tail -1 || echo "ERROR")
    has_crash=$(grep -c "Traceback" "$STAGE1_DIR/iter_$ITER/seed${seed}.log" 2>/dev/null || echo 0)
    echo "Smoke seed $seed: $result (crash lines: $has_crash)" | tee -a "$LOG"
    if [[ "$result" == "ERROR" && "$has_crash" -gt 0 ]]; then
        smoke_crash=$((smoke_crash + 1))
    fi
done
if [[ $smoke_crash -eq 3 ]]; then
    echo "SMOKE TEST FAILED: all 3 seeds crashed — diagnose and fix before running full loop" | tee -a "$LOG"
    echo "See logs: $STAGE1_DIR/iter_$ITER/seed51.log  seed52.log  seed53.log"
    exit 1
fi
echo "=== SMOKE TEST PASSED ($smoke_crash/3 crashed) — continuing full run ===" | tee -a "$LOG"
# ──────────────────────────────────────────────────────────────────────────────

# ── Parallel seed run (4 workers) ─────────────────────────────────────────────
# Seeds are independent; SAM3/GraspNet are HTTP servers so no model-load contention.
# Each worker appends one line to $LOG when done — writes are atomic for short lines.
_run_one_seed() {
    seed=$1
    WORKTREE=$2; SUITE=$3; TASK=$4; GPU=$5; CODE=$6; SEED_OUTDIR=$7; ITER_DIR=$8; LOG=$9
    trial_padded=$(printf "%02d" $seed)
    if find "$SEED_OUTDIR" -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q .; then
        echo "Seed $seed: skip (exists)" >> "$LOG"; return
    fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH="$(cd "$WORKTREE/../.." && pwd)" \
    "$WORKTREE/.venv-libero/bin/python3" scripts/libero/replay_trial.py \
        --args.suite "$SUITE" --args.task "$TASK" --args.trial $seed \
        --args.replay-code "$CODE" \
        --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
        --args.output-dir "$SEED_OUTDIR" > "$ITER_DIR/seed${seed}.log" 2>&1 || true
    result=$(grep -oE "taskcompleted_[01]" "$ITER_DIR/seed${seed}.log" | tail -1 || echo "ERROR")
    echo "Seed $seed: $result" >> "$LOG"
}
export -f _run_one_seed

ITER_DIR="$STAGE1_DIR/iter_$ITER"
seq 51 65 | xargs -P 4 -I{} bash -c \
    '_run_one_seed "$@"' _ {} \
    "$WORKTREE" "$SUITE" "$TASK" "$GPU" "$CODE" "$SEED_OUTDIR" "$ITER_DIR" "$LOG"

echo "ITER_DONE" | tee -a "$LOG"
# ──────────────────────────────────────────────────────────────────────────────

# Verify skill library was not modified
SKILL_DIRTY=$(cd "$WORKTREE" && git diff --name-only -- .claude/libero/skills/ 2>/dev/null)
if [[ -n "$SKILL_DIRTY" ]]; then
  echo "ERROR: skill library was modified during iter $ITER — this is a contamination violation"
  echo "$SKILL_DIRTY"
fi
```

### 3c — Count pass rate and write result.json

```python
import json
from pathlib import Path

WORKTREE = "<ASPIRE_ROOT_SNAPSHOT>"
SUITE = "<SUITE>"
TASK = "<TASK>"
SNAPSHOT = "<SNAPSHOT>"
ITER = 1  # current iteration
STAGE1_DIR = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{SUITE}/{TASK}/stage1")
CODE_VERSIONS_DIR = STAGE1_DIR / "code_versions"
seed_dir = STAGE1_DIR / f"iter_{ITER}" / "seed_outputs"

trials = list(seed_dir.rglob("trial_*"))  # rglob: replay_trial.py nests under suite/task/model/run/
n_pass = sum(1 for t in trials if "taskcompleted_1" in t.name)
n_total = len(trials)
pass_rate = n_pass / n_total if n_total > 0 else 0.0

result = {
    "iter": ITER,
    "suite": SUITE,
    "task": TASK,
    "seeds": "51-65",
    "n_pass": n_pass,
    "n_total": n_total,
    "pass_rate": pass_rate,
    "code": next(CODE_VERSIONS_DIR.glob(f"iter_{ITER}_*.py"), Path(f"iter_{ITER}_unknown.py")).name,
}
(STAGE1_DIR / f"iter_{ITER}" / "result.json").write_text(json.dumps(result, indent=2))
print(f"Iter {ITER}: {n_pass}/{n_total} = {pass_rate*100:.0f}%")

# ── Stop-condition check ──────────────────────────────────────────────────────
# Read all completed results so far
all_results = []
for iter_dir in sorted(STAGE1_DIR.glob("iter_*")):
    r_file = iter_dir / "result.json"
    if r_file.exists():
        all_results.append(json.loads(r_file.read_text()))
all_results.sort(key=lambda r: r["iter"])

rates = [r["pass_rate"] for r in all_results]
best_so_far = max(rates)

# 1. Success
if pass_rate >= 0.90:
    print(f"STOP: success — {pass_rate*100:.0f}% ≥ 90% → promote iter {ITER}")
# 2. Plateau: last 3 iters all at or below the best from before them
elif len(rates) >= 3 and max(rates[-3:]) <= max(rates[:-3] or [0]):
    print(f"STOP: plateau — no improvement in last 3 iters (rates: {[f'{r*100:.0f}%' for r in rates[-3:]]}) → promote best so far")
# 3. Hard limit
elif ITER >= 20:
    print(f"STOP: hard limit 20 iters reached → promote best so far")
else:
    print(f"CONTINUE: best={best_so_far*100:.0f}%, iter {ITER} done, proceed to iter {ITER+1}")
# ─────────────────────────────────────────────────────────────────────────────
```

**If any STOP condition printed above, skip to Step 4 (promotion). Otherwise diagnose and do iter+1.**

### 3d — Diagnose failures (before next iteration)

Look at failed seeds. Identify the top-2 failure modes.

```bash
ITER=1   # current iteration
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
SUITE="<SUITE>"
TASK="<TASK>"
SNAPSHOT="<SNAPSHOT>"
STAGE1_DIR="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$SUITE/$TASK/stage1"

for seed in $(seq 51 65); do
    log="$STAGE1_DIR/iter_${ITER}/seed${seed}.log"
    result=$(grep -oE "taskcompleted_[01]" "$log" 2>/dev/null | tail -1 || echo "MISSING")
    if [ "$result" != "taskcompleted_1" ]; then
        echo "=== FAILED seed $seed ==="
        grep -E "(Error|error|Traceback|localize|SAM3|grasp|place|None|mask)" "$log" | tail -20
        echo ""
    fi
done
```

Read keyframe images from a sample of failed trials:

```bash
OUTDIR="$STAGE1_DIR/iter_${ITER}/seed_outputs"
for trial_dir in $(find "$OUTDIR" -type d -name "*taskcompleted_0*" | head -3); do
    echo "=== Failed trial: $(basename $trial_dir) ==="
    ls "$trial_dir/"
    # Read any keyframe images present (e.g. obs_*.png, keyframe_*.png)
done
```

Based on failure analysis, identify root causes:
- **Localization failure**: SAM3 can't find object → improve prompts, add fallback prompts
- **Grasp failure**: plan_grasp returns None or bad grasp → add top-down fallback
- **Placement failure**: wrong target position → check surface_z estimation, adjust z_offset
- **Subtask sequencing**: subtask 1 state affects subtask 2 → add intermediate observation after subtask 1
- **Close step missing/wrong approach**: wrong direction for drawer/microwave

Write revised code using the Step 3a pattern with `ITER=N+1` — the filename will be `code_versions/iter_N+1_YYYYMMDD_HHMMSS.py`. Repeat 3b–3d until a stop condition triggers (success, plateau, or hard limit of 20).

---

## Step 4 — Promote best code

After all iterations (or early stop at ≥90%), find the best iteration and promote.
`task_code.py` is a copy of the best iter's code from `code_versions/` — the canonical version for Stage 2.

```python
import json, shutil
from pathlib import Path

WORKTREE = "<ASPIRE_ROOT_SNAPSHOT>"
SUITE = "<SUITE>"
TASK = "<TASK>"
SNAPSHOT = "<SNAPSHOT>"
STAGE1_DIR = Path(f"{WORKTREE}/outputs/scaling_eval/{SNAPSHOT}/debug_eval/{SUITE}/{TASK}/stage1")
CODE_VERSIONS_DIR = STAGE1_DIR / "code_versions"

# Read all result.json files
results = []
for iter_dir in sorted(STAGE1_DIR.glob("iter_*")):
    r_file = iter_dir / "result.json"
    if r_file.exists():
        r = json.loads(r_file.read_text())
        results.append(r)
        print(f"Iter {r['iter']}: {r['n_pass']}/{r['n_total']} = {r['pass_rate']*100:.0f}%")

if not results:
    print("ERROR: no results found")
    raise SystemExit(1)

# List all code versions for audit
print("\nCode versions saved:")
for f in sorted(CODE_VERSIONS_DIR.glob("iter_*.py")):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")

best = max(results, key=lambda r: (r["pass_rate"], -r["iter"]))
best_iter = best["iter"]
best_code = next(CODE_VERSIONS_DIR.glob(f"iter_{best_iter}_*.py"), None)
if best_code is None:
    print(f"ERROR: no code file found for iter {best_iter} in {CODE_VERSIONS_DIR}")
    raise SystemExit(1)
task_code = STAGE1_DIR / "task_code.py"
shutil.copy(best_code, task_code)

summary = {
    "snapshot": SNAPSHOT,
    "suite": SUITE,
    "task": TASK,
    "seeds_debug": "51-65",
    "iters_run": len(results),
    "best_iter": best_iter,
    "best_pass_rate": best["pass_rate"],
    "best_n_pass": best["n_pass"],
    "best_n_total": best["n_total"],
    "promoted": best["pass_rate"] >= 0.90,
    "code_versions_dir": str(CODE_VERSIONS_DIR),
    "all_iters": [{
        "iter": r["iter"],
        "pass_rate": r["pass_rate"],
        "n_pass": r["n_pass"],
        "n_total": r["n_total"],
        "code": r.get("code", next((f.name for f in CODE_VERSIONS_DIR.glob(f"iter_{r['iter']}_*.py")), f"iter_{r['iter']}_unknown.py")),
    } for r in results],
}
(STAGE1_DIR / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nPromoted iter {best_iter}: {best['n_pass']}/{best['n_total']} = {best['pass_rate']*100:.0f}%")
print(f"task_code.py: {task_code}")
print(f"stage1_summary.json: {STAGE1_DIR / 'stage1_summary.json'}")
print(f"promoted: {summary['promoted']}")
```

---

## Step 5 — Verify output and return

```bash
WORKTREE="<ASPIRE_ROOT_SNAPSHOT>"
SNAPSHOT="<SNAPSHOT>"
SUITE="<SUITE>"
TASK="<TASK>"
STAGE1_DIR="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$SUITE/$TASK/stage1"

echo "=== Stage 1 output ==="
ls "$STAGE1_DIR/"
echo ""
echo "Code versions:"
ls "$STAGE1_DIR/code_versions/"
echo ""
echo "stage1_summary.json:"
cat "$STAGE1_DIR/stage1_summary.json"
echo ""
echo "task_code.py lines: $(wc -l < $STAGE1_DIR/task_code.py)"

# Final contamination check
SKILL_DIRTY=$(cd "$WORKTREE" && git diff --name-only -- .claude/libero/skills/ 2>/dev/null)
if [[ -n "$SKILL_DIRTY" ]]; then
  echo "WARNING: skill library was modified — this is a protocol violation"
  echo "$SKILL_DIRTY"
else
  echo "✓ Skill library unmodified"
fi
```

**STOP HERE.** Do NOT run seeds 1–50. Do NOT trigger Stage 2. The coordinator handles Stage 2.

---

## What to Return

```
SNAPSHOT: <tag>
SUITE: <suite>
TASK: <task>
GPU: <N>

Stage 1 result:
  Iters run: N
  Code versions: code_versions/iter_1_YYYYMMDD_HHMMSS.py ... iter_N_YYYYMMDD_HHMMSS.py
  Best iter: M  (pass_rate: X/15)
  Promoted: yes/no  (threshold 90%)
  task_code.py: written
  stage1_summary.json: written

Top failure modes diagnosed: <1-2 lines>
Library patterns used: <which skill entries helped>
Skill library modified: no  (required)

Stage 2: NOT run (coordinator handles seeds 1-50)
```

Do NOT include code, full traces, or seed-by-seed breakdown.
```

---
