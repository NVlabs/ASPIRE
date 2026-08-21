---
name: libero-long-pro-subagent-prompt
description: Self-contained prompt template for LIBERO Long Pro actor subagents. Claude Code writes task code from scratch using skill library. Copy, fill in SUITE/TASK/GPU, and pass to Agent tool.
---

# LIBERO Long Pro Actor Subagent Prompt Template

Copy the block below, fill in SUITE/TASK/GPU/TASKSHORT, and pass as `prompt` to `Agent(subagent_type="general-purpose", model="opus", run_in_background=True)`.

> **Model:** Use `model="opus"` — resolves to `claude-opus-4-6`. Long-horizon task decomposition, multi-object localization, and multi-seed generalization all benefit significantly from Opus 4.6.

---

```
## Task Assignment

SUITE:      <libero_10_swap|libero_10_task>
TASK:       <full_task_name_with_SCENE_prefix>
GPU:        <3|4|5|6|7>
TASKSHORT:  <short_unique_name_for_logs>

Working directory: $ASPIRE_ROOT

---

## ⛔ EVAL SET LOCKOUT — READ THIS FIRST

**Seeds 1–50 are the evaluation set. They are LOCKED during Stage 1.**

- During Stage 1 (write + debug phase), you may ONLY run seeds 51–65.
- Do NOT run seeds 1–50 for any reason during Stage 1 — not for "testing", not for "spot checking", not into /tmp.
- Running seeds 1–50 with any version of code before task_code.py is finalized invalidates the benchmark.
- Stage 2 (seeds 1–50) runs exactly once, after task_code.py is finalized and frozen, into the official eval dir.

Violation of this rule produces invalid benchmark results.

---

## What You Are

Actor subagent for ASPIRE/LIBERO Long Pro. Write robot control code from scratch for one long-horizon task, debug on seeds 51–65, run Stage 2 eval on seeds 1–50, return a findings report.

All commands from working directory `$ASPIRE_ROOT`.
Set `PYTHONPATH=$PYTHON_ROOT` in every python call.
**.venv-libero/bin/python3** — use the LIBERO experiment env for every replay/eval command.

---

## Context

**What is LIBERO Long Pro?**
LIBERO Long Pro (libero_10_swap / libero_10_task) is a long-horizon robotic manipulation benchmark.
Each task requires completing **2–3 sequential subtasks** in one episode (e.g., "pick A AND pick B AND place both in basket", "open drawer, place object, close drawer").

**Your code must handle the entire task in a single Python script** — no multi-turn; one monolithic code.py that executes all subtasks in sequence.

**Perturbation types:**
- `libero_10_swap`: object positions randomized per seed — SAM3 handles naturally
- `libero_10_task`: language goal remapped (the bddl filename is misleading) — ALWAYS use `env.handle.task_language` for the actual instruction

**Config to use for replay:**
```
env_configs/libero/franka_libero_libero10_traced.yaml
```

**Perception servers must be running** before any replay:
```bash
for p in 8114 8115 8116; do echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"; done
```
(404 = UP, 000 = DOWN)

---

## ⛔ FORBIDDEN APIs

**Using these invalidates benchmark results — they don't transfer to real robots:**
```
sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
inner.parsed_problem, inner._eval_predicate, inner.obj_body_id,
env.handle.env (unwrapping), sim.model.*, sim.data.qpos, sim.forward(),
env._step_once(), reading .bddl/.xml/.urdf asset files for geometry
```

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
goto_home_joint_position()               → move to robot home config (safe reset between subtasks)
get_oriented_bounding_box_from_3d_points(pts) → {center, extent, R}
point_prompt_molmo(image, text)           → pixel (x, y) for named object
env.handle.task_language                  → ACTUAL task instruction (use this, not bddl name)
numpy, scipy
```

---

## Stage 1: Write and Debug on Seeds 51–65

### Step 0 — Check if task_code.py already exists

```bash
cd $ASPIRE_ROOT
ls outputs/aspire_libero_long_actor/$SUITE/$TASK/task_code.py 2>/dev/null && echo EXISTS || echo MISSING
```

If it **EXISTS** → skip Stage 1 entirely. Jump to Stage 2 guard.

---

### Step 1 — Read the skill library (MANDATORY before writing any code)

```bash
cd $ASPIRE_ROOT
cat .claude/libero/skills/grasp.md
cat .claude/libero/skills/localize.md
cat .claude/libero/skills/transport.md
cat .claude/libero/skills/manipulation.md
```

Also look at working code examples for similar subtasks:
```bash
if [ -d outputs/working_codes ]; then
    ls outputs/working_codes/ | grep -E "(pick|place|bowl|stove|drawer|basket|moka|mug|microwave)" || true
fi
# Read 1–2 relevant examples for the subtask types in your task
```

---

### Step 2 — Understand the task (MANDATORY before writing any code)

Inspect the scene on seed 51:

```bash
cd $ASPIRE_ROOT
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$PYTHON_ROOT \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite $SUITE --args.task "$TASK" --args.trial 51 \
  --args.interactive \
  --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
  --args.output-dir /tmp/repl_out 2>/dev/null << 'EOF'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
print(env.handle.task_language, flush=True)
obs = get_observation()
plt.imsave("/tmp/scene_$TASKSHORT.png", obs["agentview"]["images"]["rgb"])
print("Scene saved", flush=True)
# Probe SAM3 for objects relevant to your task — fill in from task name
for prompt in ["<object1>", "<object2>", "<target>", "<container>"]:
    masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], prompt)
    print(f"{prompt}: {len(masks)} masks" + (f", top={masks[0]['score']:.3f}" if masks else ""), flush=True)
EOF
```

Then use Read tool to view `/tmp/scene_$TASKSHORT.png` to understand object layout.

Rate each SAM3 probe: `[OK]` (≥1 mask, score > 0.7), `[WEAK]` (low score), `[FAIL]` (0 masks).

**Decompose into subtasks** — most tasks follow one of these patterns:
- **Double pick-and-place:** pick A → place, pick B → place (e.g., "put both X and Y in basket")
- **Manipulation + place:** turn/open → pick → place (e.g., "turn on stove, put pot on it")
- **Pick + place + close:** open container → place object inside → close (e.g., "put bowl in drawer, close it")
- **Spatial placement:** place A at location, place B at different location (e.g., "mug on left plate, mug on right plate")

---

### Step 3 — Write initial task_code.py

Structure your code around the subtask decomposition. Use this template:

```python
import numpy as np

# ── Utilities ────────────────────────────────────────────────────────────────

def make_topdown_quat():
    """Top-down grasp orientation (wxyz)."""
    return np.array([0.0, 1.0, 0.0, 0.0])

def localize_object(rgb, depth, K, T, prompts, min_masks=1):
    """Try prompts in order, return (mask, pts_3d) for first hit."""
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if len(masks) >= min_masks:
            mask = masks[0]["mask"]
            pts  = mask_to_world_points(mask, depth, K, T)
            if len(pts) > 0:
                return mask, pts
    return None, None

def pick_object(rgb, depth, K, T, E, prompts):
    """Locate + grasp an object. Returns True on success."""
    mask, pts = localize_object(rgb, depth, K, T, prompts)
    if mask is None:
        print("Localization failed", flush=True)
        return False
    grasp_poses, grasp_scores = plan_grasp(depth, K, mask)
    if grasp_poses is None or len(grasp_poses) == 0:
        print("No grasp found", flush=True)
        return False
    T_grasp = select_top_down_grasp(grasp_poses, grasp_scores, E)
    pos, quat = decompose_transform(T_grasp)
    open_gripper()
    goto_pose(pos, quat, z_approach=pos[2] + 0.12)
    close_gripper()
    return True

# ── Main task ─────────────────────────────────────────────────────────────────

obs  = get_observation()
rgb  = obs["agentview"]["images"]["rgb"]
depth = obs["agentview"]["images"]["depth"]
K    = obs["agentview"]["intrinsics"]
T    = obs["agentview"]["pose_mat"]
E    = obs["robot"]["eef_pose"]

# --- Subtask 1 ---
goto_home_joint_position()  # safe reset before each subtask
obs = get_observation()     # fresh obs after homing
rgb, depth = obs["agentview"]["images"]["rgb"], obs["agentview"]["images"]["depth"]

picked = pick_object(rgb, depth, K, T, E, prompts=["alphabet soup", "soup can"])
if picked:
    # transport to target
    target_masks = segment_sam3_text_prompt(rgb, "basket")
    if target_masks:
        basket_pts = mask_to_world_points(target_masks[0]["mask"], depth, K, T)
        center = basket_pts.mean(axis=0)
        center[2] += 0.15   # drop height above basket
        goto_pose(center, make_topdown_quat())
        open_gripper()

# --- Subtask 2 ---
goto_home_joint_position()
obs = get_observation()
rgb, depth = obs["agentview"]["images"]["rgb"], obs["agentview"]["images"]["depth"]
# ... repeat pattern for second object ...
```

**Long-horizon code rules:**
- Call `goto_home_joint_position()` **before every subtask** — it prevents joint limit issues between consecutive operations
- Re-call `get_observation()` after each home reset — the scene view may have shifted
- Never assume the first subtask succeeded before starting the second — write code that does reasonable best-effort for each subtask independently where possible
- For tasks with "AND close it" (drawer, microwave): always do the close step even if the place step failed partially
- **Write for eval seeds, not debug seeds.** Use robust SAM3 prompts that work across object positions (seeds 1–50), not hardcoded coordinates from one observation.

---

### Step 4 — Test and iterate

```bash
cd $ASPIRE_ROOT
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$PYTHON_ROOT \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite $SUITE --args.task "$TASK" --args.trial 51 \
  --args.replay-code /tmp/task_$TASKSHORT.py \
  --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
  --args.output-dir /tmp/long_debug_$TASKSHORT
```

Check result: `_reward_1.000_taskcompleted_1` in dir name = full success.

**Note: LIBERO Long Pro uses partial credit** — the task may complete 1/2 subtasks and get `taskcompleted_0` but non-zero reward. Read `summary.txt` for details.

Read trace.json to diagnose failures:
```python
import json
trace = json.load(open("/tmp/long_debug_.../trial_51_.../trace.json"))
for step in trace:
    fn = step.get("function", "")
    if fn in ("close_gripper", "segment_sam3_text_prompt", "solve_ik"):
        print(fn, step.get("result", {})[:200] if isinstance(step.get("result"), str) else step.get("result"), flush=True)
```

**Hard limits (ABSOLUTE — no exceptions):**
- **Per seed**: max 3 replay attempts on any one seed number.
- **Total code versions**: if you've written more than 15 `/tmp/task_$TASKSHORT_v*.py` files with zero successes, **BLOCK the task immediately.**
- Check version count before each new version: `ls /tmp/task_${TASKSHORT}_v*.py 2>/dev/null | wc -l`
- Test on seeds 51, 52, 53. Need 2/3 passing to finalize.

Read trace to diagnose failures:
```bash
cat /tmp/long_debug_$TASKSHORT/*/trace.json | python3 -c "
import json, sys
for step in json.load(sys.stdin):
    fn = step.get('function','')
    if fn in ('close_gripper','segment_sam3_text_prompt','plan_grasp','solve_ik','goto_pose'):
        print(fn, str(step.get('result',''))[:120])
" 2>/dev/null
```

---

### Step 5 — Finalize and save task_code.py

After achieving reliable success on 2+ debug seeds:

```bash
cd $ASPIRE_ROOT
mkdir -p outputs/aspire_libero_long_actor/$SUITE/$TASK
cp /tmp/task_$TASKSHORT.py outputs/aspire_libero_long_actor/$SUITE/$TASK/task_code.py
mkdir -p outputs/working_codes
cp /tmp/task_$TASKSHORT.py "outputs/working_codes/${SUITE}_${TASK}.py"
```

**Also write findings.md:**
```bash
cat > outputs/aspire_libero_long_actor/$SUITE/$TASK/findings.md << 'EOF'
## Task: $SUITE / $TASK
## Actual task language (for _task suite): "<from env.handle.task_language>"

### Subtask Decomposition
- Subtask 1: <description>
- Subtask 2: <description>
- Subtask 3 (if any): <description>

### Root Cause(s) of Failures
- <failure mode and why>

### What Fixed It
- <approach that worked>

### SAM3 Prompts That Worked
| Object | Prompts (priority order) | Notes |
|---|---|---|
| <object> | "<prompt1>", "<prompt2>" | <caveats> |

### Long-Horizon Specific Findings
- Home position between subtasks: yes/no, why
- Subtask ordering: did order matter?
- Partial failure handling: what happened when subtask 1 failed

### Generalizable Patterns
- <anything useful for other long-horizon tasks>

### Stage 2 Success Rate
N/50
EOF
```

If ALL test seeds blocked → write BLOCKED sentinel and skip Stage 2:
```bash
touch outputs/aspire_libero_long_actor/$SUITE/$TASK/BLOCKED
```

---

## Stage 2: Eval on Seeds 1–50

⚠️ **CRITICAL: Stage 2 is ONE-SHOT. Run it exactly once. Do NOT debug Stage 2 results.**

### Guard check first
```bash
cd $ASPIRE_ROOT
existing=$(find outputs/aspire_libero_long_eval -maxdepth 6 -type d -name "trial_*" 2>/dev/null | grep "$SUITE/$TASK" | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
echo "Seeds already on disk: $existing"
```
If existing >= 45: **Stage 2 already done. Return immediately.**

### Write, sanity-check, launch, and poll

Step 1 — Write the script:
```bash
cd $ASPIRE_ROOT
cat > /tmp/stage2_long_$TASKSHORT.sh << 'SCRIPT'
#!/bin/bash
# NO set -e — individual trial failures must not kill the loop
cd $ASPIRE_ROOT
SUITE="FILL_IN_SUITE"
TASK="FILL_IN_TASK"
GPU=FILL_IN_GPU
TASKSHORT="FILL_IN_TASKSHORT"
FIX_CODE="outputs/aspire_libero_long_actor/${SUITE}/${TASK}/task_code.py"
LOG="/tmp/val_long_${TASKSHORT}_progress.log"

# Sanity checks
if [[ ! -f "$FIX_CODE" ]]; then
    echo "ERROR: FIX_CODE not found: $FIX_CODE" | tee -a "$LOG"; exit 1
fi
echo "Stage 2 start: FIX_CODE=$FIX_CODE  GPU=$GPU" | tee -a "$LOG"

for trial in $(seq 1 50); do
    trial_padded=$(printf "%02d" $trial)
    # Skip already-completed trials (idempotent)
    if find outputs/aspire_libero_long_eval -maxdepth 6 -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q "${SUITE}/${TASK}"; then
        echo "Trial ${trial}: skip" | tee -a "$LOG"; continue
    fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH=$PYTHON_ROOT \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite "${SUITE}" --args.task "${TASK}" --args.trial ${trial} \
        --args.replay-code "${FIX_CODE}" \
        --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
        --args.output-dir outputs/aspire_libero_long_eval > "/tmp/val_long_${TASKSHORT}_${trial}.log" 2>&1 || true
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "/tmp/val_long_${TASKSHORT}_${trial}.log" | tail -1 | sed 's/reward_//')
    echo "Trial $trial: ${reward:-ERROR}" | tee -a "$LOG"
done
echo "STAGE2_DONE" | tee -a "$LOG"
SCRIPT
sed -i "s/FILL_IN_SUITE/$SUITE/g; s/FILL_IN_TASK/$TASK/g; s/FILL_IN_GPU/$GPU/g; s/FILL_IN_TASKSHORT/$TASKSHORT/g" /tmp/stage2_long_$TASKSHORT.sh
# Verify no unresolved placeholders before launching
grep "FILL_IN_" /tmp/stage2_long_$TASKSHORT.sh && echo "WARNING: unresolved placeholders!" || echo "OK: placeholders resolved"
chmod +x /tmp/stage2_long_$TASKSHORT.sh
```

Step 2 — Launch (`nohup &` survives agent timeout/crash):
```bash
nohup bash /tmp/stage2_long_$TASKSHORT.sh > /tmp/stage2_long_${TASKSHORT}.out 2>&1 &
echo $! > /tmp/stage2_long_${TASKSHORT}.pid
echo "Stage 2 launched PID=$(cat /tmp/stage2_long_${TASKSHORT}.pid)"
```

Step 3 — Poll until done (small Bash calls every ~5 min):
```bash
LOG="/tmp/val_long_${TASKSHORT}_progress.log"
done_flag=$(grep -c "STAGE2_DONE" "$LOG" 2>/dev/null || echo 0)
count=$(grep -c "Trial" "$LOG" 2>/dev/null || echo 0)
echo "Stage 2: $count/50 trials logged, done=$done_flag"
tail -3 "$LOG" 2>/dev/null
```
Repeat until `done=1`.

Step 4 — Read final results:
```bash
cd $ASPIRE_ROOT
actual=$(find outputs/aspire_libero_long_eval -maxdepth 6 -type d -name "trial_*" 2>/dev/null | grep "$SUITE/$TASK" | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
successes=$(find outputs/aspire_libero_long_eval -maxdepth 6 -type d -name "*taskcompleted_1*" 2>/dev/null | grep "$SUITE/$TASK" | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
echo "Stage 2: $successes/$actual"
```

---

## Final Step

Always run before returning:
```bash
cd $ASPIRE_ROOT
PYTHONPATH=$PYTHON_ROOT .venv/bin/python3 scripts/libero/gen_progress_long.py
```

---

## What to Return

```
SUITE: <suite>
TASK: <task>
GPU: <N>
Actual task language: "<from env.handle.task_language>"

Stage 1: task_code.py written: yes/no
  Test seeds passed: <list, e.g. 51✓ 52✗ 53✓>
  Seeds blocked: <count + one-line root cause>

Stage 2: <N>/50  (trials on disk: <actual find count>)
  Output dir: outputs/aspire_libero_long_eval/<suite>/<task>/

Key findings (3 bullets max):
  - <subtask decomposition approach>
  - <SAM3 prompts that worked for new objects>
  - <long-horizon pattern worth adding to skill library>
```
```

---
