---
name: libero-90-pro-subagent-prompt
description: Self-contained prompt template for LIBERO-90 actor subagents. Claude Code writes task code from scratch using skill library, seeds 51–65. Suite is always libero_90. Copy, fill in TASK/GPU/TASKSHORT, and pass to Agent tool.
---

# LIBERO-90 Actor Subagent Prompt Template

Copy the block below, fill in TASK/GPU/TASKSHORT, pass as `prompt` to `Agent(subagent_type="general-purpose", model="opus", run_in_background=True)`.

> **Model:** Use `model="opus"` — resolves to `claude-opus-4-6`. Code synthesis, trace diagnosis, and multi-seed generalization reasoning all benefit significantly from Opus 4.6.

---

```
## Task Assignment

SUITE:      libero_90
TASK:       <full_task_name>
GPU:        <3|4|5|6|7>
TASKSHORT:  <short_unique_name_for_logs>

Working directory: $ASPIRE_ROOT

---

## ⛔ EVAL SET LOCKOUT — READ THIS FIRST

**Seeds 1–50 are the evaluation set. LOCKED during Stage 1.**
- During Stage 1 (write + debug), ONLY run seeds 51–65.
- Do NOT run seeds 1–50 until task_code.py is finalized and frozen.
- Stage 2 runs exactly once after task_code.py is frozen.

Violation invalidates benchmark results.

---

## What You Are

Actor subagent for ASPIRE/LIBERO-90. Write robot control code from scratch for one task,
debug on seeds 51–65, run Stage 2 eval on seeds 1–50, return a findings report.

All commands from working directory `$ASPIRE_ROOT`.
Set `PYTHONPATH=$PYTHON_ROOT` in every python call.
**.venv-libero/bin/python3** — use the LIBERO experiment env for every replay/eval command.

---

## Context

**LIBERO-90** is a single suite with 90 tasks across Kitchen, Living Room, and Study scenes.
Task language matches the BDDL filename — no remapping. Always confirm with `env.handle.task_language`.

**Task types** (determine from task name before writing code):
- **Pick-and-place**: put X on Y / put X in basket/tray
- **Drawer**: open drawer / put X in drawer and close it / close drawer
- **Stove/microwave**: turn on stove / open microwave / turn on stove and put X on it
- **Stacking**: stack bowl A on bowl B
- **Relative placement**: put X to the left/right of Y / place in front/back/left/right compartment
- **Shelf**: place on/under cabinet shelf

**Config for replay and REPL:**
```
env_configs/libero/franka_libero_traced.yaml
```

**Servers:** 404 = UP on 8114 (SAM3), 8115 (GraspNet), 8116 (PyRoKi). Must all be UP.

---

## ⛔ FORBIDDEN APIs

```
sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
inner.parsed_problem, inner._eval_predicate, inner.obj_body_id,
env.handle.env (unwrapping), sim.model.*, sim.data.qpos, sim.forward(),
env._step_once(), reading .bddl/.xml/.urdf asset files
```

## ✅ ALLOWED APIs

```
get_observation()                              → RGB, depth, intrinsics, pose_mat, robot state
segment_sam3_text_prompt(rgb, text)            → SAM3 masks [{mask, box, score, label}]
segment_sam3_point_prompt(rgb, points)         → SAM3 masks by point
mask_to_world_points(mask, depth, K, T)       → (N,3) world-frame point cloud
plan_grasp(depth, intrinsics, mask)            → (grasp_poses, grasp_scores)
select_top_down_grasp(poses, scores, E)       → best grasp 4×4 world-frame matrix
decompose_transform(T)                        → (position, quaternion_wxyz)
solve_ik(position, quaternion_wxyz)           → joint angles or None
move_to_joints(joints)                        → blocking motor control
open_gripper() / close_gripper()
goto_pose(pos, quat, z_approach=None)
goto_home_joint_position()
get_oriented_bounding_box_from_3d_points(pts) → {center, extent, R}
point_prompt_molmo(image, text)               → pixel (x, y)
env.handle.task_language                      → task instruction string
numpy, scipy
```

---

## Stage 1: Write and Debug on Seeds 51–65

### Step 0 — Check if task_code.py already exists

```bash
cd $ASPIRE_ROOT
ls outputs/aspire_actor_90/libero_90/$TASK/task_code.py 2>/dev/null && echo EXISTS || echo MISSING
```
If EXISTS → skip to Stage 2 guard.

---

### Step 1 — Read the skill library (MANDATORY before writing code)

```bash
cd $ASPIRE_ROOT
cat .claude/libero/skills/grasp.md
cat .claude/libero/skills/localize.md
cat .claude/libero/skills/transport.md
cat .claude/libero/skills/manipulation.md
```

---

### Step 2 — Understand the task (MANDATORY before writing any code)

Inspect the scene on seed 51:

```bash
cd $ASPIRE_ROOT
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$PYTHON_ROOT \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite libero_90 --args.task "$TASK" --args.trial 51 \
  --args.interactive \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.output-dir /tmp/repl_out 2>/dev/null << 'EOF'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
print(env.handle.task_language, flush=True)
obs = get_observation()
plt.imsave("/tmp/scene_$TASKSHORT.png", obs["agentview"]["images"]["rgb"])
print("Scene saved", flush=True)
# Probe SAM3 for relevant objects from the task name
for prompt in ["<object1>", "<object2>", "<target>", "<container>"]:
    masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], prompt)
    print(f"{prompt}: {len(masks)} masks" + (f", top={masks[0]['score']:.3f}" if masks else ""), flush=True)
EOF
```

From the task name, identify:
1. **Task type** (pick-and-place / drawer / stove / stacking / relative / shelf)
2. **What to pick** (if applicable)
3. **Where to place / what to manipulate**

---

### Step 3 — Write initial task_code.py

Use the appropriate template based on task type.

**Pick-and-place / basket / tray template:**

```python
import numpy as np
from scipy.spatial.transform import Rotation

def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def localize_object(rgb, depth, K, E, prompts):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    for prompt in ([prompts] if isinstance(prompts, str) else prompts):
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        best = max(masks, key=lambda d: d["score"])
        pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 10: continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, best["mask"]
    return None, None, None

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K, E = cam["intrinsics"], cam["pose_mat"]

# Detect both objects before any arm movement
obj_center, obj_pts, obj_mask = localize_object(rgb, depth, K, E, ["<object prompt>"])
tgt_center, tgt_pts, _ = localize_object(rgb, depth, K, E, ["<target prompt>"])
if obj_center is None: raise RuntimeError("Object not found")
if tgt_center is None: raise RuntimeError("Target not found")

surface_z = tgt_pts[:, 2].max()

# Grasp
grasp_poses, grasp_scores = plan_grasp(depth, K, obj_mask)
best_grasp, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp is None:
    best_grasp = E @ grasp_poses[grasp_scores.argmax()]
grasp_pos, quat = decompose_transform(best_grasp)

open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# Lift and place
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)

above = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)

release_pos = np.array([tgt_center[0], tgt_center[1], surface_z + 0.05])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)
open_gripper()

for _ in range(3): get_observation()
```

**For other task types** (drawer, stove, microwave, stacking, relative placement):
- Read `.claude/libero/skills/manipulation.md` for validated patterns
- Adapt accordingly — add subtask steps, change approach vectors, use manipulation primitives

---

### Step 4 — Test and iterate

```bash
cd $ASPIRE_ROOT
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$PYTHON_ROOT \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite libero_90 --args.task "$TASK" --args.trial 51 \
  --args.replay-code /tmp/task_$TASKSHORT.py \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.output-dir /tmp/debug_90_$TASKSHORT
```

`_reward_1.000_taskcompleted_1` in dir name = success.

**Hard limits (ABSOLUTE — no exceptions):**
- **Per seed**: max 3 replay attempts on any one seed number.
- **Total code versions**: if you've written more than 15 `/tmp/task_$TASKSHORT_v*.py` files with zero successes, **BLOCK the task immediately.** Do not write v16. The task may require a perception primitive not yet in the skill library.
- Check version count before each new version: `ls /tmp/task_${TASKSHORT}_v*.py 2>/dev/null | wc -l`
- If blocked: write `outputs/aspire_actor_90/libero_90/$TASK/BLOCKED` and return. The coordinator will escalate.

**Test seeds 51, 52, 53. Need 2/3 passing to finalize.**

Read trace to diagnose failures:
```bash
cat /tmp/debug_90_$TASKSHORT/*/trace.json | python3 -c "
import json, sys
for step in json.load(sys.stdin):
    fn = step.get('function','')
    if fn in ('close_gripper','segment_sam3_text_prompt','plan_grasp','solve_ik','goto_pose'):
        print(fn, str(step.get('result',''))[:120])
" 2>/dev/null
```

---

### Step 5 — Finalize task_code.py

After 2/3 test seeds pass:
```bash
cd $ASPIRE_ROOT
mkdir -p outputs/aspire_actor_90/libero_90/$TASK
cp /tmp/task_$TASKSHORT.py outputs/aspire_actor_90/libero_90/$TASK/task_code.py
```

Write findings.md:
```bash
cat > outputs/aspire_actor_90/libero_90/$TASK/findings.md << 'EOF'
## Task: libero_90 / $TASK
## Task language: "<from env.handle.task_language>"
## Task type: <pick-and-place|drawer|stove|microwave|stacking|relative|shelf>

### Objects
- Pick: <object>, SAM3 prompts: "<p1>", "<p2>"
- Target: <target>, SAM3 prompts: "<p1>", "<p2>"

### Root Cause(s) of Failures
- <failure mode>

### What Fixed It
- <approach>

### SAM3 Prompts That Worked
| Object | Prompts | Notes |
|---|---|---|
| <object> | "<p1>", "<p2>" | |

### Generalizable Patterns
- <anything reusable for other tasks>

### Stage 2 Success Rate
N/50
EOF
```

If ALL test seeds blocked:
```bash
touch outputs/aspire_actor_90/libero_90/$TASK/BLOCKED
```

---

## Stage 2: Eval on Seeds 1–50

### Guard check first
```bash
cd $ASPIRE_ROOT
existing=$(find outputs/aspire_eval_90 -maxdepth 6 -type d -name "trial_*" 2>/dev/null | grep "libero_90/$TASK" | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
echo "Seeds already on disk: $existing"
```
If existing >= 45: **Stage 2 already done. Return immediately.**

### Write, sanity-check, launch, and poll

Step 1 — Write the script:
```bash
cd $ASPIRE_ROOT
cat > /tmp/stage2_90_$TASKSHORT.sh << 'SCRIPT'
#!/bin/bash
# NO set -e — individual trial failures must not kill the loop
cd $ASPIRE_ROOT
TASK="FILL_IN_TASK"
GPU=FILL_IN_GPU
TASKSHORT="FILL_IN_TASKSHORT"
FIX_CODE="outputs/aspire_actor_90/libero_90/${TASK}/task_code.py"
LOG="/tmp/val90_${TASKSHORT}_progress.log"

# Sanity checks — fail loudly rather than silently writing nothing
if [[ ! -f "$FIX_CODE" ]]; then
    echo "ERROR: FIX_CODE not found: $FIX_CODE" | tee -a "$LOG"; exit 1
fi
echo "Stage 2 start: FIX_CODE=$FIX_CODE  GPU=$GPU" | tee -a "$LOG"

for trial in $(seq 1 50); do
    trial_padded=$(printf "%02d" $trial)
    # Skip already-completed trials (idempotent)
    if find outputs/aspire_eval_90 -maxdepth 6 -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q "libero_90/$TASK"; then
        echo "Trial ${trial}: skip" | tee -a "$LOG"; continue
    fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH=$PYTHON_ROOT \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite libero_90 --args.task "${TASK}" --args.trial ${trial} \
        --args.replay-code "${FIX_CODE}" \
        --args.config env_configs/libero/franka_libero_traced.yaml \
        --args.output-dir outputs/aspire_eval_90 > "/tmp/val90_${TASKSHORT}_${trial}.log" 2>&1 || true
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "/tmp/val90_${TASKSHORT}_${trial}.log" | tail -1 | sed 's/reward_//')
    echo "Trial $trial: ${reward:-ERROR}" | tee -a "$LOG"
done
echo "STAGE2_DONE" | tee -a "$LOG"
SCRIPT
sed -i "s/FILL_IN_TASK/$TASK/g; s/FILL_IN_GPU/$GPU/g; s/FILL_IN_TASKSHORT/$TASKSHORT/g" /tmp/stage2_90_$TASKSHORT.sh
# Verify no unresolved placeholders before launching
grep "FILL_IN_" /tmp/stage2_90_$TASKSHORT.sh && echo "WARNING: unresolved placeholders!" || echo "OK: placeholders resolved"
chmod +x /tmp/stage2_90_$TASKSHORT.sh
```

Step 2 — Launch (`nohup &` survives agent timeout/crash):
```bash
nohup bash /tmp/stage2_90_$TASKSHORT.sh > /tmp/stage2_90_${TASKSHORT}.out 2>&1 &
echo $! > /tmp/stage2_90_${TASKSHORT}.pid
echo "Stage 2 launched PID=$(cat /tmp/stage2_90_${TASKSHORT}.pid)"
```

Step 3 — Poll until done (small Bash calls every ~5 min):
```bash
LOG="/tmp/val90_${TASKSHORT}_progress.log"
done_flag=$(grep -c "STAGE2_DONE" "$LOG" 2>/dev/null || echo 0)
count=$(grep -c "Trial" "$LOG" 2>/dev/null || echo 0)
echo "Stage 2: $count/50 trials logged, done=$done_flag"
tail -3 "$LOG" 2>/dev/null
```
Repeat until `done=1`.

Step 4 — Read final results:
```bash
cd $ASPIRE_ROOT
actual=$(find outputs/aspire_eval_90 -maxdepth 6 -type d -name "trial_*" 2>/dev/null | grep "libero_90/$TASK" | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
successes=$(find outputs/aspire_eval_90 -maxdepth 6 -type d -name "*taskcompleted_1*" 2>/dev/null | grep "libero_90/$TASK" | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
echo "Stage 2: $successes/$actual"
```

---

## What to Return

```
SUITE: libero_90
TASK: <task>
GPU: <N>
Task type: <pick-and-place|drawer|stove|microwave|stacking|relative|shelf>
Task language: "<from env.handle.task_language>"

Stage 1: task_code.py written: yes/no
  Test seeds passed: <e.g. 51✓ 52✗ 53✓>
  Seeds blocked: <count + root cause>

Stage 2: <N>/50  (trials on disk: <actual count>)

Key findings (3 bullets max):
  - SAM3 prompts that worked for relevant objects
  - Grasp/manipulation approach and any key parameters
  - Any pattern worth adding to skill library
```
```
