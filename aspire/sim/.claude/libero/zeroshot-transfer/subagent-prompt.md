---
name: libero-90-zeroshot-build-subagent-prompt
description: Self-contained prompt template for LIBERO-90 scaling-law build subagents. Claude Code writes task code from scratch using skill library, debug on seeds 51–80 ONLY (30 seeds). No Stage 2. Copy, fill in TASK/GPU/TASKSHORT/CHUNK, pass to Agent tool.
---

# LIBERO-90 Scaling Build — Subagent Prompt Template

Copy the block below, fill in `TASK`, `GPU`, `TASKSHORT`, `CHUNK`, pass as `prompt` to `Agent(subagent_type="general-purpose", model="opus", run_in_background=True)`.

> **Model:** Use `model="opus"` (resolves to latest Opus, currently Opus 4.7). The Agent tool only accepts aliases "sonnet"/"opus"/"haiku" — specific sub-version IDs are not supported. Code synthesis + trace diagnosis benefit significantly from Opus.

> **Difference from non-scaling pipeline:**
> - Seeds 51–80 (not 51–65) — 30 seeds for debug
> - **No Stage 2** — no eval on LIBERO-90. Eval is on LIBERO-Long-Pro, run by coordinator after chunk.
> - Per-task commits retired — coordinator commits per chunk of 5.
> - Subagent exits after task_code.py finalized OR BLOCKED sentinel written.

---

```
## Task Assignment

CHUNK:      <chunk index, 1..18>
SUITE:      libero_90
TASK:       <full_task_name>
GPU:        <3|4|5|6|7>
TASKSHORT:  <short_unique_name_for_logs>

Working directory: $ASPIRE_ROOT  (resolve from env var)

---

## ⛔ EVAL SET LOCKOUT — READ THIS FIRST

**Seeds 1–50 are the LIBERO-90 evaluation set. They are LOCKED.**

- You may ONLY run seeds 51–80 (30 seeds for debug).
- Do NOT run seeds 1–50 EVER — not for testing, not for spot checking, not into /tmp.
- There is NO Stage 2 in this pipeline. You exit after Stage 1.

Violation invalidates the benchmark and poisons the scaling law curve.

---

## What You Are

Actor subagent for ASPIRE/LIBERO-90 scaling-law build phase. Write robot control code from scratch for ONE task, debug on seeds 51–80, write a findings.md, and exit. Do not run Stage 2. Do not eval on LIBERO-Long. Those are the coordinator's job after all 5 of you (from this chunk) finish.

All commands from working directory `$ASPIRE_ROOT` (resolve via `echo "$ASPIRE_ROOT"`).
Set `PYTHONPATH=$PYTHON_ROOT` in every python call.
Use `$ASPIRE_ROOT/.venv-libero/bin/python3` — never system python or any other venv.

---

## Context

**LIBERO-90** = 90 tasks across Kitchen, Living Room, Study scenes. Task language matches BDDL filename — no remapping. Always confirm with `env.handle.task_language`.

**Task types** (determine from task name before writing code):
- **Pick-and-place**: put X on Y / put X in basket/tray
- **Drawer**: open drawer / put X in drawer and close it / close drawer
- **Stove/microwave**: turn on stove / open microwave / turn on stove and put X on it
- **Stacking**: stack bowl A on bowl B
- **Relative placement**: put X to the left/right/front/back of Y
- **Shelf**: place on/under cabinet shelf

**Config for replay:**
```
env_configs/libero/franka_libero_traced.yaml
```

**Perception servers** must be up (404 = UP on 8114/8115/8116).

---

## ⛔ FORBIDDEN APIs — DO NOT USE

```
sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
inner.parsed_problem, inner._eval_predicate, inner.obj_body_id,
env.handle.env (unwrapping), sim.model.*, sim.data.qpos, sim.forward(),
env._step_once(), reading .bddl/.xml/.urdf asset files for geometry
```

Using these invalidates the benchmark — they don't transfer to real robots.

## ✅ ALLOWED APIs

```
get_observation()                              → RGB, depth, intrinsics, pose_mat, robot state
segment_sam3_text_prompt(rgb, text)            → SAM3 masks
segment_sam3_point_prompt(rgb, points)         → SAM3 masks by point
mask_to_world_points(mask, depth, K, T)       → (N,3) world-frame point cloud
plan_grasp(depth, intrinsics, mask)            → (grasp_poses, grasp_scores)
select_top_down_grasp(poses, scores, E)       → best grasp 4×4 world-frame
decompose_transform(T)                        → (position, quaternion_wxyz)
solve_ik(position, quaternion_wxyz)           → joint angles or None
move_to_joints(joints)                        → blocking motor control
open_gripper() / close_gripper()
goto_pose(pos, quat, z_approach=None)
goto_home_joint_position()
get_oriented_bounding_box_from_3d_points(pts) → {center, extent, R}
point_prompt_molmo(image, text)               → pixel (x, y)
env.handle.task_language                      → actual task instruction
numpy, scipy
```

---

## Stage 1: Write and Debug on Seeds 51–80

### Step 0 — Check if task_code.py already exists

```bash
cd "$ASPIRE_ROOT"
ls outputs/scaling_build/libero_90/$TASK/task_code.py 2>/dev/null && echo EXISTS || echo MISSING
```
If EXISTS → skip directly to Step 6 (write findings.md summarizing the existing task_code, then exit).

---

### Step 1 — Read the skill library (MANDATORY)

```bash
cd "$ASPIRE_ROOT"
cat .claude/libero/skills/grasp.md
cat .claude/libero/skills/localize.md
cat .claude/libero/skills/transport.md
cat .claude/libero/skills/manipulation.md
```

The skill library grows across chunks. Your chunk is #$CHUNK — earlier chunks may have added patterns you can reuse.

---

### Step 2 — Understand the task

Inspect the scene on seed 51:

```bash
cd "$ASPIRE_ROOT"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$PYTHON_ROOT" \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite libero_90 --args.task "$TASK" --args.trial 51 \
  --args.interactive \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.output-dir /tmp/repl_${TASKSHORT} 2>/dev/null << 'EOF'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
print(env.handle.task_language, flush=True)
obs = get_observation()
plt.imsave("/tmp/scene_${TASKSHORT}.png", obs["agentview"]["images"]["rgb"])
# Probe SAM3 for relevant objects from task name
for prompt in ["<object1>", "<object2>", "<target>", "<container>"]:
    masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], prompt)
    print(f"{prompt}: {len(masks)} masks" + (f", top={masks[0]['score']:.3f}" if masks else ""), flush=True)
EOF
```

From task name identify:
1. **Task type** (pick-and-place / drawer / stove / stacking / relative / shelf)
2. **What to pick** (if applicable)
3. **Where to place / what to manipulate**

---

### Step 3 — Write initial task_code.py

Use the appropriate template from the skill library. Base pick-and-place pattern:

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
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

obj_center, obj_pts, obj_mask = localize_object(rgb, depth, K, E, ["<object>"])
tgt_center, tgt_pts, _ = localize_object(rgb, depth, K, E, ["<target>"])
if obj_center is None: raise RuntimeError("Object not found")
if tgt_center is None: raise RuntimeError("Target not found")

surface_z = tgt_pts[:, 2].max()

grasp_poses, grasp_scores = plan_grasp(depth, K, obj_mask)
best_grasp, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp is None:
    best_grasp = E @ grasp_poses[grasp_scores.argmax()]
grasp_pos, quat = decompose_transform(best_grasp)

open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

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

For drawer / stove / microwave / stacking / relative / shelf — read `.claude/libero/skills/manipulation.md` for validated patterns and adapt.

---

### Step 4 — Test and iterate on seeds 51–80

```bash
cd "$ASPIRE_ROOT"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$PYTHON_ROOT" \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite libero_90 --args.task "$TASK" --args.trial <SEED> \
  --args.replay-code /tmp/task_${TASKSHORT}.py \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.output-dir /tmp/debug_${TASKSHORT}
```

`_reward_1.000_taskcompleted_1` = success.

**Hard limits (ABSOLUTE):**
- **Per seed**: max 3 replay attempts on the same seed number
- **Total code versions**: max 15 `/tmp/task_${TASKSHORT}_v*.py`. On v15 without success → BLOCK the task.
- Check version count: `ls /tmp/task_${TASKSHORT}_v*.py 2>/dev/null | wc -l`
- Debug seed order: try 51 first, then 52, 53 for initial test; then 54–80 for generalization.

**Need ≥3/5 passing on seeds 51–55 to finalize.** Then run 56–80 for coverage; record in findings.

**Read trace to diagnose failures:**
```bash
cat /tmp/debug_${TASKSHORT}/*/trace.json | python3 -c "
import json, sys
for step in json.load(sys.stdin):
    fn = step.get('function','')
    if fn in ('close_gripper','segment_sam3_text_prompt','plan_grasp','solve_ik','goto_pose'):
        print(fn, str(step.get('result',''))[:120])
" 2>/dev/null
```

---

### Step 5 — Finalize task_code.py

After ≥3/5 test seeds pass on 51–55:

```bash
cd "$ASPIRE_ROOT"
mkdir -p outputs/scaling_build/libero_90/$TASK
cp /tmp/task_${TASKSHORT}.py outputs/scaling_build/libero_90/$TASK/task_code.py
```

Then run coverage sweep on seeds 56–80, saving trial outputs to `outputs/` for auditability:

```bash
# One-shot coverage run on 56–80 — save to outputs/ so results persist
mkdir -p outputs/scaling_build/libero_90/$TASK/trials
for seed in $(seq 56 80); do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$PYTHON_ROOT" \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite libero_90 --args.task "$TASK" --args.trial $seed \
        --args.replay-code outputs/scaling_build/libero_90/$TASK/task_code.py \
        --args.config env_configs/libero/franka_libero_traced.yaml \
        --args.output-dir outputs/scaling_build/libero_90/$TASK/trials > /dev/null 2>&1 || true
done

# Count successes
find outputs/scaling_build/libero_90/$TASK/trials -type d -name "*reward_1.000_taskcompleted_1*" | wc -l
```

---

### Step 6 — Write findings.md (MANDATORY for coordinator to update skills)

```bash
cat > outputs/scaling_build/libero_90/$TASK/findings.md << 'EOF'
## Task: libero_90 / $TASK
## Task language: "<from env.handle.task_language>"
## Task type: <pick-and-place|drawer|stove|microwave|stacking|relative|shelf>
## Chunk: $CHUNK

### Objects (if applicable)
- Pick: <object>, SAM3 prompts: "<p1>", "<p2>"
- Target: <target>, SAM3 prompts: "<p1>", "<p2>"

### Root Cause(s) of Failures
- <what kept failing in v1..v(n-1)>

### What Fixed It
- <the insight that unblocked this task>

### SAM3 Prompts That Worked
| Object | Prompts | Notes |
|---|---|---|
| <object> | "<p1>", "<p2>" | |

### Generalizable Patterns (COORDINATOR READS THIS)
<Only list patterns that (a) plausibly help ≥2 other task types AND (b) achieved ≥15/30 on seeds 51–80.
Do NOT propose: hardcoded RGB thresholds, scene-specific color ratios, absolute XY coordinates, or
disambiguation logic tied to one BDDL file — these overfit debug seeds and hurt eval generalization.>
- <pattern 1>: <1-line description, proposed skill> — **<X>/30 seeds**
- <pattern 2>: <1-line description> — **<X>/30 seeds**

### Per-Seed Results (seeds 51–80)
| Seed | Result |
|---|---|
| 51 | pass/fail |
| 52 | pass/fail |
| ... | |

Pass rate on 51–80: <N>/30
EOF
```

If ALL debug seeds 51–55 failed after v15:
```bash
touch outputs/scaling_build/libero_90/$TASK/BLOCKED
cat > outputs/scaling_build/libero_90/$TASK/findings.md << 'EOF'
## Task: libero_90 / $TASK — BLOCKED
## Chunk: $CHUNK

### Why Blocked
- Wrote $N code versions over <time>; best seed rate was <X>/5 on 51–55.
- Root cause: <what we think is missing>

### What's Needed
- <missing perception primitive or skill pattern>
EOF
```

---

### Step 7 — Exit cleanly

Return a short summary:
- Task: $TASK
- Status: solved | blocked
- Seeds passing on 51–55: N/5
- Seeds passing on 56–80: N/25 (if solved)
- task_code.py: path or BLOCKED

**Do NOT run Stage 2. Do NOT touch seeds 1–50. Do NOT edit `.claude/libero/skills/*.md` — coordinator does that.**

---

## Common Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `segment_sam3_text_prompt` returns empty | Poor prompt | Try synonyms ("red mug" / "red coffee cup"); try point_prompt_molmo fallback |
| `plan_grasp` returns empty | Mask too small or noisy | Dilate mask slightly; check mask area > 100 px |
| `solve_ik` returns None | Pose out of workspace | Lower z, move x/y toward robot base |
| Gripper closes on air | Z too high | Use OBB `extent[2]/2` to get grasp z relative to object top |
| Succeeds on 51 but fails on 53 | Object position varies per seed | Re-localize before each action, don't cache positions across seeds |

---

## Context Hygiene

- If your context is getting long (>100k tokens), use `/compact` to summarize — but save `task_code.py` and any `/tmp/*.py` versions first.
- Keep code files under 300 lines. If you're over 300, you're probably over-engineering.
- Do not read `scripts/libero/replay_trial.py` source unless debugging a replay error — it's large and you don't need it.

---

## Safety

- Never push to remote. Never modify `.git/`. Never edit `CLAUDE.md`.
- If a tool seems to need permission, use it as configured — the coordinator ran you with `auto` mode.
- If you see a forbidden API in existing code, STOP and flag it in findings. Do not use it to "just get this working."
```
