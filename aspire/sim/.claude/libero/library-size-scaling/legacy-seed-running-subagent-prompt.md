---
name: libero-library-size-legacy-seed-running-subagent-prompt
description: Self-contained prompt template for LIBERO-Long-Pro zero-shot eval subagents. Given a frozen strategy library snapshot, write code once per task (one-shot mode) and execute across assigned seeds. No debug, no iteration, no skill updates. Copy, fill in SUITE/TASK/GPU/SEEDS/SNAPSHOT/ASPIRE_ROOT_SNAPSHOT, pass to Agent tool.
---

# LIBERO-Long-Pro Eval — Subagent Prompt Template (One-Shot Mode)

Copy the block below, fill in `SUITE`, `TASK`, `GPU`, `TASKSHORT`, `SEED_START`, `SEED_END`, `SNAPSHOT`, `ASPIRE_ROOT_SNAPSHOT`, pass as `prompt` to `Agent(subagent_type="general-purpose", model="opus", run_in_background=True)`.

> **Model:** Use `model="opus"` — long-horizon task decomposition and multi-object localization benefit from Opus.

> **Key difference from build subagents:**
> - This is **zero-shot evaluation**. You write code ONCE per task, execute on all seeds. No debug loop.
> - Library is **frozen** — you read `.claude/libero/skills/` from a snapshot worktree. You CANNOT modify skills.
> - No `findings.md`. No `task_code.py` saved to outputs. Just execute and record success/failure.
> - Seeds 1–50 are the eval set. You run them directly — there is no "Stage 1" debug phase.

---

```
## Task Assignment

SNAPSHOT:           <snapshot-N0|snapshot-N5|snapshot-N10|...|snapshot-N90>
SUITE:              <libero_10_swap|libero_10_task>
TASK:               <full_task_name_with_SCENE_prefix>
GPU:                <3|4|5|6|7>
TASKSHORT:          <short_unique_name_for_logs>
SEED_START:         <1>
SEED_END:           <50>

Worktree bootstrap path: $ASPIRE_ROOT_SNAPSHOT
(After Step 0 sources the .env, $ASPIRE_ROOT == this path for all subsequent commands.)

---

## ⚠️ STEP 0 — WORKTREE SETUP & VERIFICATION (DO THIS BEFORE ANYTHING ELSE)

Source the worktree's .env to fence $ASPIRE_ROOT to the snapshot. After this block,
use $ASPIRE_ROOT everywhere — never relative paths, never cd again.

```bash
# ── 1. Source .env — sets ASPIRE_ROOT, SNAPSHOT, MUJOCO_GL, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, PYTHONPATH
source "$ASPIRE_ROOT_SNAPSHOT/.env" || { echo "FATAL: cannot source $ASPIRE_ROOT_SNAPSHOT/.env"; exit 1; }
echo "ASPIRE_ROOT=$ASPIRE_ROOT"
echo "SNAPSHOT=$SNAPSHOT"

# ── 2. Verify git tag
cd "$ASPIRE_ROOT" || { echo "FATAL: cannot cd to $ASPIRE_ROOT"; exit 1; }
TAG=$(git describe --tags --exact-match 2>/dev/null || echo "NOTAG")
echo "Git tag: $TAG"
if [[ "$TAG" != "$SNAPSHOT" ]]; then
  echo "FATAL: wrong worktree — expected $SNAPSHOT, got $TAG — aborting"
  exit 1
fi
echo "✓ Worktree verified: $ASPIRE_ROOT at $SNAPSHOT"

# ── 3. Set output dir (always absolute)
OUTDIR="$ASPIRE_ROOT/outputs/scaling_eval/$SNAPSHOT/one_shot"
mkdir -p "$OUTDIR"
echo "OUTDIR=$OUTDIR"

# ── 4. Contamination guard — HARD BLOCK
if [[ -d "$ASPIRE_ROOT/outputs/scaling_build/libero_90" ]]; then
  echo "FATAL: scaling_build/libero_90 present inside worktree — task_code.py files contaminate zero-shot eval"
  exit 1
fi
if find "$ASPIRE_ROOT/outputs/scaling_build" -name "task_code.py" 2>/dev/null | grep -q .; then
  echo "FATAL: stray task_code.py found inside worktree scaling_build"
  exit 1
fi
# Main repo path via relative traversal
MAIN_BUILD="$ASPIRE_ROOT/../../../outputs/scaling_build"
if [[ -d "$MAIN_BUILD/libero_90" ]]; then
  echo "FATAL: main repo scaling_build/libero_90 accessible at $(realpath $MAIN_BUILD/libero_90) — contamination risk"
  exit 1
fi
echo "✓ Contamination guard passed"
```

**If any check fails, STOP immediately. Report the failure.**
**After Step 0: $ASPIRE_ROOT is your project root. Never use cd again. All paths are absolute via $ASPIRE_ROOT.**

---

## What You Are

Zero-shot eval subagent for ASPIRE/LIBERO-Long-Pro. Your job:
1. Verify worktree (Step 0 above — mandatory)
2. Read the frozen strategy library (skills checked out at the snapshot tag)
3. Read the task description
4. Write ONE complete program for the task (no iteration, no debug)
5. Execute that program on every seed in your assigned range
6. Report success/failure counts

You do NOT debug failures. You do NOT update skills. You do NOT retry with different code.
This is a measurement — the library either helps you write good code or it doesn't.

Use `$ASPIRE_ROOT/.venv-libero/bin/python3` — never system python or any other venv.
`$PYTHONPATH`, `$MUJOCO_GL`, `$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` are already set from the .env.

---

## Context

**What is LIBERO-Long-Pro?**
LIBERO-Long-Pro (`libero_10_swap` / `libero_10_task`) is a long-horizon robotic manipulation benchmark.
Each task requires completing **2–3 sequential subtasks** in one episode (e.g., "pick A AND pick B AND place both in basket", "open drawer, place object, close drawer").

**Your code must handle the entire task in a single Python script** — all subtasks in sequence, one monolithic file.

**Perturbation types:**
- `libero_10_swap`: object positions randomized per seed — SAM3 handles naturally
- ⚠️ `libero_10_task`: **language goal is remapped — the BDDL filename is MISLEADING.** ALWAYS use `env.handle.task_language` for the actual instruction. Do NOT trust the task name for `_task` suites.

⚠️ **Suite name collision:** `libero_10_swap` and `libero_10_task` share **identical task names**. The SUITE field determines which variant you're running. Always include full SUITE in logs and output paths.

**Config for replay:**
```
env_configs/libero/franka_libero_libero10_traced.yaml
```

**Perception servers must be running** (404 = UP on 8114/8115/8116).

---

## LIBERO-Long-Pro Task List (10 tasks, shared across both suites)

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

## ⛔ FORBIDDEN APIs

**Using these invalidates benchmark results — they don't transfer to real robots:**
```
sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
inner.parsed_problem, inner._eval_predicate, inner.obj_body_id,
env.handle.env (unwrapping), sim.model.*, sim.data.qpos, sim.forward(),
env._step_once(), reading .bddl/.xml/.urdf asset files for geometry
```

## ⛔ FORBIDDEN SOURCES — BENCHMARK CONTAMINATION

**You may ONLY read `.claude/libero/skills/*.md` from your snapshot worktree. Everything else is forbidden:**
```
outputs/scaling_build/          — DO NOT READ (libero_90 task codes, built on seeds 51–80)
outputs/scaling_eval/           — DO NOT READ (other tasks' eval outputs)
Any task_code.py or findings.md — DO NOT READ
Other snapshots' worktrees      — DO NOT READ
/tmp/ code files from other tasks — DO NOT READ
```

**Self-check before writing code:** If you found useful information outside `.claude/libero/skills/`, you are contaminating the eval. Discard it and write only from skill library patterns.

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

## Step 1 — Read the frozen skill library (MANDATORY)

```bash
cat "$ASPIRE_ROOT/.claude/libero/skills/grasp.md"
cat "$ASPIRE_ROOT/.claude/libero/skills/localize.md"
cat "$ASPIRE_ROOT/.claude/libero/skills/transport.md"
cat "$ASPIRE_ROOT/.claude/libero/skills/manipulation.md"
```

These skills were accumulated during the scaling-law build phase. Use patterns, helper functions, prompt registry entries, and parameters you find — but **critically evaluate whether each pattern generalizes to YOUR task**.

### Skill Generalizability Check (MANDATORY before writing code)

The skill library was built on LIBERO-90 tasks, which differ from LIBERO-Long-Pro tasks. Before using a skill pattern, ask:
1. **Does this object match?** Registry entries (e.g., "moka pot top-knob grasp at z_max-0.012") are tuned for specific objects. If your task has a different object, adapt or use generic GraspNet instead.
2. **Does the scene geometry match?** Workspace filters were calibrated for specific LIBERO-90 scenes. LIBERO-Long-Pro scenes may have different layouts. Widen or remove filters that seem scene-specific.
3. **Is the manipulation pattern applicable?** Drawer/microwave parameters are scene-specific. Use the pattern structure but re-probe dimensions from perception.
4. **Are SAM3 prompts general enough?** Prefer generic prompts ("mug", "plate", "basket") over overly specific ones unless disambiguation requires it.

**Rule of thumb:** Use the skill library's *structure* (home-reset, lift-transit-descend, multi-pass IK) but verify *parameters* against what you observe in YOUR scene.

---

## Step 2 — Understand the task

For `_task` suites, get the ACTUAL instruction first:

```bash
CUDA_VISIBLE_DEVICES=$GPU \
"$ASPIRE_ROOT/.venv-libero/bin/python3" "$ASPIRE_ROOT/scripts/libero/replay_trial.py" \
  --args.suite $SUITE --args.task "$TASK" --args.trial $SEED_START \
  --args.interactive \
  --args.config "$ASPIRE_ROOT/env_configs/libero/franka_libero_libero10_traced.yaml" \
  --args.output-dir "/tmp/eval_repl_${TASKSHORT}" 2>/dev/null << 'EOF'
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
task_lang = env.handle.task_language
print(f"TASK_LANGUAGE: {task_lang}", flush=True)
obs = get_observation()
plt.imsave("/tmp/eval_scene_${TASKSHORT}.png", obs["agentview"]["images"]["rgb"])
print("Scene saved", flush=True)
# Probe SAM3 for objects from the task instruction
for prompt in ["<object1>", "<object2>", "<target>", "<container>"]:
    masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], prompt)
    print(f"{prompt}: {len(masks)} masks" + (f", top={masks[0]['score']:.3f}" if masks else ""), flush=True)
EOF
```

View `/tmp/eval_scene_${TASKSHORT}.png` to understand layout. Decompose into subtasks.

---

## Step 3 — Write ONE complete program

Write a single `/tmp/eval_os_${SUITE}_${TASKSHORT}.py` that handles all subtasks.

### Long-Horizon Code Rules (CRITICAL)

1. **Call `goto_home_joint_position()` BEFORE every subtask** — it prevents joint limit issues between consecutive operations and gives a clean known state.
2. **Re-call `get_observation()` after each home reset** — the camera view shifts when the arm moves home; you need fresh RGB/depth for the next subtask's localization.
3. **Never assume subtask 1 succeeded before starting subtask 2** — write best-effort code for each subtask independently where possible. Even if pick A failed, still attempt pick B.
4. **For "AND close it" tasks (drawer, microwave):** ALWAYS do the close step even if the place/put step failed partially. Closing is often the final success predicate.
5. **Write for eval seeds (1–50), not one specific scene.** Use robust SAM3 prompts that work across position variations, not hardcoded coordinates.

### Code structure template:

```python
import numpy as np
from scipy.spatial.transform import Rotation

# ── Utilities (from skill library) ───────────────────────────────────────────

def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def localize_object(rgb, depth, K, T, prompts):
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    for prompt in ([prompts] if isinstance(prompts, str) else prompts):
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        best = max(masks, key=lambda d: d["score"])
        pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, T)
        if pts is None or len(pts) < 10: continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, best["mask"]
    return None, None, None

def pick_and_place(rgb, depth, K, T, E, pick_prompts, place_prompts, place_z_offset=0.05):
    """Full pick-and-place using skill library patterns."""
    obj_center, obj_pts, obj_mask = localize_object(rgb, depth, K, T, pick_prompts)
    tgt_center, tgt_pts, _ = localize_object(rgb, depth, K, T, place_prompts)
    if obj_center is None or tgt_center is None:
        return False

    surface_z = tgt_pts[:, 2].max()
    grasp_poses, grasp_scores = plan_grasp(depth[:,:,0] if len(depth.shape)==3 else depth, K, obj_mask)
    if grasp_poses is None or len(grasp_poses) == 0:
        return False
    best_grasp, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
    if best_grasp is None:
        best_grasp = E @ grasp_poses[grasp_scores.argmax()]
    grasp_pos, quat = decompose_transform(best_grasp)

    open_gripper()
    goto_pose(grasp_pos, quat, z_approach=0.15)
    goto_pose(grasp_pos, quat)
    close_gripper()

    lift_z = grasp_pos[2] + 0.15
    joints = solve_ik([grasp_pos[0], grasp_pos[1], lift_z], quat.tolist())
    if joints is not None: move_to_joints(joints)

    above = [tgt_center[0], tgt_center[1], lift_z]
    joints = solve_ik(above, quat.tolist())
    if joints is not None: move_to_joints(joints)

    release = [tgt_center[0], tgt_center[1], surface_z + place_z_offset]
    joints = solve_ik(release, quat.tolist())
    if joints is not None: move_to_joints(joints)
    open_gripper()
    return True

# ── Main task ─────────────────────────────────────────────────────────────────

task_lang = env.handle.task_language
print(f"Task: {task_lang}", flush=True)

# --- Subtask 1 ---
goto_home_joint_position()
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
depth = obs["agentview"]["images"]["depth"]
K, T = obs["agentview"]["intrinsics"], obs["agentview"]["pose_mat"]
E = obs["robot"]["eef_pose"]

# <adapt from task decomposition — use skill library patterns>
# pick_and_place(rgb, depth, K, T, E, ["object1 prompt"], ["target prompt"])

# --- Subtask 2 ---
goto_home_joint_position()          # ALWAYS home between subtasks
obs = get_observation()              # ALWAYS re-observe after home
rgb = obs["agentview"]["images"]["rgb"]
depth = obs["agentview"]["images"]["depth"]
K, T = obs["agentview"]["intrinsics"], obs["agentview"]["pose_mat"]
E = obs["robot"]["eef_pose"]

# <second subtask — best-effort even if subtask 1 failed>

# --- Close step (if applicable) ---
# ALWAYS attempt close even if prior subtask failed
# goto_home_joint_position()
# <close drawer / microwave>

for _ in range(3): get_observation()  # let sim settle
```

**Do NOT iterate.** Write the best code you can from the skill library + task understanding, then move to Step 4.

---

## Step 4 — Execute on all assigned seeds

```bash
OUTDIR="$ASPIRE_ROOT/outputs/scaling_eval/${SNAPSHOT}/one_shot"
LOG="/tmp/eval_os_${TASKSHORT}_progress.log"
CODE="/tmp/eval_os_${SUITE}_${TASKSHORT}.py"

echo "Eval start: $SUITE/$TASK seeds ${SEED_START}–${SEED_END} snapshot=$SNAPSHOT" | tee "$LOG"
echo "ASPIRE_ROOT: $ASPIRE_ROOT" | tee -a "$LOG"
echo "OUTDIR: $OUTDIR" | tee -a "$LOG"

for seed in $(seq $SEED_START $SEED_END); do
    trial_padded=$(printf "%02d" $seed)
    if find "$OUTDIR/$SUITE/$TASK" -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q .; then
        echo "Seed $seed: skip (exists)" | tee -a "$LOG"; continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU \
    "$ASPIRE_ROOT/.venv-libero/bin/python3" "$ASPIRE_ROOT/scripts/libero/replay_trial.py" \
        --args.suite "$SUITE" --args.task "$TASK" --args.trial $seed \
        --args.replay-code "$CODE" \
        --args.config "$ASPIRE_ROOT/env_configs/libero/franka_libero_libero10_traced.yaml" \
        --args.output-dir "$OUTDIR" > "/tmp/eval_os_${TASKSHORT}_seed${seed}.log" 2>&1 || true

    result=$(grep -oE "taskcompleted_[01]" "/tmp/eval_os_${TASKSHORT}_seed${seed}.log" | tail -1 || echo "ERROR")
    echo "Seed $seed: ${result}" | tee -a "$LOG"
done

echo "EVAL_DONE" | tee -a "$LOG"
```

**Do NOT debug failures.** This is zero-shot measurement. If seed 17 fails, record it and move on.

---

## Step 5 — Report results

```bash
OUTDIR="$ASPIRE_ROOT/outputs/scaling_eval/${SNAPSHOT}/one_shot"
actual=$(find "$OUTDIR/$SUITE/$TASK" -type d -name "trial_*" 2>/dev/null | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
successes=$(find "$OUTDIR/$SUITE/$TASK" -type d -name "*taskcompleted_1*" 2>/dev/null | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
echo "Result: $successes/$actual"
echo "Output confirmed in: $OUTDIR/$SUITE/$TASK"
```

---

## What to Return

```
SNAPSHOT: <tag>
SUITE: <suite>
TASK: <task>
GPU: <N>

Seeds run: <SEED_START>–<SEED_END>
Success: <N>/<total>
Output dir: outputs/scaling_eval/<SNAPSHOT>/one_shot/<SUITE>/<TASK>/

Code approach (1 line): <e.g. "double pick-and-place with basket drop">
Library patterns used: <which skill entries helped, if any>
```

Do NOT include code, traces, or debug analysis. Just the numbers.
```

---
