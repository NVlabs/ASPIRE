---
name: libero-library-size-codegen-subagent-prompt
description: Code-generation-only subagent prompt for LIBERO-Long-Pro zero-shot eval. Subagent writes ONE program per task and returns the code path. Does NOT execute seeds — the coordinator handles that via bash scripts. This avoids the 600s watchdog stall that kills subagents during long seed loops.
---

# LIBERO-Long-Pro Eval — Code Generation Subagent (One-Shot Mode)

## Architecture

This is the **code-generation half** of the split eval pipeline:
1. **Subagent** (this prompt): reads frozen skill library → probes scene → writes ONE program → saves to known path → returns
2. **Coordinator**: collects code paths → runs all seeds via `scripts/libero/resume_eval_gpu.sh` bash scripts

The subagent does NOT run seeds. It returns as soon as code is written.

## How to use

Fill in placeholders, pass as `prompt` to `Agent(subagent_type="general-purpose", model="opus", run_in_background=True)`.

> **Model:** Use `model="opus"` — long-horizon task decomposition benefits from Opus.

---

```
## Task Assignment

SNAPSHOT:           <snapshot-N0|snapshot-N5|snapshot-N10|...|snapshot-N90>
SUITE:              <libero_10_swap|libero_10_task>
TASK:               <full_task_name_with_SCENE_prefix>
GPU:                <3|4|5|6|7>
TASKSHORT:          <short_unique_name_for_logs>

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
mkdir -p "$OUTDIR/$SUITE/$TASK"
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

Code-generation subagent for ASPIRE/LIBERO-Long-Pro zero-shot eval. Your job:
1. Verify worktree (Step 0 above — mandatory)
2. Read the frozen strategy library (skills checked out at the snapshot tag)
3. Read the task description and probe the scene
4. Write ONE complete program for the task (no iteration, no debug)
5. Save code to a known path and return

You do NOT execute seeds. You do NOT debug failures. You do NOT update skills.
The coordinator will run your code across seeds 1–50 after you return.

Use `$ASPIRE_ROOT/.venv-libero/bin/python3` — never system python.
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
1. **Does this object match?** Registry entries (e.g., "moka pot top-knob grasp at z_max-0.012") are tuned for specific objects. If your task has a different object (e.g., a mug vs a moka pot), the parameters may not transfer. Adapt or use generic GraspNet instead.
2. **Does the scene geometry match?** Workspace filters (e.g., "x in [0.35, 0.65]") were calibrated for specific LIBERO-90 scenes. LIBERO-Long-Pro scenes may have different layouts. Widen or remove filters that seem scene-specific.
3. **Is the manipulation pattern applicable?** Drawer open/close parameters are scene-specific (hinge position, pull distance). Use the pattern structure but re-probe dimensions from perception rather than hardcoding registry values.
4. **Are SAM3 prompts general enough?** Prefer generic prompts ("mug", "plate", "basket") over overly specific ones ("red cylindrical can") unless disambiguation requires it.

**Rule of thumb:** Use the skill library's *structure* (e.g., home-reset-between-subtasks, lift-transit-descend, multi-pass IK descent) but verify *parameters* against what you observe in YOUR scene. The library teaches you HOW to do things; your scene observation tells you WHAT values to use.

---

## Step 2 — Understand the task (MANDATORY — do NOT skip)

**This step MUST complete before you write any code.** Run the interactive probe below and read the full output. If it fails (timeout, crash), retry once. If it fails again, use a simpler probe (just `env.handle.task_language`). Do NOT proceed to Step 3 without knowing the task language and which SAM3 prompts detect objects.

Do NOT try to view the saved PNG image (it causes API errors). Use the SAM3 probe text output to understand what objects are present.

```bash
CUDA_VISIBLE_DEVICES=$GPU \
"$ASPIRE_ROOT/.venv-libero/bin/python3" "$ASPIRE_ROOT/scripts/libero/replay_trial.py" \
  --args.suite $SUITE --args.task "$TASK" --args.trial 1 \
  --args.interactive \
  --args.config "$ASPIRE_ROOT/env_configs/libero/franka_libero_libero10_traced.yaml" \
  --args.output-dir "/tmp/eval_repl_${TASKSHORT}" 2>/dev/null << 'EOF'
import numpy as np
task_lang = env.handle.task_language
print(f"TASK_LANGUAGE: {task_lang}", flush=True)
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
# Probe SAM3 for objects mentioned in the task instruction
# Adapt these prompts based on the task language above
for prompt in ["basket", "plate", "mug", "bowl", "stove", "drawer", "microwave", "moka pot", "book", "caddy", "butter", "cream cheese", "alphabet soup", "tomato sauce", "chocolate pudding", "bottle", "cup", "box", "pan", "frying pan", "ketchup", "white mug", "red mug", "yellow mug"]:
    masks = segment_sam3_text_prompt(rgb, prompt)
    if masks:
        print(f"{prompt}: {len(masks)} masks, top={masks[0]['score']:.3f}", flush=True)
EOF
```

**Before proceeding to Step 3, confirm you have:**
1. The actual TASK_LANGUAGE (especially for `_task` suite — it's often different from the task name)
2. SAM3 detection results showing which prompts find which objects and their scores
3. A clear decomposition of the task into subtasks

If you don't have these, your code will be blind and likely fail on all 50 seeds.

---

## Step 3 — Write ONE complete program

Write a single Python file that handles all subtasks.

### Long-Horizon Code Rules (CRITICAL)

1. **Call `goto_home_joint_position()` BEFORE every subtask** — prevents joint limit issues.
2. **Re-call `get_observation()` after each home reset** — camera view shifts when arm moves.
3. **Never assume subtask 1 succeeded before starting subtask 2** — best-effort each independently.
4. **For "AND close it" tasks (drawer, microwave):** ALWAYS close even if prior step failed.
5. **Write for seeds 1–50** — robust SAM3 prompts, no hardcoded coordinates.

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

# ... (add more helpers from skill library as needed)

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

# --- Subtask 2 ---
goto_home_joint_position()
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
depth = obs["agentview"]["images"]["depth"]
K, T = obs["agentview"]["intrinsics"], obs["agentview"]["pose_mat"]
E = obs["robot"]["eef_pose"]

# <second subtask — best-effort even if subtask 1 failed>

for _ in range(3): get_observation()  # let sim settle
```

**Do NOT iterate or debug.** Write the best code you can, then proceed to Step 4.

---

---

## Step 4 — Save code and return

Save the code to the stable output path where the coordinator expects it:

```bash
OUTDIR="$ASPIRE_ROOT/outputs/scaling_eval/${SNAPSHOT}/one_shot"
CODE_DIR="$OUTDIR/$SUITE/$TASK"
mkdir -p "$CODE_DIR"
cp "/tmp/eval_os_${SUITE}_${TASKSHORT}.py" "$CODE_DIR/code.py"
echo "Code saved to: $CODE_DIR/code.py"
ls -la "$CODE_DIR/code.py"
```

**Verify the file exists and is non-empty, then RETURN immediately.**

Do NOT run seeds. The coordinator handles execution.

---

## What to Return

```
SNAPSHOT: <tag>
SUITE: <suite>
TASK: <task>
GPU: <N>

Code path: outputs/scaling_eval/<SNAPSHOT>/one_shot/<SUITE>/<TASK>/code.py
Code approach (1 line): <e.g. "double pick-and-place with basket drop">
Library patterns used: <which skill entries helped, if any>
```

Do NOT include the full code. Just the path and a one-line summary.
```

---
