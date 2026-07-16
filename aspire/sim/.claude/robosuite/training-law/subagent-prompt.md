---
name: robosuite/training-law/subagent-prompt
description: Training-law variant of the Robosuite fix loop subagent prompt. Adds structured iteration loop, code versioning, result tracking, and 5x consecutive 25/25 stop condition. Copy, fill in TASK/CONFIG/GPU, and pass to Agent tool.
---

# Fix Loop Robosuite Subagent Prompt Template

When dispatching a subagent for one Robosuite task, you MUST copy the **entire** block below **verbatim**, fill in ONLY the four variables at the top (TASK, CONFIG, GPU, CAMERA_KEY), and pass as the `prompt` parameter to the Agent tool with `subagent_type: "general-purpose"`.

**⚠️ Do NOT rewrite, paraphrase, restructure, or omit any part of this template.** Every section is essential.

---

```
## Task Assignment

TASK:       <cube_lifting|cube_restack|cube_stack|nut_assembly|spill_wipe|two_arm_lift|two_arm_handover>
CONFIG:     <full path to traced yaml from coordinator task reference table>
GPU:        <3|4|5|6|7>
CAMERA_KEY: robot0_robotview

Working directory: $ASPIRE_ROOT

---

## ⛔ EVAL SET LOCKOUT — READ THIS FIRST

**Seeds 1–100 are the evaluation set. They are LOCKED.**

- You may ONLY replay seeds 101–125. NEVER replay seeds 1–100 for any reason.
- The coordinator will run seeds 1–100 manually after you deliver fix_code.py.

Violation of this rule produces invalid benchmark results.

---

## ⛔ FILESYSTEM BOUNDARY

**You may ONLY read, write, or access files under these two directories:**
- `$ASPIRE_ROOT/` (the simulation workspace)
- `/tmp/` (scratch space) — **ONLY files with `_tl` in the name.** Do NOT read, write, or list any `/tmp/` files without `_tl` in the name — those belong to a separate experiment running in parallel.

**Do NOT read, list, or access files anywhere else.** No other checkout, no home directories, no `/root/`, no other paths.

## Convenience Paths (substitute actual values in your commands)

BASELINE_DIR: outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel
CONFIG_STEM:  (the filename of CONFIG without the .yaml extension)

All paths below use $BASELINE_DIR and $CONFIG_STEM as shorthands. Replace them with literal values in your bash commands.

---

## ⛔ ABSOLUTE STOP CONDITION

**You MUST NOT exit, return, promote fix_code.py, or declare the task done for ANY reason other than achieving 5 consecutive 25/25 runs on all debug seeds (101–125).**

- "Perception is too stochastic" is NOT a valid reason to stop.
- "I've tried many approaches" is NOT a valid reason to stop.
- "All modifications cause regressions" is NOT a valid reason to stop.
- There is NO escape hatch. Keep trying fundamentally new approaches.
- If you completely run out of ideas, think more carefully about any incorrect assumptions you might have previously made.

**Violation of this rule means the task is incomplete and you will be redispatched.**

---

## ⛔ MANDATORY ITERATION RULES

**These rules are as important as the stop condition. Violating them means the run is INVALID.**

**The coordinator uses `code_versions/` and `iter_*_result.json` files for evaluation. If these are missing or incorrectly named, the entire run is wasted — your work cannot be scored.**

1. **Every iteration uses `scripts/robosuite/run_iteration.sh`:** This script handles code versioning, seed runs, result.json, stop condition, and checkpoint — all automatically. You MUST use it for every iteration. Do NOT run seeds manually or write result.json manually.
2. **Iteration structure:** 3a (write code) → 3b (run `run_iteration.sh`) → 3c (diagnose failures). No skipping, no reordering.
3. **Respect exit codes:** Exit 0 = CONTINUE, Exit 1 = STOP (go to Step 4), Exit 2 = SMOKE FAILED (fix code, re-run same ITER).
4. **Increment ITER every iteration.** N starts at 1 and goes up by 1 each time. Never reuse an ITER number — **except** on smoke failure (exit 2), where you fix the code and re-run with the same ITER.
5. **Output dir:** The script uses `/tmp/fix_test_${TASK}_tl_iter${ITER}`. Logs at `/tmp/fix_test_${TASK}_tl_iter${ITER}_logs/`.

**Self-check:** After every iteration, `code_versions/` should contain: `iter_{N}_*.py` AND `iter_{N}_result.json` for EVERY iteration 1 through N. If any are missing, something went wrong — investigate before continuing.

---

## Context

You are a fix loop subagent. ASPIRE: LLMs write Python code to control a robot arm via a perception+manipulation API.
Code runs in MuJoCo (Robosuite). The baseline already ran seeds 101–125 for every task —
most failed. Your job: diagnose failures, write a generalizable fix_code.py, and test it on debug seeds. The coordinator runs seeds 1–100 separately.

**FORBIDDEN APIs** (use any of these and results are invalid):
  sim.data.body_xpos, sim.data.get_site_xpos, sim.data.set_joint_qpos,
  sim.model.*, sim.data.qpos, sim.forward(), env._step_once(),
  reading .xml/.urdf asset files for geometry

**ALLOWED APIs** (the only tools available to robot code):
  get_observation()                              → RGB, depth, intrinsics, pose_mat, robot state
  segment_sam3_text_prompt(rgb, text)            → SAM3 masks [{mask, box, score, label}]
  segment_sam3_point_prompt(rgb, point_coords)   → SAM3 masks by point [{mask, score}]
  mask_to_world_points(mask, depth, K, T)        → (N,3) world-frame point cloud
  depth_to_point_cloud(depth, intrinsics)        → (H,W,3) camera-frame point cloud
  pixel_to_world_point(u, v, z, intrinsics, T)  → (3,) world point from single pixel
  plan_grasp(depth, intrinsics, segmentation)    → (grasp_poses, grasp_scores) in camera frame
  select_top_down_grasp(poses, scores, cam_to_world) → (best_grasp_4x4_world, score) or (None, -inf)
  get_oriented_bounding_box_from_3d_points(pts)  → {center, extent, R}
  decompose_transform(T)                         → (position (3,), quaternion_wxyz (4,))
  rotation_matrix_to_quaternion(R)               → quaternion wxyz (4,)
  transform_points(points, T)                    → apply 4×4 to (N,3) or (H,W,3) array
  interpolate_segment(p1, p2, step=0.03)         → list of waypoints along a line segment
  normalize_vector(v)                            → unit vector (3,)
  solve_ik(position, quaternion_wxyz)            → joint angles (7,)
  move_to_joints(joints)                         → blocking motor control
  open_gripper() / close_gripper()
  point_prompt_molmo(image, text)                → {text: (pixel_x, pixel_y)} or (None, None)
  numpy, scipy

**Bimanual tasks** (two_arm_lift, two_arm_handover) replace the single-arm IK/motion functions:
  solve_ik_arm0(position, quaternion_wxyz)       → joint angles (7,)
  solve_ik_arm1(position, quaternion_wxyz)       → joint angles (7,); input in robot0 base frame
  move_to_joints_arm0(joints)                    → blocking motor control for arm 0
  move_to_joints_arm1(joints)                    → blocking motor control for arm 1
  move_to_joints_both(joints0, joints1)          → simultaneous blocking control for both arms
  open_gripper_arm0() / close_gripper_arm0()
  open_gripper_arm1() / close_gripper_arm1()
  (All perception and geometry functions are the same as single-arm)
  (solve_ik / move_to_joints / open_gripper / close_gripper without suffix do NOT exist for bimanual)

**Camera key:** All 7 tasks use `robot0_robotview`:
    obs["robot0_robotview"]["images"]["rgb"], obs["robot0_robotview"]["images"]["depth"],
    obs["robot0_robotview"]["intrinsics"], obs["robot0_robotview"]["pose_mat"]

**spill_wipe note:** Uses a sponge tool attachment with tcp_offset=[0,0,-0.0158] (shorter than
  standard [0,0,-0.107]). The API class is otherwise identical to other single-arm tasks.

**venv:** always use `.venv-robosuite/bin/python3` for Robosuite replay/eval — never system python

**Perception servers must be running** before any replay:
  for p in 8114 8115 8116 8122; do echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"; done
  (404 = UP, 000 = DOWN; 8114=SAM3, 8115=GraspNet, 8116=PyRoKi, 8122=Molmo)

---

## ⚠️ FIRST STEP: Check for prior progress + state

**This is the FIRST thing you must do — before any other work.** A previous subagent may have been terminated mid-run. Check the checkpoint file and disk state:

```bash
echo "=== CHECKPOINT ==="
cat /tmp/fix_progress_checkpoint_${TASK}_tl.md 2>/dev/null || echo "NO PRIOR PROGRESS"

echo "=== FIX CODE ==="
ls $BASELINE_DIR/$CONFIG_STEM/fix_code.py 2>/dev/null && echo "EXISTS" || echo "MISSING"

```

**Decision tree:**
- If fix_code.py exists → task is done. Report results and return.
- If checkpoint exists AND no fix_code.py → read checkpoint, resume Stage 1 from where the previous run left off. Do not repeat work already done (e.g. don't re-test seeds that already have results on disk).
- If no checkpoint AND no fix_code.py → start fresh from Step 1.

---

## Stage 1: Debug Seeds 101–125 (You must never use seeds 1-100 during Stage 1)

### Step 1 — Fast path: try successful baseline code first

Baseline dir: $BASELINE_DIR/$CONFIG_STEM/

**Check if any seed already succeeded:**
```bash
find $BASELINE_DIR/$CONFIG_STEM -maxdepth 1 -type d -name "*reward_1.000*"
```

If successful trials exist:
1. Read the successful `code.py`
2. Test it on 2–3 failed seeds **(seeds 101–125 ONLY — never 1–100)**:
   ```bash
   MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
   .venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
     --args.config $CONFIG \
     --args.trial <failed_seed_from_101_to_125> \
     --args.replay-code /tmp/baseline_success_${TASK}_tl.py \
     --args.output-dir /tmp/fast_path_test_${TASK}_tl
   ```
3. If reward=1 on those 2-3 seeds → use this code as your starting point for Step 3's iterative loop (write it to `/tmp/fix_code_${TASK}_tl.py` and run `scripts/robosuite/run_iteration.sh --iter 1 ...`). It still must pass the 5-consecutive-25/25 gate before promotion.
4. If any failures on those 2-3 seeds → fall through to debug loop, using baseline code as a reference.

---

### Step 2 — Diagnose each failed seed (101–125)

For each failed seed, read `trace.json`, `code.py`, and `summary.txt`.

**Trace parsing script** (reuse in Step 2 and Step 3c):
```python
import json
trace = json.load(open("path/to/trace.json"))
for step in trace:
    fn = step.get("function", "")
    res = step.get("result", {})
    err = step.get("error", None)
    # gripper_width: >0.9=opened, >0.06=grasped, 0.03-0.06=marginal, <0.03=air grasp
    if fn.startswith("open_gripper") or fn.startswith("close_gripper"):
        print(f"{fn}: gripper_width={res.get('gripper_width', '?')}")
    # num_masks=0 means bad prompt
    if fn == "segment_sam3_text_prompt":
        print(f"SAM3 num_masks={res.get('num_masks', 0)}, prompt={step.get('args', {}).get('text_prompt')}")
    # "error" key (not "result") means IK failed / target out of workspace
    if fn.startswith("solve_ik"):
        if err:
            print(f"{fn}: FAILED — {err}")
        else:
            print(f"{fn}: joints={res.get('joints', '?')}")
    if fn.startswith("move_to_joints"):
        print(f"{fn}: completed={res.get('completed', '?')}")
    if fn == "plan_grasp":
        print(f"plan_grasp: num_grasps={res.get('num_grasps', 0)}")
    if fn == "select_top_down_grasp":
        print(f"select_top_down_grasp: found={res.get('found_grasp', '?')}")
```

---

### Step 3 — Iterative debug loop (until 5 consecutive 25/25)

⚠️ **ALL replays must use seeds 101–125. NEVER replay seeds 1–100.**

Read the skill files before writing code: `.claude/robosuite/training-law/skills/grasp.md`, `.claude/robosuite/training-law/skills/localize.md`, `.claude/robosuite/training-law/skills/transport.md`

**Stop condition reminder:** 5 consecutive 25/25 iterations → promote. One 25/25 is NOT enough (SAM3/Molmo stochasticity). No iteration limit, no early exit. See ABSOLUTE STOP CONDITION and MANDATORY ITERATION RULES above.

#### 3a — Write/revise code

Write your fix code to a temp file. This is the ONLY step you do manually — everything else is automated.

```bash
ITER=1   # increment each iteration: 1, 2, 3, ...
CODE="/tmp/fix_code_${TASK}_tl.py"
```

Write your code to `$CODE`.

#### 3b — Run iteration (smoke test + 25 seeds + result.json + checkpoint — ALL AUTOMATIC)

**One command does everything:** copies code to `code_versions/`, smoke tests, runs all 25 seeds in parallel, writes `iter_{N}_result.json`, checks stop condition, updates checkpoint.

```bash
scripts/robosuite/run_iteration.sh \
    --code "$CODE" \
    --config $CONFIG \
    --task $TASK \
    --iter $ITER \
    --gpu $GPU \
    --seeds 101-125 \
    --workers 5
```

**Read the exit code to decide what to do next:**

| Exit code | Meaning | Action |
|---|---|---|
| **0** | CONTINUE — need more iterations | Proceed to 3c (diagnose), then 3a with ITER+1 |
| **1** | STOP — 5 consecutive 25/25 achieved | Go to Step 4 (promote) |
| **2** | SMOKE FAILED — all 3 smoke seeds crashed | Read the crash log printed in output, fix the code, re-run 3b with SAME ITER |

**You do NOT need to:**
- Manually copy code to `code_versions/` (the script does it)
- Manually write `result.json` (the script does it)
- Manually update the checkpoint file (the script does it)
- Manually check the stop condition (the script does it)

The script prints a summary line at the end: `Iter 3: 22/25 (88%) | streak 0/5 | CONTINUE`

#### 3c — Diagnose failures, then next iteration

**If pass rate was 25/25 but streak < 5:** Re-run the same code with incremented ITER. Go back to 3a.

**If pass rate was < 25/25:** Diagnose failures using traces, keyframes, and REPL, then revise code.

**3c-i. Find failed seeds**
```bash
find /tmp/fix_test_${TASK}_tl_iter${ITER} -type d -name "*reward_0*" | head -5
```

**3c-ii. Review traces and keyframes**

For each failed trial, examine:
- `trace.json` — use the trace parsing script from Step 2
- `keyframes/video_frame_*.jpg` — **view these images** to visually diagnose what went wrong
- `summary.txt` — stdout/stderr
- Per-seed logs: `/tmp/fix_test_${TASK}_tl_iter${ITER}_logs/seed${SEED}.log`

**3c-iii. REPL debug on failed seeds**

You MUST run **at most 5** failed seeds **at most 2 times each** in interactive REPL mode before writing revised code. The REPL lets you execute API calls live — test SAM3 prompts, check point clouds, inspect gripper state, verify IK solutions — so you can confirm your hypothesis before committing to a code revision. Traces and keyframes alone often miss the root cause.

Pipe commands via heredoc (seeds 101–125 ONLY):
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
    --args.config $CONFIG \
    --args.trial <failed_seed_101_to_125> \
    --args.interactive \
    --args.output-dir /tmp/debug_repl_${TASK}_tl 2>/dev/null << 'PYEOF'
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
obs = get_observation()
rgb = obs["robot0_robotview"]["images"]["rgb"]
plt.imsave("/tmp/debug_repl_tl.png", rgb)
masks = segment_sam3_text_prompt(rgb, "<your_prompt>")
print(f"SAM3: {len(masks)} masks" + (f", top={masks[0]['score']:.3f}" if masks else ""), flush=True)
PYEOF
```

Adapt the REPL code for your specific task. The key is to **test your hypothesis live** before rewriting code.

**3c-iv. Identify top failure modes, revise code**

Synthesize findings from traces, keyframes, and REPL. Write revised code. **Increment ITER and return to 3a.**

---

### Step 4 — Promote best code to fix_code.py

Read all `iter_*_result.json` files in `$BASELINE_DIR/$CONFIG_STEM/code_versions/`. Find the iteration with the **highest pass rate** (ties broken by lowest iteration number). Copy that iteration's code file to:

  $BASELINE_DIR/$CONFIG_STEM/fix_code.py              ← task-level (gen_progress_robosuite.py looks here)
  outputs/working_codes/robosuite_${TASK}_fix.py      ← named copy

Write findings at `$BASELINE_DIR/$CONFIG_STEM/findings.md`:

```
## Task: $TASK

### Root Cause(s)
- <concise description of each failure mode found>

### What Fixed It
- <what change resolved each root cause>

### SAM3/Molmo Prompts That Worked
| Model | Object | Prompts (priority order) | Notes |
|---|---|---|---|
| <SAM 3 or Molmo> | <object> | "<prompt1>", "<prompt2>" | <any caveats> |

### Generalizable Patterns
- <patterns likely to apply to OTHER tasks>

### Task-Specific Quirks
- <things particular to this task>

### Debug Seed Success Rate
<N>/25
```

---

## Final Step Before Returning

**1. Write reasoning.txt** at `$BASELINE_DIR/$CONFIG_STEM/reasoning.txt`:

```
## Why Stage 1 Stopped
<one of: "5 consecutive 25/25 achieved", "fix_code already existed">

## Information you MUST include
- Success rate on 25 debug seeds: <N>/25  ← fix_code.py run on ALL 25, not assumed from baseline
- Result of the fix code on all 25 debug seeds: <list out 1 by 1 the result of fix_code.py on each of the 25 debug seeds>
- Reason for failures on debug seeds (if any): <count and one-line cause each, or "none">
- Key trace signals that informed the decision: <e.g. "SAM3 returned mask showing the cube was stacked on 25/25 seeds">

## Concerns / Caveats
<anything that might affect result validity, or "none">

## Anomalies / Unexpected Behavior
<anything unexpected or strange encountered during this run — be specific. Examples:
  - API call returned unexpected output
  - trace.json was empty, truncated, or missing for a seed
  - replay script crashed or hung unexpectedly
  - perception server returned an error or timed out
  - SAM3/Molmo/GraspNet gave results that seemed wrong
  - solve_ik raised an exception for a pose that should be reachable
  - any other behavior that seemed like a bug or environment issue
If nothing anomalous occurred, write "none".>
```

**2. Update the shared progress file:**

```bash
.venv-robosuite/bin/python3 scripts/robosuite/gen_progress_robosuite.py
```
```
