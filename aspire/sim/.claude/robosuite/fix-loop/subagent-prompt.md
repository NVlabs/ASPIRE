---
name: robosuite/fix-loop/subagent-prompt
description: Self-contained prompt template for Robosuite task-level fix loop subagents. Copy, fill in TASK/CONFIG/GPU, and pass to Agent tool.
---

# Fix Loop Robosuite Subagent Prompt Template

When dispatching a subagent for one Robosuite task, copy the **entire** block below **verbatim**, fill in ONLY the four variables at the top (TASK, CONFIG, GPU, CAMERA_KEY), and pass as the `prompt` parameter to the Agent tool with `subagent_type: "general-purpose"`.

**⚠️ Do NOT rewrite, paraphrase, restructure, or omit any part of this template.** Every section is essential.

---

```
## Task Assignment

TASK:       <cube_lifting|cube_restack|cube_stack|nut_assembly|spill_wipe|two_arm_lift|two_arm_handover>
CONFIG:     <full path to traced yaml from coordinator task reference table>
GPU:        <3|4|5|6|7>
CAMERA_KEY: robot0_robotview

Working directory: $ASPIRE_ROOT

## ⛔ FILESYSTEM BOUNDARY — READ THIS FIRST

**You may ONLY read, write, or access files under these two directories:**
- `$ASPIRE_ROOT/` (the simulation workspace)
- `/tmp/` (scratch space) — **ONLY files WITHOUT `_tl` in the name.** Do NOT read, write, or list any `/tmp/` files with `_tl` in the name — those belong to a separate experiment running in parallel.

**Do NOT read, list, or access files anywhere else.** No other checkout, no home directories, no `/root/`, no other paths.

## Convenience Paths (substitute actual values in your commands)

TASK_DIR:  outputs/robosuite_fix_loop/$TASK
DEBUG_DIR: outputs/robosuite_fix_loop_debug

All paths below use $TASK_DIR and $DEBUG_DIR as shorthands. Replace them with literal values in your bash commands.

---

## ⛔ EVAL SET LOCKOUT — READ THIS FIRST

**Seeds 1–100 are the evaluation set. They are LOCKED.**

- You may ONLY replay seeds 101–125. NEVER replay seeds 1–100 for any reason.
- The coordinator will run seeds 1–100 manually after you deliver fix_code.py.

Violation of this rule produces invalid benchmark results.

---

## What You Are

You are a fix loop subagent for the ASPIRE Robosuite benchmark. Your job is to debug
failed robot trials for one task, write a generalizable fix program, test it on debug seeds, and report results.
You have full tool access (Bash, Read, Write, Edit, Glob, Grep).

---

## Context

ASPIRE: LLMs write Python code to control a robot arm via a perception+manipulation API.
Code runs in MuJoCo (Robosuite). No external baseline is used. First inspect one observed scene and
generate an initial task-level program. Then diagnose its failures on debug seeds 101–125 and write a
generalizable fix_code.py. The coordinator runs seeds 1–100 separately.

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
cat /tmp/fix_progress_checkpoint_${TASK}.md 2>/dev/null || echo "NO PRIOR PROGRESS"

echo "=== FIX CODE ==="
ls $TASK_DIR/fix_code.py 2>/dev/null && echo "EXISTS" || echo "MISSING"

```

**Decision tree:**
- If fix_code.py exists → task is done. Report results and return.
- If checkpoint exists AND no fix_code.py → read checkpoint, resume from where the previous run left off. Do not repeat work already done (e.g. don't re-test seeds that already have results on disk).
- If no checkpoint AND no fix_code.py → start fresh from Stage 0.

---

## Progress Checkpointing (crash recovery)

Your session may be terminated unexpectedly (e.g. rate limits). **You MUST save progress after every meaningful step** to:

```
/tmp/fix_progress_checkpoint_${TASK}.md
```

Update this file after each milestone (e.g. after diagnosing seeds, after each fix attempt, after writing fix_code.py). Use this format:

```
## Progress: $TASK
Last updated: <timestamp>

### Current Stage: <Stage 1 Step N>
### Seeds Tested So Far: <list of seeds tested and their reward>
### Fix Code Location: <path or "not yet written">
### Diagnosis Summary: <what you've learned about failure modes>
### Next Step: <what to do next>
```

---

## Stage 0: Explore Once and Generate Initial Code

```bash
mkdir -p "$TASK_DIR/attempts" outputs/working_codes
```

Inspect one observed scene using the REPL below (seeds 101–125 ONLY), and save the scene images plus
your notes to `$TASK_DIR/task_analysis.md`. Then read the skill files before writing:
- `.claude/robosuite/fix-loop/skills/grasp.md`
- `.claude/robosuite/fix-loop/skills/localize.md`
- `.claude/robosuite/fix-loop/skills/transport.md`
- `.claude/robosuite/fix-loop/skills/manipulation.md`

**The initial analysis may be wrong.** It comes from one seed. Treat inferred identity, geometry, free
space, and strategy as hypotheses; revise them when later traces disagree. Never hardcode
snapshot-specific coordinates or mask order.

Before writing the full initial code, explore seed 101 interactively in a REPL session. You have a budget of **5 code blocks** inside ONE REPL session. Use them to incrementally build and test your approach — observe the scene, try perception calls, attempt grasps, diagnose what works and what doesn't. Each code block runs in the same environment state left by the previous block (the robot and objects stay where they are).

**Interactive exploration on seed 101 (5 code blocks, 1 REPL session):**

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
  --args.config $CONFIG \
  --args.trial 101 \
  --args.interactive \
  --args.output-dir /tmp/repl_explore_${TASK} 2>/dev/null << 'REPL_EOF' | tee $TASK_DIR/repl_explore_101.txt
# Code block 0: observe scene, run perception, print what you see
import numpy as np
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
T = cam["pose_mat"]
# ... explore segmentation, point clouds, object locations ...
print("block 0 done", flush=True)

# Code block 1: try a grasp or motion approach
# ... test your strategy step by step ...
print("block 1 done", flush=True)

# Code block 2: inspect result, adjust
# ... check gripper width, re-observe, diagnose ...
print("block 2 done", flush=True)

# Code block 3: refine approach
# ... fix issues found in block 2 ...
print("block 3 done", flush=True)

# Code block 4: final verification
# ... confirm the full sequence works ...
print("block 4 done", flush=True)
REPL_EOF
```

**How to use the 5 blocks:** Plan all 5 blocks upfront in one heredoc (the REPL reads stdin in batch mode). Each `# Code block N` section runs sequentially in the same env. Use early blocks for perception and scene understanding, middle blocks for trying manipulation, and later blocks to refine. Print intermediate results so you can see what happened. After the REPL session, read the tee'd output and any keyframes/video to understand what worked.

**After the REPL exploration**, synthesize everything you learned into `$TASK_DIR/initial_code.py` — a complete standalone program. This is NOT a copy-paste of the REPL blocks (which ran incrementally in shared state). The initial code must work from a fresh env reset.

These 5 REPL blocks do NOT count toward the Stage 1 per-seed replay limit. They are a separate exploration budget.

After writing initial_code.py, run it once on every debug seed:

```bash
for trial in $(seq 101 125); do
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
    --args.config $CONFIG \
    --args.trial $trial \
    --args.replay-code "$TASK_DIR/initial_code.py" \
    --args.output-dir "$DEBUG_DIR" \
    > "$TASK_DIR/initial_seed_${trial}.log" 2>&1
done
```

---

## Stage 1: Debug Seeds 101–125 (You must never use seeds 1-100 during Stage 1)

### Step 1 — Triage the initial run

For each seed 101–125, check the reward in the replay output dir name: `_reward_1.000` = success,
`_reward_0.000` = failure. List which seeds passed and which failed. Debug only the failed seeds.

---

### Step 2 — Diagnose each failed seed (101–125)

For each failed seed, read:
  - `trace.json` — key signals:
      open_gripper → gripper_width: >0.9 confirms gripper opened successfully
      close_gripper → gripper_width: >0.06 grasped, 0.03–0.06 marginal, <0.03 air grasp
      segment_sam3_text_prompt → num_masks=0 means bad prompt
      plan_grasp → num_grasps=0 means no grasp candidates found
      select_top_down_grasp → found_grasp=false means no top-down grasp passed threshold
      solve_ik entry has "error" key (not "result") → IK failed, target out of workspace
  - `code.py` — what did the current program try?
  - `summary.txt` — stdout/stderr/reward

Parse trace.json (handles both single-arm and bimanual function names):
```python
import json
trace = json.load(open("path/to/trace.json"))
for step in trace:
    fn = step.get("function", "")
    res = step.get("result", {})
    err = step.get("error", None)
    if fn.startswith("open_gripper") or fn.startswith("close_gripper"):
        print(f"{fn}: gripper_width={res.get('gripper_width', '?')}")
    if fn == "segment_sam3_text_prompt":
        print(f"SAM3 num_masks={res.get('num_masks', 0)}, prompt={step.get('args', {}).get('text_prompt')}")
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

### Step 3 — Write and test fix code

⚠️ **ALL replays in Stage 1 must use seeds 101–125. NEVER replay seeds 1–100 during Stage 1.**

⚠️ **fix_code.py is ONE program and you MUST test it on ALL 25 debug seeds — not just the ones that failed the initial run.** You MUST run your candidate fix on every seed 101–125, including seeds where the initial program already succeeded. Do NOT assume initial-run successes will still pass with your fix code — they may not. There can be variance in runs even if the seed and code is fixed. Your Stage 1 score is the count of reward=1 across all 25 seeds when you run fix_code.py on all of them. Reporting (N_fixed_failures + N_initial_successes)/25 without running fix_code on the initial-success seeds is invalid.

⚠️ **COMPLETION GATE: Do NOT write fix_code.py unless your fix code achieves 25/25 (100%) on the debug seeds or you reach the maximum allowed 5 replay attempts per seed for all seeds.**

For code structure, read the skill files before writing:
- `.claude/robosuite/fix-loop/skills/grasp.md`
- `.claude/robosuite/fix-loop/skills/localize.md`
- `.claude/robosuite/fix-loop/skills/transport.md`

Write a fix.py and test **(seeds 101–125 ONLY — never 1–100)**:
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
  --args.config $CONFIG \
  --args.trial <seed_from_101_to_125> \
  --args.replay-code /tmp/fix_attempt_${TASK}.py \
  --args.output-dir /tmp/fix_test_${TASK}
```

Check reward in output dir name: `_reward_1.000` = success, `_reward_0.000` = failure.

**REPL for live inspection (seeds 101–125 ONLY):**
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
  --args.config $CONFIG \
  --args.trial <seed_from_101_to_125> \
  --args.interactive \
  --args.output-dir /tmp/repl_out_${TASK} 2>/dev/null << 'EOF' | tee /tmp/repl_out_${TASK}/<seed>.txt
import numpy as np
obs = get_observation()
masks = segment_sam3_text_prompt(obs["$CAMERA_KEY"]["images"]["rgb"], "<prompt>")
print(f"num_masks={len(masks)}", flush=True)
EOF
```

**Hard limit: 5 replay attempts per seed. Absolute — no exceptions.**

If a seed is blocked after 5 attempts, write BLOCKED.md:
  $TASK_DIR/attempts/trial_<N>_BLOCKED.md
  Format:
    ## Root Cause: [Physical|Perception|Algorithmic]
    ## Details: <what exactly fails and why>
    ## What Was Tried: <list of approaches>

---

### Step 4 — Synthesize task-level fix_code.py

Save to TWO locations:
  $TASK_DIR/fix_code.py                                ← task-level (gen_progress_robosuite.py looks here)
  outputs/working_codes/robosuite_${TASK}_fix.py      ← named copy

Write findings at:
  $TASK_DIR/findings.md

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

**1. Write reasoning.txt** — always, before anything else in this step:

```
$TASK_DIR/reasoning.txt
```

You must write a plain-text file explaining why you stopped. Be specific — include seed counts, success rates, and the deciding factor. Use this format:

```
## Why Stage 1 Stopped
<one of: "fix_code generalized", "hit retry limit on N seeds", "fast-path succeeded", "fix_code already existed">

## Information you MUST include
- Success rate on 25 debug seeds: <N>/25  ← fix_code.py run on ALL 25, not assumed from the initial run
- Result of the fix code on all 25 debug seeds: <list out 1 by 1 the result of fix_code.py on each of the 25 debug seeds>
- Reason for failures on debug seeds (if any): <count and one-line cause each, or "none">
- Key trace signals that informed the decision: <e.g. "SAM3 returned mask showing the cube was stacked on 25/25 seeds">

## Concerns / Caveats
<anything that might affect result validity, or "none">

## Anomalies / Unexpected Behavior
<anything unexpected or strange encountered during this run — be specific. Examples:
  - API call returned unexpected output (e.g. plan_grasp returned 0 grasps on a clearly visible object)
  - trace.json was empty, truncated, or missing for a seed
  - replay script crashed or hung unexpectedly
  - perception server returned an error or timed out
  - SAM3/Molmo/GraspNet gave results that seemed wrong (e.g. mask covered wrong object)
  - solve_ik raised an exception for a pose that should be reachable
  - output directory structure was unexpected
  - seeds that should have existed were missing from the debug run
  - conflicting or confusing instructions in your prompt or context
  - any other behavior that seemed like a bug or environment issue
If nothing anomalous occurred, write "none".>
```

**2. Update the shared progress file:**

```bash
.venv-robosuite/bin/python3 scripts/robosuite/gen_progress_robosuite.py
```

---

## What to Return (write to $TASK_DIR/findings.md)

```
TASK: <task>
CONFIG: <config path>
GPU: <N>

fix_code.py written: yes/no
  Debug seed success rate: <N>/25
  Seeds blocked: <count + one-line root cause each>

Key findings (3 bullets max, generalizable patterns only):
  - <root cause and fix, one line>
  - <SAM3/Molmo prompt that worked, if notable>
  - <anything that would change how the next similar task is approached>
```
```
