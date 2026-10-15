---
name: evosearch-iteration
description: End-to-end guide for running the Evolutionary Search-style iterative debugging loop on CaP benchmark tasks. Covers pipeline overview, perception services, reading recorded run artifacts, writing observation-only candidate programs, running evaluations, picking the best, and iterating.
---

# Evolutionary Search Debugging Pipeline

---

## 1. Prerequisites (One-Time Setup)

### Benchmark environment
Complete the target benchmark's setup guide (for LIBERO: `scripts/bootstrap_libero.sh`, then
export `LIBERO_CONFIG_PATH="$PWD/.libero"`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
`EGL_PLATFORM=device`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`).

Key rule: **always use the pinned environment** (`.venv-libero/bin/cap-harness`), never system
python or ad hoc dependency upgrades.

### Discover available tasks for a suite
Task ids must match the registry:

```bash
.venv-libero/bin/python3 -c "
from cap_harness.libero.registry import LiberoSuiteRegistry
for task in LiberoSuiteRegistry().enumerate_tasks(['libero_goal_swap']):
    print(f'  {task.task_id:2d}: {task.task_name}')
"
```

### Perception services (start once, leave running)
```bash
scripts/supervise_services.sh
# Check: 200=UP, 000=DOWN
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$p/openapi.json)"
done
```
SAM3 → GPU 0 (8114), Contact-GraspNet → GPU 1 (8115), PyRoki → GPU 2 (8116). Reserve the
remaining GPUs for sim.

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  NEW RUN                                                 │
│  0. Create timestamped run dir (once per run)            │
│  0. Run scene_snapshot.py → READ the keyframes           │
│  0. Write task_analysis.md §1-5 FROM the images          │
│  0. Validate SAM3 prompts (probe run + overlays)         │
│                                                          │
│  ITERATION N                                             │
│  1. READ task_analysis.md IN FULL                        │
│  2. Write K=8 candidate programs (public APIs only!)     │
│  3. Evaluate ALL K candidates → wait for completion      │
│  4. READ keyframes/videos from failure runs              │
│  5. Diagnose failures across ALL candidates              │
│     → For each failure mode: find .claude/skills -name   │
│       "*.md" | sort  and read matching technique files   │
│     → Apply cross-reference rule (see §8 diversity rule) │
│  6. UPDATE task_analysis.md §1-5 + append iter log       │
│  7. Write K=8 new candidates seeded by top-3 survivors   │
│  8. Repeat until solved or plateau                       │
└─────────────────────────────────────────────────────────┘
```

**Never write candidates without first reading task_analysis.md. Never write candidates without
first updating task_analysis.md after the prior eval.**

**Key rule: Do not debug while evaluation is running. Use `run_in_background: true` on the Bash
eval command — you will be notified automatically when it finishes, then read all results at
once before writing iter_N+1.**

---

## Related Skills & Companion Files

| File | When to read |
|------|-------------|
| `../../iterative-debugging/skills/` | Shared robot skills: localize, grasp, transport, manipulation |
| `docs/api-reference.md` | Authoritative public program API and the forbidden-simulator boundary |
| `docs/run-artifacts.md` | Authoritative recorded-run artifact contract |

Motion-efficiency, contact-task, and blocking guidance is condensed into §8 and §10 below.

---

## 3. Scene Snapshot (REQUIRED Before iter_00)

**Do not write a single candidate until task_analysis.md §1-5 is populated from a real scene
image.** Skipping this wastes iterations: obstacles visible in the snapshot (shelves, clutter,
tight clearances) often do not show up in traces as errors.

**Step 1 — Capture:**
```bash
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
.venv-libero/bin/cap-harness run \
  --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID --seed 51 \
  --program .claude/skills/iterative-debugging/scripts/scene_snapshot.py \
  --output-root $RUN_DIR/snapshot --init-mode seeded
```

**Step 2 — Read the recorded artifacts** (the command prints `run: <run_dir>`):
- `<run_dir>/source/program-result.json` — authoritative `task_language`, cameras, metadata
- `<run_dir>/media/keyframes/` — lossless RGB from every camera: the wide agent view (scene
  layout) and the eye-in-hand view (close-up)

Look for: every shelf/wall/raised edge, which approach directions are blocked vs clear, object
shapes and relative positions.

**Step 3 — Validate SAM3 prompts:** copy `scene_snapshot.py` into `$RUN_DIR`, fill
`PROBE_PROMPTS` from what you see, re-run, then read the probe scores in
`program-result.json` and the rendered masks in `media/overlays/`. Fix any weak/failing prompt
before writing candidates.

**Step 4 — Populate task_analysis.md §1-5 from what you see:**
- §5 (Hypotheses for iter_00) must note any approach directions that appear blocked

---

## 4. Task Analysis (Required Before iter_00)

Before writing any candidate programs, reason through the task and write your analysis to
`$RUN_DIR/task_analysis.md`. This step is **mandatory** — it prevents wasting early iterations
on geometrically wrong approaches.

Answer all five questions from the task language and scene snapshot images. If uncertain, mark
it as a hypothesis to test in iter_00.

```markdown
# Task Analysis: <task_name>

## 1. Object Shape
What is the target object? Describe its geometry:
- Shape class: cylindrical / flat / box / irregular / articulated
- Relevant dimensions (approximate): height, width, diameter
- Grasping implications: where can the gripper contact it? Any edges/rims/handles?
- Known failure modes for this shape class (e.g. cylindrical → double-close pushes it out)

## 2. Grasp/Approach Strategy
Based on the object shape:
- Grasp type: centroid / geometry-guided / rim-edge / grasp backend
- Approach direction: top-down / side / angled
- Any obstacles that may block arm's approach to object?
- Gripper orientation: yaw angle? tilt needed? angle of wrist?
- Z offset from object center: (e.g. fraction of estimated height for cylinders)
- Lift height: how high to clear obstacles during transport?

## 3. Goal Geometry
What is the placement target? Describe its geometry:
- Type: flat surface / container / slot / cradle / constrained site
- Orientation: horizontal / angled / vertical
- Tolerance: generous (±10cm) / moderate (±5cm) / tight (±1cm)
- Key geometric constraint: what must the object's pose satisfy to succeed?

## 4. Placement/Movement Strategy
Given object shape + goal geometry:
- Release type: drop from above / slide in along angle / lower carefully / press
- Any obstacles in the intended movement path?
- Gripper orientation at release: same as grasp? needs to change?
- Approach vector into the target: from directly above / from front / along surface normal
- Release height/offset: how far above target to open gripper?
- Post-release: retreat needed? risk of collision with target structure?

## 5. Hypotheses for iter_00
List any uncertain assumptions above that iter_00 candidates should test or verify.
- Blocked approach directions observed in snapshot (name them explicitly)
- Uncertain geometry (e.g. board tilt angle, container depth)
- Placement tolerance unknown — need to bracket with tight vs loose release heights?

## 6. Iteration Log
### iter_00 — YYYY-MM-DD
Open questions going in: ...

#### Results
| Candidate | Strategy | Pass rate | Status |
|-----------|----------|-----------|--------|

**Eliminated** (tested, definitively wrong — do not re-test without a new structural reason):
- ...

**Blocked, untested reconfiguration** (approach failed due to arm-body/workspace constraint,
but NOT yet tested with arm reconfiguration such as wrist rotation, a staged pose, or an
explicit cuRobo plan):
- ...

Geometry updates (cite keyframe/trace): ...
Open questions → iter_01: ...
```

---

## 5. CRITICAL: Public-API-Only Programs

Generated robot programs **MUST NOT access the simulator directly** — see
`docs/api-reference.md` for the authoritative boundary and §Forbidden in the subagent prompt.

Programs also cannot `import` anything: no NumPy, no scipy, no filesystem. All helpers are pure
Python over the public contracts, and the program's conclusion goes into a top-level `result`
variable.

### ✅ ALLOWED (public API only):
```python
context = get_task_context()                       # authoritative task language
observation = get_observation()                    # calibrated RGB-D + robot state
state = get_robot_state()                          # joints, gripper, EEF poses, base frame
found = segment_text("agentview", "bowl")          # SAM3 segmentation (SegmentationSet)
cloud = mask_to_point_cloud(mask, "agentview")     # framed point cloud
geometry = estimate_geometry(cloud)                # framed center/orientation/extents
located = localize_object("bowl")                  # composed segment+project+estimate
grasps = generate_grasps("agentview", mask)        # grasp backend candidates
grasp = select_grasp(grasps, strategy="top_down")
ik = solve_ik(pose)                                # IKResult (.ok, .joint_positions)
plan = plan_motion(pose)                           # PlanResult (.ok, .trajectory)
execute_trajectory(plan.trajectory)
move_to_pose(pose); move_to_joints(joints)
open_gripper(); close_gripper(); set_gripper(0.5)
result = {...}                                     # REQUIRED top-level result
```

### How to localize objects without sim access:
| Need | How |
|------|-----|
| Object 3D center | `localize_object(...)` → `.geometry.pose.position` |
| Object top surface Z | `max(p[2] for p in cloud.points)` |
| Object height | `max(zs) - min(zs)` over `cloud.points` |
| Object geometry | `estimate_geometry(cloud)` → `.pose`, `.extents` |
| Object still in hand after lift? | re-segment on `robot0_eye_in_hand`; check candidates non-empty |
| Placement target | segment target, project, use max Z as surface |

**Camera names (LIBERO):** `agentview` — wide scene view (use for initial localization);
`robot0_eye_in_hand` — close-up from the gripper (use to verify grasp or localize post-lift
when agentview is obscured). Other benchmarks publish different names — read them from
`get_observation().cameras`.

---

## 6. Code Conventions (Verify Before Writing Candidates)

If every candidate returns `errors=N` with 0% pass rate, check these before diagnosing
strategy:

### 1. Typed results — always check `.ok`
Every perception/grasp/IK/plan/motion call returns a typed result with `.ok` and a typed
error. A failed backend is a typed failure, never an implicit fallback.

```python
# ✅ CORRECT
found = segment_text("agentview", "wine bottle")
if not found.ok or not found.segmentations:
    raise Exception("SAM3: no masks returned")
best = found.segmentations[0]                     # already score-ordered
cloud = mask_to_point_cloud(best, "agentview")

# ❌ WRONG — indexing without checking, or passing raw arrays
cloud = mask_to_point_cloud(found[0], "agentview")
```

### 2. Contract arrays are read-only
Copy before mutating, and pass plain lists/tuples when constructing inputs:
```python
joints = [float(value) for value in ik.joint_positions]
joints[-1] += 1.5707963268
move_to_joints(joints)
```

### 3. `Pose` quaternions are wxyz, frames are explicit
```python
base_frame = get_robot_state().base_frame
pose = Pose((x, y, z), (qw, qx, qy, qz), base_frame)
```
Reuse the observed end-effector quaternion
(`get_robot_state().end_effector_poses["primary"].quaternion_wxyz`) and compose rotations with
a pure-Python quaternion product (see
`../../iterative-debugging/skills/manipulation.md`) instead of scipy.

### 4. No imports except `math`, no numpy
Quantiles by sorting, means by `sum/len`, vectors as tuples. See the templates in
`../../iterative-debugging/skills/grasp.md`.

### 5. Top-level `result` required
```python
result = {"grasp_verified": grasp_verified, "notes": "..."}   # ← recorded as the program result
```
Everything you want to inspect after the run (localized centers, probe scores, measured EEF
positions) should go in `result` — it lands in `source/program-result.json`.

---

## 7. Output Directory Structure

**Every new run gets a timestamped directory.** Create it before writing any candidates:

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$EVOSEARCH_DIR/$BENCHMARK/$SUITE/task_$TASK_ID/$RUN_ID
mkdir -p $RUN_DIR
```

All Evolutionary Search runs live under
`outputs/evosearch/<benchmark>/<suite>/task_<id>/<YYYYMMDD_HHMMSS>/`. **Do NOT use any other
top-level directory.**

Structure within a run:
```
outputs/evosearch/<benchmark>/<suite>/task_<id>/<run_id>/
├── task_analysis.md             # written before iter_00, updated each iteration
├── snapshot/                    # scene-snapshot recorded run
├── iter_00/
│   ├── candidate_A/
│   │   ├── code.py              # candidate program
│   │   ├── eval_results.json    # per-trial results after eval
│   │   ├── logs/                # seed_51.log, seed_52.log, ...
│   │   └── eval/                # cap-harness --output-root (one immutable run per seed)
│   │       └── <benchmark>/<suite>/<id>-<slug>/<seed>/<run-id>/
│   │           ├── outcome.json         # program_ok, task_success, termination_reason
│   │           ├── source/program-result.json
│   │           ├── trace/events.jsonl   # span log (calls + providers)
│   │           └── media/               # videos, keyframes, overlays — every run
│   ├── candidate_B/
│   │   └── ...
│   └── iter_summary.json        # leaderboard across all candidates
└── iter_01/
    └── ...
```

Note: unlike the old pipeline, videos, keyframes, and overlays are recorded for **every** run —
there is no separate highlight re-run stage.

---

## 8. Writing Candidate Programs

### Runtime efficiency — CRITICAL

Each `move_to_pose`/`move_to_joints` call blocks until convergence (up to `max_steps` ticks) —
the number of motion calls drives wallclock and step budget. Budget **≤10 motion calls total**
(pre-grasp 1, descent 1, lift 1, transport 2–3, placement 1–2). Anti-patterns: fine-grained
waypoint loops (20 blocking calls for one arc), long grasp-retry loops (cap at 2
configurations), multi-step descents. Prefer a 3-point arc (apex with a +0.05 clearance bump →
above-target → release), or one `plan_motion` + `execute_trajectory` for a genuinely
constrained path. If two candidates score alike, keep the one with fewer motion calls.

### Non-prehensile and contact tasks (push, slide, wipe, press)

- **Identify a collision-free approach corridor from the snapshot before writing.** If the
  standoff ray behind the push direction is blocked (shelf, clutter), descend vertically onto
  the contact point instead of sweeping laterally at contact height.
- **Arm target ≠ object final position:** the arm contacts the back rim and stops one object
  radius behind the desired final position; for short pushes compute the actual required
  displacement — never alpha-interpolate toward a distant landmark.
- **Detect physical blocking:** a motion call returning does not mean the arm moved freely.
  Signals: `wall_duration_s` 30%+ below the candidate norm, commanded pose vs measured
  end-effector pose gap >3cm (record both in `result`), object's segmentation box not moving
  between early and late spans.
- **When blocked:** change the approach direction or arm configuration, not scalar offsets.
  One cheap reconfiguration before eliminating an approach: copy `solve_ik(...).joint_positions`
  to a list, rotate the last wrist joint ±π/2, and `move_to_joints` — changes the arm-body
  profile without changing the grasp line (or request
  `MotionStrategy(trajectory_planner="curobo")` for a collision-aware plan). An approach is
  only **Eliminated** after failing *with* a reconfiguration applied.

### Example template (copy-adapt):

```python
"""
Candidate X: <one-line description>
Hypothesis: <specific failure mode this targets>
Differs from prior: <structural difference, not just param tweak>
Expected failure if wrong: <what trace would show>
Seeded from: <candidate+iter or 'novel'>
"""

def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]


base_frame = get_robot_state().base_frame
downward = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
report = {"task": get_task_context().language}

# Localize pick object (prompt-fallback helper: iterative-debugging/skills/localize.md)
picked = localize_object("<manipulated object prompt>", camera_name="agentview",
                         target_frame=base_frame)
if not picked.ok or len(picked.point_cloud.points) < 20:
    raise Exception("localization failed for pick object")
center = picked.geometry.pose.position
grasp_pose = Pose((center[0], center[1], center[2] + 0.01), downward, base_frame)

# Grasp: pre-grasp → lower → close
open_gripper()
move_to_pose(Pose((center[0], center[1], center[2] + 0.09), downward, base_frame),
             tolerance=0.02, max_steps=250)
move_to_pose(grasp_pose, tolerance=0.015, max_steps=250)
close_gripper()

# Lift
lift_z = center[2] + 0.15
move_to_pose(Pose((center[0], center[1], lift_z), downward, base_frame),
             tolerance=0.02, max_steps=250)

# Re-observe for target
target = localize_object("<goal / surface prompt>", camera_name="agentview",
                         target_frame=base_frame)
if not target.ok:
    raise Exception("localization failed for target")
target_center = target.geometry.pose.position
surface_z = quantile((point[2] for point in target.point_cloud.points), 0.9)

# Transport: arc to above-target
move_to_pose(Pose((target_center[0], target_center[1], lift_z), downward, base_frame),
             tolerance=0.02, max_steps=300)

# Place
move_to_pose(Pose((target_center[0], target_center[1], surface_z + 0.03), downward, base_frame),
             tolerance=0.02, max_steps=250)
open_gripper()

result = report
```

### K=8 diversity rule

Each candidate must test a distinct hypothesis — no two should fail at the same stage for the
same reason. Before writing, ask: *would I learn something different from each one?*

- Do not re-test anything listed as **Eliminated** in task_analysis.md §6
- Seed from top-3 performers of any prior iterations, not just the current winner
- At least one candidate must be structurally different from all prior iterations

**Cross-reference rule — apply when a new technique is discovered:**

When you discover a technique mid-run (arm reconfiguration like a wrist rotation, an explicit
cuRobo plan, a new contact strategy, etc.), immediately ask:

> "Which approaches listed under **Blocked, untested reconfiguration** in task_analysis.md
> should be retried with this technique?"

A blocked approach is only truly eliminated after it fails **with** the reconfiguration
applied. Until then, it remains a live hypothesis.

---

## 9. Running the Evaluator

```bash
# Full iteration eval (15 fixed development seeds)
python3 .claude/skills/evosearch/scripts/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_NN \
    --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID \
    --trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 \
    --sim-gpus $GPU --parallel-per-gpu 2 \
    --cap-harness .venv-libero/bin/cap-harness \
    2>&1 | tee $RUN_DIR/iter_NN/eval.log

# Deep re-check of the top candidates only
python3 .claude/skills/evosearch/scripts/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_NN \
    --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID \
    --trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 \
    --candidates candidate_A candidate_B candidate_C \
    --sim-gpus $GPU --parallel-per-gpu 2 \
    --cap-harness .venv-libero/bin/cap-harness
```

Run with `run_in_background=True` and wait for the completion notification. The evaluator skips
a seed only when an existing finalized run matches the candidate's current code and settings
(program hash + init mode + capture), so re-running the same command resumes an interrupted
eval, while edited candidates or changed flags always re-run.

Use the same fixed `--trial-seeds` across all iterations for fair cross-iteration comparison.
Every episode records its own video, keyframes, and overlays — no post-processing stage.

---

## 10. Reading Results — After All Evaluations Complete

**Only start reading/debugging after the eval process exits.**

### Quick analysis with the dedicated script

```bash
# Full per-candidate breakdown (failed spans, empty prompts, motion counts, wall-time anomalies)
python3 .claude/skills/evosearch/scripts/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_NN

# Single candidate only
python3 .claude/skills/evosearch/scripts/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_NN --candidate candidate_F

# Leaderboard only (no per-signal stats)
python3 .claude/skills/evosearch/scripts/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_NN --summary-only
```

### Manual inspection

**iter_summary.json schema** — each entry in `candidates` has: `candidate`, `code_path`,
`trials`, `pass_count`, `pass_rate`, `mean_reward`, `errors`, `trial_results`. Top-level keys:
`best_candidate`, `best_pass_rate`, `trial_seeds`.

### Key signals in the recorded artifacts:

| Signal | Where | What it means |
|--------|-------|--------------|
| `program_ok=false` | `outcome.json` | Program crashed or was rejected — read `error` and fix API usage first |
| `task_success=false`, `termination_reason=program_completed` | `outcome.json` | Ran clean but the goal predicate failed — geometry/placement issue |
| `termination_reason=step_limit` | `outcome.json` | Too many motion ticks — cut waypoints (§8 motion budget) |
| failed `solve_ik`/`plan_motion` span | `trace/events.jsonl` (`ok:false`) | Target out of workspace or planner failure — stage the pose or switch backend |
| `segment_text` with zero segmentations | `trace/calls/*/output.json` | Bad prompt — revise from overlays |
| gripper closes, object never rises | keyframes / video | Grasp missed — verify with post-lift re-localization |
| commanded pose ≠ measured EEF | program `result` / `episode/steps.jsonl` | Physical blocking — change approach or reconfigure the arm (§8) |
| `wall_duration_s` 30%+ below group norm | `outcome.json` | Arm hit an obstacle early |

### Common failure modes:

| Failure | Signature | Fix strategy |
|---------|----------|--------------|
| Grasp miss | Object never rises in video; post-lift verify fails | Centroid offset; check approach Z |
| Drop too high | Release keyframe far above surface | Derive release Z from projected surface + 3cm |
| Wrong object | Overlay covers wrong region | Better text prompt; geometry filtering |
| IK failure | Failed `solve_ik` span | Target out of workspace; adjust or stage the pose |
| Code error | `program_error` termination | Read `outcome.json` `error`; fix API usage |
| Transport drop | Grasp OK, nothing at target | Retain gripper command; smoother arc; wrist-cam verification |

---

## 11. Visual Keyframe Debugging (only when necessary)

Every recorded run keeps its own media:

```
<run_dir>/media/
    videos/<camera>.mp4              # full episode per camera at 20 Hz
    keyframes/<NNNNNN>-<label>/      # lossless RGB before/after each motion call
    overlays/                        # top-1 SAM mask rendered over RGB
```

If you are unsure about a failure case, read from the failed seed's run:
- the first keyframe — initial scene, object positions, obstacles
- a mid-execution keyframe — is the arm blocked, colliding, or missing the object?
- the final keyframe / video — where did the object end up?

**Key signals from images:**
- Object not moving despite arm motion → physical obstacle (change approach direction, not
  parameters)
- Arm contorted near a wall/shelf → blocked approach — mark in §5
- Object moved but wrong direction → push/grasp geometry off-axis

Update task_analysis.md §1-5 with anything new you see, then append the iter log entry to §6.

---

## 12. Post-Eval Replay

Re-run a single seed for deeper inspection — pick seeds that answer a specific question, not
all seeds. Every run is recorded, so a replay is just another `cap-harness run`:

```bash
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
.venv-libero/bin/cap-harness run \
  --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID --seed <N> \
  --program $RUN_DIR/iter_NN/candidate_F/code.py \
  --output-root $RUN_DIR/iter_NN/candidate_F/debug --init-mode seeded
```

Read the printed run dir's keyframes (JPEG-readable with the Read tool), trace, and
`program-result.json`. After viewing, update the relevant section of `task_analysis.md`.

---

## 13. Probing (No REPL)

There is no interactive REPL — probe with short observation-only programs instead. Write a
program that collects what you need into `result` (SAM3 candidates and scores, projected
centers, measured EEF poses after a move), run it with `cap-harness run` on the seed in
question, and read `source/program-result.json` plus `media/overlays/`. A reusable starting
point is `.claude/skills/iterative-debugging/scripts/scene_snapshot.py` (copy and edit
`PROBE_PROMPTS`).

---

## 14. Iteration Loop Checklist

**Before iter_00 (once per run):**
- [ ] Run the scene snapshot → **read the agentview and eye-in-hand keyframes**
- [ ] Populate `task_analysis.md` §1-5 from images — §5 must list every uncertain assumption
      and blocked approach direction
- [ ] SAM3 prompts validated (probe scores > 0.5, overlays cover the intended objects)

**Per iteration:**
- [ ] Read `task_analysis.md` in full (geometry + eliminated hypotheses + open questions)
- [ ] Write K=8 candidates, each testing a distinct hypothesis
- [ ] Run full eval (background), wait for completion
- [ ] Read `iter_summary.json` leaderboard
- [ ] Read keyframes for failing candidates: first frame, an overlay, mid-execution, final frame
- [ ] For any candidate with >0% success, also read its success keyframes
- [ ] Update `task_analysis.md` §1-5 if visual inspection reveals new geometry or obstacles
- [ ] Append `### iter_NN` entry to `task_analysis.md` §6
- [ ] Write next K=8 candidates seeded from top-3

### Decision rules:
| Outcome | Action |
|---------|--------|
| > 5pp improvement | Keep direction, refine further |
| Flat / no improvement | Different hypothesis — re-read traces |
| Regression (worse) | New bug introduced — diff vs previous winner |
| Consistent program errors | Fix API usage before strategy |
| All candidates 0% | Visual inspection + probe programs before writing more |
| Plateau (< 3pp for 2+ iters) | Check §5 for untried directions; force structural variety |
