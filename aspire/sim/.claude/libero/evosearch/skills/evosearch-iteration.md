---
name: libero-evosearch-iteration
description: End-to-end guide for Claude Code to run the Evolutionary Search-style iterative debugging loop on LIBERO-Pro tasks. Covers pipeline overview, starting persistent perception servers, reading failure traces, writing observation-only candidate programs, running evaluations, picking the best, and iterating.
---

# Evolutionary Search + Claude Code Debugging Pipeline

---

## 1. Prerequisites (One-Time Setup)

### LIBERO config (required by evosearch_eval.py)
`evosearch_eval.py` calls `_find_task_id()` which imports `libero.benchmark`, which reads `~/.libero/config.yaml`. If missing, it hangs on an interactive prompt.

```bash
# Check if it exists
cat ~/.libero/config.yaml 2>/dev/null || echo "MISSING — create it!"

# Create it (adjust REPO_ROOT to your path)
mkdir -p ~/.libero
REPO_ROOT=$REPO_ROOT
cat > ~/.libero/config.yaml << EOF
benchmark_root: ${REPO_ROOT}/cap/third_party/LIBERO-PRO/libero/libero
bddl_files: ${REPO_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/bddl_files
init_states: ${REPO_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/init_files
datasets: ${REPO_ROOT}/cap/third_party/LIBERO-PRO/libero/datasets
assets: ${REPO_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/assets
EOF
```

### Discover available tasks for a suite
Task names must be exact strings (as returned by LIBERO benchmark):

```bash
.venv-libero/bin/python3 -c "
from libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_goal_swap']()
for i in range(suite.n_tasks):
    print(f'  {i:2d}: {suite.get_task(i).name}')
"
```

**Available suites** (partial list): `libero_goal`, `libero_goal_swap`, `libero_goal_task`, `libero_10`, `libero_10_swap`, `libero_90`

### Venv and dependencies
Key rule: **always use `.venv-libero/bin/python3`**, never `uv run` or system python3.

### Perception servers (start once, leave running)
```bash
bash scripts/common/start_perception_servers.sh
# Check: 404=UP, 000=DOWN
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$p/health)"
done
```
SAM3 → GPU 0 (8114), GraspNet → GPU 1 (8115), PyRoKi → CPU (8116). Reserve GPUs 4-7 for sim.

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  NEW RUN                                                 │
│  0. Create timestamped run dir (once per run)            │
│  0. Run scene_snapshot.py → READ both images             │
│  0. Write task_analysis.md §1-5 FROM the images          │
│  0. Validate SAM3 prompts                                │
│                                                          │
│  ITERATION N                                             │
│  1. READ task_analysis.md IN FULL                        │
│  2. Write K=8 candidate programs (observation-only!)     │
│  3. Evaluate ALL K candidates → wait for completion      │
│  4. Extract + READ keyframes from failure videos         │
│  5. Diagnose failures across ALL candidates              │
│     → For each failure mode: find .claude/libero -name   │
│       "*.md" | sort  and read matching technique files   │
│     → Apply cross-reference rule (see §8 diversity rule) │
│  6. UPDATE task_analysis.md §1-5 + append iter log       │
│  7. Write K=8 new candidates seeded by top-3 survivors   │
│  8. Repeat until solved or plateau                       │
└─────────────────────────────────────────────────────────┘
```

**Never write candidates without first reading task_analysis.md. Never write candidates without first updating task_analysis.md after the prior eval.**

**Key rule: Do not debug while evaluation is running. Use `run_in_background: true` on the Bash eval command — Claude Code will be notified automatically when it finishes, then read all results at once before writing iter_N+1.**

---

## Related Skills & Companion Files

| File | When to read |
|------|-------------|
| `motion-efficiency.md` | Before writing any candidate — ≤10 `move_to_joints` calls budget, anti-patterns, preferred arc patterns |
| `push-contact-tasks.md` | Any non-prehensile task (push, slide, press) — approach corridors, vertical descent, arm-target geometry, blocking detection |
| `wrist-rotation-blocking.md` | ARM BLOCKED detected — try `j[6] += π/2` before declaring an approach infeasible |
| `../CLAUDE.md` | Shared LIBERO benchmark, API, replay, and preflight reference |
| `CLAUDE.md` "FORBIDDEN APIs" | Authoritative list of simulator APIs that must never appear in candidate code |

---

## 3. Scene Snapshot (REQUIRED Before iter_00)

**Do not write a single candidate until task_analysis.md §1-5 is populated from a real scene image.** Skipping this wastes iterations: obstacles visible in the snapshot (shelves, clutter, tight clearances) often do not show up in traces as errors.

**Step 1 — Capture (pass `SNAPSHOT_DIR` to save images to run directory):**
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
SNAPSHOT_DIR=$RUN_DIR \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --suite $SUITE --task "$TASK" --trial 1 \
  --replay-code scripts/libero/scene_snapshot.py \
  --config env_configs/libero/franka_libero_traced.yaml \
  2>&1 | grep -E "(snapshot|SAM3|\[OK\]|\[WEAK\]|\[FAIL\]|Saved)"
```

**Step 2 — Read both images with the Read tool:**
```
$RUN_DIR/scene_snapshot.jpg        ← agentview (main scene layout)
$RUN_DIR/scene_snapshot_wrist.jpg  ← wrist cam (close-up of objects)
```
(Falls back to `/tmp/scene_snapshot.jpg` if `SNAPSHOT_DIR` not set.)

Look for: every shelf/wall/raised edge, which approach directions are blocked vs clear, object shapes and relative positions.

**Step 3 — Populate task_analysis.md §1-5 from what you see:**
- §5 (Hypotheses for iter_00) must note any approach directions that appear blocked
- Fix any `[WEAK]`/`[FAIL]` SAM3 prompts before writing any candidates

---

## 4. Task Analysis (Required Before iter_00)

Before writing any candidate programs, reason through the task and write your analysis to `$RUN_DIR/task_analysis.md`. This step is **mandatory** — it prevents wasting early iterations on geometrically wrong approaches.

Answer all five questions from the task name and scene snapshot images. If uncertain, mark it as a hypothesis to test in iter_00.

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
- Grasp type: centroid / OBB-guided / rim-edge / GraspNet
- Approach direction: top-down / side / angled
- Any obstacles that may block arm's approach to object?
- Gripper orientation: yaw angle? tilt needed? angle of wrist?
- Z offset from object center: (e.g. 80% of OBB height for cylinders)
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

**Blocked, untested reconfiguration** (approach failed due to arm-body/workspace constraint, but NOT yet tested with arm reconfiguration such as j[6] rotation, different IK seed, or wrist yaw change):
- ...

Geometry updates (cite keyframe/trace): ...
Open questions → iter_01: ...
```

---

## 5. CRITICAL: Observation-Only Programs

Generated robot programs **MUST NOT access the MuJoCo simulator directly** — see CLAUDE.md "FORBIDDEN APIs" for the authoritative list.

### ✅ ALLOWED (observation API only):
```python
obs = get_observation()                                          # RGB + depth from cameras
masks = segment_sam3_text_prompt(rgb, "bowl")                    # SAM3 segmentation
pt = point_prompt_molmo(rgb, "bowl")                             # Molmo point grounding
grasps, scores = plan_grasp(depth, intrinsics, mask)             # GraspNet grasp poses
pts = mask_to_world_points(mask, depth, intrinsics, extrinsics)  # 3D from depth+mask
obb = get_oriented_bounding_box_from_3d_points(pts)              # OBB geometry
pos, quat = decompose_transform(T)                               # 4x4 → pos + quat
joints = solve_ik(pos, quat)                                     # PyRoKi IK
move_to_joints(joints)
open_gripper(); close_gripper()
import numpy as np  # must import explicitly
```

**Depth shape:** `depth` from `get_observation()` may be `(H, W, 1)` or `(H, W)`. Always normalize:
```python
depth_img = depth[:,:,0] if len(depth.shape) == 3 else depth
```

**Camera keys:** `obs` has two cameras:
- `obs["agentview"]` — wide scene view (use for initial localization)
- `obs["robot0_eye_in_hand"]` — close-up from the gripper (use to verify grasp, confirm object in hand, or localize target post-lift when agentview is obscured)

**CRITICAL**: The wrist camera key is `"robot0_eye_in_hand"` — NOT `"wristview"`. Using `"wristview"` raises `KeyError`.

Both have the same sub-keys: `images/rgb`, `images/depth`, `intrinsics`, `pose_mat`.

### How to localize objects without sim access:
| Need | How |
|------|-----|
| Object 3D center | `mask_to_world_points()` → `pts.mean(axis=0)` |
| Object top surface Z | `pts[:,2].max()` |
| Object height | `pts[:,2].max() - pts[:,2].min()` |
| Object geometry | `get_oriented_bounding_box_from_3d_points(pts)` |
| Object still in hand after lift? | `obs["robot0_eye_in_hand"]` → re-segment; check mask is non-empty |
| Placement target | Segment target, get point cloud, use max Z as surface |

---

## 6. Code Conventions (Verify Before Writing Candidates)

If every candidate returns `errors=N` with 0% pass rate, check these before diagnosing strategy:

### 1. SAM3 mask dict — extract `["mask"]`, guard for empty
`segment_sam3_text_prompt` returns a **list of dicts** sorted by score (descending).

```python
# ✅ CORRECT
masks = segment_sam3_text_prompt(rgb, "wine bottle")
if not masks:
    raise RuntimeError("SAM3: no masks returned")
best = max(masks, key=lambda d: d["score"])
obj_pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
obj_center = obj_pts.mean(axis=0)

# ❌ WRONG — dict not array; mask_to_world_points needs a numpy array, not a dict
pts = mask_to_world_points(masks[0], depth, K, T)
```

If keyframes show the mask covering the table/background instead of the object, add an area cap (`m["mask"].sum() / (h*w) < 0.15`) as a diagnostic fix.

### 2. Molmo returns dict keyed by prompt string
`point_prompt_molmo(rgb, text)` returns `dict[str, tuple[int|None, int|None]]`.
Key is the prompt string, NOT an integer index.

```python
# ✅ CORRECT
result = point_prompt_molmo(rgb, "the target object")
px, py = result["the target object"]   # or: next(iter(result.values()))
if px is None:
    ...  # Molmo failed to find the object

# ❌ WRONG — KeyError: 0
px, py = result[0], result[1]
```

### 3. OBB keys: `"center"`, `"extent"`, `"R"`
`get_oriented_bounding_box_from_3d_points` returns `{"center", "extent", "R"}`.
- `"extent"`: full extents (not half), shape (3,)
- `"R"`: rotation matrix (3,3) — use as axes

```python
# ✅ CORRECT
obb = get_oriented_bounding_box_from_3d_points(pts)
center = obb["center"]
extent = obb["extent"]   # NOT "extents"
R = obb["R"]             # rotation matrix — NOT "axes"
```

### 4. Calling convention — API functions are standalone globals

```python
# ✅ CORRECT
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]       # nested key — NOT obs["rgb"]
depth = obs["agentview"]["images"]["depth"]
K = obs["agentview"]["intrinsics"]
T = obs["agentview"]["pose_mat"]

# ❌ WRONG — these methods don't exist on the low-level env
obs = env.get_observation()
env.segment_sam3_text_prompt(...)
```

### 5. Top-level call required
Code must execute at the top level. If you define helper functions, call them at the end:

```python
def run():
    obs = get_observation()
    ...

run()   # ← REQUIRED — without this, nothing runs and eval silently scores 0%
```

---

## 7. Output Directory Structure

**Every new run gets a timestamped directory.** Create it before writing any candidates:

```bash
SUITE=libero_goal_swap
TASK="put_the_bowl_on_the_stove"
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=outputs/claude_evosearch/$SUITE/$TASK/$RUN_ID
mkdir -p $RUN_DIR
```

All Evolutionary Search runs live under `outputs/claude_evosearch/<suite>/<task>/<YYYYMMDD_HHMMSS>/`. **Do NOT use `evosearch_runs/` or any other top-level directory.**

Structure within a run:
```
outputs/claude_evosearch/<suite>/<task>/<run_id>/
├── task_analysis.md             # written before iter_00, updated each iteration
├── scene_snapshot.jpg           # agentview from scene_snapshot.py
├── scene_snapshot_wrist.jpg     # wrist cam from scene_snapshot.py
├── iter_00/
│   ├── candidate_A/
│   │   ├── code.py              # candidate program
│   │   ├── eval_results.json    # per-trial results after eval
│   │   ├── logs/                # trial_01.log, trial_02.log, ...
│   │   ├── videos/              # saved after eval completes
│   │   │   ├── trial_07_success.mp4
│   │   │   ├── trial_03_failure.mp4
│   │   │   ├── success.log
│   │   │   ├── failure.log
│   │   │   └── trial_07_deep_debug/   # only if pass rate ≤ threshold
│   │   │       ├── keyframes/   # per-step RGB/depth/mask/grasp arrays
│   │   │       ├── trace.json
│   │   │       └── video_*.mp4
│   │   └── eval/                # replay_trial.py output per seed
│   │       └── trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/
│   │           ├── trace.json   # API call log (always saved)
│   │           └── summary.txt  # stdout + reward
│   ├── candidate_B/
│   │   └── ...
│   ├── highlights.json
│   └── iter_summary.json        # leaderboard across all candidates
└── iter_01/
    └── ...
```

Note: `trace.json` is always saved per trial. Video and numpy keyframes are NOT saved during bulk eval (disabled for speed) — only in highlight re-runs after eval completes.

---

## 8. Writing Candidate Programs

### Runtime efficiency — CRITICAL
Each `move_to_joints()` call is expensive, minimize them (`motion-efficiency.md`).

### Non-prehensile and contact tasks (push, slide, wipe, press)
(`push-contact-tasks.md`)

### Example template (copy-adapt):

```python
"""
Candidate X: <one-line description>
Hypothesis: <specific failure mode this targets>
Differs from prior: <structural difference, not just param tweak>
Expected failure if wrong: <what trace would show>
Seeded from: <candidate+iter or 'novel'>
"""
import numpy as np
from scipy.spatial.transform import Rotation

def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz for API

obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:,:,0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]
h, w = rgb.shape[:2]

# Segment pick object — max(score) is fine for specific objects; add area cap if keyframes
# show wrong mask (see §6.1 for when to use area filtering)
masks = segment_sam3_text_prompt(rgb, "<manipulated object prompt>")
if not masks:
    raise RuntimeError("SAM3: no masks returned")
best = max(masks, key=lambda d: d["score"])
obj_pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
obj_center = obj_pts.mean(axis=0)

quat = make_topdown_quat()
grasp_pos = np.array([obj_center[0], obj_center[1], obj_center[2] + 0.01])

# Grasp: pre-grasp → lower → close
open_gripper()
joints = solve_ik((grasp_pos + np.array([0,0,0.08])).tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)
joints = solve_ik(grasp_pos.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)
close_gripper()

# Lift
lift_z = grasp_pos[2] + 0.15
joints = solve_ik([grasp_pos[0], grasp_pos[1], lift_z], quat.tolist())
if joints is not None: move_to_joints(joints)

# Re-observe for target
obs2 = get_observation()
rgb2 = obs2["agentview"]["images"]["rgb"]
d2 = obs2["agentview"]["images"]["depth"]
d2 = d2[:,:,0] if len(d2.shape) == 3 else d2
K2 = obs2["agentview"]["intrinsics"]
E2 = obs2["agentview"]["pose_mat"]

target_masks = segment_sam3_text_prompt(rgb2, "<goal / surface prompt>")
if not target_masks:
    raise RuntimeError("SAM3: no masks returned for target")
best_target = max(target_masks, key=lambda d: d["score"])
target_pts = mask_to_world_points(best_target["mask"].astype(np.uint8), d2, K2, E2)
target_center = target_pts.mean(axis=0)
surface_z = target_pts[:,2].max()

# Transport: arc to above-target
above_target = np.array([target_center[0], target_center[1], lift_z])
joints = solve_ik(above_target.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)

# Place
release_z = surface_z + 0.03
joints = solve_ik([target_center[0], target_center[1], release_z], quat.tolist())
if joints is not None: move_to_joints(joints)
open_gripper()
```

### K=8 diversity rule

Each candidate must test a distinct hypothesis — no two should fail at the same stage for the same reason. Before writing, ask: *would I learn something different from each one?*

- Do not re-test anything listed as **Eliminated** in task_analysis.md §6
- Seed from top-3 performers of any prior iterations, not just the current winner
- At least one candidate must be structurally different from all prior iterations

**Cross-reference rule — apply when a new technique is discovered:**

When you discover a technique mid-run (arm reconfiguration like j[6] rotation, a different IK seed, a new contact strategy, etc.), immediately ask:

> "Which approaches listed under **Blocked, untested reconfiguration** in task_analysis.md should be retried with this technique?"
A blocked approach is only truly eliminated after it fails **with** the reconfiguration applied. Until then, it remains a live hypothesis.

---

## 9. Running the Evaluator

**ALWAYS use `--no-highlights` for quick screens.** After all trials finish, `save_highlight_videos` re-runs one success + one failure trial per candidate via sequential `subprocess.run()`. With 8 candidates × 2 replays × ~60-90s each = **16-24 minutes of post-processing** before the process exits and the notification fires. `iter_summary.json` is written before this — results are available immediately — but the background ping is delayed until highlight saving finishes.

```bash
# Quick screen (10 trials) — ALWAYS add --no-highlights for fast notification
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv-libero/bin/python3 scripts/libero/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_NN \
    --suite $SUITE --task "$TASK" \
    --trials 10 \
    --sim-gpus 4 5 6 7 --parallel-per-gpu 2 \
    --no-highlights \
    2>&1 | tee $RUN_DIR/iter_NN/eval.log

# Deep eval (30 trials, top candidates only) — highlights OK here
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv-libero/bin/python3 scripts/libero/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_NN \
    --suite $SUITE --task "$TASK" \
    --trials 30 \
    --candidates candidate_A candidate_B candidate_C \
    --sim-gpus 4 5 6 7 --parallel-per-gpu 2 \
    --best-only-highlights \
    2>&1 | tee $RUN_DIR/iter_NN/eval_deep.log
```

**Fixed seeds** (recommended for clean cross-iteration comparison):
```bash
SEEDS=$(python3 -c "import json; print(' '.join(str(s) for s in json.load(open('$RUN_DIR/iter_00/iter_summary.json'))['trial_seeds']))")
# add --trial-seeds $SEEDS to subsequent iters
```

**Automatic deep debug** (disabled by default):
```bash
... --deep-debug-threshold 0.10   # auto-trigger when best ≤ 10%
```
Output lands in `candidate_X/videos/trial_NN_deep_debug/`.

### Speed expectations:
| Phase | Time |
|-------|------|
| Pool warmup | ~30s × N_workers (sequential) — 8 workers ≈ 4min |
| Per trial | ~8-15s (reset + code + perception) |
| Highlight saving | ~60-90s × 2 per candidate, sequential — 16-24min for 8 |
| Quick screen (10 trials, --no-highlights, 8 workers) | ~6min total |
| Deep eval (30 trials, 3 candidates, --best-only-highlights) | ~8min total |

All candidates run concurrently (flat task dispatch, not sequential per-candidate).

---

## 10. Reading Results — After All Evaluations Complete

**Only start reading/debugging after the eval process exits.**

### Quick analysis with the dedicated script

```bash
# Full per-candidate breakdown (gripper widths, IK failures, SAM3 prompts, EEF displacement, blocking)
python3 scripts/libero/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_NN

# Single candidate only
python3 scripts/libero/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_NN --candidate candidate_F

# Leaderboard only (no per-signal stats)
python3 scripts/libero/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_NN --summary-only
```

#### ARM MOVEMENT OFF detection (automatic)

The script automatically compares each `solve_ik` target position against the actual `robot_cartesian_pos` from the next `get_observation()`. If they differ by a significant margin, the arm did not reach its target. Could indicate a physical blocking of the arm.
```
*** ARM BLOCKING: 6 events across 6/10 trials ***
    target=[0.33, -0.12, 0.05]  actual=[0.33, -0.10, 0.05]  error=0.021m
```
When blocking is detected: read `push-contact-tasks.md` and `wrist-rotation-blocking.md`.

### Manual trace inspection

**iter_summary.json schema** — each entry in `s['candidates']` has: `candidate`, `code_path`, `trials`, `pass_count`, `pass_rate`, `mean_reward`, `errors`, `trial_results`. Top-level keys: `best_candidate`, `best_pass_rate`, `trial_seeds`.

```bash
# Read leaderboard
python3 -c "
import json; from pathlib import Path
s = json.loads(Path('$RUN_DIR/iter_00/iter_summary.json').read_text())
for c in s['candidates']:
    print(f\"{c['candidate']:<22} {c['pass_rate']:.1%}  errors={c['errors']}  passes={c['pass_count']}/{c['trials']}\")
print(f'Winner: {s[\"best_candidate\"]} ({s[\"best_pass_rate\"]:.1%})')
"
```

### Key signals in trace.json:

| Signal | What it means |
|--------|--------------|
| `close_gripper → gripper_width ≈ 0.003` | Grasp missed (near-zero = empty fingers) |
| `close_gripper → gripper_width ≈ 0.015` | Thin rim grasped (marginal) |
| `close_gripper → gripper_width ≈ 0.080` | Solid grasp |
| `open_gripper → width ≈ start_width` | Nothing was held (same as initial open) |
| `open_gripper → width < start_width` | Object WAS held, now released |
| `plan_grasp → 0 candidates` | GraspNet found nothing — use centroid fallback |
| `solve_ik` missing from trace | IK returned None — target out of workspace |
| `segment_sam3_text_prompt → num_masks: 0` | SAM3 found nothing — bad prompt |
| Many `move_to_joints` + little state change | Blocked interaction or wrong mask |
| `*** ARM BLOCKED ***` in analyze output | EEF >3cm from IK target — physical obstacle |

**Blocked vs missed:** both show zero displacement. `analyze_evosearch_traces.py` detects blocking automatically. To distinguish manually: compare `solve_ik` target vs `robot_cartesian_pos` in the next `get_observation()`.

### Common failure modes:

| Failure | Trace signature | Fix strategy |
|---------|----------------|--------------|
| Grasp miss | `gripper_width < 0.003 after close` | Centroid offset; check approach Z |
| Drop too high | `eef_z >> surface_z at release` | `pts[:,2].max()` + 3cm; multi-step lower |
| Wrong object | bbox far from expected | Better text prompt |
| IK failure | `solve_ik` missing or move stalls | Target out of workspace; adjust position |
| Code error | `sandboxrc_1` in folder name | Check `summary.txt` stderr |
| Transport drop | Grasp OK, nothing at target | Slower transport; wrist cam verification |

---

## 11. Visual Keyframe Debugging (only when necessary)

The top failure **and** top success trial for each candidate are saved as full replay directories:
```
candidate_*/videos/trial_NN_{failure,success}/
    video_*.mp4
    keyframes/
        step_000_obs_agentview.jpg   ← initial scene
        step_NNN_sam3.jpg            ← SAM3 segmentation overlay
        step_NNN_obs_agentview.jpg   ← arm state after each action
    trace.json
    summary.txt
```

If you are unsure about failure case, you can read the failure replay:
- `step_000_obs_agentview.jpg` — initial scene, object positions, obstacles
- A mid-execution frame — is the arm blocked, colliding, or missing the object?
- The final frame — where did the object end up?

**Key signals from images:**
- Object not moving despite arm motion → physical obstacle (change approach direction, not parameters)
- Arm contorted near a wall/shelf → blocked approach — mark in §5
- Object moved but wrong direction → push/grasp geometry off-axis

Update task_analysis.md §1-5 with anything new you see, then append the iter log entry to §6.

---

## 12. Post-Eval Replay

Run a replay for deeper inspection — pick trials that answer a specific question, not all trials.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --suite $SUITE --task "$TASK" --trial 7 \
  --replay-code $RUN_DIR/iter_NN/candidate_F/code.py \
  --config env_configs/libero/franka_libero_traced.yaml \
  --output-dir $RUN_DIR/iter_NN/candidate_F/videos/trial_07_debug
```

Saves: `keyframes/`, `trace.json`, `video_*.mp4`. Keyframes are JPEG — read with the Read tool.

After viewing keyframes, update the relevant section of `task_analysis.md`.

---

## 13. Interactive Debugging (REPL Mode)

Quick start (full workflow: `debug-workflow` skill):

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --suite $SUITE \
  --task "$TASK" \
  --trial 1 \
  --interactive \
  --config env_configs/libero/franka_libero_traced.yaml
```

All API functions are in scope. Use to test SAM3 prompts, verify IK targets, or step through candidate code.

---

## 14. Iteration Loop Checklist

**Before iter_00 (once per run):**
- [ ] Run `scene_snapshot.py` → **read both `$RUN_DIR/scene_snapshot.jpg` and `$RUN_DIR/scene_snapshot_wrist.jpg`**
- [ ] Populate `task_analysis.md` §1-5 from images — §5 must list every uncertain assumption and blocked approach direction
- [ ] SAM3 prompts validated (all `[OK]`, score > 0.5)

**Per iteration:**
- [ ] Read `task_analysis.md` in full (geometry + eliminated hypotheses + open questions)
- [ ] Write K=8 candidates, each testing a distinct hypothesis
- [ ] Run full eval (background), wait for completion
- [ ] Read `iter_summary.json` leaderboard
- [ ] Read keyframes for candidates: `step_000`, a SAM3 overlay, mid-execution, final frame
- [ ] For any candidate with >0% success, also read its success keyframes
- [ ] Update `task_analysis.md` §1-5 if visual inspection reveals new geometry or obstacles
- [ ] Append `### iter_NN` entry to `task_analysis.md` §6
- [ ] Write next K=8 candidates seeded from top-3
- [ ] Append to `docs/logs/YYYY-MM-DD.md`

### Decision rules:
| Outcome | Action |
|---------|--------|
| > 5pp improvement | Keep direction, refine further |
| Flat / no improvement | Different hypothesis — re-read traces |
| Regression (worse) | New bug introduced — diff vs previous winner |
| Consistent stderr errors | Fix API usage before strategy |
| All candidates 0% | Visual inspection + REPL before writing more |
| Plateau (< 3pp for 2+ iters) | Check §5 for untried directions; force structural variety |
