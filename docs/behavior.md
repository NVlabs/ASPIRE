# BEHAVIOR-1K

R1 Pro in OmniGibson on NVIDIA Isaac Sim, restricted to two pickup tasks that mirror the ASPIRE
reference: `turning_on_radio` ("pick up the red radio") and `picking_up_trash` ("pick up the blue
can of soda"), both in the `house_double_floor_lower` scene. The public API is the one documented
in the [API reference](api-reference.md); the extensions under `behavior.*` add the holonomic base
and the torso.

## Setup

```bash
cd /path/to/ASPIRE
scripts/bootstrap_behavior.sh --accept-dataset-tos       # Python 3.11, Isaac Sim 5.1, cuRobo, datasets
# cuRobo is compiled for the profile whose architecture matches GPU 0 (nvidia-smi compute
# capability); pass --profile <name> or CAP_HARNESS_PROFILE to build for another card.
source .venv-behavior/bin/activate
export OMNIGIBSON_DATA_PATH="$HOME/behavior-data" OMNIGIBSON_APPDATA_PATH="$PWD/.behavior-appdata"
export OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=0
ulimit -c 0
```

The runtime is deliberately outside `uv.lock`: BEHAVIOR-1K v3.9.2 pins Isaac Sim 5.1.0 wheels,
which exist only for Python 3.11, plus torch 2.7 (CUDA 12.8) and `numpy<2`. The bootstrap replays
the upstream installer with `uv pip`, compiles cuRobo against a CUDA 12.x toolkit
(`CAP_HARNESS_CUDA_HOME` if it is not under `/usr/local/cuda-12*` or `~/cuda-12.8`), installs
`cap-harness` editable with `--no-deps`, and refuses to proceed without the datasets unless
`--skip-datasets` is given. Never run `uv sync` in `.venv-behavior`; it removes the Isaac wheels.

Datasets (about 40 GB) are downloaded only with `--accept-dataset-tos`, which accepts the BEHAVIOR
Data Bundle terms (non-commercial academic research, encrypted assets usable only inside
OmniGibson, never redistributed). They live under `CAP_HARNESS_BEHAVIOR_DATA` (default
`~/behavior-data`), outside the repository. Isaac selects its GPU through `OMNIGIBSON_GPU_ID`, not
`CUDA_VISIBLE_DEVICES`; the first launch on a machine compiles shaders for several minutes into
`OMNIGIBSON_APPDATA_PATH`, later launches take tens of seconds. Measured on an RTX 4090 with a
warm shader cache: Kit ready in 7 s, the first reset of a task about 190 s on a cold cache and about 35 s once the scene has
loaded on the machine before, later
task-instance reloads about 5 s, roughly 16 GB of system RAM and 12 GB of VRAM for the
simulator alone. Both tasks ship 300 training instances (ids 0–299).

## Semantics

Arms are `primary` (left) and `secondary` (right), seven joints each, with normalized grippers
(0 = closed, 1 = open) mapped onto the two finger joints. A `RobotAction` advances one 30 Hz tick;
every tick sends the full R1 Pro joint target through `robot.q_to_action`, so arms, torso and base
move together and nothing drifts while one of them is commanded.

The public base frame is `odom`: the base-footprint pose captured when the task instance was loaded,
with z on the floor and yaw only. Cameras (`head`, `left_wrist`, `right_wrist`, RGB-D with
intrinsics), end-effector poses, point clouds and navigation goals are all expressed in `odom`, so a
localized object stays valid after the base drives. `behavior.get_base_pose()` returns the current
`(x, y, yaw)`; `behavior.navigate_to_pose(x, y, yaw, planner="curobo" | "servo")` drives there and
reports `moved_x`, `moved_y`, `moved_yaw`; `behavior.move_torso(target)` and
`behavior.reset_torso()` servo the four torso joints; `behavior.plan_standoff_pose(support_cloud,
object_cloud, standoff_m)` and `behavior.plan_approach_pose(object_cloud, distance_m)` are pure
geometry: the first returns a base pose just outside the support surface's nearest edge, the second
a pose short of a free-standing object along the current line of sight, both facing the object;
`behavior.base_pose_is_free(x, y)` says whether the base footprint fits at an `odom` position
according to the scene's traversability map, and `navigate_to_pose` refuses a blocked goal before
planning. When cuRobo's direct base plan fails (its wrapper optimises trajectories without graph
search, so a goal behind furniture is out of reach in one plan), the harness follows the scene's
traversability map in 1 m hops, planning each hop with cuRobo and servoing a hop that cannot be
planned; the result's diagnostics report `route`, `hops`, `servo_hops` and cuRobo's `direct_status`.
The servo path steps its commanded pose toward the goal rather than commanding the goal outright,
so the base cannot move fast enough to sweep furniture aside, and navigation fails loudly if the
base leaves the floor, which a map cell with no ground geometry beneath it can otherwise cause
without the planar pose showing anything wrong.

Motion planning runs in-process on OmniGibson's cuRobo wrapper with the simulator's own robot model
and collision world, registered as the `curobo` and `curobo-integrated` backends and used by
default (`MotionStrategy(ik_solver="curobo", trajectory_planner="curobo",
pose_planner="curobo-integrated")`). Arm plans use upstream's arm embodiment, in which the four
torso joints move with the arm; the adapter executes the planned torso motion alongside the arm
targets (`CAP_HARNESS_BEHAVIOR_LOCK_TORSO=1` plans with the torso locked instead). A scene
object whose bounding box contains a pose goal (the
object the arm is reaching into) is left out of the collision world for that plan, as OmniGibson's
own grasp primitive does; the ignored objects are recorded in the plan's diagnostics. PyRoKi and
the HTTP cuRobo service are not used. SAM3 (8114)
and Contact-GraspNet (8115) are reached over HTTP as for the other benchmarks; grasp candidates
follow the harness convention (the pose's +Z column is the approach axis).

Seeds are challenge task-instance ids, one to one; the registry enumerates the ids present under
`2026-challenge-task-instances/scenes/house_double_floor_lower/json/` and rejects a seed that has
no instance. Success is the ASPIRE-comparable pickup rule evaluated by a host-side witness: the
robot's grasp holds the target object and the object is at least 5 mm above the height it had after
the instance load (baseline captured after the load, unlike the reference). The witness writes
`evaluation/protocol.json`; generated programs never see it, the BDDL predicates, the object
registry, or instance segmentation.

## Validation

```bash
cap-harness doctor --environment-root "$VIRTUAL_ENV" \
  --expected-environment-name .venv-behavior --runtime behavior \
  --reset --reset-embodiment behavior --reset-suite turning_on_radio --seed 1 \
  --output validation-artifacts/behavior/doctor.json

scripts/validate_behavior.sh structural      # both tasks, instances 1-3, no providers
scripts/validate_behavior.sh pickup          # example programs; needs SAM3 and Contact-GraspNet
```

The structural plan (`configs/validation/behavior-structural.yaml`) resets each task on instances
1–3, checks the observation schema, that one hold tick advances exactly 1/30 s, and that the
primary arm moves. The pickup plan (`configs/validation/behavior-pickup.yaml`) runs the example
programs in a fresh process per seed with `evaluator: behavior_pickup` and requires at least one
success per task. Episodes take minutes; run one Isaac process per GPU and read `outcome.json`
rather than the process exit code, which Kit can leave at 139 after every artifact is written.

Measured 2026-09-12 on an RTX 4090: the structural plan passed 6/6 (both tasks, instances 1-3)
in 197 s of wall time in one process, including the scene rebuild when the suite changes; the
process then exited 139 with `summary.json` complete.
The same doctor (with a live reset) and the structural plan (6/6, 204 s) also passed from a fresh
clone of the repository with its own `.venv-behavior` built by the bootstrap, sharing only the
downloaded datasets. The pickup plan passed twice the same day: prelude v8 radio 2/3 (seeds 2, 3) and trash 1/3
(seed 3) in 632 s; prelude v10 radio 2/3 (seeds 1, 3) and trash 2/3 (seeds 2, 3) in 708 s, six
fresh processes each. The misses were all at the grasp (fingers closing without an assisted
grasp); navigation, search and re-acquisition succeeded on every seed.

## Recorded run

```bash
OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=0 \
  .venv-behavior/bin/cap-harness run \
  --benchmark behavior --suite turning_on_radio --task-id 0 --seed 1 \
  --program examples/behavior/turning_on_radio_seed1.py \
  --output-root outputs/examples --max-steps 6000 \
  --camera-width 512 --camera-height 512
```

Videos are recorded at every third simulator frame (10 fps) for the three cameras; everything else
in [Recorded runs](run-artifacts.md) applies unchanged.

A successful radio run on this bench takes 80-95 s of wall time (about 35 s of it Kit start-up
and scene load) and 370-440 ticks; a trash run 115-130 s and 650-780 ticks. `outcome.json` reports
`termination_reason: task_succeeded` the tick the witness sees the object held and 5 mm up, and
`evaluation/protocol.json` carries the tick indices and the measured clearance.
