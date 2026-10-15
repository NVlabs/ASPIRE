# Robosuite

## Setup

```bash
cd /path/to/ASPIRE
scripts/bootstrap_robosuite.sh
source .venv-robosuite/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
```

The canonical environment path is `.venv-robosuite`. When `CAP_HARNESS_VENV_ROOT` is set, this path
is a repository-local symlink to node-local storage. It uses this checkout as editable source but
remains independent of `.venv-libero`; provider services are shared over HTTP.

## Semantics

The registry contains `cube_lifting`, `cube_restack`, `cube_stack`, `nut_assembly`, `spill_wipe`,
`two_arm_lift`, and `two_arm_handover`. Single-arm Panda is `primary`; bimanual robot0 and robot1
are `primary` and `secondary`. Public poses use `robot0_base`; secondary transforms remain inside
the adapter/provider boundary.

A `RobotAction` advances one 20 Hz native tick. A bimanual action moves both arms in that tick, and
`SynchronizedTrajectory` requires equal waypoint counts. The generated program sees only normalized
observations and typed results, never native MuJoCo state or task-success data.
Synchronized requests and trajectories must name every arm in the embodiment. Trajectory joint
names must exactly match the controlled arm, and a gripper command for a fixed-tool arm is rejected
instead of being silently ignored.
Every trajectory carries its full expected `RobotState`; execution rejects active-arm,
inactive-arm, joint-name, or gripper drift before the first control tick. `set_grippers()` commands
all arm grippers in one tick while preserving cached joint targets. Terminal, truncated, timed-out,
cancelled, stale, planner-failed, and controller-failed motions have distinct typed statuses and
cannot be reported as successful execution.
The shared tools, Robosuite metadata extensions, and host-only adapter methods are catalogued in
the [API reference](api-reference.md).

## Validation

```bash
cap-harness doctor \
  --environment-root "$VIRTUAL_ENV" \
  --expected-environment-name .venv-robosuite \
  --runtime robosuite \
  --reset --reset-embodiment robosuite --reset-suite cube_lifting \
  --output validation-artifacts/robosuite/doctor.json

cap-harness validate --plan configs/validation/robosuite-structural.yaml \
  --output-dir validation-artifacts/robosuite/structural
cap-harness validate --plan configs/validation/robosuite-control.yaml \
  --output-dir validation-artifacts/robosuite/control
```

The structural matrix covers seven tasks with seeds 1–3. The control gate verifies independent and
same-tick bimanual motion, persistent gripper targets, and home error below 0.01 rad.

TwoArmLift and TwoArmHandover have a separate six-case task-success gate. It runs each task on seeds
1–3 in a fresh process and requires `program_ok=true`, `task_success=true`, and evaluator-only
`protocol_success=true`:

```bash
CAP_HARNESS_ROBOSUITE_GPU=3 scripts/validate_robosuite_bimanual.sh
```

To software-render the simulator, set `CAP_HARNESS_ROBOSUITE_RENDERER=cpu`; OSMesa must be
available to the process, either system-wide or through `CAP_HARNESS_OSMESA_LIBRARY_DIR`. This only
makes simulation rendering CPU-only: the policies still call the separately managed SAM3, PyRoki,
and cuRobo services (canonical ports 8114, 8116, and 8118) from the shared provider stack. Provider
placement, ports, and concurrency are defined by the declarative topology and profile overlays; see
[Environment architecture](environments.md).

The six-case gate refuses a dirty checkout or reused output directory. Each finalized run includes
three camera videos, public API/provider traces, native outcome, summarized protocol witnesses, a
content manifest, exact program/dependency/commit identity, and the campaign environment hash.
The verifier re-hashes every run artifact and writes `summary.json` plus a campaign-level
`artifact-manifest.json`. Privileged grasp and task predicates are consumed only by the host-side
protocol evaluator and never enter the generated-program registry or public traces.

## Recorded run

```bash
cap-harness run \
  --benchmark robosuite \
  --suite cube_lifting \
  --task-id 0 --seed 1 \
  --program examples/robosuite/cube_lifting_seed1.py \
  --output-root outputs \
  --max-steps 3000
```

cuRobo uses adapter calibration for its robot model. Integrated bimanual planning optimizes a
combined 14-DoF trajectory and checks self-, scene-, and inter-arm collision; see
[Optional Backends](optional-backends.md).
