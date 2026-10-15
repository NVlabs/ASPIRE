# Example CaP Programs

The LIBERO and Robosuite directories contain public-API policies; each task's documented
validation scope is stated below and in its benchmark guide.
Run commands from the repository root with SAM3, Contact-GraspNet, and PyRoki
healthy on ports 8114–8116.

The files are CaP programs, not standalone Python modules: `cap-harness run`
validates them and injects the allowlisted API functions and contract types.

Each task also has a `_curobo_seed1.py` variant. These variants use PyRoki IK
and cuRoboV2 c-space planning for free-space moves, while retaining explicit
interpolation for contact and held-object moves. Cube lifting, two-arm lift,
two-arm handover, and LIBERO-Pro object swap are verified task-success runs with
at least one successful cuRobo trajectory. Start port 8118 with
`scripts/supervise_services.sh --profile rtx5090 --providers sam3,contact_graspnet,pyroki,curobo`,
then substitute the matching variant path in the command below.

## LIBERO-Pro Object Swap

- Benchmark: `libero-pro`
- Task: pick up the alphabet-soup can and place it in the basket
- Suite/task: `libero_object_swap:0`
- Seed: `1`

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device CUDA_VISIBLE_DEVICES=0 \
  .venv-libero/bin/cap-harness run \
  --benchmark libero-pro --suite libero_object_swap --task-id 0 --seed 1 \
  --program examples/libero/object_swap_seed1.py \
  --output-root outputs/examples --max-steps 2000 \
  --camera-width 800 --camera-height 512
```

Additional LIBERO programs exercise drawer opening, bowl placement and drawer
closing, two side-grasp bottle placements, and microwave loading/closing. The
task mapping and validation scope are listed in
[`docs/libero-pro.md`](../docs/libero-pro.md). They use the same injected public
API as object swap; several deliberately compose `plan_motion()` and
`execute_trajectory()` to preserve a grasp through contact-rich motion.

## Robosuite Cube Lifting

- Benchmark: `robosuite`
- Task: pick up and lift the red cube
- Suite/task: `cube_lifting:0`
- Seed: `1`

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device CUDA_VISIBLE_DEVICES=0 \
  .venv-robosuite/bin/cap-harness run \
  --benchmark robosuite --suite cube_lifting --task-id 0 --seed 1 \
  --program examples/robosuite/cube_lifting_seed1.py \
  --output-root outputs/examples --max-steps 3000 \
  --camera-width 512 --camera-height 512
```

## Robosuite Two-Arm Lift

- Benchmark: `robosuite`
- Task: grasp both pot handles and lift the pot with both arms
- Suite/task: `two_arm_lift:0`
- Seed: `1`

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device CUDA_VISIBLE_DEVICES=0 \
  .venv-robosuite/bin/cap-harness run \
  --benchmark robosuite --suite two_arm_lift --task-id 0 --seed 1 \
  --program examples/robosuite/two_arm_lift_seed1.py \
  --output-root outputs/examples --max-steps 3000 \
  --camera-width 512 --camera-height 512
```

## Robosuite Two-Arm Handover

- Benchmark: `robosuite`
- Task: primary lifts the hammer and transfers its handle to secondary
- Suite/task: `two_arm_handover:0`
- Seed: `1`

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device CUDA_VISIBLE_DEVICES=0 \
  .venv-robosuite/bin/cap-harness run \
  --benchmark robosuite --suite two_arm_handover --task-id 0 --seed 1 \
  --program examples/robosuite/two_arm_handover_seed1.py \
  --output-root outputs/examples --max-steps 7000 \
  --camera-width 512 --camera-height 512
```


## BEHAVIOR-1K, R1 Pro pickup tasks

- Benchmark: `behavior`
- Tasks: `turning_on_radio:0` (pick up the red radio from the living-room table) and
  `picking_up_trash:0` (pick up the blue can of soda from the living-room floor)
- Seeds: task-instance ids present in the downloaded dataset (1, 2, 3 in the validation plan)
- Status: verified 2026-09-12 on an RTX 4090 with `scripts/validate_behavior.sh pickup` (seeds 1-3):
  `turning_on_radio` 2/3, `picking_up_trash` 2/3; the headers carry the per-seed detail

Both programs embed `examples/behavior/_prelude.py` verbatim (search by turning, torso tilt,
standoff and approach poses, grasp attempts in score order); `tests/test_behavior_examples.py`
keeps the copies identical. Programs may `import math`; nothing else is importable.

```bash
OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=0 OMNIGIBSON_DATA_PATH="$HOME/behavior-data" \
  .venv-behavior/bin/cap-harness run \
  --benchmark behavior --suite turning_on_radio --task-id 0 --seed 1 \
  --program examples/behavior/turning_on_radio_seed1.py \
  --output-root outputs/examples --max-steps 6000 \
  --camera-width 512 --camera-height 512
```

## Other LIBERO-Pro programs

`examples/libero/` also carries five programs that came out of fix-loop campaigns rather than a
pinned validation scope: `close_microwave.py`, `open_middle_drawer.py`,
`put_bowl_in_bottom_drawer.py`, `put_wine_bottle_on_drawer.py` and `put_wine_bottle_on_rack.py`.
They are accepted by the generated-program validator and run with the same command shape as the
object-swap program, substituting their suite and task id; no success rate is claimed for them
here, so treat them as starting points rather than release-gated results.
