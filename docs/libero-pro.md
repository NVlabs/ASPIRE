# LIBERO-Pro

## Setup

```bash
cd /path/to/ASPIRE
scripts/bootstrap_libero.sh
source .venv-libero/bin/activate
export LIBERO_CONFIG_PATH="$PWD/.libero"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
```

The bootstrap initializes pinned submodules and installs the frozen runtime into `.venv-libero`.
Do not repair it with unpinned package upgrades; update the lock and bootstrap deliberately.

Offscreen rendering needs the vendor-neutral GLVND dispatch library `libEGL.so.1` (Debian
`libegl1`). The NVIDIA driver alone is not enough — MuJoCo loads `libEGL.so.1`, and without it
`doctor` fails the EGL check with `'NoneType' object has no attribute 'eglQueryString'`.

### Selecting a GPU

EGL device enumeration ignores `CUDA_VISIBLE_DEVICES`, so MuJoCo renders on physical GPU 0 unless
`MUJOCO_EGL_DEVICE_ID` says otherwise. Set both when GPU 0 is busy or when placing episodes on a
specific device; a contended GPU 0 yields intermittent depth failures
(`depth conversion produced invalid metric values`, `point_cloud_failed`) that look like policy
bugs rather than resource contention:

```bash
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU cap-harness run ...
```

## Semantics

LIBERO-Pro contains 80 required `(suite, task)` pairs across Goal, Object, Spatial, and Long
families. The adapter publishes calibrated RGB-D, seven Panda joints, normalized gripper state, and
the end-effector pose in `robot_base`. A `RobotAction` advances exactly one 20 Hz simulator tick.
The pinned LIBERO robosuite fork is an implementation dependency, not a separate policy API.
The shared tools, LIBERO metadata extensions, and host-only adapter methods are catalogued in the
[API reference](api-reference.md).

## Validation

```bash
cap-harness doctor --reset --output validation-artifacts/libero/doctor.json
cap-harness validate --plan configs/validation/libero-smoke.yaml \
  --output-dir validation-artifacts/libero/smoke
cap-harness validate --plan configs/validation/libero-nightly.yaml \
  --output-dir validation-artifacts/libero/nightly
```

`cap-harness validate --plan` is the canonical, cross-embodiment validation entry point (the same
command validates RoboSuite via its plans). The smoke matrix runs seed 1; nightly
runs seeds 1–3. Each case verifies reset metadata, schemas, a 0.05-second hold tick, a
state-changing movement tick, clean close, and fingerprint-safe resume.
These are structural gates, not task-completion benchmarks.

## Recorded run

```bash
cap-harness run \
  --benchmark libero-pro \
  --suite libero_object_swap \
  --task-id 0 --seed 1 \
  --program examples/libero/object_swap_seed1.py \
  --output-root outputs
```

### Initial-state modes

`--init-mode` selects how the LIBERO-Pro adapter derives the initial scene from the seed:

- `saved` (default): restore saved initial state `(seed - 1) % N` after reset. Deterministic, but
  seeds wrap around the finite saved-state set (typically 50), so seeds 51+ repeat earlier states.
- `seeded`: seed the environment RNG and let reset sample procedural placements. Do **not**
  restore a saved state afterward: that would overwrite the sampled movable-object positions.
  This avoids wrapping around the finite saved-state set, but different seeds are not a guarantee
  of distinct layouts for every task (placement constraints still apply). No saved state is
  selected, so `init_state_index` is `null`. Tasks with no saved states require this mode.

The selected mode is recorded in `run.json` (`init_mode`), the task context, and
`libero.get_task_metadata()["init_mode"]`. Earlier buggy seeded runs restored saved state 0;
those results should not be treated as evaluation across independently sampled layouts.

SAM3 requires authorized access to its gated model; request access and log in to Hugging Face as
described in [Host setup](host-setup.md#sam3-model-weights-need-a-hugging-face-token). Missing
credentials are an explicit provider blocker; credentials must never be copied into the repository or
run artifacts.

## Task examples

The public-API-only examples cover perception-driven manipulation beyond the
object-swap smoke task. They contain no seed-specific behavior; validation
claims should name the exact task and seeds that were run.

| Program | Suite/task | Behavior |
| --- | --- | --- |
| `open_middle_drawer.py` | `libero_goal_swap:0` | Localize and pull the middle drawer. |
| `put_bowl_in_bottom_drawer.py` | `libero_10_swap:3` | Rim-grasp a bowl, place it in the open bottom drawer, and close the drawer. |
| `put_wine_bottle_on_drawer.py` | `libero_goal_swap:2` | Side-grasp a bottle and place it on the cabinet. |
| `put_wine_bottle_on_rack.py` | `libero_goal_swap:9` | Side-grasp a bottle and place it on the rack's sloped receiving plane. |
| `close_microwave.py` | `libero_90:33` | Close the open microwave with a staged door sweep. |

These programs intentionally compose the documented low-level tools
(`segment_text`, `mask_to_point_cloud`, `plan_motion`, and
`execute_trajectory`) where contact-rich behavior needs more control than a
single high-level move. See the [API reference](api-reference.md) for the
complete boundary.

The five programs above have fresh task-success evidence at seed 1. The
microwave sweep additionally passes seeds 2 and 3. This is selected-example
evidence, not a broad randomized-layout robustness claim.
