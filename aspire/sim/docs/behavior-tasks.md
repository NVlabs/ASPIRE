# BEHAVIOR Tasks

BEHAVIOR tasks use the [BEHAVIOR-1K](https://behavior.stanford.edu/) stack via
NVIDIA Isaac Sim and OmniGibson. This branch ships configs and runbooks for
two R1Pro BEHAVIOR tasks: soda-can pickup and radio pickup.

The broader BEHAVIOR-1K benchmark has many household activities, but this repo
does not ship YAML configs or runbooks for those tasks yet.

## Tested configuration

The instructions below were run end to end on:

| Component | Value |
|---|---|
| Host | Cluster node, 4× NVIDIA L40 (48 GB each), 128 cores |
| Driver | 580.95.05 |
| OS | Ubuntu 22.04.5 LTS, x86-64 |
| CUDA toolkit (for building curobo) | 12.6 (`/usr/local/cuda-12.6`) |
| Python | 3.10.12 |

Verified package versions:

| Package | Version |
|---|---|
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| numpy | 1.26.4 (OmniGibson requires `numpy<2`) |
| setuptools | 80.10.2 (`<81`; see troubleshooting) |
| opencv-python-headless | 4.12.0.88 |
| omnigibson | 3.7.2 (editable, from the `b1k` submodule) |
| isaacsim | 4.5.0.0 |
| curobo | StanfordVL fork @ `cbaf7d32436160956dad190a9465360fad6aba73` |

Disk footprint after a full install: ~33 GB BEHAVIOR-1K assets, ~2.4 GB robot
assets, ~400 MB 2025 challenge task instances, ~5 GB SAM3 weights, plus the
Isaac Sim wheels in the virtual environment. Budget ~60 GB.

Other hosts should work, but treat the first run on a new machine as bring-up
and record any local fixes here.

## Prerequisites

- Linux x86-64 with an NVIDIA GPU and driver 535+.
- A **CUDA 12.x toolkit with `nvcc`**. curobo compiles CUDA extensions against
  the installed torch; torch tolerates a CUDA minor-version mismatch but not a
  major one, so a CUDA 13-only host cannot build the validated cu124 stack.
- Python 3.10 (Isaac Sim wheels are cp310-only) and [uv](https://docs.astral.sh/uv/).
- EGL for headless rendering: `sudo apt-get install -y libegl1 libgl1`.
- A Hugging Face account with access to the gated `facebook/sam3` model.

## Installation

Run one command from the simulation workspace:

```bash
cd aspire/sim
export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"

# Required once, for the gated SAM3 weights.
hf auth login

scripts/behavior/setup_behavior.sh --accept-dataset-license
```

`--accept-dataset-license` is mandatory and accepts the BEHAVIOR-1K dataset
license on your behalf. Without it the script refuses to run rather than
downloading assets under an unacknowledged license. Use `--skip-datasets` to
install code only.

The script is **idempotent and resumable**: completed steps are detected and
skipped, and it never deletes downloaded assets. Re-run it after an interruption
instead of starting over. Useful options:

| Option | Effect |
|---|---|
| `--gpu-id N` | GPU used by the verification run (default 0) |
| `--skip-datasets` | Install code only |
| `--force-curobo` | Rebuild the curobo CUDA extensions |
| `--skip-verify` | Skip the mandatory verification step |
| `--venv PATH` | Install into a different virtual environment |

It finishes by running `scripts/behavior/verify_behavior.py`, which fails loudly rather
than leaving a half-working environment.

> **Do not run `cap/third_party/b1k/uv_install.sh` directly.** It does not
> complete on a clean host: it never creates the venv it installs into, aborts at
> its own verification step on a missing `cv2` (silently skipping curobo, pyroki,
> and the perception dependencies), and never installs SAM3 because its path test
> is off by one directory (`CAPX_ROOT` already resolves to the package directory,
> then the test appends it again).
> `scripts/behavior/setup_behavior.sh` invokes it for the pinned Isaac Sim wheel set and
> applies every needed deviation from the ASPIRE side, so the `b1k` submodule
> stays clean.
>
> These are **upstream defects, not ASPIRE-specific**. Each was reproduced on
> 2026-07-30 against a pristine checkout of the upstream reference repository at
> its pinned commit, on this same node, using the same pinned B1K submodule
> (`272ec5ca`) and a byte-identical `uv_install.sh`.

### Verifying an existing install

```bash
source cap/third_party/b1k/.venv/bin/activate

python scripts/behavior/verify_behavior.py --gpu-id 2   # full: servers + oracle seed
python scripts/behavior/verify_behavior.py --quick      # imports, pins, datasets only
```

It checks GPU/driver, the validated pins, Isaac Sim/OmniGibson imports, the
curobo **compiled** extensions, dataset presence including the `r1pro_ik.urdf`
overlay and `models/r1pro/usd/r1pro.usda` runtime payload, SAM3 and
Contact-GraspNet startup on real sockets, and one soda-can oracle seed with
video. It writes `environment_manifest.json` to the output directory and exits
non-zero on any failure. See
[`docs/logs/2026-07-31-final-acceptance.md`](logs/2026-07-31-final-acceptance.md)
for the independent clean-clone acceptance record.

## Environment variables

Set these before running BEHAVIOR tasks:

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
ulimit -c 0
```

Isaac Sim uses `OMNIGIBSON_GPU_ID`, **not** `CUDA_VISIBLE_DEVICES`, for GPU
selection. The perception servers use the default CUDA device, so pointing
`OMNIGIBSON_GPU_ID` at a different GPU keeps the simulator off the perception
GPU. `OMNI_KIT_ACCEPT_EULA=YES` is also required for any bare `import isaacsim`,
which otherwise prompts interactively and dies with `EOFError` in scripts.

`scripts/behavior/setup_behavior.sh` installs `aspire` as an editable package into the
B1K venv, so `python -m aspire.sim.cap.envs.launch_b1k` works from any
directory. No `PYTHONPATH` is needed.

## Supported tasks

| Task | Configs | Notes |
|---|---|---|
| Soda-can pickup | `env_configs/r1pro/r1pro_pick_up_trash.yaml`, `..._aspire.yaml`, `..._aspire_traced.yaml`, `..._oracle.yaml` | Historical filenames use `trash`, but the task prompt targets a blue can of soda. |
| Radio pickup | `env_configs/r1pro/r1pro_pick_up_radio.yaml`, `..._aspire.yaml`, `..._aspire_traced.yaml`, `..._oracle.yaml` | Table-object task using the R1Pro radio policy environment. |

## Smoke test without an LLM

The oracle configs set `use_oracle_code: true` and replay stored code, so they
need **no model endpoint and no API key**. This is the right first run on a new
machine: it exercises Isaac Sim, OmniGibson, curobo, both perception servers,
control, and video recording on their own.

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
ulimit -c 0

OMNIGIBSON_GPU_ID=2 python -m aspire.sim.cap.envs.launch_b1k \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml \
  --trial-ids 1 \
  --output-dir outputs/behavior/smoke/soda_oracle \
  --record-video True
```

Expected: a trial directory named
`trial_01_sandboxrc_0_reward_1.000_taskcompleted_1` containing `code.py`,
`summary.txt`, and four `video_combined_*.mp4` files (rgb, ego, left wrist,
right wrist), with `Reward: 1` and `Task Completed: True` in `summary.txt`.

The launcher inserts a model-name segment *before* the final component of
`--output-dir`, so with `--output-dir outputs/behavior/smoke/soda_oracle` the
trial directory lands at
`outputs/behavior/smoke/<model>/soda_oracle/trial_01_...` — for the oracle
configs, `<model>` is the launcher's default `google_gemini-3.1-pro-preview`
even though no model is contacted.

Timing on the tested host: ~15 s to Isaac Sim `app ready`, a few minutes of
scene loading, then ~80 s for the trial itself. The perception servers are
started automatically from the config's `api_servers` section.

## Runs that require a model endpoint

The non-oracle configs query an LLM. `launch_b1k` defaults to
`--server-url http://127.0.0.1:8110/chat/completions` with model
`google/gemini-3.1-pro-preview`, so you need that endpoint reachable or must
pass `--server-url`, `--model`, and `--api-key` explicitly. Keep keys in
environment variables or CLI arguments — never in YAML or committed files.

```bash
OMNIGIBSON_GPU_ID=2 python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml
```

Traced debug variants use `R1ProControlApiTraced` and write `trace.json` plus
`keyframes/` under each trial directory; they default to debug seeds 26-35
(`resume_idx: 26`, `trials: 35`):

```bash
OMNIGIBSON_GPU_ID=2 python -m aspire.sim.cap.envs.launch_b1k \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml \
  --trial-ids 26 27 28 \
  --output-dir outputs/behavior/debug/soda_aspire_traced \
  --record-video True
```

Replay saved code on one seed without any LLM call:

```bash
OMNIGIBSON_GPU_ID=2 python scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml \
  --replay-code path/to/code.py \
  --trial 1 \
  --output-dir outputs/behavior/replay/soda_trial1 \
  --record-video True
```

## Known issues

**Intermittent empty point cloud on the first perception call.** A trial can
fail with:

```
utils.py:264: RuntimeWarning: divide by zero encountered in divide
  xs = (us - cx) * d / fx
RuntimeError: QH6412 qhull input error ... Got 0 3-d points
```

`get_object_pose` deprojects the SAM3 mask through the depth map; if the
resulting cloud is empty, `get_oriented_bounding_box()` fails in qhull.

**Mechanism: the camera intrinsics are degenerate on failing runs.** In the
deprojection the only divisors are `fx` and `fy`, so `divide by zero` means
`fx == fy == 0`; every point becomes NaN, the cloud empties after outlier
removal, and qhull reports 0 points. `get_camera_intrinsics()` reads
`VisionSensor.intrinsic_matrix`, which derives `fx` from
`camera_parameters["cameraProjection"]`; OmniGibson adds the `camera_params`
annotator lazily and performs exactly three `render()` calls before the first
read, so a read that races the annotator returns zeros.

Probing intrinsics during a *passing* run is not informative — it reports the
healthy values (`fx = fy = 243.87`, 512×512, stable across extra `render()`
calls). The probe must be taken on a failing run.

**This is inherited upstream behaviour, not an ASPIRE defect.** On 2026-07-30 a
pristine checkout of the upstream reference repository, at its pinned commit on
this same node, reproduced the identical failure while executing
**byte-identical** oracle code (`md5 1d2a34a73c9f298cec3e565e13965457`) on the
same task and seed, against a semantically identical control API. A full package
comparison of the two environments showed no relevant difference.

It is intermittent and not yet tied to a deterministic trigger: ASPIRE observed
1 failure in 4 oracle runs (the same command both failed and passed), while the
upstream control failed its first 2 runs. **The root cause is not confirmed** —
specifically, why the `camera_params` annotator yields a zero projection matrix
on some runs. Retry the seed; to investigate, capture `get_camera_intrinsics()`
and the SAM3 mask pixel count at the call site **on a failing run**.
`scripts/behavior/verify_behavior.py` retries the oracle seed once (`--attempts`) for this
reason.

**Launcher exits 139 (SIGSEGV) at teardown.** The launcher can segfault while
shutting Isaac Sim down *after* writing all artifacts — including on runs that
report `reward 1.000, taskcompleted 1`. Task success and clean process exit are
separate facts; check both. This also reproduced in the pristine upstream
reference on this node (`exit 139`), so it is inherited, not an ASPIRE
regression. The commonly cited cause — a duplicate Vulkan ICD — does **not**
apply on a host where `/usr/share/vulkan/icd.d/nvidia_icd.json` is absent.

When capturing the launcher's status in a script, read `$?` directly. A trailing
`echo "EXIT=$?"` after the command reports the echo's own success and will hide a
segfault.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` during install | Nothing in the dependency chain installs OpenCV, and the upstream installer's verification step imports it. `scripts/behavior/setup_behavior.sh` pre-installs `opencv-python-headless`. |
| `EOFError` at `Do you accept the EULA?` | Export `OMNI_KIT_ACCEPT_EULA=YES` before any `import isaacsim`. |
| `error: 'float lerp(float, float, float)' conflicts with a previous declaration` | torch resolved to a cu13 build, which compiles extensions as C++20 where curobo's `helper_math.h` collides with `std::lerp`. Install the pinned `torch==2.6.0+cu124` **and** constrain the build environment (next row) — the venv pin alone is not enough. |
| cuRobo still fails after pinning torch in the venv | uv builds cuRobo in an isolated PEP 517 environment that re-resolves its **own** unpinned torch (look for a traceback rooted in `/tmp/.../builds-v0/...`). Set `UV_BUILD_CONSTRAINT` to a file pinning `torch==2.6.0+cu124` plus `UV_EXTRA_INDEX_URL`, or build with `--no-build-isolation`. `scripts/behavior/setup_behavior.sh` does the former. |
| cuRobo imports fine but you never saw it compile | A previously built wheel in the shared `~/.cache/uv` can be reused across repositories/environments. A successful `import curobo.curobolib.geom_cu` is **not** proof this environment built it — and a stale wheel silently mismatches the ABI (`undefined symbol: _ZN3c10...`). Use `--no-cache` when the build must be genuine. |
| `ptxas fatal: Unsupported .version 9.3; current version is '9.0'` | Mixed CUDA toolchain component versions. Use a system CUDA 12.x toolkit rather than assembling one from pip packages. |
| `fatal error: nv/target: No such file or directory` | Missing CCCL headers — the same mixed-toolchain problem as above. |
| `ModuleNotFoundError: No module named 'numpy'` building Contact-GraspNet | Its `setup.py` imports numpy without declaring it as a build dependency; install with `--no-build-isolation`. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥ 81 removed it and sam3's `model_builder` imports it. Pin `setuptools<81`. |
| `ModuleNotFoundError: No module named 'aspire'` | Install the package into the B1K venv: `uv pip install -e . --no-deps`. |
| SAM3 weights fail to download | The model is gated. Request access, then `hf auth login` inside the B1K venv. |
| `RuntimeError: No CUDA 12.x toolkit with nvcc found` | Install a CUDA 12.x toolkit or set `CUDA_HOME`. |

Never run `uv sync` inside the B1K virtual environment — it will tear out the
Isaac Sim wheels the upstream installer placed there.

## Architecture

- **Simulator**: `aspire.sim.cap.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel`
- **Task envs**: `aspire.sim.cap.envs.tasks.r1pro.r1pro_pickup_radio.R1ProRadioCodeEnv`
  and `aspire.sim.cap.envs.tasks.r1pro.r1pro_pickup_trash.R1ProTrashCodeEnv`
- **Control API**: `aspire.sim.cap.integrations.r1pro.control.R1ProControlApi`
- **Traced Control API**: `R1ProControlApiTraced`, registered via
  `aspire.sim.cap.integrations.trace_logger.make_traced_api`
- **OmniGibson configs**:
  `cap/third_party/b1k/OmniGibson/omnigibson/configs/r1pro_*.yaml`

## API servers

BEHAVIOR task configs auto-launch the perception servers listed in their
`api_servers` section. Ports are config-specific — the oracle and plain configs
use 8114/8115, the traced configs use 8116/8117. To start them manually:

```bash
source cap/third_party/b1k/.venv/bin/activate

python -m aspire.sim.cap.serving.launch_sam3_server --device cuda --port 8114
python -m aspire.sim.cap.serving.launch_contact_graspnet_server --port 8115
```
