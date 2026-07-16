# BEHAVIOR Tasks

BEHAVIOR tasks use the [BEHAVIOR-1K](https://behavior.stanford.edu/) stack via
NVIDIA Isaac Sim and OmniGibson. This branch ships configs and runbooks for
two R1Pro BEHAVIOR tasks: soda-can pickup and radio pickup.

The broader BEHAVIOR-1K benchmark has many household activities, but this repo
does not ship YAML configs or runbooks for those tasks yet. B1K setup and
execution are host-dependent and were not runtime-validated in this documentation
audit; use the first run on a new machine as bring-up/smoke testing.

## Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.x
- Isaac Sim 4.5.0 installed by `cap/third_party/b1k/uv_install.sh`

## Installation

From the repository root, enter the simulation workspace first:

```bash
cd aspire/sim
git submodule update --init cap/third_party/b1k cap/third_party/curobo

cd cap/third_party/b1k
./uv_install.sh --dataset --accept-dataset-tos
cd ../../..
```

After `cd ../../..`, your working directory should be `aspire/sim` again. The
installer sets up OmniGibson, Isaac Sim, BDDL, cuRobo, robot assets, and the
BEHAVIOR assets needed by the supported tasks.

### Post-Install Requirements

After `uv_install.sh`, copy the cuRobo headers needed for first-time JIT
compilation:

```bash
# Fix missing cuRobo CUDA headers for first-time JIT compilation.
source cap/third_party/b1k/.venv/bin/activate
cp cap/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

Do not delete system Vulkan ICD files as a general setup step. If Isaac Sim
reports duplicate or invalid ICDs, inspect `/etc/vulkan/icd.d/` and
`/usr/share/vulkan/icd.d/`, preserve the original files, and follow the NVIDIA
driver guidance for the host before changing system configuration.

On headless servers, also install EGL:

```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
```

On first run, cuRobo may JIT-compile CUDA kernels for several minutes.

## Environment Variables

Set these before running BEHAVIOR tasks:

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

Isaac Sim uses `OMNIGIBSON_GPU_ID`, not `CUDA_VISIBLE_DEVICES`, for GPU
selection.

## Supported Tasks

| Task | Configs | Notes |
|---|---|---|
| Soda-can pickup | `env_configs/r1pro/r1pro_pick_up_trash.yaml`, `env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml`, `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml`, `env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml` | Historical filenames use `trash`, but the task prompt targets a blue can of soda. |
| Radio pickup | `env_configs/r1pro/r1pro_pick_up_radio.yaml`, `env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml`, `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml`, `env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml` | Table-object task using the R1Pro radio policy environment. |

## Running Evaluations

Run these examples from `aspire/sim` with the B1K environment active and the
environment variables above set. For first-machine bring-up, start with one
traced debug seed through `launch_b1k --trial-ids` before launching full ranges.

Radio pickup:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml
```

Soda-can pickup:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_trash.yaml
```

ASPIRE variants:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml

uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml
```

Traced debug variants:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml

uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml
```

The traced configs use `R1ProControlApiTraced` and write `trace.json` plus
`keyframes/` under each trial directory. They default to debug seeds 26-35.

Exact seed and replay helpers:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch_b1k \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --trial-ids 26 27 28

uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --replay-code path/to/code.py \
  --trial 26
```

## Architecture

BEHAVIOR tasks use the following components:

- **Simulator**: `aspire.sim.cap.envs.simulators.r1pro_b1k.R1ProBehaviourLowLevel`
- **Task envs**: `aspire.sim.cap.envs.tasks.r1pro.r1pro_pickup_radio.R1ProRadioCodeEnv`
  and `aspire.sim.cap.envs.tasks.r1pro.r1pro_pickup_trash.R1ProTrashCodeEnv`
- **Control API**: `aspire.sim.cap.integrations.r1pro.control.R1ProControlApi`
- **Traced Control API**: `R1ProControlApiTraced`, registered via
  `aspire.sim.cap.integrations.trace_logger.make_traced_api`
- **OmniGibson configs**:
  `cap/third_party/b1k/OmniGibson/omnigibson/configs/r1pro_*.yaml`

## API Servers

BEHAVIOR tasks require perception servers running alongside the simulator. The
YAML configs auto-launch them, but you can also start them manually:

```bash
source cap/third_party/b1k/.venv/bin/activate

uv run --no-sync --active python -m aspire.sim.cap.serving.launch_sam3_server --device cuda --port 8114
uv run --no-sync --active python -m aspire.sim.cap.serving.launch_contact_graspnet_server --port 8115
```
