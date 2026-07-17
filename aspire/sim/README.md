# ASPIRE Simulation Workspace

This directory contains the simulation side of ASPIRE. Use it as the working root for simulator setup, code-as-policy execution, perception and control servers, experiment configs, run scripts, tests, agent runbooks, and generated outputs.

For the paper overview and project-level context, see [the repository README](../../README.md).

## Workspace Layout

| Path | Purpose |
| ---- | ------- |
| [`cap/`](cap/) | Python package imported as `aspire.sim.cap.*`; simulator wrappers, task runners, integrations, skill-library helpers, and service launchers. |
| [`env_configs/`](env_configs/) | YAML configs for LIBERO, Robosuite, and BEHAVIOR tasks. |
| [`scripts/`](scripts/) | Suite-specific and common analysis, replay, evaluation, and plotting scripts. |
| [`docs/`](docs/) | Simulation docs, task notes, and experiment references. |
| [`.claude/`](.claude/README.md) | Agent runbooks and skills for reproducing simulation experiments. |

## Supported Simulation Suites

| Suite | Environment | Notes |
| ----- | ----------- | ----- |
| LIBERO-PRO, LIBERO-90, LIBERO-Long | `.venv-libero` | Python 3.12; uses upstream LIBERO plus its pinned Robosuite fork. |
| Robosuite | `.venv-robosuite` | Python 3.10; standalone Robosuite stack. |
| BEHAVIOR-1K | External Isaac Sim environment | Uses the BEHAVIOR installer and is excluded from the uv workspace. |

## Working Conventions

Run setup, replay, eval, and agent workflows from this directory:

```bash
cd aspire/sim
export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"
```

Keep simulator dependencies isolated. The base `.venv` is the common ASPIRE runtime for repository tools, plotting, services, and non-simulator utilities; install suite extras into their dedicated suite venvs.

## Setup

ASPIRE uses [uv](https://docs.astral.sh/uv/) for dependency management. Start with the base repo setup, then set up whichever simulation suites you need. LIBERO-PRO, Robosuite, and BEHAVIOR-1K use incompatible simulator stacks and should not be installed into the same virtual environment; keep each suite in its dedicated venv.

Run all setup commands from this `aspire/sim` directory. If you are updating an older checkout that had venvs at the repository root, recreate them here instead of moving the old directories; virtualenv scripts contain absolute paths.

Choose the setup level that matches what you need:

| Level | Required steps | What it enables |
| ----- | -------------- | --------------- |
| Base tools | [Base Repo Setup](#base-repo-setup) | Imports, plotting, analysis, services, and non-simulator utilities. |
| Robosuite | Base + [Robosuite setup](#suite-setup-robosuite) | Standalone Robosuite tasks and its offscreen smoke test. |
| LIBERO | Base + [LIBERO setup](#suite-setup-libero) | LIBERO-PRO, LIBERO-90, LIBERO-Long, and the offscreen environment smoke test. |
| Perception | LIBERO + gated weights and [perception servers](#suite-setup-libero) | SAM3/GraspNet-backed replay and evaluation. |
| Paper experiments | Relevant suite + perception services | Full coordinator runbooks under [`.claude/`](.claude/README.md). |

The base and suite environments include large ML, CUDA, and simulator packages and can consume several gigabytes each. Check available disk space before syncing all environments. You do not need credentials or perception servers for the two offscreen environment smoke tests.

### Prerequisites

- Linux on x86-64. The simulator environments are not supported on macOS or Windows.
- An NVIDIA GPU with a working driver and CUDA runtime. BEHAVIOR-1K has additional Isaac Sim requirements documented in [`docs/behavior-tasks.md`](docs/behavior-tasks.md).
- Git, `curl`, and `tmux`. A C/C++ build toolchain may also be required by native dependencies.
- Enough local disk for simulator repositories, model weights, and benchmark assets.

The examples below use separate GPUs for SAM3, GraspNet, and simulation. Set the GPU IDs for your host before launching services. Processes may share a GPU only when it has enough memory for all selected models and the simulator.

### Credentials

Keep credentials in environment variables or protected files populated by your
approved secret manager; never put tokens in YAML, Markdown, shell history, or
committed files.

- SAM3 requires access to its gated Hugging Face model and an authenticated Hugging Face CLI session.
- LIBERO baseline runbooks use `NVIDIA_API_KEY` when the NVIDIA inference endpoint is selected.
- The Robosuite multimodel baseline reads one key per line from protected files
  selected with `CODEGEN_KEY_FILE` and `VDM_KEY_FILE`. Keep the files outside
  the repository and restrict them to the current user.
- Other model providers may require different variables; use the names documented by the selected experiment runbook.

### Base Repo Setup

From a fresh clone:

```bash
git clone https://github.com/NVlabs/ASPIRE.git
cd ASPIRE/aspire/sim
```

From an existing checkout at the repository root:

```bash
cd aspire/sim
```

Then initialize the path-source submodules:

```bash
# Do NOT init b1k here: BEHAVIOR-1K has its own installer and venv and is
# excluded from the uv workspace.
git submodule update --init \
  cap/third_party/sam3 \
  cap/third_party/robosuite \
  cap/third_party/libero_dependencies/robosuite \
  cap/third_party/LIBERO-PRO \
  cap/third_party/contact_graspnet_pytorch \
  cap/third_party/curobo

# Install uv if needed.
curl -LsSf https://astral.sh/uv/install.sh | sh
if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi
command -v uv
uv --version

# Common runtime for repository tools, plotting, services, and non-simulator utilities.
uv python install 3.10
uv venv .venv --python 3.10
uv sync --locked

export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"

# Confirm that the locked Torch build is compatible with this host's GPU driver.
.venv/bin/python -c \
  "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

Use `uv sync --locked` for normal setup and CI. It fails when `pyproject.toml` and
`uv.lock` disagree instead of silently changing dependency versions. Maintainers
should use `uv lock --upgrade` only when intentionally upgrading dependencies,
then rerun the suite smoke tests and commit both files together.

### Suite Setup: LIBERO

LIBERO-PRO (also includes LIBERO-90, LIBERO-Long) uses a dedicated `.venv-libero` environment because its Robosuite dependency conflicts with the standalone Robosuite stack.
Run the `source .../activate` and `uv sync --locked --active ...` commands in the same shell session; `--active` installs into whichever environment is currently activated.

```bash
git submodule update --init \
  cap/third_party/LIBERO-PRO \
  cap/third_party/contact_graspnet_pytorch \
  cap/third_party/libero_dependencies/robosuite \
  cap/third_party/curobo

uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
# `contactgraspnet` is required for the GraspNet perception server used by replay/eval;
# `dev` provides pytest for the smoke test below.
uv sync --locked --active --extra libero --extra contactgraspnet --extra dev
deactivate
```

Create the LIBERO path config. This file is required by upstream LIBERO even though
ASPIRE imports from `aspire.sim.cap`; without it, LIBERO may prompt interactively and
non-interactive scripts can fail with EOF:

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << EOF
benchmark_root: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero
bddl_files: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/bddl_files
init_states: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/init_files
datasets: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/datasets
assets: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/assets
EOF
```

Verify that LIBERO landed in the dedicated environment and not the base tools environment:

```bash
.venv-libero/bin/python3 -c "import libero, robosuite, sam3, contact_graspnet_pytorch; print('libero env ok')"
.venv/bin/python3 -c "import importlib.util; print(importlib.util.find_spec('libero'))"
# Expected second output: None
```

Run the fast offscreen environment smoke test before configuring perception servers.
It loads a LIBERO-10 task, resets it from a benchmark initial state, and takes ten
zero-action steps; it does not require model credentials or perception services:

```bash
ASPIRE_INTEGRATION_REAL=1 MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venv-libero/bin/python -m pytest tests/test_libero.py -q
```

SAM3 weights are gated. Request access to the SAM3 Hugging Face model, then authenticate from the LIBERO environment:

```bash
.venv-libero/bin/hf auth login
```

Perception servers are required for LIBERO-PRO replay/eval, but the experiment coordinator normally starts and verifies them during preflight. Manual prelaunch is optional, mainly useful when sharing one server set across multiple runs. If you prelaunch them manually, keep them in a persistent `tmux` pane so they survive SSH disconnects:

```bash
tmux new -s aspire-perception
cd "$ASPIRE_ROOT"
export ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3
export SAM3_GPU=0
export GRASPNET_GPU=1
export SIM_GPU=3
bash scripts/common/start_perception_servers.sh --no-molmo \
  --gpu-sam3 "$SAM3_GPU" --gpu-graspnet "$GRASPNET_GPU"

for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
# For these servers, 404 on /health is expected and means the process is responding; 000 means down.
```

Optional smoke test after the servers are up:

```bash
ASPIRE_ROOT="$(pwd)" MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
CUDA_VISIBLE_DEVICES="$SIM_GPU" .venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite libero_goal_swap \
  --args.task put_the_bowl_on_the_stove \
  --args.trial 51 \
  --args.interactive \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.no-record-video
```

At the REPL prompt, run:

```python
obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
masks = segment_sam3_text_prompt(rgb, "bowl")
print(rgb.shape, len(masks))
exit()
```

Use `.venv-libero/bin/python3` for LIBERO eval runners and perception servers. Use `.venv/bin/python3` only for lightweight scripts such as plotting/progress generation.

For LIBERO paper experiments, run the coordinator agent in a persistent terminal session because runs can take hours or days and may dispatch background subagents:

```bash
tmux new -s aspire-libero
cd "$ASPIRE_ROOT"
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Launch Claude Code or another compatible CLI agent here, then follow .claude/libero/CLAUDE.md.
```

### Suite Setup: Robosuite

Use this path for standalone Robosuite tasks. Do not install this into `.venv` or `.venv-libero`; keep `.venv` base-only and create `.venv-robosuite` for this suite.

```bash
git submodule update --init cap/third_party/robosuite

uv venv .venv-robosuite --python 3.10
source .venv-robosuite/bin/activate
# Run this in the same shell as the activation above so --active targets .venv-robosuite.
uv sync --locked --active --extra robosuite --extra dev
deactivate
```

Quick offscreen smoke test:

```bash
ASPIRE_INTEGRATION_REAL=1 MUJOCO_GL=egl \
  .venv-robosuite/bin/python -m pytest tests/test_robosuite_setup.py -q
```

Optional longer demo/eval launch. This config is not just a setup check: it is configured for
100 trials, 5 workers, and multimodel generation.

```bash
source .venv-robosuite/bin/activate
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml
```

Robosuite fix-loop and traced replay configs use the same SAM3, GraspNet, and
PyRoKi perception servers as LIBERO. Start those servers from a persistent tmux
session before replay/eval; `.venv-libero` can be used as the perception server
environment when it was synced with `--extra contactgraspnet`.

### Setup Troubleshooting

#### uv version and native build dependencies

This project uses `tool.uv.extra-build-dependencies` for legacy native packages.
Older uv releases may not apply those build dependencies correctly; uv 0.8.14 is
known to require the fallback below in this setup. If Contact-GraspNet reports
`No module named 'numpy'`, or `egl-probe` fails because of CMake policy compatibility,
first update uv and retry the locked sync.

If updating uv is not possible, seed the declared build tools into the active
LIBERO environment and disable build isolation only for the affected packages:

```bash
source .venv-libero/bin/activate
uv pip install "numpy==1.26.4" "cmake<3.27" setuptools wheel
uv sync --locked --active --extra libero --extra contactgraspnet --extra dev \
  --no-build-isolation-package contact-graspnet-pytorch \
  --no-build-isolation-package egl-probe
deactivate
```

#### Interrupted submodule checkout

Inspect the pinned checkout state when a clone or setup command is interrupted:

```bash
git submodule status
```

A leading `-` means the submodule is not initialized; a leading `+` means its
working tree is not at the commit pinned by this repository. Retry only the
affected path:

```bash
git submodule update --init cap/third_party/<affected-submodule>
```

Do not initialize `cap/third_party/b1k` as part of base setup. It has a separate
installer described below.

#### GPU compatibility

If the Torch check reports `available: False`, compare the locked Torch CUDA build
shown by `torch.version.cuda` with the host NVIDIA driver before launching GPU
services. Simulator dependency installation can succeed even when the driver is
too old to initialize the locked Torch build.

### Suite Setup: BEHAVIOR-1K

Use this path for BEHAVIOR-1K / OmniGibson tasks. BEHAVIOR installs its own environment under `cap/third_party/b1k/.venv`.

```bash
git submodule update --init cap/third_party/b1k cap/third_party/curobo

cd cap/third_party/b1k
./uv_install.sh --dataset
cd ../../..
```

The installer downloads robot assets and BEHAVIOR-1K scene/object assets used by the supported tasks. For non-interactive installation, use `./uv_install.sh --dataset --accept-dataset-tos`.

On headless GPU servers, install the EGL runtime and copy the cuRobo headers needed for first-time JIT compilation:

```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1

source cap/third_party/b1k/.venv/bin/activate
cp cap/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

Do not delete system Vulkan ICD files as a general setup step. If Isaac Sim reports duplicate or invalid ICDs, inspect `/etc/vulkan/icd.d/` and `/usr/share/vulkan/icd.d/`, preserve the original files, and follow the NVIDIA driver guidance for the host before changing system configuration.

Run a BEHAVIOR task:

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1

uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml
```

This branch ships configs and runbooks for radio pickup and soda-can pickup.
B1K setup is host-dependent; treat the first run on a new machine as smoke
testing. See [`docs/behavior-tasks.md`](docs/behavior-tasks.md) for the supported
configs.

Isaac Sim uses `OMNIGIBSON_GPU_ID`, not `CUDA_VISIBLE_DEVICES`, for GPU selection.

---

## Reproducing the Paper Experiments

The public paper reproduction runbooks are organized by suite under [`.claude/`](.claude/README.md). Complete the base setup and the suite setup for each experiment you want to run, then open the matching suite guide. These experiments are agent-driven: Claude Code or a compatible CLI agent reads the suite guide, dispatches per-task subagents, and runs the replay/eval scripts. They are not a single monolithic CLI command.

General workflow:

1. Complete [Base Repo Setup](#base-repo-setup) and the relevant suite setup.
2. Start a persistent terminal session such as `tmux new -s aspire-<suite>`.
3. Launch Claude Code or another compatible CLI agent from `$ASPIRE_ROOT`.
4. Have the agent read [`.claude/README.md`](.claude/README.md), the suite `CLAUDE.md`, and the target experiment `INSTRUCTIONS.md`.
5. Let the coordinator perform suite preflight. If you manually prelaunch long-running services, keep them in another persistent `tmux` pane.

### Suite Runbooks

| Suite | Status in this branch | Suite guide | Paper experiments |
|---|---|---|---|
| Robosuite | Available | [`.claude/robosuite/CLAUDE.md`](.claude/robosuite/CLAUDE.md) | Fix Loop; Training Law |
| LIBERO | Available | [`.claude/libero/CLAUDE.md`](.claude/libero/CLAUDE.md) | Fix Loop; Fix Loop + Evolutionary Search; Zero-Shot Transfer; Library-Size Scaling; Inference-Time Scaling |
| BEHAVIOR-1K | Available | [`.claude/behavior/CLAUDE.md`](.claude/behavior/CLAUDE.md) | Fix Loop |

### Available Experiment Guides

| Suite | Experiment | Guide |
|---|---|---|
| Robosuite | Fix Loop | [`.claude/robosuite/fix-loop/INSTRUCTIONS.md`](.claude/robosuite/fix-loop/INSTRUCTIONS.md) |
| Robosuite | Training Law | [`.claude/robosuite/training-law/INSTRUCTIONS.md`](.claude/robosuite/training-law/INSTRUCTIONS.md) |
| LIBERO | Fix Loop | [`.claude/libero/fix-loop/INSTRUCTIONS.md`](.claude/libero/fix-loop/INSTRUCTIONS.md) |
| LIBERO | Fix Loop + Evolutionary Search | [`.claude/libero/evosearch/INSTRUCTIONS.md`](.claude/libero/evosearch/INSTRUCTIONS.md) |
| LIBERO | Zero-Shot Transfer (LIBERO-90 -> LIBERO-Long-Pro) | [`.claude/libero/zeroshot-transfer/INSTRUCTIONS.md`](.claude/libero/zeroshot-transfer/INSTRUCTIONS.md) |
| LIBERO | Library-Size Scaling | [`.claude/libero/library-size-scaling/INSTRUCTIONS.md`](.claude/libero/library-size-scaling/INSTRUCTIONS.md) |
| LIBERO | Inference-Time Scaling | [`.claude/libero/inference-time-scaling/INSTRUCTIONS.md`](.claude/libero/inference-time-scaling/INSTRUCTIONS.md) |
| BEHAVIOR-1K | Fix Loop | [`.claude/behavior/fix-loop/INSTRUCTIONS.md`](.claude/behavior/fix-loop/INSTRUCTIONS.md) |

---

## Documentation

| Guide | Contents |
| ----- | -------- |
| [Adding Environments](docs/adding-environments.md) | Creating simulators, task environments, YAML configs |
| [Adding APIs](docs/adding-apis.md) | Implementing and registering new robot control APIs |
| [Configuration](docs/configuration.md) | YAML format, CLI flags, runtime options |
| [LIBERO Suite Reference](docs/libero-tasks.md) | LIBERO/LIBERO-Pro task families, env conventions, experiment entrypoints |
| [BEHAVIOR Tasks](docs/behavior-tasks.md) | Setup, R1Pro tasks, expected baselines, environment variables |
| [Skill Library Compilation](scripts/common/skill_library_compilation/README.md) | Legacy/offline utilities for analyzing eval outputs |
| [Experiment Prompts & Skills](.claude/README.md) | Per-suite agent prompts, skill snapshots, and run instructions for each paper experiment |

> **Reproducing a result, or reusing a prompt/skill?** The exact agent prompts,
> skill snapshots, and run instructions for each experiment live under
> [`.claude/`](.claude/README.md), organized by task suite.
> Start at [`.claude/README.md`](.claude/README.md).

## License

ASPIRE-owned simulation code is available under the project-level
[Apache License 2.0](../../LICENSE). Inherited code, copied material,
dependency patches, Git submodules, models, datasets, and assets retain their
original terms. Review the repository's
[third-party licenses](../../THIRD_PARTY_LICENSES.md) and
[third-party notices](../../THIRD_PARTY_NOTICES.md) before using or
redistributing the full stack.
