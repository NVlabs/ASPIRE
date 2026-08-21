# Real-Station Setup

> **Current status:** `aspire/real` contains the YAM deployment code and
> operator workflows, but a fresh public clone is not yet self-contained.
> Several vendor repositories, licensed artifacts, and station-specific files
> must be recovered or replaced before installation can succeed.

For an agent operating on the known-good workstation, use the exact search and
handoff checklist in [`WORKSTATION_RECOVERY.md`](WORKSTATION_RECOVERY.md).
The reviewed recovery artifacts and installer flow are documented in
[`install/README.md`](install/README.md).

Run all commands in this guide from the real-robot workspace:

```bash
cd aspire/real
```

## Supported Environment

The checked-in dependency configuration currently expects:

- Linux x86-64 for the full robot and perception stack
- Python 3.11 exactly
- an NVIDIA GPU and a CUDA toolchain compatible with the PyTorch CUDA 12.8
  packages selected in `pyproject.toml`
- YAM arm CAN interfaces and the station's camera devices
- the Stereolabs ZED SDK when ZED cameras are enabled
- Intel RealSense drivers when RealSense cameras are enabled

The recovered workstation baseline is Ubuntu 22.04.5, kernel 6.8, Python
3.11.13, CUDA toolkit 12.8.93, NVIDIA driver 580.82.09, Torch 2.8.0+cu128,
RealSense Python 2.56.5.9235, and four D405 cameras on firmware 5.12.14.100.
See [`docs/recovery/HARDWARE_PREFLIGHT.md`](docs/recovery/HARDWARE_PREFLIGHT.md)
for the serial-free report. Native ZED SDK and arm motor firmware versions
remain unresolved.

The launch scripts also use these command-line tools:

```text
uv  tmux  curl  ffmpeg  ffprobe  ss  fuser  pgrep  nohup
```

Some optional workflows additionally expect `node`, `npm`, `nvcc`, and SSH.

## Workstation Recovery Checklist

Use this table when inspecting the working station. Do not commit credentials,
private keys, personal access tokens, or third-party license files to the
public repository.

| Item | Expected location | Why it is needed | Public-release action |
| --- | --- | --- | --- |
| Runtime environment | `.forge_env.example` | Documents required environment variables and paths | Copy to `.forge_env`, fill locally, and keep the populated file private |
| Station configuration | `robot/local_station.toml.example` | Selects the station profile without relying on a hostname | Copy to `robot/local_station.toml` and keep the populated file local |
| Calibrated station model | site-specific path assigned to `YAM_STATION_CALIBRATED_XML` | Real motion planning and station geometry | Document how operators generate or obtain it; publish only if redistribution is allowed |
| i2rt | `third_party/i2rt` | Core YAM dependency | Add a public submodule, package dependency, or documented download |
| PyRoki | `third_party/pyroki` | IK and trajectory service | Add a public submodule, package dependency, or documented download |
| BundleSDF | `third_party/bundlesdf` | Object tracking | Add a public submodule or documented upstream install |
| robosuite | `third_party/robosuite` | Optional simulation/robocasa dependency | Reuse an approved public source or remove the unused real-workspace dependency |
| robocasa | `third_party/robocasa` | Optional robocasa dependency | Add a public source or remove the unused dependency |
| cuRobo | `third_party/curobo` | Motion planning | Add a public submodule/package source and document the tested revision |
| AnyGrasp service | Operator-supplied `ANYGRASP_SERVICE_URL` | Grasp generation | Obtain and run the vendor service separately; ASPIRE distributes only its client protocol |
| BundleSDF runtime libraries | `third_party/bundlesdf_5090` | Current launcher library path | Replace with a reproducible build/install or document how it is produced |
| AnyGrasp checkpoint | External vendor installation | AnyGrasp inference | Obtain through the authorized vendor workflow; do not place it in this repository |
| AnyGrasp license archive | External vendor installation | Vendor runtime authorization | Keep private; do not place it in this repository |
| Unreferenced invalid mesh | `robot/models/station/assets/model2__1.stl` | Historical CAD export | The invalid HTML file was removed; the active calibrated XML does not reference it, so it is not an installation blocker |
| Demo media | `media/aspire_yam_station_real_demo.mp4` | README demo link | Add approved media or remove the placeholder section |

The six local sources declared in `pyproject.toml` must exist before `uv` can
resolve the environment. They are intentionally ignored today, so recovering
them locally is not enough for a reproducible public release: each needs a
public source and pinned revision, or the dependency configuration must change.

## Local Configuration

Copy the sanitized environment example and fill in only the values required
for the current station:

```bash
cp .forge_env.example .forge_env
$EDITOR .forge_env
source .forge_env
```

Likely configuration includes:

- `YAM_STATION_CALIBRATED_XML`
- `ASPIRE_STATION` or `robot/local_station.toml`
- camera backend, serial, or device mappings
- `CUDA_HOME` and runtime library paths
- model/cache locations
- provider API keys required by the selected VLM backend
- AnyGrasp checkpoint and license paths

Never commit the populated `.forge_env`. Avoid copying the working station's
file directly into `.forge_env.example`; replace secrets, usernames, hostnames,
private IPs, and absolute home-directory paths with documented placeholders.

Keep workstation usernames, hostnames, private-network endpoints, and absolute
home-directory paths in local configuration rather than committed defaults.

## Dependency Installation

Do not run this section until all required local sources have been recovered or
the manifest has been updated to public sources. See
[`patches/dependencies/README.md`](patches/dependencies/README.md) for the
recovered bases and patch series.

The integrated installer performs the required two-pass sync:

```bash
bash install/install_cap.sh
```

If BundleSDF must be built, the installer also requires
`BUNDLESDF_CONDA_EXPLICIT_SPEC` and `BUNDLESDF_CONDA_EXPLICIT_SHA256` for a
reviewed CUDA 12.8-compatible conda environment. The recovered conda solve is
kept as audit evidence but is not executable because it selected CUDA 13
metadata.

Optional extras in `pyproject.toml` should be installed only for the workflows
that require them. Record the exact command used on the working station and
confirm that `uv.lock` can be reproduced from a clean checkout.

## Safe Preflight

Before connecting to or moving the robot, run non-motion checks:

```bash
bash -n tmux/launch_yam_demo_services.sh
bash -n tmux/launch_realworld_localserver_realsense.sh
bash -n tools/run_yam_demo_service.sh
bash -n tools/yam_demo_preflight.sh
bash -n cap/saved_scripts/yam_demo.sh
python3 -m compileall -q cap robot tools experimental scripts
bash cap/saved_scripts/yam_demo.sh --help
bash tools/non_motion_preflight.sh
bash tools/yam_demo_preflight.sh
```

Then verify the environment imports without starting services:

```bash
uv run python -c "import cap, robot, tools; print('imports_ok')"
```

Check that the selected station and cameras resolve correctly before starting
arm processes. `tools/non_motion_preflight.sh` records the host/GPU/driver
inventory, while `tools/yam_demo_preflight.sh` validates the demo-specific
configuration, dependency sources, camera aliases, and follower CAN
interfaces. Neither script sends motion commands.

## Service Startup

Once installation and station configuration are complete, run the demo-only
static preflight:

```bash
bash tools/yam_demo_preflight.sh
```

It verifies the commands, Python modules, reviewed `i2rt` and BundleSDF source
trees, BundleSDF runtime libraries, calibrated station XML, camera aliases, and
follower CAN interfaces used by the canonical demo. It does not connect to arm
RPC endpoints or send motion commands.

Start the minimal demo service set:

```bash
bash tmux/launch_yam_demo_services.sh --no-attach
```

That launcher starts and health-checks both follower arm servers, the read-only
four-camera Portal on port 8300, SAM3 on port 6767, and BundleSDF on port 8119.
Confirm an already-running service set without sending motion commands with:

```bash
bash tools/yam_demo_preflight.sh --services
```

The canonical saved demo uses local RRTConnect/MuJoCo planning and BundleSDF
one-shot localization. It does not require AnyGrasp, cuRobo, PyRoki, provider
API keys, voice services, or RoboCasa. The broader
`tmux/launch_realworld_localserver_realsense.sh` launcher is only for workflows
that need those additional services.

Follow `.agents/skills/yam-server-setup/SKILL.md` for individual checks and
manual commands. Process presence alone is not sufficient.

The dependency recovery table above covers the full development workspace. For
the canonical demo specifically, the runtime blockers are `i2rt`, BundleSDF
source/runtime libraries, SAM3 model access, the calibrated station model and
meshes, the YAM arm/CAN configuration, and the four RGB-D cameras. PyRoki,
cuRobo, AnyGrasp, RoboCasa, and provider credentials are optional for this demo.

## Physical-Motion Gate

Real motion must remain an explicit operator action. Motion-capable commands
require:

```bash
export OPENFORGE_ALLOW_PHYSICAL_MOTION=1
```

Set it only after verifying the station profile, calibration, cameras, arm
health, workspace clearance, emergency-stop access, and recording. Start with
a bounded home/open or task-specific dry-run procedure rather than the full
demo. The repository does not yet provide a complete hardware commissioning
procedure.

## Release Completion Criteria

The real setup can be called reproducible when a clean Linux machine can:

1. obtain every redistributable dependency from a documented, pinned source;
2. supply licensed/private assets through documented local paths;
3. create `.forge_env` and station configuration from sanitized examples;
4. complete `uv sync` from the checked-in lockfile;
5. pass non-motion import, model, camera, and arm-health checks;
6. launch all services without internal hostnames or undocumented paths; and
7. follow a safe commissioning procedure before any autonomous demo.
