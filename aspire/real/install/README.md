# Real-Station Installation

The recovered installer is integrated here, but it deliberately does not
download private dependency snapshots, vendor authorization files, or opaque
runtime binaries. Complete the source recovery steps before running it.

## 1. Materialize dependency sources

Read [`../patches/dependencies/README.md`](../patches/dependencies/README.md).
Place reviewed source trees at:

```text
third_party/i2rt
third_party/pyroki
third_party/bundlesdf
third_party/curobo
```

PyRoki and cuRobo have public base commits and apply-checked patches. The i2rt
base was reconstructed from the former vendor snapshot and is not currently
fetchable from its public remote. BundleSDF remains blocked for public
redistribution; use an approved local source only.

RoboSuite and RoboCasa are optional for the real-robot flow. The installer
creates metadata-only stubs when those sources are absent. Install their full
sources only for the `robocasa` extra.

AnyGrasp must be obtained through its vendor process. Keep its SDK, checkpoint,
native extensions, and license archive outside Git.

### Canonical demo runtime scope

`cap/saved_scripts/yam_demo.sh full` uses `i2rt`, the local RRTConnect/MuJoCo
planner, the four-camera Portal, SAM3, and BundleSDF. It does not call AnyGrasp,
cuRobo, PyRoki, RoboCasa, voice services, or an LLM provider. The recovered
installer still provisions the broader development environment; the focused
runtime and health checks are:

```bash
bash tools/yam_demo_preflight.sh
bash tmux/launch_yam_demo_services.sh --no-attach
```

On a station where the reviewed environment is already installed, those
commands are the preferred path and the full installer does not need to be
rerun.

## 2. Configure the station

```bash
cp .forge_env.example .forge_env
$EDITOR .forge_env
```

At minimum, set `YAM_STATION_CALIBRATED_XML` to an approved local calibration.
Use `ASPIRE_STATION` or copy `robot/local_station.toml.example` to the ignored
`robot/local_station.toml`.

Run the read-only workstation inventory before installation:

```bash
bash tools/non_motion_preflight.sh
```

## 3. Install

The full installer targets Ubuntu 22.04, Python 3.11, CUDA 12.8, and a
compatible NVIDIA driver:

```bash
bash install/install_cap.sh
```

It performs the recovered two-pass `uv sync`, builds BundleSDF from locked
inputs when no usable local runtime exists, installs local runtime libraries,
downloads gated models after local authentication, and writes ignored runtime
environment files. It never overwrites the operator-maintained `.forge_env`;
generated paths go to `.forge_env.runtime` and credentials remain in
`.forge_env.secrets`.

Important controls:

- `SKIP_HF_MODELS=1` skips gated Hugging Face downloads.
- `INSTALL_ZED_SDK=0` skips the native ZED SDK.
- `ZED_SDK_SHA256` is required before downloading a ZED installer.
- `NODE_SHA256` is required to install the optional Node.js runtime.
- `ANYGRASP_LICENSE_ZIP` and AnyGrasp credentials must remain local.

## BundleSDF lock

[`locks/bundlesdf/`](locks/bundlesdf/) records:

- Miniforge version, URL, and SHA-256;
- OpenCV and opencv_contrib commits, archive URLs, and SHA-256 values;
- the recovered 300-package conda solver result for audit; and
- compiler, Python, CUDA, PCL, and GPU architecture settings.

The recovered explicit conda file contains CUDA 13 solver artifacts and is not
used to create the CUDA 12.8 build environment. Until a corrected explicit
spec is reviewed, supply one explicitly:

```bash
export BUNDLESDF_CONDA_EXPLICIT_SPEC=/path/to/reviewed-conda-linux-64.explicit.txt
export BUNDLESDF_CONDA_EXPLICIT_SHA256=<sha256>
```

`compile_bundlesdf.sh` verifies the corrected spec and all pinned downloaded
source inputs. The recovery metadata makes the prior resolution auditable; it
does not grant permission to redistribute recovered BundleSDF source or the
resulting library closure.

## ZED SDK

`build_zed_sdk.sh` runs a vendor installer with `sudo` and writes under
`/usr/local/zed`. Review it before use. It requires an approved
`ZED_SDK_SHA256`, does not print camera serials, and keeps station mappings in
ignored local configuration.

## Remaining limitations

- No approved public BundleSDF source is included.
- The i2rt public base commit is no longer fetchable.
- The small AnyGrasp support conda environment is version-constrained but does
  not yet have its own explicit package lock.
- The recovered BundleSDF conda solve selected CUDA 13 metadata and must be
  replaced with a reviewed CUDA 12.8-compatible explicit spec.
- Calibration XML/CAD redistribution permission is unresolved.
- AnyGrasp assets remain vendor-restricted.
- Node and ZED archive checksums must be supplied from approved sources.
