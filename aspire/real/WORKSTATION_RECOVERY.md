# Workstation Recovery Manifest

This manifest is for an agent running on the working YAM station. Its goal is
to identify the missing inputs needed to make `aspire/real` reproducible
without copying credentials, private keys, or restricted vendor licenses into
Git.

## Recovery status

The 2026-07-02 workstation handoff has been reviewed and partially integrated:

- dependency patch series are under `patches/dependencies/`;
- the installer is under `install/`;
- BundleSDF build locks are under `install/locks/bundlesdf/`;
- the calibration patch and provenance are under `patches/calibration/`; and
- the non-motion hardware preflight is `tools/non_motion_preflight.sh`.

Restricted BundleSDF source, AnyGrasp payloads, calibration/CAD files, opaque
runtime binaries, caches, and recovery archives were deliberately not copied.
This document remains useful for recollecting or independently verifying the
inputs.

Start from the workstation's known-good real-robot checkout. Record the path:

```bash
pwd
git status --short --branch 2>/dev/null || true
```

## Priority 1: Dependency Sources

The checked-in `pyproject.toml` and `uv.lock` refer to these local repositories.
For each one, find the directory, upstream URL, exact commit, active branch,
and whether the worktree has local changes.

| Expected path under `aspire/real` | Used for | What the workstation agent should return |
| --- | --- | --- |
| `third_party/i2rt` | YAM robot APIs | Upstream URL, commit SHA, branch, local patch status, license |
| `third_party/pyroki` | IK/trajectory service | Upstream URL, commit SHA, branch, local patch status, license |
| `third_party/bundlesdf` | Tracking integration | Upstream URL, commit SHA, branch, local patch status, build instructions, license |
| `third_party/robosuite` | Optional robocasa workflows | Upstream URL, commit SHA, branch, whether this dependency is still required here |
| `third_party/robocasa` | Optional robocasa workflows | Upstream URL, commit SHA, branch, whether this dependency is still required here |
| `third_party/curobo` | Motion planning | Upstream URL, commit SHA, branch, local patch status, build instructions, license |
| `third_party/anygrasp_sdk` | AnyGrasp runtime | Vendor source/version and installation instructions; do not copy restricted files |
| `third_party/bundlesdf_5090` | Current BundleSDF runtime libraries | Explain how this directory is built and whether it can be replaced by a reproducible build |

Search common checkout locations if the expected paths are absent:

```bash
find "$HOME" /opt /workspace -maxdepth 6 -type d \
  \( -name i2rt -o -name pyroki -o -name bundlesdf -o -name robosuite \
     -o -name robocasa -o -name curobo -o -name anygrasp_sdk \
     -o -name bundlesdf_5090 \) 2>/dev/null
```

For every Git dependency found, capture metadata without copying credentials:

```bash
for repo in \
  third_party/i2rt \
  third_party/pyroki \
  third_party/bundlesdf \
  third_party/robosuite \
  third_party/robocasa \
  third_party/curobo \
  third_party/anygrasp_sdk
do
  if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf '\n[%s]\n' "$repo"
    git -C "$repo" remote -v
    git -C "$repo" rev-parse HEAD
    git -C "$repo" branch --show-current
    git -C "$repo" status --short
  fi
done
```

Return the metadata first. Do not archive whole dependency worktrees until
their licenses and local modifications have been reviewed.

## Priority 2: Installation and Environment Definition

Find these files or their current equivalents:

| Item | Likely workstation location | Recovery action |
| --- | --- | --- |
| CAP installer | `install/install_cap.sh` | Return a sanitized copy or identify the replacement installer |
| Runtime environment | `.forge_env` | Return variable names and non-secret path requirements; never return secret values |
| User runtime environment | `~/.config/aspire/runtime_env.sh` or older local equivalent | Return only reusable, non-secret setup logic |
| Station selection | `robot/local_station.toml` | Return the selected profile name; do not include hostname mappings unless necessary |
| Python lock/install command | shell history, operator notes, installer | Record the exact successful `uv sync` command and extras |

Useful searches:

```bash
find "$HOME" /opt /workspace -type f \
  \( -name 'install_cap.sh' -o -name '.forge_env' \
     -o -name 'runtime_env.sh' -o -name 'local_station.toml' \) \
  2>/dev/null

# Print variable names only; this intentionally suppresses values.
if test -f .forge_env; then
  sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' \
    .forge_env | sort -u
fi
```

The repository now contains sanitized starting points:

- `.forge_env.example`
- `robot/local_station.toml.example`

Compare those examples with the workstation configuration and report missing
variable names or setup steps. Do not overwrite the examples with raw local
files.

## Priority 3: Calibration and Hardware Mapping

Locate and characterize the files and mappings used by the working station:

| Item | Runtime selector | What to record |
| --- | --- | --- |
| Calibrated station XML | `YAM_STATION_CALIBRATED_XML` | File path, generation procedure, owning project, version/date, redistribution permission |
| Camera mappings | station profile and `CAP_*CAMERA*` variables | Camera roles, backend types, udev symlinks; keep device serials local unless publication is necessary |
| CAN mappings | station profile and `YAM_*CAN_INTERFACE` variables | Stable SocketCAN/udev interface names and setup procedure |
| ZED SDK | system install | SDK version and official installation source |
| RealSense runtime | system install | `pyrealsense2`/firmware versions and udev setup |
| Arm firmware/config | motor tooling | Firmware/tool versions and any required calibration procedure |

Locate likely station descriptions:

```bash
find "$HOME" /opt /workspace -type f \
  \( -iname '*station*.xml' -o -iname '*station*.urdf' \
     -o -iname '*calibrat*.xml' -o -name 'local_station.toml' \) \
  2>/dev/null
```

Record system information needed to reproduce the environment:

```bash
uname -a
cat /etc/os-release
python3 --version
uv --version
nvidia-smi 2>/dev/null || true
nvcc --version 2>/dev/null || true
ffmpeg -version 2>/dev/null | head -1
tmux -V 2>/dev/null || true
```

Do not copy an unreviewed calibration file into Git. First establish whether it
contains proprietary geometry or workstation identifiers and whether its mesh
dependencies are redistributable.

## Priority 4: Model and Perception Assets

| Item | Expected path | Handling |
| --- | --- | --- |
| Unreferenced invalid mesh | `robot/models/station/assets/model2__1.stl` | All recovered copies were HTML and were removed; the active XML does not reference this file |
| AnyGrasp checkpoint | `checkpoint_detection.tar` | Record official download/version/checksum; keep local unless redistribution is permitted |
| AnyGrasp license | `license_*.zip` | Never commit; confirm only the expected local path and vendor setup process |
| BundleSDF libraries | `third_party/bundlesdf_5090` or configured replacement | Record build inputs and commands rather than copying opaque binaries when possible |
| Model caches | Hugging Face/cache directories | Record model identifiers and pinned revisions, not cache contents |

Search for the missing station mesh and record candidates:

```bash
find "$HOME" /opt /workspace -type f -name 'model2__1.stl' -print 2>/dev/null
find "$HOME" /opt /workspace -type f -name 'model2__1.stl' -print0 2>/dev/null \
  | xargs -0 -r file
find "$HOME" /opt /workspace -type f -name 'model2__1.stl' -print0 2>/dev/null \
  | xargs -0 -r shasum -a 256
```

A valid candidate should be a binary or ASCII stereolithography file, not HTML
or a Git hosting login page.

## Priority 5: Optional Release Material

These items do not block code recovery but should be resolved before public
release:

- approved demo media for `media/aspire_yam_station_real_demo.mp4`;
- provenance and redistribution approval for committed STL/USD/URDF assets;
- citations, notices, and licenses required by copied integration snippets;
- a non-motion hardware preflight command and expected output;
- the exact tested service launch order and troubleshooting notes.

## Do Not Copy Into Git

Do not commit or send back the raw contents of:

- `.forge_env` or other populated environment files;
- API keys, access tokens, SSH keys, cloud credentials, or model-provider keys;
- `license_*.zip` or other vendor authorization files;
- shell history;
- Codex/Claude session databases, JSONL transcripts, or agent run archives;
- robot logs, camera recordings, or operator-identifying media without review;
- private hostname/IP mappings or hardware serials unless publication is
  explicitly approved.

## Expected Handoff

The workstation agent should return:

1. a table of dependency URLs, commit SHAs, branches, licenses, and local patch
   status;
2. sanitized copies of reusable installer/configuration templates;
3. the calibrated model's provenance and generation procedure;
4. system, SDK, driver, and firmware versions;
5. checksums and licensing status for required model/perception assets; and
6. a list of anything still available only through a private or licensed
   source.

With that handoff, this repository can replace ignored local path dependencies
with pinned public sources or documented local installation steps.
