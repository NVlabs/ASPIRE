#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Idempotent BEHAVIOR-1K installer for the ASPIRE simulation workspace.
#
# This is an ASPIRE-owned layer over the upstream BEHAVIOR installer
# (cap/third_party/b1k/uv_install.sh). It keeps the b1k submodule clean: the
# upstream installer is invoked as-is for the Isaac Sim wheel set, and every
# deviation needed to make it work is applied from this side.
#
# Deviations from a bare `./uv_install.sh --dataset --accept-dataset-tos`, all
# observed on a clean host (see docs/behavior-tasks.md § Troubleshooting):
#
#   1. uv_install.sh never creates the venv it installs into.
#   2. Its verification step imports omnigibson, which imports cv2. Nothing in
#      the dependency chain installs OpenCV, so the installer aborts there under
#      `set -e` and silently skips curobo, pyroki, and the perception deps.
#   3. Isaac Sim pins no torch version, so resolution picks a cu13 build that
#      needs a CUDA 13 toolkit and compiles extensions as C++20, where curobo's
#      helper_math.h collides with std::lerp. We pin the validated cu124 stack
#      before the upstream installer runs.
#   4. Its SAM3 block probes a stale pre-rename path (capx/third_party/sam3) and
#      silently installs nothing.
#   5. contact_graspnet_pytorch imports numpy in setup.py without declaring it,
#      so it only builds with --no-build-isolation.
#   6. setuptools >= 81 removed pkg_resources, which sam3's model_builder needs.
#   7. ASPIRE's own runtime deps are never installed into the B1K venv, and
#      `uv sync` is forbidden there, so they are installed explicitly here.
#
# Safe to re-run: completed steps are detected and skipped, and downloaded
# assets are never deleted.

set -euo pipefail

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SIM_ROOT}/../.." && pwd)"
B1K_ROOT="${SIM_ROOT}/cap/third_party/b1k"

# ---- Validated stack (see docs/behavior-tasks.md § Tested configuration) ----
PYTHON_VERSION="3.10"
TORCH_VERSION="2.6.0+cu124"
TORCHVISION_VERSION="0.21.0+cu124"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
NUMPY_VERSION="1.26.4"          # OmniGibson requires numpy<2
CUROBO_COMMIT="cbaf7d32436160956dad190a9465360fad6aba73"
TORCH_CUDA_ARCH_LIST_DEFAULT="8.9"   # L40 / Ada. Override with TORCH_CUDA_ARCH_LIST.

VENV_PATH="${B1K_ROOT}/.venv"
ACCEPT_DATASET_LICENSE=false
SKIP_DATASETS=false
SKIP_VERIFY=false
FORCE_CUROBO=false
GPU_ID="${OMNIGIBSON_GPU_ID:-0}"

usage() {
  cat <<EOF
Usage: scripts/setup_behavior.sh [OPTIONS]

Options:
  --accept-dataset-license  Accept the BEHAVIOR-1K dataset license and download
                            the assets. Required for a usable install; omitting
                            it is a hard error unless --skip-datasets is passed.
  --skip-datasets           Do not download or check datasets.
  --venv PATH               Virtual environment path
                            (default: cap/third_party/b1k/.venv).
  --gpu-id N                GPU used by the verification run (default: 0).
  --force-curobo            Rebuild curobo even if it already imports.
  --skip-verify             Skip the mandatory verification step.
  -h, --help                Show this help.

Run from anywhere; paths resolve relative to the repository.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-dataset-license) ACCEPT_DATASET_LICENSE=true; shift ;;
    --skip-datasets) SKIP_DATASETS=true; shift ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --force-curobo) FORCE_CUROBO=true; shift ;;
    --skip-verify) SKIP_VERIFY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# =========================================================================
# Preflight
# =========================================================================
log "Preflight"

command -v uv >/dev/null || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git >/dev/null || die "git not found"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; BEHAVIOR-1K requires an NVIDIA GPU"

if [[ -n "${EXP_PATH:-}${CARB_APP_PATH:-}${ISAAC_PATH:-}" ]]; then
  die "Existing Isaac Sim environment variables detected (EXP_PATH/CARB_APP_PATH/ISAAC_PATH). Unset them and retry."
fi

# curobo compiles CUDA extensions, so a CUDA 12.x toolkit must be present. The
# torch build above is cu124; torch permits a minor-version mismatch (warning)
# but not a major one, so a CUDA 13 toolkit alone will not work.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 \
                   /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      cuda_major="$("${candidate}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"
      if [[ "${cuda_major}" == "12" ]]; then CUDA_HOME="${candidate}"; break; fi
    fi
  done
fi
[[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]] || \
  die "No CUDA 12.x toolkit with nvcc found. Install one or set CUDA_HOME explicitly."
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "CUDA_HOME=${CUDA_HOME} ($("${CUDA_HOME}/bin/nvcc" --version | tail -2 | head -1 | tr -s ' '))"

if ! ldconfig -p 2>/dev/null | grep -q "libEGL\.so"; then
  die "libEGL not found. On headless hosts: sudo apt-get install -y libegl1 libgl1"
fi

if [[ "${SKIP_DATASETS}" == false && "${ACCEPT_DATASET_LICENSE}" == false ]]; then
  die "BEHAVIOR-1K assets require accepting the dataset license.
Re-run with --accept-dataset-license (or --skip-datasets to install code only)."
fi

export OMNI_KIT_ACCEPT_EULA=YES
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${TORCH_CUDA_ARCH_LIST_DEFAULT}}"
export GIT_LFS_SKIP_SMUDGE=1
export MAX_JOBS="${MAX_JOBS:-32}"

# =========================================================================
# Submodules and verified third-party patches
# =========================================================================
log "Initializing submodules"
git -C "${REPO_ROOT}" submodule update --init \
  aspire/sim/cap/third_party/b1k \
  aspire/sim/cap/third_party/curobo \
  aspire/sim/cap/third_party/sam3 \
  aspire/sim/cap/third_party/contact_graspnet_pytorch

# Verifies the pinned revision and the three patched files byte-for-byte, and is
# idempotent (it re-checks rather than re-applies).
log "Applying and verifying the Contact-GraspNet patch"
bash "${SIM_ROOT}/scripts/common/apply_contact_graspnet_patch.sh"

# BEHAVIOR-1K itself needs no source patches at this pin; see patches/b1k/README.md.

# =========================================================================
# Virtual environment
# =========================================================================
if [[ -x "${VENV_PATH}/bin/python" ]]; then
  log "Reusing existing venv: ${VENV_PATH}"
else
  log "Creating venv: ${VENV_PATH} (Python ${PYTHON_VERSION})"
  uv python install "${PYTHON_VERSION}"
  uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}"
fi

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
export VIRTUAL_ENV="${VENV_PATH}"
PY="${VENV_PATH}/bin/python"

have() { "${PY}" -c "import $1" >/dev/null 2>&1; }

# The upstream installer builds curobo *with* build isolation, so its throwaway
# build environment resolves its own torch. Unconstrained, that picks a cu13
# build and reintroduces the C++20 std::lerp collision even though the venv holds
# the correct torch. Constrain build environments to the validated pins.
BUILD_CONSTRAINTS="${VENV_PATH}/aspire-build-constraints.txt"
cat > "${BUILD_CONSTRAINTS}" <<EOF
torch==${TORCH_VERSION}
numpy==${NUMPY_VERSION}
EOF
export UV_BUILD_CONSTRAINT="${BUILD_CONSTRAINTS}"
export UV_EXTRA_INDEX_URL="${TORCH_INDEX}"

# =========================================================================
# Pin the validated stack BEFORE the upstream installer runs
# =========================================================================
# uv does not upgrade an already-satisfied requirement, so pinning torch here
# keeps Isaac Sim's unpinned "torch" dependency from pulling a cu13 build.
log "Pinning validated base stack (torch ${TORCH_VERSION}, numpy ${NUMPY_VERSION}, setuptools<81)"
uv pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --extra-index-url "${TORCH_INDEX}"
uv pip install \
  "numpy==${NUMPY_VERSION}" \
  "setuptools<81" wheel ninja \
  opencv-python-headless

# =========================================================================
# Upstream installer: OmniGibson, bddl3, and the pinned Isaac Sim wheel set
# =========================================================================
if have isaacsim && have omnigibson; then
  log "OmniGibson and Isaac Sim already installed; skipping upstream installer"
else
  log "Running upstream uv_install.sh (no --dataset; assets are handled below)"
  ( cd "${B1K_ROOT}" && ./uv_install.sh )
fi

# =========================================================================
# Pieces the upstream installer cannot install correctly
# =========================================================================
log "Installing SAM3 from the vendored submodule"
uv pip install "${SIM_ROOT}/cap/third_party/sam3" --no-deps
uv pip install iopath einops timm "ftfy==6.1.1" decord pycocotools

log "Installing Contact-GraspNet (no build isolation: undeclared numpy build dep)"
uv pip install --no-build-isolation -e "${SIM_ROOT}/cap/third_party/contact_graspnet_pytorch" --no-deps
uv pip install pyrender open3d

log "Installing ASPIRE runtime dependencies"
uv pip install \
  tyro omegaconf pyyaml fastapi uvicorn openai transformers \
  pydantic rich mediapy msgpack msgpack_numpy requests cloudpickle \
  "imageio[ffmpeg]" viser

# The upstream installer may have re-pulled a newer torch through a transitive
# dependency; re-assert the validated pin before anything is compiled against it.
log "Re-asserting the validated torch pin"
uv pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  "numpy==${NUMPY_VERSION}" "setuptools<81" \
  --extra-index-url "${TORCH_INDEX}"

if [[ "${FORCE_CUROBO}" == true ]] || ! "${PY}" -c "from curobo.curobolib import geom_cu" >/dev/null 2>&1; then
  log "Building curobo CUDA extensions (arch ${TORCH_CUDA_ARCH_LIST})"
  uv pip install --no-build-isolation --force-reinstall --no-deps \
    "nvidia_curobo@git+https://github.com/StanfordVL/curobo@${CUROBO_COMMIT}"
else
  log "curobo CUDA extensions already built; skipping"
fi

if ! have pyroki; then
  log "Installing pyroki"
  uv pip install "pyroki@git+https://github.com/chungmin99/pyroki.git"
fi

# Makes `python -m aspire.sim.cap.envs.launch_b1k` work from any directory.
log "Installing the aspire package (editable)"
uv pip install -e "${SIM_ROOT}" --no-deps

# =========================================================================
# Datasets (never deletes existing downloads)
# =========================================================================
if [[ "${SKIP_DATASETS}" == true ]]; then
  log "Skipping datasets (--skip-datasets)"
else
  log "Datasets"
  DATA_PATH="$("${PY}" -c 'from omnigibson.macros import gm; print(gm.DATA_PATH)')"
  echo "DATA_PATH=${DATA_PATH}"

  ROBOT_ASSETS="${DATA_PATH}/omnigibson-robot-assets"
  IK_URDF_REL="models/r1pro/urdf/r1pro_ik.urdf"

  # The b1k submodule git-tracks the r1pro_ik.urdf overlay *inside* this dataset
  # directory, so ${ROBOT_ASSETS}/models exists on every fresh clone before any
  # download runs. Testing for the directory therefore reports the ~2.4 GB
  # download as already done on a clean host, and the run later dies with
  # FileNotFoundError on models/r1pro/usd/r1pro.usda. Test for real payload
  # instead: any file other than the tracked overlay.
  if [[ -n "$(find "${ROBOT_ASSETS}" -type f ! -path "${ROBOT_ASSETS}/${IK_URDF_REL}" -print -quit 2>/dev/null)" ]]; then
    echo "robot assets present; skipping"
  else
    # Call the unpacker directly: download_omnigibson_robot_assets() guards on
    # os.path.exists(<dataset dir>), which the same tracked file defeats, so it
    # would print "Assets already downloaded." and do nothing. extractall()
    # merges into the existing tree, so the overlay survives; the restore step
    # below is the backstop.
    "${PY}" -c "from omnigibson.utils.asset_utils import download_and_unpack_zipped_dataset; download_and_unpack_zipped_dataset('omnigibson-robot-assets')"
  fi

  # r1pro_ik.urdf has the mobile-base and gripper joints fixed for IK-only use.
  # The upstream installer deletes and re-downloads the asset tree to restore it;
  # we only ever add the file back.
  if [[ ! -f "${ROBOT_ASSETS}/${IK_URDF_REL}" ]]; then
    if [[ -f "${B1K_ROOT}/assets/r1pro_ik.urdf" ]]; then
      mkdir -p "${ROBOT_ASSETS}/$(dirname "${IK_URDF_REL}")"
      cp -a "${B1K_ROOT}/assets/r1pro_ik.urdf" "${ROBOT_ASSETS}/${IK_URDF_REL}"
      echo "installed r1pro_ik.urdf from repo assets"
    else
      die "r1pro_ik.urdf missing and no repo copy at ${B1K_ROOT}/assets/r1pro_ik.urdf"
    fi
  fi

  if [[ -d "${DATA_PATH}/behavior-1k-assets" ]]; then
    echo "BEHAVIOR-1K assets present; skipping"
  else
    "${PY}" -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
  fi

  if [[ -d "${DATA_PATH}/2025-challenge-task-instances" ]]; then
    echo "2025 challenge task instances present; skipping"
  else
    "${PY}" -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
  fi
fi

# =========================================================================
# Mandatory verification
# =========================================================================
if [[ "${SKIP_VERIFY}" == true ]]; then
  log "Skipping verification (--skip-verify)"
  echo "Environment NOT verified. Run: scripts/verify_behavior.py"
  exit 0
fi

log "Verifying the environment"
OMNIGIBSON_GPU_ID="${GPU_ID}" "${PY}" "${SIM_ROOT}/scripts/verify_behavior.py" --gpu-id "${GPU_ID}"

log "BEHAVIOR-1K setup complete and verified"
