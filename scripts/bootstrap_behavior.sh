#!/usr/bin/env bash
# Bootstrap the BEHAVIOR-1K (OmniGibson on Isaac Sim) simulator runtime into .venv-behavior.
#
# BEHAVIOR-1K ships a conda-only installer (third_party/BEHAVIOR-1K/setup.sh). This script
# replays the same steps with uv so the environment is reproducible from this checkout:
#   1. Python 3.11 venv (Isaac Sim 5.1 wheels are cp311-only).
#   2. torch/numpy pinned BEFORE anything else resolves them (numpy<2 is load-bearing).
#   3. Isaac Sim 5.1.0 wheels from pypi.nvidia.com, plus upstream's post-install fixups.
#   4. bddl3 and OmniGibson[primitives] (cuRobo, compiled against a CUDA 12.x toolkit).
#   5. cap-harness itself, editable, --no-deps, plus its runtime client dependencies.
#   6. Optionally the datasets, only with --accept-dataset-tos.
# Never run `uv sync` in this environment: it would tear out the Isaac Sim wheels.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
VENV_ROOT="${CAP_HARNESS_BEHAVIOR_VENV:-$(resolve_canonical_venv "$REPO_ROOT" ".venv-behavior")}"
PYTHON_VERSION="${CAP_HARNESS_BEHAVIOR_PYTHON:-3.11}"
DATA_ROOT="${CAP_HARNESS_BEHAVIOR_DATA:-$HOME/behavior-data}"
APPDATA_ROOT="${CAP_HARNESS_BEHAVIOR_APPDATA:-$REPO_ROOT/.behavior-appdata}"
FORCE_REBUILD=0
PROFILE="${CAP_HARNESS_PROFILE:-}"   # empty: detect from the GPU's compute capability
B1K_ROOT="$REPO_ROOT/third_party/BEHAVIOR-1K"
MARKER="sim:behavior:behavior"

# ---- Validated stack (BEHAVIOR-1K v3.9.2 setup.sh) ----
ISAACSIM_VERSION="5.1.0"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
TORCHCODEC_VERSION="0.5"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX="https://pypi.nvidia.com"

ACCEPT_DATASET_TOS=0
SKIP_DATASETS=0
GPU_ID="${OMNIGIBSON_GPU_ID:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_behavior.sh [--accept-dataset-tos | --skip-datasets] [--gpu-id N]

Install the BEHAVIOR-1K simulator runtime into .venv-behavior (Python 3.11, Isaac Sim 5.1).

Options:
  --accept-dataset-tos  Accept the BEHAVIOR Data Bundle terms (non-commercial academic research)
                        and download the assets into $CAP_HARNESS_BEHAVIOR_DATA
                        (default ~/behavior-data, ~40 GB). Without this flag the datasets are
                        checked, not downloaded, and the bootstrap fails if they are missing
                        unless --skip-datasets is passed.
  --skip-datasets       Neither download nor check datasets.
  --gpu-id N            GPU the verification import uses (default: $OMNIGIBSON_GPU_ID or 0).
  --profile NAME         hardware profile (configs/profiles/NAME.json); default: auto-detected
  --rebuild              rebuild even when the fingerprint is current
  -h, --help            Show this help.

Environment:
  CAP_HARNESS_BEHAVIOR_DATA   dataset root (default ~/behavior-data)
  CAP_HARNESS_BEHAVIOR_APPDATA  Isaac Kit appdata/shader cache (default <repo>/.behavior-appdata)
  CAP_HARNESS_CUDA_HOME       CUDA 12.x toolkit used to compile cuRobo (auto-detected)
  CAP_HARNESS_PROFILE         hardware profile for the cuRobo compile arch list; default: the
                              profile whose arch matches GPU 0's compute capability (nvidia-smi)
EOF
}

while (($#)); do
  case "$1" in
    --accept-dataset-tos) ACCEPT_DATASET_TOS=1; shift ;;
    --skip-datasets) SKIP_DATASETS=1; shift ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --rebuild) FORCE_REBUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [[ "$(basename -- "$VENV_ROOT")" != ".venv-behavior" ]]; then
  die "BEHAVIOR environment basename must be exactly .venv-behavior: $VENV_ROOT"
fi
command -v uv >/dev/null || die "uv is required"
command -v git >/dev/null || die "git is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required (NVIDIA driver)"

# The cuRobo kernels are compiled for one architecture: pick the profile whose arch matches the
# GPU that will run them unless the caller chose one (cross-compiling for another host).
gpu_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_ID:-0}" 2>/dev/null | head -1 | tr -d '[:space:]')"
if [[ -z "$PROFILE" ]]; then
  [[ -n "$gpu_capability" ]] || die "could not read GPU ${GPU_ID:-0}'s compute capability; pass --profile <name>"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU_ID:-0}" 2>/dev/null | head -1)"
  PROFILE="$(PYTHONPATH="$REPO_ROOT/src" python3 - "$REPO_ROOT" "$gpu_capability" "$gpu_name" <<'PYCAP'
import sys
from pathlib import Path
from cap_harness import environments
repo, capability, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
print(environments.profile_for_gpu(environments.arch_for_compute_capability(capability), name, repo) or "")
PYCAP
)"
  [[ -n "$PROFILE" ]] || die "no hardware profile compiles for GPU ${GPU_ID:-0} (compute capability $gpu_capability); add configs/profiles/<name>.json or pass --profile"
  printf 'profile: %s (auto-detected: GPU %s is "%s", compute capability %s)\n' "$PROFILE" "${GPU_ID:-0}" "$gpu_name" "$gpu_capability"
elif [[ -n "$gpu_capability" ]]; then
  profile_arch="$(PYTHONPATH="$REPO_ROOT/src" python3 -c 'import sys; from pathlib import Path; from cap_harness import environments; print(environments.load_profile(sys.argv[2], Path(sys.argv[1]))["arch"])' "$REPO_ROOT" "$PROFILE")"
  gpu_arch="$(PYTHONPATH="$REPO_ROOT/src" python3 -c 'import sys; from cap_harness import environments; print(environments.arch_for_compute_capability(sys.argv[1]))' "$gpu_capability")"
  if [[ "$profile_arch" != "$gpu_arch" ]]; then
    printf 'warning: profile %s compiles cuRobo for %s but GPU %s is %s; the kernels will not run here\n' "$PROFILE" "$profile_arch" "${GPU_ID:-0}" "$gpu_arch" >&2
  fi
fi
if [[ -n "${EXP_PATH:-}${CARB_APP_PATH:-}${ISAAC_PATH:-}" ]]; then
  die "existing Isaac Sim environment variables detected (EXP_PATH/CARB_APP_PATH/ISAAC_PATH); unset them"
fi
ldconfig -p 2>/dev/null | grep -q 'libEGL.so' || die "libEGL.so not found; install libegl1"

# CUDA 12.x toolkit: cuRobo compiles CUDA extensions against the venv's torch (cu128), and torch
# refuses a major-version mismatch, so a CUDA 13 toolkit alone cannot build it.
detect_cuda_home() {
  local candidate
  for candidate in "${CAP_HARNESS_CUDA_HOME:-}" "$HOME/cuda-12.8" "$HOME/cuda-12.6" \
      /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 \
      /usr/local/cuda-12 /usr/local/cuda; do
    [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
    if "$candidate/bin/nvcc" --version | grep -qE 'release 12\.'; then
      printf '%s\n' "$candidate"; return 0
    fi
  done
  return 1
}
CUDA_HOME="$(detect_cuda_home)" || die "no CUDA 12.x toolkit with nvcc found; set CAP_HARNESS_CUDA_HOME"
CUDA_ARCH_LIST="$(PYTHONPATH="$REPO_ROOT/src" python3 - "$REPO_ROOT" "$PROFILE" <<'PY'
import sys
from pathlib import Path
from cap_harness import environments
repo, name = Path(sys.argv[1]), sys.argv[2]
profile = environments.load_profile(name, repo)
print(environments.cuda_arch_list(profile["arch"], repo))
PY
)"

# uv caches built wheels by source, not by TORCH_CUDA_ARCH_LIST: keep one cache per architecture
# so a cuRobo wheel compiled for another card is never reused (and reinstall it on every rebuild).
UV_CACHE_ROOT="${CAP_HARNESS_BEHAVIOR_CACHE:-/tmp/${USER:-root}/uv-cache-behavior-${CUDA_ARCH_LIST//[^0-9A-Za-z]/_}}"

git -C "$REPO_ROOT" submodule update --init third_party/BEHAVIOR-1K
expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dependencies"]["behavior_1k"]["commit"])' "$REPO_ROOT/configs/dependency-lock.json")"
actual="$(git -C "$B1K_ROOT" rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || die "BEHAVIOR-1K checkout mismatch: $actual != $expected"

mkdir -p "$(dirname -- "$VENV_ROOT")" "$UV_CACHE_ROOT" "$DATA_ROOT" "$APPDATA_ROOT"
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "$PROFILE" "$MARKER" "third_party/BEHAVIOR-1K")"
if (( FORCE_REBUILD )); then
  INSTALL=1
elif venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  # The fingerprint covers sources and the profile, not what a cached wheel was compiled for:
  # check the installed cuRobo kernels against the profile's architecture before trusting it.
  installed_arches="$(PYTHONPATH="$REPO_ROOT/src" "$VENV_ROOT/bin/python" - <<'PYARCH' 2>/dev/null || true
import pathlib
from cap_harness.environments import compiled_cuda_arches
try:
    import curobo.curobolib as lib
except Exception:
    print("missing")
else:
    found = compiled_cuda_arches(pathlib.Path(lib.__file__).parent.glob("*.so"))
    print(" ".join(sorted(found)) or "unknown")
PYARCH
)"
  profile_arch="$(PYTHONPATH="$REPO_ROOT/src" python3 -c 'import sys; from pathlib import Path; from cap_harness import environments; print(environments.load_profile(sys.argv[2], Path(sys.argv[1]))["arch"])' "$REPO_ROOT" "$PROFILE")"
  if [[ "$installed_arches" == "missing" || ( "$installed_arches" != "unknown" && " $installed_arches " != *" $profile_arch "* ) ]]; then
    printf 'installed cuRobo kernels (%s) do not match profile %s (%s); rebuilding\n' "$installed_arches" "$PROFILE" "$profile_arch"
    INSTALL=1
  else
    printf '%s is up to date (fingerprint %s, cuRobo kernels %s); skipping rebuild\n' "$MARKER" "${FINGERPRINT:0:12}" "$installed_arches"
    INSTALL=0
  fi
else
  INSTALL=1
fi

export UV_CACHE_DIR="$UV_CACHE_ROOT"
PIP=(uv pip install --python "$VENV_ROOT/bin/python")
CONSTRAINTS="$VENV_ROOT/.cap-behavior-constraints.txt"

if (( INSTALL )); then
  uv venv --allow-existing --python "$PYTHON_VERSION" "$VENV_ROOT"
  "$VENV_ROOT/bin/python" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), f"Isaac Sim 5.1 wheels are cp311-only, got {sys.version}"
PY
  cat >"$CONSTRAINTS" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
torchcodec==${TORCHCODEC_VERSION}
numpy<2
setuptools>=71,<81
EOF
  export UV_CONSTRAINT="$CONSTRAINTS"
  # 2. Pins first, so nothing below resolves a cu13 torch or numpy 2.
  "${PIP[@]}" --index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" "torchcodec==${TORCHCODEC_VERSION}"
  "${PIP[@]}" "numpy<2" "setuptools>=71,<81" wheel ninja psutil
  # 3. Isaac Sim 5.1 (the same wheel set setup.sh downloads one by one).
  "${PIP[@]}" --extra-index-url "$NVIDIA_INDEX" --index-strategy unsafe-best-match \
    "isaacsim[all,extscache]==${ISAACSIM_VERSION}"
  # Upstream post-install fixups (setup.sh): Isaac bundles stale websockets/packaging copies that
  # shadow the venv's, and pins a cffi that mismatches OmniGibson's.
  "$VENV_ROOT/bin/python" - <<'PY'
import importlib.util, pathlib, shutil
spec = importlib.util.find_spec("isaacsim")
root = pathlib.Path(spec.origin).parent
removed = []
for pattern in ("extscache/**/pip_prebundle/websockets", "extscache/**/pip_prebundle/packaging"):
    for path in root.glob(pattern):
        shutil.rmtree(path, ignore_errors=True); removed.append(str(path.relative_to(root)))
print("removed bundled copies:", removed or "none")
PY
  "${PIP[@]}" --reinstall-package cffi "cffi==1.17.1"
  "${PIP[@]}" --reinstall-package websockets "websockets>=15.0.1"
  # 4. bddl3 and OmniGibson with cuRobo. --no-build-isolation so cuRobo sees the venv's torch.
  "${PIP[@]}" --editable "$B1K_ROOT/bddl3"
  CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
    MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
    "${PIP[@]}" --no-build-isolation --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --reinstall-package nvidia-curobo \
    --editable "$B1K_ROOT/OmniGibson[primitives]"
  # 5. cap-harness itself plus its client dependencies (from the topology), no uv sync here.
  client_deps="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["client_dependencies"]))' "$REPO_ROOT/configs/environments.json")"
  # shellcheck disable=SC2086
  "${PIP[@]}" $client_deps 'imageio[ffmpeg]' pydantic
  "${PIP[@]}" --no-deps --editable "$REPO_ROOT"
fi

# Verification that does not launch Isaac: versions, the compiled cuRobo extension, CUDA torch.
OMNIGIBSON_DATA_PATH="$DATA_ROOT" OMNIGIBSON_APPDATA_PATH="$APPDATA_ROOT" OMNIGIBSON_HEADLESS=1 \
CUDA_VISIBLE_DEVICES="$GPU_ID" "$VENV_ROOT/bin/python" - "$TORCH_VERSION" "$ISAACSIM_VERSION" <<'PY'
import importlib.metadata as md, sys
torch_expected, isaac_expected = sys.argv[1:3]
import numpy, torch
assert numpy.__version__.split(".")[0] == "1", numpy.__version__
assert torch.__version__.startswith(torch_expected + "+cu12"), torch.__version__
assert torch.cuda.is_available(), "torch cannot see a CUDA device"
assert md.version("isaacsim").startswith(isaac_expected), md.version("isaacsim")
import omnigibson
assert omnigibson.__version__ == "3.9.2", omnigibson.__version__
from curobo.curobolib import geom_cu, lbfgs_step_cu  # noqa: F401  (compiled extensions)
import cap_harness
print(f"Verified BEHAVIOR runtime: python={sys.version.split()[0]} torch={torch.__version__} "
      f"isaacsim={md.version('isaacsim')} omnigibson={omnigibson.__version__} curobo=compiled "
      f"cap_harness={cap_harness.__file__}")
PY

# 6. Datasets (BEHAVIOR Data Bundle EULA: non-commercial academic research; accepting installs the
#    decryption key). Downloaded only on explicit request; otherwise checked.
if (( ! SKIP_DATASETS )); then
  OMNIGIBSON_DATA_PATH="$DATA_ROOT" OMNIGIBSON_APPDATA_PATH="$APPDATA_ROOT" OMNIGIBSON_HEADLESS=1 \
  "$VENV_ROOT/bin/python" - "$ACCEPT_DATASET_TOS" <<'PY'
import os, sys
accept = sys.argv[1] == "1"
from omnigibson.macros import gm
from omnigibson.utils import asset_utils as au
root = gm.DATA_PATH
required = {
    "omnigibson-robot-assets": os.path.join(root, "omnigibson-robot-assets", "models", "r1pro", "r1pro.yaml"),
    "behavior-1k-assets": os.path.join(root, "behavior-1k-assets", "VERSION"),
    "2026-challenge-task-instances": os.path.join(root, "2026-challenge-task-instances", "metadata", "task.jsonl"),
    "omnigibson.key": au.get_key_path(),
}
missing = [name for name, path in required.items() if not os.path.exists(path)]
if missing and not accept:
    sys.exit("missing datasets under %s: %s (re-run with --accept-dataset-tos or --skip-datasets)" % (root, missing))
if "omnigibson-robot-assets" in missing:
    au.download_omnigibson_robot_assets()
if "behavior-1k-assets" in missing or "omnigibson.key" in missing:
    au.download_behavior_1k_assets(accept_license=True)
if "2026-challenge-task-instances" in missing:
    au.download_2026_challenge_task_instances()
still = [name for name, path in required.items() if not os.path.exists(path)]
assert not still, f"datasets still missing after download: {still}"
print("datasets present under", root)
PY
fi

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "$MARKER" "$PROFILE"
cat <<EOF
BEHAVIOR bootstrap complete.
  source:       $REPO_ROOT
  environment:  $VENV_ROOT
  datasets:     $DATA_ROOT
  appdata:      $APPDATA_ROOT
  cuda:         $CUDA_HOME (arch list $CUDA_ARCH_LIST)

Activate:
  source "$VENV_ROOT/bin/activate"
  export OMNIGIBSON_DATA_PATH="$DATA_ROOT" OMNIGIBSON_APPDATA_PATH="$APPDATA_ROOT"
  export OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=$GPU_ID
  ulimit -c 0

Diagnose:
  cap-harness doctor --environment-root "$VENV_ROOT" \\
    --expected-environment-name .venv-behavior --runtime behavior --reset \\
    --reset-embodiment behavior --reset-suite turning_on_radio
EOF
