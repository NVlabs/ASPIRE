#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/tools/runtime_env.sh"

warn_yellow() {
  printf '\033[33m[warn]\033[0m %s\n' "$*" >&2
}

kill_listener() {
  local port="$1"
  local attempt
  for attempt in 1 2 3; do
    if ! ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .; then
      return 0
    fi
    warn_yellow "stopping existing listener on port ${port} (attempt ${attempt}/3)"
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    sleep 1
  done
}

resolve_lfs_pointer() {
  local path="$1"
  if [[ -L "$path" ]]; then
    local target
    target="$(readlink "$path" || true)"
    if [[ -n "$target" && "$target" == *"/.git/lfs/objects/"* ]]; then
      local suffix="${target#*/.git/lfs/objects/}"
      local obj="$REPO_ROOT/.git/lfs/objects/$suffix"
      if [[ ! -f "$obj" ]]; then
        echo "Missing local LFS object for $path at $obj" >&2
        return 1
      fi
      echo "$obj"
      return 0
    fi
    if [[ -n "$target" && -f "$target" ]]; then
      echo "$target"
      return 0
    fi
  fi
  if [[ ! -f "$path" ]]; then
    echo "$path"
    return 0
  fi
  if head -n 1 "$path" | grep -q 'https://git-lfs.github.com/spec/v1'; then
    local sha
    sha="$(sed -n 's/^oid sha256://p' "$path")"
    if [[ -z "$sha" ]]; then
      echo "Could not parse LFS oid from $path" >&2
      return 1
    fi
    local obj="$REPO_ROOT/.git/lfs/objects/${sha:0:2}/${sha:2:2}/$sha"
    if [[ ! -f "$obj" ]]; then
      echo "Missing local LFS object for $path at $obj" >&2
      return 1
    fi
    echo "$obj"
  else
    echo "$path"
  fi
}

materialize_lfs_symlink() {
  local path="$1"
  if [[ -L "$path" ]]; then
    resolve_lfs_pointer "$path" >/dev/null || return 1
    return 0
  fi
  if [[ ! -f "$path" ]]; then
    return 0
  fi
  if head -n 1 "$path" | grep -q 'https://git-lfs.github.com/spec/v1'; then
    resolve_lfs_pointer "$path" >/dev/null || return 1
  fi
}

find_openssl11_dir() {
  local candidate
  local candidates=(
    "${ANYGRASP_OPENSSL11_DIR:-}"
    "$REPO_ROOT/third_party/anygrasp_sdk/license_registration/.compat/openssl-1.1/usr/lib/x86_64-linux-gnu"
    "${RUNTIME_DEPS_ROOT:-}/libssl11/usr/lib/x86_64-linux-gnu"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate/libcrypto.so.1.1" && -f "$candidate/libssl.so.1.1" ]]; then
      warn_yellow "using host-configured OpenSSL 1.1 runtime from $candidate"
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

find_openblas_dir() {
  local candidate
  local candidates=(
    "${OPENBLAS_HOME:-}"
    "${OPENBLAS_HOME:-}/lib"
    "${RUNTIME_DEPS_ROOT:-}/openblas/lib"
    "/usr/lib/x86_64-linux-gnu/openblas-pthread"
    "/lib/x86_64-linux-gnu"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate/libopenblas.so.0" ]]; then
      warn_yellow "using host OpenBLAS runtime from $candidate"
      echo "$candidate"
      return 0
    fi
  done

  if command -v ldconfig >/dev/null 2>&1; then
    candidate="$(
      ldconfig -p 2>/dev/null \
        | awk '/libopenblas\.so\.0[[:space:]]/ {print $NF; exit}'
    )"
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      warn_yellow "repo-relative OpenBLAS bundle is unavailable; falling back to system library $(dirname "$candidate")"
      dirname "$candidate"
      return 0
    fi
  fi

  return 1
}

find_cuda_home() {
  local candidate
  local candidates=(
    "${CUDA_HOME:-}"
    "/usr/local/cuda"
    "/usr/local/cuda-12.8"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  if command -v nvcc >/dev/null 2>&1; then
    candidate="$(cd -- "$(dirname -- "$(dirname -- "$(command -v nvcc)")")" && pwd)"
    if [[ -x "$candidate/bin/nvcc" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  return 1
}

find_uv_bin() {
  if [[ -n "${UV_BIN:-}" && -x "${UV_BIN:-}" ]]; then
    echo "$UV_BIN"
    return 0
  fi
  command -v uv 2>/dev/null || true
}

if [[ -f "$REPO_ROOT/tools/anygrasp_identity/activate.sh" ]]; then
  source "$REPO_ROOT/tools/anygrasp_identity/activate.sh"
fi

export ANYGRASP_PORT="${ANYGRASP_PORT:-8122}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if CUDA_HOME="$(find_cuda_home)"; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
else
  warn_yellow "could not locate CUDA_HOME via env or nvcc; continuing without adding a CUDA bin path"
fi
if OPENSSL11_DIR="$(find_openssl11_dir)"; then
  :
else
  warn_yellow "missing host-configured OpenSSL 1.1 runtime"
  echo "[anygrasp] set ANYGRASP_OPENSSL11_DIR or RUNTIME_DEPS_ROOT to a self-contained libssl11 directory" >&2
  exit 1
fi

if OPENBLAS_DIR="$(find_openblas_dir)"; then
  cuda_lib64=""
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
    cuda_lib64="${CUDA_HOME}/lib64:"
  fi
  export LD_LIBRARY_PATH="$OPENSSL11_DIR:${cuda_lib64}$OPENBLAS_DIR:${LD_LIBRARY_PATH:-}"
else
  echo "[anygrasp] missing libopenblas.so.0; set OPENBLAS_HOME or install/copy OpenBLAS first" >&2
  exit 1
fi

materialize_lfs_symlink "third_party/anygrasp_sdk/dependencies/MinkowskiEngine/build/lib.linux-x86_64-cpython-311/MinkowskiEngineBackend/_C.cpython-311-x86_64-linux-gnu.so"
materialize_lfs_symlink "third_party/anygrasp_sdk/pointnet2/build/lib.linux-x86_64-cpython-311/pointnet2/_ext.cpython-311-x86_64-linux-gnu.so"

if [[ -f "${ANYGRASP_REPO_CHECKPOINT_PATH:-}" ]]; then
  ANYGRASP_CHECKPOINT_PATH="${ANYGRASP_REPO_CHECKPOINT_PATH}"
elif [[ -n "${ANYGRASP_CHECKPOINT_PATH:-}" ]]; then
  warn_yellow "repo-relative AnyGrasp checkpoint is unavailable; falling back to ANYGRASP_CHECKPOINT_PATH=$ANYGRASP_CHECKPOINT_PATH"
else
  ANYGRASP_CHECKPOINT_PATH="${ANYGRASP_REPO_CHECKPOINT_PATH:-$REPO_ROOT/checkpoint_detection.tar}"
fi

if [[ -f "${ANYGRASP_REPO_LICENSE_ZIP:-}" ]]; then
  ANYGRASP_LICENSE_ZIP="${ANYGRASP_REPO_LICENSE_ZIP}"
elif [[ -n "${ANYGRASP_LICENSE_ZIP:-}" ]]; then
  warn_yellow "repo-relative AnyGrasp license zip is unavailable; falling back to ANYGRASP_LICENSE_ZIP=$ANYGRASP_LICENSE_ZIP"
else
  echo "[anygrasp] set ANYGRASP_LICENSE_ZIP to an authorized local license archive" >&2
  exit 1
fi

ANYGRASP_CHECKPOINT_PATH="$(resolve_lfs_pointer "$ANYGRASP_CHECKPOINT_PATH")"
ANYGRASP_LICENSE_ZIP="$(resolve_lfs_pointer "$ANYGRASP_LICENSE_ZIP")"
ANYGRASP_MAX_GRIPPER_WIDTH="${ANYGRASP_MAX_GRIPPER_WIDTH:-0.1}"
ANYGRASP_GRIPPER_HEIGHT="${ANYGRASP_GRIPPER_HEIGHT:-0.03}"
ANYGRASP_TOP_DOWN="${ANYGRASP_TOP_DOWN:-1}"

UV_BIN="$(find_uv_bin)"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  warn_yellow "could not locate uv on PATH"
  echo "Missing uv executable; install uv or set UV_BIN" >&2
  exit 1
fi

args=(
  python tools/vision/serve_anygrasp.py
  --port "$ANYGRASP_PORT"
  --checkpoint-path "$ANYGRASP_CHECKPOINT_PATH"
  --license-zip "$ANYGRASP_LICENSE_ZIP"
  --max-gripper-width "$ANYGRASP_MAX_GRIPPER_WIDTH"
  --gripper-height "$ANYGRASP_GRIPPER_HEIGHT"
)

if [[ "$ANYGRASP_TOP_DOWN" != "0" ]]; then
  args+=(--top-down-grasp)
fi

args+=("$@")

echo "[anygrasp] launching on :$ANYGRASP_PORT"
echo "[anygrasp] checkpoint: $ANYGRASP_CHECKPOINT_PATH"
echo "[anygrasp] license   : $ANYGRASP_LICENSE_ZIP"
echo "[anygrasp] top_down  : $ANYGRASP_TOP_DOWN"

kill_listener "$ANYGRASP_PORT"

exec "$UV_BIN" run "${args[@]}"
