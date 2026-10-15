#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
VENV_ROOT="${CAP_HARNESS_ROBOSUITE_VENV:-$(resolve_canonical_venv "$REPO_ROOT" ".venv-robosuite")}"
UV_CACHE_ROOT="${CAP_HARNESS_ROBOSUITE_CACHE:-/tmp/${USER:-root}/uv-cache-robosuite}"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"

if [[ "$(basename -- "$VENV_ROOT")" != ".venv-robosuite" ]]; then
  printf 'error: Robosuite environment basename must be exactly .venv-robosuite: %s\n' \
    "$VENV_ROOT" >&2
  exit 2
fi
command -v uv >/dev/null || { printf 'error: uv is required\n' >&2; exit 1; }
[[ -f "$REPO_ROOT/uv.lock" ]] || { printf 'error: uv.lock is missing\n' >&2; exit 1; }

git -C "$REPO_ROOT" submodule update --init third_party/robosuite
expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dependencies"]["robosuite"]["commit"])' "$REPO_ROOT/configs/dependency-lock.json")"
actual="$(git -C "$REPO_ROOT/third_party/robosuite" rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || {
  printf 'error: robosuite checkout mismatch: %s != %s\n' "$actual" "$expected" >&2
  exit 1
}

mkdir -p "$(dirname -- "$VENV_ROOT")" "$UV_CACHE_ROOT"
# No-op re-bootstrap when the resolved deps, arch, and pins are unchanged.
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "${CAP_HARNESS_PROFILE:-rtx5090}" "sim:robosuite:robosuite" "third_party/robosuite")"
if venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  printf '%s is up to date (fingerprint %s); skipping rebuild\n' "sim:robosuite:robosuite" "${FINGERPRINT:0:12}"
  exit 0
fi
UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
# Client-only: the Robosuite simulator + cap-harness runtime/clients only. No
# model-serving `providers` stack (torch/sam3/open3d); providers are reached
# over HTTP from their own per-provider venvs.
VIRTUAL_ENV="$VENV_ROOT" UV_PROJECT_ENVIRONMENT="$VENV_ROOT" UV_CACHE_DIR="$UV_CACHE_ROOT" \
  uv sync --frozen --active --project "$REPO_ROOT" \
    --extra robosuite --extra dev

"$VENV_ROOT/bin/python" - <<'PY'
from pathlib import Path
import cap_harness, robosuite
import sys

root = Path(sys.prefix).resolve()
assert root.name == ".venv-robosuite", root
print(f"Verified Robosuite runtime: {root} robosuite={robosuite.__version__}")
PY

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "sim:robosuite:robosuite" "${CAP_HARNESS_PROFILE:-rtx5090}"
cat <<EOF
Robosuite bootstrap complete.
  source:       $REPO_ROOT
  environment:  $VENV_ROOT
  cache:        $UV_CACHE_ROOT

Activate:
  source "$VENV_ROOT/bin/activate"
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device

Diagnose:
  cap-harness doctor --environment-root "$VENV_ROOT" \\
    --expected-environment-name .venv-robosuite --runtime robosuite --reset \\
    --reset-embodiment robosuite --reset-suite cube_lifting
EOF
