#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
VENV_ROOT="${CAP_HARNESS_CUROBO_VENV:-$(resolve_canonical_venv "$REPO_ROOT" ".venv-curobo")}"
UV_CACHE_ROOT="${CAP_HARNESS_CUROBO_CACHE:-/tmp/${USER:-root}/uv-cache-curobo}"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-/usr/bin/python3}"

[[ "$(basename -- "$VENV_ROOT")" == ".venv-curobo" ]] || {
  printf 'error: cuRobo environment basename must be .venv-curobo: %s\n' "$VENV_ROOT" >&2
  exit 2
}
command -v uv >/dev/null || { printf 'error: uv is required\n' >&2; exit 1; }

git -C "$REPO_ROOT" submodule update --init third_party/curobo
expected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["optional_dependencies"]["curobo"]["commit"])' "$REPO_ROOT/configs/dependency-lock.json")"
actual="$(git -C "$REPO_ROOT/third_party/curobo" rev-parse HEAD)"
[[ "$actual" == "$expected" ]] || {
  printf 'error: cuRobo checkout mismatch: %s != %s\n' "$actual" "$expected" >&2
  exit 1
}

mkdir -p "$(dirname -- "$VENV_ROOT")" "$UV_CACHE_ROOT"
# No-op re-bootstrap when the resolved deps, arch, and pins are unchanged.
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "${CAP_HARNESS_PROFILE:-rtx5090}" "provider:curobo:source" "third_party/curobo")"
if venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  printf '%s is up to date (fingerprint %s); skipping rebuild\n' "provider:curobo:source" "${FINGERPRINT:0:12}"
  exit 0
fi
UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
UV_CACHE_DIR="$UV_CACHE_ROOT" uv pip install --python "$VENV_ROOT/bin/python" \
  'torch==2.9.1' 'torchvision==0.24.1' \
  fastapi pydantic requests uvicorn
UV_CACHE_DIR="$UV_CACHE_ROOT" uv pip install --python "$VENV_ROOT/bin/python" \
  --editable "$REPO_ROOT/third_party/curobo[cu12-torch]"
UV_CACHE_DIR="$UV_CACHE_ROOT" uv pip install --python "$VENV_ROOT/bin/python" \
  --editable "$REPO_ROOT" --no-deps

CUDA_VISIBLE_DEVICES="${CAP_HARNESS_CUROBO_GPU:-3}" "$VENV_ROOT/bin/python" - <<'PY'
import curobo
import torch
assert torch.__version__.split("+", 1)[0] == "2.9.1"
print(f"Verified cuRobo runtime: torch={torch.__version__} module={curobo.__file__}")
PY

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "provider:curobo:source" "${CAP_HARNESS_PROFILE:-rtx5090}"
printf 'cuRobo bootstrap complete. Activate with: source %q/bin/activate\n' "$VENV_ROOT"
