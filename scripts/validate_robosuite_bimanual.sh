#!/usr/bin/env bash
# Thin wrapper for the sealed RoboSuite bimanual acceptance gate. It owns only the
# per-process renderer/GL/GPU environment and the provider preflight; the case
# matrix, fresh-process execution, sealing, and verification all live in the
# declarative plan run by `cap-harness validate --plan`.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CAP_HARNESS_ROBOSUITE_PYTHON:-$REPO_ROOT/.venv-robosuite/bin/python}"
RENDERER="${CAP_HARNESS_ROBOSUITE_RENDERER:-egl}"
SIMULATION_GPU="${CAP_HARNESS_ROBOSUITE_GPU:-}"
OSMESA_LIBRARY_DIR="${CAP_HARNESS_OSMESA_LIBRARY_DIR:-}"
OUTPUT_ROOT="${CAP_HARNESS_ROBOSUITE_BIMANUAL_OUTPUT:-$REPO_ROOT/validation-artifacts/robosuite-bimanual-$(date -u +%Y%m%dT%H%M%SZ)}"
PLAN="$REPO_ROOT/configs/validation/robosuite-bimanual.yaml"

if (($#)); then
  printf 'usage: %s\n' "$0" >&2
  exit 2
fi
[[ -x "$PYTHON_BIN" ]] || {
  printf 'error: Robosuite Python is missing: %s\n' "$PYTHON_BIN" >&2
  exit 1
}

# Per-process renderer/GL/GPU environment (must be set before the interpreter
# initializes MuJoCo/EGL/CUDA); exported so each fresh run process inherits it.
export PYTHONHASHSEED=0
case "$RENDERER" in
  cpu)
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
      GALLIUM_DRIVER=llvmpipe CUDA_VISIBLE_DEVICES=
    if [[ -n "$OSMESA_LIBRARY_DIR" ]]; then
      [[ -d "$OSMESA_LIBRARY_DIR" ]] || {
        printf 'error: CAP_HARNESS_OSMESA_LIBRARY_DIR is not a directory: %s\n' \
          "$OSMESA_LIBRARY_DIR" >&2
        exit 2
      }
      export LD_LIBRARY_PATH="$OSMESA_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    ;;
  egl)
    [[ "$SIMULATION_GPU" =~ ^[0-9]+$ ]] || {
      printf 'error: CAP_HARNESS_ROBOSUITE_GPU must name one GPU for EGL rendering\n' >&2
      exit 2
    }
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device \
      CUDA_VISIBLE_DEVICES="$SIMULATION_GPU" MUJOCO_EGL_DEVICE_ID="$SIMULATION_GPU"
    ;;
  *)
    printf 'error: CAP_HARNESS_ROBOSUITE_RENDERER must be cpu or egl\n' >&2
    exit 2
    ;;
esac

# Provider preflight: sam3 (8114), pyroki (8116), curobo (8118).
for port in 8114 8116 8118; do
  curl --fail --silent --show-error --max-time 2 \
    "http://127.0.0.1:${port}/openapi.json" >/dev/null || {
    printf 'error: required provider is unavailable on port %s\n' "$port" >&2
    exit 1
  }
done

"$PYTHON_BIN" -m cap_harness.cli validate --plan "$PLAN" --output-dir "$OUTPUT_ROOT"
printf 'validation artifacts: %s\n' "$OUTPUT_ROOT"
