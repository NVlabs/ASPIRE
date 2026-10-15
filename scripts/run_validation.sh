#!/usr/bin/env bash
set -euo pipefail

MODE="smoke"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_SUBMODULE="${CAP_HARNESS_LIBERO_SUBMODULE:-$REPO_ROOT/third_party/LIBERO-PRO}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
LIBERO_CONFIG_FILE="$LIBERO_CONFIG_DIR/config.yaml"
ARTIFACT_ROOT="${CAP_HARNESS_ARTIFACT_ROOT:-$REPO_ROOT/validation-artifacts}"
OUTPUT_DIR_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: scripts/run_validation.sh [--smoke|--nightly] [--artifact-root PATH]
                                 [--output-dir PATH]

Run full doctor/reset, emit the exact 80-pair manifest, and execute the resumable
LIBERO-Pro matrix. Provider services on ports 8114-8116 must already be ready.
EOF
}

while (($#)); do
  case "$1" in
    --smoke)
      MODE="smoke"
      shift
      ;;
    --nightly)
      MODE="nightly"
      shift
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v cap-harness >/dev/null 2>&1 || {
  printf 'error: cap-harness is not installed in the active environment\n' >&2
  exit 1
}

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$MODE"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$ARTIFACT_ROOT/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"

cap-harness doctor \
  --libero-submodule "$LIBERO_SUBMODULE" \
  --libero-config "$LIBERO_CONFIG_FILE" \
  --reset \
  --output "$OUTPUT_DIR/doctor.json"

# Thin wrapper: environment/setup + doctor above; the validation gate itself is
# the canonical plan command. Smoke runs the 80-pair representative plan; nightly
# runs the same representatives across seeds 1-3.
if [[ "$MODE" == "nightly" ]]; then
  PLAN="$REPO_ROOT/configs/validation/libero-nightly.yaml"
else
  PLAN="$REPO_ROOT/configs/validation/libero-smoke.yaml"
fi
cap-harness validate --plan "$PLAN" --output-dir "$OUTPUT_DIR" --retry-failures

printf 'validation artifacts: %s\n' "$OUTPUT_DIR"
