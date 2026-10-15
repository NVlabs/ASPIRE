#!/usr/bin/env bash
# Run the BEHAVIOR-1K validation plans from the dedicated .venv-behavior runtime.
#
# Usage: scripts/validate_behavior.sh [structural|pickup] [--output-dir PATH]
#   structural  (default) resets both tasks on instances 1-3 without providers
#   pickup      runs the example pickup programs; needs SAM3 (8114) and Contact-GraspNet (8115)
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CAP_HARNESS_BEHAVIOR_PYTHON_BIN:-$REPO_ROOT/.venv-behavior/bin/python}"
PLAN_NAME="structural"
OUTPUT_ROOT=""
while (($#)); do
  case "$1" in
    structural|pickup) PLAN_NAME="$1"; shift ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -x "$PYTHON_BIN" ]] || { printf 'error: %s is missing; run scripts/bootstrap_behavior.sh\n' "$PYTHON_BIN" >&2; exit 1; }
PLAN="$REPO_ROOT/configs/validation/behavior-$PLAN_NAME.yaml"
[[ -f "$PLAN" ]] || { printf 'error: plan not found: %s\n' "$PLAN" >&2; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/validation-artifacts/behavior-$PLAN_NAME-$STAMP}"

export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
export OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID:-${CAP_HARNESS_BEHAVIOR_GPU:-0}}"
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-${CAP_HARNESS_BEHAVIOR_DATA:-$HOME/behavior-data}}"
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-$REPO_ROOT/.behavior-appdata}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$OMNIGIBSON_GPU_ID}"
export PYTHONHASHSEED=0
ulimit -c 0
[[ -d "$OMNIGIBSON_DATA_PATH" ]] || { printf 'error: dataset root missing: %s\n' "$OMNIGIBSON_DATA_PATH" >&2; exit 1; }

if [[ "$PLAN_NAME" == "pickup" ]]; then
  for port in 8114 8115; do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null || {
      printf 'error: provider on port %s is not listening (start scripts/supervise_services.sh)\n' "$port" >&2
      exit 1
    }
  done
fi
printf 'plan=%s output=%s gpu=%s data=%s\n' "$PLAN" "$OUTPUT_ROOT" "$OMNIGIBSON_GPU_ID" "$OMNIGIBSON_DATA_PATH"
"$PYTHON_BIN" -m cap_harness.cli validate --plan "$PLAN" --output-dir "$OUTPUT_ROOT"
