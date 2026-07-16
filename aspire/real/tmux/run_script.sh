#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pick up NVIDIA_API_KEY, GEMINI_API_KEY, ANYGRASP_OPENSSL11_DIR, etc.
[[ -f "$PROJECT_DIR/.forge_env" ]] && source "$PROJECT_DIR/.forge_env"


SCRIPT_FILE="cap/saved_scripts/table_bussing/nclass_sorting_nvidiagemini.py"
# SCRIPT_FILE="cap/saved_scripts/gpu/gpu_test.py"
cd "$PROJECT_DIR"
CAP_TOP_CAMERA_BACKEND=realsense NAIL_PREGRASP_DEBUG_UI=1 NAIL_PREGRASP_DEBUG_UI_BLOCKING=0 \
uv run python run_script.py  \
script_file=${SCRIPT_FILE}   \
skill_library_path=cap/saved_scripts/skill_library   \
env.name=yam-real   \
robot=real_yam \
robot.dashboard=true   \
execution.record=true   \
recording.enabled=true   \
debug_ui.enabled=true   \
debug_ui.auto_open=true   \
debug_ui.auto_exit_on_run_end=false
