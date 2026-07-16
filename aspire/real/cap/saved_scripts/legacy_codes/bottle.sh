#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HOME_SCRIPT="$ROOT/cap/saved_scripts/shell_scripts/home.sh"
COMBINED_SCRIPT="cap/saved_scripts/legacy_codes/bottle_pour_then_rack_one_shot_loop.py"

usage() {
  cat <<'USAGE'
Usage: bottle.sh

Runs the bottle demo sequence:
  0. home.sh
  1. bottle_pour_then_rack_one_shot_loop.py

Options via env:
  BOTTLE_DEMO_SKIP_HOME=1   Skip the initial home script.
  BOTTLE_DEMO_SKIP_POUR=1   Skip the dry bottle-to-glass pour inside the combined Python script.
  BOTTLE_DEMO_SKIP_RACK=1   Skip placing the bottle on the top shelf inside the combined Python script.
  OPENFORGE_BOTTLE_DEBUG_PORT=8983
                              Override the debug UI port for the combined Python run.
USAGE
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if [ ! -r "$HOME_SCRIPT" ]; then
  echo "Missing readable home script: $HOME_SCRIPT" >&2
  exit 1
fi

cd "$ROOT"

if [ "${BOTTLE_DEMO_SKIP_HOME:-0}" = "1" ]; then
  echo "=== bottle_demo: skipping home because BOTTLE_DEMO_SKIP_HOME=1 ==="
else
  echo "=== bottle_demo: running home ==="
  bash "$HOME_SCRIPT"
fi

source .forge_env

export YAM_STATION_CALIBRATED_XML="${YAM_STATION_CALIBRATED_XML:?Set YAM_STATION_CALIBRATED_XML in .forge_env}"
export CAP_TOP_CAMERA_BACKEND=realsense
export CAP_TOP_CAMERA_FRAME=top_camera_d405
export CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP=0
export OPENFORGE_ALLOW_PHYSICAL_MOTION=1

uv run python run_script.py \
  script_file="$COMBINED_SCRIPT" \
  skill_library_path=cap/saved_scripts/skill_library \
  env.name=yam-real \
  robot=real_yam \
  robot.dashboard=true \
  robot.await_exit=false \
  robot.go_home_on_exit=false \
  runtime.no_cameras=true \
  recording.enabled=true \
  debug_ui.enabled=true \
  debug_ui.auto_open=false \
  debug_ui.auto_exit_on_run_end=false \
  debug_ui.host=0.0.0.0 \
  debug_ui.port="${OPENFORGE_BOTTLE_DEBUG_PORT:-8983}"
