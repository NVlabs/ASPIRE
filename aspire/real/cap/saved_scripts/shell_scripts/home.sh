#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
source .forge_env
source "$ROOT/cap/saved_scripts/shell_scripts/speed_scale.sh"
export OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S="${OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S:-8.0}"
export OPENFORGE_PREVIEW_RECORDER_BACKEND="${OPENFORGE_PREVIEW_RECORDER_BACKEND:-python}"
export OPENFORGE_PREVIEW_RECORDER_REENCODE_H264="${OPENFORGE_PREVIEW_RECORDER_REENCODE_H264:-1}"

YAM_STATION_CALIBRATED_XML="${YAM_STATION_CALIBRATED_XML:?Set YAM_STATION_CALIBRATED_XML in .forge_env}" \
CAP_TOP_CAMERA_BACKEND=realsense \
CAP_TOP_CAMERA_FRAME=top_camera_d405 \
CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP=0 \
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 \
OPENFORGE_OPEN_HOME_SPEED_RAD_S="${OPENFORGE_OPEN_HOME_SPEED_RAD_S:-$(yam_speed_mul 0.5)}" \
OPENFORGE_OPEN_HOME_SETTLE_S="${OPENFORGE_OPEN_HOME_SETTLE_S:-1.0}" \
uv run python run_script.py \
  script_file=cap/saved_scripts/open_grippers_return_home.py \
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
  debug_ui.port="${OPENFORGE_HOME_DEBUG_PORT:-8788}"
