#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  cap/saved_scripts/run_real_yam.sh health
  cap/saved_scripts/run_real_yam.sh smoke [--cameras top,left,right,bottom]
  cap/saved_scripts/run_real_yam.sh no-motion <script.py> [extra run_script args...]
  cap/saved_scripts/run_real_yam.sh physical <script.py> [extra run_script args...]

Examples:
  cap/saved_scripts/run_real_yam.sh smoke
  cap/saved_scripts/run_real_yam.sh no-motion cap/saved_scripts/debug_observation_smoke.py
  cap/saved_scripts/run_real_yam.sh physical cap/saved_scripts/reset_open_robot.py

Physical mode sets OPENFORGE_ALLOW_PHYSICAL_MOTION=1. Use it only with an
active operator-approved ticket and script-specific safety envs.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

if [[ -f .forge_env ]]; then
  # shellcheck source=/dev/null
  source .forge_env
else
  die ".forge_env is missing in $ROOT_DIR"
fi

export YAM_STATION_CALIBRATED_XML="${YAM_STATION_CALIBRATED_XML:?Set YAM_STATION_CALIBRATED_XML in .forge_env}"
export CAP_TOP_CAMERA_BACKEND="${CAP_TOP_CAMERA_BACKEND:-realsense}"
export CAP_TOP_CAMERA_FRAME="${CAP_TOP_CAMERA_FRAME:-top_camera_d405}"
export CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP="${CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP:-0}"

[[ -f "$YAM_STATION_CALIBRATED_XML" ]] || die "calibrated station XML not found: $YAM_STATION_CALIBRATED_XML"
[[ -f run_script.py ]] || die "run_script.py not found in $ROOT_DIR"

free_port() {
  local port="${1:-8788}"
  while ss -ltn "sport = :$port" | rg -q ":$port"; do
    port=$((port + 1))
  done
  printf '%s\n' "$port"
}

health_json() {
  local name="$1"
  local url="$2"
  if curl -fsS --max-time 3 "$url" >/tmp/openforge_health.json 2>/tmp/openforge_health.err; then
    printf '[ok] %s %s\n' "$name" "$url"
    python3 -m json.tool /tmp/openforge_health.json 2>/dev/null || cat /tmp/openforge_health.json
  else
    printf '[fail] %s %s\n' "$name" "$url"
    cat /tmp/openforge_health.err || true
  fi
}

check_no_competing_run() {
  local active
  active="$(pgrep -af 'run_script.py|ffmpeg' || true)"
  if [[ -n "$active" ]]; then
    printf '%s\n' "$active" >&2
    die "a run_script.py or ffmpeg process is already active"
  fi
}

base_args() {
  local port="$1"
  printf '%s\n' \
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
    "debug_ui.port=$port"
}

run_saved_script() {
  local mode="$1"
  local script_file="$2"
  shift 2

  [[ -n "$script_file" ]] || die "missing script_file"
  [[ -f "$script_file" ]] || die "script does not exist: $script_file"
  check_no_competing_run

  local port
  port="$(free_port "${OPENFORGE_DEBUG_UI_PORT:-8788}")"
  printf '[run_real_yam] mode=%s script=%s debug_ui=http://127.0.0.1:%s\n' "$mode" "$script_file" "$port"

  if [[ "$mode" == physical ]]; then
    export OPENFORGE_ALLOW_PHYSICAL_MOTION=1
  else
    unset OPENFORGE_ALLOW_PHYSICAL_MOTION
  fi

  uv run python run_script.py \
    "script_file=$script_file" \
    skill_library_path=cap/saved_scripts/skill_library \
    $(base_args "$port") \
    "$@"
}

cmd="${1:-}"
case "$cmd" in
  health)
    printf '[run_real_yam] host=%s root=%s\n' "$(hostname)" "$ROOT_DIR"
    printf '[run_real_yam] calibrated_xml=%s\n' "$YAM_STATION_CALIBRATED_XML"
    health_json "BundleSDF" "http://127.0.0.1:8119/health"
    health_json "SAM3" "http://127.0.0.1:6767/health"
    health_json "AnyGrasp" "http://127.0.0.1:8122/health"
    printf '[run_real_yam] relevant processes:\n'
    pgrep -af 'run_script.py|ffmpeg|cap.debug_ui.app|arm_server|serve_bundlesdf|serve_real_yam_camera_portal|serve_anygrasp|serve_sam3|follower|curobo|pyroki' || true
    ;;
  smoke)
    shift
    cameras="top"
    if [[ "${1:-}" == "--cameras" ]]; then
      cameras="${2:-}"
      shift 2
    fi
    export OPENFORGE_DEBUG_OBS_TIMEOUT_S="${OPENFORGE_DEBUG_OBS_TIMEOUT_S:-30}"
    export OPENFORGE_DEBUG_OBS_CAMERAS="$cameras"
    run_saved_script no-motion cap/saved_scripts/debug_observation_smoke.py "$@"
    ;;
  no-motion)
    shift
    run_saved_script no-motion "${1:-}" "${@:2}"
    ;;
  physical)
    shift
    run_saved_script physical "${1:-}" "${@:2}"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    usage >&2
    die "unknown command: $cmd"
    ;;
esac
