#!/usr/bin/env bash
# Launch only the services required by `yam_demo.sh full`.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION="yam_demo"
ATTACH=1

usage() {
  cat <<'EOF'
Usage: bash tmux/launch_yam_demo_services.sh [--session NAME] [--no-attach]

Starts, in dependency order:
  - left and right YAM follower arm servers
  - read-only four-camera Portal server
  - SAM3 text segmentation
  - BundleSDF one-shot 6-DoF localization

This intentionally does not start AnyGrasp, cuRobo, PyRoki, an LLM provider,
voice services, or evaluation agents; the canonical saved demo does not use them.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --no-attach) ATTACH=0; shift ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

bash tools/yam_demo_preflight.sh

set +u
# shellcheck disable=SC1091
source .forge_env
set -u

PYTHON="${YAM_DEMO_PYTHON:-$ROOT/.venv/bin/python}"
CAMERA_PORT="${YAM_DEMO_CAMERA_PORTAL_PORT:-8300}"
SAM3_PORT="${YAM_DEMO_SAM3_PORT:-6767}"
BUNDLESDF_PORT="${YAM_DEMO_BUNDLESDF_PORT:-8119}"
ARM_TIMEOUT_S="${YAM_DEMO_ARM_START_TIMEOUT_S:-45}"
CAMERA_TIMEOUT_S="${YAM_DEMO_CAMERA_START_TIMEOUT_S:-60}"
SAM3_TIMEOUT_S="${YAM_DEMO_SAM3_START_TIMEOUT_S:-240}"
BUNDLESDF_TIMEOUT_S="${YAM_DEMO_BUNDLESDF_START_TIMEOUT_S:-360}"

wait_until() {
  local label="$1"
  local timeout_s="$2"
  shift 2
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if "$@" >/dev/null 2>&1; then
      echo "[ready] $label"
      return 0
    fi
    sleep 1
  done
  echo "[failed] $label did not become ready within ${timeout_s}s" >&2
  return 1
}

arms_ready() {
  "$PYTHON" -c 'import portal,sys
ok=True
for port in (11333,11334):
    try:
        h=portal.Client(f"127.0.0.1:{port}").get_health().result(timeout=3)
        ok = ok and bool(h.get("connected")) and bool(h.get("send_thread_alive")) and h.get("background_error") is None
    except Exception:
        ok=False
sys.exit(0 if ok else 1)'
}

camera_ready() {
  CAMERA_PORT="$CAMERA_PORT" "$PYTHON" -c 'import os,portal,sys
try:
    port=os.environ["CAMERA_PORT"]
    h=portal.Client(f"127.0.0.1:{port}").health().result(timeout=3)
    ok={"top","left","right","bottom"}.issubset(set(h.get("available_cameras", [])))
except Exception:
    ok=False
sys.exit(0 if ok else 1)'
}

sam3_ready() {
  curl -fsS --max-time 3 "http://127.0.0.1:$SAM3_PORT/health" |
    grep -Eq '"model_loaded"[[:space:]]*:[[:space:]]*true'
}
bundlesdf_ready() { curl -fsS --max-time 3 "http://127.0.0.1:$BUNDLESDF_PORT/health"; }

tmux new-session -d -s "$SESSION" -n left-arm -c "$ROOT" \
  "bash tools/run_yam_demo_service.sh left-arm"
tmux new-window -d -t "$SESSION" -n right-arm -c "$ROOT" \
  "bash tools/run_yam_demo_service.sh right-arm"
wait_until "both follower arm servers" "$ARM_TIMEOUT_S" arms_ready

tmux new-window -d -t "$SESSION" -n cameras -c "$ROOT" \
  "bash tools/run_yam_demo_service.sh camera-portal"
wait_until "four-camera Portal :$CAMERA_PORT" "$CAMERA_TIMEOUT_S" camera_ready

tmux new-window -d -t "$SESSION" -n sam3 -c "$ROOT" \
  "bash tools/run_yam_demo_service.sh sam3"
wait_until "SAM3 :$SAM3_PORT" "$SAM3_TIMEOUT_S" sam3_ready

tmux new-window -d -t "$SESSION" -n bundlesdf -c "$ROOT" \
  "bash tools/run_yam_demo_service.sh bundlesdf"
wait_until "BundleSDF :$BUNDLESDF_PORT" "$BUNDLESDF_TIMEOUT_S" bundlesdf_ready

bash tools/yam_demo_preflight.sh --services

cat <<EOF

YAM demo services are ready in tmux session '$SESSION'.

Run the demo from another terminal:
  cd $ROOT
  source .forge_env
  bash cap/saved_scripts/yam_demo.sh full

Attach to service logs with:
  tmux attach -t $SESSION
EOF

if [[ "$ATTACH" == "1" ]]; then
  exec tmux attach -t "$SESSION"
fi
