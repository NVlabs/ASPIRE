#!/usr/bin/env bash
# Read-only prerequisite and readiness checks for the canonical YAM demo.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CHECK_SERVICES=0

usage() {
  cat <<'EOF'
Usage: bash tools/yam_demo_preflight.sh [--services]

Without arguments, checks the local installation and station configuration.
With --services, also checks both arm servers, the camera Portal, SAM3, and
BundleSDF. The script never sends a motion command.
EOF
}

case "${1:-}" in
  "") ;;
  --services) CHECK_SERVICES=1 ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

pass_count=0
fail_count=0
warn_count=0

pass() { printf '[PASS] %s\n' "$*"; pass_count=$((pass_count + 1)); }
fail() { printf '[FAIL] %s\n' "$*" >&2; fail_count=$((fail_count + 1)); }
warn() { printf '[WARN] %s\n' "$*" >&2; warn_count=$((warn_count + 1)); }

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "command available: $1"
  else
    fail "missing command: $1"
  fi
}

check_source_tree() {
  local path="$1"
  local label="$2"
  if [[ -f "$path/pyproject.toml" || -f "$path/setup.py" ]]; then
    pass "$label source: ${path#$ROOT/}"
  else
    fail "$label source missing: ${path#$ROOT/}"
  fi
}

cd "$ROOT"

if [[ -f .forge_env ]]; then
  set +u
  # shellcheck disable=SC1091
  source .forge_env
  set -u
  pass "station environment: .forge_env"
else
  fail "missing .forge_env (copy .forge_env.example and configure it)"
fi

PYTHON="${YAM_DEMO_PYTHON:-$ROOT/.venv/bin/python}"
CAMERA_PORT="${YAM_DEMO_CAMERA_PORTAL_PORT:-8300}"
SAM3_PORT="${YAM_DEMO_SAM3_PORT:-6767}"
BUNDLESDF_PORT="${YAM_DEMO_BUNDLESDF_PORT:-8119}"
if [[ -x "$PYTHON" ]]; then
  pass "Python environment: ${PYTHON#$ROOT/}"
  python_version="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [[ "$python_version" == "3.11" ]]; then
    pass "Python version: $python_version"
  else
    fail "Python 3.11 is required; found ${python_version:-unknown}"
  fi
else
  fail "missing Python environment: ${PYTHON#$ROOT/}"
fi

for command_name in uv tmux curl ffmpeg ffprobe nvidia-smi; do
  check_command "$command_name"
done

if [[ -n "${YAM_STATION_CALIBRATED_XML:-}" && -f "${YAM_STATION_CALIBRATED_XML:-}" ]]; then
  pass "calibrated station model: $YAM_STATION_CALIBRATED_XML"
else
  fail "YAM_STATION_CALIBRATED_XML is unset or does not name a readable file"
fi

check_source_tree "$ROOT/third_party/i2rt" "i2rt"
check_source_tree "$ROOT/third_party/bundlesdf" "BundleSDF"

BUNDLESDF_LIB_DIR="${BUNDLESDF_RUNTIME_LIB_DIR:-${BUNDLESDF_REPO_LIB_DIR:-$ROOT/third_party/bundlesdf_5090}}"
if [[ -d "$BUNDLESDF_LIB_DIR" ]]; then
  pass "BundleSDF runtime libraries: $BUNDLESDF_LIB_DIR"
else
  fail "BundleSDF runtime library directory missing: $BUNDLESDF_LIB_DIR"
fi

if [[ -x "$PYTHON" ]]; then
  modules=(
    cv2 fastapi hydra i2rt mink mujoco numpy omegaconf PIL portal
    pyrealsense2 scipy torch transformers uvicorn bundlesdf
  )
  missing_modules="$($PYTHON - "${modules[@]}" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)"
  if [[ -z "$missing_modules" ]]; then
    pass "required Python modules are discoverable"
  else
    fail "missing Python modules: $missing_modules"
  fi
fi

if [[ -e /dev/video_top && -e /dev/video_left && -e /dev/video_right && -e /dev/video_bottom ]]; then
  pass "camera device aliases: top, left, right, bottom"
else
  warn "one or more camera aliases are absent: /dev/video_{top,left,right,bottom}"
fi

for interface in can_follow_l can_follow_r; do
  if [[ -e "/sys/class/net/$interface" ]]; then
    pass "CAN interface present: $interface"
  else
    fail "CAN interface missing: $interface"
  fi
done

if [[ "$CHECK_SERVICES" == "1" && -x "$PYTHON" ]]; then
  if "$PYTHON" -c 'import portal,sys
ok=True
for side,port in (("left",11333),("right",11334)):
    try:
        h=portal.Client(f"127.0.0.1:{port}").get_health().result(timeout=4)
        good=bool(h.get("connected")) and bool(h.get("send_thread_alive")) and h.get("background_error") is None
    except Exception:
        good=False
    ok = ok and good
sys.exit(0 if ok else 1)' >/dev/null 2>&1; then
    pass "arm servers healthy: left :11333, right :11334"
  else
    fail "arm servers are not healthy on :11333/:11334"
  fi

  if CAMERA_PORT="$CAMERA_PORT" "$PYTHON" -c 'import os,portal,sys
try:
    port=os.environ["CAMERA_PORT"]
    h=portal.Client(f"127.0.0.1:{port}").health().result(timeout=4)
    available=set(h.get("available_cameras", []))
    ok={"top","left","right","bottom"}.issubset(available)
except Exception:
    ok=False
sys.exit(0 if ok else 1)' >/dev/null 2>&1; then
    pass "camera Portal healthy on :$CAMERA_PORT with four cameras"
  else
    fail "camera Portal is unavailable or missing cameras on :$CAMERA_PORT"
  fi

  sam3_health="$(curl -fsS --max-time 4 "http://127.0.0.1:$SAM3_PORT/health" 2>/dev/null || true)"
  if printf '%s' "$sam3_health" | grep -Eq '"model_loaded"[[:space:]]*:[[:space:]]*true'; then
    pass "SAM3 healthy with model loaded on :$SAM3_PORT"
  else
    fail "SAM3 is unavailable or its model is not loaded on :$SAM3_PORT"
  fi

  if curl -fsS --max-time 4 "http://127.0.0.1:$BUNDLESDF_PORT/health" >/dev/null; then
    pass "BundleSDF healthy on :$BUNDLESDF_PORT"
  else
    fail "BundleSDF is not healthy on :$BUNDLESDF_PORT"
  fi
fi

printf '\nPreflight summary: %d passed, %d warning(s), %d failed\n' \
  "$pass_count" "$warn_count" "$fail_count"

if [[ "$fail_count" -ne 0 ]]; then
  exit 2
fi
