#!/usr/bin/env bash
# launch_realworld_localserver_realsense.sh
#
# Launch every CAP service that is NOT cap_server / cap_agent / cap_ui:
#
#   Tmux window "perception/control serving" — 2×2
#   ┌────────────────────────────┬────────────────────────────┐
#   │ serve_bundlesdf  (+ SAM2)  │ serve_sam3                 │
#   │ :8119                      │ :6767                      │
#   ├────────────────────────────┼────────────────────────────┤
#   │ serve_anygrasp             │ serve_curobo (portal RPC)  │
#   │ :8122                      │ :8611                      │
#   └────────────────────────────┴────────────────────────────┘
#
#   Tmux window "LLM/VLM serving"
#   ┌────────────────────────────────────────────────────────────┐
#   │ cap.agent.providers.nvidia_server  :8765  (live dashboard) │
#   └────────────────────────────────────────────────────────────┘
#
#   Background process (nohup, log in $PROJECT_DIR/logs/launch_evaluation.log):
#     evaluation (launch.py --mode=evaluation --no-attach)
#
# Also handles:
#   - Killing any stale listener on the service ports above.
#   - Killing leftover SSH local-port-forward tunnels for SAM3 / BundleSDF.
#
# Bridge (:8201) and voice (:8202) are NOT launched here — they're agent-layer
# concerns owned by the table-bussing wrappers (next to cap_server/agent/ui).
#
# The table-bussing wrappers (tmux/table_bussing/launch_table_bussing_local_*.sh)
# only own window 0 ("cap server/agent/ui") and call this script with
# `--session cap_family_bucket --no-attach` to append windows 1 + 2 + the
# background processes. Single source of truth.
#
# Default session: `cap_family_bucket`. Use `--session NAME` to attach the
# two windows to an existing session.
#
# Standalone:
#   bash tmux/launch_realworld_localserver_realsense.sh
#
# Library mode (used by the table-bussing wrappers):
#   bash tmux/launch_realworld_localserver_realsense.sh \
#       --session cap_family_bucket --no-attach

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pick up OpenSSL 1.1 (AnyGrasp), node/npm (cap/ui), API keys (NVIDIA, Gemini,
# ElevenLabs), and SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO — all
# written by install/install_cap.sh.
[[ -f "$PROJECT_DIR/.forge_env" ]] && source "$PROJECT_DIR/.forge_env"

# ── Defaults ───────────────────────────────────────────────────────────────
SESSION="cap_family_bucket"
ATTACH=true
SAM3_HOST="localhost"
SAM3_PORT="6767"
ANYGRASP_HOST="localhost"
ANYGRASP_PORT="8122"
BUNDLESDF_HOST="localhost"
BUNDLESDF_PORT="8119"
CUROBO_PORT="8611"
PYROKI_PORT="9600"
NVIDIA_PROVIDER_HOST="127.0.0.1"
NVIDIA_PROVIDER_PORT="8765"
CAP_SERVER_PORT="8300"  # used only as a config argument to serve_bundlesdf.py

START_BUNDLESDF=true
START_CUROBO=true
START_PYROKI=true
START_NVIDIA=true
START_EVALUATION=true

# ── Args ───────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)         SESSION="$2"; shift 2 ;;
        --no-attach)       ATTACH=false; shift ;;
        --sam3-port)       SAM3_PORT="$2"; shift 2 ;;
        --anygrasp-port)   ANYGRASP_PORT="$2"; shift 2 ;;
        --bundlesdf-port)  BUNDLESDF_PORT="$2"; shift 2 ;;
        --curobo-port)     CUROBO_PORT="$2"; shift 2 ;;
        --pyroki-port)     PYROKI_PORT="$2"; shift 2 ;;
        --nvidia-port)     NVIDIA_PROVIDER_PORT="$2"; shift 2 ;;
        --cap-server-port) CAP_SERVER_PORT="$2"; shift 2 ;;
        --no-bundlesdf)    START_BUNDLESDF=false; shift ;;
        --no-curobo)       START_CUROBO=false; shift ;;
        --no-pyroki)       START_PYROKI=false; shift ;;
        --no-nvidia)       START_NVIDIA=false; shift ;;
        --no-evaluation)   START_EVALUATION=false; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//' | head -50
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SAM3_URL="http://${SAM3_HOST}:${SAM3_PORT}"
ANYGRASP_URL="http://${ANYGRASP_HOST}:${ANYGRASP_PORT}"

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for service readiness checks." >&2
    exit 1
fi

# ── Pre-flight: kill any stale listener / SSH tunnel ───────────────────────
kill_listener() {
    local port="$1"
    if ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .; then
        echo "[realworld] Stopping existing listener on port ${port}"
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
        sleep 1
    fi
}

stop_ssh_port_forward_tunnels() {
    local pids
    pids="$(pgrep -u "$USER" -f 'ssh .*(-L 6767:localhost:6767|-L 8119:localhost:8119)' || true)"
    if [[ -n "${pids}" ]]; then
        echo "[realworld] Stopping existing SSH local-port-forward tunnels: ${pids}"
        kill ${pids} 2>/dev/null || true
        sleep 1
    fi
}

stop_ssh_port_forward_tunnels
kill_listener "$SAM3_PORT"
kill_listener "$ANYGRASP_PORT"
$START_BUNDLESDF && kill_listener "$BUNDLESDF_PORT"
$START_CUROBO    && kill_listener "$CUROBO_PORT"
$START_PYROKI    && kill_listener "$PYROKI_PORT"
$START_NVIDIA    && kill_listener "$NVIDIA_PROVIDER_PORT"

# ── Background process (no pane; nohup'd, log in $PROJECT_DIR/logs/) ───────
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
EVALUATION_LOG_FILE="${LOG_DIR}/launch_evaluation.log"

if $START_EVALUATION; then
    echo "[realworld] Launching evaluation background process..."
    nohup bash -lc "cd \"$PROJECT_DIR\" && uv run python launch.py --mode=evaluation --no-attach" \
        >"$EVALUATION_LOG_FILE" 2>&1 &
    echo "[realworld]   Evaluation log: $EVALUATION_LOG_FILE"
fi

# ── Tmux: ensure session exists, then create the two service windows ──────
# Window names contain spaces/slashes; we never address them by name after
# creation (pane IDs from `-P -F '#{pane_id}'` are used instead).
PCS_WIN_NAME="perception/control serving"
LLM_WIN_NAME="LLM/VLM serving"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[realworld] creating tmux session '$SESSION'"
    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" -n "$PCS_WIN_NAME"
    BUNDLESDF_PANE="$(tmux list-panes -t "$SESSION:^.0" -F '#{pane_id}' | head -1)"
else
    echo "[realworld] attaching to existing tmux session '$SESSION'"
    BUNDLESDF_PANE="$(tmux new-window -t "$SESSION" -n "$PCS_WIN_NAME" -c "$PROJECT_DIR" -P -F '#{pane_id}')"
fi

# "perception/control serving" window: 2×2 (bundlesdf | sam3 / anygrasp | curobo)
SAM3_PANE="$(tmux split-window -h -t "$BUNDLESDF_PANE" -c "$PROJECT_DIR" -P -F '#{pane_id}')"
ANYGRASP_PANE="$(tmux split-window -v -t "$BUNDLESDF_PANE" -c "$PROJECT_DIR" -P -F '#{pane_id}')"
CUROBO_PANE="$(tmux split-window -v -t "$SAM3_PANE" -c "$PROJECT_DIR" -P -F '#{pane_id}')"
PYROKI_PANE="$(tmux split-window -v -t "$CUROBO_PANE" -c "$PROJECT_DIR" -P -F '#{pane_id}')"
PCS_WIN_TARGET="$(tmux display-message -p -t "$BUNDLESDF_PANE" "#{session_name}:#{window_index}")"
tmux select-layout -t "$PCS_WIN_TARGET" tiled >/dev/null

# "LLM/VLM serving" window: single pane (full width) for the NVIDIA dashboard.
if $START_NVIDIA; then
    NVIDIA_PROVIDER_PANE="$(tmux new-window -t "$SESSION" -n "$LLM_WIN_NAME" -c "$PROJECT_DIR" -P -F '#{pane_id}')"
fi

# ── Pane commands ──────────────────────────────────────────────────────────
# Every pane sources .forge_env so it picks up CAP_AGENT_NAME, NVIDIA_API_KEY,
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO, etc. even if the tmux
# server was started before those secrets existed.
FORGE_ENV_CMD="[ -f $PROJECT_DIR/.forge_env ] && source $PROJECT_DIR/.forge_env"

# SAM3
tmux send-keys -t "$SAM3_PANE" \
    "$FORGE_ENV_CMD; HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_VERBOSITY=error TRANSFORMERS_VERBOSITY=error uv run python tools/vision/serve_sam3.py --port $SAM3_PORT --preload" Enter

# AnyGrasp
tmux send-keys -t "$ANYGRASP_PANE" \
    "$FORGE_ENV_CMD; ANYGRASP_PORT=$ANYGRASP_PORT bash ./tools/vision/launch_anygrasp_server.sh" Enter

# BundleSDF
if $START_BUNDLESDF; then
    tmux send-keys -t "$BUNDLESDF_PANE" \
        "$FORGE_ENV_CMD; export BUNDLESDF_RUNTIME_LIB_DIR=\"$PROJECT_DIR/third_party/bundlesdf_5090\"; export LD_LIBRARY_PATH=\"\$BUNDLESDF_RUNTIME_LIB_DIR:\${LD_LIBRARY_PATH:-}\"; HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_VERBOSITY=error TRANSFORMERS_VERBOSITY=error uv run --extra cap_tools python tools/vision/serve_bundlesdf.py --port $BUNDLESDF_PORT --cap_server_port $CAP_SERVER_PORT --sam3_url $SAM3_URL" Enter
else
    tmux send-keys -t "$BUNDLESDF_PANE" \
        "echo '[startup] BundleSDF skipped (--no-bundlesdf). Tracking tools will fail at use time.'" Enter
fi

# cuRobo
if $START_CUROBO; then
    tmux send-keys -t "$CUROBO_PANE" \
        "$FORGE_ENV_CMD; uv run python experimental/serve_portal_motion_planner.py --port $CUROBO_PORT --solver-speed fast --robot-type yam" Enter
else
    tmux send-keys -t "$CUROBO_PANE" \
        "echo '[startup] cuRobo skipped (--no-curobo). Motion planning tools will fail at use time.'" Enter
fi

# PyRoki straight-line IK service for short refinement approaches.
if $START_PYROKI; then
    tmux send-keys -t "$PYROKI_PANE" \
        "$FORGE_ENV_CMD; CAP_TOP_CAMERA_BACKEND=realsense uv run python scripts/launch_pyroki_server.py --robot yam_station --target-link left_grasp --port $PYROKI_PORT" Enter
else
    tmux send-keys -t "$PYROKI_PANE" \
        "echo '[startup] PyRoki skipped (--no-pyroki). Refinement final approach will fall back.'" Enter
fi

# NVIDIA provider
if $START_NVIDIA; then
    tmux send-keys -t "$NVIDIA_PROVIDER_PANE" \
        "$FORGE_ENV_CMD; uv run python -m cap.agent.providers.nvidia_server serve --host $NVIDIA_PROVIDER_HOST --port $NVIDIA_PROVIDER_PORT --fresh-start --global-request-delay-s 1.0 --max-concurrent-per-key 1 --dashboard" Enter
fi

# ── Banner ────────────────────────────────────────────────────────────────
echo ""
echo "  [realworld] windows added to tmux session '$SESSION':"
echo "    '$PCS_WIN_NAME':"
echo "      serve_sam3       :$SAM3_PORT"
echo "      serve_anygrasp   :$ANYGRASP_PORT"
$START_BUNDLESDF && echo "      serve_bundlesdf  :$BUNDLESDF_PORT"
$START_CUROBO    && echo "      serve_curobo     :$CUROBO_PORT  (portal RPC)"
$START_PYROKI    && echo "      serve_pyroki     :$PYROKI_PORT  (HTTP)"
$START_NVIDIA    && echo "    '$LLM_WIN_NAME':"
$START_NVIDIA    && echo "      nvidia_provider  :$NVIDIA_PROVIDER_PORT"
$START_EVALUATION && echo "  background process:"
$START_EVALUATION && echo "      evaluation       (log: $EVALUATION_LOG_FILE)"
echo ""

if $ATTACH; then
    tmux select-window -t "$PCS_WIN_TARGET"
    tmux attach-session -t "$SESSION"
fi
