#!/usr/bin/env bash
# Attach a coding agent after the MCP runtime has started.
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ASPIRE_ROOT/.uv-cache}"
ASPIRE_PORT="${ASPIRE_MCP_PORT:-8222}"
[[ "$ASPIRE_PORT" =~ ^[0-9]+$ ]] || { printf 'ASPIRE_MCP_PORT must be numeric.\n' >&2; exit 1; }
"$ASPIRE_ROOT/.venv-system2/bin/python" - "$ASPIRE_PORT" <<'PY'
import json, sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f'http://127.0.0.1:{sys.argv[1]}/health', timeout=5) as response:
    if not json.load(response).get('ok'):
        raise SystemExit('Start the System 2 MCP runtime first.')
PY
ASPIRE_GUIDE="$(cat "$ASPIRE_ROOT/docs/system2-agent-guide.md")"
ASPIRE_PERMISSIONS=()
ASPIRE_STARTUP="This is an observation-only startup. Discover the CAP tools, observe the station, report readiness, and do not home or move the robot."
if [[ "${1:-}" == "--allow-motion" ]]; then
  shift
  ASPIRE_PERMISSIONS+=(
    -c 'mcp_servers.cap.tools.cap_program_go_home.approval_mode="approve"'
    -c 'mcp_servers.cap.tools.cap_move_to_pose.approval_mode="approve"'
    -c 'mcp_servers.cap.tools.cap_move_synchronized.approval_mode="approve"'
    -c 'mcp_servers.cap.tools.cap_move_to_joints.approval_mode="approve"'
    -c 'mcp_servers.cap.tools.cap_set_gripper.approval_mode="approve"'
  )
  ASPIRE_STARTUP="This is a motion-enabled startup. Discover the CAP tools, then call cap_program_go_home exactly once before the operator task. This opens both grippers and homes both arms. Wait for its terminal result and fresh observations. Report READY with the measured home_verified, joint residuals, and gripper state. If Home fails or either gripper is not open, report the failure and stop instead of starting the task."
fi
for ASPIRE_TOOL in get_capability_graph observe get_job read_operator_messages cap_get_robot_state cap_localize_object cap_yam_real__get_controller_metadata; do
  ASPIRE_PERMISSIONS+=(-c "mcp_servers.cap.tools.$ASPIRE_TOOL.approval_mode=\"approve\"")
done
ASPIRE_MODE=()
if [[ "${1:-}" == "exec" ]]; then ASPIRE_MODE=(exec); shift; fi
ASPIRE_TASK="${*:-After startup completes, wait for the operator task.}"
exec codex "${ASPIRE_MODE[@]}" -C "$ASPIRE_ROOT" \
  -c 'approval_policy="never"' \
  -c 'sandbox_mode="workspace-write"' \
  -c 'sandbox_workspace_write.network_access=true' \
  -c "mcp_servers.cap.url=\"http://127.0.0.1:$ASPIRE_PORT/mcp/\"" \
  -c 'mcp_servers.cap.enabled=true' -c 'mcp_servers.cap.required=true' \
  -c 'mcp_servers.cap.tool_timeout_sec=240' \
  "${ASPIRE_PERMISSIONS[@]}" \
  "$ASPIRE_GUIDE

Session startup:
$ASPIRE_STARTUP

Operator task:
$ASPIRE_TASK"
