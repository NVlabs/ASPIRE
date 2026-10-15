#!/usr/bin/env bash
# Install a dedicated client, arm-service, and MCP environment in this checkout.
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ASPIRE_ROOT"
env -u PYTHONPATH -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT="$ASPIRE_ROOT/.venv-system2" \
  uv sync --frozen --extra yam-real --extra yam-real-server --extra agent --extra dev
"$ASPIRE_ROOT/.venv-system2/bin/python" -c \
  'import cap_harness, portal, pyrealsense2, viser, yourdfpy; from cap_harness.agent.mcp import CapMcpBridge; print("System 2 environment ready:", cap_harness.__file__)'
