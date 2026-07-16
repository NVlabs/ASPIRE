#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
source .forge_env
export OPENFORGE_ALLOW_PHYSICAL_MOTION="${OPENFORGE_ALLOW_PHYSICAL_MOTION:-1}"

exec uv run python cap/saved_scripts/drawer_observe_pull_open_x.py "$@"
