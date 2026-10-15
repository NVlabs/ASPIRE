#!/usr/bin/env bash
# Start the repository's persistent MCP runtime; callers select the station.
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASPIRE_PYTHON="$ASPIRE_ROOT/.venv-system2/bin/python"
[[ -x "$ASPIRE_PYTHON" ]] || { printf 'Run scripts/bootstrap_system2.sh first.\n' >&2; exit 1; }
cd "$ASPIRE_ROOT"
export OPENBLAS_NUM_THREADS=1
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
exec env -u PYTHONPATH "$ASPIRE_PYTHON" -m cap_harness.agent "$@"
