#!/usr/bin/env bash
# Thin wrapper: bootstrap the pyroki provider from the declarative topology.
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/bootstrap_provider.sh" pyroki "$@"
