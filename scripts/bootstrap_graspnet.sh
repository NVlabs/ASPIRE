#!/usr/bin/env bash
# Thin wrapper: bootstrap the Contact-GraspNet provider from the topology.
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/bootstrap_provider.sh" contact_graspnet "$@"
