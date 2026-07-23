#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
source .forge_env
export OPENFORGE_ALLOW_PHYSICAL_MOTION="${OPENFORGE_ALLOW_PHYSICAL_MOTION:-1}"

set +e
uv run python cap/saved_scripts/drawer_push_close_x.py "$@"
status="$?"
set -e

if [ "$status" -eq 0 ] && [ "${OPENFORGE_DRAWER_CLOSE_SKIP_HOME:-0}" != "1" ]; then
  bash cap/saved_scripts/shell_scripts/home.sh
fi

exit "$status"
