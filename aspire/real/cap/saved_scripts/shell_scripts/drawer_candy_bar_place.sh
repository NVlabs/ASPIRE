#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
source .forge_env
export OPENFORGE_ALLOW_PHYSICAL_MOTION="${OPENFORGE_ALLOW_PHYSICAL_MOTION:-1}"

exec uv run python cap/saved_scripts/drawer_candy_bar_place.py "$@"
