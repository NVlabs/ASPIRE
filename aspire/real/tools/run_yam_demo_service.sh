#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start one service used by the canonical saved-script YAM demo.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
service="${1:-}"

if [[ -z "$service" || "$service" == "-h" || "$service" == "--help" || "$service" == "help" ]]; then
  echo "Usage: bash tools/run_yam_demo_service.sh {left-arm|right-arm|camera-portal|sam3|bundlesdf}"
  exit 0
fi
case "$service" in
  left-arm|right-arm|camera-portal|sam3|bundlesdf) ;;
  *) echo "Unknown YAM demo service: $service" >&2; exit 2 ;;
esac

cd "$ROOT"

if [[ ! -f .forge_env ]]; then
  echo "Missing $ROOT/.forge_env; copy .forge_env.example and configure it." >&2
  exit 2
fi

set +u
# shellcheck disable=SC1091
source .forge_env
set -u

PYTHON="${YAM_DEMO_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing executable Python environment: $PYTHON" >&2
  exit 2
fi

case "$service" in
  left-arm)
    exec "$PYTHON" robot/yam/arm_server.py --mode follower --side left
    ;;
  right-arm)
    exec "$PYTHON" robot/yam/arm_server.py --mode follower --side right
    ;;
  camera-portal)
    exec "$PYTHON" tools/vision/serve_real_yam_camera_portal.py \
      --port "${YAM_DEMO_CAMERA_PORTAL_PORT:-8300}"
    ;;
  sam3)
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_VERBOSITY=error
    export TRANSFORMERS_VERBOSITY=error
    exec "$PYTHON" tools/vision/serve_sam3.py \
      --port "${YAM_DEMO_SAM3_PORT:-6767}" --preload
    ;;
  bundlesdf)
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_VERBOSITY=error
    export TRANSFORMERS_VERBOSITY=error
    export BUNDLESDF_RUNTIME_LIB_DIR="${BUNDLESDF_RUNTIME_LIB_DIR:-${BUNDLESDF_REPO_LIB_DIR:-$ROOT/third_party/bundlesdf_5090}}"
    export LD_LIBRARY_PATH="$BUNDLESDF_RUNTIME_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    exec "$PYTHON" tools/vision/serve_bundlesdf.py \
      --port "${YAM_DEMO_BUNDLESDF_PORT:-8119}" \
      --cap_server_host 127.0.0.1 \
      --cap_server_port "${YAM_DEMO_CAMERA_PORTAL_PORT:-8300}" \
      --sam3_url "http://127.0.0.1:${YAM_DEMO_SAM3_PORT:-6767}"
    ;;
esac
