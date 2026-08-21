#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# ASPIRE intentionally does not redistribute the AnyGrasp SDK, model, vendor
# demo server, or machine-license material. Authorized users may point this
# interface launcher at a separately installed service implementation.

ANYGRASP_PORT="${ANYGRASP_PORT:-8122}"
ANYGRASP_SERVICE_URL="${ANYGRASP_SERVICE_URL:-http://127.0.0.1:${ANYGRASP_PORT}}"

if curl --fail --silent --show-error "${ANYGRASP_SERVICE_URL%/}/health" >/dev/null 2>&1; then
  echo "[anygrasp] external service is already healthy at ${ANYGRASP_SERVICE_URL}"
  exit 0
fi

if [[ -z "${ANYGRASP_SERVER_COMMAND:-}" ]]; then
  cat >&2 <<EOF
[anygrasp] ASPIRE does not include an AnyGrasp server implementation.
[anygrasp] Start your separately licensed vendor service at ${ANYGRASP_SERVICE_URL},
[anygrasp] or set ANYGRASP_SERVER_COMMAND to an authorized local launch command.
EOF
  exit 2
fi

echo "[anygrasp] launching separately licensed external service for ${ANYGRASP_SERVICE_URL}"
exec bash -lc "$ANYGRASP_SERVER_COMMAND"
