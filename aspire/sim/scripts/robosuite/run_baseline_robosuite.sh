#!/bin/bash
# Run the traced multimodel-ensemble baseline across the 7 robosuite tasks (M4 tier).
#
# Reads NVIDIA API keys from two protected files: one for code-gen (ensemble +
# synthesis) and one for VDM. Keys are round-robined via local proxies on
# :8110 and :8111. Proxies are started and torn down by this script.
#
# Uses *Traced API classes (trace.json + keyframes/ per trial).
#
# Usage:
#   CODEGEN_KEY_FILE=/secure/codegen.keys \
#   VDM_KEY_FILE=/secure/vdm.keys \
#     scripts/robosuite/run_baseline_robosuite.sh
#
# Each file contains one API key per line and should be readable only by its owner.
#
# Seeds 101-125 per task (baseline convention). Logs → logs/robosuite_baseline.log.

set -euo pipefail

if [ "$#" -ne 0 ] || [ -z "${CODEGEN_KEY_FILE:-}" ] || [ -z "${VDM_KEY_FILE:-}" ]; then
  echo "Usage: CODEGEN_KEY_FILE=/secure/codegen.keys VDM_KEY_FILE=/secure/vdm.keys $0"
  exit 1
fi

for key_file in "$CODEGEN_KEY_FILE" "$VDM_KEY_FILE"; do
  if [ ! -f "$key_file" ] || [ ! -r "$key_file" ] || [ ! -s "$key_file" ]; then
    echo "ERROR: key file must exist, be readable, and be non-empty: $key_file" >&2
    exit 1
  fi
done

CODEGEN_PORT=8110
VDM_PORT=8111
NVIDIA_BASE=https://inference-api.nvidia.com/v1/

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_ROOT="$(cd "$WORKSPACE_ROOT/../.." && pwd)"
cd "$WORKSPACE_ROOT"
export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Start proxies --------------------------------------------------------
mkdir -p logs
pkill -f "openrouter_server.*--port $CODEGEN_PORT" 2>/dev/null || true
pkill -f "openrouter_server.*--port $VDM_PORT"     2>/dev/null || true
sleep 1

echo "Starting code-gen proxy on :$CODEGEN_PORT (2-key rotation)..."
nohup .venv/bin/python3 -m aspire.sim.cap.serving.openrouter_server \
  --key-file "$CODEGEN_KEY_FILE" \
  --host 127.0.0.1 \
  --port "$CODEGEN_PORT" \
  --base-url "$NVIDIA_BASE" \
  > logs/proxy_codegen.log 2>&1 &
CODEGEN_PROXY_PID=$!

echo "Starting VDM proxy on :$VDM_PORT (2-key rotation)..."
nohup .venv/bin/python3 -m aspire.sim.cap.serving.openrouter_server \
  --key-file "$VDM_KEY_FILE" \
  --host 127.0.0.1 \
  --port "$VDM_PORT" \
  --base-url "$NVIDIA_BASE" \
  > logs/proxy_vdm.log 2>&1 &
VDM_PROXY_PID=$!

# --- Ensure cleanup on exit -----------------------------------------------
cleanup() {
  echo ""
  echo "Cleaning up proxies..."
  kill "$CODEGEN_PROXY_PID" "$VDM_PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Wait for proxies to be healthy ---------------------------------------
for port in $CODEGEN_PORT $VDM_PORT; do
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "  proxy :$port UP"
      break
    fi
    sleep 1
    if [ "$i" = "10" ]; then
      echo "ERROR: proxy on :$port did not come up in 10s"
      exit 1
    fi
  done
done

# --- Run the batch --------------------------------------------------------
.venv/bin/python3 -u cap/envs/scripts/run_robosuite_batch.py \
  --args.config-paths \
    env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml \
    env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml \
    env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml \
    env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml \
    env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml \
    env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml \
    env_configs/robosuite/spill_wipe_multimodel_aspire_traced.yaml \
  --args.server-url "http://127.0.0.1:$CODEGEN_PORT/chat/completions" \
  --args.api-key dummy \
  --args.models 'ensemble_multimodel' \
  --args.visual-differencing-model 'aws/anthropic/bedrock-claude-opus-4-6' \
  --args.visual-differencing-model-server-url "http://127.0.0.1:$VDM_PORT/chat/completions" \
  --args.visual-differencing-model-api-key dummy \
  --args.num-workers 5 \
  --args.total-trials 125 \
  --args.output-dir ./outputs/baseline_robosuite_multimodel_ensemble_traced \
  2>&1 | tee logs/robosuite_baseline.log
