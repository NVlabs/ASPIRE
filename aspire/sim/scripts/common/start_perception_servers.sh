#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start SAM3, GraspNet, PyRoKi, and optionally Molmo perception servers.
# Safe to re-run: kills existing server processes first.
#
# Usage:
#   bash scripts/common/start_perception_servers.sh [--gpu-sam3 N] [--gpu-graspnet N]
#   bash scripts/common/start_perception_servers.sh --with-molmo [--gpu-molmo N]
#   ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3 bash scripts/common/start_perception_servers.sh
#
# Python selection:
#   1. $ASPIRE_PERCEPTION_PYTHON, if set
#   2. .venv-libero/bin/python3, if present
#   3. .venv-perception/bin/python3, if present
#   4. .venv/bin/python3, if present
#
# Molmo:
#   Default is auto: start Molmo only if the selected env has a vllm executable.
#   Use --with-molmo to require it, or --no-molmo to skip it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU_SAM3=0
GPU_GRASPNET=1
GPU_MOLMO=2
MOLMO_MODE=auto
PYTHON_BIN="${ASPIRE_PERCEPTION_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-sam3)     GPU_SAM3="$2"; shift 2 ;;
    --gpu-graspnet) GPU_GRASPNET="$2"; shift 2 ;;
    --gpu-molmo)    GPU_MOLMO="$2"; shift 2 ;;
    --with-molmo)   MOLMO_MODE=true; shift 1 ;;
    --no-molmo)     MOLMO_MODE=false; shift 1 ;;
    --python)       PYTHON_BIN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    .venv-libero/bin/python3 \
    .venv-perception/bin/python3 \
    .venv/bin/python3
  do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: no perception Python found."
  echo "Set ASPIRE_PERCEPTION_PYTHON or create .venv-libero / .venv-perception / .venv."
  exit 1
fi

VENV_BIN_DIR="$(dirname "$PYTHON_BIN")"
VLLM_BIN="$VENV_BIN_DIR/vllm"
START_MOLMO=false
if [[ "$MOLMO_MODE" == "true" ]]; then
  if [[ ! -x "$VLLM_BIN" ]]; then
    echo "ERROR: --with-molmo requested, but $VLLM_BIN is not executable."
    echo "Use --python /path/to/python from an env with vllm, or pass --no-molmo."
    exit 1
  fi
  START_MOLMO=true
elif [[ "$MOLMO_MODE" == "auto" && -x "$VLLM_BIN" ]]; then
  START_MOLMO=true
fi

echo "Using perception Python: $PYTHON_BIN"
if $START_MOLMO; then
  echo "Molmo: enabled"
else
  echo "Molmo: skipped"
fi

echo "Stopping any existing perception servers..."
pkill -f launch_sam3_server.py             2>/dev/null && echo "  Killed SAM3" || true
pkill -f launch_contact_graspnet_server.py 2>/dev/null && echo "  Killed GraspNet" || true
pkill -f launch_pyroki_server.py           2>/dev/null && echo "  Killed PyRoKi" || true
pkill -f "vllm.*Molmo"                     2>/dev/null && echo "  Killed Molmo" || true
sleep 2

echo "Starting SAM3 on GPU $GPU_SAM3, port 8114..."
CUDA_VISIBLE_DEVICES=$GPU_SAM3 nohup "$PYTHON_BIN" -u \
  cap/serving/launch_sam3_server.py \
  --device cuda --port 8114 --host 127.0.0.1 \
  > /tmp/sam3.log 2>&1 &
echo "  PID $!"

echo "Starting GraspNet on GPU $GPU_GRASPNET, port 8115..."
CUDA_VISIBLE_DEVICES=$GPU_GRASPNET nohup "$PYTHON_BIN" -u \
  cap/serving/launch_contact_graspnet_server.py \
  --port 8115 --host 127.0.0.1 \
  > /tmp/graspnet.log 2>&1 &
echo "  PID $!"

echo "Starting PyRoKi on CPU, port 8116..."
nohup "$PYTHON_BIN" -u \
  cap/serving/launch_pyroki_server.py \
  --port 8116 --host 127.0.0.1 \
  --robot panda_description --target-link panda_hand \
  > /tmp/pyroki.log 2>&1 &
echo "  PID $!"

if $START_MOLMO; then
  echo "Starting Molmo on GPU $GPU_MOLMO, port 8122..."
  CUDA_VISIBLE_DEVICES=$GPU_MOLMO nohup "$VLLM_BIN" serve allenai/Molmo2-8B \
    --trust-remote-code --port 8122 --host 127.0.0.1 \
    --max-num-batched-tokens 36864 \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 2}' \
    > /tmp/molmo.log 2>&1 &
  echo "  PID $!"
fi

WAIT=90
if $START_MOLMO; then
  WAIT=180
fi

echo ""
echo "Waiting ${WAIT}s for models to load..."
for i in $(seq $WAIT -10 10); do
  printf "\r  %3ds remaining..." "$i"
  sleep 10
done
printf "\r  Checking servers...      \n"

ALL_OK=true
for port in 8114 8115 8116; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null || echo "000")
  if [[ "$CODE" == "000" ]]; then
    case $port in
      8114) LOG=/tmp/sam3.log; NAME=SAM3 ;;
      8115) LOG=/tmp/graspnet.log; NAME=GraspNet ;;
      8116) LOG=/tmp/pyroki.log; NAME=PyRoKi ;;
    esac
    echo "FAIL port $port ($NAME): DOWN"
    echo "  Last 5 lines of $LOG:"
    tail -5 "$LOG" | sed 's/^/    /'
    ALL_OK=false
  else
    case $port in
      8114) NAME=SAM3 ;;
      8115) NAME=GraspNet ;;
      8116) NAME=PyRoKi ;;
    esac
    echo "OK port $port ($NAME): UP"
  fi
done

if $START_MOLMO; then
  MOLMO_MODEL=$(curl -s --max-time 5 http://127.0.0.1:8122/v1/models 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
  if [[ -z "$MOLMO_MODEL" ]]; then
    echo "FAIL port 8122 (Molmo): DOWN"
    echo "  Last 5 lines of /tmp/molmo.log:"
    tail -5 /tmp/molmo.log | sed 's/^/    /'
    ALL_OK=false
  else
    echo "OK port 8122 (Molmo): UP [$MOLMO_MODEL]"
  fi
fi

echo ""
if $ALL_OK; then
  echo "All requested perception servers are ready."
  echo "Logs: /tmp/sam3.log /tmp/graspnet.log /tmp/pyroki.log /tmp/molmo.log"
else
  echo "Some servers failed. Check the logs above."
  exit 1
fi
