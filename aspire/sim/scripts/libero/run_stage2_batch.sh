#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Stage 2 token-scaling eval for multiple (suite, task) pairs across GPUs.
# Usage: bash scripts/libero/run_stage2_batch.sh <worktree> <snapshot> <gpu_list> <suite1:task1> <suite2:task2> ...
# Example: bash scripts/libero/run_stage2_batch.sh /path/to/worktree snapshot-N90 "3,4,5,6,7" \
#          "libero_10_swap:TASK_A" "libero_10_task:TASK_B" ...

set -euo pipefail

WORKTREE="$1"
PYTHON_ROOT="$(cd "$WORKTREE/../.." && pwd)"
SNAPSHOT="$2"
GPU_LIST="$3"
shift 3

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

TASKS=("$@")
NUM_TASKS=${#TASKS[@]}

echo "=== Stage 2 Batch Runner ==="
echo "Worktree: $WORKTREE"
echo "Snapshot: $SNAPSHOT"
echo "GPUs: ${GPUS[*]}"
echo "Tasks: $NUM_TASKS"
echo ""

PIDS=()
TASK_DESCS=()

for i in "${!TASKS[@]}"; do
    SUITE_TASK="${TASKS[$i]}"
    SUITE="${SUITE_TASK%%:*}"
    TASK="${SUITE_TASK#*:}"
    GPU_IDX=$((i % NUM_GPUS))
    GPU="${GPUS[$GPU_IDX]}"

    LOGFILE="/tmp/stage2_${SNAPSHOT}_${SUITE}_$(echo "$TASK" | cut -c1-30).log"

    echo "[$((i+1))/$NUM_TASKS] GPU $GPU: $SUITE / ${TASK:0:50}"

    PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$WORKTREE/.venv-libero/bin/python3" \
        "$WORKTREE/scripts/libero/run_stage2_token_scaling.py" \
        --worktree "$WORKTREE" \
        --snapshot "$SNAPSHOT" \
        --suite "$SUITE" \
        --task "$TASK" \
        --gpu "$GPU" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
    TASK_DESCS+=("$SUITE/${TASK:0:40}")
done

echo ""
echo "All $NUM_TASKS jobs launched. Waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { echo "FAILED: ${TASK_DESCS[$i]}"; FAILED=$((FAILED+1)); }
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All $NUM_TASKS Stage 2 jobs completed successfully."
else
    echo "$FAILED/$NUM_TASKS Stage 2 jobs FAILED."
fi

echo ""
echo "=== Results ==="
for i in "${!TASKS[@]}"; do
    SUITE_TASK="${TASKS[$i]}"
    SUITE="${SUITE_TASK%%:*}"
    TASK="${SUITE_TASK#*:}"
    BASE="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/debug_eval/$SUITE/$TASK"
    echo "--- $SUITE / ${TASK:0:50} ---"
    for pct in 0pct 25pct 50pct 75pct 100pct; do
        RESULT="$BASE/stage2_${pct}/stage2_result.json"
        if [ -f "$RESULT" ]; then
            python3 -c "import json; d=json.load(open('$RESULT')); print(f'  {d[\"pct_label\"]:>6s}: {d[\"n_pass\"]}/{d[\"n_total\"]} = {d[\"pass_rate\"]*100:.0f}%  ({d[\"code\"]})')"
        else
            echo "  $pct: NOT DONE"
        fi
    done
done
