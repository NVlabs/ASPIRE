#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Rerun Stage 1 (debug seeds 101-125) for 4 tasks with 5 workers each
# Usage: bash scripts/robosuite/run_debug_fix_code.sh
#
# This script is for rerunning the fix loop on debug seeds 101–125 (the seed range
# used during debugging).

set -e

BASELINE="outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel"
OUTDIR="outputs/robosuite_fix_debug"
GPUS=(3 4 5 6 7)
WORKERS_PER_TASK=5
SEEDS_PER_WORKER=5  # 25 / 5

declare -A TASKS
TASKS[cube_restack]="env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml|cube_restack_multimodel_aspire_traced"
TASKS[two_arm_handover]="env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml|two_arm_handover_multimodel_aspire_traced"
TASKS[two_arm_lift]="env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml|two_arm_lift_multimodel_aspire_traced"
TASKS[nut_assembly]="env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml|nut_assembly_multimodel_aspire_traced"

TASK_ORDER=(cube_restack two_arm_handover two_arm_lift nut_assembly)

gpu_idx=0
pids=()

for task in "${TASK_ORDER[@]}"; do
    IFS='|' read -r CONFIG STEM <<< "${TASKS[$task]}"
    FIX="$BASELINE/$STEM/fix_code.py"

    if [ ! -f "$FIX" ]; then
        echo "ERROR: fix_code.py not found for $task at $FIX"
        continue
    fi

    for w in $(seq 0 $((WORKERS_PER_TASK - 1))); do
        GPU=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
        START=$(( w * SEEDS_PER_WORKER + 101 ))
        END=$(( (w + 1) * SEEDS_PER_WORKER + 100 ))
        LOGDIR="/tmp/rerun_debug_${task}"
        mkdir -p "$LOGDIR"

        (
            for trial in $(seq $START $END); do
                MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
                .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
                    --args.config "$CONFIG" \
                    --args.trial $trial \
                    --args.replay-code "$FIX" \
                    --args.output-dir "$OUTDIR" > "$LOGDIR/${trial}.log" 2>&1
                reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "$LOGDIR/${trial}.log" | tail -1 | sed 's/reward_//')
                echo "[$task GPU$GPU w$w] Trial $trial: ${reward:-ERROR}"
            done
        ) &
        pids+=($!)
        gpu_idx=$((gpu_idx + 1))
    done
done

echo ""
echo "Launched ${#pids[@]} workers across GPUs ${GPUS[*]}"
echo "Waiting for all workers to finish..."
echo ""

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "========== ALL WORKERS DONE =========="
echo ""

for task in "${TASK_ORDER[@]}"; do
    IFS='|' read -r CONFIG STEM <<< "${TASKS[$task]}"
    TASK_OUTDIR="$OUTDIR/$STEM"
    succ=$(find "$TASK_OUTDIR" -maxdepth 5 -type d -name "*reward_1.000*" 2>/dev/null | wc -l)
    total=$(find "$TASK_OUTDIR" -maxdepth 5 -type d -name "trial_*" 2>/dev/null | grep -oE 'trial_[0-9]+' | sort -u | wc -l)
    echo "$task: $succ/$total"
done
