#!/bin/bash
# Rerun Stage 2 for 5 tasks with 5 workers each (25 total, 5 per GPU)
# Usage: bash scripts/robosuite/run_eval_fix_code.sh
#
# Replays each task's `fix_code.py` across eval seeds 1–100.

set -e

BASELINE="outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel"
OUTDIR="outputs/robosuite_fix_eval"
GPUS=(3 4 5 6 7)
WORKERS_PER_TASK=5
SEEDS_PER_WORKER=20  # 100 / 5

declare -A TASKS
TASKS[cube_lifting]="env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml|cube_lifting_multimodel_aspire_traced"
TASKS[cube_restack]="env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml|cube_restack_multimodel_aspire_traced"
TASKS[cube_stack]="env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml|cube_stack_multimodel_aspire_traced"
TASKS[spill_wipe]="env_configs/robosuite/spill_wipe_multimodel_aspire_traced.yaml|spill_wipe_multimodel_aspire_traced"
TASKS[two_arm_handover]="env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml|two_arm_handover_multimodel_aspire_traced"

TASK_ORDER=(cube_lifting cube_restack cube_stack spill_wipe two_arm_handover)

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
        START=$(( w * SEEDS_PER_WORKER + 1 ))
        END=$(( (w + 1) * SEEDS_PER_WORKER ))
        LOGDIR="/tmp/rerun_${task}"
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

echo ""
.venv/bin/python3 scripts/robosuite/gen_progress_robosuite.py
