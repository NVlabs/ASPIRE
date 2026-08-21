#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Evaluate 25% and 50% output-token-budget checkpoint code versions for snapshot-N50.
# Outputs go to: stage1/checkpoint_eval/{25pct,50pct}/ — NEVER touches stage2/.
# Seeds 1-50, no video, parallel across GPUs 3-7.
#
# Checkpoints (output-token budget):
#   swap/SCENE2: 25%=iter_4, 50%=iter_6
#   swap/SCENE5: 25%=iter_4, 50%=iter_4  (same file — both checkpoints identical)
#   swap/SCENE6: 25%=iter_3, 50%=iter_5
#   task/SCENE2: 25%=iter_3, 50%=iter_5
#   task/SCENE6: 25%=iter_2, 50%=iter_4
#
# Usage:
#   bash scripts/libero/eval_checkpoint_n50.sh [--gpus "3 4 5 6 7"] [--workers-per-gpu 2]

set -euo pipefail

WORKTREE="$ASPIRE_ROOT/outputs/worktrees/snapshot-N50/aspire/sim"
PYTHON="$WORKTREE/.venv-libero/bin/python3"
PYTHON_ROOT="$(cd "$WORKTREE/../.." && pwd)"
REPLAY="$WORKTREE/scripts/libero/replay_trial.py"
CONFIG="$WORKTREE/env_configs/libero/franka_libero_libero10_traced.yaml"
BASE="$WORKTREE/outputs/scaling_eval/snapshot-N50/debug_eval"
SEEDS=50
GPUS="3 4 5 6 7"
WORKERS_PER_GPU=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)            GPUS="$2";            shift 2 ;;
        --workers-per-gpu) WORKERS_PER_GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

GPU_ARRAY=($GPUS)
N_GPUS=${#GPU_ARRAY[@]}
TOTAL_WORKERS=$((N_GPUS * WORKERS_PER_GPU))

# Checkpoint map: "suite|task|pct|iter_prefix"
CHECKPOINTS=(
    "libero_10_swap|LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket|25pct|iter_4_"
    "libero_10_swap|LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate|25pct|iter_3_"
    "libero_10_swap|LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate|25pct|iter_3_"
    "libero_10_task|LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket|25pct|iter_3_"
    "libero_10_task|LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate|25pct|iter_2_"
    "libero_10_swap|LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket|50pct|iter_6_"
    "libero_10_swap|LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate|50pct|iter_4_"
    "libero_10_swap|LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate|50pct|iter_5_"
    "libero_10_task|LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket|50pct|iter_5_"
    "libero_10_task|LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate|50pct|iter_4_"
)

echo "=== snapshot-N50 checkpoint eval ==="
echo "WORKTREE:       $WORKTREE"
echo "CONFIG:         $CONFIG"
echo "SEEDS:          1-$SEEDS"
echo "GPUS:           $GPUS  ($N_GPUS GPUs x $WORKERS_PER_GPU = $TOTAL_WORKERS workers)"
echo "CHECKPOINTS:    ${#CHECKPOINTS[@]} (5 tasks x 2 budgets)"
echo ""

# Build arrays for interleaved work list
CKPT_CODES=()
CKPT_SUITES=()
CKPT_TASKS=()
CKPT_PCTS=()
CKPT_GPUS=()

ckpt_idx=0
for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r suite task pct iter_prefix <<< "$entry"
    cv_dir="$BASE/$suite/$task/stage1/code_versions"
    code=$(ls "$cv_dir"/${iter_prefix}*.py 2>/dev/null | head -1)
    if [[ -z "$code" ]]; then
        echo "WARNING: no code found for $suite/$task $pct ($iter_prefix)" >&2
        continue
    fi
    gpu="${GPU_ARRAY[$((ckpt_idx % N_GPUS))]}"
    CKPT_CODES+=("$code")
    CKPT_SUITES+=("$suite")
    CKPT_TASKS+=("$task")
    CKPT_PCTS+=("$pct")
    CKPT_GPUS+=("$gpu")
    echo "  ckpt $pct ${suite##libero_10_}/${task:14:6}: $(basename "$code") → GPU $gpu"
    ckpt_idx=$((ckpt_idx + 1))
done

# Interleave: for each seed, emit all checkpoints
WORK_LIST=$(mktemp)
for seed in $(seq 1 $SEEDS); do
    for i in "${!CKPT_CODES[@]}"; do
        printf '%s|%s|%s|%s|%d|%d\n' \
            "${CKPT_SUITES[$i]}" "${CKPT_TASKS[$i]}" "${CKPT_PCTS[$i]}" \
            "${CKPT_CODES[$i]}" "$seed" "${CKPT_GPUS[$i]}"
    done
done > "$WORK_LIST"

n_items=$(wc -l < "$WORK_LIST")
echo ""
echo "Work items: $n_items"
echo ""

_run_one_ckpt() {
    local item="$1"
    local BASE="$2" PYTHON="$3" REPLAY="$4" CONFIG="$5"

    local suite task pct code seed gpu
    suite=$(cut -d'|' -f1 <<< "$item")
    task=$( cut -d'|' -f2 <<< "$item")
    pct=$(  cut -d'|' -f3 <<< "$item")
    code=$( cut -d'|' -f4 <<< "$item")
    seed=$( cut -d'|' -f5 <<< "$item")
    gpu=$(  cut -d'|' -f6 <<< "$item")

    local out_dir="$BASE/$suite/$task/stage1/checkpoint_eval/$pct"
    local trial_padded; trial_padded=$(printf "%02d" "$seed")

    if find "$out_dir" -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q .; then
        return 0
    fi

    mkdir -p "$out_dir"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$REPLAY" \
        --args.suite   "$suite" \
        --args.task    "$task" \
        --args.trial   "$seed" \
        --args.replay-code "$code" \
        --args.config  "$CONFIG" \
        --args.no-record-video \
        --args.output-dir "$out_dir" \
        > "/tmp/n50_ckpt_${suite##libero_10_}_${task:14:6}_${pct}_s${seed}.log" 2>&1 || true
}
export -f _run_one_ckpt

echo "Starting parallel eval ($TOTAL_WORKERS workers)..."
xargs -P "$TOTAL_WORKERS" -I{} bash -c \
    '_run_one_ckpt "$@"' _ {} \
    "$BASE" "$PYTHON" "$REPLAY" "$CONFIG" \
    < "$WORK_LIST"

rm -f "$WORK_LIST"

echo ""
echo "=== Results ==="
for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r suite task pct iter_prefix <<< "$entry"
    out_dir="$BASE/$suite/$task/stage1/checkpoint_eval/$pct"
    total=$(find "$out_dir" -type d -name "trial_*" 2>/dev/null | wc -l)
    pass=$( find "$out_dir" -type d -name "*taskcompleted_1*" 2>/dev/null | wc -l)
    rate=""
    [[ $total -gt 0 ]] && rate=$("$PYTHON" -c "print(f'{$pass/$total*100:.1f}%')")
    echo "  $pct  ${suite##libero_10_}/${task:14:6}: $pass/$total $rate"

    if [[ $total -gt 0 ]]; then
        code=$(ls "$BASE/$suite/$task/stage1/code_versions"/${iter_prefix}*.py 2>/dev/null | head -1)
        iter_name=$(basename "${code%.py}")
        "$PYTHON" -c "
import json
result = {
    'snapshot': 'snapshot-N50',
    'suite': '$suite',
    'task': '$task',
    'checkpoint': '$pct',
    'code_version': '$iter_name',
    'seeds': '1-$SEEDS',
    'n_pass': $pass,
    'n_total': $total,
    'pass_rate': round($pass/$total, 4) if $total > 0 else 0,
}
out = '$out_dir/checkpoint_result.json'
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f'  wrote {out}')
"
    fi
done

echo ""
echo "Done. Results in: $BASE/<suite>/<task>/stage1/checkpoint_eval/{25pct,50pct}/"
