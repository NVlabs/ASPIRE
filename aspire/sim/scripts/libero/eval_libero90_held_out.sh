#!/usr/bin/env bash
# Evaluate all libero_90 task_code.py programs on seeds 1-50, no video.
# Skips tasks without task_code.py (BLOCKED). Skips already-completed trials.
#
# Usage:
#   cd $ASPIRE_ROOT
#   bash scripts/libero/eval_libero90_held_out.sh \
#       --out-dir outputs/scaling_eval/libero_90_eval \
#       [--gpus "3 4 5 6 7"] \
#       [--seeds 50] \
#       [--workers-per-gpu 2]

set -euo pipefail

ASPIRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_ROOT="$(cd "$ASPIRE_ROOT/../.." && pwd)"
export ASPIRE_ROOT
BUILD_DIR="$ASPIRE_ROOT/outputs/scaling_build/libero_90"
PYTHON="$ASPIRE_ROOT/.venv-libero/bin/python3"
REPLAY="$ASPIRE_ROOT/scripts/libero/replay_trial.py"
CONFIG="$ASPIRE_ROOT/env_configs/libero/franka_libero_traced.yaml"
SUITE="libero_90"

OUT_DIR="$ASPIRE_ROOT/outputs/scaling_eval/libero_90_$(date +%Y%m%d_%H%M%S)"
GPUS="3 4 5 6 7"
SEEDS=50
WORKERS_PER_GPU=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --out-dir)          OUT_DIR="$2";          shift 2 ;;
        --gpus)             GPUS="$2";             shift 2 ;;
        --seeds)            SEEDS="$2";            shift 2 ;;
        --workers-per-gpu)  WORKERS_PER_GPU="$2";  shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"
GPU_ARRAY=($GPUS)
N_GPUS=${#GPU_ARRAY[@]}
TOTAL_WORKERS=$((N_GPUS * WORKERS_PER_GPU))

echo "=== libero_90 held-out eval ==="
echo "BUILD_DIR: $BUILD_DIR"
echo "OUT_DIR:   $OUT_DIR"
echo "SUITE:     $SUITE"
echo "SEEDS:     1-$SEEDS"
echo "GPUS:      $GPUS  ($N_GPUS GPUs x $WORKERS_PER_GPU = $TOTAL_WORKERS workers)"
echo ""

# Build work list: "task|seed|gpu" — GPU pre-assigned round-robin by task index
WORK_LIST=$(mktemp)
task_idx=0
for task_dir in "$BUILD_DIR"/*/; do
    task=$(basename "$task_dir")
    [[ ! -f "$task_dir/task_code.py" ]] && continue
    gpu="${GPU_ARRAY[$((task_idx % N_GPUS))]}"
    for seed in $(seq 1 "$SEEDS"); do
        printf '%s|%d|%d\n' "$task" "$seed" "$gpu"
    done
    task_idx=$((task_idx + 1))
done > "$WORK_LIST"

n_items=$(wc -l < "$WORK_LIST")
echo "Work items: $n_items  ($(( n_items / SEEDS )) tasks x $SEEDS seeds)"
echo ""

# Run one task+seed
_run_one() {
    local item="$1"
    local BUILD_DIR="$2" OUT_DIR="$3" SUITE="$4" CONFIG="$5" PYTHON="$6" REPLAY="$7"

    local task seed gpu
    task=$(  cut -d'|' -f1 <<< "$item")
    seed=$(  cut -d'|' -f2 <<< "$item")
    gpu=$(   cut -d'|' -f3 <<< "$item")

    local code="$BUILD_DIR/$task/task_code.py"
    local trial_padded; trial_padded=$(printf "%02d" "$seed")
    local task_out="$OUT_DIR/$task"

    # Skip if trial dir already exists anywhere under task_out
    if find "$task_out" -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q .; then
        return 0
    fi

    mkdir -p "$task_out"
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$REPLAY" \
        --args.suite   "$SUITE" \
        --args.task    "$task" \
        --args.trial   "$seed" \
        --args.replay-code "$code" \
        --args.config  "$CONFIG" \
        --args.no-record-video \
        --args.output-dir "$task_out" \
        > "/tmp/libero90_eval_${task:0:40}_s${seed}.log" 2>&1 || true
}
export -f _run_one

echo "Starting parallel eval (${TOTAL_WORKERS} workers)..."
xargs -P "$TOTAL_WORKERS" -I{} bash -c \
    '_run_one "$@"' _ {} \
    "$BUILD_DIR" "$OUT_DIR" "$SUITE" "$CONFIG" "$PYTHON" "$REPLAY" \
    < "$WORK_LIST"

rm -f "$WORK_LIST"

# Summary
echo ""
echo "=== Results ==="
total=$(find "$OUT_DIR" -type d -name "trial_*" | wc -l)
pass=$( find "$OUT_DIR" -type d -name "*taskcompleted_1*" | wc -l)
echo "Trials completed: $total"
echo "Pass:             $pass"
[[ $total -gt 0 ]] && "$PYTHON" -c "print(f'Pass rate:        {$pass/$total*100:.1f}%')"
echo ""
echo "Results in: $OUT_DIR"
