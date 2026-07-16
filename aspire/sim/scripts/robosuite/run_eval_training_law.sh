#!/bin/bash
# Eval seeds 1–100 for every intermediate code version in code_versions/ (training-law).
#
# Unlike run_eval_fix_code.sh (final fix_code.py only), this replays each
# iter_{N}_{timestamp}.py on that task's traced config.
#
# Usage:
#   scripts/robosuite/run_eval_training_law.sh [--task TASK] [--iter N] [--skip-existing]
#
# Writes trial outputs:
#   outputs/training_law_eval/<config_stem>/iter_<N>/
# Writes eval summaries (next to debug result.json):
#   .../<config_stem>/code_versions/iter_<N>_eval_result.json

set -euo pipefail

BASELINE="outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel"
EVAL_ROOT="outputs/training_law_eval"
GPUS=(3 4 5 6 7)
WORKERS=5
SEEDS_PER_WORKER=20  # 100 / 5

FILTER_TASK=""
FILTER_ITER=""
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) FILTER_TASK="$2"; shift 2 ;;
        --iter) FILTER_ITER="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_ROOT="$(cd "$WORKSPACE_ROOT/../.." && pwd)"
cd "$WORKSPACE_ROOT"
export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

declare -A TASKS
TASKS[cube_lifting]="env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml|cube_lifting_multimodel_aspire_traced"
TASKS[cube_restack]="env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml|cube_restack_multimodel_aspire_traced"
TASKS[cube_stack]="env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml|cube_stack_multimodel_aspire_traced"
TASKS[nut_assembly]="env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml|nut_assembly_multimodel_aspire_traced"
TASKS[spill_wipe]="env_configs/robosuite/spill_wipe_multimodel_aspire_traced.yaml|spill_wipe_multimodel_aspire_traced"
TASKS[two_arm_lift]="env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml|two_arm_lift_multimodel_aspire_traced"
TASKS[two_arm_handover]="env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml|two_arm_handover_multimodel_aspire_traced"

TASK_ORDER=(cube_lifting cube_restack cube_stack nut_assembly spill_wipe two_arm_lift two_arm_handover)

gpu_idx=0

_run_eval_version() {
    local task="$1" config="$2" stem="$3" iter="$4" code_path="$5" code_base="$6"
    local cv_dir="$BASELINE/$stem/code_versions"
    local outdir="$EVAL_ROOT/$stem/iter_${iter}"
    local result_json="$cv_dir/iter_${iter}_eval_result.json"
    local logdir="/tmp/training_law_eval_${task}_iter${iter}"

    if [[ $SKIP_EXISTING -eq 1 && -f "$result_json" ]]; then
        echo "SKIP $task iter $iter (eval_result exists)"
        return 0
    fi

    echo ""
    echo "=== $task iter $iter: $code_base ==="
    echo "  code:   $code_path"
    echo "  config: $config"
    echo "  out:    $outdir"

    rm -rf "$outdir" "$logdir"
    mkdir -p "$outdir" "$logdir"

    local pids=()
    local w
    for w in $(seq 0 $((WORKERS - 1))); do
        local gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
        local start=$(( w * SEEDS_PER_WORKER + 1 ))
        local end=$(( (w + 1) * SEEDS_PER_WORKER ))
        (
            for trial in $(seq "$start" "$end"); do
                MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
                .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
                    --args.config "$config" \
                    --args.trial "$trial" \
                    --args.replay-code "$code_path" \
                    --args.output-dir "$outdir" \
                    --args.flat-output \
                    > "$logdir/${trial}.log" 2>&1 || true
                local reward
                reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "$logdir/${trial}.log" 2>/dev/null | tail -1 | sed 's/reward_//' || true)
                echo "[$task iter$iter GPU$gpu] trial $trial: ${reward:-ERROR}"
            done
        ) &
        pids+=($!)
        gpu_idx=$((gpu_idx + 1))
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    .venv/bin/python3 - "$outdir" "$iter" "$code_base" "$result_json" <<'PYEOF'
import json, re, sys
from pathlib import Path

outdir = Path(sys.argv[1])
iter_num = int(sys.argv[2])
code_file = sys.argv[3]
result_path = Path(sys.argv[4])

seed_results = {}
n_pass = 0
for t in outdir.rglob("trial_*"):
    m = re.search(r"trial_(\d+)", t.name)
    if not m:
        continue
    seed_str = m.group(1)
    rm = re.search(r"reward_([\d.]+)", t.name)
    reward = float(rm.group(1)) if rm else 0.0
    seed_results[seed_str] = reward
    if reward >= 1.0:
        n_pass += 1

n_total = len(seed_results)
pass_rate = n_pass / n_total if n_total else 0.0
result = {
    "iter": iter_num,
    "code_file": code_file,
    "seed_range": "1-100",
    "n_pass": n_pass,
    "n_total": n_total,
    "pass_rate": round(pass_rate, 3),
    "seeds": seed_results,
}
result_path.write_text(json.dumps(result, indent=2))
print(f"  eval_result: {n_pass}/{n_total} = {pass_rate*100:.0f}% -> {result_path}")
PYEOF
}

for task in "${TASK_ORDER[@]}"; do
    if [[ -n "$FILTER_TASK" && "$task" != "$FILTER_TASK" ]]; then
        continue
    fi

    IFS='|' read -r CONFIG STEM <<< "${TASKS[$task]}"
    CV_DIR="$BASELINE/$STEM/code_versions"

    if [[ ! -d "$CV_DIR" ]]; then
        echo "SKIP $task: no code_versions at $CV_DIR"
        continue
    fi

    shopt -s nullglob
    code_files=("$CV_DIR"/iter_*_*.py)
    shopt -u nullglob

    if [[ ${#code_files[@]} -eq 0 ]]; then
        echo "SKIP $task: no iter_*_*.py in $CV_DIR"
        continue
    fi

    for code_path in "${code_files[@]}"; do
        code_base=$(basename "$code_path")
        if [[ ! "$code_base" =~ ^iter_([0-9]+)_([0-9]{8}_[0-9]{6})\.py$ ]]; then
            continue
        fi
        iter="${BASH_REMATCH[1]}"

        if [[ -n "$FILTER_ITER" && "$iter" != "$FILTER_ITER" ]]; then
            continue
        fi

        _run_eval_version "$task" "$CONFIG" "$STEM" "$iter" "$code_path" "$code_base"
    done
done

echo ""
echo "========== TRAINING-LAW EVAL DONE =========="
echo "Summaries: $BASELINE/<config_stem>/code_versions/iter_<N>_eval_result.json"
echo "Trials:    $EVAL_ROOT/<config_stem>/iter_<N>/"
