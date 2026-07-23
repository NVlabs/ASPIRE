#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# run_iteration.sh — Single-command iteration for Robosuite fix loop.
#
# Copies code to code_versions/, runs smoke test, runs all seeds in parallel,
# writes result.json, checks stop condition, updates checkpoint.
#
# Exit codes:
#   0 = CONTINUE (need more iterations)
#   1 = STOP (5 consecutive 25/25 achieved)
#   2 = SMOKE TEST FAILED (all 3 smoke seeds crashed)
#   3 = argument/setup error

usage() {
    cat <<EOF
Usage: $0 --code PATH --config PATH --task NAME --iter N --gpu N [--seeds START-END] [--workers N]

Required:
  --code     Path to the Python fix code file
  --config   Path to traced YAML config
  --task     Task short name (cube_lifting, two_arm_lift, etc.)
  --iter     Iteration number (1, 2, 3, ...)
  --gpu      GPU index (3, 4, 5, 6, 7)

Optional:
  --seeds    Seed range (default: 101-125)
  --workers  Parallel workers (default: 5)
EOF
    exit 3
}

# ── Parse arguments ──────────────────────────────────────────────────────────
CODE="" CONFIG="" TASK="" ITER="" GPU=""
SEED_START=101 SEED_END=125 WORKERS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --code)    CODE="$2";    shift 2 ;;
        --config)  CONFIG="$2";  shift 2 ;;
        --task)    TASK="$2";    shift 2 ;;
        --iter)    ITER="$2";    shift 2 ;;
        --gpu)     GPU="$2";     shift 2 ;;
        --seeds)
            SEED_START="${2%-*}"
            SEED_END="${2#*-}"
            shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "ERROR: unknown arg: $1" >&2; usage ;;
    esac
done

for var in CODE CONFIG TASK ITER GPU; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --$(echo $var | tr '[:upper:]' '[:lower:]') is required" >&2
        usage
    fi
done

# ── Ensure we're in the simulation workspace ─────────────────────────────────
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_ROOT="$(cd "$WORKSPACE_ROOT/../.." && pwd)"
if [[ ! -f "$WORKSPACE_ROOT/pyproject.toml" ]]; then
    echo "ERROR: simulation workspace not found at $WORKSPACE_ROOT" >&2
    exit 3
fi
cd "$WORKSPACE_ROOT"
export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$CODE" ]]; then
    echo "ERROR: code file not found: $CODE" >&2
    exit 3
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 3
fi

# ── Derive paths ─────────────────────────────────────────────────────────────
BASELINE_DIR="outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel"
CONFIG_STEM=$(basename "$CONFIG" .yaml)
CODE_VERSIONS_DIR="$BASELINE_DIR/$CONFIG_STEM/code_versions"
OUTDIR="/tmp/fix_test_${TASK}_tl_iter${ITER}"
LOGDIR="/tmp/fix_test_${TASK}_tl_iter${ITER}_logs"

mkdir -p "$CODE_VERSIONS_DIR"

# ── Step 1: Copy code to code_versions/ ──────────────────────────────────────
EXISTING=$(ls "$CODE_VERSIONS_DIR"/iter_${ITER}_*.py 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" ]]; then
    # Overwrite: the subagent may have fixed the code after a smoke failure
    cp "$CODE" "$EXISTING"
    echo "Updated: $EXISTING (overwritten with new code)"
    CODE_VERSIONED="$EXISTING"
else
    TS=$(date +%Y%m%d_%H%M%S)
    CODE_VERSIONED="$CODE_VERSIONS_DIR/iter_${ITER}_${TS}.py"
    cp "$CODE" "$CODE_VERSIONED"
    echo "Saved: $CODE_VERSIONED"
fi
CODE_BASENAME=$(basename "$CODE_VERSIONED")

# ── Step 2: Smoke test (seeds START, START+1, START+2) ──────────────────────
SMOKE_SEEDS=("$SEED_START" "$((SEED_START+1))" "$((SEED_START+2))")
SMOKE_CRASH=0
SMOKE_DIR="/tmp/fix_smoke_${TASK}_tl_iter${ITER}"
SMOKE_LOGDIR="/tmp/fix_smoke_${TASK}_tl_iter${ITER}_logs"
rm -rf "$SMOKE_DIR" "$SMOKE_LOGDIR"
mkdir -p "$SMOKE_DIR" "$SMOKE_LOGDIR"

echo "=== SMOKE TEST: seeds ${SMOKE_SEEDS[*]} ==="
for seed in "${SMOKE_SEEDS[@]}"; do
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
        --args.config "$CONFIG" \
        --args.trial "$seed" \
        --args.replay-code "$CODE_VERSIONED" \
        --args.output-dir "$SMOKE_DIR" \
        --args.flat-output \
        > "$SMOKE_LOGDIR/seed${seed}.log" 2>&1 || true

    HAS_CRASH=$(grep -c "Traceback" "$SMOKE_LOGDIR/seed${seed}.log" 2>/dev/null || true)
    HAS_CRASH=${HAS_CRASH:-0}; HAS_CRASH=$(echo "$HAS_CRASH" | tr -d '[:space:]')
    REWARD=$(grep -oE "reward_[0-9]+\.[0-9]+" "$SMOKE_LOGDIR/seed${seed}.log" 2>/dev/null | tail -1 || true)
    REWARD=${REWARD:-NONE}
    echo "  Seed $seed: $REWARD (crash lines: $HAS_CRASH)"

    if [[ "$HAS_CRASH" -gt 0 && "$REWARD" == "NONE" ]]; then
        SMOKE_CRASH=$((SMOKE_CRASH + 1))
    fi
done

if [[ $SMOKE_CRASH -eq ${#SMOKE_SEEDS[@]} ]]; then
    echo ""
    echo "SMOKE TEST FAILED: all ${#SMOKE_SEEDS[@]} seeds crashed."
    echo "Logs: $SMOKE_LOGDIR/"
    echo "Code: $CODE_VERSIONED"
    echo ""
    echo "Last crash log:"
    tail -30 "$SMOKE_LOGDIR/seed${SMOKE_SEEDS[0]}.log"
    exit 2
fi
echo "=== SMOKE TEST PASSED ($SMOKE_CRASH/${#SMOKE_SEEDS[@]} crashed) ==="

# ── Step 3: Full parallel run ────────────────────────────────────────────────
rm -rf "$OUTDIR" "$LOGDIR"
mkdir -p "$OUTDIR" "$LOGDIR"

_run_seed() {
    local seed=$1 gpu=$2 config=$3 code=$4 outdir=$5 logdir=$6
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    .venv/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
        --args.config "$config" \
        --args.trial "$seed" \
        --args.replay-code "$code" \
        --args.output-dir "$outdir" \
        --args.flat-output \
        > "$logdir/seed${seed}.log" 2>&1 || true
    local reward
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "$logdir/seed${seed}.log" 2>/dev/null | tail -1 || true)
    reward=${reward:-ERROR}
    echo "Seed $seed: $reward"
}
export -f _run_seed

N_SEEDS=$(( SEED_END - SEED_START + 1 ))
echo ""
echo "=== RUNNING $N_SEEDS SEEDS ($SEED_START-$SEED_END) WITH $WORKERS WORKERS ==="

seq "$SEED_START" "$SEED_END" | xargs -P "$WORKERS" -I{} bash -c \
    '_run_seed "$@"' _ {} \
    "$GPU" "$CONFIG" "$CODE_VERSIONED" "$OUTDIR" "$LOGDIR" \
    || true

echo "=== ALL $N_SEEDS SEEDS COMPLETE ==="

# ── Step 4: Write result.json ────────────────────────────────────────────────
RESULT_JSON="$CODE_VERSIONS_DIR/iter_${ITER}_result.json"

.venv/bin/python3 - "$OUTDIR" "$ITER" "$CODE_BASENAME" "$RESULT_JSON" <<'PYEOF'
import json, re, sys
from pathlib import Path

outdir = Path(sys.argv[1])
iter_num = int(sys.argv[2])
code_file = sys.argv[3]
result_path = Path(sys.argv[4])

trials = list(outdir.rglob("trial_*"))
seed_results = {}
n_pass = 0

for t in trials:
    name = t.name
    seed_match = re.search(r"trial_(\d+)", name)
    if not seed_match:
        continue
    seed_str = seed_match.group(1)
    reward_match = re.search(r"reward_([\d.]+)", name)
    reward = float(reward_match.group(1)) if reward_match else 0.0
    seed_results[seed_str] = reward
    if reward >= 1.0:
        n_pass += 1

n_total = len(seed_results)
pass_rate = n_pass / n_total if n_total > 0 else 0.0

result = {
    "iter": iter_num,
    "code_file": code_file,
    "n_pass": n_pass,
    "n_total": n_total,
    "pass_rate": round(pass_rate, 3),
    "seeds": seed_results,
}
result_path.write_text(json.dumps(result, indent=2))
print(f"result.json: {n_pass}/{n_total} = {pass_rate*100:.0f}%")
PYEOF

echo "Wrote: $RESULT_JSON"

# ── Step 5: Check stop condition ─────────────────────────────────────────────
STOP_RESULT=$(.venv/bin/python3 - "$CODE_VERSIONS_DIR" <<'PYEOF'
import json
from pathlib import Path
import sys

cv_dir = Path(sys.argv[1])
results = []
for f in sorted(cv_dir.glob("iter_*_result.json")):
    results.append(json.loads(f.read_text()))
results.sort(key=lambda r: r["iter"])

rates = [r["pass_rate"] for r in results]
best_rate = max(rates) if rates else 0
best_iter = max(results, key=lambda r: (r["pass_rate"], -r["iter"]))["iter"] if results else 0

streak = 0
for r in reversed(rates):
    if r >= 1.0:
        streak += 1
    else:
        break

print(f"HISTORY:", flush=True)
for r in results:
    print(f"  iter {r['iter']}: {r['n_pass']}/{r['n_total']} ({r['pass_rate']*100:.0f}%)")
print(f"Best: iter {best_iter} at {best_rate*100:.0f}%")
print(f"Consecutive 25/25 streak: {streak}/5")

if streak >= 5:
    print("DECISION: STOP")
else:
    print("DECISION: CONTINUE")
PYEOF
)

echo ""
echo "$STOP_RESULT"

# Extract decision
DECISION=$(echo "$STOP_RESULT" | grep "^DECISION:" | awk '{print $2}')
STREAK=$(echo "$STOP_RESULT" | grep "Consecutive" | grep -oE "[0-9]+/5" | head -1)

# Read current pass rate from result.json
CURRENT_RATE=$(.venv/bin/python3 -c "import json; r=json.load(open('$RESULT_JSON')); print(f\"{r['n_pass']}/{r['n_total']} ({r['pass_rate']*100:.0f}%)\")")

# ── Step 6: Update checkpoint ────────────────────────────────────────────────
CHECKPOINT="/tmp/fix_progress_checkpoint_${TASK}_tl.md"

RESULTS_SUMMARY=$(.venv/bin/python3 - "$CODE_VERSIONS_DIR" <<'PYEOF'
import json
from pathlib import Path
import sys
cv_dir = Path(sys.argv[1])
for f in sorted(cv_dir.glob("iter_*_result.json")):
    r = json.loads(f.read_text())
    print(f"- iter {r['iter']}: {r['n_pass']}/{r['n_total']} ({r['pass_rate']*100:.0f}%)")
PYEOF
)

cat > "$CHECKPOINT" <<CPEOF
## Progress: $TASK
Last updated: $(date)

### Current Iteration: $ITER
### Current Step: complete (run_iteration.sh finished)
### Results Per Iteration:
$RESULTS_SUMMARY
### Consecutive 25/25 Streak: $STREAK
### Best So Far: $(echo "$STOP_RESULT" | grep "^Best:")
### Next Step: $(if [[ "$DECISION" == "STOP" ]]; then echo "Step 4 (promote fix_code.py)"; else echo "3a with ITER=$((ITER+1))"; fi)
CPEOF

echo ""
echo "Checkpoint: $CHECKPOINT"

# ── Step 7: Summary line ─────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Iter $ITER: $CURRENT_RATE | streak $STREAK | $DECISION"
echo "======================================================================"

if [[ "$DECISION" == "STOP" ]]; then
    exit 1
else
    exit 0
fi
