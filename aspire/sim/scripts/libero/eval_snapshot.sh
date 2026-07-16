#!/usr/bin/env bash
# Full eval pipeline for a snapshot: setup → codegen (subagents) → parallel seeds → summary.
#
# Usage:
#   bash scripts/libero/eval_snapshot.sh --snapshot snapshot-N70 [--seeds 50] [--seeds-per-gpu 3]
#
# Phase 1 (codegen) is manual — the coordinator dispatches subagents.
# Phase 2 (execution) uses eval_run_seeds.py for parallel seed execution.
# This script handles Phase 2 + summary.
#
# Prerequisites:
#   - Worktree set up: scripts/libero/eval_setup_worktree.sh --snapshot $SNAPSHOT
#   - Code files exist: outputs/scaling_eval/$SNAPSHOT/one_shot/$SUITE/$TASK/code.py
#   - Perception servers running on 8114/8115/8116

set -euo pipefail

SNAPSHOT=""
SEEDS=50
SEEDS_PER_GPU=3
GPUS="3 4 5 6 7"

while [[ $# -gt 0 ]]; do
    case $1 in
        --snapshot) SNAPSHOT="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --seeds-per-gpu) SEEDS_PER_GPU="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$SNAPSHOT" ]] && { echo "ERROR: --snapshot required" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$REPO/outputs/worktrees/$SNAPSHOT/aspire/sim"
PYTHON="$REPO/.venv-libero/bin/python3"
OUTDIR="$WORKTREE/outputs/scaling_eval/$SNAPSHOT/one_shot"

# Preflight
echo "=== Eval $SNAPSHOT ==="
echo "Workspace: $WORKTREE"

if [[ ! -d "$WORKTREE" ]]; then
    echo "ERROR: worktree not found. Run: scripts/libero/eval_setup_worktree.sh --snapshot $SNAPSHOT" >&2
    exit 1
fi

# Check code files exist
code_count=$(find "$OUTDIR" -name "code.py" -path "*/one_shot/*/*/code.py" 2>/dev/null | wc -l)
echo "Code files found: $code_count/20"
if [[ "$code_count" -eq 0 ]]; then
    echo "ERROR: no code.py files found. Run Phase 1 (codegen subagents) first." >&2
    exit 1
fi

# Check perception servers
for p in 8114 8115 8116; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:$p/health 2>/dev/null || echo "000")
    if [[ "$code" == "000" ]]; then
        echo "ERROR: perception server on port $p is DOWN" >&2
        exit 1
    fi
done
echo "Perception servers: OK"

# Phase 2: parallel seed execution
echo ""
echo "=== Phase 2: Parallel seed execution ==="
"$PYTHON" "$REPO/scripts/libero/eval_run_seeds.py" \
    --worktree "$WORKTREE" \
    --snapshot "$SNAPSHOT" \
    --gpus $GPUS \
    --seeds-per-gpu "$SEEDS_PER_GPU" \
    --seeds "$SEEDS" \
    --python "$PYTHON"

# Phase 3: write summary.json
echo ""
echo "=== Phase 3: Summary ==="
SNAPSHOT="$SNAPSHOT" OUTDIR="$OUTDIR" "$PYTHON" - << 'PYEOF'
import json, re, os, sys
from pathlib import Path

snapshot = os.environ.get("SNAPSHOT", sys.argv[1] if len(sys.argv) > 1 else "")
outdir = Path(os.environ.get("OUTDIR", ""))
suites = ["libero_10_swap", "libero_10_task"]

results = {}
total_success, total_trials = 0, 0

for suite in suites:
    results[suite] = {}
    suite_dir = outdir / suite
    if not suite_dir.exists():
        continue
    for task_dir in sorted(suite_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        trial_nums, success_nums = set(), set()
        for t in task_dir.rglob("trial_*"):
            if t.is_dir():
                m = re.search(r"trial_(\d+)", t.name)
                if m:
                    trial_nums.add(int(m.group(1)))
                    if "taskcompleted_1" in t.name:
                        success_nums.add(int(m.group(1)))
        results[suite][task] = {"success": len(success_nums), "total": len(trial_nums)}
        total_success += len(success_nums)
        total_trials += len(trial_nums)

summary = {
    "snapshot": snapshot,
    "mode": "one_shot",
    "seeds": "1-50",
    "suites": suites,
    "results": results,
    "aggregate": {
        "success": total_success,
        "total": total_trials,
        "rate": round(total_success / total_trials, 4) if total_trials > 0 else 0,
    },
}

out = outdir / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"Written: {out}")
print(f"Overall: {total_success}/{total_trials} = {summary['aggregate']['rate']*100:.1f}%")
PYEOF

echo ""
echo "=== Eval $SNAPSHOT COMPLETE ==="
