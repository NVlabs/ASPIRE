#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# resume_eval_gpu.sh — per-GPU resume helper for Phase 2 of the two-phase
# LIBERO-Long-Pro eval. Runs every not-yet-completed seed (1..SEEDS) for ONE
# task's pre-written code.py on a single GPU, skipping seeds that already have a
# trial directory. Mirrors the per-seed call in scripts/libero/eval_run_seeds.py.
#
# NOTE: reconstructed from scripts/libero/eval_run_seeds.py's per-seed call — the
# original per-GPU helper was never committed. The all-tasks/all-GPUs Phase 2
# runner is scripts/libero/eval_run_seeds.py; use this only for targeted single-task
# resumes.
#
# Usage: bash scripts/libero/resume_eval_gpu.sh <gpu> <suite> <task> <code_path> [seeds]
#   e.g. bash scripts/libero/resume_eval_gpu.sh 3 libero_10_swap KITCHEN_SCENE3_... \
#            "$OUTDIR/libero_10_swap/KITCHEN_SCENE3_.../code.py"
set -euo pipefail

GPU="$1"; SUITE="$2"; TASK="$3"; CODE="$4"; SEEDS="${5:-50}"

# Derive the eval output root (one_shot dir) and worktree from the code path,
# matching eval_run_seeds.py's conventions.
OUTDIR="${CODE%%/$SUITE/*}"                                # .../scaling_eval/<snapshot>/one_shot
WT="${CODE%%/outputs/*}"                                   # worktree (or repo) root
CONFIG="$WT/env_configs/libero/franka_libero_libero10_traced.yaml"
PY="$WT/.venv-libero/bin/python3"
[ -x "$PY" ] || PY="${ASPIRE_ROOT:-$(pwd)}/.venv-libero/bin/python3"
TASK_DIR="$OUTDIR/$SUITE/$TASK"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PYTHON_ROOT="$(cd "$WT/../.." && pwd)"
export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[resume_eval_gpu] GPU=$GPU $SUITE/$TASK seeds=1..$SEEDS  out=$OUTDIR"
for seed in $(seq 1 "$SEEDS"); do
  padded=$(printf '%02d' "$seed")
  # Skip seeds that already produced a trial dir (zero-padded or not).
  if find "$TASK_DIR" -type d \( -name "trial_${padded}_*" -o -name "trial_${seed}_*" \) 2>/dev/null | grep -q .; then
    continue
  fi
  "$PY" "$WT/scripts/libero/replay_trial.py" \
    --args.suite "$SUITE" --args.task "$TASK" --args.trial "$seed" \
    --args.replay-code "$CODE" --args.config "$CONFIG" --args.output-dir "$OUTDIR" || true
done
echo "[resume_eval_gpu] done: $SUITE/$TASK on GPU $GPU"
