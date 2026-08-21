---
name: libero-90-heldout-eval
description: Coordinator guide for running seeds 1-50 on stored LIBERO-90 task_code.py files after the zero-shot transfer build phase. Uses nohup+poll pattern, 5 GPUs in parallel. Output goes to outputs/scaling_eval/libero_90/.
---

# LIBERO-90 Eval — Post-Build Stage 2

> **When:** After all 18 build chunks complete. Each task that has `outputs/scaling_build/libero_90/<task>/task_code.py` gets evaluated on seeds 1–50.
> **Output:** `outputs/scaling_eval/libero_90/<task>/trial_*/`
> **Config:** `env_configs/libero/franka_libero_traced.yaml`

---

## Step 1 — Check which tasks have task_code.py

```bash
cd "$ASPIRE_ROOT"
PYTHONPATH="$PYTHON_ROOT" .venv/bin/python3 scripts/libero/gen_progress_scaling.py
# Tasks with status "done" have task_code.py and are ready to eval
grep -c "| done" docs/progress/scaling_law_progress.md
```

---

## Step 2 — Check which tasks still need eval

```bash
cd "$ASPIRE_ROOT"
python3 - << 'EOF'
from pathlib import Path
import re

build_dir = Path("outputs/scaling_build/libero_90")
eval_dir  = Path("outputs/scaling_eval/libero_90")
DONE_THRESH = 45

pending = []
for task_dir in sorted(build_dir.iterdir()):
    code = task_dir / "task_code.py"
    if not code.exists():
        continue
    trial_dirs = list((eval_dir / task_dir.name).glob("trial_*")) if (eval_dir / task_dir.name).exists() else []
    seeds_done = len(set(re.search(r'trial_(\d+)', d.name).group(1) for d in trial_dirs if re.search(r'trial_(\d+)', d.name)))
    if seeds_done < DONE_THRESH:
        pending.append((task_dir.name, seeds_done))

print(f"{len(pending)} tasks need eval:")
for t, n in pending:
    print(f"  {n:2d}/50  {t}")
EOF
```

---

## Step 3 — Launch eval (nohup+poll, one task per GPU)

Write and launch a script per task. Batch across 5 GPUs:

```bash
cd "$ASPIRE_ROOT"
TASK="<full_task_name>"
GPU=3
TASKSHORT="<short_name>"

cat > /tmp/eval90_$TASKSHORT.sh << 'SCRIPT'
#!/bin/bash
cd FILL_ASPIRE_ROOT
TASK="FILL_TASK"; GPU=FILL_GPU; TASKSHORT="FILL_TASKSHORT"
FIX_CODE="outputs/scaling_build/libero_90/${TASK}/task_code.py"
OUTDIR="outputs/scaling_eval/libero_90"
LOG="/tmp/eval90_${TASKSHORT}_progress.log"

[[ ! -f "$FIX_CODE" ]] && echo "ERROR: $FIX_CODE missing" | tee -a "$LOG" && exit 1
echo "Eval start: $TASK  GPU=$GPU" | tee -a "$LOG"

for trial in $(seq 1 50); do
    trial_padded=$(printf "%02d" $trial)
    if find "$OUTDIR/$TASK" -maxdepth 5 -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q .; then
        echo "Trial $trial: skip" | tee -a "$LOG"; continue
    fi
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH=FILL_ASPIRE_ROOT \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite libero_90 --args.task "${TASK}" --args.trial ${trial} \
        --args.replay-code "${FIX_CODE}" \
        --args.config env_configs/libero/franka_libero_traced.yaml \
        --args.output-dir "$OUTDIR" > "/tmp/eval90_${TASKSHORT}_${trial}.log" 2>&1 || true
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "/tmp/eval90_${TASKSHORT}_${trial}.log" | tail -1 | sed 's/reward_//')
    echo "Trial $trial: ${reward:-ERROR}" | tee -a "$LOG"
done
echo "EVAL_DONE" | tee -a "$LOG"
SCRIPT

sed -i "s|FILL_ASPIRE_ROOT|$ASPIRE_ROOT|g; s/FILL_TASK/$TASK/g; s/FILL_GPU/$GPU/g; s/FILL_TASKSHORT/$TASKSHORT/g" /tmp/eval90_$TASKSHORT.sh
grep "FILL_" /tmp/eval90_$TASKSHORT.sh && echo "WARNING: unresolved!" || echo "OK"
chmod +x /tmp/eval90_$TASKSHORT.sh
nohup bash /tmp/eval90_$TASKSHORT.sh > /tmp/eval90_${TASKSHORT}.out 2>&1 &
echo "PID=$!  log=/tmp/eval90_${TASKSHORT}_progress.log"
```

Launch one task per free GPU. Poll progress:

```bash
grep -c "Trial\|DONE" /tmp/eval90_${TASKSHORT}_progress.log
```

---

## Step 4 — Summarize results

```bash
cd "$ASPIRE_ROOT"
python3 - << 'EOF'
from pathlib import Path
import re

eval_dir = Path("outputs/scaling_eval/libero_90")
total_trials = total_success = 0
rows = []

for task_dir in sorted(eval_dir.iterdir()):
    trial_dirs = list(task_dir.glob("trial_*"))
    seeds = set(m.group(1) for d in trial_dirs if (m := re.search(r'trial_(\d+)', d.name)))
    successes = sum(1 for d in trial_dirs if "taskcompleted_1" in d.name
                    and re.search(r'trial_(\d+)', d.name))
    n = len(seeds)
    total_trials += n; total_success += successes
    rows.append((task_dir.name[:55], successes, n))

rows.sort(key=lambda r: r[1]/max(r[2],1))
for name, s, n in rows:
    print(f"{s:2d}/{n:2d} ({100*s//max(n,1):3d}%)  {name}")
print(f"\nOverall: {total_success}/{total_trials} = {100*total_success//max(total_trials,1)}%")
EOF
```
