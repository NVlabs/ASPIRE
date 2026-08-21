---
name: libero-90-pro-pipeline
description: Coordinator guide for the LIBERO-90 actor pipeline. Claude Code writes task code from scratch (no baseline model), seeds 51–65. Single suite libero_90 with 90 varied tasks (pick-and-place, drawer, stove, microwave, stacking, relative placement).
---

# LIBERO-90 — Actor Pipeline (Coordinator Guide)

> **What:** Actor Agent (Claude Code) writes and debugs code from scratch for all LIBERO-90 tasks.  
> **Suite:** `libero_90` — 90 tasks across Kitchen, Living Room, Study scenes.  
> **No baseline model (CaP-X) needed.** Actor writes code directly using perception APIs + skill library.  
> **Stage 1:** Debug on seeds 51–65. **Stage 2:** Eval on seeds 1–50.  
> **Progress:** `docs/progress/libero_90_progress.md`  
> **Subagent template:** [libero-90-pro-subagent-prompt.md](libero-90-pro-subagent-prompt.md)

---

## Initialization: Verify Perception Servers

```bash
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```
404 = UP, 000 = DOWN. All three must be UP before dispatching.

---

## Directory Layout

```
outputs/
  aspire_actor_90/                   ← Stage 1 artifacts
    libero_90/
      <task_name>/
        task_code.py                ← finalized code (frozen before Stage 2)
        findings.md                 ← structured handoff for skill updates
        BLOCKED                     ← sentinel if all seeds failed

  aspire_eval_90/                    ← Stage 2 trial dirs
    libero_90/<task>/trial_*_sandboxrc_*_reward_*_taskcompleted_*/
```

Debug runs during Stage 1 go to `/tmp/` only.

---

## Task List

90 tasks in a single suite. Get the full list:
```bash
cd $ASPIRE_ROOT
PYTHONPATH=$PYTHON_ROOT .venv/bin/python3 -c "
from capx.envs.configs.instantiate import get_task_names
for i, t in enumerate(get_task_names('libero_90')): print(i, t)
" 2>/dev/null
```

**Task type breakdown** (from task name):
| Type | Examples | Subtasks |
|---|---|---|
| Pick-and-place | `put_the_black_bowl_on_the_plate` | 1 |
| Basket/tray | `pick_up_the_ketchup_and_put_it_in_the_basket` | 1 |
| Drawer | `open_the_top_drawer`, `put_X_in_drawer_and_close_it` | 1–2 |
| Stove/microwave | `turn_on_the_stove`, `turn_on_the_stove_and_put_X_on_it` | 1–2 |
| Stacking | `stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle` | 1 |
| Relative placement | `put_X_to_the_right_of_Y`, caddy compartments | 1 |
| Shelf | `place_it_on/under_the_cabinet_shelf` | 1 |

No _swap/_task variants — task language matches BDDL filename exactly.

---

## The Loop

```
check progress → assign free GPUs (3–7) → dispatch subagents → GO IDLE
                                                                    ↑
on notification: redispatch freed GPU + update skills + GO IDLE ────┘
```

One task = one subagent = one GPU. All 90 tasks are independent — no parallelism constraints.

---

## Coordinator Rules

1. **Dispatch subagents — never write code yourself.**
2. **Go idle after dispatching.** You will be notified when a subagent finishes.
3. **Keep all 5 GPUs (3–7) occupied.**
4. **NEVER re-dispatch a `done` task.** `done` = Stage 2 complete (≥45 seeds in `aspire_eval_90/`).
5. **Never write task code yourself.** Subagents own Stage 1.

---

## Workflow

### 1. Read progress

```bash
cat docs/progress/libero_90_progress.md 2>/dev/null || echo "No progress file yet — all tasks pending"
```

### 2. Check free GPUs

```bash
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```

### 3. Dispatch subagents

```python
Agent(
    description="LIBERO-90 actor: <task_short>",
    subagent_type="general-purpose",
    model="opus",
    prompt=<filled template from libero-90-pro-subagent-prompt.md>,
    run_in_background=True
)
```

Send all dispatches in one message, then stop.

### 4. On each completion: redispatch + update skills

1. Dispatch next pending task on the freed GPU
2. Read `outputs/aspire_actor_90/libero_90/$TASK/findings.md`
3. Promote generalizable patterns to skill library (see table below)
4. **Commit the skill updates** (see §5)
5. Go idle

### 5. Update skills from findings

| Skill | What to add |
|---|---|
| `.claude/libero/skills/localize.md` | New SAM3 prompts, per-object prompt registry entries |
| `.claude/libero/skills/grasp.md` | Gripper width thresholds, z_offset discoveries |
| `.claude/libero/skills/transport.md` | Validated waypoint patterns |
| `.claude/libero/skills/manipulation.md` | Drawer/knob/microwave validated parameters |

After updating any skill files, make a local commit. Use the `/token-calculate` skill to get current token usage first, then:

```bash
cd $ASPIRE_ROOT
git add .claude/libero/skills/
git -c user.email="you@example.com" -c user.name="Your Name" commit -m \
  "Skills: <1-line summary of what was added> [task: $TASK, tokens: ~Xk]"
```

The commit message should note:
- **What changed** — e.g. `"localize: add milk carton arm-occlusion pattern; grasp: butter z_offset"`
- **Which task** — `[task: put_the_black_bowl_on_the_plate]`
- **Token usage** — `[tokens: ~42k]` (from `/token-calculate`)

Only commit if at least one skill file actually changed. Skip if findings had nothing generalizable.

---

## Stage 2: Direct Bash (for coordinator-run cleanup)

If a subagent returns without completing Stage 2, reuse the same nohup+poll pattern.

```bash
cd $ASPIRE_ROOT
SUITE=libero_90; TASK=<task>; GPU=<gpu>; TASKSHORT=<short_name>
# Write and launch (same script as subagent uses):
cat > /tmp/stage2_90_$TASKSHORT.sh << 'SCRIPT'
#!/bin/bash
cd $ASPIRE_ROOT
TASK="FILL_IN_TASK"; GPU=FILL_IN_GPU; TASKSHORT="FILL_IN_TASKSHORT"
FIX_CODE="outputs/aspire_actor_90/libero_90/${TASK}/task_code.py"
LOG="/tmp/val90_${TASKSHORT}_progress.log"
[[ ! -f "$FIX_CODE" ]] && echo "ERROR: $FIX_CODE missing" | tee -a "$LOG" && exit 1
echo "Stage 2 start" | tee -a "$LOG"
for trial in $(seq 1 50); do
    trial_padded=$(printf "%02d" $trial)
    find outputs/aspire_eval_90 -maxdepth 6 -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q "libero_90/$TASK" && \
        echo "Trial $trial: skip" | tee -a "$LOG" && continue
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH=$PYTHON_ROOT \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite libero_90 --args.task "$TASK" --args.trial $trial \
        --args.replay-code "$FIX_CODE" \
        --args.config env_configs/libero/franka_libero_traced.yaml \
        --args.output-dir outputs/aspire_eval_90 > "/tmp/val90_${TASKSHORT}_${trial}.log" 2>&1 || true
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "/tmp/val90_${TASKSHORT}_${trial}.log" | tail -1 | sed 's/reward_//')
    echo "Trial $trial: ${reward:-ERROR}" | tee -a "$LOG"
done
echo "STAGE2_DONE" | tee -a "$LOG"
SCRIPT
sed -i "s/FILL_IN_TASK/$TASK/g; s/FILL_IN_GPU/$GPU/g; s/FILL_IN_TASKSHORT/$TASKSHORT/g" /tmp/stage2_90_$TASKSHORT.sh
grep "FILL_IN_" /tmp/stage2_90_$TASKSHORT.sh && echo "WARNING: unresolved!" || echo "OK"
chmod +x /tmp/stage2_90_$TASKSHORT.sh
nohup bash /tmp/stage2_90_$TASKSHORT.sh > /tmp/stage2_90_${TASKSHORT}.out 2>&1 &
echo "PID=$!"
```

Poll: `grep -c "Trial\|DONE" /tmp/val90_${TASKSHORT}_progress.log`
