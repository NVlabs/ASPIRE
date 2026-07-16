---
name: libero-long-pro-pipeline
description: Coordinator guide for the LIBERO Long Pro actor pipeline. Claude Code writes task code from scratch (no baseline), using skills from LIBERO Pro. Manages GPU dispatch for libero_10_swap and libero_10_task suites.
---

# LIBERO Long Pro — Actor Pipeline (Coordinator Guide)

> **What:** Actor Agent (Claude Code) writes and debugs code for LIBERO Long Pro tasks from scratch, drawing on the skill library from LIBERO Pro (object/goal/spatial).  
> **Suites:** `libero_10_swap` (position perturbation) and `libero_10_task` (instruction perturbation) — 10 tasks each, 20 total.  
> **Key difference from fix-loop:** No baseline code exists. The actor writes code directly using the accumulated skill library.  
> **Progress:** `docs/progress/libero_long_pro_progress.md` — run `gen_progress_long.py` to regenerate.  
> **Subagent template:** [libero-long-pro-subagent-prompt.md](libero-long-pro-subagent-prompt.md)

---

## Initialization: Verify Perception Servers

Before dispatching any subagents, confirm all three servers are up (404 = UP, 000 = DOWN):

```bash
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```

| Port | Server | GPU |
|---|---|---|
| 8114 | SAM3 | 0 |
| 8115 | GraspNet | 1 |
| 8116 | PyRoKi / Molmo | 2 |

---

## Directory Layout

```
outputs/
  aspire_libero_long_actor/        ← Stage 1 artifacts (task_code.py, findings.md)
    libero_10_swap/
      <task_name>/
        task_code.py              ← finalized code (frozen before Stage 2)
        findings.md               ← structured handoff for skill updates
        BLOCKED                   ← sentinel if all seeds failed (no task_code.py written)
    libero_10_task/
      <task_name>/
        task_code.py
        findings.md

  aspire_libero_long_eval/         ← Stage 2 trial dirs (replay_trial.py appends suite/task)
    libero_10_swap/<task>/trial_*_sandboxrc_*_reward_*_taskcompleted_*/
    libero_10_task/<task>/trial_*/
```

Debug runs during Stage 1 go to `/tmp/` only — never into the eval dir.

---

## LIBERO Long Pro Task List

Both suites share the same 10 task names (bddl files differ between swap/task):

| # | Task name |
|---|---|
| 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` |
| 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` |
| 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` |
| 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` |
| 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` |
| 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` |
| 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` |
| 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` |
| 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` |
| 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` |

**Note:** `_task` suite language goals are perturbed — always use `env.handle.task_language` for the actual instruction.

---

## The Loop

```
read progress → assign free GPUs (3–7) → dispatch subagents → GO IDLE
                                                                    ↑
on notification: redispatch freed GPU + update skills + GO IDLE ────┘
```

One task = one subagent = one GPU. Run swap and task variants of different scene tasks in parallel (safe — different scenes, no resource contention).

---

## Coordinator Rules

1. **Dispatch subagents — never write code yourself.** Your job is dispatch → idle → collect findings → update skills → dispatch.
2. **Go idle after dispatching.** You will be notified when a subagent finishes. Do not poll.
3. **Keep all 5 GPUs (3–7) occupied.** Any free GPU should immediately get a new task.
4. **NEVER re-dispatch a `done` task.** `done` = Stage 2 complete (≥45 seeds on disk in `aspire_libero_long_eval/`).
5. **Never write task code yourself.** Subagents own Stage 1. You read only `findings.md` (not traces, code.py, or keyframes).

---

## Workflow

### 1. Read progress

```bash
cat docs/progress/libero_long_pro_progress.md
```

Find `pending` and `stage1-done` tasks.

### 2. Check which GPUs are free

```bash
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```

### 3. Dispatch one subagent per free GPU

```python
Agent(
    description="LIBERO Long Pro actor: <suite>/<task_short>",
    subagent_type="general-purpose",
    model="opus",
    prompt=<filled-in template from libero-long-pro-subagent-prompt.md>,
    run_in_background=True
)
```

Send all dispatches in one message, then stop.

### 4. On each completion: redispatch + update skills

When a subagent notification arrives:

1. `cat docs/progress/libero_long_pro_progress.md` — verify updated
2. Dispatch next `pending` or `stage1-done` task on the freed GPU
3. Update skills from `findings.md` (see step 5)
4. **Commit skill updates** (see step 6)
5. Go idle

### 5. Update skills from findings

Read `outputs/aspire_libero_long_actor/$SUITE/$TASK/findings.md` and promote **generalizable patterns** to the skill library.

| Skill | What to add |
|---|---|
| `.claude/libero/skills/localize.md` | New SAM3 prompts, new objects (moka pot, microwave, caddy), disambiguation filters |
| `.claude/libero/skills/grasp.md` | Gripper width thresholds for new objects, z_offset findings |
| `.claude/libero/skills/transport.md` | Multi-step waypoints, safe transit between subtasks |
| `.claude/libero/skills/manipulation.md` | Microwave open/close, drawer + close sequence, stove knob turn |

Long-horizon patterns (sequencing, home-position between subtasks, partial progress detection) are especially valuable — add to `.claude/libero/skills/manipulation.md`.

### 6. Commit skill updates

After updating any skill files, use the `/token-calculate` skill to get current token usage, then:

```bash
cd $ASPIRE_ROOT
git add .claude/libero/skills/
git -c user.email="you@example.com" -c user.name="Your Name" commit -m \
  "Skills: <1-line summary of what was added> [task: $SUITE/$TASK, tokens: ~Xk]"
```

The commit message should note:
- **What changed** — e.g. `"localize: add moka pot prompts; manipulation: drawer+close sequence"`
- **Which task** — `[task: libero_10_swap/KITCHEN_SCENE3_...]`
- **Token usage** — `[tokens: ~42k]` (from `/token-calculate`)

Only commit if at least one skill file actually changed. Skip if findings had nothing generalizable.

---

## Stage 2: Direct Bash Scripts (Preferred for Pure Stage 2 Runs)

For tasks where `task_code.py` already exists, use nohup+poll rather than `run_in_background=True` (survives agent timeout/crash).

```bash
cd $ASPIRE_ROOT
SUITE="libero_10_swap"
TASK="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
GPU=3
TASKSHORT="lr2_soup_tomato"

cat > /tmp/stage2_long_$TASKSHORT.sh << 'SCRIPT'
#!/bin/bash
# NO set -e — individual trial failures must not kill the loop
cd $ASPIRE_ROOT
SUITE="FILL_IN_SUITE"; TASK="FILL_IN_TASK"; GPU=FILL_IN_GPU; TASKSHORT="FILL_IN_TASKSHORT"
FIX_CODE="outputs/aspire_libero_long_actor/${SUITE}/${TASK}/task_code.py"
LOG="/tmp/val_long_${TASKSHORT}_progress.log"
[[ ! -f "$FIX_CODE" ]] && echo "ERROR: $FIX_CODE missing" | tee -a "$LOG" && exit 1
echo "Stage 2 start: $FIX_CODE  GPU=$GPU" | tee -a "$LOG"
for trial in $(seq 1 50); do
    trial_padded=$(printf "%02d" $trial)
    find outputs/aspire_libero_long_eval -maxdepth 6 -type d -name "trial_${trial_padded}_*" 2>/dev/null | grep -q "${SUITE}/${TASK}" && \
        echo "Trial $trial: skip" | tee -a "$LOG" && continue
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    PYTHONPATH=$PYTHON_ROOT \
    .venv-libero/bin/python3 scripts/libero/replay_trial.py \
        --args.suite "${SUITE}" --args.task "${TASK}" --args.trial ${trial} \
        --args.replay-code "${FIX_CODE}" \
        --args.config env_configs/libero/franka_libero_libero10_traced.yaml \
        --args.output-dir outputs/aspire_libero_long_eval > "/tmp/val_long_${TASKSHORT}_${trial}.log" 2>&1 || true
    reward=$(grep -oE "reward_[0-9]+\.[0-9]+" "/tmp/val_long_${TASKSHORT}_${trial}.log" | tail -1 | sed 's/reward_//')
    echo "Trial $trial: ${reward:-ERROR}" | tee -a "$LOG"
done
PYTHONPATH=$PYTHON_ROOT .venv/bin/python3 scripts/libero/gen_progress_long.py
echo "STAGE2_DONE" | tee -a "$LOG"
SCRIPT
sed -i "s/FILL_IN_SUITE/$SUITE/g; s/FILL_IN_TASK/$TASK/g; s/FILL_IN_GPU/$GPU/g; s/FILL_IN_TASKSHORT/$TASKSHORT/g" /tmp/stage2_long_$TASKSHORT.sh
grep "FILL_IN_" /tmp/stage2_long_$TASKSHORT.sh && echo "WARNING: unresolved!" || echo "OK"
chmod +x /tmp/stage2_long_$TASKSHORT.sh
nohup bash /tmp/stage2_long_$TASKSHORT.sh > /tmp/stage2_long_${TASKSHORT}.out 2>&1 &
echo "PID=$!"
```

Poll: `grep -c "Trial\|DONE" /tmp/val_long_${TASKSHORT}_progress.log`

---

## Suite Name Collision Warning

`libero_10_swap` and `libero_10_task` share identical task names. Always include the full SUITE in subagent briefs.
