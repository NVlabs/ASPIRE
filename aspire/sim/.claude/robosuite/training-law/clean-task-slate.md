---
name: robosuite/training-law/clean-task-slate
description: Checklist for resetting a Robosuite task to clean slate before rerunning the fix loop. MUST ask user for approval before executing any deletions/moves.
---

# Clean Task Slate — Reset a Robosuite Task for Fresh Fix Loop

> **IMPORTANT:** When the user asks to clean up a task, gather the full list of files/dirs that exist on disk, present it to the user, and **wait for explicit approval** before deleting or moving anything.

---

## Workflow

1. **Identify what exists** — run the verification commands below for the target task
2. **Present the list** — show the user every file/dir you plan to move or delete
3. **Wait for approval** — do NOT proceed until the user confirms
4. **Execute** — move everything to `$ASPIRE_ROOT/outputs/agent_logs/<task>/`
5. **Regenerate progress** — run `gen_progress_robosuite.py` and verify `pending`

---

## Checklist (substitute TASK and CONFIG_STEM)

### 1. Fix loop artifacts (in baseline dir)
```
BASELINE_DIR=outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel
```
- `$BASELINE_DIR/$CONFIG_STEM/fix_code.py`
- `$BASELINE_DIR/$CONFIG_STEM/findings.md`
- `$BASELINE_DIR/$CONFIG_STEM/reasoning.txt`
- `$BASELINE_DIR/$CONFIG_STEM/code_versions/` (all versioned fix attempts)
- `$BASELINE_DIR/$CONFIG_STEM/*/BLOCKED.md` (any per-trial blocked files)

### 2. Named copy of fix code
- `outputs/working_codes/robosuite_${TASK}_fix.py`

### 3. Eval outputs
- `outputs/robosuite_fix_eval/$CONFIG_STEM/` (entire dir)
- Any backup dirs: `outputs/robosuite_fix_eval_bak_*/$CONFIG_STEM/`

### 4. /tmp scratch files
- `/tmp/fix_progress_checkpoint_${TASK}.md`
- `/tmp/fix_test_${TASK}*` (all Stage 1 test replay dirs)
- `/tmp/fast_path_test_${TASK}/`
- `/tmp/fix_attempt_${TASK}.py`
- `/tmp/baseline_success_${TASK}.py`
- `/tmp/repl_out_${TASK}*` (REPL session outputs)
- `/tmp/val_robosuite_${TASK}_*.log` (old val logs)
- `/tmp/rerun_${TASK}/` (manual rerun logs)
- `/tmp/rerun_debug_${TASK}/`

### 5. Skill entries referencing this task
- `.claude/robosuite/training-law/skills/localize.md` — prompt registry rows for this task
- `.claude/robosuite/training-law/skills/grasp.md` — grasp strategies specific to this task
- `.claude/robosuite/training-law/skills/transport.md` — motion patterns specific to this task

### 6. Regenerate progress
```bash
.venv-robosuite/bin/python3 scripts/robosuite/gen_progress_robosuite.py
```
Verify task shows as `pending`.

---

## Discovery Commands (run these first to build the approval list)

```bash
TASK="<task_name>"
CONFIG_STEM="<config_stem>"
BASELINE_DIR="outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel"

echo "=== Fix loop artifacts ==="
ls -la "$BASELINE_DIR/$CONFIG_STEM/fix_code.py" "$BASELINE_DIR/$CONFIG_STEM/findings.md" "$BASELINE_DIR/$CONFIG_STEM/reasoning.txt" 2>/dev/null
find "$BASELINE_DIR/$CONFIG_STEM" -name "BLOCKED.md" 2>/dev/null

echo "=== Named copy ==="
ls -la "outputs/working_codes/robosuite_${TASK}_fix.py" 2>/dev/null

echo "=== Eval outputs ==="
ls -d "outputs/robosuite_fix_eval/$CONFIG_STEM" 2>/dev/null
ls -d outputs/robosuite_fix_eval_bak_*/"$CONFIG_STEM" 2>/dev/null

echo "=== /tmp scratch ==="
ls -d /tmp/fix_progress_checkpoint_${TASK}.md /tmp/fix_test_${TASK}* /tmp/fast_path_test_${TASK} /tmp/fix_attempt_${TASK}.py /tmp/baseline_success_${TASK}.py /tmp/repl_out_${TASK}* /tmp/rerun_${TASK} /tmp/rerun_debug_${TASK} 2>/dev/null
ls /tmp/val_robosuite_${TASK}_*.log 2>/dev/null | wc -l

echo "=== Skill entries ==="
grep -n "$TASK" .claude/robosuite/training-law/skills/localize.md .claude/robosuite/training-law/skills/grasp.md .claude/robosuite/training-law/skills/transport.md 2>/dev/null
```
