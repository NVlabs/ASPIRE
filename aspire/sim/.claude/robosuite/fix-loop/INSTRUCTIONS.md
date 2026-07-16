---
name: robosuite/fix-loop/INSTRUCTIONS
description: End-to-end run steps for the robosuite fix-loop experiment — baseline, fix loop, eval seeds 1–100.
---

## Robosuite eval workflow

This is the end-to-end process for running the Robosuite evaluation and fix loop.

### 1) Run baseline (collect baseline results + traces/logs)

Run the baseline so you have `trace.json`, `keyframes/`, and baseline logs.

- Skill doc: `.claude/robosuite/run-baseline.md`
- Command:

```bash
scripts/robosuite/run_baseline_robosuite.sh <codegen_key1> <codegen_key2> <vdm_key1> <vdm_key2>
```

Baseline artifacts are written under:

- `outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/`
- `logs/robosuite_baseline.log` (and proxy logs under `logs/`)

### 2) Fix loop (Claude Code, Auto mode)

Tell Claude Code to follow the Robosuite fix loop skill:

- `.claude/robosuite/fix-loop/main-agent-prompt.md`

If you need to reset a task to a clean slate before rerunning the fix loop, follow:

- `.claude/robosuite/fix-loop/clean-task-slate.md`

The fix loop produces `fix_code.py` for each task/config under the baseline traced directory.

### 3) Run the fix code manually on eval seeds 1–100

After `fix_code.py` exists, run eval seeds 1–100 with:

```bash
bash scripts/robosuite/run_eval_fix_code.sh
```

This replays each task’s `fix_code.py` across seeds 1–100 and writes outputs under:

- `outputs/robosuite_fix_eval/`

### 4) (Optional) Rerun debug seeds 101–125

If you want to rerun the fix loop’s debug seed range (the range used during debugging):

```bash
bash scripts/robosuite/run_debug_fix_code.sh
```

