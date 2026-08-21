---
name: robosuite/training-law/INSTRUCTIONS
description: End-to-end run steps for the robosuite training scaling-law experiment — baseline, versioned fix loop, eval, and tokens-vs-SR plots.
---

## Robosuite training-law workflow

End-to-end process for the **training scaling law** experiment.

**Goal:** For each fix-loop iteration, save the code snapshot, timestamp, and debug success rate under `code_versions/`. Then measure **eval** success rate (seeds 1–100) for every saved code version and plot **cumulative tokens (x)** vs **success rate (y)**.

**Per-task baseline root:**
```
outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/<config_stem>/
```

---

### 1) Run traced baseline

Collect baseline traces for all 7 tasks (required before the fix loop).

- Skill doc: `.claude/robosuite/run-baseline.md`
- Command:

```bash
scripts/robosuite/run_baseline_robosuite.sh <codegen_key1> <codegen_key2> <vdm_key1> <vdm_key2>
```

Artifacts:

- `outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/<config_stem>/`
- `logs/robosuite_baseline.log`

---

### 2) Fix loop (train-law) — builds `code_versions/`

Run in Claude Code (Auto mode). Each subagent iteration must use `scripts/robosuite/run_iteration.sh` (do not run seeds or write `result.json` by hand).

- Coordinator: `.claude/robosuite/training-law/main-agent-prompt.md`
- Subagent template: `.claude/robosuite/training-law/subagent-prompt.md`
- Reset a task: `.claude/robosuite/training-law/clean-task-slate.md`
- Progress: `docs/progress/fix_loop_robosuite_progress.md`

**Where artifacts are saved (every iteration N):**

```
outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/<config_stem>/code_versions/
  iter_{N}_{YYYYMMDD_HHMMSS}.py    # code snapshot (timestamp in filename)
  iter_{N}_result.json               # debug results: seeds 101–125
```

Example for `cube_lifting`:

```
.../cube_lifting_multimodel_aspire_traced/code_versions/
  iter_1_20260520_140532.py
  iter_1_result.json
  iter_2_20260520_151204.py
  iter_2_result.json
  ...
```

`iter_{N}_result.json` fields: `iter`, `code_file`, `n_pass`, `n_total`, `pass_rate`, `seeds` (debug **101–125** only).

Subagents call:

```bash
scripts/robosuite/run_iteration.sh \
  --code /tmp/fix_code_${TASK}_tl.py \
  --config <traced yaml> \
  --task <task> \
  --iter <N> \
  --gpu <3-7> \
  --seeds 101-125 \
  --workers 5
```

Stop when the script exits **1** (5 consecutive 25/25 on debug seeds). Final promoted code (separate from versioning):  
`.../<config_stem>/fix_code.py`

**Important:** Debug seeds 101–125 only during the fix loop. Eval seeds 1–100 are step 3.

---

### 3) Training scaling law — eval all code versions + tokens + plot

#### A) Eval seeds 1–100 for every code version

Scan all `iter_*_{timestamp}.py` files in `code_versions/` across the **7 tasks**, and replay each on that task’s traced config with seeds **1–100**.

`scripts/robosuite/run_eval_fix_code.sh` is **not** for this step (it only runs final `fix_code.py`).

```bash
scripts/robosuite/run_eval_training_law.sh
```

Optional filters:

```bash
scripts/robosuite/run_eval_training_law.sh --task cube_lifting
scripts/robosuite/run_eval_training_law.sh --task cube_lifting --iter 3
scripts/robosuite/run_eval_training_law.sh --skip-existing   # skip if iter_<N>_eval_result.json exists
```

5 parallel workers per iteration (20 seeds each), GPUs 3–7 round-robin. Tasks and iterations run sequentially — use `--task` / `--iter` for partial runs.

**Eval results are saved in two places:**

**i. Summary JSON** (use for scaling-law plots):

```
outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/<config_stem>/code_versions/iter_<N>_eval_result.json
```

Same folder as debug `iter_<N>_result.json`. Fields: `iter`, `code_file`, `seed_range` (`"1-100"`), `n_pass`, `n_total`, `pass_rate`, `seeds`.

**ii. Per-trial replay artifacts:**

```
outputs/training_law_eval/<config_stem>/iter_<N>/trial_<seed>_.../
```

Replay uses `--args.flat-output` (no config stem, model folder, or `run/` level).

**Logs:** `/tmp/training_law_eval_<task>_iter<N>/<seed>.log`

#### B) Scaling-law plots

Plots cumulative **tokens** (x) vs **success rate** (y). Token data comes from `~/.claude/projects/<sanitized-repo-path>/subagents/*.jsonl` (auto-discovered) plus `code_versions/` timestamps. Run **after** step 3A for eval curves.

**Eval SR (seeds 1–100)** — requires `iter_<N>_eval_result.json`:

```bash
.venv/bin/python3 scripts/robosuite/plot_tokens_vs_eval_sr.py \
  --output outputs/plots/tokens_vs_eval_sr.png \
  --print-table
```

**Debug SR (seeds 101–125)** — uses `iter_<N>_result.json` only:

```bash
.venv/bin/python3 scripts/robosuite/plot_tokens_vs_debug_sr.py \
  --output outputs/plots/tokens_vs_debug_sr.png \
  --print-table
```

---

### Quick reference

| Step | Command / artifact |
|------|-------------------|
| Baseline | `scripts/robosuite/run_baseline_robosuite.sh` |
| Fix loop | `main-agent-prompt.md` + `run_iteration.sh` |
| Code + debug SR | `.../<config_stem>/code_versions/iter_{N}_{ts}.py` + `iter_{N}_result.json` |
| Eval all versions | `run_eval_training_law.sh` → `code_versions/iter_{N}_eval_result.json` + `training_law_eval/` |
| Final fix only | `run_eval_fix_code.sh` (not for scaling-law curves) |
| Plot eval SR | `plot_tokens_vs_eval_sr.py` (tokens + eval SR; after 3A) |
| Plot debug SR | `plot_tokens_vs_debug_sr.py` (tokens + debug SR; optional) |
