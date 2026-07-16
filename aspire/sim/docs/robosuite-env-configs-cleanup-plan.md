# Robosuite env_configs Cleanup Plan

**Goal:** Flatten `env_configs/<task>/{...,hillclimb/...}` (7 robosuite tasks, ~110 yamls)
into a single flat `env_configs/robosuite/` folder containing exactly **7 yamls** — only the
multimodel-ensemble *traced* config per task — renamed to a simple scheme. Delete everything
else. Update every reference.

## Confirmed findings (reference audit)

Only the per-task **`hillclimb/multimodel_*_multiturn_vdm_reduced_api_skill_lib_traced.yaml`**
is used by the actual robosuite pipeline (baseline / fix-loop / training-law / plots / progress).
All other per-task yamls (base, privileged, reduced_api, multiturn_vf, capagent0_traced,
ensemble_*, debug_*, non-traced multimodel_*) are dead **except** a handful referenced only by
non-robosuite framework files / a dev regression test / docstrings (see "Flagged" below).

Configs are loaded **by full relative path**; yamls are self-contained (no `extends`/include —
"inherited" comments refer to Python class defaults). No `outputs/<stem>/` data exists on disk yet,
so renaming orphans nothing.

## Rename mapping (old → new)

New folder: `env_configs/robosuite/`. New name per task: `<task>_multimodel_aspire_traced.yaml`.

| Task | Old path | New path |
|---|---|---|
| cube_lifting | `env_configs/cube_lifting/hillclimb/multimodel_franka_robosuite_cube_lifting_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml` |
| cube_restack | `env_configs/cube_restack/hillclimb/multimodel_franka_robosuite_cube_restack_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml` |
| cube_stack | `env_configs/cube_stack/hillclimb/multimodel_franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml` |
| nut_assembly | `env_configs/nut_assembly/hillclimb/multimodel_franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml` |
| spill_wipe | `env_configs/spill_wipe/hillclimb/multimodel_franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/spill_wipe_multimodel_aspire_traced.yaml` |
| two_arm_handover | `env_configs/two_arm_handover/hillclimb/multimodel_two_arm_handover_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml` |
| two_arm_lift | `env_configs/two_arm_lift/hillclimb/multimodel_franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib_traced.yaml` | `env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml` |

The new stem (filename w/o `.yaml`) = `<task>_multimodel_aspire_traced`. This also removes the old
"two_arm_handover breaks the pattern" exception — all 7 now share one scheme.

## Steps

1. `git mv` the 7 traced yamls → `env_configs/robosuite/<new>.yaml`.
2. `git rm -r` the 7 old task folders (deletes all non-traced yamls + `hillclimb/`).
3. Update each moved yaml's internal `output_dir:` to `./outputs/<task>_multimodel_aspire_traced`
   (cosmetic — always overridden by `--args.output-dir`) and the usage comment header.
4. Update references (path + bare stem) in robosuite-specific scripts & prompts:
   - `.claude/robosuite/run-baseline.md`, `fix-loop/SKILL.md`, `fix-loop/main-agent-prompt.md`,
     `training-law/SKILL.md`, `training-law/main-agent-prompt.md`
   - `scripts/robosuite/run_baseline_robosuite.sh`, `run_eval_training_law.sh`, `run_eval_fix_code.sh`,
     `run_debug_fix_code.sh`
   - `scripts/robosuite/plot_tokens_vs_sr_v2.py`, `plot_tokens_vs_eval_sr.py`, `plot_tokens_vs_debug_sr.py`,
     `gen_progress_robosuite.py`
   - `scripts/robosuite/replay_trial_robosuite.py` (docstring/help examples: capagent0_traced → new traced)
   - `cap/envs/scripts/run_robosuite_batch.py` (default `config_paths`)
5. Verify: no remaining references to old robosuite task paths/stems outside the flagged files.

## Flagged — left untouched (NOT robosuite configs/prompts; reference now-deleted yamls)

These point at deleted non-traced yamls and will need a follow-up decision (kept out of scope to
avoid changing test thresholds / general docs / framework logic):

- `scripts/robosuite/regression_test.sh` → `cube_stack` base + `cube_stack_multiturn_vdm` (mode-specific
  pass thresholds 38/70 — repointing changes semantics; needs recalibration).
- `cap/envs/launch.py` docstring, `cap/serving/launch_servers.py` docstring → cube_stack base example.
- `README.md`, `docs/development.md` → cube_stack/spill_wipe/nut_assembly base+privileged examples.
- `env_configs/human_oracle_code/robosuite/*` oracle yamls — separate grouping, referenced nowhere, left as-is.
