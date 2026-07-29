---
name: robosuite/run-baseline
description: "How to run traced Robosuite baselines with run_robosuite_batch.py: launch command, config paths, output structure, and timing for all 7 Robosuite tasks."
---

# run-baseline-robosuite

How to run a full traced multimodel baseline across all 7 Robosuite tasks.

---

## Overview

The batch runner iterates over config_paths × models × seeds, writes artifacts to `./outputs/<run_name>/`, and prints per-task results. Tasks run sequentially; workers run trials concurrently within each task.

Every trial records `trace.json` and `keyframes/` via the Traced API classes (required for the fix loop).

**Robosuite tasks (7 total):**
- `cube_lifting` — pick up the red cube and lift it
- `cube_restack` — place red cube on top of green cube
- `cube_stack` — place red cube on top of green cube (different layout)
- `nut_assembly` — insert brown square nut onto brown square block
- `spill_wipe` — wipe the brown spill (sponge attached to gripper)
- `two_arm_lift` — bimanual: both arms lift a pot by its handles
- `two_arm_handover` — bimanual: arm 0 picks up hammer, hands to arm 1

**Seeds:** convention is seeds 101–125 (25 trials) for baselines, seeds 1–100 for eval.

---

## Prerequisites

1. **Env**: `.venv-robosuite` exists and can run `aspire.sim.cap` scripts.
2. **Inference credentials**: load the proxy credentials from an approved secret manager into the environment variables used below.
3. **Ports**: this baseline script launches local proxies:
   - codegen proxy on `:8110`
   - VDM proxy on `:8111`
4. **Perception servers**: SAM3 `:8114`, GraspNet `:8115`, and PyRoKi `:8116` are running in a persistent tmux session. SAM3 requires Hugging Face authentication; GraspNet requires the pinned Contact-GraspNet submodule, its compatibility patch, and `--extra contactgraspnet` in the perception environment. The common startup script verifies and applies the patch before starting any service.

---

## Launch command

```bash
: "${CODEGEN_API_KEY_1:?Set CODEGEN_API_KEY_1 from your secret manager}"
: "${CODEGEN_API_KEY_2:?Set CODEGEN_API_KEY_2 from your secret manager}"
: "${VDM_API_KEY_1:?Set VDM_API_KEY_1 from your secret manager}"
: "${VDM_API_KEY_2:?Set VDM_API_KEY_2 from your secret manager}"

scripts/robosuite/run_baseline_robosuite.sh \
  "$CODEGEN_API_KEY_1" "$CODEGEN_API_KEY_2" \
  "$VDM_API_KEY_1" "$VDM_API_KEY_2"
```

**What the script does:**
- Start two local proxies (`aspire.sim.cap.serving.openrouter_server`) for key rotation.
- Run `cap/envs/scripts/run_robosuite_batch.py` for all 7 tasks with the flat `env_configs/robosuite/` traced configs.
- Use `--args.total-trials 125` and baseline convention seeds 101–125 (`resume_idx: 101` in each config).
- Write logs under `logs/` and outputs under `outputs/`.

---

## Output structure

```
outputs/
  baseline_robosuite_multimodel_ensemble_traced/
    ensemble_multimodel/
      ensemble_multimodel/
        <config_stem>/
          trial_101_sandboxrc_0_reward_.../
            code.py
            trace.json
            keyframes/
            video_*.mp4
```

**Success counting:**
```bash
ls outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/<config_stem>/ \
  | grep "taskcompleted_1" | wc -l
```

---

## Monitoring

```bash
tail -f logs/robosuite_baseline.log | grep --line-buffered -E "Batch execution|Running Experiment|FAILED|ERROR"
```

---

## Timing

- Failing trials run to `max_steps` (default 100 steps for Robosuite ≈ 2–4 min/trial)
- With 5 workers: 25 trials / 5 workers = 5 rounds ≈ 10–20 min/task
- 7 tasks total ≈ 1–2 hours for all tasks

---

## Configs reference

| Task | Hillclimb traced multimodel config |
|---|---|
| cube_lifting | `env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml` |
| cube_restack | `env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml` |
| cube_stack | `env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml` |
| nut_assembly | `env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml` |
| spill_wipe | `env_configs/robosuite/spill_wipe_multimodel_aspire_traced.yaml` |
| two_arm_lift | `env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml` |
| two_arm_handover | `env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml` |
