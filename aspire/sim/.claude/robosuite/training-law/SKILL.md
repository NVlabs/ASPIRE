---
name: robosuite/training-law/SKILL
description: Master reference for ASPIRE Robosuite experiments. Covers system overview, 7 tasks, setup, running experiments, debugging, full API reference, and all pipeline modes (Fix Loop, Baseline).
---

# ASPIRE Experiment Pipeline

---

## What Is ASPIRE

ASPIRE — LLMs write Python code to control a robot via a structured API. Code runs in MuJoCo (Robosuite).

**7 Robosuite tasks:**

| Task | Type | Camera key |
|---|---|---|
| `cube_lifting` | Single-arm | `robot0_robotview` |
| `cube_restack` | Single-arm | `robot0_robotview` |
| `cube_stack` | Single-arm | `robot0_robotview` |
| `nut_assembly` | Single-arm | `robot0_robotview` |
| `spill_wipe` | Single-arm | `robot0_robotview` |
| `two_arm_lift` | Bimanual | `robot0_robotview` |
| `two_arm_handover` | Bimanual | `robot0_robotview` |

---

## Pipeline Modes

| Mode | File | When |
|---|---|---|
| **Robosuite Baseline** | [run-baseline.md](../run-baseline.md) | Collect baseline on all 7 tasks |
| **Robosuite Fix Loop (train-law)** | [main-agent-prompt.md](main-agent-prompt.md) | Coordinator guide: dispatch one subagent per GPU to debug failures |
| **Robosuite Fix Loop Subagent (train-law)** | [subagent-prompt.md](subagent-prompt.md) | Self-contained prompt template for dispatching one task to a background subagent |

## Reference Files

| File | Covers |
|---|---|
| [run-baseline.md](../run-baseline.md) | Launch command, config paths, output structure for all 7 tasks |
| [api-reference.md](../api-reference.md) | Full API functions, output structure, TraceLogger format, source files |
| [clean-task-slate.md](clean-task-slate.md) | Checklist for resetting a task to clean slate before rerunning fix loop |

## Companion Skills

| Skill | Covers |
|---|---|
| [`grasp`](skills/grasp.md) | Pick-and-place code template, pre-grasp/lower/close/lift/place skeleton |
| [`localize`](skills/localize.md) | Perception server (SAM3/Molmo) prompting strategy, per-object prompt registry |
| [`transport`](skills/transport.md) | Motion patterns for moving objects between locations — multi-step waypoints, safe transit sequences, interpolated Cartesian moves, collision avoidance during transport |

---

## Setup

**Two venvs:**
- `.venv-robosuite` (Python 3.10) — Robosuite replay/eval: `replay_trial_robosuite.py`
- `.venv-libero` or `.venv-perception` — perception servers only (handled by `start_perception_servers.sh`)

**Always use `.venv-robosuite/bin/python3` for Robosuite replay/eval. Never use system python after setup.**

---

## Perception Servers (required before any experiment)

```bash
tmux new -s aspire-perception
cd "$ASPIRE_ROOT"
ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3 \
  bash scripts/common/start_perception_servers.sh --with-molmo
for p in 8114 8115 8116 8122; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
# 404/non-000 = UP for 8114-8116, 000 = DOWN. Molmo health is /v1/models on 8122.
```

SAM3 uses gated Hugging Face weights; authenticate before startup. GraspNet
requires the pinned Contact-GraspNet submodule and the perception environment
to include `--extra contactgraspnet`; the startup script verifies and applies
the compatibility patch. Molmo starts by default; `--with-molmo` aborts unless
the perception environment provides a `vllm` executable, so pass `--no-molmo`
to skip it and give up point-prompt fallback. Keep
servers in tmux or another persistent terminal; one-off background shells can
exit and take child server processes down with them.

| Server | Port | GPU | Required for |
|---|---|---|---|
| SAM3 | 8114 | Configured GPU | All runs |
| GraspNet | 8115 | Configured GPU | All runs |
| PyRoKi | 8116 | CPU | All runs |
| Molmo | 8122 | Configured GPU | Point-prompt fallback |

**GPU layout:** Derive server and worker assignments from the active host configuration.

---

## Running Experiments

Scripts use `tyro.cli` with a named `args` parameter — require the **`--args.` prefix**.
The examples assume `SIM_GPU`, `SIM_GPUS`, and `DEBUG_TRIAL_ID` are set from
the active host configuration and development partition.

```bash
# Replay fix code on a single trial
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$SIM_GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
  --args.config env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml \
  --args.trial "$DEBUG_TRIAL_ID" \
  --args.replay-code /tmp/fix_attempt.py \
  --args.output-dir ./outputs/debug_fix

# Interactive REPL (all API functions in scope)
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$SIM_GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
  --args.config env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml \
  --args.trial "$DEBUG_TRIAL_ID" \
  --args.interactive \
  --args.output-dir /tmp/repl_out
```

**Config path pattern:** `env_configs/robosuite/<task>_multimodel_aspire_traced.yaml` — all 7 tasks (including `two_arm_handover`) follow this single scheme. See the task reference table in `main-agent-prompt.md` for exact paths.

**Trial split:** Read the development and held-out partitions from the active experiment config; keep them disjoint and lock held-out trials during debugging.

---

## Critical Rules

1. **NEVER git push** — local commits only
2. **No forbidden APIs**: `sim.data.body_xpos`, `sim.data.get_site_xpos`, `sim.data.set_joint_qpos`, `sim.model.*`, `sim.data.qpos`, `sim.forward()`, `env._step_once()`
3. **`replay_trial_robosuite.py` uses `--args.` prefix** (tyro wraps the `args` parameter)
4. **`MUJOCO_GL=egl`** + **`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`** for all runs
5. **Log** to `docs/logs/YYYY-MM-DD.md` after significant work
6. **Update the experiment's `skills/`** (`.claude/robosuite/training-law/skills/`) after any new pattern discovered

---

## Monitoring

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
for gpu in $SIM_GPUS; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```
