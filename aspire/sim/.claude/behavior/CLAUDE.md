# BEHAVIOR-1K Suite Guide

This is the suite-level reference for BEHAVIOR-1K / R1Pro simulation experiments in ASPIRE.
Read this file, then the target experiment `INSTRUCTIONS.md`.

## Working Root And Status

Run commands from the simulation workspace, not the repository root:

```bash
cd aspire/sim
export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"
```

This branch ships the R1Pro task configs and runbooks for radio pickup and soda-can pickup.
The B1K setup is host-dependent and has not been runtime-validated in this documentation audit;
treat the first run on a machine as bring-up/smoke testing and record any local fixes.

## Experiment Entrypoints

| File | Purpose |
|---|---|
| [api-reference.md](api-reference.md) | Public R1Pro API surface exposed to generated code |
| [run-baseline.md](run-baseline.md) | Basic batch, traced, replay, and inspection commands |
| [skills/system-pipeline.md](skills/system-pipeline.md) | Execution flow, launch modes, outputs, configs, perception servers |
| [fix-loop/INSTRUCTIONS.md](fix-loop/INSTRUCTIONS.md) | Block-by-block interactive policy runbook |
| [fix-loop/SKILL.md](fix-loop/SKILL.md) | Skill entrypoint for BEHAVIOR fix-loop agents |
| [aspire-protocol/INSTRUCTIONS.md](aspire-protocol/INSTRUCTIONS.md) | Canonical seeds 26-35 skill acquisition and isolated seeds 1-25 evaluation protocol |
| [aspire-protocol/SKILL.md](aspire-protocol/SKILL.md) | Entry point for complete Soda Can or Radio ASPIRE campaigns |
| [skills/](skills/) | Clean-slate placeholders plus API/pipeline references |

## Supported Configs

Only these BEHAVIOR configs are part of the current sim workspace:

| Task | Configs |
|---|---|
| Radio pickup | `env_configs/r1pro/r1pro_pick_up_radio.yaml`, `env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml`, `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml`, `env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml` |
| Soda-can pickup | `env_configs/r1pro/r1pro_pick_up_trash.yaml`, `env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml`, `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml`, `env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml` |

The `pick_up_trash` filename is historical; the task prompt targets a blue can of soda.
Do not claim support for other BEHAVIOR-1K tasks unless their YAML configs and runbooks are added here.

## Critical Rules

1. Run one Isaac Sim / OmniGibson trial process per node unless the host has been explicitly validated for more.
2. Use the BEHAVIOR environment at `cap/third_party/b1k/.venv`; do not install B1K into `.venv`, `.venv-libero`, or `.venv-robosuite`.
3. Never run `uv sync` inside the B1K virtual environment after the BEHAVIOR installer has provisioned it.
4. Set `OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=1` before headless runs.
5. Select Isaac Sim's GPU with `OMNIGIBSON_GPU_ID`; examples use GPU 2, but the correct value is host-specific.
6. Record video for debug/replay runs so videos and keyframes are available for analysis. The B1K Tyro launchers default to recording; use the bare `--record-video` flag when making it explicit.
7. Perception ports are config-specific. Check the YAML `api_servers` section before assuming SAM3 or ContactGraspNet ports.
8. Keep API keys and model endpoint secrets out of YAML files and docs; use CLI args or environment variables.
9. Generated task code must use only public R1Pro API calls. Do not inspect OmniGibson internals, BDDL predicates, simulator object registries, or privileged reward state.
10. Keep master clean. Add reusable learned patterns on an experiment branch such as `learned-skills`, not directly to master, unless the maintainer asks for a release update.

## Standard Commands

Activate the B1K environment:

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
ulimit -c 0
```

Run one exact traced debug seed batch:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active python -m aspire.sim.cap.envs.launch_b1k \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --trial-ids 26 27 28 \
  --output-dir outputs/behavior/debug/radio_aspire_traced \
  --record-video
```

Replay saved policy code on one seed without an LLM call:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --replay-code outputs/interactive/fix_code_interactive_radio.py \
  --trial 26 \
  --output-dir outputs/behavior/replay/radio_trial26 \
  --record-video
```

Open an interactive REPL with R1Pro API functions in scope:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --interactive \
  --trial 26 \
  --output-dir outputs/behavior/interactive/radio_repl26 \
  --record-video
```

For full setup and supported config details, read `docs/behavior-tasks.md` and
`skills/system-pipeline.md`.
