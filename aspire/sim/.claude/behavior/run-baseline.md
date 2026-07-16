# BEHAVIOR-1K Baseline Runs

BEHAVIOR runs through `python -m aspire.sim.cap.envs.launch`. Run one config at a time from
`$ASPIRE_ROOT` (`aspire/sim`) with the B1K virtual environment active. For the full launch map, read
`skills/system-pipeline.md`. B1K execution is host-dependent; treat the first run on a new machine as smoke testing.

## Environment

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
```

If perception servers are not already running, the YAML `api_servers` section
will start SAM3 and ContactGraspNet for the trial. On multi-GPU machines, prefer
running perception on a different GPU from Isaac Sim when memory allows.

## Smoke Tasks

Use these configured tasks as the first smoke targets after installation. On an unvalidated host, prefer a single exact traced seed via `launch_b1k --trial-ids` before full ranges:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml

uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_trash.yaml
```

## Supported Task Variants

| Task | Config |
|---|---|
| Radio pickup | `env_configs/r1pro/r1pro_pick_up_radio.yaml` |
| Radio pickup, ASPIRE | `env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml` |
| Radio pickup, traced debug | `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml` |
| Radio pickup, oracle | `env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml` |
| Soda-can pickup | `env_configs/r1pro/r1pro_pick_up_trash.yaml` |
| Soda-can pickup, ASPIRE | `env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml` |
| Soda-can pickup, traced debug | `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml` |
| Soda-can pickup, oracle | `env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml` |

The soda-can configs retain the historical `trash` filename because the existing
environment class is `R1ProTrashCodeEnv`, but the prompt targets a blue can of
soda.

## Traced Debug Runs

Use traced configs for offline trace analysis:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --output-dir outputs/behavior/debug/radio_aspire_traced

uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml \
  --output-dir outputs/behavior/debug/soda_aspire_traced
```

The traced configs default to debug seeds 26-35 via `resume_idx: 26` and
`trials: 35`. They write `trace.json`, `keyframes/`, VDM feedback files, videos,
prompts, and generated code under each trial directory.

Use `python -m aspire.sim.cap.envs.launch_b1k --trial-ids ...` for exact debug seed lists.
Use `scripts/behavior/replay_trial_b1k.py` for BEHAVIOR saved-code replay or REPL
debugging. `scripts/libero/replay_trial.py` is LIBERO-specific and should not be used
for BEHAVIOR replay.

## Outputs To Inspect

- `code.py` for generated code.
- `trace.json` for API calls, durations, return summaries, and failures when
  using traced configs.
- `keyframes/` for traced RGB, depth, SAM3 overlays, and mask arrays.
- `video_*.mp4` for execution playback when `record_video: true`.
- `differencing_feedback_*.txt` for ASPIRE visual-differencing feedback.
- `prompts_and_responses/` for saved initial and multi-turn prompts when enabled.
- stdout/stderr in the trial folder.
- Any saved observations from `save_current_observation(name)`.
