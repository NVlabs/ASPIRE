# BEHAVIOR-1K System Pipeline Reference

This is the release-facing ASPIRE BEHAVIOR pipeline map. It is adapted to the
files that exist in this repo; do not assume wrappers or task configs from
older B1K branches exist here. Use the current ASPIRE launchers documented here.

Read this before opening BEHAVIOR files. For API function details, see
`r1pro-api.md`. For policy pattern placeholders, see `interactive-policy.md`,
`perception-and-search.md`, `navigation.md`, and `grasping.md`.

Run commands from the simulation workspace:

```bash
cd aspire/sim
export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"
```

B1K setup and execution are host-dependent. The paths and flags below are
validated against the repository layout and launcher code, but full Isaac Sim /
OmniGibson execution still needs smoke testing on the target machine.

## 1. What Is ASPIRE / B1K

ASPIRE evaluates LLM-written Python policies in robot environments:

```text
task prompt + API reference -> LLM writes code -> sandbox executes code
  -> simulator advances -> reward/task completion is measured
```

The current fix-loop target is an interactive closed-loop policy: write a code
block, replay it, inspect observations/traces/videos, append the next block,
and repeat until the final file is a robust observe-act-observe state machine.

This branch ships BEHAVIOR-1K simulation configs for the R1Pro mobile
bimanual robot in Isaac Sim / OmniGibson. B1K tasks are longer horizon than
tabletop MuJoCo tasks because the robot must move a base, search with a head
camera, keep targets in view, and recover from perception or grasp failures.

Supported BEHAVIOR task families in this repo:

| Task family | Current config prefix | Notes |
|---|---|---|
| Radio pickup | `env_configs/r1pro/r1pro_pick_up_radio` | Pick up the red radio on the table |
| Soda-can pickup | `env_configs/r1pro/r1pro_pick_up_trash` | Historical filename; prompt targets a blue can of soda |

Do not claim support for BEHAVIOR task configs that are not present in
`env_configs/r1pro/`.

## 2. Execution Flow

```text
YAML config + CLI args
  -> aspire.sim.cap.envs.launch or aspire.sim.cap.envs.launch_b1k
    -> aspire.sim.cap.utils.launch_utils._load_config()
    -> aspire.sim.cap.envs.runner._start_api_servers()
    -> aspire.sim.cap.envs.runner._run_headless_trials()
      -> aspire.sim.cap.envs.runner._run_trial_batch()
        -> aspire.sim.cap.envs.trial._run_single_trial()
          -> env.reset(seed=trial)
          -> LLM or oracle code generation
          -> extract code blocks
          -> for each code block:
             -> env.step(code) runs sandboxed Python with R1Pro API functions
             -> optional multi-turn prompt with stdout/stderr and VDM feedback
             -> model chooses REGENERATE with new code or FINISH
          -> save code, summaries, prompts, VDM text, videos, and traces
```

Current constants:

| Constant | Value | Source |
|---|---:|---|
| Multi-turn code block limit | `10` | `cap/envs/trial.py` |
| Trial timeout per attempt | `1000` seconds | `cap/envs/runner.py` |
| Trial attempts | `2` | `cap/envs/runner.py` |

## 3. Launch Modes

ASPIRE currently has three BEHAVIOR launch paths:

| Mode | Entry point | Use |
|---|---|---|
| Batch LLM/oracle run | `python -m aspire.sim.cap.envs.launch` | Normal evaluation from YAML `resume_idx` through `trials` |
| Exact-seed batch, replay, or REPL | `python -m aspire.sim.cap.envs.launch_b1k` | B1K-specific debugging and explicit trial IDs |
| Single-trial replay/REPL helper | `scripts/behavior/replay_trial_b1k.py` | Focused replay or interactive debugging without LIBERO task injection |

Operational rules from `.claude/behavior/CLAUDE.md`:

- Run one Isaac Sim process per node unless the machine has been explicitly
  verified for more.
- Examples use GPU 2 for the trial runner; choose the correct `OMNIGIBSON_GPU_ID` for the host.
- Never run `uv sync` inside the B1K virtual environment.
- Record video for trial runs. The B1K Tyro launchers default to recording; use
  the bare `--record-video` flag when making it explicit.
- Do not write API keys into YAML or docs.
- Do not push from an automated fix-loop run.

### Batch mode

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1

OMNIGIBSON_GPU_ID=2 uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml \
  --output-dir outputs/behavior/validation/radio_aspire \
  --record-video
```

Use `python -m aspire.sim.cap.envs.launch_b1k` when you need exact seed control:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active python -m aspire.sim.cap.envs.launch_b1k \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml \
  --trial-ids 26 27 28 29 30 \
  --output-dir outputs/behavior/debug/soda_aspire_traced \
  --record-video
```

### Replay mode

Replay executes a saved `code.py` without calling the LLM. This is the fastest
way to test a hand-edited fix against the same seed.

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml \
  --replay-code outputs/interactive/fix_code_interactive.py \
  --trial 26 \
  --output-dir outputs/behavior/replay/soda_trial26 \
  --record-video
```

Replay parses saved files on `# Code block N` headers. If no headers are found,
the entire file is executed as one block. This is compatible with the
interactive-policy workflow: append 5-20 lines, replay the file, inspect the
new observations, and append the next block.

### Interactive REPL mode

Interactive mode resets one B1K trial and opens a Python REPL with `env`, `obs`,
`np`, `args`, `step(code_str)`, and all R1Pro API functions in scope.

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --interactive \
  --trial 26 \
  --output-dir outputs/behavior/interactive/radio_trial26 \
  --record-video
```

Do not use `scripts/libero/replay_trial.py` for BEHAVIOR; it is LIBERO-specific.

### Multi-GPU isolation

Select the Isaac Sim GPU with `OMNIGIBSON_GPU_ID`. `launch_b1k.py` sets
`OMNIGIBSON_APPDATA_PATH=/tmp/og_appdata_gpu<ID>` automatically when a GPU ID is
present and no appdata path is already set. For `launch.py`, set both values
explicitly when running multiple jobs:

```bash
OMNIGIBSON_GPU_ID=2 OMNIGIBSON_APPDATA_PATH=/tmp/og_appdata_gpu2 \
  uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
    --config-path <config> \
    --record-video
```

Isaac Sim is heavy; prefer one active Isaac Sim process per node unless you have
verified the machine can handle more.

## 4. API Class Hierarchy

```text
ApiBase
  -> R1ProControlApi                  # 26 public functions in control.py
       -> R1ProControlApiTraced       # registered name for make_traced_api(...)
            -> TracedApiMixin         # wraps every functions() entry
            -> TraceLogger            # records calls, results, durations, keyframes
```

Key files, relative to `$ASPIRE_ROOT` (`aspire/sim`):

| File | Role |
|---|---|
| `cap/integrations/r1pro/control.py` | R1Pro API implementation and `functions()` map |
| `cap/integrations/trace_logger.py` | `TraceLogger`, `TracedApiMixin`, and `make_traced_api()` |
| `cap/integrations/__init__.py` | Registers `R1ProControlApi` and `R1ProControlApiTraced` |

`make_traced_api(cls)` dynamically creates a subclass that wraps each callable
returned by `cls.functions()` with trace logging. Current tracing buffers
keyframes from API call results and writes them when `TraceLogger.save()` runs.

## 5. TraceLogger

Traced configs set `apis: [R1ProControlApiTraced]`. Every public R1Pro API call
then gets a trace entry with timing, serialized arguments, result summary, and
keyframe status:

```json
{
  "step": 3,
  "timestamp": 42.817,
  "function": "find_object_base_rotate",
  "args": {"object_name": "blue can of soda"},
  "duration_ms": 121350.2,
  "result": {"summary": "..."},
  "keyframe_saved": true
}
```

Trace keyframes and arrays are buffered in memory and saved under
`keyframes/` when the trial artifacts are finalized.

Current keyframe sources:

| API call | Saved artifacts |
|---|---|
| `get_env_observation()` | `step_NNN_obs_env.jpg`, `step_NNN_depth_env.npy` |
| `get_observation()` | camera RGB, depth, intrinsics, extrinsics arrays when present |
| `segment_sam3_text_prompt()` / `segment_sam3_point_prompt()` | SAM3 overlay image and top mask |
| `plan_grasp()` | compressed grasp poses and scores |

Batch trials save traces after final trial artifacts. Replay mode also saves
traces if the selected config uses `R1ProControlApiTraced`.

## 6. Output Directory Structure

The launcher normalizes configured output directories by inserting the model
name before the final path component. For example, an output dir of
`outputs/behavior/debug/radio_aspire_traced` with model `gcp/google/...` becomes:

```text
outputs/behavior/debug/gcp_google_.../radio_aspire_traced/
```

Per-trial folders are named:

```text
trial_NN_sandboxrc_X_reward_Y.YYY_taskcompleted_Z/
```

Meaning:

| Field | Meaning |
|---|---|
| `NN` | Trial ID / environment seed |
| `sandboxrc` | `0` means sandbox execution did not crash |
| `reward` | Final reward reported by the environment |
| `taskcompleted` | `1` when the task success flag is true |

Typical traced trial artifact layout:

```text
trial_26_sandboxrc_0_reward_1.000_taskcompleted_1/
  code.py
  raw_response.sh
  all_responses.json
  summary.txt
  differencing_feedback_00.txt
  visual_feedback_00.png
  prompts_and_responses/
    initial_prompt.txt
    multi_turn_prompt_00.txt
  trace.json
  keyframes/
    step_000_obs_env.jpg
    step_000_depth_env.npy
    step_001_sam3.jpg
    step_001_mask_0.npy
  video_turn_00_rgb.mp4
  video_combined_rgb.mp4
```

Exact filenames vary with enabled options. Ensemble runs may add
`ensemble_candidates*.txt` and `ensemble_synthesis*.txt`. Wrist-camera runs add
wrist video variants.

## 7. Config Files

Only these BEHAVIOR configs are currently supported:

| Task | Mode | Config |
|---|---|---|
| Radio | Single-turn | `env_configs/r1pro/r1pro_pick_up_radio.yaml` |
| Radio | ASPIRE | `env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml` |
| Radio | ASPIRE + trace | `env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml` |
| Radio | Oracle | `env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml` |
| Soda can | Single-turn | `env_configs/r1pro/r1pro_pick_up_trash.yaml` |
| Soda can | ASPIRE | `env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml` |
| Soda can | ASPIRE + trace | `env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml` |
| Soda can | Oracle | `env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml` |

Traced configs use `R1ProControlApiTraced`, `record_video: true`,
`use_img_differencing: true`, and `save_multiturn_prompts: true`. The current
debug traced configs use `resume_idx: 26`, `trials: 35`, and `num_workers: 1`,
so their default run covers seeds 26 through 35.

An ad hoc shared-policy validation can use the non-traced multi-turn configs.
The canonical ASPIRE protocol instead uses traced replay for both stages so a
fresh external agent can adapt block by block within each seed. For all
reporting, record config path, seed range, success count, common failure mode,
and whether the run used traced or non-traced API.

Keep model endpoint details and API keys in CLI args or environment variables.
Do not write API keys into YAML files or docs.

## 8. Perception Servers

The YAML `api_servers` section auto-launches perception servers unless a target
host/port is already occupied. If servers are already running, the launcher
skips startup and reuses them.

For the interactive fix-loop, the expected shared servers are SAM3 on `8114`
and ContactGraspNet on `8115`. Verify the YAML before launch; if a copied config
uses different ports, treat that as an intentional isolated run or normalize it
before debugging. Do not restart shared perception servers during active trials.

Current configured servers:

| Config family | SAM3 port | ContactGraspNet port |
|---|---:|---:|
| Radio configs | `8114` | `8115` |
| Soda single-turn and oracle configs | `8114` | `8115` |
| Soda ASPIRE configs | `8116` | `8117` |

Keep SAM3 and ContactGraspNet alive for the full run. Avoid restarting
perception servers during active trials because in-flight API calls may fail or
return partial results.

## 9. Visual Differencing Model

Multi-turn feedback modes control what the code-generation model receives
between code iterations:

| Mode | Config flag | Feedback between turns |
|---|---|---|
| Stdout/stderr only | `use_img_differencing: false` | Console output and sandbox errors |
| Image VDM | `use_img_differencing: true` | Console output plus before/after visual difference text |
| Video VDM | `use_video_differencing: true` | Console output plus turn-video difference text |

The VDM description is injected into the multi-turn decision prompt alongside
executed code, stdout, stderr, and optional raw visual feedback. The model must
respond with either `REGENERATE` plus fenced Python code or `FINISH`.

## 10. Practical Debug Loop

Use traced debug runs to collect failures, then use replay or interactive mode
to build exact fixes block by block:

1. Start from `outputs/interactive/fix_code_interactive_radio.py` for radio or
   `outputs/interactive/fix_code_interactive.py` for soda.
2. Append 5-20 lines of code.
3. Replay the same seed with `scripts/behavior/replay_trial_b1k.py --replay-code` and
   the bare `--record-video` flag.
4. Inspect `summary.txt`, `trace.json`, `keyframes/`, VDM feedback text, and
   videos for the failed trial.
5. Append the next block based on observations, not a fixed recipe.
6. Return successful patterns to the owning experiment protocol. Do not infer
   held-out seed rules from this inner-loop reference.
