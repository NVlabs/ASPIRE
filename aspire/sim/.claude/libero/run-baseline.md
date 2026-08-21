---
name: run-baseline
description: "How to run LIBERO baselines with run_libero_batch.py: launch command, PYTHONPATH fix, monitoring, output structure, timing, and image-difference reference results."
---

# run-baseline

How to run a full Gemini/Opus LIBERO baseline across LIBERO-Pro suites using `run_libero_batch.py`.

---

## Overview

The batch runner iterates over suites × tasks × seeds, writes artifacts to `./outputs/<run_name>/`, and prints per-task summary statistics. Tasks run sequentially; 5 workers run trials concurrently within each task.

**LIBERO-Pro suites** (6 total):
- `libero_goal_swap`, `libero_goal_task` — 10 tasks each
- `libero_object_swap`, `libero_object_task` — 10 tasks each
- `libero_spatial_swap`, `libero_spatial_task` — 10 tasks each

**Seeds**: convention is seeds 51–65 (15 trials) for baselines. Set via `resume_idx: 51` in config + `--args.total-trials 65`.

---

## Prerequisites

1. **Perception servers must be available**: SAM3 on `8114`, GraspNet on `8115`, and PyRoKi on `8116`. The experiment coordinator normally checks these ports and starts missing servers during preflight. To prelaunch them manually, run `bash scripts/common/start_perception_servers.sh` from a persistent tmux pane.
2. **Inference credentials** loaded from an approved secret manager into `NVIDIA_API_KEY`; the image-difference feedback model may reuse the same credential when permitted.
3. A baseline config file. See `env_configs/libero/`:
   - `franka_libero_baseline_debug.yaml` — text-only feedback (`use_img_differencing: false`)
   - `franka_libero_baseline_image_diff_debug.yaml` — image-difference feedback (`use_img_differencing: true`, `resume_idx: 51`)

---

## Launch command

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY from your secret manager}"

PYTHONPATH=$PYTHON_ROOT nohup .venv-libero/bin/python3 -u cap/envs/scripts/run_libero_batch.py \
  --args.base-config-path env_configs/libero/franka_libero_baseline_image_diff_debug.yaml \
  --args.server-url https://inference-api.nvidia.com/v1/chat/completions \
  --args.api-key "$NVIDIA_API_KEY" \
  --args.models 'gcp/google/gemini-3.1-pro-preview' \
  --args.visual-differencing-model 'gcp/google/gemini-3.1-pro-preview' \
  --args.visual-differencing-model-server-url 'https://inference-api.nvidia.com/v1/chat/completions' \
  --args.visual-differencing-model-api-key "$NVIDIA_API_KEY" \
  --args.suites libero_goal_swap libero_goal_task libero_object_task libero_object_swap libero_spatial_task libero_spatial_swap \
  --args.num-workers 5 \
  --args.total-trials 65 \
  --args.output-dir ./outputs/libero_baseline_image_diff_gemini \
  > /tmp/baseline_run.log 2>&1 &
echo "PID: $!"
```

**Key flags:**
- `PYTHONPATH=...` — required when running from the checkout so local `aspire.sim.cap` sources and configs are on the path
- `--args.suites` — space-separated list; can be a single suite or all six
- `--args.task-names` — optional substring filter, e.g. `--args.task-names wine_bottle bowl_on_stove` runs only tasks whose name contains one of those strings
- `--args.num-workers 5` — 5 concurrent trials per task (15 trials / 5 workers = 3 rounds)
- `--args.total-trials 65` — combined with `resume_idx: 51` in config → runs seeds 51–65 only
- `--args.output-dir` — artifacts go to `<output_dir>/<suite>/<task>/<model_name>/run/`

**Text-only baseline:** omit the `--args.visual-differencing-model*` flags and use `env_configs/libero/franka_libero_baseline_debug.yaml`.

---

## Output structure

```
outputs/libero_baseline_image_diff_gemini/
  libero_goal_swap/
    put_the_wine_bottle_on_top_of_the_cabinet/
      config.yaml                          # config used for this task
      gcp_google_gemini-3.1-pro-preview/
        run/
          initial_prompt.txt
          all_responses.json
          trial_51_sandboxrc_0_reward_1.000_taskcompleted_1/
            code.py
            trace.json                     # TraceLogger output (API calls)
            keyframes/                     # per-API-call images
            visual_feedback_00.png         # image-difference feedback input images
            differencing_feedback_00.txt   # image-difference text output
            video_turn_00.mp4
            video_combined.mp4
          trial_52_sandboxrc_0_reward_0.000_taskcompleted_0/
            ...
```

**Success counting:** a trial succeeds if ANY `sandboxrc_N` dir has `taskcompleted_1`. Count unique trial numbers with `taskcompleted_1`:
```bash
ls run/ | grep "taskcompleted_1" | sed 's/trial_\([0-9]*\)_.*/\1/' | sort -nu | wc -l
```

---

## Monitoring

Set up a lean monitor — only batch completion and real errors:
```bash
tail -f /tmp/baseline_run.log | grep --line-buffered -E \
  "Batch execution|Code generation success rate|Total tasks|Running Experiment [0-9]+/[0-9]+\$|FAILED|Killed|OOM"
```

The runner prints a summary line per task:
```
Code generation success rate / Average reward / Task completed:
1.000/0.667/10
```
Format: `code_gen_rate / avg_reward / num_successes` (out of 15 trials).

---

## Timing

- Failing trials run to `max_steps` (default 4000 steps ≈ 6–7 min/trial)
- With 5 workers: 3 rounds × 7 min = ~21 min/task for 0% tasks
- With 5 workers: ~8–12 min/task for tasks with >30% success
- **Rule of thumb**: ~20 min/task average; 60 tasks (all 6 suites) ≈ 20 hours total

To speed up future runs, reduce `max_steps` in the config (e.g. 2000) — halves time for failing trials.

---

## Gemini image-difference baseline results (seeds 51-65, Apr 2026)

| Suite | Rate |
|---|---|
| `libero_goal_swap` | 39/150 = **26%** |
| `libero_goal_task` | 27/150 = **18%** |
| `libero_object_task` | 33/150 = **22%** |
| `libero_object_swap` | 34/150 = **23%** |
| `libero_spatial_task` | 21/150 = **14%** |
| `libero_spatial_swap` | 12/150 = **8%** |
| **Overall** | **166/900 = 18%** |

**Failure patterns:**
- Drawer tasks (open_middle_drawer, open_top_drawer): **0% across all suites** — no tactile feedback
- Spatial localization ("next to", "between"): near-0% — SAM3 can't reliably distinguish relative positions
- Carton/box objects (milk, OJ): low success — GraspNet struggles with flat/rectangular shapes
- Best tasks: wine bottle on cabinet (67%), tomato sauce (60%), butter swap (53%)
