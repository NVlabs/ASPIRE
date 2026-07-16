---
name: behavior/fix-loop/INSTRUCTIONS
description: Runbook for block-by-block BEHAVIOR-1K R1Pro interactive policy experiments.
---

# BEHAVIOR-1K Interactive Fix Loop

This experiment builds closed-loop interactive policies for the supported R1Pro
BEHAVIOR tasks. The current focus is `pick_up_radio`; `pick_up_trash` is the
legacy filename for soda-can pickup.

Do not write a full policy in one pass. Build one long policy file block by
block:

```text
append 5-20 lines -> replay same seed -> inspect observations/traces/videos -> append next block
```

The target policy should behave as an adaptive observe-act-observe state
machine, not a static recipe.

## 1. Preflight

Read:

- `.claude/memory/MEMORY.md`
- `.claude/behavior/CLAUDE.md`
- `.claude/behavior/api-reference.md`
- `.claude/behavior/skills/system-pipeline.md` sections 1-8
- `.claude/behavior/skills/README.md`
- `.claude/behavior/skills/interactive-policy.md`
- `.claude/behavior/skills/search.md`
- `.claude/behavior/skills/radio-table-tasks.md` for radio work
- `docs/behavior-tasks.md`

Operational rules:

- One Isaac Sim process per node.
- Use GPU 2 for the trial runner unless told otherwise.
- Never run `uv sync` in the B1K virtual environment.
- Always pass `--record-video True`.
- Keep SAM3 and ContactGraspNet alive for the full trial.
- Do not push from a fix-loop run.
- If Isaac Sim creates huge `core.*` files and NFS/quota fails, remove only
  those known crash dumps with `rm -f core.*`.

Activate the B1K environment:

```bash
source cap/third_party/b1k/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
ulimit -c 0
```

## 2. Choose Task And Policy File

Radio:

```text
config: env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml
policy: outputs/interactive/fix_code_interactive_radio.py
```

Soda can:

```text
config: env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml
policy: outputs/interactive/fix_code_interactive.py
```

The soda configs keep the historical `trash` filename, but the target is a blue
can of soda.

## 3. Build Block By Block

Start with a short observation block:

```python
import time
import numpy as np

START_TIME = time.time()
TIME_BUDGET = 900

def time_left():
    return TIME_BUDGET - (time.time() - START_TIME)

rgb, depth = get_env_observation()
save_current_observation("start")
robot_pos, _, robot_yaw = get_robot_position()
print(f"start robot={robot_pos[:2]}, yaw={robot_yaw:.2f}, time_left={time_left():.1f}")
```

Replay it on one seed:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --replay-code outputs/interactive/fix_code_interactive_radio.py \
  --trial 26 \
  --output-dir outputs/behavior/interactive/radio_trial26 \
  --record-video True
```

Then inspect the trial output and append the next 5-20 lines. Repeat until the
policy has search, approach, grasp, verification, and fallback states.

Replay does not call the LLM, so `--server-url` and API keys are not needed.

## 4. Diagnose Every Replay

For each replay, inspect:

- generated `code.py`;
- `summary.txt` stdout/stderr;
- recorded videos;
- `trace.json` and `keyframes/`;
- `differencing_feedback_*.txt` from ASPIRE/VDM runs;
- `prompts_and_responses/` when present;
- saved observations from `save_current_observation(name)`;
- SAM3 or ContactGraspNet server logs when perception fails.

Classify the current blocker as perception, search, navigation, grasp planning,
IK/motion, task sequencing, time budget, or setup.

## 5. Policy Requirements

Every candidate policy should:

- observe and save at start;
- search with prompt alternatives;
- derive approach from `get_robot_position()` and `get_object_pose()`;
- avoid hardcoded scene positions unless they came from the current observation;
- observe after navigation, approach, and grasp;
- print diagnostics for every decision;
- try multiple fallback strategies;
- verify grasps with `check_object_in_hand(arm=0)` and `check_object_in_hand(arm=1)`;
- stay within public R1Pro API calls only.

Do not read simulator internals, BDDL predicates, object registry ground truth,
or reward state from generated task code.

## 6. Useful Commands

Replay radio policy on one seed:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --replay-code outputs/interactive/fix_code_interactive_radio.py \
  --trial 26 \
  --output-dir outputs/behavior/interactive/radio_trial26 \
  --record-video True
```

Replay soda policy on one seed:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_trash_aspire_traced.yaml \
  --replay-code outputs/interactive/fix_code_interactive.py \
  --trial 26 \
  --output-dir outputs/behavior/interactive/soda_trial26 \
  --record-video True
```

Open an interactive REPL for direct API probing:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active scripts/behavior/replay_trial_b1k.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --interactive \
  --trial 26 \
  --output-dir outputs/behavior/interactive/radio_repl26 \
  --record-video True
```

Run exact traced debug seeds when replay is not enough:

```bash
OMNIGIBSON_GPU_ID=2 uv run --no-sync --active python -m aspire.sim.cap.envs.launch_b1k \
  --config-path env_configs/r1pro/r1pro_pick_up_radio_aspire_traced.yaml \
  --trial-ids 26 27 28 \
  --output-dir outputs/behavior/debug/radio_aspire_traced \
  --record-video True
```

## 7. Validate

After the interactive policy works on debug seeds, validate with non-traced
configs:

```text
env_configs/r1pro/r1pro_pick_up_radio_aspire.yaml
env_configs/r1pro/r1pro_pick_up_trash_aspire.yaml
```

For release reporting, record config path, seed range, success count, common
failure mode, videos inspected, and reusable strategy.
