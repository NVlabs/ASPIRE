---
name: iterative-debugging-task-exploration
description: Minimal initial observation step for fix loop tasks. Captures authoritative task language, scene keyframes, and prompt probes while treating all inferred geometry and strategy as provisional.
---

# Initial Scene Analysis

The executable and init mode below are **LIBERO's**; use whatever your task assignment specifies.
Robosuite differs: it runs from `.venv-robosuite` and requires `--init-mode saved`, because
`seeded` is a LIBERO-only mode the harness rejects elsewhere.

Capture development seed 51 with the scene-snapshot program:

**The harness binary and `--init-mode` are per benchmark.** The two below are not
interchangeable: `--init-mode seeded` raises `init_mode 'seeded' is only supported for the
libero-pro benchmark`, and `.venv-libero/bin/cap-harness` is the wrong interpreter for a
Robosuite run. BEHAVIOR uses its separate `behavior/` workflow.

| benchmark | harness | `--init-mode` | cameras |
|---|---|---|---|
| `libero-pro` | `.venv-libero/bin/cap-harness` | `seeded` | `agentview`, `robot0_eye_in_hand` |
| `robosuite` | `.venv-robosuite/bin/cap-harness` | `saved` | `agentview`, `robot0_eye_in_hand` |

```bash
TASK_DIR="$RUN_ROOT/$BENCHMARK/$SUITE/task_$TASK_ID"
mkdir -p "$TASK_DIR/debug/explore"
# CAP_HARNESS and INIT_MODE come from this benchmark's profile.
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" \
"$CAP_HARNESS" run \
  --benchmark "$BENCHMARK" --suite "$SUITE" --task-id "$TASK_ID" --seed 51 \
  --program .claude/skills/iterative-debugging/scripts/scene_snapshot.py \
  --output-root "$TASK_DIR/debug/explore" --flat-run-dir --no-videos --init-mode "$INIT_MODE"
```

`scene_snapshot.py` probes `segment_text("agentview", ...)`. Use the camera names reported by
your assigned embodiment; copy the probe into `$TASK_DIR/code/` before changing it.

The command prints `run: <run_dir>`. Read from that run:

- `source/program-result.json` — the authoritative `task_language`, camera names, and task
  metadata. For LIBERO `_task` suites, task language overrides the task name.
- `media/keyframes/` — lossless RGB of the untouched scene from every camera (captured around
  the program's benign gripper toggle).
- To validate SAM3 prompts, copy `scene_snapshot.py` into `$TASK_DIR/code/` as
  `probe_<slug>.py`, fill `PROBE_PROMPTS` after inspecting the keyframes, re-run with the same
  `--output-root "$TASK_DIR/debug/explore" --flat-run-dir --no-videos` and your benchmark's
  `--init-mode`, and read the probe
  scores in `source/program-result.json` plus the rendered masks in `media/overlays/`.

Write only a short `$TASK_DIR/task_analysis.md`:

```markdown
# Initial Task Analysis

- Actual task language:
- Manipulated object and likely prompts:
- Goal/handle and likely prompts:
- Visible obstacles and plausible approach:
- Interaction type: pick/place | push/contact | articulated | multi-stage
- Uncertainties to verify from later seeds:
```

## Triage from disk, not from a `run:` log line

The `run: <dir>` line that `cap-harness` prints may be buried in a benchmark's own log noise.
A triage loop built on
`grep -oP '(?<=^run: ).*'` then prints nothing at all, which is indistinguishable from "no seed
completed" and can make you re-run a sweep that already succeeded.

Find the outcomes on disk instead — this works on every benchmark:

```bash
find "$TASK_DIR/debug/initial" -name outcome.json | sort | while read f; do
  seed=$(echo "$f" | grep -oP '/\d{4}/' | tr -d '/')
  python3 -c "
import json
o=json.load(open('$f'))
print('$seed', o.get('program_ok'), o.get('task_success'), o.get('termination_reason'))
"
done
```

A run directory is simply the parent of its `outcome.json`; `trace/events.jsonl`,
`media/keyframes/`, `media/overlays/` and `episode/steps.jsonl` sit beside it.

## Wrap every run in `timeout` — orphans starve the GPU

If a loop-batched shell call hits the Bash tool's own timeout mid-loop, the running
`cap-harness` / simulator child is **not** reliably killed. It keeps running and holding several
GB of GPU memory, and enough orphans starve the SAM3 / cuRobo providers into `CUDA out of memory`
— which surfaces as a *perception* failure ("fewer than N objects detected") in every later run,
sending you to debug a prompt that was never the problem.

Wrap each invocation in its own `timeout` rather than relying on the outer tool timeout, and check
between batches:

```bash
timeout 600 <cap-harness run ...> > "$LOG" 2>&1
ps aux | grep -c "[c]ap-harness run"        # expect 0 between batches
```

If perception suddenly fails across seeds that used to work, check
`validation-artifacts/service-logs/sam3.log` for `torch.OutOfMemoryError` before touching prompts.

## Sandbox constraints when writing probes

The sandbox allowlist (`runtime.py:_SAFE_BUILTINS`) is exactly:

```
Exception ValueError TypeError abs all any bool dict enumerate float int len list
max min print range reversed round set sorted str sum tuple zip
```

Three traps follow from that list, each of which costs a full episode startup to discover:

- **`RuntimeError` does not exist.** `raise RuntimeError(...)` dies with `NameError`. Use
  `Exception`, `ValueError`, or `TypeError`. (This one crashed every seed of one task's first
  sweep.)
- **`getattr` / `setattr` / `hasattr` do not exist.** Access attributes directly and wrap
  genuinely optional ones in `try/except`.
- **No name may start with an underscore.** `for _ in range(5)` is rejected with
  `private and dunder names are not allowed` (`runtime.py:175`). Name the throwaway variable,
  e.g. `for settle_tick in range(5)`.

Access attributes directly and wrap optional ones in `try/except`:

```python
ok = bool(plan.ok)                      # direct access
try:
    status = str(result.status)         # optional field
except Exception:
    status = "unknown"
```

Imports, filesystem, and network are unavailable too — helpers must be pure Python over the
public contracts.

This analysis comes from one scene and may be flawed. Treat object identity, relative position,
dimensions, free space, and motion strategy as hypotheses. Do not encode snapshot-specific pixel
ordering or coordinates. Revise the analysis when multi-seed traces or keyframes disagree.
