# Recorded CaP Runs

`cap-harness run` executes one generated Python program and records an immutable episode.
Operational success is separate from benchmark task success.

```bash
cap-harness run \
  --suite libero_object_swap \
  --task-id 0 \
  --seed 1 \
  --program policy.py
```

The default step limit is 1000. Override it with `--max-steps`; use `--output-root` to move the
output tree. `--model-trace trace.jsonl` imports local model request/response records after secret
redaction. `--flat-run-dir` writes runs as `<output-root>/<seed>/<run-id>` — use it when the
output root is already task-scoped (e.g. the iterative-debugging task directories).
`--no-videos` skips per-camera MP4 capture for observation-only probe runs; keyframes, overlays,
and traces are still recorded, and `run.json`'s capture block records `"videos": false`.

Recorded runs default to 800x512 cameras, matching the pinned upstream LIBERO reference implementation (ASPIRE). Use
`--camera-width` and `--camera-height` to override this. Release validation continues using the
adapter's lightweight defaults because it validates structure rather than visual task completion.

## Artifact contract

```text
outputs/<benchmark>/<suite>/<task-id>-<task-slug>/<seed>/<run-id>/    # default
<output-root>/<seed>/<run-id>/                                        # with --flat-run-dir
```

Seeds use four digits. Run IDs contain a sortable UTC timestamp and a random collision suffix.
Every invocation creates a new directory and never overwrites an earlier run.

- `run.json`: identity, task, capture settings, timestamps, lifecycle status, and an optional
  campaign environment hash.
- `outcome.json`: program status, final task and protocol success, aggregate reward, durations,
  step count, termination reason, and a `timing` block attributing the wall clock (below).
- `manifest.json`: relative path, size, and SHA-256 for every finalized file.
- `source/`: exact generated program, public result, and Git/source provenance.
- `episode/`: reset data and one exact action/robot-state record per simulator tick.
- `evaluation/protocol.json`: optional summarized host-only protocol checks and witness step
  indices; it contains no raw native simulator state.
- `trace/`: parent-linked program and provider spans with typed inputs and outputs.
- `media/videos/`: one H.264 MP4 per camera, named `<camera>.mp4`. A frame is written per
  recorded step at the control rate (20 Hz on LIBERO-Pro and Robosuite; every third 30 Hz step on
  BEHAVIOR-1K, so 10 fps), and `media/video-index.json` records the effective fps.
- `media/keyframes/`: lossless RGB immediately before and after public motion API calls, plus one
  depth NPY attached to the first such keyframe in the run.
- `media/overlays/`: top-1 SAM mask rendered over RGB; raw masks are not retained.

`run.json` records the backend names configured for the program; it does not assert service health.
Program and provider spans record the
`MotionStrategy` or grasp backend requested on each call, including explicit fallback attempts.
The harness never substitutes a backend silently.

Large arrays are deduplicated by content under `payloads/` and referenced from readable JSON by
relative path, dtype, shape, and hash; small arrays are stored inline. Observations retain camera
metadata rather than duplicate RGB or depth arrays. SAM3 retains metadata, scores, and bounding-box
coordinates for its top five results, plus an RGB overlay for the top result; raw masks are not
persisted. Contact-GraspNet collections retain five candidates. Point-cloud artifacts retain only
frame, point count, and color-presence metadata; XYZ and RGB point arrays are not persisted.
Reward, benchmark success, and protocol predicates never appear in generated-program results, API
traces, or per-step records. The runtime writes only final aggregate outcomes and the separate
summarized evaluator evidence.

An active run has `status: in_progress`. JSON is replaced atomically, JSONL is flushed
incrementally, and videos remain hidden `*.partial.mp4` files until closed. A completed run has
`status: finalized`; a process that cannot finalize leaves an inspectable incomplete run.

## Where the time went

`outcome.json` carries a `timing` block: `wall_s`, `attributed_s`, `unattributed_s`, and a
`buckets` map of `{seconds, calls}`. The buckets are disjoint — a region is charged only the time
nothing nested inside it already claimed — so they sum to `attributed_s`, and what no bucket
claims is the generated program's own Python plus harness glue, reported as `unattributed_s`
rather than folded into a neighbour.

| Bucket | What it holds |
|---|---|
| `adapter_step` | Between asking the adapter to step and its answer: the simulator step and its observation normalization. |
| `setup` / `reset` | Environment construction; the reset that follows, less any keyframe or video frame it wrote. |
| `record_step` | The recorder's own per-step work: the `episode/steps.jsonl` record and its arrays. |
| `video_encode` | H.264 frames, including the writer flush at finalize. |
| `keyframe` / `overlay` | PNG (and first-depth NPY) writes. |
| `trace_write` | Span input/output JSON and `trace/events.jsonl`. |
| `provider.<name>` | One backend call, excluding the trace written around it. |
| `finalize` | Closing the run, up to writing `outcome.json`. |

The manifest's SHA-256 pass runs *after* `outcome.json` is written and so is in no bucket. It is
not worth a bucket: SHA-256 runs at ~1.3 GB/s, so a 100 MB run directory hashes in under a tenth
of a second.

Recording the block costs about 4 µs per step — 4 ms on a 1000-step episode.

There is no profiler command; the block is already the answer for one run, and totalling a
campaign is a few lines:

```bash
python3 -c '
import collections, json, sys
from pathlib import Path
total, wall = collections.Counter(), 0.0
for path in Path(sys.argv[1]).rglob("outcome.json"):
    outcome = json.loads(path.read_text())
    wall += outcome["wall_duration_s"]
    total.update({name: b["seconds"] for name, b in outcome["timing"]["buckets"].items()})
total["(program + glue)"] = wall - sum(total.values())
for name, seconds in total.most_common():
    print(f"{name:<20}{seconds:8.1f}s{100 * seconds / wall:6.1f}%")
' outputs/runs/LATEST
```

Runs recorded before this block have no `timing` key and must be skipped by such a loop; their
provider costs can still be recovered from `span_end` durations in `trace/events.jsonl`.

## Sweeping seeds

`cap-harness run-batch` runs one program over many seeds without an interpreter start per seed:

```bash
cap-harness run-batch \
  --suite libero_goal_swap --task-id 0 --seeds 51-65 \
  --program fix_code.py --output-root debug/initial --flat-run-dir \
  --init-mode seeded --workers 4
```

Every seed still builds and closes its own registry, adapter, environment, and recorder, exactly
as `cap-harness run` does, and still lands in its own `<seed>/<run-id>` directory with its own
log under `--log-dir`. A worker reuses the *imports*, never a live scene. Three rules keep a
seed's result independent of what ran before it: workers are spawned rather than forked, so no
CUDA/EGL/MuJoCo state is inherited; each seed re-seeds the global `random` and `numpy.random`
streams from its own seed; and a worker retires after `--recycle-after` seeds (default 10),
bounding any driver-side leak from repeated environment construction.

`--workers N` runs N seeds at once on the assigned GPU. They share that GPU and the perception
services, so the useful setting depends on the box — measure before trusting a number.

`--results-jsonl` gets one record per seed as it finishes, so an interrupted sweep keeps what it
completed; `--summary` gets the aggregate. `run-batch` supports only LIBERO-Pro and Robosuite.
BEHAVIOR must run each episode in a fresh process because closing Isaac ends the runtime.

Before using `--workers` for a campaign, confirm on the target machine that batching changes
nothing the campaign's conclusions rest on:

```bash
python3 scripts/check_batch_equivalence.py \
  --suite libero_goal_swap --task-id 0 --program fix_code.py \
  --gpu 3 --seeds 51-56 --workers 3
```

It runs the seeds unbatched twice and batched once, so the batched-vs-unbatched difference can be
read against the task's own run-to-run variation rather than against an assumption of determinism.

## Termination reasons

`termination_reason` says why execution stopped. It is distinct from `task_success`; a program can
return normally with `program_completed` and still have `task_success: true`.

| Code | Meaning |
|---|---|
| `program_completed` | Generated Python returned normally. |
| `task_succeeded` | The environment ended after benchmark success. |
| `environment_terminated` | The environment terminated without benchmark success. |
| `environment_truncated` | The environment reached its configured horizon. |
| `step_limit` | The run-wide simulator-step limit was reached. |
| `program_error` | Generated Python was rejected or raised an exception. |
| `provider_error` | An unrecoverable provider failure stopped execution. |
| `harness_error` | The adapter, serializer, or recorder failed. |
| `user_interrupt` | The user interrupted execution. |

The earliest stopping condition is retained. Cleanup problems are listed separately in
`finalization_errors` and do not rewrite it.

## Security boundary

Generated code cannot access native MuJoCo state, reward, success predicates, credentials, or
recorder internals. The recorder does not capture process environments, cookies, authorization
headers, or Hugging Face tokens. Imported model traces and serialized mappings redact common
credential keys and bearer/token-shaped text.

## BEHAVIOR-1K differences

Runs for the `behavior` benchmark record `head.mp4`, `left_wrist.mp4` and `right_wrist.mp4` at
every third simulator frame (10 fps at the 30 Hz control rate); `media/video-index.json` reports
the effective fps. `evaluation/protocol.json` carries the pickup witness (held, lifted, the step
both first held, the maximum clearance) and `outcome.json`'s `task_success` is that witness's
verdict, not the BDDL goal. Kit may exit with status 139 after the run directory is complete;
judge a run by `outcome.json`.
