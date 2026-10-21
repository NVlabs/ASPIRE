# Cerulion runtime target

`cerulion` is a runtime target rather than a simulator: the adapter drives a
robot whose software runs on [Cerulion](https://github.com/cerulion-inc/cerulion),
an open-source robot runtime, through `cerulion-rcpd`, the runtime's MCP daemon,
over stdio. The robot is whatever the Cerulion graph behind the daemon runs; any
graph whose manifest serves `arm.reset` and `arm.step` with the same arguments
as Cerulion's `examples/aspire_arm/rcp.yaml` is reachable without changing the adapter.

The task shipped here, `pick_cube`, uses Cerulion's example simulated arm: a
six-joint arm with a gripper and one cube on a table, the `arm_sim` node in
Cerulion's `examples/aspire_arm` graph.

## What a program gets

- **No perception.** No segmentation, grasp, IK or cuRobo provider is
  configured for this target, and the camera is not a sensor. The cube's
  position is supplied by the task in
  `get_task_context().metadata["task_supplied_cube_position"]`, and
  `TaskContext.language` says so.
- **Camera `workspace`**: a synthetic drawing of the plant's reported state, a
  top view beside a side view. Depth is all zeros and the intrinsics are
  placeholders; it is for watching a run, not for geometry.
- **Kinematics** in `metadata["kinematics"]` (`H`, `L1`, `L2`, `L3`): the pitch
  joints move one planar chain in the vertical plane at the base yaw, so a
  program can solve its own IK. `examples/cerulion/pick_cube.py` does this with
  the tool pointing straight down.
- **Control**: one arm, `primary`, joint-position commands at 20 Hz; each joint
  moves at most 0.05 rad per tick and the gripper at most 0.25 of its travel.
  The gripper is 1.0 open and 0.0 closed, as everywhere in ASPIRE.
- Task success is the plant's own flag (cube grasped and at least 15 cm up). The
  harness reads it at the end of the run; programs never see it.

## Refusals

`cerulion-rcpd` checks every call against the schemas in its manifest before
anything reaches the graph, and writes a receipt for each call. Its refusals
come back as typed `ApiError`s:

| daemon code | `ErrorCode` | recoverable |
|---|---|---|
| `invalid_args`, `args_too_large` | `INVALID_REQUEST` | yes |
| `estopped` | `SAFETY_INTERLOCK` | no |
| `timeout` (`may_have_executed` in details; the cached state becomes stale) | `TIMEOUT` | no |
| `not_in_manifest`, `publish_failed`, `journal_stopped`, `receipt_failed` | `ADAPTER_FAILED` | no |

After an e-stop the daemon refuses every motion call, `arm.reset` included,
until an operator deletes `<receipts>.estop` and restarts it. `reset()` refuses
up front when that latch file exists.

## Run it

Install ASPIRE with the `cerulion` extra (`pip install -e '.[cerulion]'`; it
needs `mcp` and the imageio ffmpeg backend) and put `cerulion` and
`cerulion-rcpd` on `PATH`. The graph must be running before the episode
starts. `examples/aspire_arm/run_demo.sh` in the
[Cerulion repository](https://github.com/cerulion-inc/cerulion) does the whole
sequence: it builds the plant, starts the graph with recording on, runs the
episode below, stops the graph, and replays the recording (next section).

```bash
cap-harness run --benchmark cerulion --suite cerulion --task-id 0 --seed 0 \
  --program examples/cerulion/pick_cube.py \
  --rcpd-manifest /path/to/cerulion/examples/aspire_arm/rcp.yaml \
  --rcpd-receipts outputs/cerulion/receipts.jsonl \
  --output-root outputs/examples --camera-width 256 --camera-height 128
```

`CAP_HARNESS_CERULION_RCPD_CMD` replaces the daemon command line entirely.

The same arm can be served to a coding agent through ASPIRE's own MCP tools:

```bash
cap-harness-agent --embodiment cerulion --rcpd-manifest /path/to/rcp.yaml \
  --rcpd-receipts outputs/cerulion/receipts.jsonl --seed 0
```

## Replay on the Cerulion side

Every tool call is published into the graph as an external input, so a
`cerulion graph run --record` recording holds the episode's calls. On the
Cerulion side, `cerulion bag play <bag> --resim all --verify` re-injects them,
recomputes the graph, and compares its answers with the recorded ones byte for
byte. That replay needs no model, no ASPIRE process and no daemon.
