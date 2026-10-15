# Isaac Sim operations

What the simulator does that is not the program's fault.

## Exit codes lie

Kit can terminate the process with status 139 (SIGSEGV) after every artifact was written. A run
with `outcome.json` is complete whatever the exit code says. Judge by `outcome.json`; grep the
log for `[Error]` only when there is no outcome file.

What a healthy log looks like: dozens of `[Error] [omni.kit.app._impl] [py stderr]` lines that
are gymnasium `UserWarning`s (benign), and, at the very end of every *successful* run, `Fatal
Python error: Segmentation fault` followed by a thread dump (Kit's teardown). Judge by
`outcome.json`; in a real crash there is a Python `Traceback` above the `Fatal` banner and no
`outcome.json`.

## One process per GPU

Two Isaac processes on one card OOM or crawl. On a single-GPU bench the coordinator runs
episodes serially; subagents never launch one. Before launching: `pgrep -af 'cap-harness run' | grep -v pgrep`
must show nothing of yours, `free -g` should have 12 GB available, and the GPU should be free of
foreign simulators. Kill leftovers by PID; `pkill -f` from a shell whose command line contains the
pattern kills the shell (exit 144).

## Environment

`OMNIGIBSON_GPU_ID` selects Isaac's GPU (`CUDA_VISIBLE_DEVICES` does not); `OMNIGIBSON_HEADLESS=1`;
`OMNIGIBSON_DATA_PATH` must exist before `import omnigibson`; `OMNIGIBSON_APPDATA_PATH` holds the
shader cache on local disk. `ulimit -c 0`, or a crash writes a multi-gigabyte core file.

## Symptoms

| Symptom | Meaning | Action |
|---|---|---|
| `HydraEngine rtx failed creating scene renderer` | wrong or busy GPU | set `OMNIGIBSON_GPU_ID`, free the card |
| `r1pro is not a registered robot` | robot assets missing | `scripts/bootstrap_behavior.sh --accept-dataset-tos` |
| `AssertionError` on `import omnigibson` about a path | `OMNIGIBSON_DATA_PATH` does not exist | create or fix the path |
| minutes of silence after "app ready" | scene loading or first-launch shader compile | wait up to 5 min (10 on a cold shader cache); check `nvidia-smi` shows the process |
| no `outcome.json`, short log | crash during launch or scene load | check RAM/GPU, retry once; never count it as a task failure |
| `Observation space does not match returned observations` | camera resolution changed after reset | resolution is set at construction; do not change it mid-episode |
| degenerate intrinsics / `fx = 0` | camera parameters read before the first renders | the adapter retries renders; if it persists, report it |

## Base planning and the articulation root

OmniGibson expresses the holonomic base as virtual joints hanging off an articulation root that
only moves while the simulator is stopped; cuRobo clamps those joints to a few metres around the
root, which stays at the world origin. The harness widens that bound to 12 m
(`CAP_HARNESS_BEHAVIOR_BASE_LIMIT_M`) so house-scale scenes are reachable. The symptom of a goal
outside the bound is `planning_failed` on every `navigate_to_pose(..., planner="curobo")` while
`servo` still works (verified 2026-09-12: the upstream ±5 m bound failed every base plan for a
robot starting at (4.6, 6.0)).

## Verified so far

Runtime spike on this harness (2026-09-12, RTX 4090, Isaac Sim 5.1.0, OmniGibson 3.9.2): Kit
ready in 7 s from a warm shader cache; 300 training instances per task. Further rows are added
by campaigns with the seed and run path that produced them.
