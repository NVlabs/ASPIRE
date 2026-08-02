# BEHAVIOR-1K clean-clone acceptance — 2026-07-31

## Verdict

PASS. The documented BEHAVIOR setup completed from an empty software and data
state on its first and only invocation, and the full verifier produced a
successful soda-can oracle trial.

## Scope

- Simulation only; no `aspire/real` code or physical hardware was accessed.
- ASPIRE base: `2c440bb8ff2274ad7e069c2488ab5dcf0d413e50`.
- B1K: `272ec5ca9936453c4a8fd335c4dfba61245e33ca`.
- Host: Ubuntu 22.04, driver 580.95.05, 4× NVIDIA L40.
- Fresh state: no B1K venv, output directory, dedicated uv cache, or downloaded
  datasets. The robot-assets tree contained only the git-tracked
  `r1pro_ik.urdf` overlay plus its README and `.gitignore`.

## Command

```bash
cd aspire/sim
export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"
export UV_CACHE_DIR=/root/uv-cache-aspire-final-clean-validation-20260731

scripts/setup_behavior.sh --accept-dataset-license --gpu-id 2
```

The setup command ran exactly once. Its true exit status was captured with
plain redirection rather than a pipeline:

- Start: `2026-07-31T05:40:41Z`
- End: `2026-07-31T06:12:16Z`
- Exit status: `0`
- Elapsed: 1,895 seconds

## Cold-install evidence

- Created a new Python 3.10 environment and installed Isaac Sim 4.5 /
  OmniGibson 3.7.2.
- Built cuRobo CUDA extensions for the L40's `sm_89` target from a dedicated
  empty uv cache.
- Expanded `omnigibson-robot-assets` from the overlay-only stub to 2,816 files
  / approximately 2.4 GB.
- Verified `models/r1pro/usd/r1pro.usda` exists and is 131,100,140 bytes.
- Final data footprint matched the expected approximately 33 GB BEHAVIOR
  assets, 2.4 GB robot assets, and 400 MB challenge instances.

## Full verifier result

`scripts/verify_behavior.py --gpu-id 2` passed:

- GPU, driver, Torch/CUDA, NumPy, setuptools, and OpenCV checks.
- Isaac Sim, OmniGibson, cuRobo, SAM3, Contact-GraspNet, and ASPIRE imports.
- Dataset payload checks, including both the tracked R1Pro IK overlay and the
  real R1Pro USD runtime payload.
- SAM3 and Contact-GraspNet startup on real loopback sockets.
- Soda-can oracle trial on attempt 1; the permitted initialization retry was
  not needed.

The passing trial directory was:

```text
outputs/behavior/verify/google_gemini-3.1-pro-preview/attempt_1/
  trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/
```

`summary.txt` records `Reward: 1`, `Task Completed: True`, and zero
regenerations. Four non-empty videos were independently decoded; each contains
64.3 seconds of frames from RGB, ego, left-wrist, and right-wrist cameras.

## Known teardown behavior

Isaac Sim returned `-11` during teardown after all success artifacts were
written. The verifier recorded this in `environment_manifest.json`, then
independently confirmed the reward/task-completed markers and four videos
before passing. This is the documented upstream post-artifact teardown issue,
not a task failure.

## Repository integrity

- No ASPIRE source, script, documentation, or configuration file was edited
  during the acceptance run.
- The B1K, SAM3, and cuRobo submodules remained clean and pinned.
- Contact-GraspNet carried only the expected three-file compatibility patch;
  its gitlink did not change.
- Nothing was committed or pushed by the validator.
