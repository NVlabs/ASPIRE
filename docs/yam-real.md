# Real YAM

The real-YAM adapter exposes the shared CAP API for two six-joint arms. Motion,
including home and gripper commands, goes through `RealYamEnv.execute_action_batch`.
Arm servers own CAN; the client runtime owns the cameras and recording.

## Install and configure

```bash
scripts/bootstrap_yam_real.sh
```

The `yam-real` extra installs the client, camera, recording, and local Mink IK
dependencies. The separate `yam-real-server` extra adds the motor driver. Model
services use the existing provider environments; YAM does not require PyRoki.

Use your station's camera serials, arm endpoints, home pose, limits, gripper
measurements, and calibration. Place custom profiles in a directory passed with
`--station-config-root`. Each station directory contains `station.yaml` and its
calibration bundle. The loader verifies the bundle hashes; update the manifest
and its hash when preparing a new calibration. Robot models are package-relative.

The packaged reference profile is
[`yam-example`](../src/cap_harness/configs/yam_real/yam-example/station.yaml),
which is also the default station ID. For camera capture and calibration, follow
the [calibration workflow](../src/cap_harness/yam_real/calibration/README.md).

The default deployment runs the arm services, camera client, and model services
on the same machine. Install both YAM extras and run one server per arm, with
SocketCAN and USB access already configured. Replace the placeholders and set
these variables in each terminal:

```bash
export ASPIRE_STATION="<station-id>"
export ASPIRE_CONFIG_ROOT="/absolute/path/to/station-profiles"
```

`--station` selects the profile whose `station` field matches that ID. The loader
searches `*/station.yaml` under the configuration root; directory names may
differ. It also accepts `<station-id>.yaml` directly under that root. The profile
and calibration manifest must declare the same ID. Without a custom root, it
searches the package's `configs/yam_real` directory.

Use `127.0.0.1` for both arm hosts and distinct ports. In separate terminals:

```bash
yam-servers left --station "$ASPIRE_STATION" --config-root "$ASPIRE_CONFIG_ROOT"
yam-servers right --station "$ASPIRE_STATION" --config-root "$ASPIRE_CONFIG_ROOT"
```

Starting an arm server energizes its motors. The CAP runner connects to existing
servers; it does not start them. Closing the client does not stop the servers.

## Run

```bash
.venv-yam-real/bin/cap-harness run \
  --benchmark yam_real --station "$ASPIRE_STATION" \
  --station-config-root "$ASPIRE_CONFIG_ROOT" --suite observe_station --task-id 0 \
  --program examples/yam_real/observe_station.py --output-root outputs
```

Without `--allow-motion`, observations work and physical commands return a
`safety_interlock` error before actuation. Add it when running the home or joint
motion example. An authorized run homes during adapter reset.
`--max-steps` counts submitted motions, not servo ticks.

Use physical arm names `left`/`right` or the profile's `primary`/`secondary`
aliases. Gripper positions are normalized: `0` closed, `1` open. The calibrated
camera provides metric RGB-D; uncalibrated auxiliary views are for visual
inspection only.

For Cartesian motion, select the YAM solver explicitly:

```python
strategy = MotionStrategy(ik_solver="mink", trajectory_planner="interpolation")
plan = plan_motion(target, arm="right", strategy=strategy)
```

Collision-aware planning uses the existing cuRobo service with the packaged
dual-arm YAM description. Collision checking stays enabled by default. Planner
refusals remain typed failures. No physical planning performance is implied by
the off-robot checks.

## Recording and off-robot checks

Runs retain the normal CAP trace and a clock-sampled episode under `episode/raw`.
Video uses the station control rate and goes under `media/videos`. `--no-videos`
disables video in both recorders. Sampling stops before the run manifest is sealed.

`build_sim_station()` substitutes a MuJoCo plant and synthetic cameras under the
same `RealYamEnv` and adapter. It is a control/packaging test fixture, not a visual
manipulation benchmark. Fake-arm tests cover motion refusal and transport errors.
