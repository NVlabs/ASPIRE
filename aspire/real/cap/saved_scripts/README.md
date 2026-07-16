# Saved Scripts

Canonical demo entrypoints:

| Task | Command |
| --- | --- |
| Full demo | `bash cap/saved_scripts/yam_demo.sh full` |
| Bowls only | `bash cap/saved_scripts/yam_demo.sh bowls` |
| Bottle rack only | `bash cap/saved_scripts/yam_demo.sh bottle-rack` |
| KitKat handover/trash only | `bash cap/saved_scripts/yam_demo.sh kitkat` |
| Drawer pull/push | `bash cap/saved_scripts/yam_demo.sh drawer` |
| Drawer close only | `bash cap/saved_scripts/yam_demo.sh drawer-close` |
| Drawer candy place | `bash cap/saved_scripts/yam_demo.sh drawer-candy` |
| Home/open grippers | `bash cap/saved_scripts/yam_demo.sh home` |

Before using these commands on a newly configured station, run:

```bash
bash tools/yam_demo_preflight.sh
bash tmux/launch_yam_demo_services.sh --no-attach
bash tools/yam_demo_preflight.sh --services
```

The minimal launcher owns both follower arm servers, the read-only camera
Portal, SAM3, and BundleSDF. Attach with `tmux attach -t yam_demo` to inspect
service logs.

`yam_demo.sh full` runs drawer-close, home, white dish, orange-on-white,
can-trash, KitKat, then bottle rack. Helper shell scripts live in
`shell_scripts/`; the active Python task implementations live at the top level
of this directory.

Use `YAM_FULL_DEMO_SPEED_SCALE` as the normal single speed knob for saved
demo commands. `1.0` keeps the validated defaults, values above `1.0` speed up
planned motion and shorten direct step durations, and values below `1.0` slow
them down:

```bash
YAM_FULL_DEMO_SPEED_SCALE=1.2 bash cap/saved_scripts/yam_demo.sh full
```

The scale currently applies to home, drawer planned motion, rack dish/bowl
planning and insertion/retreat timings, can/trash planning, KitKat planning,
and bottle-rack planning. Low-level `OPENFORGE_*_PLANNING_SPEED`,
`*_STEP_S`, `*_PLAYBACK_SPEED`, and `OPENFORGE_OPEN_HOME_SPEED_RAD_S`
overrides still work for focused debugging.

`legacy_codes/` contains older probes, sweeps, alternate task scripts, and
failed-history scripts that are not part of the current full sequential demo.

## Drawer Practice

The top-drawer workflow is wired into `yam_demo.sh drawer`. It uses one Python
entrypoint, `drawer_observe_pull_open_x.py`, which observes the handle, runs
the left-arm manual pull/push cycle, and observes again in a single
`logs/<run>/` directory. The shell wrapper is only a thin compatibility
launcher; drawer defaults live in the Python file.

Run these commands from the repo root.

### Common Environment

```bash
cd aspire/real
source .forge_env

export YAM_STATION_CALIBRATED_XML=/path/to/calibrated/station.xml
export CAP_TOP_CAMERA_BACKEND=realsense
export CAP_TOP_CAMERA_FRAME=top_camera_d405
export CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP=0
export OPENFORGE_PREVIEW_RECORDER_BACKEND=python
export OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S=8.0
export OPENFORGE_PREVIEW_RECORDER_REENCODE_H264=1
export OPENFORGE_PREVIEW_RECORDER_REQUIRE_H264=1
```

### Observe Then Pull/Push

Use this wrapper for the current top-drawer work. It launches the consolidated
Python script, which records pre-observe, manual motion, and post-observe
artifacts under one run directory. The default
`OPENFORGE_DRAWER_CYCLE_MODE=open_then_close` strategy is: approach the handle,
close in stages, pull roughly `-X`, push back pull-minus-1cm, open at the
closed pose, then retreat. Current conservative defaults are
`OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M=-0.012` for 1.2cm behind the handle,
`OPENFORGE_DRAWER_MANUAL_PULL_M=0.105`, and
`OPENFORGE_DRAWER_MANUAL_PUSH_M=0.095` from pull minus
`OPENFORGE_DRAWER_MANUAL_PUSH_LESS_THAN_PULL_M=0.010`, with no default Y
bias from the detected handle, target close
`OPENFORGE_DRAWER_MANUAL_TARGET_GRIPPER=0.10`, planning speed
`OPENFORGE_DRAWER_MANUAL_PLANNING_SPEED=0.4 * YAM_FULL_DEMO_SPEED_SCALE`, and
adaptive close-push disabled unless `OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_M=1`.

```bash
bash cap/saved_scripts/yam_demo.sh drawer
```

The drawer launchers enable both the robot dashboard and debug UI by default;
the debug UI auto-opens and remains available after the run for inspecting live
motion primitives and artifacts.

The current conservative pull/push distance can be made explicit while testing:

```bash
OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M=-0.012 \
OPENFORGE_DRAWER_MANUAL_PULL_M=0.105 \
OPENFORGE_DRAWER_MANUAL_PUSH_M=0.095 \
  bash cap/saved_scripts/yam_demo.sh drawer
```

Set `OPENFORGE_DRAWER_CYCLE_MODE=open_only` to preserve the previous pull-open
behavior with no push-back phase. Set `OPENFORGE_DRAWER_MANUAL_EXECUTE=0` to do
the same observe and planning path without physical contact. Override the
`OPENFORGE_DRAWER_MANUAL_*` envs before launching when testing a different
axis, offset, gripper width, or RPY. Set `OPENFORGE_DRAWER_POST_OBSERVE=0` only
when intentionally skipping final state evidence.

### Close Open Drawer

Use this when the top drawer is already open and you only want the left arm to
grasp the handle, push it closed, open, and retreat. It reuses the same handle
observation, staged gripper close, and manual push primitive as the full drawer
script, but forces `OPENFORGE_DRAWER_MANUAL_PULL_M=0.000`. The wrapper runs
`yam_demo.sh home` after the close motion by default; set
`OPENFORGE_DRAWER_CLOSE_SKIP_HOME=1` only when intentionally leaving the arm
near the drawer.

```bash
bash cap/saved_scripts/yam_demo.sh drawer-close
```

Default close-only push is `OPENFORGE_DRAWER_CLOSE_ONLY_PUSH_M=0.095`, derived
from current pull `0.105` minus `0.010`. Override it explicitly while tuning:

```bash
OPENFORGE_DRAWER_CLOSE_ONLY_PUSH_M=0.105 \
  bash cap/saved_scripts/yam_demo.sh drawer-close
```

Direct Python launcher equivalent:

```bash
source .forge_env
uv run python cap/saved_scripts/drawer_observe_pull_open_x.py
```

### Drawer Candy Place

This wrapper composes the verified drawer open/close primitive with a top-down
yellow candy-bar pickup. The intended sequence is: open the drawer, open the
gripper off the handle, retreat another `-X` by
`OPENFORGE_DRAWER_CANDY_OPEN_CLEAR_AFTER_PULL_M=0.150`, observe the candy, pick
top-down, lift and verify the candy moved from its source, transport to a
bounded place pose inside the drawer, release, return to the drawer-open retreat
pose, then close the drawer.

```bash
bash cap/saved_scripts/yam_demo.sh drawer-candy
```

Direct shell wrapper equivalent:

```bash
bash cap/saved_scripts/shell_scripts/drawer_candy_bar_place.sh
```

Previous best Y-axis pull evidence, kept for comparison:
`logs/drawer_manual_probe_20260609T030810`.

### Recover Home/Open

Run this after drawer contact runs. A drawer contact run can leave the left
gripper only partially open near the handle; this command fully opens and parks
both arms.

```bash
export OPENFORGE_ALLOW_PHYSICAL_MOTION=1

uv run python run_script.py \
  script_file=cap/saved_scripts/open_grippers_return_home.py \
  skill_library_path=cap/saved_scripts/skill_library \
  env.name=yam-real robot=real_yam \
  robot.dashboard=false robot.await_exit=false robot.go_home_on_exit=false \
  runtime.no_cameras=true recording.enabled=true \
  debug_ui.enabled=true debug_ui.auto_open=false \
  debug_ui.auto_exit_on_run_end=false debug_ui.host=0.0.0.0
```
