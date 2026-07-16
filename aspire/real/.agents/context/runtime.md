# YAM Runtime

Compact runtime notes. Prefer focused skills for details:

- servers: `.agents/skills/yam-server-setup/SKILL.md`
- geometry: `.agents/skills/yam-geometry/SKILL.md`
- artifacts: `.agents/skills/yam-runtime-artifacts/SKILL.md`
- full demo: `.agents/skills/yam-full-demo/SKILL.md`

## Run Shape

Most Real-YAM saved scripts use:

```bash
source .forge_env

YAM_STATION_CALIBRATED_XML=/path/to/calibrated/station.xml \
CAP_TOP_CAMERA_BACKEND=realsense \
CAP_TOP_CAMERA_FRAME=top_camera_d405 \
CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP=0 \
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 \
uv run python run_script.py \
  script_file=cap/saved_scripts/<script>.py \
  skill_library_path=cap/saved_scripts/skill_library \
  env.name=yam-real \
  robot=real_yam \
  robot.dashboard=true \
  robot.await_exit=false \
  robot.go_home_on_exit=false \
  runtime.no_cameras=true \
  recording.enabled=true \
  debug_ui.enabled=true \
  debug_ui.auto_open=false \
  debug_ui.auto_exit_on_run_end=false \
  debug_ui.host=0.0.0.0 \
  debug_ui.port=<port>
```

## Evidence

Physical runs should create `logs/<script>_<timestamp>/` with videos, result
JSON, observations, overlays, and plans. Debug from those artifacts before
changing waypoints.

## Process Check

```bash
pgrep -af 'run_script.py|ffmpeg|cap.debug_ui.app|arm_server|serve_bundlesdf|serve_real_yam_camera_portal|serve_anygrasp|serve_sam3|follower|curobo|pyroki'
```
