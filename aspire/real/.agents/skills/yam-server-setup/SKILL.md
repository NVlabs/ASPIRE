---
name: yam-server-setup
description: "Use for YAM arm, camera, perception, AnyGrasp, BundleSDF, SAM3, cuRobo, PyRoki, and provider server checks/restarts."
---

# YAM Server Setup

Run these checks and launch commands from the Aspire real-robot workspace
(`aspire/real`). Paths below are relative to that directory.

Process check:

```bash
pgrep -af 'run_script.py|ffmpeg|cap.debug_ui.app|arm_server|serve_bundlesdf|serve_real_yam_camera_portal|serve_anygrasp|serve_sam3|follower|curobo|pyroki'
```

Process presence is only liveness. Before physical demos, verify readiness.

Health/readiness check:

```bash
curl -fsS --max-time 3 http://127.0.0.1:6767/health   # SAM3
curl -fsS --max-time 3 http://127.0.0.1:8119/health   # BundleSDF
curl -fsS --max-time 3 http://127.0.0.1:9600/health   # PyRoki
curl -fsS --max-time 3 http://127.0.0.1:8765/health   # provider
curl -fsS --max-time 3 http://127.0.0.1:8122/health   # AnyGrasp if needed
```

Portal services are not HTTP health servers; plain `curl` may reset. Check them
through Portal RPC:

```bash
uv run python -c "import portal; print(portal.Client('127.0.0.1:8611').health_check().result(timeout=5))"
uv run python -c "import portal, numpy as np; c=portal.Client('127.0.0.1:8300'); print(c.health().result(timeout=3)); [print(cam, np.asarray(c.get_camera_image(cam).result(timeout=3)).shape) for cam in ['top','left','right','bottom']]"
```

Arm servers must be healthy, not just connectable. `connected=True` is
insufficient if the send loop died. Do not run physical motion unless each arm
has `send_thread_alive=True` and `background_error=None`:

```bash
uv run python -c "import portal, sys; ok=True
for side,port in [('left',11333),('right',11334)]:
    h=portal.Client(f'127.0.0.1:{port}').get_health().result(timeout=5)
    print(side, h)
    ok = ok and bool(h.get('connected')) and bool(h.get('send_thread_alive')) and h.get('background_error') is None
sys.exit(0 if ok else 2)"
```

BundleSDF and recorder readiness are required before physical debugging runs.
Do not start a physical run without video evidence unless the task is explicitly
about fixing recording. The command may return success while the run is
undebuggable if preview videos are corrupt or missing. Always use the Python
preview recorder backend for real runs unless testing the recorder itself:

```bash
export OPENFORGE_PREVIEW_RECORDER_BACKEND=python
export OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S=8.0
export OPENFORGE_PREVIEW_RECORDER_REENCODE_H264=1
```

Check BundleSDF health and preview-camera availability:

```bash
curl -fsS --max-time 3 http://127.0.0.1:8119/health
```

The `/health` response should be `status=ok` and should list usable preview
cameras such as `top`, `left`, `right`, and `bottom`. If BundleSDF is unhealthy
or previews are unavailable, fix/restart it before physical motion.

Probe BundleSDF previews before launching long physical runs:

```bash
OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S=8.0 uv run python -c "from cap.agent.recorder import PreviewStreamRecorder; from pathlib import Path; out=Path('/tmp/yam_preview_probe'); out.mkdir(exist_ok=True); r=PreviewStreamRecorder('http://127.0.0.1:8119',['top','left','right','bottom'],out); r._preflight_preview_streams(); print('preview_preflight_ok')"
```

After every real run, inspect `logs/<run>/preview_recording_result.json`.
Accept the videos only when each needed camera has `ok=true`,
`ffprobe.ok=true`, nonzero `duration_s`, nonzero `nb_frames`, and
`backend="python"`. Current preview MP4s should also report
`ffprobe.codec_name="h264"` and `ffprobe.pix_fmt="yuv420p"`. The Python
preview recorder writes a temporary OpenCV MP4, preserves it as
`<camera>.pre_h264.mp4`, and makes `<camera>.mp4` the H.264 evidence file by
default. If MP4s are tiny, use `mp4v`/non-H.264 unexpectedly, or ffprobe
reports `moov atom not found`, treat the run as missing video evidence even if
robot motion happened.

Mock AnyGrasp may satisfy integration health checks but is not safe for
physical grasp selection. If `/health` says `mock=true` or
`safe_for_robot_motion=false`, only run tasks that do not use AnyGrasp for
physical planning.

Arm servers from the repo root:

```bash
source .forge_env
uv run python robot/yam/arm_server.py --mode follower --side left   # :11333
uv run python robot/yam/arm_server.py --mode follower --side right  # :11334
```

Canonical saved-demo launcher:

```bash
bash tools/yam_demo_preflight.sh
bash tmux/launch_yam_demo_services.sh --no-attach
bash tools/yam_demo_preflight.sh --services
```

It starts both follower arms, the camera Portal `:8300`, SAM3 `:6767`, and
BundleSDF `:8119`. These are the services used by `yam_demo.sh full`.

The broader development launcher is:

```bash
bash tmux/launch_realworld_localserver_realsense.sh
```

It additionally starts AnyGrasp, cuRobo, PyRoki, and optional provider panes.
Those additional services are not required by the canonical saved demo.

Common real-run env:

```bash
YAM_STATION_CALIBRATED_XML=/path/to/calibrated/station.xml
CAP_TOP_CAMERA_BACKEND=realsense
CAP_TOP_CAMERA_FRAME=top_camera_d405
CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP=0
OPENFORGE_ALLOW_PHYSICAL_MOTION=1
```

AnyGrasp is only needed for scripts that explicitly use it. Mock/synthetic
AnyGrasp is not safe for physical grasp selection.
