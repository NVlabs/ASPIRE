# YAM camera calibration

Calibrate one RealSense camera at a time using a ChArUco board and the existing
ASPIRE arm service. Adapted from ENPIRE's Apache-2.0 YAM calibration pipeline
(revision `99ee90acf65b5b18957c8382ad580db999528be3`). Everything added lives in this
directory; it uses the robot model already shipped with ASPIRE.

## Install

From the repository root, create a separate environment for the OpenCV desktop
preview (requires a graphical desktop):

```bash
uv venv --python 3.12 .venv-yam-calibration
uv pip install --python .venv-yam-calibration/bin/python \
  -r src/cap_harness/yam_real/calibration/requirements.txt
uv pip install --python .venv-yam-calibration/bin/python --no-deps -e .
source .venv-yam-calibration/bin/activate
```

## Prepare the board

```bash
python -m cap_harness.yam_real.calibration board --output board.png
```

Print at **actual size**: five squares per side, 40 mm squares, 30 mm markers,
plus a white border. Measure the printed squares and mount the board flat and
rigid. For different dimensions, pass matching `--square-length` and
`--marker-length` values in metres to both `board` and `calibrate`.

| Camera mounting | Mode | Board mounting |
| --- | --- | --- |
| Fixed to the station | `fixed` | Rigidly attached to the selected arm |
| On the wrist | `wrist` | Stationary on the table or another fixed support |

## Capture and solve

Keep the selected arm's ASPIRE follower service running. Stop the agent/MCP
session and other camera or arm clients before calibration. Use the camera's
RealSense serial (or its `/dev/video_*` alias). Select the same color resolution
you intend to use afterward.

Fixed camera example:

```bash
python -m cap_harness.yam_real.calibration calibrate \
  --serial CAMERA_SERIAL --camera-name top --mode fixed --arm left \
  --resolution 1280 720 --output-dir outputs/calibration/top --confirm-motion
```

The command connects to localhost port 11333 for the left arm or 11334 for the
right arm; override with `--host` and `--port`. `--model` selects the MuJoCo XML
used for arm forward kinematics and export; the default is the packaged
`yam_real/description/station/station.xml`. Its arm geometry and mounting must
match the physical station.

1. With gravity compensation active, guide the arm until the board is visible.
2. Press **Space** in the preview to hold this nominal pose.
3. Review the printed sweep targets; press **Space** again to move through them.
   The 15-pose sweep checks joint limits and moves at 0.2 rad/s. Choose a clear
   nominal pose: this calibration sweep has no obstacle planner.
4. The arm returns to the nominal pose, and the solver writes the results.
   **Q**, **Esc**, or **Ctrl+C** aborts and requests a hold at the measured pose.
   The gripper target stays fixed throughout capture.

Wrist camera example, retaining the previous camera calibration in the XML:

```bash
python -m cap_harness.yam_real.calibration calibrate \
  --serial WRIST_CAMERA_SERIAL --camera-name wrist_left --mode wrist --arm left \
  --model outputs/calibration/top/station_calibrated.xml \
  --output-dir outputs/calibration/wrist_left --confirm-motion
```

Repeat for each installed camera; there is no fixed camera count. `--camera-body`
can select an existing XML body. Defaults are `top_camera_d405` for `top`,
`left_camera_d405`/`right_camera_d405` for wrist cameras, and `<camera-name>_camera`
for other fixed cameras.

## Results

Each output directory contains `samples.json`, captured PNGs, `calibration.json`,
and `station_calibrated.xml` with its mesh assets. The source model is preserved.
At least 12 usable samples with rotation about multiple axes are required.
Solutions must have consistency RMS at most 10 mm and 3 degrees; inspect these
residuals before using the result. Intrinsics and distortion come from the
RealSense factory calibration; this workflow solves **camera extrinsics**.

`solution.T_parent_from_camera` maps OpenCV optical coordinates (+Z forward,
+Y down) to world for a fixed camera, or to the selected arm's `link_6` for a wrist
camera. The XML applies the optical-to-MuJoCo camera-axis conversion. The full
intrinsics and distortion are retained in JSON; XML uses the vertical field of view.

Re-solve saved samples without hardware:

```bash
python -m cap_harness.yam_real.calibration solve \
  outputs/calibration/top/samples.json --output-dir outputs/calibration/top-resolved
```

Output is calibration data, not an installed station profile. Selecting these
results for the existing runtime remains a separate station configuration step.

Offline checks:

```bash
uv pip install --python .venv-yam-calibration/bin/python pytest
python -m pytest src/cap_harness/yam_real/calibration/test_pipeline.py -q
```
