# Blue Gripper Geometry Notes

Use this reference for YAM plate/dish rim-pinch planning. These are validated
facts from RGB-D, robot state, camera calibration, and the current MuJoCo XML;
do not treat SAM3 masks as primary metric geometry.

## Evidence

- Main RGB-D artifact:
  `logs/analyze_gripper_geometry_rgbd_20260529T135559`.
- Earlier RGB-D artifact:
  `logs/analyze_gripper_geometry_rgbd_20260529T133859`.
- Physical free-space gripper presentation/calibration:
  `logs/calibrate_gripper_geometry_top_camera_20260529T130455`.
- Current calibrated XML is selected by `YAM_STATION_CALIBRATED_XML`; keep the
  workstation-specific path in `.forge_env`.

The 13:55 RGB-D run saved:

- `runtime_api_surface.json`: actual injected `run_script.py` tools plus camera
  portal availability.
- `task_result.json`: full analysis packet.
- `observations/rgbd_{top,left,right}/rgb.png`, `depth_m.npy`,
  `depth_preview.png`, `blue_mask.png`, `blue_components_overlay.png`, and
  `analysis.json`.
- `top.mp4`, `left.mp4`, and `right.mp4`.

## Runtime API Surface

The saved script saw 17 injected run_script tools. Relevant callable names were:

- `get_robot_state()`.
- `get_camera_image(camera)`.
- `detect_objects_oneshot(...)`.
- `freespace_move(..., preview_only=True)`.
- `run_in_background(...)`.
- motion-capable tools `set_gripper`, `open_gripper`, `close_gripper`,
  `go_home`.

Metric RGB-D was available through the camera portal at `127.0.0.1:8300`, not
as separate injected run_script globals. Verified portal methods:

- `get_camera_image(top|left|right)`: RGB arrays.
- `get_camera_depth(top|left|right)`: depth arrays.
- `get_camera_intrinsics(top|left|right)`.
- `get_camera_extrinsics(top|left|right)`.

## RGB-D Facts

At the 13:55 snapshot, the robot state reported grippers near normalized
`0.40` from the earlier calibration pose. Top-camera projection of current EE
centers was:

- left EE near `(242.29, 255.16)` px.
- right EE near `(399.85, 258.96)` px.

Same-side reliable blue fingertip components measured by RGB-D and intrinsics:

- Top camera reliable visible blue patches:
  - right: about `41-60 mm` long and `20-28 mm` wide.
  - left: about `44-64 mm` long and `20-21 mm` wide.
- Wrist/side close-up views:
  - left main blue surface: about `58 mm x 24 mm`.
  - right main blue surfaces: about `54 mm x 22-23 mm`.
- Combined reliable visible component range in this run:
  - long axis: `41-64 mm`.
  - short axis: `20-28 mm`.

These measurements are for visible blue surfaces only. They do not include
occluded structure, non-blue black support/body, or the whole collision model.

The visible shape is not a simple flat rectangular parallel jaw. The camera
views show a long blue ribbed/tapered finger surface with black structure
behind and under it. For plate pickup, the useful contact is near the blue
lower/inner edge, not at the abstract TCP alone.

## XML Cross-Check

In the calibrated station XML:

- `left_grasp_site` and `right_grasp_site` are at local `z=0.1347`.
- Blue `grip_pad` boxes have half-extents `0.008 0.001 0.025`, so nominal full
  size is about `16 mm x 2 mm x 50 mm`.
- Long blue support/rail boxes have half-extents around `0.006 0.002 0.04` and
  `0.006 0.002 0.039`, so nominal full sizes are about
  `12 mm x 4 mm x 80 mm` and `12 mm x 4 mm x 78 mm`.

The XML dimensions agree with the RGB-D conclusion that the visible contact
surface is narrow and roughly `50-60 mm` long.

## Caveats

- Use same-side/projection gating. Top RGB-D includes unrelated blue/greenish
  scene components and occasional opposite-side contamination.
- Side/wrist cameras are reliable for near-field visible blue geometry, but
  they can include both fingers and crop/edge effects.
- Full point-cloud PCA can overestimate dimensions at depth discontinuities.
  For component size, prefer median-depth pinhole extents plus the overlay.
- The 13:04 physical calibration swept normalized widths `0.68`, `0.54`, and
  `0.40`, but it did not save raw RGB-D per sweep step. Therefore current
  artifacts do not prove width-command-to-observed-gap. The calibration script
  now saves per-stage RGB-D for the next physical calibration run.

## Plate Grasp Implications

- Plan a low asymmetric under/over rim pinch, not a center squeeze and not a
  high TCP-side grasp.
- Treat the lower blue fingertip edge as the key contact feature. The lower
  finger should be deliberately inserted slightly under the plate lip while
  the upper finger stays above the rim/top surface.
- Use a visible fingertip contact model around `50-60 mm` long and
  `16-25 mm` effective width, with uncertainty widened to the observed
  `20-28 mm` blue patch width when checking camera evidence.
- Keep the gripper tilted slightly toward the lower edge during insertion.
- Close conservatively in stages and require contact/gripper-state evidence
  before any lift.
- First lift should keep the grasp orientation. Do not reorient the dish before
  a small verified lift clears the table.
- 2026-05-29 bowl/dish top-down rim-straddle retries repeatedly closed in free
  space even after sweeping TCP-to-visible-contact offset, flipping the
  finger-forward sign, and lowering TCP-above-rim from `0.115 m` to `0.095 m`.
  Do not treat these failures as a contact-threshold issue: the gripper reached
  the commanded close target each time. Before another top-down physical retry,
  add a pre-close visual geometry gate or run a bounded gripper/TCP contact
  calibration for the exact wrist orientation.
