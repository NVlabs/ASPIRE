# Multiview Camera Debug Policy

Use `top`, `left`, and `right` as calibrated motion/perception cameras unless a
run proves otherwise.

## Current Verified Bottom Status

Verified on 2026-05-31:

- Bottom overlay artifact support exists in the debug observation smoke path.
- `bottom` is now in the default Real-YAM station camera set and default
  recording camera set.
- Camera portal health for port `8300` reported configured and available
  cameras `top`, `left`, `right`, and `bottom`.
- BundleSDF health for port `8119` reported preview cameras `top`, `left`,
  `right`, and `bottom`.
- Direct `/preview/bottom` validation saved a real `640x480` JPEG frame.
- Verified smoke artifact:
  `logs/debug_observation_smoke_20260531T230506/bottom.mp4`.
- The smoke run wrote `top.mp4`, `left.mp4`, `right.mp4`, and `bottom.mp4` by
  default. `bottom.mp4` was `640x480`, 25 fps, 5.04 s, 126 frames.
- The smoke packet has camera keys `top`, `left`, `right`, and `bottom`.
- In that packet, `bottom` is `image_only=true` and
  `motion_source_allowed=false`.

`bottom` is debug-only until its robot/world extrinsics are configured and
verified. It may be used for:

- raw RGB snapshots such as `bottom.png`;
- debug overlays and videos;
- human/Codex visual reasoning about occlusion, gripper/fingertip placement,
  and bowl/rim contact.

Do not use bottom-camera `position_3d`, depth, or detections for physical
motion planning, candidate ranking, gripper close width, or task success until
bottom calibration is explicitly validated.

Saved scripts should pass `bottom` as an image-only camera and keep
`motion_cameras=("top", "left", "right")`.

To request bottom observation artifacts in saved scripts, set:

```bash
OPENFORGE_DEBUG_OBS_CAMERAS=top,left,right,bottom
OPENFORGE_BOWL_RACK_IMAGE_ONLY_CAMERAS=bottom
```

This requests bottom observation/overlay artifacts when `get_camera_image` can
return `bottom`, while still keeping bottom image-only and excluded from motion
planning.

Video recording no longer needs a per-run bottom override on Real-YAM; the
default recording camera list includes `bottom`.
