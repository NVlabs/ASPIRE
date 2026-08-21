# YAM Geometry Ground Truth

Use this file for reusable YAM object, camera, table, and gripper facts that
should survive across runs. Keep run-specific coordinates, hypotheses, and
failure narratives in local run artifacts. Do not commit workstation-specific
run histories or operator context to the repository.

## Table And Cameras

- World axes from calibrated XML:
  - `+Z`: up.
  - `+X`: from arm bases toward the tabletop workspace.
  - `+Y`: left-arm side; `-Y`: right-arm side.
- Arm base positions in the calibrated XML:
  left `[0.2525, 0.31, 0.75]`, right `[0.2525, -0.31, 0.75]`.
- XML `play_table` geom is center `[0.6, 0, 0.745]` with half-thickness
  `0.005`, so the modeled table top is `0.750 m`.
- Current top-camera table/support estimate used by scripts: `TABLE_Z ~= 0.760 m`.
  Evidence: read-only debug smoke `logs/debug_observation_smoke_20260530T204215`.
- Top-camera table prompt medians in that run were tightly clustered:
  `table=0.75986`, `table top=0.75987`, `table surface=0.75967`,
  `workspace table=0.75932`, `wooden table=0.76068`.
- Treat `0.750 m` as the XML/model table top and `0.760 m` as the current
  top-camera depth/support estimate. Prefer the script-configured support Z
  for contact reasoning unless revalidating the sensor/model offset.
- Top camera XML position is `[0.0887, 0.0118, 1.6918]`.
- Wrist camera XML offsets are local offsets, not world heights:
  left wrist camera local offset Z is about `0.0746 m`; right wrist camera
  local offset Z is about `0.0750 m`.
- Side-camera table Z estimates from the 2026-05-30 smoke run were not
  consistent with top-camera table Z. Do not use side or bottom table Z for
  motion planning until a dedicated depth-plane or extrinsics check verifies
  those views.
- Bottom camera is image/debug only unless calibrated and explicitly verified.
  It may help visual reasoning and overlays, but it must not drive world-frame
  motion planning, ranking, close width, or success verification.

## Object Measurements

- Orange bowl physical height: about `8 cm`.
- White bowl physical height: about `0.5 cm`.
- Orange and white bowl radii are approximately the same in physical reality.
- Orange bowl successful top-camera estimate:
  `logs/place_orange_bowl_on_rack_one_shot_loop_20260530T201429` estimated
  rim Z `0.83494 m`, giving height above table
  `0.83494 - 0.760 ~= 0.075 m`. The XML/model `0.750 m` table top gives
  `0.085 m`; treat the operator measurement and top-camera support estimate as
  the stronger contact reference.
- Radius anchors from recent bowl scripts:
  orange default radius `~0.095 m`; white top-box-scaled radius once estimated
  `~0.124 m`, but this likely overestimates the real white bowl because the
  physical white and orange radii are similar.
- `z ~= 0.8 m` in logs is world height, not object height. Object height above
  the table must be computed relative to the table estimate.

## Gripper Geometry

- Operator TCP meaning: gripper end point without the fingers.
- Physical finger length: about `11 cm`.
- Robot XML grasp site: `*_grasp_site pos="0 0 0.1347"` means the modeled
  grasp site is about `13.47 cm` from the TCP in the gripper local frame.
- Blue pad collision geometry has about `5 cm` modeled contact length, but
  current bowl scripts mostly reason about commanded TCP pose, rim/contact
  estimates, open width, and staged close rather than full pad surface geometry.
- Do not assume hard encoder stall is required for a valid bowl hold. For
  current bowl classes, use staged close, plausible final width, and
  post-action perception/video verification.

## Validated Bowl Pickup/Lift Anchors

- Orange bowl pickup/lift and return-to-table validation:
  `logs/place_orange_bowl_on_rack_one_shot_loop_20260530T201429`.
  It used top-down rim straddle, `OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M=0.010`,
  close target `0.04`, lifted, verified motion, and returned the bowl.
- White bowl pickup/lift and return-to-table validation:
  `logs/place_white_bowl_on_rack_one_shot_loop_after_analysis_20260530T211151`.
  It used right-arm `bowl_top_down_rim_straddle_0`, rim angle `-90 deg`,
  `OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M=-0.040`, close target `0.04`,
  and verified object motion with `z_delta=0.02915 m`, `xy_delta=0.01709 m`.
- Correction: earlier white-bowl notes that described close target `0.03` for
  the `195539` run were stale. The executed target was `0.04`.
- These are pickup/lift anchors only. They do not validate rack placement.
