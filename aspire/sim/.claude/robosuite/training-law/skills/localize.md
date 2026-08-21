---
name: robosuite/training-law/skills/localize
description: Object localization via SAM3 — prompting strategies, multi-prompt fallback, 3D centroid extraction, disambiguation, per-object prompt registry. Grows through experiment.
---

## Move Arm Before Perception

**Always move the robot arm out of the camera view before any SAM3/perception call.** The robot arm is a major source of SAM3 confusion — it gets segmented instead of target objects.
- Move to a neutral pose like `[0.5, 0.0, 0.15]` (up and away from workspace) before observing.
- The camera (`robot0_robotview`) is fixed, not wrist-mounted, so camera extrinsics don't change with robot position. But the arm can occlude objects.

## Multi-Object Disambiguation

When segmenting multiple objects in the same scene:
- **Overlap-based disambiguation**: after segmenting the first object, reject masks for the second object that have >30% overlap with the first object's mask.
- **Single observation + multiple SAM3 calls** is more efficient than multiple observations.

## SAM3 Color Confusion

SAM3 sometimes scores the wrong-color object's mask highest (e.g. "red cube" returns the green cube mask with highest score). When multiple same-shape objects differ only by color:
- **Color verification**: check average RGB under each mask to confirm it matches the target color before selecting by score.
- **Mask size filtering**: only accept masks between ~50px and ~5% of image area. Rejects noise masks and overly large masks.

## 3D Validation Filtering

After SAM3 returns masks, validate each mask's 3D centroid before trusting it:
- **Z-height filter**: reject masks whose centroid z is not at the expected height (e.g. table level). This catches masks on the robot arm base, background, etc.
- **X-range filter**: reject masks outside the workspace bounds.
- **Side preference scoring**: when objects are spatially separated (e.g. left vs right handle), double the score for masks on the expected side.

## Prompt Strategy: Object-Level vs Specific

- SAM3 text prompts with **object-level descriptions** ("green object") often outperform specific descriptions ("green handle") for small/distinctively colored objects.
- Try "color object" first, fall back to more specific prompts.

## XY Localization: OBB vs Median

- **OBB center** (`get_oriented_bounding_box_from_3d_points`) is more stable than point cloud median for XY localization from partial camera views. Prefer OBB center for positioning.
