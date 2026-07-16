---
name: robosuite/fix-loop/skills/localize
description: Object localization via SAM3 — prompting strategies, multi-prompt fallback pattern, 3D centroid extraction, and per-object prompt registry. Grows through experiment.
---

# Localize — SAM3 Prompting & Object Localization

> This skill tracks **what perception (SAM3/Molmo) prompts and prompting strategies work for which objects** and the standard
> localization helper pattern. Add entries to the prompt registry as you discover
> working prompts through experiment.

---

## Key Patterns (validated across tasks)

### SAM3 Mask Filtering — Robot Arm Pollution
SAM3 text prompts frequently return masks that include the robot arm. The robot arm mask is typically >12K pixels. **Always filter SAM3 masks by pixel count** (e.g. 50–12000 for cubes) to exclude arm/background masks.

### Camera Occlusion Avoidance
Move the arm to a safe height (e.g. Z=0.15) **before** calling `get_observation()`. If the arm is near the workspace, it blocks the camera view and corrupts segmentation/depth.

### Robust Centroid Extraction
Use **median** (not mean) of 3D world points from `mask_to_world_points()` for centroid estimation. Outlier points from depth noise or mask edges skew the mean.

### Molmo Stochasticity Warning
`point_prompt_molmo` returns **different pixel coordinates on repeated calls** with the same image and prompt. Do not rely on it as the sole localization source. Best pattern: use Molmo to get an approximate pixel, then refine with `segment_sam3_point_prompt` at that pixel and pick the smallest valid mask.

### Multi-Prompt Fallback
When a primary SAM3 text prompt returns 0 masks, fall back to alternative prompts or Molmo → SAM3 point prompt refinement.

### Z-Filtering for Disambiguation
When SAM3 confuses similar-looking objects (e.g. nut vs peg), filter 3D points by z-coordinate (e.g. `z_max > 0` to exclude flat table-level objects, or `z < threshold` to exclude elevated objects).

### Retry with Fresh Perception After Failed Manipulation
SAM3 is non-deterministic — the same code+seed can produce different masks across runs. When a grasp fails, **retract arms → re-observe → re-segment with a fresh SAM3 call** is more effective than complex mask filtering. This pattern turned ~50% single-attempt success into ~72% with retry on `two_arm_lift`.

### Thin Object Centroid Warning
For thin objects (handles, rods, <2cm diameter), SAM3 mask centroids are unreliable — the mask often includes surrounding surface pixels, shifting the centroid 1–3cm from the actual object center. Use the median of 3D points and consider additional filtering by principal axis.

---

## Robosuite Object Prompt Registry

| Task | Object | Model | Prompt | Filter/Notes |
|---|---|---|---|---|
| `cube_restack` | Green cube | SAM3 | `"green cube"` | Pixel count 50–12000 |
| `cube_restack` | Red cube | SAM3 | `"red cube"` | Pixel count 50–12000 |
| `cube_restack` | Green cube | Molmo | `"green cube"` | Fallback |
| `cube_restack` | Red cube | Molmo | `"red cube"` | Fallback |
| `two_arm_handover` | Hammer | SAM3 | `"hammer"` | Select by highest score; returns ~200 masks |
| `two_arm_handover` | Hammer | Molmo | `"point to the hammer"` | Fallback (rarely needed) |
| `two_arm_lift` | Green handle | SAM3 | `"green handle"` | Returns ~200 masks; use max score. Centroid unreliable (thin object ~1cm) |
| `two_arm_lift` | Blue handle | SAM3 | `"blue handle"` | Same as green handle |
| `two_arm_lift` | Pot (height check) | SAM3 | `"pot"` | Used post-lift to verify pot elevation |
| `two_arm_lift` | Green handle | Molmo | `"point to the green handle on the pot"` | Fallback for SAM3 failures |
| `two_arm_lift` | Blue handle | Molmo | `"point to the blue handle on the pot"` | Fallback for SAM3 failures |
| `cube_lifting` | Red cube | SAM3 | `"red cube"` | Pixel count 50–12000; "red block", "cube" also work |
| `cube_stack` | Red cube | SAM3 | `"red cube"` | Pixel count 50–12000 |
| `cube_stack` | Green cube | SAM3 | `"green cube"` | Pixel count 50–12000 |
| `spill_wipe` | Spill | SAM3 | `"brown spill"` | Reliable across all seed positions |
| `nut_assembly` | Brown square peg | SAM3 | `"brown block"`, `"small block"` | Filter: Z>-0.05, Y>0, area 50–2000 |
| `nut_assembly` | Nut body | SAM3 | `"brown square nut"` | Filter: Z<-0.05, area 800–6000 |
| `nut_assembly` | Handle | SAM3 | `"extruded handle of the brown square nut"` | Filter: area 50–1500, within 12cm of nut center |
| `nut_assembly` | Handle | Molmo | `"extruded handle of the brown square nut"` | Fallback |

---
