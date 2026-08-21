---
name: bimanual
description: Consolidated patterns for bimanual (two-arm) manipulation in Robosuite — two_arm_lift and two_arm_handover. Combines grasp geometry, SAM3 localization, and transport/sequencing. Sourced from the robosuite-branch grasp/localize/transport skills.
---

# Bimanual Manipulation — Two-Arm Lift & Two-Arm Handover

> Consolidated reference for the two bimanual Robosuite tasks: `two_arm_lift`
> (lift a pot by both handles) and `two_arm_handover` (one arm grasps an object
> near one end and hands it to the other arm). Merges the bimanual-relevant
> content from the `grasp`, `localize`, and `transport` skills.

---

## 1. Localization (SAM3 / Molmo)

### Camera Occlusion Avoidance
Move arms to a safe height (e.g. Z=0.15) **before** calling `get_observation()`.
Arms near the workspace block the camera and corrupt segmentation/depth. Always
retract before (re-)observing.

### SAM3 Mask Filtering — Robot Arm Pollution
SAM3 text prompts frequently return masks that include the robot arm (typically
>12K pixels). Filter masks by pixel count to exclude arm/background masks.

### Robust Centroid Extraction
Use the **median** (not mean) of 3D world points from `mask_to_world_points()`.
Outliers from depth noise / mask edges skew the mean.

### Thin Object Centroid Warning (critical for handles)
For thin objects (pot handles, hammer handle/rods, <2cm diameter), SAM3 mask
centroids are unreliable — the mask often grabs surrounding surface pixels,
shifting the centroid 1–3cm. Use the median of 3D points and consider extra
filtering by principal axis.

### Molmo Stochasticity Warning
`point_prompt_molmo` returns different pixel coords on repeated calls with the
same input. Don't use it as the sole source. Best pattern: Molmo for an
approximate pixel → refine with `segment_sam3_point_prompt` at that pixel →
pick the smallest valid mask.

### Multi-Prompt Fallback
If a primary SAM3 text prompt returns 0 masks, fall back to alternative prompts
or Molmo → SAM3 point-prompt refinement.

### Retry with Fresh Perception After Failed Manipulation
SAM3 is non-deterministic — same code+seed can yield different masks across runs.
On a failed grasp: **retract arms → re-observe → re-segment with a fresh SAM3
call**. This is more effective than complex mask filtering. On `two_arm_lift`
this turned ~50% single-attempt success into ~72% with retry.

### Prompt Registry (bimanual objects)

| Task | Object | Model | Prompt | Filter/Notes |
|---|---|---|---|---|
| `two_arm_handover` | Hammer | SAM3 | `"hammer"` | Select by highest score; returns ~200 masks |
| `two_arm_handover` | Hammer | Molmo | `"point to the hammer"` | Fallback (rarely needed) |
| `two_arm_lift` | Green handle | SAM3 | `"green handle"` | ~200 masks; use max score. Centroid unreliable (~1cm thin) |
| `two_arm_lift` | Blue handle | SAM3 | `"blue handle"` | Same as green handle |
| `two_arm_lift` | Pot (height check) | SAM3 | `"pot"` | Used post-lift to verify pot elevation |
| `two_arm_lift` | Green handle | Molmo | `"point to the green handle on the pot"` | Fallback for SAM3 failures |
| `two_arm_lift` | Blue handle | Molmo | `"point to the blue handle on the pot"` | Fallback for SAM3 failures |

---

## 2. Grasp Geometry

### Side-Grasp for Handles / Protrusions (two_arm_lift)
For protruding objects like pot handles: gripper y-axis = world z, z-axis =
approach direction (from handle toward object center). Pre-grasp 10cm outward +
5cm above, then lower to handle height and close.

### Bimanual Handover Geometry (two_arm_handover)
Grasp the object near one end (e.g. hammer head) so the other end (handle)
protrudes for the receiving arm. Handover height **z=0.175** is a safe sweet spot
within [0.15, 0.20]. Maintain **>8cm gripper separation** during handover.
Compute handle offset dynamically: `handle_offset = max(0.10, object_length * 0.4)`.

### OBB for Grasp Orientation
`get_oriented_bounding_box_from_3d_points(pts)` returns `{center, extent, R}`.
The longest extent axis reveals the object's principal direction — use it to
align the gripper opening or to determine approach direction (especially for the
hammer in handover and for handle orientation in lift).

### Gripper Contact Detection (air-grasp / retry trigger)
After `close_gripper()`, check `robot0_gripper_qpos[0]` from observation:
- `<0.003` = air grasp
- `>0.005` = object contact
- `0.003–0.005` = marginal

Use this to trigger retry logic (see fresh-perception retry above).

---

## 3. Transport & Sequencing

### Safe Height Transit
Always lift to a safe Z (e.g. 0.15) before lateral moves. Prevents collisions
with table objects and keeps camera FOV clear for re-observation.

### Interpolated Moves
Use `interpolate_segment(p1, p2, step=0.02–0.03)` for smooth, collision-free
Cartesian motion. Small steps (0.02m) for critical phases (approach, placement,
handover), larger (0.03m) for transit.

### Separate Translation and Rotation
Separate reorientation from translation into distinct steps. Moving + rotating
simultaneously can cause collisions (e.g. the hammer sweeping through the
workspace during handover reorientation). Pattern: **lift → reorient in place →
translate to target**.

### Table Gap Awareness
The table does **not** span the full region between the two arms. Never release
objects over the central gap. Plan handover positions over solid table surface.

### Retract Before Re-Observation
After failed manipulation, move arms to a fixed safe position above the workspace
**before** `get_observation()`. Arms in the workspace corrupt depth and SAM3 masks.

---

## 4. Full Task Sequences

### two_arm_lift
1. Retract arms to safe height; `get_observation()`.
2. Segment both handles (green/blue) via SAM3.
3. Compute approach direction: handle center → pot center (midpoint of handles).
4. Side-grasp each handle: pre-grasp 10cm outward + 5cm above → lower to handle
   height + 1cm inset → close.
5. Both arms close simultaneously, then lift 15cm.
6. Verify lift with `"pot"` SAM3 height check.
7. On failure (air grasp via gripper qpos < 0.003): open grippers → retract arms
   to safe position above workspace → re-observe → re-segment → retry.

### two_arm_handover
1. Arm0 grasps object near head with DOWN_X orientation.
2. Lift to safe height.
3. Reorient to DOWN_Y (object extends along +X toward Arm1) — reorient in place.
4. Translate to handover position at z=0.175 (over solid table, not the gap).
5. Arm1 approaches from above, grasps handle with >8cm separation.
6. Arm0 releases and retreats.

---
