---
name: robosuite/fix-loop/skills/transport
description: Motion patterns for moving objects between locations — multi-step waypoints, safe transit sequences, interpolated Cartesian moves, collision avoidance during transport. Grows through experiment.
---

# Transport — Motion Patterns

> This skill covers **how to move** once an object is grasped: waypointing, safe transit,
> intermediate stops, and collision avoidance. Discovered through experiment — add entries as
> you validate them.

---

## Validated Patterns (Robosuite)

### Safe Height Transit
Always lift to a safe Z (e.g. 0.15) before lateral moves. Prevents collisions with table objects and keeps the camera FOV clear for re-observation.

### Interpolated Moves
Use `interpolate_segment(p1, p2, step=0.02–0.03)` for smooth, collision-free Cartesian motion. Small steps (0.02m) during critical phases (approach, placement), larger (0.03m) for transit.

### Separate Translation and Rotation
For bimanual tasks, separate reorientation from translation into distinct steps. Moving + rotating simultaneously can cause collisions (e.g. hammer sweeping through workspace during handover reorientation). Pattern: lift → reorient in place → translate to target.

### Bimanual Handover Sequence (two_arm_handover)
1. Arm0 grasps object near head with DOWN_X orientation
2. Lift to safe height
3. Reorient to DOWN_Y (object extends along +X toward Arm1)
4. Move to handover position at z=0.175
5. Arm1 approaches from above, grasps handle with >8cm separation
6. Arm0 releases and retreats


### Table Gap Awareness (bimanual)
The table does not span the full region between arms. Never release objects over the central gap. Plan handover positions over solid table surface.

### Bimanual Lift Sequence (two_arm_lift)
1. Segment both handles (green/blue) via SAM3
2. Compute approach direction: handle center → pot center (midpoint of handles)
3. Side-grasp: pre-grasp 10cm outward + 5cm above → lower to handle height + 1cm inset → close
4. Both arms close simultaneously, then lift 15cm
5. On failure (air grasp detected via gripper qpos < 0.003): open grippers → retract arms to safe position above workspace → re-observe → re-segment → retry

### Spill Wipe — Double-Pass Serpentine
For spill_wipe: horizontal + vertical serpentine passes at z=0.0 with 0.02m step interpolation achieve complete coverage across all spill shapes/positions. SAM3 "brown spill" prompt segments reliably.

### Retract Before Re-Observation
After failed manipulation, always move arms to a fixed safe position above the workspace before calling `get_observation()`. Arms in the workspace corrupt depth data and SAM3 masks.

---
