---
name: robosuite/fix-loop/skills/grasp
description: Structural template for pick-and-place programs. Covers the standard grasp-lift-place code skeleton, make_topdown_quat(), pre-grasp/lower/close sequence, and placement pattern. No task-specific strategies — those are discovered through experiment.
---
# Grasp — Code Template

> This skill provides the **structural skeleton** for pick-and-place programs.
> Task-specific strategies (object-specific offsets, SAM3/Molmo prompts, placement geometry, etc.)
> are discovered through experiment and added here as the skill library grows.

---

## Validated Grasp Strategies (Robosuite)

### Cube Grasping — Center Height
Grasp at cube **center Z** = `(top_z + bot_z) / 2`, not top surface. Top-surface grasps place fingertips at the edge; center grasps wrap fingers around the cube for a solid hold. Works reliably for ~4cm cubes.


### Cube Grasping — GraspNet Z Override
When using GraspNet (`plan_grasp` + `select_top_down_grasp`), the returned grasp Z is often at the cube top surface rather than center. Override the grasp Z to `(top_z + bot_z) / 2` from the 3D point cloud for more reliable grasps. Validated on cube_lifting, cube_restack, cube_stack (25/25 each).

### Bimanual Handover Geometry
Grasp the object near one end (e.g. hammer head) so the other end (handle) protrudes for the receiving arm. Handover height z=0.175 is a safe sweet spot within [0.15, 0.20]. Maintain >8cm gripper separation during handover. Compute handle_offset dynamically: `max(0.10, object_length * 0.4)`.

### OBB for Grasp Orientation
`get_oriented_bounding_box_from_3d_points(pts)` returns `{center, extent, R}`. The longest extent axis reveals the object's principal direction — use it to align gripper opening or determine approach direction.

### Flat Object Grasping — Mid-Height TCP
For thin/flat objects on the table (e.g. nuts, washers, <2cm thick), set TCP Z to object **mid-height** = `(table_z + top_z) / 2`. GraspNet often targets table-level Z, producing air grasps. Nut assembly: table Z=-0.102, nut top Z=-0.082, grasp Z=-0.092.

### Side-Grasp for Handles / Protrusions
For protruding objects like pot handles: gripper y-axis = world z, z-axis = approach direction (from handle toward object center). Pre-grasp 10cm outward + 5cm above, then lower to handle height and close. Works for `two_arm_lift` pot handles.

### Gripper Contact Detection
After `close_gripper()`, check `robot0_gripper_qpos[0]` from observation: <0.003 = air grasp, >0.005 = object contact, 0.003–0.005 = marginal. Use this to trigger retry logic.

---
