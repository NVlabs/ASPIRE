---
name: grasp
description: Structural template for pick-and-place programs. Covers the standard grasp-lift-place code skeleton, make_topdown_quat(), pre-grasp/lower/close sequence, and placement pattern. No task-specific strategies — those are discovered through experiment.
---

# Grasp — Code Template

> This skill provides the **structural skeleton** for pick-and-place programs.
> Task-specific strategies (object-specific offsets, SAM3 prompts, placement geometry)
> are discovered through experiment and added here as the skill library grows.

---

## Standard Pick-and-Place Template

```python
import numpy as np
from scipy.spatial.transform import Rotation

def make_topdown_quat(yaw_deg=0):
    """Top-down gripper orientation, rotated yaw_deg around Z."""
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz for API

# --- Observe ---
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

# --- Localize pick object via SAM3 ---
masks = segment_sam3_text_prompt(rgb, "<object prompt>")
if not masks:
    raise RuntimeError("SAM3: no masks for pick object")
best = max(masks, key=lambda d: d["score"])
obj_pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
obj_center = obj_pts.mean(axis=0)

# --- Grasp via GraspNet (preferred for irregular objects) ---
grasp_poses, grasp_scores = plan_grasp(depth, K, best["mask"])
best_grasp_world, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp_world is None:
    best_grasp_world = E @ grasp_poses[grasp_scores.argmax()]
grasp_pos, quat = decompose_transform(best_grasp_world)

open_gripper()
goto_pose(grasp_pos, quat, z_approach=0.15)
goto_pose(grasp_pos, quat)
close_gripper()

# --- OR: Grasp via make_topdown_quat (simple flat objects) ---
# quat = make_topdown_quat(yaw_deg=0)
# grasp_pos = np.array([obj_center[0], obj_center[1], obj_center[2] + <z_offset>])
# open_gripper()
# joints = solve_ik((grasp_pos + np.array([0, 0, 0.08])).tolist(), quat.tolist())
# if joints is not None: move_to_joints(joints)
# joints = solve_ik(grasp_pos.tolist(), quat.tolist())
# if joints is not None: move_to_joints(joints)
# close_gripper()

# --- Lift ---
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15])
joints = solve_ik(lift_pos.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)

# --- Re-observe for placement target ---
obs2 = get_observation()
rgb2 = obs2["agentview"]["images"]["rgb"]
d2 = obs2["agentview"]["images"]["depth"]
d2 = d2[:, :, 0] if len(d2.shape) == 3 else d2
K2 = obs2["agentview"]["intrinsics"]
E2 = obs2["agentview"]["pose_mat"]

target_masks = segment_sam3_text_prompt(rgb2, "<target prompt>")
if not target_masks:
    raise RuntimeError("SAM3: no masks for target")
best_t = max(target_masks, key=lambda d: d["score"])
tgt_pts = mask_to_world_points(best_t["mask"].astype(np.uint8), d2, K2, E2)
tgt_center = tgt_pts.mean(axis=0)
surface_z = tgt_pts[:, 2].max()

# --- Transport ---
above_target = np.array([tgt_center[0], tgt_center[1], lift_pos[2]])
joints = solve_ik(above_target.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)

# --- Place ---
release_pos = np.array([tgt_center[0], tgt_center[1], surface_z + 0.03])
joints = solve_ik(release_pos.tolist(), quat.tolist())
if joints is not None: move_to_joints(joints)
open_gripper()
```

---

## Key Conventions

**Depth normalization** — always do this before any 3D call:
```python
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
```

**SAM3 result** — list of dicts, pick highest score:
```python
best = max(masks, key=lambda d: d["score"])
obj_pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
```

**GraspNet TCP offset** — `plan_grasp` already applies a 0.12 m offset along the grasp Z axis
to its returned poses; do not add another TCP compensation on top of GraspNet grasps.

**GraspNet grasp selection** — use `select_top_down_grasp`, not a manual z-axis loop:
```python
grasp_poses, grasp_scores = plan_grasp(depth, K, mask)
best_grasp_world, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if best_grasp_world is None:
    best_grasp_world = E @ grasp_poses[grasp_scores.argmax()]
```

**IK guard** — always check `if joints is not None` before `move_to_joints`.

**TCP offset for placement**: `goto_pose(pos, quat)` moves the **WRIST** to `pos`; fingertips extend below the wrist. Calibrate per robot/gripper config and adjust placement Z accordingly.

**Top-level call** — if you define a `run()` function, call it at the end:
```python
def run():
    ...
run()
```

**Gripper width thresholds** — add validated per-object thresholds here as you discover them:

| Object | Good grasp (gw >) | Air grasp (gw <) | Notes |
|---|---|---|---|

---

## Object Geometry Utilities

```python
# Z range (grasp height tuning)
z_min, z_max = obj_pts[:, 2].min(), obj_pts[:, 2].max()
height = z_max - z_min

# OBB (orientation-aware bounding box)
obb = get_oriented_bounding_box_from_3d_points(obj_pts)
# Keys: "center" (3,), "extent" (3,), "R" (3x3 rotation matrix)
obb_center = obb["center"]
```
