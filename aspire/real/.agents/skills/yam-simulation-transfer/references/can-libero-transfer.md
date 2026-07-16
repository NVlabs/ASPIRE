# LIBERO Can Handling Skills

Consolidated from the full skill-library branches:

This file combines the reusable "soda can" / cylinder handling guidance. In LIBERO, can-like objects include alphabet soup, tomato sauce, ketchup/bbq/salad-dressing bottle variants, and other narrow cylinders.

## When To Use

Use this skill when the target is a small or narrow cylindrical object:

- soda-can-like object
- soup can
- tomato sauce can
- jar
- narrow bottle
- wine bottle
- short round can on a tray or elevated surface

Typical failure signatures:

- SAM3 returns multiple similar cans and the highest score is the wrong one.
- GraspNet returns a tilted orientation and the gripper misses or pushes the can over.
- The gripper closes on the rim instead of the center.
- A tall box/carton is selected instead of a can because its bbox is smaller.
- OBB center is biased by oblique depth back-projection.

## Localize The Correct Can

Filter by geometry first, then by SAM3 score. Do not trust `masks[0]` or max score alone when there are multiple similar objects.

For can-shaped masks, useful image-space filters are:

```python
bbox = m.get("box", None)
w = bbox[2] - bbox[0]
h = bbox[3] - bbox[1]
cy = (bbox[1] + bbox[3]) / 2
ratio = h / w if w > 0 else 99

is_can_shaped = (
    25 <= w <= 70 and
    35 <= h <= 80 and
    ratio < 2.0
)
```

The `ratio < 2.0` filter is important. In one LIBERO object task, smallest bbox area incorrectly selected an orange-juice box over the alphabet-soup can:

- alphabet soup can: `w=38`, `h=53`, `h/w=1.39`
- orange juice box: `w=28`, `h=69`, `h/w=2.46`

Among can-shaped candidates, select by score or by scene-specific constraints such as expected image row or world position.

```python
best_mask = None
best_score = -1.0

for m in masks[:10]:
    bbox = m.get("box", None)
    if bbox is None:
        continue

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cy = (bbox[1] + bbox[3]) / 2
    ratio = h / w if w > 0 else 99
    score = m.get("score", 0)

    if 25 <= w <= 70 and 35 <= h <= 80 and ratio < 2.0:
        if score > best_score:
            best_score = score
            best_mask = m
```

When two similar cylindrical cans are both valid, use bbox pixel area or task-specific position constraints:

```python
px_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
```

Do not use trace `area_pct` for this; trace logs can truncate it. Use the mask dict's `box` field directly.

## 3D Geometry Filters

After SAM3 selection, convert the mask to world points and reject masks whose 3D geometry does not match a can.

```python
pts = mask_to_world_points(mask, depth_img, K, E)
if pts is None or len(pts) < 30:
    reject()

ctr = np.median(pts, axis=0)
z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
h_range = z_max - z_min

if not (0.04 <= ctr[2] <= 0.10):
    reject()  # common tabletop can center-height range

if h_range < 0.04:
    reject()  # likely flat box/package

if h_range > 0.20:
    reject()  # likely tall clutter or wrong object
```

General height cues:

- upright carton / tall bottle: `h_range >= 0.10`
- flat box / cream cheese / butter: `h_range <= 0.04`
- can: often `h_range ~= 0.11`, with roughly equal XY extents

Use scene-specific filters when known. Example for alphabet soup vs tomato sauce:

```python
# Alphabet soup can on the left side of the table.
if ctr[1] > -0.05:
    reject()

if not (0.04 <= ctr[2] <= 0.10):
    reject()
```

## Grasp Orientation

For narrow cylindrical objects, do not trust GraspNet orientation. Even small tilts can miss or push the can over. Use GraspNet for XY only, and force a pure top-down orientation.

```python
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, 180 deg around X
```

If an existing grasp quaternion is not sufficiently top-down, replace it:

```python
from scipy.spatial.transform import Rotation

R_grasp = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
if R_grasp[2, 2] > -0.7:
    quat = TOP_DOWN_QUAT
```

If using SAM3 centroid XY without GraspNet refinement and the centroid may be off-axis, try a 90-degree yaw top-down grasp so the finger spread crosses a wider slice of the cylinder:

```python
quat = make_topdown_quat(yaw_deg=90)
```

This was validated for a wine-bottle cylinder case where yaw 0 failed and yaw 90 succeeded.

## Grasp XY

Prefer a robust cylinder center. Depending on the scene, use one of these methods.

### GraspNet XY With Forced Top-Down Orientation

Use the most vertical GraspNet candidate for XY only.

```python
grasps, scores = plan_grasp(depth_img, K, obj_mask)
grasp_xy = None

if len(grasps) > 0:
    best_vert_score = -np.inf
    for i in range(len(grasps)):
        g_world = E @ grasps[i]
        verticality = abs(g_world[:3, 2][2])
        combined = scores[i] * (verticality ** 2)
        if combined > best_vert_score:
            best_vert_score = combined
            grasp_xy = g_world[:2, 3]
```

### Short Can XY Correction

For short cans viewed obliquely, depth back-projection can bias the OBB X-center by 1-2 cm and cause rim grasps. Use `plan_grasp()` XY as primary if it is close to the OBB center; otherwise keep the OBB center.

```python
obj_obb = get_oriented_bounding_box_from_3d_points(obj_pts)
grasp_poses, grasp_scores = plan_grasp(depth_img, K, obj_mask)

if grasp_poses is not None and len(grasp_poses) > 0:
    best_world = E @ grasp_poses[grasp_scores.argmax()]
    plan_xy = best_world[:3, 3][:2]
    dist_to_obb = np.linalg.norm(plan_xy - obj_obb["center"][:2])

    if dist_to_obb < 0.04:
        grasp_pos[0], grasp_pos[1] = plan_xy[0], plan_xy[1]
```

### Snap Bad Grasp XY Back To Object Center

If a top-down grasp planner returns an XY point more than half the object radius away from the OBB center, snap back to the OBB center.

```python
obj_obb = get_oriented_bounding_box_from_3d_points(obj_pts)
obj_radius = max(obj_obb["extent"][0], obj_obb["extent"][1]) / 2
dist_xy = np.linalg.norm(grasp_pos[:2] - obj_obb["center"][:2])

if dist_xy > 0.5 * obj_radius:
    grasp_pos[0], grasp_pos[1] = obj_obb["center"][0], obj_obb["center"][1]
```

### Rim-Percentile XY For Round Objects On Elevated Surfaces

For round/cylindrical objects on elevated surfaces, OBB center can be biased by oblique camera angle. Use the top rim's 5th-95th percentile midpoint.

```python
rim_pts = obj_pts[obj_pts[:, 2] > obj_pts[:, 2].max() - 0.012]
obj_xy = np.array([
    0.5 * (np.percentile(rim_pts[:, 0], 5) + np.percentile(rim_pts[:, 0], 95)),
    0.5 * (np.percentile(rim_pts[:, 1], 5) + np.percentile(rim_pts[:, 1], 95)),
])
```

### Percentile Bbox Center For Symmetric Cans

For clean can masks, the 3D percentile bbox center is often stable:

```python
cx = 0.5 * (np.percentile(obj_pts[:, 0], 10) + np.percentile(obj_pts[:, 0], 90))
cy = 0.5 * (np.percentile(obj_pts[:, 1], 10) + np.percentile(obj_pts[:, 1], 90))
```

## Grasp Z

For tall cans and bottle-like cylinders, grasp around 60-75% of object height. This avoids unstable base grasps and neck/cap grasps.

```python
z_base = obj_pts[:, 2].min()
z_top = obj_pts[:, 2].max()
grasp_z = z_base + 0.70 * (z_top - z_base)
```

For short soup-can-like objects, successful working code used center plus a small upward offset:

```python
grasp_pos = np.array([cx, cy, np.median(obj_pts, axis=0)[2] + 0.02])
```

Use an approach height high enough to avoid pushing the can:

```python
hover = grasp_pos.copy()
hover[2] += 0.15
goto_pose(hover, TOP_DOWN_QUAT)
goto_pose(grasp_pos, TOP_DOWN_QUAT)
close_gripper()
```

or:

```python
goto_pose(grasp_pos, TOP_DOWN_QUAT, z_approach=0.12)
goto_pose(grasp_pos, TOP_DOWN_QUAT)
close_gripper()
```

## Retry On Air Grasp

Check gripper width after close. If it indicates an air grasp, retry at slightly deeper Z values.

```python
close_gripper()
obs2 = get_observation()
gw = obs2["robot_cartesian_pos"][-1]

if gw < 0.10:
    for retry_dz in [-0.015, -0.030, +0.010]:
        open_gripper()
        retry = grasp_pos.copy()
        retry[2] += retry_dz
        goto_pose(hover, TOP_DOWN_QUAT)
        goto_pose(retry, TOP_DOWN_QUAT)
        close_gripper()

        obs2 = get_observation()
        gw = obs2["robot_cartesian_pos"][-1]
        if gw >= 0.10:
            break
```

For hard-to-reach grasps near workspace edges, repeated IK descent can help:

```python
for _ in range(5):
    j = solve_ik(grasp_pos.tolist(), TOP_DOWN_QUAT.tolist())
    if j is not None:
        move_to_joints(j)
close_gripper()
```

## Place Into Basket Or Container

When placing a can into a basket, localize the basket before grasping, because the arm may occlude it after lift.

For basket center, SAM3 centroid can be biased toward a wall. Use percentile midpoint over elevated basket points:

```python
pts_f = basket_pts[
    (basket_pts[:, 2] > 0.05) &
    (basket_pts[:, 1] > 0.15) &
    (basket_pts[:, 0] > 0.40)
]

bx = 0.5 * (np.percentile(pts_f[:, 0], 10) + np.percentile(pts_f[:, 0], 90))
by = 0.5 * (np.percentile(pts_f[:, 1], 10) + np.percentile(pts_f[:, 1], 90))
basket_top_z = np.percentile(pts_f[:, 2], 90)
```

Use a lift and lateral transport before lowering:

```python
lift = grasp_pos.copy()
lift[2] += 0.18
goto_pose(lift, TOP_DOWN_QUAT)

pre_drop = np.array([bx, by, basket_top_z + 0.12])
goto_pose(pre_drop, TOP_DOWN_QUAT)

drop = np.array([bx, by, basket_top_z + 0.04])
goto_pose(drop, TOP_DOWN_QUAT)
open_gripper()
```

## LIBERO Object-Task Remaps

Some LIBERO object-task filenames do not match the actual runtime object. Relevant can/cylinder remaps from the old object/spatial-goal skill library:

| BDDL task file | Actual pick target | Fix approach |
|---|---|---|
| `pick_up_the_cream_cheese` | alphabet soup can | SAM3 `"alphabet soup"`, cylinder/can template |
| `pick_up_the_salad_dressing` | tomato sauce can | SAM3 `"tomato sauce can"`, can template |
| `pick_up_the_bbq_sauce` | ketchup bottle | SAM3 `"red tomato ketchup bottle"`, cylinder template |
| `pick_up_the_chocolate_pudding` | salad dressing bottle | bottle/cylinder template |
| `pick_up_the_tomato_sauce` | bbq sauce bottle | SAM3 `"bbq sauce bottle"`, cylinder template |

Do not assume the BDDL filename is the visual target. Inspect baseline code, scene observations, successful traces, or object positions before writing the final prompt/filter.

## Minimal Can Pickup Template

```python
import numpy as np

TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

def pick_can(can_prompts):
    goto_home_joint_position()

    obs = get_observation()
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    K = cam["intrinsics"]
    E = cam["pose_mat"]

    obj_mask = None
    obj_pts = None
    best_score = -1.0

    for prompt in can_prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue

        for m in masks[:10]:
            bbox = m.get("box", None)
            if bbox is None:
                continue

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            ratio = h / w if w > 0 else 99
            score = m.get("score", 0)

            if not (25 <= w <= 70 and 35 <= h <= 80 and ratio < 2.0):
                continue

            mask = m["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 30:
                continue

            z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
            h_range = z_max - z_min
            if h_range < 0.04 or h_range > 0.20:
                continue

            if score > best_score:
                best_score = score
                obj_mask = mask
                obj_pts = pts

    if obj_mask is None:
        return False

    cx = 0.5 * (np.percentile(obj_pts[:, 0], 10) + np.percentile(obj_pts[:, 0], 90))
    cy = 0.5 * (np.percentile(obj_pts[:, 1], 10) + np.percentile(obj_pts[:, 1], 90))
    z_base = obj_pts[:, 2].min()
    z_top = obj_pts[:, 2].max()
    grasp_z = z_base + 0.70 * (z_top - z_base)

    grasp_pos = np.array([cx, cy, grasp_z])
    hover = grasp_pos.copy()
    hover[2] += 0.15

    open_gripper()
    goto_pose(hover, TOP_DOWN_QUAT)
    goto_pose(grasp_pos, TOP_DOWN_QUAT)
    close_gripper()

    obs2 = get_observation()
    gw = obs2["robot_cartesian_pos"][-1]
    if gw < 0.10:
        for retry_dz in [-0.015, -0.030, +0.010]:
            open_gripper()
            retry = grasp_pos.copy()
            retry[2] += retry_dz
            goto_pose(hover, TOP_DOWN_QUAT)
            goto_pose(retry, TOP_DOWN_QUAT)
            close_gripper()
            obs2 = get_observation()
            gw = obs2["robot_cartesian_pos"][-1]
            if gw >= 0.10:
                break

    return gw >= 0.10
```
