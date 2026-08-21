# Plate Reference — Synthesized Cross-Skill Guide

Consolidated plate knowledge from `localize`, `grasp`, `transport`, and `manipulation` skills.

---

## 1. Localization (SAM3 Prompt Registry)

### SCENE6 — Single plate (libero_90)
- **Prompts**: `"plate"`, `"dinner plate"`, `"white plate"`
- **Description**: White plate with red stripes at table level (Z ~ 0.030)
- **Position**: Typically X ~ 0.638–0.684, but varies (some seeds X ~ 0.574)
- **SAM3 score**: ~ 0.883–0.926, n_pts ~ 7000+
- **Geometry**: `plate_half_y ~ 0.068` from point cloud
- **Placement right of plate**: `target_y = plate_center[1] + plate_half_y + object_half_y + 0.015`, cap at `plate_center[1] + plate_half_y + 0.10` to avoid off-table
- **Known issue**: Seeds with plate at X = 0.574 consistently fail (mysterious physics)

### Spatial Swap — Plate on table
- **Prompts**: `"plate with red stripes"`, `"plate"`, `"white plate"`
- **Description**: White plate with red stripes at table level (Z ~ 0.006)
- **Detection**: First prompt reliable (~250 pts)
- **Center**: Use `np.median(pts, axis=0)`

### Spatial Swap — Plate on raised silver stand
- **Prompts**: `"plate with red stripes"`, `"white plate with red circular stripes"`, `"plate"`
- **Surface Z**: ~ 0.034m
- **Filtering**: X < 0.60, mask area < 4000px, median Z < 0.06
- **Surface Z calc**: `np.percentile(pts[:,2], 85)`
- **CRITICAL**: MUST detect plate BEFORE grasping — arm occludes plate from camera after first grasp attempt
- **Placement**: `place_z = plate_surface_z + 0.10` (TCP offset correction)

### SCENE5 — Two plates side by side (libero_10)
- **Prompts**: `"plate"`, `"white plate"`, `"ceramic plate"`
- **Disambiguation**: LEFT plate = LOWER Y (min Y), RIGHT plate = HIGHER Y (max Y)

### Ramekin tasks — Plate surface Z
- **Filtering**: `plate_pts[plate_pts[:,2] > 0.005]` to exclude table outliers
- **Center XY**: `np.median`
- **Surface Z**: `np.percentile(pts[:,2], 85)`

---

## 2. Grasp & Placement

### TCP offset correction
`goto_pose(pos, quat)` moves the **wrist** to `pos`; fingertips are 10cm below wrist (TCP_OFFSET_Z). For top-down placement:

```python
place_z = plate_surface_z + 0.10  # fingertips land at plate surface
```

Using `+0.03` puts fingertips 7cm below plate surface — IK clamps, gripper releases in air, bowl drops and may miss.

### Plate push-off prevention
Placing a bowl > 1.5–2cm off plate center causes lateral forces that push the plate during descent. `check_ontop` then fails.

**Fix**: Correct EEF XY precisely before descend. Post-release, go directly to `goto_home_joint_position()` to avoid disturbing the bowl.

### EEF-body Y offset for top-down placement
Bowl body position is ~3.65cm below EEF in Y at placement height (~Z = 0.164). To land bowl at plate center:

```python
eef_y = plate_y + 0.0365   # Y offset correction
eef_x = plate_x + 0.018    # systematic X drift correction
```

Validated: `libero_spatial_swap/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` (42% Stage 2).

### GraspNet XY offset correction
When GraspNet grasps near the object rim (not centroid), the object hangs offset from TCP. Correct placement XY:

```python
xy_offset = grasp_pos[:2] - obj_center[:2]
place_xy = target_center[:2] + xy_offset
```

Validated: `libero_spatial_task/on_the_wooden_cabinet` (94% on clean trials).

### Always override GraspNet orientation with TOP_DOWN_QUAT
GraspNet often returns tilted quaternions that limit arm workspace. For transport to plate:

```python
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
# Use for ALL phases: approach, grasp, lift, transport, placement
```

Validated: `libero_spatial_swap/stove_to_plate` — GraspNet quat: 2/15, TOP_DOWN: 48/50 (96%).

**Exception — wide bowls beyond IK X range (X = 0.76–0.86)**: Top-down quat cannot reach. Use GraspNet quat for the ENTIRE sequence. Transport via `goto_home_joint_position()` then approach plate — stabilizes tilted-quat trajectory and reduces Y error from 130mm to ~20mm.

### Kinematic workspace at plate positions
- At X = 0.687, Y = -0.281 (spatial_swap plate): min wrist Z ~ 0.154 with top-down; fingertips at Z = 0.054 above plate
- IK X-clamp at ~0.75m: targets beyond X = 0.75 cannot be reached with top-down quat

---

## 3. Transport

### Pre-probe IK conditioning
Before grasping, pre-visit the plate at decreasing heights to seed the IK solver:

```python
for probe_z in [0.080, 0.060, 0.040, 0.020]:
    goto_pose(np.array([plate_x, plate_y, probe_z]), TOP_DOWN_QUAT)
goto_home_joint_position()  # return to home after probing
# NOW grasp the object — IK is conditioned for placement
```

Without this, `goto_pose` at plate XY can stall at Z ~ 0.16–0.18 even after home reset. Adds ~4 moves but prevents IK stalls entirely.

**When to use**: Placement target at low Z (< 0.10) with known XY (plate, stove top, cabinet surface).

### Skip-home transit (direct waypoint sequence)
When home transit is too slow or unnecessary:

```python
# Step 1: move to plate XY at LOW Z (arm near plate, IK on correct branch)
goto_pose(np.array([plate_x, plate_y, 0.15]), TOP_DOWN_QUAT)
# Step 2: rise to transit height
goto_pose(np.array([plate_x, plate_y, 0.30]), TOP_DOWN_QUAT)
# Step 3: descend to placement
goto_pose(np.array([plate_x, plate_y, place_z]), TOP_DOWN_QUAT)
```

Validated: `libero_spatial_task/next_to_ramekin` (58% Stage 2 with weak grasps).

### Joint-replay descent for extreme-Y plates
Left plate at Y ~ -0.295 to -0.321 (SCENE5) requires joint-replay descent.

Validated: `libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate` — seeds 51–53 all reward = 1.0 (v19).

### Sideways retreat after release
After placing on plate, retreat sideways (+0.30 X) before going home to avoid disturbing the placed object.

Validated: `libero_spatial_swap/next_to_plate` — the +0.30 X clears the placement area before diagonal sweep home.

### Physics settling after release
After releasing onto plate, add settling calls before reward check — without them, reward can check before bowl has fully stopped rolling.

---

## 4. Manipulation (Non-Pick-and-Place)

### Arm-body nudge for flat objects
For flat objects (plates, cream cheese) that are too flat for reliable grasp (gw < 0.04), nudge toward the target before grasping:

```python
nudge_pos = obj_center.copy()
nudge_pos[1] = target_y - 0.025  # push toward target

# Approach at mid-height — arm body contacts object top
j_nudge = solve_ik(nudge_pos + np.array([0, 0, 0.08]), TOP_DOWN_QUAT)
if j_nudge is not None:
    move_to_joints(j_nudge)
    # Descend — arm body pushes object along Y
    j_contact = solve_ik(nudge_pos + np.array([0, 0, 0.02]), TOP_DOWN_QUAT)
    if j_contact is not None:
        move_to_joints(j_contact)
# Now attempt grasp at nudged position
```

Validated: `libero_goal_task/push_the_plate_to_the_front_of_the_stove` — 31/50 (62%). Arm-body push was primary success mechanism for seeds with gw < 0.04.

### Lift with original grasp quat before reorienting
For thin flat objects (plates, cream cheese), GraspNet returns angled grasps. Reorienting to TOP_DOWN_QUAT immediately after close_gripper causes the object to drop.

```python
close_gripper()
# Lift with ORIGINAL grasp quat first
pos_lifted, grasp_quat = decompose_transform(grasp_pose_world)
goto_pose(pos_lifted + np.array([0, 0, 0.05]), grasp_quat)  # lift in grasp orientation
# Now reorient safely at lift height
goto_pose(pos_lifted + np.array([0, 0, 0.10]), TOP_DOWN_QUAT)
```

TOP_DOWN reorientation at Z = 0 rotates the gripper while near the table, causing slip. Lifting 5cm first clears the surface.

---

## 5. Validated Task Index

| Task | Suite | Best Score | Key Pattern |
|------|-------|------------|-------------|
| `put_the_bowl_on_the_plate` | goal_task | 88% (44/50) | TOP_DOWN_QUAT for all phases |
| `bowl_on_stove_to_plate` | spatial_swap | 96% (48/50) | TOP_DOWN override, XY correction |
| `bowl_on_cabinet_to_plate` | spatial_task | 94% | GraspNet XY offset correction |
| `bowl_between_plate_ramekin_to_plate` | spatial_swap | — | Pre-grasp plate detection |
| `bowl_next_to_cookie_box_to_plate` | spatial_swap | 20–30% | GraspNet quat (extreme X), home transit |
| `push_plate_to_stove` | goal_task | 62% (31/50) | Arm-body nudge, lift with grasp quat |
| `red_mug_on_left_plate` | libero_90 (SCENE5) | 100% (3/3) | Joint-replay descent, extreme-Y |
| `next_to_plate` | spatial_swap | 100% (50/50) | Lower-first XY correction, sideways retreat |

---

## 6. Quick Decision Tree

```
Is the plate a PLACEMENT TARGET or the MANIPULATED OBJECT?

PLACEMENT TARGET (place bowl on plate):
  1. Localize plate BEFORE grasping (arm occludes camera)
  2. Compute: place_z = plate_surface_z + 0.10 (TCP offset)
  3. Pre-probe plate at decreasing Z if plate Z < 0.10
  4. Use TOP_DOWN_QUAT unless bowl is at X > 0.75
  5. Correct EEF XY: +0.0365 Y, +0.018 X for bowl offset
  6. Descend precisely to plate center (< 1.5cm error)
  7. Release, settle, sideways retreat

MANIPULATED OBJECT (push/move plate):
  1. Plate is flat → gw likely < 0.04 → arm-body nudge
  2. If grasped: lift with original GraspNet quat before reorienting
  3. Nudge toward target before grasp attempt
```
