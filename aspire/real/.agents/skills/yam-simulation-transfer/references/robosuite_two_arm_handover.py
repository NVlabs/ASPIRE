# Code block 0
import numpy as np

# Helper functions
def move_arm0_pose(pos, quat):
    j = solve_ik_arm0(np.array(pos, dtype=float), np.array(quat, dtype=float))
    move_to_joints_arm0(j)

def move_arm1_pose(pos, quat):
    j = solve_ik_arm1(np.array(pos, dtype=float), np.array(quat, dtype=float))
    move_to_joints_arm1(j)

def move_arm0_path(points, quat):
    for p in points:
        move_arm0_pose(p, quat)

def move_arm1_path(points, quat):
    for p in points:
        move_arm1_pose(p, quat)

# Reference quaternions
DOWN_X_ARM0 = np.array([0.0, 0.707, 0.707, 0.0])  # Arm0 down, opening along X
DOWN_Y_ARM0 = np.array([0.0, 1.0, 0.0, 0.0])       # Arm0 down, opening along Y
DOWN_Y_ARM1 = np.array([0.0, 0.0, 1.0, 0.0])       # Arm1 down, opening along Y

# Step 1: Observe and detect hammer
obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]

open_gripper_arm0()
open_gripper_arm1()

# Segment hammer
masks = segment_sam3_text_prompt(rgb, "hammer")
if not masks:
    pt = point_prompt_molmo(rgb, "point to the hammer")
    px = list(pt.values())[0]
    masks = segment_sam3_point_prompt(rgb, px)

best = max(masks, key=lambda d: d.get("score", 0.0))
mask = best["mask"]

# Get 3D points and OBB
pts = mask_to_world_points(mask, depth, K, E)
obb = get_oriented_bounding_box_from_3d_points(pts)
center = np.array(obb["center"], dtype=float)
extent = np.array(obb["extent"], dtype=float)
R_obb = np.array(obb["R"], dtype=float)

# Find long axis (hammer length direction)
long_idx = int(np.argmax(extent))
long_axis = normalize_vector(R_obb[:, long_idx])
# Handle is toward +Y initially
if long_axis[1] < 0:
    long_axis = -long_axis

hammer_len = float(extent[long_idx])
obj_top_z = float(np.max(pts[:, 2]))
obj_min_z = float(np.min(pts[:, 2]))
table_z = obj_min_z

print(f"Hammer center: {center}")
print(f"Hammer length: {hammer_len:.3f}")
print(f"Long axis (head->handle): {long_axis}")
print(f"Table z: {table_z:.4f}, Top z: {obj_top_z:.4f}")

# Separate handle vs head points using projection along long axis
proj = (pts - center) @ long_axis
handle_pts = pts[proj > 0]
head_pts = pts[proj < 0]

handle_center = handle_pts.mean(axis=0) if len(handle_pts) > 0 else center + long_axis * 0.05
head_center = head_pts.mean(axis=0) if len(head_pts) > 0 else center - long_axis * 0.05

print(f"Handle center: {handle_center}")
print(f"Head center: {head_center}")

# Step 2: Arm0 picks up hammer near the HEAD so handle stays free for Arm1
# Grasp near head/center area
pick_point = 0.7 * head_center + 0.3 * center
pick_point[2] = table_z + 0.018  # just above table surface

# Gripper opening along X to grasp hammer lying along Y
grasp_quat_arm0 = DOWN_X_ARM0

pre_pick = pick_point.copy()
pre_pick[2] = obj_top_z + 0.12

# Move Arm1 to safe waiting position first
arm1_wait = np.array([1.18, -0.10, 0.30])
move_arm1_pose(arm1_wait, DOWN_Y_ARM1)

# Arm0: approach and grasp
move_arm0_pose(pre_pick, grasp_quat_arm0)
move_arm0_path(interpolate_segment(pre_pick, pick_point, step=0.02), grasp_quat_arm0)
close_gripper_arm0()
print("Arm0 grasped hammer near head")

# Lift up
lift_pos = pick_point.copy()
lift_pos[2] = max(obj_top_z + 0.15, 0.25)
move_arm0_path(interpolate_segment(pick_point, lift_pos, step=0.02), grasp_quat_arm0)
print(f"Lifted to z={lift_pos[2]:.3f}")

# Step 3: Reorient hammer so handle extends along +X toward Arm1
# With gripper opening along Y, the hammer (held perpendicular) extends along X
# So switch to DOWN_Y_ARM0 orientation
handover_quat_arm0 = DOWN_Y_ARM0

# Reorient at a safe intermediate position
intermediate = np.array([0.55, 0.0, 0.30])
move_arm0_pose(intermediate, grasp_quat_arm0)  # translate first
move_arm0_pose(intermediate, handover_quat_arm0)  # then rotate
print("Reoriented hammer along X-axis")

# Step 4: Move Arm0 to handover position
# Arm0 holds near head end; handle extends in +X toward Arm1
# Handover z must be between 0.15 and 0.20
handover_z = 0.175

# Position Arm0 so handle extends toward Arm1
# Arm0 gripper at ~0.70, handle extends to ~0.70 + hammer_len*0.6
arm0_handover = np.array([0.70, 0.0, handover_z])

waypoint_high = np.array([0.70, 0.0, 0.30])
move_arm0_pose(waypoint_high, handover_quat_arm0)
move_arm0_path(interpolate_segment(waypoint_high, arm0_handover, step=0.03), handover_quat_arm0)
print(f"Arm0 at handover: {arm0_handover}")

# Step 5: Arm1 approaches and grasps the handle
# Handle extends from Arm0 gripper in +X direction
# Arm1 grasps at roughly x = 0.70 + handle_offset
handle_offset = max(0.10, hammer_len * 0.4)
arm1_grasp_x = arm0_handover[0] + handle_offset
# Ensure at least 8cm separation
if arm1_grasp_x - arm0_handover[0] < 0.09:
    arm1_grasp_x = arm0_handover[0] + 0.10

arm1_grasp_pos = np.array([arm1_grasp_x, 0.0, handover_z])
arm1_pre = arm1_grasp_pos.copy()
arm1_pre[2] = handover_z + 0.12

print(f"Arm1 target: {arm1_grasp_pos}, separation: {arm1_grasp_x - arm0_handover[0]:.3f}m")

move_arm1_pose(arm1_pre, DOWN_Y_ARM1)
move_arm1_path(interpolate_segment(arm1_pre, arm1_grasp_pos, step=0.02), DOWN_Y_ARM1)
close_gripper_arm1()
print("Arm1 grasped handle")

# Step 6: Release from Arm0 and retreat
open_gripper_arm0()
print("Arm0 released hammer")

# Arm0 retreat upward first to avoid collision
retreat0 = arm0_handover.copy()
retreat0[2] += 0.12
move_arm0_path(interpolate_segment(arm0_handover, retreat0, step=0.02), handover_quat_arm0)
retreat0b = retreat0.copy()
retreat0b[0] -= 0.15
move_arm0_pose(retreat0b, handover_quat_arm0)
print("Arm0 retreated")

# Arm1 lift slightly to secure
post1 = arm1_grasp_pos.copy()
post1[2] += 0.05
move_arm1_path(interpolate_segment(arm1_grasp_pos, post1, step=0.02), DOWN_Y_ARM1)

print(f"Handover complete at z={handover_z} (target: 0.15-0.20)")
print(f"Arm1 holding hammer handle at {arm1_grasp_pos}")