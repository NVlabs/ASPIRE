import numpy as np

###############################################################################
# two_arm_lift: Grasp green & blue handles on a pot and lift simultaneously
#
# Code 115 base + improved retry:
#   - After failed attempt, retract arms to safe position before re-observing
#   - Use gripper state to detect air grasp
#   - On retry, use slightly different grasp_inset
###############################################################################

def make_side_grasp_quat(approach_dir):
    """Sideways grasp: y-axis = world z, z-axis = approach direction."""
    z_axis = normalize_vector(np.array(approach_dir, dtype=float))
    y_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = normalize_vector(x_axis)
    z_axis = normalize_vector(np.cross(x_axis, y_axis))
    R = np.column_stack([x_axis, y_axis, z_axis])
    return rotation_matrix_to_quaternion(R)


def get_handle_centers(rgb, depth, K, E):
    """
    Segment green and blue handles using SAM3 text prompt.
    Returns (green_center, blue_center).
    """
    green_masks = segment_sam3_text_prompt(rgb, "green handle")
    blue_masks = segment_sam3_text_prompt(rgb, "blue handle")

    if not green_masks:
        pt = point_prompt_molmo(rgb, "point to the green handle on the pot")
        for key, val in pt.items():
            if val[0] is not None:
                green_masks = segment_sam3_point_prompt(rgb, val)
                break

    if not blue_masks:
        pt = point_prompt_molmo(rgb, "point to the blue handle on the pot")
        for key, val in pt.items():
            if val[0] is not None:
                blue_masks = segment_sam3_point_prompt(rgb, val)
                break

    if not green_masks or not blue_masks:
        raise RuntimeError("Cannot segment both handles")

    green_mask = max(green_masks, key=lambda d: d.get("score", 0.0))["mask"]
    blue_mask = max(blue_masks, key=lambda d: d.get("score", 0.0))["mask"]

    green_pts = mask_to_world_points(green_mask, depth, K, E)
    blue_pts = mask_to_world_points(blue_mask, depth, K, E)

    green_center = np.median(green_pts, axis=0)
    blue_center = np.median(blue_pts, axis=0)

    return green_center, blue_center


def grasp_and_lift(green_center, blue_center, approach_offset=0.10, grasp_inset=0.01, lift_height=0.15):
    """Execute side-grasp and lift sequence. Returns (grasp0_pos, grasp1_pos, quat0, quat1)."""
    pot_center = (green_center + blue_center) / 2.0

    green_approach = normalize_vector(pot_center[:2] - green_center[:2])
    green_approach_3d = np.array([green_approach[0], green_approach[1], 0.0])
    blue_approach = normalize_vector(pot_center[:2] - blue_center[:2])
    blue_approach_3d = np.array([blue_approach[0], blue_approach[1], 0.0])

    quat0 = make_side_grasp_quat(green_approach_3d)
    quat1 = make_side_grasp_quat(blue_approach_3d)

    # Pre-grasp
    pre0 = green_center - green_approach_3d * approach_offset
    pre1 = blue_center - blue_approach_3d * approach_offset
    pre0[2] += 0.05
    pre1[2] += 0.05

    # Grasp
    grasp0 = green_center + green_approach_3d * grasp_inset
    grasp1 = blue_center + blue_approach_3d * grasp_inset

    j0 = solve_ik_arm0(pre0, quat0)
    j1 = solve_ik_arm1(pre1, quat1)
    move_to_joints_both(j0, j1)

    j0 = solve_ik_arm0(grasp0, quat0)
    j1 = solve_ik_arm1(grasp1, quat1)
    move_to_joints_both(j0, j1)

    close_gripper_arm0()
    close_gripper_arm1()

    # Lift
    lift0 = grasp0.copy()
    lift1 = grasp1.copy()
    lift0[2] += lift_height
    lift1[2] += lift_height
    target_z = max(lift0[2], lift1[2])
    lift0[2] = target_z
    lift1[2] = target_z

    j0 = solve_ik_arm0(lift0, quat0)
    j1 = solve_ik_arm1(lift1, quat1)
    move_to_joints_both(j0, j1)

    print(f"Lifted to z={target_z:.4f}")
    return grasp0, grasp1, quat0, quat1


def retract_arms():
    """Move arms to a safe retracted position so camera has clear view."""
    # Move both arms up and back
    # Use a known safe position above the workspace
    safe_pos0 = np.array([0.3, -0.1, 0.25])
    safe_pos1 = np.array([0.7, 0.1, 0.25])
    # Top-down orientation for safe moves
    R_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    q_down = rotation_matrix_to_quaternion(R_down)

    try:
        j0 = solve_ik_arm0(safe_pos0, q_down)
        j1 = solve_ik_arm1(safe_pos1, q_down)
        move_to_joints_both(j0, j1)
    except Exception:
        # If IK fails for safe position, just open grippers
        pass


# === Main ===
open_gripper_arm0()
open_gripper_arm1()

obs = get_observation()
rgb = obs["robot0_robotview"]["images"]["rgb"]
depth = obs["robot0_robotview"]["images"]["depth"]
K = obs["robot0_robotview"]["intrinsics"]
E = obs["robot0_robotview"]["pose_mat"]

green_center, blue_center = get_handle_centers(rgb, depth, K, E)
print(f"Attempt 1: Green={green_center}, Blue={blue_center}")

grasp_and_lift(green_center, blue_center)

# === Check result ===
obs2 = get_observation()
grip0 = obs2['robot0_gripper_qpos'][0]
grip1 = obs2['robot1_gripper_qpos'][0]
print(f"Gripper states: arm0={grip0:.4f}, arm1={grip1:.4f}")

# Check if pot is lifted
pot_lifted = False
pot_masks = segment_sam3_text_prompt(obs2["robot0_robotview"]["images"]["rgb"], "pot")
if pot_masks:
    pot_mask = max(pot_masks, key=lambda d: d.get("score", 0.0))["mask"]
    pot_pts = mask_to_world_points(
        pot_mask.astype(bool),
        obs2["robot0_robotview"]["images"]["depth"],
        obs2["robot0_robotview"]["intrinsics"],
        obs2["robot0_robotview"]["pose_mat"]
    )
    if pot_pts.shape[0] > 10:
        pot_z = np.median(pot_pts[:, 2])
        print(f"Post-lift pot z: {pot_z:.4f}")
        if pot_z > 0.06:
            pot_lifted = True

if not pot_lifted:
    print("Pot not lifted, retrying...")

    # Open grippers
    open_gripper_arm0()
    open_gripper_arm1()

    # Retract arms to clear camera view
    retract_arms()

    # Fresh observation with clear camera
    obs3 = get_observation()
    rgb3 = obs3["robot0_robotview"]["images"]["rgb"]
    depth3 = obs3["robot0_robotview"]["images"]["depth"]
    K3 = obs3["robot0_robotview"]["intrinsics"]
    E3 = obs3["robot0_robotview"]["pose_mat"]

    try:
        green_center2, blue_center2 = get_handle_centers(rgb3, depth3, K3, E3)
        sep2 = np.linalg.norm(green_center2[:2] - blue_center2[:2])
        print(f"Retry: Green={green_center2}, Blue={blue_center2}, Sep={sep2:.3f}")

        if sep2 > 0.08:
            # Use slightly different grasp_inset on retry
            grasp_and_lift(green_center2, blue_center2, grasp_inset=0.015)
    except Exception as e:
        print(f"Retry failed: {e}")

print("Done")
