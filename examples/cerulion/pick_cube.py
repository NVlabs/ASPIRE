# Pick up the cube on the Cerulion simulated arm (benchmark `cerulion`).
#
# This task has no perception: the cube position comes from task metadata
# (`task_supplied_cube_position`). The program solves the arm's planar IK with
# the tool pointing straight down, then opens, moves 8 cm above the cube,
# descends, closes and lifts to 25 cm. See docs/cerulion.md.
import math

context = get_task_context()
cube = context.metadata["task_supplied_cube_position"]
links = context.metadata["kinematics"]
H, L1, L2, L3 = links["H"], links["L1"], links["L2"], links["L3"]


def tool_down_ik(x, y, z):
    """Joint target putting the tool tip at (x, y, z) with the tool pointing straight down."""
    reach = math.hypot(x, y)
    wrist_r = reach
    wrist_z = z + L3 - H
    cos_elbow = (wrist_r * wrist_r + wrist_z * wrist_z - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if abs(cos_elbow) > 1.0:
        raise ValueError(f"target ({x:.3f}, {y:.3f}, {z:.3f}) is out of reach")
    elbow = -math.acos(cos_elbow)
    shoulder = math.atan2(wrist_z, wrist_r) - math.atan2(
        L2 * math.sin(elbow), L1 + L2 * math.cos(elbow)
    )
    wrist = -math.pi / 2.0 - shoulder - elbow
    return [math.atan2(y, x), shoulder, elbow, wrist, 0.0, 0.0]


stages = {}
stages["open"] = open_gripper()
stages["pregrasp"] = move_to_joints(tool_down_ik(cube[0], cube[1], cube[2] + 0.08))
stages["descend"] = move_to_joints(tool_down_ik(cube[0], cube[1], cube[2]))
stages["grasp"] = close_gripper()
stages["lift"] = move_to_joints(tool_down_ik(cube[0], cube[1], 0.25))

failed = [name for name, outcome in stages.items() if not outcome.ok]
state = get_robot_state()
result = {
    "failed_stages": failed,
    "end_effector": [float(v) for v in state.end_effector_poses["primary"].position],
    "gripper": state.gripper_positions["primary"],
}
print(f"pick_cube: failed stages {failed}; end effector {result['end_effector']}")
