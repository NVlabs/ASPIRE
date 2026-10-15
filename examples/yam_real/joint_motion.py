state = get_robot_state()
target = state.joint_positions["right"].copy()
target[0] = target[0] + 0.03
result = move_to_joints(target, arm="right")
