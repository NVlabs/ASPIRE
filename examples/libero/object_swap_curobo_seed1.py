report = {}
motion = MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo")
contact_motion = MotionStrategy()
opened = open_gripper()
target = localize_object("blue soup can", camera_name="agentview")
report["localized_target"] = target.ok

if target.ok:
    points = target.point_cloud.points
    xs = sorted([float(point[0]) for point in points])
    ys = sorted([float(point[1]) for point in points])
    zs = sorted([float(point[2]) for point in points])
    count = len(points)
    grasp_center = (xs[count // 4], ys[count // 4], zs[9 * count // 10] - 0.025)
    state = get_robot_state()
    downward = state.end_effector_poses["primary"].quaternion_wxyz
    pregrasp = Pose(
        position=(grasp_center[0], grasp_center[1], grasp_center[2] + 0.20),
        quaternion_wxyz=downward,
        frame="robot_base",
    )
    moved_pregrasp = move_to_pose(pregrasp, tolerance=0.025, max_steps=450, strategy=motion)
    report["moved_pregrasp"] = moved_pregrasp.ok
else:
    moved_pregrasp = None

if moved_pregrasp is not None and moved_pregrasp.ok:
    grasp_pose = Pose(
        position=grasp_center,
        quaternion_wxyz=downward,
        frame="robot_base",
    )
    moved_grasp = move_to_pose(
        grasp_pose,
        tolerance=0.02,
        max_steps=350,
        strategy=contact_motion,
    )
    report["moved_grasp"] = moved_grasp.ok
else:
    moved_grasp = None

if moved_grasp is not None and moved_grasp.ok:
    closed = close_gripper()
    report["closed"] = closed.ok
    report["gripper_after_close"] = get_robot_state().gripper_positions["primary"]
else:
    closed = None

lifted = closed
if closed is not None and closed.ok:
    for lift_index in range(5):
        state = get_robot_state()
        pose = state.end_effector_poses["primary"]
        lift_pose = Pose(
            position=pose.position + (0.0, 0.0, 0.04),
            quaternion_wxyz=pose.quaternion_wxyz,
            frame=pose.frame,
        )
        lifted = move_to_pose(
            lift_pose,
            tolerance=0.02,
            max_steps=150,
            strategy=contact_motion,
        )
        if not lifted.ok:
            break
    report["lifted"] = lifted.ok
    report["lift_steps"] = lift_index + 1

if lifted is not None and lifted.ok:
    basket = localize_object("the basket", camera_name="agentview")
    report["localized_basket"] = basket.ok
else:
    basket = None

if basket is not None and basket.ok:
    basket_center = basket.geometry.pose.position
    state = get_robot_state()
    held = state.end_effector_poses["primary"]
    midpoint = Pose(
        position=(held.position[0], (held.position[1] + basket_center[1]) / 2.0, 0.36),
        quaternion_wxyz=held.quaternion_wxyz,
        frame=held.frame,
    )
    moved_midpoint = move_to_pose(
        midpoint,
        tolerance=0.03,
        max_steps=350,
        strategy=contact_motion,
    )
    report["moved_midpoint"] = moved_midpoint.ok
else:
    moved_midpoint = None

if moved_midpoint is not None and moved_midpoint.ok:
    state = get_robot_state()
    held = state.end_effector_poses["primary"]
    above = Pose(
        position=(basket_center[0], basket_center[1], 0.36),
        quaternion_wxyz=held.quaternion_wxyz,
        frame=held.frame,
    )
    moved_above = move_to_pose(
        above,
        tolerance=0.03,
        max_steps=350,
        strategy=contact_motion,
    )
    report["moved_above"] = moved_above.ok
else:
    moved_above = None

if moved_above is not None and moved_above.ok:
    state = get_robot_state()
    held = state.end_effector_poses["primary"]
    release_pose = Pose(
        position=(basket_center[0] + 0.05, basket_center[1], basket_center[2] + 0.22),
        quaternion_wxyz=held.quaternion_wxyz,
        frame=held.frame,
    )
    moved_release = move_to_pose(
        release_pose,
        tolerance=0.03,
        max_steps=250,
        strategy=contact_motion,
    )
    report["moved_release"] = moved_release.ok
else:
    moved_release = None

if moved_release is not None and moved_release.ok:
    released = open_gripper()
    report["released"] = released.ok
    state = get_robot_state()
    pose = state.end_effector_poses["primary"]
    retreat = Pose(
        position=pose.position + (0.0, 0.0, 0.12),
        quaternion_wxyz=pose.quaternion_wxyz,
        frame=pose.frame,
    )
    retreated = move_to_pose(retreat, tolerance=0.03, max_steps=250, strategy=motion)
    report["retreated"] = retreated.ok

result = report
