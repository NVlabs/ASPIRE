report = {}
opened = open_gripper()
cube = localize_object("red cube", camera_name="agentview")
report["localized"] = cube.ok

if cube.ok:
    points = cube.point_cloud.points
    xs = sorted([float(point[0]) for point in points])
    ys = sorted([float(point[1]) for point in points])
    zs = sorted([float(point[2]) for point in points])
    count = len(points)
    center = (xs[count // 2], ys[count // 2], zs[count // 2])
    state = get_robot_state()
    quat = state.end_effector_poses["primary"].quaternion_wxyz
    above = Pose(
        position=(center[0], center[1], center[2] + 0.12),
        quaternion_wxyz=quat,
        frame=state.base_frame,
    )
    approached = move_to_pose(above, tolerance=0.02, max_steps=300)
    report["approached"] = approached.ok
else:
    approached = None

if approached is not None and approached.ok:
    grasp = Pose(
        position=(center[0], center[1], center[2] - 0.005),
        quaternion_wxyz=quat,
        frame=state.base_frame,
    )
    descended = move_to_pose(grasp, tolerance=0.015, max_steps=250)
    report["descended"] = descended.ok
else:
    descended = None

if descended is not None and descended.ok:
    closed = close_gripper()
    report["closed"] = closed.ok
    pose = get_robot_state().end_effector_poses["primary"]
    lifted = move_to_pose(
        Pose(
            position=pose.position + (0.0, 0.0, 0.18),
            quaternion_wxyz=pose.quaternion_wxyz,
            frame=pose.frame,
        ),
        tolerance=0.02,
        max_steps=350,
    )
    report["lifted"] = lifted.ok

result = report
