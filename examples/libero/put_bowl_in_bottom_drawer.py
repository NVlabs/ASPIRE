report = {"task": libero.get_task_metadata()["task_ref"]}


def find_object(queries, camera="agentview"):
    for query in queries:
        found = localize_object(query, camera_name=camera)
        if found.ok and len(found.point_cloud.points) >= 20:
            return found
    return None


def point_quantiles(points, x_fraction=0.5, y_fraction=0.5, z_fraction=0.5):
    xs = sorted(float(point[0]) for point in points)
    ys = sorted(float(point[1]) for point in points)
    zs = sorted(float(point[2]) for point in points)
    last = len(points) - 1
    return (
        xs[int(last * x_fraction)],
        ys[int(last * y_fraction)],
        zs[int(last * z_fraction)],
    )


def cloud_quantiles(found, x_fraction=0.5, y_fraction=0.5, z_fraction=0.5):
    return point_quantiles(found.point_cloud.points, x_fraction, y_fraction, z_fraction)


def move_with_closed_gripper(target_pose):
    plan = plan_motion(target_pose)
    if not plan.ok:
        return None
    trajectory = plan.trajectory
    held_trajectory = Trajectory(
        joint_positions=trajectory.joint_positions,
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner="program_bowl_hold",
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        arm=trajectory.arm,
        embodiment=trajectory.embodiment,
        gripper_positions=[0.0 for waypoint in trajectory.joint_positions],
    )
    return execute_trajectory(held_trajectory)


def quaternion_product(first, second):
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


bowl = find_object(("black bowl", "dark bowl"))
report["localized_bowl"] = bowl is not None

if bowl is not None:
    # Pinch the positive-Y rim instead of descending into the cup.
    bowl_grasp = cloud_quantiles(bowl, 0.50, 0.75, 0.88)
    bowl_center = cloud_quantiles(bowl)
    state = get_robot_state()
    downward = state.end_effector_poses["primary"].quaternion_wxyz
    frame = state.base_frame
    opened = open_gripper()
    approach = move_to_pose(
        Pose(
            (bowl_grasp[0], bowl_grasp[1], bowl_grasp[2] + 0.15),
            downward,
            frame,
        ),
        tolerance=0.025,
        max_steps=350,
    )
    report["approached_bowl"] = approach.ok
else:
    approach = None

if approach is not None and approach.ok:
    contact = move_to_pose(
        Pose(bowl_grasp, downward, frame),
        tolerance=0.018,
        max_steps=300,
    )
    report["contacted_bowl"] = contact.ok
else:
    contact = None

if contact is not None and contact.ok:
    closed = close_gripper()
    report["grasped_bowl"] = closed.ok
else:
    closed = None

lifted = closed
if closed is not None and closed.ok:
    for lift in (0.06, 0.12, 0.20):
        lifted = move_with_closed_gripper(
            Pose(
                (bowl_grasp[0], bowl_grasp[1], bowl_grasp[2] + lift),
                downward,
                frame,
            )
        )
        if lifted is None or not lifted.ok:
            break
    verification = find_object(("black bowl", "dark bowl"))
    if verification is not None:
        lifted_center = cloud_quantiles(verification)
        held = (
            lifted is not None
            and lifted.ok
            and lifted_center[2] > bowl_center[2] + 0.10
        )
    else:
        held = False
    report["lifted_bowl"] = held
else:
    held = False

if held:
    drawer = find_object(
        (
            "inside of the open bottom drawer",
            "open bottom drawer",
            "bottom drawer interior",
        )
    )
    report["localized_drawer"] = drawer is not None
else:
    drawer = None

if drawer is not None:
    drawer_center = cloud_quantiles(drawer, 0.50, 0.48, 0.30)
    above = move_with_closed_gripper(
        Pose(
            (drawer_center[0], drawer_center[1], drawer_center[2] + 0.22),
            downward,
            frame,
        )
    )
    report["above_drawer"] = above is not None and above.ok
else:
    above = None

if above is not None and above.ok:
    lowered = move_with_closed_gripper(
        Pose(
            (drawer_center[0], drawer_center[1], drawer_center[2] + 0.04),
            downward,
            frame,
        )
    )
    report["lowered_bowl"] = lowered is not None and lowered.ok
else:
    lowered = None

if lowered is not None:
    released = open_gripper()
    report["released_bowl"] = released.ok
    pose = get_robot_state().end_effector_poses["primary"]
    retreat = move_to_pose(
        Pose(
            (pose.position[0], pose.position[1], pose.position[2] + 0.15),
            pose.quaternion_wxyz,
            pose.frame,
        ),
        tolerance=0.03,
        max_steps=300,
    )
    report["retreated_from_drawer"] = retreat.ok
else:
    retreat = None

if retreat is not None and retreat.ok:
    # The open drawer interior is already localized for placement. Its front
    # face shares X/Y with that cloud and is contacted near the lower edge.
    handle_center = (
        drawer_center[0] + 0.05,
        drawer_center[1] + 0.11,
        drawer_center[2] + 0.02,
    )
    handle = True
else:
    handle = None

if handle is not None:
    push_start = move_to_pose(
        Pose(
            (handle_center[0], handle_center[1] + 0.08, handle_center[2] - 0.03),
            downward,
            frame,
        ),
        tolerance=0.025,
        max_steps=350,
    )
    report["approached_handle"] = push_start.ok
else:
    push_start = None

if push_start is not None and push_start.ok:
    forward = quaternion_product((0.7071067812, -0.7071067812, 0.0, 0.0), downward)
    pushed = move_to_pose(
        Pose(
            (handle_center[0], handle_center[1] + 0.08, handle_center[2] - 0.03),
            forward,
            frame,
        ),
        tolerance=0.04,
        max_steps=400,
    )
    for offset in (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40):
        pushed = move_to_pose(
            Pose(
                (
                    handle_center[0],
                    handle_center[1] + 0.08 - offset,
                    handle_center[2] - 0.03,
                ),
                forward,
                frame,
            ),
            tolerance=0.02,
            max_steps=400,
        )
    for push_repeat in range(3):
        pushed = move_to_pose(
            Pose(
                (
                    handle_center[0],
                    handle_center[1] - 0.40,
                    handle_center[2] - 0.03,
                ),
                forward,
                frame,
            ),
            tolerance=0.02,
            max_steps=500,
        )
    report["closed_drawer"] = pushed.ok

result = report
