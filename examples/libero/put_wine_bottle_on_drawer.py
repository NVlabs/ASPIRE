report = {"task": libero.get_task_metadata()["task_ref"]}


def find_object(queries, camera="agentview"):
    for query in queries:
        found = localize_object(query, camera_name=camera)
        if found.ok and len(found.point_cloud.points) >= 20:
            return found
    return None


def cloud_quantiles(found, x_fraction=0.5, y_fraction=0.5, z_fraction=0.5):
    points = found.point_cloud.points
    xs = sorted(float(point[0]) for point in points)
    ys = sorted(float(point[1]) for point in points)
    zs = sorted(float(point[2]) for point in points)
    last = len(points) - 1
    return (
        xs[int(last * x_fraction)],
        ys[int(last * y_fraction)],
        zs[int(last * z_fraction)],
    )


def quaternion_product(first, second):
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def move_with_closed_gripper(target_pose):
    plan = plan_motion(target_pose)
    if not plan.ok:
        return None
    trajectory = plan.trajectory
    held_trajectory = Trajectory(
        joint_positions=trajectory.joint_positions,
        dt_s=trajectory.dt_s,
        joint_names=trajectory.joint_names,
        planner="program_side_grasp_hold",
        collision_aware=trajectory.collision_aware,
        expected_start=trajectory.expected_start,
        arm=trajectory.arm,
        embodiment=trajectory.embodiment,
        gripper_positions=[0.0 for waypoint in trajectory.joint_positions],
    )
    return execute_trajectory(held_trajectory)


state = get_robot_state()
frame = state.base_frame
downward = state.end_effector_poses["primary"].quaternion_wxyz
rotations = (
    ((0.7071067812, -0.7071067812, 0.0, 0.0), (0.0, 0.13, 0.0)),
    ((0.7071067812, 0.7071067812, 0.0, 0.0), (0.0, -0.13, 0.0)),
    ((0.7071067812, 0.0, -0.7071067812, 0.0), (0.13, 0.0, 0.012)),
)
held = False
held_orientation = downward

for rotation, pre_offset in rotations:
    if held:
        break
    bottle = find_object(("wine bottle", "dark green wine bottle"))
    if bottle is None:
        continue
    center = cloud_quantiles(bottle, 0.50, 0.50, 0.48)
    grasp_height = center[2] + 0.05 + pre_offset[2]
    side = quaternion_product(rotation, downward)
    orientation = quaternion_product(side, (0.7071067812, 0.0, 0.0, 0.7071067812))
    opened = open_gripper()
    above = move_to_pose(
        Pose(
            (
                center[0] + pre_offset[0],
                center[1] + pre_offset[1],
                grasp_height + 0.20,
            ),
            downward,
            frame,
        ),
        tolerance=0.03,
        max_steps=450,
    )
    if not above.ok:
        continue
    oriented = move_to_pose(
        Pose(
            (
                center[0] + pre_offset[0],
                center[1] + pre_offset[1],
                grasp_height + 0.20,
            ),
            orientation,
            frame,
        ),
        tolerance=0.04,
        max_steps=450,
    )
    if not oriented.ok:
        continue
    pregrasp = move_to_pose(
        Pose(
            (
                center[0] + pre_offset[0],
                center[1] + pre_offset[1],
                grasp_height,
            ),
            orientation,
            frame,
        ),
        tolerance=0.035,
        max_steps=400,
    )
    if not pregrasp.ok:
        continue
    grasped = move_to_pose(
        Pose((center[0], center[1], grasp_height), orientation, frame),
        tolerance=0.018,
        max_steps=300,
    )
    if not grasped.ok:
        continue
    closed = close_gripper()
    lifted = move_with_closed_gripper(
        Pose((center[0], center[1], grasp_height + 0.16), orientation, frame)
    )
    verification = find_object(("wine bottle", "dark green wine bottle"))
    if verification is not None:
        lifted_center = cloud_quantiles(verification, 0.50, 0.50, 0.48)
        held = lifted is not None and lifted.ok and lifted_center[2] > center[2] + 0.08
    else:
        held = False
    if held:
        held_orientation = orientation
    else:
        open_gripper()

report["side_grasped_bottle"] = held

if held:
    cabinet = find_object(
        (
            "black wooden drawer cabinet",
            "wooden cabinet with drawers",
            "drawer cabinet",
        )
    )
    report["localized_drawer"] = cabinet is not None
else:
    cabinet = None

if cabinet is not None:
    target = cloud_quantiles(cabinet, 0.50, 0.50, 0.94)
    above = move_with_closed_gripper(
        Pose((target[0], target[1], target[2] + 0.22), held_orientation, frame)
    )
    report["above_drawer"] = above is not None and above.ok
else:
    above = None

if above is not None and above.ok:
    placed = move_with_closed_gripper(
        Pose((target[0], target[1], target[2] + 0.09), held_orientation, frame)
    )
    report["lowered_bottle"] = placed is not None and placed.ok
else:
    placed = None

if placed is not None and placed.ok:
    released = open_gripper()
    report["released_bottle"] = released.ok
    pose = get_robot_state().end_effector_poses["primary"]
    retreat = move_to_pose(
        Pose(
            (pose.position[0], pose.position[1], pose.position[2] + 0.12),
            pose.quaternion_wxyz,
            pose.frame,
        ),
        tolerance=0.03,
        max_steps=300,
    )
    report["retreated"] = retreat.ok

result = report
