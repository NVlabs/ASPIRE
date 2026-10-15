report = {}
motion = MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo")
contact_motion = MotionStrategy()
head_masks = segment_text("agentview", "hammer head")
head = None
if head_masks.ok and head_masks.segmentations:
    cloud = mask_to_point_cloud(
        head_masks.segmentations[0], "agentview", target_frame="robot0_base"
    )
    xs = sorted(float(point[0]) for point in cloud.points)
    ys = sorted(float(point[1]) for point in cloud.points)
    zs = sorted(float(point[2]) for point in cloud.points)
    count = len(xs)
    head = (xs[count // 2], ys[count // 2], zs[count // 2] - 0.025)
report["head"] = head

if head is not None:
    state = get_robot_state()
    primary_quaternion = state.end_effector_poses["primary"].quaternion_wxyz
    secondary_quaternion = state.end_effector_poses["secondary"].quaternion_wxyz

    receiver_raw = (
        secondary_quaternion[0],
        secondary_quaternion[1],
        secondary_quaternion[2] - 1.0,
        secondary_quaternion[3],
    )
    receiver_norm = sum(value * value for value in receiver_raw) ** 0.5
    receiver_quaternion = tuple(value / receiver_norm for value in receiver_raw)

    secondary_home = state.end_effector_poses["secondary"].position
    move_to_pose(
        Pose(secondary_home, receiver_quaternion, "robot0_base"),
        arm="secondary",
        tolerance=0.02,
        max_steps=700,
        strategy=motion,
    )

    open_gripper(arm="primary")
    for height in (0.15, 0.05, 0.0):
        move_to_pose(
            Pose(
                (head[0], head[1], head[2] + height),
                primary_quaternion,
                "robot0_base",
            ),
            arm="primary",
            tolerance=0.02,
            max_steps=500,
            strategy=motion if height == 0.15 else contact_motion,
        )
    close_gripper(arm="primary")
    for position in (
        (head[0], head[1], 0.10),
        (head[0], head[1], 0.25),
    ):
        move_to_pose(
            Pose(position, primary_quaternion, "robot0_base"),
            arm="primary",
            tolerance=0.02,
            max_steps=500,
            strategy=contact_motion,
        )

    rotation_half_angles = (
        (0.9238795325, 0.3826834324),
        (0.7071067812, 0.7071067812),
        (0.3826834324, 0.9238795325),
        (0.0, 1.0),
        (-0.3826834324, 0.9238795325),
        (-0.7071067812, 0.7071067812),
    )
    presentation_quaternion = primary_quaternion
    for cosine, sine in rotation_half_angles:
        w, x, y, z = primary_quaternion
        presentation_quaternion = (
            cosine * w - sine * z,
            cosine * x - sine * y,
            cosine * y + sine * x,
            cosine * z + sine * w,
        )
        move_to_pose(
            Pose(
                (head[0], head[1], 0.25),
                presentation_quaternion,
                "robot0_base",
            ),
            arm="primary",
            tolerance=0.02,
            max_steps=700,
            strategy=contact_motion,
        )

    for position in (
        (0.55, 0.02, 0.25),
        (0.68, 0.0, 0.25),
        (0.72, 0.0, 0.15),
    ):
        move_to_pose(
            Pose(position, presentation_quaternion, "robot0_base"),
            arm="primary",
            tolerance=0.02,
            max_steps=500,
            strategy=contact_motion,
        )

    open_gripper(arm="secondary")
    for position in (
        (1.05, -0.03, 0.30),
        (0.95, -0.03, 0.30),
        (0.88, -0.03, 0.30),
        (0.84, -0.03, 0.22),
        (0.84, -0.03, 0.13),
    ):
        move_to_pose(
            Pose(position, receiver_quaternion, "robot0_base"),
            arm="secondary",
            tolerance=0.02,
            max_steps=700,
            strategy=motion if position[0] == 1.05 else contact_motion,
        )
    close_gripper(arm="secondary")
    open_gripper(arm="primary")
    report["transferred"] = True

result = report
