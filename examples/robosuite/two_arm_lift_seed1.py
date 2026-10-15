report = {}
observation = get_observation()
image_height = observation.cameras["agentview"].rgb.shape[0]
image_width = observation.cameras["agentview"].rgb.shape[1]

left_masks = segment_text("agentview", "handle on the left side of the pot")
left_segmentation = None
left_center_x = None
if left_masks.ok:
    for candidate in left_masks.segmentations:
        box = candidate.box_xyxy
        if box is None:
            continue
        center_x = (float(box[0]) + float(box[2])) / 2.0
        center_y = (float(box[1]) + float(box[3])) / 2.0
        if (
            center_x > 0.30 * image_width
            and center_x < 0.55 * image_width
            and center_y > 0.18 * image_height
        ):
            left_segmentation = candidate
            left_center_x = center_x
            break

right_masks = segment_text("agentview", "pot handle")
right_segmentation = None
if right_masks.ok and left_center_x is not None:
    for candidate in right_masks.segmentations:
        box = candidate.box_xyxy
        if box is None:
            continue
        center_x = (float(box[0]) + float(box[2])) / 2.0
        if center_x > left_center_x + 0.10 * image_width:
            right_segmentation = candidate
            break
report["segmented"] = left_segmentation is not None and right_segmentation is not None

if report["segmented"]:
    left_cloud = None
    right_cloud = None
    try:
        left_cloud = mask_to_point_cloud(
            left_segmentation, "agentview", target_frame="robot0_base"
        )
        right_cloud = mask_to_point_cloud(
            right_segmentation, "agentview", target_frame="robot0_base"
        )
    except ValueError:
        report["projection_ok"] = False
    report["projection_ok"] = left_cloud is not None and right_cloud is not None

if report["segmented"] and report["projection_ok"]:
    left_x = sorted([float(point[0]) for point in left_cloud.points])
    left_y = sorted([float(point[1]) for point in left_cloud.points])
    left_z = sorted([float(point[2]) for point in left_cloud.points])
    right_x = sorted([float(point[0]) for point in right_cloud.points])
    right_y = sorted([float(point[1]) for point in right_cloud.points])
    right_z = sorted([float(point[2]) for point in right_cloud.points])
    nl = len(left_x)
    nr = len(right_x)
    report["clouds_nonempty"] = nl > 0 and nr > 0

if report["segmented"] and report["projection_ok"] and report["clouds_nonempty"]:
    left = (left_x[nl // 10], left_y[nl // 2], left_z[nl // 5])
    right = (
        right_x[7 * nr // 10] - 0.002,
        right_y[nr // 2] - 0.005,
        right_z[nr // 5] - 0.010,
    )
    handle_y_separation = left[1] - right[1]
    if handle_y_separation > -0.03 and handle_y_separation < 0.03:
        left = (left[0] + 0.02, left[1], left[2])
        right = (right[0] - 0.02, right[1], right[2])
    report["left"] = left
    report["right"] = right

    open0 = open_gripper(arm="primary")
    open1 = open_gripper(arm="secondary")
    motion_ok = open0.ok and open1.ok
    state = get_robot_state()
    orientations = {
        "primary": state.end_effector_poses["primary"].quaternion_wxyz,
        "secondary": state.end_effector_poses["secondary"].quaternion_wxyz,
    }
    targets = {"primary": left, "secondary": right}
    for arm in ("primary", "secondary"):
        target = targets[arm]
        for height in (0.15, 0.05, 0.0):
            if motion_ok:
                target_pose = Pose(
                    (target[0], target[1], target[2] + height),
                    orientations[arm],
                    "robot0_base",
                )
                if height == 0.0:
                    contact_tolerance = 0.035
                    planned = plan_motion(target_pose, arm=arm)
                    if planned.ok:
                        moved = move_to_joints(
                            planned.trajectory.joint_positions[-1],
                            arm=arm,
                            tolerance=contact_tolerance,
                            max_steps=500,
                        )
                    else:
                        motion_ok = False
                else:
                    moved = move_to_pose(
                        target_pose,
                        arm=arm,
                        tolerance=0.02,
                        max_steps=500,
                    )
                if motion_ok:
                    motion_ok = moved.ok
    report["retry_used"] = False
    positioned = False
    if motion_ok:
        final_state = get_robot_state()
        primary_position = final_state.end_effector_poses["primary"].position
        secondary_position = final_state.end_effector_poses["secondary"].position
        primary_error = sum(
            [(float(primary_position[index]) - left[index]) ** 2 for index in range(3)]
        ) ** 0.5
        secondary_error = sum(
            [(float(secondary_position[index]) - right[index]) ** 2 for index in range(3)]
        ) ** 0.5
        positioned = primary_error < 0.03 and secondary_error < 0.03
        report["position_errors"] = (primary_error, secondary_error)
    report["positioned"] = positioned
else:
    positioned = False

if positioned:
    closed = set_grippers({"primary": 0.0, "secondary": 0.0})
    lift_ok = closed.ok
    if lift_ok:
        hold_state = get_robot_state()
        for hold_index in range(30):
            if lift_ok:
                held = step(
                    RobotAction(
                        {
                            arm: ArmCommand(
                                "joint_position",
                                hold_state.joint_positions[arm],
                                0.0,
                            )
                            for arm in ("primary", "secondary")
                        }
                    )
                )
                lift_ok = held.ok and not held.terminated and not held.truncated
    if lift_ok:
        state = get_robot_state()
        lift = move_synchronized(
            {
                arm: Pose(
                    state.end_effector_poses[arm].position + (0.0, 0.0, 0.15),
                    state.end_effector_poses[arm].quaternion_wxyz,
                    "robot0_base",
                )
                for arm in ("primary", "secondary")
            },
            tolerance=0.02,
            max_steps=500,
        )
        lift_ok = lift.ok
    report["lifted"] = lift_ok

result = report
