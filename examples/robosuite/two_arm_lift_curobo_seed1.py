report = {}
motion = MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo")
contact_motion = MotionStrategy()
observation = get_observation()

left_masks = segment_text("agentview", "handle on the left side of the pot")
left_segmentation = None
if left_masks.ok:
    for candidate in left_masks.segmentations:
        box = candidate.box_xyxy
        center_x = (float(box[0]) + float(box[2])) / 2.0
        center_y = (float(box[1]) + float(box[3])) / 2.0
        if center_x > 170.0 and center_x < 260.0 and center_y > 100.0:
            left_segmentation = candidate
            break

right_masks = segment_text("agentview", "pot handle")
right_segmentation = right_masks.segmentations[0] if right_masks.ok else None
report["segmented"] = left_segmentation is not None and right_segmentation is not None

if report["segmented"]:
    left_cloud = mask_to_point_cloud(left_segmentation, "agentview", target_frame="robot0_base")
    right_cloud = mask_to_point_cloud(right_segmentation, "agentview", target_frame="robot0_base")
    left_x = sorted([float(point[0]) for point in left_cloud.points])
    left_y = sorted([float(point[1]) for point in left_cloud.points])
    left_z = sorted([float(point[2]) for point in left_cloud.points])
    right_x = sorted([float(point[0]) for point in right_cloud.points])
    right_y = sorted([float(point[1]) for point in right_cloud.points])
    right_z = sorted([float(point[2]) for point in right_cloud.points])
    nl = len(left_x)
    nr = len(right_x)
    left = (left_x[nl // 10], left_y[nl // 2], left_z[nl // 5])
    right = (right_x[7 * nr // 10], right_y[nr // 2], right_z[nr // 5])
    right = (right[0] - 0.002, right[1] - 0.005, right[2] - 0.010)
    report["left"] = left
    report["right"] = right

    open0 = open_gripper(arm="primary")
    open1 = open_gripper(arm="secondary")
    state = get_robot_state()
    orientations = {
        "primary": state.end_effector_poses["primary"].quaternion_wxyz,
        "secondary": state.end_effector_poses["secondary"].quaternion_wxyz,
    }
    targets = {"primary": left, "secondary": right}
    for arm in ("primary", "secondary"):
        target = targets[arm]
        for height in (0.15, 0.05, 0.0):
            moved = move_to_pose(
                Pose(
                    (target[0], target[1], target[2] + height),
                    orientations[arm],
                    "robot0_base",
                ),
                arm=arm,
                tolerance=0.02,
                max_steps=500,
                strategy=motion if height == 0.15 else contact_motion,
            )
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
    report["positioned"] = positioned
    report["position_errors"] = (primary_error, secondary_error)
else:
    positioned = False

if positioned:
    closed = set_grippers({"primary": 0.0, "secondary": 0.0})
    lifted = closed.ok
    if lifted:
        state = get_robot_state()
        lift = move_synchronized(
            {
                arm: Pose(
                    state.end_effector_poses[arm].position + (0.0, 0.0, 0.20),
                    state.end_effector_poses[arm].quaternion_wxyz,
                    "robot0_base",
                )
                for arm in ("primary", "secondary")
            },
            tolerance=0.02,
            max_steps=500,
        )
        lifted = lift.ok
    report["lifted"] = lifted

result = report
