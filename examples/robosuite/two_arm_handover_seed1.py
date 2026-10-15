# Fixed seed-agnostic handover policy validated through cumulative public-API stages.
report = {
    "stage": "canonical_live_transfer",
    "canonicalized": False,
    "giver_release_commanded": False,
    "handle_toward_receiver": False,
    "motion_completed": False,
    "pickup_prompt": None,
    "receiver_close_commanded": False,
    "receiver_conditioned": False,
}
observation = get_observation()
image_height = observation.cameras["agentview"].rgb.shape[0]
image_width = observation.cameras["agentview"].rgb.shape[1]

# Preserve the pickup selector that reached an elevated evaluator grasp on
# 37/39 attempt 16-34 rollouts. The presentation experiment changes the
# transfer strategy, not this already-discriminated pickup stage.
pickup_masks = segment_text("agentview", "wooden hammer handle")
pickup_segmentation = None
pickup_prompt = None
if pickup_masks.ok:
    for candidate in pickup_masks.segmentations:
        box = candidate.box_xyxy
        if box is None:
            continue
        center_x = (float(box[0]) + float(box[2])) / 2.0
        center_y = (float(box[1]) + float(box[3])) / 2.0
        box_width = float(box[2]) - float(box[0])
        box_height = float(box[3]) - float(box[1])
        if (
            center_x < 0.30 * image_width
            and center_y > 0.35 * image_height
            and box_width >= box_height
            and float(candidate.score) >= 0.10
        ):
            pickup_segmentation = candidate
            pickup_prompt = "wooden hammer handle"
            break

if pickup_segmentation is None:
    head_masks = segment_text("agentview", "hammer head")
    if head_masks.ok:
        for candidate in head_masks.segmentations:
            box = candidate.box_xyxy
            if box is None:
                continue
            center_x = (float(box[0]) + float(box[2])) / 2.0
            center_y = (float(box[1]) + float(box[3])) / 2.0
            if center_x < 0.30 * image_width and center_y > 0.35 * image_height:
                pickup_segmentation = candidate
                pickup_prompt = "hammer head"
                break

pickup_point = None
if pickup_segmentation is not None:
    try:
        pickup_cloud = mask_to_point_cloud(
            pickup_segmentation,
            "agentview",
            target_frame="robot0_base",
        )
    except ValueError:
        pickup_cloud = None
    if pickup_cloud is not None and len(pickup_cloud.points) > 0:
        xs = sorted(float(point[0]) for point in pickup_cloud.points)
        ys = sorted(float(point[1]) for point in pickup_cloud.points)
        zs = sorted(float(point[2]) for point in pickup_cloud.points)
        count = len(xs)
        depth = 0.025 if pickup_prompt == "hammer head" else 0.005
        pickup_point = (xs[count // 2], ys[count // 2], zs[count // 2] - depth)

report["pickup_prompt"] = pickup_prompt
report["pickup_point"] = pickup_point
if pickup_point is not None:
    state = get_robot_state()
    primary_quaternion = state.end_effector_poses["primary"].quaternion_wxyz
    primary_home = state.end_effector_poses["primary"].position
    secondary_pose = state.end_effector_poses["secondary"]
    transfer_x = (float(primary_home[0]) + float(secondary_pose.position[0])) / 2.0
    transfer_y = (float(primary_home[1]) + float(secondary_pose.position[1])) / 2.0
    receiver_quaternion = secondary_pose.quaternion_wxyz
    motion_ok = open_gripper(arm="secondary").ok
    if motion_ok:
        motion_ok = open_gripper(arm="primary").ok

    # Preserve the receiver's demonstrated reachable frame. Its local X axis
    # is the parallel-jaw closing direction. Canonicalize the hammer long axis
    # perpendicular to that live horizontal direction instead of forcing an
    # unreachable published receiver quaternion.
    rw, rx, ry, rz = (float(value) for value in receiver_quaternion)
    closing_x = 1.0 - 2.0 * (ry * ry + rz * rz)
    closing_y = 2.0 * (rx * ry + rw * rz)
    approach_axis = (
        2.0 * (rx * rz + rw * ry),
        2.0 * (ry * rz - rw * rx),
        1.0 - 2.0 * (rx * rx + ry * ry),
    )
    closing_horizontal_norm = (closing_x * closing_x + closing_y * closing_y) ** 0.5
    canonical_quaternion = None
    if closing_horizontal_norm > 0.5:
        handle_x = closing_y / closing_horizontal_norm
        handle_y = -closing_x / closing_horizontal_norm
        half_cos = ((1.0 + handle_x) / 2.0) ** 0.5
        half_sin = ((1.0 - handle_x) / 2.0) ** 0.5
        if handle_y < 0.0:
            half_sin = -half_sin
        canonical_quaternion = (0.0, half_cos, half_sin, 0.0)
    else:
        motion_ok = False
    report["receiver_closing_axis_xy"] = (closing_x, closing_y)
    report["receiver_approach_axis"] = approach_axis
    report["canonical_handle_axis_xy"] = (
        None if canonical_quaternion is None else handle_x,
        None if canonical_quaternion is None else handle_y,
    )
    report["receiver_conditioned"] = motion_ok

    for height in (0.15, 0.05, 0.0):
        if motion_ok:
            target = Pose(
                (pickup_point[0], pickup_point[1], pickup_point[2] + height),
                primary_quaternion,
                "robot0_base",
            )
            if height == 0.0:
                planned = plan_motion(target, arm="primary")
                if planned.ok:
                    moved = move_to_joints(
                        planned.trajectory.joint_positions[-1],
                        arm="primary",
                        tolerance=0.04,
                        max_steps=500,
                    )
                else:
                    motion_ok = False
            else:
                moved = move_to_pose(
                    target,
                    arm="primary",
                    tolerance=0.02,
                    max_steps=500,
                )
            if motion_ok:
                motion_ok = moved.ok

    if motion_ok:
        motion_ok = close_gripper(arm="primary").ok
    if motion_ok:
        pickup_state = get_robot_state()
        pickup_position = pickup_state.end_effector_poses["primary"].position
    for height in (0.08, 0.16, 0.20, 0.24, 0.28, 0.32):
        if motion_ok:
            moved = move_to_pose(
                Pose(
                    pickup_position + (0.0, 0.0, height),
                    primary_quaternion,
                    "robot0_base",
                ),
                arm="primary",
                tolerance=0.02,
                max_steps=500,
            )
            motion_ok = moved.ok

    # Hybrid design D: use the proven pickup, then canonicalize at clearance
    # using the ASPIRE giver orientation before entering the transfer pose.
    for position in ((0.55, 0.02, 0.30),):
        if motion_ok:
            planned = plan_motion(
                Pose(position, primary_quaternion, "robot0_base"),
                arm="primary",
            )
            if planned.ok:
                moved = move_to_joints(
                    planned.trajectory.joint_positions[-1],
                    arm="primary",
                    tolerance=0.04,
                    max_steps=700,
                )
                motion_ok = moved.ok
            else:
                motion_ok = False

    if motion_ok:
        canonical_plan = plan_motion(
            Pose((0.55, 0.02, 0.30), canonical_quaternion, "robot0_base"),
            arm="primary",
        )
        if canonical_plan.ok:
            canonical_move = move_to_joints(
                canonical_plan.trajectory.joint_positions[-1],
                arm="primary",
                tolerance=0.02,
                max_steps=700,
            )
            motion_ok = canonical_move.ok
            report["canonicalized"] = canonical_move.ok
        else:
            motion_ok = False

    for position in (
        (transfer_x, transfer_y, 0.30),
        (transfer_x, transfer_y, 0.175),
    ):
        if motion_ok:
            planned = plan_motion(
                Pose(position, canonical_quaternion, "robot0_base"),
                arm="primary",
            )
            if planned.ok:
                moved = move_to_joints(
                    planned.trajectory.joint_positions[-1],
                    arm="primary",
                    tolerance=0.02,
                    max_steps=700,
                )
                motion_ok = moved.ok
            else:
                motion_ok = False

    report["motion_completed"] = motion_ok
    report["post_geometry"] = None
    post_geometry = None
    if motion_ok:
        post_observation = get_observation()
        post_masks = segment_text("agentview", "wooden hammer handle")
        post_segmentation = None
        if post_masks.ok:
            for candidate in post_masks.segmentations:
                box = candidate.box_xyxy
                if box is None:
                    continue
                center_x = (float(box[0]) + float(box[2])) / 2.0
                center_y = (float(box[1]) + float(box[3])) / 2.0
                in_transfer_view = (
                    0.20 * image_width < center_x < 0.80 * image_width
                    and 0.10 * image_height < center_y < 0.80 * image_height
                )
                if in_transfer_view and (
                    post_segmentation is None
                    or candidate.score > post_segmentation.score
                ):
                    post_segmentation = candidate
        if post_segmentation is not None:
            try:
                post_cloud = mask_to_point_cloud(
                    post_segmentation,
                    "agentview",
                    target_frame="robot0_base",
                )
                raw_point_count = len(post_cloud.points)
                post_cloud = crop_point_cloud(
                    post_cloud,
                    (0.20, -0.35, -0.20),
                    (1.10, 0.35, 0.65),
                )
                workspace_point_count = len(post_cloud.points)
                post_xs = sorted(float(point[0]) for point in post_cloud.points)
                post_ys = sorted(float(point[1]) for point in post_cloud.points)
                post_zs = sorted(float(point[2]) for point in post_cloud.points)
                post_count = len(post_xs)
                quarter = post_count // 4
                three_quarters = (3 * post_count) // 4
                robust_lower = (
                    post_xs[quarter]
                    - 1.5 * (post_xs[three_quarters] - post_xs[quarter]),
                    post_ys[quarter]
                    - 1.5 * (post_ys[three_quarters] - post_ys[quarter]),
                    post_zs[quarter]
                    - 1.5 * (post_zs[three_quarters] - post_zs[quarter]),
                )
                robust_upper = (
                    post_xs[three_quarters]
                    + 1.5 * (post_xs[three_quarters] - post_xs[quarter]),
                    post_ys[three_quarters]
                    + 1.5 * (post_ys[three_quarters] - post_ys[quarter]),
                    post_zs[three_quarters]
                    + 1.5 * (post_zs[three_quarters] - post_zs[quarter]),
                )
                post_cloud = crop_point_cloud(
                    post_cloud,
                    robust_lower,
                    robust_upper,
                )
                post_geometry = estimate_geometry(post_cloud)
            except ValueError:
                post_cloud = None
                post_geometry = None
            if post_cloud is not None and post_geometry is not None:
                report["post_geometry"] = {
                    "score": float(post_segmentation.score),
                    "raw_point_count": raw_point_count,
                    "workspace_point_count": workspace_point_count,
                    "point_count": len(post_cloud.points),
                    "robust_lower": robust_lower,
                    "robust_upper": robust_upper,
                    "center": tuple(
                        float(value) for value in post_geometry.pose.position
                    ),
                    "quaternion_wxyz": tuple(
                        float(value)
                        for value in post_geometry.pose.quaternion_wxyz
                    ),
                    "extents": tuple(float(value) for value in post_geometry.extents),
                }

    verified_geometry = post_geometry

    if motion_ok and verified_geometry is not None:
        verified_giver = get_robot_state().end_effector_poses["primary"].position
        verified_center = tuple(
            float(value) for value in verified_geometry.pose.position
        )
        report["verified_handle_center"] = verified_center
        report["verified_handle_extents"] = tuple(
            float(value) for value in verified_geometry.extents
        )
        giver_to_handle = sum(
            (verified_center[index] - float(verified_giver[index])) ** 2
            for index in range(3)
        ) ** 0.5
        report["handle_toward_receiver"] = giver_to_handle >= (
            0.25 * max(report["verified_handle_extents"])
        )

    receiver_target = None
    if motion_ok and report["handle_toward_receiver"]:
        geometry_quaternion = verified_geometry.pose.quaternion_wxyz
        gw, gx, gy, gz = (float(value) for value in geometry_quaternion)
        handle_axis = [
            1.0 - 2.0 * (gy * gy + gz * gz),
            2.0 * (gx * gy + gw * gz),
            2.0 * (gx * gz - gw * gy),
        ]
        giver_to_center = [
            verified_center[index] - float(verified_giver[index])
            for index in range(3)
        ]
        if sum(handle_axis[index] * giver_to_center[index] for index in range(3)) < 0.0:
            handle_axis = [-value for value in handle_axis]
        sorted_extents = sorted(report["verified_handle_extents"])
        exposed_offset = 0.25 * sorted_extents[-1]
        surface_to_center = 0.5 * sorted_extents[-2]
        receiver_target = tuple(
            verified_center[index]
            + exposed_offset * handle_axis[index]
            + surface_to_center * approach_axis[index]
            for index in range(3)
        )
        report["receiver_handle_axis"] = tuple(handle_axis)
        report["surface_to_center"] = surface_to_center
        safe_clearance = 0.5 * sorted_extents[-1]
        grasp_clearance = 0.5 * sorted_extents[-2]
        report["receiver_clearances"] = (safe_clearance, grasp_clearance)
        for clearance in (safe_clearance, grasp_clearance, 0.0):
            if motion_ok:
                receiver_pose = Pose(
                    (
                        receiver_target[0],
                        receiver_target[1],
                        receiver_target[2] + clearance,
                    ),
                    receiver_quaternion,
                    "robot0_base",
                )
                receiver_plan = plan_motion(receiver_pose, arm="secondary")
                if receiver_plan.ok:
                    receiver_move = move_to_joints(
                        receiver_plan.trajectory.joint_positions[-1],
                        arm="secondary",
                        tolerance=0.02 if clearance == 0.0 else 0.04,
                        max_steps=700,
                    )
                    motion_ok = receiver_move.ok
                else:
                    motion_ok = False
        if motion_ok:
            receiver_close = close_gripper(arm="secondary")
            motion_ok = receiver_close.ok
            report["receiver_close_commanded"] = receiver_close.ok
        if motion_ok:
            giver_release = open_gripper(arm="primary")
            motion_ok = giver_release.ok
            report["giver_release_commanded"] = giver_release.ok
    report["receiver_target"] = receiver_target
    report["receiver_quaternion"] = receiver_quaternion

    report["motion_completed"] = motion_ok

result = report
