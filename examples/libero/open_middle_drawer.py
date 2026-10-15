report = {"task": libero.get_task_metadata()["task_ref"], "attempts": []}


def cloud_quantiles(points, x_fraction=0.5, y_fraction=0.5, z_fraction=0.5):
    xs = sorted(float(point[0]) for point in points)
    ys = sorted(float(point[1]) for point in points)
    zs = sorted(float(point[2]) for point in points)
    last = len(points) - 1
    return (
        xs[int(last * x_fraction)],
        ys[int(last * y_fraction)],
        zs[int(last * z_fraction)],
    )


def box_vertical_center(segmentation):
    box = segmentation.box_xyxy
    return (float(box[1]) + float(box[3])) / 2.0


def quaternion_product(first, second):
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def locate_middle_handle():
    handles = segment_text("agentview", "gray horizontal drawer handle")
    candidates = []
    if handles.ok:
        for candidate in handles.segmentations:
            box = candidate.box_xyxy
            width = float(box[2]) - float(box[0])
            height = float(box[3]) - float(box[1])
            horizontal_center = (float(box[0]) + float(box[2])) / 2.0
            if width < 60.0 and height > 25.0 and horizontal_center < 350.0:
                candidates.append(candidate)
    if len(candidates) < 3:
        return None
    ordered = sorted(candidates, key=box_vertical_center)
    mask = ordered[len(ordered) // 2]
    return mask_to_point_cloud(mask, "agentview")


def pull_once():
    attempt = {"localized": False}
    handle_cloud = locate_middle_handle()
    if handle_cloud is None:
        return attempt
    attempt["localized"] = True
    handle = cloud_quantiles(handle_cloud.points, 0.10, 0.50, 0.50)
    state = get_robot_state()
    frame = state.base_frame
    downward = state.end_effector_poses["primary"].quaternion_wxyz
    forward = quaternion_product((0.7071067812, -0.7071067812, 0.0, 0.0), downward)
    open_gripper()
    above = move_to_pose(
        Pose((handle[0], handle[1] + 0.10, handle[2] + 0.18), downward, frame),
        tolerance=0.03,
        max_steps=300,
    )
    attempt["above"] = above.ok
    if not above.ok:
        return attempt
    oriented = move_to_pose(
        Pose((handle[0], handle[1] + 0.10, handle[2] + 0.18), forward, frame),
        tolerance=0.04,
        max_steps=400,
    )
    attempt["oriented"] = oriented.ok
    if not oriented.ok:
        return attempt
    staged = move_to_pose(
        Pose((handle[0], handle[1] + 0.10, handle[2] + 0.025), forward, frame),
        tolerance=0.04,
        max_steps=320,
    )
    attempt["staged"] = staged.ok
    if not staged.ok:
        return attempt
    contacted = move_to_pose(
        Pose((handle[0], handle[1] + 0.01, handle[2] + 0.025), forward, frame),
        tolerance=0.04,
        max_steps=320,
    )
    attempt["contacted"] = contacted.ok
    if not contacted.ok:
        return attempt
    gripped = set_gripper(0.15)
    attempt["gripped"] = gripped.ok
    held = get_robot_state().end_effector_poses["primary"]
    pulled = gripped
    for offset in (0.04, 0.08, 0.12, 0.16, 0.20, 0.24):
        pulled = move_to_pose(
            Pose(
                (held.position[0], held.position[1] + offset, held.position[2]),
                held.quaternion_wxyz,
                held.frame,
            ),
            tolerance=0.04,
            max_steps=220,
        )
    attempt["pulled"] = pulled.ok
    open_gripper()
    pose = get_robot_state().end_effector_poses["primary"]
    attempt["retreated"] = move_to_pose(
        Pose(
            (pose.position[0], pose.position[1] + 0.05, pose.position[2] + 0.10),
            pose.quaternion_wxyz,
            pose.frame,
        ),
        tolerance=0.04,
        max_steps=250,
    ).ok
    return attempt


for attempt_index in range(1):
    attempt_result = pull_once()
    report["attempts"].append(attempt_result)

result = report
