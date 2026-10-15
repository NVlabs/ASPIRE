report = {"task": libero.get_task_metadata()["task_ref"], "attempts": []}

state = get_robot_state()
frame = state.base_frame
downward = state.end_effector_poses["primary"].quaternion_wxyz

# The fixed task region varies mainly along X. Sweep a small set of geometric
# candidates rather than branching on the seed or reading simulator state.
for handle_x in (0.34, 0.39, 0.44, 0.48):
    attempt = {"handle_x": handle_x}
    go_home()
    open_gripper()
    handle = (handle_x, 0.04, 0.104)
    above = move_to_pose(
        Pose((handle[0], handle[1], handle[2] + 0.15), downward, frame),
        tolerance=0.05,
        max_steps=400,
    )
    attempt["above"] = above.ok
    if above.ok:
        contact = move_to_pose(
            Pose(handle, downward, frame),
            tolerance=0.05,
            max_steps=350,
        )
        attempt["contact"] = contact.ok
    else:
        contact = None

    closed = contact
    if contact is not None:
        for x_offset, y_offset in (
            (0.05, 0.01),
            (0.12, 0.04),
            (0.18, 0.09),
            (0.23, 0.15),
            (0.25, 0.19),
        ):
            closed = move_to_pose(
                Pose(
                    (handle[0] + x_offset, handle[1] + y_offset, handle[2]),
                    downward,
                    frame,
                ),
                tolerance=0.05,
                max_steps=400,
            )
        attempt["closed"] = closed.ok
    report["attempts"].append(attempt)

result = report
