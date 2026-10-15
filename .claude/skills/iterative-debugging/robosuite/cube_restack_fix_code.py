def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def observe(prompt):
    found = localize_object(prompt, camera_name="agentview", target_frame=frame)
    if not found.ok or len(found.point_cloud.points) < 20:
        raise Exception("Localization failed: " + prompt)
    points = found.point_cloud.points
    top = quantile((p[2] for p in points), 0.9)
    upper = [p for p in points if float(p[2]) > top - 0.004]
    if len(upper) < 20:
        upper = points
    return {"prompt": prompt, "x": (quantile((p[0] for p in upper), 0.05) + quantile((p[0] for p in upper), 0.95)) * 0.5,
            "y": (quantile((p[1] for p in upper), 0.05) + quantile((p[1] for p in upper), 0.95)) * 0.5,
            "top": top, "bottom": quantile((p[2] for p in points), 0.02),
            "width": max(quantile((p[0] for p in points), 0.95) - quantile((p[0] for p in points), 0.05), quantile((p[1] for p in points), 0.95) - quantile((p[1] for p in points), 0.05))}

def move(position):
    for attempt in range(2):
        moved = move_to_pose(Pose(position, downward, frame), tolerance=0.015, max_steps=120,
                             strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation"))
        if moved.ok:
            return moved
    raise Exception("Motion failed: " + str(moved.status))

def transfer(picked, target_x, target_y, surface_z):
    grasp_z = (picked["top"] + picked["bottom"]) * 0.5 - 0.007
    hover_z = max(picked["top"], surface_z) + 0.13
    release_z = surface_z + grasp_z - picked["bottom"] + 0.006
    open_gripper()
    move((picked["x"], picked["y"], hover_z))
    move((picked["x"], picked["y"], grasp_z))
    close_gripper()
    move((picked["x"], picked["y"], hover_z))
    move((target_x, target_y, hover_z))
    move((target_x, target_y, release_z))
    open_gripper()
    move((target_x, target_y, hover_z))

context = get_task_context()
frame = get_robot_state().base_frame
if "primary" not in robosuite.get_controller_metadata()["controllable_gripper_arms"]:
    raise Exception("Task requires a controllable gripper")
downward = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)
report = {"language": context.language}
first = observe("red cube")
second = observe("green cube")
cubes = sorted([first, second], key=lambda cube: cube["top"])
lower = cubes[0]
upper = cubes[1]
table = localize_object("table", camera_name="agentview", target_frame=frame)
if not table.ok:
    raise Exception("Table support localization failed")
table_z = quantile((p[2] for p in table.point_cloud.points), 0.5)
table_y = (quantile((p[1] for p in table.point_cloud.points), 0.05) + quantile((p[1] for p in table.point_cloud.points), 0.95)) * 0.5
upper["bottom"] = lower["top"]
parking_direction = 1.0 if table_y > upper["y"] else -1.0
parking_y = upper["y"] + parking_direction * 3.0 * max(upper["width"], lower["width"])
report["initial_upper"] = upper
report["initial_lower"] = lower
report["table_z"] = table_z
report["parking"] = [upper["x"], parking_y]
transfer(upper, upper["x"], parking_y, table_z)
go_home()
upper_now = observe(upper["prompt"])
lower_now = observe(lower["prompt"])
lower_now["bottom"] = table_z
report["parked_upper"] = upper_now
report["uncovered_lower"] = lower_now
transfer(lower_now, upper_now["x"], upper_now["y"], upper_now["top"])
for settle_tick in range(1):
    set_gripper(1.0)
result = report
