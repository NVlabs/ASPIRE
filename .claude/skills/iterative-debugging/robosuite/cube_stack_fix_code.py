def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def observe_cube(query, camera, frame):
    found = localize_object(query, camera_name=camera, target_frame=frame)
    if not found.ok or len(found.point_cloud.points) < 20:
        raise Exception("Cube localization failed: " + query)
    points = found.point_cloud.points
    top = quantile((p[2] for p in points), 0.9)
    bottom = quantile((p[2] for p in points), 0.01)
    face = [p for p in points if float(p[2]) >= top - 0.002]
    center = []
    for axis in range(2):
        center.append(0.5 * (quantile((p[axis] for p in face), 0.02) + quantile((p[axis] for p in face), 0.98)))
    return {"x": center[0], "y": center[1], "top": top, "bottom": bottom}

def move(position, orientation, frame):
    for motion_attempt in range(2):
        moved = move_to_pose(Pose(position, orientation, frame), tolerance=0.015, max_steps=120, strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation"))
        if moved.ok:
            return moved
    raise Exception("Motion failed: " + str(moved.status))

context = get_task_context()
state = get_robot_state()
frame = state.base_frame
cameras = list(get_observation().cameras)
camera = "agentview" if "agentview" in cameras else cameras[0]
if "primary" not in robosuite.get_controller_metadata()["controllable_gripper_arms"]:
    raise Exception("Task requires a controllable gripper")
red = observe_cube("red cube", camera, frame)
green = observe_cube("green cube", camera, frame)
home_orientation = state.end_effector_poses["primary"].quaternion_wxyz
horizontal_norm = (float(home_orientation[1]) ** 2 + float(home_orientation[2]) ** 2) ** 0.5
downward = (0.0, float(home_orientation[1]) / horizontal_norm, float(home_orientation[2]) / horizontal_norm, 0.0)
# Align finger separation perpendicular to the perceived neighbor direction.
dx = green["x"] - red["x"]
dy = green["y"] - red["y"]
distance = (dx * dx + dy * dy) ** 0.5
opening_x = -dy / distance
opening_y = dx / distance
if opening_y < 0.0:
    opening_x = -opening_x
    opening_y = -opening_y
downward = (0.0, ((1.0 + opening_x) * 0.5) ** 0.5, ((1.0 - opening_x) * 0.5) ** 0.5, 0.0)
grasp_z = 0.5 * (red["top"] + red["bottom"]) - 0.007
hover_z = max(red["top"], green["top"]) + 0.14
release_z = green["top"] + (grasp_z - red["bottom"]) + 0.006
open_gripper()
move((red["x"], red["y"], hover_z), downward, frame)
move((red["x"], red["y"], grasp_z), downward, frame)
close_gripper()
move((red["x"], red["y"], hover_z), downward, frame)
move((green["x"], green["y"], hover_z), downward, frame)
move((green["x"], green["y"], release_z), downward, frame)
open_gripper()
move((green["x"], green["y"], hover_z), downward, frame)
for settle_tick in range(20):
    held = get_robot_state()
    step(RobotAction({"primary": ArmCommand("joint_position", held.joint_positions["primary"], 1.0)}))
result = {"task": context.language, "red": red, "green": green, "grasp_z": grasp_z, "release_z": release_z}
