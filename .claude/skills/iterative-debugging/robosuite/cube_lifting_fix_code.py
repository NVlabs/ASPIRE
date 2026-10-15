# Fresh policy from the public seed-101 scene and shared grasp/transport skills.
def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def move(position, orientation, frame):
    for motion_attempt in range(2):
        moved = move_to_pose(
            Pose(position, orientation, frame), tolerance=0.015, max_steps=120,
            strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation"),
        )
        if moved.ok:
            return moved
    raise Exception("Motion failed: " + str(moved.status))

context = get_task_context()
state = get_robot_state()
frame = state.base_frame
observation = get_observation()
camera = "agentview" if "agentview" in observation.cameras else list(observation.cameras)[0]
metadata = robosuite.get_controller_metadata()
if "primary" not in metadata["controllable_gripper_arms"]:
    raise Exception("The primary gripper must be controllable")
picked = None
for prompt in ("red cube", "cube", "red block"):
    localized = localize_object(prompt, camera_name=camera, target_frame=frame)
    if localized.ok and len(localized.point_cloud.points) >= 20:
        picked = localized
        break
if picked is None:
    raise Exception("Cube localization failed")
points = picked.point_cloud.points
x = 0.5 * (quantile((point[0] for point in points), 0.05) + quantile((point[0] for point in points), 0.95))
y = 0.5 * (quantile((point[1] for point in points), 0.05) + quantile((point[1] for point in points), 0.95))
bottom = quantile((point[2] for point in points), 0.10)
top = quantile((point[2] for point in points), 0.90)
grasp_z = 0.5 * (top + bottom) - 0.007
downward = (0.0, 0.5 ** 0.5, 0.5 ** 0.5, 0.0)
report = {"task": context.language, "prompt": prompt, "x": x, "y": y, "bottom": bottom, "top": top, "grasp_z": grasp_z}
open_gripper()
move((x, y, top + 0.14), downward, frame)
move((x, y, grasp_z), downward, frame)
report["close"] = close_gripper()
report["lift"] = move((x, y, grasp_z + 0.20), downward, frame)
after = localize_object(prompt, camera_name=camera, target_frame=frame)
report["observed_lift"] = after.ok and len(after.point_cloud.points) >= 20 and quantile((point[2] for point in after.point_cloud.points), 0.5) - quantile((point[2] for point in points), 0.5) > 0.03
result = report
