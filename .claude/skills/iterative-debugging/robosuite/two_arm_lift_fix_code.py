def quantile(values, fraction):
    values = sorted(float(value) for value in values)
    return values[int((len(values) - 1) * fraction)]

def distance(a, b):
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5

def move_pair_once(targets, grip, count):
    state = get_robot_state()
    solutions = {}
    for arm in arms:
        solved = solve_ik(targets[arm], seed_joints=state.joint_positions[arm], arm=arm, backend="pyroki")
        if not solved.ok:
            raise Exception("IK failed for " + arm + ": " + str(solved.error))
        solutions[arm] = solved.joint_positions
    paths = {}
    for arm in arms:
        paths[arm] = []
        for tick in range(count):
            fraction = float(tick + 1) / count
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            paths[arm].append([float(state.joint_positions[arm][j]) + fraction * (float(solutions[arm][j]) - float(state.joint_positions[arm][j])) for j in range(7)])
    trajectory = SynchronizedTrajectory(paths, period, state.joint_names, "public-pyroki-synchronized-interpolation", False, state, {arm: [grip] * count for arm in arms})
    moved = execute_trajectory(trajectory)
    report["motions"].append({"targets": targets, "ok": moved.ok, "status": moved.status, "errors": moved.final_errors})
    return moved

def move_pair(targets, grip, count=70):
    if grip == 0.0:
        move_loaded_pair(targets, count)
        return
    moved = move_pair_once(targets, grip, count)
    if not moved.ok and str(moved.status) == "ExecutionStatus.TIMEOUT":
        moved = move_pair_once(targets, grip, 35)
    if not moved.ok:
        raise Exception("Synchronized execution failed: " + str(moved.status))

def move_loaded_pair(targets, count):
    state = get_robot_state()
    solutions = {}
    for arm in arms:
        solved = solve_ik(targets[arm], seed_joints=state.joint_positions[arm], arm=arm, backend="pyroki")
        if not solved.ok:
            raise Exception("Loaded IK failed for " + arm)
        solutions[arm] = solved.joint_positions
    for tick in range(count + 40):
        fraction = min(1.0, float(tick + 1) / count)
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        commands = {arm: ArmCommand("joint_position", [float(state.joint_positions[arm][j]) + fraction * (float(solutions[arm][j]) - float(state.joint_positions[arm][j])) for j in range(7)], 0.0) for arm in arms}
        stepped = step(RobotAction(commands))
        if stepped.error is not None or stepped.terminated or stepped.truncated:
            raise Exception("Loaded synchronized step stopped: " + str(stepped.error))
        if tick + 1 >= count:
            measured = get_robot_state()
            errors = {arm: distance(measured.end_effector_poses[arm].position, targets[arm].position) for arm in arms}
            aligned = all(abs(sum(float(measured.end_effector_poses[arm].quaternion_wxyz[j]) * float(targets[arm].quaternion_wxyz[j]) for j in range(4))) > 0.995 for arm in arms)
            if max(errors.values()) < 0.01 and aligned:
                report["motions"].append({"targets": targets, "controller": "public-synchronized-step", "position_errors_m": errors, "cartesian_tolerance_reached": True})
                return
    raise Exception("Loaded wrists did not reach Cartesian tolerance")

def handle_cloud(prompt):
    channel = 1 if "green" in prompt else 2
    selected = None
    for query in (prompt, prompt.replace("handle", "object")):
        segmented = segment_text("agentview", query)
        if not segmented.ok:
            continue
        for mask in sorted(segmented.segmentations, key=lambda candidate: candidate.score, reverse=True):
            cloud = mask_to_point_cloud(mask, "agentview", target_frame=frame)
            if cloud.colors is None or len(cloud.points) < 30:
                continue
            colored = [point for point, rgb in zip(cloud.points, cloud.colors) if float(rgb[channel]) > 1.25 * max(float(rgb[other]) for other in range(3) if other != channel)]
            if len(colored) >= 30 and len(colored) > 0.5 * len(cloud.points):
                selected = {"points": colored, "score": mask.score}
                break
        if selected is not None:
            break
    if selected is None:
        raise Exception("Cannot identify colored " + prompt)
    top = quantile([point[2] for point in selected["points"]], 0.8)
    points = [point for point in selected["points"] if abs(float(point[2]) - top) < 0.004]
    center = [quantile([point[axis] for point in points], 0.5) for axis in range(3)]
    return {"points": points, "center": center, "top": top, "score": selected["score"]}

context = get_task_context()
state = get_robot_state()
metadata = robosuite.get_controller_metadata()
arms = list(metadata["arm_names"])
frame = state.base_frame
period = metadata["control_period_s"]
report = {"language": context.language, "motions": [], "handles": []}
result = report
if len(arms) != 2 or len(metadata["controllable_gripper_arms"]) != 2:
    raise Exception("Two controllable arms required")

# Raise the observed wrists to expose both handles before selecting grasps.
targets = {arm: Pose((state.end_effector_poses[arm].position[0], state.end_effector_poses[arm].position[1], state.end_effector_poses[arm].position[2] + 0.15), state.end_effector_poses[arm].quaternion_wxyz, frame) for arm in arms}
move_pair(targets, 1.0, 65)
handles = [handle_cloud("green handle"), handle_cloud("blue handle")]
direction = [handles[1]["center"][i] - handles[0]["center"][i] for i in range(2)]
length = sum(value * value for value in direction) ** 0.5
axis = [value / length for value in direction]
grasps = []
for index, handle in enumerate(handles):
    outward = [value * (-1.0 if index == 0 else 1.0) for value in axis]
    projections = [float(point[0]) * outward[0] + float(point[1]) * outward[1] for point in handle["points"]]
    threshold = quantile(projections, 0.75)
    bar = [point for point, projection in zip(handle["points"], projections) if projection >= threshold]
    center = [quantile([point[i] for point in bar], 0.5) for i in range(3)]
    center[2] = handle["top"] - 0.008
    grasps.append(center)
    report["handles"].append({"score": handle["score"], "grasp": center, "direction": outward})

state = get_robot_state()
direct = sum(distance(state.end_effector_poses[arms[i]].position, grasps[i]) for i in range(2))
crossed = sum(distance(state.end_effector_poses[arms[i]].position, grasps[1-i]) for i in range(2))
assignment = [0, 1] if direct <= crossed else [1, 0]
orientations = {}
for arm in arms:
    options = []
    for sign in (1.0, -1.0):
        opening_x, opening_y = axis[0] * sign, axis[1] * sign
        qx = max(0.0, (1.0 + opening_x) * 0.5) ** 0.5
        qy = max(0.0, (1.0 - opening_x) * 0.5) ** 0.5
        if opening_y < 0.0:
            qy = -qy
        options.append((0.0, qx, qy, 0.0))
    orientations[arm] = max(options, key=lambda q: abs(sum(float(q[j]) * float(state.end_effector_poses[arm].quaternion_wxyz[j]) for j in range(4))))

for height, count, grip in ((0.12, 90, 1.0), (0.0, 65, 1.0)):
    targets = {arm: Pose((grasps[assignment[i]][0], grasps[assignment[i]][1], grasps[assignment[i]][2] + height), orientations[arm], frame) for i, arm in enumerate(arms)}
    move_pair(targets, grip, count)
for close_tick in range(45):
    set_grippers({arm: 0.0 for arm in arms})
for height in (0.06, 0.16, 0.27):
    targets = {arm: Pose((grasps[assignment[i]][0], grasps[assignment[i]][1], grasps[assignment[i]][2] + height), orientations[arm], frame) for i, arm in enumerate(arms)}
    move_pair(targets, 0.0, 70)
for hold_tick in range(25):
    set_grippers({arm: 0.0 for arm in arms})
report["final_state"] = get_robot_state()
result = report
