def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def locate(query, reference=None):
    found = localize_object(query, camera_name="agentview", target_frame=frame)
    if not found.ok or len(found.point_cloud.points) < 20:
        raise Exception("Could not localize " + query)
    points = found.point_cloud.points
    if reference is not None:
        margin = max(0.015, reference["high"][2]-reference["low"][2])
        points = [p for p in points if reference["low"][2]-margin < float(p[2]) < reference["high"][2]+margin]
        if len(points) < 20:
            raise Exception("No head points near the observed handle height")
    top = quantile((p[2] for p in points), 0.90)
    points = [p for p in points if float(p[2]) > top-0.05]
    low = [quantile((p[a] for p in points), 0.02) for a in range(3)]
    high = [quantile((p[a] for p in points), 0.98) for a in range(3)]
    geometry = {"center": [(low[a]+high[a])*0.5 for a in range(3)], "low": low, "high": high}
    if reference is not None:
        center = geometry["center"]
        dx = reference["center"][0]-center[0]
        dy = reference["center"][1]-center[1]
        norm = (dx*dx+dy*dy)**0.5
        along = [-dy/norm, dx/norm]
        across = [dx/norm, dy/norm]
        coordinates = [(float(p[0])-center[0])*along[0]+(float(p[1])-center[1])*along[1] for p in points]
        first = quantile(coordinates,0.02)
        last = quantile(coordinates,0.98)
        options = []
        for fraction in (0.25,0.75):
            location = first+fraction*(last-first)
            band = [p for p,t in zip(points,coordinates) if abs(t-location)<0.08*(last-first)]
            if len(band)<10:
                continue
            offsets = [(float(p[0])-center[0])*across[0]+(float(p[1])-center[1])*across[1] for p in band]
            cross = (quantile(offsets,0.1)+quantile(offsets,0.9))*0.5
            ztop = quantile((p[2] for p in band),0.9)
            zbottom = quantile((p[2] for p in band),0.1)
            curvature = ztop-quantile((p[2] for p in band),0.5)
            options.append({"position":[center[0]+along[0]*location+across[0]*cross, center[1]+along[1]*location+across[1]*cross, ztop-0.4*(ztop-zbottom)], "curvature":curvature})
            outer = first+(0.15 if fraction < 0.5 else 0.85)*(last-first)
            outer_band = [p for p,t in zip(points,coordinates) if abs(t-outer)<0.08*(last-first)]
            if len(outer_band) >= 10:
                offsets = [(float(p[0])-center[0])*across[0]+(float(p[1])-center[1])*across[1] for p in outer_band]
                cross = (quantile(offsets,0.1)+quantile(offsets,0.9))*0.5
                ztop = quantile((p[2] for p in outer_band),0.9)
                zbottom = quantile((p[2] for p in outer_band),0.1)
                options[-1]["outer_position"] = [center[0]+along[0]*outer+across[0]*cross,center[1]+along[1]*outer+across[1]*cross,ztop-0.4*(ztop-zbottom)]
        geometry["grasp_options"] = options
    return geometry

def downward(x, y):
    magnitude = (x*x+y*y)**0.5
    x = x/magnitude
    y = y/magnitude
    return (0.0, ((1.0+x)*0.5)**0.5, (1.0 if y >= 0.0 else -1.0)*((1.0-x)*0.5)**0.5, 0.0)

def receiver_orientation(direction):
    q = downward(direction[1], -direction[0])
    span = max(handle["high"][a]-handle["low"][a] for a in range(2))
    separation = sum((wc[a]-grasp[a])**2 for a in range(2))**0.5+0.25*span
    if separation >= 0.17:
        return q
    r = (0.9238795325112867, -0.3826834323650898*direction[1], 0.3826834323650898*direction[0], 0.0)
    w, x, y, z = r
    a, b, c, d = q
    return (w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c, w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a)

def move(arm, position, orientation, contact=False):
    target = Pose(position, orientation, frame)
    for attempt in range(2):
        moved = move_to_pose(target, arm=arm, tolerance=0.015, max_steps=120)
        if moved.ok:
            return True
        actual_pose = get_robot_state().end_effector_poses[arm]
        distance = sum((float(actual_pose.position[a])-float(position[a]))**2 for a in range(3))**0.5
        alignment = abs(sum(float(actual_pose.quaternion_wxyz[a])*float(orientation[a]) for a in range(4)))
        if distance < 0.015 and alignment > 0.999:
            return True
        if contact:
            reached = get_robot_state().end_effector_poses[arm].position
            if sum((float(reached[a])-float(position[a]))**2 for a in range(3))**0.5 < 0.025:
                return False
        if str(moved.status) == "ExecutionStatus.PLANNER_FAILED":
            moved = move_to_pose(target, arm=arm, tolerance=0.015, max_steps=120, strategy=MotionStrategy(ik_solver="curobo", trajectory_planner="interpolation"))
            if moved.ok:
                return True
    raise Exception("Motion failed for " + arm + ": " + str(moved.status))

def paired_move(targets):
    start = get_robot_state()
    solutions = {}
    joints = {}
    grippers = {}
    for arm in targets:
        solved = solve_ik(targets[arm], seed_joints=state.joint_positions[arm], arm=arm)
        if not solved.ok:
            solved = solve_ik(targets[arm], seed_joints=state.joint_positions[arm], arm=arm, backend="curobo")
        if not solved.ok:
            raise Exception("Paired IK failed for " + arm)
        solutions[arm] = solved.joint_positions
        joints[arm] = [[float(start.joint_positions[arm][j]) + tick/180.0*(float(solutions[arm][j])-float(start.joint_positions[arm][j])) for j in range(7)] for tick in range(1,181)]
        grippers[arm] = [0.0 if arm == "primary" else 1.0 for tick in range(180)]
    trajectory = SynchronizedTrajectory(joints, 0.05, start.joint_names, "interpolation", False, start, grippers)
    executed = execute_trajectory(trajectory)
    report["paired_execution"] = {"ok": executed.ok, "status": str(executed.status), "steps": executed.steps_executed}
    for arm in targets:
        reached = get_robot_state().end_effector_poses[arm]
        actual = reached.position
        alignment = abs(sum(float(reached.quaternion_wxyz[a])*float(targets[arm].quaternion_wxyz[a]) for a in range(4)))
        if sum((float(actual[a])-float(targets[arm].position[a]))**2 for a in range(3))**0.5 > 0.025 or alignment < 0.999:
            move(arm, targets[arm].position, targets[arm].quaternion_wxyz)

context = get_task_context()
state = get_robot_state()
frame = state.base_frame
arms = robosuite.get_controller_metadata()["controllable_gripper_arms"]
report = {"language": context.language, "arms": list(arms)}
handle = locate("wooden handle")
head = locate("hammer head", handle)
report["head"] = head
report["handle"] = handle
hc = head["center"]
wc = handle["center"]
dx = wc[0]-hc[0]
dy = wc[1]-hc[1]
length = (dx*dx+dy*dy)**0.5
direction = [dx/length, dy/length]
# Select the equivalent finger-opening direction nearest the initial gripper yaw.
sign = 1.0 if direction[1] >= 0.0 else -1.0
grasp_q = downward(sign*direction[0], sign*direction[1])
head_axis = [-direction[1], direction[0]]
p0 = state.end_effector_poses["primary"].position
p1 = state.end_effector_poses["secondary"].position
if sum(head_axis[a]*(float(p1[a])-float(p0[a])) for a in range(2)) < 0.0:
    head_axis = [-head_axis[0], -head_axis[1]]
head_span = sum(abs(head_axis[a])*(head["high"][a]-head["low"][a]) for a in range(2))
grasp = [hc[0]+0.25*head_span*head_axis[0], hc[1]+0.25*head_span*head_axis[1], head["high"][2]-0.40*(head["high"][2]-head["low"][2])]
options = head["grasp_options"]
selected_option = None
if len(options) == 2 and abs(options[0]["curvature"]-options[1]["curvature"]) > 0.001:
    selected_option = max(options, key=lambda item: item["curvature"])
    grasp = selected_option["position"]
opening_direction = direction
arm_line = [float(p1[a])-float(p0[a]) for a in range(2)]
arm_span = sum(v*v for v in arm_line)**0.5
behind_home = sum((float(p0[a])-grasp[a])*arm_line[a]/arm_span for a in range(2))
radial_reach = sum(grasp[a]*grasp[a] for a in range(2))**0.5
home_reach = sum(float(p0[a])*float(p0[a]) for a in range(2))**0.5
# Include the requested Cartesian tolerance in the near-base clearance test.
handle_grasp = radial_reach-0.015 < 0.60*home_reach
if handle_grasp:
    handle_length = sum(abs(direction[a])*(handle["high"][a]-handle["low"][a]) for a in range(2))
    for fraction in (0.30,0.15,0.0,-0.15):
        grasp = [wc[0]-fraction*handle_length*direction[0], wc[1]-fraction*handle_length*direction[1], handle["high"][2]-0.45*(handle["high"][2]-handle["low"][2])]
        if sum(grasp[a]*grasp[a] for a in range(2))**0.5 >= 0.65*home_reach:
            break
    opening_direction = [-direction[1],direction[0]]
grasp_q = downward(sign*opening_direction[0],sign*opening_direction[1])
report["handle_grasp"] = handle_grasp
best_cost = None
for candidate_sign in (sign, -sign):
    candidate_q = downward(candidate_sign*opening_direction[0], candidate_sign*opening_direction[1])
    checked = solve_ik(Pose(grasp, candidate_q, frame), seed_joints=state.joint_positions["primary"], arm="primary")
    if checked.ok:
        changes = [abs(float(checked.joint_positions[j])-float(state.joint_positions["primary"][j])) for j in range(7)]
        cost = sum(v*v for v in changes)+2.0*max(changes)**2
        if best_cost is None or cost < best_cost:
            best_cost = cost
            sign = candidate_sign
            grasp_q = candidate_q
open_gripper(arm="primary")
move("primary", [grasp[0], grasp[1], head["high"][2]+0.12], grasp_q)
if handle_grasp:
    move("primary", [grasp[0],grasp[1],grasp[2]+0.065], grasp_q)
seated = move("primary", grasp, grasp_q, contact=not handle_grasp)
if not seated and not handle_grasp and selected_option is not None and "outer_position" in selected_option:
    move("primary", [grasp[0],grasp[1],grasp[2]+0.065], grasp_q)
    grasp = selected_option["outer_position"]
    move("primary",grasp,grasp_q,contact=True)
    report["outer_head_recovery"] = True
close_gripper(arm="primary")
move("primary", [grasp[0], grasp[1], grasp[2]+0.065], grasp_q)
lift_z = head["high"][2]+0.25

p0 = state.end_effector_poses["primary"].position
p1 = state.end_effector_poses["secondary"].position
rx = p1[0]-p0[0]
ry = p1[1]-p0[1]
norm = (rx*rx+ry*ry)**0.5
receiver = [rx/norm, ry/norm]
transfer_q = downward(sign*receiver[0], sign*receiver[1])
if handle_grasp:
    transfer_q = downward(-sign*receiver[1], sign*receiver[0])
transfer = [(p0[0]+p1[0])*0.5-length*0.5*receiver[0], (p0[1]+p1[1])*0.5-length*0.5*receiver[1], lift_z]
secondary_q = receiver_orientation(receiver)
cos_yaw = direction[0]*receiver[0]+direction[1]*receiver[1]
sin_yaw = direction[0]*receiver[1]-direction[1]*receiver[0]
offset_x = wc[0]-grasp[0]
offset_y = wc[1]-grasp[1]
predicted = [transfer[0]+cos_yaw*offset_x-sin_yaw*offset_y, transfer[1]+sin_yaw*offset_x+cos_yaw*offset_y, transfer[2]+wc[2]-grasp[2]+0.10]
handle_span = sum(abs(direction[a])*(handle["high"][a]-handle["low"][a]) for a in range(2))
for axis in range(2):
    predicted[axis] = predicted[axis]+0.25*handle_span*receiver[axis]
paired_move({"primary": Pose(transfer, transfer_q, frame), "secondary": Pose(predicted, secondary_q, frame)})
received_handle = locate("wooden handle")
report["handover_handle"] = received_handle
target = received_handle["center"]
receiver_span = sum(abs(receiver[a])*(received_handle["high"][a]-received_handle["low"][a]) for a in range(2))
target = [target[0]+0.25*receiver_span*receiver[0], target[1]+0.25*receiver_span*receiver[1], target[2]]
secondary_q = receiver_orientation(receiver)
move("secondary", [target[0], target[1], target[2]-0.004], secondary_q)
close_gripper(arm="secondary")
open_gripper(arm="primary")
move("primary", [transfer[0]-0.12*receiver[0], transfer[1]-0.12*receiver[1], transfer[2]+0.08], transfer_q)
for settle_tick in range(5):
    set_grippers({"primary": 1.0, "secondary": 0.0})
report["final_state"] = get_robot_state()
result = report
