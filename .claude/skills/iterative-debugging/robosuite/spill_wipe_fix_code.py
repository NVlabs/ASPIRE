context = get_task_context()
state = get_robot_state()
metadata = robosuite.get_controller_metadata()
if metadata["controllable_gripper_arms"]:
    raise Exception("Expected the fixed wiping tool")
frame = state.base_frame
spill = localize_object("brown spill",camera_name="agentview",target_frame=frame)
if not spill.ok or len(spill.point_cloud.points) < 30:
    raise Exception("Spill not localized")
points = list(spill.point_cloud.points)
xs = sorted(float(p[0]) for p in points)
ys = sorted(float(p[1]) for p in points)
zs = sorted(float(p[2]) for p in points)
surface = zs[len(zs)//2]
spill_bounds = [[values[int(len(values)*0.005)]-0.04,values[int(len(values)*0.995)]+0.04] for values in (xs,ys)]
initial_q = state.end_effector_poses["primary"].quaternion_wxyz
norm = (float(initial_q[1])**2+float(initial_q[2])**2)**0.5
down = (0.0,float(initial_q[1])/norm,float(initial_q[2])/norm,0.0)
bias = [0.0,0.0,0.0]
motions = []
hover_point = [xs[int(len(xs)*0.005)]-0.04,ys[len(ys)//2],surface+0.10]
def free_move(point,hold):
    before = get_robot_state()
    target = [point[j]-bias[j] for j in range(3)]
    solved = solve_ik(Pose(target,down,frame),seed_joints=before.joint_positions["primary"])
    if not solved.ok:
        raise Exception("Free-space IK failed")
    start = before.joint_positions["primary"]
    goal = solved.joint_positions
    ticks = max(12,int(max(abs(float(goal[j])-float(start[j])) for j in range(7))/0.025)+1)
    path = []
    for tick in range(ticks+hold):
        alpha = min(1.0,float(tick+1)/ticks)
        path.append([float(start[j])+(float(goal[j])-float(start[j]))*alpha for j in range(7)])
    moved = execute_trajectory(Trajectory(path,metadata["control_period_s"],before.joint_names["primary"],"public_ik_interpolation",False,before))
    motions.append({"kind":"calibration","target":point,"pose":get_robot_state().end_effector_poses["primary"],"status":moved.status,"steps":moved.steps_executed})
for calibration in range(2):
    free_move(hover_point,45)
    actual = get_robot_state().end_effector_poses["primary"].position
    bias = [bias[j]+float(actual[j])-hover_point[j] for j in range(3)]
passes = []
for sweep in range(3):
    if not points:
        break
    xs = sorted(float(p[0]) for p in points)
    low = xs[int(len(xs)*0.01)]-0.012
    high = xs[int(len(xs)*0.99)]+0.012
    rows = max(2,int((high-low)/0.018)+2)
    waypoints = []
    for row in range(rows):
        x = low+(high-low)*row/(rows-1)
        band = sorted(float(p[1]) for p in points if abs(float(p[0])-x) < 0.025)
        if len(band) >= 10:
            clusters = [[band[0]]]
            for y in band[1:]:
                if y-clusters[-1][-1] > 0.02:
                    clusters.append([])
                clusters[-1].append(y)
            for cluster in clusters:
                if len(cluster) >= 10:
                    y = (cluster[int(len(cluster)*0.1)]+cluster[int(len(cluster)*0.9)])/2.0
                    waypoints.append([x,y,surface-0.02-0.004*sweep])
    if not waypoints:
        break
    before = get_robot_state()
    previous = list(before.end_effector_poses["primary"].position)
    ordered = []
    cursor = previous
    while waypoints:
        chosen = min(range(len(waypoints)),key=lambda i:sum((waypoints[i][j]-cursor[j])**2 for j in range(2)))
        cursor = waypoints.pop(chosen)
        ordered.append(cursor)
    waypoints = ordered
    previous_joints = before.joint_positions["primary"]
    last = waypoints[-1]
    path = []
    complete_path = [[waypoints[0][0],waypoints[0][1],surface+0.05]]+waypoints+[[last[0],last[1],surface+0.06],hover_point]
    return_start = 0
    for point_index, point in enumerate(complete_path):
        if point_index == len(complete_path)-1:
            return_start = len(path)
        distance = sum((point[j]-previous[j])**2 for j in range(3))**0.5
        divisions = max(1,int(distance/0.01)+1)
        ticks_per_division = 2 if point[2] > surface+0.03 else 3
        for division in range(divisions):
            fraction = float(division+1)/divisions
            target = [previous[j]+fraction*(point[j]-previous[j])-bias[j] for j in range(3)]
            solved = solve_ik(Pose(target,down,frame),seed_joints=previous_joints)
            if not solved.ok:
                raise Exception("Cartesian wipe IK failed")
            goal = solved.joint_positions
            for tick in range(ticks_per_division):
                alpha = float(tick+1)/ticks_per_division
                path.append([float(previous_joints[j])+alpha*(float(goal[j])-float(previous_joints[j])) for j in range(7)])
            previous_joints = goal
        for hold_tick in range(4):
            path.append(list(previous_joints))
        previous = point
    remaining = 1000-int(before.timestamp_s/metadata["control_period_s"])
    if len(path)+30 > remaining:
        if remaining < 60:
            break
        path = path[:return_start]
        stride = max(1,int((len(path)+remaining-26)/(remaining-25)))
        final_joints = path[-1]
        path = path[::stride]
        path.append(final_joints)
    moved = execute_trajectory(Trajectory(path,metadata["control_period_s"],before.joint_names["primary"],"public_cartesian_ik",False,before))
    passes.append({"point_count":len(points),"rows":rows,"status":moved.status,"steps":moved.steps_executed})
    if moved.terminated or moved.truncated:
        break
    points = []
    segmented = segment_text("agentview","brown spill")
    if segmented.ok and segmented.segmentations:
        best_score = max(float(candidate.score) for candidate in segmented.segmentations)
        for candidate in segmented.segmentations:
            if candidate.score >= max(0.005,best_score*0.4):
                cloud = mask_to_point_cloud(candidate,"agentview",target_frame=frame)
                points.extend([p for p in cloud.points if abs(float(p[2])-surface) < 0.006 and all(spill_bounds[j][0] <= float(p[j]) <= spill_bounds[j][1] for j in range(2))])
result = {"language":context.language,"surface":surface,"bias":bias,"motions":motions,"passes":passes,"residual_points":len(points)}
