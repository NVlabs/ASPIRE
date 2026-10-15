def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def rectangle(points):
    top = quantile((p[2] for p in points), 0.85)
    surface = [p for p in points if float(p[2]) > top - 0.002]
    surface = surface[::max(1, len(surface) // 700)]
    best = None
    for index in range(401):
        t = (index - 200) / 200.0
        norm = (1.0 + t * t) ** 0.5
        ux, uy = 1.0 / norm, t / norm
        aa = sorted(float(p[0]) * ux + float(p[1]) * uy for p in surface)
        bb = sorted(-float(p[0]) * uy + float(p[1]) * ux for p in surface)
        lowa, higha = aa[0], aa[-1]
        lowb, highb = bb[0], bb[-1]
        area = (higha - lowa) * (highb - lowb)
        if best is None or area < best[0]:
            best = (area, ux, uy, lowa, higha, lowb, highb)
    area, ux, uy, lowa, higha, lowb, highb = best
    if highb - lowb > higha - lowa:
        ux, uy, lowa, higha, lowb, highb = -uy, ux, lowb, highb, -higha, -lowa
    return {"axis": (ux, uy), "a": (lowa, higha), "b": (lowb, highb), "top": top, "bottom": quantile((p[2] for p in points), 0.06), "surface": surface}

def observe(prompt):
    located = localize_object(prompt, camera_name=camera, target_frame=frame)
    if not located.ok or len(located.point_cloud.points) < 30:
        raise Exception("Could not locate " + prompt)
    return rectangle(located.point_cloud.points)

def downward(axis, opening_sign):
    ox, oy = -axis[1] * opening_sign, axis[0] * opening_sign
    qy = ((1.0 - ox) * 0.5) ** 0.5
    if oy < 0:
        qy = -qy
    return (0.0, ((1.0 + ox) * 0.5) ** 0.5, qy, 0.0)

def move(position, orientation):
    for attempt in range(2):
        motion = move_to_pose(Pose(position, orientation, frame), tolerance=0.012, max_steps=120, strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation"))
        if motion.ok:
            return motion
    raise Exception("Motion failed: " + str(motion.status))

report = {"task": get_task_context().language, "moves": []}
frame = get_robot_state().base_frame
camera = "agentview"
if camera not in get_observation().cameras:
    raise Exception("Missing wide camera")
if "primary" not in robosuite.get_controller_metadata()["controllable_gripper_arms"]:
    raise Exception("Task requires a gripper")
open_gripper()
nut = observe("brown square metal nut")
peg = observe("square peg")
ux, uy = nut["axis"]
amin, amax = nut["a"]
bmin, bmax = nut["b"]
length, width = amax - amin, bmax - bmin
low_end = [(-float(p[0]) * uy + float(p[1]) * ux) for p in nut["surface"] if float(p[0]) * ux + float(p[1]) * uy < amin + length * 0.18]
high_end = [(-float(p[0]) * uy + float(p[1]) * ux) for p in nut["surface"] if float(p[0]) * ux + float(p[1]) * uy > amax - length * 0.18]
if max(low_end) - min(low_end) < max(high_end) - min(high_end):
    ux, uy, amin, amax, bmin, bmax = -ux, -uy, -amax, -amin, -bmax, -bmin
hole_a, center_b = amin + width * 0.5, (bmin + bmax) * 0.5
handle_a = amax - (length - width) * 0.25
hole = (ux * hole_a - uy * center_b, uy * hole_a + ux * center_b)
handle = (ux * handle_a - uy * center_b, uy * handle_a + ux * center_b)
pu, pv = peg["axis"]
pa, pb = sum(peg["a"]) * 0.5, sum(peg["b"]) * 0.5
target = (pu * pa - pv * pb, pv * pa + pu * pb)
directions = ((pu, pv), (-pv, pu), (-pu, -pv), (pv, -pu))
opening_sign = -1.0 if ux < 0 else 1.0
grasp_q = downward((ux, uy), opening_sign)
grasp_z = 0.5 * (nut["top"] + nut["bottom"]) - 0.003
hover = max(nut["top"], peg["top"]) + 0.14
offset = handle_a - hole_a
release_z = peg["top"] + (grasp_z - nut["bottom"]) + 0.012
dest_axis = None
for candidate_axis in sorted(directions, key=lambda axis: -(axis[0] * ux + axis[1] * uy)):
    candidate_q = downward(candidate_axis, opening_sign)
    candidate_handle = (target[0] + candidate_axis[0] * offset, target[1] + candidate_axis[1] * offset)
    high_ik = solve_ik(Pose((candidate_handle[0], candidate_handle[1], hover), candidate_q, frame), backend="pyroki")
    low_ik = solve_ik(Pose((candidate_handle[0], candidate_handle[1], release_z), candidate_q, frame), backend="pyroki")
    if high_ik.ok and low_ik.ok:
        dest_axis, dest_q, dest_handle = candidate_axis, candidate_q, candidate_handle
        break
if dest_axis is None:
    raise Exception("No reachable square-symmetric placement")
report["geometry"] = {"hole": hole, "handle": handle, "target": target, "nut_axis": (ux, uy), "dest_axis": dest_axis, "nut_size": (length, width), "nut_z": (nut["bottom"], nut["top"]), "peg_top": peg["top"], "grasp_z": grasp_z, "dest_handle": dest_handle}
print(report)
move((handle[0], handle[1], hover), grasp_q)
move((handle[0], handle[1], grasp_z), grasp_q)
report["close"] = close_gripper()
move((handle[0], handle[1], hover), grasp_q)
if ux * dest_axis[0] + uy * dest_axis[1] < 0.8:
    for fraction in (0.25, 0.5, 0.75):
        ax = (1.0 - fraction) * ux + fraction * dest_axis[0]
        ay = (1.0 - fraction) * uy + fraction * dest_axis[1]
        norm = (ax * ax + ay * ay) ** 0.5
        intermediate_q = downward((ax / norm, ay / norm), opening_sign)
        intermediate = ((1.0 - fraction) * handle[0] + fraction * dest_handle[0], (1.0 - fraction) * handle[1] + fraction * dest_handle[1], hover)
        move(intermediate, intermediate_q)
move((dest_handle[0], dest_handle[1], hover), dest_q)
release_z = peg["top"] + (grasp_z - nut["bottom"]) + 0.012
move((dest_handle[0], dest_handle[1], release_z), dest_q)
open_gripper()
move((dest_handle[0], dest_handle[1], hover), dest_q)
for settle_tick in range(3):
    set_gripper(1.0)
result = report
