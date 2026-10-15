# BEHAVIOR-1K, R1 Pro, suite picking_up_trash, task 0: pick up the blue can of soda from the floor.
#
# Status: VERIFIED on the RTX 4090 bench, 2026-09-12 (prelude v10, pickup plan
# configs/validation/behavior-pickup.yaml): 2/3 on seeds 1-3 (seeds 2 and 3; seed 1 closed the
# fingers without a hold). The previous plan run (prelude v8) passed seed 3 only. A success takes
# 115-130 s wall time and 650-780 ticks. The robot starts in the kitchen and the cans lie
# on the living-room floor, so the policy searches by turning, drives to a free pose 0.7 m short
# of the can, tilts the torso to see the floor, re-approaches the can it sees from close range,
# and grasps across the can's body.
#
# prelude: behavior-r1pro-v10
# --- prelude begin ---
# Shared scaffolding for the BEHAVIOR-1K R1 Pro pickup programs. Pasted verbatim at the top of
# each example so every program shares the same tested search, approach and grasp helpers. Uses
# only the public program API plus `import math` (the one permitted import); no simulator internals.
#
# CALIBRATION (measured on the harness, 2026-09-12): the R1 Pro end-effector's +Z axis is the
# approach direction (it points straight down while the arms hang at reset) and the fingers
# separate along the end-effector's Y axis (12.7 cm apart when open). The eef origin is the tool
# centre between the fingertips, so EEF_TO_FINGERTIP_M is 0; GRASP_DEPTH_M is how far below the
# object's top the fingertips go.
import math

EEF_TO_FINGERTIP_M = 0.0
GRASP_DEPTH_M = 0.03
MIN_LOCALIZE_SCORE = 0.1
MAX_OBJECT_EXTENT_M = 0.8
CLOUD_MAD_SCALE = 4.0
CLOUD_MIN_HALF_M = 0.06
SEARCH_YAW_STEP_RAD = 0.5
SEARCH_MAX_TURNS = 12
EXPLORE_STEP_M = 1.0
EXPLORE_MAX_HOPS = 3
SLAB_M = 0.03
SLAB_MAX_WIDTH_M = 0.10
APPROACH_ANGLE_OFFSETS = (0.0, 0.5, -0.5, 1.0, -1.0)
APPROACH_DISTANCE_STEPS = (0.0, 0.25, 0.5)
STANDOFF_STEPS = (0.0, 0.15, 0.3)
REACQUIRE_RADIUS_M = 0.5
GRASP_REACH_M = 0.8

HOME = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
HOME_QUAT = (float(HOME[0]), float(HOME[1]), float(HOME[2]), float(HOME[3]))


def quat_multiply(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_about_z(angle):
    return (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))


def quat_axis(q, column):
    """Unit column of the rotation matrix of a wxyz quaternion (0 = x, 1 = y, 2 = z)."""
    w, x, y, z = q
    if column == 0:
        return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    if column == 1:
        return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def pose_axis(pose, column):
    return quat_axis(tuple(float(v) for v in pose.quaternion_wxyz), column)


def make_pose(position, quaternion_wxyz, frame):
    return Pose(
        position=(float(position[0]), float(position[1]), float(position[2])),
        quaternion_wxyz=tuple(float(v) for v in quaternion_wxyz),
        frame=frame,
    )


def offset_pose(pose, dx, dy, dz):
    p = pose.position
    return make_pose((float(p[0]) + dx, float(p[1]) + dy, float(p[2]) + dz), pose.quaternion_wxyz, pose.frame)


def along_axis(pose, column, distance):
    """The pose translated by ``distance`` metres along one of its own axes."""
    axis = pose_axis(pose, column)
    return offset_pose(pose, axis[0] * distance, axis[1] * distance, axis[2] * distance)


def trim_cloud(cloud, mad_scale=CLOUD_MAD_SCALE, min_half_m=CLOUD_MIN_HALF_M, margin_m=0.02):
    """Drop the depth tail a mask's edge pixels pick up from the background.

    Keeps the points within ``mad_scale`` median absolute deviations of the per-axis median (at
    least ``min_half_m`` each side). Verified 2026-09-12: the radio's raw cloud spanned 1.39 m
    along the view axis from a 27x40 px mask; the object itself is 0.32 m.
    """
    points = cloud.points
    n = len(points)
    if n < 12:
        return cloud
    lower = []
    upper = []
    for axis in range(3):
        values = sorted(float(p[axis]) for p in points)
        median = values[n // 2]
        deviations = sorted(abs(v - median) for v in values)
        half = max(mad_scale * deviations[n // 2], min_half_m) + margin_m
        lower.append(median - half)
        upper.append(median + half)
    try:
        return crop_point_cloud(cloud, lower, upper)
    except Exception:
        return cloud


def find_object(
    prompt, camera_name="head", min_score=MIN_LOCALIZE_SCORE, max_extent=MAX_OBJECT_EXTENT_M, near=None
):
    """Localize ``prompt`` with a score gate and a size sanity check.

    Returns a dict with ``segmentation``, ``point_cloud`` (trimmed of its background tail),
    ``geometry`` and ``score``, or None. A trimmed cloud that still spans more than
    ``max_extent`` metres is a background blob, not an object. ``near`` (an odom point) keeps
    the same instance when re-acquiring: candidates farther than REACQUIRE_RADIUS_M are skipped
    (verified 2026-09-12: three cans lie on the floor; from the approach pose the best-scoring
    "blue can of soda" was a different can 0.7 m to the side, out of reach).
    """
    masks = segment_text(camera_name, prompt)
    if not masks.ok:
        return None
    frame = get_robot_state().base_frame
    for candidate in sorted(masks.segmentations, key=lambda s: -float(s.score)):
        if float(candidate.score) < min_score:
            break
        try:
            cloud = trim_cloud(mask_to_point_cloud(candidate, camera_name, target_frame=frame))
            geometry = estimate_geometry(cloud)
        except Exception:
            continue
        if max(float(v) for v in geometry.extents) > max_extent:
            continue
        if near is not None:
            cx, cy, cz = cloud_centre(cloud)
            if math.hypot(cx - float(near[0]), cy - float(near[1])) > REACQUIRE_RADIUS_M:
                continue
        return {"segmentation": candidate, "point_cloud": cloud, "geometry": geometry, "score": float(candidate.score)}
    return None


def turn_in_place(delta_yaw, max_steps=200):
    x, y, yaw = behavior.get_base_pose()
    return behavior.navigate_to_pose(x, y, yaw + delta_yaw, planner="servo", max_steps=max_steps)


def back_up(distance_m, max_steps=300):
    """Reverse the base along its current heading without turning."""
    x, y, yaw = behavior.get_base_pose()
    return behavior.navigate_to_pose(
        x - math.cos(yaw) * distance_m, y - math.sin(yaw) * distance_m, yaw, planner="servo", max_steps=max_steps
    )


def search_by_turning(prompt, camera_name="head", step_rad=SEARCH_YAW_STEP_RAD, turns=SEARCH_MAX_TURNS):
    """Rotate the base in steps until ``prompt`` localizes; returns the find_object dict or None."""
    for turn in range(turns):
        found = find_object(prompt, camera_name)
        if found is not None:
            return found
        if not turn_in_place(step_rad).ok:
            break
    return None


def search_by_torso(prompt, camera_name="head", steps=4, near=None):
    """Tilt the torso down in stages until ``prompt`` localizes; returns the find_object dict or None."""
    meta = behavior.get_controller_metadata()
    current = list(meta["torso_positions"])
    lower = meta["torso_lower_bounds"]
    upper = meta["torso_upper_bounds"]
    for stage in range(1, steps + 1):
        fraction = stage / steps
        target = list(current)
        # Torso joint 1 leans back, joint 2 leans forward: together they pitch the head down.
        target[0] = current[0] + fraction * (lower[0] - current[0]) * 0.5
        target[1] = current[1] + fraction * (upper[1] - current[1]) * 0.5
        if not behavior.move_torso(target).ok:
            break
        found = find_object(prompt, camera_name, near=near)
        if found is not None:
            return found
    return None


def explore_for(prompt, hops=EXPLORE_MAX_HOPS, step_m=EXPLORE_STEP_M):
    """Turn to search; when nothing answers, drive ``step_m`` ahead (collision-aware) and retry."""
    for hop in range(hops + 1):
        found = search_by_turning(prompt)
        if found is not None:
            return found
        if hop == hops:
            break
        x, y, yaw = behavior.get_base_pose()
        moved = behavior.navigate_to_pose(
            x + math.cos(yaw) * step_m, y + math.sin(yaw) * step_m, yaw, planner="curobo", max_steps=1500
        )
        if not moved.ok:
            turn_in_place(math.pi / 2.0)
    return None


def cloud_centre(cloud):
    """Per-axis median of a point cloud."""
    points = cloud.points
    n = len(points)
    return tuple(sorted(float(p[axis]) for p in points)[n // 2] for axis in range(3))


def go_to_standoff(support_prompt, target, standoff_m=0.3):
    """Localize the support surface and drive to a free standoff pose facing ``target``.

    Falls back to free poses around the object itself when every standoff candidate is blocked
    (verified 2026-09-12: a partial table hull put the "nearest edge" at the far end, inside the
    couch, three times over).
    """
    support = find_object(support_prompt, "head", max_extent=3.0)
    if support is not None:
        for extra in STANDOFF_STEPS:
            x, y, yaw = behavior.plan_standoff_pose(
                support["point_cloud"], target["point_cloud"], standoff_m + extra
            )
            if not behavior.base_pose_is_free(x, y):
                continue
            moved = behavior.navigate_to_pose(x, y, yaw, planner="curobo", max_steps=1500)
            if moved.ok:
                return moved
    return approach_object(target, standoff_m + 0.35)


def within_reach(target, reach_m=GRASP_REACH_M):
    """Whether the object's centre is within ``reach_m`` of the base (the arms reach about
    0.65 m ahead at floor and table height with the torso helping)."""
    bx, by, byaw = behavior.get_base_pose()
    cx, cy, cz = cloud_centre(target["point_cloud"])
    return math.hypot(cx - bx, cy - by) <= reach_m


def approach_object(target, distance_m=0.7, max_steps=1500):
    """Drive to a free pose ``distance_m`` short of a free-standing object, facing it.

    Candidates step outward (0, +0.25, +0.5 m) and around the line of sight (0, +-0.5, +-1 rad);
    the traversability check skips poses inside furniture before any planning.
    """
    cx, cy, cz = cloud_centre(target["point_cloud"])
    bx, by, byaw = behavior.get_base_pose()
    line_of_sight = math.atan2(cy - by, cx - bx)
    for extra in APPROACH_DISTANCE_STEPS:
        distance = distance_m + extra
        for offset in APPROACH_ANGLE_OFFSETS:
            theta = line_of_sight + offset
            gx = cx - distance * math.cos(theta)
            gy = cy - distance * math.sin(theta)
            if not behavior.base_pose_is_free(gx, gy):
                continue
            moved = behavior.navigate_to_pose(gx, gy, theta, planner="curobo", max_steps=max_steps)
            if moved.ok:
                return moved
    return None


def top_down_grasp_poses(target, arm="primary"):
    """Two top-down eef targets on the object's top, fingers across its short then long axis."""
    points = target["point_cloud"].points
    xs = sorted(float(p[0]) for p in points)
    ys = sorted(float(p[1]) for p in points)
    zs = sorted(float(p[2]) for p in points)
    n = len(xs)
    top_z = zs[int(0.95 * (n - 1))]
    centre = (xs[n // 2], ys[n // 2], top_z - GRASP_DEPTH_M + EEF_TO_FINGERTIP_M)
    geometry = target["geometry"]
    # Horizontal object axes from the oriented box: the shortest horizontal extent first.
    axes = []
    for column in range(3):
        axis = pose_axis(geometry.pose, column)
        horizontal = math.hypot(axis[0], axis[1])
        if horizontal > 0.5:
            axes.append((float(geometry.extents[column]), math.atan2(axis[1], axis[0])))
    axes.sort()
    yaws = [yaw for extent, yaw in axes] or [0.0]
    finger_axis = quat_axis(HOME_QUAT, 1)
    home_yaw = math.atan2(finger_axis[1], finger_axis[0])
    frame = target["point_cloud"].frame
    poses = []
    for yaw in yaws[:2]:
        q = quat_multiply(quat_about_z(yaw - home_yaw), HOME_QUAT)
        poses.append(make_pose(centre, q, frame))
    return poses


def choose_arm(target):
    """The arm on the object's side of the base: ``primary`` (left) when the object is to the left."""
    x, y, yaw = behavior.get_base_pose()
    points = target["point_cloud"].points
    n = len(points)
    cx = sorted(float(p[0]) for p in points)[n // 2]
    cy = sorted(float(p[1]) for p in points)[n // 2]
    lateral = -math.sin(yaw) * (cx - x) + math.cos(yaw) * (cy - y)
    return "primary" if lateral >= 0.0 else "secondary"


def horizontal_axis(points):
    """Yaw of the principal horizontal axis of ``points`` (2D covariance)."""
    count = float(len(points))
    mx = sum(float(p[0]) for p in points) / count
    my = sum(float(p[1]) for p in points) / count
    sxx = sum((float(p[0]) - mx) ** 2 for p in points) / count
    syy = sum((float(p[1]) - my) ** 2 for p in points) / count
    sxy = sum((float(p[0]) - mx) * (float(p[1]) - my) for p in points) / count
    return 0.5 * math.atan2(2.0 * sxy, sxx - syy)


def fingers_pose(centre, finger_yaw, frame):
    """A top-down eef pose at ``centre`` with the fingers separated along ``finger_yaw``."""
    finger_axis = quat_axis(HOME_QUAT, 1)
    home_yaw = math.atan2(finger_axis[1], finger_axis[0])
    q = quat_multiply(quat_about_z(finger_yaw - home_yaw), HOME_QUAT)
    return make_pose(centre, q, frame)


def top_slab_grasp_pose(target, slab_m=SLAB_M, max_width_m=SLAB_MAX_WIDTH_M, across=True):
    """A grasp on the object's topmost feature when it is narrow enough to fit the fingers.

    The top ``slab_m`` of the cloud (a handle bar, a rim, a can's upper half) gives the grasp
    centre and height; the WHOLE cloud's principal horizontal axis gives the object's length
    direction (a 3 cm slab of a lying can is nearly square, so its own axis is unreliable:
    verified 2026-09-12, the fingers closed along the can and pushed it). ``across`` puts the
    fingers across that axis (the grasp that holds), ``across=False`` along it (a fallback).
    None when the slab is wider than the gripper opening (12.7 cm) or too sparse.
    """
    points = target["point_cloud"].points
    zs = sorted(float(p[2]) for p in points)
    n = len(zs)
    top_z = zs[int(0.98 * (n - 1))]
    slab = [(float(p[0]), float(p[1])) for p in points if float(p[2]) >= top_z - slab_m]
    if len(slab) < 10:
        return None
    count = float(len(slab))
    mx = sum(p[0] for p in slab) / count
    my = sum(p[1] for p in slab) / count
    major = horizontal_axis(points)
    finger_yaw = major + math.pi / 2.0 if across else major
    ca, sa = math.cos(finger_yaw), math.sin(finger_yaw)
    along_fingers = [(p[0] - mx) * ca + (p[1] - my) * sa for p in slab]
    if max(along_fingers) - min(along_fingers) > max_width_m:
        return None
    return fingers_pose((mx, my, top_z - GRASP_DEPTH_M), finger_yaw, target["point_cloud"].frame)


def attempt_grasps(target, arm=None, max_attempts=4, pregrasp_m=0.10, lift_m=0.15):
    """Try top-down grasps (top feature first, then Contact-GraspNet's top-down candidate, then
    the object's box); report.

    ``arm`` defaults to the arm on the object's side. Arm plans move the torso as well (the
    planner's arm embodiment keeps the four torso joints active), so a plan that fails with
    ``IK Fail`` means the pose is out of reach even with the torso: step closer or change arm.
    """
    if arm is None:
        arm = choose_arm(target)
    report = {"grasps_ok": False, "attempts": 0, "lifted": False, "arm": arm}
    candidates = []
    slab = top_slab_grasp_pose(target)
    if slab is not None:
        candidates.append(slab)
    slab_along = top_slab_grasp_pose(target, across=False)
    grasps = generate_grasps("head", target["segmentation"], backend="contact-graspnet", max_candidates=5)
    report["grasps_ok"] = grasps.ok
    if grasps.ok:
        best = select_grasp(grasps, strategy="top_down")
        if best is not None:
            candidates.append(make_pose(best.pose.position, best.pose.quaternion_wxyz, best.pose.frame))
    if slab_along is not None:
        candidates.append(slab_along)
    candidates.extend(top_down_grasp_poses(target, arm))
    for eef in candidates[:max_attempts]:
        report["attempts"] += 1
        pregrasp = along_axis(eef, 2, -pregrasp_m)
        open_gripper(arm=arm)
        if not move_to_pose(pregrasp, tolerance=0.03, max_steps=900, arm=arm).ok:
            continue
        if not move_to_pose(eef, tolerance=0.02, max_steps=600, arm=arm).ok:
            continue
        close_gripper(arm=arm)
        lifted = move_to_pose(offset_pose(eef, 0.0, 0.0, lift_m), tolerance=0.03, max_steps=600, arm=arm)
        # The episode ends the moment the witness sees the object held and 5 mm up, which can
        # interrupt the lift itself: a terminated lift is the task succeeding, not a failure.
        report["lifted"] = bool(lifted.ok or lifted.terminated)
        report["terminated"] = bool(lifted.terminated)
        if report["lifted"]:
            return report
        open_gripper(arm=arm)
    return report
# --- prelude end ---

report = {"stage": "search", "found": False, "approached": False, "reacquired": False}
open_gripper()

# 1. The cans are in the living room and the robot starts in the kitchen: turn, then drive ahead
#    a metre at a time and turn again, until a confident detection.
can = explore_for("blue can of soda")
if can is None:
    can = search_by_torso("blue can of soda")
report["found"] = can is not None

# 2. Drive to a free pose 0.7 m short of the can, facing it (candidates around the line of sight).
if can is not None:
    report["stage"] = "approach"
    moved = approach_object(can, 0.7)
    report["approached"] = bool(moved is not None and moved.ok)

# 3. Look down and re-acquire. Three cans lie on the floor and the close-range detection is the
#    one to trust (verified: from the kitchen the best "blue can" match was a different can), so
#    when the can seen from here is out of reach, drive to it and look again.
if report["approached"]:
    report["stage"] = "reacquire"
    for hop in range(2):
        can = search_by_torso("blue can of soda")
        if can is None:
            back_up(0.2)
            can = search_by_torso("blue can of soda")
        if can is None or within_reach(can):
            break
        behavior.reset_torso()
        if approach_object(can, 0.6) is None:
            break
    report["reacquired"] = bool(can is not None and within_reach(can))

# 4. Top-down grasp on the can, lift on success.
if report["reacquired"]:
    report["stage"] = "grasp"
    report.update(attempt_grasps(can, pregrasp_m=0.12))
    if not report["lifted"]:
        behavior.reset_torso()

result = report
