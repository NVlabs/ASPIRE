---
name: grasp
description: Structural template for pick-and-place programs. Covers the standard grasp-lift-place code skeleton, top-down and cylindrical side grasps, staged approach, and post-lift grasp verification. No task-specific strategies — those are discovered through experiment.
---

# Grasp — Code Template

> This skill provides the **structural skeleton** for pick-and-place programs.
> Task-specific strategies (object-specific offsets, SAM3 prompts, placement geometry)
> are discovered through experiment and added here as the skill library grows — each new
> strategy as its own subsection (trigger → code → evidence), modeled on *Cylindrical Side
> Grasp* below; tables are only for one-line thresholds.
> Programs cannot import modules — all helpers below are pure Python over public contracts.

---

## Standard Pick-and-Place Template

```python
def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * fraction)]

def cloud_quantiles(points, x_fraction=0.5, y_fraction=0.5, z_fraction=0.5):
    return (
        quantile((point[0] for point in points), x_fraction),
        quantile((point[1] for point in points), y_fraction),
        quantile((point[2] for point in points), z_fraction),
    )

base_frame = get_robot_state().base_frame
report = {"task": get_task_context().language}

# --- Localize pick object ---
picked = localize_object("<object prompt>", camera_name="agentview", target_frame=base_frame)
if not picked.ok or len(picked.point_cloud.points) < 20:
    raise ValueError("localization failed for pick object")
object_center = cloud_quantiles(picked.point_cloud.points)

# --- Grasp via grasp backend (preferred for irregular objects) ---
grasps = generate_grasps("agentview", picked.segmentation, backend="contact-graspnet")
grasp = select_grasp(grasps, strategy="top_down") if grasps.ok else None
if grasp is not None:
    grasp_pose = grasp.pose
else:
    # --- OR: top-down grasp from observed geometry (simple flat objects) ---
    downward = get_robot_state().end_effector_poses["primary"].quaternion_wxyz
    grasp_pose = Pose(
        (object_center[0], object_center[1], object_center[2] + 0.01),
        downward,
        base_frame,
    )

# --- Staged approach: pre-grasp above, descend, close ---
open_gripper()
pre_grasp = Pose(
    (grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + 0.12),
    grasp_pose.quaternion_wxyz,
    grasp_pose.frame,
)
if not move_to_pose(pre_grasp, tolerance=0.02, max_steps=250).ok:
    raise ValueError("pre-grasp approach failed")
if not move_to_pose(grasp_pose, tolerance=0.015, max_steps=250).ok:
    raise ValueError("grasp approach failed")
close_gripper()

# --- Lift, then VERIFY the object actually moved with the gripper ---
before_z = quantile((point[2] for point in picked.point_cloud.points), 0.5)
lift = Pose(
    (grasp_pose.position[0], grasp_pose.position[1], grasp_pose.position[2] + 0.15),
    grasp_pose.quaternion_wxyz,
    grasp_pose.frame,
)
move_to_pose(lift, tolerance=0.02, max_steps=250)

after = localize_object("<object prompt>", camera_name="agentview", target_frame=base_frame)
grasp_verified = (
    after.ok
    and len(after.point_cloud.points) >= 20
    and quantile((point[2] for point in after.point_cloud.points), 0.5) - before_z > 0.02
)
report["grasp_verified"] = grasp_verified

# --- Re-observe for placement target ---
target = localize_object("<target prompt>", camera_name="agentview", target_frame=base_frame)
if not target.ok:
    raise ValueError("localization failed for placement target")
target_center = cloud_quantiles(target.point_cloud.points)
surface_z = quantile((point[2] for point in target.point_cloud.points), 0.9)

# --- Transport ---
above_target = Pose(
    (target_center[0], target_center[1], lift.position[2]),
    grasp_pose.quaternion_wxyz,
    base_frame,
)
move_to_pose(above_target, tolerance=0.02, max_steps=300)

# --- Place ---
release = Pose(
    (target_center[0], target_center[1], surface_z + 0.03),
    grasp_pose.quaternion_wxyz,
    base_frame,
)
move_to_pose(release, tolerance=0.02, max_steps=250)
open_gripper()

result = report
```

---

## Cylindrical Side Grasp (bottles, cans)

Derive the grasp height from robust point-cloud quantiles instead of a fixed world height.
Handwritten poses from observed geometry (OBB, side-grasp, top-down) are often more reliable
than backend candidates for specific object classes — record which per object below.

```python
zs = sorted(float(point[2]) for point in picked.point_cloud.points)
z_low = zs[len(zs) // 10]
z_high = zs[(9 * len(zs)) // 10]
grasp_z = z_low + 0.5 * (z_high - z_low)
```

Form pre-grasp, grasp, and lift poses from the observed center and extent. Stage the approach
outside the object, preserve the grasp orientation and closed-gripper command through the lift,
and avoid abrupt transport.

---

## Key Conventions

**Grasp with `close_gripper()`, never `set_gripper(0.0)`** — they are not equivalent. `close_gripper()` accepts a *contact stall* as success: the jaw stops on the object, the call returns `ok=True`, and the harness records the commanded gripper as closed.

```python
close_gripper()                      # correct: contact stall counts as closed
# set_gripper(0.0)                   # WRONG when grasping: times out, leaves stale open command
lift = move_to_pose(above_pose)      # keeps commanding closed, object is retained
```

**Never infer grasp success from a completed call** — `close_gripper()` returning does not mean
the object is held. Verify with the post-lift re-localization pattern above.

**Motion result guard** — every `move_to_pose` / `move_to_joints` / `plan_motion` returns a typed
result; check `.ok` before depending on the motion. Requested unavailable backends return typed
failures; the harness never falls back implicitly.

**Backend choice** — `MotionStrategy()` defaults to PyRoki IK plus straight-line joint
interpolation. That default is **not collision-aware**, and it does not say so: the arm sweeps
whatever the interpolated path crosses, the call returns `ok=True`, and the damage shows up later
as a missing object or an unexplained failure. Select `trajectory_planner="curobo"` (planned
transit, PyRoki IK) or `pose_planner="curobo-integrated"` (cuRobo solves pose and path together)
for free-space transit, and keep interpolation for segments where contact is intended — the
observed cloud contains the target and any held object, so a collision-aware planner refuses the
approach into a grasp by construction.

`ik_solver="curobo"` with `trajectory_planner="interpolation"` is checked *nowhere*: standalone
IK answers against an empty world, and interpolation checks nothing. The returned `Trajectory`
reports `collision_aware=False` — read it rather than assuming that naming cuRobo anywhere in the
strategy bought a scene check.

For `solve_ik`, PyRoki or cuRobo are usually needed.

**Quaternions are wxyz** in every `Pose`.

**Wrist vs fingertips**: a public `Pose` targets the **end-effector frame** (LIBERO's
`robot0_eef`) — the grasp point between the fingertips, not the wrist. The −0.1 m conversion to
`panda_hand` happens inside the IK provider (`providers/pyroki/client.py`, `PANDA_EEF_TO_HAND_OFFSET_M`),
so a program never applies it. Treating the target as the wrist and adding your own offset puts
the grip 10 cm off.

**Grasp verification thresholds** — add validated per-object entries here as you discover them.
This table is for one-line signals and numeric thresholds only; a grasp *strategy* that needs
procedure or code (e.g. constructing a manual grasp pose when a backend fails) gets its own
subsection like *Cylindrical Side Grasp* above, with the working snippet:

| Object | Verification signal | Threshold | Notes |
|---|---|---|---|

---

## Object Geometry Utilities

```python
geometry = estimate_geometry(picked.point_cloud)
# geometry.pose (framed center + orientation), geometry.extents, geometry.point_count

# Z range (grasp height tuning), pure Python:
zs = [float(point[2]) for point in picked.point_cloud.points]
height = max(zs) - min(zs)
```
