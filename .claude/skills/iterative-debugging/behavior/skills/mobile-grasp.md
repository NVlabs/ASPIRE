# Mobile grasping on R1 Pro

## Geometry top-down grasp first, Contact-GraspNet when it is top-down

```python
target = find_object("red radio", "head")          # score-gated, size-checked dict
report = attempt_grasps(target, max_attempts=4, pregrasp_m=0.10, lift_m=0.15)   # arm chosen by side
```

`attempt_grasps` (prelude v10) tries, in order: a grasp across the object's topmost narrow
feature (`top_slab_grasp_pose`: the radio's handle, a can's body), Contact-GraspNet's top-down
candidate if there is one, the same top feature with the fingers rotated 90 degrees, then the
two box grasps on the object's top (`top_down_grasp_poses`): pregrasp along the approach axis,
grasp, close, lift, and it returns after the first lift (or when the episode ends on success).
the harness convention: the pose's +Z column is the approach axis, so `along_axis(pose, 2, -0.10)`
is the pregrasp.

## Conventions that decide success

Measured on this harness (2026-09-12): the R1 Pro end-effector's +Z is the approach axis (it
points straight down with the arms hanging) and the fingers separate along eef Y. The prelude's
`top_down_grasp_poses` therefore rotates the arm-hanging orientation about world Z to put the
fingers across the object's short axis; `EEF_TO_FINGERTIP_M` and `GRASP_DEPTH_M` set how deep the
fingers go. If grasps consistently stop above or beside the object, these two constants are the
first suspect, not the perception. Contact-GraspNet candidates on the head camera came back with
their approach axis pointing up or sideways on the radio (verified 2026-09-12, seed 1), so
`select_grasp(strategy="top_down")` filters them and the geometry grasp is the primary path.

## The target is not an obstacle for the descent

The in-process planner drops any scene object whose bounding box contains the pose goal from
cuRobo's collision world for that plan (upstream's grasp primitive does the same with
`ignore_objects=[obj]`), so a grasp pose that straddles the object is plannable; the pregrasp
10 cm above it is planned with the object present. A plan that still fails at the grasp pose is a
reach or table-collision problem, not the object itself. Objects wider than the 12.7 cm finger
span (the radio's short side is 14.6 cm) cannot be grasped across that side; grasp a corner, a
handle or the other axis.

## Arm choice and reach

- `primary` is the left arm, `secondary` the right; the prelude's `choose_arm` picks the arm on
  the object's side of the base (`y > 0` in the base's own heading means left). Verified
  2026-09-12: the can approached 0.7 m ahead sat 26 degrees to the right and the left arm's
  plans all failed.
- Arm plans move the torso: the planner uses upstream's arm embodiment (only the base and the
  fingers are locked) and the adapter executes the planned torso motion with the arm. Measured
  2026-09-12 with the torso locked instead: from the hanging posture neither arm reaches any
  top-down target 0.35-0.65 m ahead at any height (0/48 IK checks); a forward lean
  (`move_torso([0.4, 0, -0.4, 0])`) reaches table height (0.6-0.75 m) but no fixed posture
  reaches the floor (0.12-0.25 m). So `solve_ik` failures with the torso free mean the pose is
  out of reach altogether: step closer (a floor can needs the base within about 0.5 m) or use the
  other arm; do not raise the pregrasp above 0.15 m, the workspace shrinks quickly overhead.
- Reset the torso (`behavior.reset_torso()`) after a failed attempt so the next plan starts from a
  known posture; `solve_ik(pose, arm=...)` is a 0.3 s reach check, a failed `move_to_pose` plan
  costs 6-10 s.

## The episode ends on success, mid-lift

`termination_reason: task_succeeded` arrives the moment the witness sees the object held and
5 mm up, usually while the lift is still executing: `move_to_pose` then returns
`status: terminated` with `ok=False`. Treat `result.terminated` on the lift as success (the
prelude does); do not open the gripper or try the next candidate. Verified 2026-09-12: the
first radio success latched at step 508 of an 898-step run because the program kept going.

## Retention

Lift straight up in `odom` z with `move_to_pose(offset_pose(eef, 0, 0, lift_m))`. The witness
needs only 5 mm of clearance while the grasp holds, so a short, slow lift is enough; a fast lift
that drops the object still fails.

## Unverified (ASPIRE prose)

The reference backed the Contact-GraspNet pose off by 5 cm along the approach axis before
transforming to the world, closed the gripper for about 100 ticks, and tried candidates until
`check_object_in_hand` reported a grasp; the closest equivalent here is the witness-visible lift.
