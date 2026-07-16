# Drawer Skill Transfer Guide

Purpose: a compact guide for an agent transferring the LIBERO drawer skill to a real robot or a new environment. The key idea is not to memorize drawer coordinates; it is to preserve the manipulation structure and re-measure the scene every time.

## Skill Summary

Drawer opening is an articulated-object manipulation skill:

1. Localize the drawer handle.
2. Approach with a horizontal gripper, not top-down.
3. Push slightly past the handle so the fingers straddle the bar.
4. Close on the handle.
5. Pull outward.
6. Lift while still gripping.
7. Release only after lifting clear.

For tasks that include "and close it", always attempt the close step even if the place step partially failed. Closing is often the final success predicate.

## Transfer Principles

Use the learned pattern, but re-probe the real scene:

- Drawer geometry is scene-specific: handle pose, pull direction, travel distance, friction, and end stops vary.
- Do not hardcode LIBERO coordinates on a real robot.
- Use perception to find the handle, then use small guarded motions to confirm contact.
- Treat the opening direction as a hypothesis. In LIBERO it was usually +Y toward the robot; in the real world infer it from cabinet pose or probe with small motions.
- Keep forces and velocities low near the handle and drawer stops.
- Require the cabinet/drawer body to be physically stable before contact probes.
  A freestanding wooden drawer box tipped during a real low precontact retry
  (`logs/reset_open_robot_20260608T222949`, 2026-06-08), so continuing was
  unsafe until the scene was manually reset or fixture-stabilized.
- Do not blindly take the highest top-camera "drawer handle" mask. On
  2026-06-08, the top detector selected a rear/upright handle at
  `[1.108, 0.351, 0.909]` instead of the small front drawer pull
  (`logs/drawer_top_pull_push_one_shot_loop_20260608T222520`). Cross-check with
  drawer-front masks and bottom-camera video before physical contact.
- Re-observe after opening before any pick/place subtask.
- Home or move to a neutral pose between subtasks if the previous manipulation left the arm in an awkward IK branch.

## Open Drawer Pattern

### Gripper Orientation

Use a horizontal grasp with fingers aligned along the handle axis. In LIBERO this was:

```python
HORIZONTAL_DRAWER_QUAT = [0.707, 0.707, 0.0, 0.0]  # wxyz
```

For a real robot, compute the equivalent orientation from the handle frame:

- gripper closing axis spans across the handle bar
- approach axis points into or slightly behind the handle
- wrist clears cabinet edges

### Contact Sequence

Do this sequence slowly:

```text
pre-approach -> near handle -> push slightly behind handle -> close -> pull -> lift -> release
```

The important LIBERO detail was the "push-past" step. If you close too early, the fingers hit the face of the handle instead of catching behind it.

### Release Sequence

After pulling open:

```text
pull to open limit -> lift while still gripping -> open gripper -> retreat/home
```

Do not open the gripper before lifting clear. In simulation, the drawer could spring back before success registered; in the real world, the same ordering prevents the handle from dragging the gripper or reclosing the drawer.

## Close Drawer Pattern

Closing is less parameterized in the LIBERO notes, but the structure is clear:

1. Re-localize the open drawer or handle.
2. Approach from the safe outside direction.
3. Make gentle contact with the handle/front.
4. Push opposite the opening direction.
5. Stop on visual closure, force/contact threshold, or a motion limit.
6. Retreat.

For real transfer, closing should usually be a guarded push, not a committed hardcoded displacement.

## Explicit Code

This code is written in the style of the Aspire/LIBERO reduced API, but the same structure maps to a real robot controller. Replace perception and motion primitives with the real equivalents.

```python
import numpy as np

HORIZONTAL_DRAWER_QUAT = [0.707, 0.707, 0.0, 0.0]  # LIBERO wxyz example


def safe_move(pos, quat=HORIZONTAL_DRAWER_QUAT):
    joints = solve_ik(list(pos), list(quat))
    if joints is None:
        print(f"IK failed: {pos}")
        return False
    move_to_joints(joints)
    return True


def observe_agentview():
    obs = get_observation()
    cam = obs["agentview"]
    return (
        obs,
        cam["images"]["rgb"],
        cam["images"]["depth"],
        cam["intrinsics"],
        cam["pose_mat"],
    )


def select_drawer_handle(rgb, depth, K, E, drawer="best"):
    prompts = ["silver handle", "metal drawer handle", "drawer handle"]
    candidates = []

    for prompt in prompts:
        for det in segment_sam3_text_prompt(rgb, prompt):
            mask = det["mask"].astype(np.uint8)
            pts = mask_to_world_points(mask, depth, K, E)
            if pts is None or len(pts) < 20:
                continue

            center = np.median(pts, axis=0)
            candidates.append({
                "prompt": prompt,
                "score": float(det.get("score", 0.0)),
                "mask": mask,
                "center": center,
                "pts": pts,
            })

    if not candidates:
        return None
    if drawer == "top":
        return max(candidates, key=lambda c: c["center"][2])
    if drawer == "bottom":
        return min(candidates, key=lambda c: c["center"][2])
    return max(candidates, key=lambda c: c["score"])
```

### Open Drawer

```python
def open_drawer(drawer="top", pull_dir=np.array([0.0, 1.0, 0.0]), pull_distance=0.30):
    goto_home_joint_position()
    obs, rgb, depth, K, E = observe_agentview()

    handle = select_drawer_handle(rgb, depth, K, E, drawer=drawer)
    if handle is None:
        print("No drawer handle found")
        return False

    # Use grasp planner for handle center, but override orientation with drawer orientation.
    grasp_poses_cam, grasp_scores = plan_grasp(depth, K, handle["mask"])
    if grasp_poses_cam is None or len(grasp_scores) == 0:
        print("No handle grasp found")
        return False

    best = int(np.argmax(grasp_scores))
    grasp_world = E @ grasp_poses_cam[best]
    grasp_pos, _ = decompose_transform(grasp_world)
    quat = HORIZONTAL_DRAWER_QUAT

    open_gripper()

    # Approach from outside, then push slightly past handle before closing.
    for offset in [0.10, 0.05, 0.02, -0.015]:
        p = grasp_pos + np.array([0.0, offset, 0.0])
        safe_move(p, quat)

    close_gripper()

    pull_target = grasp_pos + pull_dir / (np.linalg.norm(pull_dir) + 1e-6) * pull_distance
    safe_move(pull_target, quat)

    # Lift before release.
    lift_target = pull_target + np.array([0.0, 0.0, 0.18])
    lift_target[2] = max(lift_target[2], 0.45)
    safe_move(lift_target, quat)

    open_gripper()
    goto_home_joint_position()
    return True
```

### Close Drawer

For real robots, make this a guarded push if possible. In LIBERO-style code:

```python
def close_drawer(drawer="bottom", close_dir=np.array([0.0, -1.0, 0.0]), push_distance=0.30):
    goto_home_joint_position()
    obs, rgb, depth, K, E = observe_agentview()

    handle = select_drawer_handle(rgb, depth, K, E, drawer=drawer)
    if handle is None:
        print("No drawer handle found for close")
        return False

    center = handle["center"]
    quat = HORIZONTAL_DRAWER_QUAT
    close_dir = close_dir / (np.linalg.norm(close_dir) + 1e-6)

    open_gripper()

    # Approach from outside, then make contact.
    for offset in [0.10, 0.05, 0.02]:
        safe_move(center - close_dir * offset, quat)

    close_gripper()
    safe_move(center + close_dir * push_distance, quat)

    open_gripper()
    safe_move(center - close_dir * 0.05 + np.array([0.0, 0.0, 0.10]), quat)
    goto_home_joint_position()
    return True
```

### Open, Place, Close Skeleton

```python
def drawer_task():
    opened = open_drawer(drawer="bottom")

    # Always re-observe after opening. Do not trust stale object poses.
    goto_home_joint_position()
    obs, rgb, depth, K, E = observe_agentview()

    # Insert object-specific pick/place here:
    # 1. localize object
    # 2. grasp object
    # 3. localize drawer interior or opening
    # 4. place object inside

    # Always attempt close if the task asks for it.
    closed = close_drawer(drawer="bottom")

    for _ in range(3):
        get_observation()

    return opened, closed
```

## Real YAM Transfer Note: Top Drawer, 2026-06-08

The LIBERO push-past sequence transferred cleanly as structure, but not yet as a safe real contact motion.

- Evidence `logs/drawer_top_pull_push_one_shot_loop_20260608T210030`: the only full-preview plan for the top drawer needed a high handle z offset and had large orientation residuals.
- Evidence `logs/drawer_top_pull_push_one_shot_loop_20260608T210201`: physical `pregrasp` reached a diagnostic pose, but camera video showed the gripper high/outside the handle rather than centered for closing.
- Evidence `logs/drawer_top_pull_push_one_shot_loop_20260608T210441`: removing the high offset made handle-height candidates fail before selection, with planner/collision failures around the low side approach.

Reusable rule: before `grasp_handle`, `pull_open_small`, or `push_close_small`, require both a selected no-motion preview and a visual check from the latest physical `pregrasp` showing the fingers centered at handle height. If the plan is only reachable by moving above the detected handle, keep it diagnostic and rework the approach geometry or camera-clearance strategy before contact.

## Real YAM Transfer Note: X-Axis Top Drawer Pull, 2026-06-09

The top-drawer layout behaved like an X-axis drawer, not a Y-axis drawer. A
Y-axis close/push attempt reached the handle region but mostly slid/pressed
sideways (`logs/drawer_manual_probe_20260609T032037`). A later wrapper run
observed the handle, seated slightly past it, closed in stages, pulled roughly
`-X`, lifted/retreated while gripping, then opened after retreat:

- Pre-observe: `logs/drawer_top_pull_push_one_shot_loop_20260609T123305`
- X-axis pull run: `logs/drawer_manual_probe_20260609T123311`
- Post-observe: `logs/drawer_top_pull_push_one_shot_loop_20260609T123522`

Video frames showed visible drawer opening during the `-X` pull window, and
post-observe changed from a front handle detection near
`[0.75365, 0.23038, 0.71743]` to a much more exposed/open drawer view near
`[0.76424, 0.46381, 0.88304]`. Reusable rule: for this physical top-drawer
setup, start with a fresh observe and an X-axis opening hypothesis
(`OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY=-1,0`). Do not increase Y travel when
video shows handle contact without capture; fix seating, gripper close, and
X-axis pull geometry first.

Follow-up video review showed the `20260609T123311` opening was still not a
clean handle grasp: the gripper did not seat deeply behind the handle and
`target_gripper=0.48` remained too open. Updated wrapper defaults should use a
deeper X push-past around `0.050 m`, staged close through `0.54 -> 0.10`, and
close validation before treating an open as a true handle-capture success. The
current wrapper default pull is `0.155 m` along `-X`.

## Real YAM Transfer Note: Left-Arm Top Drawer, 2026-06-09

After reset, the useful real pull axis for the tabletop top drawer was `0,-1`
with a side/horizontal wrist near display RPY `[60, -90, -180]`. The transferred
LIBERO sequence worked only after two real-robot fixes:

- Close commands through the normal `set_gripper` path could report success
  while the measured left gripper stayed near open. Use the
  `servo_ee_delta(..., delta=[0,0,0], gripper_pos=...)` workaround for drawer
  close targets; evidence `logs/drawer_manual_probe_20260609T025432`.
- For opening, do not release at the pulled pose. Keep the gripper closed
  through the retreat/pull-away motion, then release after retreat. Evidence
  `logs/drawer_manual_probe_20260609T030810` shows the drawer visibly opened
  and post-observe `logs/drawer_top_pull_push_one_shot_loop_20260609T031050`
  measured left handle y moving from about `0.22376` to `0.23746`.

The best left-arm open primitive was:

```text
handle from fresh side observe
x_bias ~= +0.015 m
y_bias ~= +0.18 m
pull_axis_xy = 0,-1
push_past ~= 0.05 m
pull ~= 0.09 m
retreat ~= 0.13 m
target/contact/retreat gripper ~= 0.55
skip open after pull; open after retreat
```

Closing partially transferred as a guarded push opposite the same axis, but was
not fully solved. Evidence `logs/drawer_manual_probe_20260609T031211`,
`logs/drawer_manual_probe_20260609T031635`, and
`logs/drawer_manual_probe_20260609T032037` showed inward drawer motion during
contact; final observes through
`logs/drawer_top_pull_push_one_shot_loop_20260609T032246` still showed a
residual opening. The likely remaining issue is release/retreat geometry: the
gripper reopens only to about `0.76` near the handle, so retreat can catch the
handle and undo some of the push. Stronger push distance alone did not close the
last gap. Next attempts should push the drawer face or lower front surface and
clear laterally/upward before any frontward retreat, rather than pushing through
the handle center and then retreating through the handle path.

## Real-World Checklist

Before execution:

- Is the handle detected and spatially plausible?
- Is the selected drawer correct (top vs bottom)?
- Is the pull direction known from geometry, or should the robot probe?
- Is there enough clearance for the wrist near cabinet edges?
- Are force/velocity limits low near contact?

During execution:

- Stop if contact is much earlier than expected.
- Stop if the handle is lost from the gripper.
- Stop at a travel limit; do not over-pull.
- Re-observe after opening.

For close:

- Push opposite the observed opening direction.
- Prefer guarded force/contact termination over fixed distance.
- Always attempt close when the task includes "and close it".
