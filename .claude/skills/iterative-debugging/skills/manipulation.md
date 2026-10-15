---
name: manipulation
description: Non-pick-and-place manipulation patterns — drawer opening/closing, knob/switch turning, pushing, and other articulated object interactions. Grows through experiment.
---

# Manipulation — Articulated Object Patterns

> This skill covers tasks that are **not** pick-and-place: interacting with drawers, knobs,
> switches, and other articulated objects. Discovered through experiment — add each validated
> pattern as a subsection or bolded principle with its working code snippet (trigger → code →
> evidence), like the *Drawer Interaction* principles below.

---

## Knob / Switch Turning

**General approach**: Localize the knob, approach top-down, close the gripper at the knob
surface, then rotate the wrist through sequential yaw angles using `solve_ik` +
`move_to_joints`.

**Guard every motion** — `solve_ik` and `move_to_pose` return typed results; check `.ok` and
stage the pose (or switch backend explicitly) instead of assuming the far workspace is
reachable:

```python
ik = solve_ik(grasp_pose)
if not ik.ok:
    raise ValueError("IK failed for grasp pose")
move_to_joints(ik.joint_positions, tolerance=0.02, max_steps=200)
```

**Z disambiguation**: When multiple objects match the same prompt, filter candidates by
projected max Z to select the correct one by height.

Add validated patterns here as you discover them — trigger condition, the working code snippet,
and which task/seeds it fixed.

---

## Hinged Door Opening / Closing

**General approach**: Localize the handle, compute hinge geometry from the observed object
center + handle position, then arc-pull (or arc-push) the door via a sequence of staged poses
along the arc (a sweep of small `move_to_pose` steps).

**Critical**: Door/object rotation in the scene affects hinge computation — confirm object
orientation from the scene keyframes before deriving hinge offsets.

Add validated patterns here as you discover them — trigger condition, the working code snippet,
and which task/seeds it fixed.

---

## Pushing / Sliding

Add a subsection here (trigger → code → evidence) for pushing objects without grasping: contact
approach, force direction, step size, distance control.

---

## General Notes

- Add observations about which joint (e.g. the wrist) is most useful for in-place rotation
- Add notes on approach directions that avoid wrist limits
- Add notes on contact detection via gripper diagnostics or motion-result failures
