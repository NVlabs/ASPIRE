---
name: wrist-rotation-blocking
description: Use j[6] wrist rotation to reduce arm body footprint when descending to a position that is blocked by a nearby obstacle (shelf, wall, clutter). Allows reaching contact heights that are otherwise inaccessible due to arm-body (not just gripper-tip) collision.
---

# Wrist Rotation for Arm-Body Blocking

## Symptom

An approach position passes IK (`solve_ik` returns joints, `move_to_joints` reports completed), but the arm physically stops well above the target Z — typically 0.15–0.30m above the requested height. Confirmed by:
- `robot_cartesian_pos` after the move differs from IK target by >5cm in Z
- Keyframes show arm at a higher-than-expected height with the object uncontacted
- Trace: elapsed time anomaly (move completes faster than expected — motor controller hit saturation)

This happens when the **arm body** (elbow, forearm link, or wrist base) collides with the obstacle during descent, even though the gripper tip would have clearance. IK has no knowledge of these collisions.

## Technique

After `solve_ik()`, modify joint 6 (last wrist rotation) before calling `move_to_joints`:

```python
def mj(pos, quat):
    j = solve_ik(pos, quat)
    if j is not None:
        j[6] += np.pi / 2   # rotate wrist 90° — changes arm body configuration
    return j
```

**Relative (`+= π/2`) vs absolute (`= π/2`):**
- Use **relative** (`+= π/2`) as the default — it adjusts whatever IK returned, works across different IK seeds and configurations.
- Absolute (`= π/2`) can be useful for consistency if you need the same wrist orientation across all moves, but may overshoot or conflict with IK solutions that returned a large j[6].

The wrist rotation changes which arm links occupy which spatial positions, reducing physical profile in the obstacle's critical axis. It does **not** change the EEF position or orientation — IK pose is preserved.

## When to apply

- Arm-body blocks descent at a target Y/Z that is near a shelf, wall, or raised obstacle
- The approach is **directionally correct** but physically blocked
- Before declaring an approach "impossible" due to blocking, test it once with `j[6] += π/2`

## When NOT to apply

- If the obstacle is directly in the gripper's path (tip-level collision) — wrist rotation won't help
- If the arm body is colliding with the **target object itself** (wrong height estimate, not obstacle blocking)

## Cross-reference rule

When arm-body blocking is diagnosed (approach fails at target Z but IK succeeds), immediately check task_analysis.md §6 "Blocked, untested reconfiguration" list and generate at least one candidate that retries the blocked approach with this technique.

## Validated context

Franka Panda, top-down (`j[6]`) wrist rotation reduces arm profile in the lateral (Y) axis during descent. Confirmed effective for approaches near shelves/cabinets at Y offsets of 0.10–0.13m.
