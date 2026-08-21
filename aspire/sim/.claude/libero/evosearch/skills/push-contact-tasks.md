---
name: push-contact-tasks
description: Writing Evolutionary Search candidates for non-prehensile and contact tasks (push, slide, wipe, press). Covers collision-free approach corridors, vertical descent vs. lateral sweep, arm-target vs. object-target geometry, and blocking detection.
---

# Push / Contact Task Patterns

Non-prehensile tasks fail differently from pick-and-place. Key failure points:

---

## 1. Approach Corridor — Identify Before Writing

"Stand off opposite to where the object should go" is the default, but that ray often runs through clutter, shelves, or another object. Before writing, identify a **collision-free approach corridor** from the scene snapshot (task_analysis.md §5) and use it explicitly — even if it means approaching from the side rather than from behind the push direction.

**Shelf/obstacle avoidance — descend vertically, don't sweep laterally:**

If the approach direction is blocked (e.g. a shelf or cabinet is behind the object), do NOT sweep in at contact height from the standoff position. Instead, descend vertically onto the contact point:

```python
# BAD: sweeps laterally at table height through shelf zone
approach_xy = plate_center[:2] - push_dir * (plate_radius + 0.07)  # may hit shelf
move_to_joints(solve_ik([approach_xy[0], approach_xy[1], push_z], ...))  # arm rams shelf

# GOOD: drop straight down onto the back rim, then push
back_rim_xy = plate_center[:2] - push_dir * plate_radius
move_to_joints(solve_ik([back_rim_xy[0], back_rim_xy[1], push_z + 0.15], ...))  # above
move_to_joints(solve_ik([back_rim_xy[0], back_rim_xy[1], push_z], ...))          # contact height
move_to_joints(solve_ik([arm_target_xy[0], arm_target_xy[1], push_z], ...))      # push
```

---

## 2. Arm Target ≠ Object Final Position

The arm contacts the object's **back rim** and pushes. The arm target must be offset by the object radius so the object ends up at the desired location:

```python
# WRONG: arm goes to desired_plate_final — plate ends up radius PAST the target
arm_target = desired_plate_final           # plate overshoots by plate_radius!

# CORRECT: arm stops radius behind the desired plate final
arm_target = desired_plate_final - push_dir * plate_radius
# Verification: plate_final = arm_target + push_dir * plate_radius = desired_plate_final ✓
```

For short pushes where the object only needs to travel a small distance (e.g. 50mm), alpha-interpolation toward a distant landmark (e.g. stove center 250mm away) will massively overshoot. Compute the actual required displacement and work backward:

```python
# desired_plate_final determined from stove localization
desired_plate_final = np.array([stove_center[0], plate_center[1] + beta*(stove_center[1]-plate_center[1])])
push_dir = normalize(desired_plate_final - plate_center[:2])
arm_target = desired_plate_final - push_dir * plate_radius  # arm stops here, plate ends at desired
```

---

## 3. Detecting Physical Blocking (Fast Path, No Keyframes)

`move_to_joints` returning does NOT mean the arm moved freely. Joint solution can be satisfied while the arm is jammed against an obstacle.

- **Elapsed time anomaly**: trials completing 13–18s vs. the ~25s norm are often blocked early.
- **Trace: compare IK target vs actual EEF**: after each `move_to_joints`, call `get_observation()`. If `robot_cartesian_pos` differs from the IK target by >3cm, the arm was physically blocked.
- **Object displacement check** (fastest): compare object mask centroid at step 0 vs. final step without reading images:

```python
import numpy as np
m0 = np.load('keyframes/step_000_mask_0.npy')
mf = np.load('keyframes/step_010_mask_0.npy')
if abs(np.where(m0)[1].mean() - np.where(mf)[1].mean()) < 5:
    print("OBJECT DID NOT MOVE — arm was blocked or missed")
```

Also check `elapsed_s` in iter_summary.json — trials completing 30%+ faster than average typically hit an obstacle early.

---

## 4. When Blocking Is Detected

1. Read the blocked target positions — which obstacle is in the way (shelf? adjacent object?)
2. Check skill catalog: `find .claude/libero -name "*.md" | sort` — read matching files (e.g. `wrist-rotation-blocking.md`)
3. Change the *approach direction* or *arm configuration*, not just scalar offsets
4. Consider descending vertically onto the contact point instead of sweeping laterally
5. Consider wrist joint rotation (`j[6] += π/2`) to reduce arm-body footprint — see `wrist-rotation-blocking.md`
6. Consider two-phase approach — move object partially away from obstacle first, then approach from clear direction
7. **Do not mark an approach as Eliminated** until it has been retried with at least one arm reconfiguration technique from the skill catalog.
