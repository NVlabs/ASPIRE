# Motion Efficiency Patterns for Evolutionary Search Candidates

Each `move_to_joints()` call blocks until the robot converges to a joint target (up to 120 sim steps). The number of calls is the primary driver of wallclock time. A pick-and-place that takes >30s is using too many waypoints.

---

## Call Budget

| Phase | Target calls |
|-------|-------------|
| Pre-grasp approach | 1 |
| Grasp descent | 1 |
| Lift | 1 |
| Transport to target | 2–3 |
| Placement descent | 1–2 |
| **Total** | **≤ 10** |

---

## Anti-patterns

**Fine-grained arc loops** — biggest offender. Each loop iteration is a full blocking move.
```python
# BAD: 20 calls = up to 2400 sim steps
for i in range(1, 21):
    t = i / 20
    wp = lift_pos + t * (above_target - lift_pos)
    wp[2] += 0.05 * 4 * t * (1 - t)
    move_to_joints(solve_ik(wp.tolist(), quat.tolist()))
```

**Long retry loops** — retrying the grasp 5 times × 3 moves each = 15 calls before even lifting.

**Fine-grained descent** — 8-step placement descent is 8 calls; 2 is enough.

---

## Preferred patterns

**3-point arc transport** (~3 calls):
```python
above_target = np.array([target_center[0], target_center[1], lift_z])
lift_pos = np.array([grasp_pos[0], grasp_pos[1], lift_z])
apex = (lift_pos + above_target) / 2
apex[2] += 0.05  # clearance bump
move_to_joints(solve_ik(apex.tolist(), quat.tolist()))
move_to_joints(solve_ik(above_target.tolist(), quat.tolist()))
```

**2-step grasp retry** (cap at 2 configs, not 5):
```python
for yaw_deg in [30, 45]:
    ...
    if gw > 0.03:
        break
    open_gripper()
    move_to_joints(solve_ik(pre_grasp.tolist(), quat.tolist()))
```

**Direct placement** (1–2 calls):
```python
release_z = surface_z + 0.03
move_to_joints(solve_ik([tx, ty, release_z], quat.tolist()))
open_gripper()
```

---

## Tradeoff rule

If two candidates achieve similar success rates, always prefer the one with fewer `move_to_joints` calls. More waypoints rarely help once the robot is close enough — the controller handles the final convergence internally.
