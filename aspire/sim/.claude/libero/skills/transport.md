---
name: transport
description: Motion patterns for moving objects between locations — multi-step waypoints, safe transit sequences, interpolated Cartesian moves, collision avoidance during transport. Grows through experiment.
---

# Transport — Motion Patterns

> This skill covers **how to move** once an object is grasped: waypointing, safe transit,
> intermediate stops, and collision avoidance. Discovered through experiment — add entries as
> you validate them.

---

## Waypoint Sequences

Add entries here when you find a multi-step motion pattern that prevents collisions, slip, or
IK failures during transport. Include: trigger condition, waypoint sequence, why it works.

| Pattern | Trigger | Sequence | Notes |
|---|---|---|---|

---

## Pre-Probe IK Conditioning

Before grasping, pre-visit the placement target at multiple approach heights. This seeds the
IK solver so it stays on the correct branch when placing after the grasp.

```python
# BEFORE grasping, probe the target at decreasing heights:
for probe_z in [target_z + 0.08, target_z + 0.06, target_z + 0.04, target_z + 0.02]:
    goto_pose(np.array([target_x, target_y, probe_z]), TOP_DOWN_QUAT)
goto_home_joint_position()
# NOW grasp — IK is conditioned for placement
```

**Why it works**: IK solvers maintain local branch continuity from their last configuration.
Visiting the placement target before grasping seeds the solver on the branch that can reach
that XY at low Z — so after grasp + home reset, the arm finds the same branch.

**When to use**: Any task where the placement target is at low Z and has a fixed known XY.

---

## Interpolated Cartesian Motion

Add entries here when fine-grained step-by-step moves outperform direct `goto_pose` (e.g. for
fragile placements, long horizontal transport, or constrained workspaces).

---

## Safe Transit (Lateral Escape Before Lift)

Add entries here when the arm must move laterally before lifting to avoid sweeping through
obstacles (opened drawers, adjacent objects, cabinet edges).

---

## Placement Approach

Add entries here for approach sequences above the target (hover height, descent speed,
drop vs. lower) that prevent bounce, tip, or miss.

### Physics Settling After Release

After releasing a grasped object, call `get_observation()` several times to give the physics
engine time to settle the object before the episode's reward predicate fires.

```python
open_gripper()
for _ in range(3):
    get_observation()
```

**NEVER call `goto_home_joint_position()` while holding a grasped object.** Going home
during transport opens the arm configuration and drops the object. Transport in one continuous
arc: lift → translate → descend → release.

| Object type | Transit height | Release height | Notes |
|---|---|---|---|
