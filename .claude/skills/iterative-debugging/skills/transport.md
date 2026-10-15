---
name: transport
description: Motion patterns for moving objects between locations — multi-step waypoints, safe transit sequences, staged Cartesian moves, collision avoidance during transport. Grows through experiment.
---

# Transport — Motion Patterns

> This skill covers **how to move** once an object is grasped: waypointing, safe transit,
> intermediate stops, and collision avoidance. Discovered through experiment — add each
> validated pattern as its own subsection (trigger → code → evidence), modeled on
> *Pre-Probe IK Conditioning* below; tables are only for one-line lookups.

---

## Waypoint Sequences

Add a subsection per multi-step motion pattern that prevents collisions, slip, or IK failures
during transport (trigger condition → waypoint code → why it works + evidence). Use the table
only for one-line pointers to those subsections.

| Pattern | Trigger | Section |
|---|---|---|
| IK pre-probe | Low-Z placement target | *Pre-Probe IK Conditioning* below |

---

## Pre-Probe IK Conditioning

Before grasping, pre-visit the placement target at multiple approach heights. This seeds the
IK solver so it stays on the correct branch when placing after the grasp.

```python
# BEFORE grasping, probe the target at decreasing heights:
for probe_z in (target_z + 0.08, target_z + 0.06, target_z + 0.04, target_z + 0.02):
    move_to_pose(Pose((target_x, target_y, probe_z), downward, base_frame),
                 tolerance=0.03, max_steps=200)
go_home()
# NOW grasp — IK is conditioned for placement
```

**Why it works**: IK solvers maintain local branch continuity from their last configuration.
Visiting the placement target before grasping seeds the solver on the branch that can reach
that XY at low Z — so after grasp + home reset, the arm finds the same branch.

**When to use**: Any task where the placement target is at low Z with an observed XY. When
branch conditioning is not enough, request collision-aware planning explicitly:
`move_to_pose(..., strategy=MotionStrategy(trajectory_planner="curobo"))`.

---

## Safe Transit (Lateral Escape Before Lift)

Add a subsection here (trigger → code → evidence) when the arm must move laterally before
lifting to avoid sweeping through obstacles (opened drawers, adjacent objects, cabinet edges).

---

## Placement Approach

Add a subsection here (trigger → code → evidence) for approach sequences above the target
(hover height, descent speed, drop vs. lower) that prevent bounce, tip, or miss.

### Physics Settling After Release

After releasing a grasped object, the object needs a few control ticks to settle before the
episode's success predicate is evaluated — observation calls do not advance time.

**`set_gripper(1.0)` does NOT buy those ticks.** When the jaw is already at the requested
width the adapter returns immediately with `steps_executed = 0`, so a program relying on it to
settle gets **no** ticks at all. Measured on `libero_goal_task/task_1` and independently on
`libero_spatial_task/task_4`, which lost a whole program version to the false hypothesis. On
LIBERO, Robosuite and RoboCasa `set_gripper` now also *refuses* any value between 0.0 and 1.0
with `unsupported`, so there is no partial-width call to fall back on either.

Advance time with something that actually steps. A one-tick hold of the arm's own measured
joints is the cheapest, and it commands the gripper explicitly rather than relying on a latched
value:

```python
open_gripper()
state = get_robot_state()
for tick in range(settle_ticks):       # measure how many you need; do not assume
    step(RobotAction(arms={"primary": ArmCommand(
        "joint_position",
        state.joint_positions["primary"],
        gripper_position=1.0,
        embodiment=state.embodiment,   # required: the default is "robosuite"
    )}))
```

Check `steps_executed` on whatever you use, rather than assuming a call advanced time.

---

## Basket / Container Placement

General rule: transit at `rim_z + 0.30`, release at `rim_z + 0.10`. Never descend inside the
container — drop from above.

**NEVER call `go_home()` while holding a grasped object.** Going home during transport opens the
arm configuration and drops the object. Transport in one continuous arc: lift → translate →
descend → release.

| Object type | Transit height | Release height | Notes |
|---|---|---|---|
