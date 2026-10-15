# Navigation: the odom frame, standoff and approach poses

## The frame does not move

`odom` is fixed where the base stood at reset. A radio localized before driving is still at the
same `odom` coordinates after driving; `behavior.get_base_pose()` returns where the base is now
in that same frame. Never re-express a stored point after a move.

## Two planners

| planner | use for | cost |
|---|---|---|
| `servo` | turns in place, hops under a metre, backing up | cheap, no collision check |
| `curobo` | room-scale moves, going around furniture | one cuRobo base plan (about 20 s when it fails, less when it succeeds), collision-checked against the scene; falls back to hops along the scene's traversability map |

`navigate_to_pose` reports `moved_x`, `moved_y`, `moved_yaw` in its diagnostics; a result that is
`ok` but barely moved means the goal was already reached, one that timed out with a large
remaining error means the base is blocked.

OmniGibson's cuRobo wrapper plans the base with trajectory optimisation only (no graph search),
so a single plan to a goal behind furniture or in another room fails with `TrajOpt Fail`
(verified 2026-09-12: the approach pose of a can 3.7 m away across the living room, and a
standoff pose behind the coffee table). The harness then asks the scene's traversability map
for the shortest path, splits it into 1 m hops, and plans each hop with cuRobo, servoing a hop
whose plan fails. The diagnostics say which happened: `route` is `direct`, `traversability` or
`none`, `hops` and `servo_hops` count the legs, `direct_status` is cuRobo's reason for the direct
plan (`TrajOpt Fail` = blocked path, `IK Fail` = the goal pose itself is in collision). An
`IK Fail` goal needs a different goal, not a route: stand farther off or on the other side.

## Standing off a table

**Trigger:** the target is on a surface (the radio on the table).

```python
radio = find_object("red radio")                       # prelude dict: {"point_cloud", "geometry", ...}
moved = go_to_standoff("table", radio, standoff_m=0.3)   # prelude: plan_standoff_pose + free check
```

**Why:** the pose sits 0.3 m outside the table edge nearest the object, facing it; the arm
reaches over the edge. **Verified 2026-09-12** (radio seeds 1–3): this geometry, taken from ASPIRE,
with a 0.3 m buffer.

`go_to_standoff` (prelude) tries 0.3, 0.45 and 0.6 m standoffs through `base_pose_is_free` and
then falls back to `approach_object(target, 0.65)`. Verified 2026-09-12 (radio, seed 1 in the
pickup plan): the head sees only part of the table, the hull's "nearest edge" was the far short
edge, and all three standoffs sat inside the couch; the run gave up after 40 ticks.

## Approaching a floor object

**Trigger:** the target stands free (a can on the floor).

```python
moved = approach_object(can, 0.7)      # prelude: free candidates around the line of sight
```

`approach_object` tries 0.7, 0.95 and 1.2 m short of the object at five angles around the line
of sight and skips every candidate `behavior.base_pose_is_free(x, y)` rejects (the scene's
traversability map eroded by the base footprint). Verified 2026-09-12: the plain line-of-sight
pose 0.7 m short of a can sat inside the coffee table twice; cuRobo needed 27 s to say `IK Fail`
and the map has no route to a blocked cell. `navigate_to_pose` now refuses such a goal in
milliseconds (`route: goal_blocked`), so the candidate loop is cheap.

**Why:** 0.7 m short of the object leaves the arm room to reach down in front of the base.
**unverified (ASPIRE prose)**: the reference stopped 0.7 m short and backed up 0.2 m to
re-acquire when the object dropped out of view.

## Backing up

```python
back_up(0.2)     # prelude: reverse along the current heading with the servo planner
```
