# Search: finding the object before anything else

The head camera is the only wide view and the robot may start facing away from the target
(`picking_up_trash` starts in the kitchen; the cans are on the living-room floor). Search is a
program stage with its own budget.

## Turn in place, then tilt

**Trigger:** `localize_object` is not `ok` from the start pose.

```python
found = search_by_turning("red radio")           # prelude: 12 turns of 0.5 rad, servo planner
if found is None:
    found = search_by_torso("red radio")         # prelude: pitch the head down in 4 stages
if found is None:
    found = explore_for("red radio")             # prelude: drive 1 m ahead and turn again, 3 hops
```

`find_object` (what every search calls) rejects SAM3 scores below 0.1, trims the mask's point
cloud to 4 median absolute deviations around its median (`trim_cloud`), and rejects a trimmed
cloud that still spans more than 0.8 m. Verified 2026-09-12: from the kitchen start pose of
`picking_up_trash` the best "blue can of soda" match scored 0.007, a random object; without the
gate the program drove to it and Contact-GraspNet returned nothing. Also verified: a 27x40 px
radio mask at 2 m produced a raw cloud 1.39 m long along the view axis (edge pixels land on the
background), so the size gate must be applied to the trimmed cloud, never the raw one; the same
tail shifts `estimate_geometry`'s box and any standoff computed from it.

**Why:** the base yaw sweep covers azimuth, the torso sweep covers elevation. The reference
policy did exactly this; a floor object close to the base leaves the head's field of view
when standing tall. **unverified (ASPIRE prose)**: the reference used 0.5 rad steps and a torso
joint-1 back / joint-2 forward combination to look down.

## Re-acquire the same instance

Three cans lie on the living-room floor. Verified 2026-09-12 (`picking_up_trash` seed 1): from
the kitchen the best "blue can of soda" scored 0.24 and was a different can; from the approach
pose the close-range detections scored 0.6-0.8 on a can 0.7 m to the side, consistently. Trust
the close-range detection: when the re-acquired can is out of reach (`within_reach`, 0.8 m),
`approach_object(can, 0.6)` and look again (the trash program does two such hops). The
`near=` filter of `find_object` is for the opposite case, a single object with distractors.

## Prompts

| Target | Prompts that the reference used | Notes |
|---|---|---|
| radio | `"red radio"`, fallback `"radio"` | The radio sits on a low table in the living room |
| soda can | `"blue can of soda"`, fallback `"can of soda"` | Three cans on the floor; any of them may answer the prompt, the task target is the blue one |
| support | `"table"`, `"floor"` | Needed for `plan_standoff_pose`; the floor cloud is large, crop it near the object |

`localize_object` takes the best mask without a score gate and `segment_text` returns every
mask, which is why the prelude's `find_object` gates at 0.1; a `segment_text` set with several
masks means
the prompt is ambiguous, look at `media/overlays/` to see which one won.

## When search fails

Twelve turns without a detection costs about 12 servo moves plus 12 SAM3 calls. Before adding
more turns: try a shorter prompt, tilt the torso first, and check whether the object is simply
occluded from the start pose (then drive one metre into the room and search again).
