---
name: yam-transport
description: "Use for YAM held-object transport after pickup, rack/bin approach waypoint ordering, object-relative placement, avoiding shelf/rack collisions, and choosing staged axis moves."
---

# YAM Transport

For held objects, prefer simple staged waypoints over one large compound move
near racks, shelves, bins, or the other arm.

Validated patterns:

- Bottle rack: compute rack-aware `-X` clearance, shift `-Y`, lift `+Z`, then
  high place and release. Source:
  `grasp_lift_place_bottle_rack_one_shot_loop.py`.
- KitKat trash: stage outside bin, orient gripper to point `+X`, move `+X` into
  bin with Y/Z fixed, lower `-Z`, release, retreat. Source:
  `handover_chocolate_bar_left_to_right_one_shot_loop.py`.
- Bowl lower rack: transport to shelf mouth/high front pose, then use bounded
  direct axis moves for insertion/release. Source: `bowl_lower_rack_common.py`.
- Can trash: lift to transfer clearance, move over/near trash center plus bias,
  release, verify with post observation. Source:
  `pick_can_place_in_trash_can_one_shot_loop.py`.

When possible, calculate standoff from live fixture detections rather than
hardcoding a distance. If cuRobo is sensitive, split movement by axis:
clearance, lateral lane, vertical lift, approach, drop.
