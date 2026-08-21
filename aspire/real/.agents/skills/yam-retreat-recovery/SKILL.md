---
name: yam-retreat-recovery
description: "Use for YAM post-release retreat, gripper-open confirmation, home/open recovery, retreat after rack/bin placement, and safe recovery while holding an object after a failed run."
---

# YAM Retreat Recovery

After placing or dropping an object, retreat before homing. Release success is
not enough if the fingers are still inside a rack/bin/shelf.

Validated patterns:

- Bottle rack: open fully on rack, retreat `-X` by computed rack-clearance
  standoff, then home and reopen. Current successful run used `0.14 m`.
- Bowl lower rack: post-release clear may use `+Z`, `X` retreat, or split
  diagonal/axis-limited retreat depending on shelf geometry.
- KitKat handover: after right hold, left opens and retreats along the clear
  lane; right retreats after left clears before trash transport.
- Home/open: normal YAM home can close grippers; use
  `open_grippers_return_home.py` when fingers must remain open.

If a run fails while holding an object, inspect current EEF pose and videos
first. If the object is safely held, plan bounded recovery from the actual pose:
clear lane, place/release, retreat, home/open. Ask for human reset only when the
scene is physically broken or unsafe for the robot to repair.
