---
task: "Nest all the cups upside down into one stable stack."
---

# Inverted cup nesting

## Goal and setup

Make one freestanding nested stack of all requested cups, with every opening facing
the table. Each added cup surrounds the cup beneath it; this is not a pyramid or a
cup balanced on another cup's closed end. Count the actual cups at the start and
maintain that inventory through recovery. Do not assume a fixed count of three.

Use compatible tapered cups on a stable reachable tabletop. Identify the opening,
closed end, body axis, and orientation of each cup. Prefer an already inverted cup
as the base and add other already inverted cups before those needing inversion.
Keep the surviving stack's count and location separate from the loose-cup inventory.
If none is already inverted, invert one with the procedure below, place it opening
down on the table, and verify independent stability before using it as the base.

Use native CAP views, measured robot state, and localization when metric grounding
is needed. Act through `cap_move_to_pose`, `cap_set_gripper`, and, when useful,
`cap_move_synchronized` or `cap_move_to_joints`; their current MCP schemas are
authoritative. Full poses use metres in a declared frame and either fixed-axis XYZ
`rpy_deg` or `quaternion_wxyz`. Use separate grip commands and preserve collision
planning. This is a low-level manipulation recipe without WAM or a VLM dependency.
Joint targets use radians. Gripper position runs from 0 closed to 1 open and does
not measure gripping force; choose closure from retained, non-crushing pad contact.

## Acquire a retained body grasp

1. Choose an approach that reaches opposing **external sidewalls** without landing
   a finger on the closed end or catching the rim. Leave the wide opening clear
   enough to surround the receiving cup later. Plan for the final placement, not
   just a reachable pickup.
2. Reject a localization mask on the wrong cup or implausible depth/size. Separate
   nearby cups with a discriminative query or another useful calibrated view. Do
   not descend toward a new estimate that conflicts with visible geometry.
3. Approach with open fingers, establish side contact, close without crushing, and
   lift clear. Verify the cup follows and does not slide or tilt in the pads. A
   closure value or successful EEF motion alone is not a grasp check.
4. If the fingers catch the rim and tip the cup, withdraw and locate the cup in its
   new orientation. A sideways cup can be regrasped at its exposed body; do not
   repeat the former approach as though it were still upright.

## Invert only as needed

Leave an already inverted cup inverted. For another cup, lift first until its full
swept volume, the fingers, and the wrist camera clear the table and other arm.
Then change its orientation using reachable full poses. Confirm the closed end is
above the opening from an external view; the rotating wrist camera and commanded
wrist angle do not establish the cup's world orientation.

Do not relax translation constraints to obtain a rotation. An orientation-only
goal can move the cup down into the table and invalidate the grasp. If a full pose
is unreachable, change the staging pose, divide the rotation into reachable parts,
or use a supported handoff instead of forcing wrist limits.
Dividing a rotation alone does not restore exhausted joint travel; change the
presentation or establish support for a regrasp when additional range is needed.

For a handoff, the receiving hand approaches exposed opposing sidewalls clear of
the donor, closes, and establishes support. Only then open the donor, verify
retention, and retract it before turning the receiver. Either arm can retain the
cup; transferring it back is unnecessary if the new grasp is useful. Do not twist
a cup between two rigidly closed grips. A table-supported regrasp is an alternative
when a clear handoff is unavailable.

## Align, insert, then release

1. Re-ground the current receiver after any contact or shift. Account for the
   stack's increased height. Align the held cup's **opening center and axis**, not
   the EEF origin, above the receiving cup's closed end. Correct tilt and lateral
   offset while separated.
2. Use top and side evidence together when needed: overlap in one projection can
   hide an edge-to-edge collision. Grasp offset can require a non-nominal wrist
   orientation even when the cup itself is vertical.
3. Lower in bounded steps while watching the receiver enter the opening and the
   walls overlap around it. The receiving stack should remain stationary. A rim
   touch, tilted perch, or apparent alignment is not yet insertion.
4. If an edge catches or the stack moves, **keep the cup gripped**, stop descent,
   and unload the contact with a bounded correction. Withdraw/lift clear only while
   the receiver remains table-supported; if it follows, stop rather than lifting a
   coupled stack. Check which earlier nesting relations survived, then correct
   against the now-visible opening and current receiver position.
5. Once insertion provides visible support, open the gripper, inspect the result,
   and retract without brushing the stack. Partial opening may fully release a
   cup: it is not a safe probe for an unresolved nesting relation.
6. Update the verified count and repeat for the remaining loose cups. Reuse
   sufficient returned views; add an observation only for unresolved identity,
   axis, insertion, support, or clearance.

## Recovery and completion

A planner rejection establishes no movement by itself; an endpoint failure can
follow partial movement. Continue from measured pose and observed object state.
If a descent stops short near the stack, inspect for obstruction rather than
pushing lower or opening because the desired pose was commanded. Do not bypass
collision checks or loosen tolerances to manufacture a successful return.

After a failed release, invalidate that placement, locate the loose cup, and check
which earlier nesting relations survived. Preserve the supported stack, reacquire
the loose cup, and change the demonstrated cause of failure before retrying.

Finish when the initial inventory is fully accounted for in one inverted nested
stack and it stays standing after both grippers withdraw. Visible rims, remembered
verified insertions, and a clear final view should agree; do not infer hidden extra
cups from a generic stack silhouette. Home is separate and only performed on request.
