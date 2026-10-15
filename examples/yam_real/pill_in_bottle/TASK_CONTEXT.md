---
task: "Put the pill into the bottle, fit its cap, and tighten the cap."
---

# Pill insertion and bottle closure

## Goal and evidence boundary

Place the pill inside an upright bottle, engage its compatible screw cap, and
leave it closed and standing on the table after both hands withdraw. Keep three
distinct predicates: **pill contained**, **cap seated/threaded**, and **cap snug**.
A level cap that remains on the rim after release is not sufficient proof of the
last predicate. This recipe supports insertion and cap seating; it does not supply
a calibrated torque or force test for final tightness.

Assume a reachable stable tabletop, a pinchable pill that fits through the mouth
in an attainable release orientation, and a conventional right-hand thread unless
the actual closure establishes otherwise. Use native CAP observation,
localization, measured robot state, full Cartesian motion, and separate gripper
commands. Typical tools are `observe`, `cap_localize_object`, `cap_get_robot_state`,
`cap_move_to_pose`, and `cap_set_gripper`; use the current MCP schemas. Poses specify
metres in a declared frame and either fixed-axis XYZ `rpy_deg` or
`quaternion_wxyz`. Preserve collision-aware planning. No policy, visual supervisor,
historical coordinates, or external task guide is needed.
Joint targets use radians. Gripper position is normalized from 0 closed to 1 open,
not measured force; determine contact from the current pads and object.

## Ground the small object and support the bottle

Distinguish the actual pill from the cap using visible appearance and location.
Inspect the localization mask: a plausible point on the wrong object is still
wrong. Transparent bottle depth can be misleading; reconcile the mouth and table
with a useful side or wrist view before approaching. Reject inconsistent bounds
instead of averaging them into an apparently precise target.

Assign one hand to the bottle and the other to the pill/cap, based on current reach
and clearance. Side-grasp the body below its shoulder with broad opposing contact,
leaving the mouth accessible. Keep the base on the table throughout insertion and
tightening. Verify that closure does not tilt or displace the bottle. Do not use an
unsupported bottle lift just to satisfy a grasp-check ritual.

The EEF is not the fingertip or object center. Ground fingertip clearance, the pill
between the fingers, the mouth center, and the cap's offset within the grasp before
precise placement. Refresh the affected relation after slip or displacement.

## Insert the pill

1. With the bottle supported, approach the pill from above, orienting the jaws to
   pinch across its narrow dimension. Descend to side contact without driving the
   fingertips into the table, then close only enough to retain the object.
2. Lift clear and verify the pill follows the fingers. Closure or an empty-looking
   patch of table alone does not establish retention.
3. Carry the pill above the bottle mouth. Align the **held pill**, not merely the
   wrist, over the aperture using a view that resolves the rim and fingers.
4. Release over the opening with fingertip clearance. Verify the pill below the
   neck or at the bottle bottom. Disappearance behind the rim is ambiguous; retain
   bottle support and resolve that relation before closing the bottle.

## Fit and thread the cap

1. Identify the cap's open side. Acquire opposing exterior sidewalls with meaningful
   pad depth, not a shallow pinch on its upper edge. Verify it lifts and stays level.
   If its opening faces the wrong way, reorient with clearance and support before
   bringing it to the bottle.
2. Bring the cap above the mouth with matching axes. Lower toward an even meeting
   of cap and rim. Use a side view for tilt; top-view overlap cannot show whether
   one edge is perched above the other. Keep fingers clear of the bottle shoulder
   and body-holding gripper.
3. For a right-hand thread, tighten clockwise **viewed from above the cap looking
   down into the bottle**. Use bounded turns about the measured bottle axis while
   preserving alignment and table support. Derive the sign from the actual axis,
   not a fixed left/right wrist sign.
4. Watch cap features turn relative to the stationary body and the cap seat down
   the neck while staying level. Accommodate observed axial descent with small
   pose changes; do not force a fixed downward stroke or equate wrist travel with
   thread engagement.
5. To regain wrist travel, leave the cap supported on the neck, open the turning
   hand, withdraw until fingers clear, reset the unloaded wrist, and regrasp the
   sidewalls. Reversing while closed can undo tightening.

Use full position-and-orientation targets, including the orbit of an offset EEF
around the cap axis. Use bounded intermediate targets rather than assuming an
endpoint tool executes an exact circular contact path. A known aligned wrist joint
may provide a bounded rotation when its range and clearance are established; do
not bypass a collision rejection with joint commands.

## Correct seating failures

- **Tilted or binding cap:** stop tightening and retain body support. If threads
  may be engaged, use controlled counterclockwise backoff to establish disengagement
  before lifting; do not pull axially against a binding thread. Recenter and reseat
  once clear. A click or successful thread reset cannot be assumed. Check the
  subsequent level seating and relative motion.
- **Cap slips in the fingers:** correct depth, angular alignment, or side contact.
  More commanded rotation does not repair a missed grasp. If dropped, reacquire
  from its new resting pose before continuing.
- **Approach stops short:** inspect measured EEF pose and the scene; errors can
  follow partial motion. A commanded height does not establish contact. Change the
  implicated approach or orientation instead of repeating an unreachable target.
- **Bottle moves:** restore stable table support and recompute cap-to-mouth
  alignment. Do not tighten using coordinates that predate that displacement.

Keep successful geometry and use returned evidence for the next ready action;
avoid repetitive observations and long planning reports at routine transitions.

## Finish honestly

Verify containment and evidence of threaded, level seating. Release and withdraw
the cap hand without disturbing the closure, then release the bottle holder and
confirm independent stability. Do not describe a merely retained cap as securely
tightened or invent a torque measurement from gripper position. If snug closure
cannot be distinguished from slipping contact or a loose cap, retain that explicit
uncertainty and continue only with a grounded, non-damaging correction or a suitable
available verification method. If the runtime has no validated way to establish
snugness, report partial completion, not a tightened-bottle success; repeated turns
do not create the missing measurement. Report what is established and what remains
unresolved. Home only when separately requested.
