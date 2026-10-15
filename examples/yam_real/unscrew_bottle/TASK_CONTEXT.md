---
task: "Unscrew the bottle cap and pour the pill onto the table."
---

# Unscrew a bottle and pour

## Goal and scope

Remove a conventional right-hand threaded cap, place it separately, pour the pill
onto a clear tabletop patch, and leave the empty bottle upright with both grippers
clear. This recipe assumes a small rigid bottle, a compatible screw cap, and a
pill that fits through the opening. Establish a different closure's mechanism
before applying torque; do not assume a push-and-turn closure behaves the same way.

Use native CAP tools in an already configured session. `observe` supplies views,
`cap_get_robot_state` supplies measured poses, and `cap_localize_object` supplies
geometry when needed. Use `cap_move_to_pose` for full Cartesian poses and
`cap_set_gripper` for separate grip changes. Position is in metres in the declared
frame; `rpy_deg` is fixed-axis XYZ in degrees, or use `quaternion_wxyz`, not both.
Current MCP schemas are authoritative. Preserve collision-aware planning. No WAM,
visual supervisor, recorded trajectory, or preselected absolute pose is required.
Joint targets are in radians; gripper position runs from 0 closed to 1 open and is
not a calibrated force. Derive the required opening from the actual pads and object.

## Establish two useful grasps

Identify the bottle body, shoulder, mouth, cap axis, table, and clear deposition
patches. Reject a transparent-bottle localization whose mask, depth, or dimensions
conflict with the images. A visible cap or rim and a complementary side view can
resolve alignment when the clear body has poor depth. Do not substitute image
pixels for calibrated world coordinates.

Use one hand as the body holder and the other as the cap turner. Choose roles from
reach, wrist travel, and clearance; left-body/right-cap is a useful arrangement,
not a requirement.

1. Side-approach the bottle below its shoulder. Seat the body between broad opposing
   pads without obstructing the cap hand. Close firmly enough to prevent co-rotation,
   without crushing or tilting the bottle. Keep its base supported on the table;
   lifting it before unscrewing is unnecessary.
2. Approach the cap from above along the bottle axis. Grip opposing cap sidewalls,
   not just its top edge, and do not clamp the neck or shoulder. Check the full
   fingers and camera housings against the supporting hand through the intended turn.
3. Verify contacts from returned views. Gripper position is not calibrated force;
   a close command is not proof of a secure grasp. Establish a comfortable wrist
   range for the first turn instead of starting at a limit.

## Unscrew without losing progress

For a right-hand thread, counterclockwise **viewed from above the cap looking
down the bottle axis** loosens it. Derive the tool rotation from that physical
axis; neither an arm-specific joint sign nor a rotated wrist image defines it.

- Begin with a bounded engaged turn. Turns of roughly 30–60 degrees are useful
  starting segments for a small bottle when clearance and wrist travel permit;
  shorten them if contact is uncertain. Verify cap motion relative to the bottle.
- Keep the bottle stationary and table-supported. If it co-rotates, improve body
  contact before continuing. If the fingers slip over the cap, correct sidewall
  depth or alignment rather than adding more wrist rotation.
- Maintain lateral alignment while accommodating observed thread rise. Do not
  pull the bottle upward or impose an assumed pitch as a downward/upward stroke.
- Before a wrist limit, open the cap hand, withdraw until the entire finger sweep
  clears the cap and neck, reset the unloaded wrist, then reapproach the cap at its
  current height and regrasp. Never rewind while still gripping.
- If the operator specifies a minimum rotation, count only observed cap rotation
  relative to the bottle: two full turns means 720 degrees. Exclude unloaded resets,
  visible slip, and bottle co-rotation. Turn count alone does not prove detachment;
  do not keep turning a cap already confirmed free merely to satisfy a counter.

Use full absolute pose targets. If the EEF is offset from the cap axis, its position
must follow that axis as well as its orientation. Several bounded endpoints may be
needed: a pose planner does not promise a contact-preserving circular path between
arbitrary endpoints. A joint-space wrist turn is appropriate only when its axis,
joint travel, and swept clearance are established; it is not a collision bypass.

## Remove, pour, and finish

1. With evidence of unthreading, test a small axial cap lift while holding the body
   steady. Require a separated cap and exposed mouth. If the bottle follows, stop
   pulling and reassess the thread and both grasps.
2. Move the held cap clear, lower near a grounded support patch, release, and confirm
   it rests separately. Account for fingers extending below the cap before lowering.
3. Preserve the body grasp. Lift the open bottle enough to clear the table through
   its full tipping sweep, then tilt toward the deposition patch using full poses.
   Check the lowest swept bottle corner, fingers, and wrist camera, not just EEF Z.
4. A pill at the shoulder or neck is not yet poured. If wrist travel limits the
   tilt, change the reachable arm presentation while retaining support and clearance.
   Do not free XYZ to obtain an orientation or prescribe shaking as a proven remedy.
5. Once the pill is visible on the tabletop and the bottle is empty, restore the
   bottle upright, lower to supported table contact, release, and retract.

Inspect achieved state after a motion error: a failed return can follow partial
motion, and a successful return may leave the pill inside. Reuse adequate terminal
images and dispatch the next grounded action promptly. Get another view only when
it resolves a decision-critical uncertainty. If a retry is needed, change the
observed cause rather than repeating an unchanged target.

Completion requires cap separation, pill deposition, and stable released bottle
placement. Do not claim a measured number of relative turns from wrist commands
alone. If an explicitly requested minimum turn count was not established before
detachment, report that condition unverified or unmet separately from cap removal;
unloaded rotations cannot satisfy it. Home is a separate operator request, not part
of this manipulation recipe.
