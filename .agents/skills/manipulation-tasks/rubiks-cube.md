# Rubik's Cube

Solve or manipulate the physical cube with native CAP tools. A layer turn holds
two layers stationary while turning the remaining outer layer. Whole-cube
reorientation changes the presentation, not the puzzle state.

## Reusable solve cycle

1. **Scan and solve once.** Ground the new cube, pick it up with a whole-cube grasp,
   and inspect six faces through supported reorientations. Record each face as a
   3×3 grid with an explicit image-to-canonical rotation. Map colors using centers,
   serialize in the solver's URFDLB convention, and validate color counts, cubie
   identities, orientation, and parity. Compute and independently apply the solution
   in a symbolic cube model before physical execution. A legal reconstruction is
   not proof that an occluded sticker was observed; resolve alternatives with views.
2. **Choose the next presentation.** Retain the solution and track a separate
   canonical-face-to-world-normal map. Whole-cube motion updates this map without
   consuming a solver move. Choose either arm as turner, preserving useful support
   and enough joint-6 travel for the next turn. Use measured configurations as seeds.
3. **Establish support, then isolate.** If the holding hand needs a new grip, first
   let the other hand support the whole cube. Release and reposition the holding
   hand onto exactly the stationary two layers. Then set the turning hand on the
   free outer layer and check the closed-contact turn gate below.
4. **Turn and verify.** Apply the turn gate below, then verify the predicted sticker
   rows and square seams before consuming the move. Small seam corrections depend
   on visible residual angle, not a fixed extra-angle recipe.
5. **Release, withdraw, reuse.** Open the turner and clear its fingers along the
   approach direction before resetting the wrist. Retain or transfer support only
   as the next presentation requires. Continue from terminal evidence without
   redundant observations or a new plan for each action.
6. **Verify and place.** Account for all six uniform faces using supported
   whole-cube rotations. Place with a horizontal side grasp that leaves the bottom
   face and table-contact path clear; verify support, release, retract, and confirm
   the cube remains upright, aligned, and solved. Home only on request.

Maintain a compact ledger: canonical face grids; face-to-world mapping; current
support/turner and measured EEF poses; closed pad coverage; verified moves and
remaining suffix; unresolved stickers/seams; and the recovery return point.

### Turn sign and wrist range

Solver clockwise is viewed from outside the named face looking inward. Derive the
signed rotation from that face's current world normal and the actual EEF axis;
do not use a fixed sign for an arm. With local EEF +Z pointing inward through the
face, +90° is clockwise from outside and −90° is prime. Confirm the small initial
turn against the predicted sticker motion. Split a half-turn if needed for wrist
range or contact verification; a wrist reset requires release and clear withdrawal.

## Cube state and plan

Track sticker colors, layer alignment, cube orientation relative to the grippers,
the supporting hand, and remaining solution moves. For a full solve, reconstruct
the six faces from current evidence, validate the state, and compute a solution.
Keep that sequence through correctly tracked reorientations; reconcile unexpected
sticker changes before continuing. Historical images are not the current cube state.

## Check each action result, then continue or correct

Before executing an action, retain its expected cube alignment, cube-to-pad
contacts, and visible sticker colors from the current state and action plan.
After each action's terminal tool return:

1. **Check alignment and colors.** Compare the returned camera evidence with the
   expected result: cube orientation, pad contacts, layer coverage, square seams,
   and the affected sticker rows. Wrist angle, gripper closure, and a tool success
   flag do not prove the expected cube state.
2. **If they match, continue immediately.** Update the verified state and execute
   the next action in the existing plan. Do not rebuild the plan, repeat sufficient
   observations, or make unnecessary adjustments.
3. **If they do not match, diagnose and correct.** Identify what changed, such as
   slip, tilt, an off-center turn, contact across a seam, or a retraction collision.
   Make a targeted adjustment and check its returned evidence before resuming.
   Reconcile unexpected colors with the actual puzzle state; revise only the
   affected remainder of the plan. Do not count an unverified turn as progress.

If a decisive relation is occluded, use the smallest useful additional camera view;
do not imagine the missing state. Before every layer rotation, the latest evidence
must establish alignment with the actual cube and secure support of the two
stationary layers. Visible tilt or lost contact requires correction before turning.
If the cube is insecure in hand, restore support with the other hand when feasible;
otherwise set it on the table and regrasp. Avoid unsupported in-air adjustments.

## Grasp and alignment

1. **Flat opposing contacts.** Match wrist orientation to the measured cube edges,
   not the image frame or a nominal horizontal pose. The closing direction is
   normal to the opposing cube faces; pad contact surfaces lie against those
   faces, with finger edges aligned to the cubie rows. Both pads have useful flat
   contact, rather than diagonal, corner-only, or fingertip-only contact.
2. **Correct layer coverage.** Support contacts span the two stationary layers
   without bridging onto the free layer. Turning contacts engage only that outer
   layer without crossing its seam. Check insertion depth as well as jaw width;
   fingers closing in front of the face do not establish a grasp.
3. **No displacement during closure.** Close while watching the edges and seams.
   The cube must not tip, slide, or acquire a layer offset. A close command or a
   gripper-width reading alone is not evidence of secure contact. More closure
   does not repair an angular mismatch.
4. **Retention and axis alignment.** After a table pickup, a short lift verifies
   cube-to-pad retention and table clearance; after an in-air transfer, check
   retention as the giver releases. Release/retract conflicting contacts before
   a lift. The turning wrist is aligned with the measured face normal and
   centered on the layer axis, or its
   motion explicitly accounts for the offset. Check both pad contacts and the
   stationary-layer seam; obtain a second view if either is occluded.

Use enough grip to retain the cube without deforming it. Imbalanced pressure or
more closure does not fix poor alignment; gripper position is not calibrated force.
The current setup uses soft UMI grippers; establish a firm, visibly retained grasp.
Ground contact depth and closing position in these pads; use the broad pad area
and check coverage after compression without bridging the seam of the free layer.

## Preferred turn setup

### Required turn gate

Before applying torque, establish all four predicates from current complementary
views: broad opposing support contacts span both stationary layers; those contacts
leave the target layer and its swept path free; turning contacts engage only the
outer layer; and the rotation axis passes through the measured face center along
its normal. Check pad coverage after compression, including insertion depth on
both sides. A reachable EEF pose, closure reading, or nominal opposing-wrist
arrangement cannot satisfy this gate. Resolve any hidden contact with another view.

For a new or adjusted contact setup, execute a small initial rotation and inspect
the returned evidence before completing the quarter-turn. Continue only if the
target layer moves independently while both stationary layers and support contacts
remain fixed. On slip, middle-layer motion, obstruction, or cube tilt, stop the turn
and diagnose the geometry. Do not complete the commanded angle or blindly reverse
it: a reverse wrist motion need not undo a slipped or wide turn. Secure the cube,
correct the implicated grip or axis, and reconcile the actual sticker state first.

After an unintended brush, drop, ambiguous turn, or support slip, mark the affected
puzzle state unknown. Resolve the affected faces, validate the reconstructed state,
and update the remaining solution before another solution move. Preserve unaffected
verified information; do not assume either that progress survived or was reversed.

The operator has found an **opposing-gripper arrangement** useful. Prefer it when
reachable. Either hand can support the two stationary layers while the other turns
the free layer. Choose roles from the current valid grasp, reachability, wrist range,
and clearance to minimize regrasping and reorientation; there is no fixed left/right
assignment or mandatory handoff from an already useful grasp.

1. **Present the target face to the turning hand.** Support the cube from the
   opposite side, securing the two stationary layers and leaving the target layer
   and its seam clear. Reorient only as needed for this arrangement.
2. **Approach and turn with the free hand.** Bring the turning hand toward the exposed
   face, with the grippers pointing toward each other. Grip just that outer layer;
   keep the supporting hand fixed and rotate about the layer's center and face normal.
   Opposite-side wrists avoid crossing around the cube, but still need finger and
   camera-housing clearance and wrist range through the intended turn.
3. **Clear, retain, reuse.** Open the turning hand and withdraw along its approach
   direction until the full fingers clear the cube before changing wrist orientation.
   This avoids catching an adjacent layer during withdrawal. Retain a useful support
   grasp and bring subsequent target faces into the same proven turning arrangement,
   rather than inventing a new contact setup for each move.

## Transport, reorientation, and contact

Plan with the held cube's full size and observed pose relative to the gripper,
not just the EEF pose. Its swept volume must clear the tabletop and the other
gripper's fingers, palm, and camera housing, except for intended pad contact.

Before whole-cube reorientation, clear the other gripper from the intended motion.
If table clearance is marginal or uncertain, **lift while preserving the current
orientation, assess clearance from the returned view, then rotate**. Do not combine
that initial clearance lift with a large orientation change. Allow a clearance margin
for the cube's lowest swept corner and the fingers along the entire trajectory,
including intermediate dips toward the table.

Table contact can lever the cube out of a secure grasp or rotate it inside the pads.
When contact causes a drop or slip, correct the path and clearance rather than
assuming insufficient grip or tightening further. Re-establish cube-to-gripper
alignment before the next face turn. Unexpected cube deflection, stalled descent,
or tracking error near the tabletop warrants checking for contact before advancing;
do not push through it. During intentional placement, stop lowering once the cube
is supported by the table.

Align the turning gripper with the cube, not merely with the supporting gripper.
If the EEF is offset from the layer axis, rotate both its position and orientation
about that axis. Verify actual layer motion and square seams, not wrist angle alone;
restore lost contact or alignment rather than forcing a jammed cube.

For placement, reject implausibly broad or inconsistent tabletop localization even
when segmentation reports success. Use a clear side view for bounded descent,
reducing step size as the gap closes. Do not inherit table height from a historical
EEF pose. Watch the cube, finger undersides, palm, and camera housing; release only
at supported contact.

### Partially shifted slice recovery

For a small, understood slice offset, retain support and use an aligned, broad
parallel grasp across the affected layers to gently square them. Do not force a
jammed or unknown state. Stop if closure causes tilt, translation, or growing
misalignment. Squaring is not a solver move; verify colors afterward.

## Adjust cube-to-gripper pose with the other hand

Use `cap_move_synchronized(targets={left: pose, right: pose})` for jointly planned
clear staging motions, such as presenting the held cube while bringing the free
hand to a separated pregrasp. Keep support-transfer steps ordered; synchronized
timing does not maintain a fixed hand-to-hand transform while both grip the cube.

Moving a securely closed holding hand moves the cube with it; this does not change
their relative pose. To recenter, deepen, or change the grasp angle:

1. The free hand approaches exposed grasp faces with finger clearance and establishes
   secure temporary support without turning a layer.
2. The original hand releases and clears its contacts, then repositions its open
   fingers around the supported cube to the desired depth, center, and orientation.
3. Regrasp and verify the new cube-to-pad alignment and retention before releasing
   the helper. Clear released fingers before transporting or reorienting the cube.

Either hand can remain the support if that avoids an unnecessary transfer back.
Do not force a grasp adjustment by twisting the cube between two closed grips.
Use table-supported regrasping when a clear, secure in-air transfer is unavailable.

## Learn from failed motion

Retain failed approaches with their arm, target orientation, other-arm arrangement,
and reported cause. An IK/planner rejection is conditional evidence, not proof that
the XYZ position is unreachable. Change the implicated orientation, clearance, or
cube presentation instead of repeating nearby targets with the same failure cause.
Endpoint failure can follow real motion; continue from the achieved pose and cube
state. Do not relax contact constraints to obtain success.

## Throughput and completion

Optimize verified face turns per attempt, not merely call frequency. Keep a compact
record of proven setups: station and pad configuration, supporting/turning roles,
cube-to-EEF offsets, grip depths and layer coverage, face-axis alignment, wrist
range, clear withdrawal paths, and the images establishing success. Reuse these
as seeds with fresh alignment checks, not universal absolute targets. A retry must
correct an observed cause; repeated nearby pose guesses without resolving grip
depth, support, or axis alignment are not useful recovery.

Optimize throughput and reduce pauses between action calls. Use `cap_move_to_pose`
or `cap_move_synchronized`, continuing the grounded sequence from each terminal result.
Reuse fresh returned images and verified contact geometry instead of repeating
settled checks or rebuilding the setup after every
move. Correct from achieved poses and observed cube-to-gripper relations, not from
commanded targets treated as measurements.

Count verified layer turns as solving progress; staging and regrasping are not
solution moves. For a full solve, verify all six faces are uniform and seams
aligned, then place the cube stably, release, and confirm it remains solved.
