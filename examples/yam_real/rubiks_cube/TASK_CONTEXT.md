---
task: "Solve the physical Rubik's Cube and put it down on the table."
---

# Rubik's Cube manipulation

Use the current scene and native low-level CAP tools, including
`cap_move_to_pose`, `cap_move_synchronized`, and gripper tools. Follow the
[Rubik's Cube guide](../../../.agents/skills/manipulation-tasks/rubiks-cube.md)
for grasp, layer isolation, object-state tracking, and throughput.

Current hardware: soft UMI grippers. The flexible tips provide limited support;
use broad pad contact without bridging the layer to be turned. Re-ground grasp
depth and closing position after a gripper change; rigid-gripper settings are not
a reference calibration.

The requested outcome is a solved cube: all six faces uniform, all layers aligned,
and the cube released stably on the table. A solver solution or successful robot
motion alone does not establish that outcome.

Do not stop working if the cube is not verified to be solved. The task is fully-solvable
and existing tools are good enough to solve it.

## Recipe for a new session

Read the guide's **Reusable solve cycle**.

For each reset scene:

1. Reconstruct and validate the current six-face state; compute its own solution.
   Keep canonical sticker state separate from whole-cube orientation in world.
2. Reuse reachable presentations with fresh cube-to-pad alignment. Have the helper
   support the whole cube while adjusting the other hand's two-layer support grip.
3. Verify **closed** support pads cover both stationary layers, leaving the target
   outer layer free. The turning pads must touch only that layer. Open-pad geometry
   alone is insufficient.
4. Test a new grip with a small turn, then finish only if the intended outer layer
   moves independently. Check sticker motion and seams before consuming the move.
5. Open and withdraw the turner clear before changing its angle. Continue promptly
   from achieved poses and returned images; do not redo settled observations.
6. Verify every face, place with the bottom face unobstructed, release at supported
   table contact, retract, and confirm stability. Home only when requested.

Historical XYZ, gripper positions, and placement EEF height are seeds, not a calibration for the
new cube. Derive clockwise/prime from the currently presented face normal and EEF
axis. Keep collision planning and full pose constraints; adapt rejected poses from
their reported cause rather than weakening checks.
