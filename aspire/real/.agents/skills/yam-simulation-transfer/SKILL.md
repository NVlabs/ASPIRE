---
name: yam-simulation-transfer
description: "Use when borrowing LIBERO/robosuite strategy ideas for YAM grasping, handover, pouring, placement, rack, or bin tasks."
---

# YAM Simulation Transfer

Borrow strategy, not simulator coordinates/API/constants.

The physical-robot implementation lives in `aspire/real`; the source
simulation workspace is its sibling `aspire/sim` (`../sim` when working from
the real workspace). Inspect simulation code and agent guidance there, then
convert the strategy into YAM-native behavior here.

References:

- `references/plate-libero-transfer.md`
- `references/bowl-libero-transfer.md`
- `references/can-libero-transfer.md`
- `references/libero-cup-mug-skills.md`
- `references/robosuite-two-arm-handover.md`
- `references/robosuite_two_arm_handover.py`
- `references/drawer-libero-transfer.md`

Convert ideas into YAM-native detections, frames, gripper geometry, planner
previews, and artifact checks before physical retries.
