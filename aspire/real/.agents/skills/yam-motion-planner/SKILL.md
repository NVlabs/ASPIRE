---
name: yam-motion-planner
description: "Use for YAM cuRobo planning/IK failures, no-motion vs real-robot planning mismatch, waypoint robustness sweeps, perturbation testing, and recovery after motion planner failures."
---

# YAM Motion Planner

cuRobo is sensitive to the real start state. A no-motion/pre-motion plan can
pass, then a real run can fail because physical execution replans from the
actual EEF pose, joint state, gripper load, or a slightly different scene pose.

When cuRobo fails:

1. Inspect the current scene, robot state, failed target pose, and failed stage.
2. Decide whether the target is physically unreasonable or just planner-sensitive.
3. If the motion is feasible, do not treat one IK/path failure as final.
4. Try nearby target offsets and extra waypoints that make the same task easier:
   retreat lanes, lift-before-shift, shift-before-approach, lower/higher
   staging poses, or less extreme rack/bin/place coordinates.
5. Preview from the current robot state, not only from an ideal scripted state.
6. Prefer waypoint sets that pass perturbation sweeps, not just one clean pass.

Robustness sweep pattern:

- Perturb candidate waypoints and final targets by small XYZ/RPY offsets that
  represent likely real-world variance.
- Include start-state sensitivity when possible: preview from the current arm
  posture, or rerun previews after moving to the actual preceding waypoint.
- Record pass/fail rate, first failed stage, pose, and planner error in
  `logs/<run>/plans/`.
- Pick the candidate with high pass rate, clear margins from obstacles, and
  simple task semantics.

Useful sweep examples:

- `legacy_codes/kitkat_trash_waypoint_robustness_sweep.py`
- `legacy_codes/kitkat_handover_clearance_waypoint_sweep.py`
- `legacy_codes/bottle_rack_waypoint_robustness_sweep.py`
- `legacy_codes/bottle_fixed_pour_waypoint_robustness_sweep.py`

Typical useful perturbations:

- placement/drop targets: `x,y,z +/- 2-5 cm`
- staging/retreat waypoints: `x,y,z +/- 3-8 cm`
- orientation: yaw/roll/pitch variants near the intended gripper axis
- approach order: split compound moves into one-axis waypoints

If a real run stops after a cuRobo failure while holding an object, first reason
from the current scene. If the object is still safely held, plan a held-object
recovery from the current EEF pose: preview nearby rack/bin/table targets, add
intermediate waypoints, release, retreat, then home/open. Ask the human to reset
only if the scene is physically broken or unsafe for the robot to fix.
