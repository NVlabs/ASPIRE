---
name: yam-grasp-pickup
description: "Use for YAM object detection, grasp candidate generation, side/top-down/rim pickup choices, staged gripper close, contact evidence, lift-only checks, and pickup failures."
---

# YAM Grasp Pickup

Start from fresh live perception. Do not use fixed object XYZ for a physical
pickup unless it is explicitly a no-motion calibration run.

Preferred helper layer:

- `yam_runtime.capture_scene`
- `yam_runtime.generate_side_grasp_candidates`
- `yam_runtime.rank_motion_candidates`
- `yam_runtime.execute_grasp_lift_attempt`
- `yam_runtime.staged_close_with_contact`
- `yam_runtime.verify_lift`

Patterns from current successful scripts:

- Bottle: side body grasp, fixed/biased body Z only after live XY detection,
  staged close, lift before transport.
- Can: cylinder model, top-down yaw candidates, lift to transfer clearance,
  verify by post observation.
- Bowl/dish/plate: rim or wall contact, high approach then low pregrasp,
  gentle staged close, lift only a few centimeters before transport.
- KitKat: top-down endpoint pinch on one short end, then handover the exposed
  end.

Do not count close command return as pickup success. Require gripper
width/contact evidence, lift/post-observation motion, video evidence, or an
explicit task artifact.
