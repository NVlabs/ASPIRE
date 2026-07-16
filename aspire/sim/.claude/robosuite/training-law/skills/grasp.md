---
name: robosuite/training-law/skills/grasp
description: Structural template for pick-and-place programs — topdown-quat construction, pre-grasp approach, lower-close sequence, lift-transport-place pattern. No task-specific parameters; those grow through experiment.
---

## Grasp Verification

- `close_gripper()` returns `None` — you CANNOT check gripper_width from user code. The trace framework logs it but it's not accessible at runtime.
- **Observation-based verification** is the reliable pattern: after lifting, re-observe with SAM3 and check if the object is still at table level. If the object centroid z is below a threshold (e.g. z < -0.03 for Robosuite), the grasp failed.
- On failed grasp: move to safe position → open gripper → re-observe → re-grasp with slightly lower z offset (-0.005).

## Grip Tightening

- "Slight lift + re-close" after initial close: lift ~2cm, then close_gripper() again. Helps secure marginal grasps where fingers partially wrapped around the object.

## GraspNet vs Geometric Grasps

- **For small objects (~35mm cubes), geometric grasps at the object center are MORE reliable than GraspNet.** GraspNet often returns edge/off-center grasps on small objects, producing marginal gripper widths that drop during transport.
- Geometric grasp: target OBB center in XY, 40-50% from the object top in Z (mid-height). This maximizes gripper finger contact surface.
- Do NOT grasp at the object top surface — fingertips get insufficient contact and the grasp slips.
- `solve_ik` positions represent where the fingertips will be (TCP offset is internally applied).

## SAM3 Prompts — Cubes (Robosuite)

| Object | Prompts (priority order) |
|---|---|
| red cube | "red cube", "red block", "cube", "red object" |
| green cube | "green cube", "green block" |
