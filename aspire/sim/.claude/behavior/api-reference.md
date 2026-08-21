# BEHAVIOR-1K API Reference

Source: `cap/integrations/r1pro/control.py`

Environment configs instantiate `R1ProControlApi` through the `R1ProControlApi`
API registration. Generated task code receives the functions returned by
`R1ProControlApi.functions()`.

Traced debug configs instantiate `R1ProControlApiTraced`, which wraps the same
function surface and saves `trace.json` plus keyframes through
`aspire.sim.cap.integrations.trace_logger`.

## Core Perception

| Function | Purpose |
|---|---|
| `get_env_observation()` | Return RGB/depth observation from the active camera |
| `segment_sam3_text_prompt(rgb, text)` | Segment objects using a SAM3 text prompt |
| `segment_sam3_point_prompt(rgb, point_coords)` | Segment from a point prompt |
| `point_prompt_molmo(image, text)` | Use Molmo to propose relevant image points |
| `get_sam3_mask(object_name)` | Convenience SAM3 object mask lookup |
| `get_object_pose(object_name, ...)` | Estimate object pose from perception outputs |
| `save_current_observation(name)` | Save the current RGB/depth observation for debugging |

## Navigation And Manipulation

| Function | Purpose |
|---|---|
| `get_robot_position()` | Return current base position |
| `get_navigation_pose(P_table, P_object)` | Compute navigation target near an object |
| `navigate_to_pose(position, orientation)` | Move the mobile base to a target pose |
| `find_object_base_rotate(object_name)` | Search for an object by rotating the base |
| `find_object_torso_rotate(object_name)` | Search for an object by rotating the torso |
| `reset_torso()` | Reset torso orientation |
| `sample_grasp_pose(object_name)` | Produce grasp candidates for an object |
| `grasp_object(pregrasp_pose, grasp_pose, object_name, arm=0)` | Execute a grasp sequence |
| `check_object_in_hand(arm=0)` | Check whether the gripper is holding an object |

## Arm Control

| Function | Purpose |
|---|---|
| `solve_ik(position, quaternion_wxyz, arm=0)` | Solve IK for an end-effector target |
| `move_hand(target_pose, arm=0)` | Move the selected hand to a pose |
| `move_to_joint_positions(target_joint_positions, ...)` | Execute arm joint targets |
| `get_current_joint_positions()` | Return current robot joint positions |
| `get_current_eef_pose(arm=0)` | Return current end-effector pose |
| `get_robot_relative_eef_pose(arm=0)` | Return end-effector pose relative to robot frame |
| `open_gripper(arm=0)` | Open a gripper |
| `close_gripper(arm=0)` | Close a gripper |
| `lift_arm(arm=0)` | Lift after grasping |

## Artifacts

| Function | Purpose |
|---|---|
| `write_video(name)` | Save current recorded frames to a video |

## Trace Artifacts

Use `env_configs/r1pro/*_aspire_traced.yaml` for offline trace
collection. Each trial can include:

- `trace.json` with API calls, arguments, return summaries, errors, and timing;
- `keyframes/step_*_obs_env.jpg` and `step_*_depth_env.npy` from
  `get_env_observation`;
- `keyframes/step_*_sam3.jpg` and `step_*_mask_0.npy` from SAM3 calls;
- videos and VDM feedback files from the normal launcher artifacts.

## Debugging Rule

Do not access simulator internals, BDDL predicates, object registry ground truth,
or OmniGibson/Isaac state directly from generated task code. Use camera
observations, perception APIs, robot state APIs, saved videos, and stdout/stderr.
