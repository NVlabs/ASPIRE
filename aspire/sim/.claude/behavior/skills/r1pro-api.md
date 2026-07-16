# R1Pro API Notes

Source: `cap/integrations/r1pro/control.py`

## Robot Model

- Mobile base: `x`, `y`, `yaw`.
- Torso: camera height and tilt control.
- Arms: two 7-DOF arms. `arm=0` is left, `arm=1` is right.
- Camera: head RGB-D camera used for perception and debugging.

## Quaternion Conventions

- `get_object_pose`, `sample_grasp_pose`, and `solve_ik` use `quaternion_wxyz`.
- `move_hand` and `get_current_eef_pose` use `quaternion_xyzw`.
- Treat quaternion conversion mistakes as a primary suspect when grasps miss.

## Sandbox Functions

| Function | Use |
|---|---|
| `get_env_observation()` | Read current head RGB-D observation |
| `save_current_observation(name)` | Save head/external camera observations at decision points |
| `segment_sam3_text_prompt(rgb, text_prompt)` | Run text-prompted SAM3 on an RGB image |
| `segment_sam3_point_prompt(rgb, points, labels)` | Run point-prompted SAM3 |
| `point_prompt_molmo(rgb, text_prompt)` | Generate point prompt candidates |
| `get_sam3_mask(name)` | Quick visibility/mask check |
| `find_object_base_rotate(name)` | Rotate base and search with SAM3 |
| `find_object_torso_rotate(name)` | Tilt torso/head and search with SAM3 |
| `get_object_pose(name)` | Estimate object pose, bbox, point cloud, OBB |
| `sample_grasp_pose(name)` | Generate pregrasp/grasp pose pairs |
| `get_navigation_pose(P_table, P_object)` | Compute table-object approach pose |
| `navigate_to_pose([x, y, yaw])` | Move mobile base |
| `get_robot_position()` | Return base position, orientation, and yaw |
| `reset_torso()` | Reset torso/head posture |
| `move_hand((pos, quat_xyzw), arm)` | Move end-effector to a pose |
| `move_to_joint_positions(joint_positions)` | Move arm joints to a target configuration |
| `get_current_eef_pose(arm)` | Return end-effector pose for one arm |
| `get_current_joint_positions()` | Return current arm joint positions |
| `solve_ik(target_pos, target_quat_wxyz, arm)` | Solve IK for an end-effector target |
| `grasp_object(pregrasp, grasp, name, arm)` | Full grasp sequence with planning |
| `open_gripper(arm)` / `close_gripper(arm)` | Control gripper state |
| `lift_arm(arm)` | Lift after grasp |
| `check_object_in_hand(arm)` | Verify gripper/object contact |
| `write_video(name)` | Save video from current buffers |

Only functions returned by `cap/integrations/r1pro/control.py::functions()`
are automatically available in generated task code. Do not call historical
helpers such as `aspire_launch.py`-specific APIs, `detect_object_sam3`,
`execute_motion_plan`, `get_camera_pose`, or `get_camera_intrinsics` unless
they are explicitly added to the registered function map.

## Interactive Policy Requirements

- Import `numpy`, `time`, and any other library explicitly inside generated
  code.
- Call `get_env_observation()` and `save_current_observation(name)` after every
  meaningful navigation, search, approach, and grasp step.
- Print diagnostic values: robot pose, object prompt, object pose, requested
  goal, movement distance, arm choice, grasp index, and `check_object_in_hand`
  result.
- Guard every perception and planning call with retries or `try/except`.
- Use only public R1Pro API calls; do not inspect OmniGibson internals, BDDL
  predicates, object registries, or reward state.

## Trace Interpretation

Some high-level API calls invoke lower-level perception or planning internally.
For example, `find_object_base_rotate` repeatedly calls SAM3, and
`sample_grasp_pose` calls SAM3 plus ContactGraspNet. Traces may show only the
top-level call, so infer internal failures from stdout/stderr, videos, saved
observations, and timing.
