---
name: robosuite/api-reference
description: Full Robosuite control API — functions, signatures, output structure, TraceLogger format, and source files. Shared by all robosuite experiments.
---

# API Reference

Source: `cap/integrations/franka/control_reduced_skill_library.py` (extends `control_reduced.py`)
Traced wrapper: `cap/integrations/robosuite_trace_logger.py` (`TracedFrankaControlApiReducedSkillLibrary`)

---

## Observation & Perception

| Function | Returns |
|---|---|
| `get_observation()` | Camera data + robot state dict (see keys below) |
| `segment_sam3_text_prompt(rgb, text_prompt)` | `[{"mask": np.bool_, "box": [x1,y1,x2,y2], "score": float, "label": str}, ...]` |
| `segment_sam3_point_prompt(rgb, point_coords)` | `[{"mask": np.bool_, "score": float}, ...]` |
| `point_prompt_molmo(image, text_prompt)` | `{text_prompt: (pixel_x, pixel_y)}` or `{text_prompt: (None, None)}` if not found |

**`get_observation()` keys:**

Single-arm tasks (`cube_lifting`, `cube_restack`, `cube_stack`, `nut_assembly`, `spill_wipe`):
```python
obs = get_observation()
rgb   = obs["robot0_robotview"]["images"]["rgb"]    # (H, W, 3) uint8
depth = obs["robot0_robotview"]["images"]["depth"]  # (H, W) float32, meters
K     = obs["robot0_robotview"]["intrinsics"]       # (3, 3) camera intrinsic matrix
T     = obs["robot0_robotview"]["pose_mat"]         # (4, 4) camera-to-world (robot0 base frame)
```

Bimanual tasks (`two_arm_lift`, `two_arm_handover`) — use the same `robot0_robotview` key as single-arm tasks:
```python
obs = get_observation()
rgb   = obs["robot0_robotview"]["images"]["rgb"]    # (H, W, 3) uint8
depth = obs["robot0_robotview"]["images"]["depth"]  # (H, W) float32, meters
K     = obs["robot0_robotview"]["intrinsics"]       # (3, 3)
T     = obs["robot0_robotview"]["pose_mat"]         # (4, 4) camera-to-world (robot0 base frame)
# Also available for bimanual:
arm0_pose = obs["robot0_cartesian_pos"]  # (8,) [x,y,z,w,qx,qy,qz, gripper_fraction]
arm1_pose = obs["robot1_cartesian_pos"]  # (8,) same, in robot0 base frame
```

---

## Grasp Planning & IK

| Function | Returns |
|---|---|
| `plan_grasp(depth, intrinsics, segmentation)` | `(grasp_poses: (K,4,4), grasp_scores: (K,))` — poses in **camera frame**, TCP-offset adjusted |
| `select_top_down_grasp(poses, scores, cam_to_world, vertical_threshold=0.8)` | `(best_grasp_4x4, best_score)` — pose in **world frame**; `(None, -inf)` if no top-down grasp found |
| `solve_ik(position, quaternion_wxyz)` | Joint angles `(7,)` |
| `solve_ik_arm0(position, quaternion_wxyz)` | Joint angles `(7,)` — bimanual only |
| `solve_ik_arm1(position, quaternion_wxyz)` | Joint angles `(7,)` — bimanual only; input in robot0 base frame, auto-transforms to arm1 frame |

**Typical grasp workflow (manual selection):**
```python
T = obs["robot0_robotview"]["pose_mat"]            # camera-to-world (all 7 tasks)
grasp_poses, grasp_scores = plan_grasp(depth, K, mask)
# Option A: select top-down grasp (returns world-frame pose directly)
best_grasp_world, score = select_top_down_grasp(grasp_poses, grasp_scores, T)
# Option B: manually pick best and convert
best_T_cam = grasp_poses[grasp_scores.argmax()]
best_T_world = T @ best_T_cam
```

---

## Motion Execution

| Function | Returns |
|---|---|
| `move_to_joints(joints)` | None (blocking) |
| `open_gripper()` / `close_gripper()` | None |
| `move_to_joints_arm0(joints)` | None (blocking) — bimanual only |
| `move_to_joints_arm1(joints)` | None (blocking) — bimanual only |
| `move_to_joints_both(joints0, joints1)` | None (blocking, simultaneous) — bimanual only |
| `open_gripper_arm0()` / `close_gripper_arm0()` | None — bimanual only |
| `open_gripper_arm1()` / `close_gripper_arm1()` | None — bimanual only |

---

## Geometry Utilities

| Function | Description |
|---|---|
| `decompose_transform(T)` | 4×4 → `(position (3,), quaternion_wxyz (4,))` |
| `rotation_matrix_to_quaternion(R)` | 3×3 rotation matrix → quaternion `(4,)` [w,x,y,z] |
| `mask_to_world_points(mask, depth, intrinsics, extrinsics)` | 2D bool mask → `(N,3)` world-frame point cloud |
| `depth_to_point_cloud(depth, intrinsics)` | Depth image → `(H,W,3)` camera-frame point cloud |
| `pixel_to_world_point(u, v, z, intrinsics, extrinsics)` | Single pixel (col, row) + depth → `(3,)` world point |
| `get_oriented_bounding_box_from_3d_points(pts)` | `(N,3)` → OBB `{center, extent, R}` |
| `transform_points(points, T)` | Apply 4×4 transform to `(N,3)` or `(H,W,3)` points |
| `interpolate_segment(p1, p2, step=0.03)` | Generate waypoints along a line segment — returns `list[np.ndarray]` |
| `normalize_vector(v)` | Normalize 3D vector to unit length |

---

## Success Conditions

Reward is binary and encoded in the trial output folder name: `_reward_1.000` = success, `_reward_0.000` = failure. Check `summary.txt` for `Task Completed: True/False`.

---

## Output Directory Structure

```
outputs/baseline_robosuite_multimodel_ensemble_traced/
└── ensemble_multimodel/
    └── ensemble_multimodel/
        └── <config_stem>/
            ├── fix_code.py             # written after Stage 1 debug
            ├── findings.md             # root causes + SAM3 prompts that worked
            ├── reasoning.txt           # why the fix loop stopped + anomalies
            └── trial_<N>/
                ├── code.py             # generated/fix code
                ├── trace.json          # PRIMARY: every API call logged
                ├── summary.txt         # stdout/stderr/reward
                ├── keyframes/          # RGB + depth + mask arrays at perception steps
                └── video_combined.mp4

outputs/robosuite_fix_eval/
└── <config_stem>/                      # Stage 2 validation output (seeds 1–100)
    └── trial_<N>_reward_<R>/
```

---

## TraceLogger Format

```json
{
  "step": 5,
  "function": "segment_sam3_text_prompt",
  "args": {"rgb_shape": [512, 800, 3], "text_prompt": "red cube"},
  "duration_ms": 1463.8,
  "result": {
    "num_masks": 3,
    "mask_0_score": 0.917,
    "mask_0_bbox": [369.2, 257.1, 438.4, 310.5],
    "mask_0_area_pct": 0.68
  },
  "keyframe_saved": true
}
```

Key signals to read from trace:
- `open_gripper` / `open_gripper_arm0/arm1` → `gripper_width`: high value (>0.9) confirms gripper opened successfully
- `close_gripper` / `close_gripper_arm0/arm1` → `gripper_width`: >0.06 grasped, 0.03–0.06 marginal, <0.03 air grasp
- `segment_sam3_text_prompt` → `num_masks=0` means bad prompt
- `plan_grasp` → `num_grasps=0` means no grasp candidates found
- `select_top_down_grasp` → `found_grasp=false` means no top-down grasp passed threshold
- `move_to_joints` → `completed=true` always logged (motion always runs to completion or times out)

---

## API Class per Task

| Task | API registered name | Notes |
|---|---|---|
| `cube_lifting` | `FrankaControlApiReducedSkillLibraryTraced` | Standard single-arm |
| `cube_restack` | `FrankaControlApiReducedSkillLibraryTraced` | Standard single-arm |
| `cube_stack` | `FrankaControlApiReducedSkillLibraryTraced` | Standard single-arm |
| `nut_assembly` | `FrankaControlApiReducedSkillLibraryTraced` | Standard single-arm |
| `spill_wipe` | `FrankaControlApiReducedSkillLibrarySpillWipeTraced` | tcp_offset=[0,0,-0.0158] — sponge attachment |
| `two_arm_lift` | `FrankaControlApiReducedSkillLibraryBimanualTraced` | bimanual=True |
| `two_arm_handover` | `FrankaControlApiReducedSkillLibraryBimanualHandoverTraced` | bimanual=True, is_handover=True (wider z_range in plan_grasp) |

---

## Key Source Files

| File | Role |
|---|---|
| `scripts/robosuite/replay_trial_robosuite.py` | Single trial runner (replay saved code or interactive REPL) |
| `scripts/robosuite/gen_progress_robosuite.py` | Generate/update fix loop progress file from disk state |
| `cap/integrations/franka/control_reduced_skill_library.py` | Geometry/grasp helper functions added to base API |
| `cap/integrations/franka/control_reduced.py` | Core API: get_observation, SAM3, plan_grasp, IK, motion |
| `cap/integrations/trace_logger.py` | TraceLogger + TracedApiMixin |
| `cap/integrations/robosuite_trace_logger.py` | Robosuite traced API wrapper |
| `cap/integrations/__init__.py` | All API registrations — source of truth for which class each task uses |
| `env_configs/robosuite/<task>_multimodel_aspire_traced.yaml` | Per-task config |
