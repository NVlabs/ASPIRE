# API Reference

Full source: `cap/integrations/franka/libero_reduced_skill_library.py`

---

## Observation & Perception

| Function | Returns |
|---|---|
| `get_observation()` | `obs["agentview"]["images"]["rgb/depth"]`, `["intrinsics"]`, `["pose_mat"]` |
| `segment_sam3_text_prompt(rgb, text_prompt)` | `[{"mask": np.bool_, "box": [x1,y1,x2,y2], "score": float, "label": str}, ...]` |
| `segment_sam3_point_prompt(rgb, point_coords)` | `[{"mask": np.bool_, "score": float}, ...]` |
| `point_prompt_molmo(image, text_prompt)` | `{text_prompt: (pixel_x, pixel_y)}` |

## Grasp Planning & IK

| Function | Returns |
|---|---|
| `plan_grasp(depth, intrinsics, segmentation)` | `(grasp_poses: (K,4,4), grasp_scores: (K,))` — poses in **camera frame** |
| `select_top_down_grasp(poses, scores, ...)` | Best grasp 4×4 matrix, score |
| `solve_ik(position, quaternion_wxyz)` | Joint angles `(7,)` |

**Camera-to-world conversion** (required after `plan_grasp`):
```python
camera_extrinsics = obs["agentview"]["pose_mat"]  # (4,4) camera-to-world
grasp_world = camera_extrinsics @ grasp_camera_frame
```

## Motion Execution

| Function | Returns |
|---|---|
| `move_to_joints(joints)` | None (blocking) |
| `open_gripper()` / `close_gripper()` | None |
| `goto_home_joint_position()` | None (moves to robot home config) |

## Geometry Utilities

| Function | Description |
|---|---|
| `decompose_transform(T)` | 4×4 → `(position (3,), quaternion_wxyz (4,))` |
| `mask_to_world_points(mask, depth, intrinsics, extrinsics)` | 2D mask → `(N,3)` world-frame point cloud |
| `pixel_to_world_point(u, v, depth, intrinsics, extrinsics)` | Single pixel → 3D world point |
| `get_oriented_bounding_box_from_3d_points(pts)` | `(N,3)` → OBB `{center, extent, R}` |
| `depth_to_point_cloud(depth, intrinsics)` | Depth → `(N,3)` camera-frame point cloud |
| `rotation_matrix_to_quaternion(R)` | 3×3 → quaternion wxyz |
| `transform_points(points, T)` | Apply 4×4 transform to `(N,3)` |
| `normalize_vector(v)` | Normalize 3D vector |

## Task Language

```python
env.handle.task_language  # e.g. "Pick up the cream cheese and put it in the bowl"
```
Only reliable source for the actual goal — especially for `_task` suites where BDDL filenames are misleading.

---

## Success Conditions

| Placement type | Tolerance |
|---|---|
| Regular object `On` (bowl on plate, cream cheese in bowl) | XY < 3cm + contact. **Tight.** |
| Site object `In`/`On` (cabinet top, basket) | ±12cm × ±9cm site-local. **Forgiving.** |

---

## Scene Coordinate Frames

Use SAM3 depth projection for actual positions — never hardcode values. For reference only:

| Scene | Table Z (MuJoCo) | Robot base |
|---|---|---|
| Kitchen (`libero_goal*`, `libero_spatial*`) | 0.9 | `[-0.66, 0, 0.912]` |
| Grocery (`libero_object*`) | 0.0 | `[-0.6, 0, 0.0]` |
| `libero_10` | varies by task | varies |

---

## Output Directory Structure

```
outputs/libero_baseline_image_diff_gemini/
└── libero_goal_swap/
    └── put_the_bowl_on_the_stove/
        └── gcp_google_gemini-3.1-pro-preview/
            └── run/
                ├── aaa_done_flag           # EXISTS = experiment complete
                └── trial_05_sandboxrc_0_reward_0.000_taskcompleted_0/
                    ├── code.py             # generated/fix code
                    ├── trace.json          # PRIMARY: every API call
                    ├── keyframes/          # RGB snapshots at perception steps
                    ├── video_combined.mp4
                    ├── visual_feedback_*.png
                    ├── summary.txt         # stdout/stderr/reward
                    └── all_responses.json  # full LLM conversation
```

**Trial folder name:** `trial_NN_sandboxrc_X_reward_Y.YYY_taskcompleted_Z`
- `sandboxrc`: 0 = clean exit, 1 = crashed
- `reward`: 0.000–1.000

## TraceLogger Format

```json
{
  "step": 5,
  "function": "segment_sam3_text_prompt",
  "args": {"rgb_shape": [512, 800, 3], "text_prompt": "bowl"},
  "duration_ms": 1463.8,
  "result": {
    "num_masks": 200,
    "mask_0_score": 0.917,
    "mask_0_bbox": [369.2, 257.1, 438.4, 310.5],
    "mask_0_area_pct": 0.68
  },
  "keyframe_saved": true
}
```

---

## Key Source Files

| File | Role |
|---|---|
| `cap/envs/scripts/run_libero_batch.py` | Batch runner (suites × tasks × models) |
| `cap/envs/launch.py` | Core trial loop (reset → code gen → multi-turn → save) |
| `scripts/libero/replay_trial.py` | Single trial runner (fresh LLM or replay saved code) |
| `scripts/common/analyze_trial.py` | Structured trace analysis for a single trial |
| `scripts/libero/analyze_evosearch_traces.py` | Per-candidate trace analysis for Evolutionary Search runs |
| `scripts/common/scan_progress.py` | Aggregate success rates across an output dir |
| `cap/integrations/franka/libero_reduced_skill_library.py` | Extended API (geometry utilities) — read for exact signatures |
| `cap/integrations/trace_logger.py` | TraceLogger + TracedApiMixin |
| `cap/integrations/libero_trace_logger.py` | LIBERO traced API wrapper |
| `env_configs/libero/franka_libero_traced.yaml` | Traced replay/eval config |
| `env_configs/libero/franka_libero_baseline_debug.yaml` | Baseline debug config (seeds 51-65, `resume_idx: 51`) |
