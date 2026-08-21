# LIBERO Cup and Mug Skills

Simulation workspace: `aspire/sim` (from `aspire/real`, use `../sim`)
Branch inspected: `libero-integration-skills`
Generated: 2026-06-05

Scope: cup, mug, coffee-cup, ramekin-as-cup, and handled drinking-vessel related skills from the LIBERO skill docs. I found no true `glass` / `glasses` skill entries on this branch. Most reusable material is for mugs; "cup" mostly appears as a ramekin synonym, pudding packaging wording, or a fallback prompt for mugs.

## Quick Index

| Object / pattern | Suites or task families | Core skill |
|---|---|---|
| White mug | `libero_90`, `libero_10_*` mug/plate tasks | Use specific prompts, body XY centroid, top-down grasp at `top - 0.025`. |
| Red mug | `libero_90` mug/plate and caddy tasks | Bottom-slice body localization, yaw retries, far-Y plate transport handling. |
| Yellow-and-white mug | `libero_90`, `libero_10_*`, SS1 caddy tasks | Handle localization and handle grasp; body too wide for gripper. |
| Ramekin / "white cup" | `libero_spatial_swap`, `libero_spatial_task` | Treat as a small ceramic bowl/cup anchor; filter low-Z points. |
| "Chocolate pudding cup" | `libero_object_*`, `libero_90` SCENE6 | Not a drinking cup. Treat as a flat box/pudding container with Z filtering. |
| Round/rim grasped mug-like objects | General LIBERO transport | Incremental lift to avoid IK snap and drops. |

## Relevant Tasks

LIBERO-90 ordering entries involving mugs:

```text
LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate
LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate
LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate
STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy
LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate
LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate
LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate
KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug
STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy
STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy
```

LIBERO-10 long-pro prompt entries involving mugs:

```text
KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it
LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
```

## Localization Skills

### Body XY For Handled Objects

Use `body_xy_centroid(obj_pts)` instead of raw OBB center for mugs, handled pots, ketchup bottles, or any object where the SAM3 mask includes a handle or shadow outliers. The observed failure mode is a 3-5 cm XY bias toward the handle or leaked mask region.

When prompt/localization has left/right ambiguity, use world-Y sorting consistently:

- Left plate: min world-Y.
- Right plate: max world-Y.
- Caddy compartments: detect all compartments, dedupe by 3D center, sort by world-Y ascending.
- Side-by-side mugs/moka pots: apply the same convention unless the task language or BDDL says otherwise.

### Two-Pass Handle Localization

Use this when `segment_sam3_text_prompt("mug handle")` returns a merged body+handle mask. The signal is `OBB ext_xy > 0.10m`. Filter mask points by distance from the known body center, keeping only handle-side points.

```python
def localize_handle(rgb, depth_img, K, E, body_center, handle_prompts=None, min_d_body=0.030):
    if handle_prompts is None:
        handle_prompts = ["mug handle", "handle"]

    for prompt in handle_prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue

        best = max(masks, key=lambda d: d["score"])
        pts = mask_to_world_points(best["mask"].astype(np.uint8), depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue

        obb = get_oriented_bounding_box_from_3d_points(pts)
        if max(obb["extent"][0], obb["extent"][1]) > 0.10:
            d_body = np.sqrt((pts[:, 0] - body_center[0]) ** 2 + (pts[:, 1] - body_center[1]) ** 2)
            pts = pts[d_body > min_d_body]
            if len(pts) < 5:
                continue

        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts

    return None, None
```

Validated on SS1 yellow-and-white mug where the merged body+handle mask was recovered by `d_body > 0.030`.

### Object-Specific Prompt Notes

| Object | Prompts | Suite / task family | Notes |
|---|---|---|---|
| White mug | `"white mug"`, `"white coffee mug"`, `"white ceramic mug"` | `mug_plate` | Do not use `"mug"` alone when red/decorated mugs are present. Use body XY centroid. |
| Red mug | `"red mug"` | `red_mug_plate`, `red_mug_left_lr5`, SS3 caddy | Tall, floral, handled, flared rim. Compute body axis from bottom slice: `pts[pts[:,2] < pts[:,2].min()+0.06]`. |
| Yellow-and-white mug | `"yellow and white mug"`, `"ceramic mug"`, `"coffee mug"`, `"mug"` | SS1 mug/caddy, LIBERO-10/90 | Body is about 9 cm diameter, too wide for gripper. Must grip handle. Handle center is about 6.1 cm in -Y from body center in SS1. |
| Ramekin / cup | `"small white ceramic bowl"`, `"ramekin"`, `"small white ramekin"`, `"ceramic cup"`, `"white cup"` | `libero_spatial_swap` | Filter points with Z in `(-0.01, 0.04)`. Use as an anchor for bowl-on-ramekin localization. |
| Chocolate pudding "cup" | `"chocolate pudding"`, `"pudding"`, `"chocolate pudding cup"`, `"brown pudding cup"` | `libero_object_task`, `libero_object_swap` | This is a flat package/container, not a drinking cup. In object-task scenes, filter Z to `[0.018, 0.045]` before score ranking. |
| Chocolate pudding box | `"chocolate pudding box"`, `"small box"`, `"brown box"`, `"dark box"` | `libero_90` SCENE6 / pudding plate | Add `"box"` qualifier. Filter `ext_z < 0.045`. Avoid confusing with mugs/bottles by limiting Z. |

## Grasp Skills

### Top-Down Fallback For Cylindrical Objects

`select_top_down_grasp` can return angled or side-approach grasps for small cylindrical objects such as mugs, bottles, bowls, and cans. If the gripper z-axis is not pointing sufficiently downward, override with a forced top-down quaternion.

```python
grasp_pos, quat = decompose_transform(best_grasp)
R_grasp = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
if R_grasp[2, 2] > -0.7:
    quat = make_topdown_quat(0)
```

Seen in `mug_plate` and similar short-cylinder tasks.

### Object-Height-Aware Release

For tall objects such as mugs and bottles, avoid a fixed `surface_z + 0.05` drop. Use actual object height:

```python
obj_height = obj_pts[:, 2].max() - obj_pts[:, 2].min()
release_z = surface_z + obj_height + 0.005
```

This prevents bouncing or crushing when placing mugs on plates.

### Mug Grasp Parameters

| Object | Grasp Z | Yaw | Notes |
|---|---:|---:|---|
| White mug | `top - 0.025` | `0` | Use body XY inlier/centroid filter because handle biases OBB. Prompt must specify `"white mug"`. |
| Yellow-and-white mug | `handle_top - 0.020` | `90` | Body is too wide for gripper, so grip the handle. Use body-offset compensation for placement: `gripper_target = place_target - (body_center - handle_center)`. |
| Red mug | `top - 0.020` | retry `0, 90, 45, 135` | Handle direction changes by seed. Try multiple yaws and keep the close with the best gripper-width signal. |

### Gripper-Width Diagnostic

After `close_gripper()`, read `obs["robot_cartesian_pos"][7]`.

| Width | Interpretation | Action |
|---:|---|---|
| `< 0.05` | Empty or air grasp | Re-localize and re-approach. |
| `0.10-0.30` | Likely handle/rim contact | For red mug, retry yaw. For SS1 handle-grip, `0.20-0.22` can be stable after the wide-grip retry below. |
| `> 0.23` on intended handle close | Body wedge through handle gap | Re-open, shift away from body, and re-close. |
| `> 0.30` | Body grip | Usually stable for red/white mug body transport. |

### Wide-Grip Retry For Handled Mugs

When yaw=90 on a handle gives `gw > 0.23`, the gripper may be wedged through the handle gap into the body wall. That grip drops during transport. Shift 5 mm away from the body along the handle direction and re-close.

```python
obs_after = get_observation()
gw0 = obs_after["robot_cartesian_pos"][7]
if gw0 > 0.23:
    open_gripper()
    dxy = np.array([handle_x - body_x, handle_y - body_y])
    dxy /= np.linalg.norm(dxy) + 1e-6
    retry_pos = [handle_x + 0.005 * dxy[0], handle_y + 0.005 * dxy[1], grasp_z]
    joints = solve_ik(retry_pos, quat.tolist())
    if joints is not None:
        move_to_joints(joints)
    close_gripper()
```

Validated on SS1 yellow-and-white mug; applies to any handled mug/cup where the handle gap allows a body wedge.

### Workspace Limit

In the M1 traced config, `panda_hand` X is effectively clamped near `0.75`. If `body_x > 0.75`, full top-down body grasps can fail because the fingers only reach the top edge. After descent, check `obs["robot_cartesian_pos"][0]`; if it is clamped at about `0.750` while target X is beyond that, treat the task as IK-limited.

Confirmed for SS3 red mug with body X roughly `0.73-0.78`; only seeds with body X at or below about `0.745` are reliably reachable.

## Transport and Placement Skills

### Fragile Rim Grasp Lift

For round bowls/mugs or rim grasps, do not jump directly to a high lift pose. Use incremental IK waypoints at about 5 cm spacing from the achieved hand Z.

```python
hand_z = obs["robot_cartesian_pos"][2]
lift_zs = [hand_z + 0.05, hand_z + 0.10, hand_z + 0.18, 0.30, 0.40]
```

This avoids IK joint snaps that jerk objects out of rim or handle contact.

### Far-Y Mug Plate Transport

Use the special transport pattern when the placement target has `abs(Y) > 0.25` and the arm must make a large X/Y transport delta. This was validated on:

```text
libero_90/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate
```

Critical details:

- Use 15 steps for the initial transition from current transport config to the first pre-probed config.
- Use 5 steps between subsequent levels.
- Call `close_gripper()` after the initial transition to re-secure the mug.
- Even if the EEF remains about 18-21 cm above the plate due to IK limits, accurate XY can still place the mug upright within success radius.

### Handle-Grasped Placement Offset

For handle-grasped mugs, the controlled gripper point is not the mug body center. Place using body-offset compensation:

```python
gripper_target_xy = place_target_xy - (body_center_xy - handle_center_xy)
```

This is called out for the SS1 yellow-and-white mug caddy task and generalizes to other handled cups/mugs.

### Caddy Placement Semantics

Do not assume "right of caddy" means "right compartment". In some LIBERO tasks the directory/BDDL predicate means exterior placement to the right of the whole caddy:

```python
target = (caddy_cx, caddy_y_max + 0.10)
```

Use `desk organizer` for the whole caddy and `compartment` with `return_all=True` for individual compartments. Trust the task directory/BDDL semantics over ambiguous task language.

## Prompt Fallbacks

When SAM3 text segmentation returns empty for a mug:

- Try specific color/object prompts first: `"red mug"`, `"white mug"`, `"yellow and white mug"`.
- Try synonym fallback: `"red coffee cup"`.
- For handled mugs, localize `"mug handle"` and fallback to `"handle"` if needed.
- In broad prompt smoke tests, include `"cup"`, `"mug"`, `"white mug"`, `"red mug"`, and `"yellow mug"`.

## Source Pointers

Primary sources on `libero-integration-skills`:

- `.claude/libero/library-size-scaling/skills/localize.md:72-107` - body XY centroid and two-pass handle localization.
- `.claude/libero/library-size-scaling/skills/localize.md:501-535` - object-specific mug/caddy prompt notes.
- `.claude/libero/library-size-scaling/skills/grasp.md:108-164` - top-down fallback and object-height-aware release.
- `.claude/libero/library-size-scaling/skills/grasp.md:297-356` - white mug, yellow-and-white mug, red mug, gripper-width, and wide-grip retry.
- `.claude/libero/library-size-scaling/skills/transport.md:112` - fragile rim grasp lift for round bowls/mugs.
- `.claude/libero/eureka/skills/transport.md:150-159` - red mug far-Y plate transport workaround.
- `.claude/libero/eureka/skills/localize.md:74-87` - pudding cup and ramekin/cup-adjacent localization notes.
- `ordering.txt:5,6,13,22,28,41,42,45,68,72` - LIBERO-90 mug task list.
