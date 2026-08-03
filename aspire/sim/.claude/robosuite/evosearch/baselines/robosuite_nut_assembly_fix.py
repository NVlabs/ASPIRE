import numpy as np

###############################################################################
# nut_assembly fix code (final)
#
# Key findings:
# - Nut+handle is FLAT 2cm-thick on table. Table Z=-0.102, Nut top Z=-0.082
# - TCP at nut MID-HEIGHT Z=-0.092 for successful grasp
# - Peg is raised (Z~0.028), Y>0 (vs cylindrical peg at Y<0)
# - Grasp handle, compute offset to nut body center (hole proxy)
# - Align hole over peg, insert gradually, release
###############################################################################

TABLE_Z = -0.102


def safe_pixel_to_world(u, v, depth, K, E, sr=15):
    u, v = int(u), int(v)
    z = float(depth[v, u])
    if np.isfinite(z) and z > 0:
        return pixel_to_world_point(u, v, z, K, E)
    for r in range(1, sr):
        for vv in range(max(0, v - r), min(depth.shape[0], v + r + 1)):
            for uu in range(max(0, u - r), min(depth.shape[1], u + r + 1)):
                zz = float(depth[vv, uu])
                if np.isfinite(zz) and zz > 0:
                    return pixel_to_world_point(uu, vv, zz, K, E)
    return None


def find_peg(rgb, depth, K, E):
    candidates = []
    for prompt in ["brown block", "small block"]:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for m in masks:
            area = m['mask'].sum()
            if 50 < area < 2000:
                pts = mask_to_world_points(m['mask'].astype(np.uint8), depth, K, E)
                if len(pts) > 10:
                    center = np.median(pts, axis=0)
                    top_z = np.max(pts[:, 2])
                    if center[2] > -0.05 and center[1] > 0:
                        candidates.append({
                            'center': center, 'top_z': top_z,
                            'score': m.get('score', 0),
                        })
        if candidates:
            break
    if not candidates:
        for p in ["small brown cube peg", "brown square block"]:
            try:
                res = point_prompt_molmo(rgb, p)
                uv = list(res.values())[0]
                if uv[0] is not None:
                    pt = safe_pixel_to_world(uv[0], uv[1], depth, K, E)
                    if pt is not None and pt[2] > -0.05:
                        candidates.append({'center': pt, 'top_z': pt[2], 'score': 0.5})
                        break
            except Exception:
                pass
    if not candidates:
        raise RuntimeError("No peg")
    best = max(candidates, key=lambda c: c['score'])
    return best['center'], best['top_z']


def find_nut_handle_hole(rgb, depth, K, E):
    """
    Returns: handle_center, handle_pts, hole_center_xy, nut_top_z
    hole_center_xy is the centroid of nut body mask MINUS handle mask.
    """
    # Nut body
    nut_masks = segment_sam3_text_prompt(rgb, "brown square nut")
    best_nut_mask = None
    nut_center = None
    nut_top_z = None
    for m in sorted(nut_masks, key=lambda d: d.get('score', 0), reverse=True):
        area = m['mask'].sum()
        if 800 < area < 6000:
            pts = mask_to_world_points(m['mask'].astype(np.uint8), depth, K, E)
            if len(pts) > 50:
                center = np.median(pts, axis=0)
                if -0.15 < center[2] < -0.05:
                    best_nut_mask = m['mask']
                    nut_center = center
                    nut_top_z = np.max(pts[:, 2])
                    break
    if nut_center is None:
        raise RuntimeError("No nut")

    # Handle
    handle_cands = []
    hmasks = segment_sam3_text_prompt(rgb, "extruded handle of the brown square nut")
    for m in hmasks:
        area = m['mask'].sum()
        if 50 < area < 1500:
            pts = mask_to_world_points(m['mask'].astype(np.uint8), depth, K, E)
            if len(pts) > 5:
                center = np.median(pts, axis=0)
                if -0.15 < center[2] < -0.05:
                    dist = np.linalg.norm(center[:2] - nut_center[:2])
                    if dist < 0.12:
                        handle_cands.append({
                            'mask': m['mask'], 'pts': pts, 'center': center,
                            'area': area, 'score': m.get('score', 0),
                        })

    if not handle_cands:
        try:
            res = point_prompt_molmo(rgb, "extruded handle of the brown square nut")
            uv = list(res.values())[0]
            if uv[0] is not None:
                pm = segment_sam3_point_prompt(rgb, (uv[0], uv[1]))
                for m in pm:
                    area = m['mask'].sum()
                    if 30 < area < 3000:
                        pts = mask_to_world_points(m['mask'].astype(np.uint8), depth, K, E)
                        if len(pts) > 5:
                            center = np.median(pts, axis=0)
                            if -0.15 < center[2] < -0.05:
                                handle_cands.append({
                                    'mask': m['mask'], 'pts': pts, 'center': center,
                                    'area': area, 'score': m.get('score', 0),
                                })
        except Exception:
            pass

    if not handle_cands:
        raise RuntimeError("No handle")

    best_h = max(handle_cands, key=lambda c: c['score'])
    handle_center = best_h['center']
    handle_pts = best_h['pts']
    handle_mask = best_h['mask']

    # Hole center = centroid of nut body MINUS handle
    body_mask = best_nut_mask.astype(bool) & ~handle_mask.astype(bool)
    body_pts = mask_to_world_points(body_mask.astype(np.uint8), depth, K, E)
    if len(body_pts) > 20:
        hole_xy = np.median(body_pts, axis=0)[:2]
    else:
        hole_xy = nut_center[:2]

    print(f"Handle: {handle_center[:2]}, Hole: {hole_xy}, Nut: {nut_center[:2]}")
    return handle_center, handle_pts, hole_xy, nut_top_z


def make_topdown_quat(handle_pts=None):
    down_z = np.array([0, 0, -1])
    if handle_pts is not None and len(handle_pts) > 20:
        try:
            obb = get_oriented_bounding_box_from_3d_points(handle_pts)
            principal = obb['R'][:, np.argmax(obb['extent'])].copy()
            principal[2] = 0
            norm = np.linalg.norm(principal)
            if norm > 1e-6:
                principal /= norm
                grip_y = principal
                grip_x = np.cross(grip_y, down_z)
                grip_x /= (np.linalg.norm(grip_x) + 1e-8)
                grip_y = np.cross(down_z, grip_x)
                grip_y /= (np.linalg.norm(grip_y) + 1e-8)
                return rotation_matrix_to_quaternion(np.column_stack([grip_x, grip_y, down_z]))
        except Exception:
            pass
    R = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    return rotation_matrix_to_quaternion(R)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

neutral = np.array([0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78])
open_gripper()
move_to_joints(neutral)

obs = get_observation()
rgb = obs["robot0_robotview"]["images"]["rgb"]
depth = obs["robot0_robotview"]["images"]["depth"]
K = obs["robot0_robotview"]["intrinsics"]
E = obs["robot0_robotview"]["pose_mat"]

peg_center, peg_top_z = find_peg(rgb, depth, K, E)
handle_center, handle_pts, hole_xy, nut_top_z = find_nut_handle_hole(rgb, depth, K, E)

nut_mid_z = (nut_top_z + TABLE_Z) / 2.0
grasp_quat = make_topdown_quat(handle_pts)
grasp_pos = handle_center.copy()
grasp_pos[2] = nut_mid_z

offset_xy = hole_xy - grasp_pos[:2]
print(f"Peg: {peg_center[:2]}, Grasp: {grasp_pos[:2]}, Offset: {offset_xy}")

# GRASP
open_gripper()
pre = grasp_pos.copy()
pre[2] += 0.05
move_to_joints(solve_ik(pre, grasp_quat))
move_to_joints(solve_ik(grasp_pos, grasp_quat))
close_gripper()

# LIFT
lift_z = 0.10
move_to_joints(solve_ik(np.array([grasp_pos[0], grasp_pos[1], lift_z]), grasp_quat))

# ALIGN
target_x = peg_center[0] - offset_xy[0]
target_y = peg_center[1] - offset_xy[1]
move_to_joints(solve_ik(np.array([target_x, target_y, lift_z]), grasp_quat))

# INSERT
for ez in [0.06, 0.02, -0.02, -0.06, -0.085, -0.095]:
    try:
        move_to_joints(solve_ik(np.array([target_x, target_y, ez]), grasp_quat))
    except Exception as e:
        print(f"IK fail z={ez}: {e}")
        break

# RELEASE
open_gripper()

print("Done")
