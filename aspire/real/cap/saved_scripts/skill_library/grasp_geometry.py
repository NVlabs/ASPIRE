# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from skill_library.namespace import *  # noqa: F401,F403
except AttributeError:
    pass

import collections
import os

import numpy as np

try:
    from skill_library.constants.vision import (  # type: ignore
        ANYGRASP_DISABLE_PLANNER_Z_CLIPPING,
        ANYGRASP_TCP_OFFSET_Z_M,
    )
except Exception:
    ANYGRASP_TCP_OFFSET_Z_M = 0.0
    ANYGRASP_DISABLE_PLANNER_Z_CLIPPING = True

try:
    from skill_library.constants.manipulation import GRIPPER_WIDTH_M  # type: ignore
except Exception:
    GRIPPER_WIDTH_M = 0.08


def _tool(name):
    fn = globals().get(name)
    if fn is not None:
        return fn
    import skill_library.namespace as namespace

    fn = getattr(namespace, name, None)
    if fn is None:
        raise RuntimeError(f"Required tool is not available in skill_library.namespace: {name}")
    return fn


def _as_grasp(position, rpy, score=1.0, width=None, trajectory_cache_key=None):
    Grasp = collections.namedtuple(
        "GraspCandidate", ["position", "rpy", "score", "width", "trajectory_cache_key"]
    )
    return Grasp(
        position=[float(x) for x in position],
        rpy=[float(x) for x in rpy],
        score=float(score),
        width=float(GRIPPER_WIDTH_M if width is None else width),
        trajectory_cache_key=trajectory_cache_key,
    )


def _det_value(det, name, default=None):
    if isinstance(det, dict):
        return det.get(name, default)
    return getattr(det, name, default)


def _parse_float_list(value, default):
    if value is None:
        return [float(x) for x in default]
    if isinstance(value, str):
        if not value.strip():
            return [float(x) for x in default]
        parts = [part.strip() for part in value.replace(";", ",").split(",")]
        return [float(part) for part in parts if part]
    return [float(x) for x in value]


def _first_detection(det_map, object_name):
    if isinstance(det_map, dict):
        dets = det_map.get(object_name) or []
        if not dets:
            for maybe_dets in det_map.values():
                if maybe_dets:
                    dets = maybe_dets
                    break
    else:
        dets = det_map or []
    return dets[0] if dets else None


def sample_topdown_geometric(
    object_name,
    camera="top",
    max_grasps=None,
    yaws=None,
    pitches=None,
    z_offsets=None,
    z_offset_m=None,
    width=None,
    clip_min_z=None,
):
    """Generate simple top-down grasp candidates from BundleSDF/SAM3 3D detection."""
    try:
        det_map = _tool("detect_objects_oneshot")(object_name, camera=camera)
    except TypeError:
        det_map = _tool("detect_objects_oneshot")(object_name)
    except Exception as exc:
        print(f"  Top-down geometric localization {object_name!r} failed: {exc}")
        return []

    det = _first_detection(det_map, object_name)
    if det is None:
        print(f"  Top-down geometric localization found no detections for {object_name!r}")
        return []

    position = _det_value(det, "position_3d")
    if position is None:
        position = _det_value(det, "position")
    if position is None:
        print(f"  Top-down geometric detection for {object_name!r} has no 3D position")
        return []
    center = np.asarray(position, dtype=float).reshape(-1)
    if center.shape[0] < 3:
        print(f"  Top-down geometric detection for {object_name!r} has invalid position={position}")
        return []
    center = center[:3]

    default_yaws = (90.0, 0.0, -45.0, 45.0, -90.0, 135.0, -135.0, 180.0)
    yaws = _parse_float_list(
        yaws if yaws is not None else os.environ.get("OPENFORGE_TOPDOWN_GRASP_YAWS"),
        default_yaws,
    )
    pitches = _parse_float_list(
        pitches
        if pitches is not None
        else os.environ.get("OPENFORGE_TOPDOWN_GRASP_PITCHES"),
        (180.0,),
    )

    if z_offset_m is None:
        z_offset_m = float(os.environ.get("OPENFORGE_TOPDOWN_GRASP_Z_OFFSET_M", "0.0"))
    default_z_offsets = (float(z_offset_m), float(z_offset_m) + 0.01, float(z_offset_m) - 0.01)
    z_offsets = _parse_float_list(
        z_offsets
        if z_offsets is not None
        else os.environ.get("OPENFORGE_TOPDOWN_GRASP_Z_OFFSETS_M"),
        default_z_offsets,
    )

    if width is None:
        width = float(os.environ.get("OPENFORGE_TOPDOWN_GRASP_WIDTH_M", GRIPPER_WIDTH_M))
    max_count = int(max_grasps or len(yaws) * len(z_offsets))
    base_score = float(_det_value(det, "score", 1.0) or 1.0)
    base_score = max(0.05, min(1.0, base_score))

    grasps = []
    for z_index, z_offset in enumerate(z_offsets):
        for pitch_index, pitch in enumerate(pitches):
            for yaw_index, yaw in enumerate(yaws):
                grasp_pos = center.copy()
                grasp_pos[2] = float(grasp_pos[2] + z_offset)
                if clip_min_z is not None:
                    grasp_pos[2] = max(float(grasp_pos[2]), float(clip_min_z))
                score = max(
                    0.01,
                    base_score - 0.02 * yaw_index - 0.04 * pitch_index - 0.05 * z_index,
                )
                grasps.append(
                    _as_grasp(
                        grasp_pos,
                        [0.0, float(pitch), float(yaw)],
                        score=score,
                        width=width,
                    )
                )
                if len(grasps) >= max_count:
                    break
            if len(grasps) >= max_count:
                break
        if len(grasps) >= max_count:
            break

    print(
        f"  Top-down geometric grasps for {object_name!r}: "
        f"center={[round(float(x), 4) for x in center]}, camera={camera}, "
        f"candidates={len(grasps)}, yaws={[round(float(x), 1) for x in yaws]}, "
        f"pitches={[round(float(x), 1) for x in pitches]}, "
        f"z_offsets={[round(float(x), 3) for x in z_offsets]}, width={float(width):.3f}"
    )
    return grasps


def _clip_grasp_min_z(grasp, min_z):
    clipped_z = max(float(grasp.position[2]), float(min_z))
    if clipped_z <= float(grasp.position[2]):
        return grasp
    position = [float(x) for x in grasp.position]
    original_z = position[2]
    position[2] = clipped_z
    print(f"  Clipping AnyGrasp z from {original_z:.4f}m to {clipped_z:.4f}m")
    return _as_grasp(
        position,
        getattr(grasp, "rpy", [0.0, 180.0, 0.0]),
        score=getattr(grasp, "score", 1.0),
        width=getattr(grasp, "width", GRIPPER_WIDTH_M),
        trajectory_cache_key=getattr(grasp, "trajectory_cache_key", None),
    )


def _display_rpy_to_rotation(rpy):
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    return Rotation.from_euler("xyz", [-pitch, roll, -yaw - 90.0], degrees=True)


def _rotation_to_display_rpy(rot):
    ex, ey, ez = rot.as_euler("xyz", degrees=True)
    disp = np.array([ey, -ex, -ez - 90.0], dtype=float)
    return ((disp + 180.0) % 360.0 - 180.0).tolist()


def _grasp_rpy(grasp):
    if isinstance(grasp, dict):
        return grasp.get("rpy")
    return getattr(grasp, "rpy", None)


def _copy_grasp_with_rpy(grasp, rpy):
    if isinstance(grasp, dict):
        out = dict(grasp)
        out["rpy"] = [float(x) for x in rpy]
        return out
    return _as_grasp(
        getattr(grasp, "position"),
        rpy,
        score=getattr(grasp, "score", 1.0),
        width=getattr(grasp, "width", GRIPPER_WIDTH_M),
        trajectory_cache_key=getattr(grasp, "trajectory_cache_key", None),
    )


def _wrist_camera_y_dot_from_rpy(rpy):
    rot = _display_rpy_to_rotation(rpy)
    return float(rot.as_matrix()[1, 1])


def _yaw_flip_rpy_for_wrist_camera(rpy):
    from scipy.spatial.transform import Rotation

    rot = _display_rpy_to_rotation(rpy)
    flipped = rot * Rotation.from_euler("z", 180.0, degrees=True)
    return _rotation_to_display_rpy(flipped)


def filter_anygrasp_wrist_camera_y(
    grasps,
    *,
    threshold=0.0,
    allow_yaw_flip=True,
    log=True,
):
    """Keep AnyGrasp poses whose wrist-camera/local +Y axis points toward world +Y."""
    if not grasps:
        return []

    kept = []
    n_flipped = 0
    n_filtered = 0
    n_unchecked = 0
    threshold = float(threshold)
    for grasp in grasps:
        rpy = _grasp_rpy(grasp)
        if rpy is None:
            kept.append(grasp)
            n_unchecked += 1
            continue
        y_dot = _wrist_camera_y_dot_from_rpy(rpy)
        if y_dot >= threshold:
            kept.append(grasp)
            continue
        if allow_yaw_flip:
            flipped_rpy = _yaw_flip_rpy_for_wrist_camera(rpy)
            flipped_y_dot = _wrist_camera_y_dot_from_rpy(flipped_rpy)
            if flipped_y_dot >= threshold:
                kept.append(_copy_grasp_with_rpy(grasp, flipped_rpy))
                n_flipped += 1
                continue
        n_filtered += 1

    if log and (n_flipped or n_filtered or n_unchecked):
        print(
            "  AnyGrasp wrist-camera +Y filter: "
            f"kept={len(kept)}/{len(grasps)} yaw_flipped={n_flipped} "
            f"filtered={n_filtered} unchecked={n_unchecked} threshold={threshold:.3f}"
        )
    return kept


def sample_anygrasp(
    object_name,
    camera="top",
    max_grasps=None,
    tcp_offset_z_m=None,
    disable_planner_z_clipping=None,
    clip_min_z=None,
    filter_wrist_camera_y=True,
    wrist_camera_y_threshold=0.0,
    allow_wrist_camera_yaw_flip=True,
):
    kwargs = {
        "object_name": object_name,
        "camera": camera,
        "object_input_mode": "segmented_object_cloud",
    }
    if max_grasps is not None:
        kwargs["max_grasps"] = int(max_grasps)
    tcp_offset = ANYGRASP_TCP_OFFSET_Z_M if tcp_offset_z_m is None else tcp_offset_z_m
    disable_clip = (
        ANYGRASP_DISABLE_PLANNER_Z_CLIPPING
        if disable_planner_z_clipping is None
        else disable_planner_z_clipping
    )
    try:
        grasps = _tool("sample_grasp_pose_anygrasp")(
            **kwargs,
            tcp_offset_z_m=tcp_offset,
            disable_planner_z_clipping=disable_clip,
            filter_wrist_camera_y=bool(filter_wrist_camera_y),
            wrist_camera_y_threshold=float(wrist_camera_y_threshold),
            allow_wrist_camera_yaw_flip=bool(allow_wrist_camera_yaw_flip),
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        grasps = _tool("sample_grasp_pose_anygrasp")(**kwargs)
    except Exception as exc:
        print(f"  AnyGrasp query {object_name!r} failed: {exc}")
        return []
    if filter_wrist_camera_y:
        try:
            grasps = filter_anygrasp_wrist_camera_y(
                grasps,
                threshold=wrist_camera_y_threshold,
                allow_yaw_flip=allow_wrist_camera_yaw_flip,
            )
        except Exception as exc:
            print(f"  AnyGrasp wrist-camera +Y filter skipped: {exc}")
    if clip_min_z is None:
        return grasps
    return [_clip_grasp_min_z(grasp, clip_min_z) for grasp in grasps]
