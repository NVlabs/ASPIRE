try:
    from skill_library.namespace import *  # noqa: F401,F403
except AttributeError:
    pass

import collections

import numpy as np

from skill_library import grasp_geometry

try:
    from skill_library.constants.planning import (  # type: ignore
        BATCH_SOLVER_SPEED,
        BATCH_TOP_K,
        BATCH_VALIDATE_TRAJECTORY,
        IK_ERROR_THRESHOLD_M,
        IK_RPY_WEIGHT,
        IK_XYZ_WEIGHT,
        MOTION_PLANNER_BACKEND,
        PLANNING_SPEED,
    )
except Exception:
    PLANNING_SPEED = 1.5
    IK_ERROR_THRESHOLD_M = 0.01
    IK_XYZ_WEIGHT = 1.0
    IK_RPY_WEIGHT = 0.3
    BATCH_TOP_K = 16
    BATCH_SOLVER_SPEED = "fast"
    BATCH_VALIDATE_TRAJECTORY = False
    MOTION_PLANNER_BACKEND = "curobo"

try:
    from skill_library.constants.vision import (  # type: ignore
        ANYGRASP_DISABLE_PLANNER_Z_CLIPPING,
        ANYGRASP_TCP_OFFSET_Z_M,
        BUNDLESDF_CAMERA,
        DEFAULT_TARGET_DROP_Z_OFFSET,
        TABLE_TARGET_DROP_Z_OFFSETS,
        TOP_GRASP_MAX,
        TOP_GRASP_TRY,
    )
except Exception:
    ANYGRASP_TCP_OFFSET_Z_M = 0.0
    ANYGRASP_DISABLE_PLANNER_Z_CLIPPING = True
    BUNDLESDF_CAMERA = "top"
    DEFAULT_TARGET_DROP_Z_OFFSET = 0.10
    TABLE_TARGET_DROP_Z_OFFSETS = {
        "blue plate": 0.10,
        "yellow plate": 0.10,
        "brown cardboard box": 0.10,
    }
    TOP_GRASP_MAX = 16
    TOP_GRASP_TRY = 16

try:
    from skill_library.constants.robot import (  # type: ignore
        HOME_VIEW_Z_OFFSET,
        LEFT_BIRDEYE_VIEW_RPY,
        LEFT_HOME_XYZ,
        RIGHT_BIRDEYE_VIEW_RPY,
        RIGHT_HOME_XYZ,
    )
except Exception:
    LEFT_HOME_XYZ = [0.4975, 0.31, 0.914]
    RIGHT_HOME_XYZ = [0.4975, -0.31, 0.914]
    HOME_VIEW_Z_OFFSET = 0.15
    LEFT_BIRDEYE_VIEW_RPY = [0.0, 125.0, 20.0]
    RIGHT_BIRDEYE_VIEW_RPY = [0.0, 125.0, -20.0]

try:
    from skill_library.constants.manipulation import MAX_GRASP_ATTEMPTS  # type: ignore
except Exception:
    MAX_GRASP_ATTEMPTS = 5

TARGET_DROP_Z_OFFSETS = dict(TABLE_TARGET_DROP_Z_OFFSETS)


def _base_freespace_kwargs(
    preview_only=False,
    planning_speed=None,
    ik_error_threshold=None,
    ik_xyz_weight=None,
    ik_rpy_weight=None,
    planner_backend=None,
):
    return {
        "planning_speed": PLANNING_SPEED if planning_speed is None else planning_speed,
        "ik_error_threshold": IK_ERROR_THRESHOLD_M if ik_error_threshold is None else ik_error_threshold,
        "ik_xyz_weight": IK_XYZ_WEIGHT if ik_xyz_weight is None else ik_xyz_weight,
        "ik_rpy_weight": IK_RPY_WEIGHT if ik_rpy_weight is None else ik_rpy_weight,
        "planner_backend": MOTION_PLANNER_BACKEND if planner_backend is None else planner_backend,
        "preview_only": preview_only,
    }


def birdseye_pose(
    side,
    left_home_xyz=None,
    right_home_xyz=None,
    home_view_z_offset=None,
    left_birdeye_view_rpy=None,
    right_birdeye_view_rpy=None,
):
    left_home_xyz = LEFT_HOME_XYZ if left_home_xyz is None else left_home_xyz
    right_home_xyz = RIGHT_HOME_XYZ if right_home_xyz is None else right_home_xyz
    home_view_z_offset = HOME_VIEW_Z_OFFSET if home_view_z_offset is None else home_view_z_offset
    left_birdeye_view_rpy = LEFT_BIRDEYE_VIEW_RPY if left_birdeye_view_rpy is None else left_birdeye_view_rpy
    right_birdeye_view_rpy = RIGHT_BIRDEYE_VIEW_RPY if right_birdeye_view_rpy is None else right_birdeye_view_rpy
    if side == "left":
        return (
            [left_home_xyz[0], left_home_xyz[1], left_home_xyz[2] + home_view_z_offset],
            list(left_birdeye_view_rpy),
        )
    return (
        [right_home_xyz[0], right_home_xyz[1], right_home_xyz[2] + home_view_z_offset],
        list(right_birdeye_view_rpy),
    )


def go_birdeye(side="both", planning_speed=None, **config):
    kwargs = _base_freespace_kwargs(
        preview_only=False,
        planning_speed=planning_speed,
        ik_error_threshold=config.get("ik_error_threshold"),
        ik_xyz_weight=config.get("ik_xyz_weight"),
        ik_rpy_weight=config.get("ik_rpy_weight"),
        planner_backend=config.get("planner_backend"),
    )
    if side in ("both", "left"):
        left_pos, left_rpy = birdseye_pose("left", **_birdseye_config(config))
        kwargs.update(
            {
                "left_target_pos": left_pos,
                "left_target_rpy": left_rpy,
                "left_gripper_target_width": 1.0,
            }
        )
    if side in ("both", "right"):
        right_pos, right_rpy = birdseye_pose("right", **_birdseye_config(config))
        kwargs.update(
            {
                "right_target_pos": right_pos,
                "right_target_rpy": right_rpy,
                "right_gripper_target_width": 1.0,
            }
        )
    return freespace_move(**kwargs)


def _birdseye_config(config):
    return {
        "left_home_xyz": config.get("left_home_xyz"),
        "right_home_xyz": config.get("right_home_xyz"),
        "home_view_z_offset": config.get("home_view_z_offset"),
        "left_birdeye_view_rpy": config.get("left_birdeye_view_rpy"),
        "right_birdeye_view_rpy": config.get("right_birdeye_view_rpy"),
    }


def _home_pose(side, left_home_xyz=None, right_home_xyz=None):
    left_home_xyz = LEFT_HOME_XYZ if left_home_xyz is None else left_home_xyz
    right_home_xyz = RIGHT_HOME_XYZ if right_home_xyz is None else right_home_xyz
    return list(left_home_xyz if side == "left" else right_home_xyz)


def _go_home_best_effort():
    try:
        go_home()
        return True
    except TypeError:
        go_home(None)
        return True
    except Exception as exc:
        print(f"  go_home failed: {exc}")
        return False


def _normalize_side(side):
    if side is None:
        return None
    normalized = str(side).strip().lower()
    if not normalized:
        return None
    if normalized not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return normalized


def choose_arm(xyz, force_side=None):
    forced = _normalize_side(force_side)
    if forced is not None:
        print(f"  Forced arm override: {forced}")
        return forced

    obj_pos = np.asarray(xyz, dtype=float)
    state = get_robot_state()
    left_dist = float(np.linalg.norm(obj_pos - np.asarray(state.left_ee_pos, dtype=float)))
    right_dist = float(np.linalg.norm(obj_pos - np.asarray(state.right_ee_pos, dtype=float)))
    side = "left" if left_dist <= right_dist else "right"
    print(f"  Distance to left gripper: {left_dist:.3f} m")
    print(f"  Distance to right gripper: {right_dist:.3f} m")
    print(f"  Closer arm: {side}")
    return side


def _gripper_pos_for_side(state, side):
    return float(state.left_gripper_pos if side == "left" else state.right_gripper_pos)


def _safe_move(
    side,
    target_pos=None,
    target_rpy=None,
    target_quat=None,
    trajectory_cache_key=None,
    **config,
):
    try:
        if trajectory_cache_key is not None:
            freespace_move(preview_only=False, trajectory_cache_key=trajectory_cache_key)
            return True
        kwargs = _base_freespace_kwargs(
            preview_only=False,
            planning_speed=config.get("planning_speed"),
            ik_error_threshold=config.get("ik_error_threshold"),
            ik_xyz_weight=config.get("ik_xyz_weight"),
            ik_rpy_weight=config.get("ik_rpy_weight"),
            planner_backend=config.get("planner_backend"),
        )
        prefix = "left" if side == "left" else "right"
        if target_pos is not None:
            kwargs[f"{prefix}_target_pos"] = [float(x) for x in target_pos]
        if target_rpy is not None:
            kwargs[f"{prefix}_target_rpy"] = [float(x) for x in target_rpy]
        if target_quat is not None:
            kwargs[f"{prefix}_target_quat"] = [float(x) for x in target_quat]
        freespace_move(**kwargs)
        return True
    except Exception as exc:
        print(f"  freespace_move failed: {exc}")
        return False


def _preview_move(side, target_pos=None, target_rpy=None, target_quat=None, label="pose", **config):
    try:
        kwargs = _base_freespace_kwargs(
            preview_only=True,
            planning_speed=config.get("planning_speed"),
            ik_error_threshold=config.get("ik_error_threshold"),
            ik_xyz_weight=config.get("ik_xyz_weight"),
            ik_rpy_weight=config.get("ik_rpy_weight"),
            planner_backend=config.get("planner_backend"),
        )
        prefix = "left" if side == "left" else "right"
        if target_pos is not None:
            kwargs[f"{prefix}_target_pos"] = [float(x) for x in target_pos]
        if target_rpy is not None:
            kwargs[f"{prefix}_target_rpy"] = [float(x) for x in target_rpy]
        if target_quat is not None:
            kwargs[f"{prefix}_target_quat"] = [float(x) for x in target_quat]
        result = freespace_move(**kwargs)
        print(f"  Preview {label} [{side}] succeeded: {getattr(result, 'status', 'Success')}")
        return True
    except Exception as exc:
        print(f"  Preview {label} [{side}] failed: {exc}")
        return False


def _preview_place_for_side(side, drop_pos, **config):
    if drop_pos is None:
        return True
    _, transport_rpy = birdseye_pose(side, **_birdseye_config(config))
    return _preview_move(side, drop_pos, transport_rpy, label="place", **config)


def _select_best_grasp(grasps, side, label="grasp", **config):
    if not grasps:
        return None
    kwargs = _base_freespace_kwargs(
        preview_only=False,
        planning_speed=config.get("planning_speed"),
        ik_error_threshold=config.get("ik_error_threshold"),
        ik_xyz_weight=config.get("ik_xyz_weight"),
        ik_rpy_weight=config.get("ik_rpy_weight"),
        planner_backend=config.get("planner_backend"),
    )
    kwargs.update(
        {
            "grasp_candidates": list(grasps),
            "batch_side": side,
            "batch_top_k": int(config.get("batch_top_k", BATCH_TOP_K)),
            "solver_speed": config.get("solver_speed", BATCH_SOLVER_SPEED),
            "batch_validate_trajectory": config.get(
                "batch_validate_trajectory", BATCH_VALIDATE_TRAJECTORY
            ),
        }
    )
    batch = freespace_move(**kwargs)
    print(
        f"  Batched cuRobo {label} ranking [{side}]: "
        f"input={int(getattr(batch, 'input_candidate_count', len(grasps)))}, "
        f"evaluated={int(getattr(batch, 'evaluated_candidate_count', 0))}, "
        f"truncated={int(getattr(batch, 'truncated_input_count', 0))}, "
        f"solve={float(getattr(batch, 'curobo_solve_time_ms', 0.0)):.1f}ms, "
        f"graph={float(getattr(batch, 'curobo_graph_time_ms', 0.0)):.1f}ms, "
        f"ik={float(getattr(batch, 'curobo_ik_time_ms', 0.0)):.1f}ms, "
        f"mode={getattr(batch, 'planning_mode', 'unknown')}"
    )
    best = getattr(batch, "best_candidate", None)
    if best is not None and getattr(best, "motion_plan_error", True) is False:
        SelectedGrasp = collections.namedtuple(
            "SelectedGrasp", ["position", "rpy", "score", "width", "trajectory_cache_key"]
        )
        selected = SelectedGrasp(
            position=[float(x) for x in best.position],
            rpy=[float(x) for x in best.rpy],
            score=float(best.score),
            width=float(best.width),
            trajectory_cache_key=getattr(best, "trajectory_cache_key", None),
        )
        print(
            f"  Selected {label}: rank={int(best.rank)}, side={side}, "
            f"xyz={[round(float(x), 4) for x in selected.position]}, "
            f"rpy={[round(float(x), 1) for x in selected.rpy]}, "
            f"score={selected.score:.3f}"
        )
        return selected

    failures = []
    for candidate in list(getattr(batch, "batch_candidates", []) or [])[:5]:
        if getattr(candidate, "motion_plan_error", False):
            failures.append(
                f"rank={int(getattr(candidate, 'rank', 0))}: "
                f"{getattr(candidate, 'motion_plan_reason', None) or 'Motion plan error'}"
            )
    raise RuntimeError(
        f"Batch cuRobo returned no feasible {label} on {side}. "
        f"{getattr(batch, 'reason', '') or ''}"
        + (f" Top failures: {'; '.join(failures)}" if failures else "")
    )


def _execute_grasp(side, grasp, label="grasp", **config):
    open_gripper(side)
    print(
        f"  Moving directly to {label} target: "
        f"xyz={[round(float(x), 4) for x in grasp.position]}, "
        f"rpy={[round(float(x), 1) for x in grasp.rpy]}, score={grasp.score:.3f}"
    )
    if not _safe_move(
        side,
        grasp.position,
        grasp.rpy,
        trajectory_cache_key=getattr(grasp, "trajectory_cache_key", None),
        **config,
    ):
        return None
    close_gripper(side)
    gripper_pos = _gripper_pos_for_side(get_robot_state(), side)
    print(f"  Gripper pos after close: {gripper_pos:.4f}")
    if gripper_pos > 0.0:
        print("  Grasp check passed")
        return True
    print("  Grasp check failed (gripper at/near zero)")
    open_gripper(side)
    return False


def _grasp_candidates(object_name, camera, grasp_mode, max_grasps, **kwargs):
    mode = str(grasp_mode).lower()
    if mode in ("anygrasp", "top_anygrasp"):
        return grasp_geometry.sample_anygrasp(
            object_name,
            camera=camera,
            max_grasps=max_grasps,
            tcp_offset_z_m=kwargs.get("tcp_offset_z_m", ANYGRASP_TCP_OFFSET_Z_M),
            disable_planner_z_clipping=kwargs.get(
                "disable_planner_z_clipping", ANYGRASP_DISABLE_PLANNER_Z_CLIPPING
            ),
            clip_min_z=kwargs.get("clip_min_z"),
        )
    if mode in ("topdown_geom", "topdown_geometric", "geometric_topdown"):
        return grasp_geometry.sample_topdown_geometric(
            object_name,
            camera=camera,
            max_grasps=max_grasps,
            yaws=kwargs.get("topdown_yaws"),
            pitches=kwargs.get("topdown_pitches"),
            z_offsets=kwargs.get("topdown_z_offsets") or kwargs.get("topdown_z_offsets_m"),
            z_offset_m=kwargs.get("topdown_z_offset_m"),
            width=kwargs.get("topdown_width_m"),
            clip_min_z=kwargs.get("clip_min_z"),
        )
    raise ValueError(
        "Pruned table-bussing runtime supports grasp_mode in "
        "('anygrasp', 'top_anygrasp', 'topdown_geom'), "
        f"got {grasp_mode!r}"
    )


def _rank_grasp_candidates(
    top_grasp_try,
    grasps,
    side,
    label,
    **kwargs,
):
    return _select_best_grasp(
        grasps[: min(top_grasp_try, len(grasps))],
        side,
        label=label,
        **kwargs,
    )


def _rank_grasp_candidates_reach_aware(
    top_grasp_try,
    grasps,
    preferred_side,
    label,
    **kwargs,
):
    force_side = _normalize_side(kwargs.get("force_side") or kwargs.get("preferred_side"))
    if force_side is not None:
        sides = [force_side]
    else:
        sides = [preferred_side, "right" if preferred_side == "left" else "left"]
    place_drop_pos = kwargs.get("place_drop_pos")
    place_aware = bool(kwargs.get("place_aware_arm_selection", False))
    failures = []
    for side in sides:
        try:
            selected = _rank_grasp_candidates(
                top_grasp_try,
                grasps,
                side,
                label,
                **kwargs,
            )
            if place_aware and not _preview_place_for_side(side, place_drop_pos, **kwargs):
                failures.append(f"{side}: place preview failed")
                continue
            return side, selected
        except Exception as exc:
            failures.append(f"{side}: {exc}")
            print(f"  Grasp ranking failed on {side}: {exc}")
    raise RuntimeError("No feasible grasp on either arm. " + " | ".join(failures))


def pick_object(object_name, camera="top", grasp_mode="anygrasp", max_attempts=None, **kwargs):
    attempts = int(MAX_GRASP_ATTEMPTS if max_attempts is None else max_attempts)
    max_grasps = int(kwargs.pop("max_grasps", TOP_GRASP_MAX))
    top_grasp_try = int(kwargs.pop("top_grasp_try", TOP_GRASP_TRY))

    for attempt in range(1, attempts + 1):
        print(f"\n  --- Grasp attempt {attempt}/{attempts} for {object_name} ---")
        if camera in ("left", "right"):
            raise ValueError("Pruned table-bussing runtime only supports top-camera grasping")
        grasps = _grasp_candidates(object_name, camera, grasp_mode, max_grasps, **kwargs)
        if not grasps:
            print("  No grasp candidates found")
            continue
        preferred_side = choose_arm(
            grasps[0].position if hasattr(grasps[0], "position") else grasps[0]["position"],
            force_side=kwargs.get("force_side") or kwargs.get("preferred_side"),
        )
        label = grasp_mode

        home_retried = False
        while True:
            try:
                side, selected = _rank_grasp_candidates_reach_aware(
                    top_grasp_try,
                    grasps,
                    preferred_side,
                    label,
                    **kwargs,
                )
            except Exception as exc:
                print(f"  Grasp ranking failed: {exc}")
                selected = None

            grasp_result = None
            if selected is not None:
                grasp_result = _execute_grasp(side, selected, label=label, **kwargs)
                if grasp_result:
                    return side

            if selected is not None and grasp_result is False:
                break
            if home_retried:
                break

            print("  Direct grasp planning from current pose failed; going home and retrying")
            if not _go_home_best_effort():
                break
            home_retried = True
            grasps = _grasp_candidates(object_name, camera, grasp_mode, max_grasps, **kwargs)
            if not grasps:
                print("  No grasp candidates found after go_home")
                break
            preferred_side = choose_arm(
                grasps[0].position if hasattr(grasps[0], "position") else grasps[0]["position"],
                force_side=kwargs.get("force_side") or kwargs.get("preferred_side"),
            )
    print(f"  Giving up on {object_name} after {attempts} attempts")
    return None


def _detect_bundlesdf_pose(object_name, camera=BUNDLESDF_CAMERA):
    det_map = detect_objects_oneshot(object_name, camera=camera)
    dets = det_map.get(object_name, []) if isinstance(det_map, dict) else []
    if not dets:
        raise RuntimeError(f"No BundleSDF one-shot detections found for {object_name!r}")
    det = dets[0]
    if not getattr(det, "position_3d", None):
        raise RuntimeError(f"BundleSDF detection for {object_name!r} did not return position_3d")
    return det


def estimate_drop(target_name, z_offset=None, camera=None, target_drop_z_offsets=None):
    det = _detect_bundlesdf_pose(target_name, camera=camera or BUNDLESDF_CAMERA)
    target_pos = np.asarray(det.position_3d, dtype=float)
    drop = target_pos.copy()
    offsets = TARGET_DROP_Z_OFFSETS if target_drop_z_offsets is None else target_drop_z_offsets
    offset = float(
        offsets.get(target_name, DEFAULT_TARGET_DROP_Z_OFFSET) if z_offset is None else z_offset
    )
    drop[2] = float(target_pos[2] + offset)
    print(
        f"  Target {target_name!r} drop estimate from BundleSDF pose: "
        f"xyz={[round(float(x), 4) for x in drop.tolist()]}, z_offset={offset:.3f}"
    )
    return drop.tolist()


def place_object(side, target_name=None, drop_pos=None, transport_rpy=None, **kwargs):
    open_on_failure = bool(kwargs.pop("open_on_failure", True))
    if transport_rpy is None:
        _, transport_rpy = birdseye_pose(side, **_birdseye_config(kwargs))
    if drop_pos is None:
        if target_name is None:
            raise ValueError("target_name or drop_pos is required")
        drop_pos = estimate_drop(
            target_name,
            z_offset=kwargs.get("target_drop_z_offset"),
            camera=kwargs.get("bundlesdf_camera", BUNDLESDF_CAMERA),
            target_drop_z_offsets=kwargs.get("target_drop_z_offsets"),
        )
    print(
        f"  Moving directly to target with BirdEyeView transport orientation: "
        f"xyz={[round(float(x), 4) for x in drop_pos]}, "
        f"rpy={[round(float(x), 1) for x in transport_rpy]}"
    )
    ok = _safe_move(side, drop_pos, transport_rpy, **kwargs)
    if ok or open_on_failure:
        open_gripper(side)
    return ok


def _move_holding_to_birdeye(side, **kwargs):
    birdseye_pos, transport_rpy = birdseye_pose(side, **_birdseye_config(kwargs))
    print(
        f"  Moving {side} arm to BirdEyeView while preserving gripper: "
        f"xyz={[round(float(x), 4) for x in birdseye_pos]}, "
        f"rpy={[round(float(x), 1) for x in transport_rpy]}"
    )
    _safe_move(side, birdseye_pos, transport_rpy, **kwargs)
    return transport_rpy


def _move_holding_to_home(side, transport_rpy, **kwargs):
    home_pos = _home_pose(
        side,
        left_home_xyz=kwargs.get("left_home_xyz"),
        right_home_xyz=kwargs.get("right_home_xyz"),
    )
    print(
        f"  Moving {side} arm home while preserving gripper: "
        f"xyz={[round(float(x), 4) for x in home_pos]}, "
        f"rpy={[round(float(x), 1) for x in transport_rpy]}"
    )
    _safe_move(side, home_pos, transport_rpy, **kwargs)


def _try_place_without_failure_release(side, object_name, target_name, transport_rpy, **kwargs):
    try:
        return place_object(
            side,
            target_name=target_name,
            drop_pos=kwargs.get("place_drop_pos"),
            transport_rpy=transport_rpy,
            open_on_failure=False,
            **kwargs,
        )
    except Exception as exc:
        print(f"  Failed to place {object_name} at {target_name}: {exc}")
        return False


def _place_with_recovery(side, object_name, target_name, transport_rpy, **kwargs):
    if _try_place_without_failure_release(side, object_name, target_name, transport_rpy, **kwargs):
        return True

    print("  Place planning failed; keeping gripper closed and retrying from BirdEyeView")
    transport_rpy = _move_holding_to_birdeye(side, **kwargs)
    if _try_place_without_failure_release(side, object_name, target_name, transport_rpy, **kwargs):
        return True

    print("  Place planning still failed; keeping gripper closed and retrying from home")
    _move_holding_to_home(side, transport_rpy, **kwargs)
    if _try_place_without_failure_release(side, object_name, target_name, transport_rpy, **kwargs):
        return True

    print("  Place recovery failed after BirdEyeView and home retries; opening gripper")
    open_gripper(side)
    return False


def pick_and_place(object_name, target_name, grasp_mode="anygrasp", **kwargs):
    print(f"\n{'=' * 50}")
    print(f"Picking: {object_name} -> {target_name} with {grasp_mode}")
    print(f"{'=' * 50}")
    if kwargs.get("place_aware_arm_selection", False) and kwargs.get("place_drop_pos") is None:
        try:
            kwargs["place_drop_pos"] = estimate_drop(
                target_name,
                z_offset=kwargs.get("target_drop_z_offset"),
                camera=kwargs.get("bundlesdf_camera", BUNDLESDF_CAMERA),
                target_drop_z_offsets=kwargs.get("target_drop_z_offsets"),
            )
            print("  Auto arm selection will require a feasible preview to the basket drop pose.")
        except Exception as exc:
            print(f"  Place-aware arm selection disabled; target preview failed: {exc}")
            kwargs["place_aware_arm_selection"] = False
    side = pick_object(object_name, grasp_mode=grasp_mode, **kwargs)
    if side is None:
        try:
            go_home()
        except Exception:
            go_birdeye()
        return False

    birdseye_pos, transport_rpy = birdseye_pose(side, **_birdseye_config(kwargs))
    print(
        f"  Moving {side} arm to BirdEyeView: "
        f"xyz={[round(float(x), 4) for x in birdseye_pos]}, "
        f"rpy={[round(float(x), 1) for x in transport_rpy]}"
    )
    _safe_move(side, birdseye_pos, transport_rpy, **kwargs)
    ok = _place_with_recovery(side, object_name, target_name, transport_rpy, **kwargs)
    if not ok:
        return False
    try:
        go_home()
    except Exception:
        go_birdeye()
    print(f"  {object_name} placed at {target_name}!")
    return bool(ok)
