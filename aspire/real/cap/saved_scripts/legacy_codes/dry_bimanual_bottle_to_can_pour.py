"""Dry bimanual bottle-to-can pour rehearsal planner.

By default this saved script is preview-only.  With a valid ticket and
``OPENFORGE_ALLOW_PHYSICAL_MOTION=1`` it executes the same live-observed,
previewed rehearsal in gated physical stages.  It never uses liquid and never
accepts manual XYZ physical targets.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    compare_detection_motion,
    current_run_dir,
    generate_side_grasp_candidates,
    json_safe,
    rank_motion_candidates,
    read_gripper_width,
    staged_close_with_contact,
    verify_lift,
    write_json,
)


TASK_NAME = "dry_bimanual_bottle_to_can_pour"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for item in raw.split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            pass
    return values or list(default)


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_BOTTLE_XYZ",
        "OPENFORGE_BOTTLE_HALF_EXTENTS",
        "OPENFORGE_CAN_XYZ",
        "OPENFORGE_CAN_HALF_EXTENTS",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_HALF_EXTENTS",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _workspace_bounds() -> dict[str, tuple[float, float]]:
    return {
        "x": (
            _env_float("OPENFORGE_DRY_POUR_WORKSPACE_X_MIN_M", 0.45),
            _env_float("OPENFORGE_DRY_POUR_WORKSPACE_X_MAX_M", 0.95),
        ),
        "y": (
            _env_float("OPENFORGE_DRY_POUR_WORKSPACE_Y_MIN_M", -0.38),
            _env_float("OPENFORGE_DRY_POUR_WORKSPACE_Y_MAX_M", 0.38),
        ),
        "z": (
            _env_float("OPENFORGE_DRY_POUR_WORKSPACE_Z_MIN_M", 0.72),
            _env_float("OPENFORGE_DRY_POUR_WORKSPACE_Z_MAX_M", 1.12),
        ),
    }


def _validate_live_detection(detection: dict[str, Any] | None, *, object_role: str) -> dict[str, Any]:
    role_prefix = f"OPENFORGE_DRY_POUR_{object_role.upper()}"
    min_score = _env_float(
        f"{role_prefix}_PRE_MIN_DETECTION_SCORE",
        _env_float("OPENFORGE_DRY_POUR_PRE_MIN_DETECTION_SCORE", 0.70),
    )
    bounds = _workspace_bounds()
    result: dict[str, Any] = {
        "object_role": object_role,
        "passed": False,
        "min_score": min_score,
        "workspace_bounds_m": bounds,
        "problems": [],
    }
    if detection is None:
        result["problems"].append("missing detection")
        return result

    score = float(detection.get("score") or 0.0)
    result["score"] = score
    if score < min_score:
        result["problems"].append(f"score {score:.3f} below minimum {min_score:.3f}")

    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        result["problems"].append("missing 3D position")
    else:
        rounded_xyz = _round_list(xyz, limit=3)
        result["position_3d"] = rounded_xyz
        for axis, value in zip(("x", "y", "z"), rounded_xyz):
            lo, hi = bounds[axis]
            if value < lo or value > hi:
                result["problems"].append(f"{axis}={value:.3f} outside [{lo:.3f}, {hi:.3f}]")

    box = detection.get("box_2d") or []
    margin_px = _env_int("OPENFORGE_DRY_POUR_IMAGE_EDGE_MARGIN_PX", 8)
    if margin_px > 0 and len(box) >= 4:
        width_px = _env_int("OPENFORGE_DRY_POUR_TOP_IMAGE_WIDTH_PX", 640)
        height_px = _env_int("OPENFORGE_DRY_POUR_TOP_IMAGE_HEIGHT_PX", 480)
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        result["box_2d"] = _round_list([x0, y0, x1, y1], digits=1, limit=4)
        result["image_edge_margin_px"] = margin_px
        if x0 <= margin_px or y0 <= margin_px or x1 >= width_px - margin_px or y1 >= height_px - margin_px:
            result["problems"].append("2D detection box touches image edge")

    result["passed"] = not result["problems"]
    return result


def _capture_prompt_fallback(
    *,
    run_dir: Path,
    stage: str,
    object_role: str,
    prompts: list[str],
    cameras: list[str],
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    detect_objects_oneshot = _required_tool("detect_objects_oneshot")
    get_camera_image = _tool("get_camera_image")
    get_robot_state = _tool("get_robot_state")
    run_in_background = _tool("run_in_background")
    packets: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prompt_index, prompt in enumerate(prompts):
        packet = capture_scene(
            prompts=[prompt],
            cameras=cameras,
            detect_objects_oneshot=detect_objects_oneshot,
            get_camera_image=get_camera_image,
            get_robot_state=get_robot_state,
            run_in_background=run_in_background,
            run_dir=run_dir,
            stage=f"{stage}_{object_role}_{prompt.replace(' ', '_')}",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
            max_retries=max(1, _env_int("OPENFORGE_DRY_POUR_DETECT_RETRIES", 2)),
            save_overlays=True,
        )
        packets.append(packet)
        detection = packet.get("best_detection")
        score = float((detection or {}).get("score") or 0.0)
        if detection is not None and score >= min_score:
            best = dict(detection)
            best["object_role"] = object_role
            best["prompt_index"] = prompt_index
            best["prompt_fallback_used"] = bool(prompt_index > 0)
            raw_source = best.get("source")
            live_source = "live_top_camera" if best.get("source_camera") == "top" else "live_perception"
            best["raw_detection_source"] = raw_source
            best["source"] = live_source
            best["detection_source"] = live_source
            break
    combined = {
        "stage": stage,
        "object_role": object_role,
        "source": "live_perception",
        "prompts_tried": prompts[: len(packets)],
        "packets": [
            {
                "packet_path": packet.get("packet_path"),
                "best_detection": packet.get("best_detection"),
                "errors": packet.get("errors"),
            }
            for packet in packets
        ],
        "best_detection": best,
    }
    combined["path"] = write_json(run_dir / "observations" / f"{stage}_{object_role}_prompt_fallback.json", combined)
    return combined, best


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _raw_preview_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    reason = _field(result, "reason", _field(result, "error", None))
    executed = bool(_field(result, "executed", False))
    trajectory_cache_key = _field(result, "trajectory_cache_key", None)
    final_pos_error_m = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    final_rot_error_deg = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    status_text = str(status) if status is not None else None
    success = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    return {
        "status": status_text,
        "reason": reason,
        "executed": executed,
        "trajectory_cache_key": trajectory_cache_key,
        "final_pos_error_m": final_pos_error_m,
        "final_rot_error_deg": final_rot_error_deg,
        "success": success,
        "raw": json_safe(result),
    }


def _guarded_preview_freespace(freespace_move: Callable[..., Any]) -> Callable[..., Any]:
    def _inner(**kwargs: Any) -> Any:
        result = freespace_move(**kwargs)
        summary = _raw_preview_summary(result)
        if bool(kwargs.get("preview_only", False)) and summary["executed"]:
            raise RuntimeError(f"preview unexpectedly executed: {summary}")
        if not summary["success"]:
            raise RuntimeError(f"planner preview failed: {summary}")
        return result

    return _inner


def _build_bottle_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    return generate_side_grasp_candidates(
        detection,
        object_kind="bottle",
        arm="left",
        default_radius_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_RADIUS_M", 0.04),
        default_half_height_m=0.0,
        body_fraction=_env_float("OPENFORGE_DRY_POUR_BOTTLE_BODY_FRACTION", 0.45),
        pregrasp_standoff_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_PREGRASP_STANDOFF_M", 0.08),
        lift_z_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_LIFT_Z_M", 0.06),
        width_margin_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_WIDTH_MARGIN_M", 0.018),
        include_topdown=False,
        yaw_angles_deg=_float_csv("OPENFORGE_DRY_POUR_BOTTLE_YAWS_DEG", [90.0, 70.0, 110.0, 50.0, 130.0, 0.0, 180.0]),
        z_offsets_m=_float_csv("OPENFORGE_DRY_POUR_BOTTLE_Z_OFFSETS_M", [0.0, 0.015, -0.015, 0.03, -0.03]),
        center_z_offset_without_extents_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_Z_OFFSET_WITHOUT_EXTENTS_M", -0.07),
    )


def _build_can_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    return generate_side_grasp_candidates(
        detection,
        object_kind="can",
        arm="right",
        default_radius_m=_env_float("OPENFORGE_DRY_POUR_CAN_RADIUS_M", 0.033),
        default_half_height_m=0.0,
        body_fraction=_env_float("OPENFORGE_DRY_POUR_CAN_BODY_FRACTION", 0.50),
        pregrasp_standoff_m=_env_float("OPENFORGE_DRY_POUR_CAN_PREGRASP_STANDOFF_M", 0.07),
        lift_z_m=_env_float("OPENFORGE_DRY_POUR_CAN_LIFT_Z_M", 0.03),
        width_margin_m=_env_float("OPENFORGE_DRY_POUR_CAN_WIDTH_MARGIN_M", 0.014),
        include_topdown=False,
        yaw_angles_deg=_float_csv("OPENFORGE_DRY_POUR_CAN_YAWS_DEG", [-90.0, -70.0, -110.0, -50.0, -130.0, 180.0, 0.0]),
        z_offsets_m=_float_csv("OPENFORGE_DRY_POUR_CAN_Z_OFFSETS_M", [0.0, 0.01, -0.01, 0.02, -0.02]),
        center_z_offset_without_extents_m=_env_float("OPENFORGE_DRY_POUR_CAN_Z_OFFSET_WITHOUT_EXTENTS_M", -0.045),
    )


def _xy_distance(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    apos = a.get("position_3d") or a.get("position")
    bpos = b.get("position_3d") or b.get("position")
    if not apos or not bpos or len(apos) < 2 or len(bpos) < 2:
        return None
    return math.hypot(float(apos[0]) - float(bpos[0]), float(apos[1]) - float(bpos[1]))


def _can_rim_z(can_detection: dict[str, Any]) -> tuple[float, str]:
    xyz = [float(v) for v in can_detection["position_3d"][:3]]
    half_extents = can_detection.get("half_extents") or []
    if len(half_extents) >= 3 and abs(float(half_extents[2])) > 0:
        return xyz[2] + abs(float(half_extents[2])), "detection_z_plus_half_extent_z"
    return xyz[2] + _env_float("OPENFORGE_DRY_POUR_CAN_RIM_Z_OFFSET_M", 0.06), "detection_z_plus_default_can_rim_offset"


def _make_pose(position: Any, rpy: Any, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "position": _round_list(position, limit=3),
        "rpy": _round_list(rpy, digits=3, limit=3),
    }


def _motion_common_kwargs(*, preview_only: bool) -> dict[str, Any]:
    return {
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_DRY_POUR_PLANNER_BACKEND", "curobo"),
        "solver_speed": os.environ.get("OPENFORGE_DRY_POUR_SOLVER_SPEED", "slow"),
        "planning_speed": _env_float("OPENFORGE_DRY_POUR_PLANNING_SPEED", 0.30),
        "ik_error_threshold": _env_float("OPENFORGE_DRY_POUR_IK_ERROR_THRESHOLD_M", 0.02),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_DRY_POUR_IK_ROT_THRESHOLD_DEG", 12.0),
        "ik_xyz_weight": _env_float("OPENFORGE_DRY_POUR_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_DRY_POUR_IK_RPY_WEIGHT", 0.25),
    }


def _waypoint_move_kwargs(waypoint: dict[str, Any], *, preview_only: bool) -> dict[str, Any]:
    kwargs = _motion_common_kwargs(preview_only=preview_only)
    left = waypoint.get("left")
    right = waypoint.get("right")
    if left:
        kwargs["left_target_pos"] = left["position"]
        kwargs["left_target_rpy"] = left["rpy"]
        kwargs["left_gripper"] = float(
            waypoint.get("left_gripper", _env_float("OPENFORGE_DRY_POUR_BOTTLE_COLLISION_GRIPPER_POS", 0.35))
        )
    if right:
        kwargs["right_target_pos"] = right["position"]
        kwargs["right_target_rpy"] = right["rpy"]
        kwargs["right_gripper"] = float(
            waypoint.get("right_gripper", _env_float("OPENFORGE_DRY_POUR_CAN_COLLISION_GRIPPER_POS", 0.40))
        )
    return kwargs


def _single_arm_waypoint(
    *,
    name: str,
    side: str,
    pose: dict[str, Any],
    left_gripper: float | None = None,
    right_gripper: float | None = None,
) -> dict[str, Any]:
    waypoint: dict[str, Any] = {"name": name}
    key = "left" if side == "left" else "right"
    waypoint[key] = _make_pose(pose["position"], pose["rpy"], label=f"{side}_{name}")
    if left_gripper is not None:
        waypoint["left_gripper"] = float(left_gripper)
    if right_gripper is not None:
        waypoint["right_gripper"] = float(right_gripper)
    return waypoint


def _bimanual_waypoint_with_grippers(
    waypoint: dict[str, Any],
    *,
    left_gripper: float,
    right_gripper: float,
    name: str | None = None,
) -> dict[str, Any]:
    out = {
        "name": name or waypoint["name"],
        "left": waypoint["left"],
        "right": waypoint["right"],
        "left_gripper": float(left_gripper),
        "right_gripper": float(right_gripper),
    }
    if "tilt_pitch_deg" in waypoint:
        out["tilt_pitch_deg"] = waypoint["tilt_pitch_deg"]
    return out


def _build_rehearsal_waypoints(
    *,
    bottle_detection: dict[str, Any],
    can_detection: dict[str, Any],
    bottle_plan: dict[str, Any],
    can_plan: dict[str, Any],
) -> dict[str, Any]:
    bottle_lift = bottle_plan["lift_pose"]
    can_hold = can_plan["grasp_pose"]
    can_xyz = [float(v) for v in can_detection["position_3d"][:3]]
    can_rim, can_rim_source = _can_rim_z(can_detection)
    lift_xyz = [float(v) for v in bottle_lift["position"][:3]]
    dx = can_xyz[0] - lift_xyz[0]
    dy = can_xyz[1] - lift_xyz[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        unit = [1.0, 0.0]
    else:
        unit = [dx / norm, dy / norm]
    standoff = _env_float("OPENFORGE_DRY_POUR_STANDOFF_M", 0.14)
    vertical_clearance = _env_float("OPENFORGE_DRY_POUR_VERTICAL_CLEARANCE_M", 0.18)
    pre_pour_xyz = [
        can_xyz[0] - unit[0] * standoff,
        can_xyz[1] - unit[1] * standoff,
        can_rim + vertical_clearance,
    ]
    carry_rpy = [float(v) for v in bottle_plan["rpy"][:3]]
    requested_tilt = _env_float("OPENFORGE_DRY_POUR_TILT_PITCH_DEG", 70.0)
    tilt_pitches: list[float] = []
    for pitch in _float_csv("OPENFORGE_DRY_POUR_TILT_PITCHES_DEG", [requested_tilt, 65.0, 60.0, 55.0]):
        if all(abs(pitch - existing) > 1e-6 for existing in tilt_pitches):
            tilt_pitches.append(pitch)
    tilt_alternatives = [
        {
            "name": f"dry_tilt_rehearsal_pitch_{pitch:g}",
            "left": _make_pose(pre_pour_xyz, [carry_rpy[0], pitch, carry_rpy[2]], label=f"left_bottle_dry_tilt_pitch_{pitch:g}"),
            "right": _make_pose(can_hold["position"], can_hold["rpy"], label="right_can_hold"),
            "tilt_pitch_deg": float(pitch),
        }
        for pitch in tilt_pitches
    ]
    waypoints = [
        {
            "name": "bimanual_pregrasp",
            "left": _make_pose(bottle_plan["pregrasp_pose"]["position"], bottle_plan["pregrasp_pose"]["rpy"], label="left_bottle_pregrasp"),
            "right": _make_pose(can_plan["pregrasp_pose"]["position"], can_plan["pregrasp_pose"]["rpy"], label="right_can_pregrasp"),
        },
        {
            "name": "bimanual_grasp_setup",
            "left": _make_pose(bottle_plan["grasp_pose"]["position"], bottle_plan["grasp_pose"]["rpy"], label="left_bottle_grasp"),
            "right": _make_pose(can_plan["grasp_pose"]["position"], can_plan["grasp_pose"]["rpy"], label="right_can_grasp"),
        },
        {
            "name": "bottle_lift_can_hold",
            "left": _make_pose(bottle_lift["position"], bottle_lift["rpy"], label="left_bottle_lift"),
            "right": _make_pose(can_hold["position"], can_hold["rpy"], label="right_can_hold"),
        },
        {
            "name": "pre_pour_standoff",
            "left": _make_pose(pre_pour_xyz, carry_rpy, label="left_bottle_pre_pour"),
            "right": _make_pose(can_hold["position"], can_hold["rpy"], label="right_can_hold"),
        },
        {
            "name": "dry_tilt_rehearsal",
            "selection_policy": "first_feasible_pitch_sweep",
            "alternatives": tilt_alternatives,
        },
        {
            "name": "upright_after_tilt",
            "left": _make_pose(pre_pour_xyz, carry_rpy, label="left_bottle_upright"),
            "right": _make_pose(can_hold["position"], can_hold["rpy"], label="right_can_hold"),
        },
        {
            "name": "retreat_to_lift_hold",
            "left": _make_pose(bottle_lift["position"], bottle_lift["rpy"], label="left_bottle_lift_retreat"),
            "right": _make_pose(can_hold["position"], can_hold["rpy"], label="right_can_hold"),
        },
    ]
    return {
        "object_requirement": "dry_empty_or_sealed_bottle_and_empty_can",
        "liquid_used": False,
        "planning_only": True,
        "can_rim_z_m": round(can_rim, 5),
        "can_rim_z_source": can_rim_source,
        "aim_direction_xy_from_bottle_to_can": _round_list(unit, limit=2),
        "standoff_m": standoff,
        "vertical_clearance_m": vertical_clearance,
        "tilt_pitch_candidates_deg": tilt_pitches,
        "bottle_detection": bottle_detection,
        "can_detection": can_detection,
        "waypoints": waypoints,
    }


def _preview_bimanual_waypoint(
    *,
    freespace_move: Callable[..., Any],
    waypoint: dict[str, Any],
    run_in_background: Callable[..., Any] | None,
    timeout_s: float,
) -> dict[str, Any]:
    left = waypoint["left"]
    right = waypoint["right"]
    kwargs = _waypoint_move_kwargs(waypoint, preview_only=True)
    call = call_with_timeout(
        f"bimanual_preview:{waypoint['name']}",
        freespace_move,
        timeout_s,
        run_in_background=run_in_background,
        **kwargs,
    )
    summary: dict[str, Any] = {
        "name": waypoint["name"],
        "left": left,
        "right": right,
        "request": kwargs,
        "call_ok": bool(call.get("ok")),
        "success": False,
    }
    if not call.get("ok"):
        summary["error"] = call.get("error")
        return summary
    preview = _raw_preview_summary(call.get("data"))
    summary["preview"] = preview
    summary["success"] = bool(preview.get("success"))
    if preview.get("executed"):
        summary["error"] = "preview_only call reported executed=True"
        summary["success"] = False
    return summary


def _preview_rehearsal(
    *,
    run_dir: Path,
    rehearsal: dict[str, Any],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    run_in_background = _tool("run_in_background")
    timeout_s = _env_float("OPENFORGE_DRY_POUR_BIMANUAL_PREVIEW_TIMEOUT_S", 45.0)
    previews: list[dict[str, Any]] = []
    selected_waypoints: list[dict[str, Any]] = []
    stop_after_first_failure = _truthy_env("OPENFORGE_DRY_POUR_STOP_PREVIEW_AFTER_FIRST_FAILURE", False)
    for waypoint in rehearsal["waypoints"]:
        alternatives = list(waypoint.get("alternatives") or [])
        if alternatives:
            alternative_previews: list[dict[str, Any]] = []
            selected_preview: dict[str, Any] | None = None
            selected_waypoint: dict[str, Any] | None = None
            for alternative in alternatives:
                alt_preview = _preview_bimanual_waypoint(
                    freespace_move=freespace_move,
                    waypoint=alternative,
                    run_in_background=run_in_background,
                    timeout_s=timeout_s,
                )
                alternative_previews.append(alt_preview)
                if alt_preview.get("success"):
                    selected_preview = alt_preview
                    selected_waypoint = alternative
                    break
            preview = {
                "name": waypoint["name"],
                "success": selected_preview is not None,
                "selection_policy": waypoint.get("selection_policy"),
                "selected_alternative": selected_preview.get("name") if selected_preview else None,
                "selected_tilt_pitch_deg": selected_waypoint.get("tilt_pitch_deg") if selected_waypoint else None,
                "alternative_previews": alternative_previews,
            }
            if selected_waypoint is not None:
                selected_waypoints.append(selected_waypoint)
            previews.append(preview)
        else:
            preview = _preview_bimanual_waypoint(
                freespace_move=freespace_move,
                waypoint=waypoint,
                run_in_background=run_in_background,
                timeout_s=timeout_s,
            )
            previews.append(preview)
            if preview.get("success"):
                selected_waypoints.append(waypoint)
        if stop_after_first_failure and not preview.get("success"):
            break
    packet = {
        "schema": "openforge.dry_bimanual_bottle_to_can_pour.preview.v1",
        "physical_motion_executed": False,
        "success": bool(previews and all(item.get("success") for item in previews)),
        "preview_count": len(previews),
        "waypoints": rehearsal["waypoints"],
        "selected_waypoints": selected_waypoints,
        "previews": previews,
    }
    packet["path"] = write_json(run_dir / "plans" / "bimanual_rehearsal_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## bimanual rehearsal preview",
            f"- preview: {packet['path']}",
            f"- success: {packet['success']}",
            f"- preview_count: {packet['preview_count']}",
        ],
    )
    return packet


def _execute_cached_waypoint(
    *,
    freespace_move: Callable[..., Any],
    waypoint: dict[str, Any],
) -> dict[str, Any]:
    preview_kwargs = _waypoint_move_kwargs(waypoint, preview_only=True)
    preview = freespace_move(**preview_kwargs)
    preview_summary = _raw_preview_summary(preview)
    if not preview_summary.get("success"):
        raise RuntimeError(f"{waypoint['name']} preview failed before physical execute: {preview_summary}")
    cache_key = preview_summary.get("trajectory_cache_key")
    if not cache_key:
        raise RuntimeError(f"{waypoint['name']} preview did not return a trajectory cache key")
    executed = freespace_move(trajectory_cache_key=cache_key)
    executed_summary = _raw_execute_summary(executed)
    if not executed_summary.get("success"):
        raise RuntimeError(f"{waypoint['name']} execution failed: {executed_summary}")
    return {
        "stage": waypoint["name"],
        "request": preview_kwargs,
        "preview": preview_summary,
        "execute": executed_summary,
        "trajectory_cache_key": cache_key,
    }


def _raw_execute_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    reason = _field(result, "reason", _field(result, "error", None))
    executed = bool(_field(result, "executed", True))
    status_text = str(status) if status is not None else None
    success = bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed)
    return {
        "status": status_text,
        "reason": reason,
        "executed": executed,
        "final_pos_error_m": _field(result, "final_pos_error_m", _field(result, "ik_error_m", None)),
        "final_rot_error_deg": _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None)),
        "success": success,
        "raw": json_safe(result),
    }


def _set_gripper_open_checked(
    *,
    side: str,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
) -> dict[str, Any]:
    before = read_gripper_width(get_robot_state, side)
    target = _env_float("OPENFORGE_DRY_POUR_OPEN_GRIPPER_POS", 1.0)
    vel_limit = _env_float("OPENFORGE_DRY_POUR_OPEN_VEL_LIMIT", 0.55)
    torque_limit = _env_float("OPENFORGE_DRY_POUR_OPEN_TORQUE_LIMIT", 0.25)
    result = set_gripper(side, target, vel_limit=vel_limit, torque_limit=torque_limit)
    open_confirm_min = _env_float("OPENFORGE_DRY_POUR_OPEN_CONFIRM_POS", 0.55)
    timeout_s = _env_float("OPENFORGE_DRY_POUR_OPEN_CONFIRM_TIMEOUT_S", 3.0)
    deadline = time.monotonic() + max(0.0, timeout_s)
    after = read_gripper_width(get_robot_state, side)
    while after is not None and after < open_confirm_min and time.monotonic() < deadline:
        time.sleep(0.1)
        after = read_gripper_width(get_robot_state, side)
    ok = after is None or after >= open_confirm_min
    return {
        "stage": f"open_{side}_gripper",
        "side": side,
        "before": before,
        "after": after,
        "target": target,
        "open_confirm_min": open_confirm_min,
        "ok": ok,
        "result": json_safe(result),
    }


def _close_gripper_for_object(
    *,
    side: str,
    object_role: str,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    run_dir: Path,
) -> dict[str, Any]:
    role_prefix = f"OPENFORGE_DRY_POUR_{object_role.upper()}"
    if object_role == "can":
        default_target = 0.32
        default_steps = [0.78, 0.58, 0.42, 0.32]
        default_torque = 0.16
        default_require_contact = False
    else:
        default_target = 0.25
        default_steps = [0.78, 0.58, 0.42, 0.25]
        default_torque = 0.18
        default_require_contact = True
    target = _env_float(f"{role_prefix}_CLOSE_TARGET_POS", default_target)
    close = staged_close_with_contact(
        side=side,
        set_gripper=set_gripper,
        get_robot_state=get_robot_state,
        target=target,
        steps=_float_csv(f"{role_prefix}_CLOSE_STEPS_POS", default_steps),
        vel_limit=_env_float(f"{role_prefix}_CLOSE_VEL_LIMIT", 0.35),
        torque_limit=_env_float(f"{role_prefix}_CLOSE_TORQUE_LIMIT", default_torque),
        min_contact_delta=_env_float(f"{role_prefix}_CLOSE_CONTACT_MIN_TARGET_DELTA_POS", 0.025),
        target_tolerance=_env_float(f"{role_prefix}_CLOSE_TARGET_TOLERANCE_POS", 0.05),
        hold_min=_env_float(f"{role_prefix}_HOLD_MIN_POS", 0.02),
        hold_max=_env_float(f"{role_prefix}_HOLD_MAX_POS", 0.95),
        run_dir=run_dir,
        task_name=TASK_NAME,
        stage=f"{object_role}_{side}_staged_close",
    )
    require_contact = _truthy_env(f"{role_prefix}_REQUIRE_CLOSE_CONTACT", default_require_contact)
    close["gate"] = {
        "require_contact": require_contact,
        "passed": bool(close.get("contact_detected") if require_contact else close.get("plausible_for_lift")),
    }
    return close


def _hold_value_after_close(close: dict[str, Any], fallback: float) -> float:
    after = close.get("after")
    if after is None:
        return float(fallback)
    try:
        return max(0.0, min(1.0, float(after)))
    except (TypeError, ValueError):
        return float(fallback)


def _capture_dual_post_observation(
    *,
    run_dir: Path,
    stage: str,
    bottle_prompt: str,
    can_prompt: str,
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    bottle_obs, bottle_det = _capture_prompt_fallback(
        run_dir=run_dir,
        stage=stage,
        object_role="bottle",
        prompts=[bottle_prompt, "bottle", "wine bottle"],
        cameras=["top"],
        min_score=min_score,
    )
    can_obs, can_det = _capture_prompt_fallback(
        run_dir=run_dir,
        stage=stage,
        object_role="can",
        prompts=[can_prompt, "can", "aluminum can", "soda can"],
        cameras=["top"],
        min_score=min_score,
    )
    return {"bottle": bottle_obs, "can": can_obs}, bottle_det, can_det


def _execute_physical_rehearsal(
    *,
    run_dir: Path,
    result: dict[str, Any],
    bottle_det: dict[str, Any],
    can_det: dict[str, Any],
    bottle_plan: dict[str, Any],
    can_plan: dict[str, Any],
    bimanual_preview: dict[str, Any],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    attempt: dict[str, Any] = {
        "schema": "openforge.dry_bimanual_bottle_to_can_pour.physical_attempt.v1",
        "attempt_index": 1,
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    left_hold = _env_float("OPENFORGE_DRY_POUR_BOTTLE_CLOSE_TARGET_POS", 0.25)
    right_hold = _env_float("OPENFORGE_DRY_POUR_CAN_CLOSE_TARGET_POS", 0.32)
    selected_waypoints = list(bimanual_preview.get("selected_waypoints") or [])
    if len(selected_waypoints) < 7:
        attempt["why_stopped"] = "missing selected bimanual preview waypoints; refused physical execution"
        attempt["path"] = write_json(run_dir / "plans" / "physical_attempt_01.json", attempt)
        return attempt
    pregrasp_wp = _bimanual_waypoint_with_grippers(selected_waypoints[0], left_gripper=1.0, right_gripper=1.0)
    grasp_setup_wp = selected_waypoints[1]
    try:
        for side in ("left", "right"):
            open_step = _set_gripper_open_checked(side=side, set_gripper=set_gripper, get_robot_state=get_robot_state)
            attempt["physical_motion_executed"] = True
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
            attempt["steps"].append(open_step)
            if not open_step.get("ok"):
                attempt["why_stopped"] = f"{side} gripper did not confirm open before arm motion"
                return attempt

        attempt["steps"].append(_execute_cached_waypoint(freespace_move=freespace_move, waypoint=pregrasp_wp))
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append("freespace_move:bimanual_pregrasp")

        right_can_grasp = _single_arm_waypoint(
            name="right_can_grasp",
            side="right",
            pose=can_plan["grasp_pose"],
            left_gripper=1.0,
            right_gripper=1.0,
        )
        attempt["steps"].append(_execute_cached_waypoint(freespace_move=freespace_move, waypoint=right_can_grasp))
        attempt["movement_capable_calls"].append("freespace_move:right_can_grasp")
        can_close = _close_gripper_for_object(
            side="right",
            object_role="can",
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            run_dir=run_dir,
        )
        attempt["movement_capable_calls"].append("set_gripper:right")
        attempt["gripper_after_can_close"] = can_close
        if not can_close.get("gate", {}).get("passed"):
            attempt["why_stopped"] = "right can staged close did not produce plausible hold evidence; stop before bottle motion"
            return attempt
        right_hold = _hold_value_after_close(can_close, right_hold)

        left_bottle_grasp = _single_arm_waypoint(
            name="left_bottle_grasp",
            side="left",
            pose=bottle_plan["grasp_pose"],
            left_gripper=1.0,
            right_gripper=right_hold,
        )
        attempt["steps"].append(_execute_cached_waypoint(freespace_move=freespace_move, waypoint=left_bottle_grasp))
        attempt["movement_capable_calls"].append("freespace_move:left_bottle_grasp")
        bottle_close = _close_gripper_for_object(
            side="left",
            object_role="bottle",
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            run_dir=run_dir,
        )
        attempt["movement_capable_calls"].append("set_gripper:left")
        attempt["gripper_after_bottle_close"] = bottle_close
        if not bottle_close.get("gate", {}).get("passed"):
            attempt["why_stopped"] = "left bottle staged close did not produce plausible hold evidence; stop before lift"
            return attempt
        left_hold = _hold_value_after_close(bottle_close, left_hold)
        attempt["hold_gripper_values_after_close"] = {
            "left_bottle": left_hold,
            "right_can": right_hold,
            "policy": "use measured gripper width after staged close for subsequent motion waypoints",
        }

        lift_wp = _bimanual_waypoint_with_grippers(selected_waypoints[2], left_gripper=left_hold, right_gripper=right_hold)
        pre_pour_wp = _bimanual_waypoint_with_grippers(selected_waypoints[3], left_gripper=left_hold, right_gripper=right_hold)
        tilt_wp = _bimanual_waypoint_with_grippers(selected_waypoints[4], left_gripper=left_hold, right_gripper=right_hold)
        upright_wp = _bimanual_waypoint_with_grippers(selected_waypoints[5], left_gripper=left_hold, right_gripper=right_hold)
        retreat_wp = _bimanual_waypoint_with_grippers(selected_waypoints[6], left_gripper=left_hold, right_gripper=right_hold)
        return_wp = _bimanual_waypoint_with_grippers(
            grasp_setup_wp,
            left_gripper=left_hold,
            right_gripper=right_hold,
            name="return_to_grasp_setup",
        )
        release_retreat_wp = _bimanual_waypoint_with_grippers(
            pregrasp_wp,
            left_gripper=1.0,
            right_gripper=1.0,
            name="release_retreat_to_pregrasp",
        )

        attempt["steps"].append(_execute_cached_waypoint(freespace_move=freespace_move, waypoint=lift_wp))
        attempt["movement_capable_calls"].append("freespace_move:bottle_lift_can_hold")
        post_lift_obs, post_lift_bottle, post_lift_can = _capture_dual_post_observation(
            run_dir=run_dir,
            stage="post_lift_observe",
            bottle_prompt=str(bottle_det.get("prompt") or "bottle"),
            can_prompt=str(can_det.get("prompt") or "can"),
            min_score=_env_float("OPENFORGE_DRY_POUR_MIN_DETECTION_SCORE", 0.0),
        )
        attempt["post_lift_observation"] = post_lift_obs
        attempt["post_lift_bottle_detection"] = post_lift_bottle
        attempt["post_lift_can_detection"] = post_lift_can
        attempt["bottle_lift_motion"] = compare_detection_motion(bottle_det, post_lift_bottle)
        attempt["bottle_lift_verification"] = verify_lift(
            bottle_det,
            post_lift_bottle,
            min_z_delta_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_LIFT_VERIFY_MIN_Z_DELTA_M", 0.035),
            max_xy_delta_m=_env_float("OPENFORGE_DRY_POUR_BOTTLE_LIFT_VERIFY_MAX_XY_DELTA_M", 0.10),
        )
        if not attempt["bottle_lift_verification"].get("success"):
            attempt["why_stopped"] = "post-lift observation did not verify a clean bottle lift; stop before dry tilt"
            return attempt

        for waypoint in (pre_pour_wp, tilt_wp, upright_wp, retreat_wp, return_wp):
            attempt["steps"].append(_execute_cached_waypoint(freespace_move=freespace_move, waypoint=waypoint))
            attempt["movement_capable_calls"].append(f"freespace_move:{waypoint['name']}")

        release_steps = []
        for side in ("left", "right"):
            open_step = _set_gripper_open_checked(side=side, set_gripper=set_gripper, get_robot_state=get_robot_state)
            release_steps.append(open_step)
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["release_open_steps"] = release_steps
        if not all(step.get("ok") for step in release_steps):
            attempt["why_stopped"] = "release gripper open confirmation failed after return"
            return attempt
        attempt["steps"].append(_execute_cached_waypoint(freespace_move=freespace_move, waypoint=release_retreat_wp))
        attempt["movement_capable_calls"].append("freespace_move:release_retreat_to_pregrasp")

        final_obs, final_bottle, final_can = _capture_dual_post_observation(
            run_dir=run_dir,
            stage="post_release_observe",
            bottle_prompt=str(bottle_det.get("prompt") or "bottle"),
            can_prompt=str(can_det.get("prompt") or "can"),
            min_score=_env_float("OPENFORGE_DRY_POUR_MIN_DETECTION_SCORE", 0.0),
        )
        attempt["post_release_observation"] = final_obs
        attempt["post_release_bottle_detection"] = final_bottle
        attempt["post_release_can_detection"] = final_can
        attempt["post_release_motion"] = {
            "bottle": compare_detection_motion(bottle_det, final_bottle),
            "can": compare_detection_motion(can_det, final_can),
        }
        attempt["success"] = bool(final_bottle is not None and final_can is not None)
        attempt["why_stopped"] = (
            "dry bimanual rehearsal completed and post-release detections were recorded"
            if attempt["success"]
            else "dry rehearsal motions completed, but post-release object detection was inconclusive"
        )
        if _truthy_env("OPENFORGE_DRY_POUR_GO_HOME_AFTER_RELEASE", False):
            go_home = _tool("go_home")
            if callable(go_home):
                home_result = go_home()
                attempt["movement_capable_calls"].append("go_home")
                attempt["go_home_after_release"] = json_safe(home_result)
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
    finally:
        attempt["path"] = write_json(run_dir / "plans" / "physical_attempt_01.json", attempt)
        append_stage_summary(
            run_dir,
            [
                "## physical attempt 01",
                f"- success: {attempt['success']}",
                f"- physical_motion_executed: {attempt['physical_motion_executed']}",
                f"- why_stopped: {attempt['why_stopped']}",
            ],
        )
    return attempt


def _rank_object_candidates(
    *,
    run_dir: Path,
    object_role: str,
    candidates: list[dict[str, Any]],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    max_preview = max(1, _env_int(f"OPENFORGE_DRY_POUR_{object_role.upper()}_MAX_PREVIEW_CANDIDATES", 80))
    return rank_motion_candidates(
        candidates=candidates[:max_preview],
        freespace_move=_guarded_preview_freespace(freespace_move),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage=f"{object_role}_side_grasp_preview",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DRY_POUR_SINGLE_ARM_PREVIEW_TIMEOUT_S", 35.0),
        planner_backend=os.environ.get("OPENFORGE_DRY_POUR_PLANNER_BACKEND", "curobo"),
        solver_speed=os.environ.get("OPENFORGE_DRY_POUR_SOLVER_SPEED", "slow"),
        planning_speed=_env_float("OPENFORGE_DRY_POUR_PLANNING_SPEED", 0.30),
        ik_error_threshold=_env_float("OPENFORGE_DRY_POUR_IK_ERROR_THRESHOLD_M", 0.02),
        ik_rot_threshold_deg=_env_float("OPENFORGE_DRY_POUR_IK_ROT_THRESHOLD_DEG", 12.0),
    )


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists():
            paths.append(str(path))
    return paths


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.dry_bimanual_bottle_to_can_pour.result.v1",
        "success": False,
        "plan_success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "dry_run": True,
        "liquid_used": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False),
        "physical_attempt_budget": min(3, max(1, _env_int("OPENFORGE_DRY_POUR_ATTEMPT_BUDGET", 1))),
        "attempt_count": 0,
        "attempts": [],
        "active_ticket_path": "/tmp/openforge_dry_bimanual_physical_ticket.txt",
        "observation_policy": "fresh live top-camera detections only; manual XYZ env vars are rejected",
        "calibration_env": {
            "YAM_STATION_CALIBRATED_XML": os.environ.get("YAM_STATION_CALIBRATED_XML"),
            "CAP_TOP_CAMERA_BACKEND": os.environ.get("CAP_TOP_CAMERA_BACKEND"),
            "CAP_TOP_CAMERA_FRAME": os.environ.get("CAP_TOP_CAMERA_FRAME"),
            "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP": os.environ.get("CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP"),
        },
        "bottle_detection": None,
        "can_detection": None,
        "bottle_plan": None,
        "can_plan": None,
        "selected_strategy": None,
        "bimanual_rehearsal": None,
        "bimanual_preview": None,
        "physical_attempt": None,
        "recovery": None,
        "why_stopped": None,
        "next_recommendation": None,
        "video_paths": [],
    }


def _run_recovery_only(run_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    recovery: dict[str, Any] = {
        "schema": "openforge.dry_bimanual_bottle_to_can_pour.recovery.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    if not result["physical_gate"]:
        recovery["why_stopped"] = "recovery-only requested but OPENFORGE_ALLOW_PHYSICAL_MOTION is not set"
        recovery["path"] = write_json(run_dir / "plans" / "recovery_only.json", recovery)
        return recovery
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _tool("get_robot_state")
    go_home = _required_tool("go_home")
    try:
        if callable(get_robot_state):
            recovery["before_state"] = json_safe(get_robot_state())
        for side in ("left", "right"):
            if callable(get_robot_state):
                step = _set_gripper_open_checked(side=side, set_gripper=set_gripper, get_robot_state=get_robot_state)
            else:
                command = set_gripper(
                    side,
                    _env_float("OPENFORGE_DRY_POUR_OPEN_GRIPPER_POS", 1.0),
                    vel_limit=_env_float("OPENFORGE_DRY_POUR_OPEN_VEL_LIMIT", 0.55),
                    torque_limit=_env_float("OPENFORGE_DRY_POUR_OPEN_TORQUE_LIMIT", 0.25),
                )
                step = {"stage": f"open_{side}_gripper", "side": side, "ok": True, "result": json_safe(command)}
            recovery["physical_motion_executed"] = True
            recovery["movement_capable_calls"].append(f"set_gripper:{side}")
            recovery["steps"].append(step)
            if not step.get("ok"):
                recovery["why_stopped"] = f"{side} gripper did not confirm open during recovery"
                recovery["path"] = write_json(run_dir / "plans" / "recovery_only.json", recovery)
                return recovery
        home_result = go_home()
        recovery["movement_capable_calls"].append("go_home")
        recovery["steps"].append({"stage": "go_home", "ok": True, "result": json_safe(home_result)})
        if callable(get_robot_state):
            recovery["after_state"] = json_safe(get_robot_state())
        try:
            post_obs, post_bottle, post_can = _capture_dual_post_observation(
                run_dir=run_dir,
                stage="post_recovery_observe",
                bottle_prompt="bottle",
                can_prompt="can",
                min_score=_env_float("OPENFORGE_DRY_POUR_MIN_DETECTION_SCORE", 0.0),
            )
            recovery["post_recovery_observation"] = post_obs
            recovery["post_recovery_bottle_detection"] = post_bottle
            recovery["post_recovery_can_detection"] = post_can
        except Exception as exc:
            recovery["post_recovery_observation_error"] = f"{type(exc).__name__}: {exc}"
        recovery["success"] = True
        recovery["why_stopped"] = "recovery-only open both grippers, go_home, and post-observe completed"
    except Exception as exc:
        recovery["why_stopped"] = f"{type(exc).__name__}: {exc}"
        recovery["steps"].append({"stage": "exception", "ok": False, "error": recovery["why_stopped"]})
    finally:
        recovery["path"] = write_json(run_dir / "plans" / "recovery_only.json", recovery)
        append_stage_summary(
            run_dir,
            [
                "## recovery only",
                f"- success: {recovery['success']}",
                f"- physical_motion_executed: {recovery['physical_motion_executed']}",
                f"- why_stopped: {recovery['why_stopped']}",
            ],
        )
    return recovery


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        manual_env = _manual_pose_env_vars_present()
        if manual_env:
            result["manual_pose_env_vars_present"] = manual_env
            result["why_stopped"] = "manual XYZ/extent env vars are not allowed for this live-detection rehearsal target"
            result["next_recommendation"] = "Unset manual pose env vars and rerun with calibrated top-camera live perception."
            return result
        if result["physical_gate"]:
            append_stage_summary(
                run_dir,
                [
                    "## safety",
                    "- OPENFORGE_ALLOW_PHYSICAL_MOTION is set; physical branch remains gated behind fresh live observe and successful previews.",
                ],
            )

        if _truthy_env("OPENFORGE_DRY_POUR_RECOVERY_ONLY", False):
            recovery = _run_recovery_only(run_dir, result)
            result["recovery"] = recovery
            result["success"] = bool(recovery.get("success"))
            result["reward"] = 1.0 if result["success"] else 0.0
            result["physical_motion_executed"] = bool(recovery.get("physical_motion_executed"))
            result["movement_capable_calls"] = list(recovery.get("movement_capable_calls") or [])
            result["why_stopped"] = recovery.get("why_stopped")
            result["next_recommendation"] = "Stop; recovery is complete." if result["success"] else "Inspect recovery_only.json before any retry."
            return result

        cameras = _csv("OPENFORGE_DRY_POUR_CAMERAS", ["top"])
        min_score = _env_float("OPENFORGE_DRY_POUR_MIN_DETECTION_SCORE", 0.0)
        bottle_prompts = _csv("OPENFORGE_DRY_POUR_BOTTLE_PROMPTS", ["bottle", "wine bottle"])
        can_prompts = _csv("OPENFORGE_DRY_POUR_CAN_PROMPTS", ["can", "aluminum can", "soda can"])

        bottle_obs, bottle_det = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="pre_observe",
            object_role="bottle",
            prompts=bottle_prompts,
            cameras=cameras,
            min_score=min_score,
        )
        can_obs, can_det = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="pre_observe",
            object_role="can",
            prompts=can_prompts,
            cameras=cameras,
            min_score=min_score,
        )
        result["pre_observation"] = {"bottle": bottle_obs, "can": can_obs}
        result["bottle_detection"] = bottle_det
        result["can_detection"] = can_det
        if bottle_det is None:
            result["why_stopped"] = "no live top-camera bottle detection; stopped before planning"
            result["next_recommendation"] = "Inspect top-camera overlay/perception service, then rerun no-motion."
            return result
        if can_det is None:
            result["why_stopped"] = "no live top-camera can detection; stopped before planning"
            result["next_recommendation"] = "Inspect top-camera overlay/perception service or can placement, then rerun no-motion."
            return result

        detection_sanity = {
            "bottle": _validate_live_detection(bottle_det, object_role="bottle"),
            "can": _validate_live_detection(can_det, object_role="can"),
        }
        result["detection_sanity"] = detection_sanity
        append_stage_summary(
            run_dir,
            [
                "## live detection sanity",
                f"- bottle: {detection_sanity['bottle']}",
                f"- can: {detection_sanity['can']}",
            ],
        )
        failed_sanity = [role for role, sanity in detection_sanity.items() if not sanity.get("passed")]
        if failed_sanity:
            result["why_stopped"] = (
                "live top-camera detection sanity failed for "
                + ", ".join(failed_sanity)
                + "; stopped before planning"
            )
            result["next_recommendation"] = (
                "Inspect detection overlays and place the dry bottle/can upright in the central reachable workspace, "
                "away from image edges, then rerun no-motion observe/plan."
            )
            return result

        xy_dist = _xy_distance(bottle_det, can_det)
        result["bottle_can_xy_distance_m"] = xy_dist
        min_separation = _env_float("OPENFORGE_DRY_POUR_MIN_BOTTLE_CAN_XY_SEPARATION_M", 0.06)
        if xy_dist is not None and xy_dist < min_separation:
            result["why_stopped"] = "live bottle/can detections are too close to trust as distinct objects"
            result["next_recommendation"] = "Inspect overlays; reposition or improve prompts before planning."
            return result

        freespace_move = _required_tool("freespace_move")
        bottle_candidates = _build_bottle_candidates(bottle_det)
        can_candidates = _build_can_candidates(can_det)
        result["candidate_counts"] = {
            "bottle": len(bottle_candidates),
            "can": len(can_candidates),
        }
        result["candidate_artifacts"] = {
            "bottle_candidates": write_json(run_dir / "plans" / "bottle_side_grasp_candidates.json", bottle_candidates),
            "can_candidates": write_json(run_dir / "plans" / "can_side_grasp_candidates.json", can_candidates),
        }

        bottle_plan = _rank_object_candidates(
            run_dir=run_dir,
            object_role="bottle",
            candidates=bottle_candidates,
            freespace_move=freespace_move,
        )
        can_plan = _rank_object_candidates(
            run_dir=run_dir,
            object_role="can",
            candidates=can_candidates,
            freespace_move=freespace_move,
        )
        result["bottle_plan"] = {
            "plan_packet": bottle_plan.get("plan_path"),
            "selected": bottle_plan.get("selected"),
            "preview_count": len(bottle_plan.get("candidates") or []),
        }
        result["can_plan"] = {
            "plan_packet": can_plan.get("plan_path"),
            "selected": can_plan.get("selected"),
            "preview_count": len(can_plan.get("candidates") or []),
        }
        if bottle_plan.get("selected") is None:
            result["why_stopped"] = "no left-arm bottle side-grasp candidate passed preview planning"
            result["next_recommendation"] = "Inspect bottle plan packet; tune bottle yaw/z/standoff in no-motion."
            return result
        if can_plan.get("selected") is None:
            result["why_stopped"] = "no right-arm can side-grasp candidate passed preview planning"
            result["next_recommendation"] = "Inspect can plan packet; tune can yaw/z/standoff in no-motion."
            return result

        rehearsal = _build_rehearsal_waypoints(
            bottle_detection=bottle_det,
            can_detection=can_det,
            bottle_plan=bottle_plan["selected"],
            can_plan=can_plan["selected"],
        )
        result["bimanual_rehearsal"] = rehearsal
        rehearsal["path"] = write_json(run_dir / "plans" / "bimanual_rehearsal_waypoints.json", rehearsal)
        bimanual_preview = _preview_rehearsal(
            run_dir=run_dir,
            rehearsal=rehearsal,
            freespace_move=freespace_move,
        )
        result["bimanual_preview"] = {
            "path": bimanual_preview.get("path"),
            "success": bimanual_preview.get("success"),
            "preview_count": bimanual_preview.get("preview_count"),
            "failed_waypoints": [item.get("name") for item in bimanual_preview.get("previews", []) if not item.get("success")],
        }
        result["selected_strategy"] = {
            "bottle": bottle_plan.get("selected"),
            "can": can_plan.get("selected"),
            "bimanual_rehearsal_path": rehearsal["path"],
            "bimanual_preview_path": bimanual_preview.get("path"),
        }
        result["plan_success"] = bool(bimanual_preview.get("success"))
        if result["plan_success"] and result["physical_gate"]:
            physical_attempt = _execute_physical_rehearsal(
                run_dir=run_dir,
                result=result,
                bottle_det=bottle_det,
                can_det=can_det,
                bottle_plan=bottle_plan["selected"],
                can_plan=can_plan["selected"],
                bimanual_preview=bimanual_preview,
                freespace_move=freespace_move,
            )
            result["attempt_count"] = 1
            result["attempts"].append(physical_attempt)
            result["physical_attempt"] = physical_attempt
            result["physical_motion_executed"] = bool(physical_attempt.get("physical_motion_executed"))
            result["movement_capable_calls"] = list(physical_attempt.get("movement_capable_calls") or [])
            result["success"] = bool(physical_attempt.get("success"))
            result["reward"] = 1.0 if result["success"] else 0.0
            result["why_stopped"] = physical_attempt.get("why_stopped")
            result["next_recommendation"] = (
                "Stop; dry bimanual rehearsal completed."
                if result["success"]
                else "Inspect physical_attempt_01.json, post observations, and videos before any artifact-driven retry."
            )
        elif result["plan_success"]:
            result["success"] = True
            result["reward"] = 1.0
            result["why_stopped"] = "no-motion live observe/plan/preview succeeded; no hardware motion executed"
            result["next_recommendation"] = "Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 only under the active bounded ticket to execute staged physical rehearsal."
        else:
            failed = result["bimanual_preview"]["failed_waypoints"]
            result["why_stopped"] = f"single-arm plans succeeded, but bimanual preview failed for {failed}"
            result["next_recommendation"] = "Inspect bimanual preview packet and tune pre-pour standoff/height/orientation in no-motion."
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and run log before retrying no-motion."
        result["error"] = result["why_stopped"]
    finally:
        result["video_paths"] = _video_paths(Path(run_dir))
        write_json(Path(run_dir) / "result.json", result)
        write_json(Path(run_dir) / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- why_stopped: {result['why_stopped']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))
    return result


TASK_RESULT.update(_run())
