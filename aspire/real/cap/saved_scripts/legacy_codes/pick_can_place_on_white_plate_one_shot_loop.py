# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick up a can and place it into/on a white plate on real YAM.

High-level transfer from ``can-libero-transfer.md`` only:
- localize the can and target plate from fresh perception before motion;
- treat the can as a simple cylinder and force a clean top-down grasp instead
  of trusting tilted grasp orientations;
- lift, transport above the plate, lower, release, retreat, then verify from
  post-action perception.

Default behavior is observe/preview only. Physical execution requires
``OPENFORGE_ALLOW_PHYSICAL_MOTION=1`` plus the can-plate ticket envs checked in
``_physical_ticket_status``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    current_run_dir,
    json_safe,
    read_gripper_width,
    staged_close_with_contact,
    write_json,
)


TASK_NAME = "pick_can_place_on_white_plate_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            continue
    return values or list(default)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _xyz(value: Any) -> list[float] | None:
    values = _round_list(value, limit=3)
    return values if len(values) == 3 else None


def _dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _can_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_CAN_PLATE_CAN_PROMPTS",
        ["can", "soda can", "spirit can", "aluminum can", "tin can"],
    )


def _plate_prompts() -> list[str]:
    return _csv("OPENFORGE_CAN_PLATE_PLATE_PROMPTS", ["white plate", "plate", "dish"])


def _all_prompts() -> list[str]:
    prompts: list[str] = []
    for prompt in _can_prompts() + _plate_prompts() + _csv("OPENFORGE_CAN_PLATE_EXTRA_PROMPTS", []):
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts


def _cameras() -> list[str]:
    return _csv("OPENFORGE_CAN_PLATE_CAMERAS", ["top", "left", "right", "bottom"])


def _image_only_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_CAN_PLATE_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _motion_cameras() -> tuple[str, ...]:
    return tuple(camera for camera in _csv("OPENFORGE_CAN_PLATE_MOTION_CAMERAS", ["top", "left", "right"]) if camera)


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_name = globals().get("__file__") or f"cap/saved_scripts/{TASK_NAME}.py"
    script_path = Path(str(script_name)).resolve()
    snapshot: dict[str, Any] = {
        "schema": "openforge.can_plate.source_snapshot.v1",
        "script": str(script_path),
        "env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("OPENFORGE_CAN_PLATE_")
            or name
            in {
                "OPENFORGE_ALLOW_PHYSICAL_MOTION",
                "YAM_STATION_CALIBRATED_XML",
                "CAP_TOP_CAMERA_BACKEND",
                "CAP_TOP_CAMERA_FRAME",
                "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP",
            }
        },
    }
    try:
        snapshot["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        snapshot["git_status_short"] = subprocess.check_output(
            ["git", "status", "--short", "--", str(script_path.relative_to(Path.cwd()))],
            text=True,
        ).strip()
    except Exception as exc:
        snapshot["git_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["path"] = write_json(run_dir / "source_snapshot.json", snapshot)
    return snapshot


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_CAN_PLATE_RECORDING_WAIT_S", 3.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    return {
        "required": True,
        "passed": bool(videos),
        "video_paths": videos,
        "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
    }


def _detections(scene: dict[str, Any], prompts: list[str]) -> list[dict[str, Any]]:
    prompt_set = {prompt.strip() for prompt in prompts}
    rows: list[dict[str, Any]] = []
    for detection in scene.get("all_detections") or []:
        prompt = str(detection.get("prompt") or "").strip()
        xyz = _xyz(detection.get("position_3d") or detection.get("position"))
        if prompt in prompt_set and xyz is not None:
            row = dict(detection)
            row["position_3d"] = xyz
            rows.append(row)
    return rows


def _half_extents(det: dict[str, Any] | None) -> list[float]:
    values = _round_list((det or {}).get("half_extents") or [], limit=3)
    return values if len(values) == 3 else []


def _prompt_rank(prompt: str, prompts: list[str]) -> int:
    try:
        return prompts.index(prompt)
    except ValueError:
        return len(prompts) + 1


def _camera_rank(camera: str | None) -> int:
    order = {"top": 0, "left": 1, "right": 2}
    return order.get(str(camera or ""), 9)


def _can_geometry_score(det: dict[str, Any]) -> float:
    half_extents = _half_extents(det)
    if len(half_extents) < 3:
        return 0.0
    xy_vals = sorted(abs(float(v)) for v in half_extents[:2])
    radius = xy_vals[0] if xy_vals else 0.0
    half_height = abs(float(half_extents[2]))
    if radius <= 1e-6:
        return 0.0
    height = 2.0 * half_height
    diameter = 2.0 * radius
    score = 0.0
    if 0.025 <= diameter <= 0.120:
        score += 0.20
    if 0.035 <= height <= 0.220:
        score += 0.25
    if height >= 0.75 * diameter:
        score += 0.15
    if height <= 0.030:
        score -= 0.40
    return score


def _select_detection(scene: dict[str, Any], prompts: list[str], *, kind: str) -> dict[str, Any] | None:
    candidates = _detections(scene, prompts)
    min_score = _env_float(f"OPENFORGE_CAN_PLATE_{kind.upper()}_MIN_SCORE", 0.20)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for det in candidates:
        score = float(det.get("score") or min_score)
        if score < min_score:
            continue
        prompt = str(det.get("prompt") or "")
        rank = score - 0.04 * _prompt_rank(prompt, prompts) - 0.01 * _camera_rank(det.get("source_camera"))
        if kind == "can":
            rank += _can_geometry_score(det)
        ranked.append((rank, det))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = dict(ranked[0][1])
    selected["selection_rank_score"] = round(float(ranked[0][0]), 5)
    selected["selection_kind"] = kind
    return selected


def _arms_from_can(can_xyz: list[float]) -> list[str]:
    requested = os.environ.get("OPENFORGE_CAN_PLATE_ARM", "auto").strip().lower()
    if requested in {"left", "right"}:
        return [requested]
    preferred = "left" if float(can_xyz[1]) > 0.04 else "right"
    alternate = "right" if preferred == "left" else "left"
    return [preferred, alternate]


def _estimate_radius(det: dict[str, Any]) -> float:
    half_extents = _half_extents(det)
    if len(half_extents) >= 2:
        xy_vals = sorted(abs(float(v)) for v in half_extents[:2])
        if xy_vals[0] > 0.005:
            return max(0.018, min(0.060, xy_vals[0]))
    return _env_float("OPENFORGE_CAN_PLATE_DEFAULT_CAN_RADIUS_M", 0.035)


def _estimate_half_height(det: dict[str, Any]) -> float:
    half_extents = _half_extents(det)
    if len(half_extents) >= 3 and abs(float(half_extents[2])) > 0.01:
        return max(0.025, min(0.120, abs(float(half_extents[2]))))
    return _env_float("OPENFORGE_CAN_PLATE_DEFAULT_CAN_HALF_HEIGHT_M", 0.055)


def _release_z(plate_det: dict[str, Any], can_half_height: float, grasp_z: float) -> float:
    if _truthy_env("OPENFORGE_CAN_PLATE_USE_PLATE_Z_FOR_RELEASE", True):
        plate_xyz = _xyz(plate_det.get("position_3d") or plate_det.get("position")) or [0.0, 0.0, grasp_z]
        return float(plate_xyz[2]) + float(can_half_height) + _env_float("OPENFORGE_CAN_PLATE_RELEASE_CLEARANCE_M", 0.025)
    return float(grasp_z) + _env_float("OPENFORGE_CAN_PLATE_RELEASE_Z_DELTA_FROM_GRASP_M", 0.0)


def _make_candidate(
    *,
    index: int,
    arm: str,
    can_det: dict[str, Any],
    plate_det: dict[str, Any],
    yaw_deg: float,
    z_offset_m: float,
) -> dict[str, Any]:
    can_xyz = _xyz(can_det.get("position_3d") or can_det.get("position"))
    plate_xyz = _xyz(plate_det.get("position_3d") or plate_det.get("position"))
    if can_xyz is None or plate_xyz is None:
        raise ValueError("can and plate detections must have 3D positions")
    can_half_height = _estimate_half_height(can_det)
    can_radius = _estimate_radius(can_det)
    grasp_z = float(can_xyz[2]) + _env_float("OPENFORGE_CAN_PLATE_GRASP_Z_BIAS_M", 0.020) + float(z_offset_m)
    release_z = _release_z(plate_det, can_half_height, grasp_z)
    approach_z = _env_float("OPENFORGE_CAN_PLATE_APPROACH_Z_M", 0.090)
    lift_z = _env_float("OPENFORGE_CAN_PLATE_LIFT_Z_M", 0.050)
    place_approach_z = _env_float("OPENFORGE_CAN_PLATE_PLACE_APPROACH_Z_M", 0.080)
    retreat_z = _env_float("OPENFORGE_CAN_PLATE_RETREAT_Z_M", 0.080)
    rpy = [0.0, _env_float("OPENFORGE_CAN_PLATE_TOPDOWN_PITCH_DEG", 180.0), float(yaw_deg)]
    place_xy_bias = [
        _env_float("OPENFORGE_CAN_PLATE_PLACE_X_BIAS_M", 0.0),
        _env_float("OPENFORGE_CAN_PLATE_PLACE_Y_BIAS_M", 0.0),
    ]
    plate_xy = [float(plate_xyz[0]) + place_xy_bias[0], float(plate_xyz[1]) + place_xy_bias[1]]
    lift_abs_z = grasp_z + lift_z
    pre_place_z = max(lift_abs_z, release_z + place_approach_z)
    can_xy = [float(can_xyz[0]), float(can_xyz[1])]
    width = max(0.02, min(0.95, 2.0 * can_radius + _env_float("OPENFORGE_CAN_PLATE_GRIPPER_WIDTH_MARGIN_M", 0.020)))
    return {
        "index": int(index),
        "schema": "openforge.can_plate.candidate.v1",
        "arm": arm,
        "strategy": "top_down_can_to_white_plate",
        "score": round(1.0 - 0.03 * index - 0.4 * abs(float(z_offset_m)), 5),
        "rpy": _round_list(rpy, digits=4, limit=3),
        "width": round(float(width), 5),
        "can_radius_m": round(float(can_radius), 5),
        "can_half_height_m": round(float(can_half_height), 5),
        "topdown_yaw_deg": round(float(yaw_deg), 4),
        "z_offset_m": round(float(z_offset_m), 5),
        "place_xy_bias_m": _round_list(place_xy_bias, limit=2),
        "source_can_detection": can_det,
        "source_plate_detection": plate_det,
        "poses": {
            "pregrasp": {"position": [can_xy[0], can_xy[1], grasp_z + approach_z], "rpy": rpy},
            "grasp": {"position": [can_xy[0], can_xy[1], grasp_z], "rpy": rpy},
            "lift": {"position": [can_xy[0], can_xy[1], lift_abs_z], "rpy": rpy},
            "pre_place": {"position": [plate_xy[0], plate_xy[1], pre_place_z], "rpy": rpy},
            "place": {"position": [plate_xy[0], plate_xy[1], release_z], "rpy": rpy},
            "retreat": {"position": [plate_xy[0], plate_xy[1], release_z + retreat_z], "rpy": rpy},
        },
        "geometry_policy": (
            "Can is treated as a cylinder; top-down gripper opening yaw is swept, "
            "plate placement is based on fresh white-plate center plus optional XY bias."
        ),
    }


def _generate_candidates(can_det: dict[str, Any], plate_det: dict[str, Any]) -> list[dict[str, Any]]:
    can_xyz = _xyz(can_det.get("position_3d") or can_det.get("position"))
    if can_xyz is None:
        return []
    arms = _arms_from_can(can_xyz)
    yaws = _float_csv("OPENFORGE_CAN_PLATE_TOPDOWN_YAWS_DEG", [90.0, 0.0, -90.0, 180.0, 45.0, -45.0])
    z_offsets = _float_csv("OPENFORGE_CAN_PLATE_TOPDOWN_Z_OFFSETS_M", [0.000, 0.010, -0.010, 0.020])
    max_candidates = max(1, _env_int("OPENFORGE_CAN_PLATE_MAX_CANDIDATES", 48))
    candidates: list[dict[str, Any]] = []
    idx = 1
    for z_offset in z_offsets:
        for yaw in yaws:
            for arm in arms:
                candidates.append(
                    _make_candidate(
                        index=idx,
                        arm=arm,
                        can_det=can_det,
                        plate_det=plate_det,
                        yaw_deg=float(yaw),
                        z_offset_m=float(z_offset),
                    )
                )
                idx += 1
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _move_kwargs(arm: str, pose: dict[str, Any], *, preview_only: bool, gripper: float | None = None) -> dict[str, Any]:
    prefix = "left" if arm == "left" else "right"
    kwargs: dict[str, Any] = {
        f"{prefix}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{prefix}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_CAN_PLATE_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_CAN_PLATE_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_CAN_PLATE_PLANNING_SPEED", 0.30),
        "ik_error_threshold": _env_float("OPENFORGE_CAN_PLATE_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_CAN_PLATE_IK_ROT_THRESHOLD_DEG", 14.0),
        "ik_xyz_weight": _env_float("OPENFORGE_CAN_PLATE_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_CAN_PLATE_IK_RPY_WEIGHT", 0.35),
    }
    if gripper is not None:
        kwargs[f"{prefix}_gripper"] = float(gripper)
    return kwargs


def _preview_summary(raw: Any) -> dict[str, Any]:
    safe = json_safe(raw)
    if isinstance(safe, dict) and "data" in safe and isinstance(safe["data"], dict):
        safe = safe["data"]
    status = str(_field(safe, "status", "") or "")
    executed = bool(_field(safe, "executed", False))
    pos_error = _field(safe, "final_pos_error_m", _field(safe, "ik_error_m", None))
    rot_error = _field(safe, "final_rot_error_deg", _field(safe, "ik_rot_error_deg", None))
    max_pos = _env_float("OPENFORGE_CAN_PLATE_PREVIEW_MAX_POS_ERROR_M", 0.030)
    max_rot = _env_float("OPENFORGE_CAN_PLATE_PREVIEW_MAX_ROT_ERROR_DEG", 16.0)
    status_ok = status.lower() in {"success", "succeeded", "ok"} and not executed
    pos_ok = pos_error is None or float(pos_error) <= max_pos
    rot_ok = rot_error is None or float(rot_error) <= max_rot
    return {
        "status": status or None,
        "reason": _field(safe, "reason", _field(safe, "error", None)),
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": max_pos,
        "max_rot_error_deg": max_rot,
        "trajectory_cache_key": _field(safe, "trajectory_cache_key", None),
        "success": bool(status_ok and pos_ok and rot_ok),
        "raw": safe,
    }


def _preview_pose(label: str, arm: str, pose: dict[str, Any], *, gripper: float | None = None) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        _required_tool("freespace_move"),
        _env_float("OPENFORGE_CAN_PLATE_PREVIEW_TIMEOUT_S", 25.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(arm, pose, preview_only=True, gripper=gripper),
    )
    if not result.get("ok"):
        return {"label": label, "success": False, "reason": result.get("error"), "executed": False}
    summary = _preview_summary(result.get("data"))
    summary["label"] = label
    summary["arm"] = arm
    summary["target_pos"] = _round_list(pose.get("position"), limit=3)
    summary["target_rpy"] = _round_list(pose.get("rpy"), digits=4, limit=3)
    return summary


def _preview_sequence(candidate: dict[str, Any]) -> dict[str, Any]:
    arm = str(candidate["arm"])
    poses = candidate["poses"]
    open_gripper_pos = _env_float("OPENFORGE_CAN_PLATE_OPEN_GRIPPER_POS", 1.0)
    hold_gripper_pos = _env_float("OPENFORGE_CAN_PLATE_HOLD_GRIPPER_POS", 0.20)
    steps = [
        _preview_pose("can_pregrasp", arm, poses["pregrasp"], gripper=open_gripper_pos),
        _preview_pose("can_grasp", arm, poses["grasp"], gripper=open_gripper_pos),
        _preview_pose("can_lift", arm, poses["lift"], gripper=hold_gripper_pos),
        _preview_pose("can_pre_place", arm, poses["pre_place"], gripper=hold_gripper_pos),
        _preview_pose("can_place", arm, poses["place"], gripper=hold_gripper_pos),
        _preview_pose("can_retreat_after_release", arm, poses["retreat"], gripper=open_gripper_pos),
    ]
    return {
        "schema": "openforge.can_plate.preview_sequence.v1",
        "success": all(bool(step.get("success")) for step in steps),
        "steps": steps,
    }


def _preview_and_select(candidates: list[dict[str, Any]], run_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    limit = min(len(candidates), max(1, _env_int("OPENFORGE_CAN_PLATE_PREVIEW_LIMIT", 24)))
    previews: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in sorted(candidates, key=lambda row: float(row.get("score") or 0.0), reverse=True)[:limit]:
        row = dict(candidate)
        row["preview_sequence"] = _preview_sequence(candidate)
        previews.append(row)
        if row["preview_sequence"].get("success"):
            selected = row
            break
    packet = {
        "schema": "openforge.can_plate.preview_rank.v1",
        "candidate_count": len(candidates),
        "previewed_count": len(previews),
        "selected": selected,
        "previews": previews,
    }
    packet["path"] = write_json(run_dir / "plans" / "can_plate_sequence_preview.json", packet)
    return selected, previews


def _physical_ticket_status() -> dict[str, Any]:
    required = {
        "OPENFORGE_CAN_PLATE_PHYSICAL_TICKET": "1",
        "OPENFORGE_CAN_PLATE_OPERATOR_PRESENT": "1",
        "OPENFORGE_CAN_PLATE_ESTOP_REACHABLE": "1",
        "OPENFORGE_CAN_PLATE_WORKSPACE_CLEAR": "1",
        "OPENFORGE_CAN_PLATE_HANDS_TOOLS_CABLES_CLEAR": "1",
        "OPENFORGE_CAN_PLATE_FOLLOWER_SERVERS_UP": "1",
        "OPENFORGE_CAN_PLATE_NO_OTHER_MOTION_PRECHECK": "1",
        "OPENFORGE_CAN_PLATE_RECORDING_DEBUG_CONFIRMED": "1",
        "OPENFORGE_CAN_PLATE_CAN_VISIBLE": "1",
        "OPENFORGE_CAN_PLATE_WHITE_PLATE_VISIBLE": "1",
        "OPENFORGE_CAN_PLATE_NO_MANUAL_XYZ": "1",
    }
    missing = [name for name, expected in required.items() if os.environ.get(name) != expected]
    return {"passed": not missing, "missing_or_invalid": missing, "required": required}


def _set_gripper_checked(arm: str, target: float, stage: str, run_dir: Path) -> dict[str, Any]:
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    before = read_gripper_width(get_robot_state, arm)
    kwargs: dict[str, float] = {}
    vel = _env_float_or_none("OPENFORGE_CAN_PLATE_GRIPPER_VEL_LIMIT")
    torque = _env_float_or_none("OPENFORGE_CAN_PLATE_GRIPPER_TORQUE_LIMIT")
    if vel is not None:
        kwargs["vel_limit"] = vel
    if torque is not None:
        kwargs["torque_limit"] = torque
    command = set_gripper(arm, float(target), **kwargs)
    after = read_gripper_width(get_robot_state, arm)
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_CAN_PLATE_GRIPPER_CONFIRM_TIMEOUT_S", 4.0))
    confirm_min = _env_float("OPENFORGE_CAN_PLATE_OPEN_CONFIRM_POS", 0.75)
    if target >= confirm_min:
        while after is not None and after < confirm_min and time.monotonic() < deadline:
            time.sleep(max(0.02, _env_float("OPENFORGE_CAN_PLATE_STATE_POLL_S", 0.15)))
            after = read_gripper_width(get_robot_state, arm)
    packet = {
        "schema": "openforge.can_plate.set_gripper.v1",
        "stage": stage,
        "arm": arm,
        "target": float(target),
        "before": before,
        "after": after,
        "command": json_safe(command),
        "confirm_min": confirm_min if target >= confirm_min else None,
        "confirmed": bool(after is None or target < confirm_min or after >= confirm_min),
    }
    packet["path"] = write_json(run_dir / "plans" / f"{stage}.json", packet)
    return packet


def _execute_pose(label: str, arm: str, pose: dict[str, Any], run_dir: Path, *, gripper: float | None = None) -> dict[str, Any]:
    preview = _preview_pose(label, arm, pose, gripper=gripper)
    if not preview.get("success") or not preview.get("trajectory_cache_key"):
        raise RuntimeError(f"{label} preview failed before execution: {preview.get('reason') or preview}")
    executed = _required_tool("freespace_move")(trajectory_cache_key=preview["trajectory_cache_key"])
    packet = {"schema": "openforge.can_plate.execute_pose.v1", "stage": label, "arm": arm, "preview": preview, "execute": json_safe(executed)}
    packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
    return packet


def _verify_post_place(
    run_dir: Path,
    pre_can: dict[str, Any] | None,
    pre_plate: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = capture_scene(
        prompts=_all_prompts(),
        cameras=_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="post_can_place",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_CAN_PLATE_OBS_TIMEOUT_S", 30.0),
        max_retries=1,
        motion_cameras=_motion_cameras(),
        image_only_cameras=_image_only_cameras(),
    )
    post_can = _select_detection(scene, _can_prompts(), kind="can")
    post_plate = _select_detection(scene, _plate_prompts(), kind="plate") or pre_plate
    pre_can_xyz = _xyz((pre_can or {}).get("position_3d") or (pre_can or {}).get("position"))
    post_can_xyz = _xyz((post_can or {}).get("position_3d") or (post_can or {}).get("position"))
    plate_xyz = _xyz((post_plate or {}).get("position_3d") or (post_plate or {}).get("position"))
    packet: dict[str, Any] = {
        "schema": "openforge.can_plate.post_verify.v1",
        "success": False,
        "confidence": 0.0,
        "post_scene": scene,
        "pre_can_xyz": pre_can_xyz,
        "post_can_detection": post_can,
        "post_can_xyz": post_can_xyz,
        "plate_detection": post_plate,
        "plate_xyz": plate_xyz,
        "can_to_plate_xy_m": None,
        "can_from_source_xy_m": None,
        "why": None,
    }
    if post_can_xyz is None:
        packet["why"] = "post observation did not find the can; inspect videos before counting success"
    elif plate_xyz is None:
        packet["why"] = "post observation did not find the plate target"
    else:
        can_to_plate = _dist_xy(post_can_xyz, plate_xyz)
        packet["can_to_plate_xy_m"] = round(float(can_to_plate), 5)
        if pre_can_xyz is not None:
            packet["can_from_source_xy_m"] = round(float(_dist_xy(post_can_xyz, pre_can_xyz)), 5)
        threshold = _env_float("OPENFORGE_CAN_PLATE_SUCCESS_MAX_CAN_TO_PLATE_XY_M", 0.080)
        moved_min = _env_float("OPENFORGE_CAN_PLATE_SUCCESS_MIN_SOURCE_XY_DELTA_M", 0.040)
        moved_ok = packet["can_from_source_xy_m"] is None or float(packet["can_from_source_xy_m"]) >= moved_min
        packet["success"] = bool(can_to_plate <= threshold and moved_ok)
        packet["confidence"] = 0.85 if packet["success"] else 0.25
        packet["why"] = (
            "post can detection is centered near the white plate"
            if packet["success"]
            else "post can detection is not close enough to the white plate or did not move from source"
        )
    packet["path"] = write_json(run_dir / "plans" / "post_can_place_verify.json", packet)
    return packet


def _execute_physical(
    selected: dict[str, Any],
    run_dir: Path,
    pre_can: dict[str, Any] | None,
    pre_plate: dict[str, Any] | None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.can_plate.physical_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        packet["why_stopped"] = "physical gate not set; no motion executed"
        packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
        return packet
    ticket = _physical_ticket_status()
    packet["physical_ticket_status"] = ticket
    if not ticket["passed"]:
        packet["why_stopped"] = f"missing physical ticket envs: {ticket['missing_or_invalid']}"
        packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
        return packet
    if not selected.get("preview_sequence", {}).get("success"):
        packet["why_stopped"] = "selected sequence preview did not pass; refusing physical execution"
        packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
        return packet

    arm = str(selected["arm"])
    poses = selected["poses"]
    open_pos = _env_float("OPENFORGE_CAN_PLATE_OPEN_GRIPPER_POS", 1.0)
    hold_pos = _env_float("OPENFORGE_CAN_PLATE_HOLD_GRIPPER_POS", 0.20)
    try:
        packet["steps"].append(_set_gripper_checked(arm, open_pos, "open_before_can_pick", run_dir))
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        packet["physical_motion_executed"] = True
        if not packet["steps"][-1].get("confirmed"):
            packet["why_stopped"] = "gripper did not confirm open before can approach"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        for label in ("pregrasp", "grasp"):
            step = _execute_pose(f"can_{label}", arm, poses[label], run_dir, gripper=open_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append(f"freespace_move:can_{label}")

        close = staged_close_with_contact(
            side=arm,
            set_gripper=_required_tool("set_gripper"),
            get_robot_state=_required_tool("get_robot_state"),
            target=_env_float("OPENFORGE_CAN_PLATE_CLOSE_TARGET_POS", 0.04),
            steps=tuple(_float_csv("OPENFORGE_CAN_PLATE_CLOSE_STEPS", [0.68, 0.40, 0.18, 0.08, 0.04])),
            vel_limit=_env_float_or_none("OPENFORGE_CAN_PLATE_CLOSE_VEL_LIMIT"),
            torque_limit=_env_float_or_none("OPENFORGE_CAN_PLATE_CLOSE_TORQUE_LIMIT"),
            min_contact_delta=_env_float("OPENFORGE_CAN_PLATE_CLOSE_CONTACT_MIN_DELTA_POS", 0.015),
            target_tolerance=_env_float("OPENFORGE_CAN_PLATE_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_CAN_PLATE_HOLD_MIN_POS", 0.015),
            hold_max=_env_float("OPENFORGE_CAN_PLATE_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="can_close",
        )
        packet["steps"].append(close)
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        packet["gripper_after_close"] = close
        require_contact = _truthy_env("OPENFORGE_CAN_PLATE_REQUIRE_CONTACT_BEFORE_LIFT", True)
        close_ok = bool(close.get("contact_detected")) if require_contact else bool(close.get("plausible_for_lift"))
        packet["close_gate"] = {
            "require_contact_before_lift": require_contact,
            "passed": close_ok,
            "contact_detected": bool(close.get("contact_detected")),
            "target_reached": bool(close.get("target_reached")),
            "plausible_for_lift": bool(close.get("plausible_for_lift")),
        }
        if not close_ok:
            packet["why_stopped"] = "can close did not produce required hold/contact evidence; stop before lift"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        for label in ("lift", "pre_place", "place"):
            step = _execute_pose(f"can_{label}", arm, poses[label], run_dir, gripper=hold_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append(f"freespace_move:can_{label}")

        release = _set_gripper_checked(arm, open_pos, "open_release_in_plate", run_dir)
        packet["steps"].append(release)
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        if not release.get("confirmed"):
            packet["why_stopped"] = "release open did not confirm; do not retreat blindly"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        step = _execute_pose("can_retreat_after_release", arm, poses["retreat"], run_dir, gripper=open_pos)
        packet["steps"].append(step)
        packet["movement_capable_calls"].append("freespace_move:can_retreat_after_release")
        post = _verify_post_place(run_dir, pre_can, pre_plate)
        packet["post_verify"] = post
        packet["success"] = bool(post.get("success"))
        packet["confidence"] = post.get("confidence")
        packet["why_stopped"] = (
            "can placement verified by post observation"
            if packet["success"]
            else f"motions completed but verification failed: {post.get('why')}"
        )
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
    packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
    return packet


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result: dict[str, Any] = {
        "schema": "openforge.can_plate.task_result.v1",
        "success": False,
        "plan_success": False,
        "confidence": 0.0,
        "physical_motion_executed": False,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "source_snapshot": None,
        "pre_observation": None,
        "selected_can_detection": None,
        "selected_plate_detection": None,
        "candidate_count": 0,
        "selected_strategy": None,
        "physical_attempt": None,
        "post_observation": None,
        "gripper_after_close": None,
        "object_motion": None,
        "video_paths": [],
        "recording_evidence": None,
        "why_stopped": None,
        "next_recommendation": None,
    }
    try:
        result["source_snapshot"] = _source_snapshot(run_dir)
        scene = capture_scene(
            prompts=_all_prompts(),
            cameras=_cameras(),
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="pre_can_plate",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_CAN_PLATE_OBS_TIMEOUT_S", 30.0),
            max_retries=1,
            motion_cameras=_motion_cameras(),
            image_only_cameras=_image_only_cameras(),
        )
        result["pre_observation"] = scene
        can_det = _select_detection(scene, _can_prompts(), kind="can")
        plate_det = _select_detection(scene, _plate_prompts(), kind="plate")
        result["selected_can_detection"] = can_det
        result["selected_plate_detection"] = plate_det
        if can_det is None:
            result["why_stopped"] = "missing can detection"
            result["next_recommendation"] = "Adjust can prompts or scene placement, then rerun no-motion preview."
            return result
        if plate_det is None:
            result["why_stopped"] = "missing white plate detection"
            result["next_recommendation"] = "Adjust plate prompts or scene placement, then rerun no-motion preview."
            return result

        candidates = _generate_candidates(can_det, plate_det)
        result["candidate_count"] = len(candidates)
        write_json(
            run_dir / "plans" / "can_plate_candidates.json",
            {"schema": "openforge.can_plate.candidates.v1", "candidates": candidates},
        )
        if not candidates:
            result["why_stopped"] = "no top-down can candidates generated"
            result["next_recommendation"] = "Inspect selected detections and candidate geometry."
            return result

        selected, previews = _preview_and_select(candidates, run_dir)
        if selected is None:
            result["why_stopped"] = "no can-to-plate sequence candidate passed planner preview"
            result["next_recommendation"] = "Inspect plans/can_plate_sequence_preview.json and tune yaw/Z/place offsets."
            return result
        result["selected_strategy"] = selected
        result["plan_success"] = True

        physical_attempt = _execute_physical(selected, run_dir, can_det, plate_det)
        result["physical_attempt"] = physical_attempt
        result["physical_motion_executed"] = bool(physical_attempt.get("physical_motion_executed"))
        result["gripper_after_close"] = physical_attempt.get("gripper_after_close")
        post_verify = physical_attempt.get("post_verify") or {}
        result["post_observation"] = post_verify.get("post_scene")
        result["object_motion"] = {
            "pre_can_xyz": post_verify.get("pre_can_xyz"),
            "post_can_xyz": post_verify.get("post_can_xyz"),
            "plate_xyz": post_verify.get("plate_xyz"),
            "can_to_plate_xy_m": post_verify.get("can_to_plate_xy_m"),
            "can_from_source_xy_m": post_verify.get("can_from_source_xy_m"),
        }
        if result["physical_motion_executed"]:
            result["success"] = bool(physical_attempt.get("success"))
            result["confidence"] = float(physical_attempt.get("confidence") or (0.8 if result["success"] else 0.2))
            result["why_stopped"] = physical_attempt.get("why_stopped")
            result["next_recommendation"] = "Inspect videos and post verification before retrying."
        else:
            result["success"] = False
            result["confidence"] = 0.0
            result["why_stopped"] = "no-motion can-to-white-plate preview passed; physical motion not executed"
            result["next_recommendation"] = "Physical attempt needs a fresh ticket and OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and service logs before retrying."
        return result
    finally:
        result["recording_evidence"] = _recording_evidence(run_dir)
        result["video_paths"] = result["recording_evidence"].get("video_paths") or []
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## Can To White Plate",
                f"- success: {result.get('success')}",
                f"- plan_success: {result.get('plan_success')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- selected_can_detection: {result.get('selected_can_detection')}",
                f"- selected_plate_detection: {result.get('selected_plate_detection')}",
                f"- selected_strategy: {result.get('selected_strategy')}",
                f"- gripper_after_close: {result.get('gripper_after_close')}",
                f"- object_motion: {result.get('object_motion')}",
                f"- recording_evidence: {result.get('recording_evidence')}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- next_recommendation: {result.get('next_recommendation')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
