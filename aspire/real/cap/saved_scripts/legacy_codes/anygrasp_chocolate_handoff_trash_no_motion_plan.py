# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion AnyGrasp plan for chocolate-in-bowl handoff and trash release.

This is a planning/debug script only. It refuses physical motion, observes the
current scene, samples real AnyGrasp candidates for a flat chocolate bar that
may be inside a white bowl, and previews candidate waypoints:

1. left arm top-camera AnyGrasp approach/grasp/lift,
2. bimanual center handoff staging,
3. right arm controlled release above the live-detected white trash can.

The final "throw" should be implemented as a controlled release/drop into the
can unless a separate safety review authorizes ballistic motion.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    current_run_dir,
    json_safe,
    write_json,
)


TASK_NAME = "anygrasp_chocolate_handoff_trash_no_motion_plan"
OBJECT_NAME = os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_OBJECT_NAME", "chocolate")
PROMPTS = [
    part.strip()
    for part in os.environ.get(
        "OPENFORGE_CHOCOLATE_HANDOFF_PROMPTS",
        "chocolate bar,chocolate,white bowl,white trash can,trash can,bin",
    ).split(",")
    if part.strip()
]
CAMERAS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_CAMERAS", "top,left,right,bottom").split(",")
    if part.strip()
]
IMAGE_ONLY_CAMERAS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_IMAGE_ONLY_CAMERAS", "left,right,bottom").split(",")
    if part.strip()
]

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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values or list(default)


def _unique_floats(values: list[float]) -> list[float]:
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        result.append(float(value))
    return result


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _best_detection(scene: dict[str, Any], prompts: tuple[str, ...]) -> dict[str, Any] | None:
    top = (scene.get("cameras") or {}).get("top") or {}
    detections = top.get("serialized_detections") or []
    candidates = [d for d in detections if str(d.get("prompt")) in prompts]
    if not candidates:
        return None
    return sorted(candidates, key=lambda d: float(d.get("score") or 0.0), reverse=True)[0]


def _anygrasp_health() -> dict[str, Any]:
    url = os.environ.get("ANYGRASP_SERVICE_URL", "http://127.0.0.1:8122").rstrip("/")
    health_url = f"{url}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_HEALTH_TIMEOUT_S", 5.0)) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        data["url"] = health_url
        return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "url": health_url, "error": f"{type(exc).__name__}: {exc}"}


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_path = Path(
        globals().get(
            "__file__",
            "cap/saved_scripts/anygrasp_chocolate_handoff_trash_no_motion_plan.py",
        )
    ).resolve()
    snapshot = {
        "script_path": str(script_path),
        "task_name": TASK_NAME,
        "object_name": OBJECT_NAME,
        "prompts": PROMPTS,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_CHOCOLATE_HANDOFF_")
            or name.startswith("ANYGRASP_")
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


def _candidate_to_dict(candidate: Any, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "position": _xyz(_field(candidate, "position")),
        "rpy": _round_list(_field(candidate, "rpy"), digits=4, limit=3),
        "score": _field(candidate, "score", None),
        "width_m": _field(candidate, "width", None),
        "source": "real_anygrasp",
    }


def _unwrap_tool_data(value: Any) -> Any:
    safe = json_safe(value)
    if isinstance(safe, dict) and safe.get("success") is False:
        raise RuntimeError(str(safe.get("error") or safe))
    if isinstance(safe, dict) and "data" in safe:
        return safe["data"]
    return safe


def _sample_anygrasp() -> dict[str, Any]:
    sample = _required_tool("sample_grasp_pose_anygrasp")
    result = call_with_timeout(
        "sample_grasp_pose_anygrasp",
        sample,
        _env_float("OPENFORGE_CHOCOLATE_HANDOFF_ANYGRASP_TIMEOUT_S", 90.0),
        object_name=OBJECT_NAME,
        camera=os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_CAMERA", "top"),
        max_grasps=_env_int("OPENFORGE_CHOCOLATE_HANDOFF_MAX_GRASPS", 20),
        top_down_only=_truthy_env("OPENFORGE_CHOCOLATE_HANDOFF_TOP_DOWN_ONLY", True),
        vertical_threshold=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_VERTICAL_THRESHOLD", 0.70),
        object_input_mode=os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_OBJECT_INPUT_MODE", "segmented_object_cloud"),
        tcp_offset_z_m=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_TCP_OFFSET_Z_M", 0.0),
        disable_planner_z_clipping=_truthy_env("OPENFORGE_CHOCOLATE_HANDOFF_DISABLE_PLANNER_Z_CLIPPING", True),
        filter_wrist_camera_y=_truthy_env("OPENFORGE_CHOCOLATE_HANDOFF_FILTER_WRIST_Y", False),
        allow_wrist_camera_yaw_flip=True,
        run_in_background=_tool("run_in_background"),
    )
    if not result.get("ok"):
        return {"success": False, "error": result.get("error"), "candidates": []}
    try:
        raw_candidates = _unwrap_tool_data(result.get("data")) or []
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}", "candidates": []}
    candidates = [_candidate_to_dict(candidate, index) for index, candidate in enumerate(raw_candidates, start=1)]
    candidates = [c for c in candidates if c.get("position") and len(c.get("rpy") or []) == 3]
    return {"success": bool(candidates), "candidates": candidates}


def _move_kwargs(side: str, pos: list[float], rpy: list[float], *, gripper: float | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        f"{side}_target_pos": [float(v) for v in pos[:3]],
        f"{side}_target_rpy": [float(v) for v in rpy[:3]],
        "preview_only": True,
        "planner_backend": os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_CHOCOLATE_HANDOFF_PLANNING_SPEED", 0.25),
        "ik_error_threshold": _env_float("OPENFORGE_CHOCOLATE_HANDOFF_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_CHOCOLATE_HANDOFF_IK_ROT_THRESHOLD_DEG", 14.0),
        "ik_xyz_weight": _env_float("OPENFORGE_CHOCOLATE_HANDOFF_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_CHOCOLATE_HANDOFF_IK_RPY_WEIGHT", 0.5),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _preview_summary(raw: Any) -> dict[str, Any]:
    safe = json_safe(raw)
    if isinstance(safe, dict) and "data" in safe and isinstance(safe["data"], dict):
        safe = safe["data"]
    status = str(_field(safe, "status", "") or "")
    executed = bool(_field(safe, "executed", False))
    pos_error = _field(safe, "final_pos_error_m", _field(safe, "ik_error_m", None))
    rot_error = _field(safe, "final_rot_error_deg", _field(safe, "ik_rot_error_deg", None))
    max_pos = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_PREVIEW_MAX_POS_ERROR_M", 0.030)
    max_rot = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_PREVIEW_MAX_ROT_ERROR_DEG", 14.0)
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
        "success": bool(status_ok and pos_ok and rot_ok),
        "raw": safe,
    }


def _preview_pose(label: str, side: str, pos: list[float], rpy: list[float], *, gripper: float | None = None) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        _required_tool("freespace_move"),
        _env_float("OPENFORGE_CHOCOLATE_HANDOFF_PREVIEW_TIMEOUT_S", 25.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pos, rpy, gripper=gripper),
    )
    if not result.get("ok"):
        return {"label": label, "success": False, "reason": result.get("error"), "executed": False}
    summary = _preview_summary(result.get("data"))
    summary["label"] = label
    summary["side"] = side
    summary["target_pos"] = _round_list(pos, limit=3)
    summary["target_rpy"] = _round_list(rpy, digits=4, limit=3)
    return summary


def _preview_bimanual(label: str, left_pos: list[float], left_rpy: list[float], right_pos: list[float], right_rpy: list[float]) -> dict[str, Any]:
    kwargs = _move_kwargs("left", left_pos, left_rpy, gripper=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_LEFT_HOLD_GRIPPER", 0.25))
    kwargs.update(_move_kwargs("right", right_pos, right_rpy, gripper=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_OPEN_GRIPPER", 0.70)))
    result = call_with_timeout(
        f"preview:{label}",
        _required_tool("freespace_move"),
        _env_float("OPENFORGE_CHOCOLATE_HANDOFF_PREVIEW_TIMEOUT_S", 25.0),
        run_in_background=_tool("run_in_background"),
        **kwargs,
    )
    if not result.get("ok"):
        return {"label": label, "success": False, "reason": result.get("error"), "executed": False}
    summary = _preview_summary(result.get("data"))
    summary["label"] = label
    summary["side"] = "bimanual"
    summary["left_target_pos"] = _round_list(left_pos, limit=3)
    summary["right_target_pos"] = _round_list(right_pos, limit=3)
    return summary


def _preview_left_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    pos = candidate["position"]
    rpy = candidate["rpy"]
    approach_z = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_APPROACH_Z_LIFT_M", 0.080)
    lift_z = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_LIFT_Z_M", 0.120)
    open_gripper = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_LEFT_OPEN_GRIPPER", 0.70)
    hold_gripper = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_LEFT_HOLD_GRIPPER", 0.25)
    steps = [
        _preview_pose(f"left_approach_{candidate['index']}", "left", [pos[0], pos[1], pos[2] + approach_z], rpy, gripper=open_gripper),
        _preview_pose(f"left_grasp_{candidate['index']}", "left", pos, rpy, gripper=open_gripper),
        _preview_pose(f"left_lift_{candidate['index']}", "left", [pos[0], pos[1], pos[2] + lift_z], rpy, gripper=hold_gripper),
    ]
    return {"success": all(bool(step.get("success")) for step in steps), "steps": steps}


def _right_receive_rpy_candidates() -> list[list[float]]:
    roll_default = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_ROLL_DEG", 0.0)
    pitch_default = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_PITCH_DEG", 90.0)
    yaw_default = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_YAW_DEG", 0.0)
    rolls = _env_float_list("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_ROLL_CANDIDATES_DEG", [roll_default])
    pitches = _env_float_list("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_PITCH_CANDIDATES_DEG", [pitch_default])
    # The right arm's home/trash-drop orientation is pitch ~= 90, yaw ~= 0.
    # Yaw 90 previewed as IK-infeasible in the first chocolate handoff scout.
    yaws = _env_float_list(
        "OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_YAW_CANDIDATES_DEG",
        _unique_floats([yaw_default, 0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0]),
    )
    return [[roll, pitch, yaw] for roll in rolls for pitch in pitches for yaw in yaws]


def _left_handoff_rpy_candidates(base_rpy: list[float]) -> list[list[float]]:
    raw = os.environ.get("OPENFORGE_CHOCOLATE_HANDOFF_LEFT_RPY_CANDIDATES_DEG")
    if raw and raw.strip():
        candidates: list[list[float]] = []
        for chunk in raw.split(";"):
            parts = []
            for value in chunk.split(","):
                try:
                    parts.append(float(value.strip()))
                except ValueError:
                    continue
            if len(parts) == 3:
                candidates.append(parts)
        if candidates:
            return candidates
    # First try to preserve the pickup orientation. If that is not reachable
    # at handoff, try normal wrist-up YAM handoff orientations.
    return [
        _round_list(base_rpy, digits=4, limit=3),
        [0.0, 90.0, 0.0],
        [0.0, 90.0, -30.0],
        [0.0, 90.0, 30.0],
        [0.0, 90.0, -90.0],
        [0.0, 90.0, 90.0],
    ]


def _make_handoff_and_trash_plan(candidate: dict[str, Any], trash_det: dict[str, Any]) -> dict[str, Any]:
    pos = candidate["position"]
    rpy = candidate["rpy"]
    handoff_x = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_CENTER_X_M", 0.58)
    handoff_y = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_CENTER_Y_M", 0.02)
    handoff_z = max(float(pos[2]) + _env_float("OPENFORGE_CHOCOLATE_HANDOFF_LIFT_Z_M", 0.120), _env_float("OPENFORGE_CHOCOLATE_HANDOFF_CENTER_Z_M", 0.93))
    left_handoff = [handoff_x, handoff_y, handoff_z]
    right_receive_pre = [
        handoff_x,
        handoff_y - _env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_STANDOFF_M", 0.120),
        handoff_z,
    ]
    default_close_y = _env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_CLOSE_Y_M", 0.035)
    right_receive_close_offsets = _env_float_list(
        "OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_CLOSE_Y_CANDIDATES_M",
        _unique_floats([default_close_y, 0.050, 0.060, 0.070, 0.080]),
    )
    right_receive_default = [
        handoff_x,
        handoff_y - default_close_y,
        handoff_z,
    ]
    trash_xyz = _xyz(trash_det.get("position_3d") or trash_det.get("position"))
    if trash_xyz is None:
        raise RuntimeError("trash can detection is missing position_3d")
    drop = [
        trash_xyz[0],
        trash_xyz[1],
        trash_xyz[2] + _env_float("OPENFORGE_CHOCOLATE_HANDOFF_TRASH_DROP_Z_CLEARANCE_M", 0.160),
    ]
    drop_rpy = [
        _env_float("OPENFORGE_CHOCOLATE_HANDOFF_TRASH_DROP_ROLL_DEG", 0.0),
        _env_float("OPENFORGE_CHOCOLATE_HANDOFF_TRASH_DROP_PITCH_DEG", 90.0),
        _env_float("OPENFORGE_CHOCOLATE_HANDOFF_TRASH_DROP_YAW_DEG", 0.0),
    ]
    left_handoff_attempts: list[dict[str, Any]] = []
    selected_left_handoff: dict[str, Any] | None = None
    max_left_rpy_candidates = max(1, _env_int("OPENFORGE_CHOCOLATE_HANDOFF_LEFT_RPY_PREVIEW_LIMIT", 6))
    for index, left_rpy in enumerate(_left_handoff_rpy_candidates(rpy)[:max_left_rpy_candidates], start=1):
        suffix = f"{index}_r{left_rpy[0]:.0f}_p{left_rpy[1]:.0f}_y{left_rpy[2]:.0f}".replace("-", "m")
        left_preview = _preview_pose(
            f"left_handoff_hold_{suffix}",
            "left",
            left_handoff,
            left_rpy,
            gripper=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_LEFT_HOLD_GRIPPER", 0.25),
        )
        attempt = {
            "index": index,
            "left_rpy": _round_list(left_rpy, digits=4, limit=3),
            "left_preview": left_preview,
        }
        left_handoff_attempts.append(attempt)
        if left_preview.get("success"):
            selected_left_handoff = attempt
            break
    receive_attempts: list[dict[str, Any]] = []
    selected_receive: dict[str, Any] | None = None
    max_receive_candidates = max(1, _env_int("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_RECEIVE_PREVIEW_LIMIT", 12))
    selected_left_preview = (selected_left_handoff or {}).get("left_preview")
    selected_left_rpy = (selected_left_handoff or {}).get("left_rpy")
    receive_index = 0
    for close_offset in right_receive_close_offsets:
        right_receive = [handoff_x, handoff_y - float(close_offset), handoff_z]
        for right_rpy in _right_receive_rpy_candidates():
            receive_index += 1
            if receive_index > max_receive_candidates:
                break
            suffix = (
                f"{receive_index}_dy{close_offset:.3f}_r{right_rpy[0]:.0f}_p{right_rpy[1]:.0f}_y{right_rpy[2]:.0f}"
                .replace("-", "m")
                .replace(".", "p")
            )
            right_pre = _preview_pose(
                f"right_receive_pre_{suffix}",
                "right",
                right_receive_pre,
                right_rpy,
                gripper=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_OPEN_GRIPPER", 0.70),
            )
            right_close = _preview_pose(
                f"right_receive_close_{suffix}",
                "right",
                right_receive,
                right_rpy,
                gripper=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_OPEN_GRIPPER", 0.70),
            )
            attempt: dict[str, Any] = {
                "index": receive_index,
                "right_rpy": _round_list(right_rpy, digits=4, limit=3),
                "right_receive_close_y_offset_m": round(float(close_offset), 5),
                "right_receive": _round_list(right_receive, limit=3),
                "right_receive_pre_preview": right_pre,
                "right_receive_close_preview": right_close,
                "right_only_success": bool(right_pre.get("success") and right_close.get("success")),
                "bimanual_pre_preview": None,
                "bimanual_close_preview": None,
                "bimanual_success": False,
            }
            if attempt["right_only_success"] and selected_left_preview and selected_left_preview.get("success") and selected_left_rpy:
                bimanual_pre = _preview_bimanual(f"handoff_receive_pre_{suffix}", left_handoff, selected_left_rpy, right_receive_pre, right_rpy)
                bimanual_close = _preview_bimanual(f"handoff_receive_close_{suffix}", left_handoff, selected_left_rpy, right_receive, right_rpy)
                attempt["bimanual_pre_preview"] = bimanual_pre
                attempt["bimanual_close_preview"] = bimanual_close
                attempt["bimanual_success"] = bool(bimanual_pre.get("success") and bimanual_close.get("success"))
            receive_attempts.append(attempt)
            if attempt["bimanual_success"]:
                selected_receive = attempt
                break
        if selected_receive is not None:
            break
        if receive_index >= max_receive_candidates:
            break
    drop_preview = _preview_pose(
        "right_trash_drop_pose",
        "right",
        drop,
        drop_rpy,
        gripper=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_RIGHT_HOLD_GRIPPER", 0.25),
    )
    selected_previews = []
    if selected_left_preview:
        selected_previews.append(selected_left_preview)
    selected_previews.append(drop_preview)
    if selected_receive is not None:
        selected_previews.extend(
            [
                selected_receive["right_receive_pre_preview"],
                selected_receive["right_receive_close_preview"],
                selected_receive["bimanual_pre_preview"],
                selected_receive["bimanual_close_preview"],
            ]
        )
    return {
        "handoff_left_tcp": _round_list(left_handoff, limit=3),
        "selected_left_handoff": selected_left_handoff,
        "left_handoff_attempts": left_handoff_attempts,
        "handoff_right_receive_pre": _round_list(right_receive_pre, limit=3),
        "handoff_right_receive": (
            _round_list((selected_receive or {}).get("right_receive"), limit=3)
            if selected_receive is not None
            else _round_list(right_receive_default, limit=3)
        ),
        "selected_right_receive": selected_receive,
        "right_receive_attempts": receive_attempts,
        "trash_drop_pose": {"position": _round_list(drop, limit=3), "rpy": _round_list(drop_rpy, limit=3)},
        "trash_detection": trash_det,
        "trash_xy_distance_from_chocolate_m": round(_dist_xy(pos, trash_xyz), 5),
        "previews": selected_previews,
        "success": bool(selected_left_preview and selected_left_preview.get("success") and drop_preview.get("success") and selected_receive is not None),
        "note": "This is a controlled release/drop plan, not a ballistic throw.",
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result: dict[str, Any] = {
        "schema": "openforge.chocolate_handoff_trash.no_motion.v1",
        "success": False,
        "physical_motion_executed": False,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "source_snapshot": None,
        "anygrasp_health": None,
        "scene_observation": None,
        "selected_chocolate_detection": None,
        "selected_source_container_detection": None,
        "selected_trash_detection": None,
        "anygrasp": None,
        "selected_anygrasp_candidate": None,
        "handoff_and_trash_plan": None,
        "why_stopped": None,
        "next_recommendation": None,
    }
    try:
        result["source_snapshot"] = _source_snapshot(run_dir)
        if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
            result["why_stopped"] = "refusing physical gate; this planner is no-motion only"
            result["next_recommendation"] = "Rerun without OPENFORGE_ALLOW_PHYSICAL_MOTION."
            return result

        result["anygrasp_health"] = _anygrasp_health()
        if result["anygrasp_health"].get("status") not in {"ok", "healthy"}:
            result["why_stopped"] = "real AnyGrasp health check failed"
            result["next_recommendation"] = "Fix AnyGrasp service before planning chocolate grasp."
            return result
        if result["anygrasp_health"].get("mock") or "mock" in str(result["anygrasp_health"]).lower():
            result["why_stopped"] = "refusing mock AnyGrasp for YAM planning"
            result["next_recommendation"] = (
                "Start an authorized external AnyGrasp service and configure "
                "ANYGRASP_SERVICE_URL."
            )
            return result

        scene = capture_scene(
            prompts=PROMPTS,
            cameras=CAMERAS,
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="pre_anygrasp_chocolate",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_CHOCOLATE_HANDOFF_OBS_TIMEOUT_S", 30.0),
            max_retries=1,
            image_only_cameras=tuple(IMAGE_ONLY_CAMERAS),
        )
        result["scene_observation"] = scene
        chocolate = _best_detection(scene, ("chocolate", "chocolate bar"))
        source_container = _best_detection(scene, ("white bowl",))
        trash = _best_detection(scene, ("white trash can", "trash can", "bin"))
        result["selected_chocolate_detection"] = chocolate
        result["selected_source_container_detection"] = source_container
        result["selected_trash_detection"] = trash
        if chocolate is None or trash is None:
            result["why_stopped"] = "missing chocolate or trash can detection"
            result["next_recommendation"] = "Adjust prompts or scene placement, then rerun no-motion observation."
            return result

        anygrasp = _sample_anygrasp()
        result["anygrasp"] = anygrasp
        write_json(run_dir / "plans" / "anygrasp_candidates.json", anygrasp)
        if not anygrasp.get("success"):
            result["why_stopped"] = f"AnyGrasp returned no usable candidates: {anygrasp.get('error')}"
            result["next_recommendation"] = "Try prompt 'chocolate' vs 'chocolate bar', inspect SAM3 mask and AnyGrasp overlay."
            return result

        initial_preview_limit = min(len(anygrasp["candidates"]), _env_int("OPENFORGE_CHOCOLATE_HANDOFF_PREVIEW_LIMIT", 8))
        preview_limit = initial_preview_limit
        for candidate in anygrasp["candidates"][:preview_limit]:
            candidate["left_pick_preview"] = _preview_left_candidate(candidate)
        previewed = [c for c in anygrasp["candidates"][:preview_limit] if (c.get("left_pick_preview") or {}).get("success")]
        if not previewed and preview_limit < len(anygrasp["candidates"]):
            for candidate in anygrasp["candidates"][preview_limit:]:
                candidate["left_pick_preview"] = _preview_left_candidate(candidate)
            preview_limit = len(anygrasp["candidates"])
            previewed = [c for c in anygrasp["candidates"] if (c.get("left_pick_preview") or {}).get("success")]
            anygrasp["expanded_preview_after_initial_failure"] = {
                "initial_preview_limit": initial_preview_limit,
                "expanded_preview_limit": preview_limit,
            }
        write_json(run_dir / "plans" / "anygrasp_candidates.json", anygrasp)
        if not previewed:
            result["why_stopped"] = "no AnyGrasp candidate passed left-arm pickup/lift preview"
            result["next_recommendation"] = "Inspect plans/anygrasp_candidates.json and lower/raise approach or IK thresholds."
            return result
        selected = sorted(previewed, key=lambda c: float(c.get("score") or 0.0), reverse=True)[0]
        if source_container is not None:
            src_xyz = _xyz(source_container.get("position_3d") or source_container.get("position"))
            cand_xyz = _xyz(selected.get("position"))
            if src_xyz is not None and cand_xyz is not None:
                selected["source_container_xy_distance_m"] = round(_dist_xy(cand_xyz, src_xyz), 5)
                selected["source_container_z_delta_m"] = round(float(cand_xyz[2]) - float(src_xyz[2]), 5)
                selected["source_container_note"] = (
                    "Chocolate appears to be in/near the white bowl; physical pickup needs a "
                    "bowl-clearance visual gate before close/lift."
                )
        result["selected_anygrasp_candidate"] = selected
        handoff_plan = _make_handoff_and_trash_plan(selected, trash)
        result["handoff_and_trash_plan"] = handoff_plan
        write_json(run_dir / "plans" / "selected_plan.json", result)

        result["success"] = bool(handoff_plan.get("success"))
        result["why_stopped"] = (
            "no-motion AnyGrasp chocolate pickup, handoff staging, and trash drop previews passed"
            if result["success"]
            else "left pickup preview passed, but handoff/trash preview has failures"
        )
        result["next_recommendation"] = (
            "Inspect selected_plan.json and videos. Physical work needs a fresh ticket and should start with left pickup/lift only, then handoff, then controlled trash release."
        )
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and service logs before retrying."
        return result
    finally:
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## AnyGrasp Chocolate Handoff Trash No-Motion Plan",
                f"- success: {result.get('success')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- chocolate_detection: {result.get('selected_chocolate_detection')}",
                f"- source_container_detection: {result.get('selected_source_container_detection')}",
                f"- trash_detection: {result.get('selected_trash_detection')}",
                f"- anygrasp_health: {result.get('anygrasp_health')}",
                f"- selected_anygrasp_candidate: {result.get('selected_anygrasp_candidate')}",
                f"- handoff_and_trash_plan_success: {((result.get('handoff_and_trash_plan') or {}).get('success'))}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- next_recommendation: {result.get('next_recommendation')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
