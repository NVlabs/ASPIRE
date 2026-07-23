# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick up the cyan cup and place it on the top shelf of the wooden rack.

The script is standalone:
- localize the cyan cup and wooden rack from fresh perception before motion;
- treat the cup as a simple cylinder and force a clean top-down grasp;
- lift, transport above the rack top shelf, lower to a conservative release
  height, open, retreat, then verify from post-action perception.

Default behavior is observe/preview only. Physical execution requires
``OPENFORGE_ALLOW_PHYSICAL_MOTION=1`` plus the cup-rack ticket envs checked in
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

from skill_library.simple_grasp_lift_loop import _side_grasp_display_rpy_from_approach

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


TASK_NAME = "pick_cyan_cup_place_on_wooden_rack_top_shelf_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}


def _set_default_cup_rack_env() -> None:
    defaults = {
        "OPENFORGE_CUP_RACK_CUP_PROMPTS": "cyan cup,blue cup,cup,plastic cup,empty cup,teal cup",
        "OPENFORGE_CUP_RACK_RACK_PROMPTS": "wooden rack,wood rack,wooden shelf,rack,shelf,wooden stand",
        "OPENFORGE_CUP_RACK_CAMERAS": "top,left,right,bottom",
        "OPENFORGE_CUP_RACK_MOTION_CAMERAS": "top,left,right",
        "OPENFORGE_CUP_RACK_IMAGE_ONLY_CAMERAS": "bottom",
        "OPENFORGE_CUP_RACK_OBS_TIMEOUT_S": "30.0",
        "OPENFORGE_CUP_RACK_RECORDING_WAIT_S": "3.0",
        "OPENFORGE_CUP_RACK_CUP_MIN_SCORE": "0.20",
        "OPENFORGE_CUP_RACK_RACK_MIN_SCORE": "0.18",
        "OPENFORGE_CUP_RACK_CUP_X_MIN_M": "0.300",
        "OPENFORGE_CUP_RACK_CUP_X_MAX_M": "0.850",
        "OPENFORGE_CUP_RACK_CUP_Y_MIN_M": "-0.150",
        "OPENFORGE_CUP_RACK_CUP_Y_MAX_M": "0.450",
        "OPENFORGE_CUP_RACK_CUP_Z_MIN_M": "0.700",
        "OPENFORGE_CUP_RACK_CUP_Z_MAX_M": "0.880",
        "OPENFORGE_CUP_RACK_RACK_PRIMARY_CAMERAS": "top",
        "OPENFORGE_CUP_RACK_REQUIRE_PRIMARY_RACK_CAMERA": "1",
        "OPENFORGE_CUP_RACK_ARM": "left",
        "OPENFORGE_CUP_RACK_TOPDOWN_YAWS_DEG": "0,90,-90,45,-45",
        "OPENFORGE_CUP_RACK_TOPDOWN_Z_OFFSETS_M": "0.035,0.020,0.050",
        "OPENFORGE_CUP_RACK_ENABLE_SIDE_GRASP": "1",
        "OPENFORGE_CUP_RACK_SIDE_APPROACH_OFFSETS_DEG": "0,-20,20,-35,35",
        "OPENFORGE_CUP_RACK_SIDE_GRASP_Z_ABOVE_DETECTION_M": "0.060",
        "OPENFORGE_CUP_RACK_SIDE_PREGRASP_STANDOFF_M": "0.100",
        "OPENFORGE_CUP_RACK_SIDE_RELEASE_Z_M": "0.990",
        "OPENFORGE_CUP_RACK_GRASP_Z_BIAS_M": "0.000",
        "OPENFORGE_CUP_RACK_APPROACH_Z_M": "0.080",
        "OPENFORGE_CUP_RACK_LIFT_Z_M": "0.100",
        "OPENFORGE_CUP_RACK_PLACE_APPROACH_Z_M": "0.140",
        "OPENFORGE_CUP_RACK_RETREAT_Z_M": "0.160",
        "OPENFORGE_CUP_RACK_USE_RACK_Z_FOR_RELEASE": "0",
        "OPENFORGE_CUP_RACK_FIXED_RELEASE_Z_M": "0.970",
        "OPENFORGE_CUP_RACK_RELEASE_GRASP_ABOVE_SHELF_M": "0.050",
        "OPENFORGE_CUP_RACK_RELEASE_CLEARANCE_M": "0.010",
        "OPENFORGE_CUP_RACK_PLACE_X_BIAS_M": "-0.015",
        "OPENFORGE_CUP_RACK_PLACE_Y_BIAS_M": "0.075",
        "OPENFORGE_CUP_RACK_DEFAULT_CUP_RADIUS_M": "0.045",
        "OPENFORGE_CUP_RACK_DEFAULT_CUP_HALF_HEIGHT_M": "0.060",
        "OPENFORGE_CUP_RACK_GRIPPER_WIDTH_MARGIN_M": "0.012",
        "OPENFORGE_CUP_RACK_OPEN_GRIPPER_POS": "1.0",
        "OPENFORGE_CUP_RACK_HOLD_GRIPPER_POS": "0.24",
        "OPENFORGE_CUP_RACK_CLOSE_TARGET_POS": "0.08",
        "OPENFORGE_CUP_RACK_CLOSE_STEPS": "0.08",
        "OPENFORGE_CUP_RACK_CLOSE_VEL_LIMIT": "0.75",
        "OPENFORGE_CUP_RACK_CLOSE_TORQUE_LIMIT": "0.60",
        "OPENFORGE_CUP_RACK_CLOSE_CONFIRM_TIMEOUT_S": "2.0",
        "OPENFORGE_CUP_RACK_CLOSE_CONTACT_MIN_DELTA_POS": "0.015",
        "OPENFORGE_CUP_RACK_CLOSE_TARGET_TOLERANCE_POS": "0.08",
        "OPENFORGE_CUP_RACK_HOLD_MIN_POS": "0.010",
        "OPENFORGE_CUP_RACK_HOLD_MAX_POS": "0.98",
        "OPENFORGE_CUP_RACK_REQUIRE_CONTACT_BEFORE_LIFT": "1",
        "OPENFORGE_CUP_RACK_GRIPPER_CONFIRM_TIMEOUT_S": "4.0",
        "OPENFORGE_CUP_RACK_OPEN_CONFIRM_POS": "0.75",
        "OPENFORGE_CUP_RACK_STATE_POLL_S": "0.15",
        "OPENFORGE_CUP_RACK_MAX_CANDIDATES": "100",
        "OPENFORGE_CUP_RACK_PREVIEW_LIMIT": "100",
        "OPENFORGE_CUP_RACK_PREVIEW_TIMEOUT_S": "25.0",
        "OPENFORGE_CUP_RACK_PREVIEW_MAX_POS_ERROR_M": "0.030",
        "OPENFORGE_CUP_RACK_PREVIEW_MAX_ROT_ERROR_DEG": "16.0",
        "OPENFORGE_CUP_RACK_PLANNER_BACKEND": "rrtconnect",
        "OPENFORGE_CUP_RACK_SOLVER_SPEED": "fast",
        "OPENFORGE_CUP_RACK_PLANNING_SPEED": "0.30",
        "OPENFORGE_CUP_RACK_IK_ERROR_THRESHOLD_M": "0.025",
        "OPENFORGE_CUP_RACK_IK_ROT_THRESHOLD_DEG": "14.0",
        "OPENFORGE_CUP_RACK_IK_XYZ_WEIGHT": "1.0",
        "OPENFORGE_CUP_RACK_IK_RPY_WEIGHT": "0.35",
        "OPENFORGE_CUP_RACK_SUCCESS_MAX_CUP_TO_RACK_XY_M": "0.090",
        "OPENFORGE_CUP_RACK_SUCCESS_MIN_SOURCE_XY_DELTA_M": "0.040",
        "OPENFORGE_CUP_RACK_SUCCESS_MIN_POST_Z_M": "0.850",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


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


def _cup_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_CUP_RACK_CUP_PROMPTS",
        ["cyan cup", "blue cup", "cup", "plastic cup", "empty cup", "teal cup"],
    )


def _rack_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_CUP_RACK_RACK_PROMPTS",
        ["wooden rack", "wood rack", "wooden shelf", "rack", "shelf", "wooden stand"],
    )


def _all_prompts() -> list[str]:
    prompts: list[str] = []
    for prompt in _cup_prompts() + _rack_prompts() + _csv("OPENFORGE_CUP_RACK_EXTRA_PROMPTS", []):
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts


def _cameras() -> list[str]:
    return _csv("OPENFORGE_CUP_RACK_CAMERAS", ["top", "left", "right", "bottom"])


def _image_only_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_CUP_RACK_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _motion_cameras() -> tuple[str, ...]:
    return tuple(camera for camera in _csv("OPENFORGE_CUP_RACK_MOTION_CAMERAS", ["top", "left", "right"]) if camera)


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_name = globals().get("__file__") or f"cap/saved_scripts/{TASK_NAME}.py"
    script_path = Path(str(script_name)).resolve()
    snapshot: dict[str, Any] = {
        "schema": "openforge.cup_rack.source_snapshot.v1",
        "script": str(script_path),
        "env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("OPENFORGE_CUP_RACK_")
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
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_CUP_RACK_RECORDING_WAIT_S", 3.0))
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


def _cup_geometry_score(det: dict[str, Any]) -> float:
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


def _select_detection(
    scene: dict[str, Any],
    prompts: list[str],
    *,
    kind: str,
    enforce_source_bounds: bool = True,
) -> dict[str, Any] | None:
    candidates = _detections(scene, prompts)
    min_score = _env_float(f"OPENFORGE_CUP_RACK_{kind.upper()}_MIN_SCORE", 0.20)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for det in candidates:
        score = float(det.get("score") or min_score)
        if score < min_score:
            continue
        xyz = _xyz(det.get("position_3d") or det.get("position"))
        if xyz is None:
            continue
        if kind == "cup" and enforce_source_bounds:
            if not (
                _env_float("OPENFORGE_CUP_RACK_CUP_X_MIN_M", 0.300)
                <= float(xyz[0])
                <= _env_float("OPENFORGE_CUP_RACK_CUP_X_MAX_M", 0.850)
                and _env_float("OPENFORGE_CUP_RACK_CUP_Y_MIN_M", -0.150)
                <= float(xyz[1])
                <= _env_float("OPENFORGE_CUP_RACK_CUP_Y_MAX_M", 0.450)
                and _env_float("OPENFORGE_CUP_RACK_CUP_Z_MIN_M", 0.700)
                <= float(xyz[2])
                <= _env_float("OPENFORGE_CUP_RACK_CUP_Z_MAX_M", 0.880)
            ):
                continue
        prompt = str(det.get("prompt") or "")
        rank = score - 0.04 * _prompt_rank(prompt, prompts) - 0.01 * _camera_rank(det.get("source_camera"))
        if kind == "cup":
            rank += _cup_geometry_score(det)
        if kind == "rack":
            primary = set(_csv("OPENFORGE_CUP_RACK_RACK_PRIMARY_CAMERAS", ["top"]))
            source_camera = str(det.get("source_camera") or "")
            if source_camera in primary:
                rank += 0.10
            elif _truthy_env("OPENFORGE_CUP_RACK_REQUIRE_PRIMARY_RACK_CAMERA", True):
                continue
        ranked.append((rank, det))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = dict(ranked[0][1])
    selected["selection_rank_score"] = round(float(ranked[0][0]), 5)
    selected["selection_kind"] = kind
    return selected


def _arms_from_cup(cup_xyz: list[float]) -> list[str]:
    requested = os.environ.get("OPENFORGE_CUP_RACK_ARM", "auto").strip().lower()
    if requested in {"left", "right"}:
        return [requested]
    preferred = "left" if float(cup_xyz[1]) > 0.04 else "right"
    alternate = "right" if preferred == "left" else "left"
    return [preferred, alternate]


def _estimate_radius(det: dict[str, Any]) -> float:
    half_extents = _half_extents(det)
    if len(half_extents) >= 2:
        xy_vals = sorted(abs(float(v)) for v in half_extents[:2])
        if xy_vals[0] > 0.005:
            return max(0.018, min(0.060, xy_vals[0]))
    return _env_float("OPENFORGE_CUP_RACK_DEFAULT_CUP_RADIUS_M", 0.045)


def _estimate_half_height(det: dict[str, Any]) -> float:
    half_extents = _half_extents(det)
    if len(half_extents) >= 3 and abs(float(half_extents[2])) > 0.01:
        return max(0.025, min(0.120, abs(float(half_extents[2]))))
    return _env_float("OPENFORGE_CUP_RACK_DEFAULT_CUP_HALF_HEIGHT_M", 0.060)


def _release_z(rack_det: dict[str, Any], cup_half_height: float, grasp_z: float) -> float:
    fixed_release_z = _env_float_or_none("OPENFORGE_CUP_RACK_FIXED_RELEASE_Z_M")
    if fixed_release_z is not None and not _truthy_env("OPENFORGE_CUP_RACK_USE_RACK_Z_FOR_RELEASE", False):
        return float(fixed_release_z)
    rack_xyz = _xyz(rack_det.get("position_3d") or rack_det.get("position")) or [0.0, 0.0, grasp_z]
    return (
        float(rack_xyz[2])
        + _env_float("OPENFORGE_CUP_RACK_RELEASE_GRASP_ABOVE_SHELF_M", 0.050)
        + _env_float("OPENFORGE_CUP_RACK_RELEASE_CLEARANCE_M", 0.010)
    )


def _make_candidate(
    *,
    index: int,
    arm: str,
    cup_det: dict[str, Any],
    rack_det: dict[str, Any],
    yaw_deg: float,
    z_offset_m: float,
) -> dict[str, Any]:
    cup_xyz = _xyz(cup_det.get("position_3d") or cup_det.get("position"))
    rack_xyz = _xyz(rack_det.get("position_3d") or rack_det.get("position"))
    if cup_xyz is None or rack_xyz is None:
        raise ValueError("cup and rack detections must have 3D positions")
    cup_half_height = _estimate_half_height(cup_det)
    cup_radius = _estimate_radius(cup_det)
    grasp_z = float(cup_xyz[2]) + _env_float("OPENFORGE_CUP_RACK_GRASP_Z_BIAS_M", 0.000) + float(z_offset_m)
    release_z = _release_z(rack_det, cup_half_height, grasp_z)
    approach_z = _env_float("OPENFORGE_CUP_RACK_APPROACH_Z_M", 0.090)
    lift_z = _env_float("OPENFORGE_CUP_RACK_LIFT_Z_M", 0.050)
    place_approach_z = _env_float("OPENFORGE_CUP_RACK_PLACE_APPROACH_Z_M", 0.080)
    retreat_z = _env_float("OPENFORGE_CUP_RACK_RETREAT_Z_M", 0.080)
    rpy = [0.0, _env_float("OPENFORGE_CUP_RACK_TOPDOWN_PITCH_DEG", 180.0), float(yaw_deg)]
    place_xy_bias = [
        _env_float("OPENFORGE_CUP_RACK_PLACE_X_BIAS_M", 0.0),
        _env_float("OPENFORGE_CUP_RACK_PLACE_Y_BIAS_M", 0.0),
    ]
    rack_xy = [float(rack_xyz[0]) + place_xy_bias[0], float(rack_xyz[1]) + place_xy_bias[1]]
    lift_abs_z = grasp_z + lift_z
    pre_place_z = max(lift_abs_z, release_z + place_approach_z)
    cup_xy = [float(cup_xyz[0]), float(cup_xyz[1])]
    width = max(0.02, min(0.95, 2.0 * cup_radius + _env_float("OPENFORGE_CUP_RACK_GRIPPER_WIDTH_MARGIN_M", 0.012)))
    return {
        "index": int(index),
        "schema": "openforge.cup_rack.candidate.v1",
        "arm": arm,
        "strategy": "top_down_cyan_cup_to_wooden_rack_top_shelf",
        "score": round(1.0 - 0.03 * index - 0.4 * abs(float(z_offset_m)), 5),
        "rpy": _round_list(rpy, digits=4, limit=3),
        "width": round(float(width), 5),
        "cup_radius_m": round(float(cup_radius), 5),
        "cup_half_height_m": round(float(cup_half_height), 5),
        "release_z_m": round(float(release_z), 5),
        "topdown_yaw_deg": round(float(yaw_deg), 4),
        "z_offset_m": round(float(z_offset_m), 5),
        "place_xy_bias_m": _round_list(place_xy_bias, limit=2),
        "source_cup_detection": cup_det,
        "source_rack_detection": rack_det,
        "poses": {
            "pregrasp": {"position": [cup_xy[0], cup_xy[1], grasp_z + approach_z], "rpy": rpy},
            "grasp": {"position": [cup_xy[0], cup_xy[1], grasp_z], "rpy": rpy},
            "lift": {"position": [cup_xy[0], cup_xy[1], lift_abs_z], "rpy": rpy},
            "pre_place": {"position": [rack_xy[0], rack_xy[1], pre_place_z], "rpy": rpy},
            "place": {"position": [rack_xy[0], rack_xy[1], release_z], "rpy": rpy},
            "retreat": {"position": [rack_xy[0], rack_xy[1], release_z + retreat_z], "rpy": rpy},
        },
        "geometry_policy": (
            "Cup is treated as a cylinder; top-down gripper opening yaw is swept, "
            "rack placement is based on fresh wooden-rack detection plus optional XY bias. "
            "Release height defaults to a fixed top-shelf-safe Z unless explicitly configured "
            "to use detected rack Z."
        ),
    }


def _make_side_candidate(
    *,
    index: int,
    arm: str,
    cup_det: dict[str, Any],
    rack_det: dict[str, Any],
    approach_angle_deg: float,
) -> dict[str, Any]:
    cup_xyz = _xyz(cup_det.get("position_3d") or cup_det.get("position"))
    rack_xyz = _xyz(rack_det.get("position_3d") or rack_det.get("position"))
    if cup_xyz is None or rack_xyz is None:
        raise ValueError("cup and rack detections must have 3D positions")
    cup_radius = _estimate_radius(cup_det)
    angle_rad = math.radians(float(approach_angle_deg))
    approach = [math.cos(angle_rad), math.sin(angle_rad), 0.0]
    rpy, axes = _side_grasp_display_rpy_from_approach(approach, wrist_roll_deg=0.0)
    grasp_z = float(cup_xyz[2]) + _env_float("OPENFORGE_CUP_RACK_SIDE_GRASP_Z_ABOVE_DETECTION_M", 0.060)
    grasp = [float(cup_xyz[0]), float(cup_xyz[1]), grasp_z]
    standoff = _env_float("OPENFORGE_CUP_RACK_SIDE_PREGRASP_STANDOFF_M", 0.100)
    pregrasp = [grasp[0] - approach[0] * standoff, grasp[1] - approach[1] * standoff, grasp[2]]
    lift = [grasp[0], grasp[1], grasp[2] + _env_float("OPENFORGE_CUP_RACK_LIFT_Z_M", 0.100)]
    rack_xy = [
        float(rack_xyz[0]) + _env_float("OPENFORGE_CUP_RACK_PLACE_X_BIAS_M", 0.0),
        float(rack_xyz[1]) + _env_float("OPENFORGE_CUP_RACK_PLACE_Y_BIAS_M", 0.0),
    ]
    release_z = _env_float("OPENFORGE_CUP_RACK_SIDE_RELEASE_Z_M", _env_float("OPENFORGE_CUP_RACK_FIXED_RELEASE_Z_M", 0.990))
    pre_place_z = max(lift[2], release_z + _env_float("OPENFORGE_CUP_RACK_PLACE_APPROACH_Z_M", 0.140))
    width = max(0.02, min(0.95, 2.0 * cup_radius + _env_float("OPENFORGE_CUP_RACK_GRIPPER_WIDTH_MARGIN_M", 0.012)))
    return {
        "index": int(index),
        "schema": "openforge.cup_rack.candidate.v1",
        "arm": arm,
        "strategy": "side_body_cyan_cup_to_wooden_rack_top_shelf",
        "score": round(0.72 - 0.03 * index - 0.001 * abs(float(approach_angle_deg)), 5),
        "rpy": _round_list(rpy, digits=4, limit=3),
        "width": round(float(width), 5),
        "cup_radius_m": round(float(cup_radius), 5),
        "release_z_m": round(float(release_z), 5),
        "approach_angle_deg": round(float(approach_angle_deg), 4),
        "approach_direction_world": _round_list(approach, limit=3),
        "gripper_local_axes_world": axes,
        "place_xy_bias_m": [
            _env_float("OPENFORGE_CUP_RACK_PLACE_X_BIAS_M", 0.0),
            _env_float("OPENFORGE_CUP_RACK_PLACE_Y_BIAS_M", 0.0),
        ],
        "source_cup_detection": cup_det,
        "source_rack_detection": rack_det,
        "poses": {
            "pregrasp": {"position": _round_list(pregrasp, limit=3), "rpy": rpy},
            "grasp": {"position": _round_list(grasp, limit=3), "rpy": rpy},
            "lift": {"position": _round_list(lift, limit=3), "rpy": rpy},
            "pre_place": {"position": [rack_xy[0], rack_xy[1], pre_place_z], "rpy": rpy},
            "place": {"position": [rack_xy[0], rack_xy[1], release_z], "rpy": rpy},
            "retreat": {
                "position": [rack_xy[0], rack_xy[1], release_z + _env_float("OPENFORGE_CUP_RACK_RETREAT_Z_M", 0.160)],
                "rpy": rpy,
            },
        },
        "geometry_policy": (
            "Fallback side grasp: local +Z approaches the cup horizontally, "
            "the cup remains upright while held around its body, and release "
            "uses a high shelf-safe side-release Z."
        ),
    }


def _generate_candidates(cup_det: dict[str, Any], rack_det: dict[str, Any]) -> list[dict[str, Any]]:
    cup_xyz = _xyz(cup_det.get("position_3d") or cup_det.get("position"))
    if cup_xyz is None:
        return []
    arms = _arms_from_cup(cup_xyz)
    yaws = _float_csv("OPENFORGE_CUP_RACK_TOPDOWN_YAWS_DEG", [0.0, 90.0, -90.0, 45.0, -45.0])
    z_offsets = _float_csv("OPENFORGE_CUP_RACK_TOPDOWN_Z_OFFSETS_M", [0.035, 0.020, 0.050])
    max_candidates = max(1, _env_int("OPENFORGE_CUP_RACK_MAX_CANDIDATES", 48))
    candidates: list[dict[str, Any]] = []
    idx = 1
    for z_offset in z_offsets:
        for yaw in yaws:
            for arm in arms:
                candidates.append(
                    _make_candidate(
                        index=idx,
                        arm=arm,
                        cup_det=cup_det,
                        rack_det=rack_det,
                        yaw_deg=float(yaw),
                        z_offset_m=float(z_offset),
                    )
                )
                idx += 1
                if len(candidates) >= max_candidates:
                    return candidates
    if _truthy_env("OPENFORGE_CUP_RACK_ENABLE_SIDE_GRASP", True):
        for arm in arms:
            base_angle = -90.0 if arm == "left" else 90.0
            for offset in _float_csv("OPENFORGE_CUP_RACK_SIDE_APPROACH_OFFSETS_DEG", [0.0, -20.0, 20.0]):
                candidates.append(
                    _make_side_candidate(
                        index=idx,
                        arm=arm,
                        cup_det=cup_det,
                        rack_det=rack_det,
                        approach_angle_deg=base_angle + float(offset),
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
        "planner_backend": os.environ.get("OPENFORGE_CUP_RACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_CUP_RACK_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_CUP_RACK_PLANNING_SPEED", 0.30),
        "ik_error_threshold": _env_float("OPENFORGE_CUP_RACK_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_CUP_RACK_IK_ROT_THRESHOLD_DEG", 14.0),
        "ik_xyz_weight": _env_float("OPENFORGE_CUP_RACK_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_CUP_RACK_IK_RPY_WEIGHT", 0.35),
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
    max_pos = _env_float("OPENFORGE_CUP_RACK_PREVIEW_MAX_POS_ERROR_M", 0.030)
    max_rot = _env_float("OPENFORGE_CUP_RACK_PREVIEW_MAX_ROT_ERROR_DEG", 16.0)
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
        _env_float("OPENFORGE_CUP_RACK_PREVIEW_TIMEOUT_S", 25.0),
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
    open_gripper_pos = _env_float("OPENFORGE_CUP_RACK_OPEN_GRIPPER_POS", 1.0)
    hold_gripper_pos = _env_float("OPENFORGE_CUP_RACK_HOLD_GRIPPER_POS", 0.20)
    steps = [
        _preview_pose("cup_pregrasp", arm, poses["pregrasp"], gripper=open_gripper_pos),
        _preview_pose("cup_grasp", arm, poses["grasp"], gripper=open_gripper_pos),
        _preview_pose("cup_lift", arm, poses["lift"], gripper=hold_gripper_pos),
        _preview_pose("cup_pre_place", arm, poses["pre_place"], gripper=hold_gripper_pos),
        _preview_pose("cup_place", arm, poses["place"], gripper=hold_gripper_pos),
        _preview_pose("cup_retreat_after_release", arm, poses["retreat"], gripper=open_gripper_pos),
    ]
    return {
        "schema": "openforge.cup_rack.preview_sequence.v1",
        "success": all(bool(step.get("success")) for step in steps),
        "steps": steps,
    }


def _preview_and_select(candidates: list[dict[str, Any]], run_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    limit = min(len(candidates), max(1, _env_int("OPENFORGE_CUP_RACK_PREVIEW_LIMIT", 24)))
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
        "schema": "openforge.cup_rack.preview_rank.v1",
        "candidate_count": len(candidates),
        "previewed_count": len(previews),
        "selected": selected,
        "previews": previews,
    }
    packet["path"] = write_json(run_dir / "plans" / "cup_rack_sequence_preview.json", packet)
    return selected, previews


def _physical_ticket_status() -> dict[str, Any]:
    required = {
        "OPENFORGE_CUP_RACK_PHYSICAL_TICKET": "1",
        "OPENFORGE_CUP_RACK_OPERATOR_PRESENT": "1",
        "OPENFORGE_CUP_RACK_ESTOP_REACHABLE": "1",
        "OPENFORGE_CUP_RACK_WORKSPACE_CLEAR": "1",
        "OPENFORGE_CUP_RACK_HANDS_TOOLS_CABLES_CLEAR": "1",
        "OPENFORGE_CUP_RACK_FOLLOWER_SERVERS_UP": "1",
        "OPENFORGE_CUP_RACK_NO_OTHER_MOTION_PRECHECK": "1",
        "OPENFORGE_CUP_RACK_RECORDING_DEBUG_CONFIRMED": "1",
        "OPENFORGE_CUP_RACK_CUP_VISIBLE": "1",
        "OPENFORGE_CUP_RACK_WOODEN_RACK_VISIBLE": "1",
        "OPENFORGE_CUP_RACK_TOP_SHELF_CLEAR": "1",
        "OPENFORGE_CUP_RACK_NO_MANUAL_XYZ": "1",
    }
    missing = [name for name, expected in required.items() if os.environ.get(name) != expected]
    return {"passed": not missing, "missing_or_invalid": missing, "required": required}


def _set_gripper_checked(arm: str, target: float, stage: str, run_dir: Path) -> dict[str, Any]:
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    before = read_gripper_width(get_robot_state, arm)
    kwargs: dict[str, float] = {}
    vel = _env_float_or_none("OPENFORGE_CUP_RACK_GRIPPER_VEL_LIMIT")
    torque = _env_float_or_none("OPENFORGE_CUP_RACK_GRIPPER_TORQUE_LIMIT")
    if vel is not None:
        kwargs["vel_limit"] = vel
    if torque is not None:
        kwargs["torque_limit"] = torque
    command = set_gripper(arm, float(target), **kwargs)
    after = read_gripper_width(get_robot_state, arm)
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_CUP_RACK_GRIPPER_CONFIRM_TIMEOUT_S", 4.0))
    confirm_min = _env_float("OPENFORGE_CUP_RACK_OPEN_CONFIRM_POS", 0.75)
    if target >= confirm_min:
        while after is not None and after < confirm_min and time.monotonic() < deadline:
            time.sleep(max(0.02, _env_float("OPENFORGE_CUP_RACK_STATE_POLL_S", 0.15)))
            after = read_gripper_width(get_robot_state, arm)
    packet = {
        "schema": "openforge.cup_rack.set_gripper.v1",
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
    packet = {"schema": "openforge.cup_rack.execute_pose.v1", "stage": label, "arm": arm, "preview": preview, "execute": json_safe(executed)}
    packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
    return packet


def _verify_post_place(
    run_dir: Path,
    pre_cup: dict[str, Any] | None,
    pre_rack: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = capture_scene(
        prompts=_all_prompts(),
        cameras=_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="post_cup_rack_place",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_CUP_RACK_OBS_TIMEOUT_S", 30.0),
        max_retries=1,
        motion_cameras=_motion_cameras(),
        image_only_cameras=_image_only_cameras(),
    )
    post_cup = _select_detection(scene, _cup_prompts(), kind="cup", enforce_source_bounds=False)
    post_rack = _select_detection(scene, _rack_prompts(), kind="rack") or pre_rack
    pre_cup_xyz = _xyz((pre_cup or {}).get("position_3d") or (pre_cup or {}).get("position"))
    post_cup_xyz = _xyz((post_cup or {}).get("position_3d") or (post_cup or {}).get("position"))
    rack_xyz = _xyz((post_rack or {}).get("position_3d") or (post_rack or {}).get("position"))
    packet: dict[str, Any] = {
        "schema": "openforge.cup_rack.post_verify.v1",
        "success": False,
        "confidence": 0.0,
        "post_scene": scene,
        "pre_cup_xyz": pre_cup_xyz,
        "post_cup_detection": post_cup,
        "post_cup_xyz": post_cup_xyz,
        "rack_detection": post_rack,
        "rack_xyz": rack_xyz,
        "cup_to_rack_xy_m": None,
        "cup_from_source_xy_m": None,
        "post_cup_z_m": None,
        "why": None,
    }
    if post_cup_xyz is None:
        packet["why"] = "post observation did not find the cup; inspect videos before counting success"
    elif rack_xyz is None:
        packet["why"] = "post observation did not find the wooden rack target"
    else:
        cup_to_rack = _dist_xy(post_cup_xyz, rack_xyz)
        packet["cup_to_rack_xy_m"] = round(float(cup_to_rack), 5)
        packet["post_cup_z_m"] = round(float(post_cup_xyz[2]), 5)
        if pre_cup_xyz is not None:
            packet["cup_from_source_xy_m"] = round(float(_dist_xy(post_cup_xyz, pre_cup_xyz)), 5)
        threshold = _env_float("OPENFORGE_CUP_RACK_SUCCESS_MAX_CUP_TO_RACK_XY_M", 0.090)
        moved_min = _env_float("OPENFORGE_CUP_RACK_SUCCESS_MIN_SOURCE_XY_DELTA_M", 0.040)
        min_post_z = _env_float("OPENFORGE_CUP_RACK_SUCCESS_MIN_POST_Z_M", 0.850)
        moved_ok = packet["cup_from_source_xy_m"] is None or float(packet["cup_from_source_xy_m"]) >= moved_min
        height_ok = float(post_cup_xyz[2]) >= min_post_z
        packet["success"] = bool(cup_to_rack <= threshold and moved_ok and height_ok)
        packet["confidence"] = 0.85 if packet["success"] else 0.25
        packet["why"] = (
            "post cup detection is centered near the wooden rack and above shelf-height threshold"
            if packet["success"]
            else "post cup detection is not close enough to the rack, not high enough, or did not move from source"
        )
    packet["path"] = write_json(run_dir / "plans" / "post_cup_rack_place_verify.json", packet)
    return packet


def _execute_physical(
    selected: dict[str, Any],
    run_dir: Path,
    pre_cup: dict[str, Any] | None,
    pre_rack: dict[str, Any] | None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.cup_rack.physical_attempt.v1",
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
    open_pos = _env_float("OPENFORGE_CUP_RACK_OPEN_GRIPPER_POS", 1.0)
    hold_pos = _env_float("OPENFORGE_CUP_RACK_HOLD_GRIPPER_POS", 0.20)
    try:
        packet["steps"].append(_set_gripper_checked(arm, open_pos, "open_before_cup_pick", run_dir))
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        packet["physical_motion_executed"] = True
        if not packet["steps"][-1].get("confirmed"):
            packet["why_stopped"] = "gripper did not confirm open before cup approach"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        for label in ("pregrasp", "grasp"):
            step = _execute_pose(f"cup_{label}", arm, poses[label], run_dir, gripper=open_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append(f"freespace_move:cup_{label}")

        close = staged_close_with_contact(
            side=arm,
            set_gripper=_required_tool("set_gripper"),
            get_robot_state=_required_tool("get_robot_state"),
            target=_env_float("OPENFORGE_CUP_RACK_CLOSE_TARGET_POS", 0.04),
            steps=tuple(_float_csv("OPENFORGE_CUP_RACK_CLOSE_STEPS", [0.68, 0.40, 0.18, 0.08, 0.04])),
            vel_limit=_env_float_or_none("OPENFORGE_CUP_RACK_CLOSE_VEL_LIMIT"),
            torque_limit=_env_float_or_none("OPENFORGE_CUP_RACK_CLOSE_TORQUE_LIMIT"),
            min_contact_delta=_env_float("OPENFORGE_CUP_RACK_CLOSE_CONTACT_MIN_DELTA_POS", 0.015),
            target_tolerance=_env_float("OPENFORGE_CUP_RACK_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_CUP_RACK_HOLD_MIN_POS", 0.015),
            hold_max=_env_float("OPENFORGE_CUP_RACK_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="cup_close",
        )
        packet["steps"].append(close)
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        packet["gripper_after_close"] = close
        require_contact = _truthy_env("OPENFORGE_CUP_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True)
        close_ok = bool(close.get("contact_detected")) if require_contact else bool(close.get("plausible_for_lift"))
        packet["close_gate"] = {
            "require_contact_before_lift": require_contact,
            "passed": close_ok,
            "contact_detected": bool(close.get("contact_detected")),
            "target_reached": bool(close.get("target_reached")),
            "plausible_for_lift": bool(close.get("plausible_for_lift")),
        }
        if not close_ok:
            packet["why_stopped"] = "cup close did not produce required hold/contact evidence; stop before lift"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        for label in ("lift", "pre_place", "place"):
            step = _execute_pose(f"cup_{label}", arm, poses[label], run_dir, gripper=hold_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append(f"freespace_move:cup_{label}")

        release = _set_gripper_checked(arm, open_pos, "open_release_on_rack", run_dir)
        packet["steps"].append(release)
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        if not release.get("confirmed"):
            packet["why_stopped"] = "release open did not confirm; do not retreat blindly"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        step = _execute_pose("cup_retreat_after_release", arm, poses["retreat"], run_dir, gripper=open_pos)
        packet["steps"].append(step)
        packet["movement_capable_calls"].append("freespace_move:cup_retreat_after_release")
        post = _verify_post_place(run_dir, pre_cup, pre_rack)
        packet["post_verify"] = post
        packet["success"] = bool(post.get("success"))
        packet["confidence"] = post.get("confidence")
        packet["why_stopped"] = (
            "cup placement verified by post observation"
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
        "schema": "openforge.cup_rack.task_result.v1",
        "success": False,
        "plan_success": False,
        "confidence": 0.0,
        "physical_motion_executed": False,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "source_snapshot": None,
        "pre_observation": None,
        "selected_cup_detection": None,
        "selected_rack_detection": None,
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
            stage="pre_cup_rack",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_CUP_RACK_OBS_TIMEOUT_S", 30.0),
            max_retries=1,
            motion_cameras=_motion_cameras(),
            image_only_cameras=_image_only_cameras(),
        )
        result["pre_observation"] = scene
        cup_det = _select_detection(scene, _cup_prompts(), kind="cup")
        rack_det = _select_detection(scene, _rack_prompts(), kind="rack")
        result["selected_cup_detection"] = cup_det
        result["selected_rack_detection"] = rack_det
        if cup_det is None:
            result["why_stopped"] = "missing cup detection"
            result["next_recommendation"] = "Adjust cup prompts or scene placement, then rerun no-motion preview."
            return result
        if rack_det is None:
            result["why_stopped"] = "missing wooden rack detection"
            result["next_recommendation"] = "Adjust rack prompts or scene placement, then rerun no-motion preview."
            return result

        candidates = _generate_candidates(cup_det, rack_det)
        result["candidate_count"] = len(candidates)
        write_json(
            run_dir / "plans" / "cup_rack_candidates.json",
            {"schema": "openforge.cup_rack.candidates.v1", "candidates": candidates},
        )
        if not candidates:
            result["why_stopped"] = "no top-down cup candidates generated"
            result["next_recommendation"] = "Inspect selected detections and candidate geometry."
            return result

        selected, previews = _preview_and_select(candidates, run_dir)
        if selected is None:
            result["why_stopped"] = "no cup-to-rack sequence candidate passed planner preview"
            result["next_recommendation"] = "Inspect plans/cup_rack_sequence_preview.json and tune yaw/Z/place offsets."
            return result
        result["selected_strategy"] = selected
        result["plan_success"] = True

        physical_attempt = _execute_physical(selected, run_dir, cup_det, rack_det)
        result["physical_attempt"] = physical_attempt
        result["physical_motion_executed"] = bool(physical_attempt.get("physical_motion_executed"))
        result["gripper_after_close"] = physical_attempt.get("gripper_after_close")
        post_verify = physical_attempt.get("post_verify") or {}
        result["post_observation"] = post_verify.get("post_scene")
        result["object_motion"] = {
            "pre_cup_xyz": post_verify.get("pre_cup_xyz"),
            "post_cup_xyz": post_verify.get("post_cup_xyz"),
            "rack_xyz": post_verify.get("rack_xyz"),
            "cup_to_rack_xy_m": post_verify.get("cup_to_rack_xy_m"),
            "cup_from_source_xy_m": post_verify.get("cup_from_source_xy_m"),
            "post_cup_z_m": post_verify.get("post_cup_z_m"),
        }
        if result["physical_motion_executed"]:
            result["success"] = bool(physical_attempt.get("success"))
            result["confidence"] = float(physical_attempt.get("confidence") or (0.8 if result["success"] else 0.2))
            result["why_stopped"] = physical_attempt.get("why_stopped")
            result["next_recommendation"] = "Inspect videos and post verification before retrying."
        else:
            result["success"] = False
            result["confidence"] = 0.0
            result["why_stopped"] = "no-motion cup-to-rack preview passed; physical motion not executed"
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
                "## Cyan Cup To Wooden Rack Top Shelf",
                f"- success: {result.get('success')}",
                f"- plan_success: {result.get('plan_success')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- selected_cup_detection: {result.get('selected_cup_detection')}",
                f"- selected_rack_detection: {result.get('selected_rack_detection')}",
                f"- selected_strategy: {result.get('selected_strategy')}",
                f"- gripper_after_close: {result.get('gripper_after_close')}",
                f"- object_motion: {result.get('object_motion')}",
                f"- recording_evidence: {result.get('recording_evidence')}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- next_recommendation: {result.get('next_recommendation')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))

_set_default_cup_rack_env()

TASK_RESULT.update(_run())
