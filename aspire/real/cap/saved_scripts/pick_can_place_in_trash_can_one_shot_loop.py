"""Pick up a can and place it into/on a trash can on real YAM.

High-level transfer from ``can-libero-transfer.md`` only:
- localize the can and target trash from fresh perception before motion;
- treat the can as a simple cylinder and force a clean top-down grasp instead
  of trusting tilted grasp orientations;
- lift, transport above the trash, lower, release, retreat, then verify from
  post-action perception.

The script is standalone: each invocation performs fresh perception, generates
top-down candidates, previews the selected sequence, and, when physical gates
are set, executes that same run's planned trajectories. A no-motion run is only
a dry-run diagnostic, not a prerequisite for a physical run. Default behavior is
observe/preview only. Physical execution requires
``OPENFORGE_ALLOW_PHYSICAL_MOTION=1`` plus the can-trash ticket envs checked in
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

from cap.saved_scripts.full_demo_speed import scaled_speed_str
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


TASK_NAME = "pick_can_place_in_trash_can_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}


def _set_default_can_trash_env() -> None:
    """Default this task to the successful top-down can grasp policy."""
    defaults = {
        "OPENFORGE_CAN_TRASH_CAN_PROMPTS": (
            "can,sprite can,green can,red can,coke can,coca cola can,red soda can,"
            "soda can,aluminum can,tin can"
        ),
        "OPENFORGE_CAN_TRASH_TRASH_PROMPTS": "white trash can,trash can,trash bin,bin,white bin,waste bin,garbage can,bucket",
        "OPENFORGE_CAN_TRASH_CAMERAS": "top,left,right,bottom",
        "OPENFORGE_CAN_TRASH_MOTION_CAMERAS": "top,left,right",
        "OPENFORGE_CAN_TRASH_IMAGE_ONLY_CAMERAS": "bottom",
        "OPENFORGE_CAN_TRASH_OBS_TIMEOUT_S": "30.0",
        "OPENFORGE_CAN_TRASH_RECORDING_WAIT_S": "3.0",
        "OPENFORGE_CAN_TRASH_CAN_MIN_SCORE": "0.20",
        "OPENFORGE_CAN_TRASH_TRASH_MIN_SCORE": "0.20",
        "OPENFORGE_CAN_TRASH_CAN_X_MIN_M": "0.300",
        "OPENFORGE_CAN_TRASH_CAN_X_MAX_M": "0.850",
        "OPENFORGE_CAN_TRASH_CAN_Y_MIN_M": "-0.550",
        "OPENFORGE_CAN_TRASH_CAN_Y_MAX_M": "0.150",
        "OPENFORGE_CAN_TRASH_CAN_Z_MIN_M": "0.720",
        "OPENFORGE_CAN_TRASH_CAN_Z_MAX_M": "0.900",
        "OPENFORGE_CAN_TRASH_HELD_CAN_MIN_SCORE": "0.20",
        "OPENFORGE_CAN_TRASH_HELD_CAN_PRIMARY_CAMERAS": "top,right",
        "OPENFORGE_CAN_TRASH_HELD_CAN_X_MIN_M": "0.250",
        "OPENFORGE_CAN_TRASH_HELD_CAN_X_MAX_M": "0.950",
        "OPENFORGE_CAN_TRASH_HELD_CAN_Y_MIN_M": "-0.750",
        "OPENFORGE_CAN_TRASH_HELD_CAN_Y_MAX_M": "0.200",
        "OPENFORGE_CAN_TRASH_HELD_CAN_Z_MIN_M": "0.720",
        "OPENFORGE_CAN_TRASH_HELD_CAN_Z_MAX_M": "1.350",
        "OPENFORGE_CAN_TRASH_TRASH_PRIMARY_CAMERAS": "top",
        "OPENFORGE_CAN_TRASH_REQUIRE_PRIMARY_TRASH_CAMERA": "1",
        "OPENFORGE_CAN_TRASH_ARM": "right",
        "OPENFORGE_CAN_TRASH_TOPDOWN_YAWS_DEG": "0,-90",
        "OPENFORGE_CAN_TRASH_TOPDOWN_Z_OFFSETS_M": "0.025",
        "OPENFORGE_CAN_TRASH_GRASP_Z_MAX_M": "0.825",
        "OPENFORGE_CAN_TRASH_TOPDOWN_PITCH_DEG": "180.0",
        "OPENFORGE_CAN_TRASH_DROP_RPY_CANDIDATES_DEG": "0,180,0",
        "OPENFORGE_CAN_TRASH_KEEP_GRASP_ORIENTATION_FOR_DROP": "1",
        "OPENFORGE_CAN_TRASH_APPROACH_Z_M": "0.120",
        "OPENFORGE_CAN_TRASH_APPROACH_ZS_M": "0.120,0.080,0.060",
        "OPENFORGE_CAN_TRASH_LIFT_Z_M": "0.080",
        "OPENFORGE_CAN_TRASH_HIGH_VERTICAL_LIFT_BEFORE_TRANSFER": "1",
        "OPENFORGE_CAN_TRASH_TRANSFER_CLEAR_Z_M": "1.100",
        "OPENFORGE_CAN_TRASH_DIRECT_DROP_SEQUENCE": "1",
        "OPENFORGE_CAN_TRASH_RELEASE_AT_DROP_APPROACH": "1",
        "OPENFORGE_CAN_TRASH_PLACE_APPROACH_Z_M": "0.120",
        "OPENFORGE_CAN_TRASH_RETREAT_Z_M": "0.160",
        "OPENFORGE_CAN_TRASH_RETREAT_AFTER_RELEASE": "0",
        "OPENFORGE_CAN_TRASH_USE_TRASH_Z_FOR_RELEASE": "0",
        "OPENFORGE_CAN_TRASH_DROP_Z_CLEARANCE_M": "0.180",
        "OPENFORGE_CAN_TRASH_MIN_RELEASE_Z_M": "1.120",
        "OPENFORGE_CAN_TRASH_FIXED_RELEASE_Z_M": "1.120",
        "OPENFORGE_CAN_TRASH_MAX_RELEASE_Z_M": "1.200",
        "OPENFORGE_CAN_TRASH_RELEASE_Z_DELTA_FROM_GRASP_M": "0.000",
        "OPENFORGE_CAN_TRASH_PLACE_X_BIAS_M": "0.0",
        "OPENFORGE_CAN_TRASH_PLACE_Y_BIAS_M": "0.0",
        "OPENFORGE_CAN_TRASH_PLACE_X_BIASES_M": "0.0",
        "OPENFORGE_CAN_TRASH_PLACE_Y_BIASES_M": "0.0",
        "OPENFORGE_CAN_TRASH_DEFAULT_CAN_RADIUS_M": "0.035",
        "OPENFORGE_CAN_TRASH_DEFAULT_CAN_HALF_HEIGHT_M": "0.055",
        "OPENFORGE_CAN_TRASH_GRIPPER_WIDTH_MARGIN_M": "0.020",
        "OPENFORGE_CAN_TRASH_OPEN_GRIPPER_POS": "1.0",
        "OPENFORGE_CAN_TRASH_HOLD_GRIPPER_POS": "0.20",
        "OPENFORGE_CAN_TRASH_CLOSE_TARGET_POS": "0.05",
        "OPENFORGE_CAN_TRASH_CLOSE_STEPS": "0.05",
        "OPENFORGE_CAN_TRASH_CLOSE_VEL_LIMIT": "0.80",
        "OPENFORGE_CAN_TRASH_CLOSE_TORQUE_LIMIT": "0.80",
        "OPENFORGE_CAN_TRASH_CLOSE_CONFIRM_TIMEOUT_S": "2.0",
        "OPENFORGE_CAN_TRASH_CLOSE_CONTACT_MIN_DELTA_POS": "0.015",
        "OPENFORGE_CAN_TRASH_CLOSE_TARGET_TOLERANCE_POS": "0.05",
        "OPENFORGE_CAN_TRASH_HOLD_MIN_POS": "0.015",
        "OPENFORGE_CAN_TRASH_HOLD_MAX_POS": "0.95",
        "OPENFORGE_CAN_TRASH_REQUIRE_CONTACT_BEFORE_LIFT": "1",
        "OPENFORGE_CAN_TRASH_GRIPPER_CONFIRM_TIMEOUT_S": "4.0",
        "OPENFORGE_CAN_TRASH_OPEN_CONFIRM_POS": "0.75",
        "OPENFORGE_CAN_TRASH_STATE_POLL_S": "0.15",
        "OPENFORGE_CAN_TRASH_MAX_CANDIDATES": "96",
        "OPENFORGE_CAN_TRASH_PREVIEW_LIMIT": "96",
        "OPENFORGE_CAN_TRASH_PREVIEW_TIMEOUT_S": "25.0",
        "OPENFORGE_CAN_TRASH_PREVIEW_MAX_POS_ERROR_M": "0.030",
        "OPENFORGE_CAN_TRASH_PREVIEW_MAX_ROT_ERROR_DEG": "16.0",
        "OPENFORGE_CAN_TRASH_TRANSPORT_PREVIEW_MAX_ROT_ERROR_DEG": "95.0",
        "OPENFORGE_CAN_TRASH_DROP_PREVIEW_MAX_ROT_ERROR_DEG": "95.0",
        "OPENFORGE_CAN_TRASH_PLANNER_BACKEND": "rrtconnect",
        "OPENFORGE_CAN_TRASH_SOLVER_SPEED": "fast",
        "OPENFORGE_CAN_TRASH_PLANNING_SPEED": scaled_speed_str(0.30),
        "OPENFORGE_CAN_TRASH_IK_ERROR_THRESHOLD_M": "0.025",
        "OPENFORGE_CAN_TRASH_IK_ROT_THRESHOLD_DEG": "14.0",
        "OPENFORGE_CAN_TRASH_IK_XYZ_WEIGHT": "1.0",
        "OPENFORGE_CAN_TRASH_IK_RPY_WEIGHT": "0.35",
        "OPENFORGE_CAN_TRASH_DROP_IK_RPY_WEIGHT": "0.05",
        "OPENFORGE_CAN_TRASH_SUCCESS_MAX_CAN_TO_TRASH_XY_M": "0.090",
        "OPENFORGE_CAN_TRASH_SUCCESS_MIN_SOURCE_XY_DELTA_M": "0.040",
        "OPENFORGE_CAN_TRASH_ADAPT_PLACE_FROM_HELD_OBS": "0",
        "OPENFORGE_CAN_TRASH_ADAPT_MAX_CORRECTION_M": "0.320",
        "OPENFORGE_CAN_TRASH_ADAPT_EXTRA_X_BIAS_M": "0.000",
        "OPENFORGE_CAN_TRASH_ADAPT_EXTRA_Y_BIAS_M": "0.000",
        "OPENFORGE_CAN_TRASH_ADAPT_REFRESH_TRASH": "0",
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


def _rpy_list_env(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [list(row[:3]) for row in default]
    rows: list[list[float]] = []
    for chunk in raw.split(";"):
        values: list[float] = []
        for part in chunk.split(","):
            try:
                values.append(float(part.strip()))
            except ValueError:
                continue
        if len(values) >= 3:
            rows.append(values[:3])
    return rows or [list(row[:3]) for row in default]


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
        "OPENFORGE_CAN_TRASH_CAN_PROMPTS",
        ["can", "sprite can", "green can", "soda can", "aluminum can", "tin can"],
    )


def _trash_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_CAN_TRASH_TRASH_PROMPTS",
        ["white trash can", "trash can", "trash bin", "bin"],
    )


def _all_prompts() -> list[str]:
    prompts: list[str] = []
    for prompt in _can_prompts() + _trash_prompts() + _csv("OPENFORGE_CAN_TRASH_EXTRA_PROMPTS", []):
        if prompt not in prompts:
            prompts.append(prompt)
    return prompts


def _cameras() -> list[str]:
    return _csv("OPENFORGE_CAN_TRASH_CAMERAS", ["top", "left", "right", "bottom"])


def _image_only_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_CAN_TRASH_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _motion_cameras() -> tuple[str, ...]:
    return tuple(camera for camera in _csv("OPENFORGE_CAN_TRASH_MOTION_CAMERAS", ["top", "left", "right"]) if camera)


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_name = globals().get("__file__") or f"cap/saved_scripts/{TASK_NAME}.py"
    script_path = Path(str(script_name)).resolve()
    snapshot: dict[str, Any] = {
        "schema": "openforge.can_trash.source_snapshot.v1",
        "script": str(script_path),
        "env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith("OPENFORGE_CAN_TRASH_")
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
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_CAN_TRASH_RECORDING_WAIT_S", 3.0))
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


def _preferred_kind_cameras(kind: str) -> list[str]:
    raw = os.environ.get(f"OPENFORGE_CAN_TRASH_{kind.upper()}_PRIMARY_CAMERAS")
    if raw is None or not raw.strip():
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def _uses_preferred_kind_camera(det: dict[str, Any] | None, kind: str) -> bool:
    preferred = _preferred_kind_cameras(kind)
    if not preferred:
        return True
    return str((det or {}).get("source_camera") or "") in preferred


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


def _within_kind_workspace(det: dict[str, Any], *, kind: str) -> bool:
    xyz = _xyz(det.get("position_3d") or det.get("position"))
    if xyz is None:
        det["selection_reject_reason"] = "missing_3d_position"
        return False
    for axis, value in zip(("x", "y", "z"), xyz):
        min_value = _env_float_or_none(f"OPENFORGE_CAN_TRASH_{kind.upper()}_{axis.upper()}_MIN_M")
        max_value = _env_float_or_none(f"OPENFORGE_CAN_TRASH_{kind.upper()}_{axis.upper()}_MAX_M")
        if min_value is not None and float(value) < float(min_value):
            det["selection_reject_reason"] = f"{kind}_{axis}_below_min"
            return False
        if max_value is not None and float(value) > float(max_value):
            det["selection_reject_reason"] = f"{kind}_{axis}_above_max"
            return False
    return True


def _select_detection(scene: dict[str, Any], prompts: list[str], *, kind: str) -> dict[str, Any] | None:
    candidates = [det for det in _detections(scene, prompts) if _within_kind_workspace(det, kind=kind)]
    min_score = _env_float(f"OPENFORGE_CAN_TRASH_{kind.upper()}_MIN_SCORE", 0.20)
    preferred_cameras = _preferred_kind_cameras(kind)
    if preferred_cameras:
        preferred = [
            det
            for det in candidates
            if float(det.get("score") or min_score) >= min_score
            and str(det.get("source_camera") or "") in preferred_cameras
        ]
        if preferred:
            candidates = preferred
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
    requested = os.environ.get("OPENFORGE_CAN_TRASH_ARM", "right").strip().lower()
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
    return _env_float("OPENFORGE_CAN_TRASH_DEFAULT_CAN_RADIUS_M", 0.035)


def _estimate_half_height(det: dict[str, Any]) -> float:
    half_extents = _half_extents(det)
    if len(half_extents) >= 3 and abs(float(half_extents[2])) > 0.01:
        return max(0.025, min(0.120, abs(float(half_extents[2]))))
    return _env_float("OPENFORGE_CAN_TRASH_DEFAULT_CAN_HALF_HEIGHT_M", 0.055)


def _place_biases_for_arm(arm: str) -> list[list[float]]:
    x_biases = _float_csv(
        "OPENFORGE_CAN_TRASH_PLACE_X_BIASES_M",
        [_env_float("OPENFORGE_CAN_TRASH_PLACE_X_BIAS_M", 0.0)],
    )
    if os.environ.get("OPENFORGE_CAN_TRASH_PLACE_Y_BIASES_M"):
        y_biases = _float_csv("OPENFORGE_CAN_TRASH_PLACE_Y_BIASES_M", [0.0])
    else:
        fallback_y = [-0.04, 0.0, -0.08] if arm == "right" else [0.04, 0.0, 0.08]
        y_biases = _float_csv("OPENFORGE_CAN_TRASH_PLACE_Y_BIASES_M", fallback_y)
    biases: list[list[float]] = []
    for x_bias in x_biases:
        for y_bias in y_biases:
            row = [float(x_bias), float(y_bias)]
            if row not in biases:
                biases.append(row)
    return biases or [[0.0, 0.0]]


def _drop_rpys_for_arm(arm: str) -> list[list[float]]:
    default = [
        [0.0, 90.0, -90.0],
        [0.0, 90.0, -60.0],
        [0.0, 90.0, -30.0],
        [0.0, 90.0, 0.0],
        [0.0, 90.0, 30.0],
        [0.0, 90.0, 60.0],
        [0.0, 90.0, 90.0],
    ]
    if arm == "left":
        default = [
            [0.0, 90.0, 90.0],
            [0.0, 90.0, 60.0],
            [0.0, 90.0, 30.0],
            [0.0, 90.0, 0.0],
            [0.0, 90.0, -30.0],
            [0.0, 90.0, -60.0],
            [0.0, 90.0, -90.0],
        ]
    return _rpy_list_env("OPENFORGE_CAN_TRASH_DROP_RPY_CANDIDATES_DEG", default)


def _release_z_policy(trash_det: dict[str, Any], can_half_height: float, grasp_z: float) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    trash_xyz = _xyz(trash_det.get("position_3d") or trash_det.get("position"))
    if _truthy_env("OPENFORGE_CAN_TRASH_USE_TRASH_Z_FOR_RELEASE", True):
        trash_base_z = float((trash_xyz or [0.0, 0.0, grasp_z])[2])
        clearance = _env_float("OPENFORGE_CAN_TRASH_DROP_Z_CLEARANCE_M", 0.180)
        candidates.append({"source": "trash_z_plus_clearance", "z_m": trash_base_z + clearance})
    candidates.append(
        {
            "source": "grasp_z_plus_delta",
            "z_m": float(grasp_z) + _env_float("OPENFORGE_CAN_TRASH_RELEASE_Z_DELTA_FROM_GRASP_M", 0.250),
        }
    )
    fixed_release_z = _env_float_or_none("OPENFORGE_CAN_TRASH_FIXED_RELEASE_Z_M")
    if fixed_release_z is not None:
        candidates.append({"source": "fixed_release_z", "z_m": float(fixed_release_z)})
    min_release_z = _env_float("OPENFORGE_CAN_TRASH_MIN_RELEASE_Z_M", 1.080)
    candidates.append({"source": "min_release_z", "z_m": min_release_z})
    chosen = max(candidates, key=lambda row: float(row["z_m"])) if candidates else {"source": "fallback", "z_m": grasp_z}
    release_z = float(chosen["z_m"])
    max_release_z = _env_float_or_none("OPENFORGE_CAN_TRASH_MAX_RELEASE_Z_M")
    clamped_by_max = False
    if max_release_z is not None and release_z > float(max_release_z):
        release_z = float(max_release_z)
        clamped_by_max = True
    return {
        "release_z_m": release_z,
        "chosen_source": chosen["source"],
        "candidates": [
            {"source": row["source"], "z_m": round(float(row["z_m"]), 5)}
            for row in candidates
        ],
        "trash_xyz": trash_xyz,
        "can_half_height_m": round(float(can_half_height), 5),
        "grasp_z_m": round(float(grasp_z), 5),
        "max_release_z_m": None if max_release_z is None else round(float(max_release_z), 5),
        "clamped_by_max": clamped_by_max,
    }


def _release_z(trash_det: dict[str, Any], can_half_height: float, grasp_z: float) -> float:
    return float(_release_z_policy(trash_det, can_half_height, grasp_z)["release_z_m"])


def _make_candidate(
    *,
    index: int,
    arm: str,
    can_det: dict[str, Any],
    trash_det: dict[str, Any],
    yaw_deg: float,
    z_offset_m: float,
    approach_z_m: float,
    place_xy_bias: list[float],
    drop_rpy: list[float],
) -> dict[str, Any]:
    can_xyz = _xyz(can_det.get("position_3d") or can_det.get("position"))
    trash_xyz = _xyz(trash_det.get("position_3d") or trash_det.get("position"))
    if can_xyz is None or trash_xyz is None:
        raise ValueError("can and trash detections must have 3D positions")
    can_half_height = _estimate_half_height(can_det)
    can_radius = _estimate_radius(can_det)
    raw_grasp_z = float(can_xyz[2]) + _env_float("OPENFORGE_CAN_TRASH_GRASP_Z_BIAS_M", 0.000) + float(z_offset_m)
    grasp_z = raw_grasp_z
    grasp_z_min = _env_float_or_none("OPENFORGE_CAN_TRASH_GRASP_Z_MIN_M")
    grasp_z_max = _env_float_or_none("OPENFORGE_CAN_TRASH_GRASP_Z_MAX_M")
    if grasp_z_min is not None:
        grasp_z = max(float(grasp_z), float(grasp_z_min))
    if grasp_z_max is not None:
        grasp_z = min(float(grasp_z), float(grasp_z_max))
    release_policy = _release_z_policy(trash_det, can_half_height, grasp_z)
    release_z = float(release_policy["release_z_m"])
    approach_z = max(0.030, float(approach_z_m))
    lift_z = _env_float("OPENFORGE_CAN_TRASH_LIFT_Z_M", 0.080)
    place_approach_z = _env_float("OPENFORGE_CAN_TRASH_PLACE_APPROACH_Z_M", 0.060)
    retreat_z = _env_float("OPENFORGE_CAN_TRASH_RETREAT_Z_M", 0.080)
    grasp_rpy = [0.0, _env_float("OPENFORGE_CAN_TRASH_TOPDOWN_PITCH_DEG", 180.0), float(yaw_deg)]
    if _truthy_env("OPENFORGE_CAN_TRASH_KEEP_GRASP_ORIENTATION_FOR_DROP", True):
        release_rpy = list(grasp_rpy)
        drop_orientation_policy = "keep_grasp_orientation"
    else:
        release_rpy = [float(v) for v in drop_rpy[:3]]
        drop_orientation_policy = "use_drop_rpy_candidates"
    trash_xy = [float(trash_xyz[0]) + place_xy_bias[0], float(trash_xyz[1]) + place_xy_bias[1]]
    min_lift_abs_z = grasp_z + lift_z
    pre_place_z = max(min_lift_abs_z, release_z + place_approach_z)
    pickup_lift_z = max(
        min_lift_abs_z,
        min(pre_place_z, _env_float("OPENFORGE_CAN_TRASH_TRANSFER_CLEAR_Z_M", 1.100)),
    )
    lift_abs_z = pickup_lift_z if _truthy_env("OPENFORGE_CAN_TRASH_HIGH_VERTICAL_LIFT_BEFORE_TRANSFER", True) else min_lift_abs_z
    direct_drop_sequence = _truthy_env("OPENFORGE_CAN_TRASH_DIRECT_DROP_SEQUENCE", True)
    transfer_z = pre_place_z if direct_drop_sequence else lift_abs_z
    release_at_drop_approach = _truthy_env("OPENFORGE_CAN_TRASH_RELEASE_AT_DROP_APPROACH", direct_drop_sequence)
    place_z = transfer_z if release_at_drop_approach else release_z
    can_xy = [float(can_xyz[0]), float(can_xyz[1])]
    width = max(0.02, min(0.95, 2.0 * can_radius + _env_float("OPENFORGE_CAN_TRASH_GRIPPER_WIDTH_MARGIN_M", 0.020)))
    return {
        "index": int(index),
        "schema": "openforge.can_trash.candidate.v1",
        "arm": arm,
        "strategy": "top_down_can_pickup_flexible_trash_drop",
        "score": round(1.0 - 0.03 * index - 0.4 * abs(float(z_offset_m)), 5),
        "grasp_rpy": _round_list(grasp_rpy, digits=4, limit=3),
        "drop_rpy": _round_list(release_rpy, digits=4, limit=3),
        "drop_orientation_policy": drop_orientation_policy,
        "width": round(float(width), 5),
        "can_radius_m": round(float(can_radius), 5),
        "can_half_height_m": round(float(can_half_height), 5),
        "topdown_yaw_deg": round(float(yaw_deg), 4),
        "z_offset_m": round(float(z_offset_m), 5),
        "approach_z_m": round(float(approach_z), 5),
        "lift_policy": (
            "high_vertical_lift_before_transfer"
            if _truthy_env("OPENFORGE_CAN_TRASH_HIGH_VERTICAL_LIFT_BEFORE_TRANSFER", True)
            else "minimum_post_grasp_lift"
        ),
        "drop_sequence": "direct_transfer_to_drop_approach" if direct_drop_sequence else "transfer_then_pre_place",
        "release_at_drop_approach": release_at_drop_approach,
        "pickup_lift_z_m": round(float(lift_abs_z), 5),
        "transfer_clear_z_m": round(float(transfer_z), 5),
        "raw_grasp_z_m": round(float(raw_grasp_z), 5),
        "grasp_z_min_m": None if grasp_z_min is None else round(float(grasp_z_min), 5),
        "grasp_z_max_m": None if grasp_z_max is None else round(float(grasp_z_max), 5),
        "grasp_z_clamped": bool(abs(float(grasp_z) - float(raw_grasp_z)) > 1e-6),
        "release_z_policy": release_policy,
        "place_xy_bias_m": _round_list(place_xy_bias, limit=2),
        "source_can_detection": can_det,
        "source_trash_detection": trash_det,
        "poses": {
            "pregrasp": {
                "position": [can_xy[0], can_xy[1], grasp_z + approach_z],
                "rpy": grasp_rpy,
                "orientation_role": "grasp",
            },
            "grasp": {
                "position": [can_xy[0], can_xy[1], grasp_z],
                "rpy": grasp_rpy,
                "orientation_role": "grasp",
            },
            "lift": {
                "position": [can_xy[0], can_xy[1], lift_abs_z],
                "rpy": grasp_rpy,
                "orientation_role": "transport",
            },
            "transfer": {
                "position": [trash_xy[0], trash_xy[1], transfer_z],
                "rpy": release_rpy,
                "orientation_role": "transport",
            },
            "pre_place": {
                "position": [trash_xy[0], trash_xy[1], pre_place_z],
                "rpy": release_rpy,
                "orientation_role": "drop",
            },
            "place": {
                "position": [trash_xy[0], trash_xy[1], place_z],
                "rpy": release_rpy,
                "orientation_role": "drop",
            },
            "retreat": {
                "position": [trash_xy[0], trash_xy[1], place_z + retreat_z],
                "rpy": release_rpy,
                "orientation_role": "drop",
            },
        },
        "geometry_policy": (
            "Can is treated as a cylinder; top-down gripper opening yaw is swept, "
            "trash placement is based on fresh trash-can center plus optional XY bias, "
            "trash transfer lands directly at the drop-approach Z by default, "
            "and trash release keeps the grasp orientation by default so the held "
            "can stays aligned with the EE target."
        ),
    }


def _generate_candidates(can_det: dict[str, Any], trash_det: dict[str, Any]) -> list[dict[str, Any]]:
    can_xyz = _xyz(can_det.get("position_3d") or can_det.get("position"))
    if can_xyz is None:
        return []
    arms = _arms_from_can(can_xyz)
    yaws = _float_csv("OPENFORGE_CAN_TRASH_TOPDOWN_YAWS_DEG", [0.0, -90.0])
    z_offsets = _float_csv("OPENFORGE_CAN_TRASH_TOPDOWN_Z_OFFSETS_M", [0.025])
    approach_zs = _float_csv("OPENFORGE_CAN_TRASH_APPROACH_ZS_M", [_env_float("OPENFORGE_CAN_TRASH_APPROACH_Z_M", 0.120)])
    max_candidates = max(1, _env_int("OPENFORGE_CAN_TRASH_MAX_CANDIDATES", 96))
    candidates: list[dict[str, Any]] = []
    idx = 1
    for arm in arms:
        for place_xy_bias in _place_biases_for_arm(arm):
            for drop_rpy in _drop_rpys_for_arm(arm):
                for z_offset in z_offsets:
                    for approach_z in approach_zs:
                        for yaw in yaws:
                            candidates.append(
                                _make_candidate(
                                    index=idx,
                                    arm=arm,
                                    can_det=can_det,
                                    trash_det=trash_det,
                                    yaw_deg=float(yaw),
                                    z_offset_m=float(z_offset),
                                    approach_z_m=float(approach_z),
                                    place_xy_bias=place_xy_bias,
                                    drop_rpy=drop_rpy,
                                )
                            )
                            idx += 1
                            if len(candidates) >= max_candidates:
                                return candidates
    return candidates


def _move_kwargs(arm: str, pose: dict[str, Any], *, preview_only: bool, gripper: float | None = None) -> dict[str, Any]:
    prefix = "left" if arm == "left" else "right"
    role = str(pose.get("orientation_role") or "")
    soft_orientation = role in {"drop", "transport"}
    kwargs: dict[str, Any] = {
        f"{prefix}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{prefix}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_CAN_TRASH_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_CAN_TRASH_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_CAN_TRASH_PLANNING_SPEED", 0.30),
        "ik_error_threshold": _env_float("OPENFORGE_CAN_TRASH_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_CAN_TRASH_IK_ROT_THRESHOLD_DEG", 14.0),
        "ik_xyz_weight": _env_float("OPENFORGE_CAN_TRASH_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float(
            "OPENFORGE_CAN_TRASH_DROP_IK_RPY_WEIGHT" if soft_orientation else "OPENFORGE_CAN_TRASH_IK_RPY_WEIGHT",
            0.05 if soft_orientation else 0.35,
        ),
    }
    if gripper is not None:
        kwargs[f"{prefix}_gripper"] = float(gripper)
    return kwargs


def _preview_summary(raw: Any, *, max_rot: float | None = None) -> dict[str, Any]:
    safe = json_safe(raw)
    if isinstance(safe, dict) and "data" in safe and isinstance(safe["data"], dict):
        safe = safe["data"]
    status = str(_field(safe, "status", "") or "")
    executed = bool(_field(safe, "executed", False))
    pos_error = _field(safe, "final_pos_error_m", _field(safe, "ik_error_m", None))
    rot_error = _field(safe, "final_rot_error_deg", _field(safe, "ik_rot_error_deg", None))
    max_pos = _env_float("OPENFORGE_CAN_TRASH_PREVIEW_MAX_POS_ERROR_M", 0.030)
    max_rot = float(max_rot) if max_rot is not None else _env_float("OPENFORGE_CAN_TRASH_PREVIEW_MAX_ROT_ERROR_DEG", 16.0)
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
        _env_float("OPENFORGE_CAN_TRASH_PREVIEW_TIMEOUT_S", 25.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(arm, pose, preview_only=True, gripper=gripper),
    )
    if not result.get("ok"):
        return {"label": label, "success": False, "reason": result.get("error"), "executed": False}
    max_rot = None
    role = str(pose.get("orientation_role") or "")
    if role == "drop":
        max_rot = _env_float("OPENFORGE_CAN_TRASH_DROP_PREVIEW_MAX_ROT_ERROR_DEG", 95.0)
    elif role == "transport":
        max_rot = _env_float("OPENFORGE_CAN_TRASH_TRANSPORT_PREVIEW_MAX_ROT_ERROR_DEG", 95.0)
    summary = _preview_summary(result.get("data"), max_rot=max_rot)
    summary["label"] = label
    summary["arm"] = arm
    summary["target_pos"] = _round_list(pose.get("position"), limit=3)
    summary["target_rpy"] = _round_list(pose.get("rpy"), digits=4, limit=3)
    summary["orientation_role"] = pose.get("orientation_role")
    return summary


def _preview_sequence(candidate: dict[str, Any]) -> dict[str, Any]:
    arm = str(candidate["arm"])
    poses = candidate["poses"]
    open_gripper_pos = _env_float("OPENFORGE_CAN_TRASH_OPEN_GRIPPER_POS", 1.0)
    hold_gripper_pos = _env_float("OPENFORGE_CAN_TRASH_HOLD_GRIPPER_POS", 0.20)
    direct_drop_sequence = _truthy_env("OPENFORGE_CAN_TRASH_DIRECT_DROP_SEQUENCE", True)
    retreat_after_release = _truthy_env("OPENFORGE_CAN_TRASH_RETREAT_AFTER_RELEASE", False)
    release_at_drop_approach = bool(candidate.get("release_at_drop_approach"))
    steps = [
        _preview_pose("can_pregrasp", arm, poses["pregrasp"], gripper=open_gripper_pos),
        _preview_pose("can_grasp", arm, poses["grasp"], gripper=open_gripper_pos),
        _preview_pose("can_lift", arm, poses["lift"], gripper=hold_gripper_pos),
        _preview_pose("can_transfer_at_clearance", arm, poses["transfer"], gripper=hold_gripper_pos),
    ]
    if not direct_drop_sequence:
        steps.append(_preview_pose("can_pre_place", arm, poses["pre_place"], gripper=hold_gripper_pos))
    if not release_at_drop_approach:
        steps.append(_preview_pose("can_place", arm, poses["place"], gripper=hold_gripper_pos))
    if retreat_after_release:
        steps.append(_preview_pose("can_retreat_after_release", arm, poses["retreat"], gripper=open_gripper_pos))
    return {
        "schema": "openforge.can_trash.preview_sequence.v1",
        "success": all(bool(step.get("success")) for step in steps),
        "steps": steps,
    }


def _preview_and_select(candidates: list[dict[str, Any]], run_dir: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    limit = min(len(candidates), max(1, _env_int("OPENFORGE_CAN_TRASH_PREVIEW_LIMIT", 24)))
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
        "schema": "openforge.can_trash.preview_rank.v1",
        "candidate_count": len(candidates),
        "previewed_count": len(previews),
        "selected": selected,
        "previews": previews,
    }
    packet["path"] = write_json(run_dir / "plans" / "can_trash_sequence_preview.json", packet)
    return selected, previews


def _physical_ticket_status() -> dict[str, Any]:
    required = {
        "OPENFORGE_CAN_TRASH_PHYSICAL_TICKET": "1",
        "OPENFORGE_CAN_TRASH_OPERATOR_PRESENT": "1",
        "OPENFORGE_CAN_TRASH_ESTOP_REACHABLE": "1",
        "OPENFORGE_CAN_TRASH_WORKSPACE_CLEAR": "1",
        "OPENFORGE_CAN_TRASH_HANDS_TOOLS_CABLES_CLEAR": "1",
        "OPENFORGE_CAN_TRASH_FOLLOWER_SERVERS_UP": "1",
        "OPENFORGE_CAN_TRASH_NO_OTHER_MOTION_PRECHECK": "1",
        "OPENFORGE_CAN_TRASH_RECORDING_DEBUG_CONFIRMED": "1",
        "OPENFORGE_CAN_TRASH_CAN_VISIBLE": "1",
        "OPENFORGE_CAN_TRASH_TRASH_CAN_VISIBLE": "1",
        "OPENFORGE_CAN_TRASH_NO_MANUAL_XYZ": "1",
    }
    missing = [name for name, expected in required.items() if os.environ.get(name) != expected]
    return {"passed": not missing, "missing_or_invalid": missing, "required": required}


def _set_gripper_checked(arm: str, target: float, stage: str, run_dir: Path) -> dict[str, Any]:
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    before = read_gripper_width(get_robot_state, arm)
    kwargs: dict[str, float] = {}
    vel = _env_float_or_none("OPENFORGE_CAN_TRASH_GRIPPER_VEL_LIMIT")
    torque = _env_float_or_none("OPENFORGE_CAN_TRASH_GRIPPER_TORQUE_LIMIT")
    if vel is not None:
        kwargs["vel_limit"] = vel
    if torque is not None:
        kwargs["torque_limit"] = torque
    command = set_gripper(arm, float(target), **kwargs)
    after = read_gripper_width(get_robot_state, arm)
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_CAN_TRASH_GRIPPER_CONFIRM_TIMEOUT_S", 4.0))
    confirm_min = _env_float("OPENFORGE_CAN_TRASH_OPEN_CONFIRM_POS", 0.75)
    if target >= confirm_min:
        while after is not None and after < confirm_min and time.monotonic() < deadline:
            time.sleep(max(0.02, _env_float("OPENFORGE_CAN_TRASH_STATE_POLL_S", 0.15)))
            after = read_gripper_width(get_robot_state, arm)
    packet = {
        "schema": "openforge.can_trash.set_gripper.v1",
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
    packet = {"schema": "openforge.can_trash.execute_pose.v1", "stage": label, "arm": arm, "preview": preview, "execute": json_safe(executed)}
    packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
    return packet


def _go_home_after_finish(run_dir: Path) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.can_trash.go_home_after_finish.v1",
        "success": False,
        "stage": "go_home_after_finish",
    }
    try:
        packet["result"] = json_safe(_required_tool("go_home")())
        packet["success"] = True
    except Exception as exc:
        packet["error"] = f"{type(exc).__name__}: {exc}"
    packet["path"] = write_json(run_dir / "plans" / "go_home_after_finish.json", packet)
    return packet


def _verify_post_place(
    run_dir: Path,
    pre_can: dict[str, Any] | None,
    pre_trash: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = capture_scene(
        prompts=_all_prompts(),
        cameras=_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="post_can_trash",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_CAN_TRASH_OBS_TIMEOUT_S", 30.0),
        max_retries=1,
        motion_cameras=_motion_cameras(),
        image_only_cameras=_image_only_cameras(),
    )
    post_can = _select_detection(scene, _can_prompts(), kind="can")
    post_trash = _select_detection(scene, _trash_prompts(), kind="trash") or pre_trash
    pre_can_xyz = _xyz((pre_can or {}).get("position_3d") or (pre_can or {}).get("position"))
    post_can_xyz = _xyz((post_can or {}).get("position_3d") or (post_can or {}).get("position"))
    trash_xyz = _xyz((post_trash or {}).get("position_3d") or (post_trash or {}).get("position"))
    packet: dict[str, Any] = {
        "schema": "openforge.can_trash.post_verify.v1",
        "success": False,
        "confidence": 0.0,
        "post_scene": scene,
        "pre_can_xyz": pre_can_xyz,
        "post_can_detection": post_can,
        "post_can_xyz": post_can_xyz,
        "trash_detection": post_trash,
        "trash_xyz": trash_xyz,
        "can_to_trash_xy_m": None,
        "can_from_source_xy_m": None,
        "source_cleared": None,
        "trash_near_or_occluded": None,
        "why": None,
    }
    if trash_xyz is None:
        packet["why"] = "post observation did not find the trash target"
    elif post_can_xyz is None:
        packet["source_cleared"] = True
        packet["trash_near_or_occluded"] = True
        packet["success"] = True
        packet["confidence"] = 0.70
        packet["why"] = "post observation no longer finds the can; after release over the trash can this is treated as inside/occluded"
    else:
        can_to_trash = _dist_xy(post_can_xyz, trash_xyz)
        packet["can_to_trash_xy_m"] = round(float(can_to_trash), 5)
        if pre_can_xyz is not None:
            packet["can_from_source_xy_m"] = round(float(_dist_xy(post_can_xyz, pre_can_xyz)), 5)
        threshold = _env_float("OPENFORGE_CAN_TRASH_SUCCESS_MAX_CAN_TO_TRASH_XY_M", 0.160)
        moved_min = _env_float("OPENFORGE_CAN_TRASH_SUCCESS_MIN_SOURCE_XY_DELTA_M", 0.040)
        moved_ok = packet["can_from_source_xy_m"] is None or float(packet["can_from_source_xy_m"]) >= moved_min
        packet["source_cleared"] = bool(moved_ok)
        packet["trash_near_or_occluded"] = bool(can_to_trash <= threshold)
        packet["success"] = bool(can_to_trash <= threshold and moved_ok)
        packet["confidence"] = 0.85 if packet["success"] else 0.25
        packet["why"] = (
            "post can detection is centered near the trash can"
            if packet["success"]
            else "post can detection is not close enough to the trash can or did not move from source"
        )
    packet["path"] = write_json(run_dir / "plans" / "post_can_trash_verify.json", packet)
    return packet


def _arm_ee_xyz(scene: dict[str, Any], arm: str) -> list[float] | None:
    state = scene.get("robot_state") if isinstance(scene, dict) else None
    arms = state.get("arms") if isinstance(state, dict) else None
    arm_state = arms.get(arm) if isinstance(arms, dict) else None
    if not isinstance(arm_state, dict):
        return None
    return _xyz(arm_state.get("ee_pos") or arm_state.get("ee_position"))


def _with_pose_xy(pose: dict[str, Any], xy: list[float]) -> dict[str, Any]:
    updated = dict(pose)
    pos = list(updated.get("position") or [])
    if len(pos) >= 3:
        pos[0] = float(xy[0])
        pos[1] = float(xy[1])
    updated["position"] = pos
    return updated


def _adapt_place_from_held_observation(
    *,
    arm: str,
    poses: dict[str, Any],
    run_dir: Path,
    pre_trash: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = capture_scene(
        prompts=_all_prompts(),
        cameras=_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="held_can_pre_place",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_CAN_TRASH_OBS_TIMEOUT_S", 30.0),
        max_retries=1,
        motion_cameras=_motion_cameras(),
        image_only_cameras=_image_only_cameras(),
    )
    held_can = _select_detection(scene, _can_prompts(), kind="held_can")
    refreshed_trash = _select_detection(scene, _trash_prompts(), kind="trash") if _truthy_env(
        "OPENFORGE_CAN_TRASH_ADAPT_REFRESH_TRASH",
        False,
    ) else None
    trash = refreshed_trash or pre_trash
    held_xyz = _xyz((held_can or {}).get("position_3d") or (held_can or {}).get("position"))
    trash_xyz = _xyz((trash or {}).get("position_3d") or (trash or {}).get("position"))
    ee_xyz = _arm_ee_xyz(scene, arm)
    original_place = list((poses.get("place") or {}).get("position") or [])
    packet: dict[str, Any] = {
        "schema": "openforge.can_trash.held_can_place_adaptation.v1",
        "success": False,
        "scene": scene,
        "held_can_detection": held_can,
        "trash_detection": trash,
        "trash_source": "held_observation" if refreshed_trash is not None else "pre_observation",
        "held_can_xyz": held_xyz,
        "trash_xyz": trash_xyz,
        "ee_xyz": ee_xyz,
        "original_place_xyz": _xyz(original_place),
        "held_can_from_ee_xy_m": None,
        "corrected_place_xyz": None,
        "correction_xy_m": None,
        "correction_norm_m": None,
        "max_correction_m": _env_float("OPENFORGE_CAN_TRASH_ADAPT_MAX_CORRECTION_M", 0.320),
        "poses": None,
        "why": None,
    }
    if held_xyz is None:
        packet["why"] = "held can was not detected at pre-place"
    elif trash_xyz is None:
        packet["why"] = "trash can was not detected during held-can adaptation"
    elif ee_xyz is None:
        packet["why"] = "robot state did not include current arm EE position"
    elif len(original_place) < 3:
        packet["why"] = "selected strategy did not include a valid place pose"
    else:
        held_offset_xy = [float(held_xyz[0]) - float(ee_xyz[0]), float(held_xyz[1]) - float(ee_xyz[1])]
        corrected_xy = [
            float(trash_xyz[0]) - held_offset_xy[0] + _env_float("OPENFORGE_CAN_TRASH_ADAPT_EXTRA_X_BIAS_M", 0.0),
            float(trash_xyz[1]) - held_offset_xy[1] + _env_float("OPENFORGE_CAN_TRASH_ADAPT_EXTRA_Y_BIAS_M", 0.0),
        ]
        correction_xy = [corrected_xy[0] - float(original_place[0]), corrected_xy[1] - float(original_place[1])]
        correction_norm = math.hypot(correction_xy[0], correction_xy[1])
        packet["held_can_from_ee_xy_m"] = _round_list(held_offset_xy)
        packet["corrected_place_xyz"] = _round_list([corrected_xy[0], corrected_xy[1], float(original_place[2])])
        packet["correction_xy_m"] = _round_list(correction_xy)
        packet["correction_norm_m"] = round(float(correction_norm), 5)
        if correction_norm > float(packet["max_correction_m"]):
            packet["why"] = "held-can correction exceeds configured maximum"
        else:
            corrected_poses = dict(poses)
            if "transfer" in corrected_poses:
                corrected_poses["transfer"] = _with_pose_xy(poses["transfer"], corrected_xy)
            corrected_poses["pre_place"] = _with_pose_xy(poses["pre_place"], corrected_xy)
            corrected_poses["place"] = _with_pose_xy(poses["place"], corrected_xy)
            corrected_poses["retreat"] = _with_pose_xy(poses["retreat"], corrected_xy)
            packet["poses"] = corrected_poses
            packet["success"] = True
            packet["why"] = "corrected place target from held-can/TCP offset"
    packet["path"] = write_json(run_dir / "plans" / "held_can_place_adaptation.json", packet)
    return packet


def _execute_physical(
    selected: dict[str, Any],
    run_dir: Path,
    pre_can: dict[str, Any] | None,
    pre_trash: dict[str, Any] | None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.can_trash.physical_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "held_can_place_adaptation": None,
        "home_after_finish": None,
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
    open_pos = _env_float("OPENFORGE_CAN_TRASH_OPEN_GRIPPER_POS", 1.0)
    hold_pos = _env_float("OPENFORGE_CAN_TRASH_HOLD_GRIPPER_POS", 0.20)
    direct_drop_sequence = _truthy_env("OPENFORGE_CAN_TRASH_DIRECT_DROP_SEQUENCE", True)
    retreat_after_release = _truthy_env("OPENFORGE_CAN_TRASH_RETREAT_AFTER_RELEASE", False)
    release_at_drop_approach = bool(selected.get("release_at_drop_approach"))
    packet["sequence_policy"] = {
        "direct_drop_sequence": direct_drop_sequence,
        "retreat_after_release": retreat_after_release,
        "release_at_drop_approach": release_at_drop_approach,
    }
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
            target=_env_float("OPENFORGE_CAN_TRASH_CLOSE_TARGET_POS", 0.05),
            steps=tuple(_float_csv("OPENFORGE_CAN_TRASH_CLOSE_STEPS", [0.05])),
            vel_limit=_env_float("OPENFORGE_CAN_TRASH_CLOSE_VEL_LIMIT", 0.80),
            torque_limit=_env_float("OPENFORGE_CAN_TRASH_CLOSE_TORQUE_LIMIT", 0.80),
            min_contact_delta=_env_float("OPENFORGE_CAN_TRASH_CLOSE_CONTACT_MIN_DELTA_POS", 0.015),
            target_tolerance=_env_float("OPENFORGE_CAN_TRASH_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            confirm_timeout_s=_env_float(
                "OPENFORGE_CAN_TRASH_CLOSE_CONFIRM_TIMEOUT_S",
                _env_float("OPENFORGE_CAN_TRASH_GRIPPER_CONFIRM_TIMEOUT_S", 2.0),
            ),
            state_poll_s=_env_float("OPENFORGE_CAN_TRASH_STATE_POLL_S", 0.15),
            hold_min=_env_float("OPENFORGE_CAN_TRASH_HOLD_MIN_POS", 0.015),
            hold_max=_env_float("OPENFORGE_CAN_TRASH_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="can_close",
        )
        packet["steps"].append(close)
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        packet["gripper_after_close"] = close
        require_contact = _truthy_env("OPENFORGE_CAN_TRASH_REQUIRE_CONTACT_BEFORE_LIFT", True)
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

        transfer_labels = ("lift", "transfer") if direct_drop_sequence else ("lift", "transfer", "pre_place")
        for label in transfer_labels:
            step = _execute_pose(f"can_{label}", arm, poses[label], run_dir, gripper=hold_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append(f"freespace_move:can_{label}")

        active_poses = poses
        if _truthy_env("OPENFORGE_CAN_TRASH_ADAPT_PLACE_FROM_HELD_OBS", True):
            adaptation = _adapt_place_from_held_observation(
                arm=arm,
                poses=poses,
                run_dir=run_dir,
                pre_trash=pre_trash,
            )
            packet["held_can_place_adaptation"] = adaptation
            packet["steps"].append(adaptation)
            if adaptation.get("success") and isinstance(adaptation.get("poses"), dict):
                active_poses = adaptation["poses"]
                step = _execute_pose("can_adapted_pre_place", arm, active_poses["pre_place"], run_dir, gripper=hold_pos)
                packet["steps"].append(step)
                packet["movement_capable_calls"].append("freespace_move:can_adapted_pre_place")
            elif _truthy_env("OPENFORGE_CAN_TRASH_REQUIRE_ADAPT_PLACE", False):
                packet["why_stopped"] = f"held-can placement adaptation failed: {adaptation.get('why')}"
                packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
                return packet

        if release_at_drop_approach:
            packet["movement_capable_calls"].append("skip_freespace_move:can_place_release_at_drop_approach")
        else:
            step = _execute_pose("can_place", arm, active_poses["place"], run_dir, gripper=hold_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append("freespace_move:can_place")

        release = _set_gripper_checked(arm, open_pos, "open_release_in_trash", run_dir)
        packet["steps"].append(release)
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        if not release.get("confirmed"):
            packet["why_stopped"] = "release open did not confirm; do not retreat blindly"
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet

        if retreat_after_release:
            step = _execute_pose("can_retreat_after_release", arm, active_poses["retreat"], run_dir, gripper=open_pos)
            packet["steps"].append(step)
            packet["movement_capable_calls"].append("freespace_move:can_retreat_after_release")
        home = _go_home_after_finish(run_dir)
        packet["home_after_finish"] = home
        packet["steps"].append(home)
        packet["movement_capable_calls"].append("go_home")
        if not home.get("success"):
            packet["why_stopped"] = (
                "can released and retreated, but final go_home failed"
                if retreat_after_release
                else "can released, but final go_home failed"
            )
            packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
            return packet
        post = _verify_post_place(run_dir, pre_can, pre_trash)
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
        "schema": "openforge.can_trash.task_result.v1",
        "success": False,
        "plan_success": False,
        "confidence": 0.0,
        "physical_motion_executed": False,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "source_snapshot": None,
        "pre_observation": None,
        "selected_can_detection": None,
        "selected_trash_detection": None,
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
            stage="pre_can_trash",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_CAN_TRASH_OBS_TIMEOUT_S", 30.0),
            max_retries=1,
            motion_cameras=_motion_cameras(),
            image_only_cameras=_image_only_cameras(),
        )
        result["pre_observation"] = scene
        can_det = _select_detection(scene, _can_prompts(), kind="can")
        trash_det = _select_detection(scene, _trash_prompts(), kind="trash")
        result["selected_can_detection"] = can_det
        result["selected_trash_detection"] = trash_det
        if can_det is None:
            result["why_stopped"] = "missing can detection"
            result["next_recommendation"] = "Adjust can prompts or scene placement, then rerun no-motion preview."
            return result
        if trash_det is None:
            result["why_stopped"] = "missing trash can detection"
            result["next_recommendation"] = "Adjust trash prompts or scene placement, then rerun no-motion preview."
            return result
        if _truthy_env("OPENFORGE_CAN_TRASH_REQUIRE_PRIMARY_TRASH_CAMERA", True) and not _uses_preferred_kind_camera(
            trash_det,
            "trash",
        ):
            result["why_stopped"] = (
                "trash can was not detected by the required primary camera; refusing to use fallback "
                f"{trash_det.get('source_camera')} target for physical placement"
            )
            result["next_recommendation"] = (
                "Move the bin into the right camera view or override "
                "OPENFORGE_CAN_TRASH_REQUIRE_PRIMARY_TRASH_CAMERA only after a no-motion inspection."
            )
            return result

        candidates = _generate_candidates(can_det, trash_det)
        result["candidate_count"] = len(candidates)
        write_json(
            run_dir / "plans" / "can_trash_candidates.json",
            {"schema": "openforge.can_trash.candidates.v1", "candidates": candidates},
        )
        if not candidates:
            result["why_stopped"] = "no top-down can candidates generated"
            result["next_recommendation"] = "Inspect selected detections and candidate geometry."
            return result

        selected, previews = _preview_and_select(candidates, run_dir)
        if selected is None:
            result["why_stopped"] = "no can-to-trash sequence candidate passed planner preview"
            result["next_recommendation"] = "Inspect plans/can_trash_sequence_preview.json and tune yaw/Z/place offsets."
            return result
        result["selected_strategy"] = selected
        result["plan_success"] = True

        physical_attempt = _execute_physical(selected, run_dir, can_det, trash_det)
        result["physical_attempt"] = physical_attempt
        result["physical_motion_executed"] = bool(physical_attempt.get("physical_motion_executed"))
        result["gripper_after_close"] = physical_attempt.get("gripper_after_close")
        post_verify = physical_attempt.get("post_verify") or {}
        result["post_observation"] = post_verify.get("post_scene")
        result["object_motion"] = {
            "pre_can_xyz": post_verify.get("pre_can_xyz"),
            "post_can_xyz": post_verify.get("post_can_xyz"),
            "trash_xyz": post_verify.get("trash_xyz"),
            "can_to_trash_xy_m": post_verify.get("can_to_trash_xy_m"),
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
            result["why_stopped"] = "no-motion can-to-trash-can preview passed; physical motion not executed"
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
                "## Can To Trash Can",
                f"- success: {result.get('success')}",
                f"- plan_success: {result.get('plan_success')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- selected_can_detection: {result.get('selected_can_detection')}",
                f"- selected_trash_detection: {result.get('selected_trash_detection')}",
                f"- selected_strategy: {result.get('selected_strategy')}",
                f"- gripper_after_close: {result.get('gripper_after_close')}",
                f"- object_motion: {result.get('object_motion')}",
                f"- recording_evidence: {result.get('recording_evidence')}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- next_recommendation: {result.get('next_recommendation')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))

_set_default_can_trash_env()

TASK_RESULT.update(_run())
