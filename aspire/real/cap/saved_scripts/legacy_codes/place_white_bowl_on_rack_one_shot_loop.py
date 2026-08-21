# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded white-bowl-to-rack one-shot script for real YAM.

The script always starts with fresh live perception and a no-motion pickup
planner run. It then detects the rack, previews a bounded pickup/lift/place
sequence, and executes at most one physical attempt only when the physical gate
and ticket preconditions are present.

This file is intentionally standalone. It copies the guarded bowl loop instead
of wrapping the orange-bowl script, so the orange baseline can remain stable.

Rim-angle tuning note for white-bowl rack work:

- Trying another rim is reasonable, especially for lower-rack placement.
- Current white-bowl pickup has used:
  OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG=-90
- Planner rim angles are world-frame radial directions:
  0 deg    = +X rim
  90 deg   = +Y rim
  180 deg  = -X rim
  -90 deg  = -Y rim
- If the right-side rim in the top view is a bad grasp side, do not keep only
  -90. Prefer an ordered sweep such as:
  OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG=-90,0,180,90
  This tries the known pickup first, then lets no-motion preview choose another
  valid rim if the current side fails.
- If the desired grasp is the bottom part in the top image, first run a
  no-motion overlay and inspect the selected rim_angle_deg. Image-bottom is
  camera-frame language, but the planner uses world-frame angles; put the
  mapped world angle first only after checking the overlay.
- Do not blindly switch physical grasp side without the no-motion overlay.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    compare_detection_motion,
    current_run_dir,
    json_safe,
    read_gripper_width,
    staged_close_with_contact,
    verify_lift,
    write_json,
)


def _set_default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _configure_white_bowl_defaults() -> None:
    _set_default_env("OPENFORGE_BOWL_RACK_TASK_NAME", "place_white_bowl_on_rack_one_shot_loop")
    _set_default_env("OPENFORGE_BOWL_RACK_OBJECT_NAME", "white bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", "white_bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_SCRIPT_PATH", "cap/saved_scripts/place_white_bowl_on_rack_one_shot_loop.py")
    _set_default_env("OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH", "cap/saved_scripts/place_white_bowl_on_rack_one_shot_loop.py")
    _set_default_env("OPENFORGE_BOWL_RACK_VISIBLE_ENV", "OPENFORGE_BOWL_RACK_WHITE_BOWL_VISIBLE")
    _set_default_env("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "white_bowl_to_rack")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", "white bowl,bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS", "white bowl,bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_POST_PROMPTS", "white bowl,bowl,rack,wooden rack,blue gripper fingers")
    _set_default_env("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", "white bowl,bowl")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "top_down_bowl_rim")
    _set_default_env("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "top_down_bowl_rim_straddle")
    _set_default_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", "0")
    _set_default_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", "1")
    _set_default_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", "0")
    _set_default_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", "0")
    _set_default_env("OPENFORGE_PLATE_USE_TOP_BOX_RADIUS", "1")
    _set_default_env("OPENFORGE_PLATE_REFERENCE_BOX_LONG_AXIS_PX", "56.0")
    _set_default_env("OPENFORGE_PLATE_REFERENCE_RADIUS_M", "0.095")
    _set_default_env("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", "0.68")
    _set_default_env("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", "0.60")
    _set_default_env("OPENFORGE_BOWL_RACK_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.04")
    _set_default_env("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", "0.04")
    _set_default_env("OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS", "0.02")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_USE_MULTIVIEW_RIM_Z", "1")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_RAISE_M", "0.035")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_XY_DISAGREEMENT_M", "0.080")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M", "0.010")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_LIFT_Z_M", "0.150")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG", "-90")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_TCP_TO_CONTACT_OFFSETS_M", "0.000")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_FINGER_FORWARD_SIGNS", "-1.0")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_STEPS", "0.68,0.40,0.18,0.08,0.04")
    _set_default_env("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TARGET_POS", "0.04")
    _set_default_env("OPENFORGE_DEBUG_OBS_CAMERAS", "top,left,right,bottom")
    _set_default_env("OPENFORGE_DEBUG_OBS_IMAGE_ONLY_CAMERAS", "bottom")
    _set_default_env("OPENFORGE_BOWL_RACK_IMAGE_ONLY_CAMERAS", "bottom")


_configure_white_bowl_defaults()

TASK_NAME = os.environ.get("OPENFORGE_BOWL_RACK_TASK_NAME", "place_white_bowl_on_rack_one_shot_loop").strip()
TASK_NAME = TASK_NAME or "place_white_bowl_on_rack_one_shot_loop"
TASK_OBJECT_NAME = os.environ.get("OPENFORGE_BOWL_RACK_OBJECT_NAME", "white bowl").strip() or "white bowl"
TASK_ARTIFACT_PREFIX = (
    os.environ.get("OPENFORGE_BOWL_RACK_ARTIFACT_PREFIX", TASK_OBJECT_NAME.replace(" ", "_")).strip()
    or TASK_OBJECT_NAME.replace(" ", "_")
)
TASK_TICKET_SCOPE = os.environ.get("OPENFORGE_BOWL_RACK_EXPECTED_TICKET_SCOPE", "white_bowl_to_rack").strip()
TASK_TICKET_SCOPE = TASK_TICKET_SCOPE or "white_bowl_to_rack"
TASK_VISIBLE_ENV = os.environ.get(
    "OPENFORGE_BOWL_RACK_VISIBLE_ENV",
    "OPENFORGE_BOWL_RACK_WHITE_BOWL_VISIBLE",
).strip() or "OPENFORGE_BOWL_RACK_WHITE_BOWL_VISIBLE"
TASK_SCRIPT_PATH = os.environ.get(
    "OPENFORGE_BOWL_RACK_SCRIPT_PATH",
    "cap/saved_scripts/place_white_bowl_on_rack_one_shot_loop.py",
).strip() or "cap/saved_scripts/place_white_bowl_on_rack_one_shot_loop.py"
TASK_IMPLEMENTATION_PATH = os.environ.get(
    "OPENFORGE_BOWL_RACK_IMPLEMENTATION_PATH",
    "cap/saved_scripts/place_white_bowl_on_rack_one_shot_loop.py",
).strip() or "cap/saved_scripts/place_white_bowl_on_rack_one_shot_loop.py"
PLANNER_PATH = Path(os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_PATH", "cap/saved_scripts/pick_plate_no_motion_plan.py"))
REQUIRED_PICKUP_STRATEGY_MODE = "top_down_bowl_rim"
REQUIRED_PICKUP_STRATEGY_TYPE = "top_down_bowl_rim_straddle"
ALLOWED_TOP_DOWN_MODES = {"top_down_bowl_rim", "bowl_top_down", "top_down_rim_straddle"}
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


def _env_optional_float_setting(name: str) -> tuple[bool, float | None]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return False, None
    value = raw.strip()
    if value.lower() in {"none", "null", "omit", "default"}:
        return True, None
    try:
        return True, float(value)
    except ValueError:
        return False, None


def _gripper_limit_kwargs() -> dict[str, float | None]:
    kwargs: dict[str, float | None] = {}
    vel_is_set, vel_limit = _env_optional_float_setting("OPENFORGE_BOWL_RACK_GRIPPER_VEL_LIMIT")
    torque_is_set, torque_limit = _env_optional_float_setting("OPENFORGE_BOWL_RACK_GRIPPER_TORQUE_LIMIT")
    if vel_is_set:
        kwargs["vel_limit"] = vel_limit
    if torque_is_set:
        kwargs["torque_limit"] = torque_limit
    return kwargs


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
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _csv_float(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            pass
    return values or list(default)


def _default_pickup_prompts() -> str:
    return os.environ.get(
        "OPENFORGE_BOWL_RACK_DEFAULT_PICKUP_PROMPTS",
        f"{TASK_OBJECT_NAME},bowl",
    )


def _default_post_prompts() -> list[str]:
    return _csv(
        "OPENFORGE_BOWL_RACK_DEFAULT_POST_PROMPTS",
        [TASK_OBJECT_NAME, "bowl", "rack", "wooden rack", "blue gripper fingers"],
    )


def _object_detection_prompts() -> tuple[str, ...]:
    values: list[str] = []
    for prompt in _csv("OPENFORGE_BOWL_RACK_OBJECT_DETECTION_PROMPTS", [TASK_OBJECT_NAME, "bowl"]):
        lower = prompt.lower()
        if lower not in values:
            values.append(lower)
    return tuple(values)


def _offset_triples(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [list(item) for item in default]
    values: list[list[float]] = []
    for chunk in raw.replace("|", ";").split(";"):
        parts = [part.strip() for part in chunk.replace(":", ",").split(",") if part.strip()]
        if len(parts) not in {2, 3}:
            continue
        try:
            item = [float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) == 3 else 0.0]
        except ValueError:
            continue
        values.append(item)
    return values or [list(item) for item in default]


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


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
    if not value:
        return None
    raw = value.get("position_3d") or value.get("position") if isinstance(value, dict) else value
    if raw is None or len(raw) < 3:
        return None
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def _distance3(a: Any, b: Any) -> float | None:
    av = _xyz(a)
    bv = _xyz(b)
    if av is None or bv is None:
        return None
    return math.sqrt(sum((float(av[index]) - float(bv[index])) ** 2 for index in range(3)))


def _dot3(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        av = [float(value) for value in list(a)[:3]]
        bv = [float(value) for value in list(b)[:3]]
    except Exception:
        return None
    if len(av) < 3 or len(bv) < 3:
        return None
    return sum(av[index] * bv[index] for index in range(3))


def _quat_xyzw_axes_world(quat: Any) -> dict[str, Any]:
    try:
        x, y, z, w = [float(value) for value in list(quat)[:4]]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return {"error": "zero-length quaternion"}
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    return {
        "source": "robot_state_ee_quat_xyzw",
        "local_x_world": _round_list([matrix[row][0] for row in range(3)], limit=3),
        "local_y_world": _round_list([matrix[row][1] for row in range(3)], limit=3),
        "local_z_world": _round_list([matrix[row][2] for row in range(3)], limit=3),
    }


def _arm_state_from_observation(packet: dict[str, Any], side: str) -> dict[str, Any] | None:
    robot_state = packet.get("robot_state") or {}
    arms = robot_state.get("arms") if isinstance(robot_state, dict) else None
    arm = (arms or {}).get(side)
    return dict(arm) if isinstance(arm, dict) else None


def _pose_available(pose: Any) -> bool:
    return bool(
        isinstance(pose, dict)
        and len(pose.get("position") or []) >= 3
        and len(pose.get("rpy") or []) >= 3
    )


def _artifact_safe_label(value: Any) -> str:
    text = str(value or "candidate")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:80] or "candidate"


def _debug_observation_cameras() -> list[str]:
    raw = os.environ.get("OPENFORGE_BOWL_RACK_OBSERVATION_CAMERAS")
    if raw and raw.strip():
        cameras = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    else:
        cameras = _csv("OPENFORGE_DEBUG_OBS_CAMERAS", ["top", "left", "right", "bottom"])
    return cameras or ["top", "left", "right", "bottom"]


def _image_only_debug_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_BOWL_RACK_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_BOWL_RACK_RECORDING_WAIT_S", 5.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    packet: dict[str, Any] = {"required": True, "passed": bool(videos), "video_paths": videos}
    if not videos:
        packet.update(
            {
                "problem": "no mp4 files found; recording/debug evidence is required before physical motion",
                "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
                "blocker_before_physical": True,
            }
        )
    return packet


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_PLATE_XYZ",
        "OPENFORGE_PLATE_HALF_EXTENTS",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
        "OPENFORGE_BOWL_XYZ",
        "OPENFORGE_RACK_XYZ",
        "OPENFORGE_BOWL_RACK_OBJECT_XYZ",
        "OPENFORGE_BOWL_RACK_RACK_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _forbidden_pickup_strategy_env() -> list[str]:
    problems: list[str] = []
    requested_mode = os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE", "").strip().lower()
    if requested_mode and requested_mode not in ALLOWED_TOP_DOWN_MODES:
        problems.append(
            "OPENFORGE_BOWL_RACK_PICKUP_STRATEGY_MODE must be one of "
            f"{sorted(ALLOWED_TOP_DOWN_MODES)} for this task; got {requested_mode!r}"
        )
    raw_allowed = os.environ.get("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", "").strip()
    if raw_allowed:
        requested_types = set(_csv("OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES", []))
        disallowed = sorted(item for item in requested_types if item != REQUIRED_PICKUP_STRATEGY_TYPE)
        if disallowed:
            problems.append(
                "OPENFORGE_BOWL_RACK_ALLOWED_PICKUP_STRATEGY_TYPES may not admit non-top-down strategies; "
                f"disallowed={disallowed}"
            )
    return problems


def _pgrep_competing_motion(run_dir: Path | None = None) -> dict[str, Any]:
    try:
        output = subprocess.check_output(["pgrep", "-af", "run_script.py|ffmpeg"], text=True)
    except subprocess.CalledProcessError:
        output = ""
    self_pid = os.getpid()
    rows: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            pid = -1
        cmd = parts[1] if len(parts) > 1 else ""
        item = {"pid": pid, "cmd": cmd}
        rows.append(item)
        first_token = cmd.split(maxsplit=1)[0] if cmd else ""
        first_name = Path(first_token).name
        actual_ffmpeg = first_name == "ffmpeg" or first_token.endswith("/ffmpeg")
        actual_run_script = "run_script.py" in cmd and (
            first_name.startswith("python") or first_name in {"uv", "uvx"} or "/python" in first_token
        )
        own_recording = run_dir is not None and str(run_dir) in cmd and actual_ffmpeg
        if pid != self_pid and not own_recording and "debug_ui" not in cmd and (actual_ffmpeg or actual_run_script):
            competitors.append(item)
    return {"self_pid": self_pid, "rows": rows, "competitors": competitors}


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_path = Path(TASK_SCRIPT_PATH).resolve()
    implementation_path = Path(TASK_IMPLEMENTATION_PATH).resolve()
    snapshot = {
        "script_path": str(script_path),
        "implementation_path": str(implementation_path),
        "planner_path": str((Path.cwd() / PLANNER_PATH).resolve()),
        "task_name": TASK_NAME,
        "object_name": TASK_OBJECT_NAME,
        "artifact_prefix": TASK_ARTIFACT_PREFIX,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_BOWL_RACK_")
            or name.startswith("OPENFORGE_BOWL_TOP_DOWN_")
            or name
            in {
                "OPENFORGE_ALLOW_PHYSICAL_MOTION",
                "OPENFORGE_DEBUG_OBS_TIMEOUT_S",
                "YAM_STATION_CALIBRATED_XML",
                "CAP_TOP_CAMERA_BACKEND",
                "CAP_TOP_CAMERA_FRAME",
                "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP",
            }
        },
    }
    try:
        snapshot["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        paths = [str(script_path.relative_to(Path.cwd()))]
        implementation_rel = str(implementation_path.relative_to(Path.cwd()))
        if implementation_rel not in paths:
            paths.append(implementation_rel)
        snapshot["git_status_short"] = subprocess.check_output(["git", "status", "--short", "--", *paths], text=True).strip()
    except Exception as exc:
        snapshot["git_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["path"] = write_json(run_dir / "source_snapshot.json", snapshot)
    return snapshot


def _with_env_overrides(overrides: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for name, value in overrides.items():
        previous[name] = os.environ.get(name)
        os.environ[name] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _run_fresh_pickup_planner() -> dict[str, Any]:
    previous_gate = os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
    overrides = {
        "OPENFORGE_PLATE_TASK_NAME": f"{TASK_NAME}_pickup_plan",
        "OPENFORGE_PLATE_OBJECT_NAME": TASK_OBJECT_NAME,
        "OPENFORGE_PLATE_PROMPTS": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_PROMPTS", _default_pickup_prompts()),
        "OPENFORGE_PLATE_COLLECT_ALL_PROMPTS": "1",
        "OPENFORGE_PLATE_TARGET_Y_SIGN": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_TARGET_Y_SIGN", "negative"),
        "OPENFORGE_PLATE_STRATEGY_MODE": REQUIRED_PICKUP_STRATEGY_MODE,
        "OPENFORGE_PLATE_ARM": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_ARM", "right"),
        "OPENFORGE_PLATE_MAX_CANDIDATES": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_MAX_CANDIDATES", "96"),
        "OPENFORGE_PLATE_PLANNER_BACKEND": os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_BACKEND", "rrtconnect"),
        "OPENFORGE_PLATE_SOLVER_SPEED": os.environ.get("OPENFORGE_BOWL_RACK_SOLVER_SPEED", "fast"),
        "OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M": os.environ.get("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", "0.025"),
        "OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG": os.environ.get("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", "16"),
        "OPENFORGE_PLATE_IK_XYZ_WEIGHT": os.environ.get("OPENFORGE_BOWL_RACK_IK_XYZ_WEIGHT", "1.0"),
        "OPENFORGE_PLATE_IK_RPY_WEIGHT": os.environ.get("OPENFORGE_BOWL_RACK_IK_RPY_WEIGHT", "0.3"),
        "OPENFORGE_PLATE_PREVIEW_MAX_POS_ERROR_M": os.environ.get(
            "OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", "0.018"
        ),
        "OPENFORGE_PLATE_PREVIEW_MAX_ROT_ERROR_DEG": os.environ.get(
            "OPENFORGE_BOWL_RACK_PREVIEW_MAX_ROT_ERROR_DEG", "10"
        ),
    }
    previous = _with_env_overrides(overrides)
    try:
        planner_path = Path.cwd() / PLANNER_PATH
        spec = importlib.util.spec_from_file_location(f"openforge_{TASK_ARTIFACT_PREFIX}_pickup_plan", planner_path)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"could not locate planner path {PLANNER_PATH}")
        tool_globals = {
            name: value
            for name, value in globals().items()
            if callable(value)
            and name
            in {
                "detect_objects_oneshot",
                "freespace_move",
                "get_camera_image",
                "get_robot_state",
                "run_in_background",
            }
        }
        namespace = runpy.run_path(spec.origin, init_globals=tool_globals, run_name=f"__openforge_{TASK_ARTIFACT_PREFIX}_plan__")
        return dict(namespace.get("TASK_RESULT") or {})
    finally:
        _restore_env(previous)
        if previous_gate is not None:
            os.environ["OPENFORGE_ALLOW_PHYSICAL_MOTION"] = previous_gate


def _validate_pickup_strategy(planner_result: dict[str, Any]) -> dict[str, Any]:
    selected = planner_result.get("selected_strategy") or {}
    source = selected.get("source_detection") or {}
    source_name = str(source.get("source") or source.get("detection_source") or "")
    allowed_strategy_types = {REQUIRED_PICKUP_STRATEGY_TYPE}
    required = {
        "selected.label": selected.get("label"),
        "selected.arm": selected.get("arm"),
        "selected.strategy_type": selected.get("strategy_type"),
        "pregrasp_pose.position": (selected.get("pregrasp_pose") or {}).get("position"),
        "grasp_pose.position": (selected.get("grasp_pose") or {}).get("position"),
        "lift_pose.position": (selected.get("lift_pose") or {}).get("position"),
        "source_detection.position_3d": source.get("position_3d"),
    }
    missing = [name for name, value in required.items() if not value]
    strategy_type = selected.get("strategy_type")
    strategy_ok = strategy_type == REQUIRED_PICKUP_STRATEGY_TYPE
    residual_gate = selected.get("preview_residual_gate") or {}
    residual_ok = bool(residual_gate.get("passed", True))
    axis_gate = selected.get("axis_gate") or {}
    axis_ok = bool(axis_gate) and axis_gate.get("passed") is True
    label_text = str(selected.get("label") or "")
    label_ok = not label_text.startswith("plate_rim_side")
    rim_edge = selected.get("selected_rim_edge") or {}
    rim_edge_ok = bool(rim_edge.get("gap_center_m") or rim_edge.get("contact_center_m"))
    source_camera = str(source.get("source_camera") or source.get("camera") or "")
    top_camera_required = _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_TOP_CAMERA_PICKUP", True)
    camera_ok = (not top_camera_required) or source_camera == "top"
    manual_source = "manual" in source_name.lower()
    passed = bool(
        planner_result.get("success")
        and selected
        and not missing
        and strategy_ok
        and label_ok
        and rim_edge_ok
        and residual_ok
        and axis_ok
        and camera_ok
        and not manual_source
    )
    return {
        "passed": passed,
        "missing": missing,
        "strategy_ok": strategy_ok,
        "strategy_type": strategy_type,
        "label_ok": label_ok,
        "rim_edge_ok": rim_edge_ok,
        "residual_gate_passed": residual_ok,
        "axis_gate_passed": axis_ok,
        "axis_gate": axis_gate,
        "axis_gate_required": True,
        "camera_ok": camera_ok,
        "source_camera": source_camera,
        "manual_source": manual_source,
        "selected_label": selected.get("label"),
        "arm": selected.get("arm"),
        "allowed_strategy_types": sorted(allowed_strategy_types),
        "source_detection": source,
        "selected_strategy": selected if passed else None,
    }


def _detection_touches_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_BOWL_RACK_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_BOWL_RACK_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_BOWL_RACK_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _select_rack_detection(packet: dict[str, Any]) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    min_score = _env_float("OPENFORGE_BOWL_RACK_MIN_SCORE", 0.20)
    prompt_order = {
        "wooden rack": 0,
        "rack": 1,
        "wooden shelf": 2,
        "shelf": 3,
        "box": 4,
    }
    filtered: list[dict[str, Any]] = []
    for det in detections:
        xyz = _xyz(det)
        camera = str(det.get("source_camera") or det.get("camera") or "")
        score = float(det.get("score") or 0.0)
        if xyz is None or camera not in {"top", "left", "right"} or score < min_score:
            continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        filtered.append(det)
    if not filtered:
        return None
    target_y_sign = os.environ.get("OPENFORGE_BOWL_RACK_TARGET_Y_SIGN", "positive").strip().lower()

    def _key(det: dict[str, Any]) -> tuple[int, int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "").lower()
        xyz = _xyz(det) or [0.0, 0.0, 0.0]
        y_rank = 0
        if target_y_sign in {"positive", "pos", "left", "left_side"}:
            y_rank = 0 if xyz[1] > 0.0 else 1
        elif target_y_sign in {"negative", "neg", "right", "right_side"}:
            y_rank = 0 if xyz[1] < 0.0 else 1
        return (
            0 if camera == "top" else 1,
            y_rank,
            1 if det.get("touches_image_edge") and camera != "top" else 0,
            prompt_order.get(prompt, 9),
            -float(det.get("score") or 0.0),
        )

    selected = sorted(filtered, key=_key)[0]
    selected["detection_source"] = "live_top_camera" if selected.get("source_camera") == "top" else "live_calibrated_camera"
    return selected


def _observe_rack(run_dir: Path) -> dict[str, Any]:
    prompts = _csv("OPENFORGE_BOWL_RACK_TARGET_PROMPTS", ["wooden rack", "rack", "wooden shelf", "shelf", "box"])
    packet = capture_scene(
        prompts=prompts,
        cameras=_debug_observation_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="observe_rack_target",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=1,
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
        image_only_cameras=_image_only_debug_cameras(),
    )
    selected = _select_rack_detection(packet)
    packet["selected_rack_detection"] = selected
    packet["path"] = write_json(run_dir / "observations" / "observe_rack_target_selected.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## rack detection",
            f"- selected_rack_detection: {selected}",
            f"- packet: {packet.get('packet_path')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _make_place_candidates(selected_pickup: dict[str, Any], rack_detection: dict[str, Any]) -> list[dict[str, Any]]:
    rack_xyz = _xyz(rack_detection)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    grasp_xyz = grasp_pose.get("position") or []
    if rack_xyz is None or source_xyz is None or len(grasp_xyz) < 3:
        raise RuntimeError("cannot build place candidates without rack, source, and grasp poses")
    tcp_above_source = float(grasp_xyz[2]) - float(source_xyz[2])
    if tcp_above_source < 0.05 or tcp_above_source > 0.25:
        tcp_above_source = _env_float("OPENFORGE_BOWL_RACK_DEFAULT_TCP_ABOVE_OBJECT_M", 0.135)
    object_ref_above_support = _env_float("OPENFORGE_BOWL_RACK_OBJECT_REF_ABOVE_SUPPORT_M", 0.035)
    release_clearance = _env_float("OPENFORGE_BOWL_RACK_RELEASE_CLEARANCE_M", 0.015)
    high_lift = _env_float("OPENFORGE_BOWL_RACK_HIGH_PLACE_Z_LIFT_M", 0.065)
    retreat_lift = _env_float("OPENFORGE_BOWL_RACK_RETREAT_Z_LIFT_M", 0.075)
    rpy = [float(v) for v in list(grasp_pose.get("rpy") or [])[:3]]
    if len(rpy) < 3:
        raise RuntimeError("selected pickup grasp pose has no rpy")
    offsets = _offset_triples(
        "OPENFORGE_BOWL_RACK_PLACE_OFFSETS_M",
        [[0.0, -0.06, 0.0], [0.0, -0.09, 0.0], [0.0, 0.0, 0.0], [-0.03, -0.06, 0.0], [0.03, -0.06, 0.0]],
    )
    max_candidates = max(1, _env_int("OPENFORGE_BOWL_RACK_MAX_PLACE_CANDIDATES", 8))
    candidates: list[dict[str, Any]] = []
    for index, offset in enumerate(offsets[:max_candidates]):
        place_xy = [rack_xyz[0] + offset[0], rack_xyz[1] + offset[1]]
        release_z = (
            rack_xyz[2]
            + tcp_above_source
            + object_ref_above_support
            + release_clearance
            + float(offset[2])
        )
        release_pose = {"position": [place_xy[0], place_xy[1], release_z], "rpy": rpy}
        high_pose = {"position": [place_xy[0], place_xy[1], release_z + high_lift], "rpy": rpy}
        retreat_pose = {"position": [place_xy[0], place_xy[1], release_z + retreat_lift], "rpy": rpy}
        candidates.append(
            {
                "label": f"rack_place_{index}",
                "arm": selected_pickup["arm"],
                "strategy_type": f"{selected_pickup.get('strategy_type', 'bowl')}_release_on_rack",
                "rack_detection": rack_detection,
                "pickup_strategy_label": selected_pickup.get("label"),
                "place_target_object_ref_m": [
                    round(place_xy[0], 5),
                    round(place_xy[1], 5),
                    round(rack_xyz[2] + object_ref_above_support + release_clearance + float(offset[2]), 5),
                ],
                "rack_surface_reference_m": _round_list(rack_xyz, limit=3),
                "relative_offset_m": _round_list(offset, limit=3),
                "release_pose": release_pose,
                "high_place_pose": high_pose,
                "retreat_pose": retreat_pose,
                "geometry_model": {
                    "source": "fresh pickup and rack detections",
                    "tcp_above_source_detection_m": round(tcp_above_source, 5),
                    "object_ref_above_support_m": round(object_ref_above_support, 5),
                    "release_clearance_m": round(release_clearance, 5),
                    "high_place_z_lift_m": round(high_lift, 5),
                    "retreat_z_lift_m": round(retreat_lift, 5),
                    "policy": "release with the same pickup grasp orientation, then retreat upward open",
                },
            }
        )
    return candidates


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None, preview_only: bool = False) -> dict[str, Any]:
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_BOWL_RACK_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_BOWL_RACK_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", 16.0),
        "ik_xyz_weight": _env_float("OPENFORGE_BOWL_RACK_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_BOWL_RACK_IK_RPY_WEIGHT", 0.3),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _error_within_limit(value: Any, limit: float) -> bool:
    if value is None:
        return True
    try:
        return float(value) <= float(limit)
    except Exception:
        return False


def _stage_allows_axis_gated_rotation_residual(label: str) -> bool:
    if not _truthy_env("OPENFORGE_BOWL_RACK_ALLOW_AXIS_GATED_ROTATION_RESIDUAL", True):
        return False
    label_text = str(label)
    return label_text.startswith("bowl_")


def _stage_tolerance(label: str, *, execution: bool = False) -> dict[str, float]:
    prefix = "OPENFORGE_BOWL_RACK_EXEC" if execution else "OPENFORGE_BOWL_RACK_PREVIEW"
    default_pos = _env_float(f"{prefix}_MAX_POS_ERROR_M", 0.018)
    default_rot = _env_float(f"{prefix}_MAX_ROT_ERROR_DEG", 10.0)
    loose_pos = max(default_pos, 0.025)
    loose_rot = max(default_rot, 16.0)
    label_text = str(label)
    if "high" in label_text or "transport" in label_text or "retreat" in label_text:
        return {"max_pos_error_m": loose_pos, "max_rot_error_deg": loose_rot}
    return {"max_pos_error_m": default_pos, "max_rot_error_deg": default_rot}


def _preview_summary(result: Any, *, stage_label: str = "") -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", False))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    tolerance = _stage_tolerance(stage_label, execution=False)
    status_ok = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    axis_gated_rotation = _stage_allows_axis_gated_rotation_residual(stage_label)
    pos_ok = _error_within_limit(pos_error, tolerance["max_pos_error_m"])
    rot_ok = axis_gated_rotation or _error_within_limit(rot_error, tolerance["max_rot_error_deg"])
    tolerance_ok = pos_ok and rot_ok
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "trajectory_cache_key": _field(result, "trajectory_cache_key", None),
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": tolerance["max_pos_error_m"],
        "max_rot_error_deg": tolerance["max_rot_error_deg"],
        "tolerance_ok": tolerance_ok,
        "position_tolerance_ok": pos_ok,
        "rotation_tolerance_ok": rot_ok,
        "axis_gated_rotation_residual": axis_gated_rotation,
        "rotation_residual_policy": (
            "ignored for bowl stage because selected pickup already passed top-down axis_gate"
            if axis_gated_rotation
            else "raw motion-tool rotation residual must be within limit"
        ),
        "success": bool(status_ok and tolerance_ok),
        "raw": json_safe(result),
    }


def _execute_summary(result: Any, *, stage_label: str = "") -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", True))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    tolerance = _stage_tolerance(stage_label, execution=True)
    status_ok = bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed)
    axis_gated_rotation = _stage_allows_axis_gated_rotation_residual(stage_label)
    pos_ok = _error_within_limit(pos_error, tolerance["max_pos_error_m"])
    rot_ok = axis_gated_rotation or _error_within_limit(rot_error, tolerance["max_rot_error_deg"])
    tolerance_ok = pos_ok and rot_ok
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": tolerance["max_pos_error_m"],
        "max_rot_error_deg": tolerance["max_rot_error_deg"],
        "tolerance_ok": tolerance_ok,
        "position_tolerance_ok": pos_ok,
        "rotation_tolerance_ok": rot_ok,
        "axis_gated_rotation_residual": axis_gated_rotation,
        "rotation_residual_policy": (
            "ignored for bowl stage because selected pickup already passed top-down axis_gate"
            if axis_gated_rotation
            else "raw motion-tool rotation residual must be within limit"
        ),
        "success": bool(status_ok and tolerance_ok),
        "raw": json_safe(result),
    }


def _preview_pose(
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    *,
    label: str,
    gripper: float | None,
) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        freespace_move,
        _env_float("OPENFORGE_BOWL_RACK_PREVIEW_TIMEOUT_S", 20.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pose, gripper=gripper, preview_only=True),
    )
    if not result.get("ok"):
        return {
            "stage_label": label,
            "success": False,
            "status": None,
            "reason": result.get("error"),
            "executed": False,
        }
    return _preview_summary(result.get("data"), stage_label=label)


def _preshape_gripper(selected: dict[str, Any]) -> float:
    override = os.environ.get("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS")
    if override not in {None, ""}:
        return float(override)
    preshape_plan = selected.get("preshape_plan") or {}
    return float(
        preshape_plan.get(
            "expected_gripper_open_before_close_normalized",
            _env_float("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", 0.68),
        )
    )


def _hold_gripper(selected: dict[str, Any]) -> float:
    close_plan = selected.get("staged_close_plan") or {}
    return _env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", float(close_plan.get("target_normalized_position", 0.40)))


def _use_high_approach(selected_pickup: dict[str, Any]) -> bool:
    default = str(selected_pickup.get("strategy_type") or "") != REQUIRED_PICKUP_STRATEGY_TYPE
    return _truthy_env("OPENFORGE_BOWL_RACK_USE_HIGH_APPROACH", default)


def _pickup_retreat_pose(selected_pickup: dict[str, Any]) -> dict[str, Any]:
    if _use_high_approach(selected_pickup) and _pose_available(selected_pickup.get("approach_pose")):
        return selected_pickup["approach_pose"]
    if _pose_available(selected_pickup.get("low_pregrasp_pose")):
        return selected_pickup["low_pregrasp_pose"]
    return selected_pickup["pregrasp_pose"]


def _pickup_lift_specs(selected_pickup: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    pregrasp_open = _preshape_gripper(selected_pickup)
    hold = _hold_gripper(selected_pickup)
    specs: list[tuple[str, dict[str, Any], float]] = []
    if _use_high_approach(selected_pickup) and _pose_available(selected_pickup.get("approach_pose")):
        specs.append(("bowl_high_approach", selected_pickup["approach_pose"], pregrasp_open))
    if _pose_available(selected_pickup.get("low_pregrasp_pose")):
        specs.append(("bowl_low_pregrasp", selected_pickup["low_pregrasp_pose"], pregrasp_open))
    else:
        specs.append(("bowl_pregrasp", selected_pickup["pregrasp_pose"], pregrasp_open))
    specs.extend([("bowl_grasp_pose", selected_pickup["grasp_pose"], pregrasp_open), ("bowl_small_lift", selected_pickup["lift_pose"], hold)])
    return specs


def _sequence_specs(selected_pickup: dict[str, Any], place_candidate: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    hold = _hold_gripper(selected_pickup)
    specs = _pickup_lift_specs(selected_pickup)
    specs.extend(
        [
            ("rack_high_place", place_candidate["high_place_pose"], hold),
            ("rack_release_pose", place_candidate["release_pose"], hold),
            ("rack_retreat_after_open", place_candidate["retreat_pose"], 1.0),
        ]
    )
    return specs


def _split_physical_sequence(
    selected_pickup: dict[str, Any], place_candidate: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, Any], float]], list[tuple[str, dict[str, Any], float]]]:
    before_close: list[tuple[str, dict[str, Any], float]] = []
    after_close: list[tuple[str, dict[str, Any], float]] = []
    saw_lift = False
    for spec in _sequence_specs(selected_pickup, place_candidate):
        label = spec[0]
        if label == "bowl_small_lift":
            saw_lift = True
        if saw_lift:
            after_close.append(spec)
        else:
            before_close.append(spec)
    if not saw_lift:
        raise RuntimeError("physical sequence missing bowl_small_lift stage")
    return before_close, after_close


def _preview_place_sequence(
    run_dir: Path,
    selected_pickup: dict[str, Any],
    place_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    attempts: list[dict[str, Any]] = []
    for candidate in place_candidates:
        steps: list[dict[str, Any]] = []
        success = True
        for label, pose, gripper in _sequence_specs(selected_pickup, candidate):
            preview = _preview_pose(freespace_move, side, pose, label=label, gripper=gripper)
            step = {"stage": label, "pose": pose, "gripper": gripper, "preview": preview, "ok": bool(preview.get("success"))}
            steps.append(step)
            if not step["ok"]:
                success = False
                break
        attempt = {
            "label": candidate.get("label"),
            "success": success,
            "failed_stage": next((step["stage"] for step in steps if not step.get("ok")), None),
            "place_candidate": candidate,
            "steps": steps,
        }
        attempts.append(attempt)
        if success:
            packet = {
                "schema": "openforge.bowl_rack.sequence_preview.v1",
                "success": True,
                "side": side,
                "selected_pickup": selected_pickup,
                "selected_place": candidate,
                "attempts": attempts,
            }
            packet["path"] = write_json(run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_to_rack_sequence_preview.json", packet)
            append_stage_summary(
                run_dir,
                [
                    f"## {TASK_OBJECT_NAME} to rack sequence preview",
                    "- success: True",
                    f"- selected_place: {candidate.get('label')}",
                    f"- release_pose: {candidate.get('release_pose')}",
                    f"- path: {packet['path']}",
                ],
            )
            return packet
    packet = {
        "schema": "openforge.bowl_rack.sequence_preview.v1",
        "success": False,
        "side": side,
        "selected_pickup": selected_pickup,
        "selected_place": None,
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_to_rack_sequence_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            f"## {TASK_OBJECT_NAME} to rack sequence preview",
            "- success: False",
            f"- attempts: {[(item.get('label'), item.get('failed_stage')) for item in attempts]}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _preview_pickup_lift_sequence(run_dir: Path, selected_pickup: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    steps: list[dict[str, Any]] = []
    success = True
    for label, pose, gripper in _pickup_lift_specs(selected_pickup):
        preview = _preview_pose(freespace_move, side, pose, label=label, gripper=gripper)
        if not preview.get("success"):
            accepted = _accept_staged_bowl_transition_from_direct_preview(selected_pickup, label, pose, steps, preview)
            if accepted:
                preview = accepted
        step = {"stage": label, "pose": pose, "gripper": gripper, "preview": preview, "ok": bool(preview.get("success"))}
        steps.append(step)
        if not step["ok"]:
            success = False
            break
    packet = {
        "schema": "openforge.bowl_rack.pickup_lift_preview.v1",
        "success": success,
        "side": side,
        "selected_pickup": selected_pickup,
        "failed_stage": next((step["stage"] for step in steps if not step.get("ok")), None),
        "steps": steps,
    }
    packet["path"] = write_json(run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_pickup_lift_sequence_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            f"## {TASK_OBJECT_NAME} pickup/lift sequence preview",
            f"- success: {success}",
            f"- failed_stage: {packet['failed_stage']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _direct_preview_gate_check(selected_pickup: dict[str, Any], stage: str) -> dict[str, Any] | None:
    gate = selected_pickup.get("preview_residual_gate") or {}
    for check in gate.get("checks") or []:
        if check.get("stage") == stage and check.get("passed") is True:
            return check
    return None


def _pose_close(a: Any, b: Any, *, max_pos_m: float = 0.002, max_rpy_deg: float = 0.1) -> bool:
    if not (_pose_available(a) and _pose_available(b)):
        return False
    pos_error = _distance3(a, b)
    if pos_error is None or pos_error > max_pos_m:
        return False
    try:
        ar = [float(v) for v in a["rpy"][:3]]
        br = [float(v) for v in b["rpy"][:3]]
    except Exception:
        return False
    return all(abs(ar[index] - br[index]) <= max_rpy_deg for index in range(3))


def _accept_staged_bowl_transition_from_direct_preview(
    selected_pickup: dict[str, Any],
    label: str,
    pose: dict[str, Any],
    prior_steps: list[dict[str, Any]],
    failed_preview: dict[str, Any],
) -> dict[str, Any] | None:
    if not _truthy_env("OPENFORGE_BOWL_RACK_ACCEPT_DIRECT_STAGE_FOR_SEQUENCE_PREVIEW", True):
        return None
    stage_map = {
        "bowl_low_pregrasp": ("pregrasp_pose", {"bowl_high_approach"}, "pregrasp_pose"),
        "bowl_pregrasp": ("pregrasp_pose", {"bowl_high_approach"}, "pregrasp_pose"),
        "bowl_grasp_pose": ("grasp_pose", {"bowl_low_pregrasp", "bowl_pregrasp"}, "grasp_pose"),
        "bowl_small_lift": ("lift_pose", {"bowl_grasp_pose"}, "lift_pose"),
    }
    if label not in stage_map:
        return None
    direct_stage, required_prior_stages, pose_key = stage_map[label]
    prior_ok = any(step.get("stage") in required_prior_stages and step.get("ok") for step in prior_steps)
    if label in {"bowl_low_pregrasp", "bowl_pregrasp"} and not _use_high_approach(selected_pickup):
        return None
    if not prior_ok:
        return None
    direct_check = _direct_preview_gate_check(selected_pickup, direct_stage)
    if not direct_check:
        return None
    if not _pose_close(pose, selected_pickup.get(pose_key)):
        return None
    if label in {"bowl_low_pregrasp", "bowl_pregrasp"}:
        approach_xyz = _xyz(selected_pickup.get("approach_pose") or {})
        low_xyz = _xyz(pose)
        if not approach_xyz or not low_xyz or float(approach_xyz[2]) <= float(low_xyz[2]):
            return None
    accepted = dict(failed_preview)
    accepted.update(
        {
            "success": True,
            "status": "AcceptedDirectStagedPreview",
            "reason": (
                f"accepted {label} sequence preview from selected candidate's direct "
                f"{direct_stage} residual gate plus a reachable prior staged pose; "
                "physical execution will still re-preview from the actually reached "
                "previous pose before executing this stage"
            ),
            "accepted_direct_staged_preview": True,
            "direct_stage": direct_stage,
            "direct_gate_check": direct_check,
            "failed_sequence_preview": failed_preview,
        }
    )
    return accepted


def _ticket_status(run_dir: Path) -> dict[str, Any]:
    required_true = [
        "OPENFORGE_BOWL_RACK_PHYSICAL_TICKET",
        "OPENFORGE_BOWL_RACK_OPERATOR_PRESENT",
        "OPENFORGE_BOWL_RACK_ESTOP_REACHABLE",
        "OPENFORGE_BOWL_RACK_WORKSPACE_CLEAR",
        "OPENFORGE_BOWL_RACK_HANDS_TOOLS_CABLES_CLEAR",
        TASK_VISIBLE_ENV,
        "OPENFORGE_BOWL_RACK_RACK_CLEAR",
        "OPENFORGE_BOWL_RACK_NO_LIQUID",
        "OPENFORGE_BOWL_RACK_NO_MANUAL_XYZ",
        "OPENFORGE_BOWL_RACK_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_BOWL_RACK_FOLLOWER_SERVERS_UP",
        "OPENFORGE_BOWL_RACK_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_BOWL_RACK_TICKET_SCOPE", "").strip()
    if scope != TASK_TICKET_SCOPE:
        missing.append(f"OPENFORGE_BOWL_RACK_TICKET_SCOPE={TASK_TICKET_SCOPE}")
    budget = _env_int("OPENFORGE_BOWL_RACK_PHYSICAL_ATTEMPT_BUDGET", 0)
    max_budget = _env_int("OPENFORGE_BOWL_RACK_MAX_ALLOWED_ATTEMPT_BUDGET", 20)
    if budget < 1 or budget > max_budget:
        missing.append(f"OPENFORGE_BOWL_RACK_PHYSICAL_ATTEMPT_BUDGET=1..{max_budget}")
    manual_env = _manual_pose_env_vars_present()
    if manual_env:
        missing.extend(manual_env)
    pgrep_status = _pgrep_competing_motion(run_dir)
    if pgrep_status["competitors"]:
        missing.append("no competing run_script.py/ffmpeg process")
    return {
        "passed": not missing,
        "missing_or_invalid": missing,
        "ticket_scope": scope,
        "attempt_budget": budget,
        "max_allowed_attempt_budget": max_budget,
        "pgrep": pgrep_status,
    }


def _set_gripper_checked(
    *,
    side: str,
    target: float,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    stage: str,
    confirm_min: float | None = None,
    confirm_max: float | None = None,
    vel_limit: float | None = None,
    torque_limit: float | None = None,
) -> dict[str, Any]:
    before = read_gripper_width(get_robot_state, side)
    command_kwargs: dict[str, float] = {}
    if vel_limit is not None:
        command_kwargs["vel_limit"] = float(vel_limit)
    if torque_limit is not None:
        command_kwargs["torque_limit"] = float(torque_limit)
    result = set_gripper(side, target, **command_kwargs)
    time.sleep(_env_float("OPENFORGE_BOWL_RACK_GRIPPER_SETTLE_S", 0.25))
    after = read_gripper_width(get_robot_state, side)
    ok = True
    if confirm_min is not None and after is not None:
        deadline = time.monotonic() + _env_float("OPENFORGE_BOWL_RACK_GRIPPER_CONFIRM_TIMEOUT_S", 2.5)
        while after < confirm_min and time.monotonic() < deadline:
            time.sleep(0.1)
            after = read_gripper_width(get_robot_state, side)
        ok = after is None or after >= confirm_min
    if confirm_max is not None and after is not None:
        ok = ok and after <= confirm_max
    return {
        "stage": stage,
        "side": side,
        "target": target,
        "before": before,
        "after": after,
        "confirm_min": confirm_min,
        "confirm_max": confirm_max,
        "ok": ok,
        "result": json_safe(result),
    }


def _execute_pose(
    *,
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    label: str,
    gripper: float | None,
) -> dict[str, Any]:
    preview = freespace_move(**_move_kwargs(side, pose, gripper=gripper, preview_only=True))
    preview_packet = _preview_summary(preview, stage_label=label)
    if not preview_packet["success"] or not preview_packet.get("trajectory_cache_key"):
        raise RuntimeError(f"{label} preview failed before execution: {preview_packet}")
    executed = freespace_move(trajectory_cache_key=preview_packet["trajectory_cache_key"])
    execute_packet = _execute_summary(executed, stage_label=label)
    if not execute_packet["success"]:
        raise RuntimeError(f"{label} execution failed: {execute_packet}")
    return {
        "stage": label,
        "side": side,
        "pose": pose,
        "gripper": gripper,
        "preview": preview_packet,
        "execute": execute_packet,
        "trajectory_cache_key": preview_packet["trajectory_cache_key"],
    }


def _capture_object_observation(run_dir: Path, stage: str) -> dict[str, Any]:
    prompts = _csv("OPENFORGE_BOWL_RACK_POST_PROMPTS", _default_post_prompts())
    packet = capture_scene(
        prompts=prompts,
        cameras=_debug_observation_cameras(),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=1,
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
        image_only_cameras=_image_only_debug_cameras(),
    )
    packet["path"] = write_json(run_dir / "observations" / f"{_artifact_safe_label(stage)}.json", packet)
    return packet


def _select_object_detection(
    packet: dict[str, Any],
    preferred_prompts: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    if preferred_prompts is None:
        preferred_prompts = _object_detection_prompts()
    detections = [dict(det) for det in packet.get("all_detections") or []]
    prompt_rank = {prompt: index for index, prompt in enumerate(preferred_prompts)}
    filtered = []
    for det in detections:
        prompt = str(det.get("prompt") or "").lower()
        camera = str(det.get("source_camera") or det.get("camera") or "")
        if prompt not in prompt_rank or camera not in {"top", "left", "right"}:
            continue
        if float(det.get("score") or 0.0) < _env_float("OPENFORGE_BOWL_RACK_POST_MIN_SCORE", 0.20):
            continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        filtered.append(det)
    if not filtered:
        return None

    def _key(det: dict[str, Any]) -> tuple[int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "").lower()
        return (
            0 if camera == "top" else 1,
            1 if det.get("touches_image_edge") and camera != "top" else 0,
            prompt_rank.get(prompt, 9),
            -float(det.get("score") or 0.0),
        )

    return sorted(filtered, key=_key)[0]


def _xy_distance(a: Any, b: Any) -> float | None:
    av = _xyz(a)
    bv = _xyz(b)
    if av is None or bv is None:
        return None
    dx = av[0] - bv[0]
    dy = av[1] - bv[1]
    return math.sqrt(dx * dx + dy * dy)


def _select_gripper_detection(packet: dict[str, Any], side: str) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    side_camera = "right" if side == "right" else "left"
    prompts = {
        "blue gripper fingers": 0,
        f"{side} gripper": 1,
        "gripper fingers": 2,
        "robot gripper": 3,
    }
    filtered: list[dict[str, Any]] = []
    for det in detections:
        prompt = str(det.get("prompt") or "").lower()
        camera = str(det.get("source_camera") or det.get("camera") or "")
        if prompt not in prompts or camera not in {"top", "left", "right"}:
            continue
        if float(det.get("score") or 0.0) < _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_GRIPPER_MIN_SCORE", 0.18):
            continue
        xyz = _xyz(det)
        if xyz is not None and camera == "top":
            side_y_margin = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_GRIPPER_SIDE_Y_MARGIN_M", 0.08)
            if side == "right" and xyz[1] > side_y_margin:
                continue
            if side == "left" and xyz[1] < -side_y_margin:
                continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        filtered.append(det)
    if not filtered:
        return None

    def _key(det: dict[str, Any]) -> tuple[int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "").lower()
        return (
            0 if camera == side_camera else 1 if camera == "top" else 2,
            1 if det.get("touches_image_edge") and camera != side_camera else 0,
            prompts.get(prompt, 9),
            -float(det.get("score") or 0.0),
        )

    return sorted(filtered, key=_key)[0]


def _adjusted_contact_from_current_object(selected_pickup: dict[str, Any], current_object: dict[str, Any]) -> list[float] | None:
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    current_xyz = _xyz(current_object)
    rim = selected_pickup.get("selected_rim_edge") or {}
    planned_contact = _xyz(
        rim.get("contact_center_m")
        or rim.get("gap_center_m")
        or (selected_pickup.get("grasp_pose") or {}).get("position")
        or selected_pickup.get("position")
        or {}
    )
    if source_xyz is None or current_xyz is None or planned_contact is None:
        return None
    # Keep the planned rim/contact offset but translate it to the current object
    # detection. This catches stale or occlusion-shifted object centers before close.
    return [
        planned_contact[0] + (current_xyz[0] - source_xyz[0]),
        planned_contact[1] + (current_xyz[1] - source_xyz[1]),
        planned_contact[2] + (current_xyz[2] - source_xyz[2]),
    ]


def _pre_close_visual_gate(run_dir: Path, selected_pickup: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    side = str(selected_pickup.get("arm") or "right")
    obj = _select_object_detection(observation)
    gripper = _select_gripper_detection(observation, side)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    rim = selected_pickup.get("selected_rim_edge") or {}
    planned_contact = _xyz(
        rim.get("contact_center_m")
        or rim.get("gap_center_m")
        or (selected_pickup.get("grasp_pose") or {}).get("position")
        or selected_pickup.get("position")
        or {}
    )
    adjusted_contact = _adjusted_contact_from_current_object(selected_pickup, obj or {})
    object_xyz = _xyz(obj or {})
    gripper_xyz = _xyz(gripper or {})
    axis_gate = selected_pickup.get("axis_gate") or {}
    arm_state = _arm_state_from_observation(observation, side)
    ee_xyz = _xyz((arm_state or {}).get("ee_pos") or [])
    ee_quat = (arm_state or {}).get("ee_quat")
    ee_axes = _quat_xyzw_axes_world(ee_quat)

    max_object_xy_drift = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_XY_DRIFT_M", 0.055)
    max_object_z_jump = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_Z_JUMP_M", 0.060)
    max_gripper_xy_to_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_GRIPPER_XY_TO_CONTACT_M", 0.075)
    max_gripper_above_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_GRIPPER_ABOVE_CONTACT_M", 0.045)
    max_gripper_below_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_GRIPPER_BELOW_CONTACT_M", 0.035)
    max_ee_to_grasp = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_EE_TO_GRASP_M", 0.030)
    max_tcp_above_deviation = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_TCP_ABOVE_DEVIATION_M", 0.030)
    min_robot_axis_alignment = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MIN_ROBOT_AXIS_ALIGNMENT", 0.90)
    max_object_xy_to_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_XY_TO_CONTACT_M", 0.050)
    max_object_z_to_contact = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_OBJECT_Z_TO_CONTACT_M", 0.040)
    max_contact_above_object_z = _env_float("OPENFORGE_BOWL_RACK_PRE_CLOSE_MAX_CONTACT_ABOVE_OBJECT_Z_M", 0.045)

    object_xy_drift = _xy_distance(source_xyz, object_xyz) if source_xyz and object_xyz else None
    object_z_delta = float(object_xyz[2]) - float(source_xyz[2]) if source_xyz and object_xyz else None
    object_depth_jump = object_z_delta is not None and abs(object_z_delta) > max_object_z_jump
    object_drift_ok = object_xy_drift is not None and object_xy_drift <= max_object_xy_drift
    object_xy_to_contact = _xy_distance(object_xyz, planned_contact) if object_xyz and planned_contact else None
    object_z_delta_to_contact = (
        float(object_xyz[2]) - float(planned_contact[2]) if object_xyz is not None and planned_contact is not None else None
    )
    contact_above_object_z = (
        float(planned_contact[2]) - float(object_xyz[2]) if object_xyz is not None and planned_contact is not None else None
    )
    object_contact_xy_ok = object_xy_to_contact is not None and object_xy_to_contact <= max_object_xy_to_contact
    object_contact_z_ok = object_z_delta_to_contact is not None and abs(object_z_delta_to_contact) <= max_object_z_to_contact
    contact_above_object_z_ok = contact_above_object_z is not None and contact_above_object_z <= max_contact_above_object_z
    object_contact_coherent = bool(obj and object_xyz and object_contact_xy_ok and object_contact_z_ok)
    object_center_coherent = bool(obj and object_xyz and object_drift_ok and not object_depth_jump)
    object_detection_coherent = bool(object_contact_coherent or object_center_coherent)
    object_reference_mode = "planned_contact_unverified_by_current_rim_detection"
    contact_for_gate = planned_contact
    if object_contact_coherent:
        object_reference_mode = "planned_contact_verified_by_current_rim_detection"
        contact_for_gate = planned_contact
    elif object_center_coherent:
        object_reference_mode = "planned_contact_with_current_center_drift_check"
        contact_for_gate = planned_contact
    elif not object_detection_coherent and _truthy_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_OCCLUDED_SOURCE_REFERENCE", True):
        object_reference_mode = "planned_source_due_occluded_or_incoherent_detection"
        contact_for_gate = planned_contact

    gripper_xy_to_contact = _xy_distance(gripper_xyz, adjusted_contact) if gripper_xyz and adjusted_contact else None
    gripper_z_delta_to_contact = (
        float(gripper_xyz[2]) - float(adjusted_contact[2]) if gripper_xyz is not None and adjusted_contact is not None else None
    )
    grasp_pose = selected_pickup.get("grasp_pose") or selected_pickup
    grasp_xyz = _xyz(grasp_pose)
    ee_to_grasp = _distance3(ee_xyz, grasp_xyz) if ee_xyz and grasp_xyz else None
    ee_z_above_contact = (
        float(ee_xyz[2]) - float(contact_for_gate[2]) if ee_xyz is not None and contact_for_gate is not None else None
    )
    rim_model = selected_pickup.get("rim_geometry_estimate") or {}
    expected_tcp_above = rim_model.get("tcp_above_rim_m")
    if expected_tcp_above is None and grasp_xyz is not None and planned_contact is not None:
        expected_tcp_above = float(grasp_xyz[2]) - float(planned_contact[2])
    expected_opening_axis = axis_gate.get("expected_opening_axis_world")
    expected_approach_axis = axis_gate.get("expected_approach_axis_world")
    robot_opening_alignment = _dot3(ee_axes.get("local_x_world"), expected_opening_axis)
    robot_approach_alignment = _dot3(ee_axes.get("local_z_world"), expected_approach_axis)
    robot_axis_ok = (
        robot_opening_alignment is not None
        and robot_approach_alignment is not None
        and robot_opening_alignment >= min_robot_axis_alignment
        and robot_approach_alignment >= min_robot_axis_alignment
    )
    robot_ee_to_grasp_ok = ee_to_grasp is not None and ee_to_grasp <= max_ee_to_grasp
    robot_ee_z_to_contact_ok = (
        ee_z_above_contact is not None
        and expected_tcp_above is not None
        and abs(ee_z_above_contact - float(expected_tcp_above)) <= max_tcp_above_deviation
    )
    robot_state_gate = {
        "available": bool(arm_state and ee_xyz and ee_quat),
        "passed": bool(robot_ee_to_grasp_ok and robot_ee_z_to_contact_ok and robot_axis_ok),
        "side": side,
        "ee_pos": _round_list(ee_xyz, limit=3) if ee_xyz else None,
        "ee_rpy": _round_list((arm_state or {}).get("ee_rpy") or [], limit=3),
        "ee_quat_xyzw": _round_list(ee_quat or [], limit=4),
        "ee_axes_world": ee_axes,
        "target_grasp_xyz": _round_list(grasp_xyz, limit=3) if grasp_xyz else None,
        "contact_center_used_m": _round_list(contact_for_gate, limit=3) if contact_for_gate else None,
        "contact_reference_mode": object_reference_mode,
        "ee_to_grasp_m": round(ee_to_grasp, 5) if ee_to_grasp is not None else None,
        "ee_z_above_contact_m": round(ee_z_above_contact, 5) if ee_z_above_contact is not None else None,
        "expected_tcp_above_rim_m": round(float(expected_tcp_above), 5) if expected_tcp_above is not None else None,
        "opening_axis_alignment": round(robot_opening_alignment, 5) if robot_opening_alignment is not None else None,
        "approach_axis_alignment": round(robot_approach_alignment, 5) if robot_approach_alignment is not None else None,
        "checks": {
            "ee_to_grasp_ok": robot_ee_to_grasp_ok,
            "ee_z_to_contact_ok": robot_ee_z_to_contact_ok,
            "axis_ok": robot_axis_ok,
        },
        "limits": {
            "max_ee_to_grasp_m": max_ee_to_grasp,
            "max_tcp_above_deviation_m": max_tcp_above_deviation,
            "min_robot_axis_alignment": min_robot_axis_alignment,
        },
    }
    gripper_detection_ok = gripper_xy_to_contact is not None and gripper_xy_to_contact <= max_gripper_xy_to_contact
    gripper_z_ok = (
        gripper_z_delta_to_contact is not None
        and -max_gripper_below_contact <= gripper_z_delta_to_contact <= max_gripper_above_contact
    )
    require_sam_gripper = _truthy_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_REQUIRE_SAM_GRIPPER", False)
    visible_object_mismatch = bool(obj and object_xyz and not object_detection_coherent)
    allow_visible_object_mismatch = _truthy_env("OPENFORGE_BOWL_RACK_PRE_CLOSE_ALLOW_VISIBLE_OBJECT_MISMATCH", False)
    object_reference_ok = bool(
        object_detection_coherent
        or (
            object_reference_mode != "current_detection"
            and (not visible_object_mismatch or allow_visible_object_mismatch)
        )
    )

    checks = {
        "object_detected": obj is not None and object_xyz is not None,
        "gripper_detected": gripper is not None and gripper_xyz is not None,
        "axis_gate_passed": bool(axis_gate) and axis_gate.get("passed") is True,
        "object_xy_drift_ok": object_drift_ok,
        "object_depth_ok": not object_depth_jump if object_z_delta is not None else False,
        "object_xy_to_contact_ok": object_contact_xy_ok,
        "object_z_to_contact_ok": object_contact_z_ok,
        "contact_above_object_z_ok": contact_above_object_z_ok,
        "object_reference_ok": object_reference_ok,
        "robot_state_gate_ok": bool(robot_state_gate["passed"]),
        "gripper_xy_to_contact_ok": gripper_detection_ok,
        "gripper_z_to_contact_ok": gripper_z_ok,
    }
    required_checks = [
        checks["object_detected"],
        checks["axis_gate_passed"],
        checks["object_reference_ok"],
        checks["robot_state_gate_ok"],
        checks["contact_above_object_z_ok"],
    ]
    if require_sam_gripper:
        required_checks.extend([checks["gripper_detected"], checks["gripper_xy_to_contact_ok"], checks["gripper_z_to_contact_ok"]])
    packet = {
        "schema": "openforge.bowl_rack.pre_close_visual_gate.v1",
        "passed": all(required_checks),
        "checks": checks,
        "required_checks": {
            "object_detected": checks["object_detected"],
            "axis_gate_passed": checks["axis_gate_passed"],
            "object_reference_ok": checks["object_reference_ok"],
            "robot_state_gate_ok": checks["robot_state_gate_ok"],
            "contact_above_object_z_ok": checks["contact_above_object_z_ok"],
            "sam_gripper_required": require_sam_gripper,
        },
        "selected_strategy_label": selected_pickup.get("label"),
        "selected_strategy_type": selected_pickup.get("strategy_type"),
        "side": side,
        "object_detection": obj,
        "gripper_detection": gripper,
        "source_object_xyz": source_xyz,
        "current_object_xyz": object_xyz,
        "planned_contact_center_m": _round_list(planned_contact, limit=3) if planned_contact else None,
        "adjusted_contact_center_m": _round_list(adjusted_contact, limit=3) if adjusted_contact else None,
        "contact_center_used_m": _round_list(contact_for_gate, limit=3) if contact_for_gate else None,
        "object_reference_mode": object_reference_mode,
        "object_detection_coherent": object_detection_coherent,
        "object_center_coherent": object_center_coherent,
        "object_contact_coherent": object_contact_coherent,
        "visible_object_mismatch": visible_object_mismatch,
        "allow_visible_object_mismatch": allow_visible_object_mismatch,
        "current_gripper_xyz": gripper_xyz,
        "object_xy_drift_m": round(object_xy_drift, 5) if object_xy_drift is not None else None,
        "object_z_delta_m": round(object_z_delta, 5) if object_z_delta is not None else None,
        "object_xy_to_contact_m": round(object_xy_to_contact, 5) if object_xy_to_contact is not None else None,
        "object_z_delta_to_contact_m": round(object_z_delta_to_contact, 5) if object_z_delta_to_contact is not None else None,
        "contact_above_object_z_m": round(contact_above_object_z, 5) if contact_above_object_z is not None else None,
        "gripper_xy_to_contact_m": round(gripper_xy_to_contact, 5) if gripper_xy_to_contact is not None else None,
        "gripper_z_delta_to_contact_m": round(gripper_z_delta_to_contact, 5) if gripper_z_delta_to_contact is not None else None,
        "limits": {
            "max_object_xy_drift_m": max_object_xy_drift,
            "max_object_z_jump_m": max_object_z_jump,
            "max_object_xy_to_contact_m": max_object_xy_to_contact,
            "max_object_z_to_contact_m": max_object_z_to_contact,
            "max_contact_above_object_z_m": max_contact_above_object_z,
            "max_gripper_xy_to_contact_m": max_gripper_xy_to_contact,
            "max_gripper_above_contact_m": max_gripper_above_contact,
            "max_gripper_below_contact_m": max_gripper_below_contact,
        },
        "axis_gate": axis_gate,
        "robot_state_gate": robot_state_gate,
        "axis_gate_required": True,
    }
    packet["path"] = write_json(run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_pre_close_visual_gate.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## pre-close visual geometry gate",
            f"- passed: {packet['passed']}",
            f"- checks: {checks}",
            f"- required_checks: {packet['required_checks']}",
            f"- object_reference_mode: {packet['object_reference_mode']}",
            f"- object_xy_drift_m: {packet['object_xy_drift_m']}",
            f"- object_z_delta_m: {packet['object_z_delta_m']}",
            f"- object_xy_to_contact_m: {packet['object_xy_to_contact_m']}",
            f"- object_z_delta_to_contact_m: {packet['object_z_delta_to_contact_m']}",
            f"- contact_above_object_z_m: {packet['contact_above_object_z_m']}",
            f"- robot_state_gate: {robot_state_gate}",
            f"- gripper_xy_to_contact_m: {packet['gripper_xy_to_contact_m']}",
            f"- gripper_z_delta_to_contact_m: {packet['gripper_z_delta_to_contact_m']}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _verify_place(
    *,
    selected_pickup: dict[str, Any],
    selected_place: dict[str, Any],
    post_observation: dict[str, Any],
) -> dict[str, Any]:
    source = selected_pickup.get("source_detection") or {}
    post = _select_object_detection(post_observation)
    rack = selected_place.get("rack_detection") or {}
    release_target = selected_place.get("place_target_object_ref_m")
    motion = compare_detection_motion(source, post)
    if post is None or not release_target:
        return {
            "success": False,
            "confidence": 0.2,
            "reason": f"missing post-place {TASK_OBJECT_NAME} detection or release target",
            "post_detection": post,
            "object_motion": motion,
        }
    post_xyz = _xyz(post)
    rack_xyz = _xyz(rack)
    if post_xyz is None:
        return {
            "success": False,
            "confidence": 0.2,
            "reason": "post-place detection has no 3D position",
            "post_detection": post,
            "object_motion": motion,
        }
    dx = post_xyz[0] - float(release_target[0])
    dy = post_xyz[1] - float(release_target[1])
    xy_to_target = (dx * dx + dy * dy) ** 0.5
    z_floor = (rack_xyz[2] - _env_float("OPENFORGE_BOWL_RACK_VERIFY_Z_BELOW_RACK_TOL_M", 0.04)) if rack_xyz else -999.0
    xy_ok = xy_to_target <= _env_float("OPENFORGE_BOWL_RACK_VERIFY_MAX_XY_TO_TARGET_M", 0.18)
    z_ok = post_xyz[2] >= z_floor
    success = bool(xy_ok and z_ok)
    return {
        "success": success,
        "confidence": 0.75 if success else 0.3,
        "reason": f"{TASK_OBJECT_NAME} detected near rack placement target" if success else "post detection not close enough to rack target",
        "post_detection": post,
        "object_motion": motion,
        "xy_to_target_m": round(xy_to_target, 5),
        "z_floor_m": round(z_floor, 5),
        "checks": {"xy_ok": xy_ok, "z_ok": z_ok},
    }


def _execute_pickup_lift_only(run_dir: Path, selected_pickup: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    side = str(selected_pickup["arm"])
    close_plan = selected_pickup.get("staged_close_plan") or {}
    attempt: dict[str, Any] = {
        "schema": "openforge.bowl_rack.pickup_lift_only_physical_attempt.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "steps": [],
        "gripper_after_close": None,
        "why_stopped": None,
    }
    try:
        gripper_limit_kwargs = _gripper_limit_kwargs()
        pregrasp_open = _preshape_gripper(selected_pickup)
        hold = _hold_gripper(selected_pickup)
        open_step = _set_gripper_checked(
            side=side,
            target=pregrasp_open,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="checked_preshape_open",
            confirm_min=_env_float("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(open_step)
        if not open_step["ok"]:
            attempt["why_stopped"] = "gripper did not confirm open before arm motion"
            return attempt

        pickup_specs = _pickup_lift_specs(selected_pickup)
        before_close_specs = [spec for spec in pickup_specs if spec[0] != "bowl_small_lift"]
        lift_specs = [spec for spec in pickup_specs if spec[0] == "bowl_small_lift"]
        if not lift_specs:
            raise RuntimeError("pickup/lift sequence missing bowl_small_lift stage")

        def _open_and_retreat(stage_prefix: str) -> None:
            release = _set_gripper_checked(
                side=side,
                target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                stage=f"{stage_prefix}_open",
                confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
                **gripper_limit_kwargs,
            )
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
            attempt["steps"].append(release)
            retreat_pose = _pickup_retreat_pose(selected_pickup)
            retreat = _execute_pose(
                freespace_move=freespace_move,
                side=side,
                pose=retreat_pose,
                label=f"{stage_prefix}_retreat",
                gripper=1.0,
            )
            attempt["movement_capable_calls"].append(f"freespace_move:{stage_prefix}_retreat")
            attempt["steps"].append(retreat)

        for label, pose, gripper in before_close_specs:
            try:
                step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append(step)
            except Exception as exc:
                attempt["movement_capable_calls"].append(f"freespace_move:{label}")
                attempt["steps"].append({"stage": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                attempt["why_stopped"] = f"{label} motion failed before close; stop before close/lift"
                try:
                    _open_and_retreat("before_close_motion_failed")
                except Exception as retreat_exc:
                    attempt["steps"].append(
                        {
                            "stage": "before_close_motion_failed_retreat_exception",
                            "ok": False,
                            "error": f"{type(retreat_exc).__name__}: {retreat_exc}",
                        }
                    )
                return attempt

        attempt["pre_close_observation"] = _capture_object_observation(run_dir, "pre_close_grasp_pose_observe")
        pre_close_gate = _pre_close_visual_gate(run_dir, selected_pickup, attempt["pre_close_observation"])
        attempt["pre_close_visual_gate"] = pre_close_gate
        if _truthy_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", True) and not bool(pre_close_gate.get("passed")):
            attempt["why_stopped"] = "pre-close visual geometry gate failed; stop before close/lift"
            _open_and_retreat("pre_close_gate_failed")
            return attempt

        close = staged_close_with_contact(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=hold,
            steps=_csv_float("OPENFORGE_BOWL_RACK_CLOSE_STEPS", list(close_plan.get("steps") or [0.68, 0.56, 0.48, 0.40])),
            **gripper_limit_kwargs,
            min_contact_delta=_env_float(
                "OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS",
                float(close_plan.get("expected_contact_delta_min", 0.010)),
            ),
            target_tolerance=_env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_BOWL_RACK_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_BOWL_RACK_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage=f"{TASK_ARTIFACT_PREFIX}_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["gripper_after_close"] = close
        attempt["post_close_observation"] = _capture_object_observation(run_dir, "post_close_before_lift_observe")
        if _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True) and not bool(close.get("contact_detected")):
            attempt["why_stopped"] = "staged close did not produce contact evidence; stop before lift"
            _open_and_retreat("no_contact_after_close")
            return attempt
        if not bool(close.get("plausible_for_lift")):
            attempt["why_stopped"] = "staged close did not produce plausible gripper-state evidence; stop before lift"
            _open_and_retreat("implausible_hold_after_close")
            return attempt

        label, pose, gripper = lift_specs[0]
        lift_step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
        attempt["movement_capable_calls"].append(f"freespace_move:{label}")
        attempt["steps"].append(lift_step)

        post_lift = _capture_object_observation(run_dir, "post_lift_pickup_only_observe")
        post_lift_det = _select_object_detection(post_lift)
        attempt["post_lift_observation"] = post_lift
        attempt["lift_verification"] = verify_lift(
            selected_pickup.get("source_detection"),
            post_lift_det,
            min_z_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MIN_Z_DELTA_M", 0.015),
            max_xy_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MAX_XY_DELTA_M", 0.24),
        )
        attempt["verification_success"] = bool(attempt["lift_verification"].get("success"))
        attempt["success"] = bool(attempt["verification_success"])

        if _truthy_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY_RETURN_TO_TABLE", True):
            return_steps: list[dict[str, Any]] = []
            for return_label, return_pose, return_gripper in [
                ("return_to_grasp_pose", selected_pickup["grasp_pose"], hold),
                (
                    "retreat_after_table_release",
                    _pickup_retreat_pose(selected_pickup),
                    1.0,
                ),
            ]:
                if return_label == "retreat_after_table_release":
                    release = _set_gripper_checked(
                        side=side,
                        target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
                        set_gripper=set_gripper,
                        get_robot_state=get_robot_state,
                        stage="open_release_back_on_table",
                        confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
                        **gripper_limit_kwargs,
                    )
                    attempt["movement_capable_calls"].append(f"set_gripper:{side}")
                    return_steps.append(release)
                    attempt["steps"].append(release)
                step = _execute_pose(
                    freespace_move=freespace_move,
                    side=side,
                    pose=return_pose,
                    label=return_label,
                    gripper=return_gripper,
                )
                attempt["movement_capable_calls"].append(f"freespace_move:{return_label}")
                return_steps.append(step)
                attempt["steps"].append(step)
            attempt["return_to_table"] = {"enabled": True, "steps": return_steps}
            attempt["post_return_observation"] = _capture_object_observation(run_dir, "post_return_table_observe")

        attempt["why_stopped"] = (
            "pickup/lift verified; returned bowl to table"
            if attempt["success"] and attempt.get("return_to_table", {}).get("enabled")
            else "lift command completed; lift verification failed or inconclusive"
            if not attempt["success"]
            else "pickup/lift verified"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _execute_physical(run_dir: Path, selected_pickup: dict[str, Any], selected_place: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    side = str(selected_pickup["arm"])
    close_plan = selected_pickup.get("staged_close_plan") or {}
    attempt: dict[str, Any] = {
        "schema": "openforge.bowl_rack.physical_attempt.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "selected_place": selected_place,
        "steps": [],
        "gripper_after_close": None,
        "why_stopped": None,
    }
    try:
        gripper_limit_kwargs = _gripper_limit_kwargs()
        pregrasp_open = _preshape_gripper(selected_pickup)
        hold = _hold_gripper(selected_pickup)
        open_step = _set_gripper_checked(
            side=side,
            target=pregrasp_open,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="checked_preshape_open",
            confirm_min=_env_float("OPENFORGE_BOWL_RACK_OPEN_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(open_step)
        if not open_step["ok"]:
            attempt["why_stopped"] = "gripper did not confirm open before arm motion"
            return attempt

        before_close_specs, after_close_specs = _split_physical_sequence(selected_pickup, selected_place)

        for label, pose, gripper in before_close_specs:
            step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        attempt["pre_close_observation"] = _capture_object_observation(run_dir, "pre_close_grasp_pose_observe")
        pre_close_gate = _pre_close_visual_gate(run_dir, selected_pickup, attempt["pre_close_observation"])
        attempt["pre_close_visual_gate"] = pre_close_gate
        if _truthy_env("OPENFORGE_BOWL_RACK_ENABLE_PRE_CLOSE_VISUAL_GATE", True) and not bool(
            pre_close_gate.get("passed")
        ):
            attempt["why_stopped"] = "pre-close visual geometry gate failed; stop before close/lift/place"
            return attempt
        close = staged_close_with_contact(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=hold,
            steps=_csv_float("OPENFORGE_BOWL_RACK_CLOSE_STEPS", list(close_plan.get("steps") or [0.68, 0.56, 0.48, 0.40])),
            **gripper_limit_kwargs,
            min_contact_delta=_env_float(
                "OPENFORGE_BOWL_RACK_CLOSE_CONTACT_MIN_DELTA_POS",
                float(close_plan.get("expected_contact_delta_min", 0.010)),
            ),
            target_tolerance=_env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_BOWL_RACK_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_BOWL_RACK_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage=f"{TASK_ARTIFACT_PREFIX}_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["gripper_after_close"] = close
        attempt["post_close_observation"] = _capture_object_observation(run_dir, "post_close_before_lift_observe")
        require_contact = _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True)
        if require_contact and not bool(close.get("contact_detected")):
            attempt["why_stopped"] = "staged close did not produce contact evidence; stop before lift/place"
            return attempt
        if not bool(close.get("plausible_for_lift")):
            attempt["why_stopped"] = "staged close did not produce plausible gripper-state evidence; stop before lift/place"
            return attempt

        for label, pose, gripper in after_close_specs[:1]:
            step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        post_lift = _capture_object_observation(run_dir, "post_lift_before_rack_transport_observe")
        post_lift_det = _select_object_detection(post_lift)
        attempt["post_lift_observation"] = post_lift
        attempt["lift_verification"] = verify_lift(
            selected_pickup.get("source_detection"),
            post_lift_det,
            min_z_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MIN_Z_DELTA_M", 0.035),
            max_xy_delta_m=_env_float("OPENFORGE_BOWL_RACK_VERIFY_LIFT_MAX_XY_DELTA_M", 0.24),
        )
        if _truthy_env("OPENFORGE_BOWL_RACK_REQUIRE_LIFT_PERCEPTION_BEFORE_PLACE", False) and not bool(
            attempt["lift_verification"].get("success")
        ):
            attempt["why_stopped"] = "lift perception verification failed; stop before rack transport by configured policy"
            return attempt

        for label, pose, gripper in after_close_specs[1:3]:
            step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        release = _set_gripper_checked(
            side=side,
            target=_env_float("OPENFORGE_BOWL_RACK_RELEASE_GRIPPER_POS", 1.0),
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="open_release_on_rack",
            confirm_min=_env_float("OPENFORGE_BOWL_RACK_RELEASE_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(release)
        if not release["ok"]:
            attempt["why_stopped"] = "release gripper did not confirm open"
            return attempt

        for label, pose, gripper in after_close_specs[3:]:
            step = _execute_pose(freespace_move=freespace_move, side=side, pose=pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        post_place = _capture_object_observation(run_dir, "post_place_on_rack_observe")
        attempt["post_observation"] = post_place
        verification = _verify_place(
            selected_pickup=selected_pickup,
            selected_place=selected_place,
            post_observation=post_place,
        )
        attempt["place_verification"] = verification
        attempt["verification_success"] = bool(verification.get("success"))
        attempt["success"] = bool(attempt["verification_success"])
        attempt["why_stopped"] = (
            f"{TASK_OBJECT_NAME} placement verified on/near rack"
            if attempt["success"]
            else f"place sequence completed but verification failed/inconclusive: {verification.get('reason')}"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.bowl_rack.result.v1",
        "success": False,
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "observation_policy": "fresh live detections only; manual XYZ env vars are rejected",
        "selected_strategy": None,
        "selected_strategy_type": None,
        "planner_result": None,
        "pickup_validation": None,
        "pickup_lift_preview": None,
        "pickup_lift_attempt": None,
        "rack_observation": None,
        "place_candidates_path": None,
        "sequence_preview": None,
        "physical_ticket_status": None,
        "physical_attempt": None,
        "source_snapshot": None,
        "recording_evidence": None,
        "video_paths": [],
        "why_stopped": None,
        "next_recommendation": None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        result["source_snapshot"] = _source_snapshot(run_dir)
        manual_vars = _manual_pose_env_vars_present()
        if manual_vars:
            result["why_stopped"] = "manual XYZ/debug env vars are not allowed for this live bowl/rack task"
            result["manual_pose_env_vars_present"] = manual_vars
            result["next_recommendation"] = "Clear manual pose env vars and rerun from fresh live perception."
            return result
        strategy_env_problems = _forbidden_pickup_strategy_env()
        if strategy_env_problems:
            result["why_stopped"] = f"refusing non-top-down {TASK_OBJECT_NAME} pickup strategy configuration"
            result["pickup_strategy_env_problems"] = strategy_env_problems
            result["next_recommendation"] = (
                "Clear side-rim strategy env vars and rerun a recorded no-motion top-down rim-straddle validation."
            )
            return result

        planner_result = _run_fresh_pickup_planner()
        result["planner_result"] = planner_result
        write_json(run_dir / "plans" / f"fresh_{TASK_ARTIFACT_PREFIX}_pickup_planner_result.json", planner_result)
        pickup_validation = _validate_pickup_strategy(planner_result)
        result["pickup_validation"] = pickup_validation
        append_stage_summary(
            run_dir,
            [
                f"## {TASK_OBJECT_NAME} pickup validation",
                f"- passed: {pickup_validation['passed']}",
                f"- selected_label: {pickup_validation.get('selected_label')}",
                f"- arm: {pickup_validation.get('arm')}",
                f"- source_detection: {pickup_validation.get('source_detection')}",
            ],
        )
        if not pickup_validation.get("passed"):
            result["why_stopped"] = f"fresh pickup planner did not produce a valid {TASK_OBJECT_NAME} rim-straddle strategy"
            result["next_recommendation"] = "Inspect pickup planner artifacts; no physical motion executed."
            return result

        selected_pickup = pickup_validation["selected_strategy"]
        result["selected_strategy"] = selected_pickup.get("label")
        result["selected_strategy_type"] = selected_pickup.get("strategy_type")
        if _truthy_env("OPENFORGE_BOWL_RACK_PICKUP_LIFT_ONLY", False):
            pickup_lift_preview = _preview_pickup_lift_sequence(run_dir, selected_pickup)
            result["pickup_lift_preview"] = pickup_lift_preview
            result["plan_success"] = bool(pickup_lift_preview.get("success"))
            if not pickup_lift_preview.get("success"):
                result["why_stopped"] = "pickup/lift no-motion sequence preview failed"
                result["next_recommendation"] = (
                    f"Inspect {TASK_ARTIFACT_PREFIX}_pickup_lift_sequence_preview.json before physical motion."
                )
                return result
            if not result["physical_gate"]:
                result["success"] = True
                result["why_stopped"] = "recorded no-motion pickup/lift validation succeeded; physical gate was not set"
                result["next_recommendation"] = (
                    f"For physical pickup/lift validation, rerun under the active {TASK_TICKET_SCOPE} ticket "
                    "with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and complete precondition env vars."
                )
                return result

            recording = _recording_evidence(run_dir)
            result["recording_evidence"] = recording
            if not recording.get("passed"):
                result["why_stopped"] = "recording evidence is missing; refusing physical motion"
                result["next_recommendation"] = "Fix recording/debug UI evidence and rerun no-motion validation."
                return result

            ticket_status = _ticket_status(run_dir)
            result["physical_ticket_status"] = ticket_status
            append_stage_summary(
                run_dir,
                [
                    "## physical ticket gate",
                    f"- passed: {ticket_status['passed']}",
                    f"- missing_or_invalid: {ticket_status['missing_or_invalid']}",
                ],
            )
            if not ticket_status.get("passed"):
                result["why_stopped"] = "physical ticket/precondition env acknowledgements are incomplete; refusing physical motion"
                result["next_recommendation"] = "Continue no-motion or rerun physical with complete ticket/precondition env vars."
                return result

            attempt = _execute_pickup_lift_only(run_dir, selected_pickup)
            result["pickup_lift_attempt"] = attempt
            result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
            result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
            result["success"] = bool(attempt.get("success"))
            result["why_stopped"] = attempt.get("why_stopped")
            result["next_recommendation"] = (
                "Review pickup/lift result, overlays, and videos before any rack-placement retry."
            )
            return result

        rack_observation = _observe_rack(run_dir)
        result["rack_observation"] = rack_observation
        rack_detection = rack_observation.get("selected_rack_detection")
        if not rack_detection:
            result["why_stopped"] = "fresh rack detection failed"
            result["next_recommendation"] = "Make the rack visible to top/side cameras and rerun recorded no-motion validation."
            return result

        place_candidates = _make_place_candidates(selected_pickup, rack_detection)
        result["place_candidates_path"] = write_json(
            run_dir / "plans" / f"{TASK_ARTIFACT_PREFIX}_rack_place_candidates.json",
            {
                "schema": "openforge.bowl_rack.place_candidates.v1",
                "selected_pickup": selected_pickup,
                "rack_detection": rack_detection,
                "candidate_count": len(place_candidates),
                "candidates": place_candidates,
            },
        )
        sequence = _preview_place_sequence(run_dir, selected_pickup, place_candidates)
        result["sequence_preview"] = sequence
        result["plan_success"] = bool(sequence.get("success"))
        if not sequence.get("success"):
            result["why_stopped"] = "pickup-to-rack no-motion sequence preview failed"
            result["next_recommendation"] = "Tune rack place offsets/z clearance or choose a different rack target after inspecting previews."
            return result

        if not result["physical_gate"]:
            result["success"] = True
            result["why_stopped"] = "recorded no-motion validation succeeded; physical gate was not set"
            result["next_recommendation"] = (
                f"For physical mode, rerun once under the active {TASK_TICKET_SCOPE} ticket "
                "with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and complete precondition env vars."
            )
            return result

        recording = _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        if not recording.get("passed"):
            result["why_stopped"] = "recording evidence is missing; refusing physical motion"
            result["next_recommendation"] = "Fix recording/debug UI evidence and rerun no-motion validation."
            return result

        ticket_status = _ticket_status(run_dir)
        result["physical_ticket_status"] = ticket_status
        append_stage_summary(
            run_dir,
            [
                "## physical ticket gate",
                f"- passed: {ticket_status['passed']}",
                f"- missing_or_invalid: {ticket_status['missing_or_invalid']}",
            ],
        )
        if not ticket_status.get("passed"):
            result["why_stopped"] = "physical ticket/precondition env acknowledgements are incomplete; refusing physical motion"
            result["next_recommendation"] = "Continue no-motion or rerun physical with complete ticket/precondition env vars."
            return result

        attempt = _execute_physical(run_dir, selected_pickup, sequence["selected_place"])
        result["physical_attempt"] = attempt
        result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
        result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
        result["success"] = bool(attempt.get("success"))
        result["why_stopped"] = attempt.get("why_stopped")
        result["next_recommendation"] = (
            "Do not retry under this ticket without reviewing result.json, stage_summary.md, overlays, and videos."
        )
        return result
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and run log before retrying."
        return result
    finally:
        recording = result.get("recording_evidence") or _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## bowl rack result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- recording_evidence: {result['recording_evidence']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
