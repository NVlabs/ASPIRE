# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded orange-bowl-into-white-bowl stacking script for real YAM.

This script is independent from the rack scripts. It reuses the top-down orange
bowl pickup planner, observes the white bowl as the placement target, previews a
bounded pickup/lift/center-place/release sequence, and executes at most one
physical attempt only when a stack-specific physical ticket is present.
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


TASK_NAME = "stack_orange_bowl_into_white_bowl_one_shot_loop"
PLANNER_PATH = Path(os.environ.get("OPENFORGE_BOWL_STACK_PLANNER_PATH", "cap/saved_scripts/pick_plate_no_motion_plan.py"))
REQUIRED_PICKUP_STRATEGY_MODE = "top_down_bowl_rim"
REQUIRED_PICKUP_STRATEGY_TYPE = "top_down_bowl_rim_straddle"
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
            values.append([float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) == 3 else 0.0])
        except ValueError:
            pass
    return values or [list(item) for item in default]


def _gripper_limit_kwargs() -> dict[str, float | None]:
    kwargs: dict[str, float | None] = {}
    vel_is_set, vel_limit = _env_optional_float_setting("OPENFORGE_BOWL_STACK_GRIPPER_VEL_LIMIT")
    torque_is_set, torque_limit = _env_optional_float_setting("OPENFORGE_BOWL_STACK_GRIPPER_TORQUE_LIMIT")
    if vel_is_set:
        kwargs["vel_limit"] = vel_limit
    if torque_is_set:
        kwargs["torque_limit"] = torque_limit
    return kwargs


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required injected tool missing: {name}")
    return value


def _xyz(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        raw = value.get("position_3d") or value.get("position")
    else:
        raw = value
    if not raw or len(raw) < 3:
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2])]
    except Exception:
        return None


def _round_list(values: Any, *, limit: int = 5) -> list[float] | None:
    if values is None:
        return None
    try:
        return [round(float(v), limit) for v in values]
    except Exception:
        return None


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _artifact_safe_label(value: str) -> str:
    text = str(value or "artifact")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:80] or "artifact"


def _debug_observation_cameras() -> list[str]:
    return _csv("OPENFORGE_DEBUG_OBS_CAMERAS", ["top", "left", "right", "bottom"])


def _image_only_debug_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_BOWL_STACK_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_BOWL_STACK_RECORDING_WAIT_S", 5.0))
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
        "OPENFORGE_BOWL_STACK_ORANGE_XYZ",
        "OPENFORGE_BOWL_STACK_WHITE_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


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
    script_path = Path("cap/saved_scripts/stack_orange_bowl_into_white_bowl_one_shot_loop.py").resolve()
    snapshot = {
        "script_path": str(script_path),
        "planner_path": str((Path.cwd() / PLANNER_PATH).resolve()),
        "task_name": TASK_NAME,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_BOWL_STACK_")
            or name.startswith("OPENFORGE_BOWL_TOP_DOWN_")
            or name
            in {
                "OPENFORGE_ALLOW_PHYSICAL_MOTION",
                "OPENFORGE_DEBUG_OBS_TIMEOUT_S",
                "OPENFORGE_DEBUG_OBS_CAMERAS",
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


def _run_fresh_orange_pickup_planner() -> dict[str, Any]:
    previous_gate = os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
    overrides = {
        "OPENFORGE_PLATE_TASK_NAME": f"{TASK_NAME}_orange_pickup_plan",
        "OPENFORGE_PLATE_OBJECT_NAME": "orange bowl",
        "OPENFORGE_PLATE_PROMPTS": os.environ.get("OPENFORGE_BOWL_STACK_PICKUP_PROMPTS", "orange bowl,bowl,white bowl"),
        "OPENFORGE_PLATE_COLLECT_ALL_PROMPTS": "1",
        "OPENFORGE_PLATE_TARGET_Y_SIGN": os.environ.get("OPENFORGE_BOWL_STACK_PICKUP_TARGET_Y_SIGN", "negative"),
        "OPENFORGE_PLATE_STRATEGY_MODE": REQUIRED_PICKUP_STRATEGY_MODE,
        "OPENFORGE_PLATE_ARM": os.environ.get("OPENFORGE_BOWL_STACK_PICKUP_ARM", "right"),
        "OPENFORGE_PLATE_MAX_CANDIDATES": os.environ.get("OPENFORGE_BOWL_STACK_PICKUP_MAX_CANDIDATES", "3"),
        "OPENFORGE_PLATE_PLANNER_BACKEND": os.environ.get("OPENFORGE_BOWL_STACK_PLANNER_BACKEND", "rrtconnect"),
        "OPENFORGE_PLATE_SOLVER_SPEED": os.environ.get("OPENFORGE_BOWL_STACK_SOLVER_SPEED", "fast"),
        "OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M": os.environ.get("OPENFORGE_BOWL_STACK_IK_ERROR_THRESHOLD_M", "0.025"),
        "OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG": os.environ.get("OPENFORGE_BOWL_STACK_IK_ROT_THRESHOLD_DEG", "16"),
        "OPENFORGE_PLATE_IK_XYZ_WEIGHT": os.environ.get("OPENFORGE_BOWL_STACK_IK_XYZ_WEIGHT", "1.0"),
        "OPENFORGE_PLATE_IK_RPY_WEIGHT": os.environ.get("OPENFORGE_BOWL_STACK_IK_RPY_WEIGHT", "0.3"),
        "OPENFORGE_PLATE_PREVIEW_MAX_POS_ERROR_M": os.environ.get("OPENFORGE_BOWL_STACK_PREVIEW_MAX_POS_ERROR_M", "0.025"),
        "OPENFORGE_PLATE_PREVIEW_MAX_ROT_ERROR_DEG": os.environ.get("OPENFORGE_BOWL_STACK_PREVIEW_MAX_ROT_ERROR_DEG", "10"),
    }
    previous = _with_env_overrides(overrides)
    try:
        planner_path = Path.cwd() / PLANNER_PATH
        spec = importlib.util.spec_from_file_location("openforge_stack_orange_pickup_plan", planner_path)
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
        namespace = runpy.run_path(spec.origin, init_globals=tool_globals, run_name="__openforge_stack_orange_plan__")
        return dict(namespace.get("TASK_RESULT") or {})
    finally:
        _restore_env(previous)
        if previous_gate is not None:
            os.environ["OPENFORGE_ALLOW_PHYSICAL_MOTION"] = previous_gate


def _validate_pickup_strategy(planner_result: dict[str, Any]) -> dict[str, Any]:
    selected = planner_result.get("selected_strategy") or {}
    source = selected.get("source_detection") or {}
    source_name = str(source.get("source") or source.get("detection_source") or "")
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
    strategy_ok = selected.get("strategy_type") == REQUIRED_PICKUP_STRATEGY_TYPE
    residual_ok = bool((selected.get("preview_residual_gate") or {}).get("passed", True))
    axis_ok = bool(selected.get("axis_gate")) and (selected.get("axis_gate") or {}).get("passed") is True
    source_camera = str(source.get("source_camera") or source.get("camera") or "")
    camera_ok = source_camera == "top" if _truthy_env("OPENFORGE_BOWL_STACK_REQUIRE_TOP_CAMERA_PICKUP", True) else True
    manual_source = "manual" in source_name.lower()
    passed = bool(
        planner_result.get("success")
        and selected
        and not missing
        and strategy_ok
        and residual_ok
        and axis_ok
        and camera_ok
        and not manual_source
    )
    return {
        "passed": passed,
        "missing": missing,
        "strategy_ok": strategy_ok,
        "strategy_type": selected.get("strategy_type"),
        "residual_gate_passed": residual_ok,
        "axis_gate_passed": axis_ok,
        "camera_ok": camera_ok,
        "source_camera": source_camera,
        "manual_source": manual_source,
        "selected_label": selected.get("label"),
        "arm": selected.get("arm"),
        "source_detection": source,
        "selected_strategy": selected if passed else None,
    }


def _detection_touches_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_BOWL_STACK_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_BOWL_STACK_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_BOWL_STACK_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _capture_scene(run_dir: Path, stage: str, prompts: list[str]) -> dict[str, Any]:
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


def _select_detection(
    packet: dict[str, Any],
    *,
    preferred_prompts: tuple[str, ...],
    min_score_env: str,
    default_min_score: float,
    require_top: bool = True,
) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    prompt_rank = {prompt: index for index, prompt in enumerate(preferred_prompts)}
    filtered: list[dict[str, Any]] = []
    for det in detections:
        prompt = str(det.get("prompt") or "").lower()
        camera = str(det.get("source_camera") or det.get("camera") or "")
        if prompt not in prompt_rank or camera not in {"top", "left", "right"}:
            continue
        if require_top and camera != "top":
            continue
        if float(det.get("score") or 0.0) < _env_float(min_score_env, default_min_score):
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
            1 if det.get("touches_image_edge") else 0,
            prompt_rank.get(prompt, 9),
            -float(det.get("score") or 0.0),
        )

    selected = sorted(filtered, key=_key)[0]
    selected["detection_source"] = "live_top_camera" if selected.get("source_camera") == "top" else "live_calibrated_camera"
    return selected


def _observe_stack_targets(run_dir: Path) -> dict[str, Any]:
    prompts = _csv("OPENFORGE_BOWL_STACK_OBSERVE_PROMPTS", ["orange bowl", "white bowl", "bowl", "blue gripper fingers"])
    packet = _capture_scene(run_dir, "observe_stack_targets", prompts)
    orange = _select_detection(
        packet,
        preferred_prompts=("orange bowl",),
        min_score_env="OPENFORGE_BOWL_STACK_ORANGE_MIN_SCORE",
        default_min_score=0.25,
        require_top=True,
    )
    white = _select_detection(
        packet,
        preferred_prompts=("white bowl",),
        min_score_env="OPENFORGE_BOWL_STACK_WHITE_MIN_SCORE",
        default_min_score=0.20,
        require_top=True,
    )
    packet["selected_orange_detection"] = orange
    packet["selected_white_detection"] = white
    packet["path"] = write_json(run_dir / "observations" / "observe_stack_targets_selected.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## stack target observation",
            f"- selected_orange_detection: {orange}",
            f"- selected_white_detection: {white}",
            f"- packet: {packet.get('packet_path')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _bbox_radius_estimate(detection: dict[str, Any] | None, *, default_radius_m: float) -> dict[str, Any]:
    if not detection:
        return {"radius_m": default_radius_m, "source": "default_no_detection"}
    box = detection.get("box_2d") or []
    if len(box) >= 4 and _truthy_env("OPENFORGE_BOWL_STACK_USE_TOP_BOX_RADIUS", True):
        long_axis = max(abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1])))
        reference_axis = _env_float("OPENFORGE_BOWL_STACK_REFERENCE_BOX_LONG_AXIS_PX", 56.0)
        reference_radius = _env_float("OPENFORGE_BOWL_STACK_REFERENCE_RADIUS_M", 0.095)
        if reference_axis > 0:
            raw = reference_radius * long_axis / reference_axis
            clipped = max(
                _env_float("OPENFORGE_BOWL_STACK_MIN_RADIUS_M", 0.045),
                min(_env_float("OPENFORGE_BOWL_STACK_MAX_RADIUS_M", 0.18), raw),
            )
            return {
                "radius_m": clipped,
                "raw_radius_m": raw,
                "source": "top_box_scaled_from_reference_radius",
                "box_long_axis_px": long_axis,
                "reference_box_long_axis_px": reference_axis,
                "reference_radius_m": reference_radius,
            }
    return {"radius_m": default_radius_m, "source": "default"}


def _target_geometry_gate(selected_pickup: dict[str, Any], stack_observation: dict[str, Any]) -> dict[str, Any]:
    white = stack_observation.get("selected_white_detection")
    orange = stack_observation.get("selected_orange_detection") or selected_pickup.get("source_detection")
    orange_radius = _bbox_radius_estimate(orange, default_radius_m=_env_float("OPENFORGE_BOWL_STACK_ORANGE_RADIUS_M", 0.095))
    white_radius = _bbox_radius_estimate(white, default_radius_m=_env_float("OPENFORGE_BOWL_STACK_WHITE_RADIUS_M", 0.125))
    orange_xyz = _xyz(orange)
    white_xyz = _xyz(white)
    xy_distance = None
    if orange_xyz and white_xyz:
        xy_distance = math.hypot(orange_xyz[0] - white_xyz[0], orange_xyz[1] - white_xyz[1])
    radius_margin = float(white_radius["radius_m"]) - float(orange_radius["radius_m"])
    min_radius_margin = _env_float("OPENFORGE_BOWL_STACK_MIN_TARGET_RADIUS_MARGIN_M", 0.015)
    min_xy_separation = _env_float("OPENFORGE_BOWL_STACK_MIN_START_XY_SEPARATION_M", 0.06)
    passed = bool(white and white_xyz and radius_margin >= min_radius_margin and (xy_distance is None or xy_distance >= min_xy_separation))
    packet = {
        "passed": passed,
        "white_detection_present": bool(white and white_xyz),
        "orange_detection_present": bool(orange and orange_xyz),
        "orange_radius_estimate": orange_radius,
        "white_radius_estimate": white_radius,
        "radius_margin_m": round(radius_margin, 5),
        "xy_distance_m": round(xy_distance, 5) if xy_distance is not None else None,
        "limits": {
            "min_radius_margin_m": min_radius_margin,
            "min_start_xy_separation_m": min_xy_separation,
        },
        "reason": "white bowl appears larger and separately localized" if passed else "target bowl geometry gate failed",
    }
    return packet


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None, preview_only: bool = False) -> dict[str, Any]:
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_BOWL_STACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_BOWL_STACK_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_BOWL_STACK_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_BOWL_STACK_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_BOWL_STACK_IK_ROT_THRESHOLD_DEG", 16.0),
        "ik_xyz_weight": _env_float("OPENFORGE_BOWL_STACK_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_BOWL_STACK_IK_RPY_WEIGHT", 0.3),
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
    if not _truthy_env("OPENFORGE_BOWL_STACK_ALLOW_AXIS_GATED_ROTATION_RESIDUAL", True):
        return False
    label_text = str(label)
    return label_text.startswith("bowl_") or label_text.startswith("stack_")


def _stage_tolerance(label: str, *, execution: bool = False) -> dict[str, float]:
    prefix = "OPENFORGE_BOWL_STACK_EXEC" if execution else "OPENFORGE_BOWL_STACK_PREVIEW"
    default_pos = _env_float(f"{prefix}_MAX_POS_ERROR_M", 0.025)
    default_rot = _env_float(f"{prefix}_MAX_ROT_ERROR_DEG", 10.0)
    loose_pos = max(default_pos, 0.030)
    loose_rot = max(default_rot, 16.0)
    label_text = str(label)
    if "high" in label_text or "transport" in label_text or "retreat" in label_text:
        return {"max_pos_error_m": loose_pos, "max_rot_error_deg": loose_rot}
    return {"max_pos_error_m": default_pos, "max_rot_error_deg": default_rot}


def _summary_from_move(result: Any, *, stage_label: str, execution: bool) -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", execution))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    tolerance = _stage_tolerance(stage_label, execution=execution)
    status_ok = bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed == execution)
    axis_gated_rotation = _stage_allows_axis_gated_rotation_residual(stage_label)
    pos_ok = _error_within_limit(pos_error, tolerance["max_pos_error_m"])
    rot_ok = axis_gated_rotation or _error_within_limit(rot_error, tolerance["max_rot_error_deg"])
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": tolerance["max_pos_error_m"],
        "max_rot_error_deg": tolerance["max_rot_error_deg"],
        "tolerance_ok": bool(pos_ok and rot_ok),
        "position_tolerance_ok": pos_ok,
        "rotation_tolerance_ok": rot_ok,
        "axis_gated_rotation_residual": axis_gated_rotation,
        "rotation_residual_policy": (
            "ignored because selected orange pickup already passed top-down axis_gate and stack preserves that orientation"
            if axis_gated_rotation
            else "raw motion-tool rotation residual must be within limit"
        ),
        "success": bool(status_ok and pos_ok and rot_ok),
        "raw": json_safe(result),
    }


def _preview_pose(freespace_move: Callable[..., Any], side: str, pose: dict[str, Any], *, label: str, gripper: float | None) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        freespace_move,
        _env_float("OPENFORGE_BOWL_STACK_PREVIEW_TIMEOUT_S", 20.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pose, gripper=gripper, preview_only=True),
    )
    if not result.get("ok"):
        return {"stage_label": label, "success": False, "status": None, "reason": result.get("error"), "executed": False}
    return _summary_from_move(result.get("data"), stage_label=label, execution=False)


def _execute_pose(freespace_move: Callable[..., Any], side: str, pose: dict[str, Any], *, label: str, gripper: float | None) -> dict[str, Any]:
    preview_result = call_with_timeout(
        f"preview-before-execute:{label}",
        freespace_move,
        _env_float("OPENFORGE_BOWL_STACK_PREVIEW_TIMEOUT_S", 20.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pose, gripper=gripper, preview_only=True),
    )
    if not preview_result.get("ok"):
        raise RuntimeError(f"{label} preview failed before execute: {preview_result.get('error')}")
    preview_packet = _summary_from_move(preview_result.get("data"), stage_label=label, execution=False)
    cache_key = _field(preview_result.get("data"), "trajectory_cache_key", None)
    if not cache_key:
        raise RuntimeError(f"{label} preview did not return trajectory_cache_key")
    execute_result = call_with_timeout(
        f"execute:{label}",
        freespace_move,
        _env_float("OPENFORGE_BOWL_STACK_EXEC_TIMEOUT_S", 40.0),
        run_in_background=_tool("run_in_background"),
        trajectory_cache_key=cache_key,
    )
    if not execute_result.get("ok"):
        raise RuntimeError(f"{label} execute failed: {execute_result.get('error')}")
    execute_packet = _summary_from_move(execute_result.get("data"), stage_label=label, execution=True)
    if not execute_packet.get("success"):
        raise RuntimeError(f"{label} executed but residual/status gate failed: {execute_packet}")
    return {
        "stage": label,
        "side": side,
        "pose": pose,
        "gripper": gripper,
        "preview": preview_packet,
        "execute": execute_packet,
        "trajectory_cache_key": cache_key,
    }


def _preshape_gripper(selected: dict[str, Any]) -> float:
    override = os.environ.get("OPENFORGE_BOWL_STACK_APPROACH_GRIPPER_POS")
    if override not in {None, ""}:
        return float(override)
    return float(((selected.get("preshape_plan") or {}).get("expected_gripper_open_before_close_normalized", 0.68)))


def _hold_gripper(selected: dict[str, Any]) -> float:
    close_plan = selected.get("staged_close_plan") or {}
    return _env_float("OPENFORGE_BOWL_STACK_CLOSE_TARGET_POS", float(close_plan.get("target_normalized_position", 0.04)))


def _pickup_lift_specs(selected_pickup: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    pregrasp_open = _preshape_gripper(selected_pickup)
    hold = _hold_gripper(selected_pickup)
    specs: list[tuple[str, dict[str, Any], float]] = []
    if _truthy_env("OPENFORGE_BOWL_STACK_USE_HIGH_APPROACH", False) and selected_pickup.get("approach_pose"):
        specs.append(("bowl_high_approach", selected_pickup["approach_pose"], pregrasp_open))
    specs.append(("bowl_low_pregrasp", selected_pickup.get("low_pregrasp_pose") or selected_pickup["pregrasp_pose"], pregrasp_open))
    specs.extend([("bowl_grasp_pose", selected_pickup["grasp_pose"], pregrasp_open), ("bowl_small_lift", selected_pickup["lift_pose"], hold)])
    return specs


def _make_stack_candidates(selected_pickup: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    target_xyz = _xyz(target)
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    base_rpy = [float(v) for v in list(grasp_pose.get("rpy") or [])[:3]]
    if target_xyz is None or source_xyz is None or len(base_rpy) < 3:
        raise RuntimeError("cannot build stack candidates without target/source detections and grasp rpy")
    release_above_target = _env_float("OPENFORGE_BOWL_STACK_RELEASE_TCP_ABOVE_WHITE_Z_M", 0.090)
    approach_lift = _env_float("OPENFORGE_BOWL_STACK_APPROACH_Z_LIFT_M", 0.080)
    transport_lift = _env_float("OPENFORGE_BOWL_STACK_TRANSPORT_Z_LIFT_M", 0.140)
    retreat_lift = _env_float("OPENFORGE_BOWL_STACK_RETREAT_Z_LIFT_M", 0.100)
    offsets = _offset_triples(
        "OPENFORGE_BOWL_STACK_CENTER_OFFSETS_M",
        [[0.0, 0.0, 0.0], [0.0, -0.015, 0.0], [0.0, 0.015, 0.0], [-0.015, 0.0, 0.0], [0.015, 0.0, 0.0]],
    )
    yaw_deltas = _csv_float("OPENFORGE_BOWL_STACK_PLACE_YAW_DELTAS_DEG", [0.0, -90.0, 90.0, -45.0, 45.0])
    max_candidates = max(1, _env_int("OPENFORGE_BOWL_STACK_MAX_PLACE_CANDIDATES", 15))
    candidates: list[dict[str, Any]] = []
    for offset_index, offset in enumerate(offsets):
        place_xy = [target_xyz[0] + float(offset[0]), target_xyz[1] + float(offset[1])]
        release_z = target_xyz[2] + release_above_target + float(offset[2])
        high_z = release_z + approach_lift
        pickup_lift_z = float((selected_pickup.get("lift_pose") or {}).get("position", [0, 0, 0])[2])
        transport_z = max(high_z, pickup_lift_z, target_xyz[2] + transport_lift)
        for yaw_index, yaw_delta in enumerate(yaw_deltas):
            if len(candidates) >= max_candidates:
                return candidates
            rpy = [base_rpy[0], base_rpy[1], base_rpy[2] + float(yaw_delta)]
            release_pose = {"position": [place_xy[0], place_xy[1], release_z], "rpy": rpy}
            high_pose = {"position": [place_xy[0], place_xy[1], high_z], "rpy": rpy}
            transport_pose = {"position": [place_xy[0], place_xy[1], transport_z], "rpy": rpy}
            retreat_pose = {"position": [place_xy[0], place_xy[1], release_z + retreat_lift], "rpy": rpy}
            candidates.append(
                {
                    "label": f"white_bowl_center_stack_o{offset_index}_y{yaw_index}",
                    "arm": selected_pickup["arm"],
                    "strategy_type": "orange_bowl_release_into_white_bowl_center",
                    "white_bowl_detection": target,
                    "pickup_strategy_label": selected_pickup.get("label"),
                    "place_target_object_ref_m": [round(place_xy[0], 5), round(place_xy[1], 5), round(target_xyz[2], 5)],
                    "relative_offset_m": _round_list(offset, limit=3),
                    "place_rpy": _round_list(rpy, limit=3),
                    "place_yaw_delta_deg": round(float(yaw_delta), 5),
                    "transport_high_pose": transport_pose,
                    "high_place_pose": high_pose,
                    "release_pose": release_pose,
                    "retreat_pose": retreat_pose,
                    "geometry_model": {
                        "source": "fresh orange pickup and fresh white bowl top-camera target detections",
                        "policy": "lift orange bowl, move above white-bowl center, descend slowly, open, retreat upward",
                        "release_tcp_above_white_z_m": round(release_above_target, 5),
                        "approach_z_lift_m": round(approach_lift, 5),
                        "transport_z_lift_m": round(transport_lift, 5),
                        "transport_z_policy": "max(release_z+approach_lift, pickup_lift_z, white_target_z+transport_lift)",
                        "retreat_z_lift_m": round(retreat_lift, 5),
                        "base_pickup_rpy": _round_list(base_rpy, limit=3),
                    },
                }
            )
    return candidates


def _sequence_specs(selected_pickup: dict[str, Any], stack_candidate: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    hold = _hold_gripper(selected_pickup)
    specs = _pickup_lift_specs(selected_pickup)
    specs.extend(
        [
            ("stack_transport_high", stack_candidate["transport_high_pose"], hold),
            ("stack_high_above_white_center", stack_candidate["high_place_pose"], hold),
            ("stack_release_pose", stack_candidate["release_pose"], hold),
            ("stack_retreat_after_open", stack_candidate["retreat_pose"], 1.0),
        ]
    )
    return specs


def _preview_stack_sequence(run_dir: Path, selected_pickup: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
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
            "stack_candidate": candidate,
            "steps": steps,
        }
        attempts.append(attempt)
        if success:
            packet = {
                "schema": "openforge.bowl_stack.sequence_preview.v1",
                "success": True,
                "side": side,
                "selected_pickup": selected_pickup,
                "selected_stack": candidate,
                "attempts": attempts,
            }
            packet["path"] = write_json(run_dir / "plans" / "orange_bowl_into_white_bowl_sequence_preview.json", packet)
            append_stage_summary(
                run_dir,
                [
                    "## orange bowl into white bowl sequence preview",
                    "- success: True",
                    f"- selected_stack: {candidate.get('label')}",
                    f"- release_pose: {candidate.get('release_pose')}",
                    f"- path: {packet['path']}",
                ],
            )
            return packet
    packet = {
        "schema": "openforge.bowl_stack.sequence_preview.v1",
        "success": False,
        "side": side,
        "selected_pickup": selected_pickup,
        "selected_stack": None,
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "plans" / "orange_bowl_into_white_bowl_sequence_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## orange bowl into white bowl sequence preview",
            "- success: False",
            f"- attempts: {[(item.get('label'), item.get('failed_stage')) for item in attempts]}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _ticket_status(run_dir: Path) -> dict[str, Any]:
    required_true = [
        "OPENFORGE_BOWL_STACK_PHYSICAL_TICKET",
        "OPENFORGE_BOWL_STACK_OPERATOR_PRESENT",
        "OPENFORGE_BOWL_STACK_ESTOP_REACHABLE",
        "OPENFORGE_BOWL_STACK_WORKSPACE_CLEAR",
        "OPENFORGE_BOWL_STACK_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_BOWL_STACK_ORANGE_BOWL_VISIBLE",
        "OPENFORGE_BOWL_STACK_WHITE_BOWL_VISIBLE",
        "OPENFORGE_BOWL_STACK_NO_LIQUID",
        "OPENFORGE_BOWL_STACK_NO_MANUAL_XYZ",
        "OPENFORGE_BOWL_STACK_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_BOWL_STACK_FOLLOWER_SERVERS_UP",
        "OPENFORGE_BOWL_STACK_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_BOWL_STACK_TICKET_SCOPE", "").strip()
    if scope != "orange_bowl_into_white_bowl":
        missing.append("OPENFORGE_BOWL_STACK_TICKET_SCOPE=orange_bowl_into_white_bowl")
    budget = _env_int("OPENFORGE_BOWL_STACK_PHYSICAL_ATTEMPT_BUDGET", 0)
    max_budget = _env_int("OPENFORGE_BOWL_STACK_MAX_ALLOWED_ATTEMPT_BUDGET", 20)
    if budget < 1 or budget > max_budget:
        missing.append(f"OPENFORGE_BOWL_STACK_PHYSICAL_ATTEMPT_BUDGET=1..{max_budget}")
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
    time.sleep(_env_float("OPENFORGE_BOWL_STACK_GRIPPER_SETTLE_S", 0.25))
    after = read_gripper_width(get_robot_state, side)
    deadline = time.monotonic() + _env_float("OPENFORGE_BOWL_STACK_GRIPPER_CONFIRM_TIMEOUT_S", 2.5)

    def _in_confirm_window(value: float | None) -> bool:
        if value is None:
            return True
        if confirm_min is not None and value < confirm_min:
            return False
        if confirm_max is not None and value > confirm_max:
            return False
        return True

    if after is not None and (confirm_min is not None or confirm_max is not None):
        while not _in_confirm_window(after) and time.monotonic() < deadline:
            time.sleep(0.1)
            after = read_gripper_width(get_robot_state, side)
    ok = _in_confirm_window(after)
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
        "error": None if ok else f"gripper state {after} outside confirmed window",
    }


def _capture_post_action_observation(run_dir: Path, stage: str) -> dict[str, Any]:
    return _capture_scene(
        run_dir,
        stage,
        _csv("OPENFORGE_BOWL_STACK_POST_PROMPTS", ["orange bowl", "white bowl", "bowl", "blue gripper fingers"]),
    )


def _pre_close_gate(selected_pickup: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    orange = _select_detection(
        observation,
        preferred_prompts=("orange bowl",),
        min_score_env="OPENFORGE_BOWL_STACK_ORANGE_MIN_SCORE",
        default_min_score=0.25,
        require_top=True,
    )
    source_xyz = _xyz(selected_pickup.get("source_detection") or {})
    current_xyz = _xyz(orange)
    xy_drift = None
    z_delta = None
    if source_xyz and current_xyz:
        xy_drift = math.hypot(current_xyz[0] - source_xyz[0], current_xyz[1] - source_xyz[1])
        z_delta = current_xyz[2] - source_xyz[2]
    checks = {
        "orange_detected": orange is not None and current_xyz is not None,
        "orange_xy_drift_ok": xy_drift is not None and xy_drift <= _env_float("OPENFORGE_BOWL_STACK_PRE_CLOSE_MAX_ORANGE_XY_DRIFT_M", 0.065),
        "orange_z_jump_ok": z_delta is not None and abs(z_delta) <= _env_float("OPENFORGE_BOWL_STACK_PRE_CLOSE_MAX_ORANGE_Z_JUMP_M", 0.080),
    }
    packet = {
        "schema": "openforge.bowl_stack.pre_close_gate.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "selected_orange_detection": orange,
        "source_orange_xyz": source_xyz,
        "current_orange_xyz": current_xyz,
        "orange_xy_drift_m": round(xy_drift, 5) if xy_drift is not None else None,
        "orange_z_delta_m": round(z_delta, 5) if z_delta is not None else None,
    }
    return packet


def _verify_stack(selected_stack: dict[str, Any], post_observation: dict[str, Any]) -> dict[str, Any]:
    orange = _select_detection(
        post_observation,
        preferred_prompts=("orange bowl",),
        min_score_env="OPENFORGE_BOWL_STACK_ORANGE_MIN_SCORE",
        default_min_score=0.20,
        require_top=True,
    )
    white = _select_detection(
        post_observation,
        preferred_prompts=("white bowl",),
        min_score_env="OPENFORGE_BOWL_STACK_WHITE_MIN_SCORE",
        default_min_score=0.20,
        require_top=True,
    )
    orange_xyz = _xyz(orange)
    white_xyz = _xyz(white) or _xyz(selected_stack.get("white_bowl_detection") or {})
    target = selected_stack.get("place_target_object_ref_m") or white_xyz
    xy_to_target = None
    orange_above_white = None
    if orange_xyz and target:
        xy_to_target = math.hypot(orange_xyz[0] - float(target[0]), orange_xyz[1] - float(target[1]))
    if orange_xyz and white_xyz:
        orange_above_white = orange_xyz[2] - white_xyz[2]
    xy_ok = xy_to_target is not None and xy_to_target <= _env_float("OPENFORGE_BOWL_STACK_VERIFY_MAX_ORANGE_XY_TO_WHITE_M", 0.090)
    z_ok = orange_above_white is None or orange_above_white >= _env_float("OPENFORGE_BOWL_STACK_VERIFY_MIN_ORANGE_ABOVE_WHITE_Z_M", -0.020)
    success = bool(orange and orange_xyz and xy_ok and z_ok)
    return {
        "schema": "openforge.bowl_stack.verify_stack.v1",
        "success": success,
        "confidence": 0.75 if success else 0.3,
        "reason": "orange bowl detected near white bowl center" if success else "post-stack detection is not centered in white bowl",
        "orange_detection": orange,
        "white_detection": white,
        "xy_to_target_m": round(xy_to_target, 5) if xy_to_target is not None else None,
        "orange_above_white_z_m": round(orange_above_white, 5) if orange_above_white is not None else None,
        "checks": {"xy_ok": xy_ok, "z_ok": z_ok},
    }


def _execute_physical(run_dir: Path, selected_pickup: dict[str, Any], selected_stack: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    side = str(selected_pickup["arm"])
    hold = _hold_gripper(selected_pickup)
    pregrasp_open = _preshape_gripper(selected_pickup)
    gripper_limit_kwargs = _gripper_limit_kwargs()
    attempt: dict[str, Any] = {
        "schema": "openforge.bowl_stack.physical_attempt.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "selected_stack": selected_stack,
        "steps": [],
        "gripper_after_close": None,
        "why_stopped": None,
    }
    try:
        open_step = _set_gripper_checked(
            side=side,
            target=pregrasp_open,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="checked_preshape_open",
            confirm_min=_env_float("OPENFORGE_BOWL_STACK_OPEN_CONFIRM_POS", 0.60),
            confirm_max=pregrasp_open + _env_float("OPENFORGE_BOWL_STACK_PRESHAPE_TOLERANCE_POS", 0.08),
            **gripper_limit_kwargs,
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(open_step)
        if not open_step["ok"]:
            attempt["why_stopped"] = "gripper did not confirm open before arm motion"
            return attempt

        specs = _sequence_specs(selected_pickup, selected_stack)
        before_close = [spec for spec in specs if spec[0] not in {"bowl_small_lift", "stack_transport_high", "stack_high_above_white_center", "stack_release_pose", "stack_retreat_after_open"}]
        after_close = [spec for spec in specs if spec[0] not in {item[0] for item in before_close}]
        for label, pose, gripper in before_close:
            step = _execute_pose(freespace_move, side, pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        pre_close_observation = _capture_post_action_observation(run_dir, "pre_close_grasp_pose_observe")
        pre_close_gate = _pre_close_gate(selected_pickup, pre_close_observation)
        pre_close_gate["path"] = write_json(run_dir / "plans" / "orange_bowl_pre_close_gate.json", pre_close_gate)
        attempt["pre_close_observation"] = pre_close_observation
        attempt["pre_close_gate"] = pre_close_gate
        if _truthy_env("OPENFORGE_BOWL_STACK_ENABLE_PRE_CLOSE_GATE", True) and not bool(pre_close_gate.get("passed")):
            attempt["why_stopped"] = "pre-close orange bowl gate failed; stop before close/lift/stack"
            return attempt

        close = staged_close_with_contact(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=hold,
            steps=_csv_float("OPENFORGE_BOWL_STACK_CLOSE_STEPS", [0.68, 0.40, 0.18, 0.08, 0.04]),
            **gripper_limit_kwargs,
            min_contact_delta=_env_float("OPENFORGE_BOWL_STACK_CLOSE_CONTACT_MIN_DELTA_POS", 0.02),
            target_tolerance=_env_float("OPENFORGE_BOWL_STACK_CLOSE_TARGET_TOLERANCE_POS", 0.05),
            hold_min=_env_float("OPENFORGE_BOWL_STACK_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_BOWL_STACK_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="orange_bowl_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["gripper_after_close"] = close
        if _truthy_env("OPENFORGE_BOWL_STACK_REQUIRE_CONTACT_BEFORE_LIFT", False) and not bool(close.get("contact_detected")):
            attempt["why_stopped"] = "staged close did not produce contact evidence; stop before lift/stack"
            return attempt
        if not bool(close.get("plausible_for_lift")):
            attempt["why_stopped"] = "staged close did not produce plausible gripper-state evidence; stop before lift/stack"
            return attempt

        lift_label, lift_pose, lift_gripper = after_close[0]
        lift_step = _execute_pose(freespace_move, side, lift_pose, label=lift_label, gripper=lift_gripper)
        attempt["movement_capable_calls"].append(f"freespace_move:{lift_label}")
        attempt["steps"].append(lift_step)
        post_lift = _capture_post_action_observation(run_dir, "post_lift_before_stack_observe")
        post_lift_det = _select_detection(
            post_lift,
            preferred_prompts=("orange bowl",),
            min_score_env="OPENFORGE_BOWL_STACK_ORANGE_MIN_SCORE",
            default_min_score=0.20,
            require_top=True,
        )
        attempt["post_lift_observation"] = post_lift
        attempt["lift_verification"] = verify_lift(
            selected_pickup.get("source_detection"),
            post_lift_det,
            min_z_delta_m=_env_float("OPENFORGE_BOWL_STACK_VERIFY_LIFT_MIN_Z_DELTA_M", 0.025),
            max_xy_delta_m=_env_float("OPENFORGE_BOWL_STACK_VERIFY_LIFT_MAX_XY_DELTA_M", 0.24),
        )
        if _truthy_env("OPENFORGE_BOWL_STACK_REQUIRE_LIFT_PERCEPTION_BEFORE_STACK", True) and not bool(
            attempt["lift_verification"].get("success")
        ):
            attempt["why_stopped"] = "lift perception verification failed; stop before transport/stack"
            return attempt

        for label, pose, gripper in after_close[1:4]:
            step = _execute_pose(freespace_move, side, pose, label=label, gripper=gripper)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        release = _set_gripper_checked(
            side=side,
            target=_env_float("OPENFORGE_BOWL_STACK_RELEASE_GRIPPER_POS", 1.0),
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            stage="open_release_into_white_bowl",
            confirm_min=_env_float("OPENFORGE_BOWL_STACK_RELEASE_CONFIRM_POS", 0.55),
            **gripper_limit_kwargs,
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(release)
        if not release["ok"]:
            attempt["why_stopped"] = "release gripper did not confirm open"
            return attempt

        label, pose, gripper = after_close[4]
        retreat = _execute_pose(freespace_move, side, pose, label=label, gripper=gripper)
        attempt["movement_capable_calls"].append(f"freespace_move:{label}")
        attempt["steps"].append(retreat)

        post_stack = _capture_post_action_observation(run_dir, "post_stack_observe")
        attempt["post_observation"] = post_stack
        verification = _verify_stack(selected_stack, post_stack)
        attempt["stack_verification"] = verification
        attempt["verification_success"] = bool(verification.get("success"))
        attempt["success"] = bool(attempt["verification_success"])
        attempt["why_stopped"] = (
            "orange bowl stacked into white bowl center"
            if attempt["success"]
            else f"stack sequence completed but verification failed/inconclusive: {verification.get('reason')}"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.bowl_stack.result.v1",
        "success": False,
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "observation_policy": "fresh live detections only; manual XYZ env vars are rejected",
        "planner_result": None,
        "pickup_validation": None,
        "stack_observation": None,
        "target_geometry_gate": None,
        "stack_candidates_path": None,
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
            result["why_stopped"] = "manual XYZ/debug env vars are not allowed for this live stacking task"
            result["manual_pose_env_vars_present"] = manual_vars
            result["next_recommendation"] = "Clear manual pose env vars and rerun from fresh live perception."
            return result

        planner_result = _run_fresh_orange_pickup_planner()
        result["planner_result"] = planner_result
        write_json(run_dir / "plans" / "fresh_orange_bowl_pickup_planner_result.json", planner_result)
        pickup_validation = _validate_pickup_strategy(planner_result)
        result["pickup_validation"] = pickup_validation
        append_stage_summary(
            run_dir,
            [
                "## orange bowl pickup validation",
                f"- passed: {pickup_validation['passed']}",
                f"- selected_label: {pickup_validation.get('selected_label')}",
                f"- arm: {pickup_validation.get('arm')}",
                f"- source_detection: {pickup_validation.get('source_detection')}",
            ],
        )
        if not pickup_validation.get("passed"):
            result["why_stopped"] = "fresh pickup planner did not produce a valid orange bowl rim-straddle strategy"
            result["next_recommendation"] = "Inspect pickup planner artifacts; no physical motion executed."
            return result

        selected_pickup = pickup_validation["selected_strategy"]
        stack_observation = _observe_stack_targets(run_dir)
        result["stack_observation"] = stack_observation
        white_detection = stack_observation.get("selected_white_detection")
        if not white_detection:
            result["why_stopped"] = "fresh white bowl target detection failed"
            result["next_recommendation"] = "Make the white bowl visible to the top camera and rerun no-motion validation."
            return result

        geometry_gate = _target_geometry_gate(selected_pickup, stack_observation)
        result["target_geometry_gate"] = geometry_gate
        append_stage_summary(
            run_dir,
            [
                "## stack target geometry gate",
                f"- passed: {geometry_gate['passed']}",
                f"- orange_radius_estimate: {geometry_gate.get('orange_radius_estimate')}",
                f"- white_radius_estimate: {geometry_gate.get('white_radius_estimate')}",
                f"- radius_margin_m: {geometry_gate.get('radius_margin_m')}",
                f"- xy_distance_m: {geometry_gate.get('xy_distance_m')}",
            ],
        )
        if not geometry_gate.get("passed"):
            result["why_stopped"] = "white bowl target geometry gate failed"
            result["next_recommendation"] = "Inspect target overlays; the target must be a larger visible bowl separated from the orange bowl."
            return result

        candidates = _make_stack_candidates(selected_pickup, white_detection)
        result["stack_candidates_path"] = write_json(
            run_dir / "plans" / "orange_into_white_stack_candidates.json",
            {
                "schema": "openforge.bowl_stack.candidates.v1",
                "selected_pickup": selected_pickup,
                "white_bowl_detection": white_detection,
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
        )
        sequence = _preview_stack_sequence(run_dir, selected_pickup, candidates)
        result["sequence_preview"] = sequence
        result["plan_success"] = bool(sequence.get("success"))
        if not sequence.get("success"):
            result["why_stopped"] = "pickup-to-white-bowl stack sequence preview failed"
            result["next_recommendation"] = "Inspect sequence preview artifacts before any physical attempt."
            return result

        if not result["physical_gate"]:
            result["success"] = True
            result["why_stopped"] = "recorded no-motion stacking validation succeeded; physical gate was not set"
            result["next_recommendation"] = (
                "For physical stacking, rerun under a fresh orange_bowl_into_white_bowl ticket "
                "with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and complete OPENFORGE_BOWL_STACK_* preconditions."
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
            result["next_recommendation"] = "Continue no-motion or rerun physical with complete stack ticket/precondition env vars."
            return result

        attempt = _execute_physical(run_dir, selected_pickup, sequence["selected_stack"])
        result["physical_attempt"] = attempt
        result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
        result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
        result["success"] = bool(attempt.get("success"))
        result["why_stopped"] = attempt.get("why_stopped")
        result["next_recommendation"] = "Do not retry without reviewing result.json, stage_summary.md, overlays, and videos."
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
                "## bowl stack result",
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
