# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion real AnyGrasp reference for orange-bowl top-down pickup.

This script is diagnostic only. It refuses physical motion, records the current
scene, runs the existing geometric top-down rim-straddle planner, samples real
AnyGrasp top-down candidates, previews those candidates without execution, and
writes a compact comparison artifact.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import runpy
import subprocess
import time
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


TASK_NAME = "anygrasp_orange_bowl_topdown_reference_no_motion"
PLANNER_PATH = Path(os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_PATH", "cap/saved_scripts/pick_plate_no_motion_plan.py"))
OBJECT_NAME = os.environ.get("OPENFORGE_ANYGRASP_BOWL_OBJECT_NAME", "orange bowl")
PROMPTS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_ANYGRASP_BOWL_PROMPTS", "orange bowl,bowl,white bowl").split(",")
    if part.strip()
]
CAMERAS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_ANYGRASP_BOWL_CAMERAS", "top,left,right").split(",")
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


def _sub3(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _norm3(v: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v[:3]))


def _dot3(a: list[float], b: list[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _display_rpy_world_axes(rpy: Any) -> dict[str, Any]:
    values = _round_list(rpy, digits=8, limit=3)
    if len(values) != 3:
        return {"error": "missing rpy"}
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation

        roll, pitch, yaw = np.asarray(values, dtype=np.float64)
        euler_xyz = [-pitch, roll, -yaw - 90.0]
        matrix = Rotation.from_euler("xyz", euler_xyz, degrees=True).as_matrix()
        return {
            "local_x_world": _round_list(matrix[:, 0].tolist(), limit=3),
            "local_y_world": _round_list(matrix[:, 1].tolist(), limit=3),
            "local_z_world": _round_list(matrix[:, 2].tolist(), limit=3),
            "local_z_dot_world_down": round(float(_dot3(matrix[:, 2].tolist(), [0.0, 0.0, -1.0])), 5),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_path = Path("cap/saved_scripts/anygrasp_orange_bowl_topdown_reference_no_motion.py").resolve()
    snapshot = {
        "script_path": str(script_path),
        "planner_path": str((Path.cwd() / PLANNER_PATH).resolve()),
        "task_name": TASK_NAME,
        "object_name": OBJECT_NAME,
        "prompts": PROMPTS,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_ANYGRASP_BOWL_")
            or name.startswith("OPENFORGE_BOWL_RACK_")
            or name.startswith("OPENFORGE_BOWL_TOP_DOWN_")
            or name.startswith("ANYGRASP_")
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
        snapshot["git_status_short"] = subprocess.check_output(
            ["git", "status", "--short", "--", str(script_path.relative_to(Path.cwd()))],
            text=True,
        ).strip()
    except Exception as exc:
        snapshot["git_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["path"] = write_json(run_dir / "source_snapshot.json", snapshot)
    return snapshot


def _anygrasp_health() -> dict[str, Any]:
    url = os.environ.get("ANYGRASP_SERVICE_URL", "http://127.0.0.1:8122").rstrip("/")
    health_url = f"{url}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=_env_float("OPENFORGE_ANYGRASP_BOWL_HEALTH_TIMEOUT_S", 5.0)) as response:
            body = response.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        data["url"] = health_url
        return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "url": health_url, "error": f"{type(exc).__name__}: {exc}"}


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


def _run_geometric_topdown_planner() -> dict[str, Any]:
    previous_gate = os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
    overrides = {
        "OPENFORGE_PLATE_TASK_NAME": f"{TASK_NAME}_geometric_plan",
        "OPENFORGE_PLATE_OBJECT_NAME": OBJECT_NAME,
        "OPENFORGE_PLATE_PROMPTS": ",".join(PROMPTS),
        "OPENFORGE_PLATE_COLLECT_ALL_PROMPTS": "1",
        "OPENFORGE_PLATE_TARGET_Y_SIGN": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_TARGET_Y_SIGN", "negative"),
        "OPENFORGE_PLATE_STRATEGY_MODE": "top_down_bowl_rim",
        "OPENFORGE_PLATE_ARM": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_ARM", "right"),
        "OPENFORGE_PLATE_MAX_CANDIDATES": os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_MAX_CANDIDATES", "96"),
        "OPENFORGE_PLATE_PLANNER_BACKEND": os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_BACKEND", "rrtconnect"),
        "OPENFORGE_PLATE_SOLVER_SPEED": os.environ.get("OPENFORGE_BOWL_RACK_SOLVER_SPEED", "fast"),
        "OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M": os.environ.get("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", "0.025"),
        "OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG": os.environ.get("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", "16"),
        "OPENFORGE_PLATE_IK_XYZ_WEIGHT": os.environ.get("OPENFORGE_BOWL_RACK_IK_XYZ_WEIGHT", "1.0"),
        "OPENFORGE_PLATE_IK_RPY_WEIGHT": os.environ.get("OPENFORGE_BOWL_RACK_IK_RPY_WEIGHT", "1.0"),
        "OPENFORGE_PLATE_PREVIEW_MAX_POS_ERROR_M": os.environ.get("OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", "0.018"),
        "OPENFORGE_PLATE_PREVIEW_MAX_ROT_ERROR_DEG": os.environ.get("OPENFORGE_BOWL_RACK_PREVIEW_MAX_ROT_ERROR_DEG", "8"),
        "OPENFORGE_PLATE_PREVIEW_STOP_AFTER_SUCCESSES": os.environ.get(
            "OPENFORGE_ANYGRASP_BOWL_GEOM_STOP_AFTER_SUCCESSES", "0"
        ),
    }
    previous = _with_env_overrides(overrides)
    try:
        planner_path = Path.cwd() / PLANNER_PATH
        spec = importlib.util.spec_from_file_location("openforge_anygrasp_bowl_geometric_plan", planner_path)
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
        namespace = runpy.run_path(spec.origin, init_globals=tool_globals, run_name="__openforge_anygrasp_bowl_geometric_plan__")
        return dict(namespace.get("TASK_RESULT") or {})
    finally:
        _restore_env(previous)
        if previous_gate is not None:
            os.environ["OPENFORGE_ALLOW_PHYSICAL_MOTION"] = previous_gate


def _candidate_to_dict(candidate: Any, index: int) -> dict[str, Any]:
    position = _xyz(_field(candidate, "position"))
    rpy = _round_list(_field(candidate, "rpy"), digits=4, limit=3)
    width = _field(candidate, "width", None)
    score = _field(candidate, "score", None)
    axes = _display_rpy_world_axes(rpy)
    return {
        "index": int(index),
        "source": "real_anygrasp_reference",
        "position": position,
        "rpy": rpy,
        "width_m": round(float(width), 5) if width is not None else None,
        "score": round(float(score), 5) if score is not None else None,
        "axes": axes,
        "top_down_axis_ok": bool(float(axes.get("local_z_dot_world_down", -1.0)) >= _env_float("OPENFORGE_ANYGRASP_BOWL_MIN_DOWN_DOT", 0.80)),
    }


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None) -> dict[str, Any]:
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": True,
        "planner_backend": os.environ.get("OPENFORGE_BOWL_RACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_BOWL_RACK_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_BOWL_RACK_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_BOWL_RACK_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_BOWL_RACK_IK_ROT_THRESHOLD_DEG", 16.0),
        "ik_xyz_weight": _env_float("OPENFORGE_BOWL_RACK_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_BOWL_RACK_IK_RPY_WEIGHT", 1.0),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _preview_summary(raw: Any) -> dict[str, Any]:
    safe = json_safe(raw)
    if isinstance(safe, dict) and "data" in safe and isinstance(safe["data"], dict):
        safe = safe["data"]
    status = _field(safe, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(safe, "executed", False))
    pos_error = _field(safe, "final_pos_error_m", _field(safe, "ik_error_m", None))
    rot_error = _field(safe, "final_rot_error_deg", _field(safe, "ik_rot_error_deg", None))
    max_pos = _env_float("OPENFORGE_BOWL_RACK_PREVIEW_MAX_POS_ERROR_M", 0.018)
    max_rot = _env_float("OPENFORGE_BOWL_RACK_PREVIEW_MAX_ROT_ERROR_DEG", 8.0)
    status_ok = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    pos_ok = pos_error is None or float(pos_error) <= max_pos
    rot_ok = rot_error is None or float(rot_error) <= max_rot
    return {
        "status": status_text,
        "reason": _field(safe, "reason", _field(safe, "error", None)),
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": max_pos,
        "max_rot_error_deg": max_rot,
        "success": bool(status_ok and pos_ok and rot_ok),
        "raw": safe,
    }


def _preview_pose(freespace_move: Callable[..., Any], side: str, pose: dict[str, Any], *, label: str, gripper: float) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        freespace_move,
        _env_float("OPENFORGE_ANYGRASP_BOWL_PREVIEW_TIMEOUT_S", 20.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pose, gripper=gripper),
    )
    if not result.get("ok"):
        return {"label": label, "success": False, "reason": result.get("error"), "executed": False}
    summary = _preview_summary(result.get("data"))
    summary["label"] = label
    return summary


def _preview_anygrasp_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    position = _xyz(candidate.get("position"))
    rpy = _round_list(candidate.get("rpy"), digits=4, limit=3)
    if position is None or len(rpy) != 3:
        return {"success": False, "reason": "candidate missing position/rpy", "steps": []}
    side = os.environ.get("OPENFORGE_BOWL_RACK_PICKUP_ARM", "right").strip() or "right"
    freespace_move = _required_tool("freespace_move")
    approach_lift = _env_float("OPENFORGE_ANYGRASP_BOWL_APPROACH_Z_LIFT_M", 0.040)
    lift_z = _env_float("OPENFORGE_ANYGRASP_BOWL_LIFT_Z_M", 0.025)
    gripper = _env_float("OPENFORGE_BOWL_RACK_APPROACH_GRIPPER_POS", 0.68)
    width = candidate.get("width_m")
    hold = max(0.02, min(0.95, float(width) if width is not None else _env_float("OPENFORGE_BOWL_RACK_CLOSE_TARGET_POS", 0.25)))
    poses = [
        ("anygrasp_high_approach", {"position": [position[0], position[1], position[2] + approach_lift], "rpy": rpy}, gripper),
        ("anygrasp_grasp_pose", {"position": position, "rpy": rpy}, gripper),
        ("anygrasp_small_lift", {"position": [position[0], position[1], position[2] + lift_z], "rpy": rpy}, hold),
    ]
    steps = [_preview_pose(freespace_move, side, pose, label=f"{label}_{candidate['index']}", gripper=g) for label, pose, g in poses]
    return {"success": all(bool(step.get("success")) for step in steps), "steps": steps}


def _sample_anygrasp() -> dict[str, Any]:
    sample = _required_tool("sample_grasp_pose_anygrasp")
    result = call_with_timeout(
        "sample_grasp_pose_anygrasp",
        sample,
        _env_float("OPENFORGE_ANYGRASP_BOWL_SAMPLE_TIMEOUT_S", 80.0),
        object_name=OBJECT_NAME,
        camera=os.environ.get("OPENFORGE_ANYGRASP_BOWL_CAMERA", "top"),
        max_grasps=_env_int("OPENFORGE_ANYGRASP_BOWL_MAX_GRASPS", 16),
        top_down_only=True,
        vertical_threshold=_env_float("OPENFORGE_ANYGRASP_BOWL_VERTICAL_THRESHOLD", 0.80),
        object_input_mode=os.environ.get("OPENFORGE_ANYGRASP_BOWL_OBJECT_INPUT_MODE", "segmented_object_cloud"),
        tcp_offset_z_m=_env_float("OPENFORGE_ANYGRASP_BOWL_TCP_OFFSET_Z_M", 0.0),
        disable_planner_z_clipping=_truthy_env("OPENFORGE_ANYGRASP_BOWL_DISABLE_PLANNER_Z_CLIPPING", True),
        filter_wrist_camera_y=_truthy_env("OPENFORGE_ANYGRASP_BOWL_FILTER_WRIST_Y", False),
        allow_wrist_camera_yaw_flip=True,
        run_in_background=_tool("run_in_background"),
    )
    if not result.get("ok"):
        return {"success": False, "error": result.get("error"), "candidates": []}
    raw_candidates = result.get("data") or []
    candidates = [_candidate_to_dict(candidate, index) for index, candidate in enumerate(raw_candidates, start=1)]
    preview_limit = min(len(candidates), _env_int("OPENFORGE_ANYGRASP_BOWL_PREVIEW_LIMIT", 8))
    for candidate in candidates[:preview_limit]:
        candidate["preview"] = _preview_anygrasp_candidate(candidate)
    return {"success": bool(candidates), "candidates": candidates}


def _compare(geometric: dict[str, Any], anygrasp: dict[str, Any]) -> dict[str, Any]:
    selected = geometric.get("selected_strategy") or {}
    source = selected.get("source_detection") or {}
    source_xyz = _xyz(source.get("position_3d") or source.get("position"))
    rim = selected.get("selected_rim_edge") or {}
    contact = _xyz(rim.get("contact_center_m") or rim.get("gap_center_m") or rim.get("edge_point_m"))
    grasp = _xyz((selected.get("grasp_pose") or {}).get("position"))
    rows: list[dict[str, Any]] = []
    for candidate in anygrasp.get("candidates") or []:
        pos = _xyz(candidate.get("position"))
        row = {
            "index": candidate.get("index"),
            "score": candidate.get("score"),
            "position": pos,
            "rpy": candidate.get("rpy"),
            "width_m": candidate.get("width_m"),
            "top_down_axis_ok": candidate.get("top_down_axis_ok"),
            "preview_success": bool((candidate.get("preview") or {}).get("success")),
        }
        if pos and source_xyz:
            delta = _sub3(pos, source_xyz)
            row["delta_to_source_xyz_m"] = _round_list(delta, limit=3)
            row["xy_to_source_m"] = round(math.hypot(delta[0], delta[1]), 5)
            row["z_delta_to_source_m"] = round(delta[2], 5)
        if pos and contact:
            delta = _sub3(pos, contact)
            row["delta_to_geometric_contact_m"] = _round_list(delta, limit=3)
            row["xy_to_geometric_contact_m"] = round(math.hypot(delta[0], delta[1]), 5)
            row["z_delta_to_geometric_contact_m"] = round(delta[2], 5)
        if pos and grasp:
            delta = _sub3(pos, grasp)
            row["delta_to_geometric_grasp_tcp_m"] = _round_list(delta, limit=3)
            row["distance_to_geometric_grasp_tcp_m"] = round(_norm3(delta), 5)
        rows.append(row)
    return {
        "geometric_selected_label": selected.get("label"),
        "geometric_source_xyz": source_xyz,
        "geometric_contact_center": contact,
        "geometric_grasp_pose": selected.get("grasp_pose"),
        "geometric_axis_gate": selected.get("axis_gate"),
        "anygrasp_count": len(anygrasp.get("candidates") or []),
        "preview_success_count": sum(1 for row in rows if row.get("preview_success")),
        "rows": rows,
        "interpretation": (
            "AnyGrasp rows are no-motion reference evidence only. "
            "Physical motion still requires the normal ticket, preview, visual gate, contact gate, and post-action verification."
        ),
    }


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_ANYGRASP_BOWL_RECORDING_WAIT_S", 5.0))
    videos = [str(path) for path in (run_dir / name for name in ("top.mp4", "left.mp4", "right.mp4")) if path.exists() and path.stat().st_size > 0]
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = [str(path) for path in (run_dir / name for name in ("top.mp4", "left.mp4", "right.mp4")) if path.exists() and path.stat().st_size > 0]
    return {"required": True, "passed": bool(videos), "video_paths": videos}


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result: dict[str, Any] = {
        "schema": "openforge.anygrasp_bowl_reference.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "source_snapshot": None,
        "scene_observation": None,
        "anygrasp_health": None,
        "geometric_planner_result": None,
        "anygrasp_result": None,
        "comparison": None,
        "recording_evidence": None,
        "video_paths": [],
        "why_stopped": None,
        "next_recommendation": None,
    }
    try:
        result["source_snapshot"] = _source_snapshot(run_dir)
        if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
            result["why_stopped"] = "refusing physical gate; this AnyGrasp reference script is no-motion only"
            result["next_recommendation"] = "Rerun without OPENFORGE_ALLOW_PHYSICAL_MOTION."
            return result
        result["anygrasp_health"] = _anygrasp_health()
        if result["anygrasp_health"].get("status") not in {"ok", "healthy"}:
            result["why_stopped"] = "real AnyGrasp health check failed"
            result["next_recommendation"] = "Fix AnyGrasp service before using it as reference evidence."
            return result

        result["scene_observation"] = capture_scene(
            prompts=PROMPTS,
            cameras=CAMERAS,
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="pre_anygrasp_reference",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
            max_retries=1,
        )

        geometric = _run_geometric_topdown_planner()
        result["geometric_planner_result"] = geometric
        write_json(run_dir / "plans" / "geometric_topdown_planner_result.json", geometric)
        anygrasp = _sample_anygrasp()
        result["anygrasp_result"] = anygrasp
        comparison = _compare(geometric, anygrasp)
        result["comparison"] = comparison
        write_json(run_dir / "plans" / "anygrasp_topdown_reference.json", anygrasp)
        write_json(run_dir / "plans" / "anygrasp_vs_geometric_comparison.json", comparison)

        result["success"] = bool(anygrasp.get("success"))
        result["why_stopped"] = (
            "recorded no-motion AnyGrasp top-down reference and geometric comparison"
            if result["success"]
            else f"AnyGrasp reference returned no usable top-down candidates: {anygrasp.get('error')}"
        )
        result["next_recommendation"] = (
            "Inspect anygrasp_vs_geometric_comparison.json and video frames; revise lateral/rim straddle geometry before another physical close."
        )
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and service logs before retrying."
        return result
    finally:
        result["recording_evidence"] = _recording_evidence(run_dir)
        result["video_paths"] = result["recording_evidence"].get("video_paths", [])
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## AnyGrasp orange bowl top-down reference",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- anygrasp_health: {result.get('anygrasp_health')}",
                f"- anygrasp_count: {((result.get('comparison') or {}).get('anygrasp_count'))}",
                f"- preview_success_count: {((result.get('comparison') or {}).get('preview_success_count'))}",
                f"- recording_evidence: {result['recording_evidence']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
