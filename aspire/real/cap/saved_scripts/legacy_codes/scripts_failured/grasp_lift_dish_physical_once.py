# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded one-attempt physical wrapper for dish/plate pickup on YAM.

The wrapper always runs the committed no-motion dish planner first with fresh
live detections and preview-only motion checks.  Physical mode requires
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 plus explicit ticket/precondition env
acknowledgements.  It does not place, release, or go home.
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    read_gripper_width,
    staged_close_with_contact,
    verify_lift,
    write_json,
)


TASK_NAME = "grasp_lift_dish_physical_once"
PLANNER_PATH = Path(os.environ.get("OPENFORGE_DISH_PLANNER_PATH", "cap/saved_scripts/grasp_lift_dish_no_motion_plan.py"))
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


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


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


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_PLATE_XYZ",
        "OPENFORGE_PLATE_HALF_EXTENTS",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
        "OPENFORGE_BOTTLE_XYZ",
        "OPENFORGE_CAN_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_DISH_RECORDING_WAIT_S", 5.0))
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
    script_path = Path("cap/saved_scripts/grasp_lift_dish_physical_once.py").resolve()
    snapshot = {
        "script_path": str(script_path),
        "planner_path": str(PLANNER_PATH.resolve()),
        "task_name": TASK_NAME,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_DISH_")
            or name.startswith("OPENFORGE_PLATE_")
            or name
            in {
                "YAM_STATION_CALIBRATED_XML",
                "CAP_TOP_CAMERA_BACKEND",
                "CAP_TOP_CAMERA_FRAME",
                "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP",
                "OPENFORGE_ALLOW_PHYSICAL_MOTION",
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


def _run_fresh_no_motion_planner() -> dict[str, Any]:
    """Run the dish planner with physical gate temporarily removed."""
    previous_gate = os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
    previous_task_name = os.environ.get("OPENFORGE_PLATE_TASK_NAME")
    previous_entrypoint = os.environ.get("OPENFORGE_PLATE_ENTRYPOINT")
    try:
        os.environ.setdefault("OPENFORGE_PLATE_TASK_NAME", "grasp_lift_dish_physical_once_plan")
        os.environ.setdefault("OPENFORGE_PLATE_ENTRYPOINT", str(Path.cwd() / PLANNER_PATH))
        spec = importlib.util.spec_from_file_location("openforge_dish_no_motion_plan", Path.cwd() / PLANNER_PATH)
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
        namespace = runpy.run_path(spec.origin, init_globals=tool_globals, run_name="__openforge_dish_plan__")
        return dict(namespace.get("TASK_RESULT") or {})
    finally:
        if previous_gate is not None:
            os.environ["OPENFORGE_ALLOW_PHYSICAL_MOTION"] = previous_gate
        elif "OPENFORGE_ALLOW_PHYSICAL_MOTION" in os.environ:
            os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
        if previous_task_name is None:
            os.environ.pop("OPENFORGE_PLATE_TASK_NAME", None)
        else:
            os.environ["OPENFORGE_PLATE_TASK_NAME"] = previous_task_name
        if previous_entrypoint is None:
            os.environ.pop("OPENFORGE_PLATE_ENTRYPOINT", None)
        else:
            os.environ["OPENFORGE_PLATE_ENTRYPOINT"] = previous_entrypoint


def _validate_selected_strategy(planner_result: dict[str, Any]) -> dict[str, Any]:
    selected = planner_result.get("selected_strategy") or {}
    source = selected.get("source_detection") or {}
    expected_strategy_type = os.environ.get("OPENFORGE_DISH_EXPECT_STRATEGY_TYPE", "").strip()
    required = {
        "selected.label": selected.get("label"),
        "selected.arm": selected.get("arm"),
        "pregrasp_pose.position": (selected.get("pregrasp_pose") or {}).get("position"),
        "grasp_pose.position": (selected.get("grasp_pose") or {}).get("position"),
        "lift_pose.position": (selected.get("lift_pose") or {}).get("position"),
        "source_detection.position_3d": source.get("position_3d"),
    }
    missing = [name for name, value in required.items() if not value]
    source_name = str(source.get("source") or source.get("detection_source") or "")
    manual_source = "manual" in source_name.lower()
    strategy_mismatch = bool(expected_strategy_type and selected.get("strategy_type") != expected_strategy_type)
    residual_gate = selected.get("preview_residual_gate") or {}
    residual_gate_passed = bool(residual_gate.get("passed", True))
    passed = bool(
        planner_result.get("success")
        and selected
        and not missing
        and not manual_source
        and not strategy_mismatch
        and residual_gate_passed
    )
    return {
        "passed": passed,
        "missing": missing,
        "manual_source": manual_source,
        "expected_strategy_type": expected_strategy_type or None,
        "strategy_mismatch": strategy_mismatch,
        "preview_residual_gate_passed": residual_gate_passed,
        "preview_residual_gate": residual_gate,
        "selected_label": selected.get("label"),
        "selected_strategy_type": selected.get("strategy_type"),
        "arm": selected.get("arm"),
        "source_detection": source,
        "selected_strategy": selected if passed else None,
    }


def _ticket_status(run_dir: Path) -> dict[str, Any]:
    required_true = [
        "OPENFORGE_DISH_PHYSICAL_TICKET",
        "OPENFORGE_DISH_OPERATOR_PRESENT",
        "OPENFORGE_DISH_ESTOP_REACHABLE",
        "OPENFORGE_DISH_WORKSPACE_CLEAR",
        "OPENFORGE_DISH_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_DISH_OBJECT_PLACED",
        "OPENFORGE_DISH_NO_LIQUID",
        "OPENFORGE_DISH_NO_MANUAL_XYZ",
        "OPENFORGE_DISH_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_DISH_FOLLOWER_SERVERS_UP",
        "OPENFORGE_DISH_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_DISH_TICKET_SCOPE", "").strip()
    if scope != "dish_plate_pickup_lift_only":
        missing.append("OPENFORGE_DISH_TICKET_SCOPE=dish_plate_pickup_lift_only")
    budget = _env_int("OPENFORGE_DISH_PHYSICAL_ATTEMPT_BUDGET", 0)
    if budget != 1:
        missing.append("OPENFORGE_DISH_PHYSICAL_ATTEMPT_BUDGET=1")
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
        "pgrep": pgrep_status,
    }


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None, preview_only: bool = False) -> dict[str, Any]:
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_DISH_PHYSICAL_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_DISH_PHYSICAL_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_DISH_PHYSICAL_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_DISH_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_DISH_IK_ROT_THRESHOLD_DEG", 16.0),
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


def _stage_tolerance(label: str, *, execution: bool = False) -> dict[str, float]:
    """Return pose residual limits for a stage.

    The high approach is a free-space staging pose, so its exact wrist angle is
    less important than avoiding planner false negatives.  Contact/near-contact
    stages keep the strict default orientation gate.
    """
    prefix = "OPENFORGE_DISH_EXEC" if execution else "OPENFORGE_DISH_PREVIEW"
    default_pos = _env_float(f"{prefix}_MAX_POS_ERROR_M", 0.012)
    default_rot = _env_float(f"{prefix}_MAX_ROT_ERROR_DEG", 4.0)
    if "high_approach" in str(label):
        return {
            "max_pos_error_m": _env_float("OPENFORGE_DISH_HIGH_APPROACH_MAX_POS_ERROR_M", max(default_pos, 0.02)),
            "max_rot_error_deg": _env_float("OPENFORGE_DISH_HIGH_APPROACH_MAX_ROT_ERROR_DEG", max(default_rot, 12.0)),
        }
    if "low_pregrasp" in str(label):
        return {
            "max_pos_error_m": _env_float("OPENFORGE_DISH_LOW_PREGRASP_MAX_POS_ERROR_M", max(default_pos, 0.015)),
            "max_rot_error_deg": _env_float("OPENFORGE_DISH_LOW_PREGRASP_MAX_ROT_ERROR_DEG", max(default_rot, 8.0)),
        }
    return {"max_pos_error_m": default_pos, "max_rot_error_deg": default_rot}


def _preview_summary(result: Any, *, stage_label: str = "") -> dict[str, Any]:
    status = _field(result, "status", None)
    status_text = str(status) if status is not None else None
    executed = bool(_field(result, "executed", False))
    pos_error = _field(result, "final_pos_error_m", _field(result, "ik_error_m", None))
    rot_error = _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None))
    tolerance = _stage_tolerance(stage_label, execution=False)
    max_pos_error = tolerance["max_pos_error_m"]
    max_rot_error = tolerance["max_rot_error_deg"]
    status_ok = bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed)
    tolerance_ok = _error_within_limit(pos_error, max_pos_error) and _error_within_limit(rot_error, max_rot_error)
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "trajectory_cache_key": _field(result, "trajectory_cache_key", None),
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": max_pos_error,
        "max_rot_error_deg": max_rot_error,
        "tolerance_ok": tolerance_ok,
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
    max_pos_error = tolerance["max_pos_error_m"]
    max_rot_error = tolerance["max_rot_error_deg"]
    status_ok = bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed)
    tolerance_ok = _error_within_limit(pos_error, max_pos_error) and _error_within_limit(rot_error, max_rot_error)
    return {
        "stage_label": stage_label,
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": max_pos_error,
        "max_rot_error_deg": max_rot_error,
        "tolerance_ok": tolerance_ok,
        "success": bool(status_ok and tolerance_ok),
        "raw": json_safe(result),
    }


def _execute_pose(
    *,
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    label: str,
    gripper: float | None = None,
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


def _artifact_safe_label(value: Any) -> str:
    text = str(value or "candidate")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:80] or "candidate"


def _preshape_gripper(selected: dict[str, Any]) -> float:
    preshape_plan = selected.get("preshape_plan") or {}
    return float(
        preshape_plan.get(
            "expected_gripper_open_before_close_normalized",
            _env_float("OPENFORGE_DISH_APPROACH_GRIPPER_POS", 0.68),
        )
    )


def _pregrasp_gripper(selected: dict[str, Any]) -> float:
    configured = _env_float_or_none("OPENFORGE_DISH_APPROACH_GRIPPER_POS")
    return float(configured if configured is not None else _preshape_gripper(selected))


def _same_rpy(a: Any, b: Any, tolerance_deg: float = 1e-3) -> bool:
    try:
        av = [float(v) for v in list(a)[:3]]
        bv = [float(v) for v in list(b)[:3]]
    except Exception:
        return False
    if len(av) < 3 or len(bv) < 3:
        return False
    return all(abs(av[i] - bv[i]) <= tolerance_deg for i in range(3))


def _lift_orientation_policy(selected: dict[str, Any]) -> dict[str, Any]:
    grasp_rpy = (selected.get("grasp_pose") or {}).get("rpy")
    lift_rpy = (selected.get("lift_pose") or {}).get("rpy")
    same_orientation = _same_rpy(grasp_rpy, lift_rpy, _env_float("OPENFORGE_DISH_LIFT_RPY_MATCH_TOL_DEG", 0.01))
    return {
        "source": "plate-libero-transfer high-level flat-object guidance",
        "first_lift_uses_grasp_orientation": same_orientation,
        "reorientation_before_verified_lift": False,
        "grasp_rpy": grasp_rpy,
        "lift_rpy": lift_rpy,
        "passed": same_orientation,
        "problem": None if same_orientation else "lift_pose.rpy differs from grasp_pose.rpy",
    }


def _pose_available(pose: Any) -> bool:
    return bool(
        isinstance(pose, dict)
        and len(pose.get("position") or []) >= 3
        and len(pose.get("rpy") or []) >= 3
    )


def _set_gripper_checked_open(
    *,
    side: str,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    target: float | None = None,
) -> dict[str, Any]:
    configured = _env_float_or_none("OPENFORGE_DISH_OPEN_GRIPPER_POS")
    target = float(configured if configured is not None else (target if target is not None else 1.0))
    before = read_gripper_width(get_robot_state, side)
    result = set_gripper(
        side,
        target,
        vel_limit=_env_float("OPENFORGE_DISH_OPEN_VEL_LIMIT", 0.45),
        torque_limit=_env_float("OPENFORGE_DISH_OPEN_TORQUE_LIMIT", 0.18),
    )
    time.sleep(_env_float("OPENFORGE_DISH_GRIPPER_SETTLE_S", 0.25))
    after = read_gripper_width(get_robot_state, side)
    deadline = time.monotonic() + _env_float("OPENFORGE_DISH_OPEN_CONFIRM_TIMEOUT_S", 3.0)
    confirm_min = _env_float("OPENFORGE_DISH_OPEN_CONFIRM_POS", 0.55)
    while after is not None and after < confirm_min and time.monotonic() < deadline:
        time.sleep(0.1)
        after = read_gripper_width(get_robot_state, side)
    return {
        "stage": "checked_gripper_open",
        "side": side,
        "target": target,
        "before": before,
        "after": after,
        "confirm_min": confirm_min,
        "ok": after is None or after >= confirm_min,
        "result": json_safe(result),
    }


def _preview_physical_motion_sequence(
    run_dir: Path,
    selected: dict[str, Any],
    *,
    artifact_label: str = "physical_sequence_preview",
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected["arm"])
    close_plan = selected.get("staged_close_plan") or {}
    pregrasp_open = _pregrasp_gripper(selected)
    final_approach_gripper = _preshape_gripper(selected)
    hold_gripper = _env_float("OPENFORGE_DISH_CLOSE_TARGET_POS", float(close_plan.get("target_normalized_position", 0.54)))
    lift_policy = _lift_orientation_policy(selected)
    steps: list[dict[str, Any]] = []
    preview_specs: list[tuple[str, dict[str, Any], float]] = []
    if _pose_available(selected.get("approach_pose")):
        preview_specs.append(("plate_high_approach_with_physical_gripper", selected["approach_pose"], pregrasp_open))
    if _pose_available(selected.get("low_pregrasp_pose")):
        preview_specs.append(("plate_low_pregrasp_near_table", selected["low_pregrasp_pose"], final_approach_gripper))
    else:
        preview_specs.append(("plate_pregrasp_with_physical_gripper", selected["pregrasp_pose"], pregrasp_open))
    preview_specs.extend(
        [
            ("plate_tilted_insertion_grasp_pose", selected["grasp_pose"], final_approach_gripper),
            ("plate_small_lift_with_hold_width", selected["lift_pose"], hold_gripper),
        ]
    )
    success = bool(lift_policy.get("passed"))
    for label, pose, gripper in preview_specs:
        try:
            raw = freespace_move(**_move_kwargs(side, pose, gripper=gripper, preview_only=True))
            preview = _preview_summary(raw, stage_label=label)
        except Exception as exc:
            preview = {
                "status": None,
                "success": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "executed": False,
            }
        step = {
            "stage": label,
            "side": side,
            "pose": pose,
            "gripper": gripper,
            "preview": preview,
            "ok": bool(preview.get("success")),
        }
        steps.append(step)
        if not step["ok"]:
            success = False
            break
    packet = {
        "schema": "openforge.dish_plate.physical_sequence_preview.v1",
        "success": success,
        "side": side,
        "pregrasp_gripper": pregrasp_open,
        "final_approach_gripper": final_approach_gripper,
        "hold_gripper": hold_gripper,
        "lift_orientation_policy": lift_policy,
        "insertion_path": selected.get("insertion_path"),
        "gripper_length_model": selected.get("gripper_length_model"),
        "steps": steps,
        "path": None,
    }
    packet["candidate_label"] = selected.get("label")
    packet["path"] = write_json(run_dir / "plans" / f"{_artifact_safe_label(artifact_label)}.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## physical sequence no-motion preview",
            f"- success: {packet['success']}",
            f"- side: {side}",
            f"- pregrasp_gripper: {pregrasp_open}",
            f"- final_approach_gripper: {final_approach_gripper}",
            f"- hold_gripper: {hold_gripper}",
            f"- lift_orientation_policy: {lift_policy}",
            f"- insertion_path: {selected.get('insertion_path')}",
            f"- gripper_length_model: {selected.get('gripper_length_model')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _candidate_pool_from_planner_result(
    planner_result: dict[str, Any],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_strategy_type = os.environ.get("OPENFORGE_DISH_EXPECT_STRATEGY_TYPE", "").strip()
    inner_planner = planner_result.get("planner_result") or {}
    candidates = list(inner_planner.get("candidates") or [])
    plan_path = inner_planner.get("plan_path")
    if not candidates and plan_path:
        path = Path(str(plan_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            plan_packet = json.loads(path.read_text())
            candidates = list(plan_packet.get("candidates") or [])
        except Exception:
            candidates = []
    pool: list[dict[str, Any]] = []

    def _eligible(candidate: dict[str, Any]) -> bool:
        if not candidate:
            return False
        if expected_strategy_type and candidate.get("strategy_type") != expected_strategy_type:
            return False
        source = candidate.get("source_detection") or {}
        source_name = str(source.get("source") or source.get("detection_source") or "")
        if "manual" in source_name.lower():
            return False
        if not candidate.get("pregrasp_pose") or not candidate.get("grasp_pose") or not candidate.get("lift_pose"):
            return False
        return True

    if _eligible(selected):
        pool.append(selected)
    for candidate in candidates:
        if not candidate.get("preview_success"):
            continue
        if not _eligible(candidate):
            continue
        if any(existing.get("label") == candidate.get("label") for existing in pool):
            continue
        pool.append(candidate)
    return pool


def _recovery_before_pickup_status(planner_result: dict[str, Any]) -> dict[str, Any]:
    proximity = planner_result.get("robot_plate_proximity") or {}
    blocking = bool(proximity.get("blocking_risk"))
    return {
        "required": blocking,
        "blocking_risk": blocking,
        "warnings": list(proximity.get("warnings") or []),
        "arms": proximity.get("arms") or {},
        "policy": (
            "If an end effector is already near the plate before the pickup attempt, "
            "run a bounded recovery/open/retreat/go-home step before any grasp/lift motion."
        ),
        "source": "plate-libero-transfer high-level lesson: plan low flat-object motions from a clean start state",
    }


def _select_physical_ready_strategy(
    run_dir: Path,
    planner_result: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    max_scan = max(1, _env_int("OPENFORGE_DISH_PHYSICAL_READY_SCAN_MAX", 12))
    attempts: list[dict[str, Any]] = []
    for candidate in _candidate_pool_from_planner_result(planner_result, selected)[:max_scan]:
        preview = _preview_physical_motion_sequence(
            run_dir,
            candidate,
            artifact_label=f"physical_sequence_preview_{candidate.get('label')}",
        )
        attempts.append(
            {
                "label": candidate.get("label"),
                "arm": candidate.get("arm"),
                "strategy_type": candidate.get("strategy_type"),
                "success": bool(preview.get("success")),
                "preview_path": preview.get("path"),
                "failed_stage": next(
                    (step.get("stage") for step in preview.get("steps", []) if not step.get("ok")),
                    None,
                ),
            }
        )
        if preview.get("success"):
            packet = {
                "success": True,
                "selected_strategy": candidate,
                "selected_label": candidate.get("label"),
                "sequence_preview": preview,
                "attempts": attempts,
            }
            packet["path"] = write_json(run_dir / "plans" / "physical_ready_selection.json", packet)
            append_stage_summary(
                run_dir,
                [
                    "## physical-ready candidate selection",
                    "- success: True",
                    f"- selected_label: {candidate.get('label')}",
                    f"- arm: {candidate.get('arm')}",
                    f"- path: {packet['path']}",
                ],
            )
            return packet

    packet = {
        "success": False,
        "selected_strategy": None,
        "selected_label": None,
        "sequence_preview": None,
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "plans" / "physical_ready_selection.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## physical-ready candidate selection",
            "- success: False",
            f"- attempts: {attempts}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _detection_touches_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_DISH_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_DISH_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_DISH_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _select_post_detection(packet: dict[str, Any], pre_prompt: str | None) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    if pre_prompt:
        prompt_rank = {pre_prompt: 0}
    else:
        prompt_rank = {}

    def _eligible(det: dict[str, Any]) -> bool:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        score = float(det.get("score") or 0.0)
        return score >= _env_float("OPENFORGE_DISH_POST_MIN_SCORE", 0.20) and camera in {"top", "left", "right"}

    filtered = [det for det in detections if _eligible(det)]
    if not filtered:
        return None

    def _key(det: dict[str, Any]) -> tuple[int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        prompt = str(det.get("prompt") or "")
        edge = _detection_touches_edge(det)
        return (
            0 if camera == "top" else 1,
            1 if edge and camera != "top" else 0,
            prompt_rank.get(prompt, 1),
            -float(det.get("score") or 0.0),
        )

    selected = sorted(filtered, key=_key)[0]
    selected["touches_image_edge"] = _detection_touches_edge(selected)
    selected["detection_source"] = "live_top_camera" if selected.get("source_camera") == "top" else "live_calibrated_camera"
    return selected


def _post_observe(run_dir: Path, selected: dict[str, Any]) -> dict[str, Any]:
    source = selected.get("source_detection") or {}
    source_prompt = str(source.get("prompt") or "dish")
    prompts = []
    for prompt in [source_prompt, "dish", "plate", "dinner plate", "white plate"]:
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    packet = capture_scene(
        prompts=prompts,
        cameras=["top", "left", "right"],
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="post_lift_observe_dish",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=2,
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
    )
    packet["selected_post_detection"] = _select_post_detection(packet, source_prompt)
    packet["path"] = write_json(run_dir / "observations" / "post_lift_observe_dish_selected.json", packet)
    return packet


def _capture_attempt_observation(run_dir: Path, selected: dict[str, Any], stage: str) -> dict[str, Any]:
    source = selected.get("source_detection") or {}
    prompts: list[str] = []
    for prompt in [
        str(source.get("prompt") or "dish"),
        "dish",
        "plate",
        "white dish",
        "blue gripper fingers",
    ]:
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    try:
        packet = capture_scene(
            prompts=prompts,
            cameras=["top", "left", "right"],
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
        )
        packet["path"] = write_json(run_dir / "observations" / f"{_artifact_safe_label(stage)}.json", packet)
        append_stage_summary(
            run_dir,
            [
                f"## observation {stage}",
                f"- packet: {packet.get('packet_path')}",
                f"- best_detection: {packet.get('best_detection')}",
                f"- selected_source_detection: {source}",
                f"- path: {packet['path']}",
            ],
        )
        return packet
    except Exception as exc:
        packet = {"stage": stage, "error": f"{type(exc).__name__}: {exc}", "path": None}
        packet["path"] = write_json(run_dir / "observations" / f"{_artifact_safe_label(stage)}_error.json", packet)
        append_stage_summary(
            run_dir,
            [
                f"## observation {stage}",
                f"- error: {packet['error']}",
                f"- path: {packet['path']}",
            ],
        )
        return packet


def _execute_physical(run_dir: Path, selected: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    side = str(selected["arm"])
    close_plan = selected.get("staged_close_plan") or {}
    attempt: dict[str, Any] = {
        "schema": "openforge.dish_plate.physical_attempt.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_strategy": selected,
        "steps": [],
        "why_stopped": None,
        "object_state_after_attempt": "unknown",
        "release_or_return_executed": False,
    }
    try:
        pregrasp_open = _pregrasp_gripper(selected)
        open_step = _set_gripper_checked_open(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=pregrasp_open,
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["steps"].append(open_step)
        if not open_step["ok"]:
            attempt["why_stopped"] = "gripper did not confirm open before arm motion"
            return attempt

        preshape_plan = selected.get("preshape_plan") or {}
        final_approach_gripper = _preshape_gripper(selected)
        if final_approach_gripper < pregrasp_open:
            preshape_result = set_gripper(
                side,
                final_approach_gripper,
                vel_limit=_env_float("OPENFORGE_DISH_PRESHAPE_VEL_LIMIT", 0.08),
                torque_limit=_env_float("OPENFORGE_DISH_PRESHAPE_TORQUE_LIMIT", 0.12),
            )
            time.sleep(_env_float("OPENFORGE_DISH_GRIPPER_SETTLE_S", 0.25))
            preshape_after = read_gripper_width(get_robot_state, side)
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
            attempt["steps"].append(
                {
                    "stage": "plate_preshape_before_under_over_insertion",
                    "side": side,
                    "target": final_approach_gripper,
                    "after": preshape_after,
                    "ok": True,
                    "result": json_safe(preshape_result),
                    "preshape_plan": preshape_plan,
                }
            )

        approach_steps: list[tuple[str, dict[str, Any], float]] = []
        if _pose_available(selected.get("approach_pose")):
            approach_steps.append(("plate_high_approach", selected["approach_pose"], pregrasp_open))
        if _pose_available(selected.get("low_pregrasp_pose")):
            approach_steps.append(("plate_low_pregrasp_near_table", selected["low_pregrasp_pose"], final_approach_gripper))
        else:
            approach_steps.append(("plate_pregrasp", selected["pregrasp_pose"], pregrasp_open))
        approach_steps.append(("plate_tilted_insertion_grasp_pose", selected["grasp_pose"], final_approach_gripper))
        attempt["insertion_path"] = selected.get("insertion_path")
        attempt["gripper_length_model"] = selected.get("gripper_length_model")

        for label, pose, gripper in approach_steps:
            step = _execute_pose(
                freespace_move=freespace_move,
                side=side,
                pose=pose,
                label=label,
                gripper=gripper,
            )
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        attempt["pre_close_observation"] = _capture_attempt_observation(
            run_dir,
            selected,
            "pre_close_grasp_pose_observe",
        )
        close = staged_close_with_contact(
            side=side,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=_env_float("OPENFORGE_DISH_CLOSE_TARGET_POS", float(close_plan.get("target_normalized_position", 0.45))),
            steps=_csv_float("OPENFORGE_DISH_CLOSE_STEPS", list(close_plan.get("steps") or [0.75, 0.60, 0.50, 0.45])),
            vel_limit=_env_float("OPENFORGE_DISH_CLOSE_VEL_LIMIT", float(close_plan.get("vel_limit", 0.08))),
            torque_limit=_env_float("OPENFORGE_DISH_CLOSE_TORQUE_LIMIT", float(close_plan.get("torque_limit", 0.18))),
            min_contact_delta=_env_float(
                "OPENFORGE_DISH_CLOSE_CONTACT_MIN_DELTA_POS",
                float(close_plan.get("expected_contact_delta_min", 0.015)),
            ),
            target_tolerance=_env_float("OPENFORGE_DISH_CLOSE_TARGET_TOLERANCE_POS", 0.06),
            hold_min=_env_float("OPENFORGE_DISH_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_DISH_HOLD_MAX_POS", 0.95),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="dish_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{side}")
        attempt["gripper_after_close"] = close
        attempt["post_close_observation"] = _capture_attempt_observation(
            run_dir,
            selected,
            "post_close_before_lift_observe",
        )
        require_contact = _truthy_env("OPENFORGE_DISH_REQUIRE_CONTACT_BEFORE_LIFT", True)
        if require_contact and not bool(close.get("contact_detected")):
            attempt["why_stopped"] = "staged close did not produce contact evidence; stop before lift"
            attempt["object_state_after_attempt"] = "at_grasp_pose_no_verified_hold"
            return attempt
        if not bool(close.get("plausible_for_lift")):
            attempt["why_stopped"] = "staged close did not produce plausible gripper-state evidence; stop before lift"
            attempt["object_state_after_attempt"] = "at_grasp_pose_no_verified_hold"
            return attempt

        hold = float(close.get("after") or close.get("target") or 0.45)
        step = _execute_pose(
            freespace_move=freespace_move,
            side=side,
            pose=selected["lift_pose"],
            label="plate_small_lift",
            gripper=hold,
        )
        attempt["movement_capable_calls"].append("freespace_move:plate_small_lift")
        attempt["steps"].append(step)
        attempt["object_state_after_attempt"] = "lift_pose_reached_gripper_closed"

        post = _post_observe(run_dir, selected)
        attempt["post_observation"] = post
        verification = verify_lift(
            selected.get("source_detection"),
            post.get("selected_post_detection"),
            min_z_delta_m=_env_float("OPENFORGE_DISH_VERIFY_MIN_Z_DELTA_M", 0.045),
            max_xy_delta_m=_env_float("OPENFORGE_DISH_VERIFY_MAX_XY_DELTA_M", 0.22),
        )
        attempt["lift_verification"] = verification
        attempt["verification_success"] = bool(verification.get("success"))
        attempt["success"] = bool(attempt["verification_success"])
        attempt["why_stopped"] = (
            "small lift verified; dish/plate may still be held at lift pose, no release/place authorized"
            if attempt["success"]
            else f"lift command completed but verification is inconclusive/failed: {verification.get('reason')}"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.dish_plate.physical_once.result.v1",
        "success": False,
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "observation_policy": "fresh live detections only; manual XYZ env vars are rejected",
        "planner_result": None,
        "plan_validation": None,
        "recovery_before_pickup": None,
        "physical_sequence_preview": None,
        "physical_ready_selection": None,
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
        manual_env = _manual_pose_env_vars_present()
        if manual_env:
            result["why_stopped"] = "manual XYZ/debug env vars are not allowed for this live dish/plate task"
            result["manual_pose_env_vars_present"] = manual_env
            return result

        planner_result = _run_fresh_no_motion_planner()
        result["planner_result"] = planner_result
        write_json(run_dir / "plans" / "fresh_no_motion_planner_result.json", planner_result)
        plan_validation = _validate_selected_strategy(planner_result)
        result["plan_validation"] = plan_validation
        result["plan_success"] = bool(plan_validation.get("passed"))
        append_stage_summary(
            run_dir,
            [
                "## physical wrapper plan validation",
                f"- passed: {plan_validation['passed']}",
                f"- selected_label: {plan_validation.get('selected_label')}",
                f"- arm: {plan_validation.get('arm')}",
                f"- source_detection: {plan_validation.get('source_detection')}",
            ],
        )
        if not plan_validation.get("passed"):
            result["why_stopped"] = "fresh no-motion planner did not produce a valid selected dish/plate strategy"
            result["next_recommendation"] = "Inspect planner artifacts and continue no-motion planning; no physical motion executed."
            return result

        recovery_status = _recovery_before_pickup_status(planner_result)
        result["recovery_before_pickup"] = recovery_status
        append_stage_summary(
            run_dir,
            [
                "## recovery-before-pickup gate",
                f"- required: {recovery_status['required']}",
                f"- warnings: {recovery_status['warnings']}",
                f"- policy: {recovery_status['policy']}",
            ],
        )
        if recovery_status.get("required") and not _truthy_env("OPENFORGE_DISH_IGNORE_RECOVERY_GATE", False):
            result["why_stopped"] = "current end effector is already near the plate; recovery/open/retreat/go-home is required before pickup preview"
            result["next_recommendation"] = (
                "Do not execute pickup from this start state. Obtain a bounded recovery/home ticket, run "
                "plate_pickup_recovery_home.py, then rerun fresh recorded no-motion dish preview."
            )
            return result

        physical_ready = _select_physical_ready_strategy(run_dir, planner_result, plan_validation["selected_strategy"])
        result["physical_ready_selection"] = physical_ready
        result["physical_sequence_preview"] = physical_ready.get("sequence_preview")
        if not physical_ready.get("success"):
            result["success"] = False
            result["why_stopped"] = "physical-sequence no-motion preview failed with the intended gripper widths"
            result["next_recommendation"] = (
                "Do not execute physical pickup; tune gripper pregrasp width, approach height, or candidate geometry in no-motion mode."
            )
            return result
        plan_validation["selected_strategy"] = physical_ready["selected_strategy"]
        plan_validation["selected_label"] = physical_ready.get("selected_label")
        plan_validation["arm"] = (physical_ready.get("selected_strategy") or {}).get("arm")

        if not result["physical_gate"]:
            result["success"] = True
            result["why_stopped"] = "recorded no-motion validation succeeded; physical gate was not set"
            result["next_recommendation"] = "For physical mode, rerun once under the active ticket with OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
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

        attempt = _execute_physical(run_dir, plan_validation["selected_strategy"])
        result["physical_attempt"] = attempt
        result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
        result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
        result["success"] = bool(attempt.get("success"))
        result["why_stopped"] = attempt.get("why_stopped")
        result["next_recommendation"] = (
            "Do not retry under this ticket. If the dish/plate is held or the arm is not home, request a bounded return/release or recovery ticket."
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
        write_json(run_dir / "physical_wrapper_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## physical wrapper result",
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
