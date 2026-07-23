# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded motion helpers for YAM saved scripts."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

from .artifacts import append_stage_summary, current_run_dir, write_json
from .gripper import read_gripper_width, staged_close_with_contact


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _move_to_pose(
    freespace_move: Callable[..., Any],
    arm: str,
    pose: dict[str, Any],
    *,
    preview_only: bool = False,
    **kwargs: Any,
) -> Any:
    prefix = "left" if arm == "left" else "right"
    params = {
        f"{prefix}_target_pos": pose["position"],
        f"{prefix}_target_rpy": pose["rpy"],
        "preview_only": preview_only,
    }
    params.update(kwargs)
    return freespace_move(**params)


def _preview_then_execute_pose(
    freespace_move: Callable[..., Any],
    arm: str,
    pose: dict[str, Any],
    *,
    label: str,
    **kwargs: Any,
) -> dict[str, Any]:
    preview = _move_to_pose(freespace_move, arm, pose, preview_only=True, **kwargs)
    cache_key = _field(preview, "trajectory_cache_key")
    if isinstance(preview, dict):
        cache_key = preview.get("trajectory_cache_key", cache_key)
    if not cache_key:
        raise RuntimeError(f"{label} preview did not return a trajectory_cache_key: {preview!r}")
    executed = freespace_move(trajectory_cache_key=cache_key)
    return {
        "preview": preview,
        "execute": executed,
        "trajectory_cache_key": cache_key,
    }


def execute_grasp_lift_attempt(
    *,
    plan: dict[str, Any],
    freespace_move: Callable[..., Any],
    open_gripper: Callable[..., Any],
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    physical_gate: bool,
    run_dir: str | Path | None = None,
    task_name: str = "yam_runtime",
    planning_speed: float = 0.20,
    planner_backend: str = "curobo",
    solver_speed: str = "fast",
    gripper_target: float = 0.0,
    gripper_steps: list[float] | tuple[float, ...] | None = None,
    gripper_vel_limit: float | None = 0.08,
    gripper_torque_limit: float | None = 0.35,
    gripper_min_contact_delta: float = 0.02,
    gripper_target_tolerance: float = 0.04,
    gripper_hold_min: float = 0.025,
    gripper_hold_max: float = 0.95,
    require_contact_for_lift: bool = True,
    open_confirm_min: float = 0.55,
    open_confirm_timeout_s: float = 2.0,
    state_poll_s: float = 0.1,
    ik_error_threshold: float | None = None,
    ik_rot_threshold_deg: float | None = None,
    ik_xyz_weight: float | None = None,
    ik_rpy_weight: float | None = None,
    motion_gripper_pos: float | None = 1.0,
    preview_before_each_move: bool = True,
    execute_lift: bool = True,
) -> dict[str, Any]:
    """Execute open -> pregrasp -> grasp -> close -> lift for one selected plan."""
    run_dir = Path(run_dir) if run_dir is not None else current_run_dir(task_name)
    if not physical_gate:
        packet = {
            "success": False,
            "physical_motion_executed": False,
            "why_stopped": "physical gate is false; no motion executed",
        }
        packet["path"] = write_json(run_dir / "plans" / "execute_refused.json", packet)
        return packet
    arm = plan["arm"]
    base_move_kwargs = {
        "planner_backend": planner_backend,
        "solver_speed": solver_speed,
        "planning_speed": planning_speed,
    }
    if ik_error_threshold is not None:
        base_move_kwargs["ik_error_threshold"] = ik_error_threshold
    if ik_rot_threshold_deg is not None:
        base_move_kwargs["ik_rot_threshold_deg"] = ik_rot_threshold_deg
    if ik_xyz_weight is not None:
        base_move_kwargs["ik_xyz_weight"] = ik_xyz_weight
    if ik_rpy_weight is not None:
        base_move_kwargs["ik_rpy_weight"] = ik_rpy_weight
    open_motion_kwargs = dict(base_move_kwargs)
    if motion_gripper_pos is not None:
        open_motion_kwargs[f"{arm}_gripper"] = motion_gripper_pos
    packet: dict[str, Any] = {
        "schema": "openforge.yam_runtime.execute_grasp_lift.v1",
        "success": False,
        "physical_motion_executed": False,
        "arm": arm,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    try:
        open_before = read_gripper_width(get_robot_state, arm)
        open_result = open_gripper(arm)
        packet["physical_motion_executed"] = True
        packet["movement_capable_calls"].append(f"open_gripper:{arm}")
        open_after = read_gripper_width(get_robot_state, arm)
        deadline = time.monotonic() + max(0.0, float(open_confirm_timeout_s))
        while open_after is not None and open_after < float(open_confirm_min) and time.monotonic() < deadline:
            time.sleep(max(0.02, float(state_poll_s)))
            open_after = read_gripper_width(get_robot_state, arm)
        open_ok = open_after is None or open_after >= float(open_confirm_min)
        packet["steps"].append(
            {
                "stage": "open_gripper",
                "ok": open_ok,
                "before": open_before,
                "after": open_after,
                "open_confirm_min": float(open_confirm_min),
                "result": open_result,
            }
        )
        if not open_ok:
            packet["why_stopped"] = "gripper did not confirm open before arm motion"
            packet["path"] = write_json(run_dir / "plans" / "execute_grasp_lift_attempt.json", packet)
            return packet
        for key in ("pregrasp_pose", "grasp_pose"):
            if preview_before_each_move:
                result = _preview_then_execute_pose(
                    freespace_move,
                    arm,
                    plan[key],
                    label=key,
                    **open_motion_kwargs,
                )
            else:
                result = _move_to_pose(freespace_move, arm, plan[key], **open_motion_kwargs)
            packet["movement_capable_calls"].append("freespace_move")
            packet["steps"].append({"stage": key, "ok": True, "result": result})
        close = staged_close_with_contact(
            side=arm,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=gripper_target,
            steps=gripper_steps or (0.65, 0.45, 0.25, 0.0),
            vel_limit=gripper_vel_limit,
            torque_limit=gripper_torque_limit,
            min_contact_delta=gripper_min_contact_delta,
            target_tolerance=gripper_target_tolerance,
            hold_min=gripper_hold_min,
            hold_max=gripper_hold_max,
            run_dir=run_dir,
            task_name=task_name,
        )
        packet["movement_capable_calls"].append(f"set_gripper:{arm}")
        packet["gripper_after_close"] = close
        close_ok = bool(close.get("contact_detected")) if require_contact_for_lift else bool(
            close.get("plausible_for_lift")
        )
        packet["close_gate"] = {
            "require_contact_for_lift": bool(require_contact_for_lift),
            "contact_detected": bool(close.get("contact_detected")),
            "target_reached": bool(close.get("target_reached")),
            "plausible_for_lift": bool(close.get("plausible_for_lift")),
            "passed": close_ok,
        }
        if not close_ok:
            if require_contact_for_lift:
                packet["why_stopped"] = "gripper contact evidence is not plausible; stop before lift"
            else:
                packet["why_stopped"] = "gripper close did not reach a plausible hold state; stop before lift"
        elif not execute_lift:
            packet["success"] = True
            packet["why_stopped"] = "grasp close completed; lift disabled by grasp-only gate"
        else:
            if preview_before_each_move:
                result = _preview_then_execute_pose(
                    freespace_move,
                    arm,
                    plan["lift_pose"],
                    label="lift_pose",
                    **base_move_kwargs,
                )
            else:
                result = _move_to_pose(freespace_move, arm, plan["lift_pose"], **base_move_kwargs)
            packet["movement_capable_calls"].append("freespace_move")
            packet["steps"].append(
                {
                    "stage": "lift_pose",
                    "ok": True,
                    "result": result,
                    "gripper_command": "preserve_current_after_close",
                }
            )
            packet["success"] = True
            packet["why_stopped"] = "lift command completed; verify with post-action observation"
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
        packet["steps"].append({"stage": "exception", "ok": False, "error": packet["why_stopped"]})
    packet["path"] = write_json(run_dir / "plans" / "execute_grasp_lift_attempt.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## execute grasp_lift",
            f"- success: {packet['success']}",
            f"- physical_motion_executed: {packet['physical_motion_executed']}",
            f"- why_stopped: {packet['why_stopped']}",
        ],
    )
    return packet
