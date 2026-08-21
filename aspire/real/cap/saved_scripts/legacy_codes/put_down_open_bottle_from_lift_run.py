# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safely put down and release a bottle after a bounded lift attempt.

This recovery script is intentionally narrow. It reads the selected grasp pose
from a previous ``grasp_lift_bottle_one_shot_loop`` run, verifies that the left
arm is still near that pose family, previews a return-to-grasp move, then opens
the gripper in slow stages.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    read_gripper_width,
    write_json,
)


TASK_NAME = "put_down_open_bottle_from_lift_run"
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


def _latest_bottle_run() -> Path | None:
    runs = sorted(glob.glob("logs/grasp_lift_bottle_one_shot_loop_*"))
    return Path(runs[-1]) if runs else None


def _source_run() -> Path:
    raw = os.environ.get("OPENFORGE_BOTTLE_RECOVERY_SOURCE_RUN", "").strip()
    path = Path(raw).expanduser() if raw else _latest_bottle_run()
    if path is None:
        raise RuntimeError("no source bottle lift run found")
    if not (path / "result.json").exists():
        raise RuntimeError(f"source run has no result.json: {path}")
    return path


def _load_source_result(path: Path) -> dict[str, Any]:
    raw = json.loads((path / "result.json").read_text(encoding="utf-8"))
    return raw.get("details", raw)


def _arm_state(state: Any, side: str) -> Any:
    if isinstance(state, dict):
        return (state.get("arms") or {}).get(side) or {}
    arms = getattr(state, "arms", None)
    if isinstance(arms, dict):
        return arms.get(side)
    return None


def _state_value(state: Any, side: str, name: str) -> Any:
    direct = getattr(state, f"{side}_{name}", None)
    if direct is not None:
        return direct
    arm = _arm_state(state, side)
    if isinstance(arm, dict):
        return arm.get(name)
    return getattr(arm, name, None)


def _xyz(value: Any) -> list[float] | None:
    try:
        vals = [float(v) for v in list(value)[:3]]
    except Exception:
        return None
    return vals if len(vals) == 3 else None


def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.dist(a, b))


def _move_to_pose(
    *,
    freespace_move: Callable[..., Any],
    side: str,
    pose: dict[str, Any],
    preview_only: bool,
) -> Any:
    prefix = "left" if side == "left" else "right"
    return freespace_move(
        **{
            f"{prefix}_target_pos": pose["position"],
            f"{prefix}_target_rpy": pose["rpy"],
            "preview_only": preview_only,
            "planner_backend": os.environ.get("OPENFORGE_GRASP_LIFT_PLANNER_BACKEND", "rrtconnect"),
            "solver_speed": os.environ.get("OPENFORGE_GRASP_LIFT_SOLVER_SPEED", "slow"),
            "planning_speed": _env_float("OPENFORGE_BOTTLE_RECOVERY_PLANNING_SPEED", 0.10),
        }
    )


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    allow_physical = _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False)
    plan_only = (not allow_physical) or _truthy_env("OPENFORGE_BOTTLE_RECOVERY_PLAN_ONLY", False)
    source = _source_run()
    source_result = _load_source_result(source)
    selected = source_result.get("selected_strategy") or {}
    side = str(selected.get("arm") or os.environ.get("OPENFORGE_GRASP_LIFT_ARM", "left")).lower()
    if side != "left":
        raise RuntimeError(f"refusing non-left recovery side: {side!r}")
    target_pose_key = os.environ.get("OPENFORGE_BOTTLE_RECOVERY_TARGET_POSE", "grasp_pose").strip()
    if target_pose_key not in {"pregrasp_pose", "grasp_pose", "lift_pose"}:
        raise RuntimeError(f"unsupported OPENFORGE_BOTTLE_RECOVERY_TARGET_POSE={target_pose_key!r}")
    grasp_pose = selected.get("grasp_pose")
    lift_pose = selected.get("lift_pose")
    target_pose = selected.get(target_pose_key)
    if not isinstance(grasp_pose, dict) or "position" not in grasp_pose or "rpy" not in grasp_pose:
        raise RuntimeError("source run does not contain a usable selected grasp_pose")
    if not isinstance(target_pose, dict) or "position" not in target_pose or "rpy" not in target_pose:
        raise RuntimeError(f"source run does not contain a usable selected {target_pose_key}")

    get_robot_state = _required_tool("get_robot_state")
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    state_before = get_robot_state()
    current_ee = _xyz(_state_value(state_before, side, "ee_pos"))
    grasp_xyz = _xyz(grasp_pose.get("position"))
    lift_xyz = _xyz((lift_pose or {}).get("position"))
    dist_to_grasp = _distance(current_ee, grasp_xyz)
    dist_to_lift = _distance(current_ee, lift_xyz)
    max_start_dist = _env_float("OPENFORGE_BOTTLE_RECOVERY_MAX_START_DIST_M", 0.12)
    near_pose = any(
        dist is not None and dist <= max_start_dist
        for dist in (dist_to_grasp, dist_to_lift)
    )
    result: dict[str, Any] = {
        "success": False,
        "confidence": 0.0,
        "method": TASK_NAME,
        "source_run": str(source),
        "allow_physical": allow_physical,
        "plan_only": plan_only,
        "side": side,
        "selected_grasp_pose": grasp_pose,
        "selected_lift_pose": lift_pose,
        "target_pose_key": target_pose_key,
        "target_pose": target_pose,
        "state_before": json_safe(state_before),
        "current_ee": current_ee,
        "dist_to_grasp_m": dist_to_grasp,
        "dist_to_lift_m": dist_to_lift,
        "max_start_dist_m": max_start_dist,
        "gripper_before": read_gripper_width(get_robot_state, side),
        "pre_observation": None,
        "post_observation": None,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    try:
        pre_obs = capture_scene(
            prompts=["bottle"],
            cameras=["top"],
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=get_robot_state,
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="pre_put_down_observe",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
            max_retries=1,
            save_overlays=True,
        )
        result["pre_observation"] = pre_obs
        if not near_pose:
            result["why_stopped"] = "current left EE is not near source run grasp/lift pose; refusing recovery move"
            return result

        preview = _move_to_pose(freespace_move=freespace_move, side=side, pose=target_pose, preview_only=True)
        result["steps"].append({"stage": f"preview_return_to_{target_pose_key}", "result": json_safe(preview)})
        cache_key = _field(preview, "trajectory_cache_key")
        if not cache_key:
            result["why_stopped"] = f"return-to-grasp preview did not provide trajectory cache: {preview!r}"
            return result
        if plan_only:
            result["success"] = True
            result["confidence"] = 0.5
            result["why_stopped"] = "plan-only recovery preview succeeded; no motion executed"
            return result

        executed = freespace_move(trajectory_cache_key=cache_key)
        result["movement_capable_calls"].append("freespace_move")
        result["steps"].append({"stage": f"execute_return_to_{target_pose_key}", "result": json_safe(executed)})
        for target in _float_csv("OPENFORGE_BOTTLE_RECOVERY_OPEN_STEPS", [0.72, 0.90, 1.0]):
            step_target = max(0.0, min(1.0, float(target)))
            open_result = set_gripper(
                side,
                step_target,
                vel_limit=_env_float("OPENFORGE_BOTTLE_RECOVERY_OPEN_VEL_LIMIT", 0.25),
                torque_limit=_env_float("OPENFORGE_BOTTLE_RECOVERY_OPEN_TORQUE_LIMIT", 0.30),
            )
            time.sleep(max(0.0, _env_float("OPENFORGE_BOTTLE_RECOVERY_OPEN_STEP_PAUSE_S", 0.35)))
            width = read_gripper_width(get_robot_state, side)
            result["movement_capable_calls"].append(f"set_gripper:{side}")
            result["steps"].append(
                {
                    "stage": "open_gripper_step",
                    "target": step_target,
                    "after": width,
                    "result": json_safe(open_result),
                }
            )

        post_obs = capture_scene(
            prompts=["bottle"],
            cameras=["top"],
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=get_robot_state,
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage="post_put_down_observe",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
            max_retries=1,
            save_overlays=True,
        )
        result["post_observation"] = post_obs
        result["state_after"] = json_safe(get_robot_state())
        result["gripper_after"] = read_gripper_width(get_robot_state, side)
        result["success"] = True
        result["confidence"] = 0.7
        result["why_stopped"] = f"returned to source {target_pose_key} and opened left gripper in slow stages"
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["error"] = result["why_stopped"]
        return result
    finally:
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- plan_only: {result['plan_only']}",
                f"- physical_motion_executed: {bool(result['movement_capable_calls'])}",
                f"- why_stopped: {result['why_stopped']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
