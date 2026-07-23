# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place a currently held bottle onto the rack and retreat."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from cap.saved_scripts.yam_runtime import append_stage_summary, current_run_dir, json_safe, write_json


TASK_NAME = "recover_held_bottle_to_rack"
TASK_RESULT: dict[str, Any] = {}


def _floats(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            pass
    return values or list(default)


def _tool(name: str):
    value = globals().get(name)
    if not callable(value):
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _move(label: str, pos: list[float], rpy: list[float]) -> dict[str, Any]:
    freespace_move = _tool("freespace_move")
    kwargs = {
        "left_target_pos": [float(value) for value in pos[:3]],
        "left_target_rpy": [float(value) for value in rpy[:3]],
        "planner_backend": os.environ.get("OPENFORGE_BOTTLE_RACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_BOTTLE_RACK_SOLVER_SPEED", "slow"),
        "planning_speed": float(os.environ.get("OPENFORGE_BOTTLE_RACK_PLANNING_SPEED", "0.40")),
        "preview_only": True,
    }
    preview = freespace_move(**kwargs)
    safe_preview = json_safe(preview)
    cache_key = safe_preview.get("trajectory_cache_key") if isinstance(safe_preview, dict) else None
    if not cache_key:
        return {"stage": label, "ok": False, "pose": {"position": pos, "rpy": rpy}, "preview": safe_preview}
    executed = freespace_move(trajectory_cache_key=cache_key)
    return {
        "stage": label,
        "ok": True,
        "pose": {"position": pos, "rpy": rpy},
        "preview": safe_preview,
        "execute": json_safe(executed),
    }


def _gripper_pos() -> float | None:
    get_robot_state = globals().get("get_robot_state")
    if not callable(get_robot_state):
        return None
    state = json_safe(get_robot_state())
    try:
        return float(state["arms"]["left"]["gripper_pos"])
    except (KeyError, TypeError, ValueError):
        return None


def _open_until() -> dict[str, Any]:
    set_gripper = _tool("set_gripper")
    min_pos = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_MIN_POS", "0.95"))
    timeout_s = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_TIMEOUT_S", "12.0"))
    poll_s = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_POLL_S", "0.25"))
    retry_s = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_RETRY_S", "1.5"))
    target = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_TARGET_POS", "1.0"))
    vel = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_VEL_LIMIT", "0.50"))
    torque = float(os.environ.get("OPENFORGE_BOTTLE_RACK_OPEN_TORQUE_LIMIT", "0.30"))
    started = time.time()
    last_command = 0.0
    samples = []
    while time.time() - started <= timeout_s:
        now = time.time()
        if now - last_command >= retry_s:
            samples.append({"command": json_safe(set_gripper("left", target, vel_limit=vel, torque_limit=torque))})
            last_command = now
        pos = _gripper_pos()
        samples.append({"t_s": round(now - started, 3), "gripper_pos": pos})
        if pos is not None and pos >= min_pos:
            return {"success": True, "samples": samples, "final_pos": pos}
        time.sleep(poll_s)
    return {"success": False, "samples": samples, "final_pos": _gripper_pos(), "why_stopped": "gripper did not confirm fully open"}


def main() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result: dict[str, Any] = {
        "schema": "openforge.bottle_rack.held_recovery.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": "not_started",
    }
    high = _floats("OPENFORGE_BOTTLE_RACK_RECOVERY_HIGH_POS", [0.88916, 0.038, 1.12257])[:3]
    release = _floats("OPENFORGE_BOTTLE_RACK_RECOVERY_RELEASE_POS", [0.88916, 0.038, 1.04257])[:3]
    rpy = _floats("OPENFORGE_BOTTLE_RACK_RECOVERY_RPY", [0.0, 90.0, 0.0])[:3]
    retreat = _floats(
        "OPENFORGE_BOTTLE_RACK_RECOVERY_RETREAT_POS",
        [release[0] - 0.14, release[1], release[2]],
    )[:3]
    try:
        for label, pos in (("rack_high_place", high), ("rack_release_pose", release)):
            step = _move(label, pos, rpy)
            result["steps"].append(step)
            result["movement_capable_calls"].append(f"freespace_move:{label}")
            result["physical_motion_executed"] = True
            if not step.get("ok"):
                result["why_stopped"] = f"{label} preview/execute failed"
                return result
        opened = _open_until()
        result["steps"].append({"stage": "open_release_on_rack", "ok": bool(opened.get("success")), "result": opened})
        result["movement_capable_calls"].append("set_gripper:left")
        if not opened.get("success"):
            result["why_stopped"] = str(opened.get("why_stopped") or "open failed")
            return result
        step = _move("rack_retreat_back_x", retreat, rpy)
        result["steps"].append(step)
        result["movement_capable_calls"].append("freespace_move:rack_retreat_back_x")
        if not step.get("ok"):
            result["why_stopped"] = "rack retreat failed"
            return result
        home = _tool("go_home")()
        result["steps"].append({"stage": "go_home_after_recovery", "ok": True, "result": json_safe(home)})
        result["movement_capable_calls"].append("go_home")
        result["success"] = True
        result["why_stopped"] = "held bottle placed on rack, gripper opened, retreated, and homed"
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## held bottle rack recovery",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- why_stopped: {result['why_stopped']}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))


if not globals().get("_OPENFORGE_IMPORT_ONLY", False):
    TASK_RESULT.update(main())
