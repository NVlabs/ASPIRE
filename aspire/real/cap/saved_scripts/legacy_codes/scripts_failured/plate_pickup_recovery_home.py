"""Guarded recovery/home helper for the YAM plate-pickup task.

Default mode is no-motion validation: it records current state and refuses to
move.  Physical recovery requires both OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and a
fresh plate-recovery ticket expressed through explicit env acknowledgements.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    current_run_dir,
    json_safe,
    read_gripper_width,
    write_json,
)


TASK_NAME = "plate_pickup_recovery_home"
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


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _video_paths(run_dir: Path) -> list[str]:
    return [
        str(path)
        for path in (run_dir / name for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"))
        if path.exists() and path.stat().st_size > 0
    ]


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_PLATE_RECOVERY_RECORDING_WAIT_S", 4.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    evidence: dict[str, Any] = {"required": True, "passed": bool(videos), "video_paths": videos}
    if not videos:
        evidence.update(
            {
                "problem": "no mp4 files found; recovery/home physical motion must have recording/debug evidence",
                "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
                "blocker_before_physical": True,
            }
        )
    return evidence


def _required_ticket_env() -> dict[str, Any]:
    required_truthy = [
        "OPENFORGE_PLATE_RECOVERY_TICKET",
        "OPENFORGE_PLATE_OPERATOR_PRESENT",
        "OPENFORGE_PLATE_ESTOP_REACHABLE",
        "OPENFORGE_PLATE_WORKSPACE_CLEAR",
        "OPENFORGE_PLATE_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_PLATE_NO_LIQUID",
        "OPENFORGE_PLATE_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_PLATE_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_truthy if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_PLATE_RECOVERY_SCOPE", "").strip()
    if scope not in {"plate_pickup_recovery", "plate_pickup_home"}:
        missing.append("OPENFORGE_PLATE_RECOVERY_SCOPE=plate_pickup_recovery")
    return {
        "required_truthy_env": required_truthy,
        "required_scope_values": ["plate_pickup_recovery", "plate_pickup_home"],
        "missing_or_invalid": missing,
        "passed": not missing,
    }


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "before_state": None,
        "after_state": None,
        "ticket_check": None,
        "recording_evidence": None,
        "video_paths": [],
        "why_stopped": "not_started",
        "next_recommendation": None,
    }


def _set_gripper_open_checked(
    *,
    side: str,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any] | None,
) -> dict[str, Any]:
    target = _env_float("OPENFORGE_PLATE_RECOVERY_OPEN_TARGET", 0.68)
    confirm_min = _env_float("OPENFORGE_PLATE_RECOVERY_OPEN_CONFIRM_POS", 0.55)
    before = read_gripper_width(get_robot_state, side) if callable(get_robot_state) else None
    result = set_gripper(
        side,
        target,
        vel_limit=_env_float("OPENFORGE_PLATE_RECOVERY_OPEN_VEL_LIMIT", 0.30),
        torque_limit=_env_float("OPENFORGE_PLATE_RECOVERY_OPEN_TORQUE_LIMIT", 0.18),
    )
    time.sleep(_env_float("OPENFORGE_PLATE_RECOVERY_GRIPPER_SETTLE_S", 0.25))
    after = read_gripper_width(get_robot_state, side) if callable(get_robot_state) else None
    deadline = time.monotonic() + _env_float("OPENFORGE_PLATE_RECOVERY_OPEN_CONFIRM_TIMEOUT_S", 3.0)
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


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    get_robot_state = _tool("get_robot_state")
    if callable(get_robot_state):
        try:
            result["before_state"] = json_safe(get_robot_state())
        except Exception as exc:
            result["before_state"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        ticket = _required_ticket_env()
        result["ticket_check"] = ticket
        physical_gate = _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION")
        recording = _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        if not physical_gate:
            result["success"] = True
            result["why_stopped"] = "validated no-motion refusal: OPENFORGE_ALLOW_PHYSICAL_MOTION is not set"
            result["next_recommendation"] = (
                "For actual recovery/home, obtain a fresh bounded physical ticket and set the required "
                "OPENFORGE_PLATE_RECOVERY_* env acknowledgements plus OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
            )
            return result
        if not ticket.get("passed"):
            result["why_stopped"] = f"refusing physical recovery; ticket env missing: {ticket['missing_or_invalid']}"
            result["next_recommendation"] = "Provide a complete bounded plate recovery/home ticket before motion."
            return result
        if not recording.get("passed"):
            result["why_stopped"] = "refusing physical recovery; recording evidence is missing"
            result["next_recommendation"] = "Fix recording/debug UI before any recovery/home motion."
            return result

        go_home = _tool("go_home")
        if not callable(go_home):
            result["why_stopped"] = "required go_home tool unavailable"
            return result
        if _truthy_env("OPENFORGE_PLATE_RECOVERY_OPEN_GRIPPERS", False):
            set_gripper = _tool("set_gripper")
            if not callable(set_gripper):
                result["why_stopped"] = "OPENFORGE_PLATE_RECOVERY_OPEN_GRIPPERS requested but set_gripper unavailable"
                return result
            result["open_gripper_steps"] = []
            for side in ("left", "right"):
                result["movement_capable_calls"].append(f"set_gripper:{side}")
                step = _set_gripper_open_checked(side=side, set_gripper=set_gripper, get_robot_state=get_robot_state)
                result["open_gripper_steps"].append(step)
                result["physical_motion_executed"] = True
                if not step["ok"]:
                    result["why_stopped"] = f"{side} gripper did not confirm open during recovery"
                    return result
        result["movement_capable_calls"].append("go_home")
        result["go_home_result"] = json_safe(go_home())
        result["physical_motion_executed"] = True
        if callable(get_robot_state):
            result["after_state"] = json_safe(get_robot_state())
        result["success"] = True
        result["reward"] = 1.0
        result["why_stopped"] = "physical recovery/home completed; verify with fresh no-motion plate observe/plan"
        result["next_recommendation"] = "Rerun pick_plate_no_motion_plan.py with recording/debug evidence and fresh live detection."
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Continue no-motion analysis until a complete recovery/home ticket is available."
        return result
    finally:
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## plate recovery/home",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- ticket_check: {result['ticket_check']}",
                f"- recording_evidence: {result['recording_evidence']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )


TASK_RESULT.update(_run())
print(f"[{TASK_NAME}] success={TASK_RESULT.get('success')} why={TASK_RESULT.get('why_stopped')}")
