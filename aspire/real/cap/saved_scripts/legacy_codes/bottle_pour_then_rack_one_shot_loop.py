# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the bottle dry-pour rehearsal, then the bottle rack placement.

This is a thin composition entrypoint for one ``run_script.py`` invocation.
It reuses the validated bottle subtasks instead of duplicating their motion
logic, but writes a single combined result for the bottle sequence.
"""

from __future__ import annotations

import json
import os
import runpy
import traceback
from pathlib import Path
from typing import Any

from cap.saved_scripts.yam_runtime import append_stage_summary, current_run_dir, json_safe, write_json


TASK_NAME = "bottle_pour_then_rack_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}


_THIS_FILE_RAW = globals().get("__file__")
ROOT = Path(_THIS_FILE_RAW).resolve().parents[2] if _THIS_FILE_RAW else Path.cwd().resolve()
DRY_POUR_SCRIPT = ROOT / "cap" / "saved_scripts" / "dry_bottle_to_cup_pour_one_shot_loop.py"
BOTTLE_RACK_SCRIPT = ROOT / "cap" / "saved_scripts" / "grasp_lift_place_bottle_rack_one_shot_loop.py"

BOTTLE_CHILD_DEFAULT_ENV: dict[str, str] = {
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_PHYSICAL_TICKET": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_OPERATOR_PRESENT": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_ESTOP_REACHABLE": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_WORKSPACE_CLEAR": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_HANDS_TOOLS_CABLES_CLEAR": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_FOLLOWER_SERVERS_UP": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_RECORDING_DEBUG_CONFIRMED": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_DRY_EMPTY_OR_SEALED_BOTTLE": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_EMPTY_CUP": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_NO_LIQUID": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_NO_MANUAL_XYZ": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_NO_OTHER_MOTION_PRECHECK": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_TICKET_SCOPE": "dry_bottle_to_cup_pour",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_PHYSICAL_ATTEMPT_BUDGET": "20",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_ALLOWED_ATTEMPT_BUDGET": "20",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_ARM": "left",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_BOTTLE_PROMPTS": "bottle,wine bottle",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_PROMPTS": "glass,water glass,cup,plastic cup,empty cup",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_MIN_DETECTION_SCORE": "0.25",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_ENFORCE_DETECTION_SANITY": "0",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_ENFORCE_BOTTLE_CUP_SEPARATION": "0",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_TOP_CAMERA": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_PREGRASP_CLEARANCE_M": "0.100",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_PREGRASP_XY_BIAS_M": "0.000,0.005",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_LIVE_DETECTION_Z_TO_BODY_OFFSET_M": "-0.08",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_FIXED_GRASP_Z_M": "0.87414",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIDE_GRASP_BACKOFF_M": "-0.020",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_LIFT_Z_M": "0.24",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_TCP_OFFSET_FROM_BOTTLE_M": "-0.07760,0.05267,0.23899",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_MIN_Z_ABOVE_LIFT_M": "0.006",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_YAW_OFFSET_DEG": "-2.0",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_APPROACH_Z_LIFT_M": "0.00",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SIMPLE_TILT_PITCH_SEQUENCE_DEG": "90,75,60,45,40",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_TILT_HOLD_S": "0.75",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MIN_Z_DELTA_M": "0.025",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_SUCCESS_MAX_XY_DELTA_M": "0.12",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_POST_LIFT_VERIFY": "0",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNING_SPEED": "0.40",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_PLANNER_BACKEND": "rrtconnect",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ERROR_THRESHOLD_M": "0.015",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_IK_ROT_THRESHOLD_DEG": "8.0",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_TARGET_POS": "0.35",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_STEPS_POS": "0.78,0.58,0.35",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_VEL_LIMIT": "0.60",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_CLOSE_TORQUE_LIMIT": "0.80",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_REQUIRE_CLOSE_CONTACT": "1",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_VEL_LIMIT": "0.50",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_OPEN_TORQUE_LIMIT": "0.30",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_CUP_RIM_Z_OFFSET_M": "0.045",
    "OPENFORGE_BOTTLE_CUP_DRY_POUR_MAX_POUR_CANDIDATES": "24",
    "OPENFORGE_BOTTLE_RACK_TARGET_PROMPTS": "top shelf,upper shelf,wooden rack,rack,wooden shelf,shelf",
    "OPENFORGE_BOTTLE_RACK_TARGET_Y_SIGN": "positive",
    "OPENFORGE_BOTTLE_RACK_ATTEMPT_BUDGET": "1",
    "OPENFORGE_BOTTLE_RACK_HIGH_PLACE_Z_LIFT_M": "0.08",
    "OPENFORGE_BOTTLE_RACK_RETREAT_Z_LIFT_M": "0.0",
    "OPENFORGE_BOTTLE_RACK_RETREAT_STANDOFF_M": "0.08",
    "OPENFORGE_BOTTLE_RACK_RETREAT_SHIFT_Y_M": "0.0",
    "OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_MIN_POS": "0.95",
    "OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_TIMEOUT_S": "12.0",
    "OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_POLL_S": "0.25",
    "OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_RETRY_S": "1.5",
    "OPENFORGE_BOTTLE_RACK_GO_HOME_AFTER_PLACE": "1",
    "OPENFORGE_BOTTLE_RACK_OPEN_AFTER_HOME": "1",
    "OPENFORGE_BOTTLE_RACK_ENABLE_TRANSPORT_WAYPOINTS": "1",
    "OPENFORGE_BOTTLE_RACK_TRANSPORT_RETREAT_X_M": "-0.20",
    "OPENFORGE_BOTTLE_RACK_TRANSPORT_SHIFT_Y_M": "-0.28",
    "OPENFORGE_BOTTLE_RACK_TRANSPORT_LIFT_Z_M": "0.10",
    "OPENFORGE_BOTTLE_RACK_PLACE_OFFSETS_M": "0,0,0;0,0.02,0;0.02,0,0;-0.02,0,0",
}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _apply_child_default_env() -> None:
    for name, value in BOTTLE_CHILD_DEFAULT_ENV.items():
        os.environ.setdefault(name, value)


def _video_paths(run_dir: Path) -> list[str]:
    return [str(path) for path in (run_dir / name for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4")) if path.exists()]


def _result_success(result: dict[str, Any]) -> bool:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    return bool(result.get("success")) or bool(details.get("success"))


def _result_why(result: dict[str, Any]) -> str | None:
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    value = result.get("why_stopped") or details.get("why_stopped") or result.get("feedback")
    return str(value) if value is not None else None


def _run_child(
    label: str,
    script_path: Path,
    run_dir: Path,
    *,
    run_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "label": label,
        "script_path": str(script_path),
        "success": False,
        "result": None,
        "result_path": None,
        "why_stopped": None,
        "error": None,
    }
    append_stage_summary(run_dir, [f"## combined bottle subtask: {label}", f"- script: {script_path}"])

    try:
        child_globals = dict(globals())
        child_globals["__file__"] = str(script_path)
        child_globals["__name__"] = f"__openforge_{label}__"
        if run_kwargs is not None:
            child_globals["_OPENFORGE_IMPORT_ONLY"] = True
        executed = runpy.run_path(
            str(script_path),
            init_globals=child_globals,
            run_name=f"__openforge_{label}__",
        )
        if run_kwargs is not None:
            run_fn = executed.get("_run")
            if not callable(run_fn):
                raise RuntimeError(f"{script_path} does not expose _run")
            child_result = run_fn(**run_kwargs)
        else:
            child_result = executed.get("TASK_RESULT")
            if not isinstance(child_result, dict):
                get_info = executed.get("get_task_info")
                child_result = get_info() if callable(get_info) else {}
        packet["result"] = json_safe(child_result)
        packet["success"] = _result_success(child_result)
        packet["why_stopped"] = _result_why(child_result)
    except Exception as exc:
        packet["error"] = f"{type(exc).__name__}: {exc}"
        packet["traceback"] = traceback.format_exc()
        packet["why_stopped"] = packet["error"]

    packet["result_path"] = write_json(run_dir / "subtasks" / f"{label}_result.json", packet)
    append_stage_summary(
        run_dir,
        [
            f"## combined bottle subtask result: {label}",
            f"- success: {packet['success']}",
            f"- why_stopped: {packet['why_stopped']}",
            f"- result_path: {packet['result_path']}",
        ],
    )
    return packet


def _extract_pour_pickup(pour_packet: dict[str, Any]) -> dict[str, Any] | None:
    result = pour_packet.get("result") if isinstance(pour_packet.get("result"), dict) else {}
    strategy = result.get("selected_strategy") if isinstance(result.get("selected_strategy"), dict) else {}
    pickup = strategy.get("bottle_pickup")
    if isinstance(pickup, dict):
        return pickup
    physical = result.get("physical_attempt") if isinstance(result.get("physical_attempt"), dict) else {}
    pickup = physical.get("selected_pickup")
    return pickup if isinstance(pickup, dict) else None


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.bottle_pour_then_rack.result.v1",
        "success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False),
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "subtasks": [],
        "attempt_count": 0,
        "why_stopped": "not_started",
        "next_recommendation": None,
        "video_paths": [],
    }


def _collect_child_motion(result: dict[str, Any], child_packet: dict[str, Any]) -> None:
    child_result = child_packet.get("result") if isinstance(child_packet.get("result"), dict) else {}
    result["physical_motion_executed"] = bool(
        result["physical_motion_executed"] or child_result.get("physical_motion_executed")
    )
    movement_calls = child_result.get("movement_capable_calls")
    if isinstance(movement_calls, list):
        result["movement_capable_calls"].extend(movement_calls)
    physical_attempt = child_result.get("physical_attempt")
    if isinstance(physical_attempt, dict):
        movement_calls = physical_attempt.get("movement_capable_calls")
        if isinstance(movement_calls, list):
            result["movement_capable_calls"].extend(movement_calls)


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        _apply_child_default_env()
        if not DRY_POUR_SCRIPT.is_file():
            result["why_stopped"] = f"missing dry-pour script: {DRY_POUR_SCRIPT}"
            return result
        if not BOTTLE_RACK_SCRIPT.is_file():
            result["why_stopped"] = f"missing bottle-rack script: {BOTTLE_RACK_SCRIPT}"
            return result

        pour_packet = _run_child(
            "dry_pour",
            DRY_POUR_SCRIPT,
            run_dir,
            run_kwargs={"return_to_source_after_pour": False, "simple_fixed_pour": True},
        )
        result["subtasks"].append(pour_packet)
        _collect_child_motion(result, pour_packet)
        if not pour_packet.get("success"):
            result["why_stopped"] = f"dry-pour subtask failed; stopped before rack: {pour_packet.get('why_stopped')}"
            result["next_recommendation"] = "Inspect subtasks/dry_pour_result.json, stage_summary.md, and videos before retry."
            return result

        held_pickup = _extract_pour_pickup(pour_packet)
        if held_pickup is None:
            result["why_stopped"] = "dry-pour result did not include the held bottle pickup plan"
            result["next_recommendation"] = "Inspect subtasks/dry_pour_result.json before retry."
            return result
        rack_packet = _run_child(
            "bottle_rack",
            BOTTLE_RACK_SCRIPT,
            run_dir,
            run_kwargs={"held_pickup": held_pickup},
        )
        result["subtasks"].append(rack_packet)
        _collect_child_motion(result, rack_packet)
        if not rack_packet.get("success"):
            result["why_stopped"] = f"bottle-rack subtask failed: {rack_packet.get('why_stopped')}"
            result["next_recommendation"] = "Inspect subtasks/bottle_rack_result.json, stage_summary.md, and videos before retry."
            return result

        result["success"] = True
        result["reward"] = 1.0
        result["attempt_count"] = len([item for item in result["subtasks"] if not item.get("skipped")])
        result["why_stopped"] = "dry bottle pour rehearsal and bottle rack placement completed"
        result["next_recommendation"] = "Review result.json, stage_summary.md, and the single bottle run videos."
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["error"] = result["why_stopped"]
        result["traceback"] = traceback.format_exc()
        result["next_recommendation"] = "Inspect result.json and run log before retrying."
    finally:
        result["video_paths"] = _video_paths(run_dir)
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## combined bottle result",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- video_paths: {result['video_paths']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))
    return result


TASK_RESULT.update(_run())
