# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical wrapper for green refrigerator open + box place.

Default mode is recorded no-motion validation.  Physical execution requires
``OPENFORGE_ALLOW_PHYSICAL_MOTION=1`` plus explicit ticket/precondition env
acks; otherwise this script refuses before any movement-capable call.
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
    write_json,
)


TASK_NAME = "open_green_refrigerator_put_box_inside_physical"
PLANNER_MODULE = "cap.saved_scripts.open_green_refrigerator_put_box_inside"
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


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
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


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_REFRIGERATOR_XYZ",
        "OPENFORGE_REFRIGERATOR_HANDLE_XYZ",
        "OPENFORGE_REFRIGERATOR_DOOR_XYZ",
        "OPENFORGE_BOX_XYZ",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _video_paths(run_dir: Path) -> list[str]:
    return [str(path) for path in (run_dir / name for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4")) if path.exists()]


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_FRIDGE_RECORDING_WAIT_S", 4.0))
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


def _run_fresh_no_motion_planner() -> dict[str, Any]:
    """Run the existing planner with the physical env gate temporarily removed."""
    previous_gate = os.environ.pop("OPENFORGE_ALLOW_PHYSICAL_MOTION", None)
    try:
        spec = importlib.util.find_spec(PLANNER_MODULE)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"could not locate planner module {PLANNER_MODULE!r}")
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
        namespace = runpy.run_path(spec.origin, init_globals=tool_globals, run_name="__openforge_fridge_planner__")
        planner_result = dict(namespace.get("TASK_RESULT") or {})
    finally:
        if previous_gate is not None:
            os.environ["OPENFORGE_ALLOW_PHYSICAL_MOTION"] = previous_gate
    return planner_result


def _validate_selected_strategy(planner_result: dict[str, Any]) -> dict[str, Any]:
    selected = planner_result.get("selected_strategy") or {}
    door = selected.get("door_open") or {}
    box = selected.get("box_grasp") or {}
    place = selected.get("place_inside") or {}
    required = {
        "door.pregrasp_pose": (door.get("pregrasp_pose") or {}).get("position"),
        "door.handle_grasp_pose": (door.get("handle_grasp_pose") or {}).get("position"),
        "door.door_open_small_pose": (door.get("door_open_small_pose") or {}).get("position"),
        "box.pregrasp_pose": (box.get("pregrasp_pose") or {}).get("position"),
        "box.grasp_pose": (box.get("grasp_pose") or {}).get("position"),
        "box.lift_pose": (box.get("lift_pose") or {}).get("position"),
        "place.preplace_pose": (place.get("preplace_pose") or {}).get("position"),
        "place.place_pose": (place.get("place_pose") or {}).get("position"),
        "place.retreat_pose": (place.get("retreat_pose") or {}).get("position"),
    }
    missing = [name for name, value in required.items() if not value]
    passed = bool(planner_result.get("success") and planner_result.get("plan_success") and not missing)
    return {
        "passed": passed,
        "missing": missing,
        "door_label": door.get("label"),
        "door_arm": door.get("arm"),
        "box_label": box.get("label"),
        "box_arm": box.get("arm"),
        "place_label": place.get("label"),
        "place_arm": place.get("arm"),
        "selected_strategy": selected if passed else None,
    }


def _pgrep_competing_motion(run_dir: Path | None = None) -> dict[str, Any]:
    try:
        output = subprocess.check_output(["pgrep", "-af", "run_script.py|ffmpeg"], text=True)
    except subprocess.CalledProcessError:
        output = ""
    self_pid = os.getpid()
    rows = []
    competitors = []
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
        own_recording = run_dir is not None and str(run_dir) in cmd and "ffmpeg" in cmd
        if pid != self_pid and "debug_ui" not in cmd and not own_recording:
            competitors.append(item)
    return {"self_pid": self_pid, "rows": rows, "competitors": competitors}


def _physical_ticket_status(run_dir: Path) -> dict[str, Any]:
    required_true = [
        "OPENFORGE_FRIDGE_PHYSICAL_TICKET",
        "OPENFORGE_FRIDGE_OPERATOR_PRESENT",
        "OPENFORGE_FRIDGE_ESTOP_REACHABLE",
        "OPENFORGE_FRIDGE_WORKSPACE_CLEAR",
        "OPENFORGE_FRIDGE_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_FRIDGE_NO_LIQUID",
        "OPENFORGE_FRIDGE_NO_MANUAL_XYZ",
        "OPENFORGE_FRIDGE_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_FRIDGE_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_true if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_FRIDGE_TICKET_SCOPE", "").strip()
    if scope and scope != "green_refrigerator_box":
        missing.append("OPENFORGE_FRIDGE_TICKET_SCOPE=green_refrigerator_box")
    elif not scope:
        missing.append("OPENFORGE_FRIDGE_TICKET_SCOPE=green_refrigerator_box")
    budget_raw = os.environ.get("OPENFORGE_FRIDGE_PHYSICAL_ATTEMPT_BUDGET", "").strip()
    try:
        budget = int(budget_raw) if budget_raw else 0
    except ValueError:
        budget = 0
    if budget < 1 or budget > 3:
        missing.append("OPENFORGE_FRIDGE_PHYSICAL_ATTEMPT_BUDGET in [1,3]")
    manual_env = _manual_pose_env_vars_present()
    if manual_env:
        missing.extend(manual_env)
    pgrep_status = _pgrep_competing_motion(run_dir)
    if pgrep_status["competitors"] and not _truthy_env("OPENFORGE_FRIDGE_IGNORE_COMPETING_PROCESS_CHECK"):
        missing.append("no competing run_script.py/ffmpeg process")
    return {
        "passed": not missing,
        "missing_or_invalid": missing,
        "attempt_budget": budget,
        "ticket_scope": scope,
        "pgrep": pgrep_status,
        "policy": "physical mode requires explicit env-backed ticket/precondition acknowledgements plus OPENFORGE_ALLOW_PHYSICAL_MOTION=1",
    }


def _motion_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None, preview_only: bool = False) -> dict[str, Any]:
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_FRIDGE_PLANNER_BACKEND", "curobo"),
        "solver_speed": os.environ.get("OPENFORGE_FRIDGE_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_FRIDGE_PHYSICAL_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_FRIDGE_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_FRIDGE_IK_ROT_THRESHOLD_DEG", 12.0),
        "ik_xyz_weight": _env_float("OPENFORGE_FRIDGE_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_FRIDGE_IK_RPY_WEIGHT", 0.25),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _preview_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    executed = bool(_field(result, "executed", False))
    status_text = str(status) if status is not None else None
    return {
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "trajectory_cache_key": _field(result, "trajectory_cache_key", None),
        "final_pos_error_m": _field(result, "final_pos_error_m", _field(result, "ik_error_m", None)),
        "final_rot_error_deg": _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None)),
        "success": bool(status_text and status_text.lower() in {"success", "succeeded", "ok"} and not executed),
        "raw": json_safe(result),
    }


def _execute_summary(result: Any) -> dict[str, Any]:
    status = _field(result, "status", None)
    executed = bool(_field(result, "executed", True))
    status_text = str(status) if status is not None else None
    return {
        "status": status_text,
        "reason": _field(result, "reason", _field(result, "error", None)),
        "executed": executed,
        "final_pos_error_m": _field(result, "final_pos_error_m", _field(result, "ik_error_m", None)),
        "final_rot_error_deg": _field(result, "final_rot_error_deg", _field(result, "ik_rot_error_deg", None)),
        "success": bool((status_text is None or status_text.lower() in {"success", "succeeded", "ok"}) and executed),
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
    preview = freespace_move(**_motion_kwargs(side, pose, gripper=gripper, preview_only=True))
    preview_packet = _preview_summary(preview)
    if not preview_packet["success"] or not preview_packet.get("trajectory_cache_key"):
        raise RuntimeError(f"{label} preview failed before execution: {preview_packet}")
    executed = freespace_move(trajectory_cache_key=preview_packet["trajectory_cache_key"])
    execute_packet = _execute_summary(executed)
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


def _set_gripper_checked(
    *,
    side: str,
    target: float,
    label: str,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    vel_limit: float,
    torque_limit: float,
    confirm_open_min: float | None = None,
) -> dict[str, Any]:
    before = read_gripper_width(get_robot_state, side)
    result = set_gripper(side, float(target), vel_limit=vel_limit, torque_limit=torque_limit)
    time.sleep(_env_float("OPENFORGE_FRIDGE_GRIPPER_SETTLE_S", 0.25))
    after = read_gripper_width(get_robot_state, side)
    ok = True
    if confirm_open_min is not None and after is not None:
        deadline = time.monotonic() + _env_float("OPENFORGE_FRIDGE_OPEN_CONFIRM_TIMEOUT_S", 3.0)
        while after < confirm_open_min and time.monotonic() < deadline:
            time.sleep(0.1)
            after = read_gripper_width(get_robot_state, side)
            if after is None:
                break
        ok = after is None or after >= confirm_open_min
    return {
        "stage": label,
        "side": side,
        "target": float(target),
        "before": before,
        "after": after,
        "ok": ok,
        "vel_limit": vel_limit,
        "torque_limit": torque_limit,
        "result": json_safe(result),
    }


def _post_observe(run_dir: Path, stage: str) -> dict[str, Any]:
    detect_objects_oneshot = _required_tool("detect_objects_oneshot")
    get_camera_image = _tool("get_camera_image")
    get_robot_state = _tool("get_robot_state")
    run_in_background = _tool("run_in_background")
    packet = capture_scene(
        prompts=["refrigerator door", "refrigerator handle", "box"],
        cameras=["top", "left", "right"],
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=2,
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
    )
    return packet


def _execute_physical(
    *,
    run_dir: Path,
    selected: dict[str, Any],
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    go_home = _tool("go_home")
    door = selected["door_open"]
    box = selected["box_grasp"]
    place = selected["place_inside"]
    door_arm = str(door["arm"])
    box_arm = str(box["arm"])
    attempt: dict[str, Any] = {
        "schema": "openforge.green_fridge_box.physical_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    try:
        for side in sorted({door_arm, box_arm}):
            step = _set_gripper_checked(
                side=side,
                target=_env_float("OPENFORGE_FRIDGE_OPEN_GRIPPER_POS", 1.0),
                label=f"open_{side}_gripper_initial",
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                vel_limit=_env_float("OPENFORGE_FRIDGE_OPEN_VEL_LIMIT", 0.45),
                torque_limit=_env_float("OPENFORGE_FRIDGE_OPEN_TORQUE_LIMIT", 0.18),
                confirm_open_min=_env_float("OPENFORGE_FRIDGE_OPEN_CONFIRM_POS", 0.55),
            )
            attempt["physical_motion_executed"] = True
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
            attempt["steps"].append(step)
            if not step["ok"]:
                attempt["why_stopped"] = f"{side} gripper did not confirm open"
                return attempt

        for label, pose_key, gripper in (
            ("door_pregrasp", "pregrasp_pose", 1.0),
            ("door_handle_grasp_pose", "handle_grasp_pose", 1.0),
        ):
            step = _execute_pose(
                freespace_move=freespace_move,
                side=door_arm,
                pose=door[pose_key],
                label=label,
                gripper=gripper,
            )
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        handle_close = staged_close_with_contact(
            side=door_arm,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=_env_float("OPENFORGE_FRIDGE_HANDLE_CLOSE_TARGET_POS", 0.50),
            steps=_csv("OPENFORGE_FRIDGE_HANDLE_CLOSE_STEPS_POS", ["0.80", "0.65", "0.55", "0.50"]),
            vel_limit=_env_float("OPENFORGE_FRIDGE_HANDLE_CLOSE_VEL_LIMIT", 0.20),
            torque_limit=_env_float("OPENFORGE_FRIDGE_HANDLE_CLOSE_TORQUE_LIMIT", 0.10),
            min_contact_delta=_env_float("OPENFORGE_FRIDGE_HANDLE_CLOSE_CONTACT_MIN_DELTA_POS", 0.015),
            target_tolerance=_env_float("OPENFORGE_FRIDGE_HANDLE_CLOSE_TARGET_TOLERANCE_POS", 0.08),
            hold_min=_env_float("OPENFORGE_FRIDGE_HANDLE_HOLD_MIN_POS", 0.20),
            hold_max=_env_float("OPENFORGE_FRIDGE_HANDLE_HOLD_MAX_POS", 0.90),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="door_handle_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{door_arm}")
        attempt["gripper_after_handle_close"] = handle_close
        if not bool(handle_close.get("plausible_for_lift")):
            attempt["why_stopped"] = "handle close did not produce plausible hold evidence; stop before pull"
            return attempt
        if _truthy_env("OPENFORGE_FRIDGE_HANDLE_REQUIRE_CONTACT", True) and not bool(handle_close.get("contact_detected")):
            attempt["why_stopped"] = "handle close did not detect contact; stop before pull"
            return attempt

        for label, pose_key, gripper in (
            ("door_small_pull_open", "door_open_small_pose", _field(handle_close, "after", 0.55)),
            ("door_release_retreat_closed", "release_retreat_pose", _field(handle_close, "after", 0.55)),
        ):
            step = _execute_pose(
                freespace_move=freespace_move,
                side=door_arm,
                pose=door[pose_key],
                label=label,
                gripper=float(gripper or 0.55),
            )
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        release_handle = _set_gripper_checked(
            side=door_arm,
            target=1.0,
            label="release_handle_open",
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            vel_limit=_env_float("OPENFORGE_FRIDGE_OPEN_VEL_LIMIT", 0.45),
            torque_limit=_env_float("OPENFORGE_FRIDGE_OPEN_TORQUE_LIMIT", 0.18),
            confirm_open_min=_env_float("OPENFORGE_FRIDGE_OPEN_CONFIRM_POS", 0.55),
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{door_arm}")
        attempt["steps"].append(release_handle)
        if not release_handle["ok"]:
            attempt["why_stopped"] = "door handle release did not confirm open"
            return attempt

        attempt["post_open_observation"] = _post_observe(run_dir, "post_open_observe")

        step = _execute_pose(
            freespace_move=freespace_move,
            side=box_arm,
            pose=box["pregrasp_pose"],
            label="box_pregrasp",
            gripper=1.0,
        )
        attempt["movement_capable_calls"].append("freespace_move:box_pregrasp")
        attempt["steps"].append(step)

        box_preshape = _set_gripper_checked(
            side=box_arm,
            target=_env_float("OPENFORGE_FRIDGE_BOX_PRESHAPE_POS", 0.65),
            label="box_preshape_before_final_approach",
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            vel_limit=_env_float("OPENFORGE_FRIDGE_BOX_PRESHAPE_VEL_LIMIT", 0.22),
            torque_limit=_env_float("OPENFORGE_FRIDGE_BOX_PRESHAPE_TORQUE_LIMIT", 0.10),
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{box_arm}")
        attempt["steps"].append(box_preshape)
        box_approach_collision_gripper = _env_float(
            "OPENFORGE_FRIDGE_BOX_APPROACH_COLLISION_GRIPPER_POS",
            float(box_preshape.get("after") or 0.65),
        )
        step = _execute_pose(
            freespace_move=freespace_move,
            side=box_arm,
            pose=box["grasp_pose"],
            label="box_grasp_pose",
            gripper=box_approach_collision_gripper,
        )
        attempt["movement_capable_calls"].append("freespace_move:box_grasp_pose")
        attempt["steps"].append(step)

        box_close = staged_close_with_contact(
            side=box_arm,
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            target=_env_float("OPENFORGE_FRIDGE_BOX_CLOSE_TARGET_POS", 0.22),
            steps=_csv("OPENFORGE_FRIDGE_BOX_CLOSE_STEPS_POS", ["0.75", "0.55", "0.36", "0.22"]),
            vel_limit=_env_float("OPENFORGE_FRIDGE_BOX_CLOSE_VEL_LIMIT", 0.28),
            torque_limit=_env_float("OPENFORGE_FRIDGE_BOX_CLOSE_TORQUE_LIMIT", 0.16),
            min_contact_delta=_env_float("OPENFORGE_FRIDGE_BOX_CLOSE_CONTACT_MIN_DELTA_POS", 0.02),
            target_tolerance=_env_float("OPENFORGE_FRIDGE_BOX_CLOSE_TARGET_TOLERANCE_POS", 0.06),
            hold_min=_env_float("OPENFORGE_FRIDGE_BOX_HOLD_MIN_POS", 0.02),
            hold_max=_env_float("OPENFORGE_FRIDGE_BOX_HOLD_MAX_POS", 0.90),
            run_dir=run_dir,
            task_name=TASK_NAME,
            stage="box_staged_close",
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{box_arm}")
        attempt["gripper_after_box_close"] = box_close
        if not bool(box_close.get("plausible_for_lift")):
            attempt["why_stopped"] = "box close did not produce plausible hold evidence; stop before lift/place"
            return attempt
        box_hold = float(box_close.get("after") or 0.22)

        for label, pose in (
            ("box_lift", box["lift_pose"]),
            ("place_preplace", place["preplace_pose"]),
            ("place_inside", place["place_pose"]),
        ):
            step = _execute_pose(
                freespace_move=freespace_move,
                side=box_arm,
                pose=pose,
                label=label,
                gripper=box_hold,
            )
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)

        release_box = _set_gripper_checked(
            side=box_arm,
            target=1.0,
            label="release_box_open_at_inside_place_pose",
            set_gripper=set_gripper,
            get_robot_state=get_robot_state,
            vel_limit=_env_float("OPENFORGE_FRIDGE_OPEN_VEL_LIMIT", 0.45),
            torque_limit=_env_float("OPENFORGE_FRIDGE_OPEN_TORQUE_LIMIT", 0.18),
            confirm_open_min=_env_float("OPENFORGE_FRIDGE_OPEN_CONFIRM_POS", 0.55),
        )
        attempt["movement_capable_calls"].append(f"set_gripper:{box_arm}")
        attempt["steps"].append(release_box)
        if not release_box["ok"]:
            attempt["why_stopped"] = "box release did not confirm open"
            return attempt

        retreat_open = _execute_pose(
            freespace_move=freespace_move,
            side=box_arm,
            pose=place["retreat_pose"],
            label="place_retreat_after_release",
            gripper=1.0,
        )
        attempt["movement_capable_calls"].append("freespace_move:place_retreat_after_release")
        attempt["steps"].append(retreat_open)

        attempt["post_release_observation"] = _post_observe(run_dir, "post_release_observe")
        if _truthy_env("OPENFORGE_FRIDGE_GO_HOME_AFTER_RELEASE", False):
            if not callable(go_home):
                raise RuntimeError("OPENFORGE_FRIDGE_GO_HOME_AFTER_RELEASE requested but go_home tool is unavailable")
            home_result = go_home()
            attempt["movement_capable_calls"].append("go_home")
            attempt["steps"].append({"stage": "go_home_after_release", "ok": True, "result": json_safe(home_result)})
        attempt["success"] = True
        attempt["why_stopped"] = "staged refrigerator open and box place sequence completed; post-release observation recorded"
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
    return attempt


def _execute_recovery(run_dir: Path) -> dict[str, Any]:
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _required_tool("get_robot_state")
    go_home = _required_tool("go_home")
    attempt: dict[str, Any] = {
        "schema": "openforge.green_fridge_box.recovery_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
        "policy": "recovery-only open grippers then go_home; no object pose target is used",
    }
    try:
        for side in ("left", "right"):
            step = _set_gripper_checked(
                side=side,
                target=_env_float("OPENFORGE_FRIDGE_OPEN_GRIPPER_POS", 1.0),
                label=f"recovery_open_{side}_gripper",
                set_gripper=set_gripper,
                get_robot_state=get_robot_state,
                vel_limit=_env_float("OPENFORGE_FRIDGE_OPEN_VEL_LIMIT", 0.45),
                torque_limit=_env_float("OPENFORGE_FRIDGE_OPEN_TORQUE_LIMIT", 0.18),
                confirm_open_min=_env_float("OPENFORGE_FRIDGE_OPEN_CONFIRM_POS", 0.55),
            )
            attempt["physical_motion_executed"] = True
            attempt["movement_capable_calls"].append(f"set_gripper:{side}")
            attempt["steps"].append(step)
            if not step["ok"]:
                attempt["why_stopped"] = f"recovery {side} gripper did not confirm open"
                return attempt
        home_result = go_home()
        attempt["movement_capable_calls"].append("go_home")
        attempt["steps"].append({"stage": "recovery_go_home", "ok": True, "result": json_safe(home_result)})
        attempt["success"] = True
        attempt["why_stopped"] = "recovery-only open grippers and go_home completed"
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
    return attempt


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.green_fridge_box.physical_wrapper.result.v1",
        "success": False,
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "run_dir": str(run_dir),
        "method": TASK_NAME,
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "observation_policy": "fresh live detections only; manual XYZ env vars are rejected",
        "planner_module": PLANNER_MODULE,
        "planner_result": None,
        "plan_validation": None,
        "recovery_only": _truthy_env("OPENFORGE_FRIDGE_RECOVERY_ONLY"),
        "physical_ticket_status": None,
        "physical_attempt": None,
        "recording_evidence": None,
        "video_paths": [],
        "why_stopped": None,
        "next_recommendation": None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        manual_env = _manual_pose_env_vars_present()
        if manual_env:
            result["why_stopped"] = "manual XYZ/debug env vars are not allowed for this live refrigerator task"
            result["manual_pose_env_vars_present"] = manual_env
            return result

        if result["recovery_only"]:
            append_stage_summary(
                run_dir,
                [
                    "## recovery-only mode",
                    "- requested: True",
                    f"- physical_gate: {result['physical_gate']}",
                    "- policy: open grippers then go_home; no manual object pose is used",
                ],
            )
            if not result["physical_gate"]:
                result["success"] = True
                result["why_stopped"] = "recovery-only no-motion validation succeeded; physical gate was not set"
                result["next_recommendation"] = (
                    "For recovery motion, rerun under the active ticket with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
                    "and explicit OPENFORGE_FRIDGE_* ticket/precondition env acknowledgements."
                )
                return result
            recording = _recording_evidence(run_dir)
            if not recording.get("passed"):
                result["recording_evidence"] = recording
                result["why_stopped"] = "recording evidence is missing; refusing recovery physical motion"
                result["next_recommendation"] = "Fix recording/debug UI evidence and rerun no-motion recovery validation."
                return result
            ticket_status = _physical_ticket_status(run_dir)
            result["physical_ticket_status"] = ticket_status
            append_stage_summary(
                run_dir,
                [
                    "## recovery ticket gate",
                    f"- passed: {ticket_status['passed']}",
                    f"- missing_or_invalid: {ticket_status['missing_or_invalid']}",
                ],
            )
            if not ticket_status.get("passed"):
                result["why_stopped"] = "physical ticket/precondition env acknowledgements are incomplete; refusing recovery motion"
                result["next_recommendation"] = "Continue no-motion or rerun recovery with complete explicit ticket/precondition env vars."
                return result
            attempt = _execute_recovery(run_dir)
            result["physical_attempt"] = attempt
            result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
            result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
            result["success"] = bool(attempt.get("success"))
            result["why_stopped"] = attempt.get("why_stopped")
            result["next_recommendation"] = "Inspect videos and robot state before any retry."
            return result

        planner_result = _run_fresh_no_motion_planner()
        result["planner_result"] = planner_result
        plan_validation = _validate_selected_strategy(planner_result)
        result["plan_validation"] = plan_validation
        result["plan_success"] = bool(plan_validation.get("passed"))
        append_stage_summary(
            run_dir,
            [
                "## physical wrapper plan validation",
                f"- passed: {plan_validation['passed']}",
                f"- door: {plan_validation.get('door_label')} arm={plan_validation.get('door_arm')}",
                f"- box: {plan_validation.get('box_label')} arm={plan_validation.get('box_arm')}",
                f"- place: {plan_validation.get('place_label')} arm={plan_validation.get('place_arm')}",
            ],
        )
        if not plan_validation.get("passed"):
            result["why_stopped"] = "fresh no-motion planner did not produce a complete selected door/box/place strategy"
            result["next_recommendation"] = "Inspect planner artifacts and rerun no-motion before any physical attempt."
            return result

        if not result["physical_gate"]:
            result["success"] = True
            result["why_stopped"] = "recorded no-motion validation succeeded; physical gate was not set"
            result["next_recommendation"] = (
                "For physical mode, rerun only under the active ticket with OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
                "and explicit OPENFORGE_FRIDGE_* ticket/precondition env acknowledgements."
            )
            return result

        recording = _recording_evidence(run_dir)
        if not recording.get("passed"):
            result["recording_evidence"] = recording
            result["why_stopped"] = "recording evidence is missing; refusing physical motion"
            result["next_recommendation"] = "Fix recording/debug UI evidence and rerun no-motion validation."
            return result

        ticket_status = _physical_ticket_status(run_dir)
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
            result["next_recommendation"] = "Continue no-motion or rerun physical with complete explicit ticket/precondition env vars."
            return result

        attempt = _execute_physical(run_dir=run_dir, selected=plan_validation["selected_strategy"])
        result["physical_attempt"] = attempt
        result["physical_motion_executed"] = bool(attempt.get("physical_motion_executed"))
        result["movement_capable_calls"] = list(attempt.get("movement_capable_calls") or [])
        result["success"] = bool(attempt.get("success"))
        result["why_stopped"] = attempt.get("why_stopped")
        result["next_recommendation"] = "Inspect post-action observations/videos before any retry."
        return result
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and run log before retrying."
        return result
    finally:
        recording = _recording_evidence(run_dir)
        result["recording_evidence"] = result.get("recording_evidence") or recording
        result["video_paths"] = recording.get("video_paths", [])
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## physical wrapper result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- recording_evidence: {result['recording_evidence']}",
                f"- why_stopped: {result['why_stopped']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
