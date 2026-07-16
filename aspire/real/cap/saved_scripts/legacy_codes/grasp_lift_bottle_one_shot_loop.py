"""One-command live-observe/plan/grasp/lift loop for a dry wine bottle."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    compare_detection_motion,
    current_run_dir,
    execute_grasp_lift_attempt,
    generate_side_grasp_candidates,
    rank_motion_candidates,
    verify_lift,
    write_json,
)


TASK_NAME = "grasp_lift_bottle_one_shot_loop"
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


def _optional_env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


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
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


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


def _json_safe(value: Any) -> Any:
    try:
        from cap.saved_scripts.yam_runtime import json_safe

        return json_safe(value)
    except Exception:
        try:
            return json.loads(json.dumps(value, default=repr))
        except Exception:
            return repr(value)


def _make_open_gripper() -> Callable[[str], Any]:
    set_gripper = _tool("set_gripper")
    open_gripper = _tool("open_gripper")
    open_target = _env_float("OPENFORGE_GRASP_LIFT_OPEN_TARGET_POS", 1.0)
    open_vel = _env_float("OPENFORGE_GRASP_LIFT_OPEN_VEL_LIMIT", 0.8)
    open_torque = _env_float("OPENFORGE_GRASP_LIFT_OPEN_TORQUE_LIMIT", 0.30)

    def _open(side: str) -> Any:
        if callable(set_gripper):
            return set_gripper(side, open_target, vel_limit=open_vel, torque_limit=open_torque)
        if callable(open_gripper):
            return open_gripper(side)
        raise RuntimeError("no open gripper API is available")

    return _open


def _extract_gripper_pos(state: Any, side: str) -> float | None:
    safe = _json_safe(state)
    if isinstance(safe, dict):
        arms = safe.get("arms")
        if isinstance(arms, dict):
            arm_state = arms.get(side)
            if isinstance(arm_state, dict):
                value = arm_state.get("gripper_pos")
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
        value = safe.get(f"{side}_gripper_pos")
        if value is None:
            value = safe.get(f"{side}_grip")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    arms_obj = getattr(state, "arms", None)
    if isinstance(arms_obj, dict):
        arm_obj = arms_obj.get(side)
        value = getattr(arm_obj, "gripper_pos", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _open_gripper_until(
    side: str,
    *,
    min_pos: float,
    timeout_s: float,
    poll_s: float,
    retry_s: float,
) -> dict[str, Any]:
    open_fn = _make_open_gripper()
    get_robot_state = _tool("get_robot_state")
    packet: dict[str, Any] = {
        "success": False,
        "side": side,
        "target_min_pos": float(min_pos),
        "timeout_s": float(timeout_s),
        "poll_s": float(poll_s),
        "retry_s": float(retry_s),
        "commands": [],
        "samples": [],
        "final_gripper_pos": None,
        "final_state": None,
        "why_stopped": None,
    }

    deadline = time.monotonic() + max(0.1, timeout_s)
    next_command_time = 0.0
    while time.monotonic() <= deadline:
        now = time.monotonic()
        if now >= next_command_time:
            command_result = open_fn(side)
            packet["commands"].append(_json_safe(command_result))
            next_command_time = now + max(0.1, retry_s)
        if callable(get_robot_state):
            state = get_robot_state()
            pos = _extract_gripper_pos(state, side)
            packet["final_state"] = _json_safe(state)
            packet["final_gripper_pos"] = pos
            packet["samples"].append(
                {
                    "t_remaining_s": round(max(0.0, deadline - time.monotonic()), 3),
                    "gripper_pos": pos,
                }
            )
            if pos is not None and pos >= min_pos:
                packet["success"] = True
                packet["why_stopped"] = "verified open from robot gripper encoder"
                return packet
        time.sleep(max(0.02, poll_s))

    packet["why_stopped"] = (
        f"timed out waiting for {side} gripper to open to >= {min_pos:.3f}; "
        f"final gripper_pos={packet['final_gripper_pos']}"
    )
    return packet


def _capture_prompt_fallback(
    *,
    run_dir: Path,
    stage: str,
    prompts: list[str],
    cameras: list[str],
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    detect_objects_oneshot = _required_tool("detect_objects_oneshot")
    get_camera_image = _tool("get_camera_image")
    get_robot_state = _tool("get_robot_state")
    run_in_background = _tool("run_in_background")
    packets: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prompt_index, prompt in enumerate(prompts):
        packet = capture_scene(
            prompts=[prompt],
            cameras=cameras,
            detect_objects_oneshot=detect_objects_oneshot,
            get_camera_image=get_camera_image,
            get_robot_state=get_robot_state,
            run_in_background=run_in_background,
            run_dir=run_dir,
            stage=f"{stage}_{prompt.replace(' ', '_')}",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
            max_retries=max(1, _env_int("OPENFORGE_GRASP_LIFT_DETECT_RETRIES", 2)),
            save_overlays=True,
        )
        packets.append(packet)
        detection = packet.get("best_detection")
        score = float((detection or {}).get("score") or 0.0)
        if detection is not None and score >= min_score:
            best = dict(detection)
            best["prompt_index"] = prompt_index
            best["prompt_fallback_used"] = bool(prompt_index > 0)
            raw_source = best.get("source")
            live_source = "live_top_camera" if best.get("source_camera") == "top" else "live_perception"
            best["raw_detection_source"] = raw_source
            best["source"] = live_source
            best["detection_source"] = live_source
            break
    combined = {
        "stage": stage,
        "prompts_tried": prompts[: len(packets)],
        "packets": [
            {
                "packet_path": packet.get("packet_path"),
                "best_detection": packet.get("best_detection"),
                "errors": packet.get("errors"),
            }
            for packet in packets
        ],
        "best_detection": best,
        "source": "live_perception",
    }
    combined["path"] = write_json(run_dir / "observations" / f"{stage}_prompt_fallback.json", combined)
    return combined, best


def _apply_bottle_pregrasp_clearance(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clearance = _optional_env_float("OPENFORGE_GRASP_LIFT_SIDE_PREGRASP_CLEARANCE_M")
    if clearance is None:
        return candidates
    if clearance < 0.0:
        raise ValueError("OPENFORGE_GRASP_LIFT_SIDE_PREGRASP_CLEARANCE_M must be nonnegative")

    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        cand = dict(candidate)
        object_center = cand.get("object_center")
        approach = cand.get("approach_direction_world")
        grasp_pose = dict(cand.get("grasp_pose") or {})
        grasp_pos = grasp_pose.get("position")
        pre_pose = dict(cand.get("pregrasp_pose") or {})
        if not object_center or not approach or not grasp_pos:
            cand["pregrasp_policy"] = "unchanged_missing_geometry_for_clearance"
            adjusted.append(cand)
            continue

        center = [float(value) for value in list(object_center)[:3]]
        approach_vec = [float(value) for value in list(approach)[:3]]
        grasp = [float(value) for value in list(grasp_pos)[:3]]
        pre = [
            center[0] - approach_vec[0] * clearance,
            center[1] - approach_vec[1] * clearance,
            grasp[2],
        ]
        old_pre_pose = cand.get("pregrasp_pose")
        if old_pre_pose is not None and "original_pregrasp_pose" not in cand:
            cand["original_pregrasp_pose"] = _json_safe(old_pre_pose)
        pre_pose["position"] = pre
        cand["pregrasp_pose"] = pre_pose
        cand["pregrasp_policy"] = "object_center_clearance_decoupled_from_grasp_backoff"
        cand["pregrasp_clearance_m"] = float(clearance)
        cand["pregrasp_to_grasp_distance_m"] = float(math.dist(pre, grasp))
        cand["pregrasp_to_grasp_vector_m"] = [grasp[i] - pre[i] for i in range(3)]
        adjusted.append(cand)
    return adjusted


def _build_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    requested_arm = os.environ.get("OPENFORGE_GRASP_LIFT_ARM", "auto").strip().lower()
    arm = requested_arm if requested_arm in {"left", "right"} else None
    candidates = generate_side_grasp_candidates(
        detection,
        object_kind="bottle",
        arm=arm,
        default_radius_m=_env_float("OPENFORGE_GRASP_LIFT_OBJECT_RADIUS_M", 0.04),
        default_half_height_m=0.0,
        body_fraction=_env_float("OPENFORGE_GRASP_LIFT_BODY_GRASP_FRACTION", 0.45),
        pregrasp_standoff_m=_env_float("OPENFORGE_GRASP_LIFT_SIDE_PREGRASP_STANDOFF_M", 0.08),
        lift_z_m=_env_float("OPENFORGE_GRASP_LIFT_LIFT_Z_M", 0.05),
        width_margin_m=_env_float("OPENFORGE_GRASP_LIFT_GRIPPER_WIDTH_CLEARANCE_M", 0.015),
        include_topdown=False,
        yaw_angles_deg=_float_csv("OPENFORGE_GRASP_LIFT_SIDE_YAWS_DEG", [0.0, 90.0, 70.0, 110.0, 180.0]),
        z_offsets_m=_float_csv("OPENFORGE_GRASP_LIFT_SIDE_Z_OFFSETS_M", [0.0, 0.015, -0.015, 0.03]),
        center_z_offset_without_extents_m=_env_float(
            "OPENFORGE_GRASP_LIFT_LIVE_DETECTION_Z_TO_BODY_OFFSET_M",
            -0.10,
        ),
        grasp_backoff_m=_env_float("OPENFORGE_GRASP_LIFT_SIDE_GRASP_BACKOFF_M", 0.0),
    )
    return _apply_bottle_pregrasp_clearance(candidates)


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists():
            paths.append(str(path))
    return paths


def _verify_grasp_only(
    *,
    pre_detection: dict[str, Any] | None,
    post_detection: dict[str, Any] | None,
    gripper_after_close: dict[str, Any] | None,
) -> dict[str, Any]:
    motion = compare_detection_motion(pre_detection, post_detection)
    if not motion.get("ok"):
        return {"success": False, "confidence": 0.0, "motion": motion, "reason": motion.get("reason")}
    close = gripper_after_close or {}
    xy_delta = abs(float(motion.get("xy_delta_m") or 0.0))
    z_delta = abs(float(motion.get("z_delta_m") or 0.0))
    max_xy = _env_float("OPENFORGE_GRASP_LIFT_GRASP_ONLY_MAX_XY_MOTION_M", 0.025)
    max_z = _env_float("OPENFORGE_GRASP_LIFT_GRASP_ONLY_MAX_ABS_Z_MOTION_M", 0.035)
    close_ok = bool(close.get("plausible_for_lift") or close.get("target_reached") or close.get("contact_detected"))
    stable = bool(xy_delta <= max_xy and z_delta <= max_z)
    success = bool(close_ok and stable)
    if success:
        reason = "grasp_close_plausible_and_bottle_stable"
    elif not close_ok:
        reason = "gripper close did not produce plausible enclosure evidence"
    else:
        reason = "post-grasp detection moved too much for a no-push grasp"
    return {
        "success": success,
        "confidence": 0.65 if success else 0.25,
        "motion": motion,
        "reason": reason,
        "thresholds": {
            "max_xy_delta_m": max_xy,
            "max_abs_z_delta_m": max_z,
        },
        "gripper_evidence": close,
    }


def _return_after_verified_lift(
    *,
    selected: dict[str, Any],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    side = str(selected["arm"])
    prefix = "left" if side == "left" else "right"
    packet: dict[str, Any] = {
        "success": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    try:
        def _preview_and_move_pose(pose_key: str, stage: str) -> bool:
            pose = selected.get(pose_key)
            if not pose:
                packet["why_stopped"] = f"selected strategy has no {pose_key}; cannot execute {stage}"
                return False
            move_kwargs = {
                f"{prefix}_target_pos": pose["position"],
                f"{prefix}_target_rpy": pose["rpy"],
                "planner_backend": os.environ.get("OPENFORGE_GRASP_LIFT_PLANNER_BACKEND", "curobo"),
                "solver_speed": os.environ.get("OPENFORGE_GRASP_LIFT_SOLVER_SPEED", "slow"),
                "planning_speed": _env_float("OPENFORGE_GRASP_LIFT_PLANNING_SPEED", 0.35),
            }
            preview_result = freespace_move(
                **{
                    **move_kwargs,
                    "preview_only": True,
                }
            )
            packet["steps"].append({"stage": f"preview_{stage}", "result": _json_safe(preview_result)})
            cache_key = preview_result.get("trajectory_cache_key") if isinstance(preview_result, dict) else getattr(
                preview_result,
                "trajectory_cache_key",
                None,
            )
            if not cache_key:
                packet["why_stopped"] = f"{stage} preview did not provide trajectory cache: {preview_result!r}"
                return False
            move_result = freespace_move(trajectory_cache_key=cache_key)
            packet["movement_capable_calls"].append("freespace_move")
            packet["steps"].append({"stage": stage, "result": _json_safe(move_result)})
            return True

        if not _preview_and_move_pose("grasp_pose", "return_to_grasp_pose"):
            return packet
        open_result = _open_gripper_until(
            side,
            min_pos=_env_float("OPENFORGE_GRASP_LIFT_OPEN_BEFORE_RETREAT_MIN_POS", 0.95),
            timeout_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_BEFORE_RETREAT_TIMEOUT_S", 12.0),
            poll_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_BEFORE_RETREAT_POLL_S", 0.25),
            retry_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_BEFORE_RETREAT_RETRY_S", 1.5),
        )
        packet["movement_capable_calls"].append(f"open_gripper:{side}")
        packet["steps"].append({"stage": "release_gripper_before_retreat", "result": _json_safe(open_result)})
        if not open_result.get("success"):
            packet["why_stopped"] = str(open_result.get("why_stopped") or "release open verification failed before retreat")
            return packet
        if _truthy_env("OPENFORGE_GRASP_LIFT_RETREAT_TO_PREGRASP_AFTER_RELEASE", True):
            if not _preview_and_move_pose("pregrasp_pose", "retreat_to_pregrasp_pose"):
                return packet
        if _truthy_env("OPENFORGE_GRASP_LIFT_GO_HOME_AFTER_RETURN", False):
            go_home = _required_tool("go_home")
            packet["steps"].append({"stage": "go_home_after_pregrasp_retreat", "result": _json_safe(go_home())})
            packet["movement_capable_calls"].append("go_home")
            if _truthy_env("OPENFORGE_GRASP_LIFT_OPEN_AFTER_HOME", True):
                final_open_result = _open_gripper_until(
                    side,
                    min_pos=_env_float("OPENFORGE_GRASP_LIFT_OPEN_AFTER_HOME_MIN_POS", 0.90),
                    timeout_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_AFTER_HOME_TIMEOUT_S", 12.0),
                    poll_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_AFTER_HOME_POLL_S", 0.25),
                    retry_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_AFTER_HOME_RETRY_S", 1.5),
                )
                packet["movement_capable_calls"].append(f"open_gripper:{side}")
                packet["steps"].append({"stage": "open_gripper_after_home", "result": _json_safe(final_open_result)})
                if not final_open_result.get("success"):
                    packet["why_stopped"] = str(final_open_result.get("why_stopped") or "post-home open verification failed")
                    return packet
        packet["success"] = True
        if _truthy_env("OPENFORGE_GRASP_LIFT_GO_HOME_AFTER_RETURN", False):
            packet["why_stopped"] = "returned to grasp pose, opened, retreated to pregrasp, went home, and reopened gripper"
        elif _truthy_env("OPENFORGE_GRASP_LIFT_RETREAT_TO_PREGRASP_AFTER_RELEASE", True):
            packet["why_stopped"] = "returned to grasp pose, opened, and retreated to pregrasp"
        else:
            packet["why_stopped"] = "returned to planned grasp pose and opened gripper"
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
    return packet


def _recover_open_home(run_dir: Path, *, reason: str) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "success": False,
        "reason": reason,
        "movement_capable_calls": [],
        "steps": [],
        "before_state": None,
        "after_state": None,
    }
    get_robot_state = _tool("get_robot_state")
    if callable(get_robot_state):
        packet["before_state"] = _json_safe(get_robot_state())
    try:
        open_fn = _make_open_gripper()
        for side in ("left", "right"):
            packet["steps"].append({"stage": f"open_{side}_before_home", "result": _json_safe(open_fn(side))})
            packet["movement_capable_calls"].append(f"open_gripper:{side}")
        go_home = _required_tool("go_home")
        packet["steps"].append({"stage": "go_home", "result": _json_safe(go_home())})
        packet["movement_capable_calls"].append("go_home")
        for side in ("left", "right"):
            packet["steps"].append({"stage": f"open_{side}_after_home", "result": _json_safe(open_fn(side))})
            packet["movement_capable_calls"].append(f"open_gripper:{side}")
        if callable(get_robot_state):
            packet["after_state"] = _json_safe(get_robot_state())
        packet["success"] = True
    except Exception as exc:
        packet["error"] = f"{type(exc).__name__}: {exc}"
        if callable(get_robot_state):
            packet["after_state"] = _json_safe(get_robot_state())
    packet["path"] = write_json(run_dir / "plans" / "recovery_open_home.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## recovery open_home",
            f"- success: {packet['success']}",
            f"- reason: {reason}",
        ],
    )
    return packet


def _safe_to_recover_home(execution: dict[str, Any], verification: dict[str, Any]) -> bool:
    if bool(verification.get("success")):
        return False
    if execution.get("gripper_after_close") is not None:
        return False
    steps = execution.get("steps") or []
    closed_or_lifted = any(str(step.get("stage")) in {"lift_pose", "grasp_close"} for step in steps)
    return not closed_or_lifted and bool(execution.get("physical_motion_executed"))


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    allow_physical = _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION")
    grasp_only = _truthy_env("OPENFORGE_GRASP_LIFT_GRASP_ONLY", False)
    plan_only = (not allow_physical) or _truthy_env("OPENFORGE_GRASP_LIFT_PLAN_ONLY")
    attempt_budget = 1 if plan_only else max(1, _env_int("OPENFORGE_GRASP_LIFT_ATTEMPT_BUDGET", 2))
    prompts = _csv("OPENFORGE_BOTTLE_PROMPTS", ["bottle", "wine bottle"])
    cameras = _csv("OPENFORGE_DEBUG_OBS_CAMERAS", ["top"])
    if allow_physical:
        cameras = ["top"]
    min_score = _env_float("OPENFORGE_GRASP_LIFT_MIN_DETECTION_SCORE", 0.25)
    result: dict[str, Any] = {
        "success": False,
        "reward": 0.0,
        "confidence": 0.0,
        "method": TASK_NAME,
        "object_kind": "bottle",
        "run_dir": str(run_dir),
        "prompts": prompts,
        "cameras": cameras,
        "plan_success": False,
        "attempt_budget": attempt_budget,
        "attempt_count": 0,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_strategy": None,
        "pre_observation": None,
        "pre_detection": None,
        "post_observation": None,
        "post_detection": None,
        "gripper_after_close": None,
        "object_motion": None,
        "verification": None,
        "video_paths": [],
        "why_stopped": "not_started",
        "next_recommendation": (
            "Run no-motion plan checks first. Physical motion requires a fresh bounded "
            "dry-bottle ticket and OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
        ),
        "attempts": [],
        "risk_notes": [
            "Physical object poses come from fresh live top-camera observations.",
            "Manual OPENFORGE_BOTTLE_XYZ is ignored by this script for physical planning.",
            "No AnyGrasp, liquid, pour, or cup behavior is used.",
        ],
        "config": {
            "allow_physical": allow_physical,
            "plan_only": plan_only,
            "solver_speed": os.environ.get("OPENFORGE_GRASP_LIFT_SOLVER_SPEED", "slow"),
            "close_target_pos": _env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_POS", 0.35),
            "require_contact_for_lift": _truthy_env("OPENFORGE_GRASP_LIFT_REQUIRE_CLOSE_CONTACT", not grasp_only),
            "grasp_only": grasp_only,
            "side_grasp_backoff_m": _env_float("OPENFORGE_GRASP_LIFT_SIDE_GRASP_BACKOFF_M", 0.0),
            "side_pregrasp_clearance_m": _optional_env_float("OPENFORGE_GRASP_LIFT_SIDE_PREGRASP_CLEARANCE_M"),
            "live_detection_z_to_body_offset_m": _env_float(
                "OPENFORGE_GRASP_LIFT_LIVE_DETECTION_Z_TO_BODY_OFFSET_M",
                -0.10,
            ),
        },
    }
    if os.environ.get("OPENFORGE_BOTTLE_XYZ"):
        result["manual_xyz_status"] = "ignored; manual XYZ is no-motion debug only and not used by this live script"

    try:
        if _truthy_env("OPENFORGE_GRASP_LIFT_RECOVERY_ONLY", False):
            if not allow_physical:
                result["why_stopped"] = "recovery-only requested without OPENFORGE_ALLOW_PHYSICAL_MOTION=1"
                result["next_recommendation"] = "Set the physical gate only under a bounded recovery ticket."
                return result
            recovery = _recover_open_home(run_dir, reason="explicit recovery-only request")
            result["physical_motion_executed"] = bool(recovery.get("movement_capable_calls"))
            result["movement_capable_calls"] = list(recovery.get("movement_capable_calls") or [])
            result["recovery"] = recovery
            result["success"] = bool(recovery.get("success"))
            result["confidence"] = 0.0
            result["why_stopped"] = "recovery-only open/go_home/open completed" if recovery.get("success") else "recovery-only failed"
            result["next_recommendation"] = "Run a fresh no-motion observe/plan check before another grasp attempt."
            return result

        freespace_move = _required_tool("freespace_move")
        for attempt_index in range(1, attempt_budget + 1):
            attempt: dict[str, Any] = {
                "attempt_index": attempt_index,
                "success": False,
                "plan_success": False,
                "physical_motion_executed": False,
            }
            result["attempt_count"] = attempt_index
            result["attempts"].append(attempt)

            pre_obs, pre_det = _capture_prompt_fallback(
                run_dir=run_dir,
                stage=f"attempt_{attempt_index:02d}_pre_observe",
                prompts=prompts,
                cameras=cameras,
                min_score=min_score,
            )
            attempt["pre_observation"] = pre_obs
            attempt["pre_detection"] = pre_det
            result["pre_observation"] = pre_obs
            result["pre_detection"] = pre_det
            if pre_det is None:
                result["why_stopped"] = "no live top-camera bottle detection; no candidate can be planned"
                result["next_recommendation"] = "Inspect top-camera overlay/calibration and retry no-motion observe."
                attempt["why_stopped"] = result["why_stopped"]
                break

            candidates = _build_candidates(pre_det)
            max_preview = max(1, _env_int("OPENFORGE_GRASP_LIFT_MAX_PREVIEW_CANDIDATES", 96))
            plan = rank_motion_candidates(
                candidates=candidates[:max_preview],
                freespace_move=freespace_move,
                run_in_background=_tool("run_in_background"),
                run_dir=run_dir,
                stage=f"attempt_{attempt_index:02d}_grasp_lift_preview",
                task_name=TASK_NAME,
                timeout_s=_env_float("OPENFORGE_GRASP_LIFT_PREVIEW_TIMEOUT_S", 35.0),
                planner_backend=os.environ.get("OPENFORGE_GRASP_LIFT_PLANNER_BACKEND", "curobo"),
                solver_speed=os.environ.get("OPENFORGE_GRASP_LIFT_SOLVER_SPEED", "slow"),
                planning_speed=_env_float("OPENFORGE_GRASP_LIFT_PLANNING_SPEED", 0.35),
                ik_error_threshold=_env_float("OPENFORGE_GRASP_LIFT_IK_ERROR_THRESHOLD_M", 0.015),
                ik_rot_threshold_deg=_env_float("OPENFORGE_GRASP_LIFT_IK_ROT_THRESHOLD_DEG", 8.0),
                pose_keys=("pregrasp_pose", "grasp_pose") if grasp_only else None,
            )
            selected = plan.get("selected")
            attempt["planner_result"] = {
                "plan_packet": plan.get("plan_path"),
                "selected": selected,
                "candidate_count": len(candidates),
                "preview_count": len(plan.get("candidates") or []),
            }
            attempt["selected_strategy"] = selected
            result["selected_strategy"] = selected
            if selected is None:
                result["why_stopped"] = "no side-cylinder bottle candidate passed pregrasp/grasp/lift preview"
                result["next_recommendation"] = "Inspect plan packet and tune live pose/z/approach offsets in no-motion."
                attempt["why_stopped"] = result["why_stopped"]
                break

            result["plan_success"] = True
            attempt["plan_success"] = True
            if plan_only:
                result["why_stopped"] = (
                    "no-motion plan-only run completed; physical grasp requires "
                    "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and a bounded ticket"
                )
                result["next_recommendation"] = "Use this selected live-top candidate as the next physical reference."
                attempt["why_stopped"] = result["why_stopped"]
                break

            execution = execute_grasp_lift_attempt(
                plan=selected,
                freespace_move=freespace_move,
                open_gripper=_make_open_gripper(),
                set_gripper=_required_tool("set_gripper"),
                get_robot_state=_required_tool("get_robot_state"),
                physical_gate=allow_physical,
                run_dir=run_dir,
                task_name=TASK_NAME,
                planning_speed=_env_float("OPENFORGE_GRASP_LIFT_PLANNING_SPEED", 0.35),
                planner_backend=os.environ.get("OPENFORGE_GRASP_LIFT_PLANNER_BACKEND", "curobo"),
                solver_speed=os.environ.get("OPENFORGE_GRASP_LIFT_SOLVER_SPEED", "slow"),
                gripper_target=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_POS", 0.35),
                gripper_steps=(
                    _env_float("OPENFORGE_GRASP_LIFT_CLOSE_SCOUT_POS", 0.78),
                    _env_float("OPENFORGE_GRASP_LIFT_CLOSE_MID_POS", 0.58),
                    _env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_POS", 0.35),
                ),
                gripper_vel_limit=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_VEL_LIMIT", 0.45),
                gripper_torque_limit=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TORQUE_LIMIT", 0.18),
                gripper_min_contact_delta=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_CONTACT_MIN_TARGET_DELTA_POS", 0.03),
                gripper_target_tolerance=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_TOLERANCE_POS", 0.04),
                require_contact_for_lift=_truthy_env("OPENFORGE_GRASP_LIFT_REQUIRE_CLOSE_CONTACT", not grasp_only),
                open_confirm_min=_env_float("OPENFORGE_GRASP_LIFT_OPEN_CONFIRM_POS", 0.55),
                open_confirm_timeout_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_CONFIRM_TIMEOUT_S", 3.0),
                state_poll_s=_env_float("OPENFORGE_GRASP_LIFT_STATE_POLL_S", 0.2),
                ik_error_threshold=_env_float("OPENFORGE_GRASP_LIFT_IK_ERROR_THRESHOLD_M", 0.015),
                ik_rot_threshold_deg=_env_float("OPENFORGE_GRASP_LIFT_IK_ROT_THRESHOLD_DEG", 8.0),
                ik_xyz_weight=_env_float("OPENFORGE_GRASP_LIFT_IK_XYZ_WEIGHT", 1.0),
                ik_rpy_weight=_env_float("OPENFORGE_GRASP_LIFT_IK_RPY_WEIGHT", 0.25),
                motion_gripper_pos=_env_float("OPENFORGE_GRASP_LIFT_MOTION_GRIPPER_POS", 1.0),
                preview_before_each_move=_truthy_env("OPENFORGE_GRASP_LIFT_PREVIEW_BEFORE_EACH_MOVE", True),
                execute_lift=not grasp_only,
            )
            attempt["execution"] = execution
            attempt["gripper_after_close"] = execution.get("gripper_after_close")
            attempt["physical_motion_executed"] = bool(execution.get("physical_motion_executed"))
            result["physical_motion_executed"] = bool(result["physical_motion_executed"] or execution.get("physical_motion_executed"))
            result["movement_capable_calls"].extend(execution.get("movement_capable_calls") or [])
            result["gripper_after_close"] = attempt["gripper_after_close"]

            post_obs, post_det = _capture_prompt_fallback(
                run_dir=run_dir,
                stage=f"attempt_{attempt_index:02d}_post_observe",
                prompts=[pre_det.get("prompt") or prompts[0]] + [p for p in prompts if p != pre_det.get("prompt")],
                cameras=["top"],
                min_score=min_score,
            )
            motion = compare_detection_motion(pre_det, post_det)
            if grasp_only:
                verification = _verify_grasp_only(
                    pre_detection=pre_det,
                    post_detection=post_det,
                    gripper_after_close=attempt["gripper_after_close"],
                )
            else:
                verification = verify_lift(
                    pre_det,
                    post_det,
                    min_z_delta_m=_env_float("OPENFORGE_GRASP_LIFT_SUCCESS_MIN_Z_DELTA_M", 0.035),
                    max_xy_delta_m=_env_float("OPENFORGE_GRASP_LIFT_SUCCESS_MAX_XY_DELTA_M", 0.08),
                )
            attempt["post_observation"] = post_obs
            attempt["post_detection"] = post_det
            attempt["object_motion"] = motion
            attempt["verification"] = verification
            attempt["success"] = bool(execution.get("success") and verification.get("success"))
            result["post_observation"] = post_obs
            result["post_detection"] = post_det
            result["object_motion"] = motion
            result["verification"] = verification
            result["video_paths"] = _video_paths(run_dir)
            if attempt["success"]:
                result["success"] = True
                result["reward"] = 1.0
                result["confidence"] = float(verification.get("confidence") or 0.8)
                if grasp_only:
                    result["why_stopped"] = "verified stable grasp-only close from gripper state and post-action live top-camera observation"
                    result["next_recommendation"] = "Inspect grasp video/overlay before deciding whether to authorize a small lift."
                else:
                    result["why_stopped"] = "verified bottle lift from post-action live top-camera observation"
                    result["next_recommendation"] = "Stop; task is verified complete."
                if (not grasp_only) and _truthy_env("OPENFORGE_GRASP_LIFT_RETURN_AFTER_VERIFIED_LIFT", False):
                    return_result = _return_after_verified_lift(selected=selected, freespace_move=freespace_move)
                    attempt["return_after_verified_lift"] = return_result
                    result["return_after_verified_lift"] = return_result
                    result["movement_capable_calls"].extend(return_result.get("movement_capable_calls") or [])
                    final_obs, final_det = _capture_prompt_fallback(
                        run_dir=run_dir,
                        stage=f"attempt_{attempt_index:02d}_post_return_observe",
                        prompts=[pre_det.get("prompt") or prompts[0]] + [p for p in prompts if p != pre_det.get("prompt")],
                        cameras=["top"],
                        min_score=min_score,
                    )
                    attempt["post_return_observation"] = final_obs
                    attempt["post_return_detection"] = final_det
                    result["post_return_observation"] = final_obs
                    if return_result.get("success"):
                        result["why_stopped"] = f"verified bottle lift; {return_result.get('why_stopped')}"
                        result["next_recommendation"] = "Stop; lift, release, retreat, and return are complete."
                    else:
                        result["success"] = False
                        result["reward"] = 0.0
                        result["confidence"] = min(float(result.get("confidence") or 0.0), 0.2)
                        result["why_stopped"] = "lift verified, but return/release stage failed"
                        result["next_recommendation"] = "Operator should inspect the held object before any recovery."
                break

            result["why_stopped"] = execution.get("why_stopped") or verification.get("reason") or "attempt failed"
            if execution.get("success"):
                result["next_recommendation"] = (
                    "Lift command ran but post-observation did not verify the requested object motion; "
                    "inspect videos before any recovery or retry."
                )
                if (not grasp_only) and _truthy_env("OPENFORGE_GRASP_LIFT_RETURN_AFTER_LIFT_COMMAND", False):
                    return_result = _return_after_verified_lift(selected=selected, freespace_move=freespace_move)
                    attempt["return_after_lift_command"] = return_result
                    result["return_after_lift_command"] = return_result
                    result["movement_capable_calls"].extend(return_result.get("movement_capable_calls") or [])
                    result["physical_motion_executed"] = bool(
                        result["physical_motion_executed"] or return_result.get("movement_capable_calls")
                    )
                    final_obs, final_det = _capture_prompt_fallback(
                        run_dir=run_dir,
                        stage=f"attempt_{attempt_index:02d}_post_return_observe",
                        prompts=[pre_det.get("prompt") or prompts[0]] + [p for p in prompts if p != pre_det.get("prompt")],
                        cameras=["top"],
                        min_score=min_score,
                    )
                    attempt["post_return_observation"] = final_obs
                    attempt["post_return_detection"] = final_det
                    result["post_return_observation"] = final_obs
                    if return_result.get("success"):
                        result["why_stopped"] = (
                            "lift command completed; verification was inconclusive; "
                            f"{return_result.get('why_stopped')}"
                        )
                        result["next_recommendation"] = "Inspect lift, release, retreat, and return video before another retry."
                    else:
                        result["why_stopped"] = "lift command completed; verification was inconclusive; return/release stage failed"
                        result["next_recommendation"] = "Operator should inspect the held object before any recovery."
            else:
                result["next_recommendation"] = "Inspect execution artifacts; retry only after no-motion analysis."
                if _truthy_env("OPENFORGE_GRASP_LIFT_RECOVER_HOME_AFTER_SAFE_FAIL", True) and _safe_to_recover_home(
                    execution,
                    verification,
                ):
                    recovery = _recover_open_home(run_dir, reason="failed before close/lift; gripper not holding")
                    attempt["post_failure_recovery"] = recovery
                    result["post_failure_recovery"] = recovery
                    result["movement_capable_calls"].extend(recovery.get("movement_capable_calls") or [])
                    result["physical_motion_executed"] = bool(
                        result["physical_motion_executed"] or recovery.get("movement_capable_calls")
                    )
                    result["next_recommendation"] = (
                        "Recovered home after a no-hold failure; use a changed approach/yaw and fresh observation."
                    )
            break
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json and run log before any physical retry."
        result["error"] = result["why_stopped"]
    finally:
        result["video_paths"] = _video_paths(run_dir)
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        for attempt in result.get("attempts") or []:
            write_json(run_dir / "attempts" / f"attempt_{int(attempt['attempt_index']):02d}.json", attempt)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- why_stopped: {result['why_stopped']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(_json_safe(result), indent=2))
    return result


TASK_RESULT.update(_run())
