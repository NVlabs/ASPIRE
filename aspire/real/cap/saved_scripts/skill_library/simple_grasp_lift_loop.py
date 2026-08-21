# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-command observe/plan/grasp/lift loop for simple YAM objects.

This helper intentionally uses lower-level YAM tools directly.  It does not use
``skill_library.pick_place.pick_and_place``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from skill_library.debug_observation import (
    capture_observation,
    current_run_dir,
    save_observation_packet,
    save_plan_packet,
    write_stage_summary,
)


JSON = dict[str, Any]


@dataclass
class GraspLiftConfig:
    task_name: str
    object_kind: str
    prompts: list[str]
    cameras: list[str]
    post_cameras: list[str]
    prompt_env: str
    manual_xyz_env: str
    manual_half_extents_env: str
    allow_physical: bool
    plan_only: bool
    attempt_budget: int
    min_detection_score: float
    max_candidates: int
    max_preview_candidates: int
    planner_backend: str
    solver_speed: str
    planning_speed: float
    ik_error_threshold_m: float
    ik_rot_threshold_deg: float
    ik_xyz_weight: float
    ik_rpy_weight: float
    pregrasp_z_m: float
    lift_z_m: float
    side_pregrasp_standoff_m: float
    body_grasp_fraction: float
    live_detection_z_to_body_offset_m: float
    topdown_only: bool
    hold_s: float
    object_width_m: float
    object_radius_m: float
    gripper_width_clearance_m: float
    gripper_open_width_m: float
    gripper_closed_width_m: float
    close_width_compression_m: float
    motion_gripper_pos: float
    open_target_pos: float
    open_vel_limit: float
    open_torque_limit: float
    close_target_pos: float
    close_target_cap_pos: float
    close_scout_pos: float
    close_vel_limit: float
    close_torque_limit: float
    close_confirm_timeout_s: float
    close_target_tolerance_pos: float
    close_contact_min_target_delta_pos: float
    close_require_contact_before_lift: bool
    close_min_delta_pos: float
    open_confirm_pos: float
    open_confirm_timeout_s: float
    state_poll_s: float
    gripper_plausible_min_pos: float
    gripper_plausible_max_pos: float
    success_min_z_delta_m: float
    success_max_xy_to_lift_m: float
    success_allow_mixed_real_sources: bool
    timeout_s: float
    requested_arm: str


def build_config(
    *,
    task_name: str,
    object_kind: str,
    default_prompts: list[str],
    prompt_env: str,
    manual_xyz_env: str,
    manual_half_extents_env: str,
) -> GraspLiftConfig:
    prompts = _csv(
        os.environ.get(prompt_env)
        or os.environ.get("OPENFORGE_GRASP_LIFT_PROMPTS")
        or ",".join(default_prompts)
    )
    cameras = _csv(os.environ.get("OPENFORGE_DEBUG_OBS_CAMERAS") or "top,left,right")
    post_cameras = _csv(
        os.environ.get("OPENFORGE_GRASP_LIFT_POST_OBS_CAMERAS") or ",".join(cameras)
    )
    allow_physical = _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION")
    explicit_plan_only = _truthy_env("OPENFORGE_GRASP_LIFT_PLAN_ONLY")
    is_bottle = object_kind.strip().lower() == "bottle"
    default_lift_z_m = 0.05 if is_bottle else 0.08
    default_side_standoff_m = 0.08 if is_bottle else 0.10
    default_body_fraction = 0.65 if is_bottle else 0.48
    default_live_detection_z_to_body_offset_m = -0.10 if is_bottle else 0.0
    default_radius_m = 0.04 if is_bottle else 0.035
    default_clearance_m = 0.03 if is_bottle else 0.025
    default_close_target_pos = 0.48 if is_bottle else 0.72
    default_close_target_cap_pos = 0.48 if is_bottle else 1.0
    default_close_scout_pos = 0.78 if is_bottle else 0.86
    default_close_compression_m = 0.004 if is_bottle else 0.006
    default_solver_speed = "slow" if is_bottle else "fast"
    return GraspLiftConfig(
        task_name=task_name,
        object_kind=object_kind,
        prompts=prompts,
        cameras=cameras,
        post_cameras=post_cameras,
        prompt_env=prompt_env,
        manual_xyz_env=manual_xyz_env,
        manual_half_extents_env=manual_half_extents_env,
        allow_physical=allow_physical,
        plan_only=(not allow_physical) or explicit_plan_only,
        attempt_budget=max(1, _env_int("OPENFORGE_GRASP_LIFT_ATTEMPT_BUDGET", 2)),
        min_detection_score=_env_float("OPENFORGE_GRASP_LIFT_MIN_DETECTION_SCORE", 0.25),
        max_candidates=max(1, _env_int("OPENFORGE_GRASP_LIFT_MAX_CANDIDATES", 160 if is_bottle else 24)),
        max_preview_candidates=max(1, _env_int("OPENFORGE_GRASP_LIFT_MAX_PREVIEW_CANDIDATES", 96 if is_bottle else 10)),
        planner_backend=os.environ.get("OPENFORGE_GRASP_LIFT_PLANNER_BACKEND", "curobo").strip(),
        solver_speed=os.environ.get("OPENFORGE_GRASP_LIFT_SOLVER_SPEED", default_solver_speed).strip(),
        planning_speed=_env_float("OPENFORGE_GRASP_LIFT_PLANNING_SPEED", 0.35),
        ik_error_threshold_m=_env_float("OPENFORGE_GRASP_LIFT_IK_ERROR_THRESHOLD_M", 0.015),
        ik_rot_threshold_deg=_env_float("OPENFORGE_GRASP_LIFT_IK_ROT_THRESHOLD_DEG", 8.0),
        ik_xyz_weight=_env_float("OPENFORGE_GRASP_LIFT_IK_XYZ_WEIGHT", 1.0),
        ik_rpy_weight=_env_float("OPENFORGE_GRASP_LIFT_IK_RPY_WEIGHT", 0.25),
        pregrasp_z_m=_env_float("OPENFORGE_GRASP_LIFT_PREGRASP_Z_M", 0.06),
        lift_z_m=_env_float("OPENFORGE_GRASP_LIFT_LIFT_Z_M", default_lift_z_m),
        side_pregrasp_standoff_m=_env_float(
            "OPENFORGE_GRASP_LIFT_SIDE_PREGRASP_STANDOFF_M",
            default_side_standoff_m,
        ),
        body_grasp_fraction=_env_float(
            "OPENFORGE_GRASP_LIFT_BODY_GRASP_FRACTION",
            default_body_fraction,
        ),
        live_detection_z_to_body_offset_m=_env_float(
            "OPENFORGE_GRASP_LIFT_LIVE_DETECTION_Z_TO_BODY_OFFSET_M",
            default_live_detection_z_to_body_offset_m,
        ),
        topdown_only=_truthy_env("OPENFORGE_GRASP_LIFT_TOPDOWN_ONLY"),
        hold_s=_env_float("OPENFORGE_GRASP_LIFT_HOLD_S", 0.5),
        object_width_m=_env_float("OPENFORGE_GRASP_LIFT_OBJECT_WIDTH_M", 0.08),
        object_radius_m=_env_float("OPENFORGE_GRASP_LIFT_OBJECT_RADIUS_M", default_radius_m),
        gripper_width_clearance_m=_env_float(
            "OPENFORGE_GRASP_LIFT_GRIPPER_WIDTH_CLEARANCE_M",
            default_clearance_m,
        ),
        gripper_open_width_m=_env_float("OPENFORGE_GRASP_LIFT_GRIPPER_OPEN_WIDTH_M", 0.085),
        gripper_closed_width_m=_env_float("OPENFORGE_GRASP_LIFT_GRIPPER_CLOSED_WIDTH_M", 0.004),
        close_width_compression_m=_env_float(
            "OPENFORGE_GRASP_LIFT_CLOSE_WIDTH_COMPRESSION_M",
            default_close_compression_m,
        ),
        motion_gripper_pos=_env_float("OPENFORGE_GRASP_LIFT_MOTION_GRIPPER_POS", 1.0),
        open_target_pos=_env_float("OPENFORGE_GRASP_LIFT_OPEN_TARGET_POS", 1.0),
        open_vel_limit=_env_float("OPENFORGE_GRASP_LIFT_OPEN_VEL_LIMIT", 0.8),
        open_torque_limit=_env_float("OPENFORGE_GRASP_LIFT_OPEN_TORQUE_LIMIT", 0.30),
        close_target_pos=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_POS", default_close_target_pos),
        close_target_cap_pos=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_CAP_POS", default_close_target_cap_pos),
        close_scout_pos=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_SCOUT_POS", default_close_scout_pos),
        close_vel_limit=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_VEL_LIMIT", 0.45),
        close_torque_limit=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TORQUE_LIMIT", 0.18),
        close_confirm_timeout_s=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_CONFIRM_TIMEOUT_S", 1.5),
        close_target_tolerance_pos=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_TOLERANCE_POS", 0.12),
        close_contact_min_target_delta_pos=_env_float(
            "OPENFORGE_GRASP_LIFT_CLOSE_CONTACT_MIN_TARGET_DELTA_POS",
            0.03 if is_bottle else 0.0,
        ),
        close_require_contact_before_lift=_truthy_env(
            "OPENFORGE_GRASP_LIFT_REQUIRE_CLOSE_CONTACT",
            default=is_bottle,
        ),
        close_min_delta_pos=_env_float("OPENFORGE_GRASP_LIFT_CLOSE_MIN_DELTA_POS", 0.02),
        open_confirm_pos=_env_float("OPENFORGE_GRASP_LIFT_OPEN_CONFIRM_POS", 0.55),
        open_confirm_timeout_s=_env_float("OPENFORGE_GRASP_LIFT_OPEN_CONFIRM_TIMEOUT_S", 3.0),
        state_poll_s=_env_float("OPENFORGE_GRASP_LIFT_STATE_POLL_S", 0.2),
        gripper_plausible_min_pos=_env_float("OPENFORGE_GRASP_LIFT_GRIPPER_MIN_PLAUSIBLE_POS", 0.04),
        gripper_plausible_max_pos=_env_float("OPENFORGE_GRASP_LIFT_GRIPPER_MAX_PLAUSIBLE_POS", 0.93),
        success_min_z_delta_m=_env_float("OPENFORGE_GRASP_LIFT_SUCCESS_MIN_Z_DELTA_M", 0.035),
        success_max_xy_to_lift_m=_env_float("OPENFORGE_GRASP_LIFT_SUCCESS_MAX_XY_TO_LIFT_M", 0.08),
        success_allow_mixed_real_sources=_truthy_env("OPENFORGE_GRASP_LIFT_SUCCESS_ALLOW_MIXED_REAL_SOURCES"),
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 10.0),
        requested_arm=os.environ.get("OPENFORGE_GRASP_LIFT_ARM", "auto").strip().lower(),
    )


def run_one_shot_loop(config: GraspLiftConfig) -> JSON:
    run_dir = current_run_dir(default=f"logs/{config.task_name}_{_stamp()}")
    result = _initial_result(config, run_dir)
    attempted_keys: set[str] = set()
    why_stopped = "not_started"

    try:
        if config.allow_physical and _truthy_env("OPENFORGE_GRASP_LIFT_RECOVERY_GO_HOME_FIRST"):
            recovery = _pre_recovery_open_home(config)
            result["pre_recovery"] = recovery
            result["physical_motion_executed"] = bool(recovery.get("movement_capable_calls"))
            result["movement_capable_calls"] = list(recovery.get("movement_capable_calls", []))
            _write_result(run_dir, result)
            if not recovery.get("success"):
                why_stopped = "pre-recovery open/go_home failed; no grasp attempt started"
                result["why_stopped"] = why_stopped
                result["next_recommendation"] = "Inspect recovery error and robot state before any retry."
                return _finish_result(run_dir, result, why_stopped)

        for attempt_index in range(1, config.attempt_budget + 1):
            attempt = _new_attempt(attempt_index, config)
            result["attempt_count"] = attempt_index
            result["attempts"].append(attempt)

            pre_obs = _observe(config, run_dir, f"attempt_{attempt_index:02d}_pre_observe")
            pre_det = _selected_detection(pre_obs, config, allow_manual=not config.allow_physical)
            use_real_verification = config.allow_physical or _truthy_env("OPENFORGE_GRASP_LIFT_MANUAL_PLUS_CAMERA")
            pre_verify_det = (
                _selected_detection(pre_obs, config, allow_manual=False)
                if use_real_verification
                else pre_det
            )
            attempt["pre_observation"] = _obs_ref(pre_obs)
            attempt["pre_detection"] = pre_det
            attempt["pre_verification_detection"] = pre_verify_det
            result["pre_observation"] = attempt["pre_observation"]
            result["pre_detection"] = pre_det
            result["pre_verification_detection"] = pre_verify_det
            if pre_det is not None:
                result["detection_prompt_used"] = pre_det.get("prompt")
                result["detection_source_camera"] = pre_det.get("source_camera")

            if pre_det is None:
                why_stopped = "no calibrated prompt detection; no candidate can be planned"
                attempt["why_stopped"] = why_stopped
                attempt["next_recommendation"] = (
                    f"Re-observe with {config.prompt_env} set to a prompt visible from "
                    "top/left/right, or reset the object into calibrated view."
                )
                _write_attempt(run_dir, attempt)
                break

            candidates = _generate_candidates(config, pre_det, attempted_keys)
            attempt["candidate_count"] = len(candidates)
            attempt["candidate_preview"] = candidates[: min(8, len(candidates))]
            if not candidates:
                why_stopped = "no grasp candidates generated from the selected detection"
                attempt["why_stopped"] = why_stopped
                attempt["next_recommendation"] = "Check object pose/height and candidate-generation knobs."
                _write_attempt(run_dir, attempt)
                break

            selected, previews, plan_path = _preview_and_rank(config, candidates, pre_obs, run_dir)
            attempt["planner_result"] = {
                "plan_packet": str(plan_path),
                "preview_count": len(previews),
                "selected": selected,
                "preview_failures": _preview_failures(previews),
            }
            result["selected_strategy"] = selected
            if selected is None:
                why_stopped = "no candidate passed guarded pregrasp/grasp/lift cuRobo preview"
                attempt["why_stopped"] = why_stopped
                attempt["next_recommendation"] = (
                    "Inspect plans and overlays; then adjust approach offsets, arm, "
                    "or object layout before a physical ticket."
                )
                _write_attempt(run_dir, attempt)
                break

            attempt["selected_strategy"] = selected
            attempted_keys.add(_candidate_key(selected))
            if config.plan_only:
                why_stopped = (
                    "no-motion plan-only run completed; physical grasp/lift requires "
                    "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 and a bounded ticket"
                )
                attempt["success"] = False
                attempt["plan_success"] = True
                attempt["why_stopped"] = why_stopped
                attempt["next_recommendation"] = _ticket_recommendation(config)
                result["plan_success"] = True
                result["confidence"] = 0.0
                result["next_recommendation"] = attempt["next_recommendation"]
                _write_attempt(run_dir, attempt)
                break

            execution = _execute_physical_attempt(config, selected, attempt)
            attempt["execution"] = execution
            attempt["gripper_after_close"] = execution.get("gripper_after_close")
            result["gripper_after_close"] = attempt["gripper_after_close"]
            result["physical_motion_executed"] = bool(execution.get("movement_capable_calls"))
            result["movement_capable_calls"] = list(execution.get("movement_capable_calls", []))

            post_obs = _observe(
                config,
                run_dir,
                f"attempt_{attempt_index:02d}_post_observe",
                cameras=config.post_cameras,
            )
            pre_source_camera = str((pre_verify_det or pre_det or {}).get("source_camera") or "")
            post_det = _selected_detection(
                post_obs,
                config,
                allow_manual=not config.allow_physical,
                preferred_camera=pre_source_camera or None,
            )
            attempt["post_observation"] = _obs_ref(post_obs)
            attempt["post_detection"] = post_det
            attempt["post_detection_preferred_camera"] = pre_source_camera or None
            result["post_observation"] = attempt["post_observation"]

            judgment = _judge_attempt(config, selected, pre_verify_det or pre_det, post_det, execution)
            attempt["object_motion"] = judgment.get("object_motion")
            attempt["success_evidence"] = judgment
            attempt["success"] = bool(judgment.get("success"))
            attempt["confidence"] = float(judgment.get("confidence", 0.0) or 0.0)
            attempt["why_stopped"] = judgment.get("why_stopped")
            attempt["next_recommendation"] = judgment.get("next_recommendation")
            result["object_motion"] = attempt["object_motion"]
            result["video_paths"] = _video_paths(run_dir)

            if attempt["success"]:
                _write_attempt(run_dir, attempt)
                result["success"] = True
                result["reward"] = 1.0
                result["confidence"] = attempt["confidence"]
                why_stopped = "verified grasp/lift success from post-observation and gripper evidence"
                result["next_recommendation"] = "Stop; task is verified complete."
                break

            if _should_recover_after_failed_attempt(config, execution, judgment):
                recovery = _pre_recovery_open_home(config)
                attempt["post_failure_recovery"] = recovery
                result["post_failure_recovery"] = recovery
                result["movement_capable_calls"].extend(recovery.get("movement_capable_calls", []))
                result["physical_motion_executed"] = result["physical_motion_executed"] or bool(
                    recovery.get("movement_capable_calls")
                )
                post_recovery_obs = _observe(
                    config,
                    run_dir,
                    f"attempt_{attempt_index:02d}_post_failure_recovery_observe",
                )
                attempt["post_failure_recovery_observation"] = _obs_ref(post_recovery_obs)
                result["post_observation"] = attempt["post_failure_recovery_observation"]
                if not recovery.get("success"):
                    attempt["next_recommendation"] = "Recovery open/go_home failed; inspect robot state before retry."
                    result["next_recommendation"] = attempt["next_recommendation"]
                    _write_attempt(run_dir, attempt)
                    why_stopped = "post-failure recovery open/go_home failed"
                    break

            _write_attempt(run_dir, attempt)

            if execution.get("blocker"):
                why_stopped = str(execution["blocker"])
                result["next_recommendation"] = attempt["next_recommendation"]
                break

            if attempt_index >= config.attempt_budget:
                why_stopped = "attempt budget exhausted without verified post-action success"
                result["next_recommendation"] = attempt["next_recommendation"]
                break

            result["next_recommendation"] = (
                "Retrying inside the same bounded script invocation with fresh observation "
                "and the previous selected strategy deprioritized."
            )

    except Exception as exc:
        why_stopped = f"loop failed: {type(exc).__name__}: {exc}"
        result["risk_notes"].append("Inspect result.json, stage_summary.md, observations, and plans before any retry.")
        result["error"] = why_stopped
    finally:
        result["why_stopped"] = why_stopped
        result["video_paths"] = _video_paths(run_dir)
        _write_result(run_dir, result)
        write_stage_summary(stage="result", result=result, log_dir=run_dir)
        print(f"[{config.task_name}] {why_stopped}")
        print(json.dumps(_json_safe(result), indent=2))

    return result


def _initial_result(config: GraspLiftConfig, run_dir: Path) -> JSON:
    return {
        "success": False,
        "reward": 0.0,
        "confidence": 0.0,
        "method": config.task_name,
        "object_kind": config.object_kind,
        "prompts": config.prompts,
        "cameras": config.cameras,
        "post_cameras": config.post_cameras,
        "run_dir": str(run_dir),
        "plan_success": False,
        "attempt_budget": config.attempt_budget,
        "attempt_count": 0,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "pre_recovery": None,
        "post_failure_recovery": None,
        "selected_strategy": None,
        "pre_observation": None,
        "pre_detection": None,
        "pre_verification_detection": None,
        "detection_prompt_used": None,
        "detection_source_camera": None,
        "post_observation": None,
        "gripper_after_close": None,
        "object_motion": None,
        "video_paths": [],
        "why_stopped": "not_started",
        "next_recommendation": _ticket_recommendation(config),
        "attempts": [],
        "risk_notes": [
            "Default mode is no-motion plan-only unless OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is set.",
            "A selected cuRobo preview is not task success; success requires post-action evidence.",
            "Mock AnyGrasp is not used by this helper.",
        ],
        "config": _config_summary(config),
    }


def _config_summary(config: GraspLiftConfig) -> JSON:
    return {
        "prompt_env": config.prompt_env,
        "manual_xyz_env": config.manual_xyz_env,
        "manual_half_extents_env": config.manual_half_extents_env,
        "plan_only": config.plan_only,
        "cameras": config.cameras,
        "post_cameras": config.post_cameras,
        "allow_physical_env": config.allow_physical,
        "planner_backend": config.planner_backend,
        "solver_speed": config.solver_speed,
        "planning_speed": config.planning_speed,
        "ik_error_threshold_m": config.ik_error_threshold_m,
        "ik_rot_threshold_deg": config.ik_rot_threshold_deg,
        "pregrasp_z_m": config.pregrasp_z_m,
        "lift_z_m": config.lift_z_m,
        "side_pregrasp_standoff_m": config.side_pregrasp_standoff_m,
        "body_grasp_fraction": config.body_grasp_fraction,
        "live_detection_z_to_body_offset_m": config.live_detection_z_to_body_offset_m,
        "topdown_only": config.topdown_only,
        "max_candidates": config.max_candidates,
        "max_preview_candidates": config.max_preview_candidates,
        "object_width_m": config.object_width_m,
        "object_radius_m": config.object_radius_m,
        "gripper_width_clearance_m": config.gripper_width_clearance_m,
        "motion_gripper_pos": config.motion_gripper_pos,
        "gripper_open_width_m": config.gripper_open_width_m,
        "gripper_closed_width_m": config.gripper_closed_width_m,
        "close_width_compression_m": config.close_width_compression_m,
        "close_target_pos": config.close_target_pos,
        "close_target_cap_pos": config.close_target_cap_pos,
        "close_scout_pos": config.close_scout_pos,
        "close_vel_limit": config.close_vel_limit,
        "close_torque_limit": config.close_torque_limit,
        "close_confirm_timeout_s": config.close_confirm_timeout_s,
        "close_contact_min_target_delta_pos": config.close_contact_min_target_delta_pos,
        "close_require_contact_before_lift": config.close_require_contact_before_lift,
        "close_min_delta_pos": config.close_min_delta_pos,
        "success_allow_mixed_real_sources": config.success_allow_mixed_real_sources,
        "requested_arm": config.requested_arm,
    }


def _new_attempt(index: int, config: GraspLiftConfig) -> JSON:
    return {
        "attempt_index": index,
        "plan_only": config.plan_only,
        "success": False,
        "confidence": 0.0,
        "selected_strategy": None,
        "pre_observation": None,
        "pre_verification_detection": None,
        "post_observation": None,
        "planner_result": None,
        "gripper_after_close": None,
        "object_motion": None,
        "why_stopped": "not_started",
        "next_recommendation": None,
    }


def _observe(config: GraspLiftConfig, run_dir: Path, stage: str, *, cameras: list[str] | None = None) -> JSON:
    cameras = list(cameras or config.cameras)
    manual_debug = _manual_detection(config)
    manual_ignored_for_physical = bool(config.allow_physical and manual_debug is not None)
    manual = None if config.allow_physical else manual_debug
    detect_fn = _tool("detect_objects_oneshot")
    get_state_fn = _tool("get_robot_state")
    if manual is not None:
        manual_packet = _manual_observation_packet(config, manual, stage, run_dir, save=False)
        use_real_cameras = config.allow_physical or _truthy_env("OPENFORGE_GRASP_LIFT_MANUAL_PLUS_CAMERA")
        if not use_real_cameras or detect_fn is None:
            _save_manual_like_packet(manual_packet, run_dir)
            return manual_packet
        try:
            packet = capture_observation(
                stage=stage,
                prompts=config.prompts,
                cameras=cameras,
                detect_fn=detect_fn,
                get_robot_state_fn=get_state_fn,
                log_dir=run_dir,
                capture_robot_state=True,
                per_call_timeout_s=config.timeout_s,
            )
            packet.setdefault("cameras", {})
            packet["cameras"] = {
                "manual_env": manual_packet["cameras"]["manual_env"],
                **packet["cameras"],
            }
            packet["manual_detection_present"] = True
            if packet.get("packet_path"):
                Path(str(packet["packet_path"])).write_text(
                    json.dumps(_json_safe(packet), indent=2) + "\n",
                    encoding="utf-8",
                )
            write_stage_summary(stage=stage, observation=packet, log_dir=run_dir)
            return packet
        except Exception as exc:
            manual_packet.setdefault("errors", []).append(
                f"real_camera_observation_failed: {type(exc).__name__}: {exc}"
            )
            _save_manual_like_packet(manual_packet, run_dir)
            return manual_packet

    if detect_fn is None:
        packet = {
            "schema": "openforge.observation.v1",
            "stage": stage,
            "stamp": _stamp(),
            "physical_motion_executed": False,
            "prompts": list(config.prompts),
            "cameras": {
                "unavailable": {
                    "detections": {prompt: [] for prompt in config.prompts},
                    "errors": [
                        "detect_objects_oneshot unavailable; live perception is required for physical motion"
                        if config.allow_physical
                        else "detect_objects_oneshot unavailable; run through run_script.py or use a labeled no-motion manual XYZ debug check"
                    ],
                }
            },
            "robot_state": _read_robot_state(),
            "errors": [
                "manual XYZ override ignored because physical planning requires fresh live detection"
            ]
            if manual_ignored_for_physical
            else [],
        }
        _save_manual_like_packet(packet, run_dir)
        return packet

    packet = capture_observation(
        stage=stage,
        prompts=config.prompts,
        cameras=cameras,
        detect_fn=detect_fn,
        get_robot_state_fn=get_state_fn,
        log_dir=run_dir,
        capture_robot_state=True,
        per_call_timeout_s=config.timeout_s,
    )
    if manual_ignored_for_physical:
        packet["manual_xyz_ignored_for_physical"] = True
        packet.setdefault("errors", []).append(
            "manual XYZ override ignored because physical planning requires fresh live detection"
        )
        if packet.get("packet_path"):
            Path(str(packet["packet_path"])).write_text(
                json.dumps(_json_safe(packet), indent=2) + "\n",
                encoding="utf-8",
            )
    return packet


def _manual_observation_packet(
    config: GraspLiftConfig,
    detection: JSON,
    stage: str,
    run_dir: Path,
    *,
    save: bool = True,
) -> JSON:
    first_prompt = str(detection.get("prompt") or config.prompts[0])
    detections = {prompt: [] for prompt in config.prompts}
    detections.setdefault(first_prompt, [])
    detections[first_prompt] = [detection]
    packet = {
        "schema": "openforge.observation.v1",
        "stage": stage,
        "stamp": _stamp(),
        "physical_motion_executed": False,
        "prompts": list(config.prompts),
        "cameras": {
            "manual_env": {
                "detections": detections,
                "errors": [],
            }
        },
        "robot_state": _read_robot_state(),
        "errors": [],
    }
    if save:
        _save_manual_like_packet(packet, run_dir)
    return packet


def _save_manual_like_packet(packet: JSON, run_dir: Path) -> None:
    path = save_observation_packet(packet, log_dir=run_dir)
    packet["packet_path"] = str(path)
    write_stage_summary(stage=str(packet.get("stage")), observation=packet, log_dir=run_dir)


def _manual_detection(config: GraspLiftConfig) -> JSON | None:
    xyz_raw = os.environ.get(config.manual_xyz_env) or os.environ.get("OPENFORGE_GRASP_LIFT_OBJECT_XYZ")
    xyz = _float_list(xyz_raw, 3) if xyz_raw else None
    if xyz is None:
        return None
    half_raw = os.environ.get(config.manual_half_extents_env) or os.environ.get(
        "OPENFORGE_GRASP_LIFT_OBJECT_HALF_EXTENTS"
    )
    half_extents = _float_list(half_raw, 3) if half_raw else []
    rpy_raw = os.environ.get("OPENFORGE_GRASP_LIFT_OBJECT_RPY")
    yaw_raw = os.environ.get("OPENFORGE_GRASP_LIFT_OBJECT_YAW_DEG")
    rpy = _float_list(rpy_raw, 3) if rpy_raw else None
    if rpy is None and yaw_raw is not None and yaw_raw.strip():
        rpy = [0.0, 0.0, float(yaw_raw)]
    prompt = config.prompts[0]
    detection = {
        "prompt": prompt,
        "label": prompt,
        "score": 1.0,
        "position_3d": xyz,
        "half_extents": half_extents,
        "source": "manual_env",
    }
    if rpy is not None:
        detection["rpy"] = rpy
    return detection


def _selected_detection(
    packet: JSON,
    config: GraspLiftConfig,
    *,
    allow_manual: bool = True,
    preferred_camera: str | None = None,
) -> JSON | None:
    camera_rank = {camera: idx for idx, camera in enumerate(["manual_env", "top", "left", "right"])}
    camera_passes: list[str | None] = [None]
    if preferred_camera:
        camera_passes.insert(0, preferred_camera)
    for camera_filter in camera_passes:
        for prompt_index, prompt in enumerate(config.prompts):
            best: tuple[float, int, JSON] | None = None
            for camera, data in (packet.get("cameras") or {}).items():
                if camera not in {"manual_env", "top", "left", "right"}:
                    continue
                if camera_filter is not None and camera != camera_filter:
                    continue
                if camera == "manual_env" and not allow_manual:
                    continue
                detections_by_prompt = data.get("detections") or {}
                for det in detections_by_prompt.get(prompt) or []:
                    position = det.get("position_3d") or det.get("position")
                    if not _valid_xyz(position):
                        continue
                    score = det.get("score")
                    score_value = float(score) if score is not None else config.min_detection_score
                    if score_value < config.min_detection_score:
                        continue
                    enriched = dict(det)
                    enriched["prompt"] = prompt
                    enriched["source_camera"] = camera
                    enriched["prompt_index"] = prompt_index
                    enriched["prompt_fallback_used"] = bool(prompt_index > 0)
                    enriched["preferred_camera_used"] = bool(camera_filter is not None)
                    rank_score = score_value - 0.01 * camera_rank.get(camera, 9)
                    candidate = (rank_score, -camera_rank.get(camera, 9), enriched)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
            if best is not None:
                return best[2]
    return None


def _generate_candidates(config: GraspLiftConfig, detection: JSON, attempted_keys: set[str]) -> list[JSON]:
    if config.object_kind == "bottle":
        candidates = _generate_bottle_candidates(config, detection)
    else:
        candidates = _generate_can_candidates(config, detection)
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in attempted_keys:
            candidate["score"] = round(float(candidate.get("score", 0.0)) - 0.35, 4)
            candidate["retry_adjustment"] = "previously_attempted_strategy_deprioritized"
    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return candidates[: config.max_candidates]


def _generate_can_candidates(config: GraspLiftConfig, detection: JSON) -> list[JSON]:
    center = _xyz(detection["position_3d"])
    half_extents = _half_extents(detection)
    radius = _object_radius(config, half_extents)
    gripper_width = _clip(2.0 * radius + config.gripper_width_clearance_m, 0.045, 0.095)
    estimated_width = round(float(2.0 * radius), 4)
    z_policy = (
        "height_fraction_from_half_extents"
        if len(half_extents) >= 3 and half_extents[2] > 0
        else "detection_center_z_plus_offsets"
    )
    preferred, reason = _preferred_arm(center, config)
    arms = _arm_order(preferred, config)
    yaws = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_TOPDOWN_YAWS"),
        fallback=[90.0, 0.0, -45.0, 45.0, -90.0, 135.0, -135.0, 180.0],
    )
    pitches = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_TOPDOWN_PITCHES"),
        fallback=[180.0, 165.0],
    )
    z_offsets = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_TOPDOWN_Z_OFFSETS_M"),
        fallback=[0.0, 0.01, -0.01],
    )
    candidates: list[JSON] = []
    for arm_index, arm in enumerate(arms):
        arm_bonus = 0.0 if arm_index == 0 else -0.18
        for z_index, z_offset in enumerate(z_offsets):
            for pitch_index, pitch in enumerate(pitches):
                for yaw_index, yaw in enumerate(yaws):
                    grasp_z = _body_grasp_z(config, center, half_extents, float(z_offset), detection)
                    grasp = [center[0], center[1], grasp_z]
                    score = 1.0 + arm_bonus - 0.04 * yaw_index - 0.05 * pitch_index - 0.015 * z_index
                    candidates.append(
                        _candidate_with_poses(
                            config,
                            arm=arm,
                            strategy="topdown_body",
                            grasp=grasp,
                            rpy=[0.0, float(pitch), float(yaw)],
                            width=gripper_width,
                            score=score,
                            details={
                                "geometry_policy": (
                                    "LIBERO-transfer can policy: treat the can as a cylinder, "
                                    "force top-down orientation, use cylinder body height when available, "
                                    "and sweep yaw/Z from fresh live detection."
                                ),
                                "arm_reason": reason,
                                "z_policy": z_policy,
                                "body_grasp_fraction": round(float(config.body_grasp_fraction), 4),
                                "z_offset_m": round(float(z_offset), 4),
                                "yaw_deg": round(float(yaw), 3),
                                "pitch_deg": round(float(pitch), 3),
                                "half_extents": _round_list(half_extents),
                                "estimated_radius_m": round(float(radius), 4),
                                "estimated_object_width_m": estimated_width,
                                "gripper_width_m": round(float(gripper_width), 4),
                                "source_detection": _det_ref(detection),
                            },
                        )
                    )

    if config.topdown_only:
        return candidates

    side_offsets = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_SIDE_APPROACH_OFFSETS_DEG"),
        fallback=[0.0, -20.0, 20.0],
    )
    for arm_index, arm in enumerate(arms):
        base_angle = -90.0 if arm == "left" else 90.0
        arm_bonus = -0.08 if arm_index == 0 else -0.26
        for offset_index, offset in enumerate(side_offsets):
            angle = base_angle + float(offset)
            candidates.append(
                _side_candidate(
                    config,
                    detection,
                    arm=arm,
                    approach_angle_deg=angle,
                    z_offset_m=0.0,
                    wrist_roll_deg=0.0,
                    score=0.86 + arm_bonus - 0.03 * offset_index,
                    strategy="side_body",
                    arm_reason=reason,
                )
            )
    return candidates


def _generate_bottle_candidates(config: GraspLiftConfig, detection: JSON) -> list[JSON]:
    center = _xyz(detection["position_3d"])
    preferred, reason = _preferred_arm(center, config)
    arms = _arm_order(preferred, config)
    offsets = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_SIDE_APPROACH_OFFSETS_DEG"),
        fallback=[0.0, -15.0, 15.0, -30.0, 30.0],
    )
    z_offsets = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_SIDE_Z_OFFSETS_M"),
        fallback=[0.0, 0.015, -0.015, 0.03],
    )
    wrist_rolls = _float_list(
        os.environ.get("OPENFORGE_GRASP_LIFT_SIDE_WRIST_ROLLS_DEG"),
        fallback=[0.0, -15.0, 15.0, -30.0, 30.0],
    )
    rpy_variants = _csv(os.environ.get("OPENFORGE_GRASP_LIFT_SIDE_RPY_VARIANTS") or "standard,display_pitch_flip")
    candidates: list[JSON] = []
    orientation_hints = _orientation_hints_from_detection(detection)
    absolute_angles = _absolute_approach_angles(orientation_hints)
    for arm_index, arm in enumerate(arms):
        base_angle = -90.0 if arm == "left" else 90.0
        arm_bonus = 0.0 if arm_index == 0 else -0.02
        angle_specs: list[tuple[float, str, int, float]] = [
            (base_angle + float(offset), "arm_side_yaw_sweep", offset_index, 0.004 * abs(float(offset)))
            for offset_index, offset in enumerate(offsets)
        ]
        for hint_index, absolute_angle in enumerate(absolute_angles):
            angle_delta = abs(_normalize_angle_deg(float(absolute_angle) - base_angle))
            angle_specs.append(
                (
                    float(absolute_angle),
                    "detection_orientation_yaw_hint",
                    len(angle_specs) + hint_index,
                    min(0.18, 0.003 * angle_delta),
                )
            )
        seen_angles: set[tuple[float, str]] = set()
        for approach_angle, angle_source, offset_index, angle_penalty in angle_specs:
            dedupe_key = (round(_normalize_angle_deg(approach_angle), 3), angle_source)
            if dedupe_key in seen_angles:
                continue
            seen_angles.add(dedupe_key)
            for wrist_index, wrist_roll in enumerate(wrist_rolls):
                for z_index, z_offset in enumerate(z_offsets):
                    for variant_index, rpy_variant in enumerate(rpy_variants):
                        variant = rpy_variant.strip().lower()
                        if variant not in {"standard", "display_pitch_flip"}:
                            continue
                        score = (
                            1.0
                            + arm_bonus
                            - angle_penalty
                            - 0.002 * abs(float(wrist_roll))
                            - 0.04 * z_index
                            - 0.01 * offset_index
                            - 0.015 * variant_index
                        )
                        candidates.append(
                            _side_candidate(
                                config,
                                detection,
                                arm=arm,
                                approach_angle_deg=approach_angle,
                                z_offset_m=float(z_offset),
                                wrist_roll_deg=float(wrist_roll),
                                score=score,
                                strategy="mid_body_side",
                                arm_reason=reason,
                                orientation_hints=orientation_hints,
                                angle_source=angle_source,
                                rpy_variant=variant,
                            )
                    )
    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return candidates


def _side_candidate(
    config: GraspLiftConfig,
    detection: JSON,
    *,
    arm: str,
    approach_angle_deg: float,
    z_offset_m: float,
    wrist_roll_deg: float,
    score: float,
    strategy: str,
    arm_reason: str,
    orientation_hints: JSON | None = None,
    angle_source: str = "arm_side_yaw_sweep",
    rpy_variant: str = "standard",
) -> JSON:
    center = _xyz(detection["position_3d"])
    half_extents = _half_extents(detection)
    radius = _object_radius(config, half_extents)
    grasp_z = _body_grasp_z(config, center, half_extents, z_offset_m, detection)
    grasp = [center[0], center[1], grasp_z]
    angle_rad = math.radians(float(approach_angle_deg))
    requested_approach = [math.cos(angle_rad), math.sin(angle_rad), 0.0]
    rpy, axes = _side_grasp_display_rpy_from_approach(requested_approach, wrist_roll_deg=wrist_roll_deg)
    approach = requested_approach
    requested_approach_angle_deg = float(approach_angle_deg)
    if rpy_variant == "display_pitch_flip":
        rpy = _display_pitch_flip_rpy(rpy)
        actual_axes = _display_rpy_axes(rpy)
        approach = _horizontal_unit(
            actual_axes.get("local_z_approach_axis", requested_approach),
            fallback=requested_approach,
        )
        actual_angle_deg = _normalize_angle_deg(math.degrees(math.atan2(approach[1], approach[0])))
        axes = {
            "variant": "display_pitch_flip",
            **actual_axes,
            "source_standard_axes": axes,
            "requested_approach_axis": _round_list(requested_approach),
            "actual_approach_angle_deg": round(float(actual_angle_deg), 3),
            "note": (
                "planner-reachable alternate wrist orientation; pregrasp is offset along "
                "the actual display-RPY local +Z axis"
            ),
        }
        orientation_status = (
            "mid-body side-cylinder candidate using planner-reachable display_pitch_flip "
            "wrist orientation with pregrasp derived from the actual local +Z approach axis"
        )
    else:
        orientation_status = (
            "mid-body side-cylinder grasp; local +Z follows approach, local +X spans "
            "the bottle body, and every candidate must pass pregrasp/grasp/lift preview"
        )
    width = _clip(2.0 * radius + config.gripper_width_clearance_m, 0.045, 0.095)
    pregrasp = [
        grasp[0] - approach[0] * config.side_pregrasp_standoff_m,
        grasp[1] - approach[1] * config.side_pregrasp_standoff_m,
        grasp[2],
    ]
    lift = [grasp[0], grasp[1], grasp[2] + config.lift_z_m]
    return {
        "arm": arm,
        "strategy": strategy,
        "position": _round_list(grasp),
        "rpy": _round_list(rpy, 3),
        "width": round(float(width), 4),
        "score": round(float(score), 4),
        "pregrasp_pose": {"position": _round_list(pregrasp), "rpy": _round_list(rpy, 3)},
        "grasp_pose": {"position": _round_list(grasp), "rpy": _round_list(rpy, 3)},
        "lift_pose": {"position": _round_list(lift), "rpy": _round_list(rpy, 3)},
        "approach_direction_world": _round_list(approach),
        "gripper_local_axes_world": axes,
        "approach_angle_deg": round(
            float(math.degrees(math.atan2(approach[1], approach[0]))),
            3,
        ),
        "requested_approach_angle_deg": round(float(requested_approach_angle_deg), 3),
        "approach_angle_source": angle_source,
        "rpy_variant": rpy_variant,
        "wrist_roll_deg": round(float(wrist_roll_deg), 3),
        "z_offset_m": round(float(z_offset_m), 4),
        "live_detection_z_to_body_offset_m": round(float(config.live_detection_z_to_body_offset_m), 4)
        if _uses_live_z_body_offset(config, detection, half_extents)
        else 0.0,
        "estimated_radius_m": round(float(radius), 4),
        "estimated_object_width_m": round(float(2.0 * radius), 4),
        "arm_reason": arm_reason,
        "orientation_hints": orientation_hints or _orientation_hints_from_detection(detection),
        "source_detection": _det_ref(detection),
        "orientation_status": orientation_status,
    }


def _candidate_with_poses(
    config: GraspLiftConfig,
    *,
    arm: str,
    strategy: str,
    grasp: list[float],
    rpy: list[float],
    width: float,
    score: float,
    details: JSON,
) -> JSON:
    pregrasp = [grasp[0], grasp[1], grasp[2] + config.pregrasp_z_m]
    lift = [grasp[0], grasp[1], grasp[2] + config.lift_z_m]
    return {
        "arm": arm,
        "strategy": strategy,
        "position": _round_list(grasp),
        "rpy": _round_list(rpy, 3),
        "width": round(float(width), 4),
        "score": round(float(score), 4),
        "pregrasp_pose": {"position": _round_list(pregrasp), "rpy": _round_list(rpy, 3)},
        "grasp_pose": {"position": _round_list(grasp), "rpy": _round_list(rpy, 3)},
        "lift_pose": {"position": _round_list(lift), "rpy": _round_list(rpy, 3)},
        **details,
    }


def _preview_and_rank(
    config: GraspLiftConfig,
    candidates: list[JSON],
    observation: JSON,
    run_dir: Path,
) -> tuple[JSON | None, list[JSON], Path]:
    previews: list[JSON] = []
    selected: JSON | None = None
    for index, candidate in enumerate(candidates[: config.max_preview_candidates], start=1):
        candidate = dict(candidate)
        candidate["candidate_index"] = index
        sequence: list[JSON] = []
        for pose_key in ("pregrasp_pose", "grasp_pose", "lift_pose"):
            preview = _preview_pose(config, candidate, pose_key, f"candidate_{index}_{pose_key}")
            previews.append(preview)
            sequence.append(preview)
            if not preview.get("success"):
                break
        if all(step.get("success") for step in sequence):
            selected = candidate
            selected["preview_sequence"] = [
                {
                    "label": step.get("label"),
                    "pose_key": step.get("pose_key"),
                    "status": (step.get("result") or {}).get("status"),
                    "trajectory_steps": (step.get("result") or {}).get("trajectory_steps"),
                    "final_pos_error_m": (step.get("result") or {}).get("final_pos_error_m"),
                    "final_rot_error_deg": (step.get("result") or {}).get("final_rot_error_deg"),
                }
                for step in sequence
            ]
            break

    path = save_plan_packet(
        stage="grasp_lift_preview",
        selected=selected,
        candidates=candidates,
        previews=previews,
        observation=observation,
        log_dir=run_dir,
    )
    return selected, previews, path


def _preview_pose(config: GraspLiftConfig, candidate: JSON, pose_key: str, label: str) -> JSON:
    freespace_move = _tool("freespace_move")
    pose = candidate[pose_key]
    kwargs = _move_kwargs(config, candidate["arm"], pose, preview_only=True, width=float(candidate["width"]))
    started = time.time()
    if freespace_move is None:
        return {
            "label": label,
            "pose_key": pose_key,
            "success": False,
            "duration_s": 0.0,
            "request": kwargs,
            "error": "freespace_move unavailable; run through run_script.py for cuRobo preview",
        }
    try:
        summary = _result_summary(freespace_move(**kwargs))
        success = summary.get("status") == "Success" and not bool(summary.get("executed"))
        if bool(summary.get("executed")):
            summary["unsafe_preview_error"] = "preview_only call reported executed=True"
            success = False
        return {
            "label": label,
            "pose_key": pose_key,
            "success": success,
            "duration_s": round(time.time() - started, 3),
            "request": _json_safe(kwargs),
            "result": summary,
        }
    except Exception as exc:
        return {
            "label": label,
            "pose_key": pose_key,
            "success": False,
            "duration_s": round(time.time() - started, 3),
            "request": _json_safe(kwargs),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _execute_physical_attempt(config: GraspLiftConfig, selected: JSON, attempt: JSON) -> JSON:
    if not config.allow_physical or config.plan_only:
        raise RuntimeError("Physical execution requested without OPENFORGE_ALLOW_PHYSICAL_MOTION=1.")
    execution: JSON = {
        "success": False,
        "movement_capable_calls": [],
        "moves": [],
        "gripper_after_close": None,
        "blocker": None,
    }
    side = str(selected["arm"])
    try:
        open_result = _open_gripper_checked(config, side)
        execution["open_gripper"] = open_result
        execution["movement_capable_calls"].extend(open_result.get("movement_capable_calls", []))
        if not open_result.get("confirmed_open"):
            execution["blocker"] = (
                f"{side} gripper did not confirm open before arm motion; stop physical sequence"
            )
            return execution

        for pose_key, label in (
            ("pregrasp_pose", "move_pregrasp"),
            ("grasp_pose", "move_grasp"),
        ):
            move = _execute_move(config, selected, pose_key, label)
            execution["moves"].append(move)
            execution["movement_capable_calls"].append("freespace_move")

        close_result = _close_gripper_checked(config, side, selected)
        execution["gripper_after_close"] = close_result
        execution["movement_capable_calls"].extend(close_result.get("movement_capable_calls", []))
        if not close_result.get("plausible_for_lift"):
            execution["blocker"] = "gripper close evidence is not plausible for lift; stop before lift"
            return execution

        lift_move = _execute_move(config, selected, "lift_pose", "move_lift")
        execution["moves"].append(lift_move)
        execution["movement_capable_calls"].append("freespace_move")
        if config.hold_s > 0:
            time.sleep(config.hold_s)
        execution["success"] = True
        return execution
    except Exception as exc:
        execution["error"] = f"{type(exc).__name__}: {exc}"
        execution["blocker"] = execution["blocker"] or "physical attempt raised before verified lift"
        return execution
    finally:
        attempt["movement_capable_calls"] = list(execution.get("movement_capable_calls", []))


def _pre_recovery_open_home(config: GraspLiftConfig) -> JSON:
    recovery: JSON = {
        "success": False,
        "movement_capable_calls": [],
        "open_grippers": {},
        "post_home_open_grippers": {},
        "go_home": None,
        "before_state": _read_robot_state(),
        "after_state": None,
    }
    try:
        for side in ("left", "right"):
            open_result = _open_gripper_checked(config, side)
            recovery["open_grippers"][side] = open_result
            recovery["movement_capable_calls"].extend(open_result.get("movement_capable_calls", []))
            if not open_result.get("confirmed_open"):
                recovery["reason"] = f"{side} gripper did not confirm open during recovery"
                recovery["after_state"] = _read_robot_state()
                return recovery
        go_home = _tool("go_home")
        if go_home is None:
            recovery["reason"] = "go_home tool unavailable"
            recovery["after_state"] = _read_robot_state()
            return recovery
        recovery["go_home"] = _json_safe(go_home())
        recovery["movement_capable_calls"].append("go_home")
        if _truthy_env("OPENFORGE_GRASP_LIFT_OPEN_AFTER_HOME", default=True):
            for side in ("left", "right"):
                open_result = _open_gripper_checked(config, side)
                recovery["post_home_open_grippers"][side] = open_result
                recovery["movement_capable_calls"].extend(open_result.get("movement_capable_calls", []))
                if not open_result.get("confirmed_open"):
                    recovery["reason"] = f"{side} gripper did not confirm open after go_home"
                    recovery["after_state"] = _read_robot_state()
                    return recovery
        recovery["after_state"] = _read_robot_state()
        recovery["success"] = True
        return recovery
    except Exception as exc:
        recovery["error"] = f"{type(exc).__name__}: {exc}"
        recovery["after_state"] = _read_robot_state()
        return recovery


def _should_recover_after_failed_attempt(config: GraspLiftConfig, execution: JSON, judgment: JSON) -> bool:
    if not config.allow_physical:
        return False
    if not _truthy_env("OPENFORGE_GRASP_LIFT_RECOVER_HOME_AFTER_FAILURE", default=True):
        return False
    if bool(judgment.get("success")):
        return False
    gripper = execution.get("gripper_after_close") or {}
    lift_executed = bool(judgment.get("lift_executed"))
    gripper_plausible = bool(gripper.get("plausible_for_lift"))
    if lift_executed and gripper_plausible:
        return False
    return bool(execution.get("movement_capable_calls"))


def _execute_move(config: GraspLiftConfig, selected: JSON, pose_key: str, label: str) -> JSON:
    freespace_move = _required_tool("freespace_move")
    kwargs = _move_kwargs(config, selected["arm"], selected[pose_key], preview_only=False, width=float(selected["width"]))
    summary = _result_summary(freespace_move(**kwargs))
    summary.update(
        {
            "label": label,
            "pose_key": pose_key,
            "position": selected[pose_key]["position"],
            "rpy": selected[pose_key]["rpy"],
            "preview_only": False,
        }
    )
    if summary.get("status") != "Success" or not bool(summary.get("executed")):
        raise RuntimeError(f"{label} did not execute successfully: {summary}")
    return summary


def _open_gripper_checked(config: GraspLiftConfig, side: str) -> JSON:
    set_gripper = _tool("set_gripper")
    open_gripper = _tool("open_gripper")
    before = _side_gripper_pos(side)
    movement_calls: list[str] = []
    if set_gripper is not None:
        command_result = set_gripper(
            side,
            _clip(config.open_target_pos, 0.0, 1.0),
            vel_limit=config.open_vel_limit,
            torque_limit=config.open_torque_limit,
        )
        movement_calls.append("set_gripper")
    elif open_gripper is not None:
        command_result = open_gripper(side)
        movement_calls.append("open_gripper")
    else:
        raise RuntimeError(
            "Required YAM gripper open tool is unavailable. Run through run_script.py "
            "with skill_library_path=cap/saved_scripts/skill_library."
        )
    samples = []
    deadline = time.monotonic() + max(0.0, config.open_confirm_timeout_s)
    after = _side_gripper_pos(side)
    while after is not None and after < config.open_confirm_pos and time.monotonic() < deadline:
        samples.append(after)
        time.sleep(max(0.02, config.state_poll_s))
        after = _side_gripper_pos(side)
    if after is not None:
        samples.append(after)
    return {
        "movement_capable_calls": movement_calls,
        "before_pos": before,
        "after_pos": after,
        "samples": samples,
        "command_result": _json_safe(command_result),
        "confirmed_open": after is not None and after >= config.open_confirm_pos,
        "target_pos": config.open_target_pos if set_gripper is not None else None,
        "vel_limit": config.open_vel_limit if set_gripper is not None else None,
        "torque_limit": config.open_torque_limit if set_gripper is not None else None,
        "open_confirm_pos": config.open_confirm_pos,
    }


def _close_gripper_checked(config: GraspLiftConfig, side: str, selected: JSON) -> JSON:
    set_gripper = _tool("set_gripper")
    close_gripper = _tool("close_gripper")
    before = _side_gripper_pos(side)
    movement_calls: list[str] = []
    close_target = _resolve_close_target(config, selected)
    target_pos = float(close_target["target_pos"])
    steps: list[JSON] = []
    if set_gripper is not None:
        scout_pos = max(target_pos, min(_clip(config.open_target_pos, 0.0, 1.0), config.close_scout_pos))
        command_result = None
        close_steps = (
            (("target_close", target_pos),)
            if abs(float(scout_pos) - float(target_pos)) < 1e-4
            else (("scout_close", scout_pos), ("target_close", target_pos))
        )
        for label, step_target in close_steps:
            if steps and abs(float(step_target) - float(steps[-1]["target_pos"])) < 1e-4:
                continue
            step_before = _side_gripper_pos(side)
            command_result = set_gripper(
                side,
                _clip(float(step_target), 0.0, 1.0),
                vel_limit=config.close_vel_limit,
                torque_limit=config.close_torque_limit,
            )
            movement_calls.append("set_gripper")
            time.sleep(max(0.0, config.state_poll_s))
            step_after = _side_gripper_pos(side)
            steps.append(
                {
                    "label": label,
                    "target_pos": round(float(step_target), 5),
                    "before_pos": step_before,
                    "after_pos": step_after,
                    "vel_limit": config.close_vel_limit,
                    "torque_limit": config.close_torque_limit,
                    "command_result": _json_safe(command_result),
                }
            )
    elif _truthy_env("OPENFORGE_GRASP_LIFT_ALLOW_RAW_CLOSE") and close_gripper is not None:
        command_result = close_gripper(side)
        movement_calls.append("close_gripper")
    else:
        return {
            "movement_capable_calls": [],
            "before_pos": before,
            "after_pos": None,
            "command_result": None,
            "plausible_for_lift": False,
            "reason": "set_gripper unavailable and raw close was not explicitly enabled",
        }

    samples = []
    deadline = time.monotonic() + max(0.0, config.close_confirm_timeout_s)
    after = _side_gripper_pos(side)
    while after is not None and time.monotonic() < deadline:
        samples.append(after)
        if after <= target_pos + config.close_target_tolerance_pos:
            break
        time.sleep(max(0.02, config.state_poll_s))
        after = _side_gripper_pos(side)
    if after is not None and (not samples or samples[-1] != after):
        samples.append(after)
    delta = float(before - after) if before is not None and after is not None else None
    target_reached = after is not None and after <= target_pos + config.close_target_tolerance_pos
    contact_delta = float(after - target_pos) if after is not None else None
    contact_detected = (
        contact_delta is not None
        and contact_delta >= config.close_contact_min_target_delta_pos
        and delta is not None
        and delta >= config.close_min_delta_pos
    )
    if config.close_require_contact_before_lift:
        plausible_max_pos = config.gripper_plausible_max_pos
        close_evidence_ok = bool(contact_detected)
        close_evidence_mode = "contact_above_commanded_target_required"
    else:
        plausible_max_pos = min(
            config.gripper_plausible_max_pos,
            target_pos + config.close_target_tolerance_pos,
        )
        close_evidence_ok = bool(target_reached)
        close_evidence_mode = "target_reached"
    plausible = (
        after is not None
        and config.gripper_plausible_min_pos <= after <= config.gripper_plausible_max_pos
        and after <= plausible_max_pos
        and close_evidence_ok
        and (delta is None or delta >= config.close_min_delta_pos or after <= target_pos + 0.02)
    )
    return {
        "movement_capable_calls": movement_calls,
        "before_pos": before,
        "after_pos": after,
        "delta_pos": round(float(delta), 4) if delta is not None else None,
        "steps": steps,
        "samples": samples,
        "command_result": _json_safe(command_result),
        "target_pos": target_pos if set_gripper is not None else None,
        "target_resolution": close_target,
        "target_reached": target_reached,
        "contact_delta_pos": round(float(contact_delta), 4) if contact_delta is not None else None,
        "contact_detected": contact_detected,
        "close_evidence_mode": close_evidence_mode,
        "close_require_contact_before_lift": config.close_require_contact_before_lift,
        "plausible_for_lift": plausible,
        "plausible_min_pos": config.gripper_plausible_min_pos,
        "plausible_max_pos": plausible_max_pos,
        "close_min_delta_pos": config.close_min_delta_pos,
    }


def _resolve_close_target(config: GraspLiftConfig, selected: JSON) -> JSON:
    explicit = os.environ.get("OPENFORGE_GRASP_LIFT_CLOSE_TARGET_POS")
    estimated_width = selected.get("estimated_object_width_m")
    if estimated_width is None:
        width = selected.get("width")
        if width is not None:
            estimated_width = max(0.0, float(width) - config.gripper_width_clearance_m)
    if explicit is not None and explicit.strip():
        return {
            "mode": "explicit_normalized_target",
            "target_pos": round(_clip(config.close_target_pos, 0.0, 1.0), 5),
            "target_cap_pos": None,
            "cap_applied": False,
            "estimated_object_width_m": estimated_width,
        }
    if estimated_width is None:
        return {
            "mode": "fallback_normalized_target",
            "target_pos": round(_clip(config.close_target_pos, 0.0, 1.0), 5),
            "target_cap_pos": round(_clip(config.close_target_cap_pos, 0.0, 1.0), 5),
            "cap_applied": False,
            "estimated_object_width_m": None,
            "reason": "selected candidate did not include estimated object width",
        }
    open_width = max(config.gripper_open_width_m, config.gripper_closed_width_m + 1e-4)
    target_width = _clip(
        float(estimated_width) - config.close_width_compression_m,
        config.gripper_closed_width_m,
        open_width,
    )
    ratio = (target_width - config.gripper_closed_width_m) / (open_width - config.gripper_closed_width_m)
    uncapped_target_pos = _clip(ratio, 0.0, 1.0)
    target_cap_pos = _clip(config.close_target_cap_pos, 0.0, 1.0)
    target_pos = min(uncapped_target_pos, target_cap_pos)
    return {
        "mode": "estimated_width_to_normalized_target",
        "target_pos": round(target_pos, 5),
        "uncapped_target_pos": round(uncapped_target_pos, 5),
        "target_cap_pos": round(target_cap_pos, 5),
        "cap_applied": bool(target_pos < uncapped_target_pos),
        "estimated_object_width_m": round(float(estimated_width), 5),
        "target_width_m": round(float(target_width), 5),
        "open_width_m": round(float(open_width), 5),
        "closed_width_m": round(float(config.gripper_closed_width_m), 5),
        "compression_m": round(float(config.close_width_compression_m), 5),
    }


def _judge_attempt(
    config: GraspLiftConfig,
    selected: JSON,
    pre_det: JSON,
    post_det: JSON | None,
    execution: JSON,
) -> JSON:
    gripper = execution.get("gripper_after_close") or {}
    gripper_plausible = bool(gripper.get("plausible_for_lift"))
    lift_executed = any(move.get("label") == "move_lift" and move.get("executed") for move in execution.get("moves", []))

    object_motion = _object_motion(config, selected, pre_det, post_det)
    lifted_by_vision = bool(object_motion.get("lifted_by_vision"))
    success = bool(execution.get("success") and lift_executed and gripper_plausible and lifted_by_vision)
    confidence = 0.85 if success else 0.2 if execution.get("success") and gripper_plausible else 0.05
    if success:
        why = "post-observation shows object lifted near the planned lift pose and gripper state is plausible"
        recommendation = "Stop; verified success."
    elif post_det is None:
        why = "post-observation did not find the object prompt; success is inconclusive/failed"
        recommendation = "Inspect videos/overlays; do not claim success from tool returns."
    elif not gripper_plausible:
        why = "gripper_after_close is not plausible for a held object"
        recommendation = "Inspect gripper state and object contact before any retry."
    elif not lift_executed:
        why = "lift move did not execute"
        recommendation = "Inspect execution error before any physical retry."
    else:
        why = "post-observation did not show the required object z/lateral motion"
        recommendation = "Retry only with a changed candidate from fresh observation."
    return {
        "success": success,
        "confidence": confidence,
        "lift_executed": lift_executed,
        "gripper_plausible": gripper_plausible,
        "object_motion": object_motion,
        "why_stopped": why,
        "next_recommendation": recommendation,
    }


def _object_motion(
    config: GraspLiftConfig,
    selected: JSON,
    pre_det: JSON,
    post_det: JSON | None,
) -> JSON:
    pre = _xyz(pre_det["position_3d"])
    lift = _xyz(selected["lift_pose"]["position"])
    pre_source = str(pre_det.get("source_camera") or pre_det.get("source") or "")
    out: JSON = {
        "pre_position": _round_list(pre),
        "pre_source_camera": pre_source or None,
        "post_position": None,
        "post_source_camera": None,
        "planned_lift_position": _round_list(lift),
        "z_delta_m": None,
        "xy_delta_m": None,
        "xy_distance_to_lift_m": None,
        "verification_mode": None,
        "lifted_by_vision": False,
        "success_min_z_delta_m": config.success_min_z_delta_m,
        "success_max_xy_to_lift_m": config.success_max_xy_to_lift_m,
    }
    if post_det is None:
        out["reason"] = "post_detection_missing"
        return out
    post = _xyz(post_det["position_3d"])
    post_source = str(post_det.get("source_camera") or post_det.get("source") or "")
    xy_delta = math.hypot(post[0] - pre[0], post[1] - pre[1])
    xy_to_lift = math.hypot(post[0] - lift[0], post[1] - lift[1])
    z_delta = post[2] - pre[2]
    same_real_source = bool(pre_source and pre_source == post_source and pre_source != "manual_env")
    same_manual_source = bool(pre_source == post_source == "manual_env")
    if same_real_source:
        verification_mode = "relative_real_pre_post_detection"
        lifted_by_vision = bool(
            z_delta >= config.success_min_z_delta_m
            and xy_delta <= config.success_max_xy_to_lift_m
        )
    elif config.success_allow_mixed_real_sources and pre_source != "manual_env" and post_source != "manual_env":
        verification_mode = "mixed_real_post_to_planned_lift"
        lifted_by_vision = bool(
            z_delta >= config.success_min_z_delta_m
            and xy_to_lift <= config.success_max_xy_to_lift_m
        )
    elif same_manual_source:
        verification_mode = "manual_planned_pose_match"
        lifted_by_vision = bool(
            z_delta >= config.success_min_z_delta_m
            and xy_to_lift <= config.success_max_xy_to_lift_m
        )
    else:
        verification_mode = "mixed_detection_sources_inconclusive"
        lifted_by_vision = False
    out.update(
        {
            "post_position": _round_list(post),
            "post_source_camera": post_source or None,
            "z_delta_m": round(float(z_delta), 4),
            "xy_delta_m": round(float(xy_delta), 4),
            "xy_distance_to_lift_m": round(float(xy_to_lift), 4),
            "verification_mode": verification_mode,
            "lifted_by_vision": lifted_by_vision,
        }
    )
    return out


def _move_kwargs(config: GraspLiftConfig, side: str, pose: JSON, *, preview_only: bool, width: float) -> JSON:
    return {
        f"{side}_target_pos": [float(x) for x in pose["position"]],
        f"{side}_target_rpy": [float(x) for x in pose["rpy"]],
        f"{side}_gripper": _clip(config.motion_gripper_pos, 0.0, 1.0),
        "preview_only": bool(preview_only),
        "planning_speed": config.planning_speed,
        "ik_error_threshold": config.ik_error_threshold_m,
        "ik_rot_threshold_deg": config.ik_rot_threshold_deg,
        "ik_xyz_weight": config.ik_xyz_weight,
        "ik_rpy_weight": config.ik_rpy_weight,
        "planner_backend": config.planner_backend,
        "solver_speed": config.solver_speed,
    }


def _preferred_arm(xyz: list[float], config: GraspLiftConfig) -> tuple[str, str]:
    if config.requested_arm in {"left", "right"}:
        return config.requested_arm, "OPENFORGE_GRASP_LIFT_ARM override"
    y = float(xyz[1])
    if y > 0.04:
        return "left", "object y is on YAM left side"
    if y < -0.04:
        return "right", "object y is on YAM right side"
    return "right", "object near centerline; right arm is default"


def _arm_order(preferred: str, config: GraspLiftConfig) -> list[str]:
    if config.requested_arm in {"left", "right"}:
        return [config.requested_arm]
    other = "left" if preferred == "right" else "right"
    return [preferred, other]


def _body_grasp_z(
    config: GraspLiftConfig,
    center: list[float],
    half_extents: list[float],
    z_offset_m: float,
    detection: JSON,
) -> float:
    if len(half_extents) >= 3 and half_extents[2] > 0:
        fraction = _clip(config.body_grasp_fraction, 0.25, 0.75)
        bottom = float(center[2]) - float(half_extents[2])
        return bottom + 2.0 * float(half_extents[2]) * fraction + float(z_offset_m)
    if _uses_live_z_body_offset(config, detection, half_extents):
        return float(center[2]) + float(config.live_detection_z_to_body_offset_m) + float(z_offset_m)
    return float(center[2]) + float(z_offset_m)


def _object_radius(config: GraspLiftConfig, half_extents: list[float]) -> float:
    if len(half_extents) >= 2 and max(half_extents[0], half_extents[1]) > 0:
        return _clip(max(float(half_extents[0]), float(half_extents[1])), 0.02, 0.055)
    return _clip(config.object_radius_m, 0.02, 0.055)


def _uses_live_z_body_offset(config: GraspLiftConfig, detection: JSON, half_extents: list[float]) -> bool:
    if len(half_extents) >= 3 and half_extents[2] > 0:
        return False
    source = str(detection.get("source_camera") or detection.get("source") or "")
    return bool(config.object_kind.strip().lower() == "bottle" and source and source != "manual_env")


def _orientation_hints_from_detection(detection: JSON) -> JSON:
    """Extract optional yaw hints without making them required for planning."""
    hints: JSON = {
        "sources": [],
        "absolute_approach_angles_deg": [],
        "body_axis_yaw_deg": None,
        "box_2d_axis_status": None,
    }
    yaw: float | None = None
    rpy = _coerce_float_list(detection.get("rpy"), 3)
    if rpy is not None:
        yaw = _normalize_angle_deg(rpy[2])
        hints["sources"].append("detection_rpy")

    if yaw is None:
        quat = _coerce_float_list(detection.get("quaternion_xyzw"), 4)
        if quat is not None:
            try:
                from scipy.spatial.transform import Rotation

                euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
                yaw = _normalize_angle_deg(float(euler[2]))
                hints["sources"].append("detection_quaternion_xyzw")
            except Exception as exc:
                hints["quaternion_parse_error"] = f"{type(exc).__name__}: {exc}"

    if yaw is not None:
        hints["body_axis_yaw_deg"] = round(float(yaw), 3)
        hints["absolute_approach_angles_deg"] = _dedupe_angles([yaw + 90.0, yaw - 90.0, yaw, yaw + 180.0])

    box = _coerce_float_list(detection.get("box_2d"), 4)
    if box is not None:
        width_px = abs(box[2] - box[0])
        height_px = abs(box[3] - box[1])
        hints["box_2d_size_px"] = [round(float(width_px), 1), round(float(height_px), 1)]
        hints["box_2d_axis_status"] = (
            "axis_aligned_bbox_only_no_in_plane_yaw"
            if width_px > 0 and height_px > 0
            else "bbox_unusable"
        )

    if not hints["sources"]:
        hints["sources"].append("none")
    return hints


def _absolute_approach_angles(orientation_hints: JSON) -> list[float]:
    override = _float_list(os.environ.get("OPENFORGE_GRASP_LIFT_ABSOLUTE_APPROACH_ANGLES_DEG"))
    if override:
        return _dedupe_angles(override)
    return _dedupe_angles(orientation_hints.get("absolute_approach_angles_deg") or [])


def _dedupe_angles(values: list[float]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        angle = round(_normalize_angle_deg(float(value)), 3)
        if angle in seen:
            continue
        seen.add(angle)
        out.append(angle)
    return out


def _side_grasp_display_rpy_from_approach(
    approach_dir: list[float],
    wrist_roll_deg: float = 0.0,
) -> tuple[list[float], JSON]:
    """Return display RPY with local +Z along the horizontal approach axis."""
    try:
        import numpy as np

        approach = np.asarray(approach_dir, dtype=float)
        norm = float(np.linalg.norm(approach))
        if norm < 1e-6:
            raise ValueError(f"invalid approach_dir={approach_dir!r}")
        z_axis = approach / norm
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        x_axis = np.cross(world_up, z_axis)
        if float(np.linalg.norm(x_axis)) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / float(np.linalg.norm(y_axis))
        if abs(float(wrist_roll_deg)) > 1e-6:
            roll_rad = math.radians(float(wrist_roll_deg))
            cos_t = math.cos(roll_rad)
            sin_t = math.sin(roll_rad)
            x_base = x_axis
            y_base = y_axis
            x_axis = cos_t * x_base + sin_t * y_base
            y_axis = -sin_t * x_base + cos_t * y_base
            x_axis = x_axis / float(np.linalg.norm(x_axis))
            y_axis = y_axis / float(np.linalg.norm(y_axis))
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        rpy = _display_rpy_from_rotation_matrix(rotation_matrix)
        return (
            rpy,
            {
                "local_x_opening_axis": _round_list(x_axis.tolist()),
                "local_y_height_axis": _round_list(y_axis.tolist()),
                "local_z_approach_axis": _round_list(z_axis.tolist()),
            },
        )
    except Exception:
        jaw_axis_yaw = _normalize_angle_deg(math.degrees(math.atan2(approach_dir[1], approach_dir[0])) + 90.0)
        return [0.0, 90.0, jaw_axis_yaw], {
            "fallback": "simple_side_grasp_rpy",
            "local_z_approach_axis": _round_list(approach_dir),
        }


def _display_pitch_flip_rpy(rpy: list[float]) -> list[float]:
    if len(rpy) < 3:
        return rpy
    return [
        float(rpy[0]),
        -float(rpy[1]),
        _normalize_angle_deg(float(rpy[2]) + 90.0),
    ]


def _display_rpy_axes(rpy: list[float]) -> JSON:
    import numpy as np
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    euler_xyz = [-pitch, roll, -yaw - 90.0]
    matrix = Rotation.from_euler("xyz", euler_xyz, degrees=True).as_matrix()
    return {
        "local_x_opening_axis": _round_list(matrix[:, 0].tolist()),
        "local_y_height_axis": _round_list(matrix[:, 1].tolist()),
        "local_z_approach_axis": _round_list(matrix[:, 2].tolist()),
    }


def _horizontal_unit(axis: Any, *, fallback: list[float]) -> list[float]:
    values = [float(x) for x in list(axis)[:3]]
    if len(values) < 3:
        values = [float(x) for x in fallback[:3]]
    values[2] = 0.0
    norm = math.hypot(values[0], values[1])
    if norm < 1e-6:
        values = [float(fallback[0]), float(fallback[1]), 0.0]
        norm = math.hypot(values[0], values[1])
    if norm < 1e-6:
        return [1.0, 0.0, 0.0]
    return [values[0] / norm, values[1] / norm, 0.0]


def _display_rpy_from_rotation_matrix(rotation_matrix: Any) -> list[float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    euler_xyz = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=float)).as_euler(
        "xyz",
        degrees=True,
    )
    display = np.array(
        [euler_xyz[1], -euler_xyz[0], -euler_xyz[2] - 90.0],
        dtype=float,
    )
    display = (display + 180.0) % 360.0 - 180.0
    return [float(x) for x in display]


def _candidate_key(candidate: JSON) -> str:
    return "|".join(
        [
            str(candidate.get("arm")),
            str(candidate.get("strategy")),
            ",".join(f"{float(x):.3f}" for x in candidate.get("position", [])[:3]),
            ",".join(f"{float(x):.1f}" for x in candidate.get("rpy", [])[:3]),
        ]
    )


def _preview_failures(previews: list[JSON]) -> list[JSON]:
    failures = []
    for preview in previews:
        if preview.get("success"):
            continue
        result = preview.get("result") if isinstance(preview.get("result"), dict) else {}
        failures.append(
            {
                "label": preview.get("label"),
                "pose_key": preview.get("pose_key"),
                "status": result.get("status"),
                "final_pos_error_m": result.get("final_pos_error_m"),
                "final_rot_error_deg": result.get("final_rot_error_deg"),
                "error": preview.get("error") or result.get("reason"),
            }
        )
    return failures[:10]


def _ticket_recommendation(config: GraspLiftConfig) -> str:
    return (
        "Physical ticket needed: operator present, E-stop reachable, workspace and "
        f"hands/tools/cables clear, object/target='{','.join(config.prompts)}' "
        f"for {config.task_name}, allowed script cap/saved_scripts/{config.task_name}.py, "
        "allowed stages observe/plan/pregrasp/grasp_close/lift/post_observe inside "
        f"one-command loop, no liquid, max attempts {config.attempt_budget}, "
        "stop on gripper anomaly, planner failure, unexpected contact, object slip/drop, "
        "inconclusive post-observation, operator interruption, or competing motion process; "
        "confirm no other motion-capable script is running; command must set "
        "OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
    )


def _write_attempt(run_dir: Path, attempt: JSON) -> None:
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    path = attempts_dir / f"attempt_{int(attempt['attempt_index']):02d}.json"
    path.write_text(json.dumps(_json_safe(attempt), indent=2) + "\n", encoding="utf-8")
    write_stage_summary(stage=f"attempt_{attempt['attempt_index']:02d}_result", result=attempt, log_dir=run_dir)


def _write_result(run_dir: Path, result: JSON) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(result), indent=2) + "\n"
    (run_dir / "task_result.json").write_text(payload, encoding="utf-8")
    (run_dir / "result.json").write_text(payload, encoding="utf-8")


def _video_paths(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    paths = []
    for path in sorted(run_dir.rglob("*.mp4")):
        try:
            paths.append(str(path.relative_to(run_dir)))
        except ValueError:
            paths.append(str(path))
    return paths[:20]


def _obs_ref(packet: JSON | None) -> JSON | None:
    if packet is None:
        return None
    return {
        "stage": packet.get("stage"),
        "packet_path": packet.get("packet_path"),
        "cameras": list((packet.get("cameras") or {}).keys()),
    }


def _det_ref(det: JSON) -> JSON:
    return {
        "prompt": det.get("prompt"),
        "label": det.get("label"),
        "score": det.get("score"),
        "source_camera": det.get("source_camera"),
        "prompt_fallback_used": det.get("prompt_fallback_used"),
        "position_3d": _round_list(det.get("position_3d") or det.get("position") or []),
        "box_2d": _round_list(det.get("box_2d") or []),
        "rpy": _round_list(det.get("rpy") or [], 3),
        "quaternion_xyzw": _round_list(det.get("quaternion_xyzw") or [], 4),
        "half_extents": _round_list(det.get("half_extents") or []),
    }


def _read_robot_state() -> Any:
    fn = _tool("get_robot_state")
    if fn is None:
        return "get_robot_state unavailable"
    try:
        return _json_safe(fn())
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _side_gripper_pos(side: str) -> float | None:
    state_fn = _tool("get_robot_state")
    if state_fn is None:
        return None
    try:
        state = state_fn()
        value = getattr(state, f"{side}_gripper_pos")
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else None
        return float(value)
    except Exception:
        return None


def _result_summary(result: Any) -> JSON:
    if isinstance(result, dict):
        return dict(result)
    return {
        "status": getattr(result, "status", None),
        "executed": bool(getattr(result, "executed", False)),
        "side": getattr(result, "side", None),
        "trajectory_steps": int(getattr(result, "trajectory_steps", 0) or 0),
        "trajectory_cache_key": getattr(result, "trajectory_cache_key", None),
        "final_pos_error_m": getattr(result, "final_pos_error_m", None),
        "final_rot_error_deg": getattr(result, "final_rot_error_deg", None),
        "reason": getattr(result, "reason", ""),
    }


def _tool(name: str) -> Callable[..., Any] | None:
    try:
        import skill_library.namespace as namespace

        fn = getattr(namespace, name, None)
        return fn if callable(fn) else None
    except Exception:
        return None


def _required_tool(name: str) -> Callable[..., Any]:
    fn = _tool(name)
    if fn is None:
        raise RuntimeError(
            f"Required YAM tool {name!r} is unavailable. Run through run_script.py "
            "with skill_library_path=cap/saved_scripts/skill_library."
        )
    return fn


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _float_list(
    value: str | None,
    expected: int | None = None,
    *,
    fallback: list[float] | None = None,
) -> list[float] | None:
    if value is None or not value.strip():
        return list(fallback) if fallback is not None else None
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if expected is not None and len(parts) != expected:
        raise ValueError(f"Expected {expected} floats, got {value!r}")
    return [float(part) for part in parts]


def _coerce_float_list(value: Any, expected: int | None = None) -> list[float] | None:
    if value is None:
        return None
    try:
        out = [float(part) for part in list(value)]
    except Exception:
        return None
    if expected is not None and len(out) < expected:
        return None
    if expected is not None:
        out = out[:expected]
    if not all(math.isfinite(part) for part in out):
        return None
    return out


def _xyz(value: Any) -> list[float]:
    return [float(x) for x in list(value)[:3]]


def _valid_xyz(value: Any) -> bool:
    try:
        xyz = _xyz(value)
        return len(xyz) == 3 and all(math.isfinite(x) for x in xyz)
    except Exception:
        return False


def _half_extents(detection: JSON) -> list[float]:
    value = detection.get("half_extents") or []
    try:
        return [float(x) for x in list(value)[:3]]
    except Exception:
        return []


def _round_list(values: Any, digits: int = 4) -> list[float]:
    try:
        return [round(float(x), digits) for x in list(values)]
    except Exception:
        return []


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _normalize_angle_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist() if value.size <= 64 else {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")
