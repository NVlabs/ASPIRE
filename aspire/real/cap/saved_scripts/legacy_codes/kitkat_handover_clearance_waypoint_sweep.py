# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion robustness sweep for KitKat handover clearance waypoints.

This script does not command the robot. It plans directly through the cuRobo
portal from recorded post-step joint states in recent physical handover runs:

* left post-lift -> new left post-grip clearance -> left handoff
* right post-left-retreat -> right -X retreat -> first trash-stage waypoint

Each fixed offset is also stressed with small random target and start-joint
perturbations. The output recommends env values for the physical handover
script, but does not execute any trajectory.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from cap.saved_scripts.yam_runtime import append_stage_summary, current_run_dir, json_safe, write_json


TASK_NAME = "kitkat_handover_clearance_waypoint_sweep"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    out: list[float] = []
    for part in raw.split(","):
        try:
            out.append(float(part.strip()))
        except ValueError:
            pass
    return out or list(default)


def _env_xyz(name: str, default: list[float]) -> list[float]:
    values = _env_list(name, default)
    if len(values) < 3:
        return list(default)
    return [float(values[0]), float(values[1]), float(values[2])]


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        rows = list(values)
        if limit is not None:
            rows = rows[:limit]
        return [round(float(value), digits) for value in rows]
    except Exception:
        return []


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _display_rpy_to_quat(rpy: list[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    euler_xyz = [-pitch, roll, -yaw - 90.0]
    return Rotation.from_euler("xyz", euler_xyz, degrees=True).as_quat()


def _joint_pos(packet: dict[str, Any] | None, side: str) -> list[float] | None:
    values = (((packet or {}).get("post_exec_eef") or {}).get(side) or {}).get("joint_pos")
    rows = _round_list(values, digits=6, limit=6)
    return rows if len(rows) == 6 else None


def _ee_pos(packet: dict[str, Any] | None, side: str) -> list[float] | None:
    values = (((packet or {}).get("post_exec_eef") or {}).get(side) or {}).get("ee_pos")
    rows = _round_list(values, digits=5, limit=3)
    return rows if len(rows) == 3 else None


def _pose_packet(pose: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": _round_list(pose.get("position"), limit=3),
        "rpy": _round_list(pose.get("rpy"), limit=3),
    }


def _load_seed_runs(run_dir: Path) -> list[dict[str, Any]]:
    pattern = os.environ.get(
        "OPENFORGE_KITKAT_CLEARANCE_SWEEP_SEED_RUN_GLOB",
        "logs/handover_chocolate_bar_left_to_right_one_shot_loop_20260606T16*",
    )
    max_runs = _env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_MAX_SEED_RUNS", 4)
    seeds: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for raw in sorted(glob.glob(pattern), reverse=True):
        path = Path(raw)
        physical = _read_json(path / "plans" / "physical_attempt.json")
        if not bool((physical or {}).get("success")):
            skipped.append({"run_dir": str(path), "reason": "physical_attempt not successful"})
            continue
        left_lift = _read_json(path / "plans" / "left_lift_direct.json")
        right_receive = _read_json(path / "plans" / "right_receive_close_pose_try_1.json")
        left_retreat = _read_json(path / "plans" / "left_retreat_after_release_actual_small_y_clear_1.json")
        trash_plan = _read_json(path / "plans" / "trash_place_plan.json")
        selected_plan = _read_json(path / "plans" / "selected_handover_plan.json")
        left_lift_l = _joint_pos(left_lift, "left")
        left_lift_r = _joint_pos(left_lift, "right")
        right_start_l = _joint_pos(left_retreat, "left")
        right_start_r = _joint_pos(left_retreat, "right")
        right_start_pos = _ee_pos(left_retreat, "right")
        right_receive_pos = _ee_pos(right_receive, "right")
        if not (left_lift_l and left_lift_r and right_start_l and right_start_r and right_start_pos):
            skipped.append({"run_dir": str(path), "reason": "missing required post-step joints/EEF pose"})
            continue
        trash_stage_pose = None
        transport = (trash_plan or {}).get("right_trash_transport_poses") or []
        if transport and isinstance(transport[0], dict):
            trash_stage_pose = transport[0].get("pose")
        handoff_pose = (selected_plan or {}).get("left_handoff_pose") or {}
        seeds.append(
            {
                "run_dir": str(path),
                "left_start_joints_after_lift": {"left": left_lift_l, "right": left_lift_r},
                "right_start_joints_after_left_retreat": {"left": right_start_l, "right": right_start_r},
                "right_start_pos_after_left_retreat_m": right_start_pos,
                "right_receive_pos_m": right_receive_pos,
                "trash_stage_pose": trash_stage_pose,
                "left_handoff_pose_from_run": handoff_pose,
            }
        )
        if len(seeds) >= max_runs:
            break
    write_json(run_dir / "plans" / "seed_runs.json", {"selected": seeds, "skipped": skipped, "pattern": pattern})
    return seeds


def _make_planner() -> Any:
    from experimental.portal_motion_planner import PortalMotionPlanner

    raw_port = os.environ.get("CAP_CUROBO_PORT", "").strip()
    port = int(raw_port) if raw_port else None
    start_server_default = raw_port == ""
    return PortalMotionPlanner(
        backend="curobo",
        solver_speed=os.environ.get("OPENFORGE_KITKAT_CLEARANCE_SWEEP_SOLVER_SPEED", "fast"),
        host=os.environ.get("CAP_CUROBO_HOST", "127.0.0.1"),
        port=port,
        position_threshold=_env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_IK_ERROR_THRESHOLD_M", 0.025),
        rotation_threshold=math.radians(_env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_IK_ROT_THRESHOLD_DEG", 16.0)),
        start_server=_env_bool("CAP_CUROBO_START_SERVER", start_server_default),
        robot_type=os.environ.get("CAP_ROBOT_TYPE", "yam"),
    )


def _success_status(status: Any) -> bool:
    return str(status or "").strip().lower() in {"success", "succeeded", "ok"}


def _final_joints(result: dict[str, Any], side: str, fallback: np.ndarray) -> np.ndarray:
    key = f"{side}_positions"
    values = np.asarray(result.get(key, []), dtype=np.float64)
    if values.size == 0:
        return np.asarray(fallback, dtype=np.float64).reshape(6)
    return values.reshape(-1, 6)[-1].astype(np.float64)


def _plan_pose(
    planner: Any,
    *,
    label: str,
    side: str,
    left_jp: np.ndarray,
    right_jp: np.ndarray,
    pose: dict[str, Any],
    left_gripper: float,
    right_gripper: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    kwargs: dict[str, Any] = {
        "current_left_jp": np.asarray(left_jp, dtype=np.float64),
        "current_right_jp": np.asarray(right_jp, dtype=np.float64),
        "target_left_pos": None,
        "target_left_quat_xyzw": None,
        "target_right_pos": None,
        "target_right_quat_xyzw": None,
        "side": side,
        "left_gripper": float(left_gripper),
        "right_gripper": float(right_gripper),
        "validate_trajectory": True,
    }
    if side == "left":
        kwargs["target_left_pos"] = np.asarray(pose["position"], dtype=np.float64)
        kwargs["target_left_quat_xyzw"] = _display_rpy_to_quat(pose["rpy"])
    elif side == "right":
        kwargs["target_right_pos"] = np.asarray(pose["position"], dtype=np.float64)
        kwargs["target_right_quat_xyzw"] = _display_rpy_to_quat(pose["rpy"])
    else:
        raise ValueError(f"unsupported side: {side}")
    try:
        result = planner.plan_to_pose(**kwargs)
    except Exception as exc:
        return (
            {
                "label": label,
                "side": side,
                "success": False,
                "status": "Exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "target_pose": _pose_packet(pose),
            },
            np.asarray(left_jp, dtype=np.float64),
            np.asarray(right_jp, dtype=np.float64),
        )
    success = _success_status(result.get("status"))
    left_final = _final_joints(result, "left", left_jp)
    right_final = _final_joints(result, "right", right_jp)
    summary = {
        "label": label,
        "side": side,
        "success": success,
        "status": result.get("status"),
        "status_detail": result.get("status_detail"),
        "target_pose": _pose_packet(pose),
        "position_error_m": result.get("position_error_m"),
        "rotation_error_deg": result.get("rotation_error_deg"),
        "trajectory_steps": int(np.asarray(result.get("position", [])).reshape(-1, 12).shape[0])
        if np.asarray(result.get("position", [])).size
        else 0,
    }
    return summary, left_final, right_final


def _perturb_pose(pose: dict[str, Any], rng: random.Random, *, yaw_default: float = 5.0) -> dict[str, Any]:
    pos = [float(v) for v in pose["position"][:3]]
    rpy = [float(v) for v in pose["rpy"][:3]]
    pos[0] += rng.uniform(-_env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_X_M", 0.020), _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_X_M", 0.020))
    pos[1] += rng.uniform(-_env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_Y_M", 0.020), _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_Y_M", 0.020))
    pos[2] += rng.uniform(-_env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_Z_M", 0.020), _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_Z_M", 0.020))
    rpy[2] += rng.uniform(-_env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_YAW_DEG", yaw_default), _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_YAW_DEG", yaw_default))
    return {"position": pos, "rpy": rpy}


def _perturb_joints(joints: np.ndarray, rng: random.Random) -> np.ndarray:
    limit = _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_START_JOINT_PERTURB_RAD", 0.015)
    if limit <= 0:
        return np.asarray(joints, dtype=np.float64).copy()
    return np.asarray(joints, dtype=np.float64) + np.asarray([rng.uniform(-limit, limit) for _ in range(6)], dtype=np.float64)


def _score_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [trial for trial in trials if trial.get("success")]
    step_successes = [step for trial in successes for step in trial.get("steps", []) if step.get("success")]
    pos_errors = [float(step["position_error_m"]) for step in step_successes if step.get("position_error_m") is not None]
    rot_errors = [float(step["rotation_error_deg"]) for step in step_successes if step.get("rotation_error_deg") is not None]
    return {
        "success_count": len(successes),
        "trial_count": len(trials),
        "success_rate": round(len(successes) / max(1, len(trials)), 4),
        "max_pos_error_m": round(max(pos_errors), 6) if pos_errors else None,
        "mean_pos_error_m": round(sum(pos_errors) / len(pos_errors), 6) if pos_errors else None,
        "max_rot_error_deg": round(max(rot_errors), 4) if rot_errors else None,
        "mean_rot_error_deg": round(sum(rot_errors) / len(rot_errors), 4) if rot_errors else None,
    }


def _left_clearance_pose(left_handoff_pos: list[float], x_offset: float, y_offset: float, z_offset: float) -> dict[str, Any]:
    return {
        "position": [
            float(left_handoff_pos[0] + x_offset),
            float(left_handoff_pos[1] + y_offset),
            float(left_handoff_pos[2] + z_offset),
        ],
        "rpy": [0.0, 180.0, 90.0],
    }


def _right_retreat_pose(right_start_pos: list[float], x_offset: float, y_offset: float, z_offset: float) -> dict[str, Any]:
    return {
        "position": [
            float(right_start_pos[0] + x_offset),
            float(right_start_pos[1] + y_offset),
            float(right_start_pos[2] + z_offset),
        ],
        "rpy": [0.0, 90.0, -90.0],
    }


def _left_trial(
    planner: Any,
    seed: dict[str, Any],
    left_handoff_pos: list[float],
    candidate: dict[str, float],
    rng: random.Random | None,
) -> dict[str, Any]:
    left_jp = np.asarray(seed["left_start_joints_after_lift"]["left"], dtype=np.float64)
    right_jp = np.asarray(seed["left_start_joints_after_lift"]["right"], dtype=np.float64)
    if rng is not None:
        left_jp = _perturb_joints(left_jp, rng)
        right_jp = _perturb_joints(right_jp, rng)
    clearance = _left_clearance_pose(left_handoff_pos, candidate["x_offset_m"], candidate["y_offset_m"], candidate["z_offset_m"])
    if rng is not None:
        clearance = _perturb_pose(clearance, rng)
    handoff = {"position": list(left_handoff_pos), "rpy": [0.0, 180.0, 90.0]}
    step_1, left_after, right_after = _plan_pose(
        planner,
        label="left_post_lift_to_post_grip_clearance",
        side="left",
        left_jp=left_jp,
        right_jp=right_jp,
        pose=clearance,
        left_gripper=0.04,
        right_gripper=1.0,
    )
    steps = [step_1]
    if step_1["success"]:
        step_2, _, _ = _plan_pose(
            planner,
            label="left_post_grip_clearance_to_handoff",
            side="left",
            left_jp=left_after,
            right_jp=right_after,
            pose=handoff,
            left_gripper=0.04,
            right_gripper=1.0,
        )
        steps.append(step_2)
    return {"seed_run_dir": seed["run_dir"], "success": all(step.get("success") for step in steps), "steps": steps}


def _right_trial(
    planner: Any,
    seed: dict[str, Any],
    candidate: dict[str, float],
    rng: random.Random | None,
) -> dict[str, Any]:
    left_jp = np.asarray(seed["right_start_joints_after_left_retreat"]["left"], dtype=np.float64)
    right_jp = np.asarray(seed["right_start_joints_after_left_retreat"]["right"], dtype=np.float64)
    if rng is not None:
        left_jp = _perturb_joints(left_jp, rng)
        right_jp = _perturb_joints(right_jp, rng)
    retreat = _right_retreat_pose(seed["right_start_pos_after_left_retreat_m"], candidate["x_offset_m"], candidate["y_offset_m"], candidate["z_offset_m"])
    if rng is not None:
        retreat = _perturb_pose(retreat, rng)
    step_1, left_after, right_after = _plan_pose(
        planner,
        label="right_receive_to_minus_x_retreat",
        side="right",
        left_jp=left_jp,
        right_jp=right_jp,
        pose=retreat,
        left_gripper=1.0,
        right_gripper=0.0,
    )
    steps = [step_1]
    trash_stage = seed.get("trash_stage_pose")
    if step_1["success"] and isinstance(trash_stage, dict):
        step_2, _, _ = _plan_pose(
            planner,
            label="right_minus_x_retreat_to_trash_yz_stage",
            side="right",
            left_jp=left_after,
            right_jp=right_after,
            pose=trash_stage,
            left_gripper=1.0,
            right_gripper=0.0,
        )
        steps.append(step_2)
    return {"seed_run_dir": seed["run_dir"], "success": all(step.get("success") for step in steps), "steps": steps}


def _evaluate_left(
    planner: Any,
    seeds: list[dict[str, Any]],
    left_handoff_pos: list[float],
    rng: random.Random,
) -> dict[str, Any]:
    candidates: list[dict[str, float]] = []
    for x_offset in _env_list("OPENFORGE_KITKAT_CLEARANCE_SWEEP_LEFT_X_OFFSETS_M", [0.0]):
        for y_offset in _env_list("OPENFORGE_KITKAT_CLEARANCE_SWEEP_LEFT_Y_OFFSETS_M", [0.170, 0.190, 0.210, 0.230]):
            for z_offset in _env_list("OPENFORGE_KITKAT_CLEARANCE_SWEEP_LEFT_Z_OFFSETS_M", [0.030, 0.050]):
                candidates.append({"x_offset_m": x_offset, "y_offset_m": y_offset, "z_offset_m": z_offset})
    coarse: list[dict[str, Any]] = []
    for candidate in candidates:
        trials = [_left_trial(planner, seed, left_handoff_pos, candidate, None) for seed in seeds]
        coarse.append({**candidate, "coarse_score": _score_trials(trials), "coarse_trials": trials})
    coarse.sort(
        key=lambda item: (
            -float(item["coarse_score"]["success_rate"]),
            float(item["coarse_score"]["max_pos_error_m"] or 999.0),
            abs(float(item["y_offset_m"]) - 0.190),
            abs(float(item["z_offset_m"]) - 0.030),
        )
    )
    stress: list[dict[str, Any]] = []
    top_k = _env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_TOP_K", 4)
    random_trials = _env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RANDOM_TRIALS", 5)
    for item in coarse[:top_k]:
        trials: list[dict[str, Any]] = []
        candidate = {key: float(item[key]) for key in ("x_offset_m", "y_offset_m", "z_offset_m")}
        for seed in seeds:
            for _ in range(random_trials):
                trials.append(_left_trial(planner, seed, left_handoff_pos, candidate, rng))
        stress.append({**item, "stress_score": _score_trials(trials), "stress_trials": trials})
    stress.sort(
        key=lambda item: (
            -float(item["stress_score"]["success_rate"]),
            float(item["stress_score"]["max_pos_error_m"] or 999.0),
            abs(float(item["y_offset_m"]) - 0.190),
            abs(float(item["z_offset_m"]) - 0.030),
            abs(float(item["x_offset_m"])),
        )
    )
    selected = stress[0] if stress else (coarse[0] if coarse else None)
    return {"candidates": candidates, "coarse": coarse, "stress": stress, "selected": selected}


def _evaluate_right(planner: Any, seeds: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    candidates: list[dict[str, float]] = []
    for x_offset in _env_list("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RIGHT_X_OFFSETS_M", [-0.030, -0.040, -0.060, -0.080]):
        for y_offset in _env_list("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RIGHT_Y_OFFSETS_M", [-0.045]):
            for z_offset in _env_list("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RIGHT_Z_OFFSETS_M", [0.015, 0.025]):
                candidates.append({"x_offset_m": x_offset, "y_offset_m": y_offset, "z_offset_m": z_offset})
    coarse: list[dict[str, Any]] = []
    for candidate in candidates:
        trials = [_right_trial(planner, seed, candidate, None) for seed in seeds]
        coarse.append({**candidate, "coarse_score": _score_trials(trials), "coarse_trials": trials})
    preferred_x = _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RIGHT_PREFERRED_X_M", -0.040)
    coarse.sort(
        key=lambda item: (
            -float(item["coarse_score"]["success_rate"]),
            float(item["coarse_score"]["max_pos_error_m"] or 999.0),
            abs(float(item["x_offset_m"]) - preferred_x),
            abs(float(item["z_offset_m"]) - 0.015),
        )
    )
    stress: list[dict[str, Any]] = []
    top_k = _env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_TOP_K", 4)
    random_trials = _env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RANDOM_TRIALS", 5)
    for item in coarse[:top_k]:
        trials: list[dict[str, Any]] = []
        candidate = {key: float(item[key]) for key in ("x_offset_m", "y_offset_m", "z_offset_m")}
        for seed in seeds:
            for _ in range(random_trials):
                trials.append(_right_trial(planner, seed, candidate, rng))
        stress.append({**item, "stress_score": _score_trials(trials), "stress_trials": trials})
    stress.sort(
        key=lambda item: (
            -float(item["stress_score"]["success_rate"]),
            float(item["stress_score"]["max_pos_error_m"] or 999.0),
            abs(float(item["x_offset_m"]) - preferred_x),
            abs(float(item["z_offset_m"]) - 0.015),
            abs(float(item["y_offset_m"]) + 0.045),
        )
    )
    selected = stress[0] if stress else (coarse[0] if coarse else None)
    return {"candidates": candidates, "coarse": coarse, "stress": stress, "selected": selected}


def _selected_offsets(selected: dict[str, Any] | None) -> list[float] | None:
    if not selected:
        return None
    return [float(selected["x_offset_m"]), float(selected["y_offset_m"]), float(selected["z_offset_m"])]


def _print_compact_result(result: dict[str, Any]) -> None:
    compact = {
        "success": result.get("success"),
        "physical_motion_executed": result.get("physical_motion_executed"),
        "run_dir": result.get("run_dir"),
        "why_stopped": result.get("why_stopped"),
        "seed_count": result.get("seed_count"),
        "selected_env": result.get("selected_env"),
        "path": result.get("path"),
    }
    print(json.dumps(json_safe(compact), indent=2))


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    rng = random.Random(_env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_SEED", 17))
    seeds = _load_seed_runs(run_dir)
    if not seeds:
        result = {
            "schema": "openforge.kitkat_handover_clearance_waypoint_sweep.v1",
            "success": False,
            "physical_motion_executed": False,
            "why_stopped": "no usable successful physical seed runs found",
            "run_dir": str(run_dir),
        }
        result["path"] = write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(run_dir, ["## KitKat Handover Clearance Sweep", "- success: False", "- why_stopped: no usable seed runs"])
        _print_compact_result(result)
        return result

    default_handoff = [0.560, 0.060, 0.940]
    first_handoff = seeds[0].get("left_handoff_pose_from_run") or {}
    handoff_pos = _round_list(first_handoff.get("position") or default_handoff, limit=3) or default_handoff
    left_handoff_pos = _env_xyz("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_HANDOFF_POS_M", handoff_pos)

    planner = _make_planner()
    left = _evaluate_left(planner, seeds, left_handoff_pos, rng)
    right = _evaluate_right(planner, seeds, rng)
    left_offsets = _selected_offsets(left.get("selected"))
    right_offsets = _selected_offsets(right.get("selected"))
    result = {
        "schema": "openforge.kitkat_handover_clearance_waypoint_sweep.v1",
        "success": bool(left_offsets and right_offsets),
        "physical_motion_executed": False,
        "run_dir": str(run_dir),
        "seed_count": len(seeds),
        "seed_runs": [seed["run_dir"] for seed in seeds],
        "left_handoff_pos_m": _round_list(left_handoff_pos, limit=3),
        "sweep_parameters": {
            "random_trials_per_seed": _env_int("OPENFORGE_KITKAT_CLEARANCE_SWEEP_RANDOM_TRIALS", 5),
            "start_joint_perturb_rad": _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_START_JOINT_PERTURB_RAD", 0.015),
            "target_perturb_xyz_m": [
                _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_X_M", 0.020),
                _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_Y_M", 0.020),
                _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_Z_M", 0.020),
            ],
            "target_perturb_yaw_deg": _env_float("OPENFORGE_KITKAT_CLEARANCE_SWEEP_PERTURB_YAW_DEG", 5.0),
        },
        "selected_env": {
            "OPENFORGE_CHOCOLATE_HANDOVER_LEFT_POST_GRIP_CLEARANCE_OFFSET_FROM_HANDOFF_M": ",".join(f"{v:.3f}" for v in left_offsets) if left_offsets else None,
            "OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_X_M": f"{right_offsets[0]:.3f}" if right_offsets else None,
            "OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_Y_M": f"{right_offsets[1]:.3f}" if right_offsets else None,
            "OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_Z_M": f"{right_offsets[2]:.3f}" if right_offsets else None,
        },
        "left": left,
        "right": right,
    }
    result["path"] = write_json(run_dir / "plans" / "handover_clearance_waypoint_sweep.json", result)
    write_json(run_dir / "result.json", result)
    write_json(run_dir / "task_result.json", result)
    append_stage_summary(
        run_dir,
        [
            "## KitKat Handover Clearance Waypoint Sweep",
            f"- success: {result['success']}",
            f"- physical_motion_executed: {result['physical_motion_executed']}",
            f"- seed_count: {result['seed_count']}",
            f"- selected_left_offset_from_handoff_m: {left_offsets}",
            f"- selected_left_stress_score: {(left.get('selected') or {}).get('stress_score')}",
            f"- selected_right_retreat_offset_m: {right_offsets}",
            f"- selected_right_stress_score: {(right.get('selected') or {}).get('stress_score')}",
            f"- selected_env: {result['selected_env']}",
        ],
    )
    _print_compact_result(result)
    return result


if __name__ == "__main__":
    TASK_RESULT.update(_run())
