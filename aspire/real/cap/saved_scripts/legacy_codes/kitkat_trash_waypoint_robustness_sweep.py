# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion robustness sweep for KitKat trash waypoints.

This script does not execute robot motion. It samples fixed offsets relative to
the live trash can pose and previews perturbed target poses with cuRobo. The
goal is to choose trash-stage waypoints that remain preview-reachable under
small target/start-state variance instead of sitting on an IK boundary.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.yam_runtime import append_stage_summary, call_with_timeout, current_run_dir, json_safe, write_json


TASK_NAME = "kitkat_trash_waypoint_robustness_sweep"
TASK_RESULT: dict[str, Any] = {}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


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


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        rows = list(values)
        if limit is not None:
            rows = rows[:limit]
        return [round(float(v), digits) for v in rows]
    except Exception:
        return []


def _xyz(value: Any) -> list[float] | None:
    rows = _round_list(value, limit=3)
    return rows if len(rows) == 3 else None


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _rpy_matrix_deg(rpy_deg: list[float]) -> list[list[float]]:
    roll, pitch, yaw = [math.radians(float(v)) for v in rpy_deg[:3]]
    yaw = -yaw
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(float(mat[i][j]) * float(vec[j]) for j in range(3)) for i in range(3)]


def _tcp_pos_for_gripper_center(center_pos: list[float], rpy_deg: list[float]) -> list[float]:
    offset_local = _env_list("OPENFORGE_TRASH_SWEEP_TCP_TO_GRIPPER_CENTER_LOCAL_M", [0.0, 0.0, 0.045])[:3]
    offset_world = _mat_vec(_rpy_matrix_deg(rpy_deg), offset_local)
    return [float(center_pos[i]) - float(offset_world[i]) for i in range(3)]


def _detect_trash_xyz(run_dir: Path) -> tuple[list[float], dict[str, Any]]:
    manual = _env_list("OPENFORGE_TRASH_SWEEP_TRASH_XYZ_M", [])
    if len(manual) >= 3:
        return [float(v) for v in manual[:3]], {"source": "env", "position_3d": [float(v) for v in manual[:3]]}
    prompts = ["white trash can", "trash can", "bin"]
    result = _required_tool("detect_objects_oneshot")(prompts, camera="top", max_retries=1)
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        for det in list((result or {}).get(prompt) or []):
            xyz = _xyz(_field(det, "position_3d", None))
            if xyz is None:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "score": float(_field(det, "score", 0.0) or 0.0),
                    "position_3d": xyz,
                    "detection": json_safe(det),
                }
            )
    if not rows:
        raise RuntimeError("no live trash detection with 3D position")
    rows.sort(key=lambda row: row["score"], reverse=True)
    packet = {"schema": "openforge.kitkat_trash_sweep.trash_detection.v1", "rows": rows, "selected": rows[0]}
    write_json(run_dir / "plans" / "trash_detection.json", packet)
    return list(rows[0]["position_3d"]), rows[0]


def _right_state() -> dict[str, Any]:
    state = json_safe(_required_tool("get_robot_state")())
    return ((state.get("arms") or {}).get("right") or {})


def _preview(label: str, pose: dict[str, Any], *, gripper: float) -> dict[str, Any]:
    kwargs = {
        "right_target_pos": [float(v) for v in pose["position"][:3]],
        "right_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "right_gripper": float(gripper),
        "preview_only": True,
        "planner_backend": os.environ.get("OPENFORGE_TRASH_SWEEP_PLANNER_BACKEND", "rrtconnect"),
        "planning_speed": _env_float("OPENFORGE_TRASH_SWEEP_PLANNING_SPEED", 0.30),
        "ik_error_threshold": _env_float("OPENFORGE_TRASH_SWEEP_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_TRASH_SWEEP_IK_ROT_THRESHOLD_DEG", 16.0),
    }
    result = call_with_timeout(
        label,
        _required_tool("freespace_move"),
        _env_float("OPENFORGE_TRASH_SWEEP_PREVIEW_TIMEOUT_S", 8.0),
        run_in_background=_tool("run_in_background"),
        **kwargs,
    )
    if not result.get("ok"):
        return {"success": False, "error": result.get("error")}
    data = json_safe(result.get("data"))
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    status = str((data or {}).get("status") or "")
    return {
        "success": status.lower() in {"success", "succeeded", "ok"} and not bool((data or {}).get("executed")),
        "status": status,
        "reason": (data or {}).get("reason") or (data or {}).get("error"),
        "final_pos_error_m": (data or {}).get("final_pos_error_m") or (data or {}).get("ik_error_m"),
        "final_rot_error_deg": (data or {}).get("final_rot_error_deg") or (data or {}).get("ik_rot_error_deg"),
        "trajectory_steps": (data or {}).get("trajectory_steps"),
    }


def _candidate_pose(drop_xy: list[float], x_offset: float, y_offset: float, z: float, yaw: float) -> dict[str, Any]:
    return {
        "position": [float(drop_xy[0] + x_offset), float(drop_xy[1] + y_offset), float(z)],
        "rpy": [0.0, 90.0, float(yaw)],
    }


def _perturbed_pose(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    pos = list(base["position"])
    rpy = list(base["rpy"])
    pos[0] += rng.uniform(-_env_float("OPENFORGE_TRASH_SWEEP_PERTURB_X_M", 0.020), _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_X_M", 0.020))
    pos[1] += rng.uniform(-_env_float("OPENFORGE_TRASH_SWEEP_PERTURB_Y_M", 0.020), _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_Y_M", 0.020))
    pos[2] += rng.uniform(-_env_float("OPENFORGE_TRASH_SWEEP_PERTURB_Z_M", 0.020), _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_Z_M", 0.020))
    rpy[2] += rng.uniform(-_env_float("OPENFORGE_TRASH_SWEEP_PERTURB_YAW_DEG", 10.0), _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_YAW_DEG", 10.0))
    return {"position": pos, "rpy": rpy}


def _score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in results if item.get("success")]
    pos_errors = [float(item["final_pos_error_m"]) for item in successes if item.get("final_pos_error_m") is not None]
    rot_errors = [float(item["final_rot_error_deg"]) for item in successes if item.get("final_rot_error_deg") is not None]
    return {
        "success_count": len(successes),
        "trial_count": len(results),
        "success_rate": round(len(successes) / max(1, len(results)), 4),
        "max_pos_error_m": round(max(pos_errors), 6) if pos_errors else None,
        "mean_pos_error_m": round(sum(pos_errors) / len(pos_errors), 6) if pos_errors else None,
        "max_rot_error_deg": round(max(rot_errors), 4) if rot_errors else None,
        "mean_rot_error_deg": round(sum(rot_errors) / len(rot_errors), 4) if rot_errors else None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    rng = random.Random(_env_int("OPENFORGE_TRASH_SWEEP_SEED", 7))
    trash_xyz, trash_det = _detect_trash_xyz(run_dir)
    bias = _env_list("OPENFORGE_TRASH_SWEEP_TRASH_XY_BIAS_M", [0.010, -0.035])[:2]
    drop_xy = [trash_xyz[0] + bias[0], trash_xyz[1] + bias[1]]
    right = _right_state()
    gripper = float(right.get("gripper_pos") or 0.0)
    x_offsets = _env_list("OPENFORGE_TRASH_SWEEP_X_OFFSETS_M", [-0.24, -0.20, -0.16, -0.12, -0.08])
    y_offsets = _env_list("OPENFORGE_TRASH_SWEEP_Y_OFFSETS_M", [0.0])
    z_values = _env_list("OPENFORGE_TRASH_SWEEP_Z_VALUES_M", [1.12, 1.14, 1.16, 1.18])
    yaw_values = _env_list("OPENFORGE_TRASH_SWEEP_YAWS_DEG", [-75.0, -60.0, -45.0, -30.0, -15.0, 0.0])
    random_trials = _env_int("OPENFORGE_TRASH_SWEEP_RANDOM_TRIALS", 16)
    keep = _env_int("OPENFORGE_TRASH_SWEEP_TOP_K", 10)

    coarse: list[dict[str, Any]] = []
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for z in z_values:
                for yaw in yaw_values:
                    pose = _candidate_pose(drop_xy, x_offset, y_offset, z, yaw)
                    preview = _preview(f"coarse_x{x_offset:.3f}_z{z:.3f}_yaw{yaw:.1f}", pose, gripper=gripper)
                    coarse.append(
                        {
                            "x_offset_m": x_offset,
                            "y_offset_m": y_offset,
                            "z_m": z,
                            "yaw_deg": yaw,
                            "pose": {"position": _round_list(pose["position"], limit=3), "rpy": _round_list(pose["rpy"], limit=3)},
                            "preview": preview,
                            "coarse_success": bool(preview.get("success")),
                        }
                    )
    coarse_successes = [item for item in coarse if item.get("coarse_success")]
    coarse_successes.sort(
        key=lambda item: (
            abs(float(item["x_offset_m"]) + 0.16),
            abs(float(item["z_m"]) - 1.16),
            abs(float(item["yaw_deg"]) + 30.0),
        )
    )

    stress: list[dict[str, Any]] = []
    for item in coarse_successes[:keep]:
        base = _candidate_pose(drop_xy, item["x_offset_m"], item["y_offset_m"], item["z_m"], item["yaw_deg"])
        trials: list[dict[str, Any]] = []
        for trial in range(random_trials):
            pose = _perturbed_pose(base, rng)
            preview = _preview(f"stress_{len(stress):02d}_{trial:02d}", pose, gripper=gripper)
            trials.append(
                {
                    "trial": trial,
                    "pose": {"position": _round_list(pose["position"], limit=3), "rpy": _round_list(pose["rpy"], limit=3)},
                    **preview,
                }
            )
        score = _score_results(trials)
        stress.append({**item, "stress_score": score, "stress_trials": trials})
    stress.sort(
        key=lambda item: (
            -float((item.get("stress_score") or {}).get("success_rate") or 0.0),
            float((item.get("stress_score") or {}).get("max_pos_error_m") or 999.0),
            abs(float(item["z_m"]) - 1.16),
        )
    )
    selected = stress[0] if stress else None

    final_rpy = [0.0, 90.0, 0.0]
    drop_z = _env_float("OPENFORGE_TRASH_SWEEP_DROP_Z_M", 1.120)
    transport_z = float(selected["z_m"]) if selected else _env_float("OPENFORGE_TRASH_SWEEP_TRANSPORT_Z_M", 1.160)
    route_check = []
    if selected is not None:
        stage_pose = _candidate_pose(drop_xy, selected["x_offset_m"], selected["y_offset_m"], selected["z_m"], selected["yaw_deg"])
        orient_pose = {"position": list(stage_pose["position"]), "rpy": final_rpy}
        over_pose = {"position": _tcp_pos_for_gripper_center([drop_xy[0], drop_xy[1], transport_z], final_rpy), "rpy": final_rpy}
        drop_pose = {"position": _tcp_pos_for_gripper_center([drop_xy[0], drop_xy[1], drop_z], final_rpy), "rpy": final_rpy}
        for label, pose in [
            ("selected_stage", stage_pose),
            ("selected_orient_plus_x", orient_pose),
            ("selected_over_bin", over_pose),
            ("selected_drop", drop_pose),
        ]:
            route_check.append(
                {
                    "label": label,
                    "pose": {"position": _round_list(pose["position"], limit=3), "rpy": _round_list(pose["rpy"], limit=3)},
                    "preview": _preview(label, pose, gripper=gripper),
                }
            )

    result = {
        "schema": "openforge.kitkat_trash_waypoint_robustness_sweep.v1",
        "success": selected is not None,
        "physical_motion_executed": False,
        "run_dir": str(run_dir),
        "trash_detection": trash_det,
        "trash_xyz_m": _round_list(trash_xyz, limit=3),
        "drop_xy_m": _round_list(drop_xy, limit=2),
        "right_state": {
            "ee_pos": _round_list(right.get("ee_pos") or [], limit=3),
            "joint_pos": _round_list(right.get("joint_pos") or [], limit=6),
            "gripper_pos": gripper,
        },
        "sweep_parameters": {
            "x_offsets_m": x_offsets,
            "y_offsets_m": y_offsets,
            "z_values_m": z_values,
            "yaw_values_deg": yaw_values,
            "random_trials": random_trials,
            "perturb_x_m": _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_X_M", 0.020),
            "perturb_y_m": _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_Y_M", 0.020),
            "perturb_z_m": _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_Z_M", 0.020),
            "perturb_yaw_deg": _env_float("OPENFORGE_TRASH_SWEEP_PERTURB_YAW_DEG", 10.0),
        },
        "coarse_count": len(coarse),
        "coarse_success_count": len(coarse_successes),
        "selected": selected,
        "route_check": route_check,
        "coarse": coarse,
        "stress": stress,
    }
    result["path"] = write_json(run_dir / "plans" / "trash_waypoint_robustness_sweep.json", result)
    write_json(run_dir / "result.json", result)
    write_json(run_dir / "task_result.json", result)
    append_stage_summary(
        run_dir,
        [
            "## KitKat Trash Waypoint Robustness Sweep",
            f"- success: {result['success']}",
            f"- physical_motion_executed: {result['physical_motion_executed']}",
            f"- trash_xyz_m: {result['trash_xyz_m']}",
            f"- drop_xy_m: {result['drop_xy_m']}",
            f"- coarse_success_count: {result['coarse_success_count']} / {result['coarse_count']}",
            f"- selected: {selected and {k: selected[k] for k in ('x_offset_m', 'y_offset_m', 'z_m', 'yaw_deg')}}",
            f"- selected_stress_score: {selected and selected.get('stress_score')}",
            f"- route_check_success: {[bool((item.get('preview') or {}).get('success')) for item in route_check]}",
        ],
    )
    print(json.dumps(json_safe(result), indent=2))
    return result


TASK_RESULT.update(_run())
