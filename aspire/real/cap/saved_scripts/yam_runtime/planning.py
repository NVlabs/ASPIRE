# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion-candidate helpers for YAM saved scripts."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import append_stage_summary, call_with_timeout, current_run_dir, write_json


def _arm_from_y(y: float) -> str:
    if y > 0.04:
        return "left"
    if y < -0.04:
        return "right"
    return "right"


def _radius(half_extents: list[float] | None, default: float) -> float:
    if half_extents and len(half_extents) >= 2:
        vals = sorted(abs(float(v)) for v in half_extents[:2])
        if vals[0] > 0:
            return vals[0]
    return default


def _body_z(center_z: float, half_extents: list[float] | None, default_half_height: float, body_fraction: float) -> float:
    half_height = default_half_height
    if half_extents and len(half_extents) >= 3 and abs(float(half_extents[2])) > 0:
        half_height = abs(float(half_extents[2]))
    bottom = float(center_z) - half_height
    return bottom + 2.0 * half_height * max(0.2, min(0.75, float(body_fraction)))


def _has_height_extent(half_extents: list[float] | None) -> bool:
    return bool(half_extents and len(half_extents) >= 3 and abs(float(half_extents[2])) > 0)


def _round_list(values: Any, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in list(values)[:3]]


def _normalize_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _horizontal_unit(axis: Any, fallback: list[float]) -> list[float]:
    values = [float(v) for v in list(axis)[:3]]
    if len(values) < 3:
        values = list(fallback)
    values[2] = 0.0
    norm = math.hypot(values[0], values[1])
    if norm < 1e-6:
        values = [float(fallback[0]), float(fallback[1]), 0.0]
        norm = math.hypot(values[0], values[1])
    if norm < 1e-6:
        return [1.0, 0.0, 0.0]
    return [values[0] / norm, values[1] / norm, 0.0]


def _display_rpy_axes(rpy: list[float]) -> dict[str, Any]:
    roll_f = float(rpy[0])
    pitch_f = float(rpy[1])
    yaw_f = float(rpy[2])
    if abs(roll_f) < 1e-6 and abs(abs(pitch_f) - 90.0) < 1e-6:
        yaw_rad = math.radians(yaw_f)
        if pitch_f >= 0.0:
            x_axis = [-math.sin(yaw_rad), -math.cos(yaw_rad), 0.0]
            y_axis = [0.0, 0.0, -1.0]
            z_axis = [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0]
        else:
            x_axis = [math.sin(yaw_rad), math.cos(yaw_rad), 0.0]
            y_axis = [0.0, 0.0, 1.0]
            z_axis = [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0]
        return {
            "local_x_opening_axis": _round_list(x_axis),
            "local_y_height_axis": _round_list(y_axis),
            "local_z_approach_axis": _round_list(z_axis),
            "source": "analytic_side_display_rpy",
        }
    try:
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
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _candidate(
    *,
    center: list[float],
    arm: str,
    yaw_deg: float,
    pitch_deg: float,
    radius_m: float,
    pregrasp_standoff_m: float,
    lift_z_m: float,
    width_margin_m: float,
    score: float,
    label: str,
    z_offset_m: float = 0.0,
    source_detection: dict[str, Any] | None = None,
    body_z_source: str | None = None,
    grasp_backoff_m: float = 0.0,
) -> dict[str, Any]:
    yaw_rad = math.radians(yaw_deg)
    requested_approach = [math.cos(yaw_rad), math.sin(yaw_rad), 0.0]
    rpy = [0.0, float(pitch_deg), float(yaw_deg)]
    axes = _display_rpy_axes(rpy)
    approach = _horizontal_unit(axes.get("local_z_approach_axis", requested_approach), requested_approach)
    actual_yaw_deg = _normalize_angle_deg(math.degrees(math.atan2(approach[1], approach[0])))
    object_center = [float(center[0]), float(center[1]), float(center[2])]
    grasp = [
        object_center[0] - approach[0] * float(grasp_backoff_m),
        object_center[1] - approach[1] * float(grasp_backoff_m),
        object_center[2],
    ]
    pre = [
        grasp[0] - approach[0] * pregrasp_standoff_m,
        grasp[1] - approach[1] * pregrasp_standoff_m,
        grasp[2],
    ]
    lift = [grasp[0], grasp[1], grasp[2] + lift_z_m]
    width = max(0.02, min(0.95, 2.0 * radius_m + width_margin_m))
    return {
        "label": label,
        "arm": arm,
        "position": grasp,
        "rpy": rpy,
        "score": float(score),
        "width": width,
        "pregrasp_pose": {"position": pre, "rpy": rpy},
        "grasp_pose": {"position": grasp, "rpy": rpy},
        "lift_pose": {"position": lift, "rpy": rpy},
        "approach_yaw_deg": float(actual_yaw_deg),
        "requested_approach_yaw_deg": float(yaw_deg),
        "approach_direction_world": _round_list(approach),
        "gripper_local_axes_world": axes,
        "z_offset_m": float(z_offset_m),
        "estimated_radius_m": float(radius_m),
        "estimated_object_width_m": float(2.0 * radius_m),
        "source_detection": source_detection,
        "body_z_source": body_z_source,
        "object_center": _round_list(object_center),
        "grasp_backoff_m": float(grasp_backoff_m),
    }


def generate_side_grasp_candidates(
    detection: dict[str, Any],
    *,
    object_kind: str = "object",
    arm: str | None = None,
    default_radius_m: float = 0.035,
    default_half_height_m: float = 0.12,
    body_fraction: float = 0.45,
    pregrasp_standoff_m: float = 0.085,
    lift_z_m: float = 0.08,
    width_margin_m: float = 0.015,
    include_topdown: bool = False,
    yaw_angles_deg: list[float] | tuple[float, ...] | None = None,
    z_offsets_m: list[float] | tuple[float, ...] | None = None,
    center_z_offset_without_extents_m: float | None = None,
    grasp_backoff_m: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate simple side-grasp candidates from a live detection."""
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        raise ValueError("detection must contain position_3d")
    half_extents = detection.get("half_extents") or []
    if center_z_offset_without_extents_m is not None and not _has_height_extent(half_extents):
        center_z = float(xyz[2]) + float(center_z_offset_without_extents_m)
        body_z_source = "detection_z_plus_offset_without_height_extents"
    else:
        center_z = _body_z(float(xyz[2]), half_extents, default_half_height_m, body_fraction)
        body_z_source = "detection_center_and_height_extents"
    center = [float(xyz[0]), float(xyz[1]), center_z]
    selected_arm = arm or _arm_from_y(float(center[1]))
    radius = _radius(half_extents, default_radius_m)
    base_yaw = 90.0 if selected_arm == "left" else -90.0
    yaws = list(yaw_angles_deg) if yaw_angles_deg is not None else [
        base_yaw,
        base_yaw + 20.0,
        base_yaw - 20.0,
        0.0,
        180.0,
    ]
    z_offsets = list(z_offsets_m) if z_offsets_m is not None else [0.0]
    candidates = []
    idx = 0
    for z_index, z_offset in enumerate(z_offsets):
        z_center = [center[0], center[1], center[2] + float(z_offset)]
        for yaw_index, yaw in enumerate(yaws):
            candidates.append(
                _candidate(
                    center=z_center,
                    arm=selected_arm,
                    yaw_deg=yaw,
                    pitch_deg=90.0,
                    radius_m=radius,
                    pregrasp_standoff_m=pregrasp_standoff_m,
                    lift_z_m=lift_z_m,
                    width_margin_m=width_margin_m,
                    score=1.0 - 0.05 * yaw_index - 0.04 * z_index,
                    label=f"{object_kind}_side_{idx}",
                    z_offset_m=float(z_offset),
                    source_detection=detection,
                    body_z_source=body_z_source,
                    grasp_backoff_m=grasp_backoff_m,
                )
            )
            idx += 1
    if include_topdown:
        candidates.append(
            _candidate(
                center=center,
                arm=selected_arm,
                yaw_deg=base_yaw,
                pitch_deg=180.0,
                radius_m=radius,
                pregrasp_standoff_m=pregrasp_standoff_m,
                lift_z_m=lift_z_m,
                width_margin_m=width_margin_m,
                score=0.5,
                label=f"{object_kind}_topdown",
                source_detection=detection,
                body_z_source=body_z_source,
                grasp_backoff_m=grasp_backoff_m,
            )
        )
    return candidates


def _preview_pose(
    freespace_move: Callable[..., Any],
    arm: str,
    pose: dict[str, Any],
    *,
    timeout_s: float,
    run_in_background: Callable[..., Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    prefix = "left" if arm == "left" else "right"
    params = {
        f"{prefix}_target_pos": pose["position"],
        f"{prefix}_target_rpy": pose["rpy"],
        "preview_only": True,
    }
    params.update(kwargs)
    result = call_with_timeout(
        f"freespace_preview:{arm}",
        freespace_move,
        timeout_s,
        run_in_background=run_in_background,
        **params,
    )
    return result


def rank_motion_candidates(
    *,
    candidates: list[dict[str, Any]],
    freespace_move: Callable[..., Any],
    run_in_background: Callable[..., Any] | None = None,
    run_dir: str | Path | None = None,
    stage: str = "plan",
    task_name: str = "yam_runtime",
    timeout_s: float = 30.0,
    planner_backend: str = "curobo",
    solver_speed: str = "fast",
    planning_speed: float = 0.20,
    ik_error_threshold: float = 0.02,
    ik_rot_threshold_deg: float = 12.0,
    ik_xyz_weight: float | None = None,
    ik_rpy_weight: float | None = None,
    stop_after_successes: int | None = None,
    pose_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Preview pregrasp/grasp/lift poses and select the first feasible candidate."""
    run_dir = Path(run_dir) if run_dir is not None else current_run_dir(task_name)
    keys = tuple(pose_keys or ("pregrasp_pose", "grasp_pose", "lift_pose"))
    ranked: list[dict[str, Any]] = []
    success_count = 0
    for idx, cand in enumerate(candidates):
        arm = cand["arm"]
        preview_kwargs = {
            "planner_backend": planner_backend,
            "solver_speed": solver_speed,
            "planning_speed": planning_speed,
            "ik_error_threshold": ik_error_threshold,
            "ik_rot_threshold_deg": ik_rot_threshold_deg,
        }
        if ik_xyz_weight is not None:
            preview_kwargs["ik_xyz_weight"] = ik_xyz_weight
        if ik_rpy_weight is not None:
            preview_kwargs["ik_rpy_weight"] = ik_rpy_weight
        previews = []
        ok = True
        for key in keys:
            result = _preview_pose(
                freespace_move,
                arm,
                cand[key],
                timeout_s=timeout_s,
                run_in_background=run_in_background,
                **preview_kwargs,
            )
            previews.append({"stage": key, "ok": result["ok"], "result": result})
            if not result["ok"]:
                ok = False
                break
        item = dict(cand)
        item.update({"candidate_index": idx, "preview_success": ok, "previews": previews})
        ranked.append(item)
        if ok:
            success_count += 1
            if stop_after_successes is not None and success_count >= max(1, int(stop_after_successes)):
                break
    selected = next((item for item in ranked if item["preview_success"]), None)
    packet = {
        "schema": "openforge.yam_runtime.plan.v1",
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected": selected,
        "candidates": ranked,
    }
    plan_path = run_dir / "plans" / f"{time.strftime('%Y%m%dT%H%M%S')}_{stage}" / "plan.json"
    packet["plan_path"] = write_json(plan_path, packet)
    append_stage_summary(
        run_dir,
        [
            f"## plan {stage}",
            f"- plan: {packet['plan_path']}",
            f"- candidates: {len(ranked)}",
            f"- selected: {selected.get('label') if selected else None}",
        ],
    )
    return packet
