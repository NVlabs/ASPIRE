# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion robustness sweep for the simple fixed bottle pour waypoints."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

from cap.saved_scripts.yam_runtime import current_run_dir, json_safe, write_json


TASK_NAME = "bottle_fixed_pour_waypoint_robustness_sweep"
TASK_RESULT: dict[str, Any] = {}


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values or list(default)


def _load_source_result(source_run: Path) -> dict[str, Any]:
    packet_path = source_run / "subtasks" / "dry_pour_result.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    result = packet.get("result") if isinstance(packet.get("result"), dict) else packet
    if not isinstance(result, dict):
        raise RuntimeError(f"could not parse dry-pour result from {packet_path}")
    return result


def _xyz(item: dict[str, Any], key: str = "position_3d") -> list[float]:
    values = item.get(key) or item.get("position")
    if not values or len(values) < 3:
        raise RuntimeError(f"missing xyz in {item}")
    return [float(values[0]), float(values[1]), float(values[2])]


def _yaw_deg_from_xy(vec: list[float]) -> float:
    return math.degrees(math.atan2(float(vec[1]), float(vec[0])))


def _ok(result: Any) -> bool:
    safe = json_safe(result)
    if isinstance(safe, dict):
        status = safe.get("status")
        return bool(safe.get("ok") or safe.get("success") or status == "Success")
    return False


def _preview_pose(freespace_move: Any, pos: list[float], rpy: list[float], label: str) -> dict[str, Any]:
    try:
        result = freespace_move(
            left_target_pos=[float(value) for value in pos[:3]],
            left_target_rpy=[float(value) for value in rpy[:3]],
            preview_only=True,
            planner_backend=os.environ.get("OPENFORGE_BOTTLE_SWEEP_PLANNER_BACKEND", "rrtconnect"),
            solver_speed=os.environ.get("OPENFORGE_BOTTLE_SWEEP_SOLVER_SPEED", "slow"),
            planning_speed=float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_PLANNING_SPEED", "0.40")),
            ik_error_threshold=float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_IK_ERROR_THRESHOLD_M", "0.015")),
            ik_rot_threshold_deg=float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_IK_ROT_THRESHOLD_DEG", "8.0")),
        )
        return {"label": label, "ok": _ok(result), "result": json_safe(result)}
    except Exception as exc:
        return {"label": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _make_perturbations(count: int, *, seed: int, xyz_radius_m: float, z_radius_m: float, yaw_radius_deg: float) -> list[dict[str, float]]:
    rng = random.Random(seed)
    samples = [{"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}]
    for _ in range(max(0, count - 1)):
        samples.append(
            {
                "dx": rng.uniform(-xyz_radius_m, xyz_radius_m),
                "dy": rng.uniform(-xyz_radius_m, xyz_radius_m),
                "dz": rng.uniform(-z_radius_m, z_radius_m),
                "dyaw": rng.uniform(-yaw_radius_deg, yaw_radius_deg),
            }
        )
    return samples


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    source_run = Path(
        os.environ.get(
            "OPENFORGE_BOTTLE_SWEEP_SOURCE_RUN",
            "logs/bottle_pour_then_rack_one_shot_loop_20260606T205237",
        )
    )
    result = _load_source_result(source_run)
    bottle_xyz = _xyz(result["bottle_detection"])
    cup_xyz = _xyz(result["cup_detection"])
    pickup = (result.get("bottle_plan") or {}).get("selected") or {}
    lift_pose = pickup.get("lift_pose") or {}
    lift_xyz = _xyz(lift_pose, key="position")
    base_offset_xy = _float_csv("OPENFORGE_BOTTLE_SWEEP_BASE_OFFSET_XY_M", [-0.07760, 0.05267])
    base_xy = [bottle_xyz[0] + base_offset_xy[0], bottle_xyz[1] + base_offset_xy[1]]
    base_yaw = _yaw_deg_from_xy([bottle_xyz[0] - cup_xyz[0], bottle_xyz[1] - cup_xyz[1]])
    z_values = _float_csv("OPENFORGE_BOTTLE_SWEEP_TARGET_Z_VALUES_M", [1.105, 1.115, 1.125, 1.135, 1.145])
    yaw_deltas = _float_csv("OPENFORGE_BOTTLE_SWEEP_BASE_YAW_DELTAS_DEG", [0.0])
    pitch_sequence = _float_csv("OPENFORGE_BOTTLE_SWEEP_PITCH_SEQUENCE_DEG", [90.0, 75.0, 60.0, 45.0, 40.0])
    sample_count = int(os.environ.get("OPENFORGE_BOTTLE_SWEEP_SAMPLE_COUNT", "14"))
    perturbations = _make_perturbations(
        sample_count,
        seed=int(os.environ.get("OPENFORGE_BOTTLE_SWEEP_SEED", "7")),
        xyz_radius_m=float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_XY_RADIUS_M", "0.012")),
        z_radius_m=float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_Z_RADIUS_M", "0.008")),
        yaw_radius_deg=float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_YAW_RADIUS_DEG", "3.0")),
    )
    freespace_move = globals()["freespace_move"]
    attempts: list[dict[str, Any]] = []
    for target_z in z_values:
        for yaw_delta in yaw_deltas:
            sample_packets: list[dict[str, Any]] = []
            full_success = True
            passed_samples = 0
            candidate_yaw = base_yaw + float(yaw_delta)
            for sample_index, perturb in enumerate(perturbations):
                sample_xy = [base_xy[0] + perturb["dx"], base_xy[1] + perturb["dy"]]
                sample_z = float(target_z) + perturb["dz"]
                sample_yaw = candidate_yaw + perturb["dyaw"]
                steps: list[dict[str, Any]] = []
                sample_ok = True
                for pitch in pitch_sequence:
                    step = _preview_pose(
                        freespace_move,
                        [sample_xy[0], sample_xy[1], sample_z],
                        [0.0, float(pitch), sample_yaw],
                        f"z{target_z:.3f}_yaw{yaw_delta:+.1f}_sample{sample_index:02d}_pitch{pitch:.1f}",
                    )
                    steps.append(step)
                    if not step["ok"]:
                        sample_ok = False
                        full_success = False
                        break
                if sample_ok:
                    passed_samples += 1
                sample_packets.append(
                    {
                        "sample_index": sample_index,
                        "perturbation": perturb,
                        "target_xyz": [round(sample_xy[0], 5), round(sample_xy[1], 5), round(sample_z, 5)],
                        "yaw_deg": round(sample_yaw, 5),
                        "success": sample_ok,
                        "steps": steps,
                    }
                )
            attempts.append(
                {
                    "target_z_m": float(target_z),
                    "base_yaw_delta_deg": float(yaw_delta),
                    "base_candidate_yaw_deg": candidate_yaw,
                    "success": full_success,
                    "passed_samples": passed_samples,
                    "sample_count": len(perturbations),
                    "samples": sample_packets,
                }
            )
    best = max(
        attempts,
        key=lambda item: (
            int(item["success"]),
            int(item["passed_samples"]),
            float(item["target_z_m"]),
            -abs(float(item["base_yaw_delta_deg"])),
        ),
    )
    packet = {
        "schema": "openforge.bottle_fixed_pour_waypoint_robustness_sweep.v1",
        "success": bool(best["success"]),
        "source_run": str(source_run),
        "bottle_xyz_m": bottle_xyz,
        "cup_xyz_m": cup_xyz,
        "lift_xyz_m": lift_xyz,
        "base_offset_xy_m": base_offset_xy[:2],
        "base_xy_m": base_xy,
        "base_yaw_deg": base_yaw,
        "base_yaw_delta_candidates_deg": yaw_deltas,
        "pitch_sequence_deg": pitch_sequence,
        "perturbation_policy": {
            "sample_count": len(perturbations),
            "xy_radius_m": float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_XY_RADIUS_M", "0.012")),
            "z_radius_m": float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_Z_RADIUS_M", "0.008")),
            "yaw_radius_deg": float(os.environ.get("OPENFORGE_BOTTLE_SWEEP_YAW_RADIUS_DEG", "3.0")),
        },
        "best": {
            "target_z_m": best["target_z_m"],
            "base_yaw_delta_deg": best["base_yaw_delta_deg"],
            "base_candidate_yaw_deg": round(float(best["base_candidate_yaw_deg"]), 5),
            "success": best["success"],
            "passed_samples": best["passed_samples"],
            "sample_count": best["sample_count"],
            "recommended_offset_z_from_bottle_m": round(float(best["target_z_m"]) - bottle_xyz[2], 5),
            "recommended_min_z_above_lift_m": round(float(best["target_z_m"]) - lift_xyz[2], 5),
        },
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "bottle_fixed_pour_waypoint_robustness_sweep.json", packet)
    print(json.dumps(json_safe(packet), indent=2))
    return packet


if not globals().get("_OPENFORGE_IMPORT_ONLY", False):
    TASK_RESULT.update(_run())
