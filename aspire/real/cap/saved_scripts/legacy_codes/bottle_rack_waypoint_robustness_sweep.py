"""No-motion robustness sweep for held-bottle rack placement waypoints."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

from cap.saved_scripts.yam_runtime import current_run_dir, json_safe, write_json


TASK_NAME = "bottle_rack_waypoint_robustness_sweep"
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


def _offset_triples(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if not raw:
        return [list(item[:3]) for item in default]
    values: list[list[float]] = []
    for chunk in raw.replace("|", ";").split(";"):
        parts = [part.strip() for part in chunk.replace(":", ",").split(",") if part.strip()]
        if len(parts) < 2:
            continue
        xyz = [float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0]
        values.append(xyz)
    return values or [list(item[:3]) for item in default]


def _xyz(item: dict[str, Any], key: str = "position_3d") -> list[float]:
    values = item.get(key) or item.get("position")
    if not values or len(values) < 3:
        raise RuntimeError(f"missing xyz in {item}")
    return [float(values[0]), float(values[1]), float(values[2])]


def _load_source(source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preview_path = source_run / "plans" / "bottle_to_rack_sequence_preview.json"
    obs_path = source_run / "observations" / "observe_rack_target_selected.json"
    if preview_path.exists():
        preview = json.loads(preview_path.read_text(encoding="utf-8"))
    else:
        result_path = source_run / "task_result.json"
        attempt_path = source_run / "plans" / "direct_bottle_to_rack_attempt.json"
        if result_path.exists():
            preview = json.loads(result_path.read_text(encoding="utf-8"))
        elif attempt_path.exists():
            preview = json.loads(attempt_path.read_text(encoding="utf-8"))
        else:
            raise RuntimeError(f"missing source pickup artifact in {source_run}")
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    pickup = preview.get("selected_pickup") or preview.get("selected_strategy")
    rack = obs.get("selected_rack_detection")
    if not isinstance(pickup, dict):
        raise RuntimeError(f"missing selected pickup in {source_run}")
    if not isinstance(rack, dict):
        raise RuntimeError(f"missing selected_rack_detection in {obs_path}")
    return pickup, rack


def _tcp_above_support(selected_pickup: dict[str, Any]) -> float:
    grasp_z = _xyz(selected_pickup.get("grasp_pose") or {}, key="position")[2]
    source_support_z = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_SOURCE_SUPPORT_Z_M", "0.760"))
    value = grasp_z - source_support_z
    if value < 0.035 or value > 0.180:
        return float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_DEFAULT_TCP_ABOVE_SUPPORT_M", "0.080"))
    return value


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _transport_poses(selected_pickup: dict[str, Any], base_rpy: list[float]) -> list[tuple[str, list[float], list[float]]]:
    if not _truthy_env("OPENFORGE_BOTTLE_RACK_SWEEP_ENABLE_TRANSPORT_WAYPOINTS", True):
        return []
    base_pose = selected_pickup.get("lift_pose") or selected_pickup.get("grasp_pose") or {}
    base_pos = _xyz(base_pose, key="position")
    retreat_x = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_TRANSPORT_RETREAT_X_M", "-0.20"))
    shift_y = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_TRANSPORT_SHIFT_Y_M", "-0.28"))
    lift_z = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_TRANSPORT_LIFT_Z_M", "0.10"))
    x_pose = [base_pos[0] + retreat_x, base_pos[1], base_pos[2]]
    y_pose = [x_pose[0], x_pose[1] + shift_y, x_pose[2]]
    z_pose = [y_pose[0], y_pose[1], y_pose[2] + lift_z]
    return [
        ("transport_retreat_x", x_pose, list(base_rpy)),
        ("transport_shift_y", y_pose, list(base_rpy)),
        ("transport_lift_z", z_pose, list(base_rpy)),
    ]


def _retreat_poses(
    *,
    release_pos: list[float],
    rpy: list[float],
    approach: list[float],
    retreat_lift: float,
    retreat_standoff: float,
    retreat_shift_y: float,
) -> list[tuple[str, list[float], list[float]]]:
    back = [
        release_pos[0] - approach[0] * retreat_standoff,
        release_pos[1] - approach[1] * retreat_standoff,
        release_pos[2] + retreat_lift,
    ]
    poses = [
        ("rack_retreat_back_x", back, list(rpy)),
    ]
    if abs(retreat_shift_y) > 1e-9:
        poses.append(("rack_retreat_lane_y", [back[0], back[1] + retreat_shift_y, back[2]], list(rpy)))
    return poses


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
            planner_backend=os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_PLANNER_BACKEND", "rrtconnect"),
            solver_speed=os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_SOLVER_SPEED", "slow"),
            planning_speed=float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_PLANNING_SPEED", "0.40")),
            ik_error_threshold=float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_IK_ERROR_THRESHOLD_M", "0.015")),
            ik_rot_threshold_deg=float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_IK_ROT_THRESHOLD_DEG", "8.0")),
        )
        return {"label": label, "ok": _ok(result), "result": json_safe(result)}
    except Exception as exc:
        return {"label": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _make_perturbations(count: int, *, seed: int, xy_radius_m: float, z_radius_m: float, yaw_radius_deg: float) -> list[dict[str, float]]:
    rng = random.Random(seed)
    samples = [{"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw": 0.0}]
    for _ in range(max(0, count - 1)):
        samples.append(
            {
                "dx": rng.uniform(-xy_radius_m, xy_radius_m),
                "dy": rng.uniform(-xy_radius_m, xy_radius_m),
                "dz": rng.uniform(-z_radius_m, z_radius_m),
                "dyaw": rng.uniform(-yaw_radius_deg, yaw_radius_deg),
            }
        )
    return samples


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    source_run = Path(
        os.environ.get(
            "OPENFORGE_BOTTLE_RACK_SWEEP_SOURCE_RUN",
            "logs/bottle_pour_then_rack_one_shot_loop_20260606T210605",
        )
    )
    pickup, rack = _load_source(source_run)
    rack_xyz = _xyz(rack)
    base_rpy = [float(value) for value in list((pickup.get("grasp_pose") or {}).get("rpy") or [0.0, 90.0, 0.0])[:3]]
    tcp_above_support = _tcp_above_support(pickup)
    rack_surface_z = rack_xyz[2] + float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_SURFACE_Z_OFFSET_M", "0.0"))
    release_clearance = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_RELEASE_CLEARANCE_M", "0.015"))
    retreat_lift = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_RETREAT_LIFT_M", "0.0"))
    retreat_standoff = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_RETREAT_STANDOFF_M", "0.08"))
    retreat_shift_y = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_RETREAT_SHIFT_Y_M", "0.0"))
    approach = [float(value) for value in list(pickup.get("approach_direction_world") or [1.0, 0.0, 0.0])[:3]]
    if len(approach) < 3:
        approach = [1.0, 0.0, 0.0]
    transport_poses = _transport_poses(pickup, base_rpy)
    offsets = _offset_triples(
        "OPENFORGE_BOTTLE_RACK_SWEEP_PLACE_OFFSETS_M",
        [[0.0, 0.03, 0.0], [0.0, 0.06, 0.0], [0.0, 0.09, 0.0], [-0.03, 0.06, 0.0], [0.03, 0.06, 0.0]],
    )
    high_lifts = _float_csv("OPENFORGE_BOTTLE_RACK_SWEEP_HIGH_LIFTS_M", [0.06, 0.08, 0.10, 0.12, 0.16])
    yaw_deltas = _float_csv("OPENFORGE_BOTTLE_RACK_SWEEP_YAW_DELTAS_DEG", [0.0, -10.0, 10.0, -20.0, 20.0])
    sample_count = int(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_SAMPLE_COUNT", "8"))
    perturbations = _make_perturbations(
        sample_count,
        seed=int(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_SEED", "11")),
        xy_radius_m=float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_XY_RADIUS_M", "0.010")),
        z_radius_m=float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_Z_RADIUS_M", "0.008")),
        yaw_radius_deg=float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_YAW_RADIUS_DEG", "2.0")),
    )
    freespace_move = globals()["freespace_move"]
    attempts: list[dict[str, Any]] = []
    for offset in offsets:
        for high_lift in high_lifts:
            for yaw_delta in yaw_deltas:
                sample_packets: list[dict[str, Any]] = []
                full_success = True
                passed_samples = 0
                place_xy = [rack_xyz[0] + float(offset[0]), rack_xyz[1] + float(offset[1])]
                release_z = rack_surface_z + tcp_above_support + release_clearance + float(offset[2])
                for sample_index, perturb in enumerate(perturbations):
                    pos_xy = [place_xy[0] + perturb["dx"], place_xy[1] + perturb["dy"]]
                    rz = release_z + perturb["dz"]
                    rpy = [base_rpy[0], base_rpy[1], base_rpy[2] + yaw_delta + perturb["dyaw"]]
                    poses = list(transport_poses) + [
                        ("rack_high_place", [pos_xy[0], pos_xy[1], rz + float(high_lift)], rpy),
                        ("rack_release_pose", [pos_xy[0], pos_xy[1], rz], rpy),
                    ]
                    poses.extend(
                        _retreat_poses(
                            release_pos=[pos_xy[0], pos_xy[1], rz],
                            rpy=rpy,
                            approach=approach,
                            retreat_lift=retreat_lift,
                            retreat_standoff=retreat_standoff,
                            retreat_shift_y=retreat_shift_y,
                        )
                    )
                    steps = []
                    sample_ok = True
                    for label, pos, step_rpy in poses:
                        step = _preview_pose(freespace_move, pos, step_rpy, f"{label}_sample{sample_index:02d}")
                        steps.append({"stage": label, "pose": {"position": pos, "rpy": step_rpy}, "preview": step, "ok": step["ok"]})
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
                            "success": sample_ok,
                            "steps": steps,
                        }
                    )
                attempts.append(
                    {
                        "relative_offset_m": [float(offset[0]), float(offset[1]), float(offset[2])],
                        "high_lift_m": float(high_lift),
                        "yaw_delta_deg": float(yaw_delta),
                        "success": full_success,
                        "passed_samples": passed_samples,
                        "sample_count": len(perturbations),
                        "release_z_m": release_z,
                        "samples": sample_packets,
                    }
                )
    best = max(
        attempts,
        key=lambda item: (
            int(item["success"]),
            int(item["passed_samples"]),
            float(item["high_lift_m"]),
            -math.hypot(float(item["relative_offset_m"][0]), float(item["relative_offset_m"][1])),
            -abs(float(item["yaw_delta_deg"])),
        ),
    )
    packet = {
        "schema": "openforge.bottle_rack_waypoint_robustness_sweep.v1",
        "success": bool(best["success"]),
        "source_run": str(source_run),
        "rack_xyz_m": rack_xyz,
        "base_rpy_deg": base_rpy,
        "tcp_above_support_m": tcp_above_support,
        "rack_surface_z_m": rack_surface_z,
        "release_clearance_m": release_clearance,
        "retreat_lift_m": retreat_lift,
        "retreat_standoff_m": retreat_standoff,
        "retreat_shift_y_m": retreat_shift_y,
        "approach_direction_world": approach,
        "transport_poses": [
            {"label": label, "position": pos, "rpy": rpy}
            for label, pos, rpy in transport_poses
        ],
        "perturbation_policy": {
            "sample_count": len(perturbations),
            "xy_radius_m": float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_XY_RADIUS_M", "0.010")),
            "z_radius_m": float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_Z_RADIUS_M", "0.008")),
            "yaw_radius_deg": float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_YAW_RADIUS_DEG", "2.0")),
        },
        "best": {
            "relative_offset_m": best["relative_offset_m"],
            "high_lift_m": best["high_lift_m"],
            "yaw_delta_deg": best["yaw_delta_deg"],
            "release_z_m": round(float(best["release_z_m"]), 5),
            "success": best["success"],
            "passed_samples": best["passed_samples"],
            "sample_count": best["sample_count"],
        },
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "bottle_rack_waypoint_robustness_sweep.json", packet)
    print(json.dumps(json_safe(packet), indent=2))
    return packet


if not globals().get("_OPENFORGE_IMPORT_ONLY", False):
    TASK_RESULT.update(_run())
