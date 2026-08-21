# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-motion probe for cyan-cup wooden-rack shelf place poses."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from cap.saved_scripts.yam_runtime import current_run_dir, json_safe, write_json


TASK_NAME = "probe_cup_rack_place_poses"


def _tool(name: str):
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str):
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_float(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    out: list[float] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out or list(default)


def _candidate_grid() -> list[dict[str, Any]]:
    rack_x = _env_float("OPENFORGE_PROBE_RACK_X_M", 0.866)
    rack_y = _env_float("OPENFORGE_PROBE_RACK_Y_M", 0.029)
    xs = [rack_x + bias for bias in _csv_float("OPENFORGE_PROBE_X_BIASES_M", [-0.015, -0.045, -0.075, -0.105])]
    ys = [rack_y + bias for bias in _csv_float("OPENFORGE_PROBE_Y_BIASES_M", [0.075, 0.105, 0.045, 0.015])]
    zs = _csv_float("OPENFORGE_PROBE_ZS_M", [0.93, 0.95, 0.97, 0.99, 1.01, 1.05])
    topdown_yaws = _csv_float("OPENFORGE_PROBE_TOPDOWN_YAWS_DEG", [0, 45, -45, 90, -90])
    side_yaws = _csv_float("OPENFORGE_PROBE_SIDE_YAWS_DEG", [-70, -90, -110, -55, -125])
    candidates: list[dict[str, Any]] = []
    idx = 0
    for x in xs:
        for y in ys:
            for z in zs:
                for yaw in topdown_yaws:
                    idx += 1
                    candidates.append(
                        {
                            "source_index": idx,
                            "label": f"topdown_x{x:.3f}_y{y:.3f}_z{z:.3f}_yaw{yaw:.0f}",
                            "family": "topdown",
                            "position": [x, y, z],
                            "rpy": [0.0, 180.0, yaw],
                            "score": 2.0 - 0.01 * idx,
                        }
                    )
                for yaw in side_yaws:
                    idx += 1
                    candidates.append(
                        {
                            "source_index": idx,
                            "label": f"side_x{x:.3f}_y{y:.3f}_z{z:.3f}_yaw{yaw:.0f}",
                            "family": "side",
                            "position": [x, y, z],
                            "rpy": [0.0, -90.0, yaw],
                            "score": 1.0 - 0.01 * idx,
                        }
                    )
                for yaw in (0.0, 45.0, -45.0, 90.0, -90.0):
                    idx += 1
                    candidates.append(
                        {
                            "source_index": idx,
                            "label": f"upright_x{x:.3f}_y{y:.3f}_z{z:.3f}_yaw{yaw:.0f}",
                            "family": "upright",
                            "position": [x, y, z],
                            "rpy": [0.0, 90.0, yaw],
                            "score": 0.5 - 0.01 * idx,
                        }
                    )
    return candidates


def _rows_by_source(raw: Any) -> dict[int, dict[str, Any]]:
    data = json_safe(raw)
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    rows = data.get("batch_candidates") if isinstance(data, dict) else []
    out: dict[int, dict[str, Any]] = {}
    for row in rows or []:
        source_index = row.get("source_index")
        if source_index is None:
            continue
        out[int(source_index)] = row
    return out


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(Path("logs") / TASK_NAME)
    candidates = _candidate_grid()
    side = os.environ.get("OPENFORGE_PROBE_ARM", "left").strip().lower() or "left"
    limit = int(_env_float("OPENFORGE_PROBE_TOP_K", min(256, len(candidates))))
    raw = _required_tool("freespace_move")(
        grasp_candidates=candidates,
        batch_side=side,
        batch_top_k=limit,
        batch_validate_trajectory=_truthy_env("OPENFORGE_PROBE_VALIDATE_TRAJECTORY", False),
        planner_backend="curobo",
        solver_speed=os.environ.get("OPENFORGE_PROBE_SOLVER_SPEED", "fast"),
        planning_speed=_env_float("OPENFORGE_PROBE_PLANNING_SPEED", 0.30),
        ik_error_threshold=_env_float("OPENFORGE_PROBE_IK_ERROR_THRESHOLD_M", 0.030),
        ik_rot_threshold_deg=_env_float("OPENFORGE_PROBE_IK_ROT_THRESHOLD_DEG", 16.0),
        ik_xyz_weight=1.0,
        ik_rpy_weight=_env_float("OPENFORGE_PROBE_IK_RPY_WEIGHT", 0.35),
    )
    rows = _rows_by_source(raw)
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        row = rows.get(int(candidate["source_index"]))
        merged = {
            "source_index": candidate["source_index"],
            "label": candidate["label"],
            "family": candidate["family"],
            "position": candidate["position"],
            "rpy": candidate["rpy"],
            "rank_row": row,
        }
        if row:
            merged["executable"] = bool(row.get("motion_plan_success") or row.get("trajectory_cache_key"))
            merged["ik_error_m"] = row.get("position_error_m", row.get("ik_error_m"))
            merged["rot_error_deg"] = row.get("rotation_error_deg", row.get("rot_error_deg"))
            merged["motion_plan_reason"] = row.get("motion_plan_reason")
        else:
            merged["executable"] = False
        annotated.append(merged)
    feasible = [
        row
        for row in annotated
        if row.get("executable")
        and (row.get("ik_error_m") is None or float(row["ik_error_m"]) <= 0.030)
        and (row.get("rot_error_deg") is None or float(row["rot_error_deg"]) <= 16.0)
    ]
    result = {
        "schema": "openforge.probe_cup_rack_place_poses.v1",
        "arm": side,
        "candidate_count": len(candidates),
        "batch_top_k": limit,
        "raw": json_safe(raw),
        "feasible_count": len(feasible),
        "best_feasible": feasible[:20],
        "top_ranked": [row for row in annotated if row.get("rank_row")][:30],
    }
    write_json(run_dir / "probe_cup_rack_place_poses.json", result)
    write_json(run_dir / "result.json", result)
    print(json.dumps(json_safe(result), indent=2))
    return result


TASK_RESULT = _run()
