"""Preview-only sweep for bottle-rack recovery poses from the current robot state."""

from __future__ import annotations

import json
import os
from itertools import product

from cap.saved_scripts.yam_runtime import append_stage_summary, current_run_dir, json_safe, write_json


TASK_NAME = "preview_bottle_rack_recovery_waypoints"


def _floats(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            pass
    return values or list(default)


def _preview(freespace_move, pos: list[float], rpy: list[float], label: str) -> dict:
    try:
        result = freespace_move(
            left_target_pos=pos,
            left_target_rpy=rpy,
            planner_backend=os.environ.get("OPENFORGE_BOTTLE_RACK_PLANNER_BACKEND", "rrtconnect"),
            solver_speed=os.environ.get("OPENFORGE_BOTTLE_RACK_SOLVER_SPEED", "slow"),
            planning_speed=float(os.environ.get("OPENFORGE_BOTTLE_RACK_PLANNING_SPEED", "0.40")),
            preview_only=True,
        )
        safe = json_safe(result)
        return {
            "label": label,
            "success": bool(safe.get("trajectory_cache_key")) if isinstance(safe, dict) else False,
            "result": safe,
        }
    except Exception as exc:
        return {"label": label, "success": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> dict:
    run_dir = current_run_dir(TASK_NAME)
    freespace_move = globals().get("freespace_move")
    if not callable(freespace_move):
        raise RuntimeError("freespace_move tool is unavailable")

    rack_xyz = _floats("OPENFORGE_BOTTLE_RACK_SWEEP_RACK_XYZ", [0.88916, -0.08158, 0.91343])[:3]
    release_z = float(os.environ.get("OPENFORGE_BOTTLE_RACK_SWEEP_RELEASE_Z", "1.04257"))
    rpy = _floats("OPENFORGE_BOTTLE_RACK_SWEEP_RPY", [0.0, 90.0, 0.0])[:3]
    dxs = _floats("OPENFORGE_BOTTLE_RACK_SWEEP_DX", [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02])
    dys = _floats("OPENFORGE_BOTTLE_RACK_SWEEP_DY", [0.0, 0.03, 0.06, 0.09, 0.12])
    dzs = _floats("OPENFORGE_BOTTLE_RACK_SWEEP_DZ", [0.0, 0.03, 0.06])
    high_lifts = _floats("OPENFORGE_BOTTLE_RACK_SWEEP_HIGH_LIFT", [0.08, 0.12, 0.16])

    attempts = []
    passing = []
    for dx, dy, dz, high_lift in product(dxs, dys, dzs, high_lifts):
        release = [rack_xyz[0] + dx, rack_xyz[1] + dy, release_z + dz]
        high = [release[0], release[1], release[2] + high_lift]
        high_preview = _preview(freespace_move, high, rpy, "rack_high_place")
        release_preview = _preview(freespace_move, release, rpy, "rack_release_pose")
        attempt = {
            "offset": [round(dx, 5), round(dy, 5), round(dz, 5)],
            "high_lift": round(high_lift, 5),
            "high_pose": {"position": high, "rpy": rpy},
            "release_pose": {"position": release, "rpy": rpy},
            "high_preview": high_preview,
            "release_preview": release_preview,
            "success": bool(high_preview.get("success") and release_preview.get("success")),
        }
        attempts.append(attempt)
        if attempt["success"]:
            passing.append(attempt)

    packet = {
        "schema": "openforge.bottle_rack.recovery_preview_sweep.v1",
        "rack_xyz": rack_xyz,
        "release_z": release_z,
        "rpy": rpy,
        "attempt_count": len(attempts),
        "passing_count": len(passing),
        "passing": passing[:20],
        "attempts": attempts,
    }
    write_json(run_dir / "sweep.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## bottle rack recovery preview sweep",
            f"- passing_count: {len(passing)}",
            f"- first_passing: {passing[0]['offset'] if passing else None}",
            f"- path: {run_dir / 'sweep.json'}",
        ],
    )
    print(json.dumps(packet, indent=2))
    return packet


TASK_RESULT = main()
