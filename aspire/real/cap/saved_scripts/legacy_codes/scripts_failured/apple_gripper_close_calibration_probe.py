"""Read-only evidence probe for apple width-limited gripper close.

This script never commands arm or gripper motion. It collects repo/model
constants, current arm-server gripper health/state, and apple close calibration
environment variables so the runtime can decide whether a physical apple close
is allowed.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from skill_library.debug_observation import current_run_dir


_SCRIPT_FILE = globals().get("__file__")
ROOT = Path(_SCRIPT_FILE).resolve().parents[2] if _SCRIPT_FILE else Path.cwd()
PLAN_LOG_DIR = os.environ.get("OPENFORGE_APPLE_PLAN_LOG_DIR", "").strip()
PLAN_RESULT_JSON = os.environ.get("OPENFORGE_APPLE_PLAN_RESULT_JSON", "").strip()
MIN_TARGET_POS = float(os.environ.get("OPENFORGE_APPLE_CLOSE_MIN_TARGET_POS", "0.08"))

TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "apple_gripper_close_calibration_probe",
    "safe_read_only": True,
    "physical_motion_executed": False,
    "movement_capable_calls": [],
}


def get_task_info() -> dict[str, Any]:
    return TASK_RESULT


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_plan_width() -> dict[str, Any]:
    raw_path = PLAN_RESULT_JSON or PLAN_LOG_DIR
    if not raw_path:
        return {"source": None, "selected_width_m": None}
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "result.json"
    data = json.loads(path.read_text())
    details = data.get("details", data)
    plan = details.get("plan", {}) if isinstance(details, dict) else {}
    selected = plan.get("selected_grasp", {}) if isinstance(plan, dict) else {}
    return {
        "source": str(path),
        "selected_width_m": selected.get("width"),
        "selected_grasp": selected,
    }


def _model_width_evidence() -> dict[str, Any]:
    sim_path = ROOT / "robot/yam/yam_sim_env.py"
    xml_path = ROOT / "robot/models/station/station.xml"
    sim_text = sim_path.read_text()
    ctrl_match = re.search(r"GRIPPER_CTRL_SCALE\s*=\s*([0-9.]+)", sim_text)
    qpos_match = re.search(r"GRIPPER_QPOS_SCALE\s*=\s*([0-9.]+)", sim_text)
    ctrl_scale = float(ctrl_match.group(1)) if ctrl_match else None
    qpos_scale = float(qpos_match.group(1)) if qpos_match else None

    xml_estimates: dict[str, Any] = {}
    tree = ET.parse(xml_path)
    for joint in tree.findall(".//joint"):
        name = str(joint.attrib.get("name", ""))
        if name not in {"left_left_finger", "right_left_finger"}:
            continue
        raw_range = str(joint.attrib.get("range", "")).split()
        if len(raw_range) != 2:
            continue
        low, high = [float(value) for value in raw_range]
        xml_estimates[name] = {
            "range_m": [low, high],
            "positive_limit_aperture_m": round(2.0 * max(abs(low), abs(high)), 6),
            "full_span_aperture_m": round(2.0 * abs(high - low), 6),
        }

    estimates = {}
    if qpos_scale is not None:
        estimates["sim_qpos_scale_open_aperture_m"] = round(2.0 * qpos_scale, 6)
    if ctrl_scale is not None:
        estimates["sim_ctrl_scale_open_aperture_m"] = round(2.0 * ctrl_scale, 6)
    return {
        "status": "model_only_not_physical_calibration",
        "sim_path": str(sim_path),
        "xml_path": str(xml_path),
        "sim_gripper_ctrl_scale": ctrl_scale,
        "sim_gripper_qpos_scale": qpos_scale,
        "model_open_width_estimates_m": estimates,
        "xml_finger_joint_estimates": xml_estimates,
        "interpretation": (
            "These are simulation/model aperture estimates. They do not prove the repaired "
            "real gripper's physical jaw width or safe apple compression target."
        ),
    }


def _arm_server_snapshot(side: str, port: int) -> dict[str, Any]:
    import portal

    client = portal.Client(f"127.0.0.1:{port}")
    snapshot: dict[str, Any] = {"side": side, "port": port}
    try:
        snapshot["health"] = _json_safe(client.get_health().result())
    except Exception as exc:
        snapshot["health_error"] = f"{type(exc).__name__}: {exc}"
    try:
        snapshot["observations"] = _json_safe(client.get_observations().result())
    except Exception as exc:
        snapshot["observations_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _calibration_env_evidence(plan_width_m: float | None) -> dict[str, Any]:
    target_pos = _optional_float("OPENFORGE_APPLE_CLOSE_TARGET_POS")
    target_width_m = _optional_float("OPENFORGE_APPLE_CLOSE_TARGET_WIDTH_M")
    closed_width_m = _optional_float("OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M")
    open_width_m = _optional_float("OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M")
    closed_pos = _optional_float("OPENFORGE_APPLE_GRIPPER_CLOSED_POS")
    open_pos = _optional_float("OPENFORGE_APPLE_GRIPPER_OPEN_POS")
    compression_m = float(os.environ.get("OPENFORGE_APPLE_CLOSE_WIDTH_COMPRESSION_M", "0.005"))

    evidence: dict[str, Any] = {
        "target_pos": target_pos,
        "target_width_m": target_width_m,
        "closed_width_m": closed_width_m,
        "open_width_m": open_width_m,
        "closed_pos": 0.0 if closed_pos is None else closed_pos,
        "open_pos": 1.0 if open_pos is None else open_pos,
        "min_target_pos": MIN_TARGET_POS,
        "close_width_compression_m": compression_m,
        "plan_width_m": plan_width_m,
    }

    if target_pos is not None:
        evidence["configured"] = bool(target_pos >= MIN_TARGET_POS)
        evidence["mode"] = "explicit_normalized_target"
        if target_pos < MIN_TARGET_POS:
            evidence["error"] = "target_pos_below_min_target"
        return evidence

    if closed_width_m is not None and open_width_m is not None:
        evidence["mode"] = "width_to_normalized_calibration"
        if open_width_m <= closed_width_m:
            evidence["configured"] = False
            evidence["error"] = "open_width_m_must_exceed_closed_width_m"
            return evidence
        if plan_width_m is None and target_width_m is None:
            evidence["configured"] = False
            evidence["error"] = "need_plan_width_or_target_width_m"
            return evidence
        width = target_width_m if target_width_m is not None else float(plan_width_m) - compression_m
        width = min(max(float(width), closed_width_m), open_width_m)
        ratio = (width - closed_width_m) / (open_width_m - closed_width_m)
        pos = float(evidence["closed_pos"]) + ratio * (float(evidence["open_pos"]) - float(evidence["closed_pos"]))
        evidence["computed_target_width_m"] = round(width, 6)
        evidence["computed_target_pos"] = round(max(0.0, min(1.0, pos)), 6)
        evidence["configured"] = bool(evidence["computed_target_pos"] >= MIN_TARGET_POS)
        if not evidence["configured"]:
            evidence["error"] = "computed_target_pos_below_min_target"
        return evidence

    evidence["mode"] = "missing"
    evidence["configured"] = False
    evidence["error"] = "no_explicit_target_or_width_calibration"
    return evidence


def _write_artifacts(payload: dict[str, Any]) -> Path:
    run_dir = current_run_dir()
    out_dir = run_dir / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "apple_gripper_close_probe.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    summary_path = run_dir / "stage_summary.md"
    decision = payload["decision"]
    lines = [
        f"## {time.strftime('%Y-%m-%d %H:%M:%S')} - apple gripper close calibration probe",
        "",
        "- read-only: no arm or gripper command was sent.",
        f"- physical close ready: `{decision['physical_close_ready']}`",
        f"- reason: {decision['reason']}",
        f"- evidence json: `{path.relative_to(run_dir)}`",
        "",
    ]
    with summary_path.open("a") as handle:
        handle.write("\n".join(lines))
    return path


print("[apple_gripper_close_calibration_probe] Read-only probe starting.")
if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
    print("[apple_gripper_close_calibration_probe] OPENFORGE_ALLOW_PHYSICAL_MOTION ignored; this script sends no motion commands.")

plan = _read_plan_width()
plan_width = plan.get("selected_width_m")
plan_width_f = None if plan_width is None else float(plan_width)
calibration = _calibration_env_evidence(plan_width_f)
payload: dict[str, Any] = {
    "schema": "openforge.apple_gripper_close_calibration_probe.v1",
    "safe_read_only": True,
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "plan": plan,
    "calibration_env": calibration,
    "model_width_evidence": _model_width_evidence(),
    "arm_servers": {
        "left": _arm_server_snapshot("left", 11333),
        "right": _arm_server_snapshot("right", 11334),
    },
}

required = (
    "Provide OPENFORGE_APPLE_CLOSE_TARGET_POS as a validated nonzero normalized "
    "target for the repaired gripper/apple, or provide measured "
    "OPENFORGE_APPLE_GRIPPER_CLOSED_WIDTH_M and OPENFORGE_APPLE_GRIPPER_OPEN_WIDTH_M "
    "plus endpoint overrides if normalized closed/open are not 0.0/1.0."
)
if calibration.get("configured"):
    decision = {
        "static_env_close_ready": True,
        "physical_close_ready": True,
        "reason": "calibration_env_configured",
        "required_evidence_if_false": None,
    }
else:
    decision = {
        "static_env_close_ready": False,
        "physical_close_ready": False,
        "reason": calibration.get("error", "calibration_env_not_configured"),
        "required_evidence_if_false": (
            required
            + " Static env absence alone is not final: the apple task script may still "
            "proceed with its online bbox/depth/model-prior width estimate after a "
            "fresh calibrated top/side observation."
        ),
    }
payload["decision"] = decision
artifact_path = _write_artifacts(payload)

TASK_RESULT.update(
    {
        "success": True,
        "reward": 1.0 if decision["physical_close_ready"] else 0.0,
        "details": payload,
        "artifact_path": str(artifact_path),
    }
)
print(f"[apple_gripper_close_calibration_probe] artifact={artifact_path}")
print(f"[apple_gripper_close_calibration_probe] physical_close_ready={decision['physical_close_ready']}")
print(f"[apple_gripper_close_calibration_probe] reason={decision['reason']}")
