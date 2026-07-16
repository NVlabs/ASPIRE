"""Recorded gripper-geometry calibration for YAM top-camera plate work.

Default mode is no-motion: capture current gripper geometry with live SAM3 /
detect_objects_oneshot prompts, write a proposed calibration sweep, and refuse
physical movement.  Physical mode is only for a bounded gripper-geometry
calibration ticket and requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1 plus explicit
ticket/precondition env acknowledgements.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    read_gripper_width,
    write_json,
)


TASK_NAME = "calibrate_gripper_geometry_top_camera"
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
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
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


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _video_paths(run_dir: Path) -> list[str]:
    return [
        str(path)
        for path in (run_dir / name for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"))
        if path.exists() and path.stat().st_size > 0
    ]


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_GRIPPER_CALIBRATION_RECORDING_WAIT_S", 5.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    packet: dict[str, Any] = {"required": True, "passed": bool(videos), "video_paths": videos}
    if not videos:
        packet.update(
            {
                "problem": "no mp4 files found; gripper calibration requires recording/debug evidence",
                "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
                "blocker_before_physical": True,
            }
        )
    return packet


def _save_depth_preview(path: Path, depth_m: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 3.0)
    if not valid.any():
        preview = np.zeros(depth_m.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth_m[valid], [5, 95])
        preview = (255.0 * np.clip((depth_m - lo) / max(1e-6, hi - lo), 0.0, 1.0)).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)
    return str(path)


def _save_portal_rgbd_snapshot(run_dir: Path, stage: str, cameras: list[str]) -> dict[str, Any]:
    """Save raw RGB-D for later metric gripper analysis without making motion calls."""
    packet: dict[str, Any] = {
        "schema": "openforge.gripper_calibration_rgbd_snapshot.v1",
        "stage": stage,
        "enabled": _truthy_env("OPENFORGE_GRIPPER_CALIBRATION_SAVE_RGBD", True),
        "camera_portal_addr": os.environ.get("OPENFORGE_CAMERA_PORTAL_ADDR", "127.0.0.1:8300").strip(),
        "cameras": {},
    }
    if not packet["enabled"]:
        packet["reason"] = "OPENFORGE_GRIPPER_CALIBRATION_SAVE_RGBD is false"
        return packet
    try:
        import portal

        client = portal.Client(str(packet["camera_portal_addr"]))
    except Exception as exc:
        packet["error"] = f"{type(exc).__name__}: {exc}"
        return packet
    out_dir = run_dir / "observations" / "rgbd_snapshots" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        item: dict[str, Any] = {"camera": camera}
        try:
            rgb = np.asarray(client.get_camera_image(camera).result())
            depth = np.asarray(client.get_camera_depth(camera).result(), dtype=np.float32)
            intrinsics = client.get_camera_intrinsics(camera).result()
            try:
                extrinsics = client.get_camera_extrinsics(camera).result()
            except Exception as exc:
                extrinsics = {"error": f"{type(exc).__name__}: {exc}"}
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                rgb_path = out_dir / f"{camera}_rgb.png"
                Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB").save(rgb_path)
                item["rgb_path"] = str(rgb_path)
            if depth.ndim == 2:
                if depth.size and float(np.nanmedian(depth)) > 10.0:
                    depth = depth / 1000.0
                depth_path = out_dir / f"{camera}_depth_m.npy"
                preview_path = out_dir / f"{camera}_depth_preview.png"
                np.save(depth_path, depth)
                valid = np.isfinite(depth) & (depth > 0.05) & (depth < 3.0)
                item.update(
                    {
                        "depth_npy_path": str(depth_path),
                        "depth_preview_path": _save_depth_preview(preview_path, depth),
                        "depth_shape": list(depth.shape),
                        "depth_summary": {
                            "valid_pixels": int(valid.sum()),
                            "total_pixels": int(depth.size),
                            "valid_fraction": round(float(valid.sum()) / float(max(1, depth.size)), 5),
                        },
                    }
                )
                if valid.any():
                    item["depth_summary"].update(
                        {
                            "median_m": round(float(np.median(depth[valid])), 5),
                            "p05_m": round(float(np.percentile(depth[valid], 5)), 5),
                            "p95_m": round(float(np.percentile(depth[valid], 95)), 5),
                        }
                    )
            item["intrinsics"] = json_safe(intrinsics)
            item["extrinsics"] = json_safe(extrinsics)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        packet["cameras"][camera] = item
    packet["path"] = write_json(out_dir / "rgbd_snapshot.json", packet)
    return packet


def _current_process_family_pids() -> set[int]:
    """Return this script process and its ancestors so pgrep ignores itself."""
    pids: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            pid = int(stat[3])
        except Exception:
            break
    return pids


def _pgrep_competing_motion(run_dir: Path | None = None) -> dict[str, Any]:
    try:
        output = subprocess.check_output(["pgrep", "-af", "run_script.py|ffmpeg"], text=True)
    except subprocess.CalledProcessError:
        output = ""
    self_pid = os.getpid()
    process_family_pids = _current_process_family_pids()
    rows: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            pid = -1
        cmd = parts[1] if len(parts) > 1 else ""
        first_token = cmd.split(maxsplit=1)[0] if cmd else ""
        first_name = Path(first_token).name
        actual_ffmpeg = first_name == "ffmpeg" or first_token.endswith("/ffmpeg")
        actual_run_script = "run_script.py" in cmd and (
            first_name.startswith("python") or first_name in {"uv", "uvx"} or "/python" in first_token
        )
        own_recording = run_dir is not None and str(run_dir) in cmd and actual_ffmpeg
        own_invocation = pid in process_family_pids
        item = {"pid": pid, "cmd": cmd}
        rows.append(item)
        if not own_invocation and not own_recording and (actual_ffmpeg or actual_run_script):
            competitors.append(item)
    return {
        "self_pid": self_pid,
        "process_family_pids": sorted(process_family_pids),
        "rows": rows,
        "competitors": competitors,
    }


def _box_center_px(box: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return None
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def _box_size_px(box: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return None
    return [max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _capture_gripper_scene(run_dir: Path, stage: str) -> dict[str, Any]:
    prompts = _csv(
        "OPENFORGE_GRIPPER_CALIBRATION_PROMPTS",
        [
            "blue gripper fingers",
            "gripper fingers",
            "right gripper",
            "left gripper",
            "robot gripper",
            "black gripper",
        ],
    )
    cameras = [camera for camera in _csv("OPENFORGE_GRIPPER_CALIBRATION_CAMERAS", ["top", "right", "left"]) if camera in {"top", "left", "right"}]
    if not cameras:
        cameras = ["top"]
    packet = capture_scene(
        prompts=prompts,
        cameras=cameras,
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=max(1, _env_int("OPENFORGE_GRIPPER_CALIBRATION_DETECT_RETRIES", 2)),
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
    )
    packet["rgbd_snapshot"] = _save_portal_rgbd_snapshot(run_dir, stage, cameras)
    packet["geometry_analysis"] = _analyze_gripper_detections(packet)
    packet["analysis_path"] = write_json(run_dir / "observations" / f"{stage}_geometry_analysis.json", packet["geometry_analysis"])
    return packet


def _analyze_gripper_detections(packet: dict[str, Any]) -> dict[str, Any]:
    min_score = _env_float("OPENFORGE_GRIPPER_CALIBRATION_MIN_SCORE", 0.15)
    detections: list[dict[str, Any]] = []
    for det in packet.get("all_detections") or []:
        item = dict(det)
        score = float(item.get("score") or 0.0)
        if score < min_score:
            continue
        box = item.get("box_2d") or []
        size = _box_size_px(box)
        center = _box_center_px(box)
        if size:
            item["box_size_px"] = _round_list(size, digits=2, limit=2)
            item["long_axis_px"] = round(max(size), 2)
            item["short_axis_px"] = round(min(size), 2)
        if center:
            item["box_center_px"] = _round_list(center, digits=2, limit=2)
        detections.append(item)

    def _key(item: dict[str, Any]) -> tuple[int, float, float]:
        camera = str(item.get("source_camera") or item.get("camera") or "")
        prompt = str(item.get("prompt") or "")
        return (
            0 if camera == "top" else 1,
            0 if "blue gripper" in prompt else 1,
            -float(item.get("score") or 0.0),
        )

    detections = sorted(detections, key=_key)
    robot_state = packet.get("robot_state") or {}
    arms = robot_state.get("arms") if isinstance(robot_state, dict) else {}
    return {
        "schema": "openforge.gripper_geometry_analysis.v1",
        "source": "fresh live gripper prompts; diagnostic only, not a motion target",
        "min_score": min_score,
        "detections": detections,
        "selected_top_detection": next((det for det in detections if det.get("source_camera") == "top"), None),
        "robot_state_summary": {
            side: {
                "ee_pos": _round_list((arm or {}).get("ee_pos") or [], limit=3),
                "ee_rpy": _round_list((arm or {}).get("ee_rpy") or [], limit=3),
                "gripper_pos": (arm or {}).get("gripper_pos"),
            }
            for side, arm in (arms or {}).items()
        },
        "interpretation": [
            "Use top-camera blue-finger masks to estimate visible finger footprint.",
            "Do not infer plate-grasp success from 2D overlap alone; compare 3D gripper z/xy against rim geometry.",
            "Physical calibration should sweep known gripper widths and small wrist rotations in free space.",
        ],
    }


def _parse_pose_env(name: str) -> dict[str, Any] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            pass
    if len(values) != 6:
        return None
    return {"position": values[:3], "rpy": values[3:6], "source": name}


def _current_pose(state: dict[str, Any], side: str) -> dict[str, Any] | None:
    arm = ((state or {}).get("arms") or {}).get(side) or {}
    pos = arm.get("ee_pos") or []
    rpy = arm.get("ee_rpy") or []
    if len(pos) >= 3 and len(rpy) >= 3:
        return {"position": [float(v) for v in pos[:3]], "rpy": [float(v) for v in rpy[:3]], "source": "current_robot_state"}
    return None


def _pose_for_side(state: dict[str, Any], side: str) -> dict[str, Any] | None:
    explicit = _parse_pose_env(f"OPENFORGE_GRIPPER_CALIBRATION_{side.upper()}_POSE")
    if explicit:
        return explicit
    if _truthy_env("OPENFORGE_GRIPPER_CALIBRATION_USE_CURRENT_POSE", True):
        return _current_pose(state, side)
    if _truthy_env("OPENFORGE_GRIPPER_CALIBRATION_ALLOW_DEFAULT_POSES", False):
        y = 0.28 if side == "left" else -0.28
        return {"position": [0.50, y, 0.92], "rpy": [0.0, 90.0, 0.0], "source": "built_in_default_pose"}
    return None


def _rpy_offsets() -> list[list[float]]:
    raw = os.environ.get("OPENFORGE_GRIPPER_CALIBRATION_RPY_OFFSETS_DEG", "").strip()
    if not raw:
        return [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [-8.0, 0.0, 0.0], [0.0, 0.0, 12.0], [0.0, 0.0, -12.0]]
    offsets: list[list[float]] = []
    for chunk in raw.split(";"):
        vals: list[float] = []
        for part in chunk.split(","):
            try:
                vals.append(float(part.strip()))
            except ValueError:
                pass
        if len(vals) == 3:
            offsets.append(vals)
    return offsets or [[0.0, 0.0, 0.0]]


def _add_rpy(base: list[float], offset: list[float]) -> list[float]:
    return [float(base[i]) + float(offset[i]) for i in range(3)]


def _build_calibration_plan(state: dict[str, Any]) -> dict[str, Any]:
    sides = [side for side in _csv("OPENFORGE_GRIPPER_CALIBRATION_SIDES", ["right", "left"]) if side in {"left", "right"}]
    widths = _float_csv("OPENFORGE_GRIPPER_CALIBRATION_WIDTHS", [0.68, 0.54, 0.40])
    offsets = _rpy_offsets()
    steps: list[dict[str, Any]] = []
    missing: list[str] = []
    for side in sides:
        base = _pose_for_side(state, side)
        if base is None:
            missing.append(f"{side} pose; set OPENFORGE_GRIPPER_CALIBRATION_{side.upper()}_POSE or allow current/default pose")
            continue
        for offset_index, offset in enumerate(offsets):
            pose = {
                "position": list(base["position"]),
                "rpy": _add_rpy(list(base["rpy"]), offset),
                "base_pose_source": base.get("source"),
                "rpy_offset_deg": offset,
            }
            for width in widths:
                steps.append(
                    {
                        "label": f"{side}_offset{offset_index}_width{width:.2f}",
                        "side": side,
                        "pose": pose,
                        "gripper_width_normalized": float(width),
                    }
                )
    return {
        "schema": "openforge.gripper_top_camera_calibration_plan.v1",
        "sides": sides,
        "widths": widths,
        "rpy_offsets_deg": offsets,
        "steps": steps,
        "missing": missing,
        "physical_plan_requires_ticket": True,
        "purpose": "record visible blue-finger geometry, TCP-to-fingertip relation, and width-to-gap behavior in free space",
    }


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None, preview_only: bool = False) -> dict[str, Any]:
    kwargs = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_GRIPPER_CALIBRATION_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_GRIPPER_CALIBRATION_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_GRIPPER_CALIBRATION_PLANNING_SPEED", 0.20),
        "ik_error_threshold": _env_float("OPENFORGE_GRIPPER_CALIBRATION_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_GRIPPER_CALIBRATION_IK_ROT_THRESHOLD_DEG", 16.0),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _preview_step(freespace_move: Callable[..., Any], step: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = freespace_move(
            **_move_kwargs(
                str(step["side"]),
                step["pose"],
                gripper=float(step["gripper_width_normalized"]),
                preview_only=True,
            )
        )
        status = raw.get("status") if isinstance(raw, dict) else getattr(raw, "status", None)
        executed = bool(raw.get("executed", False)) if isinstance(raw, dict) else bool(getattr(raw, "executed", False))
        pos_err = raw.get("final_pos_error_m") if isinstance(raw, dict) else getattr(raw, "final_pos_error_m", None)
        rot_err = raw.get("final_rot_error_deg") if isinstance(raw, dict) else getattr(raw, "final_rot_error_deg", None)
        ok = bool(str(status).lower() in {"success", "succeeded", "ok"} and not executed)
        return {
            "ok": ok,
            "status": status,
            "executed": executed,
            "final_pos_error_m": pos_err,
            "final_rot_error_deg": rot_err,
            "trajectory_cache_key": raw.get("trajectory_cache_key") if isinstance(raw, dict) else getattr(raw, "trajectory_cache_key", None),
            "raw": json_safe(raw),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "executed": False}


def _preview_plan(run_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _tool("freespace_move")
    if not callable(freespace_move):
        return {"success": False, "problem": "freespace_move unavailable", "steps": []}
    max_steps = max(1, _env_int("OPENFORGE_GRIPPER_CALIBRATION_PREVIEW_MAX_STEPS", 30))
    previews: list[dict[str, Any]] = []
    success = not plan.get("missing")
    for step in list(plan.get("steps") or [])[:max_steps]:
        preview = _preview_step(freespace_move, step)
        item = dict(step)
        item["preview"] = preview
        item["ok"] = bool(preview.get("ok"))
        previews.append(item)
        if not item["ok"]:
            success = False
            break
    packet = {
        "schema": "openforge.gripper_top_camera_calibration_preview.v1",
        "success": success,
        "previewed_step_count": len(previews),
        "total_step_count": len(plan.get("steps") or []),
        "steps": previews,
        "missing": plan.get("missing") or [],
    }
    packet["path"] = write_json(run_dir / "plans" / "calibration_preview.json", packet)
    return packet


def _ticket_status(run_dir: Path) -> dict[str, Any]:
    required_truthy = [
        "OPENFORGE_GRIPPER_CALIBRATION_TICKET",
        "OPENFORGE_GRIPPER_CALIBRATION_OPERATOR_PRESENT",
        "OPENFORGE_GRIPPER_CALIBRATION_ESTOP_REACHABLE",
        "OPENFORGE_GRIPPER_CALIBRATION_WORKSPACE_CLEAR",
        "OPENFORGE_GRIPPER_CALIBRATION_HANDS_TOOLS_CABLES_CLEAR",
        "OPENFORGE_GRIPPER_CALIBRATION_NO_LIQUID",
        "OPENFORGE_GRIPPER_CALIBRATION_NO_OTHER_MOTION_PRECHECK",
        "OPENFORGE_GRIPPER_CALIBRATION_FOLLOWER_SERVERS_UP",
        "OPENFORGE_GRIPPER_CALIBRATION_RECORDING_DEBUG_CONFIRMED",
    ]
    missing = [name for name in required_truthy if not _truthy_env(name)]
    scope = os.environ.get("OPENFORGE_GRIPPER_CALIBRATION_SCOPE", "").strip()
    if scope != "gripper_geometry_top_camera":
        missing.append("OPENFORGE_GRIPPER_CALIBRATION_SCOPE=gripper_geometry_top_camera")
    budget = _env_int("OPENFORGE_GRIPPER_CALIBRATION_ATTEMPT_BUDGET", 0)
    if budget != 1:
        missing.append("OPENFORGE_GRIPPER_CALIBRATION_ATTEMPT_BUDGET=1")
    pgrep = _pgrep_competing_motion(run_dir)
    if pgrep["competitors"]:
        missing.append("no competing run_script.py/ffmpeg process")
    return {
        "passed": not missing,
        "missing_or_invalid": missing,
        "required_truthy_env": required_truthy,
        "scope": scope,
        "attempt_budget": budget,
        "pgrep": pgrep,
    }


def _execute_calibration(run_dir: Path, plan: dict[str, Any], preview: dict[str, Any]) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    set_gripper = _required_tool("set_gripper")
    get_robot_state = _tool("get_robot_state")
    result: dict[str, Any] = {
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    preview_by_label = {step.get("label"): step for step in preview.get("steps") or []}
    for step in plan.get("steps") or []:
        label = step["label"]
        preview_step = preview_by_label.get(label)
        if not preview_step or not preview_step.get("ok"):
            result["why_stopped"] = f"missing or failed preview for {label}"
            return result
        side = str(step["side"])
        width = float(step["gripper_width_normalized"])
        set_result = set_gripper(
            side,
            width,
            vel_limit=_env_float("OPENFORGE_GRIPPER_CALIBRATION_GRIPPER_VEL_LIMIT", 0.25),
            torque_limit=_env_float("OPENFORGE_GRIPPER_CALIBRATION_GRIPPER_TORQUE_LIMIT", 0.12),
        )
        time.sleep(_env_float("OPENFORGE_GRIPPER_CALIBRATION_GRIPPER_SETTLE_S", 0.25))
        after_width = read_gripper_width(get_robot_state, side) if callable(get_robot_state) else None
        result["physical_motion_executed"] = True
        result["movement_capable_calls"].append(f"set_gripper:{side}")
        raw_preview = freespace_move(
            **_move_kwargs(
                side,
                step["pose"],
                gripper=width,
                preview_only=True,
            )
        )
        cache_key = raw_preview.get("trajectory_cache_key") if isinstance(raw_preview, dict) else getattr(raw_preview, "trajectory_cache_key", None)
        if not cache_key:
            result["why_stopped"] = f"execution preview for {label} did not return a trajectory cache key"
            return result
        executed = freespace_move(trajectory_cache_key=cache_key)
        result["movement_capable_calls"].append(f"freespace_move:{label}")
        capture = _capture_gripper_scene(run_dir, f"after_{label}")
        result["steps"].append(
            {
                "label": label,
                "side": side,
                "target_width": width,
                "gripper_after_set": after_width,
                "set_gripper_result": json_safe(set_result),
                "move_result": json_safe(executed),
                "capture_packet": capture.get("packet_path"),
                "capture_analysis": capture.get("analysis_path"),
            }
        )
    if _truthy_env("OPENFORGE_GRIPPER_CALIBRATION_GO_HOME_AFTER", False):
        if not _truthy_env("OPENFORGE_GRIPPER_CALIBRATION_ALLOW_GO_HOME", False):
            result["why_stopped"] = "GO_HOME_AFTER requested but ALLOW_GO_HOME not set"
            return result
        go_home = _required_tool("go_home")
        result["movement_capable_calls"].append("go_home")
        result["go_home_result"] = json_safe(go_home())
    result["success"] = True
    result["why_stopped"] = "physical gripper geometry calibration completed"
    return result


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema": "openforge.gripper_top_camera_calibration.result.v1",
        "success": False,
        "run_dir": str(run_dir),
        "physical_gate": _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"),
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "pre_observation": None,
        "calibration_plan": None,
        "preview": None,
        "ticket_status": None,
        "physical_execution": None,
        "recording_evidence": None,
        "video_paths": [],
        "why_stopped": "not_started",
        "next_recommendation": None,
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        observation = _capture_gripper_scene(run_dir, "pre_observe_gripper_geometry")
        result["pre_observation"] = observation
        state = observation.get("robot_state") or {}
        plan = _build_calibration_plan(state)
        result["calibration_plan"] = plan
        write_json(run_dir / "plans" / "calibration_plan.json", plan)
        preview = _preview_plan(run_dir, plan)
        result["preview"] = preview
        append_stage_summary(
            run_dir,
            [
                "## gripper calibration no-motion analysis",
                f"- observation: {observation.get('packet_path')}",
                f"- geometry_analysis: {observation.get('analysis_path')}",
                f"- plan_steps: {len(plan.get('steps') or [])}",
                f"- preview_success: {preview.get('success')}",
                f"- preview_path: {preview.get('path')}",
            ],
        )
        if not result["physical_gate"]:
            result["success"] = bool(observation and not plan.get("missing"))
            result["why_stopped"] = "recorded no-motion gripper geometry calibration validation; physical gate not set"
            result["next_recommendation"] = (
                "For physical calibration, provide a bounded gripper-geometry ticket and rerun with "
                "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 plus OPENFORGE_GRIPPER_CALIBRATION_* acknowledgements."
            )
            return result
        recording = _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        if not recording.get("passed"):
            result["why_stopped"] = "recording evidence missing; refusing physical calibration"
            result["next_recommendation"] = "Fix recording/debug UI before physical gripper calibration."
            return result
        ticket = _ticket_status(run_dir)
        result["ticket_status"] = ticket
        append_stage_summary(
            run_dir,
            [
                "## physical calibration ticket gate",
                f"- passed: {ticket['passed']}",
                f"- missing_or_invalid: {ticket['missing_or_invalid']}",
            ],
        )
        if not ticket.get("passed"):
            result["why_stopped"] = "physical calibration ticket/preconditions incomplete; refusing motion"
            result["next_recommendation"] = "Provide complete OPENFORGE_GRIPPER_CALIBRATION_* ticket env vars."
            return result
        if not preview.get("success"):
            result["why_stopped"] = "calibration sweep preview failed; refusing physical motion"
            result["next_recommendation"] = "Inspect plans/calibration_preview.json and adjust calibration poses/offsets."
            return result
        execution = _execute_calibration(run_dir, plan, preview)
        result["physical_execution"] = execution
        result["physical_motion_executed"] = bool(execution.get("physical_motion_executed"))
        result["movement_capable_calls"] = list(execution.get("movement_capable_calls") or [])
        result["success"] = bool(execution.get("success"))
        result["why_stopped"] = execution.get("why_stopped")
        result["next_recommendation"] = "Inspect all after_* observation analyses and derive fingertip/TCP offsets for plate pickup."
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Fix the calibration script/no-motion blocker before any physical calibration."
        return result
    finally:
        recording = result.get("recording_evidence") or _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- recording_evidence: {result['recording_evidence']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )
        print(f"[{TASK_NAME}] success={result['success']} why={result['why_stopped']}")


TASK_RESULT.update(_run())
