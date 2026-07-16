"""One-command live bottle grasp/lift/place-to-rack loop for YAM.

This script is intentionally independent from the existing bottle lift script:
it repeats the task-specific observation, side-grasp, rack-place, release, and
retreat logic here while reusing only the compact yam_runtime helper layer.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.full_demo_speed import scaled_speed
from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    compare_detection_motion,
    current_run_dir,
    generate_side_grasp_candidates,
    json_safe,
    write_json,
)


TASK_NAME = "grasp_lift_place_bottle_rack_one_shot_loop"
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


def _optional_env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _pickup_compat_name(name: str) -> str | None:
    prefix = "OPENFORGE_BOTTLE_RACK_"
    if not name.startswith(prefix):
        return None
    return "OPENFORGE_GRASP_LIFT_" + name[len(prefix) :]


def _env_raw_pickup(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return raw
    compat = _pickup_compat_name(name)
    if compat is None:
        return None
    raw = os.environ.get(compat)
    if raw is not None and raw.strip():
        return raw
    return None


def _env_str_pickup(name: str, default: str) -> str:
    raw = _env_raw_pickup(name)
    return raw.strip() if raw is not None else default


def _truthy_env_pickup(name: str, default: bool = False) -> bool:
    raw = _env_raw_pickup(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_pickup(name: str, default: float) -> float:
    raw = _env_raw_pickup(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_env_float_pickup(name: str) -> float | None:
    raw = _env_raw_pickup(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        compat = _pickup_compat_name(name)
        names = f"{name}/{compat}" if compat else name
        raise ValueError(f"{names} must be a float, got {raw!r}") from exc


def _env_int_pickup(name: str, default: int) -> int:
    raw = _env_raw_pickup(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    return values or list(default)


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            pass
    return values or list(default)


def _float_csv_pickup(name: str, default: list[float]) -> list[float]:
    raw = _env_raw_pickup(name)
    if raw is None:
        return list(default)
    values: list[float] = []
    for item in raw.replace(";", ",").split(","):
        try:
            values.append(float(item.strip()))
        except ValueError:
            pass
    return values or list(default)


def _offset_triples(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [list(item[:3]) for item in default]
    triples: list[list[float]] = []
    for chunk in raw.replace("|", ";").split(";"):
        parts = [part.strip() for part in chunk.replace(":", ",").split(",") if part.strip()]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) >= 3 else 0.0
        except ValueError:
            continue
        triples.append([x, y, z])
    return triples or [list(item[:3]) for item in default]


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _make_open_gripper() -> Callable[[str], Any]:
    set_gripper = _tool("set_gripper")
    open_gripper = _tool("open_gripper")
    open_target = _env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_TARGET_POS", 1.0)
    open_vel = _env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_VEL_LIMIT", 0.50)
    open_torque = _env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_TORQUE_LIMIT", 0.30)

    def _open(side: str) -> Any:
        if callable(set_gripper):
            return set_gripper(side, open_target, vel_limit=open_vel, torque_limit=open_torque)
        if callable(open_gripper):
            return open_gripper(side)
        raise RuntimeError("no open gripper API is available")

    return _open


def _extract_gripper_pos(state: Any, side: str) -> float | None:
    safe = json_safe(state)
    if isinstance(safe, dict):
        arms = safe.get("arms")
        if isinstance(arms, dict):
            arm_state = arms.get(side)
            if isinstance(arm_state, dict):
                value = arm_state.get("gripper_pos")
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
        value = safe.get(f"{side}_gripper_pos")
        if value is None:
            value = safe.get(f"{side}_grip")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    arms_obj = getattr(state, "arms", None)
    if isinstance(arms_obj, dict):
        arm_obj = arms_obj.get(side)
        value = getattr(arm_obj, "gripper_pos", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _open_gripper_until(
    side: str,
    *,
    min_pos: float,
    timeout_s: float,
    poll_s: float,
    retry_s: float,
) -> dict[str, Any]:
    open_fn = _make_open_gripper()
    get_robot_state = _tool("get_robot_state")
    packet: dict[str, Any] = {
        "success": False,
        "side": side,
        "target_min_pos": float(min_pos),
        "timeout_s": float(timeout_s),
        "poll_s": float(poll_s),
        "retry_s": float(retry_s),
        "commands": [],
        "samples": [],
        "final_gripper_pos": None,
        "final_state": None,
        "why_stopped": None,
    }

    deadline = time.monotonic() + max(0.1, timeout_s)
    next_command_time = 0.0
    while time.monotonic() <= deadline:
        now = time.monotonic()
        if now >= next_command_time:
            packet["commands"].append(json_safe(open_fn(side)))
            next_command_time = now + max(0.1, retry_s)
        if callable(get_robot_state):
            state = get_robot_state()
            pos = _extract_gripper_pos(state, side)
            packet["final_state"] = json_safe(state)
            packet["final_gripper_pos"] = pos
            packet["samples"].append(
                {
                    "t_remaining_s": round(max(0.0, deadline - time.monotonic()), 3),
                    "gripper_pos": pos,
                }
            )
            if pos is not None and pos >= min_pos:
                packet["success"] = True
                packet["why_stopped"] = "verified open from robot gripper encoder"
                return packet
        time.sleep(max(0.02, poll_s))

    packet["why_stopped"] = (
        f"timed out waiting for {side} gripper to open to >= {min_pos:.3f}; "
        f"final gripper_pos={packet['final_gripper_pos']}"
    )
    return packet


def _xyz(detection: dict[str, Any] | None) -> list[float] | None:
    if not detection:
        return None
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        return None
    try:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except (TypeError, ValueError):
        return None


def _round_list(values: Any, digits: int = 5, limit: int = 3) -> list[float]:
    return [round(float(v), digits) for v in list(values)[:limit]]


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists():
            paths.append(str(path))
    return paths


def _capture_prompt_fallback(
    *,
    run_dir: Path,
    stage: str,
    prompts: list[str],
    cameras: list[str],
    min_score: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    packets: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for prompt_index, prompt in enumerate(prompts):
        packet = capture_scene(
            prompts=[prompt],
            cameras=cameras,
            detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
            get_camera_image=_tool("get_camera_image"),
            get_robot_state=_tool("get_robot_state"),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage=f"{stage}_{prompt.replace(' ', '_')}",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
            max_retries=max(1, _env_int_pickup("OPENFORGE_BOTTLE_RACK_DETECT_RETRIES", 2)),
            save_overlays=True,
        )
        packets.append(packet)
        detection = packet.get("best_detection")
        score = float((detection or {}).get("score") or 0.0)
        if detection is not None and score >= min_score:
            best = dict(detection)
            best["prompt_index"] = prompt_index
            best["prompt_fallback_used"] = bool(prompt_index > 0)
            raw_source = best.get("source")
            live_source = "live_top_camera" if best.get("source_camera") == "top" else "live_perception"
            best["raw_detection_source"] = raw_source
            best["source"] = live_source
            best["detection_source"] = live_source
            break
    combined = {
        "stage": stage,
        "prompts_tried": prompts[: len(packets)],
        "packets": [
            {
                "packet_path": packet.get("packet_path"),
                "best_detection": packet.get("best_detection"),
                "errors": packet.get("errors"),
            }
            for packet in packets
        ],
        "best_detection": best,
        "source": "live_perception",
    }
    combined["path"] = write_json(run_dir / "observations" / f"{stage}_prompt_fallback.json", combined)
    return combined, best


def _detection_touches_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin = _env_int("OPENFORGE_BOTTLE_RACK_IMAGE_EDGE_MARGIN_PX", 6)
    width = _env_int("OPENFORGE_BOTTLE_RACK_IMAGE_WIDTH_PX", 640)
    height = _env_int("OPENFORGE_BOTTLE_RACK_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def _select_rack_detection(packet: dict[str, Any]) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    min_score = _env_float("OPENFORGE_BOTTLE_RACK_MIN_RACK_SCORE", 0.20)
    prompt_order = {
        "wooden rack": 0,
        "rack": 1,
        "wooden shelf": 2,
        "shelf": 3,
        "box": 4,
    }
    filtered: list[dict[str, Any]] = []
    for det in detections:
        xyz = _xyz(det)
        camera = str(det.get("source_camera") or det.get("camera") or "")
        score = float(det.get("score") or 0.0)
        if xyz is None or camera != "top" or score < min_score:
            continue
        det["touches_image_edge"] = _detection_touches_edge(det)
        if det["touches_image_edge"] and _truthy_env("OPENFORGE_BOTTLE_RACK_REJECT_EDGE_RACK_DETECTIONS", True):
            continue
        filtered.append(det)
    if not filtered:
        return None

    target_y_sign = os.environ.get("OPENFORGE_BOTTLE_RACK_TARGET_Y_SIGN", "positive").strip().lower()

    def _key(det: dict[str, Any]) -> tuple[int, int, float]:
        prompt = str(det.get("prompt") or "").lower()
        xyz = _xyz(det) or [0.0, 0.0, 0.0]
        y_rank = 0
        if target_y_sign in {"positive", "pos", "left", "left_side"}:
            y_rank = 0 if xyz[1] > 0.0 else 1
        elif target_y_sign in {"negative", "neg", "right", "right_side"}:
            y_rank = 0 if xyz[1] < 0.0 else 1
        return (y_rank, prompt_order.get(prompt, 9), -float(det.get("score") or 0.0))

    selected = sorted(filtered, key=_key)[0]
    selected["detection_source"] = "live_top_camera"
    return selected


def _observe_rack(run_dir: Path) -> dict[str, Any]:
    prompts = _csv("OPENFORGE_BOTTLE_RACK_TARGET_PROMPTS", ["wooden rack", "rack", "wooden shelf", "shelf"])
    packet = capture_scene(
        prompts=prompts,
        cameras=["top"],
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="observe_rack_target",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 20.0),
        max_retries=max(1, _env_int("OPENFORGE_BOTTLE_RACK_DETECT_RETRIES", 2)),
        save_overlays=True,
        motion_cameras=("top",),
    )
    selected = _select_rack_detection(packet)
    packet["selected_rack_detection"] = selected
    packet["path"] = write_json(run_dir / "observations" / "observe_rack_target_selected.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## rack detection",
            f"- selected_rack_detection: {selected}",
            f"- packet: {packet.get('packet_path')}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _apply_bottle_pregrasp_clearance(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clearance = _optional_env_float_pickup("OPENFORGE_BOTTLE_RACK_SIDE_PREGRASP_CLEARANCE_M")
    xy_bias = _float_csv_pickup("OPENFORGE_BOTTLE_RACK_SIDE_PREGRASP_XY_BIAS_M", [0.0, 0.0])
    bias_x = float(xy_bias[0]) if len(xy_bias) >= 1 else 0.0
    bias_y = float(xy_bias[1]) if len(xy_bias) >= 2 else 0.0
    if clearance is None:
        return candidates
    if clearance < 0.0:
        raise ValueError("OPENFORGE_BOTTLE_RACK_SIDE_PREGRASP_CLEARANCE_M must be nonnegative")

    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        cand = dict(candidate)
        object_center = cand.get("object_center")
        approach = cand.get("approach_direction_world")
        grasp_pose = dict(cand.get("grasp_pose") or {})
        grasp_pos = grasp_pose.get("position")
        pre_pose = dict(cand.get("pregrasp_pose") or {})
        if not object_center or not approach or not grasp_pos:
            cand["pregrasp_policy"] = "unchanged_missing_geometry_for_clearance"
            adjusted.append(cand)
            continue

        center = [float(value) for value in list(object_center)[:3]]
        approach_vec = [float(value) for value in list(approach)[:3]]
        grasp = [float(value) for value in list(grasp_pos)[:3]]
        pre = [
            center[0] - approach_vec[0] * clearance + bias_x,
            center[1] - approach_vec[1] * clearance + bias_y,
            grasp[2],
        ]
        old_pre_pose = cand.get("pregrasp_pose")
        if old_pre_pose is not None and "original_pregrasp_pose" not in cand:
            cand["original_pregrasp_pose"] = json_safe(old_pre_pose)
        pre_pose["position"] = pre
        cand["pregrasp_pose"] = pre_pose
        cand["pregrasp_policy"] = "object_center_clearance_decoupled_from_grasp_backoff"
        cand["pregrasp_clearance_m"] = float(clearance)
        cand["pregrasp_xy_bias_m"] = [bias_x, bias_y]
        cand["pregrasp_to_grasp_distance_m"] = float(math.dist(pre, grasp))
        cand["pregrasp_to_grasp_vector_m"] = [grasp[i] - pre[i] for i in range(3)]
        adjusted.append(cand)
    return adjusted


def _apply_fixed_bottle_grasp_z(
    candidates: list[dict[str, Any]],
    *,
    fixed_grasp_z_m: float,
    lift_z_m: float,
    env_name: str,
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        cand = dict(candidate)
        cand["fixed_grasp_z_env"] = env_name
        cand["fixed_grasp_z_m"] = float(fixed_grasp_z_m)
        cand["original_body_z_source"] = cand.get("body_z_source")
        cand["original_z_offset_m"] = cand.get("z_offset_m")

        for key in ("position", "object_center"):
            values = cand.get(key)
            if values and len(values) >= 3:
                updated = [float(value) for value in list(values)[:3]]
                cand[f"original_{key}"] = json_safe(values)
                updated[2] = float(fixed_grasp_z_m)
                cand[key] = _round_list(updated)

        for pose_key in ("pregrasp_pose", "grasp_pose"):
            pose = dict(cand.get(pose_key) or {})
            values = pose.get("position")
            if values and len(values) >= 3:
                updated = [float(value) for value in list(values)[:3]]
                pose["original_position"] = json_safe(values)
                updated[2] = float(fixed_grasp_z_m)
                pose["position"] = updated
                cand[pose_key] = pose

        lift_pose = dict(cand.get("lift_pose") or {})
        values = lift_pose.get("position")
        if values and len(values) >= 3:
            updated = [float(value) for value in list(values)[:3]]
            lift_pose["original_position"] = json_safe(values)
            updated[2] = float(fixed_grasp_z_m) + float(lift_z_m)
            lift_pose["position"] = updated
            cand["lift_pose"] = lift_pose

        cand["body_z_source"] = "fixed_grasp_z_m"
        cand["z_offset_m"] = 0.0
        adjusted.append(cand)
    return adjusted


def _build_bottle_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    requested_arm = _env_str_pickup("OPENFORGE_BOTTLE_RACK_ARM", "left").strip().lower()
    arm = requested_arm if requested_arm in {"left", "right"} else None
    lift_z_m = _env_float_pickup("OPENFORGE_BOTTLE_RACK_LIFT_Z_M", 0.30)
    candidates = generate_side_grasp_candidates(
        detection,
        object_kind="bottle",
        arm=arm,
        default_radius_m=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OBJECT_RADIUS_M", 0.04),
        default_half_height_m=0.0,
        body_fraction=_env_float_pickup("OPENFORGE_BOTTLE_RACK_BODY_GRASP_FRACTION", 0.45),
        pregrasp_standoff_m=_env_float_pickup("OPENFORGE_BOTTLE_RACK_SIDE_PREGRASP_STANDOFF_M", 0.08),
        lift_z_m=lift_z_m,
        width_margin_m=_env_float_pickup("OPENFORGE_BOTTLE_RACK_GRIPPER_WIDTH_CLEARANCE_M", 0.015),
        include_topdown=False,
        yaw_angles_deg=_float_csv_pickup("OPENFORGE_BOTTLE_RACK_SIDE_YAWS_DEG", [0.0, 90.0, 70.0, 110.0, 180.0]),
        z_offsets_m=_float_csv_pickup("OPENFORGE_BOTTLE_RACK_SIDE_Z_OFFSETS_M", [0.0, 0.015, -0.015, 0.03]),
        center_z_offset_without_extents_m=_env_float_pickup(
            "OPENFORGE_BOTTLE_RACK_LIVE_DETECTION_Z_TO_BODY_OFFSET_M",
            -0.08,
        ),
        grasp_backoff_m=_env_float_pickup("OPENFORGE_BOTTLE_RACK_SIDE_GRASP_BACKOFF_M", -0.02),
    )
    fixed_grasp_z = _optional_env_float_pickup("OPENFORGE_BOTTLE_RACK_FIXED_GRASP_Z_M")
    if fixed_grasp_z is not None:
        candidates = _apply_fixed_bottle_grasp_z(
            candidates,
            fixed_grasp_z_m=fixed_grasp_z,
            lift_z_m=lift_z_m,
            env_name="OPENFORGE_BOTTLE_RACK_FIXED_GRASP_Z_M",
        )
    return _apply_bottle_pregrasp_clearance(candidates)


def _bottle_tcp_above_support(selected_pickup: dict[str, Any]) -> float:
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    grasp_pos = grasp_pose.get("position") or []
    if len(grasp_pos) < 3:
        raise RuntimeError("selected pickup grasp pose has no position")
    source_support_z = _env_float("OPENFORGE_BOTTLE_RACK_SOURCE_SUPPORT_Z_M", 0.760)
    value = float(grasp_pos[2]) - source_support_z
    fallback = _env_float("OPENFORGE_BOTTLE_RACK_DEFAULT_TCP_ABOVE_SUPPORT_M", 0.080)
    min_value = _env_float("OPENFORGE_BOTTLE_RACK_MIN_TCP_ABOVE_SUPPORT_M", 0.035)
    max_value = _env_float("OPENFORGE_BOTTLE_RACK_MAX_TCP_ABOVE_SUPPORT_M", 0.180)
    if value < min_value or value > max_value:
        return fallback
    return value


def _compute_transport_retreat_x(base_x: float, rack_x: float) -> dict[str, Any]:
    fixed_raw = os.environ.get("OPENFORGE_BOTTLE_RACK_TRANSPORT_RETREAT_X_M")
    fallback = _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_RETREAT_X_FALLBACK_M", -0.20)
    if fixed_raw is not None and fixed_raw.strip():
        retreat_x = _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_RETREAT_X_M", fallback)
        return {
            "source": "fixed_env",
            "lift_x_m": round(float(base_x), 5),
            "rack_x_m": round(float(rack_x), 5),
            "retreat_x_m": round(float(retreat_x), 5),
        }

    clearance = _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_RACK_X_CLEARANCE_M", 0.25)
    min_retreat = abs(_env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_MIN_RETREAT_X_M", 0.08))
    max_retreat = abs(_env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_MAX_RETREAT_X_M", 0.30))
    if max_retreat < min_retreat:
        min_retreat, max_retreat = max_retreat, min_retreat

    safe_x = float(rack_x) - float(clearance)
    required_retreat_x = safe_x - float(base_x)
    already_clear = required_retreat_x >= 0.0
    if already_clear:
        retreat_x = 0.0
    else:
        required_mag = abs(required_retreat_x)
        if required_mag > max_retreat:
            raise RuntimeError(
                "computed bottle-rack transport retreat exceeds max: "
                f"need {required_mag:.3f}m, max {max_retreat:.3f}m "
                f"(lift_x={float(base_x):.3f}, rack_x={float(rack_x):.3f}, clearance={clearance:.3f})"
            )
        retreat_x = -max(required_mag, min_retreat)

    return {
        "source": "rack_x_clearance",
        "lift_x_m": round(float(base_x), 5),
        "rack_x_m": round(float(rack_x), 5),
        "rack_clearance_m": round(float(clearance), 5),
        "safe_x_m": round(float(safe_x), 5),
        "required_retreat_x_m": round(float(required_retreat_x), 5),
        "min_retreat_x_m": round(float(min_retreat), 5),
        "max_retreat_x_m": round(float(max_retreat), 5),
        "already_clear": bool(already_clear),
        "retreat_x_m": round(float(retreat_x), 5),
    }


def _make_transport_waypoints(selected_pickup: dict[str, Any], rack_detection: dict[str, Any]) -> list[dict[str, Any]]:
    if not _truthy_env("OPENFORGE_BOTTLE_RACK_ENABLE_TRANSPORT_WAYPOINTS", True):
        return []
    base_pose = selected_pickup.get("lift_pose") or selected_pickup.get("grasp_pose") or {}
    base_pos = [float(value) for value in list(base_pose.get("position") or [])[:3]]
    base_rpy = [float(value) for value in list(base_pose.get("rpy") or selected_pickup.get("rpy") or [])[:3]]
    if len(base_pos) < 3 or len(base_rpy) < 3:
        return []
    rack_xyz = _xyz(rack_detection)
    if rack_xyz is None:
        raise RuntimeError("cannot calculate bottle-rack transport retreat without rack xyz")
    retreat_calc = _compute_transport_retreat_x(base_pos[0], rack_xyz[0])
    retreat_x = float(retreat_calc["retreat_x_m"])
    shift_y = _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_SHIFT_Y_M", -0.28)
    lift_z = _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_LIFT_Z_M", 0.10)
    x_pose = [base_pos[0] + retreat_x, base_pos[1], base_pos[2]]
    y_pose = [x_pose[0], x_pose[1] + shift_y, x_pose[2]]
    z_pose = [y_pose[0], y_pose[1], y_pose[2] + lift_z]
    waypoints: list[dict[str, Any]] = []
    if abs(retreat_x) > 1e-6:
        waypoints.append(
            {
                "label": "transport_retreat_x",
                "position": x_pose,
                "rpy": list(base_rpy),
                "relative_move_m": [retreat_x, 0.0, 0.0],
                "calculation": retreat_calc,
            }
        )
    else:
        x_pose = list(base_pos)
    waypoints.extend(
        [
            {
                "label": "transport_shift_y",
                "position": y_pose,
                "rpy": list(base_rpy),
                "relative_move_m": [0.0, shift_y, 0.0],
            },
            {
                "label": "transport_lift_z",
                "position": z_pose,
                "rpy": list(base_rpy),
                "relative_move_m": [0.0, 0.0, lift_z],
            },
        ]
    )
    for waypoint in waypoints:
        waypoint["transport_retreat_x_calculation"] = retreat_calc
    return waypoints


def _clamp_place_xy(place_xy: list[float]) -> tuple[list[float], dict[str, Any] | None]:
    clamped = [float(place_xy[0]), float(place_xy[1])]
    bounds = {
        "min_x": _optional_env_float("OPENFORGE_BOTTLE_RACK_PLACE_TARGET_MIN_X_M"),
        "max_x": _optional_env_float("OPENFORGE_BOTTLE_RACK_PLACE_TARGET_MAX_X_M"),
        "min_y": _optional_env_float("OPENFORGE_BOTTLE_RACK_PLACE_TARGET_MIN_Y_M"),
        "max_y": _optional_env_float("OPENFORGE_BOTTLE_RACK_PLACE_TARGET_MAX_Y_M"),
    }
    original = list(clamped)
    if bounds["min_x"] is not None:
        clamped[0] = max(clamped[0], float(bounds["min_x"]))
    if bounds["max_x"] is not None:
        clamped[0] = min(clamped[0], float(bounds["max_x"]))
    if bounds["min_y"] is not None:
        clamped[1] = max(clamped[1], float(bounds["min_y"]))
    if bounds["max_y"] is not None:
        clamped[1] = min(clamped[1], float(bounds["max_y"]))
    if clamped == original:
        return clamped, None
    return clamped, {
        "source": "env_bounds",
        "original_xy_m": _round_list(original),
        "clamped_xy_m": _round_list(clamped),
        "bounds_m": {key: value for key, value in bounds.items() if value is not None},
    }


def _compute_release_retreat_standoff(
    release_pos: list[float],
    approach: list[float],
    rack_xyz: list[float],
) -> dict[str, Any]:
    fixed_raw = os.environ.get("OPENFORGE_BOTTLE_RACK_RETREAT_STANDOFF_M")
    fallback = _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_STANDOFF_FALLBACK_M", 0.10)
    if fixed_raw is not None and fixed_raw.strip():
        standoff = _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_STANDOFF_M", fallback)
        return {
            "source": "fixed_env",
            "release_x_m": round(float(release_pos[0]), 5),
            "rack_x_m": round(float(rack_xyz[0]), 5),
            "approach_x": round(float(approach[0]) if approach else 0.0, 5),
            "retreat_standoff_m": round(float(standoff), 5),
        }

    clearance = _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_RACK_X_CLEARANCE_M", 0.10)
    min_standoff = abs(_env_float("OPENFORGE_BOTTLE_RACK_RETREAT_MIN_STANDOFF_M", 0.10))
    max_standoff = abs(_env_float("OPENFORGE_BOTTLE_RACK_RETREAT_MAX_STANDOFF_M", 0.25))
    if max_standoff < min_standoff:
        min_standoff, max_standoff = max_standoff, min_standoff

    approach_x = float(approach[0]) if approach else 0.0
    if approach_x <= 1e-6:
        standoff = fallback
        source = "fallback_nonpositive_x_approach"
        safe_x = float(rack_xyz[0]) - float(clearance)
        required = None
    else:
        source = "rack_x_clearance"
        safe_x = float(rack_xyz[0]) - float(clearance)
        required = max(0.0, (float(release_pos[0]) - safe_x) / approach_x)
        standoff = max(required, min_standoff)

    if standoff > max_standoff:
        raise RuntimeError(
            "computed rack retreat exceeds max: "
            f"need {standoff:.3f}m, max {max_standoff:.3f}m "
            f"(release_x={float(release_pos[0]):.3f}, rack_x={float(rack_xyz[0]):.3f}, clearance={clearance:.3f})"
        )

    return {
        "source": source,
        "release_x_m": round(float(release_pos[0]), 5),
        "rack_x_m": round(float(rack_xyz[0]), 5),
        "approach_x": round(float(approach_x), 5),
        "rack_clearance_m": round(float(clearance), 5),
        "safe_tcp_x_m": round(float(safe_x), 5),
        "required_standoff_m": None if required is None else round(float(required), 5),
        "min_standoff_m": round(float(min_standoff), 5),
        "max_standoff_m": round(float(max_standoff), 5),
        "retreat_standoff_m": round(float(standoff), 5),
    }


def _make_retreat_waypoints(
    *,
    release_pose: dict[str, Any],
    rack_detection: dict[str, Any],
    approach: list[float],
    retreat_lift: float,
) -> list[dict[str, Any]]:
    release_pos = [float(value) for value in list(release_pose.get("position") or [])[:3]]
    rpy = [float(value) for value in list(release_pose.get("rpy") or [])[:3]]
    if len(release_pos) < 3 or len(rpy) < 3:
        return []
    rack_xyz = _xyz(rack_detection)
    if rack_xyz is None:
        raise RuntimeError("cannot calculate rack retreat without rack xyz")
    retreat_calc = _compute_release_retreat_standoff(release_pos, approach, rack_xyz)
    retreat_standoff = float(retreat_calc["retreat_standoff_m"])

    retreat_back = {
        "label": "rack_retreat_back_x",
        "position": [
            release_pos[0] - float(approach[0]) * retreat_standoff,
            release_pos[1] - float(approach[1]) * retreat_standoff,
            release_pos[2] + retreat_lift,
        ],
        "rpy": list(rpy),
        "relative_move_m": [
            -float(approach[0]) * retreat_standoff,
            -float(approach[1]) * retreat_standoff,
            retreat_lift,
        ],
        "calculation": retreat_calc,
    }
    retreat_shift_y = _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_SHIFT_Y_M", 0.0)
    if abs(retreat_shift_y) <= 1e-9:
        return [retreat_back]
    return [
        retreat_back,
        {
            "label": "rack_retreat_lane_y",
            "position": [
                retreat_back["position"][0],
                retreat_back["position"][1] + retreat_shift_y,
                retreat_back["position"][2],
            ],
            "rpy": list(rpy),
            "relative_move_m": [0.0, retreat_shift_y, 0.0],
            "retreat_calculation": retreat_calc,
        },
    ]


def _make_place_candidates(selected_pickup: dict[str, Any], rack_detection: dict[str, Any]) -> list[dict[str, Any]]:
    rack_xyz = _xyz(rack_detection)
    grasp_pose = selected_pickup.get("grasp_pose") or {}
    rpy = [float(value) for value in list(grasp_pose.get("rpy") or [])[:3]]
    if rack_xyz is None or len(rpy) < 3:
        raise RuntimeError("cannot build rack place candidates without rack xyz and pickup rpy")
    approach = [float(value) for value in list(selected_pickup.get("approach_direction_world") or [0.0, 0.0, 0.0])[:3]]
    if len(approach) < 3:
        approach = [0.0, 0.0, 0.0]
    tcp_above_support = _bottle_tcp_above_support(selected_pickup)
    rack_surface_z = float(rack_xyz[2]) + _env_float("OPENFORGE_BOTTLE_RACK_SURFACE_Z_OFFSET_M", 0.0)
    release_clearance = _env_float("OPENFORGE_BOTTLE_RACK_RELEASE_CLEARANCE_M", 0.015)
    high_lift = _env_float("OPENFORGE_BOTTLE_RACK_HIGH_PLACE_Z_LIFT_M", 0.080)
    retreat_lift = _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_Z_LIFT_M", 0.0)
    offsets = _offset_triples(
        "OPENFORGE_BOTTLE_RACK_PLACE_OFFSETS_M",
        [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]],
    )
    transport_waypoints = _make_transport_waypoints(selected_pickup, rack_detection)
    transport_retreat_calc = (
        transport_waypoints[0].get("transport_retreat_x_calculation")
        if transport_waypoints
        else None
    )
    max_candidates = max(1, _env_int("OPENFORGE_BOTTLE_RACK_MAX_PLACE_CANDIDATES", 8))
    candidates: list[dict[str, Any]] = []
    for index, offset in enumerate(offsets[:max_candidates]):
        unclamped_place_xy = [rack_xyz[0] + float(offset[0]), rack_xyz[1] + float(offset[1])]
        place_xy, place_clamp = _clamp_place_xy(unclamped_place_xy)
        release_z = rack_surface_z + tcp_above_support + release_clearance + float(offset[2])
        release_pose = {"position": [place_xy[0], place_xy[1], release_z], "rpy": rpy}
        high_pose = {"position": [place_xy[0], place_xy[1], release_z + high_lift], "rpy": rpy}
        retreat_waypoints = _make_retreat_waypoints(
            release_pose=release_pose,
            rack_detection=rack_detection,
            approach=approach,
            retreat_lift=retreat_lift,
        )
        retreat_calc = retreat_waypoints[0].get("calculation") if retreat_waypoints else None
        retreat_pose = (
            {"position": retreat_waypoints[-1]["position"], "rpy": retreat_waypoints[-1]["rpy"]}
            if retreat_waypoints
            else {"position": [place_xy[0], place_xy[1], release_z + retreat_lift], "rpy": rpy}
        )
        candidates.append(
            {
                "label": f"bottle_rack_place_{index}",
                "arm": selected_pickup["arm"],
                "strategy_type": f"{selected_pickup.get('strategy_type', 'bottle_side')}_release_on_rack",
                "rack_detection": rack_detection,
                "pickup_strategy_label": selected_pickup.get("label"),
                "rack_surface_reference_m": _round_list([rack_xyz[0], rack_xyz[1], rack_surface_z]),
                "relative_offset_m": _round_list(offset),
                "place_target_clamp": place_clamp,
                "place_target_object_ref_m": _round_list(
                    [place_xy[0], place_xy[1], rack_surface_z + release_clearance + float(offset[2])]
                ),
                "transport_waypoints": transport_waypoints,
                "release_pose": release_pose,
                "high_place_pose": high_pose,
                "retreat_waypoints": retreat_waypoints,
                "retreat_pose": retreat_pose,
                "geometry_model": {
                    "source": "fresh bottle and rack detections",
                    "source_support_z_m": _env_float("OPENFORGE_BOTTLE_RACK_SOURCE_SUPPORT_Z_M", 0.760),
                    "tcp_above_source_support_m": round(tcp_above_support, 5),
                    "rack_surface_z_offset_m": _env_float("OPENFORGE_BOTTLE_RACK_SURFACE_Z_OFFSET_M", 0.0),
                    "release_clearance_m": round(release_clearance, 5),
                    "high_place_z_lift_m": round(high_lift, 5),
                    "retreat_z_lift_m": round(retreat_lift, 5),
                    "retreat_calculation": retreat_calc,
                    "retreat_shift_y_m": round(_env_float("OPENFORGE_BOTTLE_RACK_RETREAT_SHIFT_Y_M", 0.0), 5),
                    "transport_retreat_x_calculation": transport_retreat_calc,
                    "transport_shift_y_m": round(_env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_SHIFT_Y_M", -0.28), 5),
                    "transport_lift_z_m": round(_env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_LIFT_Z_M", 0.10), 5),
                    "retreat_direction_world": _round_list([-approach[0], -approach[1], 0.0]),
                    "policy": (
                        "keep the validated side-grasp orientation, calculate a -X lane from live rack clearance, "
                        "shift -Y, lift +Z, then move to the rack, open fully, calculate a rack-clear retreat, "
                        "and back out along -approach at release height"
                    ),
                },
            }
        )
    return candidates


def _move_kwargs(side: str, pose: dict[str, Any], *, preview_only: bool = False) -> dict[str, Any]:
    prefix = "left" if side == "left" else "right"
    kwargs = {
        f"{prefix}_target_pos": [float(value) for value in pose["position"][:3]],
        f"{prefix}_target_rpy": [float(value) for value in pose["rpy"][:3]],
        "planner_backend": _env_str_pickup("OPENFORGE_BOTTLE_RACK_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": _env_str_pickup("OPENFORGE_BOTTLE_RACK_SOLVER_SPEED", "slow"),
        "planning_speed": _env_float_pickup("OPENFORGE_BOTTLE_RACK_PLANNING_SPEED", scaled_speed(0.40)),
        "preview_only": preview_only,
    }
    return kwargs


def _preview_pose(freespace_move: Callable[..., Any], side: str, pose: dict[str, Any], *, label: str) -> dict[str, Any]:
    try:
        preview = freespace_move(**_move_kwargs(side, pose, preview_only=True))
        cache_key = preview.get("trajectory_cache_key") if isinstance(preview, dict) else getattr(
            preview,
            "trajectory_cache_key",
            None,
        )
        return {
            "success": bool(cache_key),
            "label": label,
            "trajectory_cache_key": cache_key,
            "result": json_safe(preview),
        }
    except Exception as exc:
        return {"success": False, "label": label, "error": f"{type(exc).__name__}: {exc}"}


def _execute_pose(freespace_move: Callable[..., Any], side: str, pose: dict[str, Any], *, label: str) -> dict[str, Any]:
    preview = _preview_pose(freespace_move, side, pose, label=label)
    if not preview.get("success"):
        return {"stage": label, "ok": False, "preview": preview, "why_stopped": "preview failed before execution"}
    executed = freespace_move(trajectory_cache_key=preview["trajectory_cache_key"])
    return {"stage": label, "ok": True, "preview": preview, "execute": json_safe(executed)}


def _execute_pose_direct(freespace_move: Callable[..., Any], side: str, pose: dict[str, Any], *, label: str) -> dict[str, Any]:
    try:
        result = freespace_move(**_move_kwargs(side, pose, preview_only=False))
        return {"stage": label, "ok": True, "pose": pose, "execute": json_safe(result)}
    except Exception as exc:
        return {
            "stage": label,
            "ok": False,
            "pose": pose,
            "error": f"{type(exc).__name__}: {exc}",
            "why_stopped": "direct motion command failed",
        }


def _close_gripper_direct(side: str) -> dict[str, Any]:
    set_gripper = _required_tool("set_gripper")
    steps = [
        _env_float_pickup("OPENFORGE_BOTTLE_RACK_CLOSE_SCOUT_POS", 0.78),
        _env_float_pickup("OPENFORGE_BOTTLE_RACK_CLOSE_MID_POS", 0.58),
        _env_float_pickup("OPENFORGE_BOTTLE_RACK_CLOSE_TARGET_POS", 0.35),
    ]
    vel_limit = _env_float_pickup("OPENFORGE_BOTTLE_RACK_CLOSE_VEL_LIMIT", 0.60)
    torque_limit = _env_float_pickup("OPENFORGE_BOTTLE_RACK_CLOSE_TORQUE_LIMIT", 0.80)
    packet: dict[str, Any] = {"stage": "close_gripper", "ok": True, "side": side, "steps": []}
    for target in steps:
        try:
            result = set_gripper(side, target, vel_limit=vel_limit, torque_limit=torque_limit)
            packet["steps"].append({"target": target, "ok": True, "result": json_safe(result)})
        except Exception as exc:
            packet["ok"] = False
            packet["steps"].append({"target": target, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            break
    return packet


def _open_gripper_direct(side: str, *, label: str) -> dict[str, Any]:
    try:
        result = _make_open_gripper()(side)
        return {"stage": label, "ok": True, "side": side, "result": json_safe(result)}
    except Exception as exc:
        return {"stage": label, "ok": False, "side": side, "error": f"{type(exc).__name__}: {exc}"}


def _execute_direct_bottle_to_rack(
    *,
    selected_pickup: dict[str, Any],
    selected_place: dict[str, Any],
    starts_with_held_bottle: bool = False,
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    packet: dict[str, Any] = {
        "schema": "openforge.bottle_rack.direct_no_gate_attempt.v1",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "selected_place": selected_place,
        "steps": [],
        "why_stopped": None,
    }

    def _record(step: dict[str, Any], call: str | None = None) -> bool:
        packet["steps"].append(step)
        if call:
            packet["movement_capable_calls"].append(call)
            packet["physical_motion_executed"] = True
        step_ok = bool(step.get("ok", step.get("success", False)))
        if not step_ok:
            packet["why_stopped"] = f"{step.get('stage')} failed: {step.get('error') or step.get('why_stopped')}"
            return False
        return True

    if not starts_with_held_bottle:
        if not _record(_open_gripper_direct(side, label="open_gripper_before_pickup"), f"open_gripper:{side}"):
            return packet
        for label, pose in (
            ("pickup_pregrasp", selected_pickup["pregrasp_pose"]),
            ("pickup_grasp", selected_pickup["grasp_pose"]),
        ):
            if not _record(_execute_pose_direct(freespace_move, side, pose, label=label), f"freespace_move:{label}"):
                return packet
        if not _record(_close_gripper_direct(side), f"set_gripper:{side}"):
            return packet
        if not _record(
            _execute_pose_direct(freespace_move, side, selected_pickup["lift_pose"], label="pickup_lift"),
            "freespace_move:pickup_lift",
        ):
            return packet

    for waypoint in selected_place.get("transport_waypoints") or []:
        label = str(waypoint.get("label") or "transport_waypoint")
        pose = {"position": waypoint["position"], "rpy": waypoint["rpy"]}
        if not _record(_execute_pose_direct(freespace_move, side, pose, label=label), f"freespace_move:{label}"):
            return packet

    for label, pose in (
        ("rack_high_place", selected_place["high_place_pose"]),
        ("rack_release_pose", selected_place["release_pose"]),
    ):
        if not _record(_execute_pose_direct(freespace_move, side, pose, label=label), f"freespace_move:{label}"):
            return packet
    release_open = _open_gripper_until(
        side,
        min_pos=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_MIN_POS", 0.95),
        timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_TIMEOUT_S", 12.0),
        poll_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_POLL_S", 0.25),
        retry_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_RETRY_S", 1.5),
    )
    if not _record(
        {
            "stage": "open_release_on_rack",
            "ok": bool(release_open.get("success")),
            "side": side,
            "result": json_safe(release_open),
            "why_stopped": release_open.get("why_stopped"),
        },
        f"open_gripper:{side}",
    ):
        return packet
    retreat_waypoints = selected_place.get("retreat_waypoints") or [
        {
            "label": "rack_retreat_after_open",
            "position": selected_place["retreat_pose"]["position"],
            "rpy": selected_place["retreat_pose"]["rpy"],
        }
    ]
    for waypoint in retreat_waypoints:
        label = str(waypoint.get("label") or "rack_retreat_after_open")
        pose = {"position": waypoint["position"], "rpy": waypoint["rpy"]}
        if not _record(_execute_pose_direct(freespace_move, side, pose, label=label), f"freespace_move:{label}"):
            return packet
    home = _finish_home_if_requested(side)
    packet["steps"].append({"stage": "post_place_home", "ok": bool(home.get("success")), "result": json_safe(home)})
    packet["movement_capable_calls"].extend(home.get("movement_capable_calls") or [])
    if home.get("movement_capable_calls"):
        packet["physical_motion_executed"] = True
    if not home.get("success"):
        packet["why_stopped"] = f"retreat succeeded, but home/open recovery failed: {home.get('why_stopped')}"
        return packet
    packet["success"] = True
    packet["why_stopped"] = "direct bottle-to-rack command sequence completed"
    return packet


def _sequence_specs(
    selected_pickup: dict[str, Any],
    place_candidate: dict[str, Any],
    *,
    include_pickup: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    specs = []
    if include_pickup:
        specs.extend(
            [
                ("pickup_pregrasp", selected_pickup["pregrasp_pose"]),
                ("pickup_grasp", selected_pickup["grasp_pose"]),
                ("pickup_lift", selected_pickup["lift_pose"]),
            ]
        )
    specs.extend(
        [
            (
                str(waypoint.get("label") or "transport_waypoint"),
                {"position": waypoint["position"], "rpy": waypoint["rpy"]},
            )
            for waypoint in place_candidate.get("transport_waypoints") or []
        ]
    )
    specs.extend(
        [
            ("rack_high_place", place_candidate["high_place_pose"]),
            ("rack_release_pose", place_candidate["release_pose"]),
        ]
    )
    specs.extend(
        [
            (
                str(waypoint.get("label") or "rack_retreat_after_open"),
                {"position": waypoint["position"], "rpy": waypoint["rpy"]},
            )
            for waypoint in (
                place_candidate.get("retreat_waypoints")
                or [
                    {
                        "label": "rack_retreat_after_open",
                        "position": place_candidate["retreat_pose"]["position"],
                        "rpy": place_candidate["retreat_pose"]["rpy"],
                    }
                ]
            )
        ]
    )
    return specs


def _preview_place_sequence(
    run_dir: Path,
    selected_pickup: dict[str, Any],
    place_candidates: list[dict[str, Any]],
    *,
    include_pickup: bool = True,
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    attempts: list[dict[str, Any]] = []
    for candidate in place_candidates:
        steps: list[dict[str, Any]] = []
        success = True
        for label, pose in _sequence_specs(selected_pickup, candidate, include_pickup=include_pickup):
            preview = _preview_pose(freespace_move, side, pose, label=label)
            step = {"stage": label, "pose": pose, "preview": preview, "ok": bool(preview.get("success"))}
            steps.append(step)
            if not step["ok"]:
                success = False
                break
        attempt = {
            "label": candidate.get("label"),
            "success": success,
            "failed_stage": next((step["stage"] for step in steps if not step.get("ok")), None),
            "place_candidate": candidate,
            "steps": steps,
        }
        attempts.append(attempt)
        if success:
            packet = {
                "schema": "openforge.bottle_rack.sequence_preview.v1",
                "success": True,
                "side": side,
                "selected_pickup": selected_pickup,
                "selected_place": candidate,
                "include_pickup": bool(include_pickup),
                "attempts": attempts,
            }
            packet["path"] = write_json(run_dir / "plans" / "bottle_to_rack_sequence_preview.json", packet)
            append_stage_summary(
                run_dir,
                [
                    "## bottle to rack sequence preview",
                    "- success: True",
                    f"- selected_place: {candidate.get('label')}",
                    f"- release_pose: {candidate.get('release_pose')}",
                    f"- path: {packet['path']}",
                ],
            )
            return packet
    packet = {
        "schema": "openforge.bottle_rack.sequence_preview.v1",
        "success": False,
        "side": side,
        "selected_pickup": selected_pickup,
        "selected_place": None,
        "include_pickup": bool(include_pickup),
        "attempts": attempts,
    }
    packet["path"] = write_json(run_dir / "plans" / "bottle_to_rack_sequence_preview.json", packet)
    append_stage_summary(
        run_dir,
        [
            "## bottle to rack sequence preview",
            "- success: False",
            f"- attempts: {[(item.get('label'), item.get('failed_stage')) for item in attempts]}",
            f"- path: {packet['path']}",
        ],
    )
    return packet


def _return_to_source_after_lift(
    *,
    selected: dict[str, Any],
    freespace_move: Callable[..., Any],
) -> dict[str, Any]:
    side = str(selected["arm"])
    packet: dict[str, Any] = {
        "success": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    try:
        step = _execute_pose(freespace_move, side, selected["grasp_pose"], label="return_to_source_grasp_pose")
        packet["steps"].append(step)
        packet["movement_capable_calls"].append("freespace_move:return_to_source_grasp_pose")
        if not step.get("ok"):
            packet["why_stopped"] = "return-to-source grasp pose failed"
            return packet
        open_result = _open_gripper_until(
            side,
            min_pos=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_MIN_POS", 0.95),
            timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_TIMEOUT_S", 12.0),
            poll_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_POLL_S", 0.25),
            retry_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_RETRY_S", 1.5),
        )
        packet["steps"].append({"stage": "release_at_source_before_retreat", "result": json_safe(open_result)})
        packet["movement_capable_calls"].append(f"open_gripper:{side}")
        if not open_result.get("success"):
            packet["why_stopped"] = str(open_result.get("why_stopped") or "release at source failed")
            return packet
        if _truthy_env_pickup("OPENFORGE_BOTTLE_RACK_RETREAT_TO_PREGRASP_AFTER_RELEASE", True):
            step = _execute_pose(freespace_move, side, selected["pregrasp_pose"], label="retreat_to_source_pregrasp_pose")
            packet["steps"].append(step)
            packet["movement_capable_calls"].append("freespace_move:retreat_to_source_pregrasp_pose")
            if not step.get("ok"):
                packet["why_stopped"] = "retreat to source pregrasp failed"
                return packet
        packet["success"] = True
        packet["why_stopped"] = "returned to source grasp pose, opened fully, and retreated"
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
    return packet


def _finish_home_if_requested(side: str) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "success": True,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": "home not requested",
    }
    if not _truthy_env("OPENFORGE_BOTTLE_RACK_GO_HOME_AFTER_PLACE", True):
        return packet
    try:
        go_home = _required_tool("go_home")
        packet["steps"].append({"stage": "go_home_after_rack_retreat", "result": json_safe(go_home())})
        packet["movement_capable_calls"].append("go_home")
        if _truthy_env_pickup("OPENFORGE_BOTTLE_RACK_OPEN_AFTER_HOME", True):
            open_result = _open_gripper_until(
                side,
                min_pos=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_AFTER_HOME_MIN_POS", 0.90),
                timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_AFTER_HOME_TIMEOUT_S", 12.0),
                poll_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_AFTER_HOME_POLL_S", 0.25),
                retry_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_AFTER_HOME_RETRY_S", 1.5),
            )
            packet["steps"].append({"stage": "open_gripper_after_home", "result": json_safe(open_result)})
            packet["movement_capable_calls"].append(f"open_gripper:{side}")
            if not open_result.get("success"):
                packet["success"] = False
                packet["why_stopped"] = str(open_result.get("why_stopped") or "post-home open verification failed")
                return packet
        packet["why_stopped"] = "retreated, went home, and verified gripper open"
    except Exception as exc:
        packet["success"] = False
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
    return packet


def _verify_place(
    *,
    selected_pickup: dict[str, Any],
    selected_place: dict[str, Any],
    post_detection: dict[str, Any] | None,
) -> dict[str, Any]:
    source = selected_pickup.get("source_detection") or {}
    rack = selected_place.get("rack_detection") or {}
    release_target = selected_place.get("place_target_object_ref_m")
    motion = compare_detection_motion(source, post_detection)
    if post_detection is None or not release_target:
        return {
            "success": False,
            "confidence": 0.2,
            "reason": "missing post-place bottle detection or release target",
            "post_detection": post_detection,
            "object_motion": motion,
        }
    post_xyz = _xyz(post_detection)
    rack_xyz = _xyz(rack)
    if post_xyz is None:
        return {
            "success": False,
            "confidence": 0.2,
            "reason": "post-place detection has no 3D position",
            "post_detection": post_detection,
            "object_motion": motion,
        }
    dx = post_xyz[0] - float(release_target[0])
    dy = post_xyz[1] - float(release_target[1])
    xy_to_target = (dx * dx + dy * dy) ** 0.5
    rack_surface_z = (float(rack_xyz[2]) + _env_float("OPENFORGE_BOTTLE_RACK_SURFACE_Z_OFFSET_M", 0.0)) if rack_xyz else -999.0
    z_floor = rack_surface_z - _env_float("OPENFORGE_BOTTLE_RACK_VERIFY_Z_BELOW_RACK_TOL_M", 0.05)
    xy_ok = xy_to_target <= _env_float("OPENFORGE_BOTTLE_RACK_VERIFY_MAX_XY_TO_TARGET_M", 0.18)
    z_ok = post_xyz[2] >= z_floor
    success = bool(xy_ok and z_ok)
    return {
        "success": success,
        "confidence": 0.75 if success else 0.3,
        "reason": "bottle detected near rack placement target" if success else "post detection not close enough to rack target",
        "post_detection": post_detection,
        "object_motion": motion,
        "xy_to_target_m": round(xy_to_target, 5),
        "z_floor_m": round(z_floor, 5),
        "checks": {"xy_ok": xy_ok, "z_ok": z_ok},
    }


def _execute_place_after_lift(
    *,
    run_dir: Path,
    selected_pickup: dict[str, Any],
    selected_place: dict[str, Any],
    pre_detection: dict[str, Any],
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    side = str(selected_pickup["arm"])
    attempt: dict[str, Any] = {
        "schema": "openforge.bottle_rack.physical_place_after_lift.v1",
        "success": False,
        "verification_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "selected_pickup": selected_pickup,
        "selected_place": selected_place,
        "steps": [],
        "post_observation": None,
        "post_detection": None,
        "place_verification": None,
        "why_stopped": None,
    }
    try:
        for label, pose in (
            ("rack_high_place", selected_place["high_place_pose"]),
            ("rack_release_pose", selected_place["release_pose"]),
        ):
            step = _execute_pose(freespace_move, side, pose, label=label)
            attempt["physical_motion_executed"] = True
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)
            if not step.get("ok"):
                recovery = _return_to_source_after_lift(selected=selected_pickup, freespace_move=freespace_move)
                attempt["return_to_source_after_pre_release_failure"] = recovery
                attempt["movement_capable_calls"].extend(recovery.get("movement_capable_calls") or [])
                attempt["why_stopped"] = (
                    f"{label} failed before rack release; "
                    f"return-to-source recovery success={bool(recovery.get('success'))}"
                )
                return attempt

        release = _open_gripper_until(
            side,
            min_pos=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_MIN_POS", 0.95),
            timeout_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_TIMEOUT_S", 12.0),
            poll_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_POLL_S", 0.25),
            retry_s=_env_float_pickup("OPENFORGE_BOTTLE_RACK_OPEN_BEFORE_RETREAT_RETRY_S", 1.5),
        )
        attempt["physical_motion_executed"] = True
        attempt["movement_capable_calls"].append(f"open_gripper:{side}")
        attempt["steps"].append({"stage": "open_release_on_rack_before_retreat", "result": json_safe(release)})
        if not release.get("success"):
            attempt["why_stopped"] = str(release.get("why_stopped") or "release gripper did not confirm fully open")
            return attempt

        retreat_waypoints = selected_place.get("retreat_waypoints") or [
            {
                "label": "rack_retreat_after_open",
                "position": selected_place["retreat_pose"]["position"],
                "rpy": selected_place["retreat_pose"]["rpy"],
            }
        ]
        for waypoint in retreat_waypoints:
            label = str(waypoint.get("label") or "rack_retreat_after_open")
            pose = {"position": waypoint["position"], "rpy": waypoint["rpy"]}
            step = _execute_pose(freespace_move, side, pose, label=label)
            attempt["movement_capable_calls"].append(f"freespace_move:{label}")
            attempt["steps"].append(step)
            if not step.get("ok"):
                attempt["why_stopped"] = f"{label} failed after rack release"
                return attempt

        post_obs, post_det = _capture_prompt_fallback(
            run_dir=run_dir,
            stage="post_place_on_rack_observe",
            prompts=[pre_detection.get("prompt") or "bottle", "bottle", "wine bottle"],
            cameras=["top"],
            min_score=_env_float_pickup("OPENFORGE_BOTTLE_RACK_MIN_DETECTION_SCORE", 0.25),
        )
        attempt["post_observation"] = post_obs
        attempt["post_detection"] = post_det
        verification = _verify_place(
            selected_pickup=selected_pickup,
            selected_place=selected_place,
            post_detection=post_det,
        )
        attempt["place_verification"] = verification
        attempt["verification_success"] = bool(verification.get("success"))
        home = _finish_home_if_requested(side)
        attempt["post_place_home"] = home
        attempt["movement_capable_calls"].extend(home.get("movement_capable_calls") or [])
        if not home.get("success"):
            attempt["why_stopped"] = f"place sequence completed, but home/open recovery failed: {home.get('why_stopped')}"
            return attempt
        attempt["success"] = bool(verification.get("success"))
        attempt["why_stopped"] = (
            "bottle placement verified on/near rack"
            if attempt["success"]
            else f"place sequence completed but verification failed/inconclusive: {verification.get('reason')}"
        )
        return attempt
    except Exception as exc:
        attempt["why_stopped"] = f"{type(exc).__name__}: {exc}"
        attempt["steps"].append({"stage": "exception", "ok": False, "error": attempt["why_stopped"]})
        return attempt


def _initial_result(run_dir: Path, allow_physical: bool, plan_only: bool, attempt_budget: int) -> dict[str, Any]:
    return {
        "schema": "openforge.bottle_rack.result.v1",
        "success": False,
        "reward": 0.0,
        "confidence": 0.0,
        "method": TASK_NAME,
        "object_kind": "bottle",
        "target_kind": "rack",
        "run_dir": str(run_dir),
        "plan_success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "attempt_budget": attempt_budget,
        "attempt_count": 0,
        "allow_physical": allow_physical,
        "plan_only": plan_only,
        "selected_strategy": None,
        "selected_place": None,
        "pre_observation": None,
        "pre_detection": None,
        "rack_observation": None,
        "place_candidates_path": None,
        "sequence_preview": None,
        "physical_attempt": None,
        "post_observation": None,
        "post_detection": None,
        "verification": None,
        "video_paths": [],
        "why_stopped": "not_started",
        "next_recommendation": "Run no-motion plan checks first.",
        "attempts": [],
        "risk_notes": [
            "Bottle and rack poses come from fresh live top-camera observations.",
            "Manual XYZ env vars are not used for physical planning.",
            "The script verifies full gripper opening before retreating from the rack.",
        ],
        "config": {
            "arm": _env_str_pickup("OPENFORGE_BOTTLE_RACK_ARM", "left"),
            "planner_backend": _env_str_pickup("OPENFORGE_BOTTLE_RACK_PLANNER_BACKEND", "rrtconnect"),
            "planning_speed": _env_float_pickup("OPENFORGE_BOTTLE_RACK_PLANNING_SPEED", scaled_speed(0.40)),
            "lift_z_m": _env_float_pickup("OPENFORGE_BOTTLE_RACK_LIFT_Z_M", 0.30),
            "side_grasp_backoff_m": _env_float_pickup("OPENFORGE_BOTTLE_RACK_SIDE_GRASP_BACKOFF_M", -0.02),
            "side_pregrasp_clearance_m": _optional_env_float_pickup("OPENFORGE_BOTTLE_RACK_SIDE_PREGRASP_CLEARANCE_M"),
            "transport_waypoints_enabled": _truthy_env("OPENFORGE_BOTTLE_RACK_ENABLE_TRANSPORT_WAYPOINTS", True),
            "transport_retreat_x_mode": (
                "fixed_env"
                if os.environ.get("OPENFORGE_BOTTLE_RACK_TRANSPORT_RETREAT_X_M")
                else "rack_x_clearance"
            ),
            "transport_rack_x_clearance_m": _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_RACK_X_CLEARANCE_M", 0.25),
            "transport_min_retreat_x_m": _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_MIN_RETREAT_X_M", 0.08),
            "transport_max_retreat_x_m": _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_MAX_RETREAT_X_M", 0.30),
            "transport_retreat_x_fallback_m": _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_RETREAT_X_FALLBACK_M", -0.20),
            "transport_shift_y_m": _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_SHIFT_Y_M", -0.28),
            "transport_lift_z_m": _env_float("OPENFORGE_BOTTLE_RACK_TRANSPORT_LIFT_Z_M", 0.10),
            "rack_retreat_mode": (
                "fixed_env"
                if os.environ.get("OPENFORGE_BOTTLE_RACK_RETREAT_STANDOFF_M")
                else "rack_x_clearance"
            ),
            "rack_retreat_x_clearance_m": _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_RACK_X_CLEARANCE_M", 0.10),
            "rack_retreat_min_standoff_m": _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_MIN_STANDOFF_M", 0.10),
            "rack_retreat_max_standoff_m": _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_MAX_STANDOFF_M", 0.25),
            "rack_retreat_standoff_fallback_m": _env_float("OPENFORGE_BOTTLE_RACK_RETREAT_STANDOFF_FALLBACK_M", 0.10),
            "place_offsets_m": _offset_triples(
                "OPENFORGE_BOTTLE_RACK_PLACE_OFFSETS_M",
                [[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]],
            ),
        },
    }


def _run(*, held_pickup: dict[str, Any] | None = None) -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    allow_physical = _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION")
    plan_only = not allow_physical
    attempt_budget = max(1, _env_int_pickup("OPENFORGE_BOTTLE_RACK_ATTEMPT_BUDGET", 1))
    result = _initial_result(run_dir, allow_physical, plan_only, attempt_budget)
    starts_with_held_bottle = held_pickup is not None
    result["starts_with_held_bottle"] = starts_with_held_bottle

    try:
        prompts = _csv("OPENFORGE_BOTTLE_RACK_BOTTLE_PROMPTS", ["bottle", "wine bottle"])
        min_score = _env_float_pickup("OPENFORGE_BOTTLE_RACK_MIN_DETECTION_SCORE", 0.25)

        for attempt_index in range(1, attempt_budget + 1):
            attempt: dict[str, Any] = {
                "attempt_index": attempt_index,
                "success": False,
                "plan_success": False,
                "physical_motion_executed": False,
            }
            result["attempt_count"] = attempt_index
            result["attempts"].append(attempt)

            if starts_with_held_bottle:
                selected = json_safe(held_pickup)
                if not isinstance(selected, dict):
                    result["why_stopped"] = "held pickup plan was not a dict"
                    result["next_recommendation"] = "Inspect dry-pour selected_strategy.bottle_pickup before retry."
                    attempt["why_stopped"] = result["why_stopped"]
                    break
                pre_det = selected.get("source_detection") if isinstance(selected.get("source_detection"), dict) else None
                attempt["pre_observation"] = {"skipped": True, "reason": "bottle already held from dry pour"}
                attempt["pre_detection"] = pre_det
                attempt["planner_result"] = {
                    "skipped": True,
                    "reason": "bottle already held from dry pour",
                    "selected": selected,
                }
                attempt["selected_strategy"] = selected
                result["pre_observation"] = attempt["pre_observation"]
                result["pre_detection"] = pre_det
                result["selected_strategy"] = selected
            else:
                pre_obs, pre_det = _capture_prompt_fallback(
                    run_dir=run_dir,
                    stage=f"attempt_{attempt_index:02d}_pre_observe",
                    prompts=prompts,
                    cameras=["top"],
                    min_score=min_score,
                )
                attempt["pre_observation"] = pre_obs
                attempt["pre_detection"] = pre_det
                result["pre_observation"] = pre_obs
                result["pre_detection"] = pre_det
                if pre_det is None:
                    result["why_stopped"] = "fresh top-camera bottle detection failed"
                    result["next_recommendation"] = "Inspect top overlay and retry no-motion after the bottle is visible."
                    attempt["why_stopped"] = result["why_stopped"]
                    break

                candidates = _build_bottle_candidates(pre_det)
                selected = candidates[0] if candidates else None
                attempt["planner_result"] = {
                    "skipped_preview_gate": True,
                    "selected": selected,
                    "candidate_count": len(candidates),
                    "preview_count": 0,
                }
                attempt["selected_strategy"] = selected
                result["selected_strategy"] = selected
                if selected is None:
                    result["why_stopped"] = "no bottle candidate could be built from the live detection"
                    result["next_recommendation"] = "Inspect the bottle detection artifact."
                    attempt["why_stopped"] = result["why_stopped"]
                    break
                write_json(
                    run_dir / "plans" / "bottle_side_grasp_candidates.json",
                    {
                        "schema": "openforge.bottle_rack.direct_pickup_candidates.v1",
                        "selected": selected,
                        "candidate_count": len(candidates),
                        "candidates": candidates,
                    },
                )
                append_stage_summary(
                    run_dir,
                    [
                        "## direct bottle pickup candidate",
                        "- preview_gate: skipped",
                        f"- selected: {selected.get('label')}",
                        f"- pregrasp_pose: {selected.get('pregrasp_pose')}",
                        f"- grasp_pose: {selected.get('grasp_pose')}",
                        f"- lift_pose: {selected.get('lift_pose')}",
                    ],
                )

            rack_observation = _observe_rack(run_dir)
            result["rack_observation"] = rack_observation
            attempt["rack_observation"] = rack_observation
            rack_detection = rack_observation.get("selected_rack_detection")
            if not rack_detection:
                result["why_stopped"] = "fresh top-camera rack detection failed"
                result["next_recommendation"] = "Make the rack visible to the top camera and retry recorded no-motion validation."
                attempt["why_stopped"] = result["why_stopped"]
                break

            place_candidates = _make_place_candidates(selected, rack_detection)
            result["place_candidates_path"] = write_json(
                run_dir / "plans" / "bottle_rack_place_candidates.json",
                {
                    "schema": "openforge.bottle_rack.place_candidates.v1",
                    "selected_pickup": selected,
                    "rack_detection": rack_detection,
                    "candidate_count": len(place_candidates),
                    "candidates": place_candidates,
                },
            )
            selected_place = place_candidates[0] if place_candidates else None
            result["selected_place"] = selected_place
            result["plan_success"] = bool(selected and selected_place)
            attempt["plan_success"] = result["plan_success"]
            attempt["sequence_preview"] = {
                "skipped_preview_gate": True,
                "selected_place": selected_place,
            }
            result["sequence_preview"] = attempt["sequence_preview"]
            if selected_place is None:
                result["why_stopped"] = "no rack place candidate could be built from the live rack detection"
                result["next_recommendation"] = "Inspect the rack detection artifact."
                attempt["why_stopped"] = result["why_stopped"]
                break
            append_stage_summary(
                run_dir,
                [
                    "## direct rack place candidate",
                    "- preview_gate: skipped",
                    f"- selected_place: {selected_place.get('label')}",
                    f"- transport_waypoints: {selected_place.get('transport_waypoints')}",
                    f"- high_place_pose: {selected_place.get('high_place_pose')}",
                    f"- release_pose: {selected_place.get('release_pose')}",
                    f"- retreat_waypoints: {selected_place.get('retreat_waypoints')}",
                    f"- retreat_pose: {selected_place.get('retreat_pose')}",
                ],
            )

            if plan_only:
                result["why_stopped"] = "OPENFORGE_ALLOW_PHYSICAL_MOTION is not set; no motion executed"
                result["next_recommendation"] = "Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 for physical execution."
                attempt["why_stopped"] = result["why_stopped"]
                break

            direct_attempt = _execute_direct_bottle_to_rack(
                selected_pickup=selected,
                selected_place=selected_place,
                starts_with_held_bottle=starts_with_held_bottle,
            )
            direct_attempt["path"] = write_json(run_dir / "plans" / "direct_bottle_to_rack_attempt.json", direct_attempt)
            attempt["physical_attempt"] = direct_attempt
            result["physical_attempt"] = direct_attempt
            result["movement_capable_calls"].extend(direct_attempt.get("movement_capable_calls") or [])
            result["physical_motion_executed"] = bool(
                result["physical_motion_executed"] or direct_attempt.get("physical_motion_executed")
            )
            result["verification"] = {
                "skipped": True,
                "reason": "all task-level gates and post-action verification were removed for this direct script",
            }
            result["success"] = bool(direct_attempt.get("success"))
            result["reward"] = 1.0 if result["success"] else 0.0
            result["confidence"] = 0.5 if result["success"] else 0.0
            result["why_stopped"] = direct_attempt.get("why_stopped")
            result["next_recommendation"] = "Review videos and direct_bottle_to_rack_attempt.json."
            attempt["success"] = result["success"]
            attempt["why_stopped"] = result["why_stopped"]
            break
    except Exception as exc:
        result["why_stopped"] = f"script failed: {type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json and run log before retrying."
        result["error"] = result["why_stopped"]
    finally:
        result["video_paths"] = _video_paths(run_dir)
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        for attempt in result.get("attempts") or []:
            write_json(run_dir / "attempts" / f"attempt_{int(attempt['attempt_index']):02d}.json", attempt)
        append_stage_summary(
            run_dir,
            [
                "## bottle rack result",
                f"- success: {result['success']}",
                f"- plan_success: {result['plan_success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- movement_capable_calls: {result['movement_capable_calls']}",
                f"- video_paths: {result['video_paths']}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )
        print(f"[{TASK_NAME}] {result['why_stopped']}")
        print(json.dumps(json_safe(result), indent=2))
    return result


if not globals().get("_OPENFORGE_IMPORT_ONLY", False):
    TASK_RESULT.update(_run())
