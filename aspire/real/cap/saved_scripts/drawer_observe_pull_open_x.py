"""Observe the top drawer, pull/push it with the left arm, then observe again.

This is the canonical drawer practice entrypoint. When executed by
``run_script.py`` it runs the robot task. When executed directly, it sets the
same launcher defaults as the shell wrapper and execs ``run_script.py``.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from cap.saved_scripts.full_demo_speed import scaled_speed, scaled_speed_str


def _setdefault_env(values: dict[str, str]) -> None:
    for key, value in values.items():
        os.environ.setdefault(key, value)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _default_close_push_m() -> float:
    pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
    short_m = _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_LESS_THAN_PULL_M", 0.010)
    return max(0.0, pull_m - short_m)


def _set_launcher_defaults() -> None:
    _setdefault_env(
        {
            "CAP_TOP_CAMERA_BACKEND": "realsense",
            "CAP_TOP_CAMERA_FRAME": "top_camera_d405",
            "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP": "0",
            "OPENFORGE_PREVIEW_RECORDER_BACKEND": "python",
            "OPENFORGE_PREVIEW_RECORDER_PROBE_TIMEOUT_S": "8.0",
            "OPENFORGE_PREVIEW_RECORDER_REENCODE_H264": "1",
            "OPENFORGE_PREVIEW_RECORDER_REQUIRE_H264": "1",
            "OPENFORGE_ALLOW_PHYSICAL_MOTION": "1",
            "OPENFORGE_DRAWER_ENABLE_PHYSICAL_CONTACT": "1",
            "OPENFORGE_DRAWER_PROMPTS": "drawer handle,drawer,cabinet handle",
            "OPENFORGE_DEBUG_OBS_CAMERAS": "top,left,right",
            "OPENFORGE_DRAWER_HANDLE_SELECT_STRATEGY": "front_face",
            "OPENFORGE_DRAWER_FRONT_HANDLE_CAMERAS": "left,right",
            "OPENFORGE_DRAWER_POST_OBSERVE": "1",
            "OPENFORGE_DRAWER_CYCLE_MODE": "open_then_close",
            "OPENFORGE_DRAWER_MANUAL_SIDE": "left",
            "OPENFORGE_DRAWER_MANUAL_X_BIAS_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_Y_BIAS_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_Y_STANDOFF_M": "0.050",
            "OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY": "-1,0",
            "OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M": "-0.012",
            "OPENFORGE_DRAWER_MANUAL_PULL_M": "0.105",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_M": "0.085",
            "OPENFORGE_DRAWER_MANUAL_ABS_Z_M": "0.905",
            "OPENFORGE_DRAWER_MANUAL_RPY": "60,-90,-180",
            "OPENFORGE_DRAWER_MANUAL_CONTACT": "1",
            "OPENFORGE_DRAWER_MANUAL_EXECUTE": "1",
            "OPENFORGE_DRAWER_MANUAL_MAX_FIRST_MOVE_M": "0.550",
            "OPENFORGE_DRAWER_MANUAL_PLANNING_SPEED": scaled_speed_str(0.4),
            "OPENFORGE_DRAWER_MANUAL_PLANNER_BACKEND": "rrtconnect",
            "OPENFORGE_DRAWER_MANUAL_GRIPPER_METHOD": "servo",
            "OPENFORGE_DRAWER_MANUAL_PRECONTACT_GRIPPER": "0.78",
            "OPENFORGE_DRAWER_MANUAL_ADVANCE_GRIPPER": "0.72",
            "OPENFORGE_DRAWER_MANUAL_SCOUT_GRIPPER": "0.54",
            "OPENFORGE_DRAWER_MANUAL_TARGET_GRIPPER": "0.10",
            "OPENFORGE_DRAWER_MANUAL_CONTACT_GRIPPER": "0.10",
            "OPENFORGE_DRAWER_MANUAL_VALIDATE_GRIPPER_CLOSE": "1",
            "OPENFORGE_DRAWER_MANUAL_MIN_CLOSE_DELTA_POS": "0.12",
            "OPENFORGE_DRAWER_MANUAL_MAX_CLOSED_GRIPPER_POS": "0.24",
            "OPENFORGE_DRAWER_MANUAL_SKIP_CLOSE": "0",
            "OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_M": "0",
        }
    )

    operator_set_push = "OPENFORGE_DRAWER_MANUAL_PUSH_M" in os.environ
    os.environ.setdefault(
        "OPENFORGE_DRAWER_MANUAL_PUSH_M_OPERATOR_SET",
        "1" if operator_set_push else "0",
    )
    cycle_mode = _cycle_mode_from_env()
    close_cycle = cycle_mode == "open_then_close"
    _setdefault_env(
        {
            "OPENFORGE_DRAWER_MANUAL_PUSH_M": f"{_default_close_push_m():.5f}" if close_cycle else "0.000",
            "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER": "0" if close_cycle else "1",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER": "1.0" if close_cycle else "0.10",
            "OPENFORGE_DRAWER_MANUAL_OPEN_AFTER_RETREAT": "0" if close_cycle else "1",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_Z_M": "0.000" if close_cycle else "0.025",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_GRIPPER": "1.0" if close_cycle else "0.10",
        }
    )


def _cycle_mode_from_env() -> str:
    raw = os.environ.get("OPENFORGE_DRAWER_CYCLE_MODE", "open_then_close").strip().lower()
    aliases = {
        "open": "open_only",
        "pull_only": "open_only",
        "pull": "open_only",
        "open_close": "open_then_close",
        "pull_push": "open_then_close",
        "pull_then_push": "open_then_close",
        "cycle": "open_then_close",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"open_only", "open_then_close"}:
        raise RuntimeError(
            "OPENFORGE_DRAWER_CYCLE_MODE must be open_only or open_then_close "
            f"(got {raw!r})"
        )
    return mode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _running_under_run_script() -> bool:
    namespace = sys.modules.get("skill_library.namespace")
    return callable(getattr(namespace, "get_robot_state", None))


def _run_script_argv(extra_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "run_script.py",
        "script_file=cap/saved_scripts/drawer_observe_pull_open_x.py",
        "skill_library_path=cap/saved_scripts/skill_library",
        "env.name=yam-real",
        "robot=real_yam",
        "robot.dashboard=true",
        "robot.await_exit=false",
        "robot.go_home_on_exit=false",
        "runtime.no_cameras=true",
        "recording.enabled=true",
        "debug_ui.enabled=true",
        "debug_ui.auto_open=true",
        "debug_ui.auto_exit_on_run_end=false",
        "debug_ui.host=0.0.0.0",
        *extra_args,
    ]


def _launch_run_script() -> None:
    root = _repo_root()
    os.chdir(root)
    _set_launcher_defaults()
    argv = _run_script_argv(sys.argv[1:])
    forge_env = root / ".forge_env"
    if forge_env.exists() and os.environ.get("OPENFORGE_DRAWER_SKIP_FORGE_ENV_SOURCE", "0") != "1":
        command = "source .forge_env && exec " + " ".join(shlex.quote(part) for part in argv)
        os.execvp("bash", ["bash", "-lc", command])
    os.execvp(argv[0], argv)


if globals().get("__name__", "__main__") == "__main__" and not _running_under_run_script():
    _launch_run_script()


from skill_library.debug_observation import capture_observation, current_run_dir, write_stage_summary
from skill_library.namespace import detect_objects_oneshot, freespace_move, get_robot_state, servo_ee_delta, set_gripper


_setdefault_env(
    {
        "OPENFORGE_DRAWER_PROMPTS": "drawer handle,drawer,cabinet handle",
        "OPENFORGE_DEBUG_OBS_CAMERAS": "top,left,right",
        "OPENFORGE_DRAWER_HANDLE_SELECT_STRATEGY": "front_face",
        "OPENFORGE_DRAWER_FRONT_HANDLE_CAMERAS": "left,right",
    }
)

os.environ["OPENFORGE_DRAWER_STAGE"] = "observe"


def _initial_task_result() -> dict[str, Any]:
    return {
        "success": False,
        "reward": 0.0,
        "method": "drawer_observe_pull_open_x",
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "pre_observe": None,
        "manual_probe": None,
        "post_observe": None,
        "artifacts": {},
    }


TASK_RESULT: dict[str, Any] = _initial_task_result()


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def reset_task_result() -> None:
    TASK_RESULT.clear()
    TASK_RESULT.update(_initial_task_result())


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _cycle_mode() -> str:
    return _cycle_mode_from_env()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def _handle_xyz(observation: dict[str, Any]) -> list[float]:
    handle = observation.get("handle_detection")
    if not isinstance(handle, dict):
        raise RuntimeError("pre_observe did not produce handle_detection")
    xyz = handle.get("position_3d")
    if not isinstance(xyz, list) or len(xyz) < 3:
        raise RuntimeError(f"pre_observe handle_detection has invalid position_3d: {xyz!r}")
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _load_observation_packet(observation: dict[str, Any]) -> dict[str, Any] | None:
    raw_path = observation.get("packet_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        path = current_run_dir() / raw_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _handle_prompts() -> list[str]:
    prompts = [
        part.strip()
        for part in os.environ.get("OPENFORGE_DRAWER_PROMPTS", "drawer handle,drawer,cabinet handle").split(",")
        if part.strip()
    ]
    return [prompt for prompt in prompts if "handle" in prompt.lower() or "pull" in prompt.lower()]


def _packet_detections(packet: dict[str, Any], camera: str, prompt: str) -> list[dict[str, Any]]:
    cam = packet.get("cameras", {}).get(camera, {})
    dets = cam.get("detections", {}).get(prompt, [])
    return dets if isinstance(dets, list) else []


def _score(det: dict[str, Any]) -> float:
    try:
        return float(det.get("score") or 0.0)
    except Exception:
        return 0.0


def _det_xyz(det: dict[str, Any]) -> list[float] | None:
    xyz = det.get("position_3d")
    if not isinstance(xyz, list) or len(xyz) < 3:
        return None
    try:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except Exception:
        return None


def _best_top_handle(packet: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for prompt in _handle_prompts():
        for det in _packet_detections(packet, "top", prompt):
            xyz = _det_xyz(det)
            if xyz is None:
                continue
            candidates.append(
                {
                    "camera": "top",
                    "prompt": prompt,
                    "label": det.get("label", prompt),
                    "score": _score(det),
                    "box_2d": det.get("box_2d"),
                    "position_3d": xyz,
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item["score"]), item["prompt"]))
    return candidates[0]


def _all_prompts() -> list[str]:
    return [
        part.strip()
        for part in os.environ.get(
            "OPENFORGE_DRAWER_PROMPTS",
            os.environ.get("OPENFORGE_DEBUG_OBS_PROMPTS", "drawer handle,drawer,cabinet handle"),
        ).split(",")
        if part.strip()
    ]


def _all_cameras() -> list[str]:
    return [
        part.strip()
        for part in os.environ.get("OPENFORGE_DEBUG_OBS_CAMERAS", "top,left,right,bottom").split(",")
        if part.strip()
    ]


def _calibrated_motion_cameras() -> set[str]:
    return {"top", "left", "right"}


def _handle_min_score() -> float:
    return _env_float("OPENFORGE_DRAWER_HANDLE_MIN_SCORE", 0.45)


def _target_drawer() -> str:
    return os.environ.get("OPENFORGE_DRAWER_TARGET", "top").strip().lower()


def _portal_camera(camera: str = "top") -> Any:
    import portal

    return portal.Client(os.environ["OPENFORGE_DEBUG_OBS_CAMERA_PORTAL"]).get_camera_image(camera).result()


def _capture(stage: str) -> dict[str, Any]:
    capture_state = _truthy("OPENFORGE_DEBUG_OBS_CAPTURE_STATE", "1")
    packet = capture_observation(
        stage=stage,
        prompts=_all_prompts(),
        cameras=_all_cameras(),
        detect_fn=detect_objects_oneshot,
        get_camera_fn=_portal_camera if os.environ.get("OPENFORGE_DEBUG_OBS_CAMERA_PORTAL", "").strip() else None,
        get_robot_state_fn=get_robot_state if capture_state else None,
        capture_robot_state=capture_state,
        per_call_timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 8.0),
    )
    TASK_RESULT.setdefault("artifacts", {})[f"{stage}_packet"] = packet.get("packet_path")
    TASK_RESULT.setdefault("artifacts", {})["stage_summary"] = str(current_run_dir() / "stage_summary.md")
    return packet


def _detection_candidates(packet: dict[str, Any], prompts: list[str], cameras: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    prompt_rank = {prompt: idx for idx, prompt in enumerate(prompts)}
    calibrated = _calibrated_motion_cameras()
    for camera in cameras:
        if camera not in calibrated:
            continue
        for prompt in prompts:
            for det in _packet_detections(packet, camera, prompt):
                xyz = _det_xyz(det)
                if xyz is None:
                    continue
                candidates.append(
                    {
                        "camera": camera,
                        "prompt": prompt,
                        "label": det.get("label", prompt),
                        "score": _score(det),
                        "box_2d": det.get("box_2d"),
                        "position_3d": xyz,
                        "prompt_rank": prompt_rank.get(prompt, 99),
                    }
                )
    return candidates


def _round_list(values: Any, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def _best_detection(
    packet: dict[str, Any],
    prompts: list[str],
    cameras: list[str],
    *,
    drawer_target: str | None = None,
) -> dict[str, Any] | None:
    candidates = _detection_candidates(packet, prompts, cameras)
    target = (drawer_target or "").strip().lower()
    if target in {"top", "upper"}:
        candidates.sort(key=lambda item: (item["prompt_rank"], -float(item["position_3d"][2]), -item["score"]))
    elif target in {"bottom", "lower"}:
        candidates.sort(key=lambda item: (item["prompt_rank"], float(item["position_3d"][2]), -item["score"]))
    else:
        candidates.sort(key=lambda item: (item["prompt_rank"], -item["score"]))
    return candidates[0] if candidates else None


def _best_handle_detection(
    packet: dict[str, Any],
    prompts: list[str],
    cameras: list[str],
    *,
    drawer_target: str | None = None,
) -> dict[str, Any] | None:
    candidates = _detection_candidates(packet, prompts, cameras)
    if not candidates:
        return None
    min_score = _handle_min_score()
    high_conf = [candidate for candidate in candidates if float(candidate["score"]) >= min_score]
    usable = high_conf or candidates
    target = (drawer_target or "").strip().lower()
    camera_rank = {"left": 0, "top": 1, "right": 2}
    if target in {"top", "upper"}:
        usable.sort(
            key=lambda item: (
                item["prompt_rank"],
                camera_rank.get(str(item.get("camera")), 9),
                -float(item["score"]),
                -float(item["position_3d"][2]),
            )
        )
    elif target in {"bottom", "lower"}:
        usable.sort(
            key=lambda item: (
                item["prompt_rank"],
                camera_rank.get(str(item.get("camera")), 9),
                -float(item["score"]),
                float(item["position_3d"][2]),
            )
        )
    else:
        usable.sort(
            key=lambda item: (
                item["prompt_rank"],
                camera_rank.get(str(item.get("camera")), 9),
                -float(item["score"]),
            )
        )
    return usable[0]


def _drawer_body_for_handle(
    packet: dict[str, Any],
    handle: dict[str, Any] | None,
    notes: list[str],
) -> dict[str, Any] | None:
    cameras = _all_cameras()
    drawer_candidates = _detection_candidates(packet, ["drawer"], cameras)
    if not drawer_candidates:
        return None
    target = _target_drawer()
    if handle is None or not handle.get("position_3d"):
        return _best_detection(packet, ["drawer"], cameras, drawer_target=target)

    handle_xyz = [float(x) for x in handle["position_3d"][:3]]
    max_xy = _env_float("OPENFORGE_DRAWER_MAX_HANDLE_DRAWER_XY_M", 0.28)
    max_z = _env_float("OPENFORGE_DRAWER_MAX_HANDLE_DRAWER_Z_M", 0.16)
    associated: list[tuple[tuple[float, float, float, float], dict[str, Any], float, float]] = []
    for det in drawer_candidates:
        drawer_xyz = [float(x) for x in det["position_3d"][:3]]
        xy_dist = math.hypot(handle_xyz[0] - drawer_xyz[0], handle_xyz[1] - drawer_xyz[1])
        z_dist = abs(handle_xyz[2] - drawer_xyz[2])
        if xy_dist > max_xy or z_dist > max_z:
            continue
        same_camera_rank = 0.0 if det.get("camera") == handle.get("camera") else 1.0
        associated.append(((same_camera_rank, xy_dist, z_dist, -float(det["score"])), det, xy_dist, z_dist))

    if associated:
        associated.sort(key=lambda item: item[0])
        _, selected, xy_dist, z_dist = associated[0]
        notes.append(
            "Selected drawer body associated with handle "
            f"(camera={selected['camera']}, xy_dist={xy_dist:.3f}m, z_dist={z_dist:.3f}m)."
        )
        return selected

    fallback = _best_detection(packet, ["drawer"], cameras, drawer_target=target)
    if fallback is not None:
        notes.append(
            "No drawer-body detection was close enough to the selected handle; "
            "falling back to target/z-ranked drawer body."
        )
    return fallback


def _front_face_handle_detection(
    packet: dict[str, Any],
    prompts: list[str],
    cameras: list[str],
    notes: list[str],
) -> dict[str, Any] | None:
    side_cameras = [
        camera.strip()
        for camera in os.environ.get("OPENFORGE_DRAWER_FRONT_HANDLE_CAMERAS", "left,right").split(",")
        if camera.strip()
    ]
    calibrated = _calibrated_motion_cameras()
    side_cameras = [camera for camera in side_cameras if camera in cameras and camera in calibrated]
    if not side_cameras:
        return None

    min_score = _env_float("OPENFORGE_DRAWER_FRONT_HANDLE_MIN_SCORE", _handle_min_score())
    scored: list[tuple[tuple[float, float, float, float], dict[str, Any]]] = []
    for handle in _detection_candidates(packet, prompts, side_cameras):
        if float(handle["score"]) < min_score:
            continue
        bbox = handle.get("box_2d")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        drawer = _drawer_body_for_handle(packet, handle, [])
        drawer_box = drawer.get("box_2d") if isinstance(drawer, dict) else None
        if not isinstance(drawer_box, list) or len(drawer_box) < 4:
            continue

        handle_cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
        drawer_top = float(drawer_box[1])
        drawer_bottom = float(drawer_box[3])
        drawer_h = max(1.0, drawer_bottom - drawer_top)
        rel_y = (handle_cy - drawer_top) / drawer_h
        min_rel_y = _env_float("OPENFORGE_DRAWER_FRONT_HANDLE_MIN_REL_Y", 0.30)
        max_rel_y = _env_float("OPENFORGE_DRAWER_FRONT_HANDLE_MAX_REL_Y", 0.95)
        if rel_y < min_rel_y or rel_y > max_rel_y:
            continue

        handle_xyz = [float(x) for x in handle["position_3d"][:3]]
        drawer_xyz = [float(x) for x in drawer["position_3d"][:3]]
        xy_dist = math.hypot(handle_xyz[0] - drawer_xyz[0], handle_xyz[1] - drawer_xyz[1])
        max_xy = _env_float("OPENFORGE_DRAWER_FRONT_HANDLE_MAX_DRAWER_XY_M", 0.22)
        if xy_dist > max_xy:
            continue

        camera_rank = float(side_cameras.index(handle["camera"]))
        scored.append(((camera_rank, -float(handle["score"]), abs(rel_y - 0.60), xy_dist), handle))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    selected = scored[0][1]
    notes.append(
        "Selected front-face side-camera handle "
        f"(camera={selected['camera']}, xyz={_round_list(selected['position_3d'])})."
    )
    return selected


def _select_handle_and_drawer(packet: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    cameras = _all_cameras()
    target = _target_drawer()
    drawer_target = target if target in {"top", "upper", "bottom", "lower"} else None
    handle_prompts = [
        prompt for prompt in _all_prompts() if any(token in prompt.lower() for token in ("handle", "pull"))
    ] or ["drawer handle", "cabinet handle"]
    strategy = os.environ.get("OPENFORGE_DRAWER_HANDLE_SELECT_STRATEGY", "front_face").strip().lower()

    handle = None
    if strategy in {"front", "front_face", "side_front", "side"}:
        handle = _front_face_handle_detection(packet, handle_prompts, cameras, notes)
    if handle is None:
        handle = _best_handle_detection(packet, handle_prompts, cameras, drawer_target=drawer_target)
        if strategy in {"front", "front_face", "side_front", "side"}:
            notes.append("No side-camera front-face handle passed filters; fell back to score/camera-ranked handle detection.")
    drawer = _drawer_body_for_handle(packet, handle, notes)
    if drawer_target:
        notes.append(f"Selected {drawer_target} drawer target using {strategy} handle strategy.")
    if handle is None:
        notes.append("No calibrated-camera detection for 'drawer handle' or 'cabinet handle'.")
    elif float(handle["score"]) < _handle_min_score():
        notes.append(
            f"Best handle score {float(handle['score']):.3f} is below "
            f"OPENFORGE_DRAWER_HANDLE_MIN_SCORE={_handle_min_score():.3f}."
        )
    if drawer is None:
        notes.append("No calibrated-camera drawer-body detection; handle-only targeting will be used.")
    return handle, drawer, notes


def _observe(stage: str = "observe") -> dict[str, Any]:
    packet = _capture(stage)
    handle, drawer, notes = _select_handle_and_drawer(packet)
    summary = {
        "success": handle is not None and float(handle.get("score") or 0.0) >= _handle_min_score(),
        "handle_detection": handle,
        "drawer_detection": drawer,
        "notes": notes,
        "packet_path": packet.get("packet_path"),
    }
    write_stage_summary(stage=stage, result=summary, log_dir=current_run_dir())
    return summary


def _fused_handle_xyz(observation: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    """Use top view to stabilize drawer planar targeting while keeping side-view evidence.

    Side/front cameras see the drawer handle shape well, but recent videos show
    their 3D median can land on one post of the handle. The top camera is a
    better source for the world-XY centerline, so we allow it to correct Y
    strongly and X only by a capped amount.
    """
    selected = observation.get("handle_detection")
    raw_xyz = _handle_xyz(observation)
    info: dict[str, Any] = {
        "enabled": _truthy("OPENFORGE_DRAWER_TOP_PLANAR_FUSION", "1"),
        "raw_handle_xyz": [round(float(x), 5) for x in raw_xyz],
        "source_camera": selected.get("camera") if isinstance(selected, dict) else None,
        "source_score": selected.get("score") if isinstance(selected, dict) else None,
        "applied": False,
        "reason": "disabled",
    }
    if not info["enabled"]:
        return raw_xyz, info

    packet = _load_observation_packet(observation)
    if packet is None:
        info["reason"] = "observation packet unavailable"
        return raw_xyz, info

    top = _best_top_handle(packet)
    if top is None:
        info["reason"] = "no top-camera handle detection"
        return raw_xyz, info

    top_xyz = [float(x) for x in top["position_3d"]]
    top_score = float(top["score"])
    info["top_handle"] = {
        "camera": top.get("camera"),
        "prompt": top.get("prompt"),
        "score": round(top_score, 4),
        "box_2d": top.get("box_2d"),
        "position_3d": [round(float(x), 5) for x in top_xyz],
    }
    min_score = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_MIN_SCORE", 0.70)
    if top_score < min_score:
        info["reason"] = f"top score {top_score:.3f} below {min_score:.3f}"
        return raw_xyz, info

    dx = float(top_xyz[0] - raw_xyz[0])
    dy = float(top_xyz[1] - raw_xyz[1])
    planar_delta = math.hypot(dx, dy)
    max_planar = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_MAX_DELTA_M", 0.24)
    info["raw_delta_m"] = {
        "x": round(dx, 5),
        "y": round(dy, 5),
        "xy": round(planar_delta, 5),
    }
    if planar_delta > max_planar:
        info["reason"] = f"top/selected planar delta {planar_delta:.3f}m exceeds {max_planar:.3f}m"
        return raw_xyz, info

    min_x = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_MIN_X_CORRECTION_M", 0.015)
    min_y = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_MIN_Y_CORRECTION_M", 0.015)
    x_blend = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_X_BLEND", 0.35)
    y_blend = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_Y_BLEND", 1.0)
    max_x = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_MAX_X_CORRECTION_M", 0.040)
    max_y = _env_float("OPENFORGE_DRAWER_TOP_PLANAR_MAX_Y_CORRECTION_M", 0.180)

    corr_x = 0.0 if abs(dx) < min_x else _clamp(dx * x_blend, -max_x, max_x)
    corr_y = 0.0 if abs(dy) < min_y else _clamp(dy * y_blend, -max_y, max_y)
    fused = [raw_xyz[0] + corr_x, raw_xyz[1] + corr_y, raw_xyz[2]]
    info.update(
        {
            "applied": abs(corr_x) > 1e-6 or abs(corr_y) > 1e-6,
            "reason": "top planar fusion applied",
            "correction_m": {"x": round(corr_x, 5), "y": round(corr_y, 5)},
            "fused_handle_xyz": [round(float(x), 5) for x in fused],
            "limits": {
                "x_blend": x_blend,
                "y_blend": y_blend,
                "max_x_correction_m": max_x,
                "max_y_correction_m": max_y,
                "max_planar_delta_m": max_planar,
            },
        }
    )
    return fused, info


def _format_xyz(xyz: list[float]) -> str:
    return ",".join(f"{float(x):.5f}" for x in xyz)


def _set_manual_defaults(handle_xyz: list[float], *, cycle_mode: str) -> None:
    close_cycle = cycle_mode == "open_then_close"
    operator_set_push = "OPENFORGE_DRAWER_MANUAL_PUSH_M" in os.environ
    os.environ.setdefault(
        "OPENFORGE_DRAWER_MANUAL_PUSH_M_OPERATOR_SET",
        "1" if operator_set_push else "0",
    )
    _setdefault_env(
        {
            "OPENFORGE_DRAWER_MANUAL_SIDE": "left",
            "OPENFORGE_DRAWER_MANUAL_X_BIAS_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_Y_BIAS_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_Y_STANDOFF_M": "0.050",
            "OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY": "-1,0",
            "OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M": "-0.012",
            "OPENFORGE_DRAWER_MANUAL_PULL_M": "0.105",
            "OPENFORGE_DRAWER_MANUAL_PUSH_M": f"{_default_close_push_m():.5f}" if close_cycle else "0.000",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_M": "0.085",
            "OPENFORGE_DRAWER_MANUAL_ABS_Z_M": "0.905",
            "OPENFORGE_DRAWER_MANUAL_RPY": "60,-90,-180",
            "OPENFORGE_DRAWER_MANUAL_CONTACT": "1",
            "OPENFORGE_DRAWER_MANUAL_EXECUTE": "1",
            "OPENFORGE_DRAWER_MANUAL_MAX_FIRST_MOVE_M": "0.550",
            "OPENFORGE_DRAWER_MANUAL_PLANNING_SPEED": scaled_speed_str(0.4),
            "OPENFORGE_DRAWER_MANUAL_PLANNER_BACKEND": "rrtconnect",
            "OPENFORGE_DRAWER_MANUAL_GRIPPER_METHOD": "servo",
            "OPENFORGE_DRAWER_MANUAL_PRECONTACT_GRIPPER": "0.78",
            "OPENFORGE_DRAWER_MANUAL_ADVANCE_GRIPPER": "0.72",
            "OPENFORGE_DRAWER_MANUAL_SCOUT_GRIPPER": "0.54",
            "OPENFORGE_DRAWER_MANUAL_TARGET_GRIPPER": "0.10",
            "OPENFORGE_DRAWER_MANUAL_CONTACT_GRIPPER": "0.10",
            "OPENFORGE_DRAWER_MANUAL_VALIDATE_GRIPPER_CLOSE": "1",
            "OPENFORGE_DRAWER_MANUAL_MIN_CLOSE_DELTA_POS": "0.12",
            "OPENFORGE_DRAWER_MANUAL_MAX_CLOSED_GRIPPER_POS": "0.24",
            "OPENFORGE_DRAWER_MANUAL_SKIP_CLOSE": "0",
            "OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_M": "0",
            "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER": "0" if close_cycle else "1",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER": "1.0" if close_cycle else "0.10",
            "OPENFORGE_DRAWER_MANUAL_OPEN_AFTER_RETREAT": "0" if close_cycle else "1",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_Z_M": "0.000" if close_cycle else "0.025",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_GRIPPER": "1.0" if close_cycle else "0.10",
        }
    )
    os.environ["OPENFORGE_DRAWER_MANUAL_HANDLE_XYZ"] = _format_xyz(handle_xyz)


def _axis_xy_from_env() -> list[float]:
    raw = os.environ.get("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", "-1,0")
    values = [float(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(values) != 2:
        raise RuntimeError("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY must contain two comma-separated floats")
    norm = math.hypot(values[0], values[1])
    if norm < 1e-6:
        raise RuntimeError("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY must have nonzero length")
    return [values[0] / norm, values[1] / norm]


def _adapt_close_push_distance(*, cycle_mode: str, fusion: dict[str, Any]) -> dict[str, Any] | None:
    if cycle_mode != "open_then_close":
        return None
    info: dict[str, Any] = {
        "enabled": _truthy("OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_M", "0"),
        "applied": False,
    }
    if not info["enabled"]:
        info["reason"] = "disabled"
        return info
    if _truthy("OPENFORGE_DRAWER_MANUAL_PUSH_M_LOCKED", "0"):
        info["reason"] = "OPENFORGE_DRAWER_MANUAL_PUSH_M_LOCKED=1"
        return info
    if _truthy("OPENFORGE_DRAWER_MANUAL_PUSH_M_OPERATOR_SET", "0"):
        info["reason"] = "OPENFORGE_DRAWER_MANUAL_PUSH_M was operator-set before wrapper defaults"
        return info

    top = fusion.get("top_handle") if isinstance(fusion, dict) else None
    fused_xyz = fusion.get("fused_handle_xyz") if isinstance(fusion, dict) else None
    if not isinstance(top, dict) or not isinstance(fused_xyz, list) or len(fused_xyz) < 2:
        info["reason"] = "no top/fused handle pair"
        return info
    top_xyz = top.get("position_3d")
    if not isinstance(top_xyz, list) or len(top_xyz) < 2:
        info["reason"] = "top handle has no valid position"
        return info

    pull_axis = _axis_xy_from_env()
    close_axis = [-pull_axis[0], -pull_axis[1]]
    residual = (float(top_xyz[0]) - float(fused_xyz[0])) * close_axis[0] + (
        float(top_xyz[1]) - float(fused_xyz[1])
    ) * close_axis[1]
    min_extra = _env_float("OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_MIN_EXTRA_M", 0.015)
    max_extra = _env_float("OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_MAX_EXTRA_M", 0.080)
    if residual < min_extra:
        info.update(
            {
                "reason": f"close-axis residual {residual:.3f}m below {min_extra:.3f}m",
                "close_axis_xy": [round(float(x), 5) for x in close_axis],
                "residual_m": round(float(residual), 5),
            }
        )
        return info

    pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
    current_push_m = _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_M", pull_m)
    extra = _clamp(residual, 0.0, max_extra)
    adapted_push_m = max(current_push_m, pull_m + extra)
    os.environ["OPENFORGE_DRAWER_MANUAL_PUSH_M"] = f"{adapted_push_m:.5f}"
    info.update(
        {
            "applied": True,
            "reason": "added capped top-view X residual to close push distance",
            "close_axis_xy": [round(float(x), 5) for x in close_axis],
            "residual_m": round(float(residual), 5),
            "extra_push_m": round(float(extra), 5),
            "previous_push_m": round(float(current_push_m), 5),
            "adapted_push_m": round(float(adapted_push_m), 5),
        }
    )
    return info


class _StopAfterStage(Exception):
    """Internal control flow for bounded diagnostic partial runs."""


def _floats(name: str, count: int, default: str | None = None) -> list[float]:
    raw = os.environ.get(name, default or "")
    vals = [float(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
    if len(vals) != count:
        raise RuntimeError(f"{name} must contain {count} comma-separated floats")
    return vals


def _env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return float(raw)


def _arm_state(state: Any, side: str, field: str) -> list[float] | float:
    attr = f"{side}_{field}"
    if hasattr(state, attr):
        value = getattr(state, attr)
    else:
        value = state.arms[side][field] if isinstance(state, dict) else getattr(state.arms[side], field)
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [float(x) for x in value.tolist()]
    except Exception:
        pass
    return float(value)


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _unit_xy3(values: list[float]) -> list[float]:
    x = float(values[0])
    y = float(values[1])
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-6:
        raise RuntimeError("Pull axis must have nonzero XY length")
    return [x / norm, y / norm, 0.0]


def _axis_point(base: list[float], axis: list[float], scale: float) -> list[float]:
    return [
        float(base[0]) + float(axis[0]) * float(scale),
        float(base[1]) + float(axis[1]) * float(scale),
        float(base[2]),
    ]


def _vec(a: list[float], b: list[float]) -> list[float]:
    return [float(b[i]) - float(a[i]) for i in range(3)]


def _display_rpy_from_rotation_matrix(rotation_matrix: Any) -> list[float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    euler_xyz = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_euler("xyz", degrees=True)
    display = np.array([euler_xyz[1], -euler_xyz[0], -euler_xyz[2] - 90.0], dtype=np.float64)
    display = (display + 180.0) % 360.0 - 180.0
    return [float(x) for x in display]


def _side_grasp_display_rpy_from_approach(
    approach_dir: list[float],
    wrist_roll_deg: float,
) -> tuple[list[float], dict[str, list[float]]]:
    import numpy as np

    approach = np.asarray(approach_dir, dtype=np.float64)
    norm = float(np.linalg.norm(approach))
    if norm < 1e-6:
        raise RuntimeError(f"Invalid drawer approach direction: {approach_dir!r}")
    z_axis = approach / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(world_up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    if abs(float(wrist_roll_deg)) > 1e-6:
        roll_rad = math.radians(float(wrist_roll_deg))
        cos_t = math.cos(roll_rad)
        sin_t = math.sin(roll_rad)
        x_base = x_axis
        y_base = y_axis
        x_axis = cos_t * x_base + sin_t * y_base
        y_axis = -sin_t * x_base + cos_t * y_base
        x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = y_axis / float(np.linalg.norm(y_axis))
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return (
        _display_rpy_from_rotation_matrix(rotation_matrix),
        {
            "local_x_opening_axis": [round(float(x), 5) for x in x_axis.tolist()],
            "local_y_height_axis": [round(float(x), 5) for x in y_axis.tolist()],
            "local_z_approach_axis": [round(float(x), 5) for x in z_axis.tolist()],
        },
    )


def _initial_manual_result() -> dict[str, Any]:
    return {
        "success": False,
        "reward": 0.0,
        "method": "drawer_manual_probe",
        "physical_motion_executed": False,
        "movement_capable_calls": [],
    }


def _run_manual_probe() -> dict[str, Any]:
    result = _initial_manual_result()

    def _append_step(step: dict[str, Any]) -> None:
        result.setdefault("steps", []).append(step)

    def _move(
        *,
        side: str,
        label: str,
        pos: list[float],
        rpy: list[float],
        execute: bool,
        planner_backend: str,
        planning_speed: float,
        gripper_pos: float | None = 1.0,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            f"{side}_target_pos": [float(x) for x in pos],
            f"{side}_target_rpy": [float(x) for x in rpy],
            "preview_only": not execute,
            "planning_speed": planning_speed,
            "ik_error_threshold": _env_float("OPENFORGE_DRAWER_MANUAL_IK_ERROR_THRESHOLD_M", 0.035),
            "ik_rot_threshold_deg": _env_float("OPENFORGE_DRAWER_MANUAL_IK_ROT_THRESHOLD_DEG", 35.0),
            "ik_xyz_weight": _env_float("OPENFORGE_DRAWER_MANUAL_IK_XYZ_WEIGHT", 1.0),
            "ik_rpy_weight": _env_float("OPENFORGE_DRAWER_MANUAL_IK_RPY_WEIGHT", 0.05),
            "planner_backend": planner_backend,
            "solver_speed": os.environ.get("OPENFORGE_DRAWER_MANUAL_SOLVER_SPEED", "fast").strip(),
        }
        if gripper_pos is not None:
            kwargs[f"{side}_gripper"] = float(gripper_pos)
        started = time.time()
        move_result = _json_safe(freespace_move(**kwargs))
        result["movement_capable_calls"].append("freespace_move")
        step = {
            "label": label,
            "target_pos": [round(float(x), 5) for x in pos],
            "target_rpy": [round(float(x), 3) for x in rpy],
            "target_gripper_pos": None if gripper_pos is None else round(float(gripper_pos), 5),
            "execute": execute,
            "duration_s": round(time.time() - started, 3),
            "result": move_result,
            "state_after": _json_safe(get_robot_state()) if execute else None,
        }
        if execute and isinstance(step.get("state_after"), dict):
            try:
                actual = step["state_after"]["arms"][side]["ee_pos"][:3]
                step["actual_pos_error_m"] = round(_dist([float(x) for x in actual], pos), 5)
            except Exception:
                pass
        _append_step(step)
        status = move_result.get("status") if isinstance(move_result, dict) else None
        if status != "Success":
            raise RuntimeError(f"{label} failed: {move_result}")
        return step

    def _servo_delta(
        *,
        side: str,
        label: str,
        delta_pos: list[float],
        execute: bool,
        gripper_pos: float | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        if execute:
            servo_result = _json_safe(
                servo_ee_delta(
                    side,
                    [float(x) for x in delta_pos],
                    duration_s=_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_DURATION_S", 1.0),
                    steps=int(_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_STEPS", 30.0)),
                    gripper_pos=gripper_pos,
                    max_translation_m=_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_MAX_TRANSLATION_M", 0.035),
                    max_component_m=_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_MAX_COMPONENT_M", 0.030),
                    max_joint_delta_rad=_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_MAX_JOINT_DELTA_RAD", 0.22),
                    max_ik_pos_error_m=_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_MAX_IK_POS_ERROR_M", 0.020),
                    command_hz=_env_float("OPENFORGE_DRAWER_MANUAL_SERVO_COMMAND_HZ", 60.0),
                )
            )
            result["movement_capable_calls"].append("servo_ee_delta")
        else:
            servo_result = {"status": "Skipped", "reason": "preview_only"}
        step = {
            "label": label,
            "delta_pos": [round(float(x), 5) for x in delta_pos],
            "execute": execute,
            "duration_s": round(time.time() - started, 3),
            "result": servo_result,
            "state_after": _json_safe(get_robot_state()) if execute else None,
        }
        _append_step(step)
        status = servo_result.get("status") if isinstance(servo_result, dict) else None
        if execute and status != "Success":
            raise RuntimeError(f"{label} failed: {servo_result}")
        return step

    def _set_gripper_step(side: str, label: str, pos: float, *, execute: bool) -> dict[str, Any]:
        step = {"label": label, "target_pos": float(pos), "execute": execute}
        before = _json_safe(get_robot_state()) if execute else None
        before_pos = None
        try:
            before_pos = float(before["arms"][side]["gripper_pos"]) if isinstance(before, dict) else None
        except Exception:
            before_pos = None
        if execute:
            is_open_command = float(pos) >= 0.95
            method = os.environ.get("OPENFORGE_DRAWER_MANUAL_GRIPPER_METHOD", "set_gripper").strip().lower()
            if method == "servo" and not is_open_command:
                grip_result = _json_safe(
                    servo_ee_delta(
                        side,
                        [0.0, 0.0, 0.0],
                        duration_s=_env_float("OPENFORGE_DRAWER_MANUAL_GRIPPER_SERVO_DURATION_S", 0.8),
                        steps=int(_env_float("OPENFORGE_DRAWER_MANUAL_GRIPPER_SERVO_STEPS", 24.0)),
                        gripper_pos=float(pos),
                        max_translation_m=0.001,
                        max_component_m=0.001,
                        max_joint_delta_rad=_env_float(
                            "OPENFORGE_DRAWER_MANUAL_GRIPPER_SERVO_MAX_JOINT_DELTA_RAD",
                            0.04,
                        ),
                        max_ik_pos_error_m=_env_float(
                            "OPENFORGE_DRAWER_MANUAL_GRIPPER_SERVO_MAX_IK_POS_ERROR_M",
                            0.010,
                        ),
                        command_hz=_env_float("OPENFORGE_DRAWER_MANUAL_GRIPPER_SERVO_COMMAND_HZ", 60.0),
                    )
                )
                result["movement_capable_calls"].append("servo_ee_delta")
            else:
                vel_env = (
                    "OPENFORGE_DRAWER_MANUAL_OPEN_GRIPPER_VEL_LIMIT"
                    if is_open_command
                    else "OPENFORGE_DRAWER_MANUAL_GRIPPER_VEL_LIMIT"
                )
                torque_env = (
                    "OPENFORGE_DRAWER_MANUAL_OPEN_GRIPPER_TORQUE_LIMIT"
                    if is_open_command
                    else "OPENFORGE_DRAWER_MANUAL_GRIPPER_TORQUE_LIMIT"
                )
                grip_result = _json_safe(
                    set_gripper(
                        side,
                        float(pos),
                        vel_limit=_env_float(vel_env, 0.45 if is_open_command else 0.25),
                        torque_limit=_env_float(torque_env, 0.28 if is_open_command else 0.10),
                    )
                )
                result["movement_capable_calls"].append("set_gripper")
            time.sleep(_env_float("OPENFORGE_DRAWER_MANUAL_GRIPPER_SETTLE_S", 0.4))
        else:
            grip_result = {"status": "Skipped", "reason": "preview_only"}
        step["result"] = grip_result
        step["state_after"] = _json_safe(get_robot_state()) if execute else None
        after_pos = None
        try:
            after_pos = float(step["state_after"]["arms"][side]["gripper_pos"]) if isinstance(step["state_after"], dict) else None
        except Exception:
            after_pos = None
        if before_pos is not None:
            step["before_gripper_pos"] = round(before_pos, 5)
        if after_pos is not None:
            step["after_gripper_pos"] = round(after_pos, 5)
        if before_pos is not None and after_pos is not None:
            step["gripper_delta_pos"] = round(before_pos - after_pos, 5)
        _append_step(step)
        return step

    def _maybe_stop_after(label: str) -> None:
        requested = os.environ.get("OPENFORGE_DRAWER_MANUAL_STOP_AFTER", "").strip()
        if requested and requested == label:
            raise _StopAfterStage(label)

    def _validate_close_if_requested() -> None:
        if not _truthy("OPENFORGE_DRAWER_MANUAL_VALIDATE_GRIPPER_CLOSE"):
            return
        min_delta = _env_float("OPENFORGE_DRAWER_MANUAL_MIN_CLOSE_DELTA_POS", 0.05)
        max_after = _env_float("OPENFORGE_DRAWER_MANUAL_MAX_CLOSED_GRIPPER_POS", 0.92)
        close_steps = [
            step
            for step in result.get("steps", [])
            if isinstance(step, dict) and step.get("label") in {"scout_close", "target_close"}
        ]
        if not close_steps:
            raise RuntimeError("No manual close steps recorded for gripper validation")
        first_before = close_steps[0].get("before_gripper_pos")
        final_after = close_steps[-1].get("after_gripper_pos")
        if first_before is None or final_after is None:
            raise RuntimeError("Manual close validation lacks measured gripper positions")
        delta = float(first_before) - float(final_after)
        if delta < min_delta or float(final_after) > max_after:
            raise RuntimeError(
                f"Manual close did not capture handle: delta={delta:.3f}, after={float(final_after):.3f}, "
                f"required delta>={min_delta:.3f} and after<={max_after:.3f}"
            )

    def _main() -> None:
        side = os.environ.get("OPENFORGE_DRAWER_MANUAL_SIDE", "left").strip().lower()
        if side not in {"left", "right"}:
            raise RuntimeError("OPENFORGE_DRAWER_MANUAL_SIDE must be left or right")

        execute = _truthy("OPENFORGE_DRAWER_MANUAL_EXECUTE")
        contact = _truthy("OPENFORGE_DRAWER_MANUAL_CONTACT")
        if execute and not _truthy("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
            raise RuntimeError("OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required for physical manual probe")
        if execute and contact and not _truthy("OPENFORGE_DRAWER_ENABLE_PHYSICAL_CONTACT"):
            raise RuntimeError("OPENFORGE_DRAWER_ENABLE_PHYSICAL_CONTACT=1 is required for contact probe")

        handle = _floats("OPENFORGE_DRAWER_MANUAL_HANDLE_XYZ", 3)
        x_bias = _env_float("OPENFORGE_DRAWER_MANUAL_X_BIAS_M", 0.0)
        y_bias = _env_float("OPENFORGE_DRAWER_MANUAL_Y_BIAS_M", 0.0)
        y_standoff = _env_float("OPENFORGE_DRAWER_MANUAL_Y_STANDOFF_M", 0.040)
        y_contact = _env_float("OPENFORGE_DRAWER_MANUAL_Y_CONTACT_OFFSET_M", 0.006)
        axis_raw = os.environ.get("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", "").strip()
        axis_push_past = _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M", -0.012)
        pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.015)
        push_m = _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_M", 0.015)
        retreat_m = _env_float("OPENFORGE_DRAWER_MANUAL_RETREAT_M", 0.045)
        z_abs = _env_float_optional("OPENFORGE_DRAWER_MANUAL_ABS_Z_M")
        z_from_current = _env_float("OPENFORGE_DRAWER_MANUAL_Z_FROM_CURRENT_M", 0.0)
        max_first_move = _env_float("OPENFORGE_DRAWER_MANUAL_MAX_FIRST_MOVE_M", 0.080)
        planner_backend = os.environ.get("OPENFORGE_DRAWER_MANUAL_PLANNER_BACKEND", "rrtconnect").strip()
        planning_speed = _env_float("OPENFORGE_DRAWER_MANUAL_PLANNING_SPEED", scaled_speed(0.10))
        contact_motion_mode = os.environ.get("OPENFORGE_DRAWER_MANUAL_CONTACT_MOTION_MODE", "planned").strip().lower()
        if contact_motion_mode not in {"planned", "servo"}:
            raise RuntimeError("OPENFORGE_DRAWER_MANUAL_CONTACT_MOTION_MODE must be planned or servo")
        scout_gripper = _env_float("OPENFORGE_DRAWER_MANUAL_SCOUT_GRIPPER", 0.70)
        target_gripper = _env_float("OPENFORGE_DRAWER_MANUAL_TARGET_GRIPPER", 0.52)
        precontact_gripper = _env_float_optional("OPENFORGE_DRAWER_MANUAL_PRECONTACT_GRIPPER")
        advance_gripper = _env_float_optional("OPENFORGE_DRAWER_MANUAL_ADVANCE_GRIPPER")
        contact_gripper = _env_float_optional("OPENFORGE_DRAWER_MANUAL_CONTACT_GRIPPER")
        if advance_gripper is None:
            advance_gripper = precontact_gripper if precontact_gripper is not None else 1.0
        if contact_gripper is None:
            contact_gripper = target_gripper
        skip_close = _truthy("OPENFORGE_DRAWER_MANUAL_SKIP_CLOSE")
        skip_open_after = _truthy("OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER")
        open_after_retreat = _truthy("OPENFORGE_DRAWER_MANUAL_OPEN_AFTER_RETREAT")
        retreat_gripper = _env_float_optional("OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER")
        if retreat_gripper is None:
            retreat_gripper = contact_gripper if skip_open_after else 1.0
        post_contact_clearance = [
            _env_float("OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_X_M", 0.0),
            _env_float("OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_Y_M", 0.0),
            _env_float("OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_Z_M", 0.0),
        ]
        post_contact_clear_gripper = _env_float_optional("OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_GRIPPER")
        if post_contact_clear_gripper is None:
            post_contact_clear_gripper = contact_gripper

        state = get_robot_state()
        current_pos = [float(x) for x in _arm_state(state, side, "ee_pos")[:3]]  # type: ignore[index]
        current_rpy = [float(x) for x in _arm_state(state, side, "ee_rpy")[:3]]  # type: ignore[index]
        if os.environ.get("OPENFORGE_DRAWER_MANUAL_RPY"):
            current_rpy = _floats("OPENFORGE_DRAWER_MANUAL_RPY", 3)
        axis_info: dict[str, Any] | None = None
        if axis_raw and _truthy("OPENFORGE_DRAWER_MANUAL_RPY_FROM_AXIS"):
            pull_axis_for_rpy = _unit_xy3(_floats("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", 2))
            approach_dir = [-pull_axis_for_rpy[0], -pull_axis_for_rpy[1], 0.0]
            current_rpy, axes = _side_grasp_display_rpy_from_approach(
                approach_dir,
                _env_float("OPENFORGE_DRAWER_MANUAL_WRIST_ROLL_DEG", 0.0),
            )
            axis_info = {
                "pull_axis_world": [round(float(x), 5) for x in pull_axis_for_rpy],
                "approach_dir_world": [round(float(x), 5) for x in approach_dir],
                "gripper_local_axes_world": axes,
                "rpy_source": "OPENFORGE_DRAWER_MANUAL_RPY_FROM_AXIS",
            }

        z = z_abs if z_abs is not None else current_pos[2] + z_from_current
        base = [handle[0] + x_bias, handle[1] + y_bias, z]
        pull_axis: list[float] | None = None
        if axis_raw:
            pull_axis = _unit_xy3(_floats("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", 2))
            precontact = _axis_point(base, pull_axis, y_standoff)
            contact_pos = _axis_point(base, pull_axis, -axis_push_past)
            pull_pos = _axis_point(contact_pos, pull_axis, pull_m)
            push_pos = _axis_point(pull_pos, pull_axis, -push_m)
            retreat_pos = _axis_point(base, pull_axis, retreat_m)
            if push_m > 0.0:
                close_axis = [-pull_axis[0], -pull_axis[1], 0.0]
                close_from_base = sum((float(push_pos[i]) - float(base[i])) * close_axis[i] for i in range(3))
                max_close_from_base = _env_float("OPENFORGE_DRAWER_MANUAL_MAX_CLOSE_FROM_BASE_M", 0.140)
                if close_from_base > max_close_from_base:
                    raise RuntimeError(
                        "Refusing manual probe: push target is "
                        f"{close_from_base:.3f}m past handle base along close axis, exceeding "
                        f"OPENFORGE_DRAWER_MANUAL_MAX_CLOSE_FROM_BASE_M={max_close_from_base:.3f}m"
                    )
            if axis_info is None:
                axis_info = {
                    "pull_axis_world": [round(float(x), 5) for x in pull_axis],
                    "rpy_source": "current_or_env_rpy",
                }
        else:
            precontact = [handle[0] + x_bias, handle[1] + y_standoff, z]
            contact_pos = [handle[0] + x_bias, handle[1] + y_contact, z]
            pull_pos = [contact_pos[0], contact_pos[1] + pull_m, z]
            push_pos = [pull_pos[0], pull_pos[1] - push_m, z]
            retreat_pos = [handle[0] + x_bias, handle[1] + retreat_m, z]

        first_move = _dist(current_pos, precontact)
        if first_move > max_first_move:
            raise RuntimeError(
                f"Refusing manual probe: first move {first_move:.3f}m exceeds "
                f"OPENFORGE_DRAWER_MANUAL_MAX_FIRST_MOVE_M={max_first_move:.3f}m"
            )

        result.update(
            {
                "physical_motion_executed": execute,
                "contact_enabled": contact,
                "side": side,
                "handle_xyz": [round(float(x), 5) for x in handle],
                "initial_state": _json_safe(state),
                "current_ee_pos": [round(float(x), 5) for x in current_pos],
                "current_ee_rpy": [round(float(x), 3) for x in current_rpy],
                "targets": {
                    "precontact": [round(float(x), 5) for x in precontact],
                    "contact": [round(float(x), 5) for x in contact_pos],
                    "pull": [round(float(x), 5) for x in pull_pos],
                    "push": [round(float(x), 5) for x in push_pos],
                    "retreat": [round(float(x), 5) for x in retreat_pos],
                },
                "config": {
                    "x_bias_m": x_bias,
                    "y_bias_m": y_bias,
                    "y_standoff_m": y_standoff,
                    "y_contact_offset_m": y_contact,
                    "pull_axis_xy": [round(float(x), 5) for x in pull_axis[:2]] if pull_axis else None,
                    "axis_push_past_m": axis_push_past if pull_axis else None,
                    "axis_info": axis_info,
                    "pull_m": pull_m,
                    "push_m": push_m,
                    "retreat_m": retreat_m,
                    "z_from_current_m": z_from_current,
                    "abs_z_m": z_abs,
                    "planner_backend": planner_backend,
                    "planning_speed": planning_speed,
                    "contact_motion_mode": contact_motion_mode,
                    "scout_gripper": scout_gripper,
                    "target_gripper": target_gripper,
                    "precontact_gripper": precontact_gripper,
                    "advance_gripper": advance_gripper,
                    "contact_gripper": contact_gripper,
                    "skip_close": skip_close,
                    "skip_open_after": skip_open_after,
                    "open_after_retreat": open_after_retreat,
                    "retreat_gripper": retreat_gripper,
                    "post_contact_clearance_m": [round(float(x), 5) for x in post_contact_clearance],
                    "post_contact_clear_gripper": post_contact_clear_gripper,
                },
            }
        )

        if _truthy("OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_BEFORE"):
            _append_step(
                {
                    "label": "open_before_probe",
                    "target_pos": 1.0,
                    "execute": False,
                    "result": {"status": "Skipped", "reason": "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_BEFORE=1"},
                    "state_after": _json_safe(get_robot_state()) if execute else None,
                }
            )
        else:
            _set_gripper_step(side, "open_before_probe", 1.0, execute=execute)
        _maybe_stop_after("open_before_probe")

        if _truthy("OPENFORGE_DRAWER_MANUAL_SKIP_ALIGN"):
            _append_step(
                {
                    "label": "align_precontact",
                    "target_pos": [round(float(x), 5) for x in precontact],
                    "target_rpy": [round(float(x), 3) for x in current_rpy],
                    "execute": False,
                    "result": {"status": "Skipped", "reason": "OPENFORGE_DRAWER_MANUAL_SKIP_ALIGN=1"},
                    "state_after": _json_safe(get_robot_state()) if execute else None,
                }
            )
        else:
            _move(
                side=side,
                label="align_precontact",
                pos=precontact,
                rpy=current_rpy,
                execute=execute,
                planner_backend=planner_backend,
                planning_speed=planning_speed,
                gripper_pos=1.0,
            )
        _maybe_stop_after("align_precontact")

        if precontact_gripper is not None:
            _set_gripper_step(side, "precontact_gripper", precontact_gripper, execute=execute)
            _maybe_stop_after("precontact_gripper")

        if contact_motion_mode == "servo":
            _servo_delta(
                side=side,
                label="advance_to_contact_line",
                delta_pos=_vec(precontact, contact_pos),
                execute=execute and contact,
                gripper_pos=advance_gripper,
            )
        else:
            _move(
                side=side,
                label="advance_to_contact_line",
                pos=contact_pos,
                rpy=current_rpy,
                execute=execute and contact,
                planner_backend=planner_backend,
                planning_speed=planning_speed,
                gripper_pos=advance_gripper,
            )
        _maybe_stop_after("advance_to_contact_line")

        if skip_close:
            for label, target in (("scout_close", scout_gripper), ("target_close", target_gripper)):
                _append_step(
                    {
                        "label": label,
                        "target_pos": target,
                        "execute": False,
                        "result": {"status": "Skipped", "reason": "OPENFORGE_DRAWER_MANUAL_SKIP_CLOSE=1"},
                        "state_after": _json_safe(get_robot_state()) if execute else None,
                    }
                )
        else:
            _set_gripper_step(side, "scout_close", scout_gripper, execute=execute and contact)
            _maybe_stop_after("scout_close")
            _set_gripper_step(side, "target_close", target_gripper, execute=execute and contact)
            _validate_close_if_requested()
        _maybe_stop_after("target_close")

        if contact_motion_mode == "servo":
            _servo_delta(
                side=side,
                label="micro_pull",
                delta_pos=_vec(contact_pos, pull_pos),
                execute=execute and contact,
                gripper_pos=contact_gripper,
            )
        else:
            _move(
                side=side,
                label="micro_pull",
                pos=pull_pos,
                rpy=current_rpy,
                execute=execute and contact,
                planner_backend=planner_backend,
                planning_speed=planning_speed,
                gripper_pos=contact_gripper,
            )
        _maybe_stop_after("micro_pull")

        if contact_motion_mode == "servo":
            _servo_delta(
                side=side,
                label="micro_push",
                delta_pos=_vec(pull_pos, push_pos),
                execute=execute and contact,
                gripper_pos=contact_gripper,
            )
        else:
            _move(
                side=side,
                label="micro_push",
                pos=push_pos,
                rpy=current_rpy,
                execute=execute and contact,
                planner_backend=planner_backend,
                planning_speed=planning_speed,
                gripper_pos=contact_gripper,
            )
        _maybe_stop_after("micro_push")

        if any(abs(float(x)) > 1e-6 for x in post_contact_clearance):
            clear_pos = [
                float(push_pos[0]) + float(post_contact_clearance[0]),
                float(push_pos[1]) + float(post_contact_clearance[1]),
                float(push_pos[2]) + float(post_contact_clearance[2]),
            ]
            _move(
                side=side,
                label="post_contact_clear",
                pos=clear_pos,
                rpy=current_rpy,
                execute=execute and contact,
                planner_backend=planner_backend,
                planning_speed=planning_speed,
                gripper_pos=post_contact_clear_gripper,
            )
            _maybe_stop_after("post_contact_clear")

        if skip_open_after:
            _append_step(
                {
                    "label": "open_after_probe",
                    "target_pos": 1.0,
                    "execute": False,
                    "result": {"status": "Skipped", "reason": "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER=1"},
                    "state_after": _json_safe(get_robot_state()) if execute else None,
                }
            )
        else:
            _set_gripper_step(side, "open_after_probe", 1.0, execute=execute)
        _maybe_stop_after("open_after_probe")

        _move(
            side=side,
            label="retreat",
            pos=retreat_pos,
            rpy=current_rpy,
            execute=execute,
            planner_backend=planner_backend,
            planning_speed=planning_speed,
            gripper_pos=retreat_gripper,
        )
        _maybe_stop_after("retreat")
        if open_after_retreat:
            _set_gripper_step(side, "open_after_retreat", 1.0, execute=execute)
            _maybe_stop_after("open_after_retreat")

        if execute:
            result["final_state"] = _json_safe(get_robot_state())
        result["success"] = True
        result["reward"] = 1.0
        write_stage_summary(stage="drawer_manual_probe", result=result, log_dir=current_run_dir())
        print(json.dumps(_json_safe(result), indent=2))

    try:
        _main()
    except _StopAfterStage as exc:
        label = str(exc)
        result["success"] = True
        result["reward"] = 1.0
        result["why_stopped"] = f"Stopped after requested manual stage: {label}"
        if result.get("physical_motion_executed"):
            try:
                result["final_state"] = _json_safe(get_robot_state())
            except Exception:
                pass
        write_stage_summary(stage=f"drawer_manual_probe_stopped_after_{label}", result=result, log_dir=current_run_dir())
        print(json.dumps(_json_safe(result), indent=2))
    except Exception as exc:
        result["success"] = False
        result["reward"] = 0.0
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        write_stage_summary(stage="drawer_manual_probe_failed", result=result, log_dir=current_run_dir())
        print(json.dumps(_json_safe(result), indent=2))
    return dict(result)


def _write_combined_result() -> None:
    run_dir = current_run_dir()
    path = run_dir / "drawer_observe_pull_open_x_result.json"
    path.write_text(json.dumps(_json_safe(TASK_RESULT), indent=2) + "\n", encoding="utf-8")
    TASK_RESULT.setdefault("artifacts", {})["combined_result"] = str(path)


def _run() -> None:
    started = time.time()
    cycle_mode = _cycle_mode()
    TASK_RESULT.update(
        {
            "success": False,
            "reward": 0.0,
            "physical_motion_executed": False,
            "movement_capable_calls": [],
            "pre_observe": None,
            "manual_probe": None,
            "post_observe": None,
            "artifacts": {},
            "config": {
                "cycle_mode": cycle_mode,
                "post_observe": _truthy("OPENFORGE_DRAWER_POST_OBSERVE", "1"),
                "manual_execute": _truthy("OPENFORGE_DRAWER_MANUAL_EXECUTE", "1"),
                "manual_side": os.environ.get("OPENFORGE_DRAWER_MANUAL_SIDE", "left"),
                "manual_pull_axis_xy": os.environ.get("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", "-1,0"),
            },
        }
    )
    print("[drawer_observe_pull_open_x] pre_observe")
    pre = _observe("pre_observe")
    TASK_RESULT["pre_observe"] = pre
    if pre.get("packet_path"):
        TASK_RESULT["artifacts"]["pre_observe_packet"] = pre.get("packet_path")

    if not pre.get("success"):
        TASK_RESULT["why_stopped"] = "pre_observe did not produce a high-confidence drawer handle; no motion was run."
        return

    handle_xyz, fusion = _fused_handle_xyz(pre)
    TASK_RESULT["raw_handle_xyz"] = fusion.get("raw_handle_xyz")
    TASK_RESULT["handle_fusion"] = fusion
    TASK_RESULT["handle_xyz"] = [round(float(x), 5) for x in handle_xyz]
    _set_manual_defaults(handle_xyz, cycle_mode=cycle_mode)
    adaptive_push = _adapt_close_push_distance(cycle_mode=cycle_mode, fusion=fusion)
    if adaptive_push is not None:
        TASK_RESULT["adaptive_close_push"] = adaptive_push
    TASK_RESULT["config"].update(
        {
            "manual_pull_m": os.environ.get("OPENFORGE_DRAWER_MANUAL_PULL_M"),
            "manual_push_m": os.environ.get("OPENFORGE_DRAWER_MANUAL_PUSH_M"),
            "manual_skip_open_after": os.environ.get("OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER"),
            "manual_retreat_gripper": os.environ.get("OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER"),
        }
    )
    print(f"[drawer_observe_pull_open_x] handle_xyz={_format_xyz(handle_xyz)}")
    print(f"[drawer_observe_pull_open_x] handle_fusion={json.dumps(_json_safe(fusion), sort_keys=True)}")

    if _truthy("OPENFORGE_DRAWER_COMBINED_SKIP_MANUAL", "0"):
        TASK_RESULT["why_stopped"] = "OPENFORGE_DRAWER_COMBINED_SKIP_MANUAL=1; manual probe skipped after observe."
        return

    print("[drawer_observe_pull_open_x] manual_probe")
    manual = _run_manual_probe()
    TASK_RESULT["manual_probe"] = manual
    TASK_RESULT["physical_motion_executed"] = bool(manual.get("physical_motion_executed"))
    TASK_RESULT["movement_capable_calls"] = list(manual.get("movement_capable_calls") or [])

    if _truthy("OPENFORGE_DRAWER_POST_OBSERVE", "1"):
        print("[drawer_observe_pull_open_x] post_observe")
        post = _observe("post_observe")
        TASK_RESULT["post_observe"] = post
        if post.get("packet_path"):
            TASK_RESULT["artifacts"]["post_observe_packet"] = post.get("packet_path")

    manual_success = bool(manual.get("success"))
    TASK_RESULT["success"] = manual_success
    TASK_RESULT["reward"] = 1.0 if manual_success else 0.0
    TASK_RESULT["why_stopped"] = (
        "combined observe/manual/post-observe drawer run completed"
        if manual_success
        else f"manual probe failed: {manual.get('why_stopped') or 'see manual_probe details'}"
    )
    TASK_RESULT["duration_s"] = round(time.time() - started, 3)


def run_combined(*, reset: bool = True) -> dict[str, Any]:
    if reset:
        reset_task_result()
    try:
        _run()
    except Exception as exc:
        TASK_RESULT["success"] = False
        TASK_RESULT["reward"] = 0.0
        TASK_RESULT["why_stopped"] = f"{type(exc).__name__}: {exc}"
    finally:
        _write_combined_result()
        write_stage_summary(stage="drawer_observe_pull_open_x", result=TASK_RESULT, log_dir=current_run_dir())
        print(json.dumps(_json_safe(TASK_RESULT), indent=2))
    return dict(TASK_RESULT)


if globals().get("__name__", "__main__") == "__main__":
    run_combined()
