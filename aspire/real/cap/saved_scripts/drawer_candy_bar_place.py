"""Open the top drawer, place a yellow candy bar inside, and close it.

This composes the validated drawer pull/push primitive with a focused
top-down pinch for a small yellow wrapped bar.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from cap.saved_scripts.full_demo_speed import scaled_speed, scaled_speed_str


TASK_NAME = "drawer_candy_bar_place"


def _setdefault_env(values: dict[str, str]) -> None:
    for key, value in values.items():
        os.environ.setdefault(key, value)


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
            "OPENFORGE_DRAWER_CANDY_EXECUTE": "1",
        }
    )


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
        "script_file=cap/saved_scripts/drawer_candy_bar_place.py",
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
    if forge_env.exists() and os.environ.get("OPENFORGE_DRAWER_CANDY_SKIP_FORGE_ENV_SOURCE", "0") != "1":
        command = "source .forge_env && exec " + " ".join(shlex.quote(part) for part in argv)
        os.execvp("bash", ["bash", "-lc", command])
    os.execvp(argv[0], argv)


if globals().get("__name__", "__main__") == "__main__" and not _running_under_run_script():
    _launch_run_script()


from cap.saved_scripts import drawer_observe_pull_open_x as drawer
from cap.saved_scripts.yam_runtime import append_stage_summary, capture_scene, current_run_dir, json_safe, write_json
from skill_library.namespace import (
    detect_objects_oneshot,
    freespace_move,
    get_camera_image,
    get_robot_state,
    run_in_background,
    servo_ee_delta,
    set_gripper,
)


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
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


def _csv_env(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, default).split(",") if part.strip()]


def _float_list(name: str, default: list[float], *, count: int | None = None) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        values = list(default)
    else:
        values = []
        for part in raw.replace(";", ",").split(","):
            try:
                values.append(float(part.strip()))
            except ValueError:
                pass
        if not values:
            values = list(default)
    if count is not None and len(values) != count:
        return list(default)
    return values


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        rows = list(values)
        if limit is not None:
            rows = rows[:limit]
        return [round(float(value), digits) for value in rows]
    except Exception:
        return []


def _xyz(value: Any) -> list[float] | None:
    rows = _round_list(value, limit=3)
    return rows if len(rows) == 3 else None


def _unit_xy(value: Any, default: tuple[float, float]) -> list[float]:
    try:
        x = float(value[0])
        y = float(value[1])
        norm = math.hypot(x, y)
        if norm > 1e-6:
            return [x / norm, y / norm]
    except Exception:
        pass
    return [float(default[0]), float(default[1])]


def _dist3(a: Any, b: Any) -> float:
    av = [float(v) for v in list(a)[:3]]
    bv = [float(v) for v in list(b)[:3]]
    return math.sqrt(sum((av[i] - bv[i]) ** 2 for i in range(3)))


def _angle_norm_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterator[None]:
    old: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _format_xyz(xyz: list[float]) -> str:
    return ",".join(f"{float(value):.5f}" for value in xyz)


def _capture(stage: str, run_dir: Path) -> dict[str, Any]:
    return capture_scene(
        prompts=_csv_env(
            "OPENFORGE_DRAWER_CANDY_PROMPTS",
            "yellow candy bar,yellow candy wrapper,yellow wrapped candy,yellow bar,small yellow bar,candy bar,wrapped bar,bar,drawer handle,drawer",
        ),
        cameras=_csv_env("OPENFORGE_DRAWER_CANDY_CAMERAS", "top,left,right,bottom"),
        detect_objects_oneshot=detect_objects_oneshot,
        get_camera_image=get_camera_image,
        get_robot_state=get_robot_state,
        run_in_background=run_in_background,
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DRAWER_CANDY_OBS_TIMEOUT_S", 30.0),
        max_retries=_env_int("OPENFORGE_DRAWER_CANDY_OBS_RETRIES", 1),
        image_only_cameras=tuple(_csv_env("OPENFORGE_DRAWER_CANDY_IMAGE_ONLY_CAMERAS", "left,right,bottom")),
    )


def _top_detections(scene: dict[str, Any]) -> list[dict[str, Any]]:
    top = (scene.get("cameras") or {}).get("top") or {}
    rows = top.get("serialized_detections") or []
    return [row for row in rows if isinstance(row, dict)]


def _best_detection(scene: dict[str, Any], prompts: list[str]) -> dict[str, Any] | None:
    prompt_rank = {prompt.lower(): idx for idx, prompt in enumerate(prompts)}
    rows: list[dict[str, Any]] = []
    for det in _top_detections(scene):
        prompt = str(det.get("prompt") or det.get("label") or "").lower()
        if prompt not in prompt_rank:
            continue
        xyz = _xyz(det.get("position_3d") or det.get("position"))
        if xyz is None:
            continue
        item = dict(det)
        item["_rank"] = prompt_rank[prompt]
        rows.append(item)
    if not rows:
        return None
    rows.sort(key=lambda det: (int(det.get("_rank", 99)), -float(det.get("score") or 0.0)))
    selected = rows[0]
    selected.pop("_rank", None)
    return selected


def _best_candy(scene: dict[str, Any]) -> dict[str, Any] | None:
    return _best_detection(
        scene,
        _csv_env(
            "OPENFORGE_DRAWER_CANDY_OBJECT_PROMPTS",
            "yellow candy bar,yellow candy wrapper,yellow wrapped candy,yellow bar,small yellow bar,candy bar,wrapped bar,bar",
        ),
    )


def _axis_xy_from_bbox(det: dict[str, Any]) -> list[float] | None:
    if not _truthy_env("OPENFORGE_DRAWER_CANDY_USE_BBOX_AXIS", False):
        return None
    box = det.get("box_2d") or []
    if not isinstance(box, list) or len(box) < 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width < 3.0 or height < 3.0:
        return None
    # Approximate top-view image axes. Leave disabled by default because the
    # configured world-axis override is more reliable than an axis-aligned box.
    return [1.0, 0.0] if width >= height else [0.0, 1.0]


def _build_candy_plan(candy: dict[str, Any], place_pose: dict[str, Any]) -> dict[str, Any]:
    center = _xyz(candy.get("position_3d") or candy.get("position"))
    if center is None:
        raise RuntimeError("selected candy detection has no 3D position")
    axis = _axis_xy_from_bbox(candy)
    if axis is None:
        axis = _unit_xy(_float_list("OPENFORGE_DRAWER_CANDY_BAR_AXIS_XY", [0.0, 1.0], count=2), (0.0, 1.0))
    else:
        axis = _unit_xy(axis, (0.0, 1.0))
    length = _env_float("OPENFORGE_DRAWER_CANDY_BAR_LENGTH_M", 0.120)
    fraction = _env_float("OPENFORGE_DRAWER_CANDY_PICK_FRACTION_ALONG_AXIS", 0.0)
    fraction = max(-0.45, min(0.45, fraction))
    pick_xy = [
        center[0] + axis[0] * length * fraction,
        center[1] + axis[1] * length * fraction,
    ]
    grasp_z = center[2] + _env_float("OPENFORGE_DRAWER_CANDY_TOP_DOWN_Z_OFFSET_M", 0.074)
    yaw = _angle_norm_deg(math.degrees(math.atan2(axis[1], axis[0])) + _env_float("OPENFORGE_DRAWER_CANDY_YAW_OFFSET_DEG", 0.0))
    rpy = [0.0, 180.0, yaw]
    approach_z = _env_float("OPENFORGE_DRAWER_CANDY_APPROACH_Z_LIFT_M", 0.080)
    lift_z = _env_float("OPENFORGE_DRAWER_CANDY_LIFT_Z_M", 0.110)
    place_approach_z = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_APPROACH_Z_M", place_pose["position"][2] + 0.080)
    return {
        "schema": "openforge.drawer_candy_bar.plan.v1",
        "success": True,
        "side": os.environ.get("OPENFORGE_DRAWER_CANDY_SIDE", "left").strip().lower(),
        "candy_detection": candy,
        "candy_center_xyz": _round_list(center, limit=3),
        "bar_axis_xy": _round_list(axis, digits=5, limit=2),
        "bar_length_m": round(float(length), 5),
        "pick_fraction_along_axis": round(float(fraction), 5),
        "pick_pose": {"position": [pick_xy[0], pick_xy[1], grasp_z], "rpy": rpy},
        "approach_pose": {"position": [pick_xy[0], pick_xy[1], grasp_z + approach_z], "rpy": rpy},
        "lift_pose": {"position": [pick_xy[0], pick_xy[1], grasp_z + lift_z], "rpy": rpy},
        "place_approach_pose": {
            "position": [place_pose["position"][0], place_pose["position"][1], place_approach_z],
            "rpy": rpy,
        },
        "place_pose": {"position": list(place_pose["position"]), "rpy": rpy},
    }


def _preview_summary(raw: Any, *, strict: bool = True) -> dict[str, Any]:
    safe = json_safe(raw)
    if isinstance(safe, dict) and isinstance(safe.get("data"), dict):
        safe = safe["data"]
    status = str(_field(safe, "status", "") or "")
    pos_error = _field(safe, "final_pos_error_m", None)
    ik_error = _field(safe, "ik_error_m", None)
    rot_error = _field(safe, "final_rot_error_deg", None)
    ok = status.lower() in {"success", "succeeded", "ok"}
    reason = _field(safe, "reason", _field(safe, "error", None))
    if ok and strict:
        max_pos_error = _env_float("OPENFORGE_DRAWER_CANDY_MAX_PREVIEW_POS_ERROR_M", 0.035)
        max_ik_error = _env_float("OPENFORGE_DRAWER_CANDY_MAX_PREVIEW_IK_ERROR_M", 0.035)
        max_rot_error = _env_float("OPENFORGE_DRAWER_CANDY_MAX_PREVIEW_ROT_ERROR_DEG", 65.0)
        try:
            if pos_error is not None and float(pos_error) > max_pos_error:
                ok = False
                reason = f"final_pos_error_m={float(pos_error):.4f} exceeds {max_pos_error:.4f}"
        except Exception:
            pass
        try:
            if ik_error is not None and float(ik_error) > max_ik_error:
                ok = False
                reason = f"ik_error_m={float(ik_error):.4f} exceeds {max_ik_error:.4f}"
        except Exception:
            pass
        try:
            if rot_error is not None and float(rot_error) > max_rot_error:
                ok = False
                reason = f"final_rot_error_deg={float(rot_error):.2f} exceeds {max_rot_error:.2f}"
        except Exception:
            pass
    return {
        "status": status or None,
        "reason": reason,
        "executed": bool(_field(safe, "executed", False)),
        "trajectory_cache_key": _field(safe, "trajectory_cache_key", None),
        "final_pos_error_m": pos_error,
        "ik_error_m": ik_error,
        "final_rot_error_deg": rot_error,
        "strict": bool(strict),
        "success": bool(ok and not bool(_field(safe, "executed", False))),
        "raw": safe,
    }


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None, preview_only: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": bool(preview_only),
        "planner_backend": os.environ.get("OPENFORGE_DRAWER_CANDY_PLANNER_BACKEND", "rrtconnect"),
        "solver_speed": os.environ.get("OPENFORGE_DRAWER_CANDY_SOLVER_SPEED", "fast"),
        "planning_speed": _env_float("OPENFORGE_DRAWER_CANDY_PLANNING_SPEED", scaled_speed(0.25)),
        "ik_error_threshold": _env_float("OPENFORGE_DRAWER_CANDY_IK_ERROR_THRESHOLD_M", 0.030),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_DRAWER_CANDY_IK_ROT_THRESHOLD_DEG", 25.0),
        "ik_xyz_weight": _env_float("OPENFORGE_DRAWER_CANDY_IK_XYZ_WEIGHT", 1.0),
        "ik_rpy_weight": _env_float("OPENFORGE_DRAWER_CANDY_IK_RPY_WEIGHT", 0.05),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _preview_pose(
    label: str,
    side: str,
    pose: dict[str, Any],
    *,
    gripper: float | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    result = freespace_move(**_move_kwargs(side, pose, gripper=gripper, preview_only=True))
    packet = _preview_summary(result, strict=strict)
    packet["label"] = label
    packet["side"] = side
    packet["target_pose"] = {"position": _round_list(pose["position"], limit=3), "rpy": _round_list(pose["rpy"], digits=3, limit=3)}
    return packet


def _execute_pose(label: str, side: str, pose: dict[str, Any], run_dir: Path, *, gripper: float | None = None) -> dict[str, Any]:
    preview = _preview_pose(label, side, pose, gripper=gripper)
    if not preview.get("success") or not preview.get("trajectory_cache_key"):
        packet = {"stage": label, "side": side, "preview": preview, "execute": None}
        packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
        raise RuntimeError(f"{label} preview failed before execution: {preview.get('reason') or preview}")
    executed = freespace_move(trajectory_cache_key=preview["trajectory_cache_key"])
    post_state = json_safe(get_robot_state())
    packet = {
        "stage": label,
        "side": side,
        "preview": preview,
        "execute": json_safe(executed),
        "post_state": post_state,
    }
    try:
        actual = post_state["arms"][side]["ee_pos"][:3]
        packet["actual_pos_error_m"] = round(_dist3(actual, pose["position"]), 5)
    except Exception:
        packet["actual_pos_error_m"] = None
    packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
    return packet


def _set_gripper_checked(side: str, target: float, label: str, run_dir: Path) -> dict[str, Any]:
    before = json_safe(get_robot_state())
    is_open_command = float(target) >= 0.95
    method = os.environ.get("OPENFORGE_DRAWER_CANDY_GRIPPER_METHOD", "servo").strip().lower()
    if method == "servo" and not is_open_command:
        command = servo_ee_delta(
            side,
            [0.0, 0.0, 0.0],
            duration_s=_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_SERVO_DURATION_S", 0.8),
            steps=_env_int("OPENFORGE_DRAWER_CANDY_GRIPPER_SERVO_STEPS", 24),
            gripper_pos=float(target),
            max_translation_m=0.001,
            max_component_m=0.001,
            max_joint_delta_rad=_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_SERVO_MAX_JOINT_DELTA_RAD", 0.04),
            max_ik_pos_error_m=_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_SERVO_MAX_IK_POS_ERROR_M", 0.010),
            command_hz=_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_SERVO_COMMAND_HZ", 60.0),
        )
    else:
        vel_default = 0.45 if is_open_command else 0.25
        torque_default = 0.28 if is_open_command else 0.28
        command = set_gripper(
            side,
            float(target),
            vel_limit=_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_VEL_LIMIT", vel_default),
            torque_limit=_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_TORQUE_LIMIT", torque_default),
        )
    time.sleep(_env_float("OPENFORGE_DRAWER_CANDY_GRIPPER_SETTLE_S", 0.4))
    after = json_safe(get_robot_state())
    before_pos = None
    after_pos = None
    try:
        before_pos = float(before["arms"][side]["gripper_pos"])
        after_pos = float(after["arms"][side]["gripper_pos"])
    except Exception:
        pass
    packet = {
        "stage": label,
        "side": side,
        "target": float(target),
        "method": method,
        "before": before,
        "after": after,
        "command": json_safe(command),
        "before_gripper_pos": None if before_pos is None else round(before_pos, 5),
        "after_gripper_pos": None if after_pos is None else round(after_pos, 5),
        "gripper_delta_pos": None
        if before_pos is None or after_pos is None
        else round(before_pos - after_pos, 5),
        "validated": True,
    }
    if not is_open_command and _truthy_env("OPENFORGE_DRAWER_CANDY_VALIDATE_GRIPPER_CLOSE", True):
        min_delta = _env_float("OPENFORGE_DRAWER_CANDY_MIN_CLOSE_DELTA_POS", 0.08)
        max_after = _env_float("OPENFORGE_DRAWER_CANDY_MAX_CLOSED_GRIPPER_POS", 0.90)
        if before_pos is None or after_pos is None:
            packet["validated"] = False
            packet["validation_reason"] = "missing measured gripper positions"
        else:
            delta = before_pos - after_pos
            if delta < min_delta or after_pos > max_after:
                packet["validated"] = False
                packet["validation_reason"] = (
                    f"close did not move enough: delta={delta:.3f}, after={after_pos:.3f}, "
                    f"required delta>={min_delta:.3f} and after<={max_after:.3f}"
                )
    packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
    if not packet["validated"]:
        raise RuntimeError(packet.get("validation_reason") or f"{label} gripper validation failed")
    return packet


def _open_retreat_m_after_pull() -> float:
    if "OPENFORGE_DRAWER_CANDY_OPEN_RETREAT_M" in os.environ:
        return _env_float("OPENFORGE_DRAWER_CANDY_OPEN_RETREAT_M", 0.267)
    pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
    axis_push_past = _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M", -0.012)
    clear_after_pull = _env_float("OPENFORGE_DRAWER_CANDY_OPEN_CLEAR_AFTER_PULL_M", 0.150)
    return max(0.0, -axis_push_past + pull_m + clear_after_pull)


def _drawer_open(run_dir: Path) -> dict[str, Any]:
    retreat_m = _open_retreat_m_after_pull()
    with _temporary_env(
        {
            "OPENFORGE_DRAWER_CYCLE_MODE": "open_only",
            "OPENFORGE_DRAWER_POST_OBSERVE": "1",
            "OPENFORGE_DRAWER_MANUAL_EXECUTE": "1",
            "OPENFORGE_DRAWER_MANUAL_PUSH_M_OPERATOR_SET": "0",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_M": f"{retreat_m:.5f}",
            "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER": "0",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER": "1.0",
            "OPENFORGE_DRAWER_MANUAL_OPEN_AFTER_RETREAT": "0",
        }
    ):
        result = drawer.run_combined(reset=True)
    write_json(run_dir / "plans" / "drawer_open_result.json", result)
    return result


def _computed_drawer_retreat_pose(handle_xyz: list[float]) -> dict[str, Any]:
    pull_axis = _pull_axis()
    retreat_m = _open_retreat_m_after_pull()
    z_abs = _env_float("OPENFORGE_DRAWER_MANUAL_ABS_Z_M", 0.905)
    rpy = _float_list("OPENFORGE_DRAWER_MANUAL_RPY", [60.0, -90.0, -180.0], count=3)
    return {
        "position": [
            float(handle_xyz[0]) + pull_axis[0] * retreat_m,
            float(handle_xyz[1]) + pull_axis[1] * retreat_m,
            z_abs,
        ],
        "rpy": rpy,
    }


def _drawer_observe_handle(stage: str) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    obs = drawer._observe(stage)  # noqa: SLF001 - saved-script composition of local primitive
    handle_xyz, fusion = drawer._fused_handle_xyz(obs)  # noqa: SLF001
    return handle_xyz, fusion, obs


def _pull_axis() -> list[float]:
    values = _float_list("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", [-1.0, 0.0], count=2)
    return _unit_xy(values, (-1.0, 0.0))


def _place_pose_from_open_drawer(open_result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    pull_axis = _pull_axis()
    pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
    open_handle_xyz: list[float] | None = None
    source = "post_open_observe"
    try:
        post = open_result.get("post_observe")
        if isinstance(post, dict) and post.get("success"):
            open_handle_xyz, fusion = drawer._fused_handle_xyz(post)  # noqa: SLF001
            write_json(run_dir / "plans" / "post_open_handle_fusion.json", fusion)
    except Exception:
        open_handle_xyz = None
    if open_handle_xyz is None:
        closed_handle = _xyz(open_result.get("handle_xyz"))
        if closed_handle is None:
            raise RuntimeError("drawer open result did not include a usable handle pose")
        open_handle_xyz = [closed_handle[0] + pull_axis[0] * pull_m, closed_handle[1] + pull_axis[1] * pull_m, closed_handle[2]]
        source = "closed_handle_plus_configured_pull"
    raw_into = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_INTO_DRAWER_M", 0.110)
    min_into = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_MIN_INTO_DRAWER_M", 0.085)
    max_into = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_MAX_INTO_DRAWER_M", 0.120)
    into = max(min_into, min(max_into, raw_into))
    xy_bias = _float_list("OPENFORGE_DRAWER_CANDY_PLACE_XY_BIAS_M", [0.0, 0.0], count=2)
    max_xy_bias = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_MAX_ABS_XY_BIAS_M", 0.035)
    xy_bias = [
        max(-max_xy_bias, min(max_xy_bias, float(xy_bias[0]))),
        max(-max_xy_bias, min(max_xy_bias, float(xy_bias[1]))),
    ]
    raw_z = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_TCP_Z_M", 0.940)
    min_z = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_MIN_TCP_Z_M", 0.920)
    max_z = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_MAX_TCP_Z_M", 0.970)
    place_z = max(min_z, min(max_z, raw_z))
    target = [
        open_handle_xyz[0] - pull_axis[0] * into + xy_bias[0],
        open_handle_xyz[1] - pull_axis[1] * into + xy_bias[1],
        place_z,
    ]
    packet = {
        "schema": "openforge.drawer_candy_bar.place_pose.v1",
        "source": source,
        "open_handle_xyz": _round_list(open_handle_xyz, limit=3),
        "pull_axis_xy": _round_list(pull_axis, digits=5, limit=2),
        "raw_place_into_drawer_m": round(float(raw_into), 5),
        "place_into_drawer_m": round(float(into), 5),
        "place_into_limits_m": [round(float(min_into), 5), round(float(max_into), 5)],
        "place_xy_bias_m": _round_list(xy_bias, limit=2),
        "place_tcp_z_m": round(float(place_z), 5),
        "place_tcp_z_limits_m": [round(float(min_z), 5), round(float(max_z), 5)],
        "pose": {"position": target, "rpy": [0.0, 180.0, 0.0]},
    }
    packet["path"] = write_json(run_dir / "plans" / "drawer_place_pose.json", packet)
    return packet["pose"]


def _close_drawer(run_dir: Path) -> dict[str, Any]:
    try:
        handle_xyz, fusion, obs = _drawer_observe_handle("pre_close_drawer_observe")
        handle_source = "pre_close_observe"
        write_json(run_dir / "plans" / "pre_close_handle_fusion.json", fusion)
    except Exception as exc:
        raise RuntimeError(f"could not observe open drawer handle for close: {exc}") from exc
    default_push_m = max(
        0.0,
        _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
        - _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_LESS_THAN_PULL_M", 0.010),
    )
    push_m = _env_float("OPENFORGE_DRAWER_CANDY_CLOSE_PUSH_M", default_push_m)
    with _temporary_env(
        {
            "OPENFORGE_DRAWER_MANUAL_SIDE": "left",
            "OPENFORGE_DRAWER_MANUAL_HANDLE_XYZ": _format_xyz(handle_xyz),
            "OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY": os.environ.get("OPENFORGE_DRAWER_MANUAL_PULL_AXIS_XY", "-1,0"),
            "OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M": os.environ.get("OPENFORGE_DRAWER_MANUAL_PUSH_PAST_M", "-0.012"),
            "OPENFORGE_DRAWER_MANUAL_PULL_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_PUSH_M": f"{push_m:.5f}",
            "OPENFORGE_DRAWER_MANUAL_PUSH_M_OPERATOR_SET": "1",
            "OPENFORGE_DRAWER_ADAPT_CLOSE_PUSH_M": "0",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_M": os.environ.get("OPENFORGE_DRAWER_MANUAL_RETREAT_M", "0.085"),
            "OPENFORGE_DRAWER_MANUAL_ABS_Z_M": os.environ.get("OPENFORGE_DRAWER_MANUAL_ABS_Z_M", "0.905"),
            "OPENFORGE_DRAWER_MANUAL_RPY": os.environ.get("OPENFORGE_DRAWER_MANUAL_RPY", "60,-90,-180"),
            "OPENFORGE_DRAWER_MANUAL_CONTACT": "1",
            "OPENFORGE_DRAWER_MANUAL_EXECUTE": "1",
            "OPENFORGE_DRAWER_MANUAL_PLANNING_SPEED": os.environ.get(
                "OPENFORGE_DRAWER_MANUAL_PLANNING_SPEED",
                scaled_speed_str(0.4),
            ),
            "OPENFORGE_DRAWER_MANUAL_PLANNER_BACKEND": os.environ.get("OPENFORGE_DRAWER_MANUAL_PLANNER_BACKEND", "rrtconnect"),
            "OPENFORGE_DRAWER_MANUAL_GRIPPER_METHOD": os.environ.get("OPENFORGE_DRAWER_MANUAL_GRIPPER_METHOD", "servo"),
            "OPENFORGE_DRAWER_MANUAL_PRECONTACT_GRIPPER": os.environ.get("OPENFORGE_DRAWER_MANUAL_PRECONTACT_GRIPPER", "0.78"),
            "OPENFORGE_DRAWER_MANUAL_ADVANCE_GRIPPER": os.environ.get("OPENFORGE_DRAWER_MANUAL_ADVANCE_GRIPPER", "0.72"),
            "OPENFORGE_DRAWER_MANUAL_SCOUT_GRIPPER": os.environ.get("OPENFORGE_DRAWER_MANUAL_SCOUT_GRIPPER", "0.54"),
            "OPENFORGE_DRAWER_MANUAL_TARGET_GRIPPER": os.environ.get("OPENFORGE_DRAWER_MANUAL_TARGET_GRIPPER", "0.10"),
            "OPENFORGE_DRAWER_MANUAL_CONTACT_GRIPPER": os.environ.get("OPENFORGE_DRAWER_MANUAL_CONTACT_GRIPPER", "0.10"),
            "OPENFORGE_DRAWER_MANUAL_VALIDATE_GRIPPER_CLOSE": "1",
            "OPENFORGE_DRAWER_MANUAL_SKIP_CLOSE": "0",
            "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER": "0",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER": "1.0",
            "OPENFORGE_DRAWER_MANUAL_OPEN_AFTER_RETREAT": "0",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_Z_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_GRIPPER": "1.0",
        }
    ):
        result = drawer._run_manual_probe()  # noqa: SLF001
    result["handle_source"] = handle_source
    result["pre_close_observe"] = obs
    write_json(run_dir / "plans" / "drawer_close_result.json", result)
    return result


def _verify_candy_lift(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    scene = _capture("post_candy_lift_verify", run_dir)
    source_xyz = _xyz(plan.get("candy_center_xyz"))
    candy = _best_candy(scene)
    packet: dict[str, Any] = {
        "schema": "openforge.drawer_candy_bar.lift_verify.v1",
        "success": True,
        "source_candy_xyz": source_xyz,
        "detected_candy": candy,
        "observation": scene,
        "reason": "no yellow candy detected at the original tabletop source after lift",
    }
    detected_xyz = _xyz((candy or {}).get("position_3d") or (candy or {}).get("position"))
    if source_xyz is not None and detected_xyz is not None:
        xy_dist = math.hypot(float(detected_xyz[0]) - float(source_xyz[0]), float(detected_xyz[1]) - float(source_xyz[1]))
        z_delta = float(detected_xyz[2]) - float(source_xyz[2])
        packet["detected_candy_xyz"] = detected_xyz
        packet["source_xy_dist_m"] = round(xy_dist, 5)
        packet["source_z_delta_m"] = round(z_delta, 5)
        max_xy = _env_float("OPENFORGE_DRAWER_CANDY_SOURCE_VERIFY_MAX_XY_M", 0.055)
        max_abs_z = _env_float("OPENFORGE_DRAWER_CANDY_SOURCE_VERIFY_MAX_ABS_Z_M", 0.060)
        if xy_dist <= max_xy and abs(z_delta) <= max_abs_z:
            packet["success"] = False
            packet["reason"] = (
                f"yellow candy still detected at source after lift: xy={xy_dist:.3f}m, "
                f"z_delta={z_delta:.3f}m"
            )
        else:
            packet["reason"] = (
                f"yellow candy detection moved from source after lift: xy={xy_dist:.3f}m, "
                f"z_delta={z_delta:.3f}m"
            )
    packet["path"] = write_json(run_dir / "plans" / "post_candy_lift_verify.json", packet)
    return packet


def _preview_plan(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    side = plan["side"]
    open_pos = _env_float("OPENFORGE_DRAWER_CANDY_OPEN_GRIPPER", 1.0)
    close_pos = _env_float("OPENFORGE_DRAWER_CANDY_CLOSE_GRIPPER", 0.04)
    strict = _truthy_env("OPENFORGE_DRAWER_CANDY_STRICT_ADVISORY_PREVIEW", False)
    rows = []
    for label, pose, gripper in (
        ("candy_approach", plan["approach_pose"], open_pos),
        ("candy_pick", plan["pick_pose"], open_pos),
        ("candy_lift", plan["lift_pose"], close_pos),
        ("drawer_place_approach", plan["place_approach_pose"], close_pos),
        ("drawer_place_release", plan["place_pose"], close_pos),
    ):
        rows.append(_preview_pose(label, side, pose, gripper=gripper, strict=strict))
    packet = {"schema": "openforge.drawer_candy_bar.preview.v1", "success": all(bool(row.get("success")) for row in rows), "previews": rows}
    packet["path"] = write_json(run_dir / "plans" / "drawer_candy_preview.json", packet)
    return packet


def _execute_candy_pick_place(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    side = plan["side"]
    open_pos = _env_float("OPENFORGE_DRAWER_CANDY_OPEN_GRIPPER", 1.0)
    close_pos = _env_float("OPENFORGE_DRAWER_CANDY_CLOSE_GRIPPER", 0.04)
    packet: dict[str, Any] = {
        "schema": "openforge.drawer_candy_bar.pick_place.v1",
        "success": False,
        "side": side,
        "steps": [],
        "movement_capable_calls": [],
        "holding_candy": False,
    }
    try:
        packet["steps"].append(_set_gripper_checked(side, open_pos, "open_before_candy_pick", run_dir))
        packet["movement_capable_calls"].append(f"set_gripper:{side}")
        packet["steps"].append(_execute_pose("candy_approach", side, plan["approach_pose"], run_dir, gripper=open_pos))
        packet["movement_capable_calls"].append("freespace_move:candy_approach")
        packet["steps"].append(_execute_pose("candy_pick", side, plan["pick_pose"], run_dir, gripper=open_pos))
        packet["movement_capable_calls"].append("freespace_move:candy_pick")
        packet["steps"].append(_set_gripper_checked(side, close_pos, "close_on_candy", run_dir))
        packet["movement_capable_calls"].append(f"set_gripper:{side}")
        packet["holding_candy"] = True
        packet["steps"].append(_execute_pose("candy_lift", side, plan["lift_pose"], run_dir, gripper=close_pos))
        packet["movement_capable_calls"].append("freespace_move:candy_lift")
        lift_verify = _verify_candy_lift(plan, run_dir)
        packet["steps"].append({"stage": "post_candy_lift_verify", "ok": bool(lift_verify.get("success")), "result": lift_verify})
        if not lift_verify.get("success"):
            packet["holding_candy"] = False
            raise RuntimeError(lift_verify.get("reason") or "post-lift verification failed")
        packet["steps"].append(_execute_pose("drawer_place_approach", side, plan["place_approach_pose"], run_dir, gripper=close_pos))
        packet["movement_capable_calls"].append("freespace_move:drawer_place_approach")
        packet["steps"].append(_execute_pose("drawer_place_release", side, plan["place_pose"], run_dir, gripper=close_pos))
        packet["movement_capable_calls"].append("freespace_move:drawer_place_release")
        packet["steps"].append(_set_gripper_checked(side, open_pos, "release_candy_in_drawer", run_dir))
        packet["movement_capable_calls"].append(f"set_gripper:{side}")
        packet["holding_candy"] = False
        packet["success"] = True
        packet["why_stopped"] = "candy pick/place motion completed; ready to return to drawer close pose"
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
        packet["final_state"] = json_safe(get_robot_state())
    packet["path"] = write_json(run_dir / "plans" / "drawer_candy_pick_place.json", packet)
    return packet


def _current_arm_rpy(side: str) -> list[float]:
    state = json_safe(get_robot_state())
    try:
        rpy = _round_list(state["arms"][side]["ee_rpy"], digits=3, limit=3)
        if len(rpy) == 3:
            return rpy
    except Exception:
        pass
    return [0.0, 180.0, 90.0]


def _build_held_candy_place_plan(place_pose: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    side = os.environ.get("OPENFORGE_DRAWER_CANDY_SIDE", "left").strip().lower()
    rpy = _float_list("OPENFORGE_DRAWER_CANDY_HELD_RPY", _current_arm_rpy(side), count=3)
    place_approach_z = _env_float("OPENFORGE_DRAWER_CANDY_PLACE_APPROACH_Z_M", place_pose["position"][2] + 0.080)
    plan = {
        "schema": "openforge.drawer_candy_bar.held_place_plan.v1",
        "success": True,
        "side": side,
        "source": "current_left_gripper_holding_candy",
        "place_approach_pose": {
            "position": [place_pose["position"][0], place_pose["position"][1], place_approach_z],
            "rpy": rpy,
        },
        "place_pose": {"position": list(place_pose["position"]), "rpy": rpy},
    }
    plan["path"] = write_json(run_dir / "plans" / "held_candy_place_plan.json", plan)
    return plan


def _execute_held_candy_place(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    side = plan["side"]
    open_pos = _env_float("OPENFORGE_DRAWER_CANDY_OPEN_GRIPPER", 1.0)
    close_pos = _env_float("OPENFORGE_DRAWER_CANDY_CLOSE_GRIPPER", 0.04)
    packet: dict[str, Any] = {
        "schema": "openforge.drawer_candy_bar.held_place.v1",
        "success": False,
        "side": side,
        "steps": [],
        "movement_capable_calls": [],
        "holding_candy": True,
    }
    try:
        packet["steps"].append(_execute_pose("drawer_place_approach", side, plan["place_approach_pose"], run_dir, gripper=close_pos))
        packet["movement_capable_calls"].append("freespace_move:drawer_place_approach")
        packet["steps"].append(_execute_pose("drawer_place_release", side, plan["place_pose"], run_dir, gripper=close_pos))
        packet["movement_capable_calls"].append("freespace_move:drawer_place_release")
        packet["steps"].append(_set_gripper_checked(side, open_pos, "release_candy_in_drawer", run_dir))
        packet["movement_capable_calls"].append(f"set_gripper:{side}")
        packet["holding_candy"] = False
        packet["success"] = True
        packet["why_stopped"] = "held candy placement completed; ready to return to drawer close pose"
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
        packet["final_state"] = json_safe(get_robot_state())
    packet["path"] = write_json(run_dir / "plans" / "held_candy_place.json", packet)
    return packet


def _drawer_open_retreat_pose(open_result: dict[str, Any]) -> dict[str, Any] | None:
    fallback = open_result.get("close_ready_pose") if isinstance(open_result, dict) else None
    if isinstance(fallback, dict) and _xyz(fallback.get("position")) is not None:
        rpy = _round_list(fallback.get("rpy"), digits=3, limit=3)
        if len(rpy) == 3:
            return {"position": _xyz(fallback.get("position")), "rpy": rpy}
    manual = open_result.get("manual_probe") if isinstance(open_result, dict) else None
    if not isinstance(manual, dict):
        return None
    targets = manual.get("targets") if isinstance(manual.get("targets"), dict) else {}
    retreat_pos = _xyz(targets.get("retreat"))
    if retreat_pos is None:
        return None
    retreat_rpy: list[float] | None = None
    for step in manual.get("steps") or []:
        if isinstance(step, dict) and step.get("label") == "retreat":
            rows = _round_list(step.get("target_rpy"), digits=3, limit=3)
            if len(rows) == 3:
                retreat_rpy = rows
            break
    if retreat_rpy is None:
        retreat_rpy = _float_list("OPENFORGE_DRAWER_MANUAL_RPY", [60.0, -90.0, -180.0], count=3)
    return {"position": retreat_pos, "rpy": retreat_rpy}


def _return_to_drawer_close_ready(open_result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    pose = _drawer_open_retreat_pose(open_result)
    if pose is None:
        return {
            "schema": "openforge.drawer_candy_bar.close_ready_return.v1",
            "success": False,
            "reason": "drawer open result did not include a retreat pose",
        }
    side = os.environ.get("OPENFORGE_DRAWER_CANDY_SIDE", "left").strip().lower()
    open_pos = _env_float("OPENFORGE_DRAWER_CANDY_OPEN_GRIPPER", 1.0)
    packet = {
        "schema": "openforge.drawer_candy_bar.close_ready_return.v1",
        "success": False,
        "side": side,
        "source": "drawer_open.manual_probe.retreat",
        "pose": {"position": _round_list(pose["position"], limit=3), "rpy": _round_list(pose["rpy"], digits=3, limit=3)},
    }
    try:
        packet["step"] = _execute_pose("drawer_close_ready_return", side, pose, run_dir, gripper=open_pos)
        packet["success"] = True
    except Exception as exc:
        packet["reason"] = f"{type(exc).__name__}: {exc}"
    packet["path"] = write_json(run_dir / "plans" / "drawer_close_ready_return.json", packet)
    return packet


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    execute = _truthy_env("OPENFORGE_DRAWER_CANDY_EXECUTE", True)
    result: dict[str, Any] = {
        "schema": "openforge.drawer_candy_bar.task.v1",
        "success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "physical_motion_executed": False,
        "execute": execute,
        "drawer_open": None,
        "candy_observation": None,
        "selected_candy_detection": None,
        "place_pose": None,
        "candy_plan": None,
        "preview": None,
        "pick_place": None,
        "close_ready_return": None,
        "drawer_close": None,
        "post_close_observe": None,
        "why_stopped": None,
    }
    if execute and not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False):
        result["why_stopped"] = "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required"
        return result
    drawer_opened = False
    try:
        if execute:
            drawer_already_open = _truthy_env("OPENFORGE_DRAWER_CANDY_DRAWER_ALREADY_OPEN", False) or _truthy_env(
                "OPENFORGE_DRAWER_CANDY_RESUME_HELD", False
            )
            if drawer_already_open:
                handle_xyz, fusion, obs = _drawer_observe_handle("already_open_drawer_observe")
                close_ready_pose = _computed_drawer_retreat_pose(handle_xyz)
                result["drawer_open"] = {
                    "success": True,
                    "reward": 1.0,
                    "method": "already_open_observe",
                    "physical_motion_executed": False,
                    "handle_xyz": handle_xyz,
                    "handle_fusion": fusion,
                    "post_observe": obs,
                    "close_ready_pose": close_ready_pose,
                    "why_stopped": "drawer was already open; skipped open primitive",
                }
                drawer_opened = True
                write_json(run_dir / "plans" / "drawer_open_result.json", result["drawer_open"])
            else:
                result["drawer_open"] = _drawer_open(run_dir)
                drawer_opened = bool((result["drawer_open"] or {}).get("physical_motion_executed"))
                result["physical_motion_executed"] = drawer_opened
                if not bool((result["drawer_open"] or {}).get("success")):
                    result["why_stopped"] = "drawer open failed; stopped before candy pickup"
                    return result
            place_pose = _place_pose_from_open_drawer(result["drawer_open"], run_dir)
            if _truthy_env("OPENFORGE_DRAWER_CANDY_RESUME_HELD", False):
                result["place_pose"] = {"position": _round_list(place_pose["position"], limit=3), "rpy": _round_list(place_pose["rpy"], limit=3)}
                plan = _build_held_candy_place_plan(place_pose, run_dir)
                result["candy_plan"] = plan
                result["selected_candy_detection"] = {"source": "resume_held", "assumption": "left gripper is already holding the yellow candy"}
                held_place = _execute_held_candy_place(plan, run_dir)
                result["pick_place"] = held_place
                result["physical_motion_executed"] = True
                if not held_place.get("success"):
                    result["why_stopped"] = f"held candy placement failed: {held_place.get('why_stopped')}"
                    return result
                close_ready = _return_to_drawer_close_ready(result["drawer_open"], run_dir)
                result["close_ready_return"] = close_ready
                if not close_ready.get("success"):
                    result["why_stopped"] = f"failed to return to drawer close-ready pose: {close_ready.get('reason')}"
                    return result
                close = _close_drawer(run_dir)
                result["drawer_close"] = close
                result["post_close_observe"] = drawer._observe("post_close_drawer_observe")  # noqa: SLF001
                result["success"] = bool(close.get("success"))
                result["reward"] = 1.0 if result["success"] else 0.0
                result["why_stopped"] = (
                    "placed already-held yellow candy bar inside and closed drawer"
                    if result["success"]
                    else "drawer close failed after held candy placement"
                )
                return result
        else:
            handle_xyz, fusion, obs = _drawer_observe_handle("preview_closed_drawer_observe")
            pull_axis = _pull_axis()
            pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
            predicted_open = [handle_xyz[0] + pull_axis[0] * pull_m, handle_xyz[1] + pull_axis[1] * pull_m, handle_xyz[2]]
            fake_open = {"handle_xyz": handle_xyz, "post_observe": None}
            place_pose = _place_pose_from_open_drawer({**fake_open, "handle_xyz": handle_xyz}, run_dir)
            write_json(
                run_dir / "plans" / "preview_drawer_open_estimate.json",
                {"closed_handle": handle_xyz, "fusion": fusion, "observe": obs, "predicted_open_handle": predicted_open},
            )
        result["place_pose"] = {"position": _round_list(place_pose["position"], limit=3), "rpy": _round_list(place_pose["rpy"], limit=3)}

        scene = _capture("post_open_candy_observe" if execute else "preview_candy_observe", run_dir)
        result["candy_observation"] = scene
        candy = _best_candy(scene)
        result["selected_candy_detection"] = candy
        if candy is None:
            result["why_stopped"] = "missing valid top-camera yellow candy bar detection"
            return result
        plan = _build_candy_plan(candy, place_pose)
        result["candy_plan"] = plan
        write_json(run_dir / "plans" / "drawer_candy_plan.json", plan)
        preview = _preview_plan(plan, run_dir)
        result["preview"] = preview
        if not preview.get("success"):
            result["why_stopped"] = "candy pick/place preview failed; no candy motion executed"
            return result
        if not execute:
            result["success"] = True
            result["reward"] = 1.0
            result["why_stopped"] = "no-motion preview completed"
            return result
        pick_place = _execute_candy_pick_place(plan, run_dir)
        result["pick_place"] = pick_place
        result["physical_motion_executed"] = True
        if not pick_place.get("success"):
            result["why_stopped"] = f"candy pick/place failed: {pick_place.get('why_stopped')}"
            return result
        close_ready = _return_to_drawer_close_ready(result["drawer_open"], run_dir)
        result["close_ready_return"] = close_ready
        if not close_ready.get("success"):
            result["why_stopped"] = f"failed to return to drawer close-ready pose: {close_ready.get('reason')}"
            return result
        close = _close_drawer(run_dir)
        result["drawer_close"] = close
        result["post_close_observe"] = drawer._observe("post_close_drawer_observe")  # noqa: SLF001
        result["success"] = bool(close.get("success"))
        result["reward"] = 1.0 if result["success"] else 0.0
        result["why_stopped"] = "opened drawer, placed yellow candy bar inside, and closed drawer" if result["success"] else "drawer close failed after candy placement"
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if execute and drawer_opened and not result.get("drawer_close") and not _truthy_env("OPENFORGE_DRAWER_CANDY_SKIP_RECOVERY_CLOSE", False):
            holding = bool(((result.get("pick_place") or {}).get("holding_candy")))
            if not holding:
                try:
                    result["drawer_close_recovery"] = _close_drawer(run_dir)
                except Exception as close_exc:
                    result["drawer_close_recovery_error"] = f"{type(close_exc).__name__}: {close_exc}"
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## Drawer Candy Bar Place",
                f"- success: {result.get('success')}",
                f"- execute: {result.get('execute')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- selected_candy_detection: {result.get('selected_candy_detection')}",
                f"- place_pose: {result.get('place_pose')}",
                f"- preview_success: {((result.get('preview') or {}).get('success'))}",
                f"- pick_place_success: {((result.get('pick_place') or {}).get('success'))}",
                f"- close_ready_return_success: {((result.get('close_ready_return') or {}).get('success'))}",
                f"- drawer_close_success: {((result.get('drawer_close') or {}).get('success'))}",
                f"- why_stopped: {result.get('why_stopped')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
