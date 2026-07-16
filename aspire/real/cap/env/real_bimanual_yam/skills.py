"""Direct-env tool namespace for real bimanual YAM.

This module is intentionally a thin shim: user-facing tool behavior lives in
``cap.agent.tools``.  ``make_namespace`` instantiates those tools with
``env=env`` so direct script execution talks to the provided environment.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

# Imported by YamDashboard.  Tool implementations own motion behavior; these
# flags remain here for UI compatibility with the direct runner.
_stop_requested = threading.Event()
_pause_requested = threading.Event()


def _sigint_handler(sig, frame):
    if not _stop_requested.is_set():
        print("\n[YAM] Stop requested.")
    _stop_requested.set()


signal.signal(signal.SIGINT, _sigint_handler)


def display_rpy_to_quat(rpy_deg: list[float] | np.ndarray) -> np.ndarray:
    """Convert planner/display RPY degrees to quaternion xyzw."""
    roll, pitch, yaw = [float(x) for x in rpy_deg]
    return Rotation.from_euler(
        "xyz", [-pitch, roll, -yaw - 90.0], degrees=True
    ).as_quat()


def _arm(env: Any, side: str):
    return env._profile.arms[side]


def _sample_keypoints(
    timestamps: np.ndarray,
    values: np.ndarray,
    t_now: float,
) -> np.ndarray:
    if t_now <= float(timestamps[0]):
        return values[0]
    if t_now >= float(timestamps[-1]):
        return values[-1]
    idx = int(np.searchsorted(timestamps, t_now, side="right") - 1)
    idx = max(0, min(idx, len(timestamps) - 2))
    t0 = float(timestamps[idx])
    t1 = float(timestamps[idx + 1])
    alpha = 1.0 if t1 <= t0 + 1e-9 else (float(t_now) - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * values[idx] + alpha * values[idx + 1]


def _command_bimanual_joint7(env: Any, left_cmd: np.ndarray, right_cmd: np.ndarray) -> None:
    left_p = _arm(env, "left")
    right_p = _arm(env, "right")
    env.command_joint_state(
        "left",
        {
            "pos": np.asarray(left_cmd, dtype=np.float64).reshape(7),
            "vel": np.zeros(7),
            "kp": left_p.interp_kp,
            "kd": left_p.interp_kd,
        },
    )
    env.command_joint_state(
        "right",
        {
            "pos": np.asarray(right_cmd, dtype=np.float64).reshape(7),
            "vel": np.zeros(7),
            "kp": right_p.interp_kp,
            "kd": right_p.interp_kd,
        },
    )


def _prepare_joint7_waypoints(
    env: Any,
    name: str,
    joint_positions: list,
    gripper_positions: list | None,
    n: int,
) -> np.ndarray:
    arr = np.asarray(joint_positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 6:
        raise ValueError(
            f"{name}_joint_positions must have shape (N,6) or (N,7); got {arr.shape}"
        )
    joints = arr[:, :6]
    if gripper_positions is None:
        if arr.shape[1] >= 7:
            gripper = arr[:, 6]
        else:
            cur = env._arms[name].get_observations()["gripper_pos"]
            gripper = np.full(n, float(np.asarray(cur).ravel()[0]), dtype=np.float64)
    else:
        gripper = np.asarray(gripper_positions, dtype=np.float64).reshape(n, -1)[:, 0]
    gripper = np.clip(gripper, 0.0, 1.0)
    return np.column_stack([joints, gripper]).astype(np.float64)


def _move_bimanual_joint_keypoints(
    env: Any,
    timestamps: list[float] | np.ndarray,
    left_joint_positions: list | np.ndarray,
    right_joint_positions: list | np.ndarray,
    left_gripper_positions: list | np.ndarray | None = None,
    right_gripper_positions: list | np.ndarray | None = None,
    playback_speed: float = 1.0,
    command_hz: float = 60.0,
    start_interp_s: float = 1.0,
) -> dict[str, Any]:
    """Replay synchronized bimanual joint waypoints directly on RealYamEnv.

    This is private support for direct-env FreespaceMoveTool execution.  It is
    deliberately not exported into the script namespace.
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 1:
        return {"success": False, "reason": "empty timestamps"}
    if not np.all(np.isfinite(ts)):
        return {"success": False, "reason": "timestamps contain non-finite values"}
    ts = ts - float(ts[0])
    if np.any(np.diff(ts) < -1e-9):
        return {"success": False, "reason": "timestamps must be monotonically increasing"}

    keep = np.ones(ts.shape[0], dtype=bool)
    keep[1:] = np.diff(ts) > 1e-9
    original_n = int(len(keep))
    ts = ts[keep]
    n = int(ts.size)
    try:
        left7_all = _prepare_joint7_waypoints(
            env, "left", left_joint_positions, left_gripper_positions, original_n
        )[keep]
        right7_all = _prepare_joint7_waypoints(
            env, "right", right_joint_positions, right_gripper_positions, original_n
        )[keep]
    except Exception as exc:
        return {"success": False, "reason": str(exc)}
    if left7_all.shape[0] != n or right7_all.shape[0] != n:
        return {"success": False, "reason": "waypoint count mismatch after timestamp filtering"}

    speed = max(0.05, float(playback_speed))
    ts = ts / speed
    duration_s = float(ts[-1]) if ts.size else 0.0
    dt = 1.0 / max(1.0, float(command_hz))

    obs_l = env._arms["left"].get_observations()
    obs_r = env._arms["right"].get_observations()
    cur_left7 = np.concatenate([obs_l["joint_pos"], obs_l["gripper_pos"]]).astype(np.float64)
    cur_right7 = np.concatenate([obs_r["joint_pos"], obs_r["gripper_pos"]]).astype(np.float64)
    first_left7 = left7_all[0]
    first_right7 = right7_all[0]

    interp_s = max(0.0, float(start_interp_s))
    interp_steps = int(np.ceil(interp_s / dt)) if interp_s > 1e-9 else 0
    for step in range(1, interp_steps + 1):
        if _stop_requested.is_set():
            raise KeyboardInterrupt("stop during replay start interpolation")
        while _pause_requested.is_set():
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop while paused during replay")
            time.sleep(dt)
        alpha = float(step) / float(max(interp_steps, 1))
        _command_bimanual_joint7(
            env,
            (1.0 - alpha) * cur_left7 + alpha * first_left7,
            (1.0 - alpha) * cur_right7 + alpha * first_right7,
        )
        time.sleep(dt)

    t0 = time.time()
    command_count = 0
    while True:
        if _stop_requested.is_set():
            raise KeyboardInterrupt("stop during bimanual trajectory replay")
        while _pause_requested.is_set():
            if _stop_requested.is_set():
                raise KeyboardInterrupt("stop while paused during replay")
            time.sleep(dt)

        t_now = time.time() - t0
        left_cmd = _sample_keypoints(ts, left7_all, t_now)
        right_cmd = _sample_keypoints(ts, right7_all, t_now)
        _command_bimanual_joint7(env, left_cmd, right_cmd)
        command_count += 1
        if t_now >= duration_s:
            break
        time.sleep(dt)

    settle_steps = max(1, int(round(0.2 / dt)))
    for _ in range(settle_steps):
        if _stop_requested.is_set():
            raise KeyboardInterrupt("stop during replay settle")
        _command_bimanual_joint7(env, left7_all[-1], right7_all[-1])
        time.sleep(dt)

    return {
        "success": True,
        "reason": "ok",
        "waypoints": int(n),
        "duration_s": round(duration_s, 4),
        "playback_speed": float(speed),
        "command_hz": float(command_hz),
        "command_count": int(command_count),
        "start_interp_s": float(interp_s),
        "final_left_gripper": float(left7_all[-1, 6]),
        "final_right_gripper": float(right7_all[-1, 6]),
    }


def _install_direct_helpers(env: Any) -> None:
    if not hasattr(env, "_move_bimanual_joint_keypoints"):
        setattr(
            env,
            "_move_bimanual_joint_keypoints",
            lambda *args, **kwargs: _move_bimanual_joint_keypoints(env, *args, **kwargs),
        )
    if not hasattr(env, "move_bimanual_joint_keypoints"):
        setattr(
            env,
            "move_bimanual_joint_keypoints",
            lambda *args, **kwargs: _move_bimanual_joint_keypoints(env, *args, **kwargs),
        )


def _tool_with_env(tool_cls: type, env: Any, **kwargs: Any):
    try:
        return tool_cls(env=env, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{tool_cls.__module__}.{tool_cls.__name__} must support env= for "
            "real_bimanual_yam direct mode; refusing to fall back to remote transport."
        ) from exc


def _call_tool(tool: Any, *args: Any, **kwargs: Any) -> Any:
    param_names = [p.name for p in getattr(tool, "parameters", [])]
    for idx, value in enumerate(args):
        if idx < len(param_names):
            kwargs[param_names[idx]] = value
    result = tool.execute(**kwargs)
    if not result.success:
        raise RuntimeError(f"Tool {tool.name} failed: {result.error}")
    return result.data


def _tool_callable(tool: Any) -> Callable[..., Any]:
    def fn(*args: Any, **kwargs: Any) -> Any:
        return _call_tool(tool, *args, **kwargs)

    fn.__name__ = getattr(tool, "name", tool.__class__.__name__)
    fn.__doc__ = getattr(tool, "description", None)
    return fn


def _cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        cur = cfg
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return default
        return cur


def make_namespace(env, vlm_backend: str = "nvidia", cfg: Any = None) -> dict[str, Any]:
    """Build the real-YAM direct-mode namespace from shared tool classes."""
    from cap.agent.tools.detection import DetectObjectTool, DetectObjectsOneshotTool
    from cap.agent.tools.freespace_move import FreespaceMoveTool
    from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool
    from cap.agent.tools.native import (
        CloseGripperTool,
        GetCameraImageTool,
        GetRobotStateTool,
        GoHomeTool,
        OpenGripperTool,
        ServoEeDeltaTool,
        SetGripperTool,
    )
    from cap.agent.tools.vlm_query import VlmQueryTool

    _install_direct_helpers(env)

    freespace = _tool_with_env(FreespaceMoveTool, env)
    detect = _tool_with_env(DetectObjectTool, env)
    detect_oneshot = _tool_with_env(DetectObjectsOneshotTool, env, detect_tool=detect)
    anygrasp = _tool_with_env(SampleGraspPoseAnyGraspTool, env)
    vlm = _tool_with_env(VlmQueryTool, env, default_backend=vlm_backend)
    camera_image = _tool_with_env(GetCameraImageTool, env)
    robot_state = _tool_with_env(GetRobotStateTool, env)
    set_gripper = _tool_with_env(SetGripperTool, env)
    open_gripper = _tool_with_env(OpenGripperTool, env)
    close_gripper = _tool_with_env(CloseGripperTool, env)
    go_home = _tool_with_env(GoHomeTool, env)
    servo_ee_delta = _tool_with_env(ServoEeDeltaTool, env)

    def select_best_grasp(
        grasp_candidates: list[Any],
        side: str = "right",
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("batch_side", side)
        return _call_tool(freespace, grasp_candidates=grasp_candidates, **kwargs)

    def get_task_info() -> dict[str, Any]:
        """Evaluate real-hardware task success with a configured VLM reward."""
        from cap.agent.tools.vlm import query as _vlm_query

        task = str(_cfg_select(cfg, "task", "Complete the real-YAM task.") or "")
        backend = str(_cfg_select(cfg, "reward.vlm_backend", "nvidia") or "nvidia")
        model = _cfg_select(
            cfg, "reward.vlm_model", "gcp/google/gemini-3.1-pro-preview"
        )
        camera = str(_cfg_select(cfg, "reward.vlm_camera", "top") or "top")
        reasoning_effort = str(
            _cfg_select(cfg, "reward.vlm_reasoning_effort", "high") or "high"
        )

        img = env.render_rgb(camera)
        if img is None:
            return {
                "success": False,
                "reward": 0.0,
                "method": "vlm_reward",
                "error": f"No image available from camera={camera!r}",
            }

        prompt = f"""
You are evaluating whether a real robot task is complete from the camera image.

Task:
{task}

For the nail-bussing task, success means every visible small black nail/screw is
on or inside the blue plate. Failure means at least one visible nail/screw is
still outside the blue plate. If the image is ambiguous, occluded, or the blue
plate is not clearly visible, answer UNSURE.

Answer with exactly one first-line token:
SUCCESS
FAILURE
UNSURE

After the first-line token, add one short sentence explaining the visible
evidence.
""".strip()

        try:
            response = _vlm_query(
                backend=backend,
                text=prompt,
                images=[img],
                model=model,
                temperature=0.0,
                reasoning_effort=reasoning_effort,
                telemetry_source="real_yam_task_reward",
            )
        except Exception as exc:
            return {
                "success": False,
                "reward": 0.0,
                "method": "vlm_reward",
                "backend": backend,
                "model": model,
                "camera": camera,
                "error": str(exc),
            }

        text = str(response).strip()
        first = text.splitlines()[0].strip().upper() if text else "UNSURE"
        success = first.startswith("SUCCESS")
        status = (
            "success"
            if success
            else "failure"
            if first.startswith("FAILURE")
            else "unsure"
        )
        return {
            "success": success,
            "reward": 1.0 if success else 0.0,
            "method": "vlm_reward",
            "status": status,
            "task": task,
            "backend": backend,
            "model": model,
            "camera": camera,
            "vlm_response": text,
        }

    return {
        "freespace_move": _tool_callable(freespace),
        "select_best_grasp": select_best_grasp,
        "detect_objects_oneshot": _tool_callable(detect_oneshot),
        "sample_grasp_pose_anygrasp": _tool_callable(anygrasp),
        "vlm_query": _tool_callable(vlm),
        "get_camera_image": _tool_callable(camera_image),
        "get_robot_state": _tool_callable(robot_state),
        "set_gripper": _tool_callable(set_gripper),
        "open_gripper": _tool_callable(open_gripper),
        "close_gripper": _tool_callable(close_gripper),
        "go_home": _tool_callable(go_home),
        "servo_ee_delta": _tool_callable(servo_ee_delta),
        "display_rpy_to_quat": display_rpy_to_quat,
        "get_task_info": get_task_info,
    }
