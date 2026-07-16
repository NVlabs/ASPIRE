"""Open YAM gripper(s), return home while holding them open, then confirm open.

Physical motion is refused unless OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is set by a
bounded recovery ticket.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cap.saved_scripts.full_demo_speed import scaled_speed
from skill_library.debug_observation import current_run_dir, write_stage_summary


TASK_NAME = "open_grippers_return_home"
TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": TASK_NAME,
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "initial_state": None,
    "state_after_initial_open": None,
    "state_after_home": None,
    "final_state": None,
    "sides": [],
    "why_stopped": "not_started",
    "required_ticket": (
        "Physical ticket needed: operator present, E-stop reachable, workspace and "
        "hands/tools/cables clear, task open gripper(s), return home while open, "
        "then confirm open "
        "gripper(s), allowed script cap/saved_scripts/open_grippers_return_home.py, "
        "allowed stages open selected gripper(s), home arm joints with selected "
        "gripper targets held open, read state, confirm selected gripper(s) open, "
        "no liquid, max attempts 1, stop on unexpected contact, gripper/arm anomaly, "
        "operator interruption, or competing motion process; command must set "
        "OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
    ),
}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [part.strip().lower() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _tool(name: str) -> Callable[..., Any] | None:
    try:
        import skill_library.namespace as namespace

        fn = getattr(namespace, name, None)
        return fn if callable(fn) else None
    except Exception:
        return None


def _required_tool(name: str) -> Callable[..., Any]:
    fn = _tool(name)
    if fn is None:
        raise RuntimeError(
            f"Required YAM tool {name!r} is unavailable. Run through run_script.py "
            "with skill_library_path=cap/saved_scripts/skill_library."
        )
    return fn


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist() if value.size <= 64 else {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _read_state() -> Any:
    fn = _tool("get_robot_state")
    if fn is None:
        return "get_robot_state unavailable"
    try:
        return _json_safe(fn())
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _find_env(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any | None:
    if value is None or depth > 6:
        return None
    seen = seen if seen is not None else set()
    ident = id(value)
    if ident in seen:
        return None
    seen.add(ident)

    env = getattr(value, "_env", None)
    if env is not None:
        return env

    env = _find_env(getattr(value, "__wrapped__", None), depth=depth + 1, seen=seen)
    if env is not None:
        return env

    for cell in getattr(value, "__closure__", ()) or ():
        try:
            cell_value = cell.cell_contents
        except ValueError:
            continue
        env = _find_env(cell_value, depth=depth + 1, seen=seen)
        if env is not None:
            return env
    return None


def _tool_env(name: str) -> Any | None:
    return _find_env(_tool(name))


def _obs(env: Any, side: str) -> dict[str, np.ndarray]:
    if hasattr(env, "get_observations"):
        return env.get_observations(side)
    if hasattr(env, "get_arm_observation"):
        return env.get_arm_observation(side)
    raise RuntimeError("direct YAM env does not expose arm observations")


def _go_home_holding_open(
    sides: list[str],
    *,
    duration_s_override: float | None = None,
    settle_s_override: float | None = None,
    command_hz_override: float | None = None,
) -> dict[str, Any]:
    env = _tool_env("go_home") or _tool_env("get_robot_state")
    if env is None:
        raise RuntimeError(
            "Cannot access direct YAM env from the injected tool namespace; refusing "
            "to call normal go_home() because YAM home gripper targets are closed."
        )
    if not hasattr(env, "command_joint_state"):
        raise RuntimeError("direct YAM env does not expose command_joint_state")

    profiles = getattr(getattr(env, "_profile", None), "arms", None)
    if not profiles:
        raise RuntimeError("direct YAM env does not expose arm profiles")

    open_target = float(np.clip(_env_float("OPENFORGE_OPEN_HOME_OPEN_TARGET_POS", 1.0), 0.0, 1.0))
    settle_s = max(
        0.0,
        float(
            settle_s_override
            if settle_s_override is not None
            else _env_float("OPENFORGE_OPEN_HOME_SETTLE_S", 0.5)
        ),
    )
    command_hz = max(
        10.0,
        float(
            command_hz_override
            if command_hz_override is not None
            else _env_float("OPENFORGE_OPEN_HOME_COMMAND_HZ", 50.0)
        ),
    )
    dt = 1.0 / command_hz

    initial: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    selected = set(sides)
    for side in ("left", "right"):
        side_obs = _obs(env, side)
        current_joints = np.asarray(side_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        current_gripper = float(np.asarray(side_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        home_joints = np.asarray(profiles[side].home_joint_pos, dtype=np.float64).reshape(-1)[:6]
        target_gripper = open_target if side in selected else current_gripper
        initial[side] = np.concatenate([current_joints, [current_gripper]])
        target[side] = np.concatenate([home_joints, [target_gripper]])

    max_joint_delta = max(
        float(np.max(np.abs(target[side][:6] - initial[side][:6])))
        for side in ("left", "right")
    )
    duration_source = "duration_override"
    speed_rad_s: float | None = None
    if duration_s_override is not None:
        duration_s = float(duration_s_override)
    else:
        duration_env = _env_float_optional("OPENFORGE_OPEN_HOME_DURATION_S")
        if duration_env is not None:
            duration_s = duration_env
            duration_source = "OPENFORGE_OPEN_HOME_DURATION_S"
        else:
            speed_rad_s = max(0.05, _env_float("OPENFORGE_OPEN_HOME_SPEED_RAD_S", scaled_speed(0.5)))
            min_duration_s = max(0.2, _env_float("OPENFORGE_OPEN_HOME_MIN_DURATION_S", 0.8))
            max_duration_s = max(min_duration_s, _env_float("OPENFORGE_OPEN_HOME_MAX_DURATION_S", 10.0))
            duration_s = float(np.clip(max_joint_delta / speed_rad_s, min_duration_s, max_duration_s))
            duration_source = "OPENFORGE_OPEN_HOME_SPEED_RAD_S"
    duration_s = max(0.2, duration_s)

    def _command(side: str, pos7: np.ndarray) -> None:
        profile = profiles[side]
        env.command_joint_state(
            side,
            {
                "pos": np.asarray(pos7, dtype=np.float64).reshape(7),
                "vel": np.zeros(7, dtype=np.float64),
                "kp": profile.interp_kp,
                "kd": profile.interp_kd,
            },
        )

    start = time.time()
    command_count = 0
    while True:
        elapsed = time.time() - start
        alpha = float(np.clip(elapsed / duration_s, 0.0, 1.0))
        for side in ("left", "right"):
            _command(side, (1.0 - alpha) * initial[side] + alpha * target[side])
        command_count += 1
        if alpha >= 1.0:
            break
        time.sleep(dt)

    settle_start = time.time()
    while time.time() - settle_start < settle_s:
        for side in ("left", "right"):
            _command(side, target[side])
        command_count += 1
        time.sleep(dt)

    final_grippers = {}
    max_joint_error = 0.0
    for side in ("left", "right"):
        side_obs = _obs(env, side)
        joints = np.asarray(side_obs["joint_pos"], dtype=np.float64).reshape(-1)[:6]
        final_grippers[side] = float(np.asarray(side_obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        max_joint_error = max(max_joint_error, float(np.max(np.abs(joints - target[side][:6]))))

    return {
        "success": True,
        "mode": "home_joint_interpolation_holding_selected_grippers_open",
        "open_sides": list(sides),
        "open_target": open_target,
        "duration_s": duration_s,
        "duration_source": duration_source,
        "speed_rad_s": speed_rad_s,
        "max_joint_delta_rad": max_joint_delta,
        "settle_s": settle_s,
        "command_hz": command_hz,
        "command_count": command_count,
        "max_joint_error_rad": max_joint_error,
        "final_grippers": final_grippers,
    }


def _write_result() -> None:
    run_dir = current_run_dir(default=f"logs/{TASK_NAME}_{time.strftime('%Y%m%dT%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(_json_safe(TASK_RESULT), indent=2) + "\n",
        encoding="utf-8",
    )
    write_stage_summary(stage=TASK_NAME, result=TASK_RESULT, log_dir=run_dir)


try:
    TASK_RESULT["initial_state"] = _read_state()
    sides = _csv("OPENFORGE_OPEN_HOME_GRIPPER_SIDES", ("left", "right"))
    invalid = [side for side in sides if side not in {"left", "right"}]
    if invalid:
        raise RuntimeError(f"invalid gripper side(s): {invalid}; expected left,right")
    TASK_RESULT["sides"] = sides

    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError(
            "Refusing physical motion. Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 only "
            "after a bounded open-then-home recovery ticket confirms operator present, "
            "E-stop reachable, workspace clear, and no competing motion process."
        )

    open_gripper = _required_tool("open_gripper")

    for side in sides:
        print(f"[open_grippers_return_home] Opening {side} gripper...")
        open_gripper(side)
        TASK_RESULT["physical_motion_executed"] = True
        TASK_RESULT["movement_capable_calls"].append(f"open_gripper:{side}")

    TASK_RESULT["state_after_initial_open"] = _read_state()

    print("[open_grippers_return_home] Homing arm joints while holding selected gripper(s) open...")
    TASK_RESULT["home_holding_open_result"] = _json_safe(_go_home_holding_open(sides))
    TASK_RESULT["physical_motion_executed"] = True
    TASK_RESULT["movement_capable_calls"].append("go_home_holding_selected_grippers_open")

    TASK_RESULT["state_after_home"] = _read_state()

    for side in sides:
        print(f"[open_grippers_return_home] Confirming {side} gripper open after home...")
        open_gripper(side)
        TASK_RESULT["physical_motion_executed"] = True
        TASK_RESULT["movement_capable_calls"].append(f"open_gripper:{side}")

    if _truthy_env("OPENFORGE_OPEN_HOME_REHOLD_AFTER_FINAL_OPEN", True):
        print("[open_grippers_return_home] Re-holding home after final gripper open...")
        TASK_RESULT["final_rehold_home_result"] = _json_safe(
            _go_home_holding_open(
                sides,
                duration_s_override=_env_float("OPENFORGE_OPEN_HOME_FINAL_REHOLD_DURATION_S", 2.0),
                settle_s_override=_env_float("OPENFORGE_OPEN_HOME_FINAL_REHOLD_SETTLE_S", 1.0),
                command_hz_override=_env_float("OPENFORGE_OPEN_HOME_FINAL_REHOLD_COMMAND_HZ", 50.0),
            )
        )
        TASK_RESULT["physical_motion_executed"] = True
        TASK_RESULT["movement_capable_calls"].append("rehold_home_after_final_open")

    TASK_RESULT["final_state"] = _read_state()
    TASK_RESULT["success"] = True
    TASK_RESULT["reward"] = 1.0
    TASK_RESULT["why_stopped"] = (
        "opened selected gripper(s), homed arm joints while holding selected "
        "gripper(s) open, state_after_home recorded, confirmed selected "
        "gripper(s) open, then re-held home before final state when enabled"
    )
except Exception as exc:
    TASK_RESULT["success"] = False
    TASK_RESULT["reward"] = 0.0
    TASK_RESULT["why_stopped"] = f"open-home-open failed or refused: {type(exc).__name__}: {exc}"
finally:
    _write_result()
    print(f"[open_grippers_return_home] {TASK_RESULT['why_stopped']}")
    print(json.dumps(_json_safe(TASK_RESULT), indent=2))
