"""Physical gripper-only API smoke test for YAM.

This script commands one gripper through a small normalized-position sequence
and records encoder feedback after each step. It does not command arm motion.
Physical gripper motion is refused unless OPENFORGE_ALLOW_PHYSICAL_MOTION=1.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from skill_library.debug_observation import current_run_dir, write_stage_summary


TASK_NAME = "gripper_api_smoke"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    return float(raw)


def _env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return float(raw)


def _csv_float(name: str, default: tuple[float, ...]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [float(v) for v in default]
    return [float(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]


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


def _read_gripper(get_robot_state: Callable[..., Any], side: str) -> float | None:
    state = get_robot_state()
    value = getattr(state, f"{side}_gripper_pos", None)
    if value is None and isinstance(state, dict):
        value = ((state.get("arms") or {}).get(side) or {}).get("gripper_pos")
    return None if value is None else float(value)


def _read_state(get_robot_state: Callable[..., Any]) -> Any:
    try:
        return _json_safe(get_robot_state())
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _write_result(run_dir: Path, result: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    write_stage_summary(stage=TASK_NAME, result=result, log_dir=run_dir)


def get_task_info() -> dict[str, Any]:
    return {"method": TASK_NAME}


run_dir = current_run_dir(default=f"logs/{TASK_NAME}_{time.strftime('%Y%m%dT%H%M%S')}")
get_robot_state = _required_tool("get_robot_state")
set_gripper = _required_tool("set_gripper")

side = os.environ.get("OPENFORGE_GRIPPER_API_SMOKE_SIDE", "right").strip().lower()
sequence = _csv_float("OPENFORGE_GRIPPER_API_SMOKE_SEQUENCE", (1.0, 0.68, 0.40, 1.0))
settle_s = _env_float("OPENFORGE_GRIPPER_API_SMOKE_SETTLE_S", 1.0)
vel_limit = _env_float_optional("OPENFORGE_GRIPPER_API_SMOKE_VEL_LIMIT")
torque_limit = _env_float_optional("OPENFORGE_GRIPPER_API_SMOKE_TORQUE_LIMIT")
tolerance = _env_float("OPENFORGE_GRIPPER_API_SMOKE_TOLERANCE", 0.08)

result: dict[str, Any] = {
    "schema": "openforge.gripper_api_smoke.v1",
    "success": False,
    "method": TASK_NAME,
    "run_dir": str(run_dir),
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "side": side,
    "sequence": sequence,
    "settle_s": settle_s,
    "vel_limit": vel_limit,
    "torque_limit": torque_limit,
    "tolerance": tolerance,
    "initial_state": _read_state(get_robot_state),
    "steps": [],
    "final_state": None,
    "why_stopped": "not_started",
}

try:
    if side not in {"left", "right"}:
        raise RuntimeError(f"invalid side {side!r}; expected left or right")
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError("refusing gripper motion without OPENFORGE_ALLOW_PHYSICAL_MOTION=1")

    for target in sequence:
        before = _read_gripper(get_robot_state, side)
        command_kwargs: dict[str, float] = {}
        if vel_limit is not None:
            command_kwargs["vel_limit"] = float(vel_limit)
        if torque_limit is not None:
            command_kwargs["torque_limit"] = float(torque_limit)
        command_result = set_gripper(side, float(target), **command_kwargs)
        result["physical_motion_executed"] = True
        result["movement_capable_calls"].append(f"set_gripper:{side}:{target}")
        time.sleep(settle_s)
        after = _read_gripper(get_robot_state, side)
        reached = after is not None and abs(float(after) - float(target)) <= tolerance
        step = {
            "target": float(target),
            "before": before,
            "after": after,
            "delta": None if before is None or after is None else float(after) - float(before),
            "reached": reached,
            "error_abs": None if after is None else abs(float(after) - float(target)),
            "set_gripper_result": _json_safe(command_result),
        }
        result["steps"].append(step)
        print(f"[{TASK_NAME}] {side} target={target:.3f} before={before} after={after} reached={reached}")

    result["final_state"] = _read_state(get_robot_state)
    result["success"] = bool(result["steps"]) and all(bool(step["reached"]) for step in result["steps"])
    result["why_stopped"] = "gripper API smoke completed"
except Exception as exc:
    result["success"] = False
    result["why_stopped"] = f"gripper API smoke failed/refused: {type(exc).__name__}: {exc}"
finally:
    _write_result(run_dir, result)
    print(json.dumps(_json_safe(result), indent=2))
