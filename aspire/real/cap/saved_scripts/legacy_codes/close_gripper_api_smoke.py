# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical close_gripper API smoke test for YAM.

This script checks the dedicated ``close_gripper`` tool, not just
``set_gripper(..., 0.0)`` from a saved script. To make the close step meaningful,
it can first open the selected gripper, then call close_gripper and verify the
reported encoder moved toward 0.0.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from skill_library.debug_observation import current_run_dir, write_stage_summary


TASK_NAME = "close_gripper_api_smoke"


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def _call_gripper(fn: Callable[..., Any], side: str, *, vel_limit: float | None, torque_limit: float | None) -> Any:
    kwargs: dict[str, float] = {}
    if vel_limit is not None:
        kwargs["vel_limit"] = float(vel_limit)
    if torque_limit is not None:
        kwargs["torque_limit"] = float(torque_limit)
    return fn(side, **kwargs)


def _write_result(run_dir: Path, result: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(_json_safe(result), indent=2) + "\n", encoding="utf-8")
    write_stage_summary(stage=TASK_NAME, result=result, log_dir=run_dir)


def get_task_info() -> dict[str, Any]:
    return {"method": TASK_NAME}


run_dir = current_run_dir(default=f"logs/{TASK_NAME}_{time.strftime('%Y%m%dT%H%M%S')}")
get_robot_state = _required_tool("get_robot_state")
open_gripper = _required_tool("open_gripper")
close_gripper = _required_tool("close_gripper")

side = os.environ.get("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_SIDE", "right").strip().lower()
open_first = _truthy_env("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_OPEN_FIRST", True)
settle_s = _env_float("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_SETTLE_S", 1.0)
open_min = _env_float("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_OPEN_MIN", 0.90)
close_max = _env_float("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_CLOSE_MAX", 0.10)
vel_limit = _env_float_optional("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_VEL_LIMIT")
torque_limit = _env_float_optional("OPENFORGE_CLOSE_GRIPPER_API_SMOKE_TORQUE_LIMIT")

result: dict[str, Any] = {
    "schema": "openforge.close_gripper_api_smoke.v1",
    "success": False,
    "method": TASK_NAME,
    "run_dir": str(run_dir),
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "side": side,
    "open_first": open_first,
    "settle_s": settle_s,
    "open_min": open_min,
    "close_max": close_max,
    "vel_limit": vel_limit,
    "torque_limit": torque_limit,
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

    if open_first:
        before_open = _read_gripper(get_robot_state, side)
        open_result = _call_gripper(open_gripper, side, vel_limit=vel_limit, torque_limit=torque_limit)
        result["physical_motion_executed"] = True
        result["movement_capable_calls"].append(f"open_gripper:{side}")
        time.sleep(settle_s)
        after_open = _read_gripper(get_robot_state, side)
        open_passed = after_open is not None and float(after_open) >= open_min
        result["steps"].append(
            {
                "stage": "open_gripper",
                "before": before_open,
                "after": after_open,
                "delta": None if before_open is None or after_open is None else float(after_open) - float(before_open),
                "passed": open_passed,
                "open_gripper_result": _json_safe(open_result),
            }
        )
        print(f"[{TASK_NAME}] {side} open before={before_open} after={after_open} passed={open_passed}")

    before_close = _read_gripper(get_robot_state, side)
    close_result = _call_gripper(close_gripper, side, vel_limit=vel_limit, torque_limit=torque_limit)
    result["physical_motion_executed"] = True
    result["movement_capable_calls"].append(f"close_gripper:{side}")
    time.sleep(settle_s)
    after_close = _read_gripper(get_robot_state, side)
    close_passed = after_close is not None and float(after_close) <= close_max
    moved_closed = (
        before_close is not None
        and after_close is not None
        and float(before_close) - float(after_close) >= 0.25
    )
    result["steps"].append(
        {
            "stage": "close_gripper",
            "before": before_close,
            "after": after_close,
            "delta": None if before_close is None or after_close is None else float(after_close) - float(before_close),
            "passed": close_passed,
            "moved_closed": moved_closed,
            "close_gripper_result": _json_safe(close_result),
        }
    )
    print(
        f"[{TASK_NAME}] {side} close before={before_close} after={after_close} "
        f"passed={close_passed} moved_closed={moved_closed}"
    )

    result["final_state"] = _read_state(get_robot_state)
    close_step = result["steps"][-1]
    if open_first:
        result["success"] = bool(result["steps"][0]["passed"] and close_step["passed"] and close_step["moved_closed"])
    else:
        result["success"] = bool(close_step["passed"])
    result["why_stopped"] = "close gripper API smoke completed"
except Exception as exc:
    result["success"] = False
    result["why_stopped"] = f"close gripper API smoke failed/refused: {type(exc).__name__}: {exc}"
finally:
    _write_result(run_dir, result)
    print(json.dumps(_json_safe(result), indent=2))
