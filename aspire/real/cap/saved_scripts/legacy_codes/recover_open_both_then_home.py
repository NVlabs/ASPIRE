"""Recovery-only: open both grippers in place, then go home.

Use after a failed bimanual hold where either gripper may be near or holding an
object. Physical motion requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable


RESULT: dict[str, Any] = {
    "success": False,
    "method": "recover_open_both_then_home",
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "initial_state": None,
    "final_state": None,
    "why_stopped": "not_started",
}


def get_task_info() -> dict[str, Any]:
    return dict(RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _tool(name: str) -> Callable[..., Any] | None:
    fn = globals().get(name)
    if callable(fn):
        return fn
    try:
        import skill_library.namespace as namespace

        fn = getattr(namespace, name, None)
        return fn if callable(fn) else None
    except Exception:
        return None


def _required_tool(name: str) -> Callable[..., Any]:
    fn = _tool(name)
    if fn is None:
        raise RuntimeError(f"required YAM tool unavailable: {name}")
    return fn


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def _read_state() -> Any:
    fn = _tool("get_robot_state")
    if fn is None:
        return "get_robot_state unavailable"
    return _json_safe(fn())


def _run_dir() -> Path:
    try:
        from skill_library.debug_observation import current_run_dir

        return current_run_dir(default=f"logs/recover_open_both_then_home_{time.strftime('%Y%m%dT%H%M%S')}")
    except Exception:
        return Path(f"logs/recover_open_both_then_home_{time.strftime('%Y%m%dT%H%M%S')}")


try:
    RESULT["initial_state"] = _read_state()
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError("refusing physical recovery without OPENFORGE_ALLOW_PHYSICAL_MOTION=1")

    set_gripper = _required_tool("set_gripper")
    go_home = _required_tool("go_home")
    open_gripper = _tool("open_gripper")

    for side in ("left", "right"):
        RESULT[f"open_{side}"] = _json_safe(set_gripper(side, 1.0, vel_limit=0.18, torque_limit=0.12))
        RESULT["physical_motion_executed"] = True
        RESULT["movement_capable_calls"].append(f"set_gripper:{side}")
    time.sleep(0.4)

    RESULT["go_home"] = _json_safe(go_home())
    RESULT["movement_capable_calls"].append("go_home")

    if callable(open_gripper):
        for side in ("left", "right"):
            RESULT[f"open_{side}_post_home"] = _json_safe(open_gripper(side))
            RESULT["movement_capable_calls"].append(f"open_gripper:{side}")

    RESULT["final_state"] = _read_state()
    RESULT["success"] = True
    RESULT["why_stopped"] = "opened both grippers in place, then go_home completed"
except Exception as exc:
    RESULT["why_stopped"] = f"recovery failed: {type(exc).__name__}: {exc}"
finally:
    run_dir = _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(_json_safe(RESULT), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(RESULT), indent=2))
