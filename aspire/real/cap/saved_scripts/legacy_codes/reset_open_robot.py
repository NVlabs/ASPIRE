"""One-command YAM reset/open script.

Physical motion is refused unless OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is set by a
bounded reset ticket.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from skill_library.debug_observation import current_run_dir, write_stage_summary


TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "reset_open_robot",
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "initial_state": None,
    "final_state": None,
    "why_stopped": "not_started",
    "required_ticket": (
        "Physical ticket needed: operator present, E-stop reachable, workspace and "
        "hands/tools/cables clear, task reset/open robot, allowed script "
        "cap/saved_scripts/reset_open_robot.py, allowed stages go_home then open "
        "both grippers once, no liquid, max attempts 1, stop on unexpected contact, "
        "gripper/arm anomaly, operator interruption, or competing motion process; "
        "command must set OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
    ),
}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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


def _write_result() -> None:
    run_dir = current_run_dir(default=f"logs/reset_open_robot_{time.strftime('%Y%m%dT%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(_json_safe(TASK_RESULT), indent=2) + "\n",
        encoding="utf-8",
    )
    write_stage_summary(stage="reset_open_robot", result=TASK_RESULT, log_dir=run_dir)


try:
    TASK_RESULT["initial_state"] = _read_state()
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError(
            "Refusing physical motion. Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 only "
            "after a bounded reset/open ticket confirms operator present, E-stop "
            "reachable, workspace clear, and no competing motion process."
        )

    go_home = _required_tool("go_home")
    open_gripper = _required_tool("open_gripper")

    print("[reset_open_robot] Calling go_home()...")
    go_home()
    TASK_RESULT["physical_motion_executed"] = True
    TASK_RESULT["movement_capable_calls"].append("go_home")

    for side in ("left", "right"):
        print(f"[reset_open_robot] Opening {side} gripper...")
        open_gripper(side)
        TASK_RESULT["movement_capable_calls"].append(f"open_gripper:{side}")

    TASK_RESULT["final_state"] = _read_state()
    TASK_RESULT["success"] = True
    TASK_RESULT["reward"] = 1.0
    TASK_RESULT["why_stopped"] = "go_home and open both grippers completed; final state recorded"
except Exception as exc:
    TASK_RESULT["success"] = False
    TASK_RESULT["reward"] = 0.0
    TASK_RESULT["why_stopped"] = f"reset/open failed or refused: {type(exc).__name__}: {exc}"
finally:
    _write_result()
    print(f"[reset_open_robot] {TASK_RESULT['why_stopped']}")
    print(json.dumps(_json_safe(TASK_RESULT), indent=2))
