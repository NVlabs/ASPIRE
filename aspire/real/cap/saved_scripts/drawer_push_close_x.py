"""Close an already-open top drawer with the left arm.

This is the close-only companion to ``drawer_observe_pull_open_x.py``. It
observes the open drawer handle, grasps it with the same staged close logic,
pushes along +X via the existing drawer primitive, opens, and retreats.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from cap.saved_scripts.full_demo_speed import scaled_speed_str


TASK_NAME = "drawer_push_close_x"


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
            "OPENFORGE_DRAWER_CLOSE_EXECUTE": "1",
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
        "script_file=cap/saved_scripts/drawer_push_close_x.py",
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
    if forge_env.exists() and os.environ.get("OPENFORGE_DRAWER_CLOSE_SKIP_FORGE_ENV_SOURCE", "0") != "1":
        command = "source .forge_env && exec " + " ".join(shlex.quote(part) for part in argv)
        os.execvp("bash", ["bash", "-lc", command])
    os.execvp(argv[0], argv)


if globals().get("__name__", "__main__") == "__main__" and not _running_under_run_script():
    _launch_run_script()


from cap.saved_scripts import drawer_observe_pull_open_x as drawer
from cap.saved_scripts.yam_runtime import append_stage_summary, current_run_dir, json_safe, write_json


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


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        rows = list(values)
        if limit is not None:
            rows = rows[:limit]
        return [round(float(value), digits) for value in rows]
    except Exception:
        return []


def _format_xyz(xyz: list[float]) -> str:
    return ",".join(f"{float(value):.5f}" for value in xyz)


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


def _default_push_m() -> float:
    if "OPENFORGE_DRAWER_CLOSE_ONLY_PUSH_M" in os.environ:
        return _env_float("OPENFORGE_DRAWER_CLOSE_ONLY_PUSH_M", 0.095)
    if "OPENFORGE_DRAWER_MANUAL_PUSH_M" in os.environ:
        return _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_M", 0.095)
    pull_m = _env_float("OPENFORGE_DRAWER_MANUAL_PULL_M", 0.105)
    short_m = _env_float("OPENFORGE_DRAWER_MANUAL_PUSH_LESS_THAN_PULL_M", 0.010)
    return max(0.0, pull_m - short_m)


def _run_close_only(run_dir: Path, *, execute: bool, handle_xyz: list[float]) -> dict[str, Any]:
    push_m = _default_push_m()
    with _temporary_env(
        {
            "OPENFORGE_DRAWER_MANUAL_SIDE": os.environ.get("OPENFORGE_DRAWER_MANUAL_SIDE", "left"),
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
            "OPENFORGE_DRAWER_MANUAL_EXECUTE": "1" if execute else "0",
            "OPENFORGE_DRAWER_MANUAL_MAX_FIRST_MOVE_M": os.environ.get("OPENFORGE_DRAWER_MANUAL_MAX_FIRST_MOVE_M", "0.550"),
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
            "OPENFORGE_DRAWER_MANUAL_MIN_CLOSE_DELTA_POS": os.environ.get("OPENFORGE_DRAWER_MANUAL_MIN_CLOSE_DELTA_POS", "0.12"),
            "OPENFORGE_DRAWER_MANUAL_MAX_CLOSED_GRIPPER_POS": os.environ.get("OPENFORGE_DRAWER_MANUAL_MAX_CLOSED_GRIPPER_POS", "0.24"),
            "OPENFORGE_DRAWER_MANUAL_SKIP_CLOSE": "0",
            "OPENFORGE_DRAWER_MANUAL_SKIP_OPEN_AFTER": "0",
            "OPENFORGE_DRAWER_MANUAL_RETREAT_GRIPPER": "1.0",
            "OPENFORGE_DRAWER_MANUAL_OPEN_AFTER_RETREAT": "0",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_Z_M": "0.000",
            "OPENFORGE_DRAWER_MANUAL_POST_CONTACT_CLEAR_GRIPPER": "1.0",
        }
    ):
        result = drawer._run_manual_probe()  # noqa: SLF001 - close-only composition of local primitive
    write_json(run_dir / "plans" / "drawer_push_close_manual_probe.json", result)
    return result


def _run() -> dict[str, Any]:
    started = time.time()
    run_dir = current_run_dir(TASK_NAME)
    execute = _truthy_env("OPENFORGE_DRAWER_CLOSE_EXECUTE", True)
    result: dict[str, Any] = {
        "success": False,
        "reward": 0.0,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "execute": execute,
        "physical_motion_executed": False,
        "pre_close_observe": None,
        "handle_xyz": None,
        "handle_fusion": None,
        "manual_probe": None,
        "post_close_observe": None,
        "why_stopped": None,
    }
    try:
        if execute and not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False):
            result["why_stopped"] = "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is required"
            return result
        obs = drawer._observe("pre_close_drawer_observe")  # noqa: SLF001 - saved-script composition
        result["pre_close_observe"] = obs
        if not obs.get("success"):
            result["why_stopped"] = "pre_close_observe did not produce a high-confidence drawer handle; no motion was run."
            return result
        handle_xyz, fusion = drawer._fused_handle_xyz(obs)  # noqa: SLF001
        result["handle_xyz"] = _round_list(handle_xyz, limit=3)
        result["handle_fusion"] = fusion
        manual = _run_close_only(run_dir, execute=execute, handle_xyz=handle_xyz)
        result["manual_probe"] = manual
        result["physical_motion_executed"] = bool(manual.get("physical_motion_executed"))
        if _truthy_env("OPENFORGE_DRAWER_CLOSE_POST_OBSERVE", True):
            result["post_close_observe"] = drawer._observe("post_close_drawer_observe")  # noqa: SLF001
        result["success"] = bool(manual.get("success"))
        result["reward"] = 1.0 if result["success"] else 0.0
        result["why_stopped"] = (
            "observed open drawer handle and pushed drawer closed"
            if result["success"]
            else f"close-only manual probe failed: {manual.get('why_stopped') or 'see manual_probe details'}"
        )
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["duration_s"] = round(time.time() - started, 3)
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## Drawer Push Close X",
                f"- success: {result.get('success')}",
                f"- execute: {result.get('execute')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- handle_xyz: {result.get('handle_xyz')}",
                f"- manual_success: {((result.get('manual_probe') or {}).get('success'))}",
                f"- why_stopped: {result.get('why_stopped')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
