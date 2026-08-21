# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-YAM script runner backend for the openforge runtime subset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from cap.agent.recorder import ScriptRecorder


def _recording_cameras_from_env() -> list[str]:
    import os

    raw = os.environ.get("OPENFORGE_REAL_YAM_RECORDING_CAMERAS", "").strip()
    if not raw:
        return []
    cameras: list[str] = []
    for item in raw.replace(";", ",").split(","):
        camera = item.strip().lower()
        if camera and camera not in cameras:
            cameras.append(camera)
    return cameras


class ScriptRunnerBackend(ABC):
    @abstractmethod
    def vis_cameras(self, env: Any) -> list[str]:
        pass

    @property
    @abstractmethod
    def is_quiet(self) -> bool:
        pass

    @abstractmethod
    def setup_recorder(
        self, env: Any, log_dir: Path, cfg: Any
    ) -> "ScriptRecorder | None":
        pass

    @contextmanager
    @abstractmethod
    def exec_context(self, namespace: dict, script_path: Path) -> Iterator[None]:
        yield

    @abstractmethod
    def post_exec(self, exec_error: str | None, namespace: dict) -> None:
        pass


class RealYamBackend(ScriptRunnerBackend):
    """YamDashboard + await_exit + go_home for real bimanual YAM hardware."""

    def __init__(
        self,
        env: Any,
        cfg: Any,
        dashboard: bool = True,
        await_exit: bool = True,
        go_home_on_exit: bool = True,
        **_: Any,
    ):
        self._env = env
        self._cfg = cfg
        self._dashboard = bool(dashboard)
        self._await_exit = await_exit
        self._go_home_on_exit = go_home_on_exit
        self._started_dashboard = False

    def vis_cameras(self, env: Any) -> list[str]:
        import os

        _ = env
        enabled = os.environ.get("CAP_REAL_YAM_RUN_SCRIPT_CAPTURE_FRAMES", "")
        if enabled.strip().lower() in {"1", "true", "yes", "on"}:
            from cap.config import CAMERA_NAMES

            return list(CAMERA_NAMES)
        return []

    @property
    def is_quiet(self) -> bool:
        return self._dashboard

    def setup_recorder(
        self, env: Any, log_dir: Path, cfg: Any
    ) -> "ScriptRecorder | None":
        import os

        if not cfg.recording.enabled:
            return None

        cameras = _recording_cameras_from_env()
        if not cameras:
            cameras = list(getattr(cfg.recording, "cameras", None) or [])
        if not cameras:
            from cap.config import CAMERA_NAMES

            cameras = list(CAMERA_NAMES)

        recording_source = os.environ.get(
            "OPENFORGE_REAL_YAM_RECORDING_SOURCE", "preview"
        ).strip().lower()
        if not recording_source:
            recording_source = "preview"
        use_preview = recording_source in {"preview", "bundlesdf", "portal", "mjpeg"}
        use_direct_env = recording_source in {"env", "direct", "camera"}

        if bool(getattr(cfg.runtime, "no_cameras", False)):
            if use_direct_env:
                print(
                    "[run_script] Real-YAM recording source 'env' requested, "
                    "but runtime.no_cameras=true; using BundleSDF previews"
                )
            use_preview = True

        if use_preview or not use_direct_env:
            if not use_preview:
                print(
                    "[run_script] Unknown OPENFORGE_REAL_YAM_RECORDING_SOURCE="
                    f"{recording_source!r}; using BundleSDF previews"
                )
            from cap.agent.recorder import PreviewStreamRecorder

            host = str(getattr(cfg.runtime, "bundlesdf_host", "127.0.0.1"))
            port = int(getattr(cfg.runtime, "bundlesdf_port", 8119))
            print("[run_script] Real-YAM recording source: BundleSDF previews")
            recorder = PreviewStreamRecorder(f"http://{host}:{port}", cameras, log_dir)
            recorder.start()
            return recorder

        from cap.agent.recorder import ScriptRecorder

        class _RealYamRecorderSource:
            def __init__(self, real_env: Any, camera_names: list[str]):
                self._env = real_env
                self._cameras = {name: None for name in camera_names}

            def get_camera_image(self, camera: str):
                return self._env.render_rgb(camera)

        recorder = ScriptRecorder(_RealYamRecorderSource(env, cameras), log_dir)
        recorder.start()
        return recorder

    @contextmanager
    def exec_context(self, namespace: dict, script_path: Path) -> Iterator[None]:
        if self._dashboard:
            dash = getattr(self._env, "_dashboard", None)
            if dash is None:
                get_robot_state = namespace.get("get_robot_state")
                if get_robot_state is None:
                    print("[run_script] YAM dashboard not started: get_robot_state missing")
                else:
                    try:
                        from cap.env.real_bimanual_yam.dashboard import YamDashboard

                        dash = YamDashboard(self._env, get_robot_state)
                        dash.start()
                        self._env._dashboard = dash
                        self._started_dashboard = True
                    except Exception as exc:
                        print(f"[run_script] YAM dashboard failed to start: {exc}")
                        dash = None
        else:
            print("[run_script] YAM dashboard disabled (robot.dashboard=false)")

        dash = getattr(self._env, "_dashboard", None)
        if dash is None:
            yield
            return

        from cap.agent.profiler import set_tool_event_hooks

        set_tool_event_hooks(on_start=dash.on_tool_start, on_end=dash.on_tool_end)
        try:
            yield
        finally:
            set_tool_event_hooks(on_start=None, on_end=None)

    def post_exec(self, exec_error: str | None, namespace: dict) -> None:
        try:
            if exec_error == "KeyboardInterrupt":
                return

            should_home = True
            if self._await_exit:
                dash = getattr(self._env, "_dashboard", None)
                if dash is not None:
                    should_home = dash.await_exit()
                else:
                    import sys as _sys

                    _sys.stderr.write(
                        "\n[YAM] Script done - ENTER go home & exit | "
                        "S skip home | X emergency stop: "
                    )
                    _sys.stderr.flush()
                    try:
                        line = input().strip().lower()
                    except (EOFError, OSError):
                        line = ""
                    if line in ("s", "x", "q"):
                        should_home = False

            if should_home and self._go_home_on_exit:
                go_home_fn = namespace.get("go_home")
                if go_home_fn is not None:
                    try:
                        print("[run_script] Going home...")
                        go_home_fn()
                    except Exception as exc:
                        print(f"[run_script] go_home failed: {exc}")
        finally:
            if self._started_dashboard:
                dash = getattr(self._env, "_dashboard", None)
                if dash is not None:
                    dash.stop()
                try:
                    delattr(self._env, "_dashboard")
                except AttributeError:
                    pass


def _cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
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


def get_backend(env_name: str, cfg: Any, env: Any) -> ScriptRunnerBackend:
    if not str(env_name).startswith("yam-real"):
        raise ValueError(
            f"Unsupported env in this openforge runtime subset: {env_name!r}. "
            "Only yam-real is included."
        )

    robot_cfg = getattr(cfg, "robot", None)
    return RealYamBackend(
        env=env,
        cfg=cfg,
        dashboard=bool(_cfg_select(robot_cfg, "dashboard", True)),
        await_exit=bool(_cfg_select(robot_cfg, "await_exit", True)),
        go_home_on_exit=bool(_cfg_select(robot_cfg, "go_home_on_exit", True)),
    )
