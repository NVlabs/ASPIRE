"""Real bimanual YAM robot adapter."""

from __future__ import annotations

from typing import Any

from cap.agent.robot_adapters.base import bool_override, cfg_runtime_kwargs, cfg_select


class RealYamAdapter:
    """Create real-YAM env/tool namespace and child run_script overrides."""

    def __init__(self, config_group: str = "real_yam") -> None:
        self.config_group = config_group

    def create_runtime(
        self,
        *,
        cfg: Any | None = None,
        runtime_role: str = "script",
        env_name: str | None = None,
        viewer: bool = False,
        vlm_backend: str = "nvidia",
        seed: int | None = None,
        layout_id: int | None = None,
        style_id: int | None = None,
        camera_height: int | None = None,
        camera_width: int | None = None,
        curobo_host: str = "127.0.0.1",
        curobo_port: int = 0,
        mppi_host: str = "127.0.0.1",
        mppi_port: int = 0,
    ) -> tuple[Any, dict[str, Any]]:
        kwargs = cfg_runtime_kwargs(
            cfg,
            runtime_role=runtime_role,
            env_name=env_name,
            viewer=viewer,
            vlm_backend=vlm_backend,
            seed=seed,
            layout_id=layout_id,
            style_id=style_id,
            camera_height=camera_height,
            camera_width=camera_width,
            curobo_host=curobo_host,
            curobo_port=curobo_port,
            mppi_host=mppi_host,
            mppi_port=mppi_port,
        )

        from cap.agent.tool_handle import make_tool_runner_namespace
        from cap.env import create_env
        from cap.env.real_bimanual_yam.skills import make_namespace as _yam_ns

        # The parent run_agent process builds a namespace for prompting and
        # bookkeeping, while child run_script tmux windows execute the actual
        # robot script.  Real RGB-D cameras are single-owner / fragile, so the
        # parent must not open them; otherwise the child can start with no top
        # RGB frame and all segmentation calls fail.
        enable_cameras = kwargs["runtime_role"] != "agent"
        if cfg_select(cfg, "runtime.no_cameras", False):
            enable_cameras = False

        env = create_env(
            kwargs["env_name"] or "yam-real",
            viewer=kwargs["viewer"],
            enable_cameras=enable_cameras,
        )
        namespace = _yam_ns(env, vlm_backend=kwargs["vlm_backend"], cfg=cfg)
        namespace.update(make_tool_runner_namespace())

        # Dashboard is intentionally not started here: run_agent.py also builds
        # a real-YAM namespace in the parent process.  The script-execution
        # dashboard is started by RealYamBackend inside child run_script.py.
        return env, namespace

    def run_script_overrides(self, cfg: Any | None = None) -> list[str]:
        overrides = [f"robot={self.config_group}"]
        for field in ("dashboard", "await_exit", "go_home_on_exit"):
            value = cfg_select(cfg, f"robot.{field}", None)
            if value is not None:
                overrides.append(bool_override(f"robot.{field}", value))
        return overrides

    def child_env(
        self,
        cfg: Any | None = None,
        *,
        seed: int | None = None,
        slot: int = 0,
        n_seeds: int = 1,
    ) -> dict[str, str]:
        """Export service endpoints for code paths that read cap.config/env.

        run_script also receives Hydra runtime.* overrides, but several tool
        modules cache cap.config values at import time. Setting the child env
        here keeps the tmux executor aligned with the configured tunnel ports.
        """
        _ = seed
        sam3_host = str(cfg_select(cfg, "runtime.sam3_host", "127.0.0.1"))
        sam3_base = int(cfg_select(cfg, "runtime.sam3_port", 6767) or 0)
        sam3_port = sam3_base + slot if (sam3_base > 0 and n_seeds > 1) else sam3_base

        anygrasp_host = str(cfg_select(cfg, "runtime.anygrasp_host", "127.0.0.1"))
        anygrasp_base = int(cfg_select(cfg, "runtime.anygrasp_port", 8122) or 0)
        anygrasp_port = (
            anygrasp_base + slot
            if (anygrasp_base > 0 and n_seeds > 1)
            else anygrasp_base
        )

        bundlesdf_host = str(cfg_select(cfg, "runtime.bundlesdf_host", "127.0.0.1"))
        bundlesdf_base = int(cfg_select(cfg, "runtime.bundlesdf_port", 8119) or 0)
        bundlesdf_port = (
            bundlesdf_base + slot
            if (bundlesdf_base > 0 and n_seeds > 1)
            else bundlesdf_base
        )

        curobo_host = str(cfg_select(cfg, "runtime.curobo_host", "127.0.0.1"))
        curobo_base = int(cfg_select(cfg, "runtime.curobo_port", 8611) or 0)
        curobo_port = (
            curobo_base + slot if (curobo_base > 0 and n_seeds > 1) else curobo_base
        )

        return {
            "SAM3_SERVER_HOST": sam3_host,
            "SAM3_SERVER_PORT": str(sam3_port),
            "ANYGRASP_SERVER_HOST": anygrasp_host,
            "ANYGRASP_SERVER_PORT": str(anygrasp_port),
            "ANYGRASP_SERVICE_URL": f"http://{anygrasp_host}:{anygrasp_port}",
            "BUNDLESDF_SERVER_HOST": bundlesdf_host,
            "BUNDLESDF_SERVER_PORT": str(bundlesdf_port),
            "CAP_CUROBO_HOST": curobo_host,
            "CAP_CUROBO_PORT": str(curobo_port),
            "CAP_CUROBO_START_SERVER": "0" if curobo_port > 0 else "1",
            "CAP_ROBOT_TYPE": "yam",
        }
