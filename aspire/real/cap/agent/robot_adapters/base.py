# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot adapter interface for config-driven runtime wiring.

Adapters own robot/environment-specific setup that should not leak into the
agent loop.  They are instantiated from Hydra config via
``robot.adapter._target_`` and provide the small set of hooks needed by
``run_agent.py``, ``run_script.py``, and the subprocess executor.
"""

from __future__ import annotations

from typing import Any, Protocol


class RobotAdapter(Protocol):
    """Minimal protocol implemented by robot adapters."""

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
        """Create robot env + tool namespace."""

    def run_script_overrides(self, cfg: Any | None = None) -> list[str]:
        """Return Hydra CLI overrides to pass from run_agent to run_script."""
        return []

    def child_env(
        self,
        cfg: Any | None = None,
        *,
        seed: int | None = None,
        slot: int = 0,
        n_seeds: int = 1,
    ) -> dict[str, str]:
        """Return per-child environment variables for a run_script process."""
        return {}


def cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
    """OmegaConf-aware nested select with a plain-object fallback."""
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        cur = cfg
        for part in path.split("."):
            if cur is None:
                return default
            cur = getattr(cur, part, default)
        return cur


def cfg_bool(cfg: Any, path: str, default: bool) -> bool:
    value = cfg_select(cfg, path, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def bool_override(key: str, value: Any) -> str:
    """Render a Hydra boolean override."""
    if isinstance(value, bool):
        rendered = str(value).lower()
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def cfg_runtime_kwargs(
    cfg: Any,
    *,
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
) -> dict[str, Any]:
    """Merge explicit legacy args with Hydra cfg values.

    Explicit args are the defaults used by ``cap.env.setup.create_runtime``.
    When ``cfg`` is present, Hydra values take precedence.
    """
    if cfg is None:
        return {
            "runtime_role": runtime_role,
            "env_name": env_name,
            "viewer": viewer,
            "vlm_backend": vlm_backend,
            "seed": seed,
            "layout_id": layout_id,
            "style_id": style_id,
            "camera_height": camera_height,
            "camera_width": camera_width,
            "curobo_host": curobo_host,
            "curobo_port": curobo_port,
            "mppi_host": mppi_host,
            "mppi_port": mppi_port,
        }

    return {
        "runtime_role": runtime_role,
        "env_name": cfg_select(cfg, "env.name", env_name),
        "viewer": cfg_bool(cfg, "env.viewer", viewer),
        # Preserve legacy create_runtime behavior: the namespace-level default
        # VLM backend is an explicit runtime argument, not reflection.reward
        # config.  Individual VLM calls can still pass their own backend/model.
        "vlm_backend": vlm_backend,
        "seed": cfg_select(cfg, "env.seed", seed),
        "layout_id": cfg_select(cfg, "env.layout_id", layout_id),
        "style_id": cfg_select(cfg, "env.style_id", style_id),
        "camera_height": cfg_select(cfg, "env.camera_height", camera_height),
        "camera_width": cfg_select(cfg, "env.camera_width", camera_width),
        "curobo_host": cfg_select(cfg, "runtime.curobo_host", curobo_host),
        "curobo_port": cfg_select(cfg, "runtime.curobo_port", curobo_port),
        "mppi_host": cfg_select(cfg, "runtime.mppi_host", mppi_host),
        "mppi_port": cfg_select(cfg, "runtime.mppi_port", mppi_port),
    }
