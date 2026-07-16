"""Hydra-instantiated robot adapters."""

from __future__ import annotations

from typing import Any

from cap.agent.robot_adapters.base import RobotAdapter, cfg_select
from cap.agent.robot_adapters.real_yam import RealYamAdapter


def adapter_for_env(env_name: str | None) -> RobotAdapter:
    """Backward-compatible adapter selection by env name."""
    name = str(env_name or "")
    if name == "yam-real" or name.startswith("yam-real:"):
        return RealYamAdapter()
    raise ValueError(
        f"Unsupported env in this openforge runtime subset: {env_name!r}. "
        "Only yam-real is included."
    )


def get_robot_adapter(cfg: Any | None) -> RobotAdapter:
    """Instantiate ``cfg.robot.adapter`` or fall back to env-name selection."""
    adapter_cfg = cfg_select(cfg, "robot.adapter", None)
    target = cfg_select(adapter_cfg, "_target_", None)
    if target:
        from hydra.utils import instantiate

        return instantiate(adapter_cfg, _recursive_=False)
    env_name = cfg_select(cfg, "env.name", None)
    return adapter_for_env(env_name)


__all__ = [
    "RobotAdapter",
    "RealYamAdapter",
    "adapter_for_env",
    "get_robot_adapter",
]
