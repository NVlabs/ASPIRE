"""Openforge runtime environment layer.

This pruned copy keeps only the real bimanual YAM runtime needed by
``tmux/run_script.sh``.
"""

from cap.env.base import (
    EefControlProtocol,
    EnvProtocol,
    SceneProtocol,
    TaskProtocol,
)

__all__ = [
    "EefControlProtocol",
    "EnvProtocol",
    "SceneProtocol",
    "TaskProtocol",
    "create_env",
]


def create_env(env_name: str, viewer: bool = False, **kwargs):
    """Create the real-YAM env kept in this runtime subset."""
    _ = viewer
    parts = env_name.split(":")
    env_type = parts[0]

    if env_type == "yam-real":
        from cap.env.real_bimanual_yam.env import RealYamEnv

        enable_cameras = kwargs.pop("enable_cameras", True)
        return RealYamEnv(enable_cameras=enable_cameras)

    raise ValueError(
        f"Unsupported env in this openforge runtime subset: {env_name!r}. "
        "Only yam-real is included."
    )
