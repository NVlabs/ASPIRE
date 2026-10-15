"""Portal RPC client for one follower arm.

The arm servers are separate processes (see :mod:`cap_harness.yam_real.server`)
that own the CAN bus and run the 100 Hz hold loop. This module is the client
side and nothing more: it does not interpret observations, choose gains, or
retry. A dropped call surfaces to the caller, because the level-1 controller is
the layer that knows whether a missed tick matters.

``portal`` is imported lazily so the harness stays importable on a machine that
has never talked to a robot.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FollowerArmClient:
    """Portal client for one follower arm server."""

    def __init__(self, host: str, port: int, *, label: str = "follower") -> None:
        import portal

        self.host = str(host)
        self.port = int(port)
        self.label = str(label)
        self._client = portal.Client(f"{self.host}:{self.port}")

    def __repr__(self) -> str:
        return f"FollowerArmClient({self.label!r}, {self.host}:{self.port})"

    def get_observations(self) -> dict[str, np.ndarray]:
        """One arm's measured state. Blocking; no command is issued."""
        return self._client.get_observations().result()

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def get_health(self) -> dict[str, Any]:
        """Connection state, buffered-target age and any background fault."""
        return self._client.get_health().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Send a joint7 position target using the server's default gains."""
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict[str, Any]) -> None:
        """Send a joint7 target with explicit gains and gripper limits."""
        self._client.command_joint_state(joint_state)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


__all__ = ["FollowerArmClient"]
