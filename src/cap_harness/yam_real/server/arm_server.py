"""Portal RPC server for one follower arm.

The wire surface the harness client expects, and nothing more: read observations,
buffer a target. It is a follower only -- there is no leader/teleop mode here,
and no path that moves the arm except through a validated buffered target.

The server owns the CAN bus for its arm, so exactly one of these runs per arm.
It outlives any harness run: a client disconnecting must not stop the hold loop,
because stopping it would drop the arm.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from cap_harness.yam_real.server.yam_robot import YamRobot


class FollowerArmServer:
    """Serve one :class:`YamRobot` over portal."""

    def __init__(self, robot: YamRobot, port: int) -> None:
        self.robot = robot
        self.port = int(port)
        self._server: Any = None

    def _get_observations(self, _request: Any = None) -> dict[str, np.ndarray]:
        return self.robot.get_observations()

    def _get_joint_pos(self, _request: Any = None) -> np.ndarray:
        return self.robot.get_joint_pos()

    def _get_health(self, _request: Any = None) -> dict[str, Any]:
        return self.robot.get_health()

    def _command_joint_state(self, joint_state: dict[str, Any]) -> dict[str, Any]:
        return self.robot.command_joint_state(joint_state)

    def _command_joint_pos(self, joint_pos: np.ndarray) -> dict[str, Any]:
        return self.robot.command_joint_pos(joint_pos)

    def _stop(self, _request: Any = None) -> dict[str, Any]:
        self.robot.stop()
        return {"stopped": True}

    def serve(self, *, calibrate_gripper: bool = False) -> None:
        """Start the hold loop, then supervise the RPC server until it stops.

        ``errors=False`` keeps one failing RPC from taking the server down: a
        client asking for something impossible must not cost the arm its hold
        loop. The supervision loop watches the portal worker and socket threads
        so a dead transport surfaces as an error rather than a server that is
        still listening but answers nothing.
        """
        import portal

        self.robot.start()
        if calibrate_gripper:
            # After start(), so the bus is connected exactly once. Doing this
            # before serve() would open a second CAN controller on the same
            # interface, since start() connects the bus itself.
            measured = self.robot.calibrate_gripper()
            print(f"[yam-server] {self.robot.side} gripper travel: {measured}", flush=True)
        server = portal.Server(self.port, errors=False)
        server.bind("get_observations", self._get_observations)
        server.bind("get_joint_pos", self._get_joint_pos)
        server.bind("get_health", self._get_health)
        server.bind("command_joint_state", self._command_joint_state)
        server.bind("command_joint_pos", self._command_joint_pos)
        server.bind("stop", self._stop)
        self._server = server
        server.start(block=False)
        print(
            f"[yam-server] {self.robot.side} arm on "
            f"{self.robot.config.arms[self.robot.side].can_interface} "
            f"serving :{self.port}",
            flush=True,
        )
        try:
            while True:
                if not server.loop.running:
                    raise RuntimeError(
                        f"portal worker loop exited (exitcode={server.loop.exitcode})"
                    )
                socket_thread = server.socket.thread
                if not socket_thread.running:
                    error = server.socket.error
                    if error is not None:
                        raise RuntimeError("portal socket thread crashed") from error
                    raise RuntimeError("portal socket thread exited unexpectedly")
                time.sleep(0.2)
        finally:
            if server.running:
                server.close(timeout=1.0)
            self.robot.close()


__all__ = ["FollowerArmServer"]
