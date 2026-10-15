"""Calibration-only adapter to the existing YAM follower RPC contract."""

from __future__ import annotations

import time

import numpy as np

KP = np.array([80, 80, 80, 40, 10, 10, 10], dtype=float)
KD = np.array([5, 5, 5, 1.5, 1.5, 1.5, 0.5], dtype=float)


class ArmClient:
    """Preserve the current gripper target throughout board capture."""

    def __init__(self, host: str, port: int, *, client=None):
        import portal

        self._client = client if client is not None else portal.Client(f"{host}:{port}")
        self._gripper = None

    def get_joint_pos(self) -> np.ndarray:
        state = self._client.get_observations().result(timeout=3.0)
        joints = np.asarray(state["joint_pos"], dtype=float).reshape(6)
        grip = float(np.asarray(state["gripper_pos"]).reshape(1)[0])
        if not np.all(np.isfinite(joints)) or not np.isfinite(grip):
            raise ValueError("arm telemetry contains non-finite values")
        if self._gripper is None:
            self._gripper = float(np.clip(grip, 0, 1))
        return joints

    def command(self, joints: np.ndarray, *, gravity: bool = False) -> None:
        joints = np.asarray(joints, dtype=float).reshape(6)
        if not np.all(np.isfinite(joints)) or self._gripper is None:
            raise ValueError("read valid arm state before commanding calibration motion")
        kp = KP.copy()
        if gravity:
            kp[:6] = 0
        result = self._client.command_joint_state(
            {
                "pos": np.r_[joints, self._gripper],
                "kp": kp,
                "kd": KD,
            }
        ).result(timeout=3.0)
        if not result.get("accepted"):
            raise RuntimeError(f"arm server refused calibration command: {result}")

    def hold(self) -> None:
        """Restore normal stiffness at the measured pose; never home or release."""
        self.command(self.get_joint_pos())

    def move(self, target: np.ndarray, *, speed: float = 0.2, poll=lambda: None) -> None:
        """Bounded linear motion, followed by a measured convergence check."""
        target = np.asarray(target, dtype=float).reshape(6)
        if not np.all(np.isfinite(target)) or not np.isfinite(speed) or not 0 < speed <= 0.3:
            raise ValueError("use finite joint targets and a speed in (0, 0.3] rad/s")
        start = self.get_joint_pos()
        duration = max(0.5, float(np.max(np.abs(target - start))) / speed)
        began = time.monotonic()
        while True:
            poll()
            fraction = min(1.0, (time.monotonic() - began) / duration)
            self.command(start + fraction * (target - start))
            if fraction == 1.0:
                break
            time.sleep(0.02)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            poll()
            # Same gain-weighted settling criterion as the reference pipeline.
            if np.max(np.abs(self.get_joint_pos() - target) * (KP[:6] / KP[:6].max())) < 0.02:
                return
            time.sleep(0.02)
        raise RuntimeError("calibration move did not settle; no sample was accepted")

    def close(self) -> None:
        self._client.close()
