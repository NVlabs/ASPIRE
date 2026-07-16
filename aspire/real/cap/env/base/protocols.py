"""Protocol definitions for the CAP env layer.

All protocols use ``typing.Protocol`` for structural subtyping — env
implementations conform by having the right methods, no inheritance needed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Core env contract (required)
# ---------------------------------------------------------------------------


@runtime_checkable
class EnvProtocol(Protocol):
    """Core contract that every env must implement.

    Provides joint-level control (for policy execution and trajectory playback),
    state observation, and rendering.
    """

    def step(self) -> None:
        """Advance physics by one control tick."""
        ...

    def get_arm_observation(self, side: str) -> dict[str, np.ndarray]:
        """Read state for one arm.

        Must return at minimum::

            {"joint_pos": ndarray(dof,), "gripper_pos": ndarray(1,)}

        Should also return (when available)::

            {"ee_pos": ndarray(3,), "ee_quat": ndarray(4,)}  # world frame
        """
        ...

    def command_arm(self, side: str, cmd: dict) -> None:
        """Write commanded joint positions for one arm.

        ``cmd["pos"]`` is shape ``(dof + 1,)`` — arm joints + gripper (0–1).
        """
        ...

    def render_rgb(self, camera_name: str) -> np.ndarray:
        """Render RGB image (H x W x 3, uint8)."""
        ...

    def render_depth(self, camera_name: str) -> np.ndarray:
        """Render depth image (H x W, float32)."""
        ...

    def get_camera_intrinsics(self, camera_name: str) -> list[float]:
        """Return [fx, fy, cx, cy]."""
        ...

    def get_camera_extrinsics(self, camera_name: str) -> dict:
        """Return {"position": [x,y,z], "rotation": [[...],...]}}."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# EE-level motion control (optional)
# ---------------------------------------------------------------------------


@runtime_checkable
class EefControlProtocol(Protocol):
    """Per-tick EE control — the env computes one tick's action, the control loop steps physics.

    This is the key abstraction for modularity: each env implements its own
    EE control law (OSC delta, pinocchio IK, cuRobo, etc.) but never steps
    physics itself. The control loop in cap_server calls ``compute_eef_action``
    every tick and handles stepping.

    Coordinates are in world frame.
    """

    def compute_eef_action(
        self,
        side: str,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        gripper: float | None = None,
    ) -> dict:
        """Compute the command for ONE control tick to move toward an EE target.

        Returns a dict suitable for ``command_arm(side, cmd)`` — typically
        ``{"pos": np.array([...joint targets..., gripper])}``.

        Called by the control loop every tick while an EE target is active.
        Must NOT call ``step()`` — the control loop does that.
        """
        ...




# ---------------------------------------------------------------------------
# Scene management (optional — MuJoCo station envs)
# ---------------------------------------------------------------------------


@runtime_checkable
class SceneProtocol(Protocol):
    """Scene object management for MuJoCo-based envs."""

    def setup_scene(self, name: str) -> dict:
        ...

    def clear_table(self) -> dict:
        ...

    def get_object_positions(self) -> dict:
        ...

    def get_scenes(self) -> dict:
        ...

    def set_body_pose(
        self,
        name: str,
        pos: list[float],
        quat_wxyz: list[float],
        gravity_comp: bool = True,
    ) -> dict:
        ...


# ---------------------------------------------------------------------------
# Structured tasks (optional — RoboCasa, etc.)
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskProtocol(Protocol):
    """Episodic tasks with rewards and success checking."""

    def reset_env(self) -> dict:
        """Reset for a new episode. Returns ``{"ok": True, ...}``."""
        ...

    def get_last_reward(self) -> float:
        """Reward from the most recent step."""
        ...

    def get_task_info(self) -> dict:
        """Returns ``{"done": bool, "reward": float, "success": bool, ...}``."""
        ...
