"""Conversion between portable joint commands and LIBERO native actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cap_harness.contracts import JOINT_DIMENSION, RobotAction, RobotState

LIBERO_ACTION_DIMENSION = JOINT_DIMENSION + 1


class LiberoActionCodec:
    """Encode one primary-arm joint target for LIBERO's delta controller."""

    def __init__(
        self,
        control_frequency: float = 20.0,
        action_spec: Sequence[Any] | None = None,
        *,
        action_low: Any | None = None,
        action_high: Any | None = None,
    ) -> None:
        if isinstance(control_frequency, bool | np.bool_):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("control_frequency must be a positive finite number")
        try:
            frequency = float(control_frequency)
        except (TypeError, ValueError) as exc:
            raise ValueError("control_frequency must be a positive finite number") from exc
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("control_frequency must be a positive finite number")
        if action_spec is not None and (action_low is not None or action_high is not None):
            raise TypeError("pass action_spec or action_low/action_high, not both")
        if (action_low is None) != (action_high is None):
            raise TypeError("action_low and action_high must be provided together")

        self.control_frequency = frequency
        self._bounds: tuple[np.ndarray, np.ndarray] | None = None
        if action_spec is not None:
            self._bounds = self.validate_action_spec(action_spec)
        elif action_low is not None:
            self._bounds = self.validate_action_spec((action_low, action_high))

    @staticmethod
    def _finite_vector(value: Any, name: str, dimension: int) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector") from exc
        if vector.shape != (dimension,):
            raise ValueError(f"{name} must have shape ({dimension},), got {vector.shape}")
        if not bool(np.all(np.isfinite(vector))):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    @classmethod
    def validate_action_spec(cls, action_spec: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(action_spec, str | bytes) or not isinstance(action_spec, Sequence):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("action_spec must be a (low, high) pair")
        if len(action_spec) != 2:
            raise ValueError("action_spec must contain exactly (low, high)")
        low = cls._finite_vector(action_spec[0], "action_spec low", LIBERO_ACTION_DIMENSION)
        high = cls._finite_vector(action_spec[1], "action_spec high", LIBERO_ACTION_DIMENSION)
        if bool(np.any(low > high)):
            raise ValueError("action_spec lower bounds must not exceed upper bounds")
        return low.copy(), high.copy()

    @staticmethod
    def _normalized_gripper(value: Any, name: str) -> float:
        if isinstance(value, bool | np.bool_):
            # Preserve the invalid-data ValueError contract.
            raise ValueError(f"{name} must be in [0, 1]")
        try:
            gripper = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be in [0, 1]") from exc
        if not np.isfinite(gripper) or gripper < 0.0 or gripper > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return gripper

    @staticmethod
    def normalized_gripper_to_native(position: Any) -> float:
        """Map 0=closed, 1=open to LIBERO's 1=closed, -1=open convention."""
        normalized = LiberoActionCodec._normalized_gripper(position, "gripper position")
        return 1.0 - 2.0 * normalized

    def _resolve_bounds(self, action_spec: Sequence[Any] | None) -> tuple[np.ndarray, np.ndarray]:
        if action_spec is not None:
            return self.validate_action_spec(action_spec)
        if self._bounds is None:
            raise ValueError("LIBERO action bounds are required")
        return self._bounds

    def encode(
        self,
        action: RobotAction,
        current_state: RobotState | Any | None = None,
        action_spec: Sequence[Any] | None = None,
        *,
        current_joint_positions: Any | None = None,
        current_gripper_position: Any | None = None,
    ) -> np.ndarray:
        """Encode a portable action using current state and reported native bounds."""
        if not isinstance(action, RobotAction):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("action must be a RobotAction")
        if set(action.arms) != {"primary"}:
            raise ValueError("LIBERO requires exactly one arm named 'primary'")
        command = action.arms["primary"]
        if command.mode != "joint_position":
            raise ValueError("LIBERO supports only joint_position ArmCommand values")

        if current_joint_positions is not None and current_state is not None:
            raise TypeError("pass current_state or current_joint_positions, not both")
        if isinstance(current_state, RobotState):
            if "primary" not in current_state.joint_positions:
                raise ValueError("current RobotState has no 'primary' arm")
            current_joints = current_state.joint_positions["primary"]
            state_gripper = current_state.gripper_positions.get("primary")
        else:
            current_joints = (
                current_joint_positions if current_joint_positions is not None else current_state
            )
            state_gripper = current_gripper_position
        if current_joints is None:
            raise ValueError("current primary joint positions are required")
        current = self._finite_vector(
            current_joints, "current primary joint positions", JOINT_DIMENSION
        )
        target = self._finite_vector(command.target, "joint target", JOINT_DIMENSION)

        if command.gripper_position is None:
            gripper = state_gripper
            if gripper is None and isinstance(current_state, RobotState):
                gripper = current_state.gripper_positions.get("primary")
            if gripper is None:
                raise ValueError(
                    "current_gripper_position is required when the command omits gripper_position"
                )
        else:
            gripper = command.gripper_position
        gripper = self._normalized_gripper(gripper, "gripper position")

        low, high = self._resolve_bounds(action_spec)
        native = np.empty(LIBERO_ACTION_DIMENSION, dtype=np.float64)
        native[:JOINT_DIMENSION] = (target - current) * self.control_frequency
        native[-1] = self.normalized_gripper_to_native(gripper)
        return np.clip(native, low, high)


__all__ = ["LIBERO_ACTION_DIMENSION", "LiberoActionCodec"]
