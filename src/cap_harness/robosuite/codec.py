"""Portable RobotAction to Robosuite 1.4 joint-controller actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cap_harness.contracts import RobotAction, RobotState


class RobosuiteActionCodec:
    """Encode all arms into one native action for one synchronized simulator tick."""

    def __init__(
        self,
        arms: Sequence[str],
        *,
        control_frequency: float = 20.0,
        action_spec: Sequence[Any] | None = None,
        arm_action_dimensions: Sequence[int] | None = None,
    ) -> None:
        self.arms = tuple(arms)
        if self.arms not in (("primary",), ("primary", "secondary")):
            raise ValueError("arms must be ('primary',) or ('primary', 'secondary')")
        self.control_frequency = float(control_frequency)
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        self.arm_action_dimensions = tuple(arm_action_dimensions or (8,) * len(self.arms))
        if len(self.arm_action_dimensions) != len(self.arms) or any(
            dimension not in (7, 8) for dimension in self.arm_action_dimensions
        ):
            raise ValueError("each Robosuite arm action dimension must be 7 or 8")
        self.dimension = sum(self.arm_action_dimensions)
        self._bounds = self.validate_action_spec(action_spec) if action_spec is not None else None

    def validate_action_spec(self, action_spec: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        if len(action_spec) != 2:
            raise ValueError("action_spec must contain (low, high)")
        low = np.asarray(action_spec[0], dtype=np.float64)
        high = np.asarray(action_spec[1], dtype=np.float64)
        if low.shape != (self.dimension,) or high.shape != (self.dimension,):
            raise ValueError(f"Robosuite action bounds must have shape ({self.dimension},)")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low > high):
            raise ValueError("Robosuite action bounds must be finite and ordered")
        return low.copy(), high.copy()

    @staticmethod
    def normalized_gripper_to_native(position: float) -> float:
        value = float(position)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("gripper position must be in [0, 1]")
        return 1.0 - 2.0 * value

    def encode(
        self,
        action: RobotAction,
        current_state: RobotState,
        *,
        joint_targets: dict[str, np.ndarray] | None = None,
        gripper_targets: dict[str, float] | None = None,
        action_spec: Sequence[Any] | None = None,
    ) -> np.ndarray:
        if not isinstance(action, RobotAction) or not isinstance(current_state, RobotState):
            raise ValueError("action and current_state must be typed contract values")
        unknown = set(action.arms) - set(self.arms)
        if unknown:
            raise ValueError(f"unknown Robosuite arms: {sorted(unknown)}")
        targets = gripper_targets or {}
        arm_targets = joint_targets or {}
        native: list[float] = []
        for arm, dimension in zip(self.arms, self.arm_action_dimensions):
            current = current_state.joint_positions[arm]
            command = action.arms.get(arm)
            target = arm_targets.get(arm, current) if command is None else command.target
            gripper = (
                targets.get(arm, current_state.gripper_positions[arm])
                if command is None or command.gripper_position is None
                else command.gripper_position
            )
            native.extend(((target - current) * self.control_frequency).tolist())
            if dimension == 8:
                native.append(self.normalized_gripper_to_native(gripper))
        encoded = np.asarray(native, dtype=np.float64)
        bounds = self._bounds if action_spec is None else self.validate_action_spec(action_spec)
        if bounds is None:
            raise ValueError("Robosuite action bounds are required")
        return np.clip(encoded, bounds[0], bounds[1])


__all__ = ["RobosuiteActionCodec"]
