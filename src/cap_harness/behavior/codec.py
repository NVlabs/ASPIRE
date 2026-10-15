"""Bookkeeping for the R1 Pro joint-target vector behind ``robot.q_to_action``.

The adapter keeps one full joint-position target (28 values for R1 Pro) and rewrites slices of
it: arm slices from ``RobotAction`` commands, finger slices from normalized gripper positions,
base and torso slices from the embodiment extensions. Every simulator tick then sends
``robot.q_to_action(target)``. This module owns the slice arithmetic so it can be unit tested
without OmniGibson.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from cap_harness.contracts import RobotAction

NATIVE_ARMS: Mapping[str, str] = {"primary": "left", "secondary": "right"}
"""Public arm names to R1 Pro arm names."""


def _index_array(values: Sequence[int], *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have {length} entries, got {array.size}")
    if np.unique(array).size != array.size or np.any(array < 0):
        raise ValueError(f"{name} must be unique non-negative joint indices")
    return array


@dataclass(frozen=True)
class R1ProJointLayout:
    """Joint indices into the robot's full joint vector, read once from the live robot."""

    joint_count: int
    base: np.ndarray
    trunk: np.ndarray
    arms: Mapping[str, np.ndarray]
    grippers: Mapping[str, np.ndarray]
    gripper_open: Mapping[str, np.ndarray]
    gripper_closed: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive")
        object.__setattr__(self, "base", _index_array(self.base, name="base", length=6))
        object.__setattr__(self, "trunk", _index_array(self.trunk, name="trunk"))
        arms = {
            arm: _index_array(idx, name=f"arm {arm}", length=7) for arm, idx in self.arms.items()
        }
        grippers = {
            arm: _index_array(idx, name=f"gripper {arm}") for arm, idx in self.grippers.items()
        }
        if set(arms) != set(NATIVE_ARMS.values()) or set(grippers) != set(arms):
            raise ValueError("layout must describe the left and right arms and grippers")
        opens = {arm: np.asarray(self.gripper_open[arm], dtype=np.float64) for arm in arms}
        closed = {
            arm: np.asarray(
                self.gripper_closed.get(arm, np.zeros_like(opens[arm])), dtype=np.float64
            )
            for arm in arms
        }
        for arm in arms:
            if opens[arm].shape != grippers[arm].shape or closed[arm].shape != grippers[arm].shape:
                raise ValueError(f"gripper limits for {arm!r} must match its joint count")
            if np.any(opens[arm] <= closed[arm]):
                raise ValueError(f"gripper open limits for {arm!r} must exceed closed limits")
        used = np.concatenate([self.base, self.trunk, *arms.values(), *grippers.values()])
        if used.max() >= self.joint_count or np.unique(used).size != used.size:
            raise ValueError("joint groups overlap or exceed joint_count")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "grippers", grippers)
        object.__setattr__(self, "gripper_open", opens)
        object.__setattr__(self, "gripper_closed", closed)

    @property
    def base_xy_yaw(self) -> np.ndarray:
        """Indices of the base x, y and yaw virtual joints (order x, y, z, rx, ry, rz)."""
        return self.base[[0, 1, 5]]


class BehaviorActionCodec:
    """Rewrite slices of the full joint target from public commands."""

    def __init__(self, layout: R1ProJointLayout) -> None:
        self.layout = layout

    @staticmethod
    def native_arm(arm: str) -> str:
        if arm in NATIVE_ARMS:
            return NATIVE_ARMS[arm]
        if arm in NATIVE_ARMS.values():
            return arm
        raise ValueError(f"unknown arm {arm!r}")

    def arm_joints(self, target: np.ndarray, arm: str) -> np.ndarray:
        return np.asarray(target, dtype=np.float64)[self.layout.arms[self.native_arm(arm)]]

    def gripper_position(self, joint_positions: np.ndarray, arm: str) -> float:
        """Normalized gripper opening: 0 = closed, 1 = open, measured from the finger joints."""
        native = self.native_arm(arm)
        fingers = np.asarray(joint_positions, dtype=np.float64)[self.layout.grippers[native]]
        opened = self.layout.gripper_open[native]
        closed = self.layout.gripper_closed[native]
        fraction = (fingers - closed) / (opened - closed)
        return float(np.clip(np.mean(fraction), 0.0, 1.0))

    @staticmethod
    def validate_gripper(position: float) -> float:
        value = float(position)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("gripper position must be in [0, 1]")
        return value

    def apply_gripper(self, target: np.ndarray, arm: str, position: float) -> np.ndarray:
        native = self.native_arm(arm)
        value = self.validate_gripper(position)
        result = np.array(target, dtype=np.float64, copy=True)
        closed = self.layout.gripper_closed[native]
        opened = self.layout.gripper_open[native]
        result[self.layout.grippers[native]] = closed + value * (opened - closed)
        return result

    def apply_arm(self, target: np.ndarray, arm: str, joints: np.ndarray) -> np.ndarray:
        native = self.native_arm(arm)
        values = np.asarray(joints, dtype=np.float64).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError(f"arm {arm!r} target must be seven finite joint values")
        result = np.array(target, dtype=np.float64, copy=True)
        result[self.layout.arms[native]] = values
        return result

    def apply_action(self, target: np.ndarray, action: RobotAction) -> np.ndarray:
        """Return a new joint target with every commanded arm (and gripper) written in."""
        result = np.array(target, dtype=np.float64, copy=True)
        for arm, command in action.arms.items():
            result = self.apply_arm(result, arm, command.target)
            if command.gripper_position is not None:
                result = self.apply_gripper(result, arm, command.gripper_position)
        return result

    def apply_trunk(self, target: np.ndarray, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if values.shape != self.layout.trunk.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"torso target must have {self.layout.trunk.size} finite values")
        result = np.array(target, dtype=np.float64, copy=True)
        result[self.layout.trunk] = values
        return result

    def apply_base(self, target: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        values = np.asarray([x, y, yaw], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("base target must be finite")
        result = np.array(target, dtype=np.float64, copy=True)
        result[self.layout.base_xy_yaw] = values
        return result


__all__ = ["NATIVE_ARMS", "BehaviorActionCodec", "R1ProJointLayout"]
