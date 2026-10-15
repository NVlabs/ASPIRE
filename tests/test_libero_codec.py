from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, Pose, RobotAction, RobotState
from cap_harness.libero.codec import LiberoActionCodec


def _state(joints: np.ndarray, gripper: float) -> RobotState:
    return RobotState(
        joint_positions={"primary": joints},
        joint_velocities={"primary": np.zeros(7)},
        end_effector_poses={
            "primary": Pose(
                position=np.zeros(3),
                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                frame="robot_base",
            )
        },
        gripper_positions={"primary": gripper},
        base_frame="robot_base",
    )


def test_joint_deltas_are_frequency_scaled_and_clamped() -> None:
    low = np.array([-0.5] * 7 + [-0.8])
    high = np.array([0.5] * 7 + [0.8])
    codec = LiberoActionCodec(control_frequency=20.0, action_spec=(low, high))
    state = _state(np.zeros(7), gripper=0.25)
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=np.array([0.10, -0.10, 0.01, -0.01, 0.0, 0.025, -0.025]),
                gripper_position=0.75,
            )
        }
    )

    native = codec.encode(action, state)

    np.testing.assert_allclose(
        native,
        np.array([0.5, -0.5, 0.2, -0.2, 0.0, 0.5, -0.5, -0.5]),
    )


def test_omitted_gripper_preserves_current_position() -> None:
    codec = LiberoActionCodec(
        control_frequency=20.0,
        action_low=np.full(8, -1.0),
        action_high=np.full(8, 1.0),
    )
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=np.zeros(7),
            )
        }
    )

    native = codec.encode(action, _state(np.zeros(7), gripper=0.25))

    assert native[-1] == pytest.approx(0.5)


def test_codec_rejects_non_primary_or_invalid_native_dimensions() -> None:
    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        LiberoActionCodec(action_spec=(np.full(7, -1.0), np.full(7, 1.0)))

    codec = LiberoActionCodec(action_spec=(np.full(8, -1.0), np.full(8, 1.0)))
    action = RobotAction(
        arms={
            "secondary": ArmCommand(mode="joint_position", target=np.zeros(7)),
        }
    )
    with pytest.raises(ValueError, match="exactly one arm named 'primary'"):
        codec.encode(action, np.zeros(7), current_gripper_position=1.0)
