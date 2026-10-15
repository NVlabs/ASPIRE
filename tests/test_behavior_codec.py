from __future__ import annotations

from behavior_fakes import ARM_IDX, BASE_IDX, GRIPPER_IDX, GRIPPER_OPEN, JOINT_COUNT, TRUNK_IDX
import numpy as np
import pytest

from cap_harness.behavior.codec import BehaviorActionCodec, R1ProJointLayout
from cap_harness.contracts import ArmCommand, RobotAction


def _layout() -> R1ProJointLayout:
    return R1ProJointLayout(
        joint_count=JOINT_COUNT,
        base=BASE_IDX,
        trunk=TRUNK_IDX,
        arms=ARM_IDX,
        grippers=GRIPPER_IDX,
        gripper_open={arm: np.full(2, GRIPPER_OPEN) for arm in ARM_IDX},
    )


def test_layout_rejects_overlapping_or_short_groups() -> None:
    with pytest.raises(ValueError, match="overlap"):
        R1ProJointLayout(
            joint_count=JOINT_COUNT,
            base=BASE_IDX,
            trunk=np.arange(5, 9),
            arms=ARM_IDX,
            grippers=GRIPPER_IDX,
            gripper_open={arm: np.full(2, GRIPPER_OPEN) for arm in ARM_IDX},
        )
    with pytest.raises(ValueError, match="7 entries"):
        R1ProJointLayout(
            joint_count=JOINT_COUNT,
            base=BASE_IDX,
            trunk=TRUNK_IDX,
            arms={"left": np.arange(10, 16), "right": ARM_IDX["right"]},
            grippers=GRIPPER_IDX,
            gripper_open={arm: np.full(2, GRIPPER_OPEN) for arm in ARM_IDX},
        )


def test_codec_rewrites_only_the_commanded_slices() -> None:
    codec = BehaviorActionCodec(_layout())
    target = np.linspace(0.0, 1.0, JOINT_COUNT)
    action = RobotAction(
        {
            "primary": ArmCommand("joint_position", np.full(7, 0.5), 0.0, embodiment="behavior"),
            "secondary": ArmCommand("joint_position", np.full(7, -0.5), embodiment="behavior"),
        }
    )
    result = codec.apply_action(target, action)
    np.testing.assert_allclose(result[ARM_IDX["left"]], 0.5)
    np.testing.assert_allclose(result[ARM_IDX["right"]], -0.5)
    np.testing.assert_allclose(result[GRIPPER_IDX["left"]], 0.0)
    np.testing.assert_allclose(result[GRIPPER_IDX["right"]], target[GRIPPER_IDX["right"]])
    np.testing.assert_allclose(result[BASE_IDX], target[BASE_IDX])
    np.testing.assert_allclose(result[TRUNK_IDX], target[TRUNK_IDX])
    assert target[ARM_IDX["left"]][0] != 0.5, "the input target must not be mutated"


def test_gripper_normalization_round_trips_between_fingers_and_fraction() -> None:
    codec = BehaviorActionCodec(_layout())
    target = np.zeros(JOINT_COUNT)
    half = codec.apply_gripper(target, "secondary", 0.5)
    np.testing.assert_allclose(half[GRIPPER_IDX["right"]], GRIPPER_OPEN / 2)
    assert codec.gripper_position(half, "secondary") == pytest.approx(0.5)
    assert codec.gripper_position(half, "primary") == pytest.approx(0.0)
    assert codec.gripper_position(codec.apply_gripper(target, "left", 1.0), "left") == 1.0
    with pytest.raises(ValueError):
        codec.apply_gripper(target, "primary", 1.5)


def test_base_and_trunk_helpers_validate_shapes() -> None:
    codec = BehaviorActionCodec(_layout())
    target = np.zeros(JOINT_COUNT)
    moved = codec.apply_base(target, 1.0, 2.0, 0.5)
    np.testing.assert_allclose(moved[BASE_IDX[[0, 1, 5]]], [1.0, 2.0, 0.5])
    torso = codec.apply_trunk(target, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(torso[TRUNK_IDX], [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        codec.apply_trunk(target, [0.1, 0.2])
    with pytest.raises(ValueError):
        codec.apply_arm(target, "primary", np.full(6, 0.1))
    with pytest.raises(ValueError):
        codec.native_arm("tertiary")
    assert codec.native_arm("right") == "right"
    assert codec.native_arm("secondary") == "right"
