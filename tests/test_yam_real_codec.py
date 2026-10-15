"""Conversions between the YAM plant and the shared contracts.

These are the conversions that fail silently when they are wrong: a swapped
quaternion produces a pose that is only wrong in orientation, and a mislabelled
camera frame produces a point cloud that is confidently wrong in position.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction
from cap_harness.yam_real import codec
from cap_harness.yam_real.station import build_sim_station


@pytest.fixture(scope="module")
def env():
    station = build_sim_station(realtime=False)
    yield station
    station.close()


# -- quaternion order ------------------------------------------------------


def test_quaternion_round_trip():
    xyzw = np.array([0.1, 0.2, 0.3, 0.927], dtype=np.float64)
    np.testing.assert_allclose(codec.quat_wxyz_to_xyzw(codec.quat_xyzw_to_wxyz(xyzw)), xyzw)


def test_quaternion_moves_the_scalar_component():
    """The whole point of the conversion: w moves between last and first."""
    xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(codec.quat_xyzw_to_wxyz(xyzw), [1.0, 0.0, 0.0, 0.0])


def test_rotation_to_wxyz_matches_identity():
    np.testing.assert_allclose(codec.rotation_to_wxyz(np.eye(3)), [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "axis,angle",
    [((1, 0, 0), np.pi / 2), ((0, 1, 0), np.pi / 3), ((0, 0, 1), -np.pi / 4)],
)
def test_rotation_to_wxyz_matches_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    quat = codec.rotation_to_wxyz(rotation)

    expected = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    # q and -q are the same rotation.
    assert np.allclose(quat, expected, atol=1e-8) or np.allclose(quat, -expected, atol=1e-8)


# -- state -----------------------------------------------------------------


def test_robot_state_reports_six_dof_per_arm(env):
    state = codec.robot_state(env)

    assert state.embodiment == "yam_real"
    assert set(state.joint_positions) == {"left", "right"}
    for side in ("left", "right"):
        assert state.joint_positions[side].shape == (6,)
        assert state.joint_velocities[side].shape == (6,)
        assert state.end_effector_poses[side].frame == env.config.base_frame


def test_robot_state_truncates_motor_width_velocity(env):
    """The plant reports velocity at motor width (7); the contract wants 6."""
    raw = env.get_observations("left")
    assert raw["joint_vel"].shape == (6,), "env should have split the gripper entry off"
    assert codec.robot_state(env).joint_velocities["left"].shape == (6,)


def test_gripper_is_not_a_seventh_joint(env):
    state = codec.robot_state(env)
    assert isinstance(state.gripper_positions["left"], float)
    assert state.joint_positions["left"].shape == (6,)


# -- camera ----------------------------------------------------------------


def test_camera_observation_is_labelled_with_its_own_frame(env):
    """Not "base".

    ``mask_to_point_cloud`` returns the cloud unchanged when the requested target
    frame equals the observation's frame. Labelling a camera-frame image "base"
    therefore skips the extrinsics entirely and relabels camera coordinates as
    base coordinates, which aims every downstream motion at the camera mount.
    """
    observation = codec.camera_observation(env)

    assert observation is not None
    assert observation.frame != env.config.base_frame
    assert observation.frame == f"{env.config.camera.role}_optical"
    assert observation.camera_pose.frame == env.config.base_frame


def test_observation_carries_camera_and_state(env):
    observation = codec.observation(env)

    assert set(observation.cameras) == {env.config.camera.role, *env.config.aux_cameras}
    assert observation.robot_state.embodiment == "yam_real"
    # Only the calibrated camera knows where it is. The image-only ones carry a
    # self-referential frame precisely so that projecting through them raises
    # instead of quietly returning camera coordinates labelled as metres.
    assert observation.cameras[env.config.camera.role].camera_pose.frame == env.config.base_frame
    for role in env.config.aux_cameras:
        assert observation.cameras[role].camera_pose.frame == f"{role}_optical"


# -- actions ---------------------------------------------------------------


def test_arm_command_targets_splits_gripper_from_joints():
    action = RobotAction(
        arms={
            "left": ArmCommand("joint_position", np.zeros(6), 0.25, embodiment="yam_real"),
            "right": ArmCommand("joint_position", np.ones(6), None, embodiment="yam_real"),
        }
    )

    joints, grippers = codec.arm_command_targets(action)

    assert joints["left"].shape == (6,)
    assert grippers["left"] == 0.25
    # None means "leave it alone", which is distinct from 0.0 (fully closed).
    assert grippers["right"] is None


def test_arm_command_targets_rejects_a_non_command():
    action = type("Fake", (), {"arms": {"left": object()}})()
    with pytest.raises(ValueError, match="ArmCommand"):
        codec.arm_command_targets(action)
