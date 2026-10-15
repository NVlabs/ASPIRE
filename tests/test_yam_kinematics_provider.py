"""The in-process mink IK provider, exercised against the MuJoCo station.

This solver exists because the contact segment of a pick cannot be planned:
descending onto an object means entering the observed point cloud, so a
collision-aware planner refuses it by design. These tests pin the property that
makes that split safe -- the provider must not report success for a pose it did
not actually reach, because the caller uses that answer to decide whether to
move.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import MotionStrategy, Pose
from cap_harness.errors import ErrorCode
from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.station import build_sim_station

READY = np.array([-0.30, 1.35, 1.60, -0.80, 0.30, -0.25])


@pytest.fixture(scope="module")
def env():
    station = build_sim_station(realtime=False)
    yield station
    station.close()


@pytest.fixture
def state(env):
    env._plant.set_joint_positions(READY, READY)
    return YamRealAdapter(env, allow_physical_motion=False).get_robot_state()


@pytest.fixture(scope="module")
def provider(env):
    return YamKinematicsIKProvider(env.kin)


# -- the contract the caller relies on -------------------------------------


def test_solves_the_arms_own_current_pose(provider, state):
    """The weakest possible goal: where the arm already is. It must converge."""
    for arm in ("left", "right"):
        result = provider.solve_ik(state.end_effector_poses[arm], state, arm=arm)

        assert result.ok is True, result.error
        assert result.arm == arm
        assert result.joint_positions.shape == (6,)
        np.testing.assert_allclose(result.joint_positions, state.joint_positions[arm], atol=0.02)


def test_an_unreachable_pose_is_reported_before_anything_moves(provider, state):
    """Mink returns its best effort regardless of convergence.

    Without the forward-kinematics check the provider would hand back that best
    effort as a success, and the caller would drive the arm at a pose it cannot
    reach and only discover the miss on arrival.
    """
    far = Pose(position=[5.0, 5.0, 5.0], quaternion_wxyz=[1.0, 0.0, 0.0, 0.0], frame="world")

    result = provider.solve_ik(far, state, arm="right")

    assert result.ok is False
    assert result.error.code is ErrorCode.IK_UNREACHABLE
    assert result.error.details["position_error_m"] > 0.01


def test_reports_the_residual_it_converged_to(provider, state):
    """Diagnostics carry the achieved error, so a caller can judge the margin."""
    result = provider.solve_ik(state.end_effector_poses["right"], state, arm="right")

    assert result.diagnostics["solver"] == "mink"
    assert result.diagnostics["position_error_m"] <= 0.01


def test_solving_one_arm_leaves_the_other_where_it_is(provider, state):
    """The solver is bimanual; the untouched arm is pinned to its measured pose.

    Otherwise asking for a right-arm goal would quietly propose a new left-arm
    configuration too, and executing the result would move an arm the caller
    never mentioned.
    """
    target = state.end_effector_poses["right"]
    shifted = Pose(
        position=np.asarray(target.position) + np.array([0.0, 0.0, -0.03]),
        quaternion_wxyz=target.quaternion_wxyz,
        frame=target.frame,
    )

    result = provider.solve_ik(shifted, state, arm="right")

    assert result.ok is True, result.error
    # Only the right arm's six joints come back, and they differ from the seed.
    assert result.joint_positions.shape == (6,)
    assert not np.allclose(result.joint_positions, state.joint_positions["right"], atol=1e-6)


def test_a_state_without_both_arms_is_a_typed_rejection(provider, state):
    """A single-arm state cannot pin the other arm, so it is refused, not guessed."""
    from dataclasses import replace

    one_arm = replace(
        state,
        joint_positions={"right": state.joint_positions["right"]},
        joint_velocities={"right": state.joint_velocities["right"]},
        end_effector_poses={"right": state.end_effector_poses["right"]},
        gripper_positions={"right": state.gripper_positions["right"]},
        joint_names={"right": state.joint_names["right"]},
    )

    result = provider.solve_ik(state.end_effector_poses["right"], one_arm, arm="right")

    assert result.ok is False
    assert result.error.code is ErrorCode.INVALID_REQUEST


# -- the strategy that selects it ------------------------------------------


def test_mink_is_a_selectable_ik_backend():
    """The contract has to admit it, or no program can ask for it.

    This is the strategy the real station plans with: cuRobo carries the
    collision-aware transit from a Cartesian goal, mink solves the contact
    waypoints locally.
    """
    strategy = MotionStrategy(ik_solver="mink", pose_planner="curobo-integrated")

    assert strategy.ik_solver == "mink"
    assert strategy.pose_planner == "curobo-integrated"


def test_an_unknown_ik_backend_is_still_rejected():
    with pytest.raises(ValueError, match="ik_solver must be"):
        MotionStrategy(ik_solver="not-a-solver")
