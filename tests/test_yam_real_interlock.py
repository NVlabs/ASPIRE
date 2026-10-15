"""The motion interlock: what an unauthorized adapter is allowed to do.

Every assertion here is about **commands that reached the arms**, not about
return values. A refusal that still emitted an RPC would satisfy a test of the
returned error code while leaving the robot moving, so the station counts every
command it receives and the tests check that count.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.station import build_sim_station


class CountingStation:
    """A sim station that records every command reaching either arm."""

    def __init__(self) -> None:
        self.env = build_sim_station(realtime=False)
        self.commands = 0
        for arm in self.env._arms.values():
            original = arm.command_joint_state

            def counted(state, _original=original):
                self.commands += 1
                return _original(state)

            arm.command_joint_state = counted

    def rearm(self) -> None:
        """Restore the start pose and zero the counter between tests."""
        home = self.env.config.arms["left"].home_joints
        self.env._plant.set_joint_positions(home, home)
        self.commands = 0

    def close(self) -> None:
        self.env.close()


@pytest.fixture(scope="module")
def _plant():
    # Loading the station model twice per test dominates runtime, so the plant is
    # built once and re-armed between tests.
    plant = CountingStation()
    yield plant
    plant.close()


@pytest.fixture
def station(_plant):
    _plant.rearm()
    return _plant


def _action(state) -> RobotAction:
    return RobotAction(
        arms={
            side: ArmCommand(
                "joint_position", state.joint_positions[side], 0.5, embodiment="yam_real"
            )
            for side in ("left", "right")
        }
    )


def _trajectory(state) -> Trajectory:
    return Trajectory(
        joint_positions=np.tile(state.joint_positions["left"], (3, 1)),
        dt_s=1.0 / 60.0,
        joint_names=[f"joint_{index + 1}" for index in range(6)],
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm="left",
        embodiment="yam_real",
    )


def test_construction_issues_no_command(station):
    adapter = YamRealAdapter(station.env)
    assert adapter.physical_motion_authorized is False
    assert station.commands == 0
    assert adapter.command_rpc_count == 0


def test_reads_work_without_authorization(station):
    adapter = YamRealAdapter(station.env)

    state = adapter.get_robot_state()
    observation = adapter.get_observation()
    metadata = adapter.get_controller_metadata()

    assert set(state.joint_positions) == {"left", "right"}
    assert observation.cameras
    assert metadata["physical_motion_authorized"] is False
    assert station.commands == 0
    assert adapter.command_rpc_count == 0


@pytest.mark.parametrize("operation", ["step", "execute_trajectory", "set_gripper", "go_home"])
def test_motion_is_refused_and_issues_zero_commands(station, operation):
    adapter = YamRealAdapter(station.env)
    state = adapter.get_robot_state()
    calls = {
        "step": lambda: adapter.step(_action(state)),
        "execute_trajectory": lambda: adapter.execute_trajectory(_trajectory(state)),
        "set_gripper": lambda: adapter.set_gripper(0.9, arm="left"),
        "go_home": lambda: adapter.go_home(),
    }

    result = calls[operation]()

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert result.error.recoverable is False
    assert station.commands == 0
    assert adapter.command_rpc_count == 0


def test_set_grippers_is_refused_and_issues_zero_commands(station):
    adapter = YamRealAdapter(station.env)

    result = adapter.set_grippers({"left": 0.2, "right": 0.8})

    assert result.ok is False
    assert result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert station.commands == 0
    assert adapter.command_rpc_count == 0


def test_reset_does_not_bypass_the_gate(station):
    """The one motion the shared runner triggers on its own.

    ``run.py`` calls ``reset`` before every program, so if homing were exempt a
    real station would move as a side effect of loading a program.
    """
    adapter = YamRealAdapter(station.env)
    before = adapter.get_robot_state().joint_positions["left"].copy()

    observation = adapter.reset(seed=7)

    assert observation.cameras
    assert station.commands == 0
    assert adapter.command_rpc_count == 0
    np.testing.assert_allclose(adapter.get_robot_state().joint_positions["left"], before, atol=1e-9)


def test_every_refusal_leaves_the_counter_at_zero(station):
    """A sequence of refusals must not accumulate phantom commands."""
    adapter = YamRealAdapter(station.env)
    state = adapter.get_robot_state()

    adapter.reset()
    adapter.step(_action(state))
    adapter.execute_trajectory(_trajectory(state))
    adapter.set_gripper(0.9, arm="left")
    adapter.set_grippers({"left": 0.1, "right": 0.1})
    adapter.go_home()

    assert station.commands == 0
    assert adapter.command_rpc_count == 0


def test_authorized_motion_commands_the_arms(station):
    """The negative tests would pass against a broken adapter that never moves."""
    adapter = YamRealAdapter(station.env, allow_physical_motion=True)

    result = adapter.go_home(duration=0.2)

    assert result.ok is True
    assert station.commands > 0
    assert adapter.command_rpc_count == 1


def test_command_count_tracks_submissions_not_attempts(station):
    """Refused attempts are invisible to the counter; accepted ones are not."""
    denied = YamRealAdapter(station.env)
    denied.go_home()
    denied.go_home()
    assert denied.command_rpc_count == 0

    allowed = YamRealAdapter(station.env, allow_physical_motion=True)
    allowed.go_home(duration=0.2)
    allowed.set_gripper(0.5, arm="left")
    assert allowed.command_rpc_count == 2
