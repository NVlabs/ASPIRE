"""A station that REPORTS a refusal must reach the program as a typed failure.

Audit finding A2. The arm servers refuse two different ways: they raise across
the RPC boundary, which ``_plant_failure`` already classified, or they answer
``{"success": False, "reason": ...}``, which nothing handled.

That second case did not degrade gracefully. ``_validate_result_status`` rejects
a failed result carrying no ``ApiError``, so every command path raised
``ValueError: a failed result must carry an ApiError`` out of the adapter and
ended the run with a traceback -- the precise opposite of the class's own
promise that a station refusal arrives "as a result it could respond to".
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.station import build_sim_station

REFUSAL = {"success": False, "reason": "refused by plant"}


@pytest.fixture
def refusing_station():
    station = build_sim_station(realtime=False)
    station.execute_action_batch = lambda *a, **k: dict(REFUSAL)
    station.set_gripper = lambda *a, **k: dict(REFUSAL)
    station.go_home = lambda *a, **k: dict(REFUSAL)
    yield station
    station.close()


def _action(adapter):
    state = adapter.get_robot_state()
    return RobotAction(
        arms={
            side: ArmCommand(
                mode="joint_position",
                target=state.joint_positions[side],
                gripper_position=0.5,
                embodiment="yam_real",
            )
            for side in ("left", "right")
        }
    )


def _trajectory(adapter):
    state = adapter.get_robot_state()
    rows = [[float(v) for v in state.joint_positions["right"]]] * 3
    return Trajectory(
        joint_positions=rows,
        dt_s=1.0 / 30.0,
        joint_names=list(state.joint_names["right"]),
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm="right",
        embodiment="yam_real",
    )


@pytest.mark.parametrize("operation", ["step", "set_gripper", "go_home", "execute_trajectory"])
def test_reported_refusal_returns_a_typed_failure(refusing_station, operation):
    adapter = YamRealAdapter(refusing_station, allow_physical_motion=True)
    calls = {
        "step": lambda: adapter.step(_action(adapter)),
        "set_gripper": lambda: adapter.set_gripper(0.5, arm="right"),
        "go_home": lambda: adapter.go_home(),
        "execute_trajectory": lambda: adapter.execute_trajectory(_trajectory(adapter)),
    }
    result = calls[operation]()

    assert result.ok is False
    assert result.error is not None, f"{operation} returned a failure with no ApiError"
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    # The station's own words survive: a bare "it failed" is not actionable.
    assert "refused by plant" in result.error.message
    assert result.error.details["reason"] == "refused by plant"


def test_a_successful_command_carries_no_error():
    """The failure path must not leak into the success path."""
    station = build_sim_station(realtime=False)
    try:
        adapter = YamRealAdapter(station, allow_physical_motion=True)
        result = adapter.set_gripper(0.5, arm="right")
        assert result.ok is True
        assert result.error is None
    finally:
        station.close()


def test_a_foreign_embodiment_cannot_even_be_constructed():
    """Audit finding B4, embodiment half -- already closed, one layer down.

    The audit read yam_real as missing yam_sim's embodiment check. In fact the
    CONTRACT enforces it for both: ``Trajectory`` and ``SynchronizedTrajectory``
    each require ``expected_start.embodiment`` to equal their own, so a
    trajectory built against this adapter's state is a yam_real trajectory by
    construction. An adapter-level re-check would be unreachable.
    """
    station = build_sim_station(realtime=False)
    try:
        adapter = YamRealAdapter(station, allow_physical_motion=True)
        state = adapter.get_robot_state()
        with pytest.raises(ValueError, match="embodiment"):
            Trajectory(
                joint_positions=[[float(v) for v in state.joint_positions["right"]]] * 3,
                dt_s=1.0 / 30.0,
                joint_names=list(state.joint_names["right"]),
                planner="test",
                collision_aware=False,
                expected_start=state,
                arm="right",
                embodiment="yam_sim",
            )
    finally:
        station.close()


def test_dt_s_need_not_equal_the_control_period():
    """Deliberately UNLIKE yam_sim, and load-bearing for both pick programs.

    Level 1 resamples waypoints against the wall clock, so a trajectory takes
    the duration it claims however many waypoints express it. Both working pick
    programs step at 1/10 s against a 1/30 s control period, because at 1/30 the
    descent outran the arm. Copying yam_sim's dt_s check here would reject them.
    """
    station = build_sim_station(realtime=False)
    try:
        adapter = YamRealAdapter(station, allow_physical_motion=True)
        state = adapter.get_robot_state()
        assert not np.isclose(1.0 / 10.0, adapter.control_period_s)
        slow = Trajectory(
            joint_positions=[[float(v) for v in state.joint_positions["right"]]] * 3,
            dt_s=1.0 / 10.0,
            joint_names=list(state.joint_names["right"]),
            planner="test",
            collision_aware=False,
            expected_start=state,
            arm="right",
            embodiment="yam_real",
        )
        assert adapter.execute_trajectory(slow).ok is True
    finally:
        station.close()
