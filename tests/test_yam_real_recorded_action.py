"""A recorded action must be what was COMMANDED, not what was measured.

Audit findings A4 and E2. ``_commanded_action`` built every action from
``codec.robot_state``, so ``episode/steps.jsonl`` said the policy commanded
exactly where the arm already was.

On a position-controlled arm that lags its target by centimetres, that is not a
weaker record of the policy -- it is a record of a different policy, one that
never asks for anything it has not already got. Trained on, it teaches the arm
to stand still.

The existing observer test could not catch this: it asserted an action was
recorded, and a measured-pose action is still an action. So every check here
compares the recorded target against a commanded value chosen to be FAR from
the measured one, and fails if they match the arm instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, StepResult, Trajectory
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.station import build_sim_station


class CapturingObserver:
    """Records what the adapter hands the run recorder."""

    def __init__(self) -> None:
        self.actions: list[RobotAction] = []

    def before_step(self, action: RobotAction) -> None:
        pass

    def after_step(self, action: RobotAction, result: StepResult) -> None:
        self.actions.append(action)

    def on_reset(self, observation: object, metadata: object) -> None:
        pass


@pytest.fixture
def station():
    env = build_sim_station(realtime=False)
    yield env
    env.close()


def test_trajectory_records_its_final_waypoint_not_the_arm(station):
    observer = CapturingObserver()
    adapter = YamRealAdapter(station, allow_physical_motion=True, run_observer=observer)
    state = adapter.get_robot_state()
    measured = np.asarray(state.joint_positions["right"], dtype=np.float64)

    # A target the arm demonstrably is not at, and cannot reach within the
    # trajectory: the point is that the ACTION says where it was sent.
    goal = measured + 0.25
    rows = [measured.tolist(), ((measured + goal) / 2).tolist(), goal.tolist()]
    adapter.execute_trajectory(
        Trajectory(
            joint_positions=rows,
            dt_s=1.0 / 30.0,
            joint_names=list(state.joint_names["right"]),
            planner="test",
            collision_aware=False,
            expected_start=state,
            arm="right",
            embodiment="yam_real",
        )
    )

    assert observer.actions, "the trajectory recorded no action at all"
    recorded = np.asarray(observer.actions[-1].arms["right"].target, dtype=np.float64)
    assert np.allclose(recorded, goal), (
        f"recorded {recorded} instead of the commanded final waypoint {goal}"
    )
    # And explicitly NOT the arm's own pose, which is what it used to record.
    assert not np.allclose(recorded, measured)


def test_step_records_the_commanded_joints(station):
    observer = CapturingObserver()
    adapter = YamRealAdapter(station, allow_physical_motion=True, run_observer=observer)
    state = adapter.get_robot_state()
    measured = np.asarray(state.joint_positions["right"], dtype=np.float64)
    goal = measured + 0.20

    adapter.step(
        RobotAction(
            arms={
                "right": ArmCommand(
                    mode="joint_position",
                    target=goal,
                    gripper_position=0.75,
                    embodiment="yam_real",
                )
            }
        )
    )

    assert observer.actions
    command = observer.actions[-1].arms["right"]
    assert np.allclose(np.asarray(command.target, dtype=np.float64), goal)
    assert command.gripper_position == pytest.approx(0.75)


def test_set_gripper_records_the_width_it_asked_for(station):
    observer = CapturingObserver()
    adapter = YamRealAdapter(station, allow_physical_motion=True, run_observer=observer)
    adapter.set_gripper(0.33, arm="right")

    assert observer.actions
    assert observer.actions[-1].arms["right"].gripper_position == pytest.approx(0.33)


def test_go_home_records_the_home_pose(station):
    """Drives the arm away first, so measured and commanded cannot coincide.

    Written the obvious way this test passed against the OLD behaviour too: the
    station starts at home, so the measured pose and the home pose were the same
    numbers and recording either satisfied it. That is the shape of test the
    audit's E1 and E2 are about -- one that asserts a value rather than a
    mechanism, and so cannot fail when the mechanism is removed.
    """
    observer = CapturingObserver()
    adapter = YamRealAdapter(station, allow_physical_motion=True, run_observer=observer)
    state = adapter.get_robot_state()
    away = np.asarray(state.joint_positions["right"], dtype=np.float64) + 0.30
    adapter.execute_trajectory(
        Trajectory(
            joint_positions=[state.joint_positions["right"].tolist(), away.tolist()],
            dt_s=1.0 / 30.0,
            joint_names=list(state.joint_names["right"]),
            planner="test",
            collision_aware=False,
            expected_start=state,
            arm="right",
            embodiment="yam_real",
        )
    )
    adapter.go_home()

    home = np.asarray(station.config.arms["right"].home_joints, dtype=np.float64)
    recorded = np.asarray(observer.actions[-1].arms["right"].target, dtype=np.float64)
    # Exact: home_joints is the profile's declared pose, while a settling arm
    # reports something a few hundredths away from it.
    assert np.allclose(recorded, home, atol=1e-9), (
        f"recorded {recorded}, expected the commanded home pose {home}"
    )


def test_an_uncommanded_arm_falls_back_to_its_measured_pose(station):
    """Holding is a real command, and measured is the right value for it.

    A single-arm trajectory says nothing about the other arm, which the plant
    holds where it is. Recording its measured pose is correct there -- the
    fallback is not laziness, it is the only honest value.
    """
    observer = CapturingObserver()
    adapter = YamRealAdapter(station, allow_physical_motion=True, run_observer=observer)
    state = adapter.get_robot_state()
    left_before = np.asarray(state.joint_positions["left"], dtype=np.float64)
    right = np.asarray(state.joint_positions["right"], dtype=np.float64)

    adapter.execute_trajectory(
        Trajectory(
            joint_positions=[right.tolist(), (right + 0.1).tolist()],
            dt_s=1.0 / 30.0,
            joint_names=list(state.joint_names["right"]),
            planner="test",
            collision_aware=False,
            expected_start=state,
            arm="right",
            embodiment="yam_real",
        )
    )

    recorded_left = np.asarray(observer.actions[-1].arms["left"].target, dtype=np.float64)
    assert np.allclose(recorded_left, left_before, atol=1e-2)
