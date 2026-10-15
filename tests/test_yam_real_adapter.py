"""Adapter behaviour on an authorized station.

These run against the MuJoCo plant rather than a stub, so the arms have mass and
finite gain and actually lag their targets. That is what makes the trajectory
residual check testable: a stub that echoes commanded positions back as measured
ones could never fail it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.yam_real.action import YamActionBatch, resolve_action_batch
from cap_harness.yam_real.adapter import TRAJECTORY_JOINT_TOLERANCE_RAD, YamRealAdapter
from cap_harness.yam_real.registry import YamRealTaskRegistry
from cap_harness.yam_real.station import build_sim_station

HOME = np.zeros(6)


@pytest.fixture(scope="module")
def env():
    station = build_sim_station(realtime=True)
    yield station
    station.close()


@pytest.fixture
def adapter(env):
    env._plant.set_joint_positions(HOME, HOME)
    return YamRealAdapter(env, allow_physical_motion=True)


def _trajectory(state, positions, arm="left"):
    return Trajectory(
        joint_positions=positions,
        dt_s=1.0 / 60.0,
        joint_names=[f"joint_{index + 1}" for index in range(6)],
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm=arm,
        embodiment="yam_real",
    )


# -- identity and metadata -------------------------------------------------


def test_adapter_declares_six_dof_bimanual(adapter):
    assert adapter.embodiment == "yam_real"
    assert adapter.arms == ("left", "right")
    assert adapter.dof == 6


def test_planning_context_states_the_model_rather_than_leaving_it_inferred(adapter):
    """The shared fallback infers dual Panda from arm count, which rejects 6-DOF."""
    context = adapter.get_planning_context()

    assert context.embodiment == "yam_real"
    assert context.model == "yam_real"
    for side in ("left", "right"):
        assert len(context.joint_names[side]) == 6
        assert context.end_effector_links[side] == f"{side}_grasp"


def test_planning_context_uses_identity_base_transforms(adapter):
    """The planner's model already places both arms; the offset must not repeat.

    The cuRobo description is bimanual and puts the arms 0.62 m apart itself, so
    passing the profile's measured transforms here applies that separation a
    second time and sends every goal somewhere impossible. Measured symptom on
    hardware: IK refused the arm's own current pose, which is reachable by
    definition.
    """
    context = adapter.get_planning_context()
    for side in ("left", "right"):
        np.testing.assert_allclose(context.base_transforms[side], np.eye(4), atol=0)


def test_controller_metadata_carries_the_control_contract(adapter):
    metadata = adapter.get_controller_metadata()

    assert metadata["control_contract_version"] == "yam-direct-v4-fixed-gain"
    assert metadata["physical_motion_authorized"] is True
    assert len(metadata["action_lower_bounds"]) == 12
    contract = metadata["control_contract"]
    # All three cascade rates, distinctly recorded.
    assert contract["command_stream_hz"] == 60.0
    assert contract["follower_hold_hz"] == 100.0
    assert contract["station"] == "yam-example"
    assert contract["calibration_bundle"] == "yam-example-v1"


def test_check_success_is_not_a_success_claim(adapter):
    """Hardware has no success predicate; False means "cannot tell"."""
    assert adapter.check_success() is False


def test_task_metadata_names_the_station_and_bundle(adapter):
    metadata = adapter.get_task_metadata()
    assert metadata["station"] == "yam-example"
    assert metadata["calibration_bundle"] == "yam-example-v1"
    assert metadata["dof"] == 6


# -- reset -----------------------------------------------------------------


def test_reset_goes_to_home_not_ready(env):
    """Reset drives to the mechanical zero, the pose the profile calibrates against.

    Home and ready both plan, so this pins a deliberate choice rather than a
    reachability constraint. The choice is home because ready parks the forearm
    at z 1.23, inside the band where the station camera returns arm hardware the
    URDF does not model, and a scene-aware planner then reads the start state as
    in-collision.
    """
    env._plant.set_joint_positions(
        np.array([0.3, 0.4, 0.5, 0.0, 0.0, 0.0]), np.array([0.3, 0.4, 0.5, 0.0, 0.0, 0.0])
    )
    adapter = YamRealAdapter(env, allow_physical_motion=True)

    adapter.reset(seed=3)

    reached = adapter.get_robot_state().joint_positions["left"]
    home = env.config.arms["left"].home_joints
    np.testing.assert_allclose(reached, home, atol=0.1)
    assert not np.allclose(home, env.config.arms["left"].ready_joints)


def test_reset_records_the_seed_as_provenance_only(adapter):
    registry = YamRealTaskRegistry()
    adapter.reset(registry.resolve("observe_station"), seed=11)

    context = adapter.get_task_context()
    assert context.metadata["seed"] == 11
    # Scene reset is a physical act; the adapter must not claim to have done it.
    assert context.metadata["scene_reset"] == "none"


# -- motion ----------------------------------------------------------------


def test_step_moves_toward_the_commanded_joints(adapter):
    state = adapter.get_robot_state()
    target = state.joint_positions["left"] + np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    action = RobotAction(
        arms={
            "left": ArmCommand("joint_position", target, 0.5, embodiment="yam_real"),
            "right": ArmCommand(
                "joint_position", state.joint_positions["right"], 0.5, embodiment="yam_real"
            ),
        }
    )

    result = adapter.step(action)

    assert result.ok is True
    moved = adapter.get_robot_state().joint_positions["left"][1]
    assert moved > state.joint_positions["left"][1]


def test_trajectory_reports_its_residual(adapter):
    state = adapter.get_robot_state()
    start = state.joint_positions["left"]
    goal = start + np.array([0.0, 0.15, 0.0, 0.0, 0.0, 0.0])
    positions = np.linspace(start, goal, 30)

    result = adapter.execute_trajectory(_trajectory(state, positions))

    assert result.ok is True
    residual = result.diagnostics["joint_residual_rad"]["left"]
    assert residual <= TRAJECTORY_JOINT_TOLERANCE_RAD


def test_trajectory_fails_when_the_arm_cannot_track_it(adapter):
    """The check exists because a batch once reported success 0.54 rad short.

    A step far beyond what the arm can reach inside the trajectory's own duration
    leaves it well behind its final waypoint, and that must not report success.
    """
    state = adapter.get_robot_state()
    start = state.joint_positions["left"]
    unreachable = start + np.array([0.0, 1.4, 0.0, 0.0, 0.0, 0.0])
    positions = np.vstack([start, unreachable])

    result = adapter.execute_trajectory(_trajectory(state, positions))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ErrorCode.EXECUTION_FAILED
    assert result.diagnostics["joint_residual_rad"]["left"] > TRAJECTORY_JOINT_TOLERANCE_RAD


def test_step_rejects_an_unknown_arm(adapter):
    """``RobotAction`` has no start state, so it accepts any arm key.

    Both trajectory contracts validate their arms against ``expected_start``, but
    an action does not, so a mistyped arm reaches the adapter looking valid.
    Unchecked, the unknown entry is dropped, both arms hold position, and the
    step reports success having moved nothing.
    """
    state = adapter.get_robot_state()
    action = RobotAction(
        arms={
            "third": ArmCommand(
                "joint_position", state.joint_positions["left"], 0.5, embodiment="yam_real"
            )
        }
    )

    result = adapter.step(action)

    assert result.ok is False
    assert result.error.code is ErrorCode.INVALID_REQUEST


def test_set_gripper_moves_one_gripper_and_leaves_the_other(adapter):
    """Asserts direction, not arrival.

    The fingers are torque-limited, so a short command closes only part of the
    remaining travel; asserting a precise position would encode the simulated
    finger dynamics into an API test and would say nothing about hardware.
    """
    before = adapter.get_robot_state()

    result = adapter.set_gripper(0.9, arm="left")

    assert result.ok is True
    after = adapter.get_robot_state()
    assert after.gripper_positions["left"] > before.gripper_positions["left"]
    # The untouched gripper is held, not frozen: it is still commanded to its
    # measured position each tick and settles by a fraction of a percent. The
    # claim under test is that it was not driven anywhere, not that physics
    # stopped for it.
    assert after.gripper_positions["right"] == pytest.approx(
        before.gripper_positions["right"], abs=0.01
    )


# -- action resolution -----------------------------------------------------


def test_unspecified_gripper_holds_its_measured_position(env):
    """A joint-only motion must not drop what the robot is carrying."""
    adapter = YamRealAdapter(env, allow_physical_motion=True)
    adapter.set_gripper(0.7, arm="left")
    held = adapter.get_robot_state().gripper_positions["left"]

    state = adapter.get_robot_state()
    batch = YamActionBatch.joint_abs(
        [0.0, 0.1],
        np.tile(state.joint_positions["left"], (2, 1)),
        np.tile(state.joint_positions["right"], (2, 1)),
        source="test",
    )
    resolved = resolve_action_batch(env, batch)

    # Tolerance, not equality: the plant runs in real time and the fingers are
    # still settling between the read above and the resolve. What matters is that
    # the held value came from the measurement rather than from a default.
    assert resolved.left_gripper_positions[0, 0] == pytest.approx(held, abs=0.02)
    assert resolved.left_gripper_positions[0, 0] not in (0.0, 1.0)


def test_action_batch_rejects_a_non_joint_space(env):
    batch = YamActionBatch(
        space="eef_abs", timestamps=[0.0], left=np.zeros((1, 7)), right=np.zeros((1, 7))
    )
    with pytest.raises(ValueError, match="joint_abs"):
        resolve_action_batch(env, batch)


def test_action_batch_rejects_non_monotonic_timestamps(env):
    state = env.get_observations("left")["joint_pos"]
    batch = YamActionBatch.joint_abs(
        [0.0, 0.2, 0.1], np.tile(state, (3, 1)), np.tile(state, (3, 1))
    )
    with pytest.raises(ValueError, match="monotonically increasing"):
        resolve_action_batch(env, batch)


# -- the trajectory's gripper column ---------------------------------------


def test_trajectory_gripper_column_reaches_the_plant(adapter):
    """The commanded width must survive execution, not the measured one.

    During a successful grasp the two deliberately differ: fingers stalled on an
    object read wider than they were told to close to. Substituting the
    measurement re-commands that wider value and releases the squeeze -- and
    because it happens *during* the lift, every stage still reports ok and the
    failure surfaces only as "the object did not rise".
    """
    adapter.set_gripper(0.9, arm="left")
    measured = adapter.get_robot_state().gripper_positions["left"]
    assert measured > 0.5, "fixture precondition: the gripper should be open"

    state = adapter.get_robot_state()
    start = state.joint_positions["left"]
    positions = np.linspace(start, start + np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0]), 12)
    commanded = 0.1
    trajectory = Trajectory(
        joint_positions=positions,
        dt_s=1.0 / 60.0,
        joint_names=[f"joint_{index + 1}" for index in range(6)],
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm="left",
        embodiment="yam_real",
        gripper_positions=np.full(len(positions), commanded),
    )

    result = adapter.execute_trajectory(trajectory)

    assert result.ok is True, result.error
    closed = adapter.get_robot_state().gripper_positions["left"]
    # Direction, not arrival: the fingers are torque-limited and the trajectory is
    # short. The claim is that the commanded 0.1 was followed rather than the
    # measured ~0.9 being re-commanded, which would have left it open.
    assert closed < measured - 0.1, f"gripper stayed at {closed:.3f}; column was ignored"


def test_trajectory_without_a_gripper_column_holds_the_measured_width(adapter):
    """A joint-only trajectory must not drop what the robot is carrying."""
    adapter.set_gripper(0.8, arm="left")
    held = adapter.get_robot_state().gripper_positions["left"]

    state = adapter.get_robot_state()
    start = state.joint_positions["left"]
    positions = np.linspace(start, start + np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0]), 10)

    result = adapter.execute_trajectory(_trajectory(state, positions))

    assert result.ok is True, result.error
    assert adapter.get_robot_state().gripper_positions["left"] == pytest.approx(held, abs=0.05)


def test_a_mis_sized_gripper_column_is_rejected_by_the_contract(adapter):
    """The guarantee the adapter relies on, pinned where it actually lives.

    ``execute_trajectory`` does not re-check the gripper column's length against
    the waypoint count. That is safe only because the contract refuses to build
    such a trajectory at all, so this asserts the constructor rejection rather
    than a downstream one that could never fire.
    """
    state = adapter.get_robot_state()
    start = state.joint_positions["left"]
    positions = np.linspace(start, start + np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0]), 10)

    with pytest.raises(ValueError, match="gripper_positions must have shape"):
        Trajectory(
            joint_positions=positions,
            dt_s=1.0 / 60.0,
            joint_names=[f"joint_{index + 1}" for index in range(6)],
            planner="test",
            collision_aware=False,
            expected_start=state,
            arm="left",
            embodiment="yam_real",
            gripper_positions=np.full(3, 0.5),  # 3 rows for 10 waypoints
        )


# -- run recording ---------------------------------------------------------


class _Observer:
    """Stands in for RunRecorder: records the calls, enforces nothing."""

    def __init__(self) -> None:
        self.reset_calls = []
        self.before = []
        self.after = []

    def on_reset(self, observation, metadata) -> None:
        self.reset_calls.append((observation, dict(metadata)))

    def before_step(self, action) -> None:
        self.before.append(action)

    def after_step(self, action, result) -> None:
        self.after.append((action, result))


def test_an_authorized_command_is_reported_to_the_run_observer(env) -> None:
    """Regression: the adapter accepted a run_observer and never called it.

    ``run.py`` passes the recorder in for every embodiment, but this adapter
    ignored it, so a recorded yam_real run had no reset.json, no step records
    and no video at all -- while the run still reported success.

    Motion must be authorized for this to mean anything: the interlock returns
    before the recording hooks, so an unauthorized run exercises none of this.
    That is exactly why a no-motion preflight cannot catch a fault in here.
    """
    env._plant.set_joint_positions(HOME, HOME)
    observer = _Observer()
    adapter = YamRealAdapter(env, allow_physical_motion=True, run_observer=observer)

    adapter.reset(YamRealTaskRegistry().resolve("reach_home"), seed=1)
    assert len(observer.reset_calls) == 1
    _, metadata = observer.reset_calls[0]
    assert metadata["station"] == "yam-example"
    assert metadata["physical_motion_authorized"] is True

    state = adapter.get_robot_state()
    result = adapter.step(
        RobotAction(
            {
                side: ArmCommand(
                    "joint_position",
                    state.joint_positions[side],
                    0.5,
                    embodiment="yam_real",
                )
                for side in ("left", "right")
            }
        )
    )

    assert result.ok, result.error
    # The action handed to the recorder must be a well-formed contract value:
    # building it wrongly raised inside the adapter and turned every authorized
    # motion into `adapter step failed: RobotAction.__init__() got an unexpected
    # keyword argument 'embodiment'`.
    assert len(observer.before) == 1
    assert len(observer.after) == 1
    action, step = observer.after[0]
    assert isinstance(action, RobotAction)
    assert action.embodiment == "yam_real"
    assert set(action.arms) == {"left", "right"}
    assert step.observation is not None  # the frame the video is written from


def test_the_observer_is_not_called_when_motion_is_refused(env) -> None:
    """A refused command moved nothing, so it must not appear in the record."""
    env._plant.set_joint_positions(HOME, HOME)
    observer = _Observer()
    adapter = YamRealAdapter(env, allow_physical_motion=False, run_observer=observer)

    result = adapter.set_gripper(0.5, arm="right")

    assert not result.ok
    assert result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert observer.before == []
    assert observer.after == []
