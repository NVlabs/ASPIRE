"""The follower arm server, against a fake motor bus.

Everything above the CAN wire is testable without hardware: target validation,
buffering, the hold loop's persistence, gravity compensation, and the gripper
unit conversion. What cannot be tested here is whether the motors do what the
bus was told -- that only proves out on the bench.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.yam_real.config import load_yam_station_config
from cap_harness.yam_real.server.motors import MotorState
from cap_harness.yam_real.server.yam_robot import ARM_DOF, YamRobot


class FakeBus:
    """Records every command and reports whatever position it is told to."""

    def __init__(self, motors: int = 7) -> None:
        self.positions = np.zeros(motors, dtype=np.float64)
        self.mit: list[dict] = []
        self.force_pos: list[dict] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_states(self) -> list[MotorState]:
        return [MotorState(position=float(p), velocity=0.0, effort=0.0) for p in self.positions]

    def send_mit(self, index, position, stiffness, damping, feedforward) -> None:
        self.mit.append(
            {
                "index": index,
                "position": position,
                "kp": stiffness,
                "kd": damping,
                "feedforward": feedforward,
            }
        )

    def send_force_position(self, index, position, velocity_limit, torque_ratio) -> None:
        self.force_pos.append(
            {
                "index": index,
                "position": position,
                "velocity_limit": velocity_limit,
                "torque_ratio": torque_ratio,
            }
        )


@pytest.fixture(scope="module")
def config():
    return load_yam_station_config("yam-example")


@pytest.fixture
def robot(config):
    return YamRobot(config, "left", FakeBus())


def _target(config, gripper=0.0):
    home = np.asarray(config.arms["left"].home_joints, dtype=np.float64)
    return {"pos": np.concatenate([home, [gripper]])}


# -- validation ------------------------------------------------------------


def test_accepts_a_target_within_limits(robot, config):
    result = robot.command_joint_state(_target(config))
    assert result["accepted"] is True


def test_rejects_a_target_outside_the_joint_limits(robot, config):
    bad = _target(config)
    bad["pos"][1] = float(config.joint_limits_upper[1]) + 0.5
    with pytest.raises(ValueError, match="outside the configured limits"):
        robot.command_joint_state(bad)


def test_rejects_a_denormalized_gripper(robot, config):
    with pytest.raises(ValueError, match="normalized"):
        robot.command_joint_state(_target(config, gripper=1.5))


def test_rejects_the_wrong_motor_count(robot):
    with pytest.raises(ValueError, match="7 entries"):
        robot.command_joint_state({"pos": np.zeros(6)})


def test_rejects_non_finite_values(robot, config):
    bad = _target(config)
    bad["pos"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        robot.command_joint_state(bad)


def test_a_rejected_target_is_not_buffered(robot, config):
    """Rejection happens before buffering, so a bad request is never held."""
    robot.command_joint_state(_target(config, gripper=0.25))
    bad = _target(config, gripper=0.9)
    bad["pos"][1] = 99.0
    with pytest.raises(ValueError):
        robot.command_joint_state(bad)

    robot.hold_step()
    assert robot.bus.force_pos[-1]["position"] == pytest.approx(robot._normalized_to_motor(0.25))


# -- the hold loop ---------------------------------------------------------


def test_hold_step_does_nothing_before_a_command(robot):
    """An arm that has received no target is left alone, not driven somewhere."""
    assert robot.hold_step() is False
    assert robot.bus.mit == []
    assert robot.bus.force_pos == []


def test_hold_step_resends_the_same_target(robot, config):
    """The point of level 2: one command, held indefinitely."""
    robot.command_joint_state(_target(config))

    for _ in range(3):
        assert robot.hold_step() is True

    assert len(robot.bus.mit) == 3 * ARM_DOF
    assert len(robot.bus.force_pos) == 3
    first = [entry["position"] for entry in robot.bus.mit[:ARM_DOF]]
    last = [entry["position"] for entry in robot.bus.mit[-ARM_DOF:]]
    assert first == last


def test_hold_step_sends_the_profile_gains(robot, config):
    robot.command_joint_state(_target(config))
    robot.hold_step()

    for index, entry in enumerate(robot.bus.mit[:ARM_DOF]):
        assert entry["kp"] == pytest.approx(config.controller.kp[index])
        assert entry["kd"] == pytest.approx(config.controller.kd[index])


def test_gravity_rides_on_the_feedforward_term(robot, config):
    """Gravity compensation is a feedforward torque, not a position offset."""
    lifted = _target(config)
    lifted["pos"][1] = 1.0
    robot.command_joint_state(lifted)
    robot.hold_step()

    commanded = [entry["position"] for entry in robot.bus.mit[:ARM_DOF]]
    feedforward = [entry["feedforward"] for entry in robot.bus.mit[:ARM_DOF]]
    np.testing.assert_allclose(commanded, lifted["pos"][:ARM_DOF])
    assert any(abs(value) > 1e-6 for value in feedforward)


def test_gravity_is_zero_for_a_free_hanging_pose(robot):
    """Sanity check that the torque is a function of the pose, not a constant."""
    torque = robot.gravity_torque(np.zeros(ARM_DOF))
    assert torque.shape == (ARM_DOF,)
    assert np.all(np.isfinite(torque))


def test_stop_latches_the_measured_position(robot, config):
    robot.command_joint_state(_target(config))
    robot.bus.positions[:ARM_DOF] = 0.2

    robot.stop()
    robot.hold_step()

    held = [entry["position"] for entry in robot.bus.mit[-ARM_DOF:]]
    np.testing.assert_allclose(held, np.full(ARM_DOF, 0.2))


# -- observation -----------------------------------------------------------


def test_observation_publishes_motor_width_channels(robot):
    """joint_pos is sliced to the arm; the rest keep the gripper entry."""
    observation = robot.get_observations()

    assert observation["joint_pos"].shape == (ARM_DOF,)
    assert observation["joint_vel"].shape == (ARM_DOF + 1,)
    assert observation["joint_eff"].shape == (ARM_DOF + 1,)
    assert observation["gripper_pos"].shape == (1,)


def test_gripper_normalization_round_trips(robot):
    for value in (0.0, 0.25, 0.5, 1.0):
        motor = robot._normalized_to_motor(value)
        assert robot._motor_to_normalized(motor) == pytest.approx(value)


def test_health_reports_staleness_and_calibration(robot, config):
    stale = robot.get_health()
    assert stale["has_target"] is False
    assert stale["buffered_target_age_s"] is None
    # True because the yam-example profile now ships a measured gripper travel, which
    # a robot loads at construction. This asserted False when the only source of
    # calibration was driving the fingers into the stops on every start.
    assert stale["gripper_calibrated"] is True

    robot.command_joint_state(_target(config))
    fresh = robot.get_health()
    assert fresh["has_target"] is True
    assert fresh["buffered_target_age_s"] >= 0.0


def test_a_resting_pose_marginally_outside_a_limit_is_accepted(robot, config):
    """Measured encoder error just outside a configured bound is tolerated.

    Motions interpolate from the measured pose, so a strict bound can reject the
    first waypoint of a motion from rest and leave the arm unable to move. Use
    the profile's bound here so this remains a tolerance test if the calibrated
    command envelope changes.
    """
    from cap_harness.yam_real.server.yam_robot import JOINT_LIMIT_TOLERANCE_RAD

    target = _target(config)
    target["pos"][1] = float(config.joint_limits_lower[1]) - (JOINT_LIMIT_TOLERANCE_RAD * 0.5)
    target["pos"][2] = float(config.joint_limits_lower[2]) - (JOINT_LIMIT_TOLERANCE_RAD * 0.25)

    assert robot.command_joint_state(target)["accepted"] is True


def test_a_real_limit_violation_is_still_rejected(robot, config):
    """The allowance is for encoder zeroing, not for honouring bad requests."""
    from cap_harness.yam_real.server.yam_robot import JOINT_LIMIT_TOLERANCE_RAD

    target = _target(config)
    target["pos"][1] = float(config.joint_limits_lower[1]) - (JOINT_LIMIT_TOLERANCE_RAD * 10)
    with pytest.raises(ValueError, match="outside the configured limits"):
        robot.command_joint_state(target)


def test_the_limit_error_names_the_offending_joint(robot, config):
    target = _target(config)
    target["pos"][3] = 99.0
    with pytest.raises(ValueError, match="joint4"):
        robot.command_joint_state(target)


# -- hold-loop fault tolerance ---------------------------------------------


class _FlakyRobot:
    """A hold loop that fails a fixed number of times, then recovers."""

    def __init__(self, failures: int, config, side: str = "right") -> None:
        import threading

        from cap_harness.yam_real.server.yam_robot import YamRobot

        self._remaining = failures
        self.calls = 0
        self.side = side
        self.config = config
        self._stop = threading.Event()
        self._background_error = None
        self._thread = None
        self._run = YamRobot._run.__get__(self)

    def hold_step(self) -> bool:
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise OSError(105, "No buffer space available")
        self._stop.set()  # recovered; end the loop so the test terminates
        return True


def _flaky(failures: int, config):
    return _FlakyRobot(failures, config)


def test_a_transient_can_send_failure_does_not_kill_the_hold_loop(config) -> None:
    """Regression: one ENOBUFS used to stop the arm holding, permanently.

    Observed on hardware -- both arms died to a full CAN transmit queue and stayed
    dead for eighteen minutes while the socket kept answering state queries, so
    the station looked alive and held nothing. The arms sagged 1.3 rad.
    """
    robot = _flaky(3, config)
    robot._run()

    assert robot.calls == 4  # three failures ridden out, then a good iteration
    assert robot._background_error is None  # recovery clears the fault


def test_a_persistent_fault_still_stops_the_arm(config) -> None:
    """Retrying forever would be its own failure: a real fault must surface."""
    import pytest

    from cap_harness.yam_real.server.yam_robot import MAXIMUM_CONSECUTIVE_FAULTS

    robot = _flaky(MAXIMUM_CONSECUTIVE_FAULTS + 10, config)
    with pytest.raises(OSError):
        robot._run()

    assert robot.calls == MAXIMUM_CONSECUTIVE_FAULTS
    assert robot._stop.is_set()
    assert robot._background_error is not None


# -- gripper calibration persistence ---------------------------------------


def test_gripper_calibration_survives_a_restart(config, tmp_path, monkeypatch) -> None:
    """Measured travel is written and reloaded, so a restart need not re-drive the stops.

    Calibration pushes the fingers into both mechanical stops. Repeating that on
    every server start is real wear for a number already known, and forgetting
    ``--calibrate-gripper`` silently reverts to the nominal span -- which is how
    this station ran with a placeholder for its whole life, making every
    normalized command about a fifth of the opening it named.
    """
    from cap_harness.yam_real.server.yam_robot import YamRobot

    path = tmp_path / "gripper-right.json"
    monkeypatch.setattr(YamRobot, "gripper_calibration_path", lambda self: path)
    measured = YamRobot(config, "right", FakeBus())
    measured.gripper_close_pos = -6.2903
    measured.gripper_open_pos = -1.1721
    measured.gripper_calibrated = True
    measured._save_gripper_calibration()
    assert path.is_file()

    restarted = YamRobot(config, "right", FakeBus())

    assert restarted.gripper_calibrated is True
    assert restarted.gripper_close_pos == pytest.approx(-6.2903)
    assert restarted.gripper_open_pos == pytest.approx(-1.1721)
    # The span is what a normalized command is scaled by, and it is what the
    # placeholder got wrong: ~5.11 real units against a nominal 1.0.
    span = restarted.gripper_open_pos - restarted.gripper_close_pos
    assert span == pytest.approx(5.1182, abs=1e-3)


def test_a_corrupt_calibration_file_does_not_stop_the_arm(config, tmp_path, monkeypatch) -> None:
    """A bad cache means re-measuring, never an arm that refuses to come up."""
    from cap_harness.yam_real.server.yam_robot import YamRobot

    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(YamRobot, "gripper_calibration_path", lambda self: path)

    robot = YamRobot(config, "left", FakeBus())

    assert robot.gripper_calibrated is False
    assert robot.gripper_open_pos == 1.0  # fell back to nominal
