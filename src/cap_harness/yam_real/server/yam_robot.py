"""Level 2: buffer one target and re-send it to the motors at a fixed rate.

The harness streams targets at 60 Hz over RPC; this holds the most recent one and
re-sends it at ``follower_hold_frequency_hz`` regardless. That decoupling is the
point of the layer -- a late or dropped RPC leaves the arm holding its last valid
target rather than going slack, and a caller that stops commanding entirely stops
at a known pose instead of falling.

Accepting a command only *buffers* it. Acceptance is never a claim that the arm
arrived: convergence is checked by the layer above, against measurement.

Gravity compensation is solved from a single-arm MuJoCo model that carries a real
inertial per link. That is a different model from the station one the harness
uses for FK/IK, and deliberately so: the station model is kinematic, and using it
here under-compensates by more than an order of magnitude -- see
:meth:`YamRobot.gravity_torque`.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np

from cap_harness.yam_real.config import YamStationConfig
from cap_harness.yam_real.server.motors import MotorBus

ARM_DOF = 6

#: Consecutive failing hold iterations before the loop gives up and stops the
#: arm. At the follower hold rate this is a fraction of a second, so a transient
#: full transmit queue is ridden out while a real fault -- a motor off the bus,
#: a cable pulled -- still stops the loop promptly instead of being retried into
#: silence.
MAXIMUM_CONSECUTIVE_FAULTS = 20

#: Gravity torques above this are not physical for this arm and indicate a bad
#: model evaluation; compensation is skipped for that cycle rather than injecting
#: a large feedforward torque into the motors.
MAX_GRAVITY_TORQUE_NM = 20.0

#: Allowance when checking a commanded target against the joint limits.
#:
#: The limits describe the commandable range, but a *measured* position can sit
#: just outside it. Joints 2 and 3 have a nominal lower bound of exactly 0.0, and
#: on this station both arms rest a few ten-thousandths of a radian below it
#: (measured: left joint 3 at -0.0002, right joint 2 at -0.0010). Every motion
#: interpolates from the measured pose, so with a strict bound the first waypoint
#: of any motion from rest is rejected and the arm can never move at all.
#:
#: 0.01 rad is ~0.6 degrees: far above encoder zeroing error, far below a
#: violation worth honouring. The check is kept rather than dropped -- the
#: reference implementation validated shapes only, which is why it never hit this.
JOINT_LIMIT_TOLERANCE_RAD = 0.01


class YamRobot:
    """One follower arm: motor bus, hold loop, and the observation it publishes."""

    def __init__(
        self,
        config: YamStationConfig,
        side: str,
        bus: MotorBus,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        self.config = config
        self.side = side
        self.bus = bus
        self._clock = clock
        self._motors = config.motors
        self._num_motors = ARM_DOF + 1
        self._gripper_index = ARM_DOF

        self._lock = threading.RLock()
        self._target: np.ndarray | None = None
        self._kp = np.asarray(config.controller.kp, dtype=np.float64).reshape(7).copy()
        self._kd = np.asarray(config.controller.kd, dtype=np.float64).reshape(7).copy()
        self._gripper_velocity_limit = config.controller.gripper_velocity_limit
        self._gripper_torque_limit_nm = config.controller.gripper_torque_limit_nm
        self._accepted_at: float | None = None
        self._last_gravity = np.zeros(ARM_DOF, dtype=np.float64)
        self._background_error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Nominal gripper travel until :meth:`calibrate_gripper` measures the real
        # stops. Nominal is a starting point, not a substitute: an uncalibrated
        # gripper tracks a normalized command only as well as the nominal span
        # happens to match this unit.
        #
        # A measured gripper span was ~5.11 motor units against this
        # nominal 1.0, so an uncalibrated normalized command produces about a
        # fifth of the opening it names.
        self._nominal_gripper_span = 1.0
        self.gripper_close_pos = 0.0
        self.gripper_open_pos = self._nominal_gripper_span
        self.gripper_calibrated = False
        self._load_gripper_calibration()

        # Gravity-only inverse dynamics on a collision-free copy of the SINGLE-ARM
        # model. Contacts must not contribute: the question is what torque holds
        # the arm against gravity, not what the world is pushing back with.
        #
        # The station model is deliberately not used here. It is a kinematic
        # description -- its 18.5 kg sits in the frame rather than distributed
        # across the arm links, and five bodies carry no mass at all -- so it
        # under-predicts arm gravity by more than an order of magnitude. Measured
        # at home on joint 4: station model 0.068 Nm, this model 1.433 Nm, against
        # the ~2.2 Nm the arm's observed sag implies. Both arms are the same part,
        # so one single-arm model serves either side.
        model = mujoco.MjModel.from_xml_path(str(config.arm_model_xml))
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        self._model = model
        self._data = mujoco.MjData(model)
        self._qslice = slice(0, ARM_DOF)

    # -- gripper calibration persistence -----------------------------------

    def gripper_calibration_path(self) -> Path:
        """Where this arm's measured gripper travel is kept between restarts.

        Beside the station profile, named for the station and the side, and
        deliberately **outside** the content-addressed calibration bundle. The
        bundle is immutable and describes optics that are re-solved as a unit;
        gripper travel is a wear measurement for one arm that is expected to
        change on its own. Writing it into the bundle would break the digest
        every time a gripper was re-measured.
        """
        return self.config.source_path.parent / f"gripper-{self.side}.json"

    def _load_gripper_calibration(self) -> None:
        """Restore a previous measurement, so a restart does not re-drive the stops.

        Calibration means pushing the fingers into both mechanical stops. Doing
        that on every server start is real wear for a number that was already
        known, and it silently reverts to the nominal span whenever someone
        forgets ``--calibrate-gripper`` -- which is how this station ran with a
        placeholder span for its whole life so far.
        """
        path = self.gripper_calibration_path()
        if not path.is_file():
            return
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            close = float(stored["close"])
            openp = float(stored["open"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Never fail construction over a bad cache: a corrupt file must not
            # stop an arm coming up, it just means re-measuring.
            print(f"[yam-robot] ignoring unreadable gripper calibration {path}: {exc}", flush=True)
            return
        if not (abs(openp - close) > 1e-6):
            print(f"[yam-robot] ignoring degenerate gripper calibration {path}", flush=True)
            return
        self.gripper_close_pos = min(close, openp)
        self.gripper_open_pos = max(close, openp)
        self.gripper_calibrated = True
        print(
            f"[yam-robot] {self.side} gripper calibration loaded from {path.name}: "
            f"span {self.gripper_open_pos - self.gripper_close_pos:.4f}",
            flush=True,
        )

    def _save_gripper_calibration(self) -> None:
        path = self.gripper_calibration_path()
        payload = {
            "schema_version": 1,
            "station": self.config.station,
            "side": self.side,
            "close": self.gripper_close_pos,
            "open": self.gripper_open_pos,
            "span": self.gripper_open_pos - self.gripper_close_pos,
        }
        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            # A measurement that cannot be saved is still usable this session.
            print(f"[yam-robot] could not save gripper calibration to {path}: {exc}", flush=True)
            return
        print(f"[yam-robot] {self.side} gripper calibration saved to {path.name}", flush=True)

    # -- gripper units -----------------------------------------------------

    def _motor_to_normalized(self, motor_position: float) -> float:
        """Raw motor position -> normalized ``0`` closed, ``1`` open."""
        span = self.gripper_open_pos - self.gripper_close_pos
        if abs(span) < 1e-9:
            return 0.0
        travelled = self._motors.gripper_sign * motor_position - self.gripper_close_pos
        return float(np.clip(travelled / span, 0.0, 1.0))

    def _normalized_to_motor(self, normalized: float) -> float:
        span = self.gripper_open_pos - self.gripper_close_pos
        travelled = self.gripper_close_pos + float(np.clip(normalized, 0.0, 1.0)) * span
        return float(self._motors.gripper_sign * travelled)

    def calibrate_gripper(
        self, *, settle_s: float = 2.5, velocity: float = 5.0
    ) -> dict[str, float]:
        """Find the gripper's physical stops by driving gently into both.

        **This moves the gripper.** The travel between the stops is what maps a
        normalized command onto motor position, and it differs per unit and drifts
        as the fingers wear, so it is measured rather than configured. Until this
        runs, the profile's nominal range is used and a normalized command will be
        off by whatever the true travel differs by.
        """
        limit_ratio = 0.3
        # Probe with the motor's full position range, which certainly brackets
        # the mechanical stops. A smaller guess can sit inside the gripper's
        # resting position, in which case both probes push the same way and the
        # measurement silently returns two identical numbers.
        probe = self.bus.position_limit(self._gripper_index)
        positions = []
        for direction in (-1.0, 1.0):
            deadline = self._clock() + float(settle_s)
            while self._clock() < deadline:
                self.bus.send_force_position(
                    self._gripper_index,
                    position=direction * probe,
                    velocity_limit=float(velocity),
                    torque_ratio=limit_ratio,
                )
                time.sleep(0.02)
            states = self.bus.read_states()
            positions.append(self._motors.gripper_sign * states[self._gripper_index].position)
        if abs(positions[1] - positions[0]) < 1e-6:
            raise RuntimeError(
                f"{self.side} gripper did not move between its stops "
                f"(both read {positions[0]:+.4f}); calibration is not usable"
            )
        self.gripper_close_pos = min(positions)
        self.gripper_open_pos = max(positions)
        self.gripper_calibrated = True
        self._save_gripper_calibration()
        return {"close": self.gripper_close_pos, "open": self.gripper_open_pos}

    # -- gravity -----------------------------------------------------------

    def gravity_torque(self, joint_positions: np.ndarray) -> np.ndarray:
        """Gravity-holding torque for this arm's six joints."""
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        self._data.qpos[self._qslice] = np.asarray(joint_positions, dtype=np.float64).reshape(
            ARM_DOF
        )
        mujoco.mj_inverse(self._model, self._data)
        raw = np.asarray(self._data.qfrc_inverse[self._qslice], dtype=np.float64).copy()
        if not np.all(np.isfinite(raw)) or float(np.max(np.abs(raw))) > MAX_GRAVITY_TORQUE_NM:
            return np.zeros(ARM_DOF, dtype=np.float64)
        return raw * self._motors.gravity_comp_factor

    # -- commands ----------------------------------------------------------

    def validate_command(self, joint_state: dict[str, Any]) -> np.ndarray:
        """Check a joint7 target against the station limits, returning it.

        Rejection happens here, before anything is buffered, so an out-of-range
        request never becomes the target the hold loop keeps re-sending.
        """
        target = np.asarray(joint_state["pos"], dtype=np.float64).reshape(-1)
        if target.size != self._num_motors:
            raise ValueError(f"pos must contain {self._num_motors} entries; got {target.size}")
        if not np.all(np.isfinite(target)):
            raise ValueError("pos contains non-finite values")
        arm = target[:ARM_DOF]
        lower = self.config.joint_limits_lower - JOINT_LIMIT_TOLERANCE_RAD
        upper = self.config.joint_limits_upper + JOINT_LIMIT_TOLERANCE_RAD
        if np.any(arm < lower) or np.any(arm > upper):
            offending = [
                f"joint{index + 1}={arm[index]:+.4f} "
                f"(limits {self.config.joint_limits_lower[index]:+.4f}"
                f"..{self.config.joint_limits_upper[index]:+.4f})"
                for index in range(ARM_DOF)
                if arm[index] < lower[index] or arm[index] > upper[index]
            ]
            raise ValueError(
                f"joint target outside the configured limits for {self.side!r}: "
                + ", ".join(offending)
            )
        if not 0.0 <= target[self._gripper_index] <= 1.0:
            raise ValueError("gripper target must be normalized to [0, 1]")
        return target

    def command_joint_state(self, joint_state: dict[str, Any]) -> dict[str, Any]:
        """Validate and buffer one target. Acceptance is not arrival."""
        target = self.validate_command(joint_state)
        with self._lock:
            self._target = target
            if joint_state.get("kp") is not None:
                self._kp = np.asarray(joint_state["kp"], dtype=np.float64).reshape(7).copy()
            if joint_state.get("kd") is not None:
                self._kd = np.asarray(joint_state["kd"], dtype=np.float64).reshape(7).copy()
            if joint_state.get("gripper_vel_limit") is not None:
                self._gripper_velocity_limit = float(joint_state["gripper_vel_limit"])
            if joint_state.get("gripper_torque_limit_nm") is not None:
                self._gripper_torque_limit_nm = float(joint_state["gripper_torque_limit_nm"])
            self._accepted_at = self._clock()
        return {"accepted": True, "accepted_at": self._accepted_at}

    def command_joint_pos(self, joint_pos: np.ndarray) -> dict[str, Any]:
        """Buffer a target using the profile's default gains."""
        return self.command_joint_state({"pos": joint_pos})

    def hold_step(self) -> bool:
        """One iteration of the hold loop: re-send the buffered target.

        Returns False when nothing has been commanded yet, which is not an error:
        an arm that has received no target is left alone rather than driven to a
        default pose that no one asked for.
        """
        with self._lock:
            target = None if self._target is None else self._target.copy()
            kp, kd = self._kp.copy(), self._kd.copy()
            velocity_limit = self._gripper_velocity_limit
            torque_limit_nm = self._gripper_torque_limit_nm
        if target is None:
            return False

        gravity = self.gravity_torque(target[:ARM_DOF])
        self._last_gravity = gravity
        for index in range(ARM_DOF):
            self.bus.send_mit(
                index,
                position=float(target[index]),
                stiffness=float(kp[index]),
                damping=float(kd[index]),
                feedforward=float(gravity[index]),
            )
        self.bus.send_force_position(
            self._gripper_index,
            position=self._normalized_to_motor(float(target[self._gripper_index])),
            velocity_limit=float(velocity_limit),
            torque_ratio=float(
                np.clip(torque_limit_nm / self._motors.gripper_torque_limit_nm, 0.0, 1.0)
            ),
        )
        return True

    # -- observation -------------------------------------------------------

    def get_observations(self) -> dict[str, np.ndarray]:
        """Measured state, in the shape the harness client expects.

        Per-joint channels other than ``joint_pos`` are published at full MOTOR
        width -- six joints plus the gripper -- and the client splits them. That
        asymmetry is historical, and the simulated station reproduces it so a
        shape mismatch cannot hide until it reaches hardware.
        """
        states = self.bus.read_states()
        if len(states) != self._num_motors:
            raise RuntimeError(f"expected {self._num_motors} motor states; got {len(states)}")
        positions = np.array([state.position for state in states], dtype=np.float64)
        velocities = np.array([state.velocity for state in states], dtype=np.float64)
        efforts = np.array([state.effort for state in states], dtype=np.float64)
        return {
            "joint_pos": positions[:ARM_DOF],
            "joint_vel": velocities,
            "joint_eff": efforts,
            "gravity_comp": np.concatenate([self._last_gravity, [0.0]]),
            "gripper_pos": np.array(
                [self._motor_to_normalized(float(positions[self._gripper_index]))]
            ),
        }

    def get_joint_pos(self) -> np.ndarray:
        observation = self.get_observations()
        return np.concatenate([observation["joint_pos"], observation["gripper_pos"]])

    def get_health(self) -> dict[str, Any]:
        """Connection, background faults, and how stale the buffered target is."""
        with self._lock:
            accepted_at = self._accepted_at
        age = None if accepted_at is None else float(self._clock() - accepted_at)
        return {
            "side": self.side,
            "can_interface": self.config.arms[self.side].can_interface,
            "has_target": accepted_at is not None,
            "buffered_target_age_s": age,
            "gripper_calibrated": self.gripper_calibrated,
            "background_error": (
                None if self._background_error is None else repr(self._background_error)
            ),
            "running": self._thread is not None and self._thread.is_alive(),
        }

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        """Latch the measured position as the held target: a safe hold in place."""
        measured = self.get_joint_pos()
        with self._lock:
            self._target = np.asarray(measured, dtype=np.float64).reshape(self._num_motors)
            self._accepted_at = self._clock()

    def start(self) -> None:
        """Run the hold loop on a background thread."""
        if self._thread is not None:
            return
        self.bus.connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"yam-{self.side}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Hold the buffered target, surviving transient send failures.

        This loop is the only thing holding the arm up. Killing it means the
        motors stop and the arm sags, so what it does with an exception matters
        more than almost anything else in this file.

        It used to stop on any exception at all. Observed on hardware: both arms
        died to ``CanError('... No buffer space available [Error Code 105]')``
        and stayed dead for eighteen minutes, reporting ``running: False`` while
        the socket kept answering state queries -- so the station looked alive,
        read plausible joint angles, and quietly held nothing. The arms sagged
        1.3 rad out of the pose they had been left in.

        That error is ENOBUFS: the CAN transmit queue was momentarily full. The
        bus was fine -- ERROR-ACTIVE, zero bus errors across 500M frames -- and
        the default ``txqueuelen`` is ten frames, which one scheduling hiccup
        can overrun with seven motors per arm. Treating a full queue like a
        bus-off fault is what turned a transient into a dead robot.

        So transient send failures are retried: the next iteration re-sends the
        same target, which is exactly what the hold loop does anyway. A fault
        that persists past ``MAXIMUM_CONSECUTIVE_FAULTS`` still stops the loop,
        because something genuinely wrong -- a motor off the bus, a cable out --
        must not be retried into silence forever.
        """
        period = 1.0 / float(self.config.controller.follower_hold_frequency_hz)
        consecutive = 0
        while not self._stop.is_set():
            try:
                self.hold_step()
            except Exception as exc:
                consecutive += 1
                self._background_error = exc
                if consecutive >= MAXIMUM_CONSECUTIVE_FAULTS:
                    self._stop.set()
                    raise
            except BaseException as exc:
                # KeyboardInterrupt, SystemExit and friends: never retry those.
                self._background_error = exc
                self._stop.set()
                raise
            else:
                if consecutive:
                    # Recovered. Clear the record so `get_health` reports the
                    # arm as healthy rather than carrying a stale fault forever.
                    consecutive = 0
                    self._background_error = None
            time.sleep(period)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.bus.disconnect()


__all__ = ["ARM_DOF", "MAX_GRAVITY_TORQUE_NM", "YamRobot"]
