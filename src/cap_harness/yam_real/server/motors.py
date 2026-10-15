"""The motor bus an arm server drives, behind one narrow interface.

:class:`MotorBus` is what :class:`~cap_harness.yam_real.server.yam_robot.YamRobot`
talks to. Keeping it narrow -- read states, send an MIT command, send a force-position
command -- means the whole hold loop, the joint-limit checks and the gravity
compensation can be exercised against a fake on a machine with no CAN bus, which
is the only part of level 2 that is testable off-robot.

Two command shapes, because the arm and the gripper are driven differently:

* **MIT** for the arm joints: ``tau = kp*(q_d - q) + kd*(0 - qdot) + tau_ff``,
  closed in the motor firmware. Position, gains and a feedforward torque go out;
  gravity compensation rides in on the feedforward term.
* **Force-position** for the gripper, which needs to squeeze rather than track:
  a target with a velocity cap and a torque cap, so closing on an object stalls
  at a bounded force instead of driving into it at full current.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, ClassVar, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class MotorState:
    """One motor's feedback, in motor units."""

    position: float
    velocity: float
    effort: float
    temperature: float = 0.0


class MotorBus(Protocol):
    """What a follower needs from the motors it drives."""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_states(self) -> list[MotorState]:
        """Feedback for every motor, arm joints first and the gripper last."""

    def send_mit(
        self, index: int, position: float, stiffness: float, damping: float, feedforward: float
    ) -> None: ...

    def send_force_position(
        self, index: int, position: float, velocity_limit: float, torque_ratio: float
    ) -> None: ...

    def position_limit(self, index: int) -> float:
        """Magnitude of the motor's commandable position range, in motor units."""


class DaMiaoBus:
    """A CAN chain of DaMiao motors.

    ``damiao_motor`` and its CAN backend are imported lazily so the harness stays
    importable, and the tests stay runnable, on a machine with neither.
    """

    def __init__(
        self,
        can_interface: str,
        motor_ids: tuple[int, ...],
        motor_types: tuple[str, ...],
        *,
        bustype: str = "socketcan",
    ) -> None:
        if len(motor_ids) != len(motor_types):
            raise ValueError("motor_ids and motor_types must have the same length")
        self.can_interface = str(can_interface)
        self.motor_ids = tuple(int(value) for value in motor_ids)
        self.motor_types = tuple(str(value) for value in motor_types)
        self.bustype = str(bustype)
        self._controller: Any = None
        self._motors: list[Any] = []

    #: Register 10 selects the motor's control mode. The arm joints run MIT; the
    #: gripper runs FORCE_POS so it can stall against an object at a bounded
    #: torque instead of tracking a position into it.
    _MODE_REGISTER = 10
    _MODES: ClassVar[dict[str, int]] = {"MIT": 1, "POS_VEL": 2, "VEL": 3, "FORCE_POS": 4}

    def _ensure_control_mode(self, motor: Any, mode: str) -> None:
        """Read register 10 and write it only if it disagrees.

        The mode is persisted in the motor, so it survives a power cycle and can
        disagree with what this process expects -- a gripper left in MIT ignores
        the torque limit a force-position command carries. The write is verified
        by reading back rather than assumed to have taken.
        """
        desired = self._MODES[mode]
        self._invalidate_register(motor, self._MODE_REGISTER)
        if int(motor.get_register(self._MODE_REGISTER, timeout=1.0)) == desired:
            return
        motor.write_register(self._MODE_REGISTER, desired)
        time.sleep(0.2)
        self._invalidate_register(motor, self._MODE_REGISTER)
        confirmed = int(motor.get_register(self._MODE_REGISTER, timeout=1.0))
        if confirmed != desired:
            raise RuntimeError(
                f"{self.can_interface}: control mode write not accepted; "
                f"wanted {mode} ({desired}), motor reports {confirmed}"
            )

    @staticmethod
    def _invalidate_register(motor: Any, register: int) -> None:
        """Drop a cached register so the next read comes from the motor."""
        registers = getattr(motor, "registers", None)
        if registers is None:
            return
        lock = getattr(motor, "registers_lock", None)
        if lock is None:
            registers.pop(register, None)
            return
        with lock:
            registers.pop(register, None)

    def connect(self) -> None:
        from damiao_motor import DaMiaoController

        self._controller = DaMiaoController(self.can_interface, bustype=self.bustype)
        self._motors = []
        for index, (motor_id, motor_type) in enumerate(
            zip(self.motor_ids, self.motor_types, strict=True)
        ):
            # Feedback id matches the command id on this station.
            motor = self._controller.add_motor(motor_id, motor_id, motor_type=motor_type)
            is_gripper = index == len(self.motor_ids) - 1
            self._ensure_control_mode(motor, "FORCE_POS" if is_gripper else "MIT")
            motor.enable()
            self._motors.append(motor)

    def disconnect(self) -> None:
        # Every motor gets a disable attempt even if an earlier one fails: one
        # unreachable motor must not leave the rest energized. Failures are
        # reported afterwards rather than swallowed -- a motor that would not
        # disable is exactly what an operator needs to hear about.
        failures: list[str] = []
        for motor, motor_id in zip(self._motors, self.motor_ids, strict=False):
            try:
                motor.disable()
            except Exception as exc:
                failures.append(f"0x{motor_id:02x}: {type(exc).__name__}: {exc}")
        if failures:
            print(
                f"[yam-server] {self.can_interface}: motors failed to disable: "
                + "; ".join(failures),
                flush=True,
            )
        if self._controller is not None:
            self._controller.shutdown()
        self._motors = []
        self._controller = None

    def read_states(self) -> list[MotorState]:
        states = []
        for motor in self._motors:
            raw = motor.get_states()
            states.append(
                MotorState(
                    position=float(raw["pos"]),
                    velocity=float(raw.get("vel", 0.0)),
                    effort=float(raw.get("eff", raw.get("tau", 0.0))),
                    temperature=float(raw.get("t_mos", 0.0)),
                )
            )
        return states

    def send_mit(
        self, index: int, position: float, stiffness: float, damping: float, feedforward: float
    ) -> None:
        self._motors[index].send_cmd_mit(
            target_position=float(position),
            target_velocity=0.0,
            stiffness=float(stiffness),
            damping=float(damping),
            feedforward_torque=float(feedforward),
        )

    def position_limit(self, index: int) -> float:
        """The motor's commandable position magnitude, from its type preset.

        Calibration has to drive past the mechanical stops to find them, so it
        needs a target the motor will accept and that certainly brackets the
        travel. Guessing a value smaller than the resting position makes both
        probe directions push the same way and the gripper never moves.
        """
        from damiao_motor import MOTOR_TYPE_PRESETS

        return float(MOTOR_TYPE_PRESETS[self.motor_types[index]]["p_max"])

    def send_force_position(
        self, index: int, position: float, velocity_limit: float, torque_ratio: float
    ) -> None:
        self._motors[index].send_cmd_force_pos(
            target_position=float(position),
            velocity_limit=float(velocity_limit),
            torque_limit_ratio=float(np.clip(torque_ratio, 0.0, 1.0)),
        )


__all__ = ["DaMiaoBus", "MotorBus", "MotorState"]
