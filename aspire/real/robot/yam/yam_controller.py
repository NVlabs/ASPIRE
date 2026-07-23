# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAM arm controller using DaMiao motors directly."""

import logging
import os
import threading
import time
from typing import List, Optional

import numpy as np
from damiao_motor import DaMiaoController, DaMiaoMotor, MOTOR_TYPE_PRESETS
from i2rt.utils.mujoco_utils import MuJoCoKDL

# Default XML for gravity compensation (LINEAR_4310 gripper variant)
_YAM_XML_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "third_party", "i2rt", "i2rt", "robot_models", "yam", "yam_4310_linear.xml",
)

# Global scale factor matching i2rt's get_yam_robot (gravity_comp_factor=1.3)
_GRAVITY_COMP_FACTOR = 1.3


class YamRobot:
    """YAM robot interface using damiao-motor for hardware control."""

    def __init__(
        self,
        can_interface: str,
        motor_ids: List[int],
        motor_types: List[str],
        feedback_ids: Optional[List[int]] = None,
        default_kp: Optional[List[float]] = None,
        default_kd: Optional[List[float]] = None,
        gripper_index: Optional[int] = None,
        gripper_sign: int = -1,
        gripper_vel_limit: float = 30.0,
        gripper_torque_limit_nm: float = 1.0,
        send_rate_hz: float = 50.0,
        calibrate_gripper: bool = True,
        gripper_close_pos: Optional[float] = None,
        gripper_open_pos: Optional[float] = None,
        transient_send_error_max_streak: int = 0,
        transient_send_error_backoff_s: float = 0.02,
        bustype: str = "socketcan",
    ):
        if len(motor_types) != len(motor_ids):
            raise ValueError(f"motor_types length {len(motor_types)} != motor_ids length {len(motor_ids)}")

        self.NUM_JOINTS = len(motor_ids)
        self.gripper_index = gripper_index
        self.gripper_sign = gripper_sign
        self.gripper_vel_limit = gripper_vel_limit
        self.gripper_torque_limit_nm = gripper_torque_limit_nm
        self.send_rate_hz = send_rate_hz
        self.calibrate_gripper = calibrate_gripper
        self._configured_gripper_close_pos = gripper_close_pos
        self._configured_gripper_open_pos = gripper_open_pos
        self.transient_send_error_max_streak = max(0, int(transient_send_error_max_streak))
        self.transient_send_error_backoff_s = max(0.0, float(transient_send_error_backoff_s))

        self.can_interface = can_interface
        self.motor_ids = motor_ids
        self.motor_types = motor_types
        self.feedback_ids = feedback_ids or motor_ids.copy()
        self.default_kp = np.asarray(default_kp) if default_kp is not None else np.zeros(self.NUM_JOINTS)
        self.default_kd = np.asarray(default_kd) if default_kd is not None else np.zeros(self.NUM_JOINTS)

        self.bustype = bustype
        self.controller = DaMiaoController(
            channel=can_interface, bustype=bustype, bitrate=1000000
        )
        self.motors: List[Optional[DaMiaoMotor]] = [None] * self.NUM_JOINTS
        self._connected = False

        # Set by calibration in connect()
        self.gripper_close_pos: Optional[float] = None
        self.gripper_open_pos: Optional[float] = None

        # Gravity compensation
        self._kdl: Optional[MuJoCoKDL] = None
        self._kdl_lock = threading.Lock()
        self._last_gravity_comp = np.zeros(self.NUM_JOINTS)

        # Background send loop
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._send_thread: Optional[threading.Thread] = None
        self._cmd_pos: Optional[np.ndarray] = None
        self._cmd_kp: Optional[np.ndarray] = None
        self._cmd_kd: Optional[np.ndarray] = None
        self._cmd_gripper_vel_limit: Optional[float] = None
        self._cmd_gripper_torque_limit_nm: Optional[float] = None
        self._background_error: Optional[BaseException] = None
        self._transient_send_error_streak = 0

    def _raise_if_background_error(self) -> None:
        if self._background_error is not None:
            raise RuntimeError(
                f"[{self.can_interface}] background send loop failed"
            ) from self._background_error

    @staticmethod
    def _is_transient_send_error(exc: BaseException) -> bool:
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if getattr(current, "errno", None) == 105:
                return True
            text = str(current)
            if "No buffer space available" in text or "Error Code 105" in text:
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _drop_cached_register(motor: DaMiaoMotor, rid: int) -> None:
        registers = getattr(motor, "registers", None)
        registers_lock = getattr(motor, "registers_lock", None)
        if registers is None or registers_lock is None:
            return
        with registers_lock:
            registers.pop(rid, None)

    def _read_register_fresh(
        self, motor: DaMiaoMotor, rid: int, timeout: float = 1.0
    ) -> float | int:
        self._drop_cached_register(motor, rid)
        return motor.get_register(rid, timeout=timeout)

    def _ensure_control_mode_fresh(
        self, motor: DaMiaoMotor, control_mode: str
    ) -> None:
        """Set register 10 and verify with a fresh read, not the SDK cache."""
        mode_to_register = {"MIT": 1, "POS_VEL": 2, "VEL": 3, "FORCE_POS": 4}
        if control_mode not in mode_to_register:
            raise ValueError(
                f"Invalid control_mode: {control_mode}. Must be one of {list(mode_to_register.keys())}"
            )
        desired = mode_to_register[control_mode]
        current = int(self._read_register_fresh(motor, 10, timeout=1.0))
        if current == desired:
            return

        print(
            f"⚠ Control mode mismatch: register 10 = {current}, required = {desired}"
        )
        print(
            f"  Setting control mode to {control_mode} (register value: {desired})..."
        )
        motor.write_register(10, desired)
        time.sleep(0.1)
        verify = int(self._read_register_fresh(motor, 10, timeout=1.0))
        if verify != desired:
            raise RuntimeError(
                f"Control mode verification failed after write: expected {desired}, got {verify}"
            )
        print(f"✓ Control mode set to {control_mode}")

    def _motor_to_normalized(self, motor_pos: float) -> float:
        """Convert raw motor position → normalized [0=close, 1=open]."""
        env_pos = self.gripper_sign * motor_pos
        t = (env_pos - self.gripper_close_pos) / (self.gripper_open_pos - self.gripper_close_pos)
        return float(np.clip(t, 0.0, 1.0))

    def _normalized_to_motor(self, t: float) -> float:
        """Convert normalized [0=close, 1=open] → raw motor position."""
        env_pos = self.gripper_close_pos + t * (self.gripper_open_pos - self.gripper_close_pos)
        return self.gripper_sign * env_pos  # sign=±1, so 1/sign == sign

    def _calibrate_gripper(self, motor: DaMiaoMotor) -> None:
        """Move gripper to both physical stops to determine close/open positions."""
        import time
        CAL_VEL     = 5.0    # rad/s, slow for safety
        CAL_TORQUE  = 0.3    # torque limit ratio
        SETTLE_TIME = 2.5    # seconds to wait at each stop
        MOTOR_RANGE = motor._p_max  # position limit from motor type preset

        print(f"  Calibrating gripper on {self.can_interface}...")
        env_positions = []
        for cmd in [-MOTOR_RANGE, MOTOR_RANGE]:
            deadline = time.monotonic() + SETTLE_TIME
            while time.monotonic() < deadline:
                motor.send_cmd_force_pos(cmd, CAL_VEL, CAL_TORQUE)
                time.sleep(0.02)
            env_positions.append(self.gripper_sign * motor.get_states()["pos"])

        self.gripper_close_pos = min(env_positions)
        self.gripper_open_pos  = max(env_positions)
        print(f"  Gripper calibrated: close={self.gripper_close_pos:.3f}, open={self.gripper_open_pos:.3f} (env rad)")

    def _compute_gravity_compensation(
        self, joint_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gravity torques for arm joints from current positions."""
        grav = np.zeros(self.NUM_JOINTS)
        if self._kdl is None:
            return grav
        try:
            if joint_positions is None:
                # Read current raw motor positions (arm joints only, exclude gripper).
                arm_pos = []
                for i, motor in enumerate(self.motors):
                    if motor is None:
                        return grav
                    if self.gripper_index is not None and i == self.gripper_index:
                        continue
                    arm_pos.append(motor.get_states()["pos"])
                q = np.array(arm_pos)
            else:
                joint_positions = np.asarray(joint_positions, dtype=float).reshape(-1)
                if joint_positions.size != self.NUM_JOINTS:
                    raise ValueError(
                        f"Expected {self.NUM_JOINTS} joint positions, got {joint_positions.size}"
                    )
                if self.gripper_index is not None:
                    q = np.delete(joint_positions, self.gripper_index)
                else:
                    q = joint_positions
            with self._kdl_lock:
                raw = self._kdl.compute_inverse_dynamics(q, np.zeros_like(q), np.zeros_like(q))
            if np.max(np.abs(raw)) > 20.0:
                logging.warning(f"Gravity torques too large: {raw}, skipping")
                return grav
            j = 0
            for i in range(self.NUM_JOINTS):
                if self.gripper_index is not None and i == self.gripper_index:
                    continue
                grav[i] = raw[j] * _GRAVITY_COMP_FACTOR
                j += 1
        except Exception as e:
            logging.exception(
                "[%s] Gravity compensation failed; disabling compensation for this cycle",
                self.can_interface,
            )
        return grav

    def _gripper_torque_ratio_from_nm(self, motor_index: int, torque_limit_nm: float) -> float:
        """Convert torque limit in Nm to damiao FORCE_POS torque_limit_ratio."""
        motor_type = self.motor_types[motor_index]
        preset = MOTOR_TYPE_PRESETS.get(motor_type, {})
        t_max = float(preset.get("t_max", 0.0))
        if t_max <= 0.0:
            motor = self.motors[motor_index]
            t_max = float(getattr(motor, "_t_max", 0.0)) if motor is not None else 0.0
        if t_max <= 0.0:
            logging.warning(f"[{self.can_interface}] invalid T_max for gripper motor type '{motor_type}'")
            return 0.0
        tau = float(np.clip(torque_limit_nm, 0.0, t_max))
        return tau / t_max

    def _do_send(
        self,
        joint_pos: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        gripper_vel_limit: Optional[float],
        gripper_torque_limit_nm: Optional[float],
    ) -> None:
        """Send current command to all motors. Called from background loop."""
        if joint_pos.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected joint_pos shape {(self.NUM_JOINTS,)}, "
                f"got {joint_pos.shape}"
            )
        if kp.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kp shape {(self.NUM_JOINTS,)}, got {kp.shape}"
            )
        if kd.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kd shape {(self.NUM_JOINTS,)}, got {kd.shape}"
            )
        for i, motor in enumerate(self.motors):
            if motor is None:
                raise RuntimeError(f"[{self.can_interface}] motor slot {i} is not connected")
            actual = motor.get_states()["pos"]
            if self.gripper_index is not None and i == self.gripper_index:
                err = joint_pos[i] - self._motor_to_normalized(actual)
                # print(f"[{self.can_interface}] gripper err: {err:.4f} (normalized)")
            else:
                err = joint_pos[i] - actual
                # print(f"[{self.can_interface}] joint {i} err: {err:.4f} rad")

        grav = self._compute_gravity_compensation()
        self._last_gravity_comp = grav.copy()
        for i, motor in enumerate(self.motors):
            if motor is None:
                raise RuntimeError(f"[{self.can_interface}] motor slot {i} is not connected")
            if self.gripper_index is not None and i == self.gripper_index:
                vel_limit = self.gripper_vel_limit if gripper_vel_limit is None else float(gripper_vel_limit)
                torque_limit_nm = (
                    self.gripper_torque_limit_nm
                    if gripper_torque_limit_nm is None
                    else float(gripper_torque_limit_nm)
                )
                motor.send_cmd_force_pos(
                    target_position=self._normalized_to_motor(joint_pos[i]),
                    velocity_limit=vel_limit,
                    torque_limit_ratio=self._gripper_torque_ratio_from_nm(i, torque_limit_nm),
                )
            else:
                motor.send_cmd_mit(
                    target_position=joint_pos[i],
                    target_velocity=0.0,
                    stiffness=kp[i],
                    damping=kd[i],
                    feedforward_torque=grav[i],
                )

    def _send_loop(self) -> None:
        import time
        dt = 1.0 / self.send_rate_hz
        while not self._stop_event.is_set():
            with self._lock:
                pos = self._cmd_pos
                kp  = self._cmd_kp
                kd  = self._cmd_kd
                gripper_vel_limit = self._cmd_gripper_vel_limit
                gripper_torque_limit_nm = self._cmd_gripper_torque_limit_nm
            if pos is not None:
                try:
                    self._do_send(pos, kp, kd, gripper_vel_limit, gripper_torque_limit_nm)
                    self._transient_send_error_streak = 0
                except Exception as e:
                    if (
                        self._is_transient_send_error(e)
                        and self._transient_send_error_streak
                        < self.transient_send_error_max_streak
                    ):
                        self._transient_send_error_streak += 1
                        logging.warning(
                            "[%s] transient CAN send failure %s/%s; backing off %.3fs",
                            self.can_interface,
                            self._transient_send_error_streak,
                            self.transient_send_error_max_streak,
                            self.transient_send_error_backoff_s,
                            exc_info=True,
                        )
                        time.sleep(self.transient_send_error_backoff_s)
                        continue
                    self._background_error = e
                    logging.exception("[%s] background send loop crashed", self.can_interface)
                    self._stop_event.set()
                    raise
            time.sleep(dt)

    def connect(self) -> None:
        if len(self.feedback_ids) != self.NUM_JOINTS:
            raise ValueError(
                f"feedback_ids length {len(self.feedback_ids)} != motor_ids length {self.NUM_JOINTS}"
            )
        for i, (motor_id, feedback_id) in enumerate(zip(self.motor_ids, self.feedback_ids)):
            motor = self.controller.add_motor(
                motor_id, feedback_id, motor_type=self.motor_types[i]
            )
            is_gripper = self.gripper_index is not None and i == self.gripper_index
            mode = "FORCE_POS" if is_gripper else "MIT"
            self._ensure_control_mode_fresh(motor, mode)
            motor.enable()
            self.motors[i] = motor
            if is_gripper:
                if self.calibrate_gripper:
                    self._calibrate_gripper(motor)
                elif (
                    self._configured_gripper_close_pos is not None
                    and self._configured_gripper_open_pos is not None
                ):
                    self.gripper_close_pos = float(self._configured_gripper_close_pos)
                    self.gripper_open_pos = float(self._configured_gripper_open_pos)
                    print(
                        f"  Gripper calibration reused: close={self.gripper_close_pos:.3f}, "
                        f"open={self.gripper_open_pos:.3f} (env rad)"
                    )
                else:
                    raise RuntimeError(
                        f"[{self.can_interface}] gripper calibration skipped but "
                        "no cached close/open positions were provided"
                    )
        self._connected = True

        # Load MuJoCo model for gravity compensation
        xml_path = os.path.normpath(_YAM_XML_PATH)
        if os.path.exists(xml_path):
            self._kdl = MuJoCoKDL(xml_path)
            logging.info(f"Loaded MuJoCo model for gravity comp: {xml_path}")
        else:
            logging.warning(f"MuJoCo XML not found at {xml_path}, gravity comp disabled")

        self._background_error = None
        self._transient_send_error_streak = 0
        self._stop_event.clear()
        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name=f"yam-send-{self.can_interface}",
        )
        self._send_thread.start()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None
        disable_errors = []
        for motor in self.motors:
            if motor is not None:
                try:
                    motor.disable()
                except Exception as e:
                    disable_errors.append(e)
        self.motors = [None] * self.NUM_JOINTS
        self._connected = False
        if disable_errors:
            raise RuntimeError(
                f"[{self.can_interface}] failed to disable {len(disable_errors)} motor(s)"
            ) from disable_errors[0]
        self._raise_if_background_error()

    def _read_states(self) -> Optional[List[dict]]:
        self._raise_if_background_error()
        if not self._connected:
            raise RuntimeError(f"[{self.can_interface}] robot is not connected")
        states = []
        for i, motor in enumerate(self.motors):
            if motor is None:
                raise RuntimeError(f"[{self.can_interface}] motor slot {i} is not connected")
            states.append(motor.get_states())
        return states

    def get_joint_pos(self) -> np.ndarray:
        """Returns arm positions (rad) + gripper normalized [0=close, 1=open]."""
        states = self._read_states()
        pos = np.array([s["pos"] for s in states])
        if self.gripper_index is not None:
            pos[self.gripper_index] = self._motor_to_normalized(pos[self.gripper_index])
        return pos

    def get_joint_vel(self) -> np.ndarray:
        states = self._read_states()
        return np.array([s["vel"] for s in states])

    def get_observations(self) -> dict[str, np.ndarray]:
        """Returns joint_pos (arm rad), gripper_pos, joint_vel, joint_eff, gravity_comp, and force_feedback_torque."""
        states = self._read_states()
        raw_pos = np.array([s["pos"] for s in states])
        pos = raw_pos.copy()
        if self.gripper_index is not None:
            pos[self.gripper_index] = self._motor_to_normalized(pos[self.gripper_index])
        joint_eff = np.array([s["torq"] for s in states])
        gravity_comp = self._last_gravity_comp.copy()
        obs = {
            "joint_pos": pos,
            "joint_vel": np.array([s["vel"] for s in states]),
            "joint_eff": joint_eff,
            "gravity_comp": gravity_comp,
            "force_feedback_torque": joint_eff - gravity_comp,
        }
        if self.gripper_index is not None:
            obs["joint_pos"] = pos[:self.gripper_index]
            obs["gripper_pos"] = pos[self.gripper_index:self.gripper_index + 1]
        return obs

    def get_raw_gripper_diagnostics(self) -> dict:
        """Return raw gripper motor state and current normalized mapping.

        This is intentionally read-only. It is for debugging cases where the
        physical jaw state disagrees with the normalized [0=closed, 1=open]
        value reported by get_joint_pos()/get_observations().
        """
        states = self._read_states()
        if self.gripper_index is None:
            raise RuntimeError(f"[{self.can_interface}] no gripper configured")
        state = states[self.gripper_index]
        raw_pos = float(state.get("pos", float("nan")))
        raw_vel = float(state.get("vel", float("nan")))
        raw_torque = float(
            state.get(
                "torq",
                state.get("eff", state.get("torque", float("nan"))),
            )
        )

        targets = {}
        for value in (0.0, 0.4, 0.68, 1.0):
            targets[f"{value:.2f}"] = float(self._normalized_to_motor(value))

        with self._lock:
            cmd_pos = None if self._cmd_pos is None else self._cmd_pos.copy()
            cmd_gripper_vel_limit = self._cmd_gripper_vel_limit
            cmd_gripper_torque_limit_nm = self._cmd_gripper_torque_limit_nm

        cmd_normalized = None
        cmd_raw = None
        if cmd_pos is not None:
            cmd_normalized = float(cmd_pos[self.gripper_index])
            cmd_raw = float(self._normalized_to_motor(cmd_normalized))

        return {
            "can_interface": self.can_interface,
            "motor_id": int(self.motor_ids[self.gripper_index]),
            "motor_type": str(self.motor_types[self.gripper_index]),
            "gripper_index": int(self.gripper_index),
            "raw_state": {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in state.items()
            },
            "raw_pos": raw_pos,
            "raw_vel": raw_vel,
            "raw_torque": raw_torque,
            "normalized_from_raw": float(self._motor_to_normalized(raw_pos)),
            "gripper_sign": int(self.gripper_sign),
            "configured_close_env_pos": None
            if self.gripper_close_pos is None
            else float(self.gripper_close_pos),
            "configured_open_env_pos": None
            if self.gripper_open_pos is None
            else float(self.gripper_open_pos),
            "raw_targets_for_normalized": targets,
            "active_command": {
                "normalized": cmd_normalized,
                "raw_target": cmd_raw,
                "gripper_vel_limit": cmd_gripper_vel_limit,
                "gripper_torque_limit_nm": cmd_gripper_torque_limit_nm,
            },
        }

    def get_motor_temperatures(self) -> np.ndarray:
        """Returns per-motor temperature in Celsius (NaN when unavailable)."""
        states = self._read_states()

        def _extract_temp(state: dict) -> float:
            vals = [
                state.get("t_mos"),
                state.get("t_rotor"),
                state.get("temp_mos"),
                state.get("temp_rotor"),
                state.get("temperature_mos"),
                state.get("temperature_rotor"),
                state.get("temp"),
                state.get("temperature"),
            ]
            numeric: list[float] = []
            for v in vals:
                if v is None:
                    continue
                fv = float(v)
                if np.isfinite(fv):
                    numeric.append(fv)
            return max(numeric) if numeric else float("nan")

        temps = [_extract_temp(s) for s in states]
        return np.asarray(temps, dtype=np.float32)

    def get_health(self) -> dict:
        return {
            "can_interface": self.can_interface,
            "connected": bool(self._connected),
            "send_thread_alive": bool(
                self._send_thread is not None and self._send_thread.is_alive()
            ),
            "background_error": None
            if self._background_error is None
            else f"{type(self._background_error).__name__}: {self._background_error}",
            "transient_send_error_streak": int(self._transient_send_error_streak),
            "transient_send_error_max_streak": int(self.transient_send_error_max_streak),
            "send_rate_hz": float(self.send_rate_hz),
            "gripper_close_pos": self.gripper_close_pos,
            "gripper_open_pos": self.gripper_open_pos,
        }

    def command_joint_pos(
        self,
        joint_pos: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        gripper_vel_limit: Optional[float] = None,
        gripper_torque_limit_nm: Optional[float] = None,
    ) -> None:
        """Update target position. Background loop sends continuously at send_rate_hz."""
        self._raise_if_background_error()
        if not self._connected:
            raise RuntimeError(f"[{self.can_interface}] cannot command joints before connect()")
        if kp is None:
            kp = self.default_kp
        if kd is None:
            kd = self.default_kd
        joint_pos = np.asarray(joint_pos, dtype=float)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        if joint_pos.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected joint_pos shape {(self.NUM_JOINTS,)}, "
                f"got {joint_pos.shape}"
            )
        if kp.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kp shape {(self.NUM_JOINTS,)}, got {kp.shape}"
            )
        if kd.shape != (self.NUM_JOINTS,):
            raise ValueError(
                f"[{self.can_interface}] expected kd shape {(self.NUM_JOINTS,)}, got {kd.shape}"
            )
        with self._lock:
            self._cmd_pos = joint_pos
            self._cmd_kp  = kp
            self._cmd_kd  = kd
            self._cmd_gripper_vel_limit = (
                None if gripper_vel_limit is None else float(gripper_vel_limit)
            )
            self._cmd_gripper_torque_limit_nm = (
                None if gripper_torque_limit_nm is None else float(gripper_torque_limit_nm)
            )

    def command_joint_state(self, joint_state: dict) -> None:
        self.command_joint_pos(
            joint_state["pos"],
            joint_state["kp"],
            joint_state["kd"],
            joint_state.get("gripper_vel_limit"),
            joint_state.get("gripper_torque_limit_nm"),
        )
