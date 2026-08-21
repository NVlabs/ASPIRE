# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Server for controlling YAM robot arm from client process.
"""

import logging
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import portal
import tyro
from portal import contextlib as portal_contextlib

from robot.constants import (
    CAN_BUSTYPE,
    LEFT_FOLLOWER_CAN_INTERFACE,
    LEFT_FOLLOWER_PORT,
    LEFT_LEADER_CAN_INTERFACE,
    LEFT_LEADER_PORT,
    RIGHT_FOLLOWER_CAN_INTERFACE,
    RIGHT_FOLLOWER_PORT,
    RIGHT_LEADER_CAN_INTERFACE,
    RIGHT_LEADER_PORT,
    YAM_ARM_KD,
    YAM_ARM_KP,
    YAM_ARM_MOTOR_IDS,
    YAM_ARM_MOTOR_TYPES,
    YAM_GRIPPER_KD,
    YAM_GRIPPER_KP,
    YAM_GRIPPER_MOTOR_ID,
    YAM_GRIPPER_MOTOR_TYPE,
    YAM_GRIPPER_SIGN,
    YAM_GRIPPER_TORQUE_LIMIT_NM,
    YAM_GRIPPER_VEL_LIMIT,
)
from robot.yam.yam_controller import YamRobot

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType


def _install_last_chance_diagnostics() -> None:
    try:
        import faulthandler

        faulthandler.enable(all_threads=True)
        for sig in (signal.SIGUSR1,):
            try:
                faulthandler.register(sig, all_threads=True)
            except Exception:
                logging.exception("Failed to register faulthandler signal for %s", sig)
    except Exception:
        logging.exception("Failed to enable faulthandler")

    original_os_exit = os._exit

    def _patched_os_exit(code: int) -> None:
        try:
            logging.critical("os._exit(%s) called unexpectedly", code)
            current_frames = sys._current_frames()
            for thread in threading.enumerate():
                frame = current_frames.get(thread.ident)
                if frame is None:
                    continue
                stack = "".join(traceback.format_stack(frame))
                logging.critical("Thread %s stack at os._exit:\n%s", thread.name, stack)
        except Exception:
            logging.exception("Failed while logging unexpected os._exit(%s)", code)
        original_os_exit(code)

    def _signal_handler(signum, frame):
        stack = "".join(traceback.format_stack(frame))
        logging.critical("Received signal %s\n%s", signum, stack)
        raise SystemExit(128 + signum)

    def _threading_excepthook(args):
        logging.critical(
            "Unhandled thread exception in %s",
            args.thread.name if args.thread else "<unknown>",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = lambda exc_type, exc_value, exc_traceback: logging.critical(
        "Unhandled exception at sys.excepthook",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    threading.excepthook = _threading_excepthook
    os._exit = _patched_os_exit

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGABRT):
        try:
            signal.signal(sig, _signal_handler)
        except Exception:
            logging.exception("Failed to install signal handler for %s", sig)


def _disable_portal_process_exit() -> None:
    """Prevent portal worker-thread failures from calling os._exit()."""

    def _patched_shutdown(exitcode: int) -> None:
        logging.critical(
            "Portal requested process shutdown with exitcode=%s; "
            "suppressing os._exit so the main thread can raise explicitly",
            exitcode,
        )

    portal_contextlib.context.shutdown = _patched_shutdown


# ---------------------------------------------------------------------------
# Follower — uses YamRobot (damiao_motor directly)
# ---------------------------------------------------------------------------


class FollowerRobotServer:
    def __init__(self, robot: YamRobot, port: int):
        self._robot = robot
        self._server = portal.Server(port, errors=False)
        self._server.bind(
            "get_joint_pos", self._wrap_rpc("get_joint_pos", self._robot.get_joint_pos)
        )
        self._server.bind(
            "command_joint_pos",
            self._wrap_rpc("command_joint_pos", self._robot.command_joint_pos),
        )
        self._server.bind(
            "command_joint_state",
            self._wrap_rpc("command_joint_state", self._robot.command_joint_state),
        )
        self._server.bind(
            "get_observations",
            self._wrap_rpc("get_observations", self._robot.get_observations),
        )
        self._server.bind(
            "get_motor_temperatures",
            self._wrap_rpc(
                "get_motor_temperatures", self._robot.get_motor_temperatures
            ),
        )
        self._server.bind(
            "get_raw_gripper_diagnostics",
            self._wrap_rpc(
                "get_raw_gripper_diagnostics",
                self._robot.get_raw_gripper_diagnostics,
            ),
        )
        self._server.bind("get_health", self._wrap_rpc("get_health", self._robot.get_health))

    def _wrap_rpc(self, method_name: str, fn):
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logging.exception("Follower RPC %s failed", method_name)
                raise

        return wrapped

    def serve(self) -> None:
        self._serve_forever()

    def _serve_forever(self) -> None:
        self._server.start(block=False)
        try:
            while True:
                if not self._server.loop.running:
                    raise RuntimeError(
                        f"Follower portal loop exited unexpectedly with exitcode={self._server.loop.exitcode}"
                    )
                if not self._server.socket.thread.running:
                    sock_error = self._server.socket.error
                    if sock_error is not None:
                        raise RuntimeError(
                            "Follower portal socket thread crashed"
                        ) from sock_error
                    raise RuntimeError(
                        "Follower portal socket thread exited unexpectedly"
                    )
                time.sleep(0.2)
        finally:
            if self._server.running:
                self._server.close(timeout=1.0)


# ---------------------------------------------------------------------------
# Leader — uses i2rt MotorChainRobot (teaching handle + encoder chain)
# ---------------------------------------------------------------------------


class LeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self._robot.command_joint_state(joint_state)

    @staticmethod
    def _extract_temp(state: Any) -> float:
        if isinstance(state, dict):
            vals = [
                state.get("t_mos"),
                state.get("t_rotor"),
                state.get("temp_mos"),
                state.get("temperature_mos"),
                state.get("temp_rotor"),
                state.get("temperature_rotor"),
                state.get("temp"),
                state.get("temperature"),
            ]
        else:
            vals = [
                getattr(state, k, None)
                for k in (
                    "t_mos",
                    "t_rotor",
                    "temp_mos",
                    "temperature_mos",
                    "temp_rotor",
                    "temperature_rotor",
                    "temp",
                    "temperature",
                )
            ]
        numeric: list[float] = []
        for v in vals:
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                numeric.append(fv)
        return max(numeric) if numeric else float("nan")

    def get_motor_temperatures(self) -> np.ndarray:
        try:
            states = self._motor_chain.read_states()
        except Exception:
            logging.exception("Failed to read leader motor temperatures")
            raise
        if states is None:
            raise RuntimeError("Leader motor state read returned None")
        temps = [self._extract_temp(s) for s in states]
        return np.asarray(temps, dtype=np.float32)


class LeaderRobotServer:
    def __init__(self, robot: LeaderRobot, port: int):
        self._robot = robot
        self._server = portal.Server(port, errors=False)
        self._server.bind("get_info", self._wrap_rpc("get_info", self._robot.get_info))
        self._server.bind(
            "command_joint_state",
            self._wrap_rpc("command_joint_state", self._robot.command_joint_state),
        )
        self._server.bind(
            "get_motor_temperatures",
            self._wrap_rpc(
                "get_motor_temperatures", self._robot.get_motor_temperatures
            ),
        )

    def _wrap_rpc(self, method_name: str, fn):
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logging.exception("Leader RPC %s failed", method_name)
                raise

        return wrapped

    def serve(self) -> None:
        self._serve_forever()

    def _serve_forever(self) -> None:
        self._server.start(block=False)
        try:
            while True:
                if not self._server.loop.running:
                    raise RuntimeError(
                        f"Leader portal loop exited unexpectedly with exitcode={self._server.loop.exitcode}"
                    )
                if not self._server.socket.thread.running:
                    sock_error = self._server.socket.error
                    if sock_error is not None:
                        raise RuntimeError(
                            "Leader portal socket thread crashed"
                        ) from sock_error
                    raise RuntimeError(
                        "Leader portal socket thread exited unexpectedly"
                    )
                time.sleep(0.2)
        finally:
            if self._server.running:
                self._server.close(timeout=1.0)


# ---------------------------------------------------------------------------


@dataclass
class Args:
    mode: Literal["follower", "leader"] = "follower"
    side: Literal["left", "right"] = "left"


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_last_chance_diagnostics()
    _disable_portal_process_exit()
    CAN_INTERFACE_MAP = {
        "follower": {
            "left": LEFT_FOLLOWER_CAN_INTERFACE,
            "right": RIGHT_FOLLOWER_CAN_INTERFACE,
        },
        "leader": {
            "left": LEFT_LEADER_CAN_INTERFACE,
            "right": RIGHT_LEADER_CAN_INTERFACE,
        },
    }
    PORT_MAP = {
        "follower": {"left": LEFT_FOLLOWER_PORT, "right": RIGHT_FOLLOWER_PORT},
        "leader": {"left": LEFT_LEADER_PORT, "right": RIGHT_LEADER_PORT},
    }

    can_interface = CAN_INTERFACE_MAP[args.mode][args.side]
    if can_interface is None:
        raise RuntimeError(
            f"CAN interface not configured for {args.side} {args.mode} — "
            f"set {args.side.upper()}_{args.mode.upper()}_CAN_INTERFACE in robot/constants.py"
        )
    port = PORT_MAP[args.mode][args.side]
    print(
        f"Starting server for {args.side} {args.mode} arm at localhost:{port} (bustype={CAN_BUSTYPE})"
    )

    if args.mode == "follower":
        close_env = os.environ.get(f"YAM_{args.side.upper()}_GRIPPER_CLOSE_POS")
        open_env = os.environ.get(f"YAM_{args.side.upper()}_GRIPPER_OPEN_POS")
        if close_env is None:
            close_env = os.environ.get("YAM_GRIPPER_CLOSE_POS")
        if open_env is None:
            open_env = os.environ.get("YAM_GRIPPER_OPEN_POS")
        skip_calibration = os.environ.get("YAM_SKIP_GRIPPER_CALIBRATION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        send_rate_hz = float(os.environ.get("YAM_FOLLOWER_SEND_RATE_HZ", "50"))
        transient_max_streak = int(
            os.environ.get("YAM_TRANSIENT_SEND_ERROR_MAX_STREAK", "0")
        )
        transient_backoff_s = float(
            os.environ.get("YAM_TRANSIENT_SEND_ERROR_BACKOFF_S", "0.02")
        )
        if skip_calibration:
            print(
                f"Using cached gripper calibration for {args.side} follower "
                f"(close={close_env}, open={open_env})"
            )
        print(
            f"Follower send_rate_hz={send_rate_hz}, "
            f"transient_send_error_max_streak={transient_max_streak}"
        )
        robot = YamRobot(
            can_interface=can_interface,
            motor_ids=YAM_ARM_MOTOR_IDS + [YAM_GRIPPER_MOTOR_ID],
            motor_types=YAM_ARM_MOTOR_TYPES + [YAM_GRIPPER_MOTOR_TYPE],
            default_kp=YAM_ARM_KP + [YAM_GRIPPER_KP],
            default_kd=YAM_ARM_KD + [YAM_GRIPPER_KD],
            gripper_index=len(YAM_ARM_MOTOR_IDS),
            gripper_sign=YAM_GRIPPER_SIGN,
            gripper_vel_limit=YAM_GRIPPER_VEL_LIMIT,
            gripper_torque_limit_nm=YAM_GRIPPER_TORQUE_LIMIT_NM,
            send_rate_hz=send_rate_hz,
            calibrate_gripper=not skip_calibration,
            gripper_close_pos=None if close_env is None else float(close_env),
            gripper_open_pos=None if open_env is None else float(open_env),
            transient_send_error_max_streak=transient_max_streak,
            transient_send_error_backoff_s=transient_backoff_s,
            bustype=CAN_BUSTYPE,
        )
        robot.connect()
        FollowerRobotServer(robot, port).serve()
    else:
        robot = get_yam_robot(
            channel=can_interface,
            gripper_type=GripperType.from_string_name("yam_teaching_handle"),
        )
        LeaderRobotServer(LeaderRobot(robot), port).serve()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except Exception:
        logging.exception("YAM arm server crashed")
        raise
