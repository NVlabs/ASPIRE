"""Level-1 controller: waypoints resampled against the wall clock at a fixed rate.

The cascade this sits in:

* **level 1** (here, in process) turns sparse waypoints -- a planner trajectory,
  a homing target -- into a fixed-rate stream of joint7 position targets with
  the profile's gains, interpolating from wherever the arm currently is;
* **level 2** (the arm server, ~100 Hz) re-sends the latest buffered target;
* **level 3** (the CAN motor driver, 250 Hz) and the motor firmware close the
  MIT law ``tau = kp*(q_d - q) + kd*(0 - qdot) + tau_gravity``.

Resampling against the wall clock is the reason this layer exists. A planner
retimes its output at whatever period it likes -- cuRobo uses 1/30 s -- and a
position-controlled arm fed those waypoints raw either races or crawls. Sampling
by elapsed time means the trajectory takes the duration it says it takes,
independent of how many waypoints express it.

The settle window at the end matters for the same reason: the arm lags its
target, so the last waypoint has to be *held* for the joint to close its
following error before anything downstream reads the pose as final.
"""

from __future__ import annotations

import signal
import threading
import time
from typing import Any

import numpy as np

ARMS = ("left", "right")

#: Version of the plant description stamped into every recorded episode.
#:
#: v4 supersedes the reference branch's ``yam-direct-v3-control-modes``. v3
#: episodes could select a compliance mode per action batch, so their gains were
#: not a constant; this plant has one fixed gain set. The version string differs
#: so a v3 dataset is never read as though it were this plant.
CONTROL_CONTRACT_VERSION = "yam-direct-v4-fixed-gain"

_stop_requested = threading.Event()
_pause_requested = threading.Event()


def stop_requested() -> bool:
    """True once a stop has been requested and not yet reset."""
    return _stop_requested.is_set()


def pause_requested() -> bool:
    return _pause_requested.is_set()


def request_stop() -> None:
    """Ask the streaming loop to abort at its next tick."""
    _stop_requested.set()


def reset_stop() -> None:
    """Clear a previous stop, so a new run can start."""
    _stop_requested.clear()
    _pause_requested.clear()


def install_stop_handler() -> None:
    """Route SIGINT to the stop flag.

    Called by a runner that owns the process, never at import time: installing a
    signal handler as an import side effect hijacks Ctrl-C for anything that
    merely imports this module, including the test runner and any embedding
    host. A library does not get to decide what SIGINT means for a process it
    does not own.
    """

    def _handler(signum: int, frame: Any) -> None:
        del signum, frame
        if not _stop_requested.is_set():
            print("\n[YAM] Stop requested.")
        _stop_requested.set()

    signal.signal(signal.SIGINT, _handler)


def _sample(timestamps: np.ndarray, values: np.ndarray, t_now: float) -> np.ndarray:
    """Linearly interpolate a waypoint array at an elapsed time."""
    if t_now <= float(timestamps[0]):
        return values[0]
    if t_now >= float(timestamps[-1]):
        return values[-1]
    index = int(np.searchsorted(timestamps, t_now, side="right") - 1)
    index = max(0, min(index, len(timestamps) - 2))
    t0, t1 = float(timestamps[index]), float(timestamps[index + 1])
    alpha = 1.0 if t1 <= t0 + 1e-9 else (float(t_now) - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * values[index] + alpha * values[index + 1]


def _command_tick(env: Any, left7: np.ndarray, right7: np.ndarray) -> None:
    """Send one bimanual tick with the profile's fixed gains."""
    controller = env.config.controller
    for side, command in (("left", left7), ("right", right7)):
        env.command_joint_state(
            side,
            {
                "pos": np.asarray(command, dtype=np.float64).reshape(7),
                "vel": np.zeros(7),
                "kp": controller.kp,
                "kd": controller.kd,
                "gripper_vel_limit": controller.gripper_velocity_limit,
                "gripper_torque_limit_nm": controller.gripper_torque_limit_nm,
            },
        )


def _joint7(joints: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Concatenate arm joints and gripper into the arm server's wire format."""
    return np.column_stack([joints, np.asarray(gripper, dtype=np.float64).reshape(-1)])


def execute_joint_trajectory(
    env: Any,
    timestamps: np.ndarray,
    left_joint_positions: np.ndarray,
    right_joint_positions: np.ndarray,
    left_gripper_positions: np.ndarray,
    right_gripper_positions: np.ndarray,
    *,
    command_hz: float,
    start_interp_s: float = 0.0,
    settle_s: float = 0.2,
    playback_speed: float = 1.0,
) -> dict[str, Any]:
    """Stream a bimanual joint trajectory at ``command_hz``.

    Returns a result dict rather than raising for a bad trajectory: the caller is
    the adapter, which turns this into a typed ``ExecutionResult``. A *stop*, by
    contrast, raises ``KeyboardInterrupt``, because a stop is not a result.
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 1:
        return {"success": False, "reason": "empty timestamps"}
    ts = ts - float(ts[0])

    # Duplicate timestamps make the interpolation ambiguous and contribute no
    # motion; drop the later of each pair rather than dividing by zero.
    keep = np.ones(ts.shape[0], dtype=bool)
    keep[1:] = np.diff(ts) > 1e-9
    ts = ts[keep]
    left = _joint7(np.asarray(left_joint_positions, dtype=np.float64), left_gripper_positions)[keep]
    right = _joint7(np.asarray(right_joint_positions, dtype=np.float64), right_gripper_positions)[
        keep
    ]

    speed = max(0.05, float(playback_speed))
    ts = ts / speed
    duration_s = float(ts[-1]) if ts.size else 0.0
    dt = 1.0 / max(1.0, float(command_hz))

    # Interpolate from where the arms actually are to the first waypoint, so a
    # trajectory that starts away from the current pose does not step there.
    interp_s = max(0.0, float(start_interp_s))
    interp_steps = int(np.ceil(interp_s / dt)) if interp_s > 1e-9 else 0
    if interp_steps:
        current = {side: env.get_observations(side) for side in ARMS}
        # The interpolation starts from where the arms ARE, and a measured pose
        # can sit just outside the commandable range. Clamping keeps the first
        # waypoint admissible without altering the trajectory being executed.
        limits = env.config
        start = {
            side: np.concatenate(
                [
                    np.clip(
                        np.asarray(current[side]["joint_pos"], dtype=np.float64).reshape(6),
                        limits.joint_limits_lower,
                        limits.joint_limits_upper,
                    ),
                    np.asarray(current[side]["gripper_pos"], dtype=np.float64).reshape(1),
                ]
            )
            for side in ARMS
        }
        for step in range(1, interp_steps + 1):
            _raise_if_stopped("during start interpolation", dt)
            alpha = float(step) / float(interp_steps)
            _command_tick(
                env,
                (1.0 - alpha) * start["left"] + alpha * left[0],
                (1.0 - alpha) * start["right"] + alpha * right[0],
            )
            time.sleep(dt)

    t0 = time.time()
    command_count = 0
    while True:
        _raise_if_stopped("during trajectory execution", dt)
        t_now = time.time() - t0
        _command_tick(env, _sample(ts, left, t_now), _sample(ts, right, t_now))
        command_count += 1
        if t_now >= duration_s:
            break
        time.sleep(dt)

    # Hold the final waypoint so the arm can close its following error.
    for _ in range(max(0, round(max(0.0, float(settle_s)) / dt))):
        _raise_if_stopped("during settle", dt)
        _command_tick(env, left[-1], right[-1])
        time.sleep(dt)

    return {
        "success": True,
        "reason": "ok",
        "waypoints": int(ts.size),
        "duration_s": round(duration_s, 4),
        "command_hz": float(command_hz),
        "command_count": int(command_count),
        "start_interp_s": float(interp_s),
        "settle_s": float(settle_s),
        "playback_speed": float(speed),
        "final_left_gripper": float(left[-1, 6]),
        "final_right_gripper": float(right[-1, 6]),
    }


def _raise_if_stopped(context: str, dt: float) -> None:
    """Honour stop and pause between ticks."""
    if _stop_requested.is_set():
        raise KeyboardInterrupt(f"stop requested {context}")
    while _pause_requested.is_set():
        if _stop_requested.is_set():
            raise KeyboardInterrupt(f"stop requested while paused {context}")
        time.sleep(dt)


def build_control_contract(env: Any) -> dict[str, Any]:
    """Describe the plant an episode's actions were executed under.

    Data collected under one contract only transfers to a deployment that
    reproduces it -- same gains, same rate, same interpolation -- so this is
    stamped per episode and should be compared before training across datasets.
    """
    controller = env.config.controller
    return {
        "version": CONTROL_CONTRACT_VERSION,
        "action_semantics": (
            "joint6+gripper1 absolute position target, sampled at the "
            "RealYamEnv.execute_action_batch boundary"
        ),
        "action_space": "joint_abs",
        "command_stream_hz": float(controller.command_stream_hz),
        "follower_hold_hz": float(controller.follower_hold_frequency_hz),
        "interpolation": "linear, wall-clock time-sampled",
        "tracker": "arm-server PD (level 2) over the CAN motor driver (level 3)",
        "owner": "cap_harness.yam_real.control.controller (level 1)",
        "gripper_action": "continuous position target in [0, 1]; 0.0=closed, 1.0=open",
        "gripper_vel_limit": float(controller.gripper_velocity_limit),
        "gripper_torque_limit_nm": float(controller.gripper_torque_limit_nm),
        "compliance_axis": "none; gains are fixed for every batch on this plant",
        "interp_kp": np.asarray(controller.kp).tolist(),
        "interp_kd": np.asarray(controller.kd).tolist(),
        "station": env.config.station,
        "calibration_bundle": env.config.calibration.bundle_id,
        "config_sha256": env.config.config_sha256,
    }


__all__ = [
    "CONTROL_CONTRACT_VERSION",
    "build_control_contract",
    "execute_joint_trajectory",
    "install_stop_handler",
    "pause_requested",
    "request_stop",
    "reset_stop",
    "stop_requested",
]
