# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robot profile abstraction for CAP server.

Encapsulates all robot-specific constants (DOF, joint limits, motor gains,
camera names, EE frame names, model paths) into a pluggable frozen dataclass.
Adding support for a new robot requires only a new factory function here —
no changes to cap_server.py or the tool layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArmProfile:
    """Configuration for a single robot arm."""

    name: str
    """Arm identifier used as key in dicts and state key prefix, e.g. "left", "right"."""

    dof: int
    """Number of joint degrees of freedom (excluding gripper). 6 for YAM, 7 for Panda."""

    joint_limits_low: np.ndarray
    """Lower joint limits, shape (dof,)."""

    joint_limits_high: np.ndarray
    """Upper joint limits, shape (dof,)."""

    home_joint_pos: np.ndarray
    """Home/zero joint configuration, shape (dof,)."""

    home_gripper_pos: np.ndarray
    """Home gripper position, shape (1,)."""

    gripper_min: float
    """Minimum gripper value (closed)."""

    gripper_max: float
    """Maximum gripper value (open)."""

    interp_kp: np.ndarray
    """Impedance position gains, shape (dof+1,) — arm joints + gripper."""

    interp_kd: np.ndarray
    """Impedance damping gains, shape (dof+1,) — arm joints + gripper."""

    ee_frame_name: str
    """End-effector frame name in URDF for FK/IK, e.g. "left_grasp"."""

    q_slice: slice | None = None
    """Pinocchio q-vector slice for this arm's joints.
    e.g. slice(0, 6) for YAM left arm, slice(8, 14) for YAM right arm.
    None when FK/IK is handled by the backend (no URDF)."""

    def __post_init__(self):
        # Validate shapes
        assert self.joint_limits_low.shape == (self.dof,), (
            f"joint_limits_low shape {self.joint_limits_low.shape} != ({self.dof},)"
        )
        assert self.joint_limits_high.shape == (self.dof,), (
            f"joint_limits_high shape {self.joint_limits_high.shape} != ({self.dof},)"
        )
        assert self.home_joint_pos.shape == (self.dof,), (
            f"home_joint_pos shape {self.home_joint_pos.shape} != ({self.dof},)"
        )
        assert self.home_gripper_pos.shape == (1,), (
            f"home_gripper_pos shape {self.home_gripper_pos.shape} != (1,)"
        )
        assert self.interp_kp.shape == (self.dof + 1,), (
            f"interp_kp shape {self.interp_kp.shape} != ({self.dof + 1},)"
        )
        assert self.interp_kd.shape == (self.dof + 1,), (
            f"interp_kd shape {self.interp_kd.shape} != ({self.dof + 1},)"
        )

    def __hash__(self):
        return hash((self.name, self.dof, self.ee_frame_name))

    def __eq__(self, other):
        if not isinstance(other, ArmProfile):
            return NotImplemented
        return self.name == other.name and self.dof == other.dof


@dataclass(frozen=True)
class RobotProfile:
    """Complete robot configuration consumed by CapServer."""

    name: str
    """Profile identifier, e.g. "yam", "panda_omron", "gr1_arms_only"."""

    arms: dict[str, ArmProfile]
    """Per-arm configurations, keyed by arm name (e.g. "left", "right")."""

    camera_names: tuple[str, ...]
    """Logical camera names used by CAP tools, e.g. ("top", "left", "right")."""

    control_freq_hz: float
    """Control loop frequency in Hz."""

    policy_freq_hz: float
    """Policy action frequency in Hz."""

    urdf_path: Path | None = None
    """Path to URDF for FK/IK via pinocchio. None if backend provides FK directly."""

    xml_path: Path | None = None
    """Path to MuJoCo XML model. None for non-MuJoCo backends."""

    camera_obs_key_map: dict[str, str] | None = None
    """Mapping from CAP camera names to backend observation keys.
    e.g. {"top": "robot0_agentview_image"}. None for backends where
    camera names match directly (MuJoCo SimBackend)."""

    is_mobile_base: bool = False
    """True for robots whose arm base moves (PandaOmron, Tiago).
    When True, _ik_servo applies a world→arm-local frame transform
    before solving IK, since the arm URDF origin ≠ world origin."""

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(self.arms.keys())

    @property
    def is_bimanual(self) -> bool:
        return len(self.arms) >= 2

    @property
    def total_arm_dof(self) -> int:
        return sum(arm.dof for arm in self.arms.values())

    def __hash__(self):
        return hash(self.name)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def yam_profile() -> RobotProfile:
    """YAM bimanual station — the original CAP robot.

    Sources values from cap/config.py constants for exact backward compatibility.
    """
    from cap.config import (
        CAMERA_NAMES,
        CONTROL_FREQ_HZ,
        GRIPPER_MAX,
        GRIPPER_MIN,
        HOME_JOINT_STATE,
        INTERP_KD,
        INTERP_KP,
        JOINT_LIMITS_HIGH,
        JOINT_LIMITS_LOW,
        POLICY_FREQ_HZ,
    )

    left_arm = ArmProfile(
        name="left",
        dof=6,
        joint_limits_low=JOINT_LIMITS_LOW[:6].copy(),
        joint_limits_high=JOINT_LIMITS_HIGH[:6].copy(),
        home_joint_pos=HOME_JOINT_STATE["left_joint_pos"].copy(),
        home_gripper_pos=HOME_JOINT_STATE["left_gripper_pos"].copy(),
        gripper_min=GRIPPER_MIN,
        gripper_max=GRIPPER_MAX,
        interp_kp=INTERP_KP.copy(),
        interp_kd=INTERP_KD.copy(),
        ee_frame_name="left_grasp",
        q_slice=slice(0, 6),
    )

    right_arm = ArmProfile(
        name="right",
        dof=6,
        joint_limits_low=JOINT_LIMITS_LOW[6:].copy(),
        joint_limits_high=JOINT_LIMITS_HIGH[6:].copy(),
        home_joint_pos=HOME_JOINT_STATE["right_joint_pos"].copy(),
        home_gripper_pos=HOME_JOINT_STATE["right_gripper_pos"].copy(),
        gripper_min=GRIPPER_MIN,
        gripper_max=GRIPPER_MAX,
        interp_kp=INTERP_KP.copy(),
        interp_kd=INTERP_KD.copy(),
        ee_frame_name="right_grasp",
        q_slice=slice(8, 14),
    )

    # Resolve model paths (may fail if robot package not installed — that's ok,
    # paths are only needed when pinocchio FK/IK is used)
    urdf_path: Path | None = None
    xml_path: Path | None = None
    try:
        from robot.models.station.paths import get_station_urdf, get_station_xml

        urdf_path = Path(str(get_station_urdf()))
        xml_path = Path(str(get_station_xml()))
    except Exception:
        pass

    return RobotProfile(
        name="yam",
        arms={"left": left_arm, "right": right_arm},
        camera_names=CAMERA_NAMES,
        control_freq_hz=CONTROL_FREQ_HZ,
        policy_freq_hz=POLICY_FREQ_HZ,
        urdf_path=urdf_path,
        xml_path=xml_path,
    )


def robocasa_panda_omron_profile() -> RobotProfile:
    """PandaOmron single-arm mobile manipulator for RoboCasa365.

    7-DOF Panda arm + gripper + mobile base + torso.
    Uses the bullet_data Panda URDF for IK via pinocchio.
    """
    # Joint limits from the Panda URDF
    panda_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    panda_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    # Standard Panda "ready" pose (all joints within limits)
    panda_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    right_arm = ArmProfile(
        name="right",
        dof=7,
        joint_limits_low=panda_limits_low,
        joint_limits_high=panda_limits_high,
        home_joint_pos=panda_home,
        home_gripper_pos=np.array([1.0]),  # open
        gripper_min=0.0,
        gripper_max=1.0,
        interp_kp=np.full(8, 150.0),  # 7 arm + 1 gripper
        interp_kd=np.full(8, 1.0),
        ee_frame_name="panda_link8",  # pinocchio frame in panda_arm.urdf
        q_slice=slice(0, 7),  # Panda URDF: q-vector is just the 7 arm joints
    )

    # Resolve Panda URDF from robosuite
    urdf_path: Path | None = None
    try:
        _candidate = Path(__file__).resolve().parents[2] / ".." / "robosuite" / "robosuite" / "models" / "assets" / "bullet_data" / "panda_description" / "urdf" / "panda_arm.urdf"
        if not _candidate.exists():
            # Try common install locations
            import importlib.util
            _spec = importlib.util.find_spec("robosuite")
            if _spec and _spec.origin:
                _candidate = Path(_spec.origin).parent / "models" / "assets" / "bullet_data" / "panda_description" / "urdf" / "panda_arm.urdf"
        if _candidate.exists():
            urdf_path = _candidate
    except Exception:
        pass

    return RobotProfile(
        name="panda_omron",
        arms={"right": right_arm},
        camera_names=("top", "right", "wrist"),
        control_freq_hz=20.0,
        policy_freq_hz=20.0,
        urdf_path=urdf_path,
        camera_obs_key_map={
            "top": "robot0_agentview_left_image",
            "right": "robot0_agentview_right_image",
            "wrist": "robot0_eye_in_hand_image",
        },
        is_mobile_base=True,
    )


def robocasa_gr1_arms_profile() -> RobotProfile:
    """GR1ArmsOnly bimanual (7+7 DOF) for RoboCasa365.

    Fixed-base humanoid with right and left 7-DOF arms.
    FK/IK handled by the RoboCasa backend.
    """
    # GR1 arm joint limits (approximate from robosuite model)
    gr1_limits_low = np.array([-3.14, -1.57, -3.14, -2.35, -3.14, -1.57, -3.14])
    gr1_limits_high = np.array([3.14, 1.57, 3.14, 0.0, 3.14, 1.57, 3.14])
    gr1_home = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])

    right_arm = ArmProfile(
        name="right",
        dof=7,
        joint_limits_low=gr1_limits_low.copy(),
        joint_limits_high=gr1_limits_high.copy(),
        home_joint_pos=gr1_home.copy(),
        home_gripper_pos=np.array([1.0]),
        gripper_min=0.0,
        gripper_max=1.0,
        interp_kp=np.full(8, 150.0),
        interp_kd=np.full(8, 1.0),
        ee_frame_name="robot0_right_hand",
        q_slice=None,
    )

    left_arm = ArmProfile(
        name="left",
        dof=7,
        joint_limits_low=gr1_limits_low.copy(),
        joint_limits_high=gr1_limits_high.copy(),
        home_joint_pos=np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0]),
        home_gripper_pos=np.array([1.0]),
        gripper_min=0.0,
        gripper_max=1.0,
        interp_kp=np.full(8, 150.0),
        interp_kd=np.full(8, 1.0),
        ee_frame_name="robot0_left_hand",
        q_slice=None,
    )

    return RobotProfile(
        name="gr1_arms_only",
        arms={"right": right_arm, "left": left_arm},
        camera_names=("top", "wrist"),
        control_freq_hz=20.0,
        policy_freq_hz=20.0,
        urdf_path=None,
        camera_obs_key_map={
            "top": "robot0_agentview_left_image",
            "wrist": "robot0_eye_in_hand_image",
        },
    )
