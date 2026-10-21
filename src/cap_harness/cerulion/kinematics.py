"""Kinematic constants and forward kinematics of the Cerulion simulated arm.

The plant itself runs inside the Cerulion graph (the ``arm_sim`` node,
``plant.rs`` in the Cerulion repository's ``examples/aspire_arm``) and reports
its own end-effector pose. These constants mirror ``plant.rs``; they are given
to programs as task metadata and used here only to draw the camera view.

A six-joint arm whose pitch joints move one planar chain in the vertical plane
at the base yaw: ``base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch,
wrist_roll, tool_roll``. With ``a1 = j1``, ``a2 = j1 + j2``, ``a3 = j1 + j2 + j3``::

    r = L1 cos a1 + L2 cos a2 + L3 cos a3
    z = H + L1 sin a1 + L2 sin a2 + L3 sin a3
    x = r cos j0,  y = r sin j0
"""

from __future__ import annotations

import math

import numpy as np

SHOULDER_HEIGHT_M = 0.20
UPPER_ARM_M = 0.30
FOREARM_M = 0.25
TOOL_M = 0.10
CUBE_HALF_SIZE_M = 0.025
JOINT_LIMIT_RAD = 3.0
MAX_JOINT_STEP_RAD = 0.05
MAX_GRIPPER_STEP = 0.25
CONTROL_FREQUENCY_HZ = 20.0
CONTROL_PERIOD_S = 1.0 / CONTROL_FREQUENCY_HZ
SUCCESS_LIFT_M = 0.15
JOINT_NAMES = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_roll",
    "tool_roll",
)
HOME_JOINTS = (0.0, 0.9, -1.4, -1.07, 0.0, 0.0)


def kinematics_metadata() -> dict[str, float]:
    return {
        "H": SHOULDER_HEIGHT_M,
        "L1": UPPER_ARM_M,
        "L2": FOREARM_M,
        "L3": TOOL_M,
    }


def chain_points(q: np.ndarray) -> np.ndarray:
    """Base, shoulder, elbow, wrist and tool tip, as a (5, 3) array in the base frame."""
    j0, j1, j2, j3 = (float(value) for value in np.asarray(q, dtype=np.float64)[:4])
    angles = (j1, j1 + j2, j1 + j2 + j3)
    lengths = (UPPER_ARM_M, FOREARM_M, TOOL_M)
    r, z = 0.0, SHOULDER_HEIGHT_M
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, SHOULDER_HEIGHT_M)]
    for angle, length in zip(angles, lengths, strict=True):
        r += length * math.cos(angle)
        z += length * math.sin(angle)
        points.append((r * math.cos(j0), r * math.sin(j0), z))
    return np.asarray(points, dtype=np.float64)


def end_effector_position(q: np.ndarray) -> np.ndarray:
    return chain_points(q)[-1]
