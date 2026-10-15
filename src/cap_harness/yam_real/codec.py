"""Convert between the YAM plant's conventions and the shared harness contracts.

Every unit, frame and ordering difference is resolved here, so the adapter stays
thin and each conversion has exactly one home. Two conventions genuinely differ
across this boundary and are converted rather than assumed:

* **Quaternion order.** The YAM side is ``xyzw`` throughout -- mink, scipy, the
  recorded episodes. The harness contracts are ``wxyz``. Getting this backwards
  produces poses that are wrong only in orientation, which is easy to miss by
  eye, so it is centralized and tested.
* **Gripper placement.** The arm server's wire format is "joint7": six arm joints
  with the gripper at index 6. The contracts keep ``gripper_position`` as its own
  field on ``ArmCommand``. The two are never concatenated here -- index 6 means
  gripper on one side of this boundary and nothing at all on the other.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    Observation,
    Pose,
    RobotState,
    TaskContext,
)

EMBODIMENT = "yam_real"
ARMS = ("left", "right")
ARM_DOF = 6
JOINT_NAMES = tuple(f"joint_{index + 1}" for index in range(ARM_DOF))


def quat_xyzw_to_wxyz(quat: Any) -> np.ndarray:
    """Convert the YAM side's xyzw quaternion to the harness's wxyz order."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(quat: Any) -> np.ndarray:
    """Convert a harness wxyz quaternion back to the YAM side's xyzw order."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to a wxyz quaternion, without a scipy dependency."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (r[2, 1] - r[1, 2]) / scale,
            (r[0, 2] - r[2, 0]) / scale,
            (r[1, 0] - r[0, 1]) / scale,
        ]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        quat = [
            (r[2, 1] - r[1, 2]) / scale,
            0.25 * scale,
            (r[0, 1] + r[1, 0]) / scale,
            (r[0, 2] + r[2, 0]) / scale,
        ]
    elif r[1, 1] > r[2, 2]:
        scale = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        quat = [
            (r[0, 2] - r[2, 0]) / scale,
            (r[0, 1] + r[1, 0]) / scale,
            0.25 * scale,
            (r[1, 2] + r[2, 1]) / scale,
        ]
    else:
        scale = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        quat = [
            (r[1, 0] - r[0, 1]) / scale,
            (r[0, 2] + r[2, 0]) / scale,
            (r[1, 2] + r[2, 1]) / scale,
            0.25 * scale,
        ]
    result = np.array(quat, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    return result / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def robot_state(env: Any, *, timestamp_s: float | None = None) -> RobotState:
    """Build a shared ``RobotState`` from both arms of a YAM plant.

    Velocities are reported as zeros when the plant does not measure them rather
    than omitted, because the contract requires the field.
    """
    positions: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}
    poses: dict[str, Pose] = {}
    grippers: dict[str, float] = {}
    names: dict[str, tuple[str, ...]] = {}

    base_frame = env.config.base_frame
    for side in ARMS:
        obs = env.get_observations(side)
        positions[side] = np.asarray(obs["joint_pos"], dtype=np.float64).reshape(ARM_DOF)
        raw_velocity = obs.get("joint_vel")
        velocities[side] = (
            np.zeros(ARM_DOF, dtype=np.float64)
            if raw_velocity is None
            else np.asarray(raw_velocity, dtype=np.float64).reshape(-1)[:ARM_DOF]
        )
        poses[side] = Pose(
            position=np.asarray(obs["ee_pos"], dtype=np.float64).reshape(3),
            quaternion_wxyz=quat_xyzw_to_wxyz(obs["ee_quat"]),
            frame=base_frame,
        )
        grippers[side] = float(np.asarray(obs["gripper_pos"], dtype=np.float64).reshape(-1)[0])
        names[side] = JOINT_NAMES

    return RobotState(
        joint_positions=positions,
        joint_velocities=velocities,
        end_effector_poses=poses,
        gripper_positions=grippers,
        base_frame=base_frame,
        joint_names=names,
        timestamp_s=timestamp_s,
        embodiment=EMBODIMENT,
    )


def camera_observation(env: Any) -> CameraObservation | None:
    """The station camera's RGB-D observation, or None when it has no frame."""
    frame = env.read_camera()
    if frame is None:
        return None
    camera = env.config.camera
    pose = camera.base_from_camera
    return CameraObservation(
        rgb=np.asarray(frame.rgb, dtype=np.uint8),
        depth_m=np.asarray(frame.depth_m, dtype=np.float64),
        intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
        # The camera's OWN frame, not the base frame. Labelling this "base" makes
        # point-cloud conversion skip the extrinsics entirely -- it returns the
        # cloud unchanged when the requested target frame already matches the
        # observation's -- so camera-frame points would be relabelled as base
        # coordinates and everything downstream would aim at the camera mount.
        frame=f"{camera.role}_optical",
        camera_pose=Pose(
            position=np.asarray(pose[:3, 3], dtype=np.float64),
            quaternion_wxyz=rotation_to_wxyz(pose[:3, :3]),
            frame=env.config.base_frame,
        ),
        timestamp_s=float(frame.timestamp_s),
    )


def aux_camera_observation(env: Any, role: str) -> CameraObservation | None:
    """An image-only camera's observation, or None when it has no frame.

    Intrinsics are real -- RealSense reports them from the device, which is
    factory data rather than calibration. What is missing is the EXTRINSIC: these
    cameras are not in the calibration bundle, so there is no transform from them
    into the base frame.

    That is expressed by giving ``camera_pose`` the camera's own optical frame as
    its parent, which is true and uninformative. It also makes the misuse
    impossible rather than merely discouraged: ``mask_to_point_cloud`` refuses a
    target frame that is not the pose's parent, so asking one of these cameras
    for world coordinates raises instead of quietly returning camera-frame points
    relabelled as base ones.
    """
    frame = env.read_aux_camera(role)
    if frame is None:
        return None
    optical = f"{role}_optical"
    return CameraObservation(
        rgb=np.asarray(frame.rgb, dtype=np.uint8),
        depth_m=np.asarray(frame.depth_m, dtype=np.float64),
        intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
        frame=optical,
        camera_pose=Pose(
            position=np.zeros(3, dtype=np.float64),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            frame=optical,
        ),
        timestamp_s=float(frame.timestamp_s),
    )


def observation(env: Any, *, task_context: TaskContext | None = None) -> Observation:
    """Assemble a full harness observation from the plant."""
    camera = camera_observation(env)
    if camera is None:
        raise RuntimeError(
            "the station camera produced no frame; the shared Observation contract "
            "requires at least one camera"
        )
    cameras = {env.config.camera.role: camera}
    # Image-only cameras join the observation so programs can query them by name,
    # but a missing frame from one is not fatal: only the calibrated camera above
    # is load-bearing.
    for role in getattr(env, "aux_camera_roles", ()):
        aux = aux_camera_observation(env, role)
        if aux is not None:
            cameras[role] = aux
    state = robot_state(env)
    return Observation(
        cameras=cameras,
        robot_state=state,
        task_context=task_context,
    )


def arm_command_targets(action: Any) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    """Split a ``RobotAction`` into per-arm joint targets and gripper targets.

    The gripper stays separate throughout; it is recombined into the arm server's
    joint7 representation only in the level-1 controller, which is the layer that
    owns that encoding.
    """
    joints: dict[str, np.ndarray] = {}
    grippers: dict[str, float | None] = {}
    for side, command in action.arms.items():
        if not isinstance(command, ArmCommand):
            # Bad data, not a caller type error: a malformed action is invalid input, not a type bug.
            raise ValueError(f"arms[{side!r}] must be an ArmCommand")
        joints[side] = np.asarray(command.target, dtype=np.float64).reshape(-1)[:ARM_DOF]
        grippers[side] = (
            None if command.gripper_position is None else float(command.gripper_position)
        )
    return joints, grippers


__all__ = [
    "ARMS",
    "ARM_DOF",
    "EMBODIMENT",
    "JOINT_NAMES",
    "arm_command_targets",
    "camera_observation",
    "observation",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "robot_state",
    "rotation_to_wxyz",
]
