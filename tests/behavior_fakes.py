"""A small, kinematically consistent stand-in for OmniGibson's R1 Pro environment.

Joint order mirrors the real robot: 6 virtual base joints, 4 torso joints, 7 + 7 arm joints and
2 + 2 finger joints. Joint targets are reached instantly, the base footprint pose is the root
pose composed with the base joints, and cameras/end effectors hang off the footprint at fixed
offsets, so frame conversions can be checked exactly without Isaac Sim.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from cap_harness.geometry import matrix_to_quaternion_wxyz, quaternion_wxyz_to_matrix

JOINT_COUNT = 28
BASE_IDX = np.arange(0, 6)
TRUNK_IDX = np.arange(6, 10)
ARM_IDX = {"left": np.arange(10, 17), "right": np.arange(17, 24)}
GRIPPER_IDX = {"left": np.arange(24, 26), "right": np.arange(26, 28)}
GRIPPER_OPEN = 0.05
HEAD_OFFSET = np.array([0.1, 0.0, 1.4])
EEF_OFFSET = {"left": np.array([0.4, 0.25, 0.9]), "right": np.array([0.4, -0.25, 0.9])}
# OpenGL camera looking along the robot's +X: columns are camera X, Y, Z in the footprint frame.
CAMERA_GL_ROTATION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def planar(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    cos, sin = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    transform[:2, :2] = [[cos, -sin], [sin, cos]]
    transform[:3, 3] = (x, y, z)
    return transform


def pose_of(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wxyz = matrix_to_quaternion_wxyz(transform[:3, :3])
    return transform[:3, 3].copy(), np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


class _Link:
    def __init__(self, robot: FakeRobot, offset: np.ndarray) -> None:
        self._robot = robot
        self._offset = offset

    def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        transform = self._robot.footprint() @ planar(*self._offset[:2], 0.0, self._offset[2])
        return pose_of(transform)


class _Sensor:
    def __init__(self, robot: FakeRobot, offset: np.ndarray, size: int) -> None:
        self._robot = robot
        self._offset = offset
        self.size = size
        self.horizontal_aperture = 20.995
        self.reads = 0

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        self.reads += 1
        focal = 17.0 * self.size / 40.0
        return np.array([[focal, 0.0, self.size / 2], [0.0, focal, self.size / 2], [0.0, 0.0, 1.0]])

    def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        transform = np.eye(4)
        transform[:3, :3] = CAMERA_GL_ROTATION
        transform[:3, 3] = self._offset
        return pose_of(self._robot.footprint() @ transform)


class FakeRobot:
    def __init__(self, root_pose: tuple[float, float, float], size: int) -> None:
        self.name = "robot_r1"
        self.model = "r1pro"
        self.arm_names = ["left", "right"]
        self.arm_joint_names = {
            arm: [f"{arm}_arm_joint{i}" for i in range(1, 8)] for arm in self.arm_names
        }
        self.eef_link_names = {arm: f"{arm}_eef_link" for arm in self.arm_names}
        self.base_footprint_link_name = "base_link"
        self.base_idx = BASE_IDX.copy()
        self.trunk_control_idx = TRUNK_IDX.copy()
        self.arm_control_idx = {arm: idx.copy() for arm, idx in ARM_IDX.items()}
        self.gripper_control_idx = {arm: idx.copy() for arm, idx in GRIPPER_IDX.items()}
        self.joint_lower_limits = np.full(JOINT_COUNT, -3.0)
        self.joint_upper_limits = np.full(JOINT_COUNT, 3.0)
        for idx in GRIPPER_IDX.values():
            self.joint_lower_limits[idx] = 0.0
            self.joint_upper_limits[idx] = GRIPPER_OPEN
        self.reset_joint_pos = np.zeros(JOINT_COUNT)
        self.reset_joint_pos[TRUNK_IDX] = [1.0, -1.4, -0.5, 0.0]
        self.reset_joint_pos[GRIPPER_IDX["left"]] = GRIPPER_OPEN
        self.reset_joint_pos[GRIPPER_IDX["right"]] = GRIPPER_OPEN
        self.q = self.reset_joint_pos.copy()
        self.qd = np.zeros(JOINT_COUNT)
        self.root = planar(*root_pose)
        self.eef_links = {arm: _Link(self, offset) for arm, offset in EEF_OFFSET.items()}
        self.sensors = {
            f"{self.name}:zed_link:Camera:0": _Sensor(self, HEAD_OFFSET, size),
            f"{self.name}:left_realsense_link:Camera:0": _Sensor(self, EEF_OFFSET["left"], size),
            f"{self.name}:right_realsense_link:Camera:0": _Sensor(self, EEF_OFFSET["right"], size),
        }
        self.grasping: dict[str, Any] = {"left": None, "right": None}
        self.keep_still_calls = 0

    # kinematics -------------------------------------------------------
    def footprint(self) -> np.ndarray:
        base = self.q[BASE_IDX]
        return self.root @ planar(base[0], base[1], base[5], base[2])

    def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        return pose_of(self.footprint())

    def set_position_orientation(self, position: Any, orientation: Any) -> None:
        world = np.eye(4)
        xyzw = np.asarray(orientation, dtype=np.float64)
        world[:3, :3] = quaternion_wxyz_to_matrix(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]))
        world[:3, 3] = np.asarray(position, dtype=np.float64)
        relative = np.linalg.inv(self.root) @ world
        self.q[BASE_IDX[0]] = relative[0, 3]
        self.q[BASE_IDX[1]] = relative[1, 3]
        self.q[BASE_IDX[5]] = math.atan2(relative[1, 0], relative[0, 0])

    # joints ------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.qd.copy()

    def set_joint_positions(
        self, positions: Any, indices: Any = None, normalized: bool = False
    ) -> None:
        values = np.asarray(positions, dtype=np.float64).reshape(-1)
        if indices is None:
            self.q[:] = values
        else:
            self.q[np.asarray(indices)] = values

    def keep_still(self) -> None:
        self.keep_still_calls += 1
        self.qd[:] = 0.0

    def get_linear_velocity(self) -> np.ndarray:
        return np.zeros(3)

    def q_to_action(self, q: Any) -> np.ndarray:
        return np.asarray(q, dtype=np.float64).copy()

    def is_grasping(self, arm: str = "default", candidate_obj: Any = None) -> bool:
        held = self.grasping[arm]
        if held is None:
            return False
        return candidate_obj is None or held is candidate_obj


class FakeObject:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self.position = np.asarray(position, dtype=np.float64)

    def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), np.array([0.0, 0.0, 0.0, 1.0])


class _Task:
    def __init__(self, targets: dict[str, FakeObject]) -> None:
        self.object_scope = dict(targets)


class FakeOmniGibsonEnv:
    """``og.Environment`` stand-in accepted by ``BehaviorAdapter(env_factory=...)``."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        root_pose: tuple[float, float, float] = (5.0, 4.3, 0.3),
        target_scope: str = "radio_receiver.n.01_1",
    ) -> None:
        self.config = config
        robot_cfg = config["robots"][0]
        size = int(robot_cfg["sensor_config"]["VisionSensor"]["sensor_kwargs"]["image_width"])
        self.robots = [FakeRobot(root_pose, size)]
        self.size = size
        self.target = FakeObject((7.0, 4.5, 0.6))
        self.task = _Task({target_scope: self.target})
        self.loaded_instances: list[int] = []
        self.step_count = 0
        self.reset_count = 0
        self.closed = False
        self.depth_infinity_pixels = 5

    # lifecycle ---------------------------------------------------------
    def load_task_instance(self, instance_id: int) -> None:
        self.loaded_instances.append(int(instance_id))

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.reset_count += 1
        self.robots[0].q = self.robots[0].reset_joint_pos.copy()
        return self.get_obs(), {}

    def close(self) -> None:
        self.closed = True

    # stepping ----------------------------------------------------------
    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        robot = self.robots[0]
        target = np.asarray(action, dtype=np.float64)
        robot.qd = (target - robot.q) * 30.0
        robot.q = target.copy()
        self.step_count += 1
        return self.get_obs(), 0.0, False, False, {"done": {"success": False}}

    def get_obs(self) -> dict[str, Any]:
        robot = self.robots[0]
        frames = {}
        for name in robot.sensors:
            rgb = np.full((self.size, self.size, 4), 127, dtype=np.uint8)
            depth = np.full((self.size, self.size), 2.5, dtype=np.float32)
            depth[0, : self.depth_infinity_pixels] = np.inf
            frames[name] = {"rgb": rgb, "depth_linear": depth}
        frames["proprio"] = np.zeros(61, dtype=np.float32)
        return {robot.name: frames}


class RecordingObserver:
    def __init__(self, max_steps: int = 10_000) -> None:
        self.resets: list[dict[str, Any]] = []
        self.steps = 0
        self.max_steps = max_steps

    def on_reset(self, observation: Any, metadata: dict[str, Any]) -> None:
        self.resets.append(dict(metadata))

    def before_step(self, action: Any) -> None:
        from cap_harness.artifacts import StepLimitReached

        if self.steps >= self.max_steps:
            raise StepLimitReached("limit")

    def after_step(self, action: Any, result: Any) -> None:
        if result.ok:
            self.steps += 1


__all__ = [
    "ARM_IDX",
    "BASE_IDX",
    "EEF_OFFSET",
    "GRIPPER_IDX",
    "GRIPPER_OPEN",
    "HEAD_OFFSET",
    "JOINT_COUNT",
    "TRUNK_IDX",
    "FakeObject",
    "FakeOmniGibsonEnv",
    "FakeRobot",
    "RecordingObserver",
    "planar",
]
