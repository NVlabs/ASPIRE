"""Typed, frame-explicit adapter for the BEHAVIOR-1K R1 Pro pickup tasks (OmniGibson on Isaac Sim).

Frame model: everything public is expressed in ``odom``, the base-footprint pose captured when the
task instance was loaded (z on the floor, yaw only). Cameras, end effectors, point clouds and
navigation goals therefore stay valid after the base drives. Internally the simulator works in the
world frame and the planner in the robot root frame; the adapter owns both conversions.

Actuation model: one full joint-position target (28 values on R1 Pro) is rewritten slice by slice
(arms from ``RobotAction``, fingers from normalized gripper positions, base and torso from the
``behavior.*`` extensions) and sent every tick through ``robot.q_to_action``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import os
import sys
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.behavior.codec import BehaviorActionCodec, R1ProJointLayout
from cap_harness.behavior.config import HEAD_HORIZONTAL_APERTURE, build_environment_config
from cap_harness.behavior.registry import (
    BEHAVIOR_ARMS,
    BehaviorTaskMetadata,
    BehaviorTaskRegistry,
)
from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    PlanningScene,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import invert_transform, matrix_to_pose

BASE_FRAME = "odom"
CONTROL_FREQUENCY_HZ = 30.0
CAMERA_LINKS: Mapping[str, str] = MappingProxyType(
    {"head": "zed_link", "left_wrist": "left_realsense_link", "right_wrist": "right_realsense_link"}
)
TRAJECTORY_TOLERANCE_RAD = 0.01
TRUNK_TOLERANCE_RAD = 0.05
"""The torso settles more slowly than the arm and carries more of the robot's mass, so holding it
to the arm's tolerance failed motions whose arm was already on target. Two evaluation seeds recorded
stalls with the arm inside tolerance and the torso 0.021 to 0.085 rad outside it.
"""
TRAJECTORY_START_TOLERANCE_RAD = 0.02
TRAJECTORY_MAX_STEPS_PER_WAYPOINT = 120
GRIPPER_STEPS = 40
SETTLE_MIN_STEPS = 50
SETTLE_MAX_STEPS = 500
SETTLE_VELOCITY_M_S = 0.01
INSTANCE_SETTLE_PHYSICS_STEPS = 25
BASE_STEPS_PER_WAYPOINT = 10
FOOTPRINT_MARGIN_M = 0.2
"""Added to half the base footprint's diagonal when testing a base goal against the scene's
traversability map (OmniGibson erodes its map by the same radius for path planning)."""
BASE_HOP_M = 1.0
"""Spacing of the intermediate base goals taken from the scene's traversability map when cuRobo's
direct base plan fails. OmniGibson's cuRobo wrapper plans the base with trajectory optimisation
only (no graph search), so a goal across the room behind furniture is unreachable in one plan
while a chain of short hops along the map's shortest path is not.
"""
BASE_SERVO_STEP_M = 0.02
BASE_SERVO_STEP_RAD = 0.05
"""Per-tick cap on how far the servo path may move its base target. The base virtual joints are
position controlled, so commanding a goal a metre away drives them at roughly a metre in twenty
ticks, which sweeps furniture aside: measured across five evaluation seeds, a leg at that rate
displaced the target object by 16 to 51 mm before the arm moved. Stepping the target keeps the
commanded motion near the planner's own 0.005 to 0.010 m per tick.
"""
BASE_FALL_LIMIT_M = 0.5
"""A base whose height drops this far below its reset height has left the floor. The scene's
traversability map is a square raster and its margin cells need not have floor geometry beneath
them, so a route can send the base somewhere it falls: measured on one evaluation seed at -34, -45
and -158 m while the planar pose kept reporting sane x and y.
"""
HOP_TOLERANCE_M = 0.10
HOP_TOLERANCE_RAD = 0.25
SERVO_HOP_MAX_STEPS = 400
BASE_WAYPOINT_TOLERANCE_M = 0.005
BASE_WAYPOINT_TOLERANCE_RAD = 0.03
ARTICULATION_WAYPOINT_TOLERANCE_RAD = 0.005
INTRINSICS_RENDER_RETRIES = 40
GL_TO_CV = np.diag((1.0, -1.0, -1.0, 1.0))
"""OmniGibson cameras look down -Z with +Y up (OpenGL); our contracts use +Z forward, +Y down."""


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _as_tensor(value: np.ndarray) -> Any:
    """A float32 torch tensor when torch is importable (OmniGibson's maps expect one)."""
    try:
        import torch
    except ImportError:
        return np.asarray(value, dtype=np.float32)
    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def _pose_matrix(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    from cap_harness.geometry import quaternion_wxyz_to_matrix

    xyzw = _to_numpy(quaternion_xyzw).astype(np.float64).reshape(4)
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(wxyz)
    transform[:3, 3] = _to_numpy(position).astype(np.float64).reshape(3)
    return transform


def _planar_transform(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    cos, sin = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = np.array([[cos, -sin], [sin, cos]])
    transform[:3, 3] = (x, y, z)
    return transform


def _yaw_of(transform: np.ndarray) -> float:
    return float(math.atan2(transform[1, 0], transform[0, 0]))


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _planar_pose_of(transform: np.ndarray) -> tuple[float, float, float]:
    return float(transform[0, 3]), float(transform[1, 3]), _yaw_of(transform)


@dataclass
class _BaseDrive:
    """Outcome of one base motion (a planned path or a servo) inside ``navigate_to_pose``."""

    ok: bool
    steps: int = 0
    final: Observation | None = None
    error: ApiError | None = None
    terminated: bool = False
    truncated: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


def erode_free_cells(free: np.ndarray, half_window: int) -> np.ndarray:
    """Cells whose square neighbourhood (``half_window`` cells each way) is entirely free."""
    free = np.asarray(free, dtype=bool)
    if half_window <= 0:
        return free.copy()
    padded = np.pad(~free, half_window, mode="constant", constant_values=True).astype(np.int64)
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * half_window + 1
    rows, columns = free.shape
    window = (
        integral[size : size + rows, size : size + columns]
        - integral[:rows, size : size + columns]
        - integral[size : size + rows, :columns]
        + integral[:rows, :columns]
    )
    return window == 0


def resample_route(
    points: Sequence[Sequence[float]], goal: Sequence[float], hop_m: float = BASE_HOP_M
) -> list[tuple[float, float, float]]:
    """Intermediate ``(x, y, yaw)`` goals every ``hop_m`` along a polyline that ends at ``goal``.

    Intermediate hops face the next hop so the head camera looks along the route; the last hop
    is the goal itself with the goal's yaw.
    """
    polyline = [np.asarray(point, dtype=np.float64)[:2] for point in points]
    polyline.append(np.asarray(goal, dtype=np.float64)[:2])
    deduped = [polyline[0]]
    for point in polyline[1:]:
        if np.linalg.norm(point - deduped[-1]) > 1e-6:
            deduped.append(point)
    hops: list[np.ndarray] = []
    if len(deduped) > 1:
        array = np.stack(deduped)
        cumulative = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1))]
        )
        for distance in np.arange(hop_m, cumulative[-1] - hop_m / 2.0 + 1e-9, hop_m):
            hops.append(
                np.array(
                    [
                        np.interp(distance, cumulative, array[:, 0]),
                        np.interp(distance, cumulative, array[:, 1]),
                    ]
                )
            )
    hops.append(deduped[-1])
    route: list[tuple[float, float, float]] = []
    for index, point in enumerate(hops):
        if index == len(hops) - 1:
            yaw = float(goal[2])
        else:
            ahead = hops[index + 1]
            yaw = math.atan2(ahead[1] - point[1], ahead[0] - point[0])
        route.append((float(point[0]), float(point[1]), float(yaw)))
    return route


def _closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    length = float(np.dot(segment, segment))
    if length <= 1e-12:
        return a.copy()
    t = float(np.clip(np.dot(point - a, segment) / length, 0.0, 1.0))
    return a + t * segment


def _hull_vertices(points_xy: np.ndarray) -> np.ndarray:
    """Convex hull vertices in order; falls back to the axis-aligned box for degenerate input."""
    unique = np.unique(points_xy, axis=0)
    if len(unique) >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(unique)
            return unique[hull.vertices]
        except Exception:  # pragma: no cover - collinear or numerically degenerate clouds
            pass
    lower, upper = unique.min(axis=0), unique.max(axis=0)
    return np.array(
        [[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]
    )


def plan_standoff_pose(
    support_cloud: Any, object_cloud: Any, standoff_m: float = 0.3
) -> tuple[float, float, float]:
    """Pure geometry: a base pose ``standoff_m`` outside the support surface's nearest edge.

    The support cloud's convex hull in XY gives the edges; the edge closest to the object's
    median XY is chosen; the outward normal is the one pointing away from the hull centroid;
    yaw faces the object. Both clouds must share one frame, which is the frame of the answer.
    """
    standoff = float(standoff_m)
    if not np.isfinite(standoff) or standoff <= 0.0:
        raise ValueError("standoff_m must be positive")
    support = np.asarray(getattr(support_cloud, "points", support_cloud), dtype=np.float64)
    target = np.asarray(getattr(object_cloud, "points", object_cloud), dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 3 or len(support) == 0:
        raise ValueError("support_cloud must contain (N, 3) points")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
        raise ValueError("object_cloud must contain (N, 3) points")
    support_frame = getattr(support_cloud, "frame", None)
    object_frame = getattr(object_cloud, "frame", None)
    if support_frame is not None and object_frame is not None and support_frame != object_frame:
        raise ValueError("support and object clouds must share a frame")
    object_xy = np.median(target[:, :2], axis=0)
    vertices = _hull_vertices(support[:, :2])
    centroid = vertices.mean(axis=0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for index in range(len(vertices)):
        a, b = vertices[index], vertices[(index + 1) % len(vertices)]
        closest = _closest_point_on_segment(object_xy, a, b)
        distance = float(np.linalg.norm(object_xy - closest))
        if best is None or distance < best[0]:
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            norm = float(np.linalg.norm(normal))
            normal = normal / norm if norm > 1e-9 else np.array([1.0, 0.0])
            if np.dot(normal, closest - centroid) < 0.0:
                normal = -normal
            best = (distance, closest, normal)
    assert best is not None
    _, closest, normal = best
    standoff_xy = closest + normal * standoff
    yaw = math.atan2(object_xy[1] - standoff_xy[1], object_xy[0] - standoff_xy[0])
    return float(standoff_xy[0]), float(standoff_xy[1]), float(yaw)


def plan_approach_pose(
    object_cloud: Any,
    base_pose: tuple[float, float, float],
    distance_m: float = 0.7,
) -> tuple[float, float, float]:
    """Pure geometry: a base pose ``distance_m`` short of the object along the line from the base.

    Useful for free-standing objects (a can on the floor) where there is no support surface to
    stand off from. Yaw faces the object. The answer is in the cloud's frame.
    """
    distance = float(distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_m must be non-negative")
    points = np.asarray(getattr(object_cloud, "points", object_cloud), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("object_cloud must contain (N, 3) points")
    object_xy = np.median(points[:, :2], axis=0)
    base_xy = np.asarray(base_pose[:2], dtype=np.float64)
    direction = object_xy - base_xy
    span = float(np.linalg.norm(direction))
    if span <= 1e-9:
        return float(base_xy[0]), float(base_xy[1]), float(base_pose[2])
    unit = direction / span
    goal = object_xy - unit * min(distance, span)
    yaw = math.atan2(unit[1], unit[0])
    return float(goal[0]), float(goal[1]), float(yaw)


class BehaviorAdapter:
    """Normalize OmniGibson's R1 Pro without exposing native state to generated code."""

    BASE_FRAME = BASE_FRAME
    ARMS = BEHAVIOR_ARMS

    def __init__(
        self,
        *,
        registry: BehaviorTaskRegistry | None = None,
        run_observer: Any | None = None,
        camera_width: int = 512,
        camera_height: int = 512,
        horizon: int = 6000,
        gpu_id: int | None = None,
        data_root: str | os.PathLike[str] | None = None,
        env_factory: Callable[[dict[str, object]], Any] | None = None,
        head_horizontal_aperture: float = HEAD_HORIZONTAL_APERTURE,
    ) -> None:
        if camera_height <= 0 or camera_width <= 0 or horizon <= 0:
            raise ValueError("camera dimensions and horizon must be positive")
        self.registry = registry or BehaviorTaskRegistry()
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.horizon = int(horizon)
        self.control_frequency = CONTROL_FREQUENCY_HZ
        self.control_period_s = 1.0 / self.control_frequency
        self.controller = "JOINT_POSITION"
        self.head_horizontal_aperture = float(head_horizontal_aperture)
        self._gpu_id = gpu_id
        self._data_root = os.fspath(data_root) if data_root is not None else None
        self._env_factory = env_factory
        self._run_observer = run_observer
        self._protocol_evaluator: Any | None = None
        self._og: Any | None = None
        self._env: Any | None = None
        self._env_metadata: BehaviorTaskMetadata | None = None
        self._layout: R1ProJointLayout | None = None
        self._codec: BehaviorActionCodec | None = None
        self._metadata: BehaviorTaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._raw_observation: Mapping[str, Any] | None = None
        self._intrinsics: dict[str, np.ndarray] = {}
        self._sensor_names: dict[str, str] = {}
        self._q_target: np.ndarray | None = None
        self._reset_trunk: np.ndarray | None = None
        self._world_from_odom = np.eye(4)
        self._odom_from_world = np.eye(4)
        self._free_cells: tuple[Any, np.ndarray] | None = None
        self._world_from_root = np.eye(4)
        self._root_from_world = np.eye(4)
        self._elapsed_steps = 0
        self._seed: int | None = None
        self._last_reward: float | None = None
        self._last_terminated = False
        self._last_truncated = False
        self._success_latched = False
        self._success_observed_step: int | None = None
        self._closed = False
        self._planner: Any | None = None

    # ------------------------------------------------------------------ identity and hooks
    @property
    def embodiment(self) -> str:
        return "behavior"

    @property
    def arms(self) -> tuple[str, ...]:
        return self.ARMS

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    @property
    def metadata(self) -> BehaviorTaskMetadata:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before task metadata is available")
        return self._metadata

    @property
    def native_env(self) -> Any:
        """Runtime-only native environment; never registered with generated programs."""
        return self._require_env()

    @property
    def success_observed_step(self) -> int | None:
        return self._success_observed_step

    @property
    def episode_seed(self) -> int | None:
        return self._seed

    def resolve_arm(self, arm: str) -> str:
        if arm in self.ARMS:
            return arm
        if arm == "left":
            return "primary"
        if arm == "right":
            return "secondary"
        return arm

    def bind_protocol_evaluator(self, evaluator: Any) -> None:
        """Attach one host-only evaluator after reset; never exposed to programs."""
        if self._env is None:
            raise RuntimeError("reset() must be called before binding an evaluator")
        if self._protocol_evaluator is not None:
            raise RuntimeError("a protocol evaluator is already bound")
        if not callable(getattr(evaluator, "after_step", None)):
            raise ValueError("protocol evaluator must define after_step()")
        self._protocol_evaluator = evaluator

    def planner(self) -> Any:
        """The in-process cuRobo planner bound to this adapter's live robot."""
        if self._planner is None:
            from cap_harness.behavior.planning import OmniGibsonCuroboPlanner

            self._planner = OmniGibsonCuroboPlanner(self)
        return self._planner

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("reset() must be called before using the BEHAVIOR adapter")
        return self._env

    def _require_codec(self) -> BehaviorActionCodec:
        if self._codec is None or self._layout is None:
            raise RuntimeError("reset() must be called before the joint layout is known")
        return self._codec

    @property
    def layout(self) -> R1ProJointLayout:
        if self._layout is None:
            raise RuntimeError("reset() must be called before the joint layout is known")
        return self._layout

    @property
    def robot(self) -> Any:
        return self._require_env().robots[0]

    # ------------------------------------------------------------------ frames
    @property
    def world_from_odom(self) -> np.ndarray:
        return self._world_from_odom.copy()

    @property
    def odom_from_world(self) -> np.ndarray:
        return self._odom_from_world.copy()

    @property
    def world_from_root(self) -> np.ndarray:
        return self._world_from_root.copy()

    @property
    def root_from_world(self) -> np.ndarray:
        return self._root_from_world.copy()

    def _capture_frames(self) -> None:
        robot = self.robot
        footprint = _pose_matrix(*robot.get_position_orientation())
        x, y, yaw = _planar_pose_of(footprint)
        self._world_from_odom = _planar_transform(x, y, yaw)
        self._free_cells = None
        self._odom_from_world = invert_transform(self._world_from_odom)
        # The base's virtual joints measure the footprint relative to the articulation root, so
        # the root pose is world_from_footprint composed with the inverse of that offset.
        q = _to_numpy(robot.get_joint_positions()).astype(np.float64)
        base = q[self.layout.base]
        root_from_footprint = _planar_transform(float(base[0]), float(base[1]), float(base[5]))
        root_from_footprint[2, 3] = float(base[2])
        self._world_from_root = footprint @ invert_transform(root_from_footprint)
        self._root_from_world = invert_transform(self._world_from_root)

    def base_pose_odom(self) -> tuple[float, float, float]:
        footprint = _pose_matrix(*self.robot.get_position_orientation())
        return _planar_pose_of(self._odom_from_world @ footprint)

    def root_from_odom_planar(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        transform = self._root_from_world @ self._world_from_odom @ _planar_transform(x, y, yaw)
        return _planar_pose_of(transform)

    # ------------------------------------------------------------------ environment lifecycle
    def _configure_process(self) -> None:
        os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
        if self._gpu_id is not None:
            os.environ.setdefault("OMNIGIBSON_GPU_ID", str(self._gpu_id))
        if self._data_root is not None:
            os.environ.setdefault("OMNIGIBSON_DATA_PATH", self._data_root)
        if "OMNIGIBSON_APPDATA_PATH" not in os.environ:
            gpu = os.environ.get("OMNIGIBSON_GPU_ID", "0")
            os.environ["OMNIGIBSON_APPDATA_PATH"] = os.path.join(
                os.path.expanduser("~"), ".cache", f"cap-harness-behavior-appdata-gpu{gpu}"
            )
        os.makedirs(os.environ["OMNIGIBSON_APPDATA_PATH"], exist_ok=True)

    def _create_native_environment(self, config: dict[str, object]) -> Any:
        self._configure_process()
        saved_argv = list(sys.argv)
        sys.argv = saved_argv[:1]  # Isaac Sim parses sys.argv and rejects unknown flags.
        try:
            import omnigibson as og

            self._og = og
            return og.Environment(configs=config)
        except Exception:
            sys.stdout.flush()
            sys.stderr.flush()
            raise
        finally:
            sys.argv = saved_argv

    def _ensure_environment(self, metadata: BehaviorTaskMetadata) -> None:
        if (
            self._env is not None
            and self._env_metadata is not None
            and self._env_metadata.activity_name == metadata.activity_name
            and self._env_metadata.scene_model == metadata.scene_model
        ):
            return
        self._clear_environment()
        config = build_environment_config(
            metadata,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            horizon=self.horizon,
        )
        env = (
            self._env_factory(config)
            if self._env_factory is not None
            else self._create_native_environment(config)
        )
        self._env = env
        self._env_metadata = metadata
        robot = env.robots[0]
        self._layout = self._read_layout(robot)
        self._codec = BehaviorActionCodec(self._layout)
        self._sensor_names = self._resolve_sensor_names(robot)
        head = robot.sensors.get(self._sensor_names["head"])
        if head is not None and hasattr(head, "horizontal_aperture"):
            head.horizontal_aperture = self.head_horizontal_aperture
        self._planner = None

    @staticmethod
    def _read_layout(robot: Any) -> R1ProJointLayout:
        q = _to_numpy(robot.get_joint_positions())
        lower = _to_numpy(robot.joint_lower_limits).astype(np.float64)
        upper = _to_numpy(robot.joint_upper_limits).astype(np.float64)
        grippers = {arm: _to_numpy(robot.gripper_control_idx[arm]) for arm in robot.arm_names}
        return R1ProJointLayout(
            joint_count=int(q.size),
            base=_to_numpy(robot.base_idx),
            trunk=_to_numpy(robot.trunk_control_idx),
            arms={arm: _to_numpy(robot.arm_control_idx[arm]) for arm in robot.arm_names},
            grippers=grippers,
            gripper_open={arm: upper[idx] for arm, idx in grippers.items()},
            gripper_closed={arm: lower[idx] for arm, idx in grippers.items()},
        )

    @staticmethod
    def _resolve_sensor_names(robot: Any) -> dict[str, str]:
        names: dict[str, str] = {}
        for public, link in CAMERA_LINKS.items():
            matches = [name for name in robot.sensors if f":{link}:" in name]
            if not matches:
                raise RuntimeError(f"robot has no camera on link {link!r} for {public!r}")
            names[public] = sorted(matches)[0]
        return names

    def _clear_environment(self) -> None:
        env, self._env = self._env, None
        self._env_metadata = None
        self._planner = None
        self._raw_observation = None
        self._intrinsics = {}
        if env is None:
            return
        if self._env_factory is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            return
        if self._og is not None:
            self._og.clear()

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        task_ref: str | tuple[str, int] | BehaviorTaskMetadata,
        seed: int,
    ) -> Observation:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        metadata = self.registry.resolve(task_ref)
        data_root = self._data_root or os.environ.get("OMNIGIBSON_DATA_PATH")
        if data_root is not None:
            ids = self.registry.instance_ids(metadata, data_root)
            if ids and seed not in ids:
                raise ValueError(
                    f"seed {seed} is not a {metadata.task_name} task instance; available: {ids}"
                )
        self._ensure_environment(metadata)
        env = self._require_env()
        self._closed = False
        self._metadata = metadata
        self._seed = seed
        self._protocol_evaluator = None
        self._success_latched = False
        self._success_observed_step = None
        np.random.seed(seed)
        env.reset()
        self._load_task_instance(metadata, seed)
        observation, _ = env.reset()
        self._raw_observation = observation
        self._q_target = _to_numpy(self.robot.get_joint_positions()).astype(np.float64).copy()
        self._reset_trunk = _to_numpy(self.robot.reset_joint_pos).astype(np.float64)[
            self.layout.trunk
        ]
        self._open_grippers()
        self._raw_step_settle(SETTLE_MIN_STEPS, SETTLE_MAX_STEPS)
        self._capture_frames()
        self._refresh_intrinsics()
        self._elapsed_steps = 0
        self._last_reward = None
        self._last_terminated = False
        self._last_truncated = False
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={
                "task_ref": metadata.task_ref,
                "activity_name": metadata.activity_name,
                "instance_id": seed,
            },
        )
        observation = self.get_observation()
        if self._run_observer is not None:
            record = dict(metadata.to_manifest_record())
            record["instance_id"] = seed
            self._run_observer.on_reset(observation, record)
        return observation

    def _load_task_instance(self, metadata: BehaviorTaskMetadata, instance_id: int) -> None:
        env = self._require_env()
        loader = getattr(env, "load_task_instance", None)
        if callable(loader):  # test doubles and future upstream helpers
            loader(instance_id)
            return
        import json

        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.bddl_utils import is_system_bddl_inst
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        og = self._og
        task = env.task
        scene_model = task.scene_name
        tro_filename = task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=task.activity_name,
            activity_definition_id=task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        path = get_task_instance_path(
            scene_model,
            f"{scene_model}_task_{task.activity_name}_instances/{tro_filename}-tro_state",
            mode=metadata.instance_mode,
        )
        if path is None:
            raise FileNotFoundError(
                f"no {metadata.instance_mode} task instance {instance_id} for "
                f"{task.activity_name} in scene {scene_model}"
            )
        with open(path, encoding="utf-8") as handle:
            tro_state = recursively_convert_to_torch(json.load(handle))
        robot = self.robot
        for key, state in tro_state.items():
            if key == "robot_poses":
                poses = {name.lower(): value for name, value in state.items()}
                available = poses.get("robot") or poses.get(robot.model)
                if available is None:
                    raise KeyError(f"no presampled robot pose for {robot.model!r}")
                robot.set_position_orientation(
                    available[0]["position"], available[0]["orientation"]
                )
                env.scene.write_task_metadata(key=key, data=state)
            else:
                task.object_scope[key].load_state(state, serialized=False)
        og.sim.update_handles()
        for _ in range(INSTANCE_SETTLE_PHYSICS_STEPS):
            og.sim.step_physics()
            for inst, entity in task.object_scope.items():
                if not is_system_bddl_inst(inst) and entity is not None:
                    entity.keep_still()
        env.scene.update_initial_file()
        env.scene.reset()

    def _open_grippers(self) -> None:
        robot = self.robot
        codec = self._require_codec()
        for arm in self.ARMS:
            native = codec.native_arm(arm)
            assert self._q_target is not None
            self._q_target = codec.apply_gripper(self._q_target, arm, 1.0)
            robot.set_joint_positions(
                self._as_tensor(self.layout.gripper_open[native]),
                indices=self._as_tensor(self.layout.grippers[native], integer=True),
            )
        keep_still = getattr(robot, "keep_still", None)
        if callable(keep_still):
            keep_still()

    def _as_tensor(self, values: np.ndarray, *, integer: bool = False) -> Any:
        try:
            import torch
        except ImportError:  # pragma: no cover - test doubles run without torch
            return np.asarray(values)
        dtype = torch.long if integer else torch.float32
        return torch.as_tensor(np.asarray(values), dtype=dtype)

    def _native_action(self) -> Any:
        assert self._q_target is not None
        return self.robot.q_to_action(self._as_tensor(self._q_target))

    def _raw_step(self) -> None:
        """Advance one tick holding the current target, bypassing observers (reset-time only)."""
        env = self._require_env()
        observation, reward, terminated, truncated, info = env.step(self._native_action())
        self._raw_observation = observation

    def _raw_step_settle(self, min_steps: int, max_steps: int) -> None:
        for _ in range(min_steps):
            self._raw_step()
        for _ in range(max_steps):
            velocity = _to_numpy(self.robot.get_linear_velocity()).astype(np.float64)
            if float(np.linalg.norm(velocity)) < SETTLE_VELOCITY_M_S:
                return
            self._raw_step()

    def _refresh_intrinsics(self) -> None:
        """Read each camera's intrinsics, rendering until the camera parameters are populated.

        OmniGibson enables the camera-parameter annotator lazily; on the first reads after a
        reset a sensor can report all zeros (older versions) or assert "degenerate" (v3.9.2)
        until a few more frames have rendered. Either way: render, then read again.
        """
        robot = self.robot
        for public, sensor_name in self._sensor_names.items():
            sensor = robot.sensors[sensor_name]
            matrix = np.zeros((3, 3))
            last_error: Exception | None = None
            for _ in range(INTRINSICS_RENDER_RETRIES):
                try:
                    matrix = _to_numpy(sensor.intrinsic_matrix).astype(np.float64).reshape(3, 3)
                except AssertionError as exc:
                    last_error = exc
                    matrix = np.zeros((3, 3))
                if matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0:
                    break
                if self._og is not None:
                    self._og.sim.render()
            if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
                raise RuntimeError(
                    f"camera {public!r} reported degenerate intrinsics after "
                    f"{INTRINSICS_RENDER_RETRIES} renders: {last_error}"
                )
            self._intrinsics[public] = matrix

    # ------------------------------------------------------------------ observations
    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before the task context is available")
        return self._task_context

    def get_robot_state(self) -> RobotState:
        robot = self.robot
        codec = self._require_codec()
        q = _to_numpy(robot.get_joint_positions()).astype(np.float64)
        qd = _to_numpy(robot.get_joint_velocities()).astype(np.float64)
        positions: dict[str, np.ndarray] = {}
        velocities: dict[str, np.ndarray] = {}
        poses: dict[str, Pose] = {}
        grippers: dict[str, float] = {}
        names: dict[str, tuple[str, ...]] = {}
        for arm in self.ARMS:
            native = codec.native_arm(arm)
            idx = self.layout.arms[native]
            positions[arm] = q[idx].copy()
            velocities[arm] = qd[idx].copy()
            eef = _pose_matrix(*robot.eef_links[native].get_position_orientation())
            poses[arm] = matrix_to_pose(self._odom_from_world @ eef, frame=self.BASE_FRAME)
            grippers[arm] = codec.gripper_position(q, arm)
            names[arm] = tuple(str(name) for name in robot.arm_joint_names[native])
        return RobotState(
            joint_positions=positions,
            joint_velocities=velocities,
            end_effector_poses=poses,
            gripper_positions=grippers,
            base_frame=self.BASE_FRAME,
            joint_names=names,
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def _camera_observation(self, name: str) -> CameraObservation:
        raw = self._raw_observation
        if raw is None:
            raise RuntimeError("reset() must be called before reading cameras")
        robot = self.robot
        sensor_name = self._sensor_names[name]
        frame = raw[robot.name][sensor_name]
        rgb = _to_numpy(frame["rgb"])[..., :3]
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        depth = _to_numpy(frame["depth_linear"]).astype(np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        # Isaac reports non-finite distance for rays that hit nothing; the contract needs
        # finite, non-negative metres, and a single bad frame must not poison the episode.
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
        sensor = robot.sensors[sensor_name]
        world_from_camera_gl = _pose_matrix(*sensor.get_position_orientation())
        odom_from_camera = self._odom_from_world @ world_from_camera_gl @ GL_TO_CV
        return CameraObservation(
            rgb=rgb,
            depth_m=np.ascontiguousarray(depth),
            intrinsics=self._intrinsics[name].copy(),
            frame=f"camera/{name}",
            camera_pose=matrix_to_pose(odom_from_camera, frame=self.BASE_FRAME),
            timestamp_s=self.current_time_s,
        )

    def get_observation(self) -> Observation:
        cameras = {name: self._camera_observation(name) for name in self.metadata.camera_names}
        return Observation(
            cameras=cameras,
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=self.current_time_s,
        )

    # ------------------------------------------------------------------ stepping
    def native_step(self, action: RobotAction) -> StepResult:
        env = self._require_env()
        codec = self._require_codec()
        if action.embodiment != self.embodiment:
            raise ValueError(
                f"action embodiment {action.embodiment!r} does not match {self.embodiment!r}"
            )
        unknown_arms = set(action.arms) - set(self.ARMS)
        if unknown_arms:
            return StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.NOT_FOUND, f"unknown BEHAVIOR arms: {sorted(unknown_arms)}"
                ),
            )
        for arm, command in action.arms.items():
            if command.target.shape != (7,):
                return StepResult(
                    ok=False,
                    error=ApiError(
                        ErrorCode.INVALID_REQUEST, f"command dimension does not match arm {arm!r}"
                    ),
                )
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        assert self._q_target is not None
        self._q_target = codec.apply_action(self._q_target, action)
        try:
            observation, reward, terminated, truncated, info = env.step(self._native_action())
            self._elapsed_steps += 1
            self._raw_observation = observation
            self._last_reward = float(reward) if reward is not None else None
            self._last_truncated = bool(truncated)
            result = StepResult(
                ok=True,
                observation=self.get_observation(),
                terminated=False,
                truncated=self._last_truncated,
                reward=self._last_reward,
                diagnostics={"native_info": info},
            )
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.ADAPTER_FAILED,
                    f"OmniGibson native step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        if self._protocol_evaluator is not None:
            self._protocol_evaluator.after_step(action, result)
            if (
                result.ok
                and not self._success_latched
                and bool(getattr(self._protocol_evaluator, "success", False))
            ):
                self._success_latched = True
                self._success_observed_step = self._elapsed_steps
                result = StepResult(
                    ok=True,
                    observation=result.observation,
                    terminated=True,
                    truncated=result.truncated,
                    reward=result.reward,
                    diagnostics=result.diagnostics,
                )
        self._last_terminated = bool(result.terminated)
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def step(self, action: RobotAction) -> StepResult:
        runtime = self.native_step(action)
        return StepResult(
            ok=runtime.ok,
            observation=runtime.observation,
            terminated=runtime.terminated,
            truncated=runtime.truncated,
            error=runtime.public_view().error,
            diagnostics={
                "control_period_s": self.control_period_s,
                "step_index": self._elapsed_steps,
            },
        )

    def _hold_action(self) -> RobotAction:
        """Command every arm to its current target so a tick changes only what was rewritten."""
        codec = self._require_codec()
        assert self._q_target is not None
        return RobotAction(
            {
                arm: ArmCommand(
                    "joint_position",
                    codec.arm_joints(self._q_target, arm),
                    embodiment=self.embodiment,
                )
                for arm in self.ARMS
            }
        )

    def check_success(self) -> bool:
        if self._success_latched:
            return True
        evaluator = self._protocol_evaluator
        return bool(evaluator is not None and getattr(evaluator, "success", False))

    def compute_reward(self) -> float | None:
        return self._last_reward

    # ------------------------------------------------------------------ trajectories
    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        if not isinstance(trajectory, Trajectory | SynchronizedTrajectory):
            raise ValueError("trajectory must be a Trajectory or SynchronizedTrajectory")
        if trajectory.embodiment != self.embodiment:
            raise ValueError(
                f"trajectory embodiment {trajectory.embodiment!r} "
                f"does not match {self.embodiment!r}"
            )
        if not np.isclose(trajectory.dt_s, self.control_period_s, atol=1e-9, rtol=0.0):
            raise ValueError("trajectory dt_s must equal the BEHAVIOR control period (1/30 s)")
        state = self.get_robot_state()
        if isinstance(trajectory, Trajectory):
            positions = {trajectory.arm: trajectory.joint_positions}
            names = {trajectory.arm: trajectory.joint_names}
            grippers = (
                None
                if trajectory.gripper_positions is None
                else {trajectory.arm: trajectory.gripper_positions}
            )
            count = len(trajectory.joint_positions)
        else:
            positions = trajectory.joint_positions
            names = trajectory.joint_names
            grippers = trajectory.gripper_positions
            count = trajectory.waypoint_count
        if not set(positions).issubset(self.ARMS):
            raise ValueError("trajectory contains an unavailable arm")
        if isinstance(trajectory, SynchronizedTrajectory) and set(positions) != set(self.ARMS):
            raise ValueError("synchronized trajectory must name every BEHAVIOR arm")
        stale = self._stale_trajectory_result(trajectory, state)
        if stale is not None:
            return stale
        for arm, trajectory_names in names.items():
            if tuple(trajectory_names) != tuple(state.joint_names[arm]):
                raise ValueError(f"trajectory joint_names do not match arm {arm!r}")
        planned_trunk = self._planned_trunk_targets(trajectory, count)
        codec = self._require_codec()
        trunk = self.layout.trunk
        final: Observation | None = None
        steps = 0

        def trunk_error() -> float:
            if planned_trunk is None:
                return 0.0
            q = _to_numpy(self.robot.get_joint_positions()).astype(np.float64)
            return float(np.max(np.abs(q[trunk] - planned_trunk[-1])))

        def commands(waypoint: int) -> dict[str, ArmCommand]:
            return {
                arm: ArmCommand(
                    mode="joint_position",
                    target=values[waypoint],
                    gripper_position=(None if grippers is None else float(grippers[arm][waypoint])),
                    embodiment=self.embodiment,
                )
                for arm, values in positions.items()
            }

        def joint_errors(observation: Observation) -> dict[str, float]:
            return {
                arm: float(
                    np.max(np.abs(observation.robot_state.joint_positions[arm] - values[-1]))
                )
                for arm, values in positions.items()
            }

        def terminal(result: StepResult, errors: dict[str, float]) -> ExecutionResult:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                terminated=result.terminated,
                truncated=result.truncated,
                error=self._terminal_error(result.terminated, result.truncated),
                final_errors=errors,
                diagnostics={"joint_error_rad": errors},
            )

        for waypoint in range(count):
            if planned_trunk is not None:
                assert self._q_target is not None
                self._q_target = codec.apply_trunk(self._q_target, planned_trunk[waypoint])
            result = self.step(RobotAction(commands(waypoint)))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False, steps_executed=steps, final_observation=final, error=result.error
                )
            steps += 1
            final = result.observation
            if result.terminated or result.truncated:
                return terminal(result, joint_errors(final))
        assert final is not None
        errors = joint_errors(final)
        for _ in range(TRAJECTORY_MAX_STEPS_PER_WAYPOINT - 1):
            if (
                max(errors.values()) <= TRAJECTORY_TOLERANCE_RAD
                and trunk_error() <= TRUNK_TOLERANCE_RAD
            ):
                break
            result = self.step(RobotAction(commands(count - 1)))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False, steps_executed=steps, final_observation=final, error=result.error
                )
            steps += 1
            final = result.observation
            errors = joint_errors(final)
            if result.terminated or result.truncated:
                return terminal(result, errors)
        torso_error = trunk_error()
        if max(errors.values()) > TRAJECTORY_TOLERANCE_RAD or torso_error > TRUNK_TOLERANCE_RAD:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                error=ApiError(
                    ErrorCode.TIMEOUT,
                    "final trajectory waypoint did not converge",
                    details={"joint_error_rad": errors, "torso_error_rad": torso_error},
                ),
                final_errors=errors,
            )
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            final_errors=errors,
            diagnostics={
                "joint_error_rad": errors,
                "torso_planned": planned_trunk is not None,
                "torso_error_rad": torso_error,
            },
        )

    def _planned_trunk_targets(
        self, trajectory: Trajectory | SynchronizedTrajectory, count: int
    ) -> np.ndarray | None:
        """Per-waypoint torso targets when the in-process planner produced ``trajectory``."""
        planner = self._planner
        lookup = getattr(planner, "full_path_for", None)
        if lookup is None:
            return None
        full = lookup(trajectory)
        if full is None:
            return None
        full = np.asarray(full, dtype=np.float64)
        if full.ndim != 2 or len(full) != count or full.shape[1] != self.layout.joint_count:
            return None
        return full[:, self.layout.trunk]

    @staticmethod
    def _terminal_error(terminated: bool, truncated: bool) -> ApiError:
        if truncated:
            return ApiError(
                ErrorCode.TRUNCATED, "episode was truncated before the motion completed"
            )
        return ApiError(ErrorCode.TERMINATED, "episode terminated before the motion completed")

    @staticmethod
    def _stale_trajectory_result(
        trajectory: Trajectory | SynchronizedTrajectory, state: RobotState
    ) -> ExecutionResult | None:
        expected = trajectory.expected_start
        if set(expected.arms) != set(state.arms):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.STALE_PLAN, "trajectory arm set is stale"),
            )
        joint_errors: dict[str, float] = {}
        gripper_errors: dict[str, float] = {}
        for arm in state.arms:
            if tuple(expected.joint_names[arm]) != tuple(state.joint_names[arm]):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.STALE_PLAN, f"trajectory joint names are stale for arm {arm!r}"
                    ),
                )
            joint_errors[arm] = float(
                np.max(np.abs(expected.joint_positions[arm] - state.joint_positions[arm]))
            )
            gripper_errors[arm] = abs(
                expected.gripper_positions[arm] - state.gripper_positions[arm]
            )
        # The gripper is reported but never gates: cuRobo plans the arm with the fingers locked,
        # so an aperture change cannot invalidate the arm path. Gating on it made the natural
        # ordering, plan the reach then open the hand then execute, fail with zero steps.
        if max(joint_errors.values()) <= TRAJECTORY_START_TOLERANCE_RAD:
            return None
        return ExecutionResult(
            ok=False,
            steps_executed=0,
            error=ApiError(
                ErrorCode.STALE_PLAN,
                "trajectory start no longer matches measured robot state",
                details={
                    "joint_error_rad": joint_errors,
                    "gripper_error": gripper_errors,
                    "tolerance": TRAJECTORY_START_TOLERANCE_RAD,
                },
            ),
            final_errors=joint_errors,
        )

    def execute_joint_targets(
        self,
        targets: Sequence[np.ndarray],
        *,
        steps_per_waypoint: int = BASE_STEPS_PER_WAYPOINT,
        articulation_tolerance_rad: float = ARTICULATION_WAYPOINT_TOLERANCE_RAD,
        base_tolerance_m: float = BASE_WAYPOINT_TOLERANCE_M,
        base_tolerance_rad: float = BASE_WAYPOINT_TOLERANCE_RAD,
    ) -> ExecutionResult:
        """Drive the full joint vector through planner waypoints (arms, torso and base together)."""
        layout = self.layout
        articulation = np.concatenate([layout.trunk, *layout.arms.values()])
        steps = 0
        final: Observation | None = None
        last_result: StepResult | None = None
        for raw_waypoint in targets:
            waypoint = np.asarray(raw_waypoint, dtype=np.float64).reshape(-1)
            if waypoint.shape != (layout.joint_count,) or not np.all(np.isfinite(waypoint)):
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    error=ApiError(ErrorCode.INVALID_REQUEST, "waypoint has the wrong joint count"),
                )
            self._q_target = waypoint.copy()
            for _ in range(max(1, steps_per_waypoint)):
                last_result = self.step(self._hold_action())
                if not last_result.ok or last_result.observation is None:
                    return ExecutionResult(
                        ok=False,
                        steps_executed=steps,
                        final_observation=final,
                        error=last_result.error,
                    )
                steps += 1
                final = last_result.observation
                if last_result.terminated or last_result.truncated:
                    return ExecutionResult(
                        ok=False,
                        steps_executed=steps,
                        final_observation=final,
                        terminated=last_result.terminated,
                        truncated=last_result.truncated,
                        error=self._terminal_error(last_result.terminated, last_result.truncated),
                    )
                q = _to_numpy(self.robot.get_joint_positions()).astype(np.float64)
                joint_error = float(np.max(np.abs(q[articulation] - waypoint[articulation])))
                base_error = float(np.linalg.norm(q[layout.base[:2]] - waypoint[layout.base[:2]]))
                yaw_error = abs(_wrap_angle(float(q[layout.base[5]] - waypoint[layout.base[5]])))
                if (
                    joint_error <= articulation_tolerance_rad
                    and base_error <= base_tolerance_m
                    and yaw_error <= base_tolerance_rad
                ):
                    break
        return ExecutionResult(ok=True, steps_executed=steps, final_observation=final)

    # ------------------------------------------------------------------ grippers
    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        if not isinstance(positions, Mapping) or set(positions) != set(self.ARMS):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "positions must name every BEHAVIOR arm"),
            )
        codec = self._require_codec()
        normalized: dict[str, float] = {}
        for arm in self.ARMS:
            try:
                normalized[arm] = codec.validate_gripper(positions[arm])
            except (TypeError, ValueError):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.INVALID_REQUEST,
                        f"gripper position for {arm!r} must be finite and in [0, 1]",
                    ),
                )
        assert self._q_target is not None
        result = self.step(
            RobotAction(
                {
                    arm: ArmCommand(
                        "joint_position",
                        codec.arm_joints(self._q_target, arm),
                        normalized[arm],
                        embodiment=self.embodiment,
                    )
                    for arm in self.ARMS
                }
            )
        )
        terminal = result.terminated or result.truncated
        return ExecutionResult(
            ok=result.ok and not terminal,
            steps_executed=1 if result.ok else 0,
            final_observation=result.observation,
            terminated=result.terminated,
            truncated=result.truncated,
            error=result.error
            or (self._terminal_error(result.terminated, result.truncated) if terminal else None),
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        arm = self.resolve_arm(arm)
        if arm not in self.ARMS:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}"),
            )
        codec = self._require_codec()
        try:
            numeric = codec.validate_gripper(position)
        except (TypeError, ValueError):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "gripper position must be in [0, 1]"),
            )
        result: StepResult | None = None
        final: Observation | None = None
        step_index = -1
        for step_index in range(GRIPPER_STEPS):
            assert self._q_target is not None
            result = self.step(
                RobotAction(
                    {
                        arm: ArmCommand(
                            "joint_position",
                            codec.arm_joints(self._q_target, arm),
                            numeric,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok or result.observation is None:
                break
            final = result.observation
            if result.terminated or result.truncated:
                break
        assert result is not None
        terminal = result.terminated or result.truncated
        return ExecutionResult(
            ok=result.ok and not terminal,
            steps_executed=step_index + 1 if result.ok else step_index,
            final_observation=final,
            terminated=result.terminated,
            truncated=result.truncated,
            error=result.error
            or (self._terminal_error(result.terminated, result.truncated) if terminal else None),
        )

    def close_gripper(self, *, arm: str = "primary") -> ExecutionResult:
        return self.set_gripper(0.0, arm=arm)

    # ------------------------------------------------------------------ extensions (public)
    def get_base_pose(self) -> tuple[float, float, float]:
        """``(x, y, yaw)`` of the base footprint in ``odom``."""
        return self.base_pose_odom()

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        *,
        planner: str = "curobo",
        tolerance_m: float = 0.05,
        tolerance_rad: float = 0.1,
        max_steps: int = 1500,
    ) -> ExecutionResult:
        """Drive the holonomic base to an ``odom`` pose; diagnostics report the motion made."""
        goal = np.array([x, y, yaw], dtype=np.float64)
        if not np.all(np.isfinite(goal)):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "navigation goal must be finite"),
            )
        if planner not in ("curobo", "servo"):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "planner must be 'curobo' or 'servo'"),
            )
        if max_steps <= 0 or tolerance_m <= 0.0 or tolerance_rad <= 0.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST, "max_steps and tolerances must be positive"
                ),
            )
        start = self.base_pose_odom()
        if planner == "servo":
            drive = self._servo_base(goal, tolerance_m, tolerance_rad, int(max_steps))
        else:
            drive = self._drive_planned(goal, tolerance_m, tolerance_rad, int(max_steps))
        steps, final = drive.steps, drive.final
        if not drive.ok:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                terminated=drive.terminated,
                truncated=drive.truncated,
                error=drive.error,
                diagnostics={**self._motion_report(start, planner), **drive.diagnostics},
            )
        position_error, yaw_error = self._base_errors(goal)
        diagnostics = {**self._motion_report(start, planner), **drive.diagnostics}
        errors = {"position_m": position_error, "yaw_rad": yaw_error}
        if position_error > tolerance_m or yaw_error > tolerance_rad:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                error=ApiError(
                    ErrorCode.TIMEOUT,
                    "base did not reach the navigation goal within tolerance",
                    details=errors,
                ),
                final_errors=errors,
                diagnostics=diagnostics,
            )
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            final_errors=errors,
            diagnostics=diagnostics,
        )

    def _servo_base(
        self, goal: np.ndarray, tolerance_m: float, tolerance_rad: float, max_steps: int
    ) -> _BaseDrive:
        """Position-servo the base virtual joints straight to an ``odom`` goal (no planning)."""
        assert self._q_target is not None
        codec = self._require_codec()
        goal_root = np.array(
            self.root_from_odom_planar(float(goal[0]), float(goal[1]), float(goal[2])),
            dtype=np.float64,
        )
        steps = 0
        final: Observation | None = None
        for _ in range(max_steps):
            # Step the commanded base pose toward the goal instead of jumping to it, so the
            # position controller never drives the base fast enough to sweep the scene.
            current = np.array(self.root_from_odom_planar(*self.base_pose_odom()), dtype=np.float64)
            delta = goal_root - current
            delta[2] = _wrap_angle(float(delta[2]))
            distance = float(np.hypot(delta[0], delta[1]))
            if distance > BASE_SERVO_STEP_M:
                delta[:2] *= BASE_SERVO_STEP_M / distance
            delta[2] = float(np.clip(delta[2], -BASE_SERVO_STEP_RAD, BASE_SERVO_STEP_RAD))
            nxt = current + delta
            self._q_target = codec.apply_base(self._q_target, nxt[0], nxt[1], nxt[2])
            result = self.step(self._hold_action())
            if not result.ok or result.observation is None:
                return _BaseDrive(False, steps, final, result.error)
            steps += 1
            final = result.observation
            if result.terminated or result.truncated:
                return _BaseDrive(
                    False,
                    steps,
                    final,
                    self._terminal_error(result.terminated, result.truncated),
                    result.terminated,
                    result.truncated,
                )
            fallen = self._base_fall_error()
            if fallen is not None:
                return _BaseDrive(False, steps, final, fallen)
            if self._base_within(goal, tolerance_m, tolerance_rad):
                break
        return _BaseDrive(True, steps, final)

    def _base_fall_error(self) -> ApiError | None:
        """An error when the base has dropped off the floor, which the planar pose cannot show."""
        height = float(_pose_matrix(*self.robot.get_position_orientation())[2, 3])
        reference = float(self._world_from_odom[2, 3])
        if not math.isfinite(height) or height < reference - BASE_FALL_LIMIT_M:
            return ApiError(
                ErrorCode.TERMINATED,
                "the base has left the floor; the navigation goal had no ground beneath it",
                details={"base_height_m": height, "reset_height_m": reference},
            )
        return None

    def _execute_base_plan(self, waypoints: Sequence[np.ndarray], budget: int) -> _BaseDrive:
        per_waypoint = max(1, budget // max(1, len(waypoints)))
        execution = self.execute_joint_targets(
            waypoints, steps_per_waypoint=min(BASE_STEPS_PER_WAYPOINT, per_waypoint)
        )
        return _BaseDrive(
            execution.ok,
            execution.steps_executed,
            execution.final_observation,
            execution.error,
            execution.terminated,
            execution.truncated,
        )

    def _drive_planned(
        self, goal: np.ndarray, tolerance_m: float, tolerance_rad: float, max_steps: int
    ) -> _BaseDrive:
        """CuRobo base plan to the goal; when none exists, hop along the traversability map."""
        if not self.base_pose_is_free(float(goal[0]), float(goal[1])):
            return _BaseDrive(
                False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED,
                    "navigation goal is not traversable for the base footprint "
                    "(scene traversability map); choose another goal",
                    details={"route": "goal_blocked"},
                ),
                diagnostics={"route": "goal_blocked"},
            )
        xr, yr, yawr = self.root_from_odom_planar(float(goal[0]), float(goal[1]), float(goal[2]))
        waypoints, error = self.planner().plan_base(xr, yr, yawr)
        if error is None:
            drive = self._execute_base_plan(waypoints, max_steps)
            drive.diagnostics.update({"route": "direct", "hops": 1, "servo_hops": 0})
            fallen = self._base_fall_error()
            if fallen is not None:
                drive.ok, drive.error = False, fallen
            return drive
        direct_status = str(dict(error.details or {}).get("status", "unknown"))
        route = self._traversable_route(goal)
        if route is None:
            return _BaseDrive(
                False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED,
                    f"{error.message}; the scene's traversability map has no route to the goal either",
                    details={**dict(error.details or {}), "route": "none"},
                ),
                diagnostics={"route": "none", "direct_status": direct_status},
            )
        hops, geodesic_m = route
        diagnostics: dict[str, Any] = {
            "route": "traversability",
            "hops": len(hops),
            "servo_hops": 0,
            "geodesic_m": geodesic_m,
            "direct_status": direct_status,
        }
        steps = 0
        final: Observation | None = None
        for index, hop in enumerate(hops):
            last = index == len(hops) - 1
            remaining = max_steps - steps
            if remaining <= 0:
                return _BaseDrive(
                    False,
                    steps,
                    final,
                    ApiError(
                        ErrorCode.TIMEOUT, "step budget exhausted before the base reached the goal"
                    ),
                    diagnostics=diagnostics,
                )
            hop_goal = np.array(hop, dtype=np.float64)
            hx, hy, hyaw = self.root_from_odom_planar(*hop)
            waypoints, hop_error = self.planner().plan_base(hx, hy, hyaw)
            if hop_error is None:
                drive = self._execute_base_plan(waypoints, remaining)
            else:
                diagnostics["servo_hops"] += 1
                drive = self._servo_base(
                    hop_goal,
                    tolerance_m if last else HOP_TOLERANCE_M,
                    tolerance_rad if last else HOP_TOLERANCE_RAD,
                    min(remaining, SERVO_HOP_MAX_STEPS),
                )
            steps += drive.steps
            final = drive.final if drive.final is not None else final
            if not drive.ok:
                drive.steps, drive.final, drive.diagnostics = steps, final, diagnostics
                return drive
        return _BaseDrive(True, steps, final, diagnostics=diagnostics)

    def _traversable_route(
        self, goal: np.ndarray
    ) -> tuple[list[tuple[float, float, float]], float] | None:
        """Hops along the scene's shortest traversable path to ``goal`` (odom), or None."""
        robot = self.robot
        scene = getattr(robot, "scene", None)
        shortest_path = getattr(scene, "get_shortest_path", None)
        if shortest_path is None:
            return None
        sx, sy, _ = self.base_pose_odom()
        start_world = (self._world_from_odom @ np.array([sx, sy, 0.0, 1.0]))[:2]
        goal_world = (self._world_from_odom @ np.array([goal[0], goal[1], 0.0, 1.0]))[:2]
        try:
            path, geodesic = shortest_path(
                self._floor_index(scene),
                _as_tensor(start_world),
                _as_tensor(goal_world),
                entire_path=True,
                robot=robot,
            )
        except Exception:
            return None
        if path is None:
            return None
        world_points = _to_numpy(path).astype(np.float64).reshape(-1, 2)
        odom_points = [
            (self._odom_from_world @ np.array([wx, wy, 0.0, 1.0]))[:2] for wx, wy in world_points
        ]
        hops = resample_route(odom_points, goal, BASE_HOP_M)
        return hops, float(_to_numpy(geodesic).reshape(-1)[0])

    def base_pose_is_free(self, x: float, y: float) -> bool:
        """Whether the base footprint fits at ``odom`` (x, y) per the scene's traversability map.

        True when the scene has no map (nothing known); False outside the map. A blocked goal is
        refused by ``navigate_to_pose`` before any planning, so programs can test candidate
        approach poses cheaply.
        """
        cells = self._free_footprint_cells()
        if cells is None:
            return True
        trav_map, free = cells
        world = (self._world_from_odom @ np.array([float(x), float(y), 0.0, 1.0]))[:2]
        try:
            index = _to_numpy(trav_map.world_to_map(_as_tensor(world))).astype(int).reshape(2)
        except Exception:
            return True
        row, column = int(index[0]), int(index[1])
        if not (0 <= row < free.shape[0] and 0 <= column < free.shape[1]):
            return False
        return bool(free[row, column])

    def _free_footprint_cells(self) -> tuple[Any, np.ndarray] | None:
        """The traversability map eroded by the base footprint, cached per reset."""
        if self._free_cells is not None:
            return self._free_cells
        robot = self.robot
        scene = getattr(robot, "scene", None)
        trav_map = getattr(scene, "trav_map", None)
        floor_maps = getattr(trav_map, "floor_map", None)
        if not floor_maps:
            return None
        try:
            grid = _to_numpy(floor_maps[self._floor_index(scene)]) >= 255
            extent = _to_numpy(robot.reset_joint_pos_aabb_extent).astype(np.float64).reshape(-1)
            radius_m = float(np.linalg.norm(extent[:2])) / 2.0 + FOOTPRINT_MARGIN_M
            resolution = float(trav_map.map_resolution)
        except Exception:
            return None
        half = max(1, int(math.ceil(radius_m / resolution)) // 2)
        self._free_cells = (trav_map, erode_free_cells(grid, half))
        return self._free_cells

    def _floor_index(self, scene: Any) -> int:
        heights = getattr(getattr(scene, "trav_map", None), "floor_heights", None)
        if not heights:
            return 0
        z = float(_pose_matrix(*self.robot.get_position_orientation())[2, 3])
        return int(np.argmin([abs(float(h) - z) for h in heights]))

    def _base_errors(self, goal: np.ndarray) -> tuple[float, float]:
        x, y, yaw = self.base_pose_odom()
        return (
            float(np.hypot(goal[0] - x, goal[1] - y)),
            abs(_wrap_angle(float(goal[2] - yaw))),
        )

    def _base_within(self, goal: np.ndarray, tolerance_m: float, tolerance_rad: float) -> bool:
        position_error, yaw_error = self._base_errors(goal)
        return position_error <= tolerance_m and yaw_error <= tolerance_rad

    def _motion_report(
        self, start: tuple[float, float, float], planner: str
    ) -> dict[str, float | str]:
        x, y, yaw = self.base_pose_odom()
        moved_yaw = _wrap_angle(yaw - start[2])
        return {
            "planner": planner,
            "moved_x": float(x - start[0]),
            "moved_y": float(y - start[1]),
            "moved_yaw": float(moved_yaw),
            "moved_cos": float(math.cos(moved_yaw)),
            "moved_sin": float(math.sin(moved_yaw)),
        }

    def move_torso(
        self,
        target: Sequence[float],
        *,
        tolerance: float = 0.01,
        max_steps: int = 300,
    ) -> ExecutionResult:
        """Servo the torso joints to ``target`` (radians, in the robot's torso joint order)."""
        codec = self._require_codec()
        try:
            values = np.asarray(target, dtype=np.float64).reshape(-1)
            assert self._q_target is not None
            proposed = codec.apply_trunk(self._q_target, values)
        except (TypeError, ValueError, AssertionError) as exc:
            return ExecutionResult(
                ok=False, steps_executed=0, error=ApiError(ErrorCode.INVALID_REQUEST, str(exc))
            )
        if tolerance <= 0.0 or max_steps <= 0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST, "tolerance and max_steps must be positive"
                ),
            )
        self._q_target = proposed
        steps = 0
        final: Observation | None = None
        error = float("inf")
        for _ in range(int(max_steps)):
            result = self.step(self._hold_action())
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False, steps_executed=steps, final_observation=final, error=result.error
                )
            steps += 1
            final = result.observation
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=self._terminal_error(result.terminated, result.truncated),
                )
            q = _to_numpy(self.robot.get_joint_positions()).astype(np.float64)
            error = float(np.max(np.abs(q[self.layout.trunk] - values)))
            if error <= tolerance:
                break
        errors = {"torso_rad": error}
        if error > tolerance:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                error=ApiError(ErrorCode.TIMEOUT, "torso did not converge", details=errors),
                final_errors=errors,
            )
        return ExecutionResult(
            ok=True, steps_executed=steps, final_observation=final, final_errors=errors
        )

    def reset_torso(self) -> ExecutionResult:
        """Return the torso to the pose it had when the task instance was loaded."""
        if self._reset_trunk is None:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "reset() must be called first"),
            )
        return self.move_torso(self._reset_trunk)

    @staticmethod
    def plan_standoff_pose(
        support_cloud: Any, object_cloud: Any, standoff_m: float = 0.3
    ) -> tuple[float, float, float]:
        return plan_standoff_pose(support_cloud, object_cloud, standoff_m)

    def plan_approach_pose(
        self, object_cloud: Any, distance_m: float = 0.7
    ) -> tuple[float, float, float]:
        """Base pose ``distance_m`` short of the object along the current line of sight."""
        return plan_approach_pose(object_cloud, self.base_pose_odom(), distance_m)

    # ------------------------------------------------------------------ planning hooks
    def get_planning_context(self) -> RobotPlanningContext:
        robot = self.robot
        codec = self._require_codec()
        odom_from_root = self._odom_from_world @ self._world_from_root
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model=str(robot.model),
            joint_names={
                arm: tuple(str(name) for name in robot.arm_joint_names[codec.native_arm(arm)])
                for arm in self.ARMS
            },
            base_transforms={arm: odom_from_root.copy() for arm in self.ARMS},
            end_effector_links={
                arm: str(robot.eef_link_names[codec.native_arm(arm)]) for arm in self.ARMS
            },
        )

    def ik_request(self, target_pose: Pose, state: RobotState, arm: str) -> tuple[Pose, RobotState]:
        return target_pose, state

    def planning_scene_request(self, scene: PlanningScene, arm: str) -> PlanningScene:
        return scene

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = _to_numpy(self.robot.joint_lower_limits).astype(np.float64)
        upper = _to_numpy(self.robot.joint_upper_limits).astype(np.float64)
        return lower.copy(), upper.copy()

    # ------------------------------------------------------------------ metadata (public)
    def get_task_metadata(self) -> Mapping[str, object]:
        record = dict(self.metadata.to_manifest_record())
        record["instance_id"] = self._seed
        return MappingProxyType(record)

    def get_controller_metadata(self) -> Mapping[str, object]:
        layout = self.layout
        q = _to_numpy(self.robot.get_joint_positions()).astype(np.float64)
        lower, upper = self.action_bounds
        return MappingProxyType(
            {
                "action_dimension": int(layout.joint_count),
                "action_lower_bounds": tuple(float(value) for value in lower),
                "action_upper_bounds": tuple(float(value) for value in upper),
                "arm_names": self.ARMS,
                "base_frame": self.BASE_FRAME,
                "base_pose": self.base_pose_odom(),
                "camera_names": self.metadata.camera_names,
                "control_frequency_hz": self.control_frequency,
                "control_period_s": self.control_period_s,
                "controller": self.controller,
                "controllable_gripper_arms": self.ARMS,
                "gripper_semantics": {
                    "normalized": "0=closed, 1=open",
                    "native": "finger joint positions in metres",
                },
                "supported_action_modes": ("joint_position",),
                "torso_positions": tuple(float(value) for value in q[layout.trunk]),
                "torso_lower_bounds": tuple(float(value) for value in lower[layout.trunk]),
                "torso_upper_bounds": tuple(float(value) for value in upper[layout.trunk]),
            }
        )

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        if self._closed:
            return
        self._clear_environment()
        self._closed = True


__all__ = [
    "BASE_FRAME",
    "CAMERA_LINKS",
    "BehaviorAdapter",
    "plan_approach_pose",
    "plan_standoff_pose",
]
