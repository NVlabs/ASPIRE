"""Validated, frame-explicit dataclasses shared by CaP harness components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType

import numpy as np

from cap_harness.errors import ApiError, ErrorCode

DEFAULT_EMBODIMENT = "robosuite"
JOINT_DIMENSIONS: Mapping[str, int] = MappingProxyType(
    {
        "behavior": 7,
        "libero": 7,
        "robosuite": 7,
        "yam_real": 6,
    }
)
SUPPORTED_JOINT_DIMENSIONS = frozenset(JOINT_DIMENSIONS.values())
JOINT_DIMENSION = JOINT_DIMENSIONS[DEFAULT_EMBODIMENT]
DEFAULT_JOINT_NAMES = tuple(f"joint_{index}" for index in range(JOINT_DIMENSION))


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _joint_dimension(embodiment: object) -> int:
    normalized = _non_empty_string(embodiment, "embodiment")
    try:
        return JOINT_DIMENSIONS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(JOINT_DIMENSIONS))
        raise ValueError(
            f"unsupported embodiment {normalized!r}; expected one of: {supported}"
        ) from exc


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool | np.bool_):
        # Preserve the invalid-data ValueError contract.
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            operator = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _optional_timestamp(value: object, name: str = "timestamp_s") -> float | None:
    if value is None:
        return None
    return _finite_float(value, name, minimum=0.0)


def _array(
    value: object,
    name: str,
    *,
    shape: tuple[int | None, ...],
    dtype: np.dtype[object] | type[object] | None = np.float64,
    finite: bool = True,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if result.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(result.shape, shape, strict=False)
    ):
        expected_shape = tuple("*" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {expected_shape}, got {result.shape}")
    if finite:
        try:
            all_finite = bool(np.all(np.isfinite(result)))
        except TypeError as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not all_finite:
            raise ValueError(f"{name} must contain only finite values")
    result = np.array(result, copy=True, order="C")
    result.setflags(write=False)
    return result


def _string_tuple(value: Sequence[str], name: str, *, length: int | None = None) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        # Preserve the invalid-data ValueError contract.
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(_non_empty_string(item, f"{name} item") for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _mapping(
    value: Mapping[str, object],
    name: str,
    *,
    allow_empty: bool = True,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        # Preserve the invalid-data ValueError contract.
        raise ValueError(f"{name} must be a mapping")
    copied: dict[str, object] = {}
    for key, item in value.items():
        normalized_key = _non_empty_string(key, f"{name} key")
        if normalized_key in copied:
            raise ValueError(f"{name} contains a duplicate normalized key: {normalized_key!r}")
        copied[normalized_key] = item
    if not allow_empty and not copied:
        raise ValueError(f"{name} must not be empty")
    return MappingProxyType(copied)


def _diagnostics(value: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(value, "diagnostics")


def _validate_result_status(ok: object, error: ApiError | None) -> bool:
    result = _bool(ok, "ok")
    if result and error is not None:
        raise ValueError("a successful result cannot carry an error")
    if not result and not isinstance(error, ApiError):
        raise ValueError("a failed result must carry an ApiError")
    return result


def _normalized_gripper(value: object, name: str = "gripper_position") -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _arm_key(value: object, name: str = "arm") -> str:
    return _non_empty_string(value, name)


def _arm_vectors(
    value: Mapping[str, object] | object,
    name: str,
    *,
    joint_dimension: int,
) -> Mapping[str, np.ndarray]:
    source: Mapping[str, object]
    if isinstance(value, Mapping):
        source = value
    else:
        source = {"primary": value}
    source = _mapping(source, name, allow_empty=False)
    return MappingProxyType(
        {
            arm: _array(vector, f"{name}[{arm!r}]", shape=(joint_dimension,))
            for arm, vector in source.items()
        }
    )


def _joint_vector(value: object, name: str) -> np.ndarray:
    result = _array(value, name, shape=(None,))
    if result.shape[0] not in SUPPORTED_JOINT_DIMENSIONS:
        raise ValueError(f"{name} must contain 6 or 7 joints, got {result.shape[0]}")
    return result


def _joint_matrix(value: object, name: str) -> np.ndarray:
    result = _array(value, name, shape=(None, None))
    if result.shape[1] not in SUPPORTED_JOINT_DIMENSIONS:
        raise ValueError(f"{name} waypoints must contain 6 or 7 joints, got {result.shape[1]}")
    return result


def _arm_grippers(
    value: Mapping[str, object] | object,
    name: str,
) -> Mapping[str, float]:
    source: Mapping[str, object]
    if isinstance(value, Mapping):
        source = value
    else:
        source = {"primary": value}
    source = _mapping(source, name, allow_empty=False)
    return MappingProxyType(
        {arm: _normalized_gripper(item, f"{name}[{arm!r}]") for arm, item in source.items()}
    )


def _arm_poses(
    value: Mapping[str, Pose] | Pose,
    name: str,
) -> Mapping[str, Pose]:
    source: Mapping[str, Pose]
    if isinstance(value, Pose):
        source = {"primary": value}
    elif isinstance(value, Mapping):
        source = value
    else:
        # Preserve the invalid-data ValueError contract.
        raise ValueError(f"{name} must be a Pose or mapping of arm names to Pose")
    source = _mapping(source, name, allow_empty=False)  # type: ignore[arg-type]
    result: dict[str, Pose] = {}
    for arm, pose in source.items():
        if not isinstance(pose, Pose):
            # Preserve the invalid-data ValueError contract.
            raise ValueError(f"{name}[{arm!r}] must be a Pose")
        result[arm] = pose
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid pose expressed in ``frame`` using a wxyz unit quaternion."""

    position: np.ndarray
    quaternion_wxyz: np.ndarray
    frame: str

    def __post_init__(self) -> None:
        position = _array(self.position, "position", shape=(3,))
        quaternion = _array(self.quaternion_wxyz, "quaternion_wxyz", shape=(4,))
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError("quaternion_wxyz must be normalized to unit length")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)
        object.__setattr__(self, "frame", _non_empty_string(self.frame, "frame"))

    @property
    def position_m(self) -> np.ndarray:
        """Alias documenting that Cartesian positions use metres."""
        return self.position

    def as_matrix(self) -> np.ndarray:
        """Return a read-only 4x4 homogeneous transform in ``frame``."""
        w, x, y, z = self.quaternion_wxyz
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = self.position
        transform.setflags(write=False)
        return transform


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """Calibrated RGB-D observation from a named coordinate frame."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    frame: str
    camera_pose: Pose
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        try:
            rgb = np.asarray(self.rgb)
        except (TypeError, ValueError) as exc:
            raise ValueError("rgb must be a numeric array") from exc
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
            raise ValueError("rgb must have non-empty shape (H, W, 3)")
        if np.issubdtype(rgb.dtype, np.bool_) or not (
            np.issubdtype(rgb.dtype, np.integer) or np.issubdtype(rgb.dtype, np.floating)
        ):
            raise ValueError("rgb must use an integer or floating dtype")
        if not bool(np.all(np.isfinite(rgb))):
            raise ValueError("rgb must contain only finite values")
        rgb_min = float(np.min(rgb))
        rgb_max = float(np.max(rgb))
        if np.issubdtype(rgb.dtype, np.integer):
            if rgb_min < 0.0 or rgb_max > 255.0:
                raise ValueError("integer rgb values must be in [0, 255]")
        elif rgb_min < 0.0 or rgb_max > 1.0:
            raise ValueError("floating rgb values must be in [0, 1]")
        rgb = np.array(rgb, copy=True, order="C")
        rgb.setflags(write=False)

        depth = _array(self.depth_m, "depth_m", shape=rgb.shape[:2])
        if bool(np.any(depth < 0.0)):
            raise ValueError("depth_m values must be non-negative")
        intrinsics = _array(self.intrinsics, "intrinsics", shape=(3, 3))
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise ValueError("intrinsics focal lengths must be positive")
        if not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
            raise ValueError("intrinsics final row must be [0, 0, 1]")
        if not isinstance(self.camera_pose, Pose):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("camera_pose must be a Pose")

        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "frame", _non_empty_string(self.frame, "frame"))
        object.__setattr__(self, "timestamp_s", _optional_timestamp(self.timestamp_s))


@dataclass(frozen=True, slots=True)
class RobotState:
    """Per-arm robot state with end-effector poses expressed in ``base_frame``."""

    joint_positions: Mapping[str, np.ndarray] | np.ndarray
    joint_velocities: Mapping[str, np.ndarray] | np.ndarray
    end_effector_poses: Mapping[str, Pose] | Pose
    gripper_positions: Mapping[str, float] | float
    base_frame: str
    joint_names: Mapping[str, Sequence[str]] | Sequence[str] | None = None
    timestamp_s: float | None = None
    embodiment: str = DEFAULT_EMBODIMENT

    def __post_init__(self) -> None:
        embodiment = _non_empty_string(self.embodiment, "embodiment")
        joint_dimension = _joint_dimension(embodiment)
        positions = _arm_vectors(
            self.joint_positions,
            "joint_positions",
            joint_dimension=joint_dimension,
        )
        velocities = _arm_vectors(
            self.joint_velocities,
            "joint_velocities",
            joint_dimension=joint_dimension,
        )
        poses = _arm_poses(self.end_effector_poses, "end_effector_poses")
        grippers = _arm_grippers(self.gripper_positions, "gripper_positions")
        arms = set(positions)
        if set(velocities) != arms or set(poses) != arms or set(grippers) != arms:
            raise ValueError("all RobotState per-arm mappings must have identical arm names")
        for arm in arms:
            if velocities[arm].shape != positions[arm].shape:
                raise ValueError(
                    f"joint_velocities[{arm!r}] must match "
                    f"joint_positions[{arm!r}] shape {positions[arm].shape}"
                )

        base_frame = _non_empty_string(self.base_frame, "base_frame")
        for arm, pose in poses.items():
            if pose.frame != base_frame:
                raise ValueError(
                    f"end_effector_poses[{arm!r}] is in frame {pose.frame!r}, "
                    f"expected {base_frame!r}"
                )

        if self.joint_names is None:
            names = {
                arm: tuple(f"joint_{index}" for index in range(positions[arm].shape[0]))
                for arm in arms
            }
        elif isinstance(self.joint_names, Mapping):
            raw_names = _mapping(self.joint_names, "joint_names", allow_empty=False)
            if set(raw_names) != arms:
                raise ValueError("joint_names must have the same arm names as joint_positions")
            names = {
                arm: _string_tuple(
                    items,
                    f"joint_names[{arm!r}]",
                    length=positions[arm].shape[0],
                )
                for arm, items in raw_names.items()
            }
        else:
            dimensions = {positions[arm].shape[0] for arm in arms}
            if len(dimensions) != 1:
                raise ValueError("joint_names must be a per-arm mapping when arm dimensions differ")
            common_names = _string_tuple(
                self.joint_names,
                "joint_names",
                length=next(iter(dimensions)),
            )
            names = dict.fromkeys(arms, common_names)

        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "joint_velocities", velocities)
        object.__setattr__(self, "end_effector_poses", poses)
        object.__setattr__(self, "gripper_positions", grippers)
        object.__setattr__(self, "base_frame", base_frame)
        object.__setattr__(self, "joint_names", MappingProxyType(names))
        object.__setattr__(self, "timestamp_s", _optional_timestamp(self.timestamp_s))
        object.__setattr__(self, "embodiment", embodiment)

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.joint_positions))


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Authoritative benchmark task identity and natural-language instruction."""

    suite: str
    task_id: int
    task_name: str
    language: str
    family: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int | np.integer):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("task_id must be a non-negative integer")
        task_id = int(self.task_id)
        if task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        object.__setattr__(self, "suite", _non_empty_string(self.suite, "suite"))
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "task_name", _non_empty_string(self.task_name, "task_name"))
        object.__setattr__(self, "language", _non_empty_string(self.language, "language"))
        object.__setattr__(self, "family", _non_empty_string(self.family, "family"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class Observation:
    """Synchronized public observation assembled by an environment adapter."""

    cameras: Mapping[str, CameraObservation]
    robot_state: RobotState
    task_context: TaskContext | None = None
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        raw_cameras = _mapping(self.cameras, "cameras", allow_empty=False)
        cameras: dict[str, CameraObservation] = {}
        for name, camera in raw_cameras.items():
            if not isinstance(camera, CameraObservation):
                # Preserve the invalid-data ValueError contract.
                raise ValueError(f"cameras[{name!r}] must be a CameraObservation")
            cameras[name] = camera
        if not isinstance(self.robot_state, RobotState):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("robot_state must be a RobotState")
        if self.task_context is not None and not isinstance(self.task_context, TaskContext):
            raise ValueError("task_context must be a TaskContext or None")
        object.__setattr__(self, "cameras", MappingProxyType(cameras))
        object.__setattr__(self, "timestamp_s", _optional_timestamp(self.timestamp_s))


@dataclass(frozen=True, slots=True)
class ArmCommand:
    """One control-period command for one supported robot arm."""

    mode: str
    target: np.ndarray
    gripper_position: float | None = None
    embodiment: str = DEFAULT_EMBODIMENT

    def __post_init__(self) -> None:
        if self.mode != "joint_position":
            raise ValueError("mode must be 'joint_position'")
        embodiment = _non_empty_string(self.embodiment, "embodiment")
        joint_dimension = _joint_dimension(embodiment)
        object.__setattr__(
            self,
            "target",
            _array(self.target, "target", shape=(joint_dimension,)),
        )
        if self.gripper_position is not None:
            object.__setattr__(
                self,
                "gripper_position",
                _normalized_gripper(self.gripper_position),
            )
        object.__setattr__(self, "embodiment", embodiment)


@dataclass(frozen=True, slots=True)
class RobotAction:
    """A mapping from embodiment arm names to synchronized commands."""

    arms: Mapping[str, ArmCommand]

    def __post_init__(self) -> None:
        raw_arms = _mapping(self.arms, "arms", allow_empty=False)
        arms: dict[str, ArmCommand] = {}
        for name, command in raw_arms.items():
            if not isinstance(command, ArmCommand):
                # Preserve the invalid-data ValueError contract.
                raise ValueError(f"arms[{name!r}] must be an ArmCommand")
            arms[name] = command
        embodiments = {command.embodiment for command in arms.values()}
        if len(embodiments) != 1:
            raise ValueError("all RobotAction arm commands must use the same embodiment")
        object.__setattr__(self, "arms", MappingProxyType(arms))

    @property
    def embodiment(self) -> str:
        return next(iter(self.arms.values())).embodiment


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A fixed-rate trajectory bound to one arm's measured start state."""

    joint_positions: np.ndarray
    dt_s: float
    joint_names: Sequence[str]
    planner: str
    collision_aware: bool
    expected_start: RobotState
    arm: str = "primary"
    gripper_positions: np.ndarray | None = None
    embodiment: str = DEFAULT_EMBODIMENT

    def __post_init__(self) -> None:
        embodiment = _non_empty_string(self.embodiment, "embodiment")
        joint_dimension = _joint_dimension(embodiment)
        positions = _array(
            self.joint_positions,
            "joint_positions",
            shape=(None, joint_dimension),
        )
        if positions.shape[0] == 0:
            raise ValueError("joint_positions must contain at least one waypoint")
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(
            self,
            "dt_s",
            _finite_float(self.dt_s, "dt_s", minimum=0.0, strict_minimum=True),
        )
        object.__setattr__(
            self,
            "joint_names",
            _string_tuple(self.joint_names, "joint_names", length=positions.shape[1]),
        )
        object.__setattr__(self, "planner", _non_empty_string(self.planner, "planner"))
        object.__setattr__(self, "collision_aware", _bool(self.collision_aware, "collision_aware"))
        object.__setattr__(self, "arm", _arm_key(self.arm))
        object.__setattr__(self, "embodiment", embodiment)
        if not isinstance(self.expected_start, RobotState):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("expected_start must be a RobotState")
        if self.expected_start.embodiment != embodiment:
            raise ValueError("expected_start embodiment must match the trajectory")
        if self.arm not in self.expected_start.joint_positions:
            raise ValueError("expected_start must contain the trajectory arm")
        if self.expected_start.joint_positions[self.arm].shape[0] != positions.shape[1]:
            raise ValueError("expected_start must match the trajectory joint dimension")
        if tuple(self.expected_start.joint_names[self.arm]) != tuple(self.joint_names):
            raise ValueError("expected_start joint names must match the trajectory")
        if self.gripper_positions is not None:
            grippers = _array(
                self.gripper_positions,
                "gripper_positions",
                shape=(positions.shape[0],),
            )
            if bool(np.any((grippers < 0.0) | (grippers > 1.0))):
                raise ValueError("gripper_positions values must be in [0, 1]")
            object.__setattr__(self, "gripper_positions", grippers)

    @property
    def waypoints(self) -> np.ndarray:
        return self.joint_positions


@dataclass(frozen=True, slots=True)
class SynchronizedTrajectory:
    """Equal-length per-arm waypoints executed in the same control ticks."""

    joint_positions: Mapping[str, np.ndarray]
    dt_s: float
    joint_names: Mapping[str, Sequence[str]]
    planner: str
    collision_aware: bool
    expected_start: RobotState
    gripper_positions: Mapping[str, np.ndarray] | None = None
    embodiment: str = DEFAULT_EMBODIMENT

    def __post_init__(self) -> None:
        embodiment = _non_empty_string(self.embodiment, "embodiment")
        joint_dimension = _joint_dimension(embodiment)
        raw_positions = _mapping(self.joint_positions, "joint_positions", allow_empty=False)
        positions = {
            arm: _array(
                value,
                f"joint_positions[{arm!r}]",
                shape=(None, joint_dimension),
            )
            for arm, value in raw_positions.items()
        }
        lengths = {value.shape[0] for value in positions.values()}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("all synchronized arms must have the same nonzero waypoint count")
        raw_names = _mapping(self.joint_names, "joint_names", allow_empty=False)
        if set(raw_names) != set(positions):
            raise ValueError("joint_names must have the same arms as joint_positions")
        names = {
            arm: _string_tuple(
                value,
                f"joint_names[{arm!r}]",
                length=positions[arm].shape[1],
            )
            for arm, value in raw_names.items()
        }
        if not isinstance(self.expected_start, RobotState):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("expected_start must be a RobotState")
        if self.expected_start.embodiment != embodiment:
            raise ValueError("expected_start embodiment must match the trajectory")
        if set(self.expected_start.arms) != set(positions):
            raise ValueError("expected_start must have the same arms as joint_positions")
        for arm in positions:
            if self.expected_start.joint_positions[arm].shape[0] != positions[arm].shape[1]:
                raise ValueError("expected_start must match each arm's trajectory joint dimension")
            if tuple(self.expected_start.joint_names[arm]) != tuple(names[arm]):
                raise ValueError("expected_start joint names must match the trajectory")
        grippers = None
        if self.gripper_positions is not None:
            raw_grippers = _mapping(self.gripper_positions, "gripper_positions", allow_empty=False)
            if set(raw_grippers) != set(positions):
                raise ValueError("gripper_positions must have the same arms as joint_positions")
            count = next(iter(lengths))
            grippers = {
                arm: _array(value, f"gripper_positions[{arm!r}]", shape=(count,))
                for arm, value in raw_grippers.items()
            }
            if any(np.any((value < 0.0) | (value > 1.0)) for value in grippers.values()):
                raise ValueError("gripper_positions values must be in [0, 1]")
        object.__setattr__(self, "joint_positions", MappingProxyType(positions))
        object.__setattr__(self, "joint_names", MappingProxyType(names))
        object.__setattr__(
            self, "gripper_positions", None if grippers is None else MappingProxyType(grippers)
        )
        object.__setattr__(
            self,
            "dt_s",
            _finite_float(self.dt_s, "dt_s", minimum=0.0, strict_minimum=True),
        )
        object.__setattr__(self, "planner", _non_empty_string(self.planner, "planner"))
        object.__setattr__(self, "collision_aware", _bool(self.collision_aware, "collision_aware"))
        object.__setattr__(self, "embodiment", embodiment)

    @property
    def waypoint_count(self) -> int:
        return next(iter(self.joint_positions.values())).shape[0]


@dataclass(frozen=True, slots=True)
class Segmentation:
    """One scored binary mask in an explicitly named camera frame."""

    mask: np.ndarray
    label: str
    score: float
    camera_name: str
    frame: str
    box_xyxy: np.ndarray | None = None

    def __post_init__(self) -> None:
        try:
            raw_mask = np.asarray(self.mask)
        except (TypeError, ValueError) as exc:
            raise ValueError("mask must be a two-dimensional array") from exc
        if raw_mask.ndim != 2 or raw_mask.shape[0] == 0 or raw_mask.shape[1] == 0:
            raise ValueError("mask must have non-empty shape (H, W)")
        if raw_mask.dtype != np.bool_:
            if not (
                np.issubdtype(raw_mask.dtype, np.integer)
                or np.issubdtype(raw_mask.dtype, np.floating)
            ):
                raise ValueError("mask must be boolean or contain only 0/1 values")
            if not bool(np.all(np.isfinite(raw_mask))) or not bool(
                np.all((raw_mask == 0) | (raw_mask == 1))
            ):
                raise ValueError("mask must be boolean or contain only 0/1 values")
        mask = np.array(raw_mask, dtype=np.bool_, copy=True, order="C")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "label", _non_empty_string(self.label, "label"))
        object.__setattr__(
            self,
            "score",
            _finite_float(self.score, "score", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "camera_name",
            _non_empty_string(self.camera_name, "camera_name"),
        )
        object.__setattr__(self, "frame", _non_empty_string(self.frame, "frame"))
        if self.box_xyxy is not None:
            box = _array(self.box_xyxy, "box_xyxy", shape=(4,))
            x1, y1, x2, y2 = box
            if x2 < x1 or y2 < y1:
                raise ValueError("box_xyxy must satisfy x2 >= x1 and y2 >= y1")
            object.__setattr__(self, "box_xyxy", box)


@dataclass(frozen=True, slots=True)
class SegmentationSet:
    """Recoverable result of a segmentation provider call."""

    ok: bool
    segmentations: Sequence[Segmentation] = ()
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if isinstance(self.segmentations, str | bytes) or not isinstance(
            self.segmentations, Sequence
        ):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("segmentations must be a sequence")
        items = tuple(self.segmentations)
        if any(not isinstance(item, Segmentation) for item in items):
            raise ValueError("segmentations must contain only Segmentation values")
        if ok and not items:
            raise ValueError("a successful SegmentationSet must contain at least one segmentation")
        if not ok and items:
            raise ValueError("a failed SegmentationSet cannot carry segmentations")
        if items:
            expected = (items[0].camera_name, items[0].frame, items[0].mask.shape)
            if any(
                (item.camera_name, item.frame, item.mask.shape) != expected for item in items[1:]
            ):
                raise ValueError("all segmentations must share camera, frame, and mask shape")
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "segmentations", items)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    @property
    def value(self) -> tuple[Segmentation, ...] | None:
        return self.segmentations if self.ok else None


@dataclass(frozen=True, slots=True)
class DepthEstimateResult:
    """Recoverable metric-depth estimate from one or more camera frames."""

    ok: bool
    provider: str
    source_frames: Sequence[str]
    depth_m: np.ndarray | None = None
    disparity_px: np.ndarray | None = None
    valid_fraction: float = 0.0
    valid_min_m: float = 0.0
    valid_max_m: float = 0.0
    median_depth_m: float = 0.0
    prior_median_abs_error_m: float | None = None
    inference_time_s: float | None = None
    device: str | None = None
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        provider = _non_empty_string(self.provider, "provider")
        source_frames = _string_tuple(self.source_frames, "source_frames")
        if not source_frames:
            raise ValueError("source_frames must contain at least one frame")

        depth = None
        if self.depth_m is not None:
            depth = _array(self.depth_m, "depth_m", shape=(None, None), dtype=np.float32)
            if depth.shape[0] == 0 or depth.shape[1] == 0:
                raise ValueError("depth_m must have non-empty shape (H, W)")
            if bool(np.any(depth < 0.0)):
                raise ValueError("depth_m values must be non-negative")

        disparity = None
        if self.disparity_px is not None:
            disparity = _array(
                self.disparity_px,
                "disparity_px",
                shape=(None, None),
                dtype=np.float32,
            )
            if depth is None or disparity.shape != depth.shape:
                raise ValueError("disparity_px must match depth_m shape")

        valid_fraction = _finite_float(
            self.valid_fraction,
            "valid_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        valid_min = _finite_float(self.valid_min_m, "valid_min_m", minimum=0.0)
        valid_max = _finite_float(self.valid_max_m, "valid_max_m", minimum=0.0)
        median = _finite_float(self.median_depth_m, "median_depth_m", minimum=0.0)
        if ok:
            if depth is None or valid_fraction <= 0.0 or not bool(np.any(depth > 0.0)):
                raise ValueError("a successful DepthEstimateResult must carry valid metric depth")
            if not 0.0 < valid_min <= median <= valid_max:
                raise ValueError(
                    "successful depth statistics must satisfy 0 < min <= median <= max"
                )
        elif depth is not None or disparity is not None:
            raise ValueError("a failed DepthEstimateResult cannot carry depth or disparity")

        prior_error = self.prior_median_abs_error_m
        if prior_error is not None:
            prior_error = _finite_float(
                prior_error,
                "prior_median_abs_error_m",
                minimum=0.0,
            )
        inference_time = self.inference_time_s
        if inference_time is not None:
            inference_time = _finite_float(
                inference_time,
                "inference_time_s",
                minimum=0.0,
            )
        device = self.device
        if device is not None:
            device = _non_empty_string(device, "device")

        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "source_frames", source_frames)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "disparity_px", disparity)
        object.__setattr__(self, "valid_fraction", valid_fraction)
        object.__setattr__(self, "valid_min_m", valid_min)
        object.__setattr__(self, "valid_max_m", valid_max)
        object.__setattr__(self, "median_depth_m", median)
        object.__setattr__(self, "prior_median_abs_error_m", prior_error)
        object.__setattr__(self, "inference_time_s", inference_time)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    @property
    def value(self) -> np.ndarray | None:
        """Return the metric depth array for successful calls."""
        return self.depth_m if self.ok else None


@dataclass(frozen=True, slots=True)
class PointCloud:
    """Finite XYZ points, and optional RGB colors, expressed in ``frame``."""

    points: np.ndarray
    frame: str
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = _array(self.points, "points", shape=(None, 3))
        if points.shape[0] == 0:
            raise ValueError("points must contain at least one point")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "frame", _non_empty_string(self.frame, "frame"))
        if self.colors is not None:
            try:
                colors = np.asarray(self.colors)
            except (TypeError, ValueError) as exc:
                raise ValueError("colors must be a numeric array") from exc
            if colors.shape != points.shape:
                raise ValueError(f"colors must have shape {points.shape}, got {colors.shape}")
            if np.issubdtype(colors.dtype, np.bool_) or not (
                np.issubdtype(colors.dtype, np.integer) or np.issubdtype(colors.dtype, np.floating)
            ):
                raise ValueError("colors must use an integer or floating dtype")
            if not bool(np.all(np.isfinite(colors))):
                raise ValueError("colors must contain only finite values")
            color_min = float(np.min(colors))
            color_max = float(np.max(colors))
            upper = 255.0 if np.issubdtype(colors.dtype, np.integer) else 1.0
            if color_min < 0.0 or color_max > upper:
                raise ValueError(f"colors values must be in [0, {upper:g}]")
            colors = np.array(colors, copy=True, order="C")
            colors.setflags(write=False)
            object.__setattr__(self, "colors", colors)


@dataclass(frozen=True, slots=True)
class PlanningScene:
    """A public-observation-derived collision scene in one explicit frame."""

    point_cloud: PointCloud
    voxel_size_m: float = 0.01
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.point_cloud, PointCloud):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("point_cloud must be a PointCloud")
        object.__setattr__(
            self,
            "voxel_size_m",
            _finite_float(
                self.voxel_size_m,
                "voxel_size_m",
                minimum=0.0,
                strict_minimum=True,
            ),
        )
        object.__setattr__(self, "timestamp_s", _optional_timestamp(self.timestamp_s))

    @property
    def frame(self) -> str:
        return self.point_cloud.frame


@dataclass(frozen=True, slots=True)
class RobotPlanningContext:
    """Non-policy-facing robot model and arm calibration for a planner."""

    embodiment: str
    model: str
    joint_names: Mapping[str, Sequence[str]]
    base_transforms: Mapping[str, np.ndarray]
    end_effector_links: Mapping[str, str]

    def __post_init__(self) -> None:
        embodiment = _non_empty_string(self.embodiment, "embodiment")
        joint_dimension = _joint_dimension(embodiment)
        raw_names = _mapping(self.joint_names, "joint_names", allow_empty=False)
        names = {
            arm: _string_tuple(value, f"joint_names[{arm!r}]", length=joint_dimension)
            for arm, value in raw_names.items()
        }
        if any(len(value) not in SUPPORTED_JOINT_DIMENSIONS for value in names.values()):
            raise ValueError("joint_names must contain 6 or 7 entries per arm")
        raw_transforms = _mapping(self.base_transforms, "base_transforms", allow_empty=False)
        raw_links = _mapping(self.end_effector_links, "end_effector_links", allow_empty=False)
        if set(raw_transforms) != set(names) or set(raw_links) != set(names):
            raise ValueError(
                "joint_names, base_transforms, and end_effector_links must share arm names"
            )
        transforms: dict[str, np.ndarray] = {}
        for arm, value in raw_transforms.items():
            transform = _array(value, f"base_transforms[{arm!r}]", shape=(4, 4))
            if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
                raise ValueError("base transforms must be homogeneous 4x4 matrices")
            transforms[arm] = transform
        links = {
            arm: _non_empty_string(value, f"end_effector_links[{arm!r}]")
            for arm, value in raw_links.items()
        }
        object.__setattr__(self, "embodiment", embodiment)
        object.__setattr__(self, "model", _non_empty_string(self.model, "model"))
        object.__setattr__(self, "joint_names", MappingProxyType(names))
        object.__setattr__(self, "base_transforms", MappingProxyType(transforms))
        object.__setattr__(self, "end_effector_links", MappingProxyType(links))

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.joint_names)


@dataclass(frozen=True, slots=True)
class MotionStrategy:
    """Per-call selection of IK, trajectory, and optional integrated pose planning."""

    ik_solver: str = "pyroki"
    trajectory_planner: str = "interpolation"
    pose_planner: str = "composed"
    time_dilation_factor: float | None = None
    interpolation_dt_s: float | None = None
    maximum_trajectory_dt_s: float | None = None

    def __post_init__(self) -> None:
        ik_solver = _non_empty_string(self.ik_solver, "ik_solver")
        trajectory_planner = _non_empty_string(self.trajectory_planner, "trajectory_planner")
        pose_planner = _non_empty_string(self.pose_planner, "pose_planner")
        if ik_solver not in {"pyroki", "curobo", "mink"}:
            raise ValueError("ik_solver must be 'pyroki', 'curobo', or 'mink'")
        if trajectory_planner not in {"interpolation", "curobo"}:
            raise ValueError("trajectory_planner must be 'interpolation' or 'curobo'")
        if pose_planner not in {"composed", "curobo-integrated"}:
            raise ValueError("pose_planner must be 'composed' or 'curobo-integrated'")
        object.__setattr__(self, "ik_solver", ik_solver)
        object.__setattr__(self, "trajectory_planner", trajectory_planner)
        object.__setattr__(self, "pose_planner", pose_planner)
        if self.time_dilation_factor is not None:
            object.__setattr__(
                self,
                "time_dilation_factor",
                _finite_float(
                    self.time_dilation_factor,
                    "time_dilation_factor",
                    minimum=0.0,
                    maximum=1.0,
                    strict_minimum=True,
                ),
            )
        if self.interpolation_dt_s is not None:
            object.__setattr__(
                self,
                "interpolation_dt_s",
                _finite_float(
                    self.interpolation_dt_s,
                    "interpolation_dt_s",
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )
        if self.maximum_trajectory_dt_s is not None:
            object.__setattr__(
                self,
                "maximum_trajectory_dt_s",
                _finite_float(
                    self.maximum_trajectory_dt_s,
                    "maximum_trajectory_dt_s",
                    minimum=0.0,
                    strict_minimum=True,
                ),
            )


@dataclass(frozen=True, slots=True)
class ObjectGeometry:
    """An oriented bounding box represented by a pose and positive XYZ extents."""

    pose: Pose
    extents: np.ndarray
    point_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("pose must be a Pose")
        extents = _array(self.extents, "extents", shape=(3,))
        if bool(np.any(extents <= 0.0)):
            raise ValueError("extents must be strictly positive")
        object.__setattr__(self, "extents", extents)
        if self.point_count is not None:
            if isinstance(self.point_count, bool) or not isinstance(
                self.point_count, int | np.integer
            ):
                raise ValueError("point_count must be a positive integer or None")
            point_count = int(self.point_count)
            if point_count <= 0:
                raise ValueError("point_count must be a positive integer or None")
            object.__setattr__(self, "point_count", point_count)

    @property
    def frame(self) -> str:
        return self.pose.frame


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """A scored gripper pose and opening width in metres."""

    pose: Pose
    score: float
    width_m: float
    approach_distance_m: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("pose must be a Pose")
        object.__setattr__(
            self,
            "score",
            _finite_float(self.score, "score", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "width_m",
            _finite_float(self.width_m, "width_m", minimum=0.0),
        )
        object.__setattr__(
            self,
            "approach_distance_m",
            _finite_float(self.approach_distance_m, "approach_distance_m", minimum=0.0),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def frame(self) -> str:
        return self.pose.frame

    @property
    def transform(self) -> np.ndarray:
        return self.pose.as_matrix()


@dataclass(frozen=True, slots=True)
class GraspSet:
    """Recoverable result of grasp generation."""

    ok: bool
    grasps: Sequence[GraspCandidate] = ()
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if isinstance(self.grasps, str | bytes) or not isinstance(self.grasps, Sequence):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("grasps must be a sequence")
        grasps = tuple(self.grasps)
        if any(not isinstance(item, GraspCandidate) for item in grasps):
            raise ValueError("grasps must contain only GraspCandidate values")
        if ok and not grasps:
            raise ValueError("a successful GraspSet must contain at least one grasp")
        if not ok and grasps:
            raise ValueError("a failed GraspSet cannot carry grasps")
        if grasps and any(item.frame != grasps[0].frame for item in grasps[1:]):
            raise ValueError("all grasp candidates must use the same frame")
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "grasps", grasps)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    @property
    def value(self) -> tuple[GraspCandidate, ...] | None:
        return self.grasps if self.ok else None


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Recoverable composed localization result with optional intermediate values."""

    ok: bool
    geometry: ObjectGeometry | None = None
    segmentation: Segmentation | None = None
    point_cloud: PointCloud | None = None
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if ok and self.geometry is None:
            raise ValueError("a successful LocalizationResult must carry geometry")
        if self.geometry is not None and not isinstance(self.geometry, ObjectGeometry):
            raise ValueError("geometry must be an ObjectGeometry or None")
        if self.segmentation is not None and not isinstance(self.segmentation, Segmentation):
            raise ValueError("segmentation must be a Segmentation or None")
        if self.point_cloud is not None and not isinstance(self.point_cloud, PointCloud):
            raise ValueError("point_cloud must be a PointCloud or None")
        if (
            self.geometry is not None
            and self.point_cloud is not None
            and self.geometry.frame != self.point_cloud.frame
        ):
            raise ValueError("geometry and point_cloud must use the same frame")
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    @property
    def value(self) -> ObjectGeometry | None:
        return self.geometry if self.ok else None


@dataclass(frozen=True, slots=True)
class IKResult:
    """Recoverable inverse-kinematics result for one arm."""

    ok: bool
    joint_positions: np.ndarray | None = None
    arm: str = "primary"
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    embodiment: str = DEFAULT_EMBODIMENT

    def __post_init__(self) -> None:
        embodiment = _non_empty_string(self.embodiment, "embodiment")
        joint_dimension = _joint_dimension(embodiment)
        ok = _validate_result_status(self.ok, self.error)
        joints = self.joint_positions
        if ok and joints is None:
            raise ValueError("a successful IKResult must carry joint_positions")
        if not ok and joints is not None:
            raise ValueError("a failed IKResult cannot carry joint_positions")
        if joints is not None:
            joints = _array(joints, "joint_positions", shape=(joint_dimension,))
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "joint_positions", joints)
        object.__setattr__(self, "arm", _arm_key(self.arm))
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))
        object.__setattr__(self, "embodiment", embodiment)

    @property
    def value(self) -> np.ndarray | None:
        return self.joint_positions if self.ok else None


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Recoverable motion-planning result for one arm."""

    ok: bool
    trajectory: Trajectory | None = None
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if ok and self.trajectory is None:
            raise ValueError("a successful PlanResult must carry a trajectory")
        if not ok and self.trajectory is not None:
            raise ValueError("a failed PlanResult cannot carry a trajectory")
        if self.trajectory is not None and not isinstance(self.trajectory, Trajectory):
            raise ValueError("trajectory must be a Trajectory or None")
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    @property
    def value(self) -> Trajectory | None:
        return self.trajectory if self.ok else None


@dataclass(frozen=True, slots=True)
class SynchronizedPlanResult:
    """Recoverable result of synchronized multi-arm motion planning."""

    ok: bool
    trajectory: SynchronizedTrajectory | None = None
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if ok and self.trajectory is None:
            raise ValueError("a successful SynchronizedPlanResult must carry a trajectory")
        if not ok and self.trajectory is not None:
            raise ValueError("a failed SynchronizedPlanResult cannot carry a trajectory")
        if self.trajectory is not None and not isinstance(self.trajectory, SynchronizedTrajectory):
            raise ValueError("trajectory must be a SynchronizedTrajectory or None")
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    @property
    def value(self) -> SynchronizedTrajectory | None:
        return self.trajectory if self.ok else None


_SENSITIVE_DIAGNOSTIC_TERMS = (
    "_private",
    "ground_truth",
    "groundtruth",
    "mujoco",
    "object_pose",
    "privileged",
    "qpos",
    "qvel",
    "raw_observation",
    "raw_state",
    "reward",
    "sensitive",
    "sim_state",
    "success_predicate",
)
_SENSITIVE_DIAGNOSTIC_NAMES = frozenset({"private", "success"})


def _is_sensitive_diagnostic_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalized.startswith("_")
        or normalized in _SENSITIVE_DIAGNOSTIC_NAMES
        or any(term in normalized for term in _SENSITIVE_DIAGNOSTIC_TERMS)
    )


def _public_diagnostic_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _public_diagnostic_value(item)
                for key, item in value.items()
                if isinstance(key, str) and not _is_sensitive_diagnostic_key(key)
            }
        )
    if isinstance(value, tuple):
        return tuple(_public_diagnostic_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_public_diagnostic_value(item) for item in value)
    return value


def _public_diagnostics(value: Mapping[str, object]) -> Mapping[str, object]:
    sanitized = _public_diagnostic_value(value)
    if not isinstance(sanitized, Mapping):  # pragma: no cover - defensive type narrowing
        # Internal invariant failure, not invalid caller input.
        raise AssertionError("diagnostic sanitization must return a mapping")
    return sanitized


def _public_error(error: ApiError | None) -> ApiError | None:
    if error is None:
        return None
    return ApiError(
        code=error.code,
        message=error.message,
        recoverable=error.recoverable,
        details=_public_diagnostics(error.details),
    )


class ExecutionStatus(str, Enum):
    """Stable terminal classification for a motion or control attempt."""

    SUCCESS = "success"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    STALE_PLAN = "stale_plan"
    PLANNER_FAILED = "planner_failed"
    CONTROLLER_FAILED = "controller_failed"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


def _execution_status(
    ok: bool,
    error: ApiError | None,
    terminated: bool,
    truncated: bool,
) -> ExecutionStatus:
    if ok:
        return ExecutionStatus.SUCCESS
    if terminated:
        return ExecutionStatus.TERMINATED
    if truncated:
        return ExecutionStatus.TRUNCATED
    if error is None:  # _validate_result_status rejects this first; defensive typing only.
        return ExecutionStatus.FAILED
    mapping = {
        ErrorCode.TERMINATED: ExecutionStatus.TERMINATED,
        ErrorCode.TRUNCATED: ExecutionStatus.TRUNCATED,
        ErrorCode.TIMEOUT: ExecutionStatus.TIMEOUT,
        ErrorCode.CANCELLED: ExecutionStatus.CANCELLED,
        ErrorCode.STALE_PLAN: ExecutionStatus.STALE_PLAN,
        ErrorCode.PLANNING_FAILED: ExecutionStatus.PLANNER_FAILED,
        ErrorCode.IK_FAILED: ExecutionStatus.PLANNER_FAILED,
        ErrorCode.IK_UNREACHABLE: ExecutionStatus.PLANNER_FAILED,
        ErrorCode.EXECUTION_FAILED: ExecutionStatus.CONTROLLER_FAILED,
        ErrorCode.CONTROLLER_FAILED: ExecutionStatus.CONTROLLER_FAILED,
        ErrorCode.ADAPTER_FAILED: ExecutionStatus.CONTROLLER_FAILED,
        ErrorCode.INVALID_REQUEST: ExecutionStatus.INVALID_REQUEST,
        ErrorCode.UNSUPPORTED: ExecutionStatus.UNSUPPORTED,
    }
    return mapping.get(error.code, ExecutionStatus.FAILED)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of exactly one adapter control period.

    ``reward`` and diagnostics marked by sensitive key names are runtime-only.
    Call :meth:`public_view` before exposing a step result to generated code.
    """

    ok: bool
    observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    reward: float | None = None
    error: ApiError | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if ok and self.observation is None:
            raise ValueError("a successful StepResult must carry an observation")
        if self.observation is not None and not isinstance(self.observation, Observation):
            raise ValueError("observation must be an Observation or None")
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "terminated", _bool(self.terminated, "terminated"))
        object.__setattr__(self, "truncated", _bool(self.truncated, "truncated"))
        if self.reward is not None:
            object.__setattr__(self, "reward", _finite_float(self.reward, "reward"))
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    def public_view(self) -> StepResult:
        """Return a copy without runtime reward or privileged diagnostic data."""
        return replace(
            self,
            reward=None,
            error=_public_error(self.error),
            diagnostics=_public_diagnostics(self.diagnostics),
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Recoverable result of timed trajectory execution."""

    ok: bool
    steps_executed: int
    status: ExecutionStatus | str | None = None
    final_observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    error: ApiError | None = None
    final_errors: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ok = _validate_result_status(self.ok, self.error)
        if isinstance(self.steps_executed, bool) or not isinstance(
            self.steps_executed, int | np.integer
        ):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("steps_executed must be a non-negative integer")
        steps = int(self.steps_executed)
        if steps < 0:
            raise ValueError("steps_executed must be a non-negative integer")
        if self.final_observation is not None and not isinstance(
            self.final_observation, Observation
        ):
            raise ValueError("final_observation must be an Observation or None")
        terminated = _bool(self.terminated, "terminated")
        truncated = _bool(self.truncated, "truncated")
        if terminated and truncated:
            raise ValueError("execution cannot be both terminated and truncated")
        if ok and (terminated or truncated):
            raise ValueError("terminated or truncated execution cannot be successful")
        status = self.status
        if status is None:
            status = _execution_status(ok, self.error, terminated, truncated)
        elif isinstance(status, str) and not isinstance(status, ExecutionStatus):
            try:
                status = ExecutionStatus(status)
            except ValueError as exc:
                raise ValueError(f"unknown execution status: {self.status!r}") from exc
        if not isinstance(status, ExecutionStatus):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("status must be an ExecutionStatus")
        if ok != (status is ExecutionStatus.SUCCESS):
            raise ValueError("only success status may have ok=True")
        inferred_status = _execution_status(ok, self.error, terminated, truncated)
        if status is not inferred_status:
            raise ValueError(
                f"status {status.value!r} conflicts with execution outcome "
                f"{inferred_status.value!r}"
            )
        raw_final_errors = _mapping(self.final_errors, "final_errors")
        final_errors = {
            arm: _finite_float(value, f"final_errors[{arm!r}]", minimum=0.0)
            for arm, value in raw_final_errors.items()
        }
        object.__setattr__(self, "ok", ok)
        object.__setattr__(self, "steps_executed", steps)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "final_errors", MappingProxyType(final_errors))
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

    def public_view(self) -> ExecutionResult:
        """Return a copy with privileged diagnostic data removed."""
        return replace(
            self,
            error=_public_error(self.error),
            diagnostics=_public_diagnostics(self.diagnostics),
        )


__all__ = [
    "DEFAULT_EMBODIMENT",
    "DEFAULT_JOINT_NAMES",
    "JOINT_DIMENSION",
    "JOINT_DIMENSIONS",
    "SUPPORTED_JOINT_DIMENSIONS",
    "ArmCommand",
    "CameraObservation",
    "DepthEstimateResult",
    "ExecutionResult",
    "ExecutionStatus",
    "GraspCandidate",
    "GraspSet",
    "IKResult",
    "LocalizationResult",
    "MotionStrategy",
    "ObjectGeometry",
    "Observation",
    "PlanResult",
    "PlanningScene",
    "PointCloud",
    "Pose",
    "RobotAction",
    "RobotPlanningContext",
    "RobotState",
    "Segmentation",
    "SegmentationSet",
    "StepResult",
    "SynchronizedPlanResult",
    "SynchronizedTrajectory",
    "TaskContext",
    "Trajectory",
]
