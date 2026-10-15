"""Frame-safe geometry helpers for perception and grasp composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    GraspCandidate,
    GraspSet,
    ObjectGeometry,
    PointCloud,
    Pose,
    Segmentation,
)

_ROTATION_ATOL = 1e-5


def _finite_array(value: object, *, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=False)
    ):
        expected = tuple("*" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rotation_matrix(value: object, *, name: str = "rotation") -> np.ndarray:
    rotation = _finite_array(value, shape=(3, 3), name=name)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return rotation


def _transform_matrix(value: object, *, name: str = "transform") -> np.ndarray:
    transform = _finite_array(value, shape=(4, 4), name=name)
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} final row must be [0, 0, 0, 1]")
    _rotation_matrix(transform[:3, :3], name=f"{name} rotation")
    return transform


def quaternion_wxyz_to_matrix(quaternion_wxyz: object) -> np.ndarray:
    """Convert a normalized WXYZ quaternion to a 3x3 rotation matrix."""
    quaternion = _finite_array(quaternion_wxyz, shape=(4,), name="quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("quaternion_wxyz must be normalized to unit length")
    w, x, y, z = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    rotation.setflags(write=False)
    return rotation


def matrix_to_quaternion_wxyz(rotation: object) -> np.ndarray:
    """Convert a 3x3 rotation matrix to canonical unit WXYZ form."""
    matrix = _rotation_matrix(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quaternion = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quaternion = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    quaternion.setflags(write=False)
    return quaternion


def rpy_deg_to_quaternion_wxyz(rpy_deg: object) -> np.ndarray:
    """Convert fixed-axis XYZ degrees: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    angles = _finite_array(rpy_deg, shape=(3,), name="rpy_deg")
    cr, cp, cy = np.cos(np.deg2rad(angles % 360) / 2)
    sr, sp, sy = np.sin(np.deg2rad(angles % 360) / 2)
    quaternion = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    quaternion.setflags(write=False)
    return quaternion


def pose_from_mapping(value: Mapping[str, object]) -> Pose:
    """Decode an absolute pose with exactly one quaternion or RPY orientation."""
    if set(value) - {"position", "quaternion_wxyz", "rpy_deg", "frame"}:
        raise ValueError("unknown pose target field")
    if not {"position", "frame"}.issubset(value):
        raise ValueError("pose target requires position and frame")
    if ("rpy_deg" in value) == ("quaternion_wxyz" in value):
        raise ValueError("specify exactly one of rpy_deg or quaternion_wxyz")
    quaternion = (
        rpy_deg_to_quaternion_wxyz(value["rpy_deg"])
        if "rpy_deg" in value
        else value["quaternion_wxyz"]
    )
    return Pose(position=value["position"], quaternion_wxyz=quaternion, frame=value["frame"])


def pose_to_matrix(pose: Pose) -> np.ndarray:
    if not isinstance(pose, Pose):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("pose must be a Pose")
    return pose.as_matrix()


def matrix_to_pose(transform: object, *, frame: str) -> Pose:
    """Create a framed pose from a homogeneous transform using WXYZ."""
    matrix = _transform_matrix(transform)
    return Pose(
        position=matrix[:3, 3],
        quaternion_wxyz=matrix_to_quaternion_wxyz(matrix[:3, :3]),
        frame=frame,
    )


def invert_transform(transform: object) -> np.ndarray:
    matrix = _transform_matrix(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    inverse.setflags(write=False)
    return inverse


def transform_points(points: object, transform: object) -> np.ndarray:
    """Apply one validated homogeneous transform to finite XYZ points."""
    xyz = _finite_array(points, shape=(None, 3), name="points")
    matrix = _transform_matrix(transform)
    transformed = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    transformed.setflags(write=False)
    return transformed


def transform_point_cloud(
    point_cloud: PointCloud,
    transform: object,
    *,
    source_frame: str,
    target_frame: str,
) -> PointCloud:
    """Transform a point cloud while enforcing the declared source frame."""
    if not isinstance(point_cloud, PointCloud):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("point_cloud must be a PointCloud")
    if point_cloud.frame != source_frame:
        raise ValueError(
            f"point cloud is in frame {point_cloud.frame!r}, expected {source_frame!r}"
        )
    return PointCloud(
        points=transform_points(point_cloud.points, transform),
        frame=target_frame,
        colors=point_cloud.colors,
    )


def crop_point_cloud(
    point_cloud: PointCloud,
    lower_xyz: object,
    upper_xyz: object,
) -> PointCloud:
    """Crop public XYZ observations to inclusive axis-aligned bounds."""
    if not isinstance(point_cloud, PointCloud):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("point_cloud must be a PointCloud")
    lower = _finite_array(lower_xyz, shape=(3,), name="lower_xyz")
    upper = _finite_array(upper_xyz, shape=(3,), name="upper_xyz")
    if bool(np.any(lower >= upper)):
        raise ValueError("lower_xyz must be strictly less than upper_xyz on every axis")

    keep = np.all((point_cloud.points >= lower) & (point_cloud.points <= upper), axis=1)
    if not bool(np.any(keep)):
        raise ValueError("point cloud contains no points inside the requested bounds")
    colors = None if point_cloud.colors is None else point_cloud.colors[keep]
    return PointCloud(
        points=point_cloud.points[keep],
        frame=point_cloud.frame,
        colors=colors,
    )


def transform_pose(
    pose: Pose,
    transform: object,
    *,
    source_frame: str,
    target_frame: str,
) -> Pose:
    """Apply a target-from-source transform to a pose in ``source_frame``."""
    if not isinstance(pose, Pose):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("pose must be a Pose")
    if pose.frame != source_frame:
        raise ValueError(f"pose is in frame {pose.frame!r}, expected {source_frame!r}")
    return matrix_to_pose(
        _transform_matrix(transform) @ pose.as_matrix(),
        frame=target_frame,
    )


def mask_to_point_cloud(
    observation: CameraObservation,
    segmentation: Segmentation,
    *,
    target_frame: str | None = None,
) -> PointCloud:
    """Back-project a mask and optionally transform it to the camera parent frame."""
    if not isinstance(observation, CameraObservation):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("observation must be a CameraObservation")
    if not isinstance(segmentation, Segmentation):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("segmentation must be a Segmentation")
    if segmentation.frame != observation.frame:
        raise ValueError(
            f"segmentation frame {segmentation.frame!r} does not match camera frame "
            f"{observation.frame!r}"
        )
    if segmentation.mask.shape != observation.depth_m.shape:
        raise ValueError("segmentation mask shape does not match camera depth")

    rows, columns = np.nonzero(segmentation.mask)
    depths = observation.depth_m[rows, columns]
    valid = np.isfinite(depths) & (depths > 0.0)
    rows = rows[valid]
    columns = columns[valid]
    depths = depths[valid]
    if depths.size == 0:
        raise ValueError("segmentation contains no pixels with valid positive depth")

    intrinsics = observation.intrinsics
    x = (columns - intrinsics[0, 2]) * depths / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * depths / intrinsics[1, 1]
    camera_points = np.column_stack((x, y, depths))
    colors = observation.rgb[rows, columns]
    camera_cloud = PointCloud(
        points=camera_points,
        frame=observation.frame,
        colors=colors,
    )

    destination = observation.frame if target_frame is None else target_frame
    if destination == observation.frame:
        return camera_cloud
    if destination != observation.camera_pose.frame:
        raise ValueError(
            f"camera pose only defines {observation.frame!r} to "
            f"{observation.camera_pose.frame!r}, not {destination!r}"
        )
    return transform_point_cloud(
        camera_cloud,
        observation.camera_pose.as_matrix(),
        source_frame=observation.frame,
        target_frame=destination,
    )


def estimate_oriented_bounding_box(point_cloud: PointCloud) -> ObjectGeometry:
    """Estimate a deterministic PCA-oriented bounding box."""
    if not isinstance(point_cloud, PointCloud):
        # Preserve the invalid-data ValueError contract.
        raise ValueError("point_cloud must be a PointCloud")
    points = point_cloud.points
    if len(points) < 4:
        raise ValueError("at least four points are required for a 3D oriented bounding box")

    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / float(len(points))
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = axes[:, order]

    # Eigenvector signs are arbitrary.  Fix each dominant component positive,
    # then restore a right-handed basis for deterministic WXYZ output.
    for column in range(3):
        dominant = int(np.argmax(np.abs(axes[:, column])))
        if axes[dominant, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0

    local = centered @ axes
    lower = np.min(local, axis=0)
    upper = np.max(local, axis=0)
    extents = upper - lower
    if not bool(np.all(np.isfinite(extents))) or bool(np.any(extents <= 1e-9)):
        raise ValueError("point cloud is degenerate and has no finite 3D bounding box")
    center_local = 0.5 * (lower + upper)
    center_world = np.mean(points, axis=0) + axes @ center_local
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axes
    transform[:3, 3] = center_world
    return ObjectGeometry(
        pose=matrix_to_pose(transform, frame=point_cloud.frame),
        extents=extents,
        point_count=len(points),
    )


def select_top_down_grasp(
    grasps: GraspSet | Sequence[GraspCandidate],
    *,
    frame: str | None = None,
    vertical_threshold: float = 0.8,
    up_axis: object = (0.0, 0.0, 1.0),
    approach_axis: int = 2,
) -> GraspCandidate | None:
    """Select the highest-scoring grasp whose approach points down.

    The convention is explicit: ``approach_axis`` names a positive gripper
    rotation column, and top-down means that axis aligns with ``-up_axis``.
    """
    if isinstance(grasps, GraspSet):
        candidates = grasps.grasps if grasps.ok else ()
    elif isinstance(grasps, Sequence) and not isinstance(grasps, str | bytes):
        candidates = tuple(grasps)
    else:
        # Preserve the invalid-data ValueError contract.
        raise ValueError("grasps must be a GraspSet or sequence of GraspCandidate")
    if any(not isinstance(candidate, GraspCandidate) for candidate in candidates):
        raise ValueError("grasps must contain only GraspCandidate values")
    if not np.isfinite(vertical_threshold) or not 0.0 <= vertical_threshold <= 1.0:
        raise ValueError("vertical_threshold must be finite and in [0, 1]")
    if approach_axis not in (0, 1, 2):
        raise ValueError("approach_axis must be 0, 1, or 2")
    up = _finite_array(up_axis, shape=(3,), name="up_axis")
    norm = float(np.linalg.norm(up))
    if norm <= 0.0:
        raise ValueError("up_axis must be non-zero")
    up /= norm

    selected: GraspCandidate | None = None
    for candidate in candidates:
        expected_frame = candidate.frame if frame is None else frame
        if candidate.frame != expected_frame:
            raise ValueError(f"grasp is in frame {candidate.frame!r}, expected {expected_frame!r}")
        approach = candidate.pose.as_matrix()[:3, approach_axis]
        alignment = float(np.dot(-approach, up))
        if alignment >= vertical_threshold and (
            selected is None or candidate.score > selected.score
        ):
            selected = candidate
    return selected


# Concise aliases used by the API surface.
estimate_geometry = estimate_oriented_bounding_box
estimate_obb = estimate_oriented_bounding_box
matrix_to_quaternion = matrix_to_quaternion_wxyz
quaternion_to_matrix = quaternion_wxyz_to_matrix


__all__ = [
    "crop_point_cloud",
    "estimate_geometry",
    "estimate_obb",
    "estimate_oriented_bounding_box",
    "invert_transform",
    "mask_to_point_cloud",
    "matrix_to_pose",
    "matrix_to_quaternion",
    "matrix_to_quaternion_wxyz",
    "pose_from_mapping",
    "pose_to_matrix",
    "quaternion_to_matrix",
    "quaternion_wxyz_to_matrix",
    "rpy_deg_to_quaternion_wxyz",
    "select_top_down_grasp",
    "transform_point_cloud",
    "transform_points",
    "transform_pose",
]
