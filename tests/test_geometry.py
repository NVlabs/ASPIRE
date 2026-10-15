from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.contracts import (
    CameraObservation,
    GraspCandidate,
    GraspSet,
    PointCloud,
    Pose,
    Segmentation,
)
from cap_harness.geometry import (
    crop_point_cloud,
    estimate_oriented_bounding_box,
    mask_to_point_cloud,
    matrix_to_quaternion_wxyz,
    pose_from_mapping,
    quaternion_wxyz_to_matrix,
    rpy_deg_to_quaternion_wxyz,
    select_top_down_grasp,
    transform_point_cloud,
)


@pytest.mark.parametrize(
    "angles",
    [
        [0, 0, 0],
        [90, 0, 0],
        [0, 90, 0],
        [0, 0, 90],
        [23, -37, 61],
        [-40, 90, 25],
        [20, -90, -15],
        [720, -450, 1080],
    ],
)
def test_rpy_degrees_follow_fixed_axis_xyz_convention(angles):
    quaternion = rpy_deg_to_quaternion_wxyz(angles)
    expected = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()

    np.testing.assert_allclose(quaternion_wxyz_to_matrix(quaternion), expected, atol=1e-12)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    pose = pose_from_mapping({"position": [0.1, 0.2, 0.3], "rpy_deg": angles, "frame": "world"})
    np.testing.assert_allclose(pose.as_matrix()[:3, :3], expected, atol=1e-12)


@pytest.mark.parametrize("angles", [[0, 0], [[0, 0, 0]], [0, np.nan, 0], [np.inf, 0, 0], None])
def test_rpy_conversion_rejects_malformed_and_nonfinite_angles(angles):
    with pytest.raises(ValueError, match="rpy_deg"):
        rpy_deg_to_quaternion_wxyz(angles)


def test_wxyz_matrix_round_trip() -> None:
    quaternion = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    rotation = quaternion_wxyz_to_matrix(quaternion)

    np.testing.assert_allclose(matrix_to_quaternion_wxyz(rotation), quaternion)
    np.testing.assert_allclose(
        rotation @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_mask_projection_preserves_and_transforms_frames() -> None:
    camera = CameraObservation(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2)),
        intrinsics=np.eye(3),
        frame="camera",
        camera_pose=Pose(
            position=np.array([1.0, 2.0, 3.0]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="base",
        ),
    )
    segmentation = Segmentation(
        mask=np.array([[1, 0], [0, 0]], dtype=bool),
        label="object",
        score=1.0,
        camera_name="agentview",
        frame="camera",
    )

    camera_cloud = mask_to_point_cloud(camera, segmentation)
    base_cloud = mask_to_point_cloud(camera, segmentation, target_frame="base")

    assert camera_cloud.frame == "camera"
    np.testing.assert_allclose(camera_cloud.points, [[0.0, 0.0, 1.0]])
    assert base_cloud.frame == "base"
    np.testing.assert_allclose(base_cloud.points, [[1.0, 2.0, 4.0]])


def test_frame_mismatch_and_nonfinite_transform_are_rejected() -> None:
    cloud = PointCloud(points=np.array([[0.0, 0.0, 1.0]]), frame="camera")

    with pytest.raises(ValueError, match="expected 'world'"):
        transform_point_cloud(
            cloud,
            np.eye(4),
            source_frame="world",
            target_frame="base",
        )

    transform = np.eye(4)
    transform[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        transform_point_cloud(
            cloud,
            transform,
            source_frame="camera",
            target_frame="base",
        )


def test_point_cloud_crop_preserves_frame_and_aligned_colors() -> None:
    cloud = PointCloud(
        points=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [2.0, 0.0, 0.0],
            ]
        ),
        frame="base",
        colors=np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )

    cropped = crop_point_cloud(cloud, (0.0, -0.1, -0.1), (0.5, 0.5, 0.5))

    assert cropped.frame == "base"
    np.testing.assert_allclose(cropped.points, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(cropped.colors, [[0, 255, 0], [0, 0, 255]])


def test_point_cloud_crop_rejects_invalid_bounds_and_empty_result() -> None:
    cloud = PointCloud(points=np.array([[0.0, 0.0, 0.0]]), frame="base")

    with pytest.raises(ValueError, match="strictly less"):
        crop_point_cloud(cloud, (0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="no points"):
        crop_point_cloud(cloud, (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))


def test_obb_and_explicit_top_down_selection() -> None:
    points = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    geometry = estimate_oriented_bounding_box(PointCloud(points=points, frame="base"))
    np.testing.assert_allclose(np.sort(geometry.extents), [2.0, 2.0, 2.0])
    assert geometry.pose.frame == "base"

    top_down = np.eye(4)
    top_down[:3, :3] = np.diag([1.0, -1.0, -1.0])
    side = np.eye(4)
    candidates = GraspSet(
        ok=True,
        grasps=(
            GraspCandidate(
                pose=Pose(
                    position=np.zeros(3),
                    quaternion_wxyz=matrix_to_quaternion_wxyz(side[:3, :3]),
                    frame="base",
                ),
                score=0.99,
                width_m=0.08,
            ),
            GraspCandidate(
                pose=Pose(
                    position=np.zeros(3),
                    quaternion_wxyz=matrix_to_quaternion_wxyz(top_down[:3, :3]),
                    frame="base",
                ),
                score=0.7,
                width_m=0.08,
            ),
        ),
    )

    selected = select_top_down_grasp(candidates, frame="base")
    assert selected is candidates.grasps[1]
    with pytest.raises(ValueError, match="expected 'world'"):
        select_top_down_grasp(candidates, frame="world")
