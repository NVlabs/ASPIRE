from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import requests

from cap_harness.contracts import (
    CameraObservation,
    PlanningScene,
    PointCloud,
    Pose,
    RobotPlanningContext,
    RobotState,
)
from cap_harness.errors import ErrorCode
from cap_harness.providers.curobo import CuRoboProvider
from cap_harness.providers.curobo.dual import build_dual_panda_config
from cap_harness.providers.graspnet import ContactGraspNetProvider
from cap_harness.providers.http import HttpProviderClient
from cap_harness.providers.pyroki import PyRokiProvider
from cap_harness.providers.sam3 import Sam3Provider

# The cuRobo service module imports FastAPI, which only the provider venvs carry;
# in a simulator venv these service-side tests skip rather than fail collection.
curobo_service = pytest.importorskip("cap_harness.providers.curobo.service")
BaseRequest = curobo_service.BaseRequest
PlanRequest = curobo_service.PlanRequest
Target = curobo_service.Target
_filter_robot_points = curobo_service._filter_robot_points
_ordered_goal_poses = curobo_service._ordered_goal_poses
_requested_indices = curobo_service._requested_indices
_voxelized_scene_mesh = curobo_service._voxelized_scene_mesh


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome


def _camera() -> CameraObservation:
    return CameraObservation(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float64),
        intrinsics=np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]),
        frame="camera/agentview",
        camera_pose=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="robot_base",
        ),
    )


def _state() -> RobotState:
    return RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="robot_base",
        ),
        gripper_positions=0.5,
        base_frame="robot_base",
    )


def _numpy_b64(value: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        np.save(buffer, value, allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_numpy_b64(value: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(value)) as buffer:
        return np.load(buffer, allow_pickle=False)


def test_http_retries_only_up_to_configured_bound() -> None:
    session = FakeSession(
        [
            requests.Timeout("slow"),
            FakeResponse(503, {"detail": "busy"}),
            FakeResponse(200, {"ok": True}),
        ]
    )
    sleeps: list[float] = []
    client = HttpProviderClient(
        "http://service.test/",
        max_retries=2,
        retry_backoff_s=0.25,
        session=session,  # type: ignore[arg-type]
        sleep=sleeps.append,
    )

    assert client.post_json("/infer", {"x": 1}) == {"ok": True}
    assert len(session.calls) == 3
    assert sleeps == [0.25, 0.5]
    assert session.calls[-1]["timeout"] == 30.0


def test_sam3_decodes_text_mask_with_camera_identity() -> None:
    mask = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "mask_base64": base64.b64encode(mask.tobytes()).decode("ascii"),
                            "shape": [2, 3],
                            "box": [0, 0, 2, 2],
                            "score": 0.9,
                            "label": "cup",
                        }
                    ]
                },
            )
        ]
    )
    provider = Sam3Provider(session=session)  # type: ignore[arg-type]

    result = provider.segment_text(_camera(), "cup")

    assert result.ok
    assert result.segmentations[0].camera_name == "camera/agentview"
    assert result.segmentations[0].frame == "camera/agentview"
    np.testing.assert_array_equal(result.segmentations[0].box_xyxy, [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_array_equal(result.segmentations[0].mask, mask.astype(bool))
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["text_prompt"] == "cup"
    assert base64.b64decode(payload["image_base64"]).startswith(b"\x89PNG")


def test_sam3_malformed_mask_is_typed_failure() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "mask_base64": "not-base64",
                            "shape": [2, 3],
                            "score": 0.9,
                            "label": "cup",
                        }
                    ]
                },
            )
        ]
    )
    result = Sam3Provider(session=session).segment_text(_camera(), "cup")  # type: ignore[arg-type]

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.PERCEPTION_FAILED


def test_sam3_keeps_only_five_highest_scoring_masks() -> None:
    mask = np.ones((2, 3), dtype=np.uint8)
    results = [
        {
            "mask_base64": base64.b64encode(mask.tobytes()).decode("ascii"),
            "shape": [2, 3],
            "box": [0, 0, 2, 1],
            "score": score,
            "label": f"candidate-{index}",
        }
        for index, score in enumerate((0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6))
    ]
    session = FakeSession([FakeResponse(200, {"results": results})])

    result = Sam3Provider(session=session).segment_text(_camera(), "cup")  # type: ignore[arg-type]

    assert result.ok
    assert [item.score for item in result.segmentations] == [0.9, 0.8, 0.7, 0.6, 0.3]


def test_graspnet_applies_local_z_orientation_correction_in_input_frame() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "grasps_base64": _numpy_b64(transform[None]),
                    "scores_base64": _numpy_b64(np.array([0.8])),
                    "contact_pts_base64": _numpy_b64(np.array([[0.1, 0.2, 0.3]])),
                },
            )
        ]
    )
    provider = ContactGraspNetProvider(session=session)  # type: ignore[arg-type]
    cloud = PointCloud(points=np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]]), frame="camera")
    scene = PointCloud(
        points=np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.2, 0.0, 0.5]]),
        frame="camera",
    )

    result = provider.generate_grasps(cloud, scene_point_cloud=scene)

    assert result.ok
    assert result.grasps[0].frame == "camera"
    expected = transform.copy()
    expected[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(result.grasps[0].transform, expected, atol=1e-12)
    assert result.grasps[0].metadata["orientation_correction"] == "right_multiply_local_z_90_deg"
    assert session.calls[0]["url"].endswith("/plan_point_clouds")
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    np.testing.assert_array_equal(_decode_numpy_b64(payload["pc_full_base64"]), scene.points)
    np.testing.assert_array_equal(_decode_numpy_b64(payload["pc_segment_base64"]), cloud.points)


def test_pyroki_converts_eef_target_to_eight_value_panda_wire_contract() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "joint_positions": list(range(8)),
                    "position_error_m": 0.001,
                    "orientation_error_rad": 0.01,
                },
            )
        ]
    )
    provider = PyRokiProvider(session=session)  # type: ignore[arg-type]
    pose = Pose(
        position=np.array([0.1, 0.2, 0.3]),
        quaternion_wxyz=np.array([0.5, 0.5, 0.5, 0.5]),
        frame="robot_base",
    )

    result = provider.solve_ik(pose, _state())

    assert result.ok
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["target_pose_wxyz_xyz"] == pytest.approx([0.5, 0.5, 0.5, 0.5, 0.0, 0.2, 0.3])
    assert payload["prev_cfg"] == [0.0] * 8
    np.testing.assert_array_equal(result.joint_positions, np.arange(7))

    bad_session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "joint_positions": [0, 1, 2, 3, 4, 5, 6, float("nan")],
                    "position_error_m": 0.001,
                    "orientation_error_rad": 0.01,
                },
            )
        ]
    )
    bad = PyRokiProvider(session=bad_session).solve_ik(pose, _state())  # type: ignore[arg-type]
    assert not bad.ok
    assert bad.error is not None
    assert bad.error.code is ErrorCode.IK_FAILED


def test_pyroki_rejects_solution_outside_pose_residual_limits() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "joint_positions": [0.0] * 8,
                    "position_error_m": 0.07,
                    "orientation_error_rad": 0.17,
                },
            )
        ]
    )

    state = _state()
    result = PyRokiProvider(session=session).solve_ik(state.end_effector_poses["primary"], state)

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.IK_FAILED
    assert result.error.details["position_error_m"] == pytest.approx(0.07)


def _planning_context(state: RobotState) -> RobotPlanningContext:
    return RobotPlanningContext(
        embodiment=state.embodiment,
        model="panda",
        joint_names=state.joint_names,
        base_transforms={arm: np.eye(4) for arm in state.arms},
        end_effector_links=dict.fromkeys(state.arms, "panda_hand"),
    )


def test_curobo_422_refusal_is_a_typed_failure_with_the_service_message() -> None:
    """A 422 "no planning result" is a planner refusal, not an invalid request.

    The generic HTTP client maps every 4xx to INVALID_REQUEST, which made twelve
    consecutive refused plans on fold_clothes read as API misuse. The cuRobo
    client re-wraps that status as the typed failure and surfaces the message.
    """
    state = _state()
    context = _planning_context(state)
    refusal = {"detail": {"message": "cuRobo returned no planning result"}}
    # The HTTP client keeps `response.text`, not the decoded JSON, in the error.
    body = json.dumps(refusal)
    session = FakeSession([FakeResponse(422, refusal, body), FakeResponse(422, refusal, body)])
    provider = CuRoboProvider(session=session)  # type: ignore[arg-type]
    pose = Pose([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], "robot_base")

    ik = provider.solve_ik_with_context(pose, state, context=context)
    scene = PlanningScene(PointCloud(points=np.array([[0.5, 0.0, 0.0]]), frame="robot_base"))
    plan = provider.plan_to_pose(state, pose, scene=scene, context=context)

    assert not ik.ok and ik.error is not None
    assert ik.error.code is ErrorCode.IK_FAILED
    assert ik.error.message == "cuRobo returned no planning result"
    assert ik.error.details["status_code"] == 422
    assert ik.error.details["http_error_code"] == "invalid_request"
    assert not plan.ok and plan.error is not None
    assert plan.error.code is ErrorCode.PLANNING_FAILED
    assert plan.error.message == "cuRobo returned no planning result"


def test_curobo_ik_uses_explicit_robot_context() -> None:
    state = _state()
    context = _planning_context(state)
    context = RobotPlanningContext(
        embodiment=context.embodiment,
        model=context.model,
        joint_names=context.joint_names,
        base_transforms={"primary": np.diag([1.0, -1.0, -1.0, 1.0])},
        end_effector_links=context.end_effector_links,
    )
    session = FakeSession([FakeResponse(200, {"joint_positions": np.ones(7).tolist()})])

    result = CuRoboProvider(session=session).solve_ik_with_context(  # type: ignore[arg-type]
        Pose([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], "robot_base"),
        state,
        context=context,
    )

    assert result.ok
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    np.testing.assert_array_equal(payload["base_transforms"]["primary"], np.diag([1, -1, -1, 1]))
    assert payload["target"] == {
        "type": "pose",
        "position": pytest.approx([0.4, 0.0, 0.2]),
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "frame": "robot_base",
    }


def test_curobo_single_arm_maps_embodiment_names_to_canonical_panda_joints() -> None:
    request = BaseRequest(
        base_frame="robot_base",
        model="panda",
        joint_positions={"primary": [0.0] * 7},
        joint_names={"primary": [f"robot0_joint{index}" for index in range(1, 8)]},
        base_transforms={"primary": np.eye(4).tolist()},
        end_effector_links={"primary": "panda_hand"},
    )

    assert _requested_indices(
        request,
        "primary",
        [
            *(f"panda_joint{index}" for index in range(1, 8)),
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
    ) == list(range(7))


def test_curobo_robot_point_filter_uses_segmenter_padding() -> None:
    spheres = np.array([[0.0, 0.0, 0.0, 0.1]])
    points = np.array(
        [
            [0.14, 0.0, 0.0],
            [0.151, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )

    filtered = _filter_robot_points(points, spheres, padding_m=0.05)

    np.testing.assert_allclose(filtered, points[1:])


class _CapturedMesh:
    def __init__(self, name: str, *, pose: list[float], vertices: object, faces: object) -> None:
        self.name = name
        self.pose = pose
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6.0
    )


def test_curobo_scene_mesh_preserves_declared_voxel_size() -> None:
    points = np.array([[0.5, 0.0, 0.2]])
    mesh = _voxelized_scene_mesh(points, voxel_size_m=0.01, mesh_type=_CapturedMesh)

    assert mesh.name == "public_rgbd_scene"
    extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    np.testing.assert_allclose(extents, 0.01)
    assert (mesh.vertices.min(axis=0) <= points[0]).all()
    assert (points[0] <= mesh.vertices.max(axis=0)).all()


def test_curobo_scene_mesh_winds_triangles_outward() -> None:
    """Upstream from_pointcloud winds triangles inward, inverting warp's
    signed-distance field (free space near surfaces reads as penetration).
    The service's patched copy must produce outward normals: one occupied
    voxel is a closed cube whose signed volume is +pitch**3.
    """
    mesh = _voxelized_scene_mesh(
        np.array([[0.0, 0.0, 0.0]]), voxel_size_m=0.01, mesh_type=_CapturedMesh
    )

    assert len(mesh.faces) == 12
    np.testing.assert_allclose(_signed_volume(mesh.vertices, mesh.faces), 0.01**3, rtol=1e-9)

    # multi-voxel bar: shared faces suppressed, volume = 2 voxels, still positive
    bar = _voxelized_scene_mesh(
        np.array([[0.0, 0.0, 0.0], [0.011, 0.0, 0.0]]),
        voxel_size_m=0.01,
        mesh_type=_CapturedMesh,
    )
    np.testing.assert_allclose(_signed_volume(bar.vertices, bar.faces), 2 * 0.01**3, rtol=1e-9)


def test_curobo_planner_purges_stale_scene_mesh_cache() -> None:
    from types import SimpleNamespace

    from cap_harness.providers.curobo.service import _purge_scene_mesh_cache

    # cuRobo's Warp cache is keyed by mesh name only, and every request names
    # its scene "public_rgbd_scene": without a purge, update_world silently
    # reuses the first request's geometry for the life of the cached planner.
    cache = {"public_rgbd_scene": object()}
    planner = SimpleNamespace(
        scene_collision_checker=SimpleNamespace(
            data=SimpleNamespace(meshes=SimpleNamespace(wp_cache=cache))
        )
    )
    _purge_scene_mesh_cache(planner)
    assert cache == {}

    # Planners without a mesh checker (or before first load) are a no-op.
    _purge_scene_mesh_cache(SimpleNamespace())
    _purge_scene_mesh_cache(SimpleNamespace(scene_collision_checker=SimpleNamespace(data=None)))


def test_curobo_plans_to_joint_goal_without_integrated_pose_ik() -> None:
    state = _state()
    scene = PlanningScene(PointCloud(points=np.array([[0.5, 0.0, 0.0]]), frame="robot_base"))
    source = np.linspace(np.zeros(7), np.ones(7), 5)
    session = FakeSession(
        [FakeResponse(200, {"joint_positions_base64": _numpy_b64(source), "dt_s": 0.025})]
    )

    result = CuRoboProvider(session=session).plan_to_joints(  # type: ignore[arg-type]
        state,
        np.ones(7),
        scene=scene,
        context=_planning_context(state),
    )

    assert result.ok and result.trajectory is not None
    assert result.trajectory.planner == "curobo_v2_cspace"
    assert result.diagnostics["planning_mode"] == "cspace"
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["target"] == {"type": "joints", "joint_positions": np.ones(7).tolist()}


def test_curobo_individual_bimanual_pose_goal_preserves_inactive_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    target = Target(
        type="pose",
        position=[0.5, 0.0, 0.3],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        frame="robot_base",
    )
    request = PlanRequest(
        base_frame="robot_base",
        model="dual_panda",
        joint_positions={"primary": [0.0] * 7, "secondary": [0.0] * 7},
        joint_names={
            "primary": [f"panda_joint{index}" for index in range(1, 8)],
            "secondary": [f"panda_joint{index}_2" for index in range(1, 8)],
        },
        base_transforms={
            "primary": np.eye(4).tolist(),
            "secondary": np.eye(4).tolist(),
        },
        end_effector_links={"primary": "panda_hand", "secondary": "panda_hand_2"},
        arm="primary",
        target=target,
    )
    active_pose = object()
    inactive_pose = object()
    current = object()
    planner = SimpleNamespace(
        tool_frames=["panda_hand", "panda_hand_2"],
        compute_kinematics=lambda value: SimpleNamespace(
            tool_poses=SimpleNamespace(
                to_dict=lambda: {"panda_hand": object(), "panda_hand_2": inactive_pose}
            )
        ),
    )
    monkeypatch.setattr(curobo_service, "_pose", lambda value: active_pose)

    poses, ordered = _ordered_goal_poses(
        request,
        {"primary": target},
        planner,
        current,
    )

    assert ordered == ["panda_hand", "panda_hand_2"]
    assert poses == {"panda_hand": active_pose, "panda_hand_2": inactive_pose}


def test_curobo_resamples_plan_to_control_period_and_keeps_endpoint() -> None:
    source = np.linspace(np.zeros(7), np.ones(7), 5)
    session = FakeSession(
        [FakeResponse(200, {"joint_positions_base64": _numpy_b64(source), "dt_s": 0.025})]
    )
    state = _state()
    scene = PlanningScene(PointCloud(points=np.array([[0.5, 0.0, 0.0]]), frame="robot_base"))
    target = Pose([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], "robot_base")

    result = CuRoboProvider(session=session).plan_to_pose(  # type: ignore[arg-type]
        state, target, scene=scene, context=_planning_context(state)
    )

    assert result.ok and result.trajectory is not None
    assert result.trajectory.dt_s == 0.05
    assert result.trajectory.collision_aware
    np.testing.assert_array_equal(result.trajectory.joint_positions[-1], np.ones(7))
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    np.testing.assert_array_equal(
        _decode_numpy_b64(payload["scene_points_base64"]), scene.point_cloud.points
    )


def test_curobo_time_dilation_stretches_plans_onto_control_grid() -> None:
    source = np.linspace(np.zeros(7), np.ones(7), 5)  # 0.1 s of plan at source dt
    session = FakeSession(
        [FakeResponse(200, {"joint_positions_base64": _numpy_b64(source), "dt_s": 0.025})]
    )
    state = _state()
    scene = PlanningScene(PointCloud(points=np.array([[0.5, 0.0, 0.0]]), frame="robot_base"))
    target = Pose([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], "robot_base")

    result = CuRoboProvider(session=session, time_dilation=0.4).plan_to_pose(  # type: ignore[arg-type]
        state, target, scene=scene, context=_planning_context(state)
    )

    assert result.ok and result.trajectory is not None
    # 0.1 s / 0.4 = 0.25 s of execution -> 6 waypoints on the 0.05 s grid.
    assert len(result.trajectory.joint_positions) == 6
    assert result.trajectory.dt_s == 0.05
    np.testing.assert_array_equal(result.trajectory.joint_positions[-1], np.ones(7))
    # The whole timeline stretches uniformly: per-step deltas shrink to 1/5,
    # rather than the original-speed plan followed by repeated final waypoints.
    steps = np.diff(result.trajectory.joint_positions[:, 0])
    np.testing.assert_allclose(steps, 0.2)
    assert dict(result.diagnostics)["time_dilation"] == 0.4


def test_curobo_per_call_timing_overrides_are_sent_and_applied() -> None:
    source = np.linspace(np.zeros(7), np.ones(7), 3)
    session = FakeSession(
        [FakeResponse(200, {"joint_positions_base64": _numpy_b64(source), "dt_s": 0.05})]
    )
    state = _state()
    scene = PlanningScene(PointCloud([[0.5, 0.0, 0.0]], frame="robot_base"))

    result = CuRoboProvider(session=session, control_period_s=0.025).plan_to_joints(  # type: ignore[arg-type]
        state,
        np.full(7, 0.1),
        scene=scene,
        context=_planning_context(state),
        time_dilation_factor=0.5,
        interpolation_dt_s=0.05,
        maximum_trajectory_dt_s=0.5,
    )

    assert result.ok and result.trajectory is not None
    assert result.trajectory.dt_s == 0.025
    payload = session.calls[0]["json"]
    assert payload["interpolation_dt_s"] == 0.05
    assert payload["maximum_trajectory_dt_s"] == 0.5
    assert result.diagnostics["time_dilation_factor"] == 0.5


def test_curobo_decodes_equal_length_synchronized_trajectories() -> None:
    pose = Pose([0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], "robot_base")
    state = RobotState(
        joint_positions={"primary": np.zeros(7), "secondary": np.zeros(7)},
        joint_velocities={"primary": np.zeros(7), "secondary": np.zeros(7)},
        end_effector_poses={"primary": pose, "secondary": pose},
        gripper_positions={"primary": 0.0, "secondary": 1.0},
        base_frame="robot_base",
        joint_names={
            "primary": tuple(f"panda_joint{i}" for i in range(1, 8)),
            "secondary": tuple(f"panda_joint{i}" for i in range(1, 8)),
        },
    )
    context = RobotPlanningContext(
        embodiment="robosuite",
        model="dual_panda",
        joint_names={
            "primary": state.joint_names["primary"],
            "secondary": tuple(f"panda_joint{i}_2" for i in range(1, 8)),
        },
        base_transforms={"primary": np.eye(4), "secondary": np.eye(4)},
        end_effector_links={"primary": "panda_hand", "secondary": "panda_hand_2"},
    )
    source = np.linspace(np.zeros(7), np.ones(7), 5)
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "joint_positions_base64": {
                        "primary": _numpy_b64(source),
                        "secondary": _numpy_b64(-source),
                    },
                    "dt_s": 0.025,
                },
            )
        ]
    )
    scene = PlanningScene(PointCloud(np.array([[1.0, 1.0, 1.0]]), "robot_base"))

    result = CuRoboProvider(session=session, time_dilation=1.0).plan_synchronized_motion(  # type: ignore[arg-type]
        state,
        {"primary": pose, "secondary": pose},
        scene=scene,
        context=context,
    )

    assert result.ok and result.trajectory is not None
    assert result.trajectory.waypoint_count == 3
    np.testing.assert_array_equal(result.trajectory.joint_positions["primary"][-1], np.ones(7))
    np.testing.assert_array_equal(result.trajectory.joint_positions["secondary"][-1], -np.ones(7))
    np.testing.assert_array_equal(result.trajectory.gripper_positions["primary"], 0.0)
    np.testing.assert_array_equal(result.trajectory.gripper_positions["secondary"], 1.0)


def test_dual_panda_builder_duplicates_active_chains(tmp_path) -> None:
    curobo_root = Path(__file__).resolve().parents[1] / "third_party/curobo"
    transform = np.eye(4)
    transform[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    transform[0, 3] = 1.0

    path = build_dual_panda_config(curobo_root, transform, tmp_path)
    payload = path.read_text(encoding="utf-8")
    urdf = path.with_name("dual_panda.urdf").read_text(encoding="utf-8")

    assert "panda_joint7_2" in payload
    assert "panda_hand_2" in payload
    assert 'child link="base_link_2"' in urdf
    assert 'name="world_base_link"' in urdf


def test_curobo_scene_drops_far_depth_outliers() -> None:
    """A few degenerate depth pixels hundreds of meters out must not survive
    into the world: the voxel extraction allocates a dense grid over the
    cloud bounding box (observed 646 TiB for an ~800 m outlier).
    """
    from cap_harness.providers.curobo.service import _drop_distant_points

    points = np.array([[0.5, 0.0, 0.4], [1.5, -1.0, 0.2], [812.0, 3.0, -40.0]])
    kept = _drop_distant_points(points)
    np.testing.assert_array_equal(kept, points[:2])


def test_curobo_robot_carve_accepts_the_request_object(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the carve once routed the (unhashable) pydantic request through an
    # lru_cache'd helper, which turned every point-cloud plan into an HTTP 500.
    class _Spheres:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def detach(self) -> _Spheres:
            return self

        def cpu(self) -> _Spheres:
            return self

        def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
            return np.asarray(self._values, dtype=dtype)

    class _Kinematics:
        robot_spheres = _Spheres(np.array([[0.0, 0.0, 0.0, 0.1]]))

    class _Robot:
        def compute_kinematics(self, state: object) -> _Kinematics:
            return _Kinematics()

    monkeypatch.setattr(curobo_service, "_state", lambda request, planner: None)
    request = BaseRequest(
        base_frame="robot_base",
        model="dual_panda",
        joint_positions={"primary": [0.0] * 7, "secondary": [0.0] * 7},
        joint_names={
            arm: [f"{arm}_joint{index}" for index in range(1, 8)]
            for arm in ("primary", "secondary")
        },
        base_transforms={"primary": np.eye(4).tolist(), "secondary": np.eye(4).tolist()},
        end_effector_links={"primary": "panda_hand", "secondary": "panda_hand"},
    )
    points = np.array([[0.14, 0.0, 0.0], [0.5, 0.0, 0.0]])

    carved = curobo_service._carve_robot_points(request, _Robot(), points)

    np.testing.assert_allclose(carved, points[1:])
