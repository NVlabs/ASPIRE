from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import sqrt

import numpy as np
import pytest

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    ExecutionStatus,
    GraspCandidate,
    GraspSet,
    IKResult,
    LocalizationResult,
    MotionStrategy,
    ObjectGeometry,
    Observation,
    PlanResult,
    PointCloud,
    Pose,
    RobotAction,
    RobotState,
    Segmentation,
    SegmentationSet,
    StepResult,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.protocols import EnvironmentAdapter, SegmentationProvider

JOINT_NAMES = tuple(f"panda_joint_{index}" for index in range(1, 8))


def _pose(*, frame: str = "base", quaternion: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)) -> Pose:
    return Pose(
        position=np.array([0.1, -0.2, 0.3]),
        quaternion_wxyz=np.asarray(quaternion),
        frame=frame,
    )


def _robot_state() -> RobotState:
    return RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=_pose(),
        gripper_positions=0.5,
        base_frame="base",
        joint_names=JOINT_NAMES,
        timestamp_s=1.0,
    )


def _camera() -> CameraObservation:
    return CameraObservation(
        rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        depth_m=np.ones((4, 5), dtype=np.float32),
        intrinsics=np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]]),
        frame="agent_camera",
        camera_pose=_pose(frame="base"),
        timestamp_s=1.0,
    )


def _task_context() -> TaskContext:
    return TaskContext(
        suite="libero_goal_task",
        task_id=3,
        task_name="open_drawer",
        language="open the top drawer",
        family="goal",
    )


def _observation() -> Observation:
    return Observation(
        cameras={"agent": _camera()},
        robot_state=_robot_state(),
        task_context=_task_context(),
        timestamp_s=1.0,
    )


def _trajectory() -> Trajectory:
    return Trajectory(
        joint_positions=np.zeros((3, 7)),
        dt_s=0.05,
        joint_names=JOINT_NAMES,
        planner="pyroki_interpolation",
        collision_aware=False,
        expected_start=_robot_state(),
    )


def test_pose_uses_wxyz_and_builds_expected_rotation_matrix() -> None:
    pose = _pose(quaternion=(sqrt(0.5), 0.0, 0.0, sqrt(0.5)))

    np.testing.assert_allclose(
        pose.as_matrix(),
        np.array(
            [
                [0.0, -1.0, 0.0, 0.1],
                [1.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        atol=1e-12,
    )
    assert not pose.position.flags.writeable
    with pytest.raises(FrozenInstanceError):
        pose.frame = "world"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("position", "quaternion", "frame"),
    [
        (np.zeros(2), (1.0, 0.0, 0.0, 0.0), "base"),
        (np.zeros(3), (2.0, 0.0, 0.0, 0.0), "base"),
        (np.array([np.nan, 0.0, 0.0]), (1.0, 0.0, 0.0, 0.0), "base"),
        (np.zeros(3), (1.0, 0.0, 0.0, 0.0), ""),
    ],
)
def test_pose_rejects_invalid_shape_normalization_and_frame(
    position: np.ndarray,
    quaternion: tuple[float, ...],
    frame: str,
) -> None:
    with pytest.raises(ValueError):
        Pose(position=position, quaternion_wxyz=np.asarray(quaternion), frame=frame)


def test_camera_observation_validates_rgbd_calibration() -> None:
    camera = _camera()
    assert camera.rgb.shape == (4, 5, 3)
    assert camera.depth_m.shape == (4, 5)

    with pytest.raises(ValueError, match="depth_m"):
        CameraObservation(
            rgb=camera.rgb,
            depth_m=np.ones((5, 4)),
            intrinsics=camera.intrinsics,
            frame=camera.frame,
            camera_pose=camera.camera_pose,
        )
    with pytest.raises(ValueError, match="floating rgb"):
        CameraObservation(
            rgb=np.full((4, 5, 3), 1.1),
            depth_m=camera.depth_m,
            intrinsics=camera.intrinsics,
            frame=camera.frame,
            camera_pose=camera.camera_pose,
        )


def test_robot_state_enforces_arm_sets_joint_shapes_and_pose_frame() -> None:
    state = _robot_state()
    assert state.arms == ("primary",)
    np.testing.assert_array_equal(state.joint_positions["primary"], np.zeros(7))

    with pytest.raises(ValueError, match="identical arm names"):
        RobotState(
            joint_positions={"left": np.zeros(7), "right": np.zeros(7)},
            joint_velocities={"left": np.zeros(7)},
            end_effector_poses={"left": _pose(), "right": _pose()},
            gripper_positions={"left": 0.5, "right": 0.5},
            base_frame="base",
        )
    with pytest.raises(ValueError, match="expected 'base'"):
        RobotState(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            end_effector_poses=_pose(frame="world"),
            gripper_positions=0.5,
            base_frame="base",
        )


def test_arm_commands_and_actions_validate_without_requiring_primary() -> None:
    command = ArmCommand(
        mode="joint_position",
        target=np.arange(7, dtype=np.float64),
        gripper_position=0.25,
    )
    action = RobotAction(arms={"secondary": command})

    assert tuple(action.arms) == ("secondary",)
    with pytest.raises(ValueError, match="mode"):
        ArmCommand(mode="delta", target=np.zeros(7))
    with pytest.raises(ValueError, match="shape"):
        ArmCommand(mode="joint_position", target=np.zeros(6))
    with pytest.raises(ValueError, match="shape"):
        ArmCommand(mode="joint_position", target=np.zeros(5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ArmCommand(mode="joint_position", target=np.zeros(7), gripper_position=1.1)


def test_motion_strategy_validates_optional_curobo_timing() -> None:
    strategy = MotionStrategy(
        pose_planner="curobo-integrated",
        time_dilation_factor=0.5,
        interpolation_dt_s=0.05,
        maximum_trajectory_dt_s=0.5,
    )

    assert strategy.time_dilation_factor == 0.5
    assert strategy.interpolation_dt_s == 0.05
    assert strategy.maximum_trajectory_dt_s == 0.5
    with pytest.raises(ValueError, match="time_dilation_factor"):
        MotionStrategy(time_dilation_factor=0.0)
    with pytest.raises(ValueError, match="interpolation_dt_s"):
        MotionStrategy(interpolation_dt_s=float("nan"))


def test_trajectory_validates_waypoints_timing_joint_names_and_metadata() -> None:
    trajectory = _trajectory()
    assert trajectory.waypoints.shape == (3, 7)
    assert trajectory.dt_s == pytest.approx(0.05)
    assert trajectory.planner == "pyroki_interpolation"
    assert trajectory.collision_aware is False

    with pytest.raises(ValueError, match="dt_s"):
        Trajectory(np.zeros((3, 7)), 0.0, JOINT_NAMES, "planner", False, _robot_state())
    with pytest.raises(ValueError, match="exactly 7"):
        Trajectory(np.zeros((3, 7)), 0.05, JOINT_NAMES[:6], "planner", False, _robot_state())
    with pytest.raises(ValueError, match="unique"):
        Trajectory(np.zeros((3, 7)), 0.05, ("same",) * 7, "planner", False, _robot_state())


def test_robot_state_rejects_mixed_joint_dimensions_for_one_embodiment() -> None:
    pose = _pose()
    with pytest.raises(ValueError, match="shape"):
        RobotState(
            joint_positions={"primary": np.zeros(7), "secondary": np.zeros(6)},
            joint_velocities={"primary": np.zeros(7), "secondary": np.zeros(6)},
            end_effector_poses={"primary": pose, "secondary": pose},
            gripper_positions={"primary": 0.5, "secondary": 0.5},
            base_frame="base",
            joint_names={"primary": JOINT_NAMES, "secondary": JOINT_NAMES[:6]},
        )


def test_perception_geometry_and_grasp_models_enforce_shapes_and_frames() -> None:
    segmentation = Segmentation(
        mask=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        label="red mug",
        score=0.9,
        camera_name="agent",
        frame="agent_camera",
        box_xyxy=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    segmentations = SegmentationSet(ok=True, segmentations=[segmentation])
    cloud = PointCloud(points=np.ones((4, 3)), frame="base")
    geometry = ObjectGeometry(pose=_pose(), extents=np.array([0.1, 0.2, 0.3]), point_count=4)
    grasp = GraspCandidate(pose=_pose(), score=0.8, width_m=0.04)
    grasps = GraspSet(ok=True, grasps=[grasp])
    localization = LocalizationResult(
        ok=True,
        geometry=geometry,
        segmentation=segmentation,
        point_cloud=cloud,
    )

    assert segmentations.value == (segmentation,)
    np.testing.assert_array_equal(segmentation.box_xyxy, [0.0, 0.0, 1.0, 1.0])
    assert segmentation.box_xyxy is not None
    assert not segmentation.box_xyxy.flags.writeable
    assert grasps.value == (grasp,)
    assert localization.value is geometry
    assert grasp.transform.shape == (4, 4)
    with pytest.raises(ValueError, match="same frame"):
        GraspSet(
            ok=True,
            grasps=[grasp, GraspCandidate(_pose(frame="world"), 0.7, 0.03)],
        )
    with pytest.raises(ValueError, match="strictly positive"):
        ObjectGeometry(pose=_pose(), extents=np.array([0.1, 0.0, 0.3]))


def test_recoverable_results_require_consistent_ok_payload_and_error() -> None:
    error = ApiError(
        code=ErrorCode.IK_UNREACHABLE,
        message="target is outside the workspace",
        details={"iterations": 50},
    )
    failed = IKResult(ok=False, error=error)
    succeeded = IKResult(ok=True, joint_positions=np.zeros(7))
    planned = PlanResult(ok=True, trajectory=_trajectory())

    assert failed.value is None
    np.testing.assert_array_equal(succeeded.value, np.zeros(7))
    assert planned.value is planned.trajectory
    with pytest.raises(ValueError, match="must carry an ApiError"):
        IKResult(ok=False)
    with pytest.raises(ValueError, match="cannot carry an error"):
        IKResult(ok=True, joint_positions=np.zeros(7), error=error)
    with pytest.raises(ValueError, match="cannot carry joint_positions"):
        IKResult(ok=False, joint_positions=np.zeros(7), error=error)


def test_api_error_is_structured_and_raiseable() -> None:
    error = ApiError(ErrorCode.PROVIDER_ERROR, "bad response", details={"status": 502})

    with pytest.raises(ApiError, match="provider_error: bad response") as caught:
        raise error
    assert caught.value.code is ErrorCode.PROVIDER_ERROR
    assert caught.value.details == {"status": 502}


def test_step_public_view_removes_reward_and_sensitive_diagnostics_recursively() -> None:
    result = StepResult(
        ok=True,
        observation=_observation(),
        reward=1.0,
        diagnostics={
            "provider": "fake",
            "reward": 1.0,
            "raw_state": [1, 2, 3],
            "nested": {
                "latency_ms": 4.5,
                "ground_truth_pose": [0.0, 0.0, 0.0],
                "success": True,
            },
        },
    )

    public = result.public_view()

    assert result.reward == 1.0
    assert public.reward is None
    assert public.diagnostics["provider"] == "fake"
    assert "reward" not in public.diagnostics
    assert "raw_state" not in public.diagnostics
    assert public.diagnostics["nested"] == {"latency_ms": 4.5}


def test_execution_result_validation() -> None:
    execution = ExecutionResult(ok=True, steps_executed=3, final_observation=_observation())

    assert execution.steps_executed == 3
    assert execution.status is ExecutionStatus.SUCCESS
    with pytest.raises(ValueError, match="non-negative"):
        ExecutionResult(ok=True, steps_executed=-1)


@pytest.mark.parametrize(
    ("code", "status"),
    (
        (ErrorCode.TIMEOUT, ExecutionStatus.TIMEOUT),
        (ErrorCode.CANCELLED, ExecutionStatus.CANCELLED),
        (ErrorCode.STALE_PLAN, ExecutionStatus.STALE_PLAN),
        (ErrorCode.PLANNING_FAILED, ExecutionStatus.PLANNER_FAILED),
        (ErrorCode.CONTROLLER_FAILED, ExecutionStatus.CONTROLLER_FAILED),
    ),
)
def test_execution_status_maps_typed_failures_and_final_arm_errors(
    code: ErrorCode,
    status: ExecutionStatus,
) -> None:
    result = ExecutionResult(
        ok=False,
        steps_executed=2,
        error=ApiError(code, "motion failed"),
        final_errors={"primary": 0.01, "secondary": 0.02},
    )

    assert result.status is status
    assert result.final_errors == {"primary": 0.01, "secondary": 0.02}


@pytest.mark.parametrize(
    ("terminated", "truncated", "code", "status"),
    (
        (True, False, ErrorCode.TERMINATED, ExecutionStatus.TERMINATED),
        (False, True, ErrorCode.TRUNCATED, ExecutionStatus.TRUNCATED),
    ),
)
def test_terminal_execution_is_never_success(
    terminated: bool,
    truncated: bool,
    code: ErrorCode,
    status: ExecutionStatus,
) -> None:
    result = ExecutionResult(
        ok=False,
        steps_executed=1,
        terminated=terminated,
        truncated=truncated,
        error=ApiError(code, "episode ended"),
    )

    assert result.status is status
    with pytest.raises(ValueError, match="cannot be successful"):
        ExecutionResult(
            ok=True,
            steps_executed=1,
            terminated=terminated,
            truncated=truncated,
        )


def test_provider_and_adapter_protocols_are_runtime_checkable() -> None:
    class FakeSegmenter:
        def segment_text(self, observation, query):
            del observation, query

        def segment_points(self, observation, points_px, *, point_labels=None):
            del observation, points_px, point_labels

    class IncompleteAdapter:
        embodiment = "fake"

        def step(self, action):
            del action

    assert isinstance(FakeSegmenter(), SegmentationProvider)
    assert not isinstance(IncompleteAdapter(), EnvironmentAdapter)


def test_motion_strategy_is_typed_and_rejects_unknown_backends() -> None:
    strategy = MotionStrategy(ik_solver="curobo", trajectory_planner="curobo")

    assert strategy.pose_planner == "composed"
    with pytest.raises(ValueError, match="ik_solver"):
        MotionStrategy(ik_solver="unknown")
    with pytest.raises(ValueError, match="trajectory_planner"):
        MotionStrategy(trajectory_planner="unknown")
    with pytest.raises(ValueError, match="pose_planner"):
        MotionStrategy(pose_planner="unknown")
