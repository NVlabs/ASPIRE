from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.api import CapApi
from cap_harness.contracts import (
    ArmCommand,
    ExecutionResult,
    ExecutionStatus,
    GraspCandidate,
    GraspSet,
    IKResult,
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
    SynchronizedPlanResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode


def _pose() -> Pose:
    return Pose(
        position=np.zeros(3),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="base",
    )


def _state() -> RobotState:
    return RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=_pose(),
        gripper_positions=0.4,
        base_frame="base",
    )


class FakeAdapter:
    control_frequency = 20.0
    control_period_s = 0.05
    action_bounds = (np.full(8, -1.0), np.full(8, 1.0))

    def __init__(self) -> None:
        self.state = _state()
        self.executed: list[object] = []
        self.close_calls = 0

    def get_robot_state(self) -> RobotState:
        return self.state

    def get_task_context(self) -> TaskContext:
        return TaskContext("suite", 0, "task", "do task", "object")

    def get_observation(self) -> Observation:
        from cap_harness.contracts import CameraObservation

        camera = CameraObservation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_m=np.ones((2, 2)),
            intrinsics=np.eye(3),
            frame="camera",
            camera_pose=_pose(),
        )
        return Observation(cameras={"agentview": camera}, robot_state=self.state)

    def step(self, action: object) -> StepResult:
        self.executed.append(action)
        if isinstance(action, RobotAction):
            positions = dict(self.state.joint_positions)
            grippers = dict(self.state.gripper_positions)
            for arm, command in action.arms.items():
                positions[arm] = command.target
                if command.gripper_position is not None:
                    grippers[arm] = command.gripper_position
            self.state = RobotState(
                joint_positions=positions,
                joint_velocities={arm: np.zeros_like(value) for arm, value in positions.items()},
                end_effector_poses=self.state.end_effector_poses,
                gripper_positions=grippers,
                base_frame=self.state.base_frame,
                joint_names=self.state.joint_names,
            )
        return StepResult(
            ok=True,
            observation=self.get_observation(),
            reward=3.0,
            diagnostics={"success": True, "safe": "visible", "nested": {"reward": 2.0}},
        )

    def execute_trajectory(self, trajectory: object) -> ExecutionResult:
        self.executed.append(trajectory)
        raw_positions = trajectory.joint_positions  # type: ignore[attr-defined]
        target = (
            {arm: positions[-1] for arm, positions in raw_positions.items()}
            if isinstance(raw_positions, Mapping)
            else raw_positions[-1]
        )
        self.state = RobotState(
            joint_positions=target,
            joint_velocities=(
                {arm: np.zeros_like(value) for arm, value in target.items()}
                if isinstance(target, Mapping)
                else np.zeros_like(target)
            ),
            end_effector_poses=self.state.end_effector_poses,
            gripper_positions=self.state.gripper_positions,
            base_frame=self.state.base_frame,
            joint_names=self.state.joint_names,
        )
        return ExecutionResult(
            ok=True,
            steps_executed=(
                len(next(iter(raw_positions.values())))
                if isinstance(raw_positions, Mapping)
                else len(raw_positions)
            ),
            final_observation=self.get_observation(),
            diagnostics={"success": True},
        )

    def close_gripper(self, *, arm: str = "primary") -> ExecutionResult:
        assert arm == "primary"
        self.close_calls += 1
        return ExecutionResult(
            ok=True, steps_executed=5, diagnostics={"completion": "contact_stall"}
        )


class FakeIK:
    def __init__(self, goal: np.ndarray) -> None:
        self.goal = goal
        self.targets: list[Pose] = []

    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary"
    ) -> IKResult:
        del robot_state
        self.targets.append(target_pose)
        return IKResult(ok=True, joint_positions=self.goal, arm=arm)


class FakeGraspProvider:
    def __init__(self) -> None:
        self.point_cloud: PointCloud | None = None
        self.scene_point_cloud: PointCloud | None = None
        self.max_candidates: int | None = None

    def generate_grasps(
        self,
        point_cloud,
        *,
        scene_point_cloud=None,
        geometry=None,
        max_candidates=None,
    ):
        del geometry
        self.point_cloud = point_cloud
        self.scene_point_cloud = scene_point_cloud
        self.max_candidates = max_candidates
        return GraspSet(
            ok=True,
            grasps=(GraspCandidate(pose=_pose(), score=0.9, width_m=0.04),),
        )


class FakeMotionProvider:
    def __init__(self) -> None:
        self.scene = None
        self.context = None
        self.target = None

    def plan_to_joints(
        self,
        robot_state,
        target_joints,
        *,
        arm="primary",
        gripper_position=None,
        scene=None,
        context=None,
    ):
        self.scene, self.context = scene, context
        self.target = target_joints
        goal = robot_state.joint_positions[arm] + 0.01
        return PlanResult(
            ok=True,
            trajectory=Trajectory(
                joint_positions=goal[None],
                dt_s=0.05,
                joint_names=robot_state.joint_names[arm],
                planner="fake_collision_planner",
                collision_aware=True,
                expected_start=robot_state,
                arm=arm,
                gripper_positions=np.array([gripper_position]),
            ),
        )


class FakeSynchronizedProvider:
    def __init__(self) -> None:
        self.targets = None
        self.gripper_positions = None

    def plan_synchronized_motion(
        self,
        robot_state,
        targets,
        *,
        gripper_positions=None,
        scene=None,
        context=None,
    ):
        del scene, context
        self.targets = targets
        self.gripper_positions = gripper_positions
        return SynchronizedPlanResult(
            ok=True,
            trajectory=SynchronizedTrajectory(
                joint_positions={arm: np.asarray(target)[None] for arm, target in targets.items()},
                dt_s=0.05,
                joint_names=robot_state.joint_names,
                planner="fake_synchronized_planner",
                collision_aware=True,
                expected_start=robot_state,
                gripper_positions={
                    arm: np.array([gripper_positions[arm]]) for arm in robot_state.arms
                },
            ),
        )


def _dual_adapter() -> FakeAdapter:
    adapter = FakeAdapter()
    pose = _pose()
    adapter.state = RobotState(
        joint_positions={"primary": np.zeros(7), "secondary": np.zeros(7)},
        joint_velocities={"primary": np.zeros(7), "secondary": np.zeros(7)},
        end_effector_poses={"primary": pose, "secondary": pose},
        gripper_positions={"primary": 0.4, "secondary": 0.6},
        base_frame="base",
        joint_names={
            "primary": tuple(f"p{i}" for i in range(7)),
            "secondary": tuple(f"s{i}" for i in range(7)),
        },
    )
    adapter.embodiment = "test"  # type: ignore[attr-defined]
    return adapter


def test_plan_motion_uses_controller_delta_and_exact_endpoint() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    goal = np.array([0.11, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = api.plan_motion(goal)

    assert result.ok
    trajectory = result.trajectory
    assert trajectory is not None
    assert trajectory.joint_positions.shape == (3, 7)
    np.testing.assert_array_equal(trajectory.joint_positions[-1], goal)
    increments = np.diff(np.vstack((np.zeros(7), trajectory.joint_positions)), axis=0)
    assert np.max(np.abs(increments)) <= 0.05 + 1e-12
    assert trajectory.collision_aware is False
    np.testing.assert_allclose(trajectory.gripper_positions, 0.4)


def test_synchronized_planning_validates_complete_arm_set_before_provider_lookup() -> None:
    api = CapApi(_dual_adapter())  # type: ignore[arg-type]

    result = api.plan_synchronized_motion({"primary": np.zeros(7)})

    assert not result.ok
    assert result.error is not None and result.error.code == ErrorCode.INVALID_REQUEST
    assert result.error.details["expected_arms"] == ("primary", "secondary")


def test_synchronized_planning_rejects_malformed_joint_target() -> None:
    api = CapApi(_dual_adapter())  # type: ignore[arg-type]

    result = api.plan_synchronized_motion({"primary": np.zeros(6), "secondary": np.zeros(7)})

    assert not result.ok
    assert result.error is not None and result.error.code == ErrorCode.INVALID_REQUEST
    assert "primary" in result.error.message


def test_synchronized_planning_copies_and_freezes_joint_targets() -> None:
    adapter = _dual_adapter()
    provider = FakeSynchronizedProvider()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        integrated_pose_planning_providers={"curobo-integrated": provider},
    )
    primary = np.full(7, 0.1)
    secondary = np.full(7, -0.1)

    result = api.plan_synchronized_motion({"secondary": secondary, "primary": primary})
    primary[:] = 0.9
    secondary[:] = -0.9

    assert result.ok
    assert provider.targets is not None
    assert tuple(provider.targets) == ("primary", "secondary")
    assert not provider.targets["primary"].flags.writeable
    np.testing.assert_array_equal(provider.targets["primary"], 0.1)
    np.testing.assert_array_equal(provider.targets["secondary"], -0.1)


def test_synchronized_motion_uses_one_coupled_plan_and_reports_both_arm_errors() -> None:
    adapter = _dual_adapter()
    provider = FakeSynchronizedProvider()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        integrated_pose_planning_providers={"curobo-integrated": provider},
    )

    result = api.move_synchronized({"primary": np.full(7, 0.1), "secondary": np.full(7, -0.1)})

    assert result.ok and result.final_errors == {"primary": 0.0, "secondary": 0.0}
    assert provider.gripper_positions == {"primary": 0.4, "secondary": 0.6}
    assert len(adapter.executed) == 1
    assert isinstance(adapter.executed[0], SynchronizedTrajectory)


def test_synchronized_planner_failure_never_falls_back_or_executes() -> None:
    adapter = _dual_adapter()

    class FailingCoupledProvider:
        def plan_synchronized_motion(self, *args, **kwargs):
            del args, kwargs
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(ErrorCode.PLANNING_FAILED, "coupled planning failed"),
            )

    api = CapApi(  # type: ignore[arg-type]
        adapter,
        integrated_pose_planning_providers={
            "curobo-integrated": FailingCoupledProvider(),
        },
    )

    result = api.move_synchronized({"primary": np.full(7, 0.1), "secondary": np.full(7, -0.1)})

    assert not result.ok and result.status is ExecutionStatus.PLANNER_FAILED
    assert result.steps_executed == 0
    assert adapter.executed == []


def test_same_tick_grippers_use_one_action_for_both_arms() -> None:
    adapter = _dual_adapter()
    api = CapApi(adapter)  # type: ignore[arg-type]

    result = api.set_grippers({"primary": 0.0, "secondary": 1.0})

    assert result.ok and result.steps_executed == 1
    assert len(adapter.executed) == 1
    action = adapter.executed[0]
    assert isinstance(action, RobotAction)
    assert tuple(sorted(action.arms)) == ("primary", "secondary")
    assert action.arms["primary"].gripper_position == 0.0
    assert action.arms["secondary"].gripper_position == 1.0


def test_full_start_binding_rejects_inactive_arm_drift_before_execution() -> None:
    adapter = _dual_adapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    start = adapter.state
    trajectory = Trajectory(
        joint_positions=np.full((1, 7), 0.1),
        dt_s=0.05,
        joint_names=start.joint_names["primary"],
        planner="test",
        collision_aware=True,
        expected_start=start,
    )
    adapter.state = RobotState(
        joint_positions={"primary": np.zeros(7), "secondary": np.full(7, 0.1)},
        joint_velocities=start.joint_velocities,
        end_effector_poses=start.end_effector_poses,
        gripper_positions=start.gripper_positions,
        base_frame=start.base_frame,
        joint_names=start.joint_names,
    )

    result = api.execute_trajectory(trajectory)

    assert not result.ok and result.status.value == "stale_plan"
    assert result.steps_executed == 0
    assert adapter.executed == []


def test_full_start_binding_rejects_gripper_drift_before_execution() -> None:
    adapter = _dual_adapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    start = adapter.state
    trajectory = Trajectory(
        joint_positions=np.full((1, 7), 0.1),
        dt_s=0.05,
        joint_names=start.joint_names["primary"],
        planner="test",
        collision_aware=True,
        expected_start=start,
    )
    adapter.state = RobotState(
        joint_positions=start.joint_positions,
        joint_velocities=start.joint_velocities,
        end_effector_poses=start.end_effector_poses,
        gripper_positions={"primary": 0.4, "secondary": 0.9},
        base_frame=start.base_frame,
        joint_names=start.joint_names,
    )

    result = api.execute_trajectory(trajectory)

    assert not result.ok and result.error.code == ErrorCode.STALE_PLAN
    assert result.steps_executed == 0
    assert adapter.executed == []


def test_planner_cannot_rebind_a_trajectory_from_the_wrong_start() -> None:
    adapter = FakeAdapter()

    class WrongStartProvider(FakeMotionProvider):
        def plan_to_joints(self, robot_state, target_joints, **kwargs):
            wrong = RobotState(
                joint_positions=np.full(7, 0.1),
                joint_velocities=np.zeros(7),
                end_effector_poses=robot_state.end_effector_poses["primary"],
                gripper_positions=robot_state.gripper_positions["primary"],
                base_frame=robot_state.base_frame,
                joint_names=robot_state.joint_names["primary"],
            )
            return PlanResult(
                ok=True,
                trajectory=Trajectory(
                    joint_positions=np.full((1, 7), 0.2),
                    dt_s=0.05,
                    joint_names=robot_state.joint_names["primary"],
                    planner="wrong-start",
                    collision_aware=True,
                    expected_start=wrong,
                ),
            )

    api = CapApi(  # type: ignore[arg-type]
        adapter,
        trajectory_planning_providers={"curobo": WrongStartProvider()},
    )
    result = api.plan_motion(np.full(7, 0.2), strategy=MotionStrategy(trajectory_planner="curobo"))

    assert not result.ok
    assert result.error is not None and result.error.code == ErrorCode.PLANNING_FAILED


def test_external_planner_receives_public_rgbd_scene_and_robot_context() -> None:
    adapter = FakeAdapter()
    adapter.embodiment = "robosuite"  # type: ignore[attr-defined]
    planner = FakeMotionProvider()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        trajectory_planning_providers={"curobo": planner},
    )

    result = api.plan_motion(
        np.full(7, 0.1),
        strategy=MotionStrategy(trajectory_planner="curobo"),
    )

    assert result.ok and result.trajectory is not None
    assert result.trajectory.collision_aware
    assert planner.scene.frame == "base"
    assert len(planner.scene.point_cloud.points) > 0
    assert planner.context.model == "panda"


def test_pose_motion_composes_selected_ik_and_trajectory_backends() -> None:
    adapter = FakeAdapter()
    adapter.embodiment = "robosuite"  # type: ignore[attr-defined]
    goal = np.full(7, 0.2)
    target = _pose()
    ik = FakeIK(goal)
    planner = FakeMotionProvider()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        ik_providers={"pyroki": ik},
        trajectory_planning_providers={"curobo": planner},
    )

    result = api.plan_motion(
        target,
        strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="curobo"),
    )

    assert result.ok
    assert len(ik.targets) == 1 and ik.targets[0] is target
    np.testing.assert_array_equal(planner.target, goal)


def test_move_to_pose_never_upgrades_execution_timeout_to_success() -> None:
    adapter = FakeAdapter()

    def timeout(_trajectory):
        return ExecutionResult(
            ok=False,
            steps_executed=1,
            final_observation=adapter.get_observation(),
            error=ApiError(ErrorCode.TIMEOUT, "controller timed out"),
        )

    adapter.execute_trajectory = timeout  # type: ignore[method-assign]
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        ik_providers={"pyroki": FakeIK(np.zeros(7))},
    )

    result = api.move_to_pose(_pose())

    assert not result.ok
    assert result.status.value == "timeout"


def test_integrated_pose_planning_applies_primary_adapter_ik_transform() -> None:
    adapter = FakeAdapter()
    adapter.embodiment = "robosuite"  # type: ignore[attr-defined]
    target = _pose()
    converted = Pose(
        position=np.array([0.0, 0.0, -0.1]),
        quaternion_wxyz=np.array([2**-0.5, 0.0, 0.0, 2**-0.5]),
        frame="base",
    )
    calls = []

    def ik_request(target_pose, robot_state, arm):
        calls.append((target_pose, robot_state, arm))
        return converted, robot_state

    adapter.ik_request = ik_request  # type: ignore[attr-defined]

    class IntegratedProvider:
        def __init__(self):
            self.target = None
            self.state = None

        def plan_to_pose(self, robot_state, target_pose, **_kwargs):
            self.state = robot_state
            self.target = target_pose
            return PlanResult(
                ok=True,
                trajectory=Trajectory(
                    joint_positions=robot_state.joint_positions["primary"][None],
                    dt_s=0.05,
                    joint_names=robot_state.joint_names["primary"],
                    planner="fake_integrated_planner",
                    collision_aware=True,
                    expected_start=robot_state,
                ),
            )

    provider = IntegratedProvider()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        integrated_pose_planning_providers={"curobo-integrated": provider},
    )

    result = api.plan_motion(
        target,
        strategy=MotionStrategy(pose_planner="curobo-integrated"),
    )

    assert result.ok
    assert len(calls) == 1 and calls[0][0] is target and calls[0][2] == "primary"
    assert provider.target is converted
    assert provider.state is calls[0][1]


def test_integrated_pose_planning_excludes_only_the_observed_target_cloud() -> None:
    adapter = FakeAdapter()

    class IntegratedProvider:
        def __init__(self):
            self.scene = None

        def plan_to_pose(self, robot_state, target_pose, *, scene=None, **_kwargs):
            del target_pose
            self.scene = scene
            return PlanResult(
                ok=True,
                trajectory=Trajectory(
                    joint_positions=robot_state.joint_positions["primary"][None],
                    dt_s=0.05,
                    joint_names=robot_state.joint_names["primary"],
                    planner="fake_integrated_planner",
                    collision_aware=True,
                    expected_start=robot_state,
                ),
            )

    provider = IntegratedProvider()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        integrated_pose_planning_providers={"curobo-integrated": provider},
    )
    target_cloud = PointCloud(points=np.array([[0.0, 0.0, 1.0]]), frame="base")

    result = api.plan_motion(
        _pose(),
        strategy=MotionStrategy(pose_planner="curobo-integrated"),
        target_point_cloud=target_cloud,
    )

    assert result.ok
    assert provider.scene is not None
    scene_points = provider.scene.point_cloud.points
    assert not np.any(np.all(np.isclose(scene_points, [0.0, 0.0, 1.0]), axis=1))
    assert np.any(np.all(np.isclose(scene_points, [1.0, 1.0, 1.0]), axis=1))


def test_motion_target_cloud_must_use_robot_base_frame() -> None:
    api = CapApi(FakeAdapter())  # type: ignore[arg-type]
    target_cloud = PointCloud(points=np.array([[0.0, 0.0, 1.0]]), frame="camera")

    result = api.plan_motion(_pose(), target_point_cloud=target_cloud)

    assert not result.ok
    assert result.error is not None and result.error.code == ErrorCode.INVALID_REQUEST


def test_one_api_can_use_multiple_ik_strategies() -> None:
    pyroki = FakeIK(np.full(7, 0.1))
    curobo = FakeIK(np.full(7, 0.2))
    api = CapApi(  # type: ignore[arg-type]
        FakeAdapter(),
        ik_providers={"pyroki": pyroki, "curobo": curobo},
    )
    target = _pose()

    first = api.plan_motion(target)
    second = api.plan_motion(target, strategy=MotionStrategy(ik_solver="curobo"))

    assert first.ok and first.trajectory is not None
    assert second.ok and second.trajectory is not None
    np.testing.assert_array_equal(first.trajectory.joint_positions[-1], np.full(7, 0.1))
    np.testing.assert_array_equal(second.trajectory.joint_positions[-1], np.full(7, 0.2))
    assert len(pyroki.targets) == 1
    assert len(curobo.targets) == 1


def test_high_level_close_uses_adapter_contact_aware_completion() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]

    result = api.close_gripper()

    assert result.ok
    assert result.steps_executed == 5
    assert result.diagnostics["completion"] == "contact_stall"
    assert adapter.close_calls == 1


def test_arm_motion_maintains_last_commanded_gripper_target() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]

    assert api.close_gripper().ok
    assert api.move_to_joints(np.full(7, 0.1)).ok

    action = adapter.executed[-1]
    assert isinstance(action, RobotAction)
    assert action.arms["primary"].gripper_position == 0.0


def test_gripper_target_is_latched_when_the_jaw_stalls_on_the_object() -> None:
    """A successful grasp stalls short of 0.0 and is reported as a failure.

    The latch records what was commanded, so it must still update; otherwise the next
    arm command inherits the 1.0 from the pre-grasp open and drops the object.
    """

    class StallingAdapter(FakeAdapter):
        def close_gripper(self, *, arm: str = "primary") -> ExecutionResult:
            self.close_calls += 1
            return ExecutionResult(
                ok=False,
                steps_executed=5,
                error=ApiError(
                    code=ErrorCode.CONTROLLER_FAILED,
                    message="gripper did not reach the commanded width",
                ),
                diagnostics={"completion": "contact_stall"},
            )

    adapter = StallingAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]

    close = api.close_gripper()
    assert not close.ok  # the failure is still reported faithfully
    assert api.move_to_joints(np.full(7, 0.1)).ok

    action = adapter.executed[-1]
    assert isinstance(action, RobotAction)
    assert action.arms["primary"].gripper_position == 0.0


def test_planned_motion_maintains_last_commanded_gripper_target() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]

    assert api.close_gripper().ok
    plan = api.plan_motion(np.full(7, 0.1))

    assert plan.ok
    assert plan.trajectory is not None
    np.testing.assert_array_equal(plan.trajectory.gripper_positions, 0.0)


def test_go_home_uses_initial_state_after_direct_motion() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    moved = np.full(7, 0.2)

    assert api.step(RobotAction(arms={"primary": ArmCommand("joint_position", moved)})).ok
    assert api.go_home().ok

    action = adapter.executed[-1]
    assert isinstance(action, RobotAction)
    np.testing.assert_array_equal(action.arms["primary"].target, np.zeros(7))


def test_go_home_forwards_secondary_arm_to_motion() -> None:
    adapter = FakeAdapter()
    pose = _pose()
    adapter.state = RobotState(
        joint_positions={"primary": np.zeros(7), "secondary": np.ones(7)},
        joint_velocities={"primary": np.zeros(7), "secondary": np.zeros(7)},
        end_effector_poses={"primary": pose, "secondary": pose},
        gripper_positions={"primary": 0.4, "secondary": 0.4},
        base_frame="base",
        joint_names={
            "primary": tuple(f"p{i}" for i in range(7)),
            "secondary": tuple(f"s{i}" for i in range(7)),
        },
    )
    api = CapApi(adapter)  # type: ignore[arg-type]
    calls = []

    def move(target, tolerance=0.01, max_steps=120, *, arm="primary"):
        calls.append((np.array(target), arm, tolerance, max_steps))
        return ExecutionResult(ok=True, steps_executed=0)

    api.move_to_joints = move  # type: ignore[method-assign]

    assert api.go_home(arm="secondary").ok
    assert calls[0][1] == "secondary"
    np.testing.assert_array_equal(calls[0][0], np.ones(7))


def test_public_mask_projection_signature_is_policy_facing() -> None:
    signature = inspect.signature(CapApi.mask_to_point_cloud)

    assert tuple(signature.parameters) == (
        "self",
        "mask",
        "camera_name",
        "target_frame",
    )
    assert signature.parameters["target_frame"].default == "robot_base"


def test_public_point_cloud_crop_is_frame_preserving() -> None:
    api = CapApi(FakeAdapter())  # type: ignore[arg-type]
    cloud = PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
        frame="base",
    )

    cropped = api.crop_point_cloud(cloud, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

    assert cropped.frame == "base"
    np.testing.assert_allclose(cropped.points, [[0.0, 0.0, 0.0]])


def test_pose_planning_preserves_exact_requested_wxyz_pose() -> None:
    target = Pose(
        position=np.array([0.1, 0.2, 0.3]),
        quaternion_wxyz=np.array([0.5, 0.5, 0.5, 0.5]),
        frame="base",
    )
    ik = FakeIK(np.full(7, 0.1))
    api = CapApi(FakeAdapter(), ik_providers={"pyroki": ik})  # type: ignore[arg-type]

    result = api.plan_motion(target)

    assert result.ok
    assert ik.targets == [target]
    assert result.trajectory is not None
    assert result.trajectory.planner == "pyroki+joint_interpolation"


def test_localize_object_composes_atomic_calls_in_order() -> None:
    api = CapApi(FakeAdapter())  # type: ignore[arg-type]
    calls: list[str] = []
    segmentation = Segmentation(
        mask=np.ones((2, 2), dtype=bool),
        label="cube",
        score=1.0,
        camera_name="agentview",
        frame="camera",
    )
    cloud = PointCloud(
        points=np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]),
        frame="base",
    )
    geometry = ObjectGeometry(pose=_pose(), extents=np.ones(3), point_count=8)

    def segment(*args: object, **kwargs: object) -> SegmentationSet:
        del args, kwargs
        calls.append("segment_text")
        return SegmentationSet(ok=True, segmentations=(segmentation,))

    def project(*args: object, **kwargs: object) -> PointCloud:
        del args, kwargs
        calls.append("mask_to_point_cloud")
        return cloud

    def estimate(value: PointCloud) -> ObjectGeometry:
        assert value is cloud
        calls.append("estimate_geometry")
        return geometry

    api._segment_text = segment  # type: ignore[method-assign]
    api._project_mask = project  # type: ignore[method-assign]
    api.estimate_geometry = estimate  # type: ignore[method-assign]

    result = api.localize_object("cube")

    assert result.ok
    assert calls == ["segment_text", "mask_to_point_cloud", "estimate_geometry"]


def test_public_step_removes_reward_and_success() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    result = api.step(RobotAction(arms={"primary": ArmCommand("joint_position", np.zeros(7))}))

    assert result.reward is None
    assert "success" not in result.diagnostics
    assert result.diagnostics["safe"] == "visible"
    assert "reward" not in result.diagnostics["nested"]


def test_camera_name_normalization_preserves_segmentation_box() -> None:
    segmentation = Segmentation(
        mask=np.ones((2, 2), dtype=bool),
        label="cube",
        score=0.9,
        camera_name="camera/frame",
        frame="camera/frame",
        box_xyxy=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    normalized = CapApi._with_camera_name(
        SegmentationSet(ok=True, segmentations=(segmentation,)),
        "agentview",
    )

    assert normalized.segmentations[0].camera_name == "agentview"
    np.testing.assert_array_equal(
        normalized.segmentations[0].box_xyxy,
        segmentation.box_xyxy,
    )


def test_generate_grasps_projects_mask_into_robot_base_frame() -> None:
    provider = FakeGraspProvider()
    api = CapApi(  # type: ignore[arg-type]
        FakeAdapter(),
        grasp_providers={"contact-graspnet": provider},
    )

    result = api.generate_grasps("agentview", np.ones((2, 2), dtype=bool))

    assert result.ok
    assert provider.point_cloud is not None
    assert provider.scene_point_cloud is not None
    assert provider.point_cloud.frame == "base"
    assert provider.scene_point_cloud.frame == "base"
    assert provider.max_candidates == 5
    assert len(provider.scene_point_cloud.points) >= len(provider.point_cloud.points)
    assert result.grasps[0].frame == "base"


def test_move_to_joints_reissues_target_until_observed_state_converges() -> None:
    class LaggingAdapter(FakeAdapter):
        def step(self, action: object) -> StepResult:
            assert isinstance(action, RobotAction)
            self.executed.append(action)
            current = self.state.joint_positions["primary"]
            target = action.arms["primary"].target
            updated = current + 0.5 * (target - current)
            self.state = RobotState(
                joint_positions=updated,
                joint_velocities=np.zeros(7),
                end_effector_poses=self.state.end_effector_poses,
                gripper_positions=self.state.gripper_positions,
                base_frame=self.state.base_frame,
                joint_names=self.state.joint_names,
            )
            return StepResult(ok=True, observation=self.get_observation())

    adapter = LaggingAdapter()
    target = np.full(7, 0.4)
    result = CapApi(adapter).move_to_joints(target, tolerance=0.01, max_steps=10)  # type: ignore[arg-type]

    assert result.ok
    assert result.steps_executed > 1
    assert all(np.array_equal(action.arms["primary"].target, target) for action in adapter.executed)


def test_move_to_joints_verifies_final_joint_error() -> None:
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    target = np.full(7, 0.08)

    result = api.move_to_joints(target)

    assert result.ok
    assert result.final_observation is not None
    np.testing.assert_allclose(
        result.final_observation.robot_state.joint_positions["primary"],
        target,
    )


def test_move_to_pose_always_plans_and_executes_a_trajectory() -> None:
    adapter = FakeAdapter()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        ik_providers={"pyroki": FakeIK(np.full(7, 0.1))},
    )

    def unexpected_direct_move(*args: object, **kwargs: object) -> ExecutionResult:
        raise AssertionError(f"move_to_joints was called directly: {args!r}, {kwargs!r}")

    api.move_to_joints = unexpected_direct_move  # type: ignore[method-assign]
    result = api.move_to_pose(_pose(), tolerance=0.01)

    assert result.ok
    assert result.diagnostics["trajectory_executed"] is True
    assert len(adapter.executed) == 1
    trajectory = adapter.executed[0]
    assert isinstance(trajectory, Trajectory)
    assert trajectory.planner == "pyroki+joint_interpolation"


@pytest.mark.parametrize("representation", ["rpy_deg", "quaternion_wxyz"])
def test_move_to_pose_mapping_reaches_planner_as_quaternion_pose(representation):
    angles = [23, -37, 61]
    rotation = Rotation.from_euler("xyz", angles, degrees=True)
    quaternion = rotation.as_quat()[[3, 0, 1, 2]]
    expected = Pose(position=np.zeros(3), quaternion_wxyz=quaternion, frame="base")
    adapter = FakeAdapter()
    adapter.state = replace(adapter.state, end_effector_poses=expected)
    ik = FakeIK(np.full(7, 0.1))
    api = CapApi(adapter, ik_providers={"pyroki": ik})
    orientation = angles if representation == "rpy_deg" else quaternion.tolist()
    target = {"position": [0, 0, 0], representation: orientation, "frame": "base"}

    result = api.move_to_pose(target)

    assert result.ok
    assert len(adapter.executed) == 1
    assert isinstance(adapter.executed[0], Trajectory)
    assert len(ik.targets) == 1
    np.testing.assert_allclose(ik.targets[0].as_matrix(), expected.as_matrix(), atol=1e-12)
    assert set(target) == {"position", representation, "frame"}


@pytest.mark.parametrize(
    "orientation",
    [
        {},
        {"rpy_deg": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
        {"rpy_deg": [0, 0]},
        {"rpy_deg": [0, 0, np.nan]},
        {"rpy_deg": [0, np.inf, 0]},
        {"quaternion_wxyz": [2, 0, 0, 0]},
        {"rpy_deg": [0, 0, 0], "unknown": True},
    ],
)
def test_move_to_pose_invalid_mapping_is_rejected_before_planning(orientation):
    adapter = FakeAdapter()
    ik = FakeIK(np.full(7, 0.1))
    api = CapApi(adapter, ik_providers={"pyroki": ik})

    result = api.move_to_pose({"position": [0, 0, 0], "frame": "base", **orientation})

    assert not result.ok
    assert result.error.code == ErrorCode.INVALID_REQUEST
    assert not ik.targets
    assert not adapter.executed


def test_move_to_pose_rejects_a_plan_over_the_waypoint_budget() -> None:
    adapter = FakeAdapter()
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        ik_providers={"pyroki": FakeIK(np.full(7, 0.1))},
    )

    result = api.move_to_pose(_pose(), max_steps=1)

    assert not result.ok
    assert result.error is not None and result.error.code == ErrorCode.INVALID_REQUEST
    assert result.error.details["waypoint_count"] == 2
    assert not adapter.executed


def test_execute_trajectory_cannot_continue_after_termination() -> None:
    api = CapApi(FakeAdapter())  # type: ignore[arg-type]
    trajectory = api.plan_motion(np.zeros(7)).trajectory
    assert trajectory is not None

    result = api.execute_trajectory(trajectory, stop_on_termination=False)

    assert not result.ok
    assert result.error is not None
    assert result.error.code.value == "unsupported"


def test_a_settling_gripper_does_not_make_an_arm_trajectory_stale() -> None:
    """Regression: the gripper was checked against a radian tolerance.

    Gripper positions are normalized 0-1 widths; joint positions are radians.
    Comparing the former against TRAJECTORY_START_TOLERANCE_RAD (0.02) rejected
    a gripper that had drifted 2% of its travel. Measured on hardware, a
    ``set_gripper(OPEN)`` before planning leaves the jaws still moving and the
    width differs by ~0.06 by the time the plan returns -- which rejected every
    transit of a pick with ``stale_plan`` while the arm itself matched its
    planned start to 0.0004 rad.
    """
    from cap_harness.api import (
        TRAJECTORY_START_GRIPPER_TOLERANCE,
        TRAJECTORY_START_TOLERANCE_RAD,
    )

    # The two bounds are different quantities and must not be the same constant.
    assert TRAJECTORY_START_GRIPPER_TOLERANCE > TRAJECTORY_START_TOLERANCE_RAD
    # The measured settling drift has to pass.
    assert TRAJECTORY_START_GRIPPER_TOLERANCE > 0.0587
    # But a gripper that has closed on an object still trips it: that changes
    # the collision geometry the path was planned against.
    assert TRAJECTORY_START_GRIPPER_TOLERANCE < 0.6


def test_solve_ik_defaults_to_the_configured_ik_backend() -> None:
    adapter = _dual_adapter()
    curobo = FakeIK(np.full(7, 0.2))
    api = CapApi(  # type: ignore[arg-type]
        adapter,
        ik_providers={"curobo": curobo},
        default_motion_strategy=MotionStrategy(ik_solver="curobo"),
    )
    pose = Pose(
        position=(0.3, 0.0, 0.2),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        frame=adapter.get_robot_state().base_frame,
    )
    result = api.solve_ik(pose)
    assert result.ok and curobo.targets == [pose]
    missing = api.solve_ik(pose, backend="pyroki")
    assert not missing.ok and missing.error.code == ErrorCode.UNSUPPORTED


def test_set_gripper_refuses_a_width_the_actuator_cannot_hold() -> None:
    """On a velocity-driven gripper only the endpoints are reachable states.

    Robosuite's `simple_grip` assigns `goal_qvel`, so the sign picks a direction
    and the jaw runs to its stop. Measured on `libero_10_swap/task_7`,
    `set_gripper(0.669)` and `set_gripper(0.933)` each spent 60 ticks and ended
    fully OPEN -- 12% of the episode, moving away from the request. Refusing
    costs nothing and says so.
    """
    adapter = FakeAdapter()
    adapter.gripper_holds_intermediate = False
    api = CapApi(adapter)  # type: ignore[arg-type]

    refused = api.set_gripper(0.669)

    assert not refused.ok
    assert refused.steps_executed == 0
    assert refused.error is not None
    assert refused.error.code is ErrorCode.UNSUPPORTED
    assert "open_gripper" in refused.error.message


def test_set_gripper_still_accepts_both_endpoints_on_a_binary_gripper() -> None:
    """`set_gripper(1.0)` is used as a re-command; refusing it would be a regression."""
    adapter = FakeAdapter()
    adapter.gripper_holds_intermediate = False
    api = CapApi(adapter)  # type: ignore[arg-type]

    assert api.set_gripper(1.0).ok
    assert api.set_gripper(0.0).ok


def test_set_gripper_leaves_a_position_controlled_gripper_alone() -> None:
    """YAM commands a real width target, so intermediate values stay legal."""
    api = CapApi(FakeAdapter())  # type: ignore[arg-type]

    assert api.set_gripper(0.669).ok


def test_go_home_reissues_until_the_arm_arrives() -> None:
    """`go_home()` promises a configuration, not one 120-tick attempt.

    `move_to_joints` caps at 120 ticks and the joint servo droops past roughly
    0.24 rad, so a single call out of a folded pose returns ok=False having moved
    most of the way. Measured on RoboCasa `drawer_utensil_sort:0`: three calls
    (gaps 0.3724 -> 0.2743 -> 0.0029). A caller that trusts the first ok=False is
    left stranded, and every later move_to_pose crawls against the residual.
    """
    adapter = FakeAdapter()
    api = CapApi(adapter)  # type: ignore[arg-type]
    calls = {"n": 0}
    real = api.move_to_joints

    def drooping(target, *args, **kwargs):
        calls["n"] += 1
        outcome = real(target, *args, **kwargs)
        if calls["n"] < 3:
            return ExecutionResult(
                ok=False,
                steps_executed=120,
                final_observation=outcome.final_observation,
                error=ApiError(ErrorCode.TIMEOUT, "joint move did not converge"),
            ).public_view()
        return outcome

    api.move_to_joints = drooping  # type: ignore[assignment]
    result = api.go_home()

    assert result.ok, "go_home gave up while the calls were still making progress"
    assert calls["n"] == 3
    assert result.steps_executed > 120, "the earlier attempts' ticks must be counted"


@pytest.mark.parametrize("terminal_flag", ["terminated", "truncated"])
@pytest.mark.parametrize("first_is_terminal", [True, False])
def test_go_home_stops_at_terminal_results(monkeypatch, terminal_flag, first_is_terminal) -> None:
    api = CapApi(FakeAdapter())
    terminal = ExecutionResult(
        ok=False,
        steps_executed=1 if first_is_terminal else 0,
        error=ApiError(ErrorCode.TIMEOUT, "episode ended"),
        **{terminal_flag: True},
    )
    outcomes = (
        [terminal]
        if first_is_terminal
        else [
            ExecutionResult(
                ok=False, steps_executed=120, error=ApiError(ErrorCode.TIMEOUT, "still approaching")
            ),
            terminal,
        ]
    )
    calls = []

    def move(*args, **kwargs):
        calls.append(1)
        assert len(calls) <= len(outcomes), "command issued after episode ended"
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(api, "move_to_joints", move)
    result = api.go_home()
    assert len(calls) == len(outcomes)
    assert getattr(result, terminal_flag)
    assert result.error == terminal.error
    assert result.steps_executed == (1 if first_is_terminal else 120)
