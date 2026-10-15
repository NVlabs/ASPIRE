"""Hierarchical public API composed from adapters and injected providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from cap_harness.contracts import (
    DEFAULT_EMBODIMENT,
    JOINT_DIMENSIONS,
    ArmCommand,
    ExecutionResult,
    GraspCandidate,
    GraspSet,
    IKResult,
    LocalizationResult,
    MotionStrategy,
    ObjectGeometry,
    Observation,
    PlanningScene,
    PlanResult,
    PointCloud,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    Segmentation,
    SegmentationSet,
    StepResult,
    SynchronizedPlanResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.registry import ToolRegistry

from .errors import ApiError, ErrorCode
from .geometry import (
    crop_point_cloud as crop_observed_point_cloud,
)
from .geometry import (
    estimate_geometry as estimate_object_geometry,
)
from .geometry import (
    mask_to_point_cloud as project_mask_to_point_cloud,
)
from .geometry import (
    pose_from_mapping,
    select_top_down_grasp,
)
from .protocols import (
    EnvironmentAdapter,
    GraspProvider,
    IKProvider,
    IntegratedPosePlanningProvider,
    SegmentationProvider,
    TrajectoryPlanningProvider,
)

TRAJECTORY_START_TOLERANCE_RAD = 0.02

#: How far the gripper may have moved since planning before a trajectory counts
#: as stale. Separate from the joint tolerance above because it is a *different
#: unit*: gripper positions are normalized 0-1 widths, not radians, and applying
#: a 0.02 rad joint threshold to them rejects a gripper that has drifted 2% of
#: its travel.
#:
#: Measured on a real bimanual station: a ``set_gripper(OPEN)`` just before planning leaves
#: the jaws still travelling -- ``gripper_full_travel_s`` is 3.0 -- so by the
#: time the plan comes back the width has moved ~0.06. That is physically
#: nothing, and it was rejecting every transit of a pick.
#:
#: Loose on purpose. The stale-plan check exists to catch an ARM that moved
#: since planning, which would invalidate the path; the gripper is commanded
#: independently along the trajectory's own gripper column. This bound only
#: catches a gripper that has changed enough to alter the collision geometry.
TRAJECTORY_START_GRIPPER_TOLERANCE = 0.25


class CapApi:
    """Atomic robot APIs plus explicit higher-level local compositions."""

    #: How many times `go_home()` may re-issue the joint move before giving up.
    #: Three sufficed in every measured case; the cap only stops a wedged arm
    #: from spending an episode trying.
    GO_HOME_MAX_ATTEMPTS = 4

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        *,
        segmentation_provider: SegmentationProvider | None = None,
        grasp_providers: Mapping[str, GraspProvider] | None = None,
        ik_providers: Mapping[str, IKProvider] | None = None,
        trajectory_planning_providers: Mapping[str, TrajectoryPlanningProvider] | None = None,
        integrated_pose_planning_providers: (
            Mapping[str, IntegratedPosePlanningProvider] | None
        ) = None,
        default_camera: str = "agentview",
        default_motion_strategy: MotionStrategy | None = None,
        home_joint_positions: Mapping[str, np.ndarray] | np.ndarray | None = None,
        recorder: object | None = None,
    ) -> None:
        self._adapter = adapter
        declared_embodiment = str(getattr(adapter, "embodiment", DEFAULT_EMBODIMENT))
        self._embodiment = (
            declared_embodiment if declared_embodiment in JOINT_DIMENSIONS else DEFAULT_EMBODIMENT
        )
        self._segmentation_provider = segmentation_provider
        self._grasp_providers = dict(grasp_providers or {})
        self._ik_providers = dict(ik_providers or {})
        self._trajectory_planning_providers = dict(trajectory_planning_providers or {})
        self._integrated_pose_planning_providers = dict(integrated_pose_planning_providers or {})
        self.default_camera = default_camera
        self.default_motion_strategy = default_motion_strategy
        self._recorder = recorder
        self._home_joint_positions: dict[str, np.ndarray] = {}
        # The last jaw target *commanded* per arm, not the last one reached. It fills
        # `gripper_position` for arm commands a program issues without one, so it must
        # record every accepted command including those that do not converge. A grasp
        # that succeeds is exactly such a command: the jaw stalls on the object well
        # short of 0.0, and the adapter reports `ok=False`. Latching only on success
        # therefore left this at the 1.0 from the open-gripper before the approach, and
        # every later move re-commanded the jaw open and dropped the held object.
        self._commanded_gripper_positions: dict[str, float] = {}
        if home_joint_positions is None:
            try:
                self._remember_home(adapter.get_robot_state())
            except RuntimeError:
                # Adapters may be injected before reset. In that case callers must
                # provide an explicit home; never redefine home from a later pose.
                pass
        else:
            source = (
                home_joint_positions
                if isinstance(home_joint_positions, Mapping)
                else {"primary": home_joint_positions}
            )
            for arm, value in source.items():
                joints = np.asarray(value, dtype=np.float64)
                if joints.shape not in {(6,), (7,)} or not bool(np.all(np.isfinite(joints))):
                    raise ValueError(
                        "home_joint_positions must contain finite 6- or 7-joint vectors"
                    )
                self._home_joint_positions[str(arm)] = joints.copy()

    @property
    def configured_providers(self) -> dict[str, object]:
        """Describe configured backends; this is not a service health check."""
        return {
            "segmentation": self._segmentation_provider is not None,
            "ik": tuple(self._ik_providers),
            "trajectory_planning": tuple(self._trajectory_planning_providers),
            "pose_planning": tuple(self._integrated_pose_planning_providers),
        }

    def monitor_snapshot(self) -> dict[str, object]:
        """Return the adapter's live station cache."""
        monitor = getattr(self._adapter, "monitor_snapshot", None)
        if not callable(monitor):
            return {"ok": False, "error": "environment does not support live monitoring"}
        try:
            snapshot = dict(monitor())
        except Exception as exc:
            return {"ok": False, "error": f"station monitor failed: {exc}"}
        snapshot["ok"] = True
        return snapshot

    def get_task_context(self) -> TaskContext:
        """The task's identity, plus how many control ticks it actually has.

        The tick budget is the binding constraint on these benchmarks -- a task
        loses by running out of episode far more often than by being unable to
        reach anything -- and until now a program had no way to read it. Every
        generated program that cared therefore hardcoded a guess, and one that
        guessed 995 went on budgeting against 995 after the harness had given it
        1950, spending none of the extra and finishing on exactly the same tick
        as before. Publishing the number costs nothing and removes a whole class
        of silently stale constant.
        """
        context = self._adapter.get_task_context()
        if self._recorder is None:
            return context
        total = getattr(self._recorder, "max_steps", None)
        used = getattr(self._recorder, "step_count", None)
        if total is None:
            return context
        metadata = dict(context.metadata)
        metadata["episode_max_steps"] = int(total)
        if used is not None:
            metadata["episode_steps_used"] = int(used)
            metadata["episode_steps_remaining"] = max(0, int(total) - int(used))
        return TaskContext(
            suite=context.suite,
            task_id=context.task_id,
            task_name=context.task_name,
            language=context.language,
            family=context.family,
            metadata=metadata,
        )

    def get_observation(self, camera_names: Sequence[str] | str | None = None) -> Observation:
        observation = self._adapter.get_observation()
        if camera_names is not None:
            names = (camera_names,) if isinstance(camera_names, str) else tuple(camera_names)
            if not names or any(name not in observation.cameras for name in names):
                raise ValueError("camera_names must name cameras present in the observation")
            observation = Observation(
                cameras={name: observation.cameras[name] for name in names},
                robot_state=observation.robot_state,
                task_context=observation.task_context,
                timestamp_s=observation.timestamp_s,
            )
        return observation

    def get_robot_state(self) -> RobotState:
        state = self._adapter.get_robot_state()
        if state.embodiment != self._embodiment:
            raise ValueError(
                f"adapter embodiment {self._embodiment!r} returned {state.embodiment!r} RobotState"
            )
        return state

    def segment_text(
        self,
        camera_name: str,
        text: str,
    ) -> SegmentationSet:
        return self._segment_text(camera_name, text, self.get_observation(camera_name))

    def _segment_text(
        self, camera_name: str, text: str, observation: Observation
    ) -> SegmentationSet:
        if self._segmentation_provider is None:
            return self._segmentation_failure("no segmentation provider is configured")
        camera = observation.cameras[camera_name]
        result = self._segmentation_provider.segment_text(camera, text)
        return self._with_camera_name(result, camera_name)

    def segment_points(
        self,
        camera_name: str,
        points_px: np.ndarray,
    ) -> SegmentationSet:
        if self._segmentation_provider is None:
            return self._segmentation_failure("no segmentation provider is configured")
        observation = self.get_observation(camera_name)
        result = self._segmentation_provider.segment_points(
            observation.cameras[camera_name], points_px
        )
        return self._with_camera_name(result, camera_name)

    def mask_to_point_cloud(
        self,
        mask: Segmentation | np.ndarray,
        camera_name: str,
        target_frame: str = "robot_base",
    ) -> PointCloud:
        observation = self.get_observation(camera_name)
        return self._project_mask(
            mask,
            camera_name=camera_name,
            target_frame=target_frame,
            observation=observation,
        )

    def _project_mask(
        self,
        mask: Segmentation | np.ndarray,
        *,
        camera_name: str,
        target_frame: str,
        observation: Observation,
    ) -> PointCloud:
        camera = observation.cameras[camera_name]
        if isinstance(mask, Segmentation):
            segmentation = mask
        else:
            segmentation = Segmentation(
                mask=mask,
                label="object",
                score=1.0,
                camera_name=camera_name,
                frame=camera.frame,
            )
        if segmentation.camera_name not in {camera_name, camera.frame}:
            raise ValueError(
                f"segmentation camera {segmentation.camera_name!r} does not match {camera_name!r}"
            )
        return project_mask_to_point_cloud(camera, segmentation, target_frame=target_frame)

    def estimate_geometry(self, point_cloud: PointCloud) -> ObjectGeometry:
        return estimate_object_geometry(point_cloud)

    def crop_point_cloud(
        self,
        point_cloud: PointCloud,
        lower_xyz: object,
        upper_xyz: object,
    ) -> PointCloud:
        return crop_observed_point_cloud(point_cloud, lower_xyz, upper_xyz)

    def generate_grasps(
        self,
        camera_name: str,
        mask: Segmentation | np.ndarray,
        *,
        backend: str = "contact-graspnet",
        max_candidates: int = 5,
    ) -> GraspSet:
        provider = self._grasp_providers.get(backend)
        if provider is None:
            return GraspSet(
                ok=False,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=f"grasp backend {backend!r} is not configured",
                    details={"backend": backend},
                ),
            )
        observation = self.get_observation(camera_name)
        camera = observation.cameras[camera_name]
        if isinstance(mask, Segmentation):
            segmentation = mask
        else:
            segmentation = Segmentation(
                mask=mask,
                label="grasp target",
                score=1.0,
                camera_name=camera_name,
                frame=camera.frame,
            )
        try:
            point_cloud = self._stage(
                "mask_to_point_cloud",
                self._project_mask,
                segmentation,
                camera_name=camera_name,
                target_frame=observation.robot_state.base_frame,
                observation=observation,
            )
            scene_point_cloud = self._stage(
                "mask_to_point_cloud",
                self._project_mask,
                np.ones(camera.depth_m.shape, dtype=bool),
                camera_name=camera_name,
                target_frame=observation.robot_state.base_frame,
                observation=observation,
            )
        except ValueError as exc:
            return GraspSet(
                ok=False,
                error=ApiError(
                    code=ErrorCode.POINT_CLOUD_FAILED,
                    message=f"cannot project grasp mask: {exc}",
                ),
            )
        # Caller-chosen, because a caller that filters candidates itself -- by
        # approach direction, say -- needs more of them than one that takes the
        # first. Five was fine as a fixed ceiling only while nobody filtered.
        return provider.generate_grasps(
            point_cloud,
            scene_point_cloud=scene_point_cloud,
            max_candidates=max_candidates,
        )

    def _resolve_arm(self, arm: str) -> str:
        """An embodiment's own name for the arm the caller asked for.

        Arm vocabularies differ between embodiments and there is no translation
        between them here: the shared signatures default to
        ``"primary"``/``"secondary"``, single-arm backends expose exactly
        ``"primary"``, and a bimanual embodiment may name its arms
        ``"left"``/``"right"``. Without a mapping every shared default fails on
        such an embodiment -- ``set_gripper()`` with no arm reports
        ``execution_failed``, and ``plan_motion()`` reports
        ``robot state has no arm 'primary'``.

        An embodiment that knows a mapping exposes ``resolve_arm``; anything
        else is passed through unchanged, so no other backend is affected. The
        mapping is DATA, not a convention hardcoded here: which physical arm is
        "primary" is a property of how a station is set up and rigged, and its
        profile is what declares it.
        """
        resolver = getattr(self._adapter, "resolve_arm", None)
        if not callable(resolver):
            return arm
        try:
            return str(resolver(arm))
        # Best-effort arm resolution must not mask the embodiment error.
        except Exception:
            # embodiment's own error, not be masked by the resolver failing.
            return arm

    def solve_ik(
        self,
        target_pose: Pose,
        seed_joints: np.ndarray | None = None,
        *,
        arm: str = "primary",
        backend: str | None = None,
    ) -> IKResult:
        arm = self._resolve_arm(arm)
        if backend is None:
            # The benchmark's default IK solver (pyroki unless the runner configured another,
            # as the BEHAVIOR runner does with its in-process cuRobo).
            strategy = self.default_motion_strategy
            backend = strategy.ik_solver if strategy is not None else "pyroki"
        provider = self._ik_providers.get(backend)
        if provider is None:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=self._embodiment,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=f"IK backend {backend!r} is not configured",
                    details={"backend": backend},
                ),
            )
        state = self.get_robot_state()
        if arm not in state.joint_positions:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=self._embodiment,
                error=ApiError(code=ErrorCode.NOT_FOUND, message=f"unknown arm {arm!r}"),
            )
        if seed_joints is not None:
            joint_dimension = state.joint_positions[arm].shape[0]
            try:
                seed = np.asarray(seed_joints, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                return IKResult(
                    ok=False,
                    embodiment=self._embodiment,
                    error=ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=(
                            f"seed_joints must be a finite {joint_dimension}-joint vector: {exc}"
                        ),
                    ),
                )
            if seed.shape != (joint_dimension,) or not bool(np.all(np.isfinite(seed))):
                return IKResult(
                    ok=False,
                    embodiment=self._embodiment,
                    error=ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=(f"seed_joints must be a finite {joint_dimension}-joint vector"),
                    ),
                )
            state = self._state_with_arm_joints(state, arm, seed)
        adapter_transform = getattr(self._adapter, "ik_request", None)
        if callable(adapter_transform):
            try:
                target_pose, state = adapter_transform(target_pose, state, arm)
            except (TypeError, ValueError) as exc:
                return IKResult(
                    ok=False,
                    arm=arm,
                    embodiment=self._embodiment,
                    error=ApiError(code=ErrorCode.INVALID_REQUEST, message=str(exc)),
                )
        solve_with_context = getattr(provider, "solve_ik_with_context", None)
        if callable(solve_with_context):
            result = solve_with_context(
                target_pose,
                state,
                arm=arm,
                context=self._planning_context(state),
            )
        else:
            result = provider.solve_ik(target_pose, state, arm=arm)
        if result.ok and (
            result.joint_positions is None
            or result.joint_positions.shape != state.joint_positions[arm].shape
            or result.arm != arm
        ):
            return IKResult(
                ok=False,
                arm=arm,
                error=ApiError(
                    code=ErrorCode.IK_FAILED,
                    message="IK provider returned a solution for the wrong arm dimension",
                    details={
                        "arm": arm,
                        "expected_dimension": state.joint_positions[arm].shape[0],
                        "returned_dimension": (
                            None
                            if result.joint_positions is None
                            else result.joint_positions.shape[0]
                        ),
                    },
                ),
            )
        return result

    def plan_motion(
        self,
        goal: Pose | np.ndarray,
        start_joints: np.ndarray | None = None,
        *,
        arm: str = "primary",
        strategy: MotionStrategy | None = None,
        target_point_cloud: PointCloud | None = None,
    ) -> PlanResult:
        arm = self._resolve_arm(arm)
        if strategy is None:
            strategy = self.default_motion_strategy or MotionStrategy()
        if not isinstance(strategy, MotionStrategy):
            return self._plan_failure(
                ErrorCode.INVALID_REQUEST,
                "strategy must be a MotionStrategy",
            )
        if target_point_cloud is not None and not isinstance(target_point_cloud, PointCloud):
            return self._plan_failure(
                ErrorCode.INVALID_REQUEST,
                "target_point_cloud must be a PointCloud or None",
            )
        state = self.get_robot_state()
        if target_point_cloud is not None and target_point_cloud.frame != state.base_frame:
            return self._plan_failure(
                ErrorCode.INVALID_REQUEST,
                f"target point cloud is in frame {target_point_cloud.frame!r}, "
                f"expected {state.base_frame!r}",
            )
        if arm not in state.joint_positions:
            return self._plan_failure(ErrorCode.NOT_FOUND, f"robot state has no arm {arm!r}")
        joint_dimension = state.joint_positions[arm].shape[0]
        if start_joints is None:
            start = state.joint_positions[arm]
        else:
            try:
                start = np.asarray(start_joints, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                return self._plan_failure(
                    ErrorCode.INVALID_REQUEST,
                    f"start_joints must be a finite {joint_dimension}-joint vector",
                    details={"error": str(exc)},
                )
            if start.shape != (joint_dimension,) or not bool(np.all(np.isfinite(start))):
                return self._plan_failure(
                    ErrorCode.INVALID_REQUEST,
                    f"start_joints must be a finite {joint_dimension}-joint vector",
                )
        planning_state = self._state_with_arm_joints(state, arm, start)
        timing_options = {
            name: value
            for name, value in (
                ("time_dilation_factor", strategy.time_dilation_factor),
                ("interpolation_dt_s", strategy.interpolation_dt_s),
                ("maximum_trajectory_dt_s", strategy.maximum_trajectory_dt_s),
            )
            if value is not None
        }
        if isinstance(goal, Pose) and strategy.pose_planner != "composed":
            provider = self._integrated_pose_planning_providers.get(strategy.pose_planner)
            if provider is None:
                return self._plan_failure(
                    ErrorCode.UNSUPPORTED,
                    f"integrated pose planner {strategy.pose_planner!r} is not configured",
                )
            planning_goal = goal
            adapter_transform = getattr(self._adapter, "ik_request", None)
            if callable(adapter_transform):
                try:
                    planning_goal, planning_state = adapter_transform(
                        planning_goal, planning_state, arm
                    )
                except (TypeError, ValueError) as exc:
                    return self._plan_failure(ErrorCode.INVALID_REQUEST, str(exc))
            try:
                result = provider.plan_to_pose(
                    planning_state,
                    planning_goal,
                    arm=arm,
                    gripper_position=self._commanded_gripper_position(state, arm),
                    scene=self._planning_scene_request(
                        arm,
                        target_point_cloud=target_point_cloud,
                    ),
                    context=self._planning_context(planning_state),
                    **timing_options,
                )
                validated = self._validate_plan_start(result, planning_state)
                return self._bind_plan_start(validated, state)
            # Best-effort arm resolution must not mask the embodiment error.
            except Exception as exc:
                return self._plan_failure(
                    ErrorCode.PLANNING_FAILED,
                    f"integrated pose planning provider failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                )

        planner = "joint_interpolation"
        if isinstance(goal, Pose):
            ik_result = self.solve_ik(
                goal,
                start,
                arm=arm,
                backend=strategy.ik_solver,
            )
            if not ik_result.ok:
                return PlanResult(ok=False, error=ik_result.error)
            joint_goal = ik_result.joint_positions
            planner = f"{strategy.ik_solver}+joint_interpolation"
        else:
            try:
                joint_goal = np.asarray(goal, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                return self._plan_failure(
                    ErrorCode.INVALID_REQUEST,
                    f"joint target must be a finite {joint_dimension}-joint vector",
                    details={"error": str(exc)},
                )
            if joint_goal.shape != (joint_dimension,) or not bool(np.all(np.isfinite(joint_goal))):
                return self._plan_failure(
                    ErrorCode.INVALID_REQUEST,
                    f"joint target must be a finite {joint_dimension}-joint vector",
                )
        assert joint_goal is not None

        if strategy.trajectory_planner != "interpolation":
            provider = self._trajectory_planning_providers.get(strategy.trajectory_planner)
            if provider is None:
                return self._plan_failure(
                    ErrorCode.UNSUPPORTED,
                    f"trajectory planner {strategy.trajectory_planner!r} is not configured",
                )
            try:
                result = provider.plan_to_joints(
                    planning_state,
                    joint_goal,
                    arm=arm,
                    gripper_position=self._commanded_gripper_position(state, arm),
                    scene=self._planning_scene(target_point_cloud=target_point_cloud),
                    context=self._planning_context(state),
                    **timing_options,
                )
                return self._validate_plan_start(result, planning_state)
            # Best-effort arm resolution must not mask the embodiment error.
            except Exception as exc:
                return self._plan_failure(
                    ErrorCode.PLANNING_FAILED,
                    f"trajectory planning provider failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                )

        step_limits, dt_s = self._controller_step_limits(start, joint_goal, arm)
        difference = joint_goal - start
        waypoint_count = max(
            1,
            int(np.max(np.ceil(np.abs(difference) / step_limits))),
        )
        fractions = np.arange(1, waypoint_count + 1, dtype=np.float64) / waypoint_count
        waypoints = start + fractions[:, None] * difference
        waypoints[-1] = joint_goal
        gripper = self._commanded_gripper_position(state, arm)
        trajectory = Trajectory(
            joint_positions=waypoints,
            dt_s=dt_s,
            joint_names=state.joint_names[arm],
            planner=planner,
            collision_aware=False,
            expected_start=planning_state,
            arm=arm,
            gripper_positions=np.full(waypoint_count, gripper, dtype=np.float64),
            embodiment=self._embodiment,
        )
        return PlanResult(
            ok=True,
            trajectory=trajectory,
            diagnostics={
                "ik_solver": strategy.ik_solver,
                "trajectory_planner": strategy.trajectory_planner,
                "max_joint_delta_rad": float(np.max(step_limits)),
            },
        )

    def plan_synchronized_motion(
        self,
        targets: Mapping[str, Pose | np.ndarray],
        strategy: MotionStrategy | None = None,
    ) -> SynchronizedPlanResult:
        if strategy is None:
            strategy = self.default_motion_strategy or MotionStrategy(
                pose_planner="curobo-integrated"
            )
        if not isinstance(strategy, MotionStrategy):
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="strategy must be a MotionStrategy",
                ),
            )
        state = self.get_robot_state()
        normalized_targets, error = self._normalize_synchronized_targets(targets, state)
        if error is not None:
            return SynchronizedPlanResult(ok=False, error=error)
        assert normalized_targets is not None
        provider = self._integrated_pose_planning_providers.get(strategy.pose_planner)
        if provider is None:
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=(
                        f"integrated pose planner {strategy.pose_planner!r} is not configured"
                    ),
                ),
            )
        grippers = {arm: self._commanded_gripper_position(state, arm) for arm in state.arms}
        timing_options = {
            name: value
            for name, value in (
                ("time_dilation_factor", strategy.time_dilation_factor),
                ("interpolation_dt_s", strategy.interpolation_dt_s),
                ("maximum_trajectory_dt_s", strategy.maximum_trajectory_dt_s),
            )
            if value is not None
        }
        try:
            result = provider.plan_synchronized_motion(
                state,
                normalized_targets,
                gripper_positions=grippers,
                scene=self._planning_scene(),
                context=self._planning_context(state),
                **timing_options,
            )
            return self._validate_synchronized_plan_start(result, state)
        # Best-effort arm resolution must not mask the embodiment error.
        except Exception as exc:
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.PLANNING_FAILED,
                    message=f"synchronized motion planning provider failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )

    def move_synchronized(
        self,
        targets: Mapping[str, Pose | np.ndarray],
        tolerance: float = 0.01,
        max_steps: int = 120,
        strategy: MotionStrategy | None = None,
    ) -> ExecutionResult:
        validation = self._motion_options(tolerance, max_steps)
        if validation is not None:
            return validation
        state = self.get_robot_state()
        normalized_targets, error = self._normalize_synchronized_targets(targets, state)
        if error is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=error)
        assert normalized_targets is not None
        plan = self._stage(
            "plan_synchronized_motion",
            self.plan_synchronized_motion,
            normalized_targets,
            strategy,
        )
        if not plan.ok or plan.trajectory is None:
            return ExecutionResult(ok=False, steps_executed=0, error=plan.error)
        if plan.trajectory.waypoint_count > max_steps:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="planned synchronized trajectory exceeds max_steps",
                    details={
                        "waypoint_count": plan.trajectory.waypoint_count,
                        "max_steps": max_steps,
                    },
                ),
            )
        execution = self._stage("execute_trajectory", self.execute_trajectory, plan.trajectory)
        if not execution.ok:
            return execution
        final_state = self._execution_state(execution)
        final_errors: dict[str, float] = {}
        for arm, target in normalized_targets.items():
            if isinstance(target, Pose):
                current = final_state.end_effector_poses[arm]
                position_error = float(np.linalg.norm(current.position - target.position))
                orientation_error = float(
                    2.0
                    * np.arccos(
                        np.clip(
                            abs(np.dot(current.quaternion_wxyz, target.quaternion_wxyz)),
                            0.0,
                            1.0,
                        )
                    )
                )
                final_errors[arm] = position_error
                if position_error > tolerance or orientation_error > np.deg2rad(10.0):
                    return ExecutionResult(
                        ok=False,
                        steps_executed=execution.steps_executed,
                        final_observation=execution.final_observation,
                        terminated=execution.terminated,
                        truncated=execution.truncated,
                        error=ApiError(
                            code=ErrorCode.CONTROLLER_FAILED,
                            message=f"synchronized pose target for {arm!r} is outside tolerance",
                            details={
                                "arm": arm,
                                "position_error_m": position_error,
                                "orientation_error_rad": orientation_error,
                            },
                        ),
                        final_errors=final_errors,
                    )
            else:
                joint_error = float(np.max(np.abs(final_state.joint_positions[arm] - target)))
                final_errors[arm] = joint_error
                if joint_error > tolerance:
                    return ExecutionResult(
                        ok=False,
                        steps_executed=execution.steps_executed,
                        final_observation=execution.final_observation,
                        terminated=execution.terminated,
                        truncated=execution.truncated,
                        error=ApiError(
                            code=ErrorCode.CONTROLLER_FAILED,
                            message=f"synchronized joint target for {arm!r} is outside tolerance",
                            details={
                                "arm": arm,
                                "joint_error_rad": joint_error,
                                "tolerance_rad": tolerance,
                            },
                        ),
                        final_errors=final_errors,
                    )
        return ExecutionResult(
            ok=True,
            steps_executed=execution.steps_executed,
            final_observation=execution.final_observation,
            final_errors=final_errors,
            diagnostics=execution.diagnostics,
        )

    def step(self, action: RobotAction) -> StepResult:
        if action.embodiment != self._embodiment:
            return StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=(
                        f"action embodiment {action.embodiment!r} does not match "
                        f"adapter embodiment {self._embodiment!r}"
                    ),
                ),
            )
        try:
            result = self._adapter.step(action)
        # Best-effort arm resolution must not mask the embodiment error.
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.ADAPTER_FAILED,
                    message=f"adapter step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        public = result.public_view()
        if public.ok:
            for arm, command in action.arms.items():
                if command.gripper_position is not None:
                    self._commanded_gripper_positions[arm] = command.gripper_position
        return public

    def execute_trajectory(
        self,
        trajectory: Trajectory | SynchronizedTrajectory,
        stop_on_termination: bool = True,
    ) -> ExecutionResult:
        if trajectory.embodiment != self._embodiment:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=(
                        f"trajectory embodiment {trajectory.embodiment!r} does not match "
                        f"adapter embodiment {self._embodiment!r}"
                    ),
                ),
            )
        if type(stop_on_termination) is not bool:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="stop_on_termination must be a bool",
                ),
            )
        if not stop_on_termination:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=(
                        "stop_on_termination=False is unsupported; adapters must stop "
                        "on termination or truncation"
                    ),
                ),
            )
        stale = self._stale_trajectory_result(trajectory)
        if stale is not None:
            return stale
        try:
            result = self._adapter.execute_trajectory(trajectory)
        except (TypeError, ValueError) as exc:
            result = ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"invalid trajectory: {exc}",
                ),
            )
        # Best-effort arm resolution must not mask the embodiment error.
        except Exception as exc:
            result = ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"trajectory execution failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        public = result.public_view()
        if public.ok:
            if isinstance(trajectory, Trajectory):
                if trajectory.gripper_positions is not None:
                    self._commanded_gripper_positions[trajectory.arm] = float(
                        trajectory.gripper_positions[-1]
                    )
            elif trajectory.gripper_positions is not None:
                self._commanded_gripper_positions.update(
                    {arm: float(values[-1]) for arm, values in trajectory.gripper_positions.items()}
                )
        return public

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        """Command every arm's gripper together in exactly one control tick."""
        state = self.get_robot_state()
        if not isinstance(positions, Mapping) or set(positions) != set(state.arms):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="positions must name every robot arm",
                    details={
                        "expected_arms": state.arms,
                        "position_arms": tuple(positions) if isinstance(positions, Mapping) else (),
                    },
                ),
            )
        normalized: dict[str, float] = {}
        for arm in state.arms:
            try:
                numeric = float(positions[arm])
            except (TypeError, ValueError):
                numeric = float("nan")
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=f"gripper position for {arm!r} must be finite and in [0, 1]",
                    ),
                )
            if not self._gripper_target_is_holdable(numeric):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.UNSUPPORTED,
                        message=(
                            f"gripper for {arm!r} only holds fully open (1.0) or fully closed "
                            f"(0.0); {numeric} cannot be held -- use open_gripper() or "
                            "close_gripper()"
                        ),
                        details={"requested": numeric, "holdable": (0.0, 1.0)},
                    ),
                )
            normalized[arm] = numeric
        adapter_method = getattr(self._adapter, "set_grippers", None)
        if callable(adapter_method):
            try:
                execution = adapter_method(normalized).public_view()
            # Best-effort arm resolution must not mask the embodiment error.
            except Exception as exc:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.CONTROLLER_FAILED,
                        message=f"coordinated gripper execution failed: {exc}",
                        details={"exception_type": type(exc).__name__},
                    ),
                )
            self._commanded_gripper_positions.update(normalized)
            return execution
        step = self.step(
            RobotAction(
                arms={
                    arm: ArmCommand(
                        mode="joint_position",
                        target=state.joint_positions[arm],
                        gripper_position=normalized[arm],
                        embodiment=self._embodiment,
                    )
                    for arm in state.arms
                }
            )
        )
        error = step.error
        if step.ok and (step.terminated or step.truncated):
            error = self._terminal_error(step.terminated, step.truncated)
        execution = ExecutionResult(
            ok=step.ok and not step.terminated and not step.truncated,
            steps_executed=1 if step.ok else 0,
            final_observation=step.observation,
            terminated=step.terminated,
            truncated=step.truncated,
            error=error,
        ).public_view()
        self._commanded_gripper_positions.update(normalized)
        return execution

    def _gripper_target_is_holdable(self, value: float) -> bool:
        """Whether this embodiment can actually hold an intermediate jaw width.

        On LIBERO and Robosuite/RoboCasa the gripper action is a **velocity**,
        not a position: `simple_grip.set_goal` assigns `goal_qvel`, so the sign
        chooses a direction and the magnitude only chooses how fast the jaw
        travels to its stop. An intermediate target is therefore unreachable --
        the jaw passes through it and keeps going -- and the wait loop spends
        its whole budget on a state that cannot occur. Measured on
        `libero_10_swap/task_7`: `set_gripper(0.669)` and `set_gripper(0.933)`
        each burned 60 ticks and finished fully OPEN, 12% of the episode.

        YAM commands a real position target, so it is unaffected and the
        default stays permissive.
        """
        if getattr(self._adapter, "gripper_holds_intermediate", True):
            return True
        return min(abs(value), abs(1.0 - value)) <= 1e-6

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        arm = self._resolve_arm(arm)
        try:
            numeric = float(position)
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="gripper position must be finite and in [0, 1]",
                ),
            )
        if not self._gripper_target_is_holdable(numeric):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=(
                        "this gripper only holds fully open (1.0) or fully closed (0.0); "
                        f"{numeric} cannot be held -- use open_gripper() or close_gripper(), "
                        "and control approach clearance with geometry instead"
                    ),
                    details={"requested": numeric, "holdable": (0.0, 1.0)},
                ),
            )
        adapter_method = getattr(self._adapter, "set_gripper", None)
        if callable(adapter_method):
            try:
                result = adapter_method(numeric, arm=arm).public_view()
                self._commanded_gripper_positions[arm] = numeric
                return result
            # Best-effort arm resolution must not mask the embodiment error.
            except Exception as exc:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message=f"gripper execution failed: {exc}",
                    ),
                )
        state = self.get_robot_state()
        if arm not in state.joint_positions:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.NOT_FOUND, message=f"unknown arm {arm!r}"),
            )
        result = self.step(
            RobotAction(
                arms={
                    arm: ArmCommand(
                        mode="joint_position",
                        target=state.joint_positions[arm],
                        gripper_position=numeric,
                        embodiment=self._embodiment,
                    )
                }
            )
        )
        execution = ExecutionResult(
            ok=result.ok and not result.terminated and not result.truncated,
            steps_executed=1 if result.ok else 0,
            final_observation=result.observation,
            terminated=result.terminated,
            truncated=result.truncated,
            error=result.error
            or (
                self._terminal_error(result.terminated, result.truncated)
                if result.terminated or result.truncated
                else None
            ),
        ).public_view()
        self._commanded_gripper_positions[arm] = numeric
        return execution

    def localize_object(
        self,
        query: str,
        *,
        camera_name: str | None = None,
        target_frame: str | None = None,
    ) -> LocalizationResult:
        observation = self.get_observation()
        name = camera_name or self.default_camera
        segmentation_set = self._stage("segment_text", self._segment_text, name, query, observation)
        if not segmentation_set.ok:
            return LocalizationResult(ok=False, error=segmentation_set.error)
        segmentation = segmentation_set.segmentations[0]
        if target_frame is None:
            target_frame = observation.robot_state.base_frame
        try:
            point_cloud = self._stage(
                "mask_to_point_cloud",
                self._project_mask,
                segmentation,
                camera_name=name,
                target_frame=target_frame,
                observation=observation,
            )
            geometry = self._stage("estimate_geometry", self.estimate_geometry, point_cloud)
        except ValueError as exc:
            return LocalizationResult(
                ok=False,
                segmentation=segmentation,
                error=ApiError(
                    code=ErrorCode.POINT_CLOUD_FAILED,
                    message=f"object localization geometry failed: {exc}",
                ),
            )
        return LocalizationResult(
            ok=True,
            geometry=geometry,
            segmentation=segmentation,
            point_cloud=point_cloud,
        )

    def select_grasp(
        self,
        grasps: GraspSet,
        strategy: str = "top_down",
    ) -> GraspCandidate | None:
        if not grasps.ok:
            return None
        if strategy == "top_down":
            return select_top_down_grasp(grasps)
        if strategy == "highest_score":
            return max(grasps.grasps, key=lambda candidate: candidate.score, default=None)
        raise ValueError("strategy must be 'top_down' or 'highest_score'")

    def move_to_joints(
        self,
        target: np.ndarray,
        tolerance: float = 0.01,
        max_steps: int = 120,
        *,
        arm: str = "primary",
    ) -> ExecutionResult:
        arm = self._resolve_arm(arm)
        validation = self._motion_options(tolerance, max_steps)
        if validation is not None:
            return validation
        try:
            goal = np.asarray(target, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"joint target must be a finite arm joint vector: {exc}",
                ),
            )
        final_observation: Observation | None = None
        initial_state = self.get_robot_state()
        if arm not in initial_state.joint_positions:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.NOT_FOUND, message=f"unknown arm {arm!r}"),
            )
        joint_dimension = initial_state.joint_positions[arm].shape[0]
        if goal.shape != (joint_dimension,) or not bool(np.all(np.isfinite(goal))):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"joint target must be a finite {joint_dimension}-joint vector",
                ),
            )
        for step_index in range(max_steps):
            state = self.get_robot_state()
            gripper_target = self._commanded_gripper_position(state, arm)
            result = self.step(
                RobotAction(
                    arms={
                        arm: ArmCommand(
                            mode="joint_position",
                            target=goal,
                            gripper_position=gripper_target,
                            embodiment=self._embodiment,
                        )
                    }
                )
            )
            if not result.ok:
                return ExecutionResult(
                    ok=False,
                    steps_executed=step_index,
                    final_observation=final_observation,
                    error=result.error,
                )
            final_observation = result.observation
            final_state = final_observation.robot_state
            error_rad = float(np.max(np.abs(final_state.joint_positions[arm] - goal)))
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=step_index + 1,
                    final_observation=final_observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=self._terminal_error(result.terminated, result.truncated),
                    final_errors={arm: error_rad},
                )
            if error_rad <= tolerance:
                return ExecutionResult(
                    ok=True,
                    steps_executed=step_index + 1,
                    final_observation=final_observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    final_errors={arm: error_rad},
                    diagnostics={"joint_error_rad": error_rad},
                )
        return ExecutionResult(
            ok=False,
            steps_executed=step_index + 1 if final_observation is not None else 0,
            final_observation=final_observation,
            terminated=result.terminated if final_observation is not None else False,
            truncated=result.truncated if final_observation is not None else False,
            error=ApiError(
                code=ErrorCode.TIMEOUT,
                message=f"joint target did not converge within max_steps={max_steps}",
                details={
                    "reason": "joint_convergence_timeout",
                    "joint_error_rad": error_rad,
                    "tolerance_rad": tolerance,
                },
            ),
            final_errors={arm: error_rad},
        )

    def move_to_pose(
        self,
        target_pose: Pose | Mapping[str, Any],
        tolerance: float = 0.01,
        max_steps: int = 120,
        *,
        arm: str = "primary",
        strategy: MotionStrategy | None = None,
    ) -> ExecutionResult:
        """Move to a Pose or an absolute position/frame plus quaternion or rpy_deg."""
        arm = self._resolve_arm(arm)
        validation = self._motion_options(tolerance, max_steps)
        if validation is not None:
            return validation
        try:
            if isinstance(target_pose, Mapping):
                target_pose = pose_from_mapping(target_pose)
            elif not isinstance(target_pose, Pose):
                raise ValueError("target_pose must be a Pose or pose mapping")
        except (TypeError, ValueError) as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, str(exc)),
            )
        plan = self._stage(
            "plan_motion",
            self.plan_motion,
            target_pose,
            arm=arm,
            strategy=strategy,
        )
        if not plan.ok or plan.trajectory is None:
            return ExecutionResult(ok=False, steps_executed=0, error=plan.error)
        if len(plan.trajectory.joint_positions) > max_steps:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="planned trajectory exceeds max_steps",
                    details={
                        "waypoint_count": len(plan.trajectory.joint_positions),
                        "max_steps": max_steps,
                    },
                ),
            )
        execution = self._stage(
            "execute_trajectory",
            self.execute_trajectory,
            plan.trajectory,
        )
        if not execution.ok:
            return execution
        return self._validate_pose_execution(execution, target_pose, tolerance, arm)

    def _validate_pose_execution(
        self,
        execution: ExecutionResult,
        target_pose: Pose,
        tolerance: float,
        arm: str,
    ) -> ExecutionResult:
        final_state = self._execution_state(execution)
        current_pose = final_state.end_effector_poses[arm]
        if current_pose.frame != target_pose.frame:
            return ExecutionResult(
                ok=False,
                steps_executed=execution.steps_executed,
                final_observation=execution.final_observation,
                terminated=execution.terminated,
                truncated=execution.truncated,
                error=ApiError(
                    code=ErrorCode.CONTROLLER_FAILED,
                    message="final end-effector pose frame does not match target frame",
                ),
            )
        position_error_m = float(np.linalg.norm(current_pose.position - target_pose.position))
        quaternion_dot = float(
            np.clip(
                abs(np.dot(current_pose.quaternion_wxyz, target_pose.quaternion_wxyz)),
                0.0,
                1.0,
            )
        )
        orientation_error_rad = float(2.0 * np.arccos(quaternion_dot))
        orientation_tolerance_rad = float(np.deg2rad(10.0))
        if position_error_m > tolerance or orientation_error_rad > orientation_tolerance_rad:
            return ExecutionResult(
                ok=False,
                steps_executed=execution.steps_executed,
                final_observation=execution.final_observation,
                terminated=execution.terminated,
                truncated=execution.truncated,
                error=ApiError(
                    code=ErrorCode.CONTROLLER_FAILED,
                    message="final end-effector pose is outside target tolerance",
                    details={
                        "position_error_m": position_error_m,
                        "position_tolerance_m": tolerance,
                        "orientation_error_rad": orientation_error_rad,
                        "orientation_tolerance_rad": orientation_tolerance_rad,
                    },
                ),
                final_errors={arm: position_error_m},
            )
        return ExecutionResult(
            ok=True,
            steps_executed=execution.steps_executed,
            final_observation=execution.final_observation,
            terminated=execution.terminated,
            truncated=execution.truncated,
            final_errors={arm: position_error_m},
            diagnostics={
                "position_error_m": position_error_m,
                "orientation_error_rad": orientation_error_rad,
                "trajectory_executed": True,
                "joint_target_converged": execution.ok,
            },
        )

    def open_gripper(self, *, arm: str = "primary") -> ExecutionResult:
        arm = self._resolve_arm(arm)
        return self.set_gripper(1.0, arm=arm)

    def close_gripper(self, *, arm: str = "primary") -> ExecutionResult:
        arm = self._resolve_arm(arm)
        adapter_method = getattr(self._adapter, "close_gripper", None)
        if callable(adapter_method):
            try:
                result = adapter_method(arm=arm).public_view()
                self._commanded_gripper_positions[arm] = 0.0
                return result
            # Best-effort arm resolution must not mask the embodiment error.
            except Exception as exc:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message=f"gripper close failed: {exc}",
                    ),
                )
        return self.set_gripper(0.0, arm=arm)

    def go_home(self, *, arm: str = "primary") -> ExecutionResult:
        """Home through the embodiment's own primitive when it has one.

        An embodiment that ships a ``go_home`` knows something this layer does
        not: on a physical station it is one smoothly interpolated command to
        the pose the station is *calibrated* against, rather than a stepping
        loop toward whatever configuration this API happened to observe first.

        The fallback below remembers the state at construction and calls that
        home. For a harness run that is the post-reset pose, which is usually
        the same place -- but only usually, and it is not the profile's home.

        Delegating also fixes a silent no-op. This layer's arm names default to
        ``"primary"`` while a bimanual embodiment's may be ``left``/``right``,
        so the fallback path found no home joints for the default arm and
        returned ``UNSUPPORTED``: every bare ``go_home()`` in such a program did
        nothing and reported it in a result the program then discarded. An
        adapter's own ``go_home`` may home both arms and ignore the arm name by
        design, so the call works.
        """
        arm = self._resolve_arm(arm)
        adapter_method = getattr(self._adapter, "go_home", None)
        if callable(adapter_method):
            try:
                return adapter_method(arm=arm).public_view()
            # Best-effort arm resolution must not mask the embodiment error.
            except Exception as exc:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message=f"adapter go_home failed: {exc}",
                    ),
                )
        if arm not in self._home_joint_positions:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=f"no home joints are configured for arm {arm!r}",
                ),
            )
        # `move_to_joints` caps at 120 ticks and the joint servo droops: a step
        # past roughly 0.24 rad does not converge inside one cap. Out of a folded
        # pose that is not a corner case -- measured on RoboCasa
        # `drawer_utensil_sort:0`, homing from the pulled-open drawer needed three
        # consecutive calls (gaps 0.3724 -> 0.2743 -> 0.0029, 353 ticks) and one
        # call returned ok=False having moved most of the way. A caller that
        # believes that `ok=False` is left with the arm stranded, and every later
        # `move_to_pose` then crawls against a residual it cannot close.
        #
        # The calls accumulate, so repeat until it arrives. `go_home()` promises a
        # configuration, not an attempt, and unlike `move_to_joints` it takes no
        # `max_steps` for a caller to raise.
        home = self._home_joint_positions[arm]
        result = self.move_to_joints(home, arm=arm)
        attempts = 1
        while (
            not result.ok
            and not result.terminated
            and not result.truncated
            and attempts < self.GO_HOME_MAX_ATTEMPTS
        ):
            follow_up = self.move_to_joints(home, arm=arm)
            attempts += 1
            result = ExecutionResult(
                ok=follow_up.ok,
                steps_executed=result.steps_executed + follow_up.steps_executed,
                final_observation=follow_up.final_observation,
                terminated=follow_up.terminated,
                truncated=follow_up.truncated,
                error=follow_up.error,
                final_errors=follow_up.final_errors,
                diagnostics={"go_home_attempts": attempts},
            ).public_view()
            if follow_up.steps_executed == 0:
                break
        return result

    def register_tools(self, registry: ToolRegistry) -> ToolRegistry:
        """Register only approved public methods and metadata extensions."""
        tools = {
            "get_task_context": ("shared_atomic", "task"),
            "get_observation": ("shared_atomic", "observation"),
            "get_robot_state": ("shared_atomic", "state"),
            "segment_text": ("shared_atomic", "perception"),
            "segment_points": ("shared_atomic", "perception"),
            "mask_to_point_cloud": ("shared_atomic", "geometry"),
            "crop_point_cloud": ("shared_atomic", "geometry"),
            "estimate_geometry": ("shared_atomic", "geometry"),
            "generate_grasps": ("shared_atomic", "grasping"),
            "solve_ik": ("shared_atomic", "kinematics"),
            "plan_motion": ("shared_atomic", "planning"),
            "plan_synchronized_motion": ("shared_atomic", "planning"),
            "step": ("shared_atomic", "control"),
            "execute_trajectory": ("shared_atomic", "execution"),
            "set_gripper": ("shared_atomic", "control"),
            "set_grippers": ("shared_atomic", "control"),
            "localize_object": ("shared_high_level", "perception"),
            "select_grasp": ("shared_high_level", "grasping"),
            "move_to_joints": ("shared_high_level", "execution"),
            "move_to_pose": ("shared_high_level", "execution"),
            "move_synchronized": ("shared_high_level", "execution"),
            "open_gripper": ("shared_high_level", "control"),
            "close_gripper": ("shared_high_level", "control"),
            "go_home": ("shared_high_level", "execution"),
        }
        for name, (layer, capability) in tools.items():
            registry.register(
                getattr(self, name),
                name=name,
                layer=layer,
                capability=capability,
                public=True,
            )
        extensions = (
            ("get_task_metadata", "metadata"),
            ("get_controller_metadata", "metadata"),
            ("get_base_pose", "metadata"),
            ("navigate_to_pose", "execution"),
            ("move_torso", "execution"),
            ("reset_torso", "execution"),
            ("plan_standoff_pose", "geometry"),
            ("plan_approach_pose", "geometry"),
            ("base_pose_is_free", "geometry"),
        )
        for extension_name, capability in extensions:
            function = getattr(self._adapter, extension_name, None)
            if callable(function):
                registry.register(
                    function,
                    name=f"{self._adapter.embodiment}.{extension_name}",
                    layer="extension",
                    capability=capability,
                    public=True,
                )
        return registry

    def _stage(self, name: str, function: Any, *args: object, **kwargs: object) -> Any:
        """Record a nested composition stage without changing public APIs."""
        if self._recorder is None:
            return function(*args, **kwargs)
        with self._recorder.span(  # type: ignore[attr-defined]
            name, category="api", inputs={"args": args, "kwargs": kwargs}
        ) as span:
            result = function(*args, **kwargs)
            ok = getattr(result, "ok", True)
            return span.output(result, ok=ok if type(ok) is bool else True)

    def _planning_scene(
        self,
        *,
        voxel_size_m: float = 0.01,
        target_point_cloud: PointCloud | None = None,
    ) -> PlanningScene:
        """Fuse public RGB-D observations without reading native simulator geometry."""
        observation = self.get_observation()
        frame = observation.robot_state.base_frame
        if target_point_cloud is not None and target_point_cloud.frame != frame:
            raise ValueError(
                f"target point cloud is in frame {target_point_cloud.frame!r}, expected {frame!r}"
            )
        clouds: list[np.ndarray] = []
        for camera_name, camera in observation.cameras.items():
            # Only cameras that know where they are can contribute. A camera
            # whose pose is expressed in its own optical frame -- an uncalibrated
            # one, with no transform into the base frame -- cannot be fused into
            # a world-frame scene, and asking it to project raises rather than
            # returning camera-frame points mislabelled as base ones.
            #
            # Skipping it here is what keeps such a camera from breaking planning
            # for everyone else: a station that adds a wrist camera for its
            # images should not thereby lose motion planning.
            if camera.camera_pose.frame != frame:
                continue
            valid = np.isfinite(camera.depth_m) & (camera.depth_m > 0.0)
            if not bool(np.any(valid)):
                continue
            cloud = self._project_mask(
                valid,
                camera_name=camera_name,
                target_frame=frame,
                observation=observation,
            )
            clouds.append(cloud.points)
        if not clouds:
            raise ValueError("no valid public depth points are available for motion planning")
        points = np.concatenate(clouds, axis=0)
        points = self._crop_to_planning_workspace(points, frame)
        if target_point_cloud is not None:
            points = self._exclude_target_from_scene(points, target_point_cloud.points)
        cells = np.floor(points / voxel_size_m).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        downsampled = points[np.sort(first)]
        return PlanningScene(
            point_cloud=PointCloud(points=downsampled, frame=frame),
            voxel_size_m=voxel_size_m,
            timestamp_s=observation.timestamp_s,
        )

    @staticmethod
    def _exclude_target_from_scene(
        scene_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Remove one observed manipulation target while retaining surrounding obstacles."""
        lower = np.quantile(target_points, 0.01, axis=0) - np.array([0.01, 0.01, 0.005])
        upper = np.quantile(target_points, 0.99, axis=0) + np.array([0.01, 0.01, 0.01])
        inside = np.all((scene_points >= lower) & (scene_points <= upper), axis=1)
        filtered = scene_points[~inside]
        if not len(filtered):
            raise ValueError("target exclusion removed the entire observed planning scene")
        return filtered

    def _crop_to_planning_workspace(self, points: np.ndarray, frame: str) -> np.ndarray:
        """Drop depth returns the robot could never reach, if the embodiment says where.

        A depth camera sees the whole room. Points beyond the arm's reach cannot
        be collided with, so carrying them into the planner buys nothing and
        costs: they inflate the voxel grid the obstacle mesh is built over, and a
        few degenerate returns hundreds of metres out have been observed
        exploding that allocation outright.

        Purely an optimization, and deliberately not a safety mechanism: it
        removes geometry that is out of reach, never geometry that is in the way.
        Cropping tight enough to remove a real obstacle would be a bug, which is
        why the bound is derived from arm reach rather than from the table.

        Embodiments opt in by exposing ``planning_workspace()``. Adapters without
        it -- every simulator embodiment today -- get the full cloud exactly as
        before.
        """
        workspace = getattr(self._adapter, "planning_workspace", None)
        if not callable(workspace):
            return points
        bounds = workspace()
        if bounds is None:
            return points
        lower, upper = bounds
        lower = np.asarray(lower, dtype=np.float64).reshape(3)
        upper = np.asarray(upper, dtype=np.float64).reshape(3)
        if not bool(np.all(np.isfinite(lower))) or not bool(np.all(np.isfinite(upper))):
            raise ValueError("planning_workspace() bounds must be finite")
        if bool(np.any(lower >= upper)):
            raise ValueError("planning_workspace() lower bounds must be below upper bounds")
        keep = np.all((points >= lower) & (points <= upper), axis=1)
        cropped = points[keep]
        if not len(cropped):
            raise ValueError(
                f"planning workspace {lower.tolist()}..{upper.tolist()} in frame {frame!r} "
                "contains no observed points; the bounds or the camera extrinsics are wrong"
            )
        return cropped

    def _planning_scene_request(
        self,
        arm: str,
        *,
        target_point_cloud: PointCloud | None = None,
    ) -> PlanningScene:
        """Let an adapter express the public scene in a selected planner base."""
        scene = self._planning_scene(target_point_cloud=target_point_cloud)
        transform = getattr(self._adapter, "planning_scene_request", None)
        if not callable(transform):
            return scene
        transformed = transform(scene, arm)
        if not isinstance(transformed, PlanningScene):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("adapter planning_scene_request() returned an invalid value")
        return transformed

    def _planning_context(self, state: RobotState) -> RobotPlanningContext:
        factory = getattr(self._adapter, "get_planning_context", None)
        if callable(factory):
            context = factory()
            if not isinstance(context, RobotPlanningContext):
                # Preserve the invalid-data ValueError contract.
                raise ValueError("adapter get_planning_context() returned an invalid value")
            if set(context.arms) == set(state.arms):
                return context
        arms = state.arms
        identity = np.eye(4, dtype=np.float64)
        return RobotPlanningContext(
            embodiment=self._embodiment,
            model="panda" if len(arms) == 1 else "dual_panda",
            joint_names={arm: state.joint_names[arm] for arm in arms},
            base_transforms=dict.fromkeys(arms, identity),
            end_effector_links={
                arm: "panda_hand" if index == 0 else f"panda_hand_{index + 1}"
                for index, arm in enumerate(arms)
            },
        )

    def _camera(
        self,
        camera_name: str | None,
        observation: Observation | None,
    ) -> tuple[str, Any]:
        current = self.get_observation() if observation is None else observation
        name = camera_name or self.default_camera
        if name not in current.cameras:
            raise ValueError(f"observation has no camera {name!r}")
        return name, current.cameras[name]

    def _remember_home(self, state: RobotState) -> None:
        for arm, joints in state.joint_positions.items():
            self._home_joint_positions.setdefault(arm, np.array(joints, copy=True))

    def _commanded_gripper_position(self, state: RobotState, arm: str) -> float:
        return self._commanded_gripper_positions.setdefault(arm, state.gripper_positions[arm])

    def _execution_state(self, result: ExecutionResult) -> RobotState:
        if result.final_observation is not None:
            return result.final_observation.robot_state
        return self.get_robot_state()

    @staticmethod
    def _validate_plan_start(result: PlanResult, expected_start: RobotState) -> PlanResult:
        if not result.ok or result.trajectory is None:
            return result
        mismatch = CapApi._start_state_mismatch(result.trajectory.expected_start, expected_start)
        if mismatch is not None:
            return PlanResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.PLANNING_FAILED,
                    message="planner returned a trajectory bound to a different start state",
                    details={"mismatch": mismatch},
                ),
            )
        return result

    @staticmethod
    def _bind_plan_start(result: PlanResult, public_start: RobotState) -> PlanResult:
        """Bind a validated adapter-frame plan to its measured public start state."""
        if not result.ok or result.trajectory is None:
            return result
        return replace(
            result,
            trajectory=replace(result.trajectory, expected_start=public_start),
        )

    @staticmethod
    def _validate_synchronized_plan_start(
        result: SynchronizedPlanResult,
        expected_start: RobotState,
    ) -> SynchronizedPlanResult:
        if not result.ok or result.trajectory is None:
            return result
        mismatch = CapApi._start_state_mismatch(result.trajectory.expected_start, expected_start)
        if mismatch is not None:
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.PLANNING_FAILED,
                    message="planner returned a trajectory bound to a different start state",
                    details={"mismatch": mismatch},
                ),
            )
        return result

    @staticmethod
    def _start_state_mismatch(actual: RobotState, expected: RobotState) -> str | None:
        if set(actual.arms) != set(expected.arms):
            return "arm_names"
        if actual.base_frame != expected.base_frame:
            return "base_frame"
        for arm in expected.arms:
            if tuple(actual.joint_names[arm]) != tuple(expected.joint_names[arm]):
                return f"joint_names[{arm}]"
            if not np.array_equal(actual.joint_positions[arm], expected.joint_positions[arm]):
                return f"joint_positions[{arm}]"
            if actual.gripper_positions[arm] != expected.gripper_positions[arm]:
                return f"gripper_positions[{arm}]"
        return None

    def _stale_trajectory_result(
        self,
        trajectory: Trajectory | SynchronizedTrajectory,
    ) -> ExecutionResult | None:
        if not isinstance(trajectory, Trajectory | SynchronizedTrajectory):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="trajectory must be a Trajectory or SynchronizedTrajectory",
                ),
            )
        state = self.get_robot_state()
        expected_state = trajectory.expected_start
        if set(expected_state.arms) != set(state.arms):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.STALE_PLAN,
                    message="trajectory start arm set no longer matches the robot",
                ),
            )
        errors: dict[str, float] = {}
        gripper_errors: dict[str, float] = {}
        for arm in expected_state.arms:
            expected = expected_state.joint_positions[arm]
            if tuple(expected_state.joint_names[arm]) != tuple(state.joint_names[arm]):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.STALE_PLAN,
                        message=f"trajectory joint names no longer match arm {arm!r}",
                    ),
                )
            measured = state.joint_positions[arm]
            if expected.shape != measured.shape:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=f"trajectory dimension does not match arm {arm!r}",
                    ),
                )
            errors[arm] = float(np.max(np.abs(measured - expected)))
            gripper_errors[arm] = abs(
                state.gripper_positions[arm] - expected_state.gripper_positions[arm]
            )
        if errors and (
            max(errors.values()) > TRAJECTORY_START_TOLERANCE_RAD
            or max(gripper_errors.values()) > TRAJECTORY_START_GRIPPER_TOLERANCE
        ):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.STALE_PLAN,
                    message="trajectory start state no longer matches measured robot state",
                    details={
                        "start_error_rad": errors,
                        "gripper_start_error": gripper_errors,
                        "tolerance_rad": TRAJECTORY_START_TOLERANCE_RAD,
                        "gripper_tolerance": TRAJECTORY_START_GRIPPER_TOLERANCE,
                    },
                ),
                final_errors=errors,
            )
        return None

    @staticmethod
    def _terminal_error(terminated: bool, truncated: bool) -> ApiError:
        if truncated:
            return ApiError(
                code=ErrorCode.TRUNCATED,
                message="episode was truncated before control completed",
            )
        if terminated:
            return ApiError(
                code=ErrorCode.TERMINATED,
                message="episode terminated before control completed",
            )
        raise ValueError("a terminal error requires terminated or truncated")

    @staticmethod
    def _state_with_arm_joints(state: RobotState, arm: str, joints: np.ndarray) -> RobotState:
        positions = dict(state.joint_positions)
        positions[arm] = joints
        return RobotState(
            joint_positions=positions,
            joint_velocities=state.joint_velocities,
            end_effector_poses=state.end_effector_poses,
            gripper_positions=state.gripper_positions,
            base_frame=state.base_frame,
            joint_names=state.joint_names,
            timestamp_s=state.timestamp_s,
            embodiment=state.embodiment,
        )

    @staticmethod
    def _normalize_synchronized_targets(
        targets: Mapping[str, Pose | np.ndarray],
        state: RobotState,
    ) -> tuple[dict[str, Pose | np.ndarray] | None, ApiError | None]:
        if not isinstance(targets, Mapping) or not targets:
            return None, ApiError(
                code=ErrorCode.INVALID_REQUEST,
                message="targets must be a non-empty arm mapping",
            )
        if set(targets) != set(state.arms):
            return None, ApiError(
                code=ErrorCode.INVALID_REQUEST,
                message="synchronized targets must name every robot arm",
                details={"expected_arms": state.arms, "target_arms": tuple(targets)},
            )
        normalized: dict[str, Pose | np.ndarray] = {}
        for arm in state.arms:
            target = targets[arm]
            if isinstance(target, Pose):
                if target.frame != state.base_frame:
                    return None, ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=f"target pose for {arm!r} is not in the robot base frame",
                        details={
                            "target_frame": target.frame,
                            "base_frame": state.base_frame,
                        },
                    )
                normalized[arm] = target
                continue
            try:
                joint_target = np.asarray(target, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                return None, ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"joint target for {arm!r} must be a finite arm joint vector",
                    details={"error": str(exc)},
                )
            dimension = state.joint_positions[arm].shape[0]
            if joint_target.shape != (dimension,) or not bool(np.all(np.isfinite(joint_target))):
                return None, ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=(f"joint target for {arm!r} must be a finite {dimension}-joint vector"),
                )
            copied = np.array(joint_target, copy=True)
            copied.setflags(write=False)
            normalized[arm] = copied
        return normalized, None

    @staticmethod
    def _motion_options(tolerance: float, max_steps: int) -> ExecutionResult | None:
        if (
            isinstance(tolerance, bool)
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="tolerance and max_steps must be positive",
                ),
            )
        return None

    @staticmethod
    def _with_camera_name(result: SegmentationSet, camera_name: str) -> SegmentationSet:
        if not result.ok:
            return result
        return SegmentationSet(
            ok=True,
            segmentations=tuple(
                Segmentation(
                    mask=item.mask,
                    label=item.label,
                    score=item.score,
                    camera_name=camera_name,
                    frame=item.frame,
                    box_xyxy=item.box_xyxy,
                )
                for item in result.segmentations
            ),
            diagnostics=result.diagnostics,
        )

    def _controller_step_limits(
        self, start: np.ndarray, goal: np.ndarray, arm: str
    ) -> tuple[np.ndarray, float]:
        joint_dimension = start.shape[0]
        fallback = np.full(joint_dimension, 0.05, dtype=np.float64)
        dt_s = 0.05
        try:
            metadata_getter = getattr(self._adapter, "get_controller_metadata", None)
            if callable(metadata_getter):
                metadata = metadata_getter()
                low = np.asarray(metadata["action_lower_bounds"], dtype=np.float64)
                high = np.asarray(metadata["action_upper_bounds"], dtype=np.float64)
                frequency = float(metadata["control_frequency_hz"])
                dt_s = float(metadata.get("control_period_s", 1.0 / frequency))
                arm_names = tuple(metadata.get("arm_names", ("primary",)))
                offset = arm_names.index(arm) * (low.size // len(arm_names))
            else:
                low, high = self._adapter.action_bounds  # type: ignore[attr-defined]
                low = np.asarray(low, dtype=np.float64)
                high = np.asarray(high, dtype=np.float64)
                frequency = float(self._adapter.control_frequency)  # type: ignore[attr-defined]
                dt_s = float(getattr(self._adapter, "control_period_s", 1.0 / frequency))
                offset = 0
            direction_limits = np.where(
                goal - start >= 0.0,
                high[offset : offset + joint_dimension],
                -low[offset : offset + joint_dimension],
            )
            # Absolute-mode controllers report joint limits as action bounds;
            # those are goal ranges, not per-tick steps, so cap interpolation
            # steps at the conservative fallback either way.
            limits = np.minimum(direction_limits / frequency, fallback)
            if (
                low.size < offset + joint_dimension
                or high.size < offset + joint_dimension
                or not np.isfinite(frequency)
                or frequency <= 0.0
                or not bool(np.all(np.isfinite(limits)))
                or bool(np.any(limits <= 0.0))
                or not np.isfinite(dt_s)
                or dt_s <= 0.0
            ):
                raise ValueError("invalid controller metadata")
            return limits, dt_s
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return fallback, dt_s

    @staticmethod
    def _segmentation_failure(message: str) -> SegmentationSet:
        return SegmentationSet(
            ok=False,
            error=ApiError(code=ErrorCode.UNSUPPORTED, message=message),
        )

    @staticmethod
    def _plan_failure(
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> PlanResult:
        return PlanResult(
            ok=False,
            error=ApiError(code=code, message=message, details=details or {}),
        )


__all__ = ["CapApi"]
