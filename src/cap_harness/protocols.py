"""Structural provider and environment boundaries for dependency injection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import (
    CameraObservation,
    ExecutionResult,
    GraspSet,
    IKResult,
    ObjectGeometry,
    Observation,
    PlanningScene,
    PlanResult,
    PointCloud,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    SegmentationSet,
    StepResult,
    SynchronizedPlanResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)


@runtime_checkable
class SegmentationProvider(Protocol):
    """Backend capable of text- and point-prompted image segmentation."""

    def segment_text(
        self,
        observation: CameraObservation,
        query: str,
    ) -> SegmentationSet:
        """Segment instances matching ``query`` in one camera observation."""
        ...

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        """Segment from Nx2 pixel prompts and optional binary point labels."""
        ...


@runtime_checkable
class GraspProvider(Protocol):
    """Backend that proposes framed grasp candidates from object geometry."""

    def generate_grasps(
        self,
        point_cloud: PointCloud,
        *,
        scene_point_cloud: PointCloud | None = None,
        geometry: ObjectGeometry | None = None,
        max_candidates: int | None = None,
    ) -> GraspSet:
        """Generate scored grasps for a segment, optionally within a full scene cloud."""
        ...


@runtime_checkable
class IKProvider(Protocol):
    """Backend that solves a requested framed end-effector pose exactly."""

    def solve_ik(
        self,
        target_pose: Pose,
        robot_state: RobotState,
        *,
        arm: str = "primary",
    ) -> IKResult:
        """Return joint positions or a typed failure without pose substitution."""
        ...


@runtime_checkable
class TrajectoryPlanningProvider(Protocol):
    """Backend that plans a trajectory to an already-resolved joint goal."""

    def plan_to_joints(
        self,
        robot_state: RobotState,
        target_joints: np.ndarray,
        *,
        arm: str = "primary",
        gripper_position: float | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> PlanResult:
        """Plan a timed joint trajectory or return a typed planning failure."""
        ...


@runtime_checkable
class IntegratedPosePlanningProvider(Protocol):
    """Backend that jointly resolves pose IK and plans a trajectory."""

    def plan_to_pose(
        self,
        robot_state: RobotState,
        target_pose: Pose,
        *,
        arm: str = "primary",
        gripper_position: float | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> PlanResult:
        """Plan directly to a Cartesian pose without an external IK result."""
        ...

    def plan_synchronized_motion(
        self,
        robot_state: RobotState,
        targets: Mapping[str, Pose],
        *,
        gripper_positions: Mapping[str, float] | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> SynchronizedPlanResult:
        """Plan equal-length, same-tick trajectories for multiple arms."""
        ...


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Embodiment boundary with one-control-period ``step`` semantics."""

    @property
    def embodiment(self) -> str:
        """Stable embodiment identifier, for example ``libero``."""
        ...

    def reset(self, task_ref: object, seed: int) -> Observation:
        """Select and reset one task to a deterministic initial state."""
        ...

    def get_task_context(self) -> TaskContext:
        """Return authoritative task metadata and runtime language."""
        ...

    def get_observation(self) -> Observation:
        """Return the current public observation without advancing control."""
        ...

    def get_robot_state(self) -> RobotState:
        """Return the current typed robot state without advancing control."""
        ...

    def step(self, action: RobotAction) -> StepResult:
        """Apply an action for exactly one control period."""
        ...

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        """Execute a fixed-rate trajectory, stopping on terminal outcomes."""
        ...

    def set_gripper(
        self,
        position: float,
        *,
        arm: str = "primary",
    ) -> ExecutionResult:
        """Set one normalized gripper position through the adapter boundary."""
        ...

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        """Set all arm grippers in exactly one shared control period."""
        ...

    def close(self) -> None:
        """Release adapter resources idempotently."""
        ...


SegmentationProviderProtocol = SegmentationProvider
GraspProviderProtocol = GraspProvider
IKProviderProtocol = IKProvider
TrajectoryPlannerProtocol = TrajectoryPlanningProvider
IntegratedPosePlannerProtocol = IntegratedPosePlanningProvider
EnvironmentAdapterProtocol = EnvironmentAdapter


__all__ = [
    "EnvironmentAdapter",
    "EnvironmentAdapterProtocol",
    "GraspProvider",
    "GraspProviderProtocol",
    "IKProvider",
    "IKProviderProtocol",
    "IntegratedPosePlannerProtocol",
    "IntegratedPosePlanningProvider",
    "SegmentationProvider",
    "SegmentationProviderProtocol",
    "TrajectoryPlannerProtocol",
    "TrajectoryPlanningProvider",
]
