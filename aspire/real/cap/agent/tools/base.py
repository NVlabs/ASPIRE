"""Base class for CAP tools and shared result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standardised return value from every tool."""

    success: bool
    data: Any = None
    error: str | None = None


@dataclass
class ToolParameter:
    """Schema for a single tool parameter."""

    name: str
    type: str  # "str", "float", "list[float]", etc.
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """Abstract base for all CAP tools.

    Subclasses must set ``name``, ``description``, and ``parameters``,
    and implement ``execute``.
    """

    name: str
    description: str
    parameters: list[ToolParameter]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given keyword arguments."""

    def schema(self) -> dict[str, Any]:
        """Return a JSON-serialisable schema for LLM tool-use."""
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.default is not None:
                props[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


# -- Result data types used across tools ------------------------------------


@dataclass
class ArmState:
    """State of a single robot arm."""

    joint_pos: list[float]
    gripper_pos: float
    ee_pos: list[float]
    ee_quat: list[float]
    ee_rpy: list[float]


@dataclass
class RobotState:
    """Snapshot of robot arms returned by ``get_robot_state``.

    ``arms`` is keyed by side name (e.g. "left", "right").
    Backward-compatible properties provide direct access via
    ``state.left_joint_pos``, ``state.right_ee_pos``, etc.
    """

    arms: dict[str, ArmState]

    def _arm(self, side: str) -> ArmState | None:
        return self.arms.get(side)

    # --- Backward-compatible properties (return empty defaults when arm missing) ---
    @property
    def left_joint_pos(self) -> list[float]:
        a = self._arm("left")
        return a.joint_pos if a else []

    @property
    def left_gripper_pos(self) -> float:
        a = self._arm("left")
        return a.gripper_pos if a else 0.0

    @property
    def left_ee_pos(self) -> list[float]:
        a = self._arm("left")
        return a.ee_pos if a else []

    @property
    def left_ee_quat(self) -> list[float]:
        a = self._arm("left")
        return a.ee_quat if a else []

    @property
    def left_ee_rpy(self) -> list[float]:
        a = self._arm("left")
        return a.ee_rpy if a else []

    @property
    def right_joint_pos(self) -> list[float]:
        a = self._arm("right")
        return a.joint_pos if a else []

    @property
    def right_gripper_pos(self) -> float:
        a = self._arm("right")
        return a.gripper_pos if a else 0.0

    @property
    def right_ee_pos(self) -> list[float]:
        a = self._arm("right")
        return a.ee_pos if a else []

    @property
    def right_ee_quat(self) -> list[float]:
        a = self._arm("right")
        return a.ee_quat if a else []

    @property
    def right_ee_rpy(self) -> list[float]:
        a = self._arm("right")
        return a.ee_rpy if a else []


@dataclass
class MoveResult:
    """Returned by ``_ik_servo`` (internal IK move)."""

    reached: bool
    feasible: bool = True
    cmd_pos_err: float | None = None
    final_pos: list[float] = field(default_factory=list)
    final_quat: list[float] = field(default_factory=list)


@dataclass
class Detection3D:
    """Single 3D detection returned by ``detect_object``."""

    label: str
    score: float
    box_2d: list[float]
    position_3d: list[float]
    quaternion_xyzw: list[float] = field(default_factory=list)
    rpy: list[float] = field(default_factory=list)
    half_extents: list[float] = field(default_factory=list)
    position_3d_source: str | None = None
    vis_b64: str | None = None  # annotated visualization (base64 JPEG) from BundleSDF


@dataclass
class SkillResult:
    """Returned by ``execute_skill``."""

    success: bool
    steps_executed: int = 0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class FreespaceBatchCandidate:
    """Single ranked candidate returned by batched ``freespace_move`` evaluation.

    ``planner_status`` distinguishes three backend-level outcomes:

    - ``"Success"``: an executable trajectory was produced
    - ``"IK_Failed"``: IK returned a finite residual but no trajectory
    - ``"Planning_Failed"`` (or another planner failure string): planning failed

    ``motion_plan_error`` is intentionally tri-state so callers can separate these
    cases without losing the finite IK residual from ``IK_Failed`` rows:

    - ``False`` => executable trajectory available
    - ``None`` => IK failed with finite residuals, no trajectory
    - ``True`` => planning / trajectory generation failed
    """

    rank: int
    source_index: int | None = None
    position: list[float] = field(default_factory=list)
    rpy: list[float] = field(default_factory=list)
    score: float = 0.0
    width: float = 0.0
    ik_error_m: float | None = None
    ik_rot_error_deg: float | None = None
    within_ik_threshold: bool | None = None
    planner_status: str | None = None
    motion_plan_error: bool | None = None
    motion_plan_reason: str | None = None
    trajectory_cache_key: str | None = None
    trajectory_steps: int = 0

    @property
    def is_executable(self) -> bool:
        return self.planner_status == "Success" and self.motion_plan_error is False

    @property
    def is_ik_failed(self) -> bool:
        return self.planner_status == "IK_Failed" and self.motion_plan_error is None

    @property
    def is_planning_failed(self) -> bool:
        return self.motion_plan_error is True


@dataclass
class FreespaceResult:
    """Returned by ``freespace_move``.

    Attributes:
        status: "Success", "IK_Failed", "Planning_Failed", or "Execution_Failed".
        ik_error_m: Maximum IK position error in metres (0.0 when IK was not attempted).
        final_pos_error_m: Maximum final Cartesian position error in metres after the
            planned trajectory's terminal waypoint.
        final_rot_error_deg: Maximum final orientation error in degrees after the
            planned trajectory's terminal waypoint.
        final_pose_error: Combined position+orientation error metric used for
            preview-time ranking.
        trajectory_steps: Number of waypoints in the planned trajectory.
        executed: Whether the trajectory was actually executed on the robot.
        reason: Human-readable explanation when status is not "Success".
        planning_mode: ``single`` for one EE target, ``batch`` for ranked batched EE targets,
            or ``cached`` when executing a previously cached trajectory.
        batch_candidates: Ranked candidates when ``grasp_candidates`` batch mode is used.
    """

    status: str
    ik_error_m: float = 0.0
    final_pos_error_m: float = 0.0
    final_rot_error_deg: float = 0.0
    final_pose_error: float = 0.0
    trajectory_steps: int = 0
    trajectory_cache_key: str | None = None
    executed: bool = True
    reason: str = ""
    planning_mode: str = "single"
    side: str | None = None
    batch_candidates: list[FreespaceBatchCandidate] = field(default_factory=list)
    best_candidate: FreespaceBatchCandidate | None = None
    input_candidate_count: int = 0
    evaluated_candidate_count: int = 0
    truncated_input_count: int = 0
    batch_attempted: bool = False
    batch_error: str | None = None
    timing_total_ms: float = 0.0
    timing_plan_eval_ms: float = 0.0
    timing_postprocess_ms: float = 0.0
    timing_rank_ms: float = 0.0
    curobo_solve_time_ms: float = 0.0
    curobo_total_time_ms: float = 0.0
    curobo_graph_time_ms: float = 0.0
    curobo_ik_time_ms: float = 0.0
