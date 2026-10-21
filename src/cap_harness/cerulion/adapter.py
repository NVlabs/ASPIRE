"""``EnvironmentAdapter`` for a simulated arm running inside a Cerulion graph.

The plant is not in this process. It is a node in a Cerulion graph, and this
adapter reaches it through ``cerulion-rcpd``, the Cerulion MCP daemon, over
stdio. One control period is one ``arm.step`` tool call: the daemon checks the
call against its manifest, writes a receipt, publishes the call into the graph
and returns the plant's reply.

There is no perception. The camera is a synthetic top-and-side drawing of the
plant's reported state with zero depth, and the cube's position is handed to
the program in task metadata (``task_supplied_cube_position``).

Gripper convention: ASPIRE's normalized gripper is 1.0 open and 0.0 closed; the
plant's is 0.0 open and 1.0 closed. The adapter converts at this boundary only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.api import TRAJECTORY_START_GRIPPER_TOLERANCE, TRAJECTORY_START_TOLERANCE_RAD
from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode

from . import kinematics as kin
from .rcp_client import RcpClient, RcpError, RcpTransportError, rcpd_command
from .registry import CerulionTaskMetadata, CerulionTaskRegistry

EMBODIMENT = "cerulion"
ARM = "primary"
BASE_FRAME = "base"
CAMERA_NAME = "workspace"

#: Ticks a gripper command may take: full travel is 1.0 at 0.25 per tick.
GRIPPER_MAX_TICKS = 8
#: Extra ticks after the last trajectory waypoint for the arm to close any residual.
TRAJECTORY_SETTLE_TICKS = 5
TRAJECTORY_JOINT_TOLERANCE_RAD = 1e-3

#: The plant's own refusals, reported inside a successful tool result.
_PLANT_ERROR_CODES = {
    "joint_limit": (ErrorCode.INVALID_REQUEST, True),
    "estopped": (ErrorCode.SAFETY_INTERLOCK, False),
}
#: The daemon's refusals (``isError``): code -> (ErrorCode, recoverable).
_RCPD_ERROR_CODES = {
    "invalid_args": (ErrorCode.INVALID_REQUEST, True),
    "args_too_large": (ErrorCode.INVALID_REQUEST, True),
    "estopped": (ErrorCode.SAFETY_INTERLOCK, False),
    "timeout": (ErrorCode.TIMEOUT, False),
    "not_in_manifest": (ErrorCode.ADAPTER_FAILED, False),
    "publish_failed": (ErrorCode.ADAPTER_FAILED, False),
    "journal_stopped": (ErrorCode.ADAPTER_FAILED, False),
    "receipt_failed": (ErrorCode.ADAPTER_FAILED, False),
}

CAMERA_METADATA = {
    "kind": "synthetic_render",
    "calibrated": False,
    "depth": "zeros",
    "views": ["top", "side"],
    "description": "a drawing of the plant's reported state, not a sensor",
}


def _to_plant_gripper(aspire: float) -> float:
    return 1.0 - float(aspire)


def _to_aspire_gripper(plant: float) -> float:
    return min(1.0, max(0.0, 1.0 - float(plant)))


class CerulionAdapter:
    """Protocol-facing wrapper over the ``cerulion-rcpd`` tools of the ``arm_sim`` plant."""

    embodiment = EMBODIMENT
    registry_prefix = EMBODIMENT
    control_frequency = kin.CONTROL_FREQUENCY_HZ
    control_period_s = kin.CONTROL_PERIOD_S

    def __init__(
        self,
        client: Any = None,
        *,
        manifest: str | Path | None = None,
        receipts: str | Path | None = None,
        client_factory: Callable[[], Any] | None = None,
        registry: CerulionTaskRegistry | None = None,
        run_observer: object | None = None,
        camera_width: int = 256,
        camera_height: int = 128,
        horizon: int | None = None,
    ) -> None:
        del horizon  # accepted for run.py's generic adapter kwargs; the recorder enforces it
        self._latch_file = (
            None if receipts is None else Path(f"{Path(receipts).expanduser()}.estop")
        )
        if client is None:
            client = (
                client_factory()
                if client_factory is not None
                else RcpClient(rcpd_command(manifest, receipts))
            )
        self._client = client
        self._registry = registry or CerulionTaskRegistry()
        self._run_observer = run_observer
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._task_context: TaskContext | None = None
        self._state: dict[str, Any] | None = None
        self._stale_reason: str | None = None
        self._held_gripper = 1.0
        self._closed = False

    # -- identity ----------------------------------------------------------

    @property
    def arms(self) -> tuple[str, ...]:
        return (ARM,)

    @property
    def dof(self) -> int:
        return len(kin.JOINT_NAMES)

    def resolve_arm(self, arm: str) -> str:
        return str(arm)

    # -- the daemon boundary -----------------------------------------------

    def _call(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """One tool call. Returns the plant ``result`` and caches its state.

        A ``timeout`` refusal marks the cached state stale: the call was
        published and the plant may have stepped.
        """
        try:
            payload = self._client.call(tool, arguments)
        except RcpError as exc:
            if exc.code == "timeout":
                self._stale_reason = f"{tool} timed out and may have executed"
            raise
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise RcpTransportError(f"{tool}: reply carries no result object")
        state = result.get("state")
        if isinstance(state, Mapping):
            self._state = dict(state)
            self._stale_reason = None
        return dict(result)

    @staticmethod
    def _plant_error(result: Mapping[str, Any], operation: str) -> ApiError:
        error = result.get("error") or {}
        code = str(error.get("code", "plant_error"))
        mapped, recoverable = _PLANT_ERROR_CODES.get(code, (ErrorCode.EXECUTION_FAILED, True))
        return ApiError(
            code=mapped,
            message=f"{operation} refused by the plant: {code}: {error.get('message', '')}",
            recoverable=recoverable,
            details={"plant_code": code},
        )

    def _rcp_error(self, exc: Exception, operation: str) -> ApiError:
        if isinstance(exc, RcpError):
            mapped, recoverable = _RCPD_ERROR_CODES.get(exc.code, (ErrorCode.ADAPTER_FAILED, False))
            details: dict[str, object] = {"rcpd_code": exc.code}
            if exc.code == "timeout":
                details["may_have_executed"] = exc.may_have_executed
            if exc.code == "estopped" and self._latch_file is not None:
                details["latch_file"] = str(self._latch_file)
            return ApiError(
                code=mapped,
                message=f"{operation} refused by cerulion-rcpd: {exc.code}: {exc.message}",
                recoverable=recoverable,
                details=details,
            )
        return ApiError(
            code=ErrorCode.ADAPTER_FAILED,
            message=f"{operation} failed at the cerulion-rcpd transport: {exc}",
            recoverable=False,
        )

    def _estop_interlock(self, detail: str) -> ApiError:
        latch = "" if self._latch_file is None else f" (latch file {self._latch_file})"
        return ApiError(
            code=ErrorCode.SAFETY_INTERLOCK,
            message=(
                f"the e-stop is engaged{latch}: {detail}; an operator clears it by deleting "
                "the latch file and restarting cerulion-rcpd"
            ),
            recoverable=False,
            details={} if self._latch_file is None else {"latch_file": str(self._latch_file)},
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        metadata: CerulionTaskMetadata = self._registry.resolve(
            task_ref if task_ref is not None else "cerulion", None
        )
        self._state = None
        self._task_context = None
        self._stale_reason = None
        if self._latch_file is not None and self._latch_file.exists():
            raise self._estop_interlock("a leftover e-stop latch refuses arm.reset")
        try:
            result = self._call("arm.reset", {"seed": int(seed)})
        except RcpError as exc:
            if exc.code == "estopped":
                raise self._estop_interlock("cerulion-rcpd refused arm.reset") from exc
            raise self._rcp_error(exc, "reset") from exc
        if not result.get("ok", False) or self._state is None:
            raise ApiError(
                code=ErrorCode.ADAPTER_FAILED,
                message=f"arm.reset failed in the plant: {result.get('error')}",
                recoverable=False,
            )
        self._held_gripper = _to_aspire_gripper(self._state["gripper"])
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={
                "seed": int(seed),
                "perception": "none: the cube position below is supplied by the task",
                "task_supplied_cube_position": [float(v) for v in self._state["cube_position"]],
                "cube_half_size_m": kin.CUBE_HALF_SIZE_M,
                "kinematics": kin.kinematics_metadata(),
                "joint_names": list(kin.JOINT_NAMES),
                "lift_goal_m": kin.SUCCESS_LIFT_M,
                "gripper_convention": "1.0 open, 0.0 closed",
            },
        )
        observation = self.get_observation()
        if self._run_observer is not None:
            record = dict(self._task_context.metadata)
            record.update({"embodiment": self.embodiment, "transport": "cerulion-rcpd"})
            self._run_observer.on_reset(observation, record)
        return observation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    # -- observation -------------------------------------------------------

    def _require_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("the Cerulion arm has no state yet; call reset() first")
        if self._stale_reason is not None:
            raise RuntimeError(f"the cached arm state is stale: {self._stale_reason}")
        return self._state

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("no task context set; call reset() first")
        return self._task_context

    def get_task_metadata(self) -> dict[str, Any]:
        context = self.get_task_context()
        return {
            "suite": context.suite,
            "task_id": context.task_id,
            "task_name": context.task_name,
            "language": context.language,
            **dict(context.metadata),
        }

    def get_controller_metadata(self) -> dict[str, Any]:
        limit = np.full(self.dof, kin.JOINT_LIMIT_RAD)
        return {
            "action_lower_bounds": (-limit).tolist(),
            "action_upper_bounds": limit.tolist(),
            "control_frequency_hz": kin.CONTROL_FREQUENCY_HZ,
            "control_period_s": kin.CONTROL_PERIOD_S,
            "arm_names": [ARM],
            "joint_names": list(kin.JOINT_NAMES),
            "control_mode": "absolute_joint_position",
            "max_joint_step_rad": kin.MAX_JOINT_STEP_RAD,
            "max_gripper_step": kin.MAX_GRIPPER_STEP,
        }

    def get_robot_state(self) -> RobotState:
        state = self._require_state()
        q = np.asarray(state["q"], dtype=np.float64)
        qd = np.asarray(state.get("qd", np.zeros_like(q)), dtype=np.float64)
        quaternion = np.asarray(state["ee_quaternion_wxyz"], dtype=np.float64)
        quaternion = quaternion / np.linalg.norm(quaternion)
        return RobotState(
            joint_positions={ARM: q},
            joint_velocities={ARM: qd},
            end_effector_poses={
                ARM: Pose(
                    position=np.asarray(state["ee_position"], dtype=np.float64),
                    quaternion_wxyz=quaternion,
                    frame=BASE_FRAME,
                )
            },
            gripper_positions={ARM: _to_aspire_gripper(state["gripper"])},
            base_frame=BASE_FRAME,
            joint_names={ARM: kin.JOINT_NAMES},
            timestamp_s=float(state["step"]) * kin.CONTROL_PERIOD_S,
            embodiment=EMBODIMENT,
        )

    def get_observation(self) -> Observation:
        state = self._require_state()
        return Observation(
            cameras={CAMERA_NAME: self._camera(state)},
            robot_state=self.get_robot_state(),
            task_context=self._task_context,
            timestamp_s=float(state["step"]) * kin.CONTROL_PERIOD_S,
        )

    def check_success(self) -> bool:
        """Harness-side verdict from the plant's own flag; never shown to programs."""
        return bool(self._state is not None and self._state.get("task_success", False))

    def monitor_snapshot(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "cameras": {
                CAMERA_NAME: render_workspace(state, self._camera_width, self._camera_height)
            },
            "state": {
                "joint_positions": {ARM: list(state["q"])},
                "gripper_positions": {ARM: _to_aspire_gripper(state["gripper"])},
                "estopped": bool(state.get("estopped", False)),
            },
            "camera_metadata": {CAMERA_NAME: dict(CAMERA_METADATA)},
            "timestamp_s": float(state["step"]) * kin.CONTROL_PERIOD_S,
            "observation_seq": int(state["step"]),
        }

    def _camera(self, state: Mapping[str, Any]) -> CameraObservation:
        width, height = self._camera_width, self._camera_height
        # Placeholder intrinsics: the image is a drawing, not a projection.
        intrinsics = np.array(
            [[float(width), 0.0, width / 2.0], [0.0, float(width), height / 2.0], [0.0, 0.0, 1.0]]
        )
        return CameraObservation(
            rgb=render_workspace(state, width, height),
            depth_m=np.zeros((height, width), dtype=np.float64),
            intrinsics=intrinsics,
            frame="synthetic_render",
            camera_pose=Pose(
                position=np.zeros(3),
                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                frame=BASE_FRAME,
            ),
            timestamp_s=float(state["step"]) * kin.CONTROL_PERIOD_S,
        )

    # -- actuation ---------------------------------------------------------

    def _recorded_action(self, q_target: np.ndarray, gripper: float) -> RobotAction:
        return RobotAction(
            arms={
                ARM: ArmCommand(
                    mode="joint_position",
                    target=np.asarray(q_target, dtype=np.float64),
                    gripper_position=float(gripper),
                    embodiment=EMBODIMENT,
                )
            }
        )

    def _tick(self, q_target: np.ndarray, gripper: float, operation: str) -> StepResult:
        """Exactly one control period: one ``arm.step`` call, recorded once."""
        action = self._recorded_action(q_target, gripper)
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        try:
            result = self._call(
                "arm.step",
                {
                    "q_target": [float(v) for v in q_target],
                    "gripper": _to_plant_gripper(gripper),
                },
            )
        except (RcpError, RcpTransportError) as exc:
            step = StepResult(ok=False, error=self._rcp_error(exc, operation))
        else:
            if result.get("ok", False):
                self._held_gripper = float(gripper)
                step = StepResult(
                    ok=True,
                    observation=self.get_observation(),
                    diagnostics={"plant_step": int(self._require_state()["step"])},
                )
            else:
                step = StepResult(ok=False, error=self._plant_error(result, operation))
        if self._run_observer is not None:
            self._run_observer.after_step(action, step)
        return step

    def step(self, action: RobotAction) -> StepResult:
        if not isinstance(action, RobotAction) or set(action.arms) != {ARM}:
            return StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"the Cerulion arm takes one command for arm {ARM!r}",
                ),
            )
        if action.embodiment != EMBODIMENT:
            return StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"action embodiment {action.embodiment!r} is not {EMBODIMENT!r}",
                ),
            )
        command = action.arms[ARM]
        if command.mode != "joint_position":
            return StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.UNSUPPORTED,
                    message=f"unsupported command mode {command.mode!r}; use joint_position",
                ),
            )
        gripper = (
            self._held_gripper if command.gripper_position is None else command.gripper_position
        )
        return self._tick(np.asarray(command.target, dtype=np.float64), gripper, "step")

    def _stale_plan(self, expected: RobotState) -> ApiError | None:
        state = self.get_robot_state()
        if set(expected.arms) != {ARM} or tuple(expected.joint_names[ARM]) != tuple(
            state.joint_names[ARM]
        ):
            return ApiError(ErrorCode.STALE_PLAN, "trajectory start metadata is stale")
        joint_error = float(
            np.max(np.abs(expected.joint_positions[ARM] - state.joint_positions[ARM]))
        )
        gripper_error = abs(expected.gripper_positions[ARM] - state.gripper_positions[ARM])
        if (
            joint_error > TRAJECTORY_START_TOLERANCE_RAD
            or gripper_error > TRAJECTORY_START_GRIPPER_TOLERANCE
        ):
            return ApiError(
                ErrorCode.STALE_PLAN,
                "trajectory start no longer matches the measured robot state",
                details={
                    "joint_error_rad": {ARM: joint_error},
                    "gripper_error": {ARM: gripper_error},
                    "joint_tolerance_rad": TRAJECTORY_START_TOLERANCE_RAD,
                    "gripper_tolerance": TRAJECTORY_START_GRIPPER_TOLERANCE,
                },
            )
        return None

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        def refused(code: ErrorCode, message: str) -> ExecutionResult:
            return ExecutionResult(
                ok=False, steps_executed=0, error=ApiError(code=code, message=message)
            )

        if isinstance(trajectory, SynchronizedTrajectory):
            if set(trajectory.joint_positions) != {ARM}:
                return refused(ErrorCode.INVALID_REQUEST, f"the Cerulion arm has one arm, {ARM!r}")
            waypoints = np.asarray(trajectory.joint_positions[ARM], dtype=np.float64)
            grippers = (
                None
                if trajectory.gripper_positions is None
                else trajectory.gripper_positions.get(ARM)
            )
        elif isinstance(trajectory, Trajectory):
            if trajectory.arm != ARM:
                return refused(ErrorCode.NOT_FOUND, f"unknown arm {trajectory.arm!r}")
            waypoints = np.asarray(trajectory.joint_positions, dtype=np.float64)
            grippers = trajectory.gripper_positions
        else:
            return refused(ErrorCode.INVALID_REQUEST, "expected a Trajectory")
        if trajectory.embodiment != EMBODIMENT:
            return refused(
                ErrorCode.INVALID_REQUEST,
                f"trajectory embodiment {trajectory.embodiment!r} is not {EMBODIMENT!r}",
            )
        if not np.isclose(trajectory.dt_s, kin.CONTROL_PERIOD_S, rtol=0.0, atol=1e-9):
            return refused(
                ErrorCode.INVALID_REQUEST,
                f"trajectory dt_s must equal the control period {kin.CONTROL_PERIOD_S}",
            )
        stale = self._stale_plan(trajectory.expected_start)
        if stale is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=stale)
        executed = 0
        for index, waypoint in enumerate(waypoints):
            gripper = self._held_gripper if grippers is None else float(grippers[index])
            step = self._tick(waypoint, gripper, "execute_trajectory")
            if not step.ok:
                return ExecutionResult(ok=False, steps_executed=executed, error=step.error)
            executed += 1
        goal = waypoints[-1]
        for _ in range(TRAJECTORY_SETTLE_TICKS):
            error = np.abs(np.asarray(self._require_state()["q"]) - goal)
            if float(np.max(error)) <= TRAJECTORY_JOINT_TOLERANCE_RAD:
                break
            step = self._tick(goal, self._held_gripper, "execute_trajectory")
            if not step.ok:
                return ExecutionResult(ok=False, steps_executed=executed, error=step.error)
            executed += 1
        final_error = float(np.max(np.abs(np.asarray(self._require_state()["q"]) - goal)))
        if final_error > TRAJECTORY_JOINT_TOLERANCE_RAD:
            return ExecutionResult(
                ok=False,
                steps_executed=executed,
                final_observation=self.get_observation(),
                error=ApiError(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"arm stopped {final_error:.4f} rad from the final waypoint",
                ),
                final_errors={ARM: final_error},
            )
        return ExecutionResult(
            ok=True,
            steps_executed=executed,
            final_observation=self.get_observation(),
            final_errors={ARM: final_error},
        )

    def set_gripper(self, position: float, *, arm: str = ARM) -> ExecutionResult:
        if arm != ARM:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.NOT_FOUND, message=f"unknown arm {arm!r}"),
            )
        target = float(position)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST, message="gripper position must be in [0, 1]"
                ),
            )
        hold = np.asarray(self._require_state()["q"], dtype=np.float64)
        executed = 0
        reached = False
        for _ in range(GRIPPER_MAX_TICKS):
            step = self._tick(hold, target, "set_gripper")
            if not step.ok:
                return ExecutionResult(ok=False, steps_executed=executed, error=step.error)
            executed += 1
            current = _to_aspire_gripper(self._require_state()["gripper"])
            if abs(current - target) <= 1e-9:
                reached = True
                break
        if not reached:
            return ExecutionResult(
                ok=False,
                steps_executed=executed,
                final_observation=self.get_observation(),
                error=ApiError(
                    code=ErrorCode.EXECUTION_FAILED,
                    message=f"gripper did not reach {target} within {GRIPPER_MAX_TICKS} ticks",
                ),
            )
        return ExecutionResult(
            ok=True, steps_executed=executed, final_observation=self.get_observation()
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        if set(positions) != {ARM}:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST, message=f"positions must name arm {ARM!r}"
                ),
            )
        return self.set_gripper(positions[ARM], arm=ARM)

    def estop(self, reason: str) -> dict[str, Any]:
        """Engage the e-stop through cerulion-rcpd. Not exposed to programs.

        The reply is ``{estopped, graph_acknowledged, graph_error, receipt,
        result}``; ``result`` is null when the graph did not answer.
        """
        payload = self._client.call("estop", {"reason": str(reason)})
        result = payload.get("result")
        if isinstance(result, Mapping) and isinstance(result.get("state"), Mapping):
            self._state = dict(result["state"])
            self._stale_reason = None
        return {
            "estopped": bool(payload.get("estopped", False)),
            "graph_acknowledged": bool(payload.get("graph_acknowledged", False)),
            "graph_error": payload.get("graph_error"),
            "receipt": payload.get("receipt"),
        }


# -- rendering --------------------------------------------------------------

_BACKGROUND = np.array([245, 245, 240], dtype=np.uint8)
_TABLE = np.array([200, 190, 170], dtype=np.uint8)
_LINK = np.array([40, 70, 140], dtype=np.uint8)
_CUBE = np.array([40, 160, 70], dtype=np.uint8)
_CUBE_HELD = np.array([220, 120, 30], dtype=np.uint8)
_DIVIDER = np.array([160, 160, 160], dtype=np.uint8)


def _disk(image: np.ndarray, u: float, v: float, radius: int, color: np.ndarray) -> None:
    height, width = image.shape[:2]
    cu, cv = round(u), round(v)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            if du * du + dv * dv <= radius * radius:
                x, y = cu + du, cv + dv
                if 0 <= x < width and 0 <= y < height:
                    image[y, x] = color


def _segment(image: np.ndarray, a: tuple[float, float], b: tuple[float, float], color) -> None:
    samples = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1])) + 1)
    for t in np.linspace(0.0, 1.0, samples):
        _disk(image, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 1, color)


def render_workspace(state: Mapping[str, Any], width: int, height: int) -> np.ndarray:
    """Top view (left half) and side view (right half) of the arm and cube, uint8 RGB."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = _BACKGROUND
    half = width // 2
    scale = min(half, height) / 0.8
    points = kin.chain_points(np.asarray(state["q"], dtype=np.float64))
    cube = np.asarray(state["cube_position"], dtype=np.float64)
    cube_color = _CUBE_HELD if state.get("grasped") else _CUBE
    cube_px = max(1, round(kin.CUBE_HALF_SIZE_M * scale))

    def top(p: np.ndarray) -> tuple[float, float]:
        return (half * 0.15 + p[0] * scale, height / 2.0 - p[1] * scale)

    def side(p: np.ndarray) -> tuple[float, float]:
        return (half + half * 0.15 + p[0] * scale, height * 0.9 - p[2] * scale)

    image[round(height * 0.9) :, half:] = _TABLE
    image[:, half - 1 : half + 1] = _DIVIDER
    for project in (top, side):
        u, v = project(cube)
        top_row, left = max(0, round(v) - cube_px), max(0, round(u) - cube_px)
        image[top_row : round(v) + cube_px + 1, left : round(u) + cube_px + 1] = cube_color
        for a, b in zip(points[:-1], points[1:], strict=True):
            _segment(image, project(a), project(b), _LINK)
    return image


__all__ = ["ARM", "CAMERA_NAME", "EMBODIMENT", "CerulionAdapter", "render_workspace"]
