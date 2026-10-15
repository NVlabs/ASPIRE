"""Typed HTTP client for isolated cuRobo IK and motion generation."""

from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np
import requests

from cap_harness.contracts import (
    JOINT_DIMENSIONS,
    IKResult,
    PlanningScene,
    PlanResult,
    Pose,
    RobotPlanningContext,
    RobotState,
    SynchronizedPlanResult,
    SynchronizedTrajectory,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import quaternion_wxyz_to_matrix

from ..http import HttpProviderClient, ProviderHttpError
from ..wire import decode_numpy, encode_numpy
from .wire import IK_PATH, PLAN_PATH, PLAN_SYNCHRONIZED_PATH

DEFAULT_CUROBO_URL = "http://127.0.0.1:8118"

# The service answers "no IK solution" / "no planning result" with HTTP 422 and
# a JSON body {"detail": {"message": ...}}. The generic HTTP client maps every
# 4xx to INVALID_REQUEST, so a genuine planner refusal read to programs as API
# misuse ("invalid_request") and hid the real message. Re-wrap that one status
# as the typed failure the caller expects, keeping the original details.
_SERVICE_REFUSAL_STATUS = 422


def _typed_service_failure(exc: ProviderHttpError, code: ErrorCode) -> ApiError:
    error = exc.error
    details = dict(error.details or {})
    if details.get("status_code") != _SERVICE_REFUSAL_STATUS:
        return error
    message = "cuRobo refused the request"
    raw = details.get("response")
    if isinstance(raw, str):
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            message = detail["message"]
        elif isinstance(detail, str):
            message = detail
    details["http_error_code"] = error.code.value
    return ApiError(code=code, message=message, details=details, recoverable=error.recoverable)


PANDA_EEF_TO_HAND_OFFSET_M = np.array([0.0, 0.0, -0.1], dtype=np.float64)
# cuRobo times trajectories to the real Panda's joint-speed limits; the sim's
# plan-tracking controller cannot follow that timing. Stretch plans to this
# fraction of full speed before resampling onto the control grid (skillgen's
# time_dilation_factor default).
DEFAULT_TIME_DILATION = 0.4


class CuRoboProvider:
    """Use cuRobo without importing CUDA extensions into the harness process."""

    def __init__(
        self,
        base_url: str = DEFAULT_CUROBO_URL,
        *,
        timeout_s: float | tuple[float, float] = 120.0,
        max_retries: int = 1,
        retry_backoff_s: float = 0.25,
        control_period_s: float = 0.05,
        time_dilation: float = DEFAULT_TIME_DILATION,
        session: requests.Session | None = None,
    ) -> None:
        if not np.isfinite(control_period_s) or control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive and finite")
        if not np.isfinite(time_dilation) or not 0.0 < time_dilation <= 1.0:
            raise ValueError("time_dilation must be in (0, 1]")
        self.time_dilation = float(time_dilation)
        self._http = HttpProviderClient(
            base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            session=session,
        )
        self.control_period_s = float(control_period_s)

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def health(self) -> bool:
        return self._http.health()

    def solve_ik(
        self,
        target_pose: Pose,
        robot_state: RobotState,
        *,
        arm: str = "primary",
    ) -> IKResult:
        return self.solve_ik_with_context(
            target_pose,
            robot_state,
            arm=arm,
            context=self._default_context(robot_state),
        )

    def solve_ik_with_context(
        self,
        target_pose: Pose,
        robot_state: RobotState,
        *,
        arm: str = "primary",
        context: RobotPlanningContext,
    ) -> IKResult:
        joint_dimension = JOINT_DIMENSIONS[robot_state.embodiment]
        error = self._validate_request(robot_state, target_pose, arm)
        if error is not None:
            return IKResult(
                ok=False,
                arm=arm,
                error=error,
                embodiment=robot_state.embodiment,
            )
        payload = self._base_payload(robot_state, context)
        payload.update(
            {"arm": arm, "target": self._encode_target(target_pose, robot_state.embodiment)}
        )
        try:
            response = self._http.post_json(IK_PATH, payload)
            joints = np.asarray(response.get("joint_positions"), dtype=np.float64)
            if joints.shape != (joint_dimension,) or not np.all(np.isfinite(joints)):
                raise ValueError(f"joint_positions must contain {joint_dimension} finite values")
        except ProviderHttpError as exc:
            return IKResult(
                ok=False,
                arm=arm,
                error=_typed_service_failure(exc, ErrorCode.IK_FAILED),
                embodiment=robot_state.embodiment,
            )
        except (TypeError, ValueError, KeyError) as exc:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    code=ErrorCode.IK_FAILED,
                    message="cuRobo returned an invalid IK response",
                    details={"provider": "curobo", "error": str(exc)},
                ),
            )
        return IKResult(
            ok=True,
            joint_positions=joints,
            arm=arm,
            embodiment=robot_state.embodiment,
            diagnostics={"provider": "curobo", "collision_aware": True},
        )

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
        return self._plan(
            robot_state,
            target_pose,
            arm=arm,
            gripper_position=gripper_position,
            scene=scene,
            context=context,
            time_dilation_factor=time_dilation_factor,
            interpolation_dt_s=interpolation_dt_s,
            maximum_trajectory_dt_s=maximum_trajectory_dt_s,
        )

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
        return self._plan(
            robot_state,
            target_joints,
            arm=arm,
            gripper_position=gripper_position,
            scene=scene,
            context=context,
            time_dilation_factor=time_dilation_factor,
            interpolation_dt_s=interpolation_dt_s,
            maximum_trajectory_dt_s=maximum_trajectory_dt_s,
        )

    def _plan(
        self,
        robot_state: RobotState,
        target: Pose | np.ndarray,
        *,
        arm: str,
        gripper_position: float | None,
        scene: PlanningScene | None,
        context: RobotPlanningContext | None,
        time_dilation_factor: float | None,
        interpolation_dt_s: float | None,
        maximum_trajectory_dt_s: float | None,
    ) -> PlanResult:
        error = self._validate_request(robot_state, target, arm)
        if error is not None:
            return PlanResult(ok=False, error=error)
        if scene is None or context is None:
            return self._plan_failure("cuRobo planning requires a scene and robot context")
        payload = self._base_payload(robot_state, context, scene)
        payload.update({"arm": arm, "target": self._encode_target(target, robot_state.embodiment)})
        dilation = self._time_dilation(time_dilation_factor)
        if interpolation_dt_s is not None:
            payload["interpolation_dt_s"] = self._positive_timing(
                interpolation_dt_s, "interpolation_dt_s"
            )
        if maximum_trajectory_dt_s is not None:
            payload["maximum_trajectory_dt_s"] = self._positive_timing(
                maximum_trajectory_dt_s, "maximum_trajectory_dt_s"
            )
        planning_mode = "pose" if isinstance(target, Pose) else "cspace"
        try:
            response = self._http.post_json(PLAN_PATH, payload)
            raw = decode_numpy(response.get("joint_positions_base64"))
            source_dt = float(response.get("dt_s"))
            waypoints = self._resample(
                raw,
                source_dt,
                JOINT_DIMENSIONS[robot_state.embodiment],
                time_dilation_factor=dilation,
            )
            gripper = robot_state.gripper_positions[arm]
            if gripper_position is not None:
                gripper = float(gripper_position)
            trajectory = Trajectory(
                joint_positions=waypoints,
                dt_s=self.control_period_s,
                joint_names=robot_state.joint_names[arm],
                planner=f"curobo_v2_{planning_mode}",
                collision_aware=True,
                expected_start=robot_state,
                arm=arm,
                gripper_positions=np.full(len(waypoints), gripper),
                embodiment=robot_state.embodiment,
            )
        except ProviderHttpError as exc:
            return PlanResult(
                ok=False, error=_typed_service_failure(exc, ErrorCode.PLANNING_FAILED)
            )
        except (TypeError, ValueError, KeyError) as exc:
            return self._plan_failure(
                "cuRobo returned an invalid motion plan",
                {"provider": "curobo", "error": str(exc)},
            )
        return PlanResult(
            ok=True,
            trajectory=trajectory,
            diagnostics={
                "provider": "curobo",
                "planning_mode": planning_mode,
                "source_dt_s": source_dt,
                "time_dilation_factor": dilation,
                "time_dilation": dilation,
                "interpolation_dt_s": interpolation_dt_s,
                "maximum_trajectory_dt_s": maximum_trajectory_dt_s,
                "scene_points": len(scene.point_cloud.points),
            },
        )

    def plan_synchronized_motion(
        self,
        robot_state: RobotState,
        targets: Mapping[str, Pose | np.ndarray],
        *,
        gripper_positions: Mapping[str, float] | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> SynchronizedPlanResult:
        if scene is None or context is None:
            return self._sync_failure("cuRobo planning requires a scene and robot context")
        if set(targets) != set(robot_state.arms):
            return self._sync_failure(
                "targets must name every robot arm", code=ErrorCode.INVALID_REQUEST
            )
        for arm, target in targets.items():
            error = self._validate_request(robot_state, target, arm)
            if error is not None:
                return SynchronizedPlanResult(ok=False, error=error)
            if not isinstance(target, Pose):
                return self._sync_failure(
                    "cuRobo V2 synchronized planning currently requires pose targets",
                    code=ErrorCode.UNSUPPORTED,
                )
        payload = self._base_payload(robot_state, context, scene)
        payload["targets"] = {
            arm: self._encode_target(target, robot_state.embodiment)
            for arm, target in targets.items()
        }
        dilation = self._time_dilation(time_dilation_factor)
        if interpolation_dt_s is not None:
            payload["interpolation_dt_s"] = self._positive_timing(
                interpolation_dt_s, "interpolation_dt_s"
            )
        if maximum_trajectory_dt_s is not None:
            payload["maximum_trajectory_dt_s"] = self._positive_timing(
                maximum_trajectory_dt_s, "maximum_trajectory_dt_s"
            )
        try:
            response = self._http.post_json(PLAN_SYNCHRONIZED_PATH, payload)
            source_dt = float(response.get("dt_s"))
            encoded = response.get("joint_positions_base64")
            if not isinstance(encoded, Mapping) or set(encoded) != set(targets):
                raise ValueError("joint_positions_base64 must match target arms")
            positions = {
                arm: self._resample(
                    decode_numpy(encoded[arm]),
                    source_dt,
                    JOINT_DIMENSIONS[robot_state.embodiment],
                    time_dilation_factor=dilation,
                )
                for arm in targets
            }
            counts = {len(value) for value in positions.values()}
            if len(counts) != 1:
                raise ValueError("resampled arm trajectories have unequal lengths")
            grippers = gripper_positions or robot_state.gripper_positions
            trajectory = SynchronizedTrajectory(
                joint_positions=positions,
                dt_s=self.control_period_s,
                joint_names={arm: robot_state.joint_names[arm] for arm in targets},
                planner="curobo_v2",
                collision_aware=True,
                expected_start=robot_state,
                embodiment=robot_state.embodiment,
                gripper_positions={
                    arm: np.full(next(iter(counts)), float(grippers[arm])) for arm in targets
                },
            )
        except ProviderHttpError as exc:
            return SynchronizedPlanResult(
                ok=False, error=_typed_service_failure(exc, ErrorCode.PLANNING_FAILED)
            )
        except (TypeError, ValueError, KeyError) as exc:
            return self._sync_failure(
                "cuRobo returned an invalid synchronized plan",
                {"provider": "curobo", "error": str(exc)},
            )
        return SynchronizedPlanResult(
            ok=True,
            trajectory=trajectory,
            diagnostics={
                "provider": "curobo",
                "source_dt_s": source_dt,
                "time_dilation_factor": dilation,
                "time_dilation": dilation,
                "interpolation_dt_s": interpolation_dt_s,
                "maximum_trajectory_dt_s": maximum_trajectory_dt_s,
            },
        )

    def _base_payload(
        self,
        state: RobotState,
        context: RobotPlanningContext,
        scene: PlanningScene | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "base_frame": state.base_frame,
            "model": context.model,
            "joint_positions": {arm: state.joint_positions[arm].tolist() for arm in state.arms},
            "joint_names": {arm: list(context.joint_names[arm]) for arm in state.arms},
            "base_transforms": {arm: context.base_transforms[arm].tolist() for arm in state.arms},
            "end_effector_links": dict(context.end_effector_links),
        }
        if scene is not None:
            if scene.frame != state.base_frame:
                raise ValueError("planning scene and robot state frames do not match")
            payload["scene_points_base64"] = encode_numpy(
                scene.point_cloud.points.astype(np.float32)
            )
            payload["voxel_size_m"] = scene.voxel_size_m
        return payload

    @staticmethod
    def _encode_target(target: Pose | np.ndarray, embodiment: str) -> dict[str, object]:
        if isinstance(target, Pose):
            if embodiment in ("libero", "robosuite"):
                hand_position = target.position + (
                    quaternion_wxyz_to_matrix(target.quaternion_wxyz) @ PANDA_EEF_TO_HAND_OFFSET_M
                )
            else:
                hand_position = target.position
            return {
                "type": "pose",
                "position": hand_position.tolist(),
                "quaternion_wxyz": target.quaternion_wxyz.tolist(),
                "frame": target.frame,
            }
        joints = np.asarray(target, dtype=np.float64)
        if joints.ndim != 1 or len(joints) == 0 or not np.all(np.isfinite(joints)):
            raise ValueError("joint target must be a non-empty finite vector")
        return {"type": "joints", "joint_positions": joints.tolist()}

    def _resample(
        self,
        values: np.ndarray,
        source_dt: float,
        joint_dimension: int = JOINT_DIMENSIONS["robosuite"],
        *,
        time_dilation_factor: float | None = None,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != joint_dimension or len(values) == 0:
            raise ValueError(f"trajectory must have shape (N, {joint_dimension})")
        if not np.all(np.isfinite(values)) or not np.isfinite(source_dt) or source_dt <= 0.0:
            raise ValueError("trajectory and dt must be finite and dt positive")
        if len(values) == 1:
            return values
        # Stretch the plan's timeline by the dilation before resampling, so the
        # whole trajectory slows uniformly (not just extra samples at the end).
        dilation = self._time_dilation(time_dilation_factor)
        effective_dt = source_dt / dilation
        duration = (len(values) - 1) * effective_dt
        count = max(2, int(np.ceil(duration / self.control_period_s)) + 1)
        source_times = np.arange(len(values), dtype=np.float64) * effective_dt
        target_times = np.linspace(0.0, duration, count)
        result = np.column_stack(
            [
                np.interp(target_times, source_times, values[:, index])
                for index in range(joint_dimension)
            ]
        )
        result[-1] = values[-1]
        return result

    def _time_dilation(self, value: float | None) -> float:
        if value is None:
            return self.time_dilation
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.0 < numeric <= 1.0:
            raise ValueError("time_dilation_factor must be in (0, 1]")
        return numeric

    @staticmethod
    def _positive_timing(value: float, name: str) -> float:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return numeric

    @staticmethod
    def _validate_request(
        state: RobotState, target: Pose | np.ndarray, arm: str
    ) -> ApiError | None:
        if not isinstance(state, RobotState):
            return ApiError(code=ErrorCode.INVALID_REQUEST, message="robot_state is invalid")
        if arm not in state.joint_positions:
            return ApiError(code=ErrorCode.NOT_FOUND, message=f"unknown arm {arm!r}")
        if isinstance(target, Pose):
            if target.frame != state.base_frame:
                return ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="target pose and robot state frames do not match",
                )
        else:
            joint_dimension = JOINT_DIMENSIONS[state.embodiment]
            try:
                joints = np.asarray(target, dtype=np.float64)
            except (TypeError, ValueError):
                joints = np.empty(0)
            if joints.shape != (joint_dimension,) or not np.all(np.isfinite(joints)):
                return ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"joint target must contain {joint_dimension} finite values",
                )
        return None

    @staticmethod
    def _default_context(state: RobotState) -> RobotPlanningContext:
        return RobotPlanningContext(
            embodiment=state.embodiment,
            model="panda" if len(state.arms) == 1 else "dual_panda",
            joint_names=state.joint_names,
            base_transforms={arm: np.eye(4) for arm in state.arms},
            end_effector_links={
                arm: "panda_hand" if index == 0 else f"panda_hand_{index + 1}"
                for index, arm in enumerate(state.arms)
            },
        )

    @staticmethod
    def _plan_failure(
        message: str,
        details: Mapping[str, object] | None = None,
        *,
        code: ErrorCode = ErrorCode.PLANNING_FAILED,
    ) -> PlanResult:
        return PlanResult(
            ok=False, error=ApiError(code=code, message=message, details=details or {})
        )

    @staticmethod
    def _sync_failure(
        message: str,
        details: Mapping[str, object] | None = None,
        *,
        code: ErrorCode = ErrorCode.PLANNING_FAILED,
    ) -> SynchronizedPlanResult:
        return SynchronizedPlanResult(
            ok=False, error=ApiError(code=code, message=message, details=details or {})
        )


CuRoboClient = CuRoboProvider

__all__ = ["DEFAULT_CUROBO_URL", "CuRoboClient", "CuRoboProvider"]
