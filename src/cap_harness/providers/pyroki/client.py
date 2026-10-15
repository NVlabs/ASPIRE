"""Typed PyRoki IK client for the provider service on port 8116."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import requests

from cap_harness.contracts import JOINT_DIMENSION, IKResult, Pose, RobotState
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import quaternion_wxyz_to_matrix

from ..http import HttpProviderClient, ProviderHttpError
from .wire import IK_PATH

DEFAULT_PYROKI_URL = "http://127.0.0.1:8116"
PYROKI_CONFIGURATION_DIMENSION = JOINT_DIMENSION + 1
PANDA_EEF_TO_HAND_OFFSET_M = np.array([0.0, 0.0, -0.1], dtype=np.float64)
MAX_POSITION_ERROR_M = 0.02
MAX_ORIENTATION_ERROR_RAD = 0.15


class PyRokiProvider:
    """Solve the exact requested pose, optionally seeded by current joints."""

    def __init__(
        self,
        base_url: str = DEFAULT_PYROKI_URL,
        *,
        timeout_s: float | tuple[float, float] = 15.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.1,
        session: requests.Session | None = None,
    ) -> None:
        self._http = HttpProviderClient(
            base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            session=session,
        )

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
        seed_joints: np.ndarray | None = None,
    ) -> IKResult:
        embodiment = robot_state.embodiment if isinstance(robot_state, RobotState) else "robosuite"
        if not isinstance(target_pose, Pose) or not isinstance(robot_state, RobotState):
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                "target_pose and robot_state must be typed contract values",
                arm=arm,
                embodiment=embodiment,
            )
        if robot_state.embodiment not in {"libero", "robosuite"}:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                f"PyRoki does not support embodiment {robot_state.embodiment!r}",
                arm=arm,
                embodiment=robot_state.embodiment,
            )
        if target_pose.frame != robot_state.base_frame:
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                "target pose frame does not match the robot base frame",
                arm=arm,
                embodiment=embodiment,
                details={
                    "target_frame": target_pose.frame,
                    "base_frame": robot_state.base_frame,
                },
            )
        if arm not in robot_state.joint_positions:
            return self._failure(
                ErrorCode.NOT_FOUND,
                f"robot state has no arm {arm!r}",
                arm=arm,
                embodiment=embodiment,
            )

        seed = robot_state.joint_positions[arm] if seed_joints is None else seed_joints
        try:
            seed = np.asarray(seed, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                "seed joints must be a finite seven-joint vector",
                arm=arm,
                embodiment=embodiment,
                details={"error": str(exc)},
            )
        if seed.shape != (JOINT_DIMENSION,) or not bool(np.all(np.isfinite(seed))):
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                "seed joints must be a finite seven-joint vector",
                arm=arm,
                embodiment=embodiment,
            )

        # The public pose targets LIBERO's robot0_eef frame, while the pinned
        # PyRoki service solves panda_hand. Convert only at the provider boundary.
        hand_position = target_pose.position + (
            quaternion_wxyz_to_matrix(target_pose.quaternion_wxyz) @ PANDA_EEF_TO_HAND_OFFSET_M
        )
        target_wxyz_xyz = np.concatenate((target_pose.quaternion_wxyz, hand_position))
        provider_seed = np.concatenate((seed, np.zeros(1, dtype=np.float64)))
        payload = {
            "target_pose_wxyz_xyz": target_wxyz_xyz.tolist(),
            "prev_cfg": provider_seed.tolist(),
        }
        try:
            response = self._http.post_json(IK_PATH, payload)
            provider_configuration = np.asarray(response.get("joint_positions"), dtype=np.float64)
            position_error_m = float(response.get("position_error_m"))
            orientation_error_rad = float(response.get("orientation_error_rad"))
            if provider_configuration.shape != (PYROKI_CONFIGURATION_DIMENSION,) or not bool(
                np.all(np.isfinite(provider_configuration))
            ):
                raise ValueError("joint_positions must be a finite eight-value Panda configuration")
            if not np.isfinite(position_error_m) or not np.isfinite(orientation_error_rad):
                raise ValueError("IK residuals must be finite")
            joints = provider_configuration[:JOINT_DIMENSION]
        except ProviderHttpError as exc:
            return IKResult(
                ok=False,
                arm=arm,
                error=exc.error,
                embodiment=embodiment,
            )
        except (TypeError, ValueError, KeyError) as exc:
            return self._failure(
                ErrorCode.IK_FAILED,
                "PyRoki returned an invalid IK response",
                arm=arm,
                embodiment=embodiment,
                details={"provider": "pyroki", "error": str(exc)},
            )

        if (
            position_error_m > MAX_POSITION_ERROR_M
            or orientation_error_rad > MAX_ORIENTATION_ERROR_RAD
        ):
            return self._failure(
                ErrorCode.IK_FAILED,
                "PyRoki solution exceeds the public pose residual limits",
                arm=arm,
                embodiment=embodiment,
                details={
                    "provider": "pyroki",
                    "position_error_m": position_error_m,
                    "orientation_error_rad": orientation_error_rad,
                    "max_position_error_m": MAX_POSITION_ERROR_M,
                    "max_orientation_error_rad": MAX_ORIENTATION_ERROR_RAD,
                },
            )
        return IKResult(
            ok=True,
            joint_positions=joints,
            arm=arm,
            embodiment=embodiment,
            diagnostics={
                "provider": "pyroki",
                "target_frame": target_pose.frame,
                "provider_target_link": "panda_hand",
                "position_error_m": position_error_m,
                "orientation_error_rad": orientation_error_rad,
            },
        )

    @staticmethod
    def _failure(
        code: ErrorCode,
        message: str,
        *,
        arm: str,
        embodiment: str,
        details: Mapping[str, object] | None = None,
    ) -> IKResult:
        return IKResult(
            ok=False,
            arm=arm,
            embodiment=embodiment,
            error=ApiError(code=code, message=message, details=details or {}),
        )


PyrokiProvider = PyRokiProvider
PyRokiClient = PyRokiProvider

__all__ = ["DEFAULT_PYROKI_URL", "PyRokiClient", "PyRokiProvider", "PyrokiProvider"]
