# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision-free free-space movement via motion planning.

Defaults to cuRobo via ``experimental.portal_motion_planner.PortalMotionPlanner``.
Can also use ``experimental.motion_planner.YamMotionPlanner`` (RRT-Connect).
Executes planned trajectories through cap_server joint-trajectory RPCs.

Supports single-arm and bimanual simultaneous movement.

Coordinate frame contract
-------------------------
All target poses are in the **robot world frame**:

    +X  forward  (toward the work table)
    +Y  left     (toward the left arm)
    +Z  up       (sky)
    Origin: URDF base_link (floor level, centred between the two arm bases)

Positions are in metres, orientations are RPY [roll, pitch, yaw] in degrees.
The motion planner uses the same Pinocchio FK/IK as cap_server, so
all frames are internally consistent.

Usage from generated code::

    result = freespace_move(
        left_target_pos=[0.3, 0.1, 0.8],
        left_target_rpy=[0, 90, 0],
    )
    if result.status != "Success":
        print(f"Failed: {result.reason}")
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
import os
from typing import Any

import numpy as np

from cap.config import CAP_SERVER_PORT, CONTROL_PERIOD_S
from cap.constants.planning import (
    DEFAULT_IK_POSITION_THRESHOLD_M as SHARED_DEFAULT_IK_POSITION_THRESHOLD_M,
    DEFAULT_IK_ROT_THRESHOLD_DEG as SHARED_DEFAULT_IK_ROT_THRESHOLD_DEG,
)
from cap.agent.tools.base import (
    FreespaceBatchCandidate,
    FreespaceResult,
    Tool,
    ToolParameter,
    ToolResult,
)
from robot.yam.kinematics import YamKinematics

logger = logging.getLogger(__name__)

# IK error thresholds
_DEFAULT_IK_THRESHOLD = float(SHARED_DEFAULT_IK_POSITION_THRESHOLD_M)
_DEFAULT_IK_ROT_THRESHOLD_DEG = float(SHARED_DEFAULT_IK_ROT_THRESHOLD_DEG)
_MAX_IK_THRESHOLD = 0.10  # 10 cm — beyond this we warn
_PLANNING_SPEED_MIN = 0.05
_PLANNING_SPEED_MAX = 3.0
_DEFAULT_PLANNING_SPEED = 1.5
_DEFAULT_IK_XYZ_WEIGHT = 1.0
_DEFAULT_IK_RPY_WEIGHT = 0.3
_DEFAULT_SOLVER_SPEED = "fast"
_GEOMETRIC_MAX_JOINT_STEP = 0.03
_PLANNER_CACHE_MAX_SIZE = 8
_DEFAULT_BACKEND = "curobo"
_INACTIVE_ARM_POS_TOL_M = 0.002
_INACTIVE_ARM_ROT_TOL_DEG = 2.0
_DEFAULT_BATCH_TOP_K = 16
_DEFAULT_BATCH_SOLVER_SPEED = _DEFAULT_SOLVER_SPEED
_DEFAULT_BATCH_VALIDATE_TRAJECTORY = False
_TRAJECTORY_CACHE_MAX_SIZE = 128
_TRAJECTORY_STATE_TOLERANCE_RAD = 0.05


class FreespaceMoveTool(Tool):
    """Move robot arm(s) to target EE pose via collision-free motion planning.

    Replaces linear-interpolation based free-space movement with a proper
    sampling-based motion planner that avoids inter-arm and arm-environment
    collisions.

    Provide targets for one arm (single-arm move) or both arms (bimanual).
    Targets not provided keep the arm at its current pose.
    """

    name = "freespace_move"
    description = (
        "Move robot arm(s) to target end-effector pose(s) via collision-free "
        "motion planning. Defaults to cuRobo; use backend='rrt-connect' to override. "
        "Supports single-arm or synchronized "
        "bimanual movement. "
        "Provide left_target_pos/rpy and/or right_target_pos/rpy. "
        "For a single-arm move, omit the inactive arm entirely instead of "
        "passing its current pose redundantly. "
        "Returns FreespaceResult with status, IK error, and trajectory info. "
        "Also supports batched grasp-candidate ranking with fast cuRobo + CUDA-graph "
        "when grasp_candidates and batch_side are provided. "
        "Replaces low-level IK moves for free-space movements where collision avoidance matters."
    )
    parameters = [
        ToolParameter(
            "left_target_pos",
            "list[float]",
            "Left arm target position [x, y, z] in metres (world frame). "
            "Omit to keep left arm at current pose. For a right-only move, leave "
            "this omitted instead of passing the current left pose.",
            required=False,
        ),
        ToolParameter(
            "left_target_rpy",
            "list[float]",
            "Left arm target orientation [roll, pitch, yaw] in degrees. "
            "Omit to keep current orientation. For a right-only move, leave "
            "this omitted instead of passing the current left orientation.",
            required=False,
        ),
        ToolParameter(
            "right_target_pos",
            "list[float]",
            "Right arm target position [x, y, z] in metres (world frame). "
            "Omit to keep right arm at current pose. For a left-only move, leave "
            "this omitted instead of passing the current right pose.",
            required=False,
        ),
        ToolParameter(
            "right_target_rpy",
            "list[float]",
            "Right arm target orientation [roll, pitch, yaw] in degrees. "
            "Omit to keep current orientation. For a left-only move, leave "
            "this omitted instead of passing the current right orientation.",
            required=False,
        ),
        ToolParameter(
            "ik_error_threshold",
            "float",
            f"Max acceptable IK position error in metres (default {_DEFAULT_IK_THRESHOLD}). "
            f"Warning if > {_MAX_IK_THRESHOLD}.",
            required=False,
            default=_DEFAULT_IK_THRESHOLD,
        ),
        ToolParameter(
            "ik_rot_threshold_deg",
            "float",
            f"Max acceptable IK rotation error in degrees for cuRobo "
            f"(default {_DEFAULT_IK_ROT_THRESHOLD_DEG:.3f}).",
            required=False,
            default=_DEFAULT_IK_ROT_THRESHOLD_DEG,
        ),
        ToolParameter(
            "ik_xyz_weight",
            "float",
            f"IK translation / XYZ weight (default {_DEFAULT_IK_XYZ_WEIGHT}). "
            "Higher values make IK care more about Cartesian position error.",
            required=False,
            default=_DEFAULT_IK_XYZ_WEIGHT,
        ),
        ToolParameter(
            "ik_rpy_weight",
            "float",
            f"IK orientation / RPY weight (default {_DEFAULT_IK_RPY_WEIGHT}). "
            "Higher values make IK care more about orientation error.",
            required=False,
            default=_DEFAULT_IK_RPY_WEIGHT,
        ),
        ToolParameter(
            "left_gripper",
            "float",
            "Left gripper position during motion (0=closed, 1=open). "
            "Used for collision checking. Omit to use current gripper state.",
            required=False,
        ),
        ToolParameter(
            "right_gripper",
            "float",
            "Right gripper position during motion (0=closed, 1=open). "
            "Used for collision checking. Omit to use current gripper state.",
            required=False,
        ),
        ToolParameter(
            "left_gripper_target_width",
            "float",
            "Left gripper target width to reach during the move "
            "(0=closed, 1=open). When provided, the gripper is commanded "
            "along the same synchronized timeline as the arm trajectory.",
            required=False,
        ),
        ToolParameter(
            "right_gripper_target_width",
            "float",
            "Right gripper target width to reach during the move "
            "(0=closed, 1=open). When provided, the gripper is commanded "
            "along the same synchronized timeline as the arm trajectory.",
            required=False,
        ),
        ToolParameter(
            "planning_speed",
            "float",
            "Planner speed in rad/s, matching the control-loop UI slider. "
            f"Clamped to [{_PLANNING_SPEED_MIN}, {_PLANNING_SPEED_MAX}] with "
            f"default {_DEFAULT_PLANNING_SPEED}. Lower = slower, safer.",
            required=False,
            default=_DEFAULT_PLANNING_SPEED,
        ),
        ToolParameter(
            "backend",
            "str",
            "Motion planner backend. Defaults to 'curobo'. "
            "Set to 'rrt-connect' to force the legacy RRT-Connect planner.",
            required=False,
            default=_DEFAULT_BACKEND,
        ),
        ToolParameter(
            "preview_only",
            "bool",
            "If true, run the exact same planning/retiming path but do not execute. "
            "Returns the final planned position/orientation error for candidate ranking.",
            required=False,
            default=False,
        ),
        ToolParameter(
            "planner_backend",
            "str",
            "Motion planner backend: 'rrtconnect' or 'curobo'. Default is 'curobo'.",
            required=False,
            default="curobo",
        ),
        ToolParameter(
            "solver_speed",
            "str",
            "cuRobo solver preset for motion planning. Use 'fast' or 'slow'. "
            f"Default is '{_DEFAULT_SOLVER_SPEED}' for a single EE target. "
            f"Batched EE ranking also defaults to '{_DEFAULT_BATCH_SOLVER_SPEED}' when omitted.",
            required=False,
            default=_DEFAULT_SOLVER_SPEED,
        ),
        ToolParameter(
            "grasp_candidates",
            "list[dict]",
            "Optional list of grasp candidates to rank in one batched cuRobo call. "
            "Each candidate should provide position/rpy (or planner_xyz/planner_rpy), "
            "plus optional score and width. When provided, freespace_move switches to "
            "batch ranking mode instead of planning/executing one EE target.",
            required=False,
        ),
        ToolParameter(
            "batch_side",
            "str",
            "Which arm should evaluate the batch candidates: 'left' or 'right'. "
            "Required when grasp_candidates is provided.",
            required=False,
        ),
        ToolParameter(
            "batch_top_k",
            "int",
            f"Evaluate only the top-K candidates by score before batched planning. "
            f"Default {_DEFAULT_BATCH_TOP_K} to match the CUDA-graph batch planner capacity.",
            required=False,
            default=_DEFAULT_BATCH_TOP_K,
        ),
        ToolParameter(
            "batch_validate_trajectory",
            "bool",
            "If true, MuJoCo-validate each successful batched cuRobo trajectory. "
            "Defaults to false for maximum speed, matching the AnyGrasp debugger batch path.",
            required=False,
            default=_DEFAULT_BATCH_VALIDATE_TRAJECTORY,
        ),
        ToolParameter(
            "trajectory_cache_key",
            "str",
            "Execute a previously cached trajectory directly with no replanning. "
            "Use this after batch preview/ranking selected the best grasp.",
            required=False,
        ),
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = CAP_SERVER_PORT,
        env: Any | None = None,
    ):
        self._host = host
        self._port = port
        self._env = env
        self._client: Any | None = None
        self._planner_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._planner_lock = threading.RLock()
        self._diagnostic_kin_cache: dict[tuple[float, float], YamKinematics] = {}
        self._trajectory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._trajectory_cache_lock = threading.Lock()

    @staticmethod
    def _normalize_planner_backend(value: Any | None) -> str:
        backend = str(value or "curobo").strip().lower()
        if backend not in {"rrtconnect", "curobo"}:
            logger.warning("Unknown planner backend %r; falling back to curobo", value)
            backend = "curobo"
        return backend

    @staticmethod
    def _normalize_solver_speed(value: Any | None) -> str:
        speed = str(value or _DEFAULT_SOLVER_SPEED).strip().lower().replace("_", "-")
        if speed in {"fast", "slow"}:
            return speed
        logger.warning(
            "Unknown solver speed %r; falling back to %s",
            value,
            _DEFAULT_SOLVER_SPEED,
        )
        return _DEFAULT_SOLVER_SPEED

    def _get_client(self) -> Any:
        if self._client is None:
            import portal

            self._client = portal.Client(f"{self._host}:{self._port}")
        return self._client

    def _get_robot_state(self) -> dict[str, Any]:
        if self._env is None:
            return self._get_client().get_state().result()

        def _obs(side: str) -> dict[str, Any]:
            if hasattr(self._env, "get_observations"):
                return self._env.get_observations(side)
            if hasattr(self._env, "get_arm_observation"):
                return self._env.get_arm_observation(side)
            raise AttributeError(
                "env must expose get_observations(side) or get_arm_observation(side)"
            )

        left = _obs("left")
        right = _obs("right")
        return {
            "left_joint_pos": np.asarray(left["joint_pos"], dtype=np.float64),
            "right_joint_pos": np.asarray(right["joint_pos"], dtype=np.float64),
            "left_gripper_pos": np.asarray(left["gripper_pos"], dtype=np.float64).reshape(-1),
            "right_gripper_pos": np.asarray(right["gripper_pos"], dtype=np.float64).reshape(-1),
            "left_ee_pos": np.asarray(left.get("ee_pos", np.zeros(3)), dtype=np.float64),
            "right_ee_pos": np.asarray(right.get("ee_pos", np.zeros(3)), dtype=np.float64),
            "left_ee_quat_xyzw": np.asarray(
                left.get("ee_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
            ),
            "right_ee_quat_xyzw": np.asarray(
                right.get("ee_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
            ),
        }

    @staticmethod
    def _env_flag(name: str, *, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _get_curobo_portal_endpoint(cls) -> tuple[str, int | None, bool]:
        raw_port = str(os.environ.get("CAP_CUROBO_PORT", "")).strip()
        host = (
            str(os.environ.get("CAP_CUROBO_HOST", "127.0.0.1")).strip() or "127.0.0.1"
        )
        start_server = cls._env_flag(
            "CAP_CUROBO_START_SERVER",
            default=(raw_port == ""),
        )
        port: int | None = None
        if raw_port:
            try:
                port = int(raw_port)
            except ValueError as exc:
                raise ValueError(
                    f"CAP_CUROBO_PORT must be an integer, got {raw_port!r}"
                ) from exc
        if not start_server and port is None:
            raise ValueError(
                "CAP_CUROBO_PORT must be set when CAP_CUROBO_START_SERVER=0"
            )
        return host, port, start_server

    @staticmethod
    def _detect_robot_type() -> str:
        """Detect robot type from CAP_ROBOT_TYPE env var (default: yam)."""
        return os.environ.get("CAP_ROBOT_TYPE", "yam").strip().lower()

    def _get_planner(
        self,
        planner_backend: str = "curobo",
        solver_speed: str = _DEFAULT_SOLVER_SPEED,
        ik_error_threshold: float = _DEFAULT_IK_THRESHOLD,
        ik_rot_threshold_deg: float = _DEFAULT_IK_ROT_THRESHOLD_DEG,
        ik_xyz_weight: float = _DEFAULT_IK_XYZ_WEIGHT,
        ik_rpy_weight: float = _DEFAULT_IK_RPY_WEIGHT,
    ):
        """Lazy-initialise the motion planner (expensive: loads MuJoCo model)."""
        backend = self._normalize_planner_backend(planner_backend)
        solver_speed = self._normalize_solver_speed(solver_speed)
        robot_type = self._detect_robot_type()
        if backend == "curobo":
            host, port, start_server = self._get_curobo_portal_endpoint()
            key = (
                backend,
                solver_speed,
                host,
                int(port or -1),
                bool(start_server),
                robot_type,
                round(float(ik_error_threshold), 6),
                round(float(ik_rot_threshold_deg), 6),
            )
        else:
            key = (backend, solver_speed, float(ik_xyz_weight), float(ik_rpy_weight))
        with self._planner_lock:
            logger.debug("_planner_lock acquired for cache lookup/insert (key=%s)", key)
            planner = self._planner_cache.get(key)
            if planner is not None:
                self._planner_cache.move_to_end(key)
                logger.debug("_planner_lock releasing after cache hit (key=%s)", key)
                return planner

            if backend == "curobo":
                from experimental.portal_motion_planner import PortalMotionPlanner

                logger.info(
                    "Initialising cuRobo portal planner at %s:%s (start_server=%s, solver_speed=%s, robot_type=%s)",
                    host,
                    port if port is not None else "<auto>",
                    start_server,
                    solver_speed,
                    robot_type,
                )
                planner = PortalMotionPlanner(
                    backend="curobo",
                    solver_speed=solver_speed,
                    host=host,
                    port=port,
                    position_threshold=float(ik_error_threshold),
                    rotation_threshold=float(np.deg2rad(ik_rot_threshold_deg)),
                    start_server=start_server,
                    robot_type=robot_type,
                )
            else:
                from experimental.motion_planner import YamMotionPlanner

                planner = YamMotionPlanner(
                    position_cost=float(ik_xyz_weight),
                    orientation_cost=float(ik_rpy_weight),
                )
            self._planner_cache[key] = planner
            self._planner_cache.move_to_end(key)

            while len(self._planner_cache) > _PLANNER_CACHE_MAX_SIZE:
                evicted_key, evicted_planner = self._planner_cache.popitem(last=False)
                logger.warning(
                    "Evicted cached planner %s to cap planner cache at %d entries",
                    evicted_key,
                    _PLANNER_CACHE_MAX_SIZE,
                )
                try:
                    if hasattr(evicted_planner, "cleanup"):
                        evicted_planner.cleanup()
                except Exception:
                    logger.exception(
                        "Failed to cleanup evicted planner for key %s", evicted_key
                    )

            logger.debug("_planner_lock releasing after cache insert (key=%s)", key)
            return planner

    @staticmethod
    def _copy_cached_trajectory_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64).copy()
        if isinstance(value, list):
            return [
                FreespaceMoveTool._copy_cached_trajectory_value(item) for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                FreespaceMoveTool._copy_cached_trajectory_value(item) for item in value
            )
        return value

    def _store_cached_trajectory(
        self,
        *,
        side: str,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        current_left_gp: float,
        current_right_gp: float,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        timestamps: list[float],
        left_gripper_positions: np.ndarray | None = None,
        right_gripper_positions: np.ndarray | None = None,
        final_pos_error_m: float,
        final_rot_error_deg: float,
        final_pose_error: float,
    ) -> str:
        cache_key = uuid.uuid4().hex
        cache_entry = {
            "cache_key": cache_key,
            "side": str(side).strip().lower(),
            "current_left_jp": np.asarray(current_left_jp, dtype=np.float64).copy(),
            "current_right_jp": np.asarray(current_right_jp, dtype=np.float64).copy(),
            "current_left_gp": float(current_left_gp),
            "current_right_gp": float(current_right_gp),
            "left_positions": np.asarray(left_positions, dtype=np.float64).copy(),
            "right_positions": np.asarray(right_positions, dtype=np.float64).copy(),
            "timestamps": [float(t) for t in timestamps],
            "left_gripper_positions": (
                None
                if left_gripper_positions is None
                else np.asarray(left_gripper_positions, dtype=np.float64).copy()
            ),
            "right_gripper_positions": (
                None
                if right_gripper_positions is None
                else np.asarray(right_gripper_positions, dtype=np.float64).copy()
            ),
            "trajectory_steps": int(
                np.atleast_2d(np.asarray(left_positions, dtype=np.float64)).shape[0]
            ),
            "final_pos_error_m": float(final_pos_error_m),
            "final_rot_error_deg": float(final_rot_error_deg),
            "final_pose_error": float(final_pose_error),
        }
        with self._trajectory_cache_lock:
            self._trajectory_cache[cache_key] = cache_entry
            self._trajectory_cache.move_to_end(cache_key)
            while len(self._trajectory_cache) > _TRAJECTORY_CACHE_MAX_SIZE:
                self._trajectory_cache.popitem(last=False)
        return cache_key

    def _get_cached_trajectory(self, cache_key: Any) -> dict[str, Any] | None:
        key = str(cache_key or "").strip()
        if not key:
            return None
        with self._trajectory_cache_lock:
            entry = self._trajectory_cache.get(key)
            if entry is None:
                return None
            self._trajectory_cache.move_to_end(key)
            return {
                name: self._copy_cached_trajectory_value(value)
                for name, value in entry.items()
            }

    @staticmethod
    def _cached_trajectory_matches_current(
        cache_entry: dict[str, Any],
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
    ) -> bool:
        cached_left = np.asarray(
            cache_entry["current_left_jp"], dtype=np.float64
        ).reshape(-1)
        cached_right = np.asarray(
            cache_entry["current_right_jp"], dtype=np.float64
        ).reshape(-1)
        current_left = np.asarray(current_left_jp, dtype=np.float64).reshape(-1)
        current_right = np.asarray(current_right_jp, dtype=np.float64).reshape(-1)
        return bool(
            np.allclose(
                cached_left,
                current_left,
                atol=_TRAJECTORY_STATE_TOLERANCE_RAD,
                rtol=0.0,
            )
            and np.allclose(
                cached_right,
                current_right,
                atol=_TRAJECTORY_STATE_TOLERANCE_RAD,
                rtol=0.0,
            )
        )

    def _execute_cached_trajectory(
        self,
        *,
        cache_entry: dict[str, Any],
        planning_mode: str = "cached",
    ) -> ToolResult:
        side = str(cache_entry["side"]).strip().lower()
        left_positions = np.atleast_2d(
            np.asarray(cache_entry["left_positions"], dtype=np.float64)
        )
        right_positions = np.atleast_2d(
            np.asarray(cache_entry["right_positions"], dtype=np.float64)
        )
        timestamps = [float(t) for t in cache_entry.get("timestamps", [])]
        left_gripper_positions = cache_entry.get("left_gripper_positions")
        right_gripper_positions = cache_entry.get("right_gripper_positions")
        exec_error = self._execute_trajectory(
            side,
            timestamps,
            left_positions,
            right_positions,
            None
            if left_gripper_positions is None
            else np.asarray(left_gripper_positions, dtype=np.float64).reshape(-1),
            None
            if right_gripper_positions is None
            else np.asarray(right_gripper_positions, dtype=np.float64).reshape(-1),
        )
        if exec_error:
            return ToolResult(
                success=False,
                data=FreespaceResult(
                    status="Execution_Failed",
                    ik_error_m=round(
                        float(cache_entry.get("final_pos_error_m", 0.0)), 4
                    ),
                    final_pos_error_m=round(
                        float(cache_entry.get("final_pos_error_m", 0.0)), 6
                    ),
                    final_rot_error_deg=round(
                        float(cache_entry.get("final_rot_error_deg", 0.0)), 4
                    ),
                    final_pose_error=round(
                        float(cache_entry.get("final_pose_error", 0.0)), 6
                    ),
                    trajectory_steps=int(
                        cache_entry.get("trajectory_steps", len(left_positions))
                    ),
                    trajectory_cache_key=str(cache_entry.get("cache_key", "")) or None,
                    executed=False,
                    planning_mode=planning_mode,
                    side=side,
                    reason=f"Trajectory execution failed: {exec_error}",
                ),
                error=exec_error,
            )
        return ToolResult(
            success=True,
            data=FreespaceResult(
                status="Success",
                ik_error_m=round(float(cache_entry.get("final_pos_error_m", 0.0)), 4),
                final_pos_error_m=round(
                    float(cache_entry.get("final_pos_error_m", 0.0)), 6
                ),
                final_rot_error_deg=round(
                    float(cache_entry.get("final_rot_error_deg", 0.0)), 4
                ),
                final_pose_error=round(
                    float(cache_entry.get("final_pose_error", 0.0)), 6
                ),
                trajectory_steps=int(
                    cache_entry.get("trajectory_steps", len(left_positions))
                ),
                trajectory_cache_key=str(cache_entry.get("cache_key", "")) or None,
                executed=True,
                planning_mode=planning_mode,
                side=side,
            ),
        )

    @staticmethod
    def _normalize_backend(backend: Any | None) -> str:
        value = str(backend or _DEFAULT_BACKEND).strip().lower().replace("_", "-")
        if value in {"curobo", "curobo-motiongen"}:
            return "curobo"
        if value in {"rrt-connect", "rrtconnect", "rrt"}:
            return "rrt-connect"
        raise ValueError(
            f"Unsupported freespace_move backend {backend!r}. "
            "Use 'curobo' or 'rrt-connect'."
        )

    @staticmethod
    def _resolve_planning_speed(
        planning_speed: Any | None,
        legacy_max_joint_vel: Any | None,
    ) -> float:
        value = planning_speed if planning_speed is not None else legacy_max_joint_vel
        if value is None:
            return _DEFAULT_PLANNING_SPEED
        return min(_PLANNING_SPEED_MAX, max(_PLANNING_SPEED_MIN, float(value)))

    @staticmethod
    def _clip_gripper_width(value: Any | None) -> float | None:
        if value is None:
            return None
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _coerce_bool(value: Any | None, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _display_rpy_to_quat(rpy: list[float] | np.ndarray) -> np.ndarray:
        """Convert display RPY (degrees) to planner quaternion [x, y, z, w]."""
        from scipy.spatial.transform import Rotation

        roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
        euler_xyz = [-pitch, roll, -yaw - 90.0]
        return Rotation.from_euler("xyz", euler_xyz, degrees=True).as_quat()

    @staticmethod
    def _quat_error_deg(
        target_quat_xyzw: np.ndarray, actual_quat_xyzw: np.ndarray
    ) -> float:
        """Shortest-angle orientation error in degrees."""
        from scipy.spatial.transform import Rotation

        delta = Rotation.from_quat(target_quat_xyzw).inv() * Rotation.from_quat(
            actual_quat_xyzw
        )
        return float(np.degrees(delta.magnitude()))

    @staticmethod
    def _pose_error_metric(
        pos_err_m: float,
        rot_err_deg: float,
        *,
        xyz_weight: float = _DEFAULT_IK_XYZ_WEIGHT,
        rpy_weight: float = _DEFAULT_IK_RPY_WEIGHT,
    ) -> float:
        """Combined pose error metric used for ranking previewed plans."""
        return float(
            float(xyz_weight) * pos_err_m + float(rpy_weight) * np.deg2rad(rot_err_deg)
        )

    @classmethod
    def _pose_matches_current(
        cls,
        *,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        current_pos: np.ndarray,
        current_quat_xyzw: np.ndarray,
    ) -> bool:
        pos_err = float(
            np.linalg.norm(
                np.asarray(target_pos, dtype=np.float64)
                - np.asarray(current_pos, dtype=np.float64)
            )
        )
        rot_err = cls._quat_error_deg(
            np.asarray(target_quat_xyzw, dtype=np.float64),
            np.asarray(current_quat_xyzw, dtype=np.float64),
        )
        return (
            pos_err <= _INACTIVE_ARM_POS_TOL_M and rot_err <= _INACTIVE_ARM_ROT_TOL_DEG
        )

    @classmethod
    def _infer_effective_side(
        cls,
        *,
        has_left: bool,
        has_right: bool,
        tgt_l_pos: np.ndarray,
        tgt_l_q: np.ndarray,
        tgt_r_pos: np.ndarray,
        tgt_r_q: np.ndarray,
        cur_l_pos: np.ndarray,
        cur_l_q: np.ndarray,
        cur_r_pos: np.ndarray,
        cur_r_q: np.ndarray,
        left_gripper_target_width: float | None,
        right_gripper_target_width: float | None,
    ) -> str | None:
        if has_left and has_right:
            left_is_current = cls._pose_matches_current(
                target_pos=tgt_l_pos,
                target_quat_xyzw=tgt_l_q,
                current_pos=cur_l_pos,
                current_quat_xyzw=cur_l_q,
            )
            right_is_current = cls._pose_matches_current(
                target_pos=tgt_r_pos,
                target_quat_xyzw=tgt_r_q,
                current_pos=cur_r_pos,
                current_quat_xyzw=cur_r_q,
            )
            left_active = not left_is_current or left_gripper_target_width is not None
            right_active = (
                not right_is_current or right_gripper_target_width is not None
            )
            if left_active and right_active:
                return "both"
            if left_active:
                logger.info(
                    "freespace_move: treating right arm as inactive because its target matches the current pose"
                )
                return "left"
            if right_active:
                logger.info(
                    "freespace_move: treating left arm as inactive because its target matches the current pose"
                )
                return "right"
            return None
        if has_left:
            return "left"
        if has_right:
            return "right"
        return None

    @staticmethod
    def _retime_joint_trajectory(
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        *,
        current_left_joint_pos: np.ndarray,
        current_right_joint_pos: np.ndarray,
        planning_speed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match ScriptedPolicy.execute_trajectory retiming behavior."""
        left_positions = np.atleast_2d(np.asarray(left_positions, dtype=np.float64))
        right_positions = np.atleast_2d(np.asarray(right_positions, dtype=np.float64))
        if left_positions.shape[0] == 0:
            return left_positions, right_positions

        left_dof = left_positions.shape[1]
        current_left = np.asarray(current_left_joint_pos, dtype=np.float64).reshape(
            1, -1
        )[:, :left_dof]
        current_right = np.asarray(current_right_joint_pos, dtype=np.float64).reshape(
            1, -1
        )[:, : right_positions.shape[1]]
        needs_bridge = not (
            np.allclose(left_positions[0], current_left[0], atol=1e-6)
            and np.allclose(right_positions[0], current_right[0], atol=1e-6)
        )
        if needs_bridge:
            left_positions = np.concatenate([current_left, left_positions], axis=0)
            right_positions = np.concatenate([current_right, right_positions], axis=0)

        full_path = np.concatenate([left_positions, right_positions], axis=1)
        if full_path.shape[0] <= 1:
            return full_path[:, :left_dof], full_path[:, left_dof:]

        dt = CONTROL_PERIOD_S
        safe_vel = min(
            _PLANNING_SPEED_MAX, max(_PLANNING_SPEED_MIN, float(planning_speed))
        )

        positions: list[np.ndarray] = []
        for k in range(full_path.shape[0] - 1):
            diff = full_path[k + 1] - full_path[k]
            max_delta = float(np.max(np.abs(diff)))
            seg_time = max(max_delta / safe_vel, dt)
            n_steps = max(1, int(np.ceil(seg_time / dt)))
            for s in range(n_steps):
                alpha = s / n_steps
                positions.append(full_path[k] + alpha * diff)
        positions.append(full_path[-1].copy())

        retimed = np.asarray(positions, dtype=np.float64)
        return retimed[:, :left_dof], retimed[:, left_dof:]

    @staticmethod
    def _densify_joint_waypoints(
        left_waypoints: np.ndarray,
        right_waypoints: np.ndarray,
        *,
        max_joint_step: float = _GEOMETRIC_MAX_JOINT_STEP,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Densify waypoints by geometry only, independent of execution speed."""
        left_waypoints = np.atleast_2d(np.asarray(left_waypoints, dtype=np.float64))
        right_waypoints = np.atleast_2d(np.asarray(right_waypoints, dtype=np.float64))
        if left_waypoints.shape[0] <= 1:
            return left_waypoints, right_waypoints

        dense_left: list[np.ndarray] = [left_waypoints[0].copy()]
        dense_right: list[np.ndarray] = [right_waypoints[0].copy()]
        safe_step = max(1e-6, float(max_joint_step))
        for idx in range(left_waypoints.shape[0] - 1):
            left_start = left_waypoints[idx]
            left_end = left_waypoints[idx + 1]
            right_start = right_waypoints[idx]
            right_end = right_waypoints[idx + 1]
            max_delta = float(
                max(
                    np.max(np.abs(left_end - left_start)),
                    np.max(np.abs(right_end - right_start)),
                )
            )
            n_segments = max(1, int(np.ceil(max_delta / safe_step)))
            for seg in range(1, n_segments + 1):
                alpha = seg / n_segments
                dense_left.append(left_start + alpha * (left_end - left_start))
                dense_right.append(right_start + alpha * (right_end - right_start))

        return np.asarray(dense_left, dtype=np.float64), np.asarray(
            dense_right, dtype=np.float64
        )

    @staticmethod
    def _timestamps_from_waypoints(
        left_waypoints: np.ndarray,
        right_waypoints: np.ndarray,
        *,
        planning_speed: float,
    ) -> list[float]:
        """Assign one timing pass so max joint speed tracks the user knob."""
        left_waypoints = np.atleast_2d(np.asarray(left_waypoints, dtype=np.float64))
        right_waypoints = np.atleast_2d(np.asarray(right_waypoints, dtype=np.float64))
        if left_waypoints.shape[0] == 0:
            return []
        if left_waypoints.shape[0] == 1:
            return [0.0]

        safe_vel = min(
            _PLANNING_SPEED_MAX, max(_PLANNING_SPEED_MIN, float(planning_speed))
        )
        timestamps = [0.0]
        t_acc = 0.0
        for idx in range(left_waypoints.shape[0] - 1):
            max_delta = float(
                max(
                    np.max(np.abs(left_waypoints[idx + 1] - left_waypoints[idx])),
                    np.max(np.abs(right_waypoints[idx + 1] - right_waypoints[idx])),
                )
            )
            seg_dt = max(1e-6, max_delta / safe_vel)
            t_acc += seg_dt
            timestamps.append(t_acc)
        return timestamps

    def _compute_plan_diagnostics(
        self,
        kin: YamKinematics,
        planner: Any,
        *,
        side: str,
        final_left_joint_pos: np.ndarray,
        final_right_joint_pos: np.ndarray,
        target_left_pos: np.ndarray,
        target_left_quat_xyzw: np.ndarray,
        target_right_pos: np.ndarray,
        target_right_quat_xyzw: np.ndarray,
    ) -> dict[str, float]:
        """Measure final pose error at the last commanded waypoint."""
        final_l_pos, final_l_q, final_r_pos, final_r_q = kin.forward_kinematics(
            np.asarray(final_left_joint_pos, dtype=np.float64),
            np.asarray(final_right_joint_pos, dtype=np.float64),
        )

        arm_metrics: list[tuple[float, float, float]] = []
        xyz_weight = float(getattr(planner, "ik_position_cost", _DEFAULT_IK_XYZ_WEIGHT))
        rpy_weight = float(
            getattr(planner, "ik_orientation_cost", _DEFAULT_IK_RPY_WEIGHT)
        )
        if side in ("left", "both"):
            pos_err = float(np.linalg.norm(final_l_pos - target_left_pos))
            rot_err = self._quat_error_deg(target_left_quat_xyzw, final_l_q)
            arm_metrics.append(
                (
                    pos_err,
                    rot_err,
                    self._pose_error_metric(
                        pos_err, rot_err, xyz_weight=xyz_weight, rpy_weight=rpy_weight
                    ),
                )
            )
        if side in ("right", "both"):
            pos_err = float(np.linalg.norm(final_r_pos - target_right_pos))
            rot_err = self._quat_error_deg(target_right_quat_xyzw, final_r_q)
            arm_metrics.append(
                (
                    pos_err,
                    rot_err,
                    self._pose_error_metric(
                        pos_err, rot_err, xyz_weight=xyz_weight, rpy_weight=rpy_weight
                    ),
                )
            )

        if not arm_metrics:
            return {
                "final_pos_error_m": 0.0,
                "final_rot_error_deg": 0.0,
                "final_pose_error": 0.0,
            }

        max_pos_err = max(item[0] for item in arm_metrics)
        max_rot_err = max(item[1] for item in arm_metrics)
        max_pose_err = max(item[2] for item in arm_metrics)
        return {
            "final_pos_error_m": max_pos_err,
            "final_rot_error_deg": max_rot_err,
            "final_pose_error": max_pose_err,
        }

    def _get_diagnostic_kinematics(
        self,
        *,
        position_cost: float,
        orientation_cost: float,
    ) -> YamKinematics | None:
        """Return a cached public kinematics helper for plan diagnostics.

        Returns None when YamKinematics is not available (e.g. Panda/RoboCasa),
        in which case callers should fall back to cuRobo-reported errors.
        """
        key = (float(position_cost), float(orientation_cost))
        kin = self._diagnostic_kin_cache.get(key)
        if kin is None:
            try:
                kin = YamKinematics(
                    position_cost=float(position_cost),
                    orientation_cost=float(orientation_cost),
                )
            except Exception:
                return None
            self._diagnostic_kin_cache[key] = kin
        return kin

    def _ik_failed_metrics(
        self,
        *,
        planner: Any,
        plan_result: dict[str, Any],
    ) -> dict[str, float | bool]:
        """Return IK-failure residuals from cuRobo only.

        Batch preview already reports cuRobo's own ``position_error_m`` and
        ``rotation_error_deg``. Single-target preview/execute must use the same
        source instead of recomputing a different post-hoc IK diagnostic.
        """

        raw_pos_err = float(plan_result.get("position_error_m", np.nan))
        raw_rot_err = float(plan_result.get("rotation_error_deg", np.nan))
        pos_reported = bool(np.isfinite(raw_pos_err))
        rot_reported = bool(np.isfinite(raw_rot_err))
        pos_err = raw_pos_err if pos_reported else 0.0
        rot_err = raw_rot_err if rot_reported else 0.0
        xyz_weight = float(getattr(planner, "ik_position_cost", _DEFAULT_IK_XYZ_WEIGHT))
        rpy_weight = float(
            getattr(planner, "ik_orientation_cost", _DEFAULT_IK_RPY_WEIGHT)
        )
        return {
            "final_pos_error_m": pos_err,
            "final_rot_error_deg": rot_err,
            "final_pose_error": self._pose_error_metric(
                pos_err,
                rot_err,
                xyz_weight=xyz_weight,
                rpy_weight=rpy_weight,
            ),
            "reported_by_curobo": pos_reported and rot_reported,
        }

    @staticmethod
    def _batch_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        """Sort rows: executable first, then IK failures, then planning failures.

        Within a group, rank by displayed IK residuals then by score.
        """

        status_bucket = (
            0
            if row.get("planner_status") == "Success"
            else (1 if row.get("planner_status") == "IK_Failed" else 2)
        )
        pos_err = (
            round(float(row["ik_error_m"]), 4)
            if row.get("ik_error_m") is not None
            else float("inf")
        )
        rot_err = (
            round(float(row["ik_rot_error_deg"]), 2)
            if row.get("ik_rot_error_deg") is not None
            else float("inf")
        )
        score = -float(row.get("score", 0.0))
        return (status_bucket, pos_err, rot_err, score)

    @staticmethod
    def _round_list(arr: np.ndarray | list[float], digits: int) -> list[float]:
        return [
            round(float(v), digits) for v in np.asarray(arr, dtype=np.float64).tolist()
        ]

    @staticmethod
    def _normalize_batch_side(value: Any | None) -> str:
        side = str(value or "").strip().lower()
        if side not in {"left", "right"}:
            raise ValueError(
                "Batch grasp ranking requires batch_side='left' or 'right'."
            )
        return side

    @staticmethod
    def _normalize_batch_top_k(value: Any | None) -> int:
        if value is None:
            return int(_DEFAULT_BATCH_TOP_K)
        top_k = int(value)
        if top_k <= 0:
            raise ValueError("batch_top_k must be >= 1.")
        return top_k

    @staticmethod
    def _candidate_value(candidate: Any, *names: str) -> Any:
        if isinstance(candidate, dict):
            for name in names:
                if name in candidate:
                    return candidate[name]
            return None
        for name in names:
            if hasattr(candidate, name):
                return getattr(candidate, name)
        return None

    @classmethod
    def _coerce_batch_candidate(cls, candidate: Any, *, index: int) -> dict[str, Any]:
        position = cls._candidate_value(candidate, "position", "planner_xyz", "xyz")
        rpy = cls._candidate_value(candidate, "rpy", "planner_rpy")
        if position is None or rpy is None:
            raise ValueError(
                f"grasp_candidates[{index}] must define position/rpy or planner_xyz/planner_rpy."
            )
        position_arr = np.asarray(position, dtype=np.float64).reshape(-1)
        rpy_arr = np.asarray(rpy, dtype=np.float64).reshape(-1)
        if position_arr.shape != (3,):
            raise ValueError(
                f"grasp_candidates[{index}].position must have exactly 3 values."
            )
        if rpy_arr.shape != (3,):
            raise ValueError(
                f"grasp_candidates[{index}].rpy must have exactly 3 values."
            )
        score = cls._candidate_value(candidate, "score")
        width = cls._candidate_value(candidate, "width")
        return {
            "source_index": int(index),
            "position": position_arr,
            "rpy": rpy_arr,
            "score": float(score) if score is not None else 0.0,
            "width": float(width) if width is not None else 0.0,
        }

    def _rank_batch_grasp_candidates(
        self,
        *,
        grasp_candidates: list[Any],
        batch_side: str,
        ik_threshold: float,
        ik_rot_threshold_deg: float,
        ik_xyz_weight: float,
        ik_rpy_weight: float,
        planning_speed: float,
        planner_backend: str,
        solver_speed: str,
        left_gripper: float | None,
        right_gripper: float | None,
        batch_top_k: int,
        batch_validate_trajectory: bool,
    ) -> ToolResult:
        total_start = time.perf_counter()
        try:
            side = self._normalize_batch_side(batch_side)
            if planner_backend != "curobo":
                raise ValueError(
                    "Batch grasp ranking is only supported with planner_backend='curobo'."
                )
            if not grasp_candidates:
                return ToolResult(
                    success=True,
                    data=FreespaceResult(
                        status="Success",
                        executed=False,
                        planning_mode="batch",
                        side=side,
                        reason="No grasp candidates provided.",
                    ),
                )

            original_candidate_count = len(grasp_candidates)
            batch_top_k = self._normalize_batch_top_k(batch_top_k)
            candidate_records = [
                self._coerce_batch_candidate(candidate, index=i)
                for i, candidate in enumerate(grasp_candidates)
            ]
            if len(candidate_records) > batch_top_k:
                candidate_records = sorted(
                    candidate_records,
                    key=lambda candidate: float(candidate["score"]),
                    reverse=True,
                )[:batch_top_k]
            else:
                candidate_records = list(candidate_records)

            state = self._get_robot_state()
            cur_left_jp = np.asarray(state["left_joint_pos"], dtype=np.float64)
            cur_right_jp = np.asarray(state["right_joint_pos"], dtype=np.float64)
            cur_left_gp = float(state["left_gripper_pos"][0])
            cur_right_gp = float(state["right_gripper_pos"][0])
            left_gripper = cur_left_gp if left_gripper is None else float(left_gripper)
            right_gripper = (
                cur_right_gp if right_gripper is None else float(right_gripper)
            )

            planner = self._get_planner(
                planner_backend=planner_backend,
                solver_speed=solver_speed,
                ik_error_threshold=ik_threshold,
                ik_rot_threshold_deg=ik_rot_threshold_deg,
                ik_xyz_weight=ik_xyz_weight,
                ik_rpy_weight=ik_rpy_weight,
            )
            grasp_xyzs = np.asarray(
                [candidate["position"] for candidate in candidate_records],
                dtype=np.float64,
            )
            grasp_quats = np.asarray(
                [
                    self._display_rpy_to_quat(candidate["rpy"])
                    for candidate in candidate_records
                ],
                dtype=np.float64,
            )

            batch_plan_result: dict[str, Any] = {}
            batch_success_mask: np.ndarray | None = None
            batch_status_by_index: list[str | None] | None = None
            batch_status_details: list[str | None] | None = None
            batch_position_errors: np.ndarray | None = None
            batch_rotation_errors: np.ndarray | None = None
            batch_left_positions_by_index: list[Any] | None = None
            batch_right_positions_by_index: list[Any] | None = None
            batch_attempted = hasattr(planner, "plan_batch_to_pose")
            batch_error: str | None = None

            plan_eval_start = time.perf_counter()
            if batch_attempted:
                try:
                    # RPC outside _planner_lock — planner ref is held so it
                    # cannot be garbage-collected; Portal serialises on the
                    # server side, so we don't need Python-side locking here.
                    logger.debug(
                        "_planner_lock released before batch RPC plan_batch_to_pose"
                    )
                    batch_plan_result = planner.plan_batch_to_pose(
                        current_left_jp=cur_left_jp,
                        current_right_jp=cur_right_jp,
                        target_left_pos=grasp_xyzs if side == "left" else None,
                        target_left_quat_xyzw=grasp_quats if side == "left" else None,
                        target_right_pos=grasp_xyzs if side == "right" else None,
                        target_right_quat_xyzw=grasp_quats if side == "right" else None,
                        side=side,
                        ik_error_threshold=ik_threshold,
                        left_gripper=left_gripper,
                        right_gripper=right_gripper,
                        max_joint_vel=planning_speed,
                        validate_trajectory=bool(batch_validate_trajectory),
                    )
                    batch_error = batch_plan_result.get("error")
                    if batch_error:
                        batch_error = str(batch_error)
                    candidate_mask = np.asarray(
                        batch_plan_result.get("success_mask", []), dtype=bool
                    ).reshape(-1)
                    if candidate_mask.shape[0] == len(candidate_records):
                        batch_success_mask = candidate_mask
                        statuses = batch_plan_result.get("status_by_index")
                        if isinstance(statuses, list) and len(statuses) == len(
                            candidate_records
                        ):
                            batch_status_by_index = [
                                None if status is None else str(status)
                                for status in statuses
                            ]
                        else:
                            batch_status_by_index = None
                        details = batch_plan_result.get("status_detail_by_index")
                        if isinstance(details, list) and len(details) == len(
                            candidate_records
                        ):
                            batch_status_details = [
                                None if detail is None else str(detail)
                                for detail in details
                            ]
                        else:
                            batch_status_details = [None for _ in candidate_records]
                        position_errors = np.asarray(
                            batch_plan_result.get("position_error_m", []),
                            dtype=np.float64,
                        ).reshape(-1)
                        if position_errors.shape[0] == len(candidate_records):
                            batch_position_errors = position_errors
                        rotation_errors = np.asarray(
                            batch_plan_result.get("rotation_error_deg", []),
                            dtype=np.float64,
                        ).reshape(-1)
                        if rotation_errors.shape[0] == len(candidate_records):
                            batch_rotation_errors = rotation_errors
                        left_positions_by_index = batch_plan_result.get(
                            "left_positions_by_index"
                        )
                        right_positions_by_index = batch_plan_result.get(
                            "right_positions_by_index"
                        )
                        if (
                            isinstance(left_positions_by_index, list)
                            and isinstance(right_positions_by_index, list)
                            and len(left_positions_by_index) == len(candidate_records)
                            and len(right_positions_by_index) == len(candidate_records)
                        ):
                            batch_left_positions_by_index = list(
                                left_positions_by_index
                            )
                            batch_right_positions_by_index = list(
                                right_positions_by_index
                            )
                    else:
                        batch_error = batch_error or (
                            f"unexpected success_mask size {candidate_mask.shape[0]} "
                            f"(expected {len(candidate_records)})"
                        )
                except Exception as exc:
                    batch_error = str(exc)
            else:
                batch_error = "planner object does not expose plan_batch_to_pose"
            plan_eval_elapsed_ms = (time.perf_counter() - plan_eval_start) * 1000.0

            if batch_success_mask is None:
                total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
                error_msg = (
                    batch_error
                    or "batch cuRobo planning did not return a usable result"
                )
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Error",
                        executed=False,
                        reason=error_msg,
                        planning_mode="batch_error",
                        side=side,
                        input_candidate_count=original_candidate_count,
                        evaluated_candidate_count=len(candidate_records),
                        truncated_input_count=max(
                            0, original_candidate_count - len(candidate_records)
                        ),
                        batch_attempted=batch_attempted,
                        batch_error=error_msg,
                        timing_total_ms=round(total_elapsed_ms, 3),
                        timing_plan_eval_ms=round(plan_eval_elapsed_ms, 3),
                        curobo_solve_time_ms=round(
                            float(batch_plan_result.get("curobo_solve_time_ms", 0.0)), 3
                        ),
                        curobo_total_time_ms=round(
                            float(batch_plan_result.get("curobo_total_time_ms", 0.0)), 3
                        ),
                        curobo_graph_time_ms=round(
                            float(batch_plan_result.get("curobo_graph_time_ms", 0.0)), 3
                        ),
                        curobo_ik_time_ms=round(
                            float(batch_plan_result.get("curobo_ik_time_ms", 0.0)), 3
                        ),
                    ),
                    error=f"Batch cuRobo sort failed: {error_msg}",
                )

            assert batch_status_details is not None
            if batch_status_by_index is None:
                batch_status_by_index = [
                    "Success" if bool(ok) else "Planning_Failed"
                    for ok in batch_success_mask.tolist()
                ]
            if batch_position_errors is None:
                batch_position_errors = np.full(
                    (len(candidate_records),), np.nan, dtype=np.float64
                )
            if batch_rotation_errors is None:
                batch_rotation_errors = np.full(
                    (len(candidate_records),), np.nan, dtype=np.float64
                )
            if batch_left_positions_by_index is None:
                batch_left_positions_by_index = [None for _ in candidate_records]
            if batch_right_positions_by_index is None:
                batch_right_positions_by_index = [None for _ in candidate_records]

            postprocess_start = time.perf_counter()
            row_payloads: list[dict[str, Any]] = []
            for idx, candidate in enumerate(candidate_records):
                candidate_status = str(batch_status_by_index[idx] or "").strip() or (
                    "Success" if bool(batch_success_mask[idx]) else "Planning_Failed"
                )
                pos_err = float(batch_position_errors[idx])
                rot_err = float(batch_rotation_errors[idx])
                if not bool(batch_success_mask[idx]):
                    if candidate_status == "IK_Failed" and np.isfinite(pos_err):
                        row_payloads.append(
                            {
                                "source_index": int(candidate["source_index"]),
                                "position": self._round_list(candidate["position"], 5),
                                "rpy": self._round_list(candidate["rpy"], 4),
                                "score": round(float(candidate["score"]), 4),
                                "width": round(float(candidate["width"]), 5),
                                "ik_error_m": round(pos_err, 6),
                                "ik_rot_error_deg": round(rot_err, 4)
                                if np.isfinite(rot_err)
                                else None,
                                "within_ik_threshold": False,
                                "planner_status": "IK_Failed",
                                "motion_plan_error": None,
                                "motion_plan_reason": None,
                                "trajectory_cache_key": None,
                                "trajectory_steps": 0,
                            }
                        )
                        continue
                    row_payloads.append(
                        {
                            "source_index": int(candidate["source_index"]),
                            "position": self._round_list(candidate["position"], 5),
                            "rpy": self._round_list(candidate["rpy"], 4),
                            "score": round(float(candidate["score"]), 4),
                            "width": round(float(candidate["width"]), 5),
                            "ik_error_m": None,
                            "ik_rot_error_deg": None,
                            "within_ik_threshold": False,
                            "planner_status": candidate_status,
                            "motion_plan_error": True,
                            "motion_plan_reason": (
                                batch_status_details[idx]
                                or (
                                    "Planning failed"
                                    if candidate_status == "Planning_Failed"
                                    else candidate_status
                                )
                            ),
                            "trajectory_cache_key": None,
                            "trajectory_steps": 0,
                        }
                    )
                    continue

                if not np.isfinite(pos_err):
                    row_payloads.append(
                        {
                            "source_index": int(candidate["source_index"]),
                            "position": self._round_list(candidate["position"], 5),
                            "rpy": self._round_list(candidate["rpy"], 4),
                            "score": round(float(candidate["score"]), 4),
                            "width": round(float(candidate["width"]), 5),
                            "ik_error_m": None,
                            "ik_rot_error_deg": None,
                            "within_ik_threshold": False,
                            "planner_status": candidate_status,
                            "motion_plan_error": True,
                            "motion_plan_reason": "Batch planner did not return a finite IK error",
                            "trajectory_cache_key": None,
                            "trajectory_steps": 0,
                        }
                    )
                    continue

                raw_left_positions = batch_left_positions_by_index[idx]
                raw_right_positions = batch_right_positions_by_index[idx]
                if raw_left_positions is None or raw_right_positions is None:
                    row_payloads.append(
                        {
                            "source_index": int(candidate["source_index"]),
                            "position": self._round_list(candidate["position"], 5),
                            "rpy": self._round_list(candidate["rpy"], 4),
                            "score": round(float(candidate["score"]), 4),
                            "width": round(float(candidate["width"]), 5),
                            "ik_error_m": None,
                            "ik_rot_error_deg": None,
                            "within_ik_threshold": False,
                            "planner_status": candidate_status,
                            "motion_plan_error": True,
                            "motion_plan_reason": "Batch planner did not return an executable trajectory",
                            "trajectory_cache_key": None,
                            "trajectory_steps": 0,
                        }
                    )
                    continue

                left_waypoints = np.atleast_2d(
                    np.asarray(raw_left_positions, dtype=np.float64)
                )
                right_waypoints = np.atleast_2d(
                    np.asarray(raw_right_positions, dtype=np.float64)
                )
                left_positions, right_positions = self._densify_joint_waypoints(
                    left_waypoints,
                    right_waypoints,
                )
                timestamps = self._timestamps_from_waypoints(
                    left_positions,
                    right_positions,
                    planning_speed=planning_speed,
                )
                trajectory_steps = int(len(left_positions))
                trajectory_cache_key = self._store_cached_trajectory(
                    side=side,
                    current_left_jp=cur_left_jp,
                    current_right_jp=cur_right_jp,
                    current_left_gp=cur_left_gp,
                    current_right_gp=cur_right_gp,
                    left_positions=left_positions,
                    right_positions=right_positions,
                    timestamps=timestamps,
                    final_pos_error_m=pos_err,
                    final_rot_error_deg=rot_err if np.isfinite(rot_err) else 0.0,
                    final_pose_error=self._pose_error_metric(
                        pos_err,
                        rot_err if np.isfinite(rot_err) else 0.0,
                        xyz_weight=ik_xyz_weight,
                        rpy_weight=ik_rpy_weight,
                    ),
                )

                row_payloads.append(
                    {
                        "source_index": int(candidate["source_index"]),
                        "position": self._round_list(candidate["position"], 5),
                        "rpy": self._round_list(candidate["rpy"], 4),
                        "score": round(float(candidate["score"]), 4),
                        "width": round(float(candidate["width"]), 5),
                        "ik_error_m": round(pos_err, 6),
                        "ik_rot_error_deg": round(rot_err, 4)
                        if np.isfinite(rot_err)
                        else None,
                        "within_ik_threshold": bool(
                            pos_err <= ik_threshold
                            and np.isfinite(rot_err)
                            and rot_err <= ik_rot_threshold_deg
                        ),
                        "planner_status": "Success",
                        "motion_plan_error": False,
                        "motion_plan_reason": None,
                        "trajectory_cache_key": trajectory_cache_key,
                        "trajectory_steps": trajectory_steps,
                    }
                )
            postprocess_elapsed_ms = (time.perf_counter() - postprocess_start) * 1000.0

            rank_start = time.perf_counter()
            row_payloads.sort(key=self._batch_rank_key)
            ranked_rows = [
                FreespaceBatchCandidate(rank=idx, **row)
                for idx, row in enumerate(row_payloads, start=1)
            ]
            rank_elapsed_ms = (time.perf_counter() - rank_start) * 1000.0
            total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

            best_candidate = next(
                (
                    candidate
                    for candidate in ranked_rows
                    if candidate.motion_plan_error is False
                    and candidate.ik_error_m is not None
                ),
                None,
            )
            feasible_count = sum(
                candidate.motion_plan_error is False
                and candidate.ik_error_m is not None
                for candidate in ranked_rows
            )
            within_threshold_count = sum(
                candidate.within_ik_threshold is True for candidate in ranked_rows
            )
            ik_failed_count = sum(
                candidate.planner_status == "IK_Failed" for candidate in ranked_rows
            )
            motion_error_count = sum(
                candidate.motion_plan_error is True for candidate in ranked_rows
            )
            best_pos_err = (
                float(best_candidate.ik_error_m)
                if best_candidate and best_candidate.ik_error_m is not None
                else 0.0
            )
            best_rot_err = (
                float(best_candidate.ik_rot_error_deg)
                if best_candidate and best_candidate.ik_rot_error_deg is not None
                else 0.0
            )
            best_pose_err = (
                self._pose_error_metric(
                    best_pos_err,
                    best_rot_err,
                    xyz_weight=ik_xyz_weight,
                    rpy_weight=ik_rpy_weight,
                )
                if best_candidate is not None
                else 0.0
            )
            reason = (
                f"Batch evaluated {len(ranked_rows)} grasp candidate(s) on {side}: "
                f"{feasible_count} executable, {within_threshold_count} within threshold, "
                f"{ik_failed_count} IK failed, {motion_error_count} planning failed."
            )
            return ToolResult(
                success=True,
                data=FreespaceResult(
                    status="Success",
                    ik_error_m=round(best_pos_err, 6),
                    final_pos_error_m=round(best_pos_err, 6),
                    final_rot_error_deg=round(best_rot_err, 4),
                    final_pose_error=round(best_pose_err, 6),
                    trajectory_steps=0,
                    executed=False,
                    reason=reason,
                    planning_mode="batch",
                    side=side,
                    batch_candidates=ranked_rows,
                    best_candidate=best_candidate,
                    input_candidate_count=original_candidate_count,
                    evaluated_candidate_count=len(candidate_records),
                    truncated_input_count=max(
                        0, original_candidate_count - len(candidate_records)
                    ),
                    batch_attempted=batch_attempted,
                    batch_error=batch_error,
                    timing_total_ms=round(total_elapsed_ms, 3),
                    timing_plan_eval_ms=round(plan_eval_elapsed_ms, 3),
                    timing_postprocess_ms=round(postprocess_elapsed_ms, 3),
                    timing_rank_ms=round(rank_elapsed_ms, 3),
                    curobo_solve_time_ms=round(
                        float(batch_plan_result.get("curobo_solve_time_ms", 0.0)), 3
                    ),
                    curobo_total_time_ms=round(
                        float(batch_plan_result.get("curobo_total_time_ms", 0.0)), 3
                    ),
                    curobo_graph_time_ms=round(
                        float(batch_plan_result.get("curobo_graph_time_ms", 0.0)), 3
                    ),
                    curobo_ik_time_ms=round(
                        float(batch_plan_result.get("curobo_ik_time_ms", 0.0)), 3
                    ),
                ),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=FreespaceResult(
                    status="Error",
                    executed=False,
                    reason=str(e),
                    planning_mode="batch_error",
                ),
                error=str(e),
            )

    def execute(self, **kwargs: Any) -> ToolResult:
        trajectory_cache_key = kwargs.get("trajectory_cache_key")
        grasp_candidates = kwargs.get("grasp_candidates")
        if trajectory_cache_key is not None:
            has_single_pose_targets = any(
                kwargs.get(name) is not None
                for name in (
                    "left_target_pos",
                    "left_target_rpy",
                    "left_target_quat",
                    "right_target_pos",
                    "right_target_rpy",
                    "right_target_quat",
                )
            )
            if grasp_candidates is not None or has_single_pose_targets:
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Invalid",
                        executed=False,
                        planning_mode="cached",
                        reason=(
                            "Use trajectory_cache_key by itself; do not combine it with target poses "
                            "or grasp_candidates."
                        ),
                    ),
                    error="trajectory_cache_key cannot be combined with other planning inputs.",
                )
            if bool(kwargs.get("preview_only", False)):
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Invalid",
                        executed=False,
                        planning_mode="cached",
                        reason="trajectory_cache_key is execution-only; preview must happen earlier.",
                    ),
                    error="trajectory_cache_key does not support preview_only.",
                )
            try:
                state = self._get_robot_state()
                cur_left_jp = np.asarray(state["left_joint_pos"], dtype=np.float64)
                cur_right_jp = np.asarray(state["right_joint_pos"], dtype=np.float64)
                cache_entry = self._get_cached_trajectory(trajectory_cache_key)
                if cache_entry is None:
                    return ToolResult(
                        success=False,
                        data=FreespaceResult(
                            status="Preview_Missing",
                            executed=False,
                            planning_mode="cached",
                            reason=(
                                "Cached batch trajectory was not found. Preview/sort the grasp again "
                                "before executing."
                            ),
                        ),
                        error="trajectory cache entry not found",
                    )
                if not self._cached_trajectory_matches_current(
                    cache_entry,
                    cur_left_jp,
                    cur_right_jp,
                ):
                    return ToolResult(
                        success=False,
                        data=FreespaceResult(
                            status="Preview_Stale",
                            executed=False,
                            planning_mode="cached",
                            side=str(cache_entry.get("side", "")) or None,
                            reason=(
                                "Robot state changed since batch preview. Preview/sort again before "
                                "executing so execution stays identical to the previewed trajectory."
                            ),
                        ),
                        error="trajectory cache entry is stale",
                    )
                return self._execute_cached_trajectory(cache_entry=cache_entry)
            except Exception as e:
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Error", reason=str(e), planning_mode="cached"
                    ),
                    error=str(e),
                )

        if grasp_candidates is not None:
            has_single_pose_targets = any(
                kwargs.get(name) is not None
                for name in (
                    "left_target_pos",
                    "left_target_rpy",
                    "left_target_quat",
                    "right_target_pos",
                    "right_target_rpy",
                    "right_target_quat",
                )
            )
            if has_single_pose_targets:
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Invalid",
                        executed=False,
                        reason=(
                            "Use either single EE target arguments or grasp_candidates batch mode, not both."
                        ),
                        planning_mode="batch_error",
                    ),
                    error="Cannot combine single EE target arguments with grasp_candidates.",
                )
            backend = self._normalize_backend(kwargs.get("backend", _DEFAULT_BACKEND))
            planner_backend = self._normalize_planner_backend(
                kwargs.get("planner_backend", "curobo")
            )
            if backend != "curobo":
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Invalid",
                        executed=False,
                        reason="Batch grasp ranking is only supported on the cuRobo backend.",
                        planning_mode="batch_error",
                    ),
                    error="Batch grasp ranking requires backend='curobo'.",
                )
            solver_speed = self._normalize_solver_speed(
                kwargs["solver_speed"]
                if "solver_speed" in kwargs
                else _DEFAULT_BATCH_SOLVER_SPEED
            )
            ik_threshold = float(
                kwargs.get("ik_error_threshold", _DEFAULT_IK_THRESHOLD)
            )
            ik_rot_threshold_deg = float(
                kwargs.get("ik_rot_threshold_deg", _DEFAULT_IK_ROT_THRESHOLD_DEG)
            )
            ik_xyz_weight = float(kwargs.get("ik_xyz_weight", _DEFAULT_IK_XYZ_WEIGHT))
            ik_rpy_weight = float(kwargs.get("ik_rpy_weight", _DEFAULT_IK_RPY_WEIGHT))
            planning_speed = self._resolve_planning_speed(
                kwargs.get("planning_speed"),
                kwargs.get("max_joint_vel"),
            )
            return self._rank_batch_grasp_candidates(
                grasp_candidates=list(grasp_candidates),
                batch_side=kwargs.get("batch_side"),
                ik_threshold=ik_threshold,
                ik_rot_threshold_deg=ik_rot_threshold_deg,
                ik_xyz_weight=ik_xyz_weight,
                ik_rpy_weight=ik_rpy_weight,
                planning_speed=planning_speed,
                planner_backend=planner_backend,
                solver_speed=solver_speed,
                left_gripper=kwargs.get("left_gripper"),
                right_gripper=kwargs.get("right_gripper"),
                batch_top_k=kwargs.get("batch_top_k", _DEFAULT_BATCH_TOP_K),
                batch_validate_trajectory=bool(
                    kwargs.get(
                        "batch_validate_trajectory",
                        _DEFAULT_BATCH_VALIDATE_TRAJECTORY,
                    )
                ),
            )

        left_target_pos = kwargs.get("left_target_pos")
        left_target_rpy = kwargs.get("left_target_rpy")
        left_target_quat_raw = kwargs.get("left_target_quat")
        right_target_pos = kwargs.get("right_target_pos")
        right_target_rpy = kwargs.get("right_target_rpy")
        right_target_quat_raw = kwargs.get("right_target_quat")
        left_target_quat = (
            np.asarray(left_target_quat_raw, dtype=np.float64)
            if left_target_quat_raw is not None
            else (
                self._display_rpy_to_quat(left_target_rpy)
                if left_target_rpy is not None
                else None
            )
        )
        right_target_quat = (
            np.asarray(right_target_quat_raw, dtype=np.float64)
            if right_target_quat_raw is not None
            else (
                self._display_rpy_to_quat(right_target_rpy)
                if right_target_rpy is not None
                else None
            )
        )
        ik_threshold = float(kwargs.get("ik_error_threshold", _DEFAULT_IK_THRESHOLD))
        ik_rot_threshold_deg = float(
            kwargs.get("ik_rot_threshold_deg", _DEFAULT_IK_ROT_THRESHOLD_DEG)
        )
        ik_xyz_weight = float(kwargs.get("ik_xyz_weight", _DEFAULT_IK_XYZ_WEIGHT))
        ik_rpy_weight = float(kwargs.get("ik_rpy_weight", _DEFAULT_IK_RPY_WEIGHT))
        solver_speed = self._normalize_solver_speed(
            kwargs.get("solver_speed", _DEFAULT_SOLVER_SPEED)
        )
        backend = self._normalize_backend(kwargs.get("backend", _DEFAULT_BACKEND))
        left_gripper = kwargs.get("left_gripper")
        right_gripper = kwargs.get("right_gripper")
        left_gripper_target_width = self._clip_gripper_width(
            kwargs.get("left_gripper_target_width")
        )
        right_gripper_target_width = self._clip_gripper_width(
            kwargs.get("right_gripper_target_width")
        )
        planning_speed = self._resolve_planning_speed(
            kwargs.get("planning_speed"),
            kwargs.get("max_joint_vel"),
        )
        preview_only = bool(kwargs.get("preview_only", False))
        planner_backend = self._normalize_planner_backend(kwargs.get("planner_backend"))

        # Validate: at least one target must be provided
        has_left = (
            left_target_pos is not None
            or left_target_rpy is not None
            or left_target_quat is not None
        )
        has_right = (
            right_target_pos is not None
            or right_target_rpy is not None
            or right_target_quat is not None
        )
        if not has_left and not has_right:
            return ToolResult(
                success=False,
                data=FreespaceResult(
                    status="Invalid", reason="No target pose provided."
                ),
                error="Provide at least one left/right target position or RPY.",
            )

        # Warn on large IK threshold
        if ik_threshold > _MAX_IK_THRESHOLD:
            logger.warning(
                "IK error threshold %.3f m exceeds %.3f m — results may be imprecise",
                ik_threshold,
                _MAX_IK_THRESHOLD,
            )

        try:
            # 1. Get current robot state
            state = self._get_robot_state()
            cur_left_jp = np.asarray(state.get("left_joint_pos", np.zeros(0)))
            cur_right_jp = np.asarray(state.get("right_joint_pos", np.zeros(0)))
            cur_left_gp = (
                float(state["left_gripper_pos"][0])
                if "left_gripper_pos" in state
                else 0.0
            )
            cur_right_gp = (
                float(state["right_gripper_pos"][0])
                if "right_gripper_pos" in state
                else 0.0
            )

            # Use the largest width encountered during execution for collision checking.
            # If no explicit collision width is provided, fold in the commanded target width
            # so planning reflects opening while the arms are moving.
            if left_gripper is None:
                left_gripper = (
                    max(cur_left_gp, left_gripper_target_width)
                    if left_gripper_target_width is not None
                    else cur_left_gp
                )
            if right_gripper is None:
                right_gripper = (
                    max(cur_right_gp, right_gripper_target_width)
                    if right_gripper_target_width is not None
                    else cur_right_gp
                )

            # 2. Plan trajectory via motion planner
            planner = self._get_planner(
                planner_backend=planner_backend,
                solver_speed=solver_speed,
                ik_error_threshold=ik_threshold,
                ik_rot_threshold_deg=ik_rot_threshold_deg,
                ik_xyz_weight=ik_xyz_weight,
                ik_rpy_weight=ik_rpy_weight,
            )
            diagnostic_kin = (
                self._get_diagnostic_kinematics(
                    position_cost=ik_xyz_weight,
                    orientation_cost=ik_rpy_weight,
                )
                if self._detect_robot_type() == "yam"
                else None
            )
            if diagnostic_kin is not None:
                cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = (
                    diagnostic_kin.forward_kinematics(cur_left_jp, cur_right_jp)
                )
            else:
                # Non-YAM robot — use EE state from cap_server observations
                cur_l_pos = state.get("left_ee_pos", np.zeros(3))
                cur_l_q = state.get("left_ee_quat_xyzw", np.array([0, 0, 0, 1.0]))
                cur_r_pos = state.get("right_ee_pos", np.zeros(3))
                cur_r_q = state.get("right_ee_quat_xyzw", np.array([0, 0, 0, 1.0]))
            tgt_l_pos = (
                np.asarray(left_target_pos, dtype=np.float64)
                if left_target_pos is not None
                else np.asarray(cur_l_pos, dtype=np.float64)
            )
            tgt_r_pos = (
                np.asarray(right_target_pos, dtype=np.float64)
                if right_target_pos is not None
                else np.asarray(cur_r_pos, dtype=np.float64)
            )
            tgt_l_q = (
                np.asarray(left_target_quat, dtype=np.float64)
                if left_target_quat is not None
                else np.asarray(cur_l_q, dtype=np.float64)
            )
            tgt_r_q = (
                np.asarray(right_target_quat, dtype=np.float64)
                if right_target_quat is not None
                else np.asarray(cur_r_q, dtype=np.float64)
            )
            side = self._infer_effective_side(
                has_left=has_left,
                has_right=has_right,
                tgt_l_pos=tgt_l_pos,
                tgt_l_q=tgt_l_q,
                tgt_r_pos=tgt_r_pos,
                tgt_r_q=tgt_r_q,
                cur_l_pos=np.asarray(cur_l_pos, dtype=np.float64),
                cur_l_q=np.asarray(cur_l_q, dtype=np.float64),
                cur_r_pos=np.asarray(cur_r_pos, dtype=np.float64),
                cur_r_q=np.asarray(cur_r_q, dtype=np.float64),
                left_gripper_target_width=left_gripper_target_width,
                right_gripper_target_width=right_gripper_target_width,
            )
            if side is None:
                return ToolResult(
                    success=True,
                    data=FreespaceResult(
                        status="Success",
                        ik_error_m=0.0,
                        final_pos_error_m=0.0,
                        final_rot_error_deg=0.0,
                        final_pose_error=0.0,
                        trajectory_steps=0,
                        executed=False,
                        reason="Target pose already matches the current robot pose; no move executed.",
                    ),
                )
            # For mobile-base robots (e.g. PandaOmron), transform world-frame
            # EE targets to the arm base frame so cuRobo plans correctly.
            _plan_left_pos = left_target_pos
            _plan_left_quat = (
                left_target_quat if left_target_quat is not None else tgt_l_q
            )
            _plan_right_pos = right_target_pos
            _plan_right_quat = (
                right_target_quat if right_target_quat is not None else tgt_r_q
            )
            if self._detect_robot_type() == "panda" and "base_pos" in state:
                from scipy.spatial.transform import Rotation as R

                base_pos = np.asarray(state["base_pos"], dtype=np.float64)
                base_quat = np.asarray(
                    state.get("base_quat_xyzw", [0, 0, 0, 1]), dtype=np.float64
                )
                R_base = R.from_quat(base_quat)
                if _plan_left_pos is not None:
                    _plan_left_pos = R_base.inv().apply(
                        np.asarray(_plan_left_pos) - base_pos
                    )
                    if _plan_left_quat is not None:
                        R_target = R.from_quat(np.asarray(_plan_left_quat))
                        _plan_left_quat = (R_base.inv() * R_target).as_quat()
                if _plan_right_pos is not None:
                    _plan_right_pos = R_base.inv().apply(
                        np.asarray(_plan_right_pos) - base_pos
                    )
                    if _plan_right_quat is not None:
                        R_target = R.from_quat(np.asarray(_plan_right_quat))
                        _plan_right_quat = (R_base.inv() * R_target).as_quat()

            # RPC outside _planner_lock — planner ref is held so it
            # cannot be garbage-collected; Portal serialises on the
            # server side, so we don't need Python-side locking here.
            logger.debug("_planner_lock released before single-target RPC plan_to_pose")
            plan_result = planner.plan_to_pose(
                current_left_jp=cur_left_jp,
                current_right_jp=cur_right_jp,
                target_left_pos=(
                    np.asarray(_plan_left_pos)
                    if _plan_left_pos is not None and side in ("left", "both")
                    else None
                ),
                target_left_quat_xyzw=(
                    np.asarray(_plan_left_quat)
                    if _plan_left_quat is not None and side in ("left", "both")
                    else None
                ),
                target_right_pos=(
                    np.asarray(_plan_right_pos)
                    if _plan_right_pos is not None and side in ("right", "both")
                    else None
                ),
                target_right_quat_xyzw=(
                    np.asarray(_plan_right_quat)
                    if _plan_right_quat is not None and side in ("right", "both")
                    else None
                ),
                side=side,
                ik_error_threshold=ik_threshold,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                max_joint_vel=planning_speed,
            )

            status = plan_result["status"]

            if status == "IK_Failed":
                diag = self._ik_failed_metrics(
                    planner=planner,
                    plan_result=plan_result,
                )
                max_err = float(diag["final_pos_error_m"])
                max_rot_err = float(diag["final_rot_error_deg"])
                max_pose_err = float(diag["final_pose_error"])
                residual_reason = (
                    f"IK did not converge: pos_err={max_err:.4f} m, "
                    f"rot_err={max_rot_err:.2f} deg (reported by cuRobo)."
                    if bool(diag["reported_by_curobo"])
                    else "IK did not converge: cuRobo returned IK_Failed but did not report finite residuals."
                )

                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="IK_Failed",
                        ik_error_m=round(max_err, 4),
                        final_pos_error_m=round(max_err, 6),
                        final_rot_error_deg=round(max_rot_err, 4),
                        final_pose_error=round(max_pose_err, 6),
                        executed=False,
                        reason=residual_reason,
                    ),
                    error=(
                        f"IK failed: {residual_reason}"
                        if bool(diag["reported_by_curobo"])
                        else "IK failed: cuRobo returned IK_Failed without finite residuals"
                    ),
                )

            if status == "Planning_Failed":
                planner_label = (
                    "cuRobo" if planner_backend == "curobo" else "RRT-Connect"
                )
                status_detail = str(plan_result.get("status_detail") or "").strip()
                reason = (
                    f"{planner_label} could not find a collision-free path. "
                    "The target may be blocked or require a complex manoeuvre."
                )
                error = "Motion planning failed: no collision-free path found."
                if status_detail:
                    reason = f"{reason} Planner detail: {status_detail}."
                    error = f"{error} Detail: {status_detail}."
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Planning_Failed",
                        executed=False,
                        reason=reason,
                    ),
                    error=error,
                )

            # 3. Execute trajectory via cap_server
            left_waypoints = np.asarray(
                plan_result.get("left_waypoints", plan_result["left_positions"]),
                dtype=np.float64,
            )
            right_waypoints = np.asarray(
                plan_result.get("right_waypoints", plan_result["right_positions"]),
                dtype=np.float64,
            )
            left_positions, right_positions = self._densify_joint_waypoints(
                left_waypoints,
                right_waypoints,
            )
            n_steps = len(left_positions)
            timestamps = self._timestamps_from_waypoints(
                left_positions,
                right_positions,
                planning_speed=planning_speed,
            )
            total_duration = float(timestamps[-1]) if timestamps else 0.0
            progress = (
                np.asarray(timestamps, dtype=np.float64) / total_duration
                if n_steps > 1 and total_duration > 1e-9
                else np.zeros(n_steps, dtype=np.float64)
            )
            left_gripper_positions = (
                cur_left_gp + (left_gripper_target_width - cur_left_gp) * progress
                if left_gripper_target_width is not None and n_steps > 0
                else None
            )
            right_gripper_positions = (
                cur_right_gp + (right_gripper_target_width - cur_right_gp) * progress
                if right_gripper_target_width is not None and n_steps > 0
                else None
            )

            final_left_joint_pos = left_positions[-1] if n_steps > 0 else cur_left_jp
            final_right_joint_pos = right_positions[-1] if n_steps > 0 else cur_right_jp
            if diagnostic_kin is not None:
                diag = self._compute_plan_diagnostics(
                    diagnostic_kin,
                    planner,
                    side=side,
                    final_left_joint_pos=final_left_joint_pos,
                    final_right_joint_pos=final_right_joint_pos,
                    target_left_pos=tgt_l_pos,
                    target_left_quat_xyzw=tgt_l_q,
                    target_right_pos=tgt_r_pos,
                    target_right_quat_xyzw=tgt_r_q,
                )
            else:
                # Use cuRobo-reported errors
                diag = {
                    "final_pos_error_m": float(
                        plan_result.get("position_error_m", 0.0)
                    ),
                    "final_rot_error_deg": float(
                        plan_result.get("rotation_error_deg", 0.0)
                    ),
                    "final_pose_error": float(plan_result.get("position_error_m", 0.0)),
                }

            trajectory_cache_key = self._store_cached_trajectory(
                side=side,
                current_left_jp=cur_left_jp,
                current_right_jp=cur_right_jp,
                current_left_gp=cur_left_gp,
                current_right_gp=cur_right_gp,
                left_positions=left_positions,
                right_positions=right_positions,
                timestamps=timestamps,
                left_gripper_positions=left_gripper_positions,
                right_gripper_positions=right_gripper_positions,
                final_pos_error_m=float(diag["final_pos_error_m"]),
                final_rot_error_deg=float(diag["final_rot_error_deg"]),
                final_pose_error=float(diag["final_pose_error"]),
            )

            if preview_only:
                return ToolResult(
                    success=True,
                    data=FreespaceResult(
                        status="Success",
                        ik_error_m=round(float(diag["final_pos_error_m"]), 4),
                        final_pos_error_m=round(float(diag["final_pos_error_m"]), 6),
                        final_rot_error_deg=round(
                            float(diag["final_rot_error_deg"]), 4
                        ),
                        final_pose_error=round(float(diag["final_pose_error"]), 6),
                        trajectory_steps=n_steps,
                        trajectory_cache_key=trajectory_cache_key,
                        executed=False,
                        planning_mode="single",
                        side=side,
                    ),
                )

            cache_entry = self._get_cached_trajectory(trajectory_cache_key)
            if cache_entry is None:
                return ToolResult(
                    success=False,
                    data=FreespaceResult(
                        status="Error",
                        ik_error_m=round(float(diag["final_pos_error_m"]), 4),
                        final_pos_error_m=round(float(diag["final_pos_error_m"]), 6),
                        final_rot_error_deg=round(
                            float(diag["final_rot_error_deg"]), 4
                        ),
                        final_pose_error=round(float(diag["final_pose_error"]), 6),
                        trajectory_steps=n_steps,
                        executed=False,
                        planning_mode="single",
                        side=side,
                        reason="Planned trajectory could not be retrieved from cache immediately after planning.",
                    ),
                    error="planned trajectory cache retrieval failed",
                )
            return self._execute_cached_trajectory(
                cache_entry=cache_entry,
                planning_mode="single",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=FreespaceResult(status="Error", reason=str(e)),
                error=str(e),
            )

    def _execute_trajectory(
        self,
        side: str,
        timestamps: list[float],
        left_positions: np.ndarray,
        right_positions: np.ndarray,
        left_gripper_positions: np.ndarray | None = None,
        right_gripper_positions: np.ndarray | None = None,
    ) -> str | None:
        """Send joint trajectory through Portal or the direct env. Returns an error string."""

        if self._env is not None:
            from cap.agent.tools.native import execute_bimanual_joint_keypoints_direct

            result = execute_bimanual_joint_keypoints_direct(
                self._env,
                np.asarray(timestamps, dtype=np.float64),
                left_positions.astype(np.float64),
                right_positions.astype(np.float64),
                None
                if left_gripper_positions is None
                else left_gripper_positions.astype(np.float64),
                None
                if right_gripper_positions is None
                else right_gripper_positions.astype(np.float64),
            )
            if not bool(result.get("success", False)):
                return str(result.get("reason", "direct bimanual execution failed"))
            return None

        client = self._get_client()

        def _send(
            arm_side: str,
            positions: np.ndarray,
            gripper_positions: np.ndarray | None = None,
        ) -> str | None:
            result = client.move_joint_keypoints(
                arm_side,
                np.asarray(timestamps, dtype=np.float64),
                positions.astype(np.float64),
                None
                if gripper_positions is None
                else gripper_positions.astype(np.float64),
            ).result()
            if not result.get("success", False):
                return result.get("reason", f"{arm_side} arm execution failed")
            return None

        needs_bimanual = (
            side == "both"
            or (side == "left" and right_gripper_positions is not None)
            or (side == "right" and left_gripper_positions is not None)
        )

        if side == "left" and not needs_bimanual:
            return _send("left", left_positions, left_gripper_positions)
        elif side == "right" and not needs_bimanual:
            return _send("right", right_positions, right_gripper_positions)
        else:
            result = client.move_bimanual_joint_keypoints(
                np.asarray(timestamps, dtype=np.float64),
                left_positions.astype(np.float64),
                right_positions.astype(np.float64),
                None
                if left_gripper_positions is None
                else left_gripper_positions.astype(np.float64),
                None
                if right_gripper_positions is None
                else right_gripper_positions.astype(np.float64),
            ).result()
            if not result.get("success", False):
                return result.get("reason", "bimanual execution failed")
            return None
