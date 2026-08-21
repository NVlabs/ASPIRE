# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import logging
import socket
import time
from typing import Any

import numpy as np
import portal

from experimental.curobo_depth_world import intrinsics_dict_to_matrix
from experimental.key_remapping_utils import _make_arrays_contiguous
from robot.yam.kinematics import YamKinematics


logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class PortalMotionPlannerConfig:
    backend: str = "curobo"
    solver_speed: str = "fast"
    port: int = 0
    position_threshold: float = 0.005
    rotation_threshold: float = 0.05
    enable_depth_collision: bool = False
    collision_checking: bool = True
    robot_type: str = "yam"  # "yam" or "panda"


class PortalMotionPlannerServer:
    def __init__(self, config: PortalMotionPlannerConfig):
        self.config = config
        self._planners: dict[str, Any] = {}  # robot_type -> planner
        self._planner = self._get_planner(
            str(getattr(config, "robot_type", "yam")).strip().lower()
        )
        self._server = portal.Server(self.config.port)
        self._server.bind("health_check", self._handle_health_check)
        self._server.bind("set_finetune_enabled", self._handle_set_finetune_enabled)
        self._server.bind("set_gripper_qpos", self._handle_set_gripper_qpos)
        self._server.bind(
            "set_depth_collision_scene", self._handle_set_depth_collision_scene
        )
        self._server.bind(
            "clear_depth_collision_scene", self._handle_clear_depth_collision_scene
        )
        self._server.bind(
            "set_debug_collision_ball", self._handle_set_debug_collision_ball
        )
        self._server.bind(
            "clear_debug_collision_ball", self._handle_clear_debug_collision_ball
        )
        self._server.bind("plan_to_pose", self._handle_plan_to_pose)
        self._server.bind("plan_batch_to_pose", self._handle_plan_batch_to_pose)
        self._server.bind(
            "update_world_from_geoms", self._handle_update_world_from_geoms
        )

    def _get_planner(self, robot_type: str) -> Any:
        """Get or lazily create a planner for the given robot type."""
        robot_type = robot_type.strip().lower()
        if robot_type in self._planners:
            return self._planners[robot_type]

        print(
            f"[PortalMotionPlannerServer] Loading planner for robot_type={robot_type}"
        )
        backend = str(self.config.backend).strip().lower()
        if backend != "curobo":
            raise ValueError(f"Unsupported portal motion planner backend: {backend}")

        if robot_type == "panda":
            from experimental.motion_planner_curobo_panda import (
                PandaMotionPlannerCurobo,
            )

            planner = PandaMotionPlannerCurobo(
                solver_speed=str(self.config.solver_speed),
                position_threshold=float(self.config.position_threshold),
                rotation_threshold=float(self.config.rotation_threshold),
            )
        else:
            from experimental.motion_planner_curobo import YamMotionPlannerCurobo

            planner = YamMotionPlannerCurobo(
                solver_speed=str(self.config.solver_speed),
                position_threshold=float(self.config.position_threshold),
                rotation_threshold=float(self.config.rotation_threshold),
                enable_depth_collision=bool(self.config.enable_depth_collision),
                collision_checking=bool(self.config.collision_checking),
                validate_with_mujoco=bool(self.config.collision_checking),
            )

        self._planners[robot_type] = planner
        return planner

    def _handle_health_check(self) -> bool:
        return True

    def _handle_set_finetune_enabled(self, enabled: bool) -> bool:
        if hasattr(self._planner, "set_finetune_enabled"):
            self._planner.set_finetune_enabled(bool(enabled))
        return True

    def _handle_set_gripper_qpos(self, payload: dict[str, Any] | None = None) -> bool:
        payload = payload or {}
        left_gripper = payload.get("left_gripper")
        right_gripper = payload.get("right_gripper")
        if hasattr(self._planner, "set_gripper_qpos"):
            self._planner.set_gripper_qpos(left_gripper, right_gripper)
        return True

    def _handle_set_depth_collision_scene(self, payload: dict[str, Any]) -> bool:
        if not hasattr(self._planner, "set_depth_collision_scene"):
            return False
        intrinsics = payload.get("intrinsics")
        intrinsics_matrix = (
            intrinsics_dict_to_matrix(intrinsics)
            if isinstance(intrinsics, dict)
            else np.asarray(intrinsics, dtype=np.float64)
        )
        return bool(
            self._planner.set_depth_collision_scene(
                depth_image=np.asarray(payload["depth_image"], dtype=np.float32),
                intrinsics=intrinsics_matrix,
                camera_pose=(
                    None
                    if payload.get("camera_pose") is None
                    else np.asarray(payload["camera_pose"], dtype=np.float64)
                ),
                joint_position=(
                    None
                    if payload.get("joint_position") is None
                    else np.asarray(payload["joint_position"], dtype=np.float32)
                ),
                segment_robot=bool(payload.get("segment_robot", True)),
                segment_distance_threshold=float(
                    payload.get("segment_distance_threshold", 0.08)
                ),
                segment_collision_sphere_buffer=float(
                    payload.get("segment_collision_sphere_buffer", 0.015)
                ),
                segment_mask_dilation_pixels=int(
                    payload.get("segment_mask_dilation_pixels", 2)
                ),
                depth_clip_range=tuple(payload.get("depth_clip_range", (0.15, 2.0))),
                max_points=int(payload.get("max_points", 25000)),
                marching_cubes_pitch=float(payload.get("marching_cubes_pitch", 0.04)),
                scene_name=str(payload.get("scene_name", "zed_depth_scene")),
            )
        )

    def _handle_clear_depth_collision_scene(self) -> bool:
        if not hasattr(self._planner, "clear_depth_collision_scene"):
            return False
        return bool(self._planner.clear_depth_collision_scene())

    def _handle_set_debug_collision_ball(self, payload: dict[str, Any]) -> bool:
        if not hasattr(self._planner, "set_debug_collision_ball"):
            return False
        return bool(
            self._planner.set_debug_collision_ball(
                position=np.asarray(payload["position"], dtype=np.float64),
                radius=float(payload["radius"]),
                scene_name=str(payload.get("scene_name", "debug_collision_ball")),
            )
        )

    def _handle_clear_debug_collision_ball(self) -> bool:
        if not hasattr(self._planner, "clear_debug_collision_ball"):
            return False
        return bool(self._planner.clear_debug_collision_ball())

    def _resolve_planner(self, kwargs: dict[str, Any]) -> Any:
        """Switch planner if request specifies a different robot_type."""
        robot_type = str(kwargs.pop("robot_type", "")).strip().lower()
        if robot_type and robot_type != getattr(self._planner, "ROBOT_TYPE", ""):
            self._planner = self._get_planner(robot_type)
        return self._planner

    def _handle_plan_to_pose(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            planner = self._resolve_planner(kwargs)
            result = planner.plan_to_pose(**kwargs)
        except Exception as exc:
            logger.exception(
                "[PortalMotionPlannerServer] plan_to_pose failed: %s",
                exc,
            )
            return {
                "status": "Error",
                "status_detail": str(exc),
                "error": str(exc),
                "position": np.empty((0, 7), dtype=np.float64),
                "left_positions": np.empty((0, 7), dtype=np.float64),
                "right_positions": np.empty((0, 7), dtype=np.float64),
                "position_error_m": float("nan"),
                "rotation_error_deg": float("nan"),
            }
        return _make_arrays_contiguous(result)

    def _handle_update_world_from_geoms(self, payload: dict[str, Any]) -> int:
        robot_type = str(payload.pop("robot_type", "")).strip().lower()
        if robot_type:
            self._planner = self._get_planner(robot_type)
        if not hasattr(self._planner, "update_world_from_geoms"):
            return 0
        return self._planner.update_world_from_geoms(
            payload=payload,
            base_pos=np.asarray(payload["base_pos"], dtype=np.float64),
            base_quat_xyzw=np.asarray(payload["base_quat_xyzw"], dtype=np.float64),
        )

    def _handle_plan_batch_to_pose(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            planner = self._resolve_planner(kwargs)
            result = planner.plan_batch_to_pose(**kwargs)
        except Exception as exc:
            logger.exception(
                "[PortalMotionPlannerServer] plan_batch_to_pose failed: %s",
                exc,
            )
            return {
                "status": "Error",
                "error": str(exc),
                "success_mask": np.zeros((0,), dtype=bool),
                "status_by_index": [],
                "status_detail_by_index": [],
                "position_error_m": np.array([], dtype=np.float64),
                "rotation_error_deg": np.array([], dtype=np.float64),
                "left_positions_by_index": [],
                "right_positions_by_index": [],
            }
        return _make_arrays_contiguous(result)

    def serve(self) -> None:
        print(
            f"[PortalMotionPlannerServer] Starting {self.config.backend} planner "
            f"(solver_speed={self.config.solver_speed}) on port {self.config.port}"
        )
        self._server.start()


def _run_motion_planner_server(config: PortalMotionPlannerConfig) -> None:
    server = PortalMotionPlannerServer(config)
    server.serve()


class PortalMotionPlanner:
    def __init__(
        self,
        backend: str = "curobo",
        *,
        solver_speed: str = "fast",
        host: str = "127.0.0.1",
        port: int | None = None,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        enable_depth_collision: bool = False,
        startup_timeout: float = 120.0,
        start_server: bool = True,
        robot_type: str = "yam",
    ) -> None:
        self._backend = str(backend).strip().lower()
        if port is None and not start_server:
            raise ValueError(
                "PortalMotionPlanner requires an explicit port when start_server=False"
            )
        self._host = str(host or "127.0.0.1").strip() or "127.0.0.1"
        self._port = int(port or _find_free_port())
        self._robot_type = str(robot_type).strip().lower()
        self._config = PortalMotionPlannerConfig(
            backend=self._backend,
            solver_speed=str(solver_speed).strip().lower(),
            port=self._port,
            position_threshold=float(position_threshold),
            rotation_threshold=float(rotation_threshold),
            enable_depth_collision=bool(enable_depth_collision),
            robot_type=self._robot_type,
        )
        self._client: portal.Client | None = None
        self._process = None
        self._finetune_enabled = False
        self._left_gripper = 1.0
        self._right_gripper = 1.0
        if self._robot_type != "panda":
            self._kin = YamKinematics()
        else:
            self._kin = None
        self._position_threshold = float(position_threshold)
        self._rotation_threshold = float(rotation_threshold)
        self._owns_server = bool(start_server)
        if self._owns_server:
            self._start_subprocess()
        self._wait_for_ready(startup_timeout)

    def _start_subprocess(self) -> None:
        config = self._config

        def _run_server() -> None:
            _run_motion_planner_server(config)

        self._process = portal.Process(_run_server, start=True)
        print(
            f"[PortalMotionPlanner] Started {self._backend} subprocess "
            f"(solver_speed={self._config.solver_speed}) on port {self._port}"
        )

    @property
    def port(self) -> int:
        return int(self._port)

    def _wait_for_ready(self, timeout: float) -> None:
        start = time.time()
        last_error = "unknown"
        while time.time() - start < timeout:
            try:
                self._client = portal.Client(f"{self._host}:{self._port}")
                if self._client.health_check().result():
                    return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.25)
        raise TimeoutError(
            f"Timed out waiting for portal motion planner at {self._host}:{self._port}: {last_error}"
        )

    def set_finetune_enabled(self, enabled: bool) -> None:
        self._finetune_enabled = bool(enabled)
        assert self._client is not None
        self._client.set_finetune_enabled(self._finetune_enabled).result()

    @property
    def finetune_enabled(self) -> bool:
        return bool(self._finetune_enabled)

    def set_gripper_qpos(
        self,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> None:
        if left_gripper is not None:
            self._left_gripper = float(np.clip(left_gripper, 0.0, 1.0))
        if right_gripper is not None:
            self._right_gripper = float(np.clip(right_gripper, 0.0, 1.0))
        assert self._client is not None
        self._client.set_gripper_qpos(
            {
                "left_gripper": self._left_gripper,
                "right_gripper": self._right_gripper,
            }
        ).result()

    def plan_to_pose(self, **kwargs: Any) -> dict[str, Any]:
        payload = dict(kwargs)
        if payload.get("left_gripper") is None:
            payload["left_gripper"] = self._left_gripper
        if payload.get("right_gripper") is None:
            payload["right_gripper"] = self._right_gripper
        payload["robot_type"] = self._robot_type
        assert self._client is not None
        return self._client.plan_to_pose(payload).result()

    def plan_batch_to_pose(self, **kwargs: Any) -> dict[str, Any]:
        payload = dict(kwargs)
        if payload.get("left_gripper") is None:
            payload["left_gripper"] = self._left_gripper
        if payload.get("right_gripper") is None:
            payload["right_gripper"] = self._right_gripper
        payload["robot_type"] = self._robot_type
        assert self._client is not None
        return self._client.plan_batch_to_pose(payload).result()

    def set_depth_collision_scene(
        self,
        *,
        depth_image: np.ndarray,
        intrinsics: np.ndarray | dict[str, Any],
        camera_pose: np.ndarray | None = None,
        joint_position: np.ndarray | None = None,
        segment_robot: bool = True,
        segment_distance_threshold: float = 0.08,
        segment_collision_sphere_buffer: float = 0.015,
        segment_mask_dilation_pixels: int = 2,
        depth_clip_range: tuple[float, float] = (0.15, 2.0),
        max_points: int = 25000,
        marching_cubes_pitch: float = 0.04,
        scene_name: str = "zed_depth_scene",
    ) -> bool:
        assert self._client is not None
        return bool(
            self._client.set_depth_collision_scene(
                {
                    "depth_image": np.asarray(depth_image, dtype=np.float32),
                    "intrinsics": intrinsics,
                    "camera_pose": (
                        None
                        if camera_pose is None
                        else np.asarray(camera_pose, dtype=np.float64)
                    ),
                    "joint_position": (
                        None
                        if joint_position is None
                        else np.asarray(joint_position, dtype=np.float32)
                    ),
                    "segment_robot": bool(segment_robot),
                    "segment_distance_threshold": float(segment_distance_threshold),
                    "segment_collision_sphere_buffer": float(
                        segment_collision_sphere_buffer
                    ),
                    "segment_mask_dilation_pixels": int(segment_mask_dilation_pixels),
                    "depth_clip_range": tuple(depth_clip_range),
                    "max_points": int(max_points),
                    "marching_cubes_pitch": float(marching_cubes_pitch),
                    "scene_name": str(scene_name),
                }
            ).result()
        )

    def clear_depth_collision_scene(self) -> bool:
        assert self._client is not None
        return bool(self._client.clear_depth_collision_scene().result())

    def set_debug_collision_ball(
        self,
        *,
        position: np.ndarray,
        radius: float,
        scene_name: str = "debug_collision_ball",
    ) -> bool:
        assert self._client is not None
        return bool(
            self._client.set_debug_collision_ball(
                {
                    "position": np.asarray(position, dtype=np.float64),
                    "radius": float(radius),
                    "scene_name": scene_name,
                }
            ).result()
        )

    def clear_debug_collision_ball(self) -> bool:
        assert self._client is not None
        return bool(self._client.clear_debug_collision_ball().result())

    def update_world_from_geoms(
        self,
        collision_data: dict,
        base_pos: np.ndarray | list | None = None,
        base_quat_xyzw: np.ndarray | list | None = None,
        timeout: float = 30.0,
    ) -> int:
        """Send collision geometry to the remote planner to update its world.

        collision_data: dict with numpy arrays (names, positions, rot_mats, dims_array,
        base_pos, base_quat_xyzw) as returned by CapServer.get_collision_geoms().
        """
        assert self._client is not None
        # Forward the payload directly — it already contains numpy arrays
        payload = dict(collision_data)
        payload["robot_type"] = self._robot_type
        if base_pos is not None:
            payload["base_pos"] = np.asarray(base_pos, dtype=np.float64)
        if base_quat_xyzw is not None:
            payload["base_quat_xyzw"] = np.asarray(base_quat_xyzw, dtype=np.float64)
        try:
            return int(
                self._client.update_world_from_geoms(payload).result(timeout=timeout)
            )
        except TimeoutError:
            raise RuntimeError(
                "Remote cuRobo server timed out on update_world_from_geoms. "
                "Restart the server with updated code to enable collision world updates."
            )
