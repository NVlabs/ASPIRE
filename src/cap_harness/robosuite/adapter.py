"""Typed, frame-explicit adapter for the supported Robosuite tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import invert_transform, matrix_to_pose, transform_pose
from cap_harness.robosuite.codec import RobosuiteActionCodec
from cap_harness.robosuite.registry import RobosuiteTaskMetadata, RobosuiteTaskRegistry

PANDA_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
TRAJECTORY_TOLERANCE_RAD = 0.01
TRAJECTORY_START_TOLERANCE_RAD = 0.02
TRAJECTORY_MAX_STEPS_PER_WAYPOINT = 120
_EEF_TO_HAND_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def _transform(rotation: Any, position: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return result


def _metric_depth(sim: Any, depth: np.ndarray) -> np.ndarray:
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    return near / (1.0 - depth * (1.0 - near / far))


class RobosuiteAdapter:
    """Normalize Robosuite 1.4 without exposing native state to generated code."""

    #: The gripper action on this embodiment is a velocity, not a position:
    #: robosuite's `simple_grip` controller assigns `goal_qvel`, so the sign picks
    #: a direction and the jaw travels to its stop. Only the two endpoints are
    #: states the actuator can hold.
    gripper_holds_intermediate = False

    BASE_FRAME = "robot0_base"
    SECONDARY_BASE_FRAME = "robot1_base"

    def __init__(
        self,
        *,
        registry: RobosuiteTaskRegistry | None = None,
        env_factory: Callable[..., Any] | None = None,
        control_frequency: float = 20.0,
        camera_height: int = 128,
        camera_width: int = 128,
        horizon: int = 1000,
        run_observer: Any | None = None,
    ) -> None:
        if camera_height <= 0 or camera_width <= 0 or horizon <= 0:
            raise ValueError("camera dimensions and horizon must be positive")
        self.registry = registry or RobosuiteTaskRegistry()
        self._env_factory = env_factory
        self.control_frequency = float(control_frequency)
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        self.control_period_s = 1.0 / self.control_frequency
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.horizon = int(horizon)
        self.controller = "JOINT_POSITION"
        self._run_observer = run_observer
        self._protocol_evaluator: Any | None = None
        self._env: Any | None = None
        self._metadata: RobosuiteTaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._raw_observation: Mapping[str, Any] | None = None
        self._codec: RobosuiteActionCodec | None = None
        self._action_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._elapsed_steps = 0
        self._seed: int | None = None
        self._gripper_targets: dict[str, float] = {}
        self._joint_targets: dict[str, np.ndarray] = {}
        self._last_reward: float | None = None
        self._last_terminated = False
        self._last_truncated = False
        self._closed = False

    @property
    def embodiment(self) -> str:
        return "robosuite"

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    def get_planning_context(self) -> RobotPlanningContext:
        """Return calibrated Panda chains without exposing task object state."""
        transforms = {"primary": np.eye(4, dtype=np.float64)}
        links = {"primary": "panda_hand"}
        joint_names = {"primary": PANDA_JOINT_NAMES}
        if "secondary" in self.arms:
            transforms["secondary"] = self._primary_from_secondary()
            links["secondary"] = "panda_hand_2"
            joint_names["secondary"] = tuple(f"{name}_2" for name in PANDA_JOINT_NAMES)
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="panda" if len(self.arms) == 1 else "dual_panda",
            joint_names=joint_names,
            base_transforms=transforms,
            end_effector_links=links,
        )

    @property
    def native_env(self) -> Any:
        """Runtime-only native environment; never registered with generated programs."""
        return self._require_env()

    def bind_protocol_evaluator(self, evaluator: Any) -> None:
        """Attach one host-only evaluator after reset; never exposed to programs."""
        if self._env is None:
            raise RuntimeError("reset() must be called before binding an evaluator")
        if self._protocol_evaluator is not None:
            raise RuntimeError("a protocol evaluator is already bound")
        if not callable(getattr(evaluator, "after_step", None)):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("protocol evaluator must define after_step()")
        self._protocol_evaluator = evaluator

    @property
    def metadata(self) -> RobosuiteTaskMetadata:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before task metadata is available")
        return self._metadata

    @property
    def arms(self) -> tuple[str, ...]:
        return self.metadata.arms

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._action_bounds is None:
            raise RuntimeError("reset() must be called before action bounds are available")
        return self._action_bounds[0].copy(), self._action_bounds[1].copy()

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("reset() must be called before using the Robosuite adapter")
        return self._env

    def _make_environment(self, metadata: RobosuiteTaskMetadata) -> Any:
        if self._env_factory is not None:
            return self._env_factory(metadata)
        suite = importlib.import_module("robosuite")
        controllers = importlib.import_module("robosuite.controllers")
        controller = controllers.load_controller_config(default_controller=self.controller)
        kwargs: dict[str, Any] = {
            "robots": list(metadata.robots),
            "controller_configs": controller,
            "initialization_noise": None,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "use_object_obs": True,
            "camera_names": list(metadata.camera_names),
            "camera_heights": self.camera_height,
            "camera_widths": self.camera_width,
            "camera_depths": True,
            "control_freq": self.control_frequency,
            "horizon": self.horizon,
            "ignore_done": False,
            "reward_shaping": True,
        }
        if metadata.env_configuration is not None:
            kwargs["env_configuration"] = metadata.env_configuration
        if metadata.task_name == "cube_restack":
            from cap_harness.robosuite.environments import CubeRestack

            return CubeRestack(**kwargs)
        return suite.make(metadata.environment, **kwargs)

    def reset(
        self,
        task_ref: str | tuple[str, int] | RobosuiteTaskMetadata,
        seed: int,
    ) -> Observation:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        metadata = self.registry.resolve(task_ref)
        self._close_environment()
        # Seed the simulator's legacy global RNG, not a separate generator.
        np.random.seed(seed)
        self._seed = seed
        self._env = self._make_environment(metadata)
        self._closed = False
        self._metadata = metadata
        self._raw_observation = self._env.reset()
        self._after_native_reset(self._env)
        self._elapsed_steps = 0
        self._last_reward = None
        self._last_terminated = False
        self._last_truncated = False
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=self._episode_language(metadata),
            family=metadata.family,
            metadata={"task_ref": metadata.task_ref},
        )
        self._action_bounds = tuple(
            np.asarray(value, dtype=np.float64).copy() for value in self._env.action_spec
        )
        self._codec = self._build_codec(metadata)
        self._gripper_targets = {
            arm: self._measured_gripper(index) for index, arm in enumerate(metadata.arms)
        }
        state = self.get_robot_state()
        self._joint_targets = {
            arm: np.array(joints, copy=True) for arm, joints in state.joint_positions.items()
        }
        observation = self.get_observation()
        if self._run_observer is not None:
            self._run_observer.on_reset(observation, metadata.to_manifest_record())
        return observation

    def _after_native_reset(self, env: Any) -> None:
        """Embodiment hook run right after the native reset, before codec setup."""
        del env  # The default hook is a no-op; subclasses use the environment.

    def _build_codec(self, metadata: RobosuiteTaskMetadata) -> Any:
        """Construct the native action codec once bounds and robots exist."""
        return RobosuiteActionCodec(
            metadata.arms,
            control_frequency=self.control_frequency,
            action_spec=self._action_bounds,
            arm_action_dimensions=tuple(robot.action_dim for robot in self._require_env().robots),
        )

    def _episode_language(self, metadata: RobosuiteTaskMetadata) -> str:
        """Return the task language for the freshly reset episode."""
        return metadata.language

    @staticmethod
    def _gripper_qpos_indexes(robot: Any) -> Any:
        """Return simulator qpos indexes for the primary gripper joints."""
        # Simulator bridge requires native state unavailable through its public API.
        return robot._ref_gripper_joint_pos_indexes

    @staticmethod
    def _grip_site(robot: Any) -> str:
        """Return the gripper grip-site name used as the public EEF frame."""
        return robot.gripper.important_sites["grip_site"]

    @staticmethod
    def _arm_has_gripper(robot: Any) -> bool:
        """Whether the robot exposes a controllable gripper actuator."""
        return robot.action_dim == 8

    @staticmethod
    def _body_transform(sim: Any, name: str) -> np.ndarray:
        return _transform(sim.data.get_body_xmat(name), sim.data.get_body_xpos(name))

    @staticmethod
    def _site_transform(sim: Any, name: str) -> np.ndarray:
        return _transform(sim.data.get_site_xmat(name), sim.data.get_site_xpos(name))

    @staticmethod
    def _camera_transform(sim: Any, name: str) -> np.ndarray:
        return _transform(sim.data.get_camera_xmat(name), sim.data.get_camera_xpos(name))

    def _primary_from_world(self) -> np.ndarray:
        return invert_transform(self._body_transform(self._require_env().sim, self.BASE_FRAME))

    def _primary_from_secondary(self) -> np.ndarray:
        env = self._require_env()
        return self._primary_from_world() @ self._body_transform(env.sim, self.SECONDARY_BASE_FRAME)

    def _camera_observation(self, name: str) -> CameraObservation:
        raw = self._raw_observation
        if raw is None:
            raise RuntimeError("reset() must be called before reading cameras")
        rgb = np.ascontiguousarray(np.asarray(raw[f"{name}_image"])[::-1])
        depth = np.asarray(raw[f"{name}_depth"], dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = np.ascontiguousarray(depth[::-1])
        env = self._require_env()
        depth_m = _metric_depth(env.sim, depth)
        camera_id = env.sim.model.camera_name2id(name)
        fovy = float(env.sim.model.cam_fovy[camera_id])
        focal = 0.5 * rgb.shape[0] / np.tan(np.deg2rad(fovy) / 2.0)
        intrinsics = np.array(
            [
                [focal, 0.0, 0.5 * rgb.shape[1]],
                [0.0, focal, 0.5 * rgb.shape[0]],
                [0.0, 0.0, 1.0],
            ]
        )
        image_from_opengl = np.diag((1.0, -1.0, -1.0, 1.0))
        primary_from_camera = (
            self._primary_from_world() @ self._camera_transform(env.sim, name) @ image_from_opengl
        )
        return CameraObservation(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            frame=f"camera/{name}",
            camera_pose=matrix_to_pose(primary_from_camera, frame=self.BASE_FRAME),
            timestamp_s=self.current_time_s,
        )

    def _measured_gripper(self, index: int) -> float:
        env = self._require_env()
        robot = env.robots[index]
        qpos = np.asarray(env.sim.data.qpos[self._gripper_qpos_indexes(robot)])
        if qpos.size == 0:
            return 1.0
        return float(np.clip(qpos[0] / 0.04, 0.0, 1.0))

    def get_robot_state(self) -> RobotState:
        env = self._require_env()
        joints: dict[str, np.ndarray] = {}
        velocities: dict[str, np.ndarray] = {}
        poses: dict[str, Pose] = {}
        grippers: dict[str, float] = {}
        for index, arm in enumerate(self.arms):
            robot = env.robots[index]
            # Simulator bridge requires native state unavailable through its public API.
            joints[arm] = np.asarray(robot._joint_positions, dtype=np.float64)
            # Simulator bridge requires native state unavailable through its public API.
            velocities[arm] = np.asarray(robot._joint_velocities, dtype=np.float64)
            world_from_eef = self._site_transform(env.sim, self._grip_site(robot))
            poses[arm] = matrix_to_pose(
                self._primary_from_world() @ world_from_eef,
                frame=self.BASE_FRAME,
            )
            grippers[arm] = self._measured_gripper(index)
        return RobotState(
            joint_positions=joints,
            joint_velocities=velocities,
            end_effector_poses=poses,
            gripper_positions=grippers,
            base_frame=self.BASE_FRAME,
            joint_names=dict.fromkeys(self.arms, PANDA_JOINT_NAMES),
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading task context")
        return self._task_context

    def get_observation(self) -> Observation:
        return Observation(
            cameras={name: self._camera_observation(name) for name in self.metadata.camera_names},
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=self.current_time_s,
        )

    def native_step(self, action: RobotAction) -> StepResult:
        env = self._require_env()
        if action.embodiment != self.embodiment:
            raise ValueError(
                f"action embodiment {action.embodiment!r} does not match {self.embodiment!r}"
            )
        if self._codec is None:
            raise RuntimeError("Robosuite codec is unavailable before reset")
        unknown_arms = set(action.arms) - set(self.arms)
        if unknown_arms:
            return StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.NOT_FOUND,
                    f"unknown Robosuite arms: {sorted(unknown_arms)}",
                ),
            )
        for arm, command in action.arms.items():
            arm_index = self.arms.index(arm)
            expected_dimension = self.get_robot_state().joint_positions[arm].shape
            if command.target.shape != expected_dimension:
                return StepResult(
                    ok=False,
                    error=ApiError(
                        ErrorCode.INVALID_REQUEST,
                        f"command dimension does not match arm {arm!r}",
                    ),
                )
            if command.gripper_position is not None and not self._arm_has_gripper(
                env.robots[arm_index]
            ):
                return StepResult(
                    ok=False,
                    error=ApiError(
                        ErrorCode.UNSUPPORTED,
                        f"arm {arm!r} has no controllable gripper",
                    ),
                )
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        for arm, command in action.arms.items():
            self._joint_targets[arm] = np.array(command.target, copy=True)
            if command.gripper_position is not None:
                self._gripper_targets[arm] = command.gripper_position
        try:
            native = self._codec.encode(
                action,
                self.get_robot_state(),
                joint_targets=self._joint_targets,
                gripper_targets=self._gripper_targets,
            )
            raw, reward, done, info = env.step(native)
            self._elapsed_steps += 1
            self._raw_observation = raw
            self._last_reward = float(reward)
            self._last_terminated = bool(done and self.check_success())
            self._last_truncated = bool(done and not self._last_terminated)
            result = StepResult(
                ok=True,
                observation=self.get_observation(),
                terminated=self._last_terminated,
                truncated=self._last_truncated,
                reward=self._last_reward,
                diagnostics={"native_info": info},
            )
        # Convert native/provider failures into the public typed error result.
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.ADAPTER_FAILED,
                    f"Robosuite native step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        if self._protocol_evaluator is not None:
            self._protocol_evaluator.after_step(action, result)
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def step(self, action: RobotAction) -> StepResult:
        runtime = self.native_step(action)
        return StepResult(
            ok=runtime.ok,
            observation=runtime.observation,
            terminated=runtime.terminated,
            truncated=runtime.truncated,
            error=runtime.public_view().error,
            diagnostics={
                "control_period_s": self.control_period_s,
                "step_index": self._elapsed_steps,
            },
        )

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        if not isinstance(trajectory, Trajectory | SynchronizedTrajectory):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("trajectory must be a Trajectory or SynchronizedTrajectory")
        if trajectory.embodiment != self.embodiment:
            raise ValueError(
                f"trajectory embodiment {trajectory.embodiment!r} "
                f"does not match {self.embodiment!r}"
            )
        if not np.isclose(trajectory.dt_s, self.control_period_s, atol=1e-9, rtol=0.0):
            raise ValueError("trajectory dt_s must equal the Robosuite control period")
        state = self.get_robot_state()
        if isinstance(trajectory, Trajectory):
            positions = {trajectory.arm: trajectory.joint_positions}
            names = {trajectory.arm: trajectory.joint_names}
            grippers = (
                None
                if trajectory.gripper_positions is None
                else {trajectory.arm: trajectory.gripper_positions}
            )
            count = len(trajectory.joint_positions)
        elif isinstance(trajectory, SynchronizedTrajectory):
            positions = trajectory.joint_positions
            names = trajectory.joint_names
            grippers = trajectory.gripper_positions
            count = trajectory.waypoint_count
        if not set(positions).issubset(self.arms):
            raise ValueError("trajectory contains an unavailable arm")
        if isinstance(trajectory, SynchronizedTrajectory) and set(positions) != set(self.arms):
            raise ValueError("synchronized trajectory must name every Robosuite arm")
        stale = self._stale_trajectory_result(trajectory, state)
        if stale is not None:
            return stale
        for arm, trajectory_names in names.items():
            if tuple(trajectory_names) != tuple(state.joint_names[arm]):
                raise ValueError(f"trajectory joint_names do not match arm {arm!r}")
        final: Observation | None = None
        steps = 0

        def commands(waypoint: int) -> dict[str, ArmCommand]:
            return {
                arm: ArmCommand(
                    mode="joint_position",
                    target=values[waypoint],
                    gripper_position=(None if grippers is None else float(grippers[arm][waypoint])),
                    embodiment=self.embodiment,
                )
                for arm, values in positions.items()
            }

        for waypoint in range(count):
            result = self.step(RobotAction(commands(waypoint)))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    error=result.error,
                )
            steps += 1
            final = result.observation
            if result.terminated or result.truncated:
                errors = {
                    arm: float(np.max(np.abs(final.robot_state.joint_positions[arm] - values[-1])))
                    for arm, values in positions.items()
                }
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=self._terminal_error(result.terminated, result.truncated),
                    final_errors=errors,
                    diagnostics={"joint_error_rad": errors},
                )

        errors = {
            arm: float(np.max(np.abs(final.robot_state.joint_positions[arm] - values[-1])))
            for arm, values in positions.items()
        }
        for _ in range(TRAJECTORY_MAX_STEPS_PER_WAYPOINT - 1):
            if max(errors.values()) <= TRAJECTORY_TOLERANCE_RAD:
                break
            result = self.step(RobotAction(commands(count - 1)))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    error=result.error,
                )
            steps += 1
            final = result.observation
            errors = {
                arm: float(np.max(np.abs(final.robot_state.joint_positions[arm] - values[-1])))
                for arm, values in positions.items()
            }
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=final,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=self._terminal_error(result.terminated, result.truncated),
                    final_errors=errors,
                    diagnostics={"joint_error_rad": errors},
                )
        if max(errors.values()) > TRAJECTORY_TOLERANCE_RAD:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                error=ApiError(
                    ErrorCode.TIMEOUT,
                    "final trajectory waypoint did not converge",
                    details={"joint_error_rad": errors},
                ),
                final_errors=errors,
            )
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            final_errors=errors if final is not None else {},
            diagnostics={"joint_error_rad": errors if final is not None else {}},
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        """Issue validated gripper targets for every arm in one simulator tick."""
        if not isinstance(positions, Mapping) or set(positions) != set(self.arms):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "positions must name every Robosuite arm",
                ),
            )
        env = self._require_env()
        normalized: dict[str, float] = {}
        for arm in self.arms:
            if not self._arm_has_gripper(env.robots[self.arms.index(arm)]):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.UNSUPPORTED,
                        f"arm {arm!r} has no controllable gripper",
                    ),
                )
            try:
                numeric = float(positions[arm])
            except (TypeError, ValueError):
                numeric = float("nan")
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.INVALID_REQUEST,
                        f"gripper position for {arm!r} must be finite and in [0, 1]",
                    ),
                )
            normalized[arm] = numeric
        state = self.get_robot_state()
        result = self.step(
            RobotAction(
                arms={
                    arm: ArmCommand(
                        "joint_position",
                        self._joint_targets.get(arm, state.joint_positions[arm]),
                        normalized[arm],
                    )
                    for arm in self.arms
                }
            )
        )
        terminal = result.terminated or result.truncated
        return ExecutionResult(
            ok=result.ok and not terminal,
            steps_executed=1 if result.ok else 0,
            final_observation=result.observation,
            terminated=result.terminated,
            truncated=result.truncated,
            error=result.error
            or (self._terminal_error(result.terminated, result.truncated) if terminal else None),
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        if arm not in self.arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}"),
            )
        arm_index = self.arms.index(arm)
        if not self._arm_has_gripper(self._require_env().robots[arm_index]):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.UNSUPPORTED,
                    f"arm {arm!r} has a fixed task tool without a controllable gripper",
                ),
            )
        numeric = float(position)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.INVALID_REQUEST, "gripper position must be in [0, 1]"),
            )
        self._gripper_targets[arm] = numeric
        result: StepResult | None = None
        final: Observation | None = None
        for _step_index in range(30):
            result = self.step(
                RobotAction(
                    {
                        arm: ArmCommand(
                            "joint_position",
                            self._joint_targets.get(
                                arm, self.get_robot_state().joint_positions[arm]
                            ),
                            numeric,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok or result.observation is None:
                break
            final = result.observation
            if result.terminated or result.truncated:
                break
        assert result is not None
        terminal = result.terminated or result.truncated
        return ExecutionResult(
            ok=result.ok and not terminal,
            steps_executed=_step_index + 1 if result.ok else _step_index,
            final_observation=final,
            terminated=result.terminated,
            truncated=result.truncated,
            error=result.error
            or (self._terminal_error(result.terminated, result.truncated) if terminal else None),
        )

    @staticmethod
    def _terminal_error(terminated: bool, truncated: bool) -> ApiError:
        del terminated  # Callers establish terminal state; truncation takes precedence.
        if truncated:
            return ApiError(
                ErrorCode.TRUNCATED,
                "episode was truncated before trajectory execution completed",
            )
        return ApiError(
            ErrorCode.TERMINATED,
            "episode terminated before trajectory execution completed",
        )

    @staticmethod
    def _stale_trajectory_result(
        trajectory: Trajectory | SynchronizedTrajectory,
        state: RobotState,
    ) -> ExecutionResult | None:
        expected = trajectory.expected_start
        if set(expected.arms) != set(state.arms):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.STALE_PLAN, "trajectory arm set is stale"),
            )
        joint_errors: dict[str, float] = {}
        gripper_errors: dict[str, float] = {}
        for arm in state.arms:
            if tuple(expected.joint_names[arm]) != tuple(state.joint_names[arm]):
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.STALE_PLAN,
                        f"trajectory joint names are stale for arm {arm!r}",
                    ),
                )
            joint_errors[arm] = float(
                np.max(np.abs(expected.joint_positions[arm] - state.joint_positions[arm]))
            )
            gripper_errors[arm] = abs(
                expected.gripper_positions[arm] - state.gripper_positions[arm]
            )
        if (
            max(joint_errors.values()) <= TRAJECTORY_START_TOLERANCE_RAD
            and max(gripper_errors.values()) <= TRAJECTORY_START_TOLERANCE_RAD
        ):
            return None
        return ExecutionResult(
            ok=False,
            steps_executed=0,
            error=ApiError(
                ErrorCode.STALE_PLAN,
                "trajectory start no longer matches measured robot state",
                details={
                    "joint_error_rad": joint_errors,
                    "gripper_error": gripper_errors,
                    "tolerance": TRAJECTORY_START_TOLERANCE_RAD,
                },
            ),
            final_errors=joint_errors,
        )

    def ik_request(self, target_pose: Pose, state: RobotState, arm: str) -> tuple[Pose, RobotState]:
        """Express secondary IK explicitly in robot1 base; primary is already robot0-based."""
        if arm == "primary":
            hand_target = target_pose.as_matrix().copy()
            hand_target[:3, :3] = hand_target[:3, :3] @ _EEF_TO_HAND_ROTATION
            return matrix_to_pose(hand_target, frame=self.BASE_FRAME), state
        if arm != "secondary" or arm not in state.joint_positions:
            raise ValueError(f"unknown arm {arm!r}")
        secondary_from_primary = invert_transform(self._primary_from_secondary())
        secondary_pose = transform_pose(
            target_pose,
            secondary_from_primary,
            source_frame=self.BASE_FRAME,
            target_frame=self.SECONDARY_BASE_FRAME,
        )
        hand_target = secondary_pose.as_matrix().copy()
        hand_target[:3, :3] = hand_target[:3, :3] @ _EEF_TO_HAND_ROTATION
        secondary_pose = matrix_to_pose(hand_target, frame=self.SECONDARY_BASE_FRAME)
        secondary_state = RobotState(
            joint_positions={arm: state.joint_positions[arm]},
            joint_velocities={arm: state.joint_velocities[arm]},
            end_effector_poses={
                arm: transform_pose(
                    state.end_effector_poses[arm],
                    secondary_from_primary,
                    source_frame=self.BASE_FRAME,
                    target_frame=self.SECONDARY_BASE_FRAME,
                )
            },
            gripper_positions={arm: state.gripper_positions[arm]},
            base_frame=self.SECONDARY_BASE_FRAME,
            joint_names={arm: state.joint_names[arm]},
            timestamp_s=state.timestamp_s,
            embodiment=self.embodiment,
        )
        return secondary_pose, secondary_state

    def check_success(self) -> bool:
        checker = getattr(self._require_env(), "_check_success", None)
        if not callable(checker):
            # Unavailable simulator state, not an argument type error.
            raise RuntimeError("Robosuite environment has no success predicate")
        return bool(checker())

    def compute_reward(self) -> float | None:
        return self._last_reward

    def get_task_metadata(self) -> Mapping[str, object]:
        return MappingProxyType(self.metadata.to_manifest_record())

    def get_controller_metadata(self) -> Mapping[str, object]:
        low, high = self.action_bounds
        return MappingProxyType(
            {
                "action_dimension": int(low.size),
                "action_lower_bounds": tuple(float(value) for value in low),
                "action_upper_bounds": tuple(float(value) for value in high),
                "arm_names": self.arms,
                "base_frame": self.BASE_FRAME,
                "camera_names": self.metadata.camera_names,
                "control_frequency_hz": self.control_frequency,
                "control_period_s": self.control_period_s,
                "controller": self.controller,
                "gripper_semantics": {
                    "normalized": "0=closed, 1=open",
                    "native": "1=closed, -1=open",
                },
                "controllable_gripper_arms": tuple(
                    arm
                    for arm, robot in zip(self.arms, self._require_env().robots, strict=False)
                    if self._arm_has_gripper(robot)
                ),
                "supported_action_modes": ("joint_position",),
            }
        )

    def _close_environment(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
        self._env = None

    def close(self) -> None:
        if self._closed:
            return
        self._close_environment()
        self._closed = True


BASE_FRAME = RobosuiteAdapter.BASE_FRAME
"""Compatibility alias; adapter implementations use the class variable."""

__all__ = ["BASE_FRAME", "RobosuiteAdapter"]
