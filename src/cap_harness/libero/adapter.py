"""Runtime-only direct OffScreenRenderEnv adapter for LIBERO-Pro."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
from typing import Any, TypeVar

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
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.libero.codec import LIBERO_ACTION_DIMENSION, LiberoActionCodec
from cap_harness.libero.registry import LiberoSuiteRegistry, LiberoTaskMetadata

CAMERA_NAMES: tuple[str, str] = ("agentview", "robot0_eye_in_hand")
BASE_FRAME = "robot_base"
PRIMARY_ARM = "primary"
PANDA_JOINT_NAMES: tuple[str, ...] = tuple(f"robot0_joint{i}" for i in range(1, 8))
TRAJECTORY_JOINT_TOLERANCE_RAD = 0.01
TRAJECTORY_MAX_STEPS_PER_WAYPOINT = 120

_LiberoAdapterT = TypeVar("_LiberoAdapterT", bound="LiberoAdapter")


def _as_finite_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _libero_quaternion_xyzw_to_matrix(quaternion: Any) -> np.ndarray:
    x, y, z, w = _as_finite_array(quaternion, "quaternion", (4,))
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= 0.0:
        raise ValueError("quaternion must have nonzero norm")
    x, y, z, w = np.array((x, y, z, w), dtype=np.float64) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_wxyz(rotation: Any) -> np.ndarray:
    matrix = _as_finite_array(rotation, "rotation", (3, 3))
    # Project tiny simulator drift back onto SO(3) before conversion.
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vh

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _transform(rotation: Any, translation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _as_finite_array(rotation, "rotation", (3, 3))
    transform[:3, 3] = _as_finite_array(translation, "translation", (3,))
    return transform


def _default_depth_converter(sim: Any, depth: np.ndarray) -> np.ndarray:
    """Match robosuite's normalized z-buffer to metric-depth conversion."""
    try:
        extent = float(sim.model.stat.extent)
        near = float(sim.model.vis.map.znear) * extent
        far = float(sim.model.vis.map.zfar) * extent
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "metric depth conversion needs sim.model.stat.extent and vis.map near/far"
        ) from exc
    if not np.isfinite(near) or not np.isfinite(far) or near <= 0.0 or far <= near:
        raise ValueError("simulator reports invalid depth clipping planes")
    metric = near / (1.0 - depth * (1.0 - near / far))
    if not bool(np.all(np.isfinite(metric))) or bool(np.any(metric < 0.0)):
        raise ValueError("depth conversion produced invalid metric values")
    return metric


def _load_offscreen_render_env() -> Callable[..., Any]:
    failures: list[BaseException] = []
    for module_name in ("libero.envs", "libero.libero.envs"):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            failures.append(exc)
            continue
        factory = getattr(module, "OffScreenRenderEnv", None)
        if callable(factory):
            return factory
    raise ModuleNotFoundError(
        "LIBERO is not installed; install the LIBERO-Pro runtime to create an adapter"
    ) from (failures[-1] if failures else None)


class LiberoAdapter:
    """Normalize a task-specific LIBERO ``OffScreenRenderEnv`` directly."""

    #: The gripper action on this embodiment is a velocity, not a position:
    #: robosuite's `simple_grip` controller assigns `goal_qvel`, so the sign picks
    #: a direction and the jaw travels to its stop. Only the two endpoints are
    #: states the actuator can hold.
    gripper_holds_intermediate = False

    def __init__(
        self,
        *,
        registry: LiberoSuiteRegistry | Any | None = None,
        env_factory: Callable[..., Any] | Any | None = None,
        control_frequency: float = 20.0,
        camera_height: int = 128,
        camera_width: int = 128,
        controller: str = "JOINT_POSITION",
        horizon: int = 1000,
        post_settle_steps: int = 10,
        gripper_max_opening_m: float = 0.04,
        gripper_tolerance: float = 0.05,
        gripper_max_steps: int = 60,
        depth_converter: Callable[[Any, np.ndarray], np.ndarray] | None = None,
        run_observer: Any | None = None,
        init_mode: str = "saved",
    ) -> None:
        if (
            isinstance(camera_height, bool)
            or not isinstance(camera_height, int)
            or camera_height <= 0
        ):
            raise ValueError("camera_height must be a positive integer")
        if isinstance(camera_width, bool) or not isinstance(camera_width, int) or camera_width <= 0:
            raise ValueError("camera_width must be a positive integer")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if (
            isinstance(post_settle_steps, bool)
            or not isinstance(post_settle_steps, int)
            or post_settle_steps < 0
        ):
            raise ValueError("post_settle_steps must be a non-negative integer")
        if not isinstance(controller, str) or not controller.strip():
            raise ValueError("controller must be a non-empty string")
        if init_mode not in ("saved", "seeded"):
            raise ValueError("init_mode must be 'saved' or 'seeded'")
        if controller != "JOINT_POSITION":
            raise ValueError("LiberoAdapter requires the JOINT_POSITION controller")
        try:
            max_opening = float(gripper_max_opening_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper_max_opening_m must be positive and finite") from exc
        if not np.isfinite(max_opening) or max_opening <= 0.0:
            raise ValueError("gripper_max_opening_m must be positive and finite")
        try:
            gripper_tolerance = float(gripper_tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper_tolerance must be positive and finite") from exc
        if not np.isfinite(gripper_tolerance) or gripper_tolerance <= 0.0:
            raise ValueError("gripper_tolerance must be positive and finite")
        if (
            isinstance(gripper_max_steps, bool)
            or not isinstance(gripper_max_steps, int)
            or gripper_max_steps <= 0
        ):
            raise ValueError("gripper_max_steps must be a positive integer")

        # Let the codec own frequency validation so adapter and encoding semantics agree.
        self._codec = LiberoActionCodec(control_frequency=control_frequency)
        self.registry = registry if registry is not None else LiberoSuiteRegistry()
        self._env_factory = env_factory
        self.control_frequency = self._codec.control_frequency
        self.control_period_s = 1.0 / self.control_frequency
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.controller = controller
        self.horizon = horizon
        self.post_settle_steps = post_settle_steps
        self.gripper_max_opening_m = max_opening
        self.gripper_tolerance = gripper_tolerance
        self.gripper_max_steps = gripper_max_steps
        self._depth_converter = depth_converter or _default_depth_converter
        self._run_observer = run_observer
        self.init_mode = init_mode

        self._env: Any | None = None
        self._environment_task_ref: str | None = None
        self._task_metadata: LiberoTaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._selected_init_state_index: int | None = None
        self._seed: int | None = None
        self._raw_observation: Mapping[str, Any] | None = None
        self._elapsed_steps = 0
        self._last_reward: float | None = None
        self._last_native_info: Mapping[str, Any] = {}
        self._last_terminated = False
        self._last_truncated = False
        self._action_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._extensions: Any | None = None
        self._closed = False

    @property
    def embodiment(self) -> str:
        return "libero"

    @property
    def native_env(self) -> Any:
        """Runtime-only access to the direct OffScreenRenderEnv instance."""
        return self._require_env()

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    def get_current_time_s(self) -> float:
        return self.current_time_s

    def get_planning_context(self) -> RobotPlanningContext:
        """Return non-privileged Panda model calibration for external planners."""
        self._require_env()
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="panda",
            joint_names={PRIMARY_ARM: PANDA_JOINT_NAMES},
            base_transforms={PRIMARY_ARM: np.eye(4, dtype=np.float64)},
            end_effector_links={PRIMARY_ARM: "panda_hand"},
        )

    @property
    def task_metadata(self) -> LiberoTaskMetadata:
        if self._task_metadata is None:
            raise RuntimeError("reset() must be called before task metadata is available")
        return self._task_metadata

    @property
    def selected_init_state_index(self) -> int | None:
        """Saved-state index, or None in seeded mode where no saved state is applied.

        Gated on ``_seed`` rather than on the index itself, because None is a
        legitimate value here: seeded mode deliberately applies no saved state.
        """
        if self._seed is None:
            raise RuntimeError("reset() must be called before init-state metadata is available")
        return self._selected_init_state_index

    @property
    def seed(self) -> int:
        if self._seed is None:
            raise RuntimeError("reset() must be called before seed metadata is available")
        return self._seed

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._action_bounds is None:
            raise RuntimeError("reset() must be called before action bounds are available")
        return self._action_bounds[0].copy(), self._action_bounds[1].copy()

    @property
    def extensions(self) -> Any:
        if self._extensions is None:
            from cap_harness.libero.extensions import LiberoRuntimeExtensions

            self._extensions = LiberoRuntimeExtensions(self)
        return self._extensions

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("reset() must be called before using the LIBERO adapter")
        return self._env

    @staticmethod
    def _extract_observation(reset_result: Any) -> Mapping[str, Any]:
        candidate = reset_result
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            candidate = reset_result[0]
        if not isinstance(candidate, Mapping):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("LIBERO reset/set_init_state did not return an observation mapping")
        return candidate

    def _read_action_spec(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        spec: Any = getattr(env, "action_spec", None)
        if spec is None and getattr(env, "env", None) is not None:
            spec = getattr(env.env, "action_spec", None)
        if callable(spec):
            spec = spec()
        if hasattr(spec, "low") and hasattr(spec, "high"):
            spec = (spec.low, spec.high)
        return LiberoActionCodec.validate_action_spec(spec)

    def _close_environment(self) -> None:
        if self._env is None:
            return
        env, self._env = self._env, None
        self._environment_task_ref = None
        close = getattr(env, "close", None)
        if callable(close):
            close()

    def _create_environment(self, metadata: LiberoTaskMetadata) -> Any:
        if self._environment_task_ref == metadata.task_ref and self._env is not None:
            return self._env
        self._close_environment()
        factory = (
            self._env_factory if self._env_factory is not None else _load_offscreen_render_env()
        )
        kwargs = {
            "bddl_file_name": str(metadata.bddl_path),
            "camera_depths": True,
            "camera_heights": self.camera_height,
            "camera_names": list(CAMERA_NAMES),
            "camera_widths": self.camera_width,
            "control_freq": self.control_frequency,
            "controller": self.controller,
            "horizon": self.horizon,
        }
        env = factory(**kwargs) if callable(factory) else factory
        if env is None:
            raise RuntimeError("env_factory returned None")
        self._env = env
        self._environment_task_ref = metadata.task_ref
        self._action_bounds = self._read_action_spec(env)
        return env

    def _unpack_native_result(
        self, result: Any
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise ValueError("LIBERO step() must return a four- or five-element tuple")
        if len(result) == 4:
            raw_observation, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raw_observation, reward, terminated, truncated, info = result
            terminated, truncated = bool(terminated), bool(truncated)
        raw_observation = self._extract_observation(raw_observation)
        try:
            numeric_reward = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError("LIBERO step reward must be numeric") from exc
        if not np.isfinite(numeric_reward):
            raise ValueError("LIBERO step reward must be finite")
        if not isinstance(info, Mapping):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("LIBERO step info must be a mapping")
        return raw_observation, numeric_reward, terminated, truncated, info

    def _settle(self, env: Any) -> None:
        for _ in range(self.post_settle_steps):
            if self._raw_observation is None:
                raise RuntimeError("cannot settle without a raw observation")
            normalized_gripper = self._normalized_gripper(self._raw_observation)
            native_action = np.zeros(LIBERO_ACTION_DIMENSION, dtype=np.float64)
            native_action[-1] = LiberoActionCodec.normalized_gripper_to_native(normalized_gripper)
            if self._action_bounds is None:  # pragma: no cover - established at env creation
                raise RuntimeError("LIBERO action bounds are unavailable")
            native_action = np.clip(native_action, *self._action_bounds)
            (
                self._raw_observation,
                self._last_reward,
                self._last_terminated,
                self._last_truncated,
                self._last_native_info,
            ) = self._unpack_native_result(env.step(native_action))

    def reset(self, task_ref: Any, seed: int) -> Observation:
        """Reset a task, choose its deterministic init state, settle, and zero public time."""
        if self._closed:
            raise RuntimeError("the LIBERO adapter is closed")
        if isinstance(seed, bool | np.bool_) or not isinstance(seed, int | np.integer):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("seed must be an integer")
        seed = int(seed)
        metadata = self.registry.resolve(task_ref)
        init_states = self.registry.get_init_states(metadata)
        if len(init_states) != metadata.init_state_count:
            raise ValueError(
                f"init-state count changed for {metadata.task_ref}: "
                f"manifest={metadata.init_state_count}, runtime={len(init_states)}"
            )
        if self.init_mode == "seeded":
            # Procedural placement. env.seed(seed) seeds np.random, and env.reset()
            # re-samples every object pose through the placement initializer
            # (bddl_base_domain._reset_internal, guarded by deterministic_reset).
            # Restoring a saved state afterwards would overwrite exactly that
            # randomization -- which is what previously pinned all 50 seeds to one
            # scene -- so seeded mode applies no saved state at all.
            state_index = None
        elif not init_states:
            raise ValueError(
                f"{metadata.task_ref} ships no saved init states, so init_mode "
                "'saved' cannot select a layout; use init_mode 'seeded'"
            )
        else:
            state_index = (seed - 1) % len(init_states)
        env = self._create_environment(metadata)

        seed_method = getattr(env, "seed", None)
        if callable(seed_method):
            seed_method(seed)
        self._raw_observation = self._extract_observation(env.reset())
        if state_index is not None:
            set_result = env.set_init_state(init_states[state_index])
            if set_result is not None:
                self._raw_observation = self._extract_observation(set_result)

        self._task_metadata = metadata
        self._selected_init_state_index = state_index
        self._seed = seed
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={
                "init_state_count": metadata.init_state_count,
                "init_state_index": state_index,
                "init_mode": self.init_mode,
                "task_ref": metadata.task_ref,
            },
        )
        self._last_reward = None
        self._last_native_info = {}
        self._last_terminated = False
        self._last_truncated = False
        self._settle(env)

        # Settling belongs to reset, not to the public episode timeline.
        self._elapsed_steps = 0
        observation = self.get_observation()
        if self._run_observer is not None:
            self._run_observer.on_reset(observation, metadata.to_manifest_record())
        return observation

    @staticmethod
    def _named_id(model: Any, kind: str, name: str) -> int:
        legacy_lookup = getattr(model, f"{kind}_name2id", None)
        if callable(legacy_lookup):
            return int(legacy_lookup(name))
        modern_lookup = getattr(model, kind, None)
        if callable(modern_lookup):
            return int(modern_lookup(name).id)
        raise ValueError(f"simulator model cannot resolve {kind} {name!r}")

    @classmethod
    def _body_transform(cls, sim: Any, body_name: str) -> np.ndarray:
        data = sim.data
        get_rotation = getattr(data, "get_body_xmat", None)
        get_position = getattr(data, "get_body_xpos", None)
        if callable(get_rotation) and callable(get_position):
            rotation = np.asarray(get_rotation(body_name), dtype=np.float64).reshape(3, 3)
            position = get_position(body_name)
        else:
            body_id = cls._named_id(sim.model, "body", body_name)
            rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            position = data.xpos[body_id]
        return _transform(rotation, position)

    @classmethod
    def _camera_transform(cls, sim: Any, camera_name: str) -> np.ndarray:
        data = sim.data
        get_rotation = getattr(data, "get_camera_xmat", None)
        get_position = getattr(data, "get_camera_xpos", None)
        if callable(get_rotation) and callable(get_position):
            rotation = np.asarray(get_rotation(camera_name), dtype=np.float64).reshape(3, 3)
            position = get_position(camera_name)
        else:
            camera_id = cls._named_id(sim.model, "camera", camera_name)
            rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
            position = data.cam_xpos[camera_id]
        return _transform(rotation, position)

    @classmethod
    def _camera_fovy(cls, sim: Any, camera_name: str) -> float:
        camera_id = cls._named_id(sim.model, "camera", camera_name)
        fovy = float(np.asarray(sim.model.cam_fovy)[camera_id])
        if not np.isfinite(fovy) or fovy <= 0.0 or fovy >= 180.0:
            raise ValueError(f"camera {camera_name!r} reports invalid fovy {fovy}")
        return fovy

    def _base_from_world(self) -> np.ndarray:
        env = self._require_env()
        return np.linalg.inv(self._body_transform(env.sim, "robot0_base"))

    def _camera_observation(
        self, camera_name: str, raw: Mapping[str, Any], timestamp_s: float
    ) -> CameraObservation:
        rgb_key = f"{camera_name}_image"
        depth_key = f"{camera_name}_depth"
        if rgb_key not in raw or depth_key not in raw:
            raise ValueError(f"LIBERO observation is missing RGB-D for {camera_name!r}")
        rgb = np.asarray(raw[rgb_key])
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"{rgb_key} must have shape (H, W, 3)")
        rgb = np.ascontiguousarray(rgb[::-1])

        raw_depth = np.asarray(raw[depth_key], dtype=np.float64)
        if raw_depth.ndim == 3 and raw_depth.shape[2] == 1:
            raw_depth = raw_depth[:, :, 0]
        if raw_depth.shape != rgb.shape[:2]:
            raise ValueError(f"{depth_key} must have shape {rgb.shape[:2]}, got {raw_depth.shape}")
        raw_depth = np.ascontiguousarray(raw_depth[::-1])
        env = self._require_env()
        depth_m = np.asarray(self._depth_converter(env.sim, raw_depth), dtype=np.float64)
        if depth_m.shape != raw_depth.shape:
            raise ValueError("depth_converter must preserve the depth image shape")

        height, width = rgb.shape[:2]
        fovy = self._camera_fovy(env.sim, camera_name)
        focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        intrinsics = np.array(
            [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # MuJoCo cameras use OpenGL axes. Flip camera Y/Z to publish the
        # conventional +X-right, +Y-down, +Z-forward image frame.
        opengl_to_image = np.diag((1.0, -1.0, -1.0, 1.0))
        base_from_camera = (
            self._base_from_world() @ self._camera_transform(env.sim, camera_name) @ opengl_to_image
        )
        camera_pose = Pose(
            position=base_from_camera[:3, 3],
            quaternion_wxyz=_matrix_to_quaternion_wxyz(base_from_camera[:3, :3]),
            frame=BASE_FRAME,
        )
        return CameraObservation(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            frame=f"camera/{camera_name}",
            camera_pose=camera_pose,
            timestamp_s=timestamp_s,
        )

    def _normalized_gripper(self, raw: Mapping[str, Any]) -> float:
        if "robot0_gripper_qpos" not in raw:
            raise ValueError("LIBERO observation is missing robot0_gripper_qpos")
        qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
        if qpos.size == 0 or not bool(np.all(np.isfinite(qpos))):
            raise ValueError("robot0_gripper_qpos must contain finite values")
        return float(np.clip(qpos[0] / self.gripper_max_opening_m, 0.0, 1.0))

    def _end_effector_pose(self, raw: Mapping[str, Any]) -> Pose:
        base_from_world = self._base_from_world()
        if "robot0_eef_pos" in raw and "robot0_eef_quat" in raw:
            world_from_eef = _transform(
                _libero_quaternion_xyzw_to_matrix(raw["robot0_eef_quat"]),
                raw["robot0_eef_pos"],
            )
        else:
            env = self._require_env()
            world_from_eef = self._body_transform(env.sim, "gripper0_eef")
        base_from_eef = base_from_world @ world_from_eef
        return Pose(
            position=base_from_eef[:3, 3],
            quaternion_wxyz=_matrix_to_quaternion_wxyz(base_from_eef[:3, :3]),
            frame=BASE_FRAME,
        )

    def get_robot_state(self) -> RobotState:
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before reading robot state")
        raw = self._raw_observation
        joints = _as_finite_array(raw.get("robot0_joint_pos"), "robot0_joint_pos", (7,))
        if "robot0_joint_vel" in raw:
            velocities = _as_finite_array(raw["robot0_joint_vel"], "robot0_joint_vel", (7,))
        else:
            velocities = np.zeros(7, dtype=np.float64)
        return RobotState(
            joint_positions={PRIMARY_ARM: joints},
            joint_velocities={PRIMARY_ARM: velocities},
            end_effector_poses={PRIMARY_ARM: self._end_effector_pose(raw)},
            gripper_positions={PRIMARY_ARM: self._normalized_gripper(raw)},
            base_frame=BASE_FRAME,
            joint_names={PRIMARY_ARM: PANDA_JOINT_NAMES},
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading task context")
        return self._task_context

    def get_observation(self) -> Observation:
        if self._raw_observation is None:
            raise RuntimeError("reset() must be called before reading observations")
        timestamp_s = self.current_time_s
        cameras = {
            camera_name: self._camera_observation(camera_name, self._raw_observation, timestamp_s)
            for camera_name in CAMERA_NAMES
        }
        return Observation(
            cameras=cameras,
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=timestamp_s,
        )

    def native_step(self, action: RobotAction) -> StepResult:
        """Runtime-only step retaining native reward and sanitized-at-boundary info."""
        env = self._require_env()
        if action.embodiment != self.embodiment:
            raise ValueError(
                f"action embodiment {action.embodiment!r} does not match {self.embodiment!r}"
            )
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        if self._action_bounds is None:  # pragma: no cover - established by reset
            raise RuntimeError("LIBERO action bounds are unavailable")
        native_action = self._codec.encode(
            action,
            self.get_robot_state(),
            action_spec=self._action_bounds,
        )
        try:
            native_result = env.step(native_action)
        # Convert native/provider failures into the public typed error result.
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.ADAPTER_FAILED,
                    message=f"LIBERO native step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
            if self._run_observer is not None:
                self._run_observer.after_step(action, result)
            return result

        # A returned native step advanced one control period even if output
        # normalization subsequently fails.
        self._elapsed_steps += 1
        try:
            (
                self._raw_observation,
                self._last_reward,
                self._last_terminated,
                self._last_truncated,
                self._last_native_info,
            ) = self._unpack_native_result(native_result)
            observation = self.get_observation()
        # Convert native/provider failures into the public typed error result.
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    code=ErrorCode.ADAPTER_FAILED,
                    message=f"LIBERO step normalization failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
            if self._run_observer is not None:
                self._run_observer.after_step(action, result)
            return result
        result = StepResult(
            ok=True,
            observation=observation,
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            reward=self._last_reward,
            diagnostics={"native_info": self._last_native_info},
        )
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def step(self, action: RobotAction) -> StepResult:
        """Advance exactly one native control period and return a public result."""
        runtime_result = self.native_step(action)
        public_error = runtime_result.public_view().error
        return StepResult(
            ok=runtime_result.ok,
            observation=runtime_result.observation,
            terminated=runtime_result.terminated,
            truncated=runtime_result.truncated,
            reward=None,
            error=public_error,
            diagnostics={
                "control_period_s": self.control_period_s,
                "step_index": self._elapsed_steps,
            },
        )

    def execute_trajectory(self, trajectory: Trajectory) -> ExecutionResult:
        if not isinstance(trajectory, Trajectory):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("trajectory must be a Trajectory")
        if trajectory.embodiment != self.embodiment:
            raise ValueError(
                f"trajectory embodiment {trajectory.embodiment!r} "
                f"does not match {self.embodiment!r}"
            )
        if trajectory.arm != PRIMARY_ARM:
            raise ValueError("LIBERO trajectories must target the 'primary' arm")
        if not np.isclose(trajectory.dt_s, self.control_period_s, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"trajectory dt_s must equal LIBERO control period {self.control_period_s}"
            )
        state = self.get_robot_state()
        expected = trajectory.expected_start
        if set(expected.arms) != {PRIMARY_ARM} or tuple(expected.joint_names[PRIMARY_ARM]) != tuple(
            state.joint_names[PRIMARY_ARM]
        ):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(ErrorCode.STALE_PLAN, "trajectory start metadata is stale"),
            )
        start_error = float(
            np.max(
                np.abs(expected.joint_positions[PRIMARY_ARM] - state.joint_positions[PRIMARY_ARM])
            )
        )
        gripper_error = abs(
            expected.gripper_positions[PRIMARY_ARM] - state.gripper_positions[PRIMARY_ARM]
        )
        if max(start_error, gripper_error) > 0.02:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.STALE_PLAN,
                    "trajectory start no longer matches measured robot state",
                    details={
                        "joint_error_rad": {PRIMARY_ARM: start_error},
                        "gripper_error": {PRIMARY_ARM: gripper_error},
                        "tolerance": 0.02,
                    },
                ),
                final_errors={PRIMARY_ARM: start_error},
            )
        final_observation: Observation | None = None
        steps_executed = 0
        terminated = False
        truncated = False

        def apply_waypoint(index: int) -> StepResult:
            joints = trajectory.joint_positions[index]
            gripper = (
                None
                if trajectory.gripper_positions is None
                else float(trajectory.gripper_positions[index])
            )
            return self.step(
                RobotAction(
                    arms={
                        PRIMARY_ARM: ArmCommand(
                            mode="joint_position",
                            target=joints,
                            gripper_position=gripper,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )

        for index, _joints in enumerate(trajectory.joint_positions):
            result = apply_waypoint(index)
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps_executed,
                    final_observation=final_observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=result.error
                    or ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message="trajectory step returned no observation",
                    ),
                    diagnostics={"failed_waypoint": index},
                )
            steps_executed += 1
            final_observation = result.observation
            terminated, truncated = result.terminated, result.truncated
            if terminated or truncated:
                break

        final_joints = trajectory.joint_positions[-1]
        error_rad = float(
            np.max(
                np.abs(final_observation.robot_state.joint_positions[PRIMARY_ARM] - final_joints)
            )
        )
        for _ in range(TRAJECTORY_MAX_STEPS_PER_WAYPOINT - 1):
            if error_rad <= TRAJECTORY_JOINT_TOLERANCE_RAD or terminated or truncated:
                break
            result = apply_waypoint(len(trajectory.joint_positions) - 1)
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps_executed,
                    final_observation=final_observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=result.error
                    or ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message="final trajectory settle returned no observation",
                    ),
                    diagnostics={"failed_waypoint": len(trajectory.joint_positions) - 1},
                )
            steps_executed += 1
            final_observation = result.observation
            terminated, truncated = result.terminated, result.truncated
            error_rad = float(
                np.max(
                    np.abs(
                        final_observation.robot_state.joint_positions[PRIMARY_ARM] - final_joints
                    )
                )
            )
        if error_rad > TRAJECTORY_JOINT_TOLERANCE_RAD and not (terminated or truncated):
            return ExecutionResult(
                ok=False,
                steps_executed=steps_executed,
                final_observation=final_observation,
                error=ApiError(
                    code=ErrorCode.TIMEOUT,
                    message="final trajectory waypoint did not converge",
                    details={
                        "joint_error_rad": error_rad,
                        "tolerance_rad": TRAJECTORY_JOINT_TOLERANCE_RAD,
                        "max_steps": TRAJECTORY_MAX_STEPS_PER_WAYPOINT,
                    },
                ),
                diagnostics={"failed_waypoint": len(trajectory.joint_positions) - 1},
            )
        if terminated or truncated:
            return ExecutionResult(
                ok=False,
                steps_executed=steps_executed,
                final_observation=final_observation,
                terminated=terminated,
                truncated=truncated,
                error=ApiError(
                    ErrorCode.TRUNCATED if truncated else ErrorCode.TERMINATED,
                    "episode ended before trajectory execution completed",
                ),
                final_errors={PRIMARY_ARM: error_rad},
                diagnostics={
                    "control_period_s": self.control_period_s,
                    "joint_error_rad": error_rad,
                },
            )
        return ExecutionResult(
            ok=True,
            steps_executed=steps_executed,
            final_observation=final_observation,
            terminated=terminated,
            truncated=truncated,
            final_errors={PRIMARY_ARM: error_rad},
            diagnostics={
                "control_period_s": self.control_period_s,
                "joint_error_rad": error_rad,
            },
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        """One-arm compatibility wrapper with one-control-tick semantics."""
        if not isinstance(positions, Mapping) or set(positions) != {PRIMARY_ARM}:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "positions must contain only the 'primary' arm",
                ),
            )
        try:
            numeric = float(positions[PRIMARY_ARM])
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "gripper position must be finite and in [0, 1]",
                ),
            )
        state = self.get_robot_state()
        result = self.step(
            RobotAction(
                arms={
                    PRIMARY_ARM: ArmCommand(
                        mode="joint_position",
                        target=state.joint_positions[PRIMARY_ARM],
                        gripper_position=numeric,
                    )
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
            or (
                ApiError(
                    ErrorCode.TRUNCATED if result.truncated else ErrorCode.TERMINATED,
                    "episode ended during the coordinated gripper tick",
                )
                if terminal
                else None
            ),
        )

    def set_gripper(self, position: float, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        return self._drive_gripper(position, arm=arm, accept_contact_stall=False)

    def close_gripper(self, *, arm: str = PRIMARY_ARM) -> ExecutionResult:
        """Close fully, accepting a stable nonzero position as object contact."""
        return self._drive_gripper(0.0, arm=arm, accept_contact_stall=True)

    def _drive_gripper(
        self,
        position: float,
        *,
        arm: str,
        accept_contact_stall: bool,
    ) -> ExecutionResult:
        if arm != PRIMARY_ARM:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.NOT_FOUND,
                    message=f"LIBERO has no arm {arm!r}",
                ),
            )
        if isinstance(position, bool | np.bool_):
            position = float("nan")
        try:
            normalized = float(position)
        except (TypeError, ValueError):
            normalized = float("nan")
        if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="gripper position must be finite and in [0, 1]",
                ),
            )
        state = self.get_robot_state()
        actual = state.gripper_positions[PRIMARY_ARM]
        if abs(actual - normalized) <= self.gripper_tolerance:
            return ExecutionResult(
                ok=True,
                steps_executed=0,
                final_observation=self.get_observation(),
                diagnostics={
                    "target_position": normalized,
                    "final_position": actual,
                    "tolerance": self.gripper_tolerance,
                },
            )

        final_observation: Observation | None = None
        recent_positions: list[float] = []
        for step_index in range(self.gripper_max_steps):
            result = self.step(
                RobotAction(
                    arms={
                        PRIMARY_ARM: ArmCommand(
                            mode="joint_position",
                            target=state.joint_positions[PRIMARY_ARM],
                            gripper_position=normalized,
                            embodiment=self.embodiment,
                        )
                    }
                )
            )
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=step_index,
                    final_observation=final_observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=result.error
                    or ApiError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message="gripper step returned no observation",
                    ),
                )
            final_observation = result.observation
            state = final_observation.robot_state
            actual = state.gripper_positions[PRIMARY_ARM]
            recent_positions.append(actual)
            steps_executed = step_index + 1
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps_executed,
                    final_observation=final_observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=ApiError(
                        code=ErrorCode.TERMINATED,
                        message="episode ended before the gripper reached its target",
                        details={"target_position": normalized, "final_position": actual},
                    ),
                )
            if abs(actual - normalized) <= self.gripper_tolerance:
                return ExecutionResult(
                    ok=True,
                    steps_executed=steps_executed,
                    final_observation=final_observation,
                    diagnostics={
                        "target_position": normalized,
                        "final_position": actual,
                        "tolerance": self.gripper_tolerance,
                        "completion": "target_reached",
                    },
                )
            stall_window = 5
            if (
                accept_contact_stall
                and normalized == 0.0
                and len(recent_positions) >= stall_window
                and max(recent_positions[-stall_window:]) - min(recent_positions[-stall_window:])
                <= 0.005
            ):
                return ExecutionResult(
                    ok=True,
                    steps_executed=steps_executed,
                    final_observation=final_observation,
                    diagnostics={
                        "target_position": normalized,
                        "final_position": actual,
                        "completion": "contact_stall",
                        "stall_window_steps": stall_window,
                    },
                )

        return ExecutionResult(
            ok=False,
            steps_executed=self.gripper_max_steps,
            final_observation=final_observation,
            error=ApiError(
                code=ErrorCode.TIMEOUT,
                message="gripper did not reach its target within the step limit",
                details={
                    "target_position": normalized,
                    "final_position": actual,
                    "max_steps": self.gripper_max_steps,
                },
            ),
        )

    def compute_reward(self) -> float | None:
        """Return the latest native reward for evaluator/runtime use only."""
        return self._last_reward

    def check_success(self) -> bool:
        """Evaluate LIBERO's privileged success predicate for runtime use only."""
        env = self._require_env()
        checker = getattr(env, "check_success", None)
        if checker is None and getattr(env, "env", None) is not None:
            checker = getattr(env.env, "_check_success", None)
        if not callable(checker):
            # Unavailable simulator state, not an argument type error.
            raise RuntimeError("LIBERO environment has no success predicate")
        return bool(checker())

    def get_task_metadata(self) -> Mapping[str, object]:
        return self.extensions.get_task_metadata()

    def get_controller_metadata(self) -> Mapping[str, object]:
        return self.extensions.get_controller_metadata()

    def close(self) -> None:
        if self._closed:
            return
        self._close_environment()
        self._closed = True

    # Preserve subclass typing on Python 3.10 without a new typing_extensions dependency.
    def __enter__(self: _LiberoAdapterT) -> _LiberoAdapterT:  # noqa: PYI019
        if self._closed:
            raise RuntimeError("the LIBERO adapter is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()
