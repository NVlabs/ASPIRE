"""
Base class for YAM bimanual robot station. Contains code shared between simulation and real environments.

Mostly adapted from `yam_env.py` in the starter code.
"""

from typing import Literal

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robot.yam.kinematics import YamKinematics


class _BaseYamEnv(gym.Env):
    CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640

    def __init__(
        self,
        control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
        ] = "joint_position",
        enable_cameras: bool = True,
        camera_names: tuple[str, ...] = ("top", "left", "right"),
    ):
        self.control_mode = control_mode
        self.enable_cameras = bool(enable_cameras)
        self.camera_names = camera_names
        # Track current joint positions for delta control modes
        self._current_joint_pos = {
            "left_joint_pos": np.zeros(6, dtype=np.float32),
            "right_joint_pos": np.zeros(6, dtype=np.float32),
        }
        # Grasp-site offset along TCP local +Z (metres).
        # When non-zero, delta_ee_pose commands are applied at the *grasp site*
        # (fingertips) instead of the raw TCP (gripper base).  This gives more
        # intuitive control when teleoperating with grippers.
        # Set via ``set_grasp_site_offset(m)`` — default 0.0 for backward compat.
        self._grasp_site_offset_m: float = 0.0
        from robot.models.station.paths import get_station_xml

        model_path = get_station_xml()
        self._spec = mujoco.MjSpec.from_file(str(model_path))
        self._spec.copy_during_attach = True
        self._spec = self._build_task_spec(self._spec)

        # Change camera resolution to 640x480
        for side in self.camera_names:
            camera = next(x for x in self._spec.cameras if side in x.name)
            camera.resolution = (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
            camera.sensor_size = (0.003148, 0.002364)  # FOV from rs.rs2_fov()

        # Compile model
        self._model = self._spec.compile()

        # Actuator mappings
        left_actuator_names = [
            x.name for x in self._spec.actuators if x.name.startswith("left_")
        ]
        right_actuator_names = [
            x.name for x in self._spec.actuators if x.name.startswith("right_")
        ]
        self.left_actuator_ids = [
            self._model.actuator(name).id for name in left_actuator_names
        ]
        self.right_actuator_ids = [
            self._model.actuator(name).id for name in right_actuator_names
        ]

        # Kinematics
        self._kinematics = YamKinematics()

        # Observation and action spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _blank_camera_image(self) -> np.ndarray:
        return np.zeros((self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 3), dtype=np.uint8)

    def _build_task_spec(self, station_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Override this method to add task-specific objects to the scene."""
        return station_spec

    def _build_observation_space(self) -> gym.spaces.Dict:
        space = {}

        # Arm and gripper states
        for side in ["left", "right"]:
            if self.control_mode in (
                "joint_position",
                "delta_joint_position",
                "delta_ee_pose",
            ):
                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                )
            elif self.control_mode == "cartesian_position":
                space.update(
                    {
                        f"{side}_ee_pos": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        f"{side}_ee_quat_xyzw": gym.spaces.Box(
                            low=-1,
                            high=1,
                            shape=(4,),
                            dtype=np.float32,
                        ),
                    }
                )

            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            )

        # Camera images
        for camera in self.camera_names:
            space[f"{camera}_camera_image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 3),
                dtype=np.uint8,
            )

        return gym.spaces.Dict(space)

    def set_grasp_site_offset(self, offset_m: float) -> None:
        """Set the grasp-site offset for delta_ee_pose control.

        When ``offset_m > 0`` the delta commands are applied at a virtual
        *grasp site* located ``offset_m`` metres along the TCP's local +Z
        axis (i.e. further out along the finger/tool direction).  This makes
        teleoperation more intuitive because the user controls the fingertips
        rather than the gripper base.

        The offset only affects ``delta_ee_pose`` mode; other modes are
        unchanged.
        """
        self._grasp_site_offset_m = float(offset_m)

    def set_control_mode(
        self,
        control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
        ],
        observation: dict[str, np.ndarray] | None = None,
    ) -> None:
        if control_mode == self.control_mode:
            return
        self.control_mode = control_mode
        # Rebuild spaces to match new mode
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()
        # Seed delta control state if joint positions are available
        if (
            observation
            and "left_joint_pos" in observation
            and "right_joint_pos" in observation
        ):
            self._current_joint_pos["left_joint_pos"] = observation[
                "left_joint_pos"
            ].copy()
            self._current_joint_pos["right_joint_pos"] = observation[
                "right_joint_pos"
            ].copy()

    def _build_action_space(self) -> gym.spaces.Dict:
        space = {}

        for side, actuator_ids in [
            ("left", self.left_actuator_ids),
            ("right", self.right_actuator_ids),
        ]:
            if self.control_mode == "joint_position":
                # Actuator control ranges
                ctrl_ranges = self._model.actuator_ctrlrange[actuator_ids]
                low = ctrl_ranges[:, 0].astype(np.float32)
                high = ctrl_ranges[:, 1].astype(np.float32)

                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=low[:6],
                    high=high[:6],
                    dtype=np.float32,
                )
            elif self.control_mode == "delta_joint_position":
                # Delta actions: relative changes to current joint positions
                # Use reasonable limits for delta (e.g., ±0.5 rad per step)
                space[f"{side}_joint_pos"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                )
            elif self.control_mode in ("cartesian_position", "delta_ee_pose"):
                space.update(
                    {
                        f"{side}_ee_pos": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                        f"{side}_ee_quat_xyzw": gym.spaces.Box(
                            low=-1,
                            high=1,
                            shape=(4,),
                            dtype=np.float32,
                        ),
                    }
                )

            space[f"{side}_gripper_pos"] = gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )

        return gym.spaces.Dict(space)

    def _convert_from_joint_to_cartesian(
        self, obs: dict[str, np.ndarray]
    ) -> np.ndarray:
        # Extract joint positions
        left_joint_pos = obs["left_joint_pos"]
        right_joint_pos = obs["right_joint_pos"]
        del obs["left_joint_pos"]
        del obs["right_joint_pos"]

        # Run forward kinematics
        left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
            self._kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
        )

        # Construct end-effector pos and quat
        obs["left_ee_pos"] = left_ee_pos
        obs["left_ee_quat_xyzw"] = left_ee_quat_xyzw
        obs["right_ee_pos"] = right_ee_pos
        obs["right_ee_quat_xyzw"] = right_ee_quat_xyzw

        return obs

    def _convert_from_cartesian_to_joint(
        self, action: dict[str, np.ndarray]
    ) -> np.ndarray:
        # Check if action is already in joint format (shouldn't happen in cartesian mode, but handle gracefully)
        if "left_joint_pos" in action and "right_joint_pos" in action:
            # Action is already in joint format, return as-is
            return action

        # Extract end-effector pos and quat
        if "left_ee_pos" not in action or "right_ee_pos" not in action:
            raise ValueError(
                f"Expected cartesian action with 'left_ee_pos' and 'right_ee_pos' keys, "
                f"but got keys: {list(action.keys())}"
            )

        left_ee_pos = action["left_ee_pos"]
        left_ee_quat_xyzw = action["left_ee_quat_xyzw"]
        right_ee_pos = action["right_ee_pos"]
        right_ee_quat_xyzw = action["right_ee_quat_xyzw"]
        del action["left_ee_pos"]
        del action["left_ee_quat_xyzw"]
        del action["right_ee_pos"]
        del action["right_ee_quat_xyzw"]

        # Run inverse kinematics
        left_joint_pos, right_joint_pos = self._kinematics.inverse_kinematics(
            left_ee_pos,
            left_ee_quat_xyzw,
            right_ee_pos,
            right_ee_quat_xyzw,
            seeded=True,
        )

        # Construct joint positions
        action["left_joint_pos"] = left_joint_pos
        action["right_joint_pos"] = right_joint_pos

        return action

    def _convert_from_delta_to_absolute(
        self, action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert delta joint actions to absolute by adding to current joint positions.

        Args:
            action: Action dict with delta joint positions (keys: left_joint_pos, right_joint_pos)

        Returns:
            Action dict with absolute joint positions
        """
        converted = action.copy()

        # Convert delta to absolute for joints (grippers remain absolute)
        for side in ["left", "right"]:
            joint_key = f"{side}_joint_pos"
            if joint_key in action:
                delta = np.asarray(action[joint_key], dtype=np.float32)
                current = self._current_joint_pos[joint_key]
                converted[joint_key] = current + delta

        return converted

    def _convert_from_delta_ee_to_absolute(
        self, action: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert delta EE pose actions to absolute EE poses.

        This is the inverse operation of the conversion file at
        ``experimental/convert_joint_parquet_to_ee_delta_pose.py``.

        Uses FK on the current tracked joint positions to obtain the current
        end-effector pose, then applies the delta:
            new_pos  = current_pos  + delta_pos
            new_quat = delta_quat * current_quat   (quaternion pre-multiply)

        When ``_grasp_site_offset_m > 0`` the deltas are applied at a virtual
        *grasp site* (offset along TCP local +Z) and then projected back to
        the TCP so that IK targets the correct link.  Concretely::

            grasp_pos  = tcp_pos + tcp_rot @ [0, 0, offset]
            new_grasp  = grasp_pos + delta_pos
            new_rot    = delta_rot * tcp_rot
            new_tcp    = new_grasp - new_rot @ [0, 0, offset]

        The result is an absolute cartesian action that can be fed into the
        existing ``_convert_from_cartesian_to_joint`` pipeline.

        Grippers are assumed to be absolute and are passed through unchanged.

        Args:
            action: Dict with delta EE keys:
                ``{side}_ee_pos`` (3,), ``{side}_ee_quat_xyzw`` (4,),
                ``{side}_gripper_pos`` (1,).

        Returns:
            Dict with the same keys but values replaced by **absolute** EE poses.
        """
        # FK from tracked joint positions to get current EE poses
        left_jp = self._current_joint_pos["left_joint_pos"]
        right_jp = self._current_joint_pos["right_joint_pos"]
        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kinematics.forward_kinematics(
            left_jp, right_jp
        )

        converted = action.copy()  # Keep grippers absolute
        offset = self._grasp_site_offset_m
        offset_vec = np.array([0.0, 0.0, offset], dtype=np.float64)  # TCP local +Z

        for side, cur_pos, cur_q in [
            ("left", cur_l_pos, cur_l_q),
            ("right", cur_r_pos, cur_r_q),
        ]:
            pos_key = f"{side}_ee_pos"
            quat_key = f"{side}_ee_quat_xyzw"

            if pos_key in action and quat_key in action:
                delta_pos = np.asarray(action[pos_key], dtype=np.float64)
                delta_q = np.asarray(action[quat_key], dtype=np.float64)

                current_rot = Rotation.from_quat(cur_q)
                delta_rot = Rotation.from_quat(delta_q)
                new_rot = delta_rot * current_rot

                if offset > 0.0:
                    # Apply delta at the grasp site, then project back to TCP.
                    grasp_pos = cur_pos + current_rot.apply(offset_vec)
                    new_grasp = grasp_pos + delta_pos
                    new_pos = new_grasp - new_rot.apply(offset_vec)
                else:
                    # No offset — apply delta directly at TCP (original path).
                    new_pos = cur_pos + delta_pos

                new_q = new_rot.as_quat()  # xyzw

                converted[pos_key] = new_pos.astype(np.float32)
                converted[quat_key] = new_q.astype(np.float32)

        return converted
