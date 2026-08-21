# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASPIRE MuJoCo environment for the YAM bimanual robot station."""

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import tyro

from robot.yam._base_yam_env import _BaseYamEnv


class YamSimEnv(_BaseYamEnv):
    GRIPPER_CTRL_SCALE = 0.041
    GRIPPER_QPOS_SCALE = 0.0376

    def __init__(
        self,
        control_mode: Literal["joint_position", "cartesian_position", "delta_joint_position", "delta_ee_pose"] = None,
        policy_control_freq: float = 30.0,
        enable_cameras: bool = True,
    ):
        if control_mode is None:
            raise ValueError("Control mode must be specified")
        super().__init__(control_mode=control_mode, enable_cameras=enable_cameras)
        self.policy_control_freq = policy_control_freq
        self.control_period = 1.0 / policy_control_freq
        self.last_step_time = time.time()

        self._data = mujoco.MjData(self._model)
        self._n_substeps = int(self.control_period // self._model.opt.timestep)
        self._renderer: Optional[mujoco.Renderer] = None

        # Joint mappings
        self.left_joint_names = [x.name for x in self._spec.joints if x.name.startswith("left_")]
        self.right_joint_names = [x.name for x in self._spec.joints if x.name.startswith("right_")]
        self.left_joint_ids = np.array(
            [self._model.joint(name).id for name in self.left_joint_names]
        )
        self.right_joint_ids = np.array(
            [self._model.joint(name).id for name in self.right_joint_names]
        )

        # Actuator mappings
        self.actuator_ids = np.concatenate([self.left_actuator_ids, self.right_actuator_ids])

        # Camera mappings
        self.camera_ids = {}
        if self.enable_cameras:
            for side in ["top", "left", "right"]:
                camera = next(x for x in self._spec.cameras if side in x.name)
                self.camera_ids[side] = camera.name

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Convert from delta EE pose to absolute cartesian, then to joint
        _was_delta_ee = self.control_mode == "delta_ee_pose"
        if _was_delta_ee:
            action = self._convert_from_delta_ee_to_absolute(action)
            action = self._convert_from_cartesian_to_joint(action)
        # Convert from Cartesian space to absolute joint positions
        elif self.control_mode == "cartesian_position":
            action = self._convert_from_cartesian_to_joint(action)
        # Convert from delta joint space to absolute joint positions
        elif self.control_mode == "delta_joint_position":
            action = self._convert_from_delta_to_absolute(action)

        # Set control inputs
        ctrl = np.concatenate(
            [
                action["left_joint_pos"],
                action["left_gripper_pos"] * self.GRIPPER_CTRL_SCALE,
                action["right_joint_pos"],
                action["right_gripper_pos"] * self.GRIPPER_CTRL_SCALE,
            ]
        )
        self._data.ctrl[self.actuator_ids] = ctrl

        # Advance the simulation
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # Maintain desired control freq
        sleep_end_time = self.last_step_time + self.control_period
        if time.time() > sleep_end_time:
            print(
                f"Warning: Control loop timing overrun "
                f"(budget {self.control_period * 1000:.1f} ms, "
                f"actual {1000 * (time.time() - self.last_step_time):.1f} ms)"
            )
        while time.time() < sleep_end_time:
            time.sleep(0.0001)
        self.last_step_time = time.time()

        obs = self._get_obs()  # Note: Image rendering very slow on MBP (30+ ms)
        _obs_ts = obs.pop("__timestamps", None)

        # Update current joint positions for delta control modes
        if self.control_mode in ("delta_joint_position", "delta_ee_pose"):
            self._current_joint_pos["left_joint_pos"] = obs["left_joint_pos"].copy()
            self._current_joint_pos["right_joint_pos"] = obs["right_joint_pos"].copy()

        reward = self._compute_reward()
        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        # Expose the post-IK joint action for delta_ee_pose debugging.
        # Format: 14D [left_jp(6), left_grip(1), right_jp(6), right_grip(1)]
        if _was_delta_ee:
            info["_debug_joint_action"] = np.concatenate([
                action["left_joint_pos"],
                action["left_gripper_pos"],
                action["right_joint_pos"],
                action["right_gripper_pos"],
            ]).astype(np.float64)

        return obs, reward, False, False, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        print("Resetting the environment...")

        # Reset the simulation
        mujoco.mj_resetData(self._model, self._data)

        for side, joint_ids, actuator_ids in [
            ("left", self.left_joint_ids, self.left_actuator_ids),
            ("right", self.right_joint_ids, self.right_actuator_ids),
        ]:
            # Set initial state if provided
            if options is not None and "initial_state" in options:
                initial_state = options["initial_state"]

                # Joint positions
                self._data.qpos[joint_ids[:6]] = initial_state[f"{side}_joint_pos"]
                self._data.qpos[joint_ids[6:8]] = initial_state[f"{side}_gripper_pos"] * np.array(
                    [self.GRIPPER_QPOS_SCALE, -self.GRIPPER_QPOS_SCALE]
                )

                # Control inputs
                self._data.ctrl[actuator_ids[:6]] = initial_state[f"{side}_joint_pos"]
                self._data.ctrl[actuator_ids[6:7]] = (
                    initial_state[f"{side}_gripper_pos"] * self.GRIPPER_CTRL_SCALE
                )

            else:
                # If no initial state is provided, just open the grippers
                self._data.qpos[joint_ids[6:8]] = [
                    self.GRIPPER_QPOS_SCALE,
                    -self.GRIPPER_QPOS_SCALE,
                ]
                self._data.ctrl[actuator_ids[6:7]] = self.GRIPPER_CTRL_SCALE

        # Advance the simulation
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)

        # Initialize current joint positions for delta control tracking
        if self.control_mode in ("delta_joint_position", "delta_ee_pose"):
            self._current_joint_pos["left_joint_pos"] = obs["left_joint_pos"].copy()
            self._current_joint_pos["right_joint_pos"] = obs["right_joint_pos"].copy()

        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        return obs, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {}
        _ts: dict[str, float] = {}

        # Joint and gripper states
        for side, joint_ids in [
            ("left", self.left_joint_ids),
            ("right", self.right_joint_ids),
        ]:
            qpos = self._data.qpos[joint_ids]
            qpos_arm = qpos[:6]
            qpos_gripper = np.abs(qpos[6:]).mean(keepdims=True) / self.GRIPPER_QPOS_SCALE

            obs[f"{side}_joint_pos"] = qpos_arm
            obs[f"{side}_gripper_pos"] = qpos_gripper
            _ts[f"{side}_state"] = time.time()

        # Update current joint positions for delta control tracking
        if self.control_mode in ("delta_joint_position", "delta_ee_pose"):
            self._current_joint_pos["left_joint_pos"] = obs["left_joint_pos"].copy()
            self._current_joint_pos["right_joint_pos"] = obs["right_joint_pos"].copy()

        # Convert from joint space to Cartesian space
        if self.control_mode == "cartesian_position":
            obs = self._convert_from_joint_to_cartesian(obs)

        # Camera images
        for camera_name in ["top", "left", "right"]:
            if self.enable_cameras:
                obs[f"{camera_name}_camera_image"] = self._render(camera_name)
            else:
                obs[f"{camera_name}_camera_image"] = self._blank_camera_image()
            _ts[f"{camera_name}_camera"] = time.time()

        # Convert float64 to float32
        obs = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in obs.items()}

        # Attach per-component creation timestamps
        obs["__timestamps"] = _ts

        return obs

    def _render(self, camera_name: str) -> np.ndarray:
        if not self.enable_cameras:
            return self._blank_camera_image()
        camera_id = self.camera_ids[camera_name]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, self.CAMERA_HEIGHT, self.CAMERA_WIDTH)
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()
        self._renderer.update_scene(self._data, camera=camera_id)
        return self._renderer.render()

    def _compute_reward(self) -> float:
        return 0.0

    def _get_info(self) -> dict[str, Any]:
        return {}


class YamSimPickRedCubeEnv(YamSimEnv):
    """A specific task environment for picking up a red cube."""

    # Task-specific constants
    CUBE_SIZE = 0.015
    CUBE_SPAWN_POS = [0.6, -0.3, 0.753]
    GOAL_REGION_POS = [0.6, -0.3, 0.753 + 0.3]
    GOAL_REGION_SIZE = [0.25, 0.25, 0.1]
    RANDOMIZATION_RANGE = 0.15
    JOINT_NOISE_RANGE = 0.1

    def __init__(self, **kwargs):
        self._randomize_scene = kwargs.pop("randomize_scene", True)
        super().__init__(**kwargs)

    def _build_task_spec(self, station_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Add a red cube and goal region to the station."""
        self._add_cube_to_scene(station_spec)
        self._add_goal_region_to_scene(station_spec)
        return station_spec

    def _add_cube_to_scene(self, station_spec: mujoco.MjSpec) -> None:
        """Add a red cube to the scene."""
        spawn_pos = self.CUBE_SPAWN_POS.copy()
        spawn_pos[2] += self.CUBE_SIZE / 2  # Adjust for cube height
        cube_spawn_site = station_spec.worldbody.add_site(pos=spawn_pos)

        cube_spec = mujoco.MjSpec()
        body = cube_spec.worldbody.add_body(name="cube_body")
        body.add_geom(
            name="red_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.CUBE_SIZE] * 3,
            rgba=[1, 0, 0, 1],
        )

        cube_body = cube_spawn_site.attach_body(cube_spec.worldbody, "cube_", "")
        self.cube_freejoint_name = "cube_joint"
        cube_body.add_freejoint(name=self.cube_freejoint_name)

    def _add_goal_region_to_scene(self, station_spec: mujoco.MjSpec) -> None:
        """Add a transparent goal region above the cube."""
        station_spec.worldbody.add_geom(
            pos=self.GOAL_REGION_POS,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.GOAL_REGION_SIZE,
            rgba=[0.5, 1, 0.5, 0.05],
            contype=0,
            conaffinity=0,
            group=2,
            mass=0,
        )

    def step(self, action):
        """Execute action and terminate on success (cube in goal region)."""
        obs, reward, terminated, truncated, info = super().step(action)
        if reward > 0.5:
            terminated = True

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        # Cube world position from its free joint: qpos = [x, y, z, qw, qx, qy, qz]
        cube_pos = self._data.jnt(self.cube_freejoint_name).qpos[:3]

        # Axis-aligned bounds of the goal region
        min_corner = np.array(self.GOAL_REGION_POS) - np.array(self.GOAL_REGION_SIZE)
        max_corner = np.array(self.GOAL_REGION_POS) + np.array(self.GOAL_REGION_SIZE)

        # Reward if cube center is inside the goal region
        if np.all(min_corner <= cube_pos) and np.all(cube_pos <= max_corner):
            return 1.0
        return 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment with optional scene randomization."""
        super().reset(seed=seed, options=options)

        if self._randomize_scene:
            self._randomize_cube_position()
            self._randomize_robot_joints()

        mujoco.mj_forward(self._model, self._data)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _randomize_cube_position(self) -> None:
        """Randomize the cube's initial position and orientation."""
        cube_joint = self._data.jnt(self.cube_freejoint_name)

        # Randomize x, y position
        random_offset = self.np_random.uniform(
            -self.RANDOMIZATION_RANGE, self.RANDOMIZATION_RANGE, size=2
        )
        cube_joint.qpos[:2] += random_offset

        # Randomize rotation around z-axis
        perturb_axis = np.array([0.0, 0.0, 1.0])
        perturb_theta = self.np_random.uniform(-np.pi, np.pi)
        mujoco.mju_axisAngle2Quat(cube_joint.qpos[3:], perturb_axis, perturb_theta)

    def _randomize_robot_joints(self) -> None:
        """Add noise to robot joint positions."""
        all_joint_names = self.left_joint_names + self.right_joint_names
        arm_joint_names = [j for j in all_joint_names if "finger" not in j]

        for joint_name in arm_joint_names:
            joint = self._data.jnt(joint_name)
            noise = self.np_random.uniform(
                -self.JOINT_NOISE_RANGE, self.JOINT_NOISE_RANGE, size=joint.qpos.shape
            )
            joint.qpos += noise


class MujocoViewerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.viewer = mujoco.viewer.launch_passive(env.unwrapped._model, env.unwrapped._data)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update viewer
        if not self.viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed")
        self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def close(self):
        self.viewer.close()


class SimDummyPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self) -> None:
        return None

    def get_action(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Simple sinusoidal action for testing
        t = time.time()
        frequency = 0.5
        osc = np.sin(2 * np.pi * frequency * t)

        def scale(key, low=None, high=None, factor=1.0):
            if low is None:
                low = self.action_space[key].low
            if high is None:
                high = self.action_space[key].high
            mid = (low + high) / 2
            amp = factor * (high - low) / 2
            return mid + amp * osc

        action = {
            "left_gripper_pos": scale("left_gripper_pos"),
            "right_gripper_pos": scale("right_gripper_pos"),
        }

        if "left_joint_pos" in obs:
            action["left_joint_pos"] = scale("left_joint_pos", factor=0.1)
            action["right_joint_pos"] = scale("right_joint_pos", factor=0.1)

        elif "left_ee_pos" in obs:
            default_quat = np.array([-0.5, 0.5, -0.5, 0.5])
            action["left_ee_pos"] = scale(
                "left_ee_pos",
                low=np.array([0.6, 0.2, 1.0]),
                high=np.array([0.8, 0.4, 1.2]),
            )
            action["left_ee_quat"] = default_quat
            action["right_ee_pos"] = scale(
                "right_ee_pos",
                low=np.array([0.6, -0.4, 1.0]),
                high=np.array([0.8, -0.2, 1.2]),
            )
            action["right_ee_quat"] = default_quat

        return action, {}


@dataclass
class SimConfig:
    """Configuration for YamSim environment."""
    control_mode: Literal["joint_position", "cartesian_position", "delta_joint_position", "delta_ee_pose"] = "joint_position"
    """Control mode for the robot."""
    policy_control_freq: float = 30.0
    """Policy control frequency in Hz."""
    num_steps: int = 1000
    """Number of steps to run the simulation."""
    headless: bool = False
    """Run without visualization (for headless servers)."""


def main(config: SimConfig):
    from gymnasium.envs.registration import register

    # Environment
    register(id="YamSim-v0", entry_point="robot.yam.yam_sim_env:YamSimEnv")
    env = gym.make(
        "YamSim-v0",
        control_mode=config.control_mode,
        policy_control_freq=config.policy_control_freq
    )
    if not config.headless:
        env = MujocoViewerWrapper(env)  # Visualize env with Mujoco viewer
    else:
        print("Running in headless mode (no visualization)")

    # Policy
    policy = SimDummyPolicy(env.action_space)

    # Main loop
    obs, info = env.reset()
    for _ in range(config.num_steps):
        action, _ = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    tyro.cli(main)
