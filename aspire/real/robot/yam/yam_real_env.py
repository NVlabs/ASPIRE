"""Real environment for YAM bimanual robot station."""

import logging
import os
import threading
import time
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import portal

from robot.camera_factory import create_camera
from robot.yam._base_yam_env import _BaseYamEnv
from robot.constants import (
    LEFT_FOLLOWER_PORT,
    MAX_JOINT_VELOCITY_RAD_S,
    RIGHT_FOLLOWER_PORT,
)

logger = logging.getLogger(__name__)


class FollowerRobotClient:
    def __init__(self, host: str = "127.0.0.1", port: int = LEFT_FOLLOWER_PORT):
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._client.get_observations().result()


class NonBlockingCamera:
    def __init__(self, camera_name: str):
        self.camera = create_camera(
            camera_name,
            resolution=(640, 480),
            fps=30,
            enable_depth=False,
        )
        self.image = None
        self._read_error_logged = False

        # Camera worker
        self.running = True
        self.worker_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.worker_thread.start()

        # Wait for worker to start
        t0 = time.time()
        while self.image is None:
            if not self.worker_thread.is_alive():
                raise RuntimeError(
                    f"Camera worker exited before first frame for {camera_name!r}"
                )
            if time.time() - t0 > 5.0:
                raise TimeoutError(
                    f"Timed out waiting for first frame from camera {camera_name!r}"
                )
            time.sleep(0.01)

        if self.image is None:
            self.running = False
            self.worker_thread.join(timeout=1.0)
            self.camera.stop()
            raise RuntimeError(
                f"Timed out waiting for first frame from camera {camera_name!r}"
            )

    def _camera_worker(self):
        while self.running:
            try:
                data = self.camera.read()  # Blocks until the next frame is ready.
                self._read_error_logged = False
            except Exception as exc:
                if self.running and not self._read_error_logged:
                    print(f"[YamRealEnv] Camera read error: {exc}")
                    self._read_error_logged = True
                time.sleep(0.01)
                continue
            if data is not None and data.images["rgb"] is not None:
                self.image = data.images["rgb"]

    def get_image(self):
        return self.image.copy()

    def close(self):
        self.running = False
        self.worker_thread.join()
        if self.camera is not None:
            self.camera.stop()


class YamRealEnv(_BaseYamEnv):
    def __init__(
        self,
        control_mode: Literal[
            "joint_position",
            "cartesian_position",
            "delta_joint_position",
            "delta_ee_pose",
        ] = "joint_position",
        policy_control_freq: float = 30.0,
        enable_cameras: bool = True,
        enable_depth_cameras: bool = False,
        depth_camera_names: tuple[str, ...] = ("top",),
        enabled_camera_names: tuple[str, ...] | None = None,
        top_camera_source: Literal["direct", "rpc"] | None = None,
    ):
        super().__init__(control_mode=control_mode, enable_cameras=enable_cameras)
        self.policy_control_freq = policy_control_freq
        self.control_period = 1.0 / policy_control_freq
        self.last_step_time = time.time()
        self.enable_depth_cameras = bool(enable_depth_cameras)
        self.depth_camera_names = tuple(str(x) for x in depth_camera_names)
        self.enabled_camera_names = tuple(
            str(x) for x in (enabled_camera_names or ("top", "left", "right"))
        )
        # Resolve top camera source: explicit param > env var > default "direct"
        if top_camera_source is None:
            env_src = os.environ.get("YAM_TOP_CAMERA_SOURCE", "").strip().lower()
            top_camera_source = "rpc" if env_src == "rpc" else "direct"
        self._top_camera_source: Literal["direct", "rpc"] = top_camera_source
        self._cap_server_host = os.environ.get("CAP_SERVER_HOST", "127.0.0.1")
        self._cap_server_port = int(os.environ.get("CAP_SERVER_PORT", "8300"))

        # Follower arms
        self.follower_arms = {}
        for side, port in [
            ("left", LEFT_FOLLOWER_PORT),
            ("right", RIGHT_FOLLOWER_PORT),
        ]:
            follower_arm = FollowerRobotClient(host="localhost", port=port)

            # Set arm to gravcomp mode
            follower_arm.command_joint_state(
                {
                    "pos": np.zeros(7),
                    "vel": np.zeros(7),
                    "kp": np.zeros(7),
                    "kd": np.zeros(7),
                }
            )

            self.follower_arms[side] = follower_arm

        # Cameras
        self.cameras = (
            {
                camera_name: NonBlockingCamera(camera_name)
                for camera_name in self.enabled_camera_names
            }
            if self.enable_cameras
            else {}
        )

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Convert from delta EE pose to absolute cartesian, then to joint
        if self.control_mode == "delta_ee_pose":
            action = self._convert_from_delta_ee_to_absolute(action)
            action = self._convert_from_cartesian_to_joint(action)
        # Convert from Cartesian space to joint space if cartesian keys are present
        elif "left_ee_pos" in action and "right_ee_pos" in action:
            try:
                action = self._convert_from_cartesian_to_joint(action)
            except ValueError as e:
                raise ValueError(
                    f"Error converting from Cartesian to joint space: {e}, action keys: {list(action.keys())}"
                )
        # Otherwise, convert only when in cartesian mode (legacy behavior)
        elif self.control_mode == "cartesian_position":
            try:
                action = self._convert_from_cartesian_to_joint(action)
            except ValueError as e:
                raise ValueError(
                    f"Error converting from Cartesian to joint space: {e}, action keys: {list(action.keys())}"
                )
        # Convert from delta joint space to absolute
        elif self.control_mode == "delta_joint_position":
            action = self._convert_from_delta_to_absolute(action)

        # --- Safety clamp: limit max joint delta per step ---
        # max_delta = MAX_JOINT_VELOCITY_RAD_S (rad/s) / control_hz (Hz)
        max_delta = MAX_JOINT_VELOCITY_RAD_S / max(self.policy_control_freq, 1.0)
        for side in self.follower_arms:
            current_pos = self.follower_arms[side].get_joint_pos()[:6]
            commanded_pos = action[f"{side}_joint_pos"]
            delta = commanded_pos - current_pos
            if np.any(np.abs(delta) > max_delta):
                clamped_delta = np.clip(delta, -max_delta, max_delta)
                logger.warning(
                    "Joint delta clamp triggered on %s arm: "
                    "max_abs_delta=%.4f rad (limit=%.4f rad = %.1f rad/s / %.0f Hz), "
                    "commanded=%s, actual=%s, clamped_command=%s",
                    side,
                    float(np.max(np.abs(delta))),
                    max_delta,
                    MAX_JOINT_VELOCITY_RAD_S,
                    self.policy_control_freq,
                    np.array2string(commanded_pos, precision=4),
                    np.array2string(current_pos, precision=4),
                    np.array2string(current_pos + clamped_delta, precision=4),
                )
                action[f"{side}_joint_pos"] = current_pos + clamped_delta

        # Command follower arms to desired joint positions
        for side, follower_arm in self.follower_arms.items():
            follower_arm.command_joint_pos(
                np.concatenate(
                    [action[f"{side}_joint_pos"], action[f"{side}_gripper_pos"]]
                )
            )

        # Maintain desired control freq
        sleep_end_time = self.last_step_time + self.control_period
        if time.time() > sleep_end_time:  # TODO: Resolve latency issues
            print(
                f"Warning: Control loop timing overrun "
                f"(budget {self.control_period * 1000:.1f} ms, "
                f"actual {1000 * (time.time() - self.last_step_time):.1f} ms)"
            )
        while time.time() < sleep_end_time:
            time.sleep(0.0001)
        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)

        # Update current joint positions for delta control modes
        if self.control_mode in ("delta_joint_position", "delta_ee_pose"):
            self._current_joint_pos["left_joint_pos"] = obs["left_joint_pos"].copy()
            self._current_joint_pos["right_joint_pos"] = obs["right_joint_pos"].copy()

        reward = self._compute_reward()
        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        return obs, reward, False, False, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = options if options is not None else {}

        # Interpolate to initial state
        print("Resetting the environment...")
        if "initial_state" in options:
            initial_state = options["initial_state"]
        else:
            initial_state = {
                "left_joint_pos": np.zeros(6),
                "left_gripper_pos": np.ones(1) * 0.0,
                "right_joint_pos": np.zeros(6),
                "right_gripper_pos": -np.ones(1),
            }
        self._interpolate_to_target(initial_state)
        print("Done resetting the environment")

        self.last_step_time = time.time()

        obs = self._get_obs()
        _obs_ts = obs.pop("__timestamps", None)

        # Update current joint positions for delta control tracking
        if self.control_mode in ("delta_joint_position", "delta_ee_pose"):
            self._current_joint_pos["left_joint_pos"] = obs["left_joint_pos"].copy()
            self._current_joint_pos["right_joint_pos"] = obs["right_joint_pos"].copy()

        info = self._get_info()
        if _obs_ts is not None:
            info["__timestamps"] = _obs_ts

        return obs, info

    def _interpolate_to_target(
        self,
        joint_state: dict[str, np.ndarray],
        duration: float = 2.0,
    ) -> None:
        """Interpolate to target joint state over given duration."""
        initial_pos = np.concatenate(
            [
                self.follower_arms["left"].get_joint_pos(),
                self.follower_arms["right"].get_joint_pos(),
            ]
        )

        # print("DEBUG: Initial gripper pos [left]:", self.follower_arms["left"].get_joint_pos()[6])
        # print("DEBUG: Initial gripper pos [right]:", self.follower_arms["right"].get_joint_pos()[6])
        target_pos = np.concatenate(
            [
                joint_state["left_joint_pos"],
                joint_state["left_gripper_pos"],
                joint_state["right_joint_pos"],
                joint_state["right_gripper_pos"],
            ]
        )
        assert initial_pos.shape == (14,) and target_pos.shape == (14,)

        # Use lower gains during interpolation
        kp = np.array(
            [40, 40, 40, 20, 5, 5, 10]
        )  # Default: [80, 80, 80, 40, 10, 10, 20]
        kd = np.array(
            [5, 5, 5, 1.5, 1.5, 1.5, 0.5]
        )  # Default: [5, 5, 5, 1.5, 1.5, 1.5, 0.5]

        start_time = time.time()
        while time.time() - start_time < duration:
            # Linearly interpolate between initial and target joint state
            alpha = (time.time() - start_time) / duration
            interp_pos = (1 - alpha) * initial_pos + alpha * target_pos

            # Command follower arms to interpolated joint state
            for side, joint_pos in [
                ("left", interp_pos[:7]),
                ("right", interp_pos[7:]),
            ]:
                self.follower_arms[side].command_joint_state(
                    {
                        "pos": joint_pos,
                        "vel": np.zeros(7),
                        "kp": kp,
                        "kd": kd,
                    }
                )

            time.sleep(0.02)

        # Keep commanding the final target until we are close enough.
        settle_start = time.time()
        settle_timeout = 1.0
        settle_tol = 0.01
        while time.time() - settle_start < settle_timeout:
            current_pos = np.concatenate(
                [
                    self.follower_arms["left"].get_joint_pos(),
                    self.follower_arms["right"].get_joint_pos(),
                ]
            )
            max_err = float(np.max(np.abs(current_pos - target_pos)))
            for side, joint_pos in [
                ("left", target_pos[:7]),
                ("right", target_pos[7:]),
            ]:
                self.follower_arms[side].command_joint_state(
                    {
                        "pos": joint_pos,
                        "vel": np.zeros(7),
                        "kp": kp,
                        "kd": kd,
                    }
                )
            if max_err < settle_tol:
                break
            time.sleep(0.02)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = {}
        _ts: dict[str, float] = {}

        # Joint and gripper states
        for side, follower_arm in self.follower_arms.items():
            follower_obs = follower_arm.get_observations()
            _ts[f"{side}_state"] = time.time()

            obs[f"{side}_joint_pos"] = follower_obs["joint_pos"]
            obs[f"{side}_gripper_pos"] = follower_obs["gripper_pos"]

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
                obs[f"{camera_name}_camera_image"] = self.cameras[
                    camera_name
                ].get_image()
            else:
                obs[f"{camera_name}_camera_image"] = self._blank_camera_image()
            _ts[f"{camera_name}_camera"] = time.time()

        # Convert float64 to float32
        obs = {
            k: v.astype(np.float32) if v.dtype == np.float64 else v
            for k, v in obs.items()
        }

        # Attach per-component creation timestamps
        obs["__timestamps"] = _ts

        return obs

    def _compute_reward(self) -> float:
        return 0.0

    def _get_info(self) -> dict[str, Any]:
        return {}


class RealDummyPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self) -> None:
        return None

    def get_action(self, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        # Simple sinusoidal action for testing
        t = time.time()
        frequency = 0.5
        osc = np.sin(2 * np.pi * frequency * t)

        def scale(key):
            low = self.action_space[key].low
            high = self.action_space[key].high
            mid = (low + high) / 2
            amp = (high - low) / 2 * 0.5
            return mid + amp * osc

        action = {
            "left_joint_pos": scale("left_joint_pos"),
            "left_gripper_pos": scale("left_gripper_pos"),
            "right_joint_pos": scale("right_joint_pos"),
            "right_gripper_pos": scale("right_gripper_pos"),
        }

        # For real robot testing, only move last arm joint and gripper
        action["left_joint_pos"][:5] = 0.0
        action["right_joint_pos"][:5] = 0.0

        return action, {}


def main():
    from gymnasium.envs.registration import register

    # Environment
    register(id="YamReal-v0", entry_point="robot.yam.yam_real_env:YamRealEnv")
    env = gym.make("YamReal-v0")

    # Policy
    policy = RealDummyPolicy(env.action_space)

    # Main loop
    obs, info = env.reset()

    print("DEBUG: Info:", info)
    for _ in range(1000):
        action, _ = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    main()
