"""Real bimanual YAM environment — direct-mode, no CapServer required."""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from cap.env.base.profile import RobotProfile, yam_profile
from robot.camera_factory import create_camera
from robot.constants import LEFT_FOLLOWER_PORT, RIGHT_FOLLOWER_PORT
from robot.station_profiles import active_station_cameras
from robot.yam.kinematics import YamKinematics
from robot.yam.yam_real_env import FollowerRobotClient


class _CameraThread:
    """Background daemon that reads and caches RGB, depth, and intrinsics."""

    def __init__(self, name: str):
        self._name = name
        self._camera = create_camera(name, enable_depth=True)
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: dict | None = None
        self._lock = threading.Lock()
        self._running = True
        self._error_logged = False
        self._period_s = max(
            0.0,
            1.0 / float(os.environ.get("CAP_CAMERA_THREAD_MAX_FPS", "30")),
        )
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # Block until first RGB frame arrives
        t0 = time.time()
        while True:
            with self._lock:
                if self._rgb is not None:
                    break
            if not self._thread.is_alive():
                raise RuntimeError(f"Camera thread for {name!r} died before first frame")
            if time.time() - t0 > 8.0:
                raise TimeoutError(f"Camera {name!r} timed out waiting for first frame")
            time.sleep(0.02)

    def _worker(self) -> None:
        while self._running:
            try:
                data = self._camera.read()
                self._error_logged = False
            except Exception as exc:
                if self._running and not self._error_logged:
                    print(f"[CameraThread] read error: {exc}")
                    self._error_logged = True
                time.sleep(0.01)
                continue
            if data is None:
                continue
            with self._lock:
                if data.images.get("rgb") is not None:
                    self._rgb = data.images["rgb"]
                if data.depth is not None:
                    self._depth = data.depth
                if data.intrinsics is not None:
                    self._intrinsics = dict(data.intrinsics)
            # Do not spin the RGB-D backends as fast as possible.  Without
            # throttling, ZED NEURAL depth + two RealSense depth streams can
            # saturate CPU/GIL scheduling enough to stall the script runner
            # before user code executes.
            if self._period_s > 0:
                time.sleep(self._period_s)

    def get_rgb(self) -> np.ndarray | None:
        with self._lock:
            return self._rgb.copy() if self._rgb is not None else None

    def get_depth(self) -> np.ndarray | None:
        with self._lock:
            return self._depth.copy() if self._depth is not None else None

    def get_intrinsics(self) -> dict | None:
        with self._lock:
            return dict(self._intrinsics) if self._intrinsics is not None else None

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        if hasattr(self._camera, "stop"):
            self._camera.stop()


class RealYamEnv:
    """Real bimanual YAM hardware environment for direct-mode agent runs.

    Wraps FollowerRobotClient arms, depth-enabled camera threads, and
    YamKinematics (FK/IK). Provides the same observation/camera API as
    RoboCasaEnv so skills.py tools work identically on real hardware.

    Camera names come from the active station profile. Override with
    OPENFORGE_REAL_YAM_CAMERAS/CAP_REAL_YAM_CAMERAS when a station has optional
    debug cameras such as "bottom".
    """

    def __init__(
        self,
        enable_cameras: bool = True,
        camera_names: tuple[str, ...] | list[str] | None = None,
    ):
        self._profile: RobotProfile = yam_profile()
        self._arms: dict[str, FollowerRobotClient] = {
            "left": FollowerRobotClient(port=LEFT_FOLLOWER_PORT),
            "right": FollowerRobotClient(port=RIGHT_FOLLOWER_PORT),
        }
        self.kin = YamKinematics()
        # Serialize all YamKinematics mutations (FK in get_observations,
        # IK in freespace_move, FK for extrinsics — never concurrent).
        self._kin_lock = threading.Lock()

        self._cameras: dict[str, _CameraThread] = {}
        self.camera_names: tuple[str, ...] = tuple(
            camera_names if camera_names is not None else active_station_cameras().names
        )
        if enable_cameras:
            for name in self.camera_names:
                try:
                    self._cameras[name] = _CameraThread(name)
                    print(f"[RealYamEnv] Camera '{name}' ready")
                except Exception as exc:
                    # Match cap_server.py's deprecated-but-robust behavior:
                    # cameras are useful for vision tools, but arm-only scripts
                    # should not fail or hang just because one RGB-D device is
                    # temporarily unavailable / busy / slow to enumerate.
                    print(f"[RealYamEnv] Camera '{name}' unavailable: {exc}")

    # ------------------------------------------------------------------
    # Arm observations
    # ------------------------------------------------------------------

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """Return {joint_pos(6), gripper_pos(1), ee_pos(3), ee_quat(4)} for one arm.

        FK is computed for both arms simultaneously (mink requires bimanual qpos).
        """
        obs_l = self._arms["left"].get_observations()
        obs_r = self._arms["right"].get_observations()
        ljp = obs_l["joint_pos"]
        rjp = obs_r["joint_pos"]
        with self._kin_lock:
            self._seed_kin(ljp, rjp)
            l_pos, l_q, r_pos, r_q = self.kin.forward_kinematics(ljp, rjp)
        side_obs = obs_l if side == "left" else obs_r
        return {
            "joint_pos": side_obs["joint_pos"],
            "gripper_pos": side_obs["gripper_pos"],
            "ee_pos": l_pos if side == "left" else r_pos,
            "ee_quat": l_q if side == "left" else r_q,
        }

    def _seed_kin(self, ljp: np.ndarray, rjp: np.ndarray) -> None:
        """Seed mink configuration with observed joint positions. Must be called inside _kin_lock."""
        self.kin.configuration.data.qpos[:6] = ljp
        self.kin.configuration.data.qpos[8:14] = rjp
        self.kin.configuration.update()

    # ------------------------------------------------------------------
    # Arm commands
    # ------------------------------------------------------------------

    def command_joint_pos(self, side: str, joint_pos_7d: np.ndarray) -> None:
        """Send a 7D [joint×6 + gripper×1] position command. Uses arm_server default gains."""
        self._arms[side].command_joint_pos(joint_pos_7d)

    def command_joint_state(self, side: str, state: dict) -> None:
        """Send a full joint state dict (pos, vel, kp, kd) to one arm."""
        self._arms[side].command_joint_state(state)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def render_rgb(self, camera: str) -> np.ndarray | None:
        cam = self._cameras.get(str(camera).strip().lower())
        return cam.get_rgb() if cam is not None else None

    def render_depth(self, camera: str) -> np.ndarray | None:
        cam = self._cameras.get(str(camera).strip().lower())
        return cam.get_depth() if cam is not None else None

    def get_camera_intrinsics(self, camera: str) -> list[float]:
        """Return [fx, fy, cx, cy]."""
        cam = self._cameras.get(str(camera).strip().lower())
        if cam is None:
            return [0.0, 0.0, 0.0, 0.0]
        intr = cam.get_intrinsics()
        if intr is None:
            return [0.0, 0.0, 0.0, 0.0]
        return [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]

    @staticmethod
    def _pose_to_camera_extrinsics(T, *, camera: str) -> dict:
        from robot.models.station.paths import needs_optical_flip

        rot = T.rotation()
        if hasattr(rot, "as_matrix"):
            rot_mat = rot.as_matrix()
        else:
            maybe_matrix = getattr(rot, "matrix", rot)
            rot_mat = maybe_matrix() if callable(maybe_matrix) else maybe_matrix

        return {
            "position": T.translation().tolist(),
            "rotation": np.asarray(rot_mat, dtype=np.float64).reshape(3, 3).tolist(),
            "needs_optical_flip": needs_optical_flip(camera),
        }

    def get_camera_extrinsics(self, camera: str) -> dict:
        """Return {position, rotation, needs_optical_flip} via FK to camera body frame."""
        from robot.models.station.paths import get_top_camera_frame

        camera = str(camera).strip().lower()
        cam_frame_map = {
            "top": os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame()),
            "left": os.environ.get("CAP_LEFT_CAMERA_FRAME", "left_camera_d405"),
            "right": os.environ.get("CAP_RIGHT_CAMERA_FRAME", "right_camera_d405"),
            "bottom": os.environ.get("CAP_BOTTOM_CAMERA_FRAME", "").strip() or None,
        }
        frame_name = cam_frame_map.get(camera)
        if frame_name is None:
            raise RuntimeError(
                f"Camera {camera!r} has no calibrated robot/world-frame extrinsics. "
                "Use it for preview/recording/debug only, or set CAP_<NAME>_CAMERA_FRAME."
            )

        if camera == "top":
            with self._kin_lock:
                T = self.kin.configuration.get_transform_frame_to_world(frame_name, "body")
            return self._pose_to_camera_extrinsics(T, camera=camera)

        obs_l = self._arms["left"].get_observations()
        obs_r = self._arms["right"].get_observations()
        with self._kin_lock:
            self._seed_kin(obs_l["joint_pos"], obs_r["joint_pos"])
            T = self.kin.configuration.get_transform_frame_to_world(frame_name, "body")

        return self._pose_to_camera_extrinsics(T, camera=camera)

    # ------------------------------------------------------------------

    def set_recorder(self, recorder) -> None:
        """No-op — recording is not supported on real hardware."""

    def close(self) -> None:
        if hasattr(self, "_dashboard"):
            self._dashboard.stop()
        for cam in self._cameras.values():
            cam.close()
