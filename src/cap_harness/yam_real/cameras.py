"""Camera access for the YAM station, driven by the station profile.

Cameras are opened **by serial**, taken from the profile and cross-checked
against the calibration bundle, rather than by device index or udev symlink.
Index order changes when a cable is re-seated, and a symlink says nothing about
which physical camera it points at; a serial is the only identifier that ties a
frame to the extrinsics solved for that camera. The loader already refuses a
profile whose serial disagrees with its bundle, so by the time a camera is
opened here the identity has been checked.

Frames are read on a background thread and cached. Depth backends are slow
enough that a synchronous read in the observation path would stall the 60 Hz
command loop; a caller always gets the most recent frame instead of waiting for
the next one.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Protocol

import numpy as np

#: Cap on the background reader, so depth backends cannot saturate the CPU and
#: starve the control loop.
DEFAULT_MAX_FPS = 30.0

#: How long to wait for a camera's first frame before giving up on it.
FIRST_FRAME_TIMEOUT_S = 8.0


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One RGB-D sample, in the units the harness contracts expect."""

    rgb: np.ndarray
    """(H, W, 3) uint8."""

    depth_m: np.ndarray
    """(H, W) float, **metres** -- not the sensor's raw uint16."""

    intrinsics: np.ndarray
    """3x3 pinhole matrix for the colour stream."""

    timestamp_s: float
    """Host monotonic time when the sample was acquired."""


class Camera(Protocol):
    """What the env needs from any camera backend."""

    def read(self) -> CameraFrame | None: ...

    def close(self) -> None: ...


class SyntheticCamera:
    """Deterministic RGB-D for off-robot runs.

    The shared ``Observation`` contract requires at least one camera, so a
    station with no hardware cannot produce a valid observation at all. Rather
    than weaken that contract for tests, the simulated station synthesizes a
    frame: enough to exercise the codec and the contracts, and obviously
    synthetic so no one mistakes it for a render of the scene.
    """

    def __init__(self, width: int = 64, height: int = 48) -> None:
        self.width = int(width)
        self.height = int(height)
        column = np.linspace(0, 255, self.width, dtype=np.uint8)
        self._rgb = np.repeat(column[None, :, None], self.height, axis=0).repeat(3, axis=2)
        self._depth = np.full((self.height, self.width), 0.75, dtype=np.float64)
        # Nominal pinhole with a 60-degree horizontal field of view.
        focal = self.width / (2.0 * np.tan(np.deg2rad(30.0)))
        self._intrinsics = np.array(
            [[focal, 0.0, self.width / 2.0], [0.0, focal, self.height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def read(self) -> CameraFrame:
        return CameraFrame(
            rgb=self._rgb.copy(),
            depth_m=self._depth.copy(),
            intrinsics=self._intrinsics.copy(),
            timestamp_s=time.monotonic(),
        )

    def close(self) -> None:
        """Nothing to release."""


class RealSenseCamera:
    """Intel RealSense colour+depth stream, selected by serial.

    Depth is aligned to colour and converted to metres here, so every consumer
    sees one convention. ``pyrealsense2`` is imported lazily: the harness must
    stay importable on a machine with no camera SDK.
    """

    def __init__(
        self,
        serial: str,
        *,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        brightness: int = 10,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self.serial = str(serial)
        available = {
            device.get_info(rs.camera_info.serial_number): device
            for device in rs.context().query_devices()
        }
        if self.serial not in available:
            raise ValueError(
                f"RealSense {self.serial} not connected; found {sorted(available) or 'none'}"
            )

        config = rs.config()
        config.enable_device(self.serial)
        width, height = int(resolution[0]), int(resolution[1])
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, int(fps))
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(fps))

        self._pipeline = rs.pipeline()
        self.profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self._depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self._configure_exposure(brightness)

    def _configure_exposure(self, brightness: int) -> None:
        """Enable auto-exposure on whichever sensor owns the colour stream.

        The D405 exposes colour and depth through a single "Stereo Module"
        sensor, while a D435 has a separate "RGB Camera" sensor that owns colour.
        Finding the owner by inspecting stream profiles works for both without
        branching on product id. This station is currently all D405, but the
        profile chooses cameras by serial, so a mixed station stays supported.
        """
        rs = self._rs
        for sensor in self.profile.get_device().query_sensors():
            try:
                owns_colour = any(
                    sp.stream_type() == rs.stream.color for sp in sensor.get_stream_profiles()
                )
            except RuntimeError:
                continue
            if not owns_colour:
                continue
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, True)
            if sensor.supports(rs.option.brightness):
                sensor.set_option(rs.option.brightness, max(-64, min(int(brightness), 64)))
            return

    def read(self) -> CameraFrame | None:
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        acquired_s = time.monotonic()
        frames = self._align.process(frames)
        colour = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not colour:
            return None
        intr = colour.profile.as_video_stream_profile().get_intrinsics()
        depth_m = (
            np.zeros((intr.height, intr.width), dtype=np.float64)
            if not depth
            else np.asanyarray(depth.get_data()).astype(np.float64) * self._depth_scale
        )
        return CameraFrame(
            rgb=np.asanyarray(colour.get_data()).astype(np.uint8),
            depth_m=depth_m,
            intrinsics=np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            timestamp_s=acquired_s,
        )

    def close(self) -> None:
        self._pipeline.stop()


class ThreadedCamera:
    """Read a camera on a daemon thread and hand out the most recent frame."""

    def __init__(self, camera: Camera, *, max_fps: float = DEFAULT_MAX_FPS) -> None:
        self._camera = camera
        self._frame: CameraFrame | None = None
        self._lock = threading.Lock()
        self._running = True
        self._reported_error: str | None = None
        self._period_s = 0.0 if max_fps <= 0 else 1.0 / float(max_fps)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while True:
            with self._lock:
                if self._frame is not None:
                    return
            if not self._thread.is_alive():
                raise RuntimeError(f"camera thread died before first frame: {self._reported_error}")
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError(f"camera produced no frame within {FIRST_FRAME_TIMEOUT_S:g}s")
            time.sleep(0.02)

    def _worker(self) -> None:
        while self._running:
            try:
                frame = self._camera.read()
            except Exception as exc:
                self._reported_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.01)
                continue
            if frame is not None:
                with self._lock:
                    self._frame = frame
            if self._period_s > 0.0:
                time.sleep(self._period_s)

    def read(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._camera.close()


def open_station_camera(config: object, *, threaded: bool = True) -> Camera:
    """Open the profile's camera, or a synthetic one when it is unavailable.

    ``config`` is a :class:`~cap_harness.yam_real.config.YamCameraConfig`.
    """
    serial = str(getattr(config, "serial", "")).strip()
    resolution = tuple(getattr(config, "resolution", (640, 480)))
    camera: Camera = RealSenseCamera(serial, resolution=(resolution[0], resolution[1]))
    return ThreadedCamera(camera) if threaded else camera


__all__ = [
    "Camera",
    "CameraFrame",
    "RealSenseCamera",
    "SyntheticCamera",
    "ThreadedCamera",
    "open_station_camera",
]
