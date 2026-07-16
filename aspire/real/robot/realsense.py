from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from packaging import version
import pyrealsense2 as rs
import tyro


def get_device_info() -> Dict[str, str]:
    """Get device information mapping serial numbers to firmware versions."""
    ctx = rs.context()
    devices = ctx.query_devices()
    device_info = {}
    for dev in devices:
        serial_number = dev.get_info(rs.camera_info.serial_number)
        firmware_version = dev.get_info(rs.camera_info.firmware_version)
        device_info[serial_number] = firmware_version
    return device_info


@dataclass
class CameraData:
    """Container for camera frame data."""

    images: Dict[str, Optional[np.ndarray]]
    timestamp: float
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[Dict[str, float]] = None


@dataclass
class RealSenseCamera:
    """Standalone RealSense RGB camera driver."""

    device_id: Optional[str] = None
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 60
    auto_exposure: bool = True
    brightness: int = 10  # Range -64 to 64
    exposure_value: Optional[int] = None
    enable_depth: bool = False
    align_depth_to_color: bool = True

    def __post_init__(self):
        if self.exposure_value is not None:
            self.auto_exposure = False
        self.brightness = max(-64, min(self.brightness, 64))

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise ValueError("No RealSense devices found")

        self._pipeline = rs.pipeline()
        config = rs.config()

        if self.device_id is not None:
            device_info = get_device_info()
            if self.device_id not in device_info:
                raise ValueError(f"Device {self.device_id} not found")
            config.enable_device(self.device_id)

            # Check firmware compatibility
            firmware_version = device_info[self.device_id]
            if version.parse(firmware_version) > version.parse("5.13.0.50"):
                raise RuntimeWarning(
                    f"Firmware {firmware_version} might have auto exposure issues. "
                )

        config.enable_stream(
            rs.stream.color,
            self.resolution[0],
            self.resolution[1],
            rs.format.rgb8,
            self.fps,
        )

        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.resolution[0],
                self.resolution[1],
                rs.format.z16,
                self.fps,
            )

        self.profile = self._pipeline.start(config)
        self._configure_exposure()

        if self.enable_depth:
            self._align = (
                rs.align(rs.stream.color) if self.align_depth_to_color else None
            )
            self._depth_scale = (
                self.profile.get_device().first_depth_sensor().get_depth_scale()
            )

    def _configure_exposure(self) -> None:
        """Configure exposure settings."""
        device = self.profile.get_device()
        for sensor in device.query_sensors():
            if sensor.get_info(rs.camera_info.name) == "Stereo Module":
                if self.auto_exposure:
                    sensor.set_option(rs.option.enable_auto_exposure, True)
                    sensor.set_option(rs.option.brightness, self.brightness)
                else:
                    if self.exposure_value is None:
                        raise ValueError(
                            "Exposure value required when auto_exposure=False"
                        )
                    sensor.set_option(rs.option.enable_auto_exposure, False)
                    sensor.set_option(rs.option.exposure, self.exposure_value)
                break

    def read(self) -> CameraData:
        """Read a frame from the camera."""
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)

        depth_array = None
        intrinsics_dict = None

        if self.enable_depth:
            if self._align is not None:
                frames = self._align.process(frames)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_array = (
                    np.asanyarray(depth_frame.get_data()).astype(np.float32)
                    * self._depth_scale
                )
                intrinsics_dict = self.get_intrinsics()

        color_frame = frames.get_color_frame()
        return CameraData(
            images={"rgb": np.asanyarray(color_frame.get_data())},
            timestamp=frames.get_timestamp(),
            depth=depth_array,
            intrinsics=intrinsics_dict,
        )

    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics from the color stream."""
        intr = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "width": float(intr.width),
            "height": float(intr.height),
        }

    def stop(self) -> None:
        """Stop the camera."""
        self._pipeline.stop()


def visualize_camera(camera: RealSenseCamera) -> None:
    """Display camera feed with OpenCV."""
    while True:
        try:
            data = camera.read()
            img = data.images["rgb"]
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    img_bgr,
                    f"TS: {data.timestamp / 1000:.3f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("RealSense", img_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()


@dataclass
class Args:
    """Command line arguments."""

    device_id: Optional[str] = None
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 60
    auto_exposure: bool = True
    brightness: int = 10
    exposure: Optional[int] = None
    enable_depth: bool = False


def main():
    """Main function."""
    args = tyro.cli(Args)

    # Print device info
    device_info = get_device_info()
    print("Available RealSense Devices:")
    for serial, firmware in device_info.items():
        print(f"  {serial}: {firmware}")

    try:
        camera = RealSenseCamera(
            device_id=args.device_id,
            resolution=args.resolution,
            fps=args.fps,
            auto_exposure=args.auto_exposure,
            brightness=args.brightness,
            exposure_value=args.exposure,
            enable_depth=args.enable_depth,
        )
        visualize_camera(camera)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            camera.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
