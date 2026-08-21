# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl


def _center_crop(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    src_aspect = float(src_w) / float(src_h)
    dst_aspect = float(dst_w) / float(dst_h)

    if abs(src_aspect - dst_aspect) < 1e-9:
        return 0, 0, src_w, src_h

    if src_aspect > dst_aspect:
        crop_h = src_h
        crop_w = int(round(crop_h * dst_aspect))
    else:
        crop_w = src_w
        crop_h = int(round(crop_w / dst_aspect))

    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)
    x0 = max(0, (src_w - crop_w) // 2)
    y0 = max(0, (src_h - crop_h) // 2)
    return x0, y0, crop_w, crop_h


@dataclass
class CameraData:
    images: Dict[str, Optional[np.ndarray]]
    timestamp: float
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[Dict[str, float]] = None


@dataclass
class ZedCamera:
    """Standalone ZED RGB-D camera driver with a RealSense-like interface."""

    device_id: Optional[int] = None
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 60
    auto_exposure: bool = True  # compatibility placeholder
    brightness: int = 10  # compatibility placeholder
    exposure_value: Optional[int] = None  # compatibility placeholder
    enable_depth: bool = True
    align_depth_to_color: bool = True  # compatibility placeholder
    native_resolution: str = "HD720"
    depth_mode: str = "NEURAL"
    confidence_threshold: int = 50
    texture_confidence_threshold: int = 100
    camera_type: str = field(default="Stereolabs ZED 2i", init=False)

    def __post_init__(self) -> None:
        self._camera = sl.Camera()
        self._runtime = sl.RuntimeParameters()
        self._image = sl.Mat()
        self._depth = sl.Mat()

        init = sl.InitParameters()
        if self.device_id is not None:
            init.set_from_serial_number(int(self.device_id))
        try:
            init.camera_resolution = getattr(sl.RESOLUTION, str(self.native_resolution).upper())
        except AttributeError as exc:
            raise ValueError(f"Unsupported ZED resolution: {self.native_resolution!r}") from exc
        init.camera_fps = int(self.fps)
        init.coordinate_units = sl.UNIT.METER
        if self.enable_depth:
            try:
                init.depth_mode = getattr(sl.DEPTH_MODE, str(self.depth_mode).upper())
            except AttributeError as exc:
                raise ValueError(f"Unsupported ZED depth mode: {self.depth_mode!r}") from exc
        else:
            init.depth_mode = sl.DEPTH_MODE.NONE

        status = self._camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        self._runtime.confidence_threshold = int(self.confidence_threshold)
        self._runtime.texture_confidence_threshold = int(self.texture_confidence_threshold)

        info = self._camera.get_camera_information()
        cfg = info.camera_configuration
        calib = cfg.calibration_parameters.left_cam
        native_w = int(cfg.resolution.width)
        native_h = int(cfg.resolution.height)
        self._native_resolution = (native_w, native_h)
        self._crop = _center_crop(native_w, native_h, self.resolution[0], self.resolution[1])
        self._intrinsics = self._build_intrinsics(
            fx=float(calib.fx),
            fy=float(calib.fy),
            cx=float(calib.cx),
            cy=float(calib.cy),
            native_w=native_w,
            native_h=native_h,
        )

    def _build_intrinsics(
        self,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        native_w: int,
        native_h: int,
    ) -> Dict[str, float]:
        crop_x, crop_y, crop_w, crop_h = self._crop
        out_w, out_h = self.resolution

        fx = fx * (float(out_w) / float(crop_w))
        fy = fy * (float(out_h) / float(crop_h))
        cx = (cx - float(crop_x)) * (float(out_w) / float(crop_w))
        cy = (cy - float(crop_y)) * (float(out_h) / float(crop_h))

        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": float(out_w),
            "height": float(out_h),
            "native_width": float(native_w),
            "native_height": float(native_h),
        }

    def _crop_and_resize_rgb(self, image: np.ndarray) -> np.ndarray:
        x0, y0, w, h = self._crop
        if (x0, y0, w, h) != (0, 0, image.shape[1], image.shape[0]):
            image = image[y0 : y0 + h, x0 : x0 + w]
        if (image.shape[1], image.shape[0]) != self.resolution:
            image = cv2.resize(image, self.resolution, interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(image)

    def _crop_and_resize_depth(self, depth: np.ndarray) -> np.ndarray:
        x0, y0, w, h = self._crop
        if (x0, y0, w, h) != (0, 0, depth.shape[1], depth.shape[0]):
            depth = depth[y0 : y0 + h, x0 : x0 + w]
        if (depth.shape[1], depth.shape[0]) != self.resolution:
            depth = cv2.resize(depth, self.resolution, interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(depth.astype(np.float32))

    def read(self) -> CameraData:
        status = self._camera.grab(self._runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED grab failed: {status}")

        self._camera.retrieve_image(self._image, sl.VIEW.LEFT)
        raw = np.asarray(self._image.get_data())
        if raw.ndim != 3:
            raise RuntimeError(f"Unexpected ZED image shape: {raw.shape}")
        if raw.shape[2] == 4:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        elif raw.shape[2] == 3:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Unexpected ZED channel count: {raw.shape}")
        rgb = self._crop_and_resize_rgb(rgb)

        depth = None
        if self.enable_depth:
            self._camera.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            raw_depth = np.asarray(self._depth.get_data()).copy()
            raw_depth = np.where(np.isfinite(raw_depth), raw_depth, 0.0).astype(np.float32)
            depth = self._crop_and_resize_depth(raw_depth)

        timestamp_ms = self._camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() / 1_000_000.0
        if timestamp_ms <= 0:
            timestamp_ms = time.time() * 1000.0

        return CameraData(
            images={"rgb": rgb},
            timestamp=timestamp_ms,
            depth=depth,
            intrinsics=self._intrinsics.copy(),
        )

    def get_intrinsics(self) -> Dict[str, float]:
        return self._intrinsics.copy()

    def stop(self) -> None:
        self._camera.close()
