# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re

import numpy as np


def resolve_serial(alias: str) -> str:
    """Resolve /dev/video_* alias to a RealSense serial number."""
    if not re.match(r"^(/dev/)?video", alias):
        return alias
    import pyrealsense2 as rs

    dev_path = alias if alias.startswith("/") else f"/dev/{alias}"
    dev_name = os.path.basename(os.path.realpath(dev_path))

    # Resolve the USB interface for this video node so we can match any
    # video node on the same physical device (D405 exposes several per camera).
    usb_iface = None
    try:
        usb_iface = os.path.realpath(f"/sys/class/video4linux/{dev_name}/device")
    except OSError:
        pass

    for d in rs.context().query_devices():
        port = d.get_info(rs.camera_info.physical_port).rstrip("/")
        if port.endswith(dev_name):
            return d.get_info(rs.camera_info.serial_number)
        if usb_iface and port.split("/video4linux/")[0] == usb_iface:
            return d.get_info(rs.camera_info.serial_number)

    raise RuntimeError(f"No RealSense device found for {alias} ({dev_name})")


class RealSenseCamera:
    # Fallback resolution chain used only when no explicit resolution is given.
    _PROFILES = [(640, 480, 30), (848, 480, 15), (1280, 720, 15)]
    # FPS values tried (in order) when a resolution is pinned.
    _FPS_FALLBACKS = (30, 15, 6)

    def __init__(
        self,
        serial: str | None = None,
        fps: int = 15,
        resolution: tuple[int, int] | None = None,
    ):
        import pyrealsense2 as rs

        self._rs = rs
        self.serial = str(serial) if serial is not None else None
        self._fps_pref = fps
        self._resolution = tuple(resolution) if resolution is not None else None
        self._pipeline = None
        self._intr = None
        self.width = 0
        self.height = 0
        self._last_frames = None

    def _candidate_profiles(self):
        """Profiles to try, in order. A pinned resolution wins; otherwise the
        default chain (640x480 first).
        """
        if self._resolution is None:
            return list(self._PROFILES)
        w, h = self._resolution
        fps_order = [self._fps_pref, *self._FPS_FALLBACKS]
        seen = set()
        return [(w, h, f) for f in fps_order if not (f in seen or seen.add(f))]

    def open(self) -> None:
        rs = self._rs
        last_err = None
        for w, h, fps in self._candidate_profiles():
            pipeline = rs.pipeline()
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            try:
                profile = pipeline.start(cfg)
            except Exception as e:
                last_err = e
                continue
            ok = False
            for _ in range(5):
                try:
                    pipeline.wait_for_frames(2000)
                    ok = True
                    break
                except Exception as e:
                    last_err = e
            if not ok:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                continue
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._intr = color_profile.get_intrinsics()
            self.width, self.height = w, h
            self._pipeline = pipeline
            if self.serial is None:
                self.serial = profile.get_device().get_info(rs.camera_info.serial_number)
            print(f"  RealSense {self.serial}: {w}x{h} @ {fps}fps")
            return
        raise RuntimeError(f"RealSense open failed. Last error: {last_err}")

    def close(self) -> None:
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def grab(self) -> bool:
        try:
            self._last_frames = self._pipeline.wait_for_frames(1000)
            return True
        except Exception:
            return False

    def get_image(self) -> np.ndarray:
        frame = self._last_frames.get_color_frame()
        return np.ascontiguousarray(np.asanyarray(frame.get_data()))

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        intr = self._intr
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        coeffs = np.array(intr.coeffs, dtype=np.float64).ravel()
        dist[: min(5, len(coeffs))] = coeffs[: min(5, len(coeffs))]
        return K, dist, (self.width, self.height)
