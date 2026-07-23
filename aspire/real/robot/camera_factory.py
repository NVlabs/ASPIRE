# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any


_BACKEND_ALIASES = {
    "realsense": "realsense",
    "rs": "realsense",
    "d405": "realsense",
    "zed": "zed",
    "zed2i": "zed",
    "stereolabs": "zed",
}


def _env_first(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def normalize_camera_backend(value: str) -> str:
    backend = _BACKEND_ALIASES.get(str(value).strip().lower())
    if backend is None:
        supported = ", ".join(sorted(set(_BACKEND_ALIASES.values())))
        raise ValueError(f"Unsupported camera backend {value!r}. Supported backends: {supported}")
    return backend


def _parse_backend_map(spec: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, backend = item.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid CAP_CAMERA_BACKENDS entry {item!r}; expected entries like 'top=zed,left=realsense'"
            )
        mapping[name.strip().lower()] = normalize_camera_backend(backend)
    return mapping


def _default_backend_for_camera(camera_name: str) -> str:
    return "zed" if camera_name == "top" else "realsense"


def get_camera_backend(camera_name: str, default: str | None = None) -> str:
    camera_name = str(camera_name).strip().lower()

    explicit = _env_first(f"CAP_{camera_name.upper()}_CAMERA_BACKEND")
    if explicit is not None:
        return normalize_camera_backend(explicit)

    mapping_spec = _env_first("CAP_CAMERA_BACKENDS")
    if mapping_spec is not None:
        mapping = _parse_backend_map(mapping_spec)
        if camera_name in mapping:
            return mapping[camera_name]

    station_cfg = _get_station_camera_config(camera_name)
    if station_cfg is not None:
        return normalize_camera_backend(station_cfg.type)

    return normalize_camera_backend(default or _default_backend_for_camera(camera_name))


def _get_station_camera_config(camera_name: str):
    try:
        from robot.station_profiles import active_station_cameras

        for cfg in active_station_cameras().cameras:
            if cfg.name == camera_name:
                return cfg
    except Exception:
        return None
    return None


def parse_resolution(value: str) -> tuple[int, int]:
    raw = value.strip().lower().replace(" ", "")
    for sep in ("x", ","):
        if sep in raw:
            w_str, h_str = raw.split(sep, 1)
            width = int(w_str)
            height = int(h_str)
            if width <= 0 or height <= 0:
                break
            return width, height
    raise ValueError(f"Invalid resolution {value!r}; expected formats like '640x480' or '640,480'")


def get_camera_resolution(camera_name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = _env_first(
        f"CAP_{camera_name.upper()}_CAMERA_RESOLUTION",
        "CAP_CAMERA_RESOLUTION",
    )
    if value is None:
        return default
    return parse_resolution(value)


def get_camera_fps(camera_name: str, default: int) -> int:
    value = _env_first(
        f"CAP_{camera_name.upper()}_CAMERA_FPS",
        "CAP_CAMERA_FPS",
    )
    return int(value) if value is not None else int(default)


def resolve_realsense_serial(camera_name: str) -> str:
    explicit = _env_first(
        f"CAP_{camera_name.upper()}_REALSENSE_SERIAL",
        "CAP_REALSENSE_SERIAL",
    )
    if explicit is not None:
        return explicit

    station_cfg = _get_station_camera_config(camera_name)
    if station_cfg is not None and station_cfg.device_id:
        return str(station_cfg.device_id)

    import pyrealsense2 as rs

    symlink = (
        station_cfg.symlink
        if station_cfg is not None and station_cfg.symlink
        else f"/dev/video_{camera_name}"
    )
    if not os.path.exists(symlink):
        raise FileNotFoundError(f"No symlink: {symlink}")

    video = os.path.basename(os.path.realpath(symlink))
    usb_id = os.path.basename(os.path.realpath(f"/sys/class/video4linux/{video}/device"))

    ctx = rs.context()
    for dev in ctx.query_devices():
        if usb_id in dev.get_info(rs.camera_info.physical_port):
            return dev.get_info(rs.camera_info.serial_number)

    raise ValueError(f"No RealSense for symlink: {symlink}")


def resolve_zed_serial(camera_name: str) -> int:
    explicit = _env_first(
        f"CAP_{camera_name.upper()}_ZED_SERIAL",
        "CAP_ZED_SERIAL",
    )
    if explicit is not None:
        return int(explicit)

    station_cfg = _get_station_camera_config(camera_name)
    if station_cfg is not None and station_cfg.device_id:
        return int(station_cfg.device_id)

    import pyzed.sl as sl

    devices = list(sl.Camera.get_device_list())
    if len(devices) == 1:
        return int(devices[0].serial_number)

    serials = [int(dev.serial_number) for dev in devices]
    raise RuntimeError(
        f"Could not resolve ZED serial for camera {camera_name!r}. "
        f"Set CAP_{camera_name.upper()}_ZED_SERIAL. Connected ZED serials: {serials}"
    )


def _get_zed_native_resolution(camera_name: str, default: str = "HD720") -> str:
    return str(
        _env_first(
            f"CAP_{camera_name.upper()}_ZED_NATIVE_RESOLUTION",
            "CAP_ZED_NATIVE_RESOLUTION",
        )
        or default
    ).upper()


def _get_zed_depth_mode(camera_name: str, default: str = "NEURAL") -> str:
    return str(
        _env_first(
            f"CAP_{camera_name.upper()}_ZED_DEPTH_MODE",
            "CAP_ZED_DEPTH_MODE",
        )
        or default
    ).upper()


def create_camera(
    camera_name: str,
    *,
    resolution: tuple[int, int] = (640, 480),
    fps: int = 60,
    enable_depth: bool = True,
) -> Any:
    camera_name = str(camera_name).strip().lower()
    resolution = get_camera_resolution(camera_name, resolution)
    fps = get_camera_fps(camera_name, fps)
    backend = get_camera_backend(camera_name)

    if backend == "realsense":
        from robot.realsense import RealSenseCamera

        return RealSenseCamera(
            device_id=resolve_realsense_serial(camera_name),
            resolution=resolution,
            fps=fps,
            auto_exposure=True,
            brightness=10,
            enable_depth=enable_depth,
        )

    if backend == "zed":
        from robot.zed import ZedCamera

        return ZedCamera(
            device_id=resolve_zed_serial(camera_name),
            resolution=resolution,
            fps=fps,
            enable_depth=enable_depth,
            native_resolution=_get_zed_native_resolution(camera_name),
            depth_mode=_get_zed_depth_mode(camera_name),
        )

    raise AssertionError(f"Unhandled camera backend: {backend}")


def get_camera_type_name(camera: Any) -> str:
    return str(getattr(camera, "camera_type", camera.__class__.__name__))
