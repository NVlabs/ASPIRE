"""Station model path resolution based on top camera backend.

Usage::

    from robot.models.station.paths import get_station_xml, get_station_urdf
"""

from __future__ import annotations

import os
from pathlib import Path

_STATION_DIR = Path(__file__).resolve().parent

_XML_MAP = {
    "realsense": "station.xml",
    "zed": "station_zed2itop.xml",
}

_URDF_MAP = {
    "realsense": "station.urdf",
    "zed": "station_zed2itop.urdf",
}

_CAMERA_FRAME_MAP = {
    "realsense": "top_camera_d405",
    "zed": "top_camera_zed2i",
}


def _backend_from_frame(frame_name: str) -> str | None:
    frame = frame_name.strip().lower()
    for backend, known_frame in _CAMERA_FRAME_MAP.items():
        if frame == known_frame:
            return backend
    return None


def _backend_from_calibrated_xml() -> str | None:
    calibrated_xml = os.environ.get("YAM_STATION_CALIBRATED_XML", "").strip()
    if not calibrated_xml:
        return None

    path = Path(calibrated_xml).expanduser()
    try:
        text = path.read_text(errors="ignore") if path.is_file() else ""
    except OSError:
        text = ""

    for backend, frame in _CAMERA_FRAME_MAP.items():
        if frame in text:
            return backend

    # Current YAM calibration exports a D405 top-camera body. If the path is
    # set but not readable in this process, prefer the calibrated station over
    # the legacy ZED fallback that cannot exist in that XML.
    return "realsense"


def _get_top_camera_backend() -> str:
    """Return normalized top camera backend (``'realsense'`` or ``'zed'``)."""
    frame_backend = _backend_from_frame(os.environ.get("CAP_TOP_CAMERA_FRAME", ""))
    if frame_backend is not None:
        return frame_backend

    raw = os.environ.get("CAP_TOP_CAMERA_BACKEND", "").strip().lower()
    if raw in _XML_MAP:
        return raw

    mapping_str = os.environ.get("CAP_CAMERA_BACKENDS", "").strip()
    if mapping_str:
        for item in mapping_str.split(","):
            if "=" in item:
                name, value = item.split("=", 1)
                if name.strip().lower() == "top":
                    v = value.strip().lower()
                    if v in _XML_MAP:
                        return v

    calibrated_backend = _backend_from_calibrated_xml()
    if calibrated_backend is not None:
        return calibrated_backend

    return "realsense"


def get_station_xml() -> Path:
    """Return the station MuJoCo XML path for the current top camera backend."""
    calibrated_xml = os.environ.get("YAM_STATION_CALIBRATED_XML", "").strip()
    if calibrated_xml:
        return Path(calibrated_xml).expanduser()
    return _STATION_DIR / _XML_MAP[_get_top_camera_backend()]


def get_station_urdf() -> Path:
    """Return the station URDF path for the current top camera backend."""
    return _STATION_DIR / _URDF_MAP[_get_top_camera_backend()]


def get_top_camera_frame() -> str:
    """Return the top camera frame/body name for the current backend."""
    raw = os.environ.get("CAP_TOP_CAMERA_FRAME", "").strip()
    if raw:
        return raw
    return _CAMERA_FRAME_MAP[_get_top_camera_backend()]


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def needs_optical_flip(camera: str) -> bool:
    """Whether the pinocchio frame for *camera* needs an OpenCV ↔ Pinocchio flip.

    The D405 URDF body frame uses Pinocchio convention (+X left, +Y up,
    +Z forward).  Converting to OpenCV (+X right, +Y down, +Z forward)
    requires ``R @ diag(-1, -1, 1)``.

    The ZED 2i URDF frame was calibrated directly in OpenCV convention,
    so no flip is needed.

    Wrist cameras (left/right) are always D405 → always need the flip.
    """
    camera = str(camera).strip().lower()
    specific_override = _env_bool(f"CAP_{camera.upper()}_CAMERA_NEEDS_OPTICAL_FLIP")
    if specific_override is not None:
        return specific_override
    global_override = _env_bool("CAP_CAMERA_NEEDS_OPTICAL_FLIP")
    if global_override is not None:
        return global_override

    if camera == "top":
        # The calibrated YAM top-camera XML stores the D405 frame in the
        # image/OpenCV convention already. Applying another optical flip mirrors
        # X/Y, which makes detections appear on the wrong side of the table.
        if os.environ.get("YAM_STATION_CALIBRATED_XML", "").strip():
            return False
        return _get_top_camera_backend() != "zed"
    return True
