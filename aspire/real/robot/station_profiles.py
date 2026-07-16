"""Named per-machine CAN (and related) presets for shared checkouts.

**Naming:** Every committed profile key MUST be an MCU *Avengers*-saga character
slug (lowercase), e.g. ``thanos``, ``thor``, ``shuri``, ``strange``. No generic
codenames — keeps benches recognizable in logs and env vars.

Resolution order (first hit wins):
  1. Environment variable ``ASPIRE_STATION`` (profile key).
  2. Gitignored ``robot/local_station.toml`` with ``station = "thor"``.
  3. ``STATION_BY_HOSTNAME``: ``socket.gethostname().lower()`` → profile key.
  4. Profile ``default`` — matches historical ``robot/constants.py`` values.

Add a new Ubuntu box by extending ``PROFILES`` with an Avengers slug and the
Linux/macOS CAN strings for that bench. On each machine set ``ASPIRE_STATION``,
``local_station.toml``, or a hostname row in ``STATION_BY_HOSTNAME``.

Camera configuration
--------------------
Each profile also specifies a ``StationCameras`` describing which cameras
are attached, their type (``"realsense"`` or ``"zed"``), and either a stable
device serial or a udev symlink used to locate the device. See
``active_station_cameras()``.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

# ---------------------------------------------------------------------------
# Profile table: name -> (linux StationCAN, darwin StationCAN)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationCAN:
    left_follower: str
    right_follower: str
    left_leader: str
    right_leader: str


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraConfig:
    """One camera attached to a station."""

    name: str  # semantic name: "top", "left", "right", …
    type: Literal["realsense", "zed"]  # driver backend
    symlink: str | None = None  # udev device symlink, e.g. "/dev/video_top_zed2i"
    device_id: str | None = None  # stable camera serial when supported by the backend


@dataclass(frozen=True)
class StationCameras:
    """Ordered set of cameras for a station."""

    cameras: tuple[CameraConfig, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.cameras)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CAMERAS = StationCameras((
    CameraConfig("top", "realsense", "/dev/video_top"),
    CameraConfig("left", "realsense", "/dev/video_left"),
    CameraConfig("right", "realsense", "/dev/video_right"),
    CameraConfig("bottom", "realsense", "/dev/video_bottom"),
))

_THANOS_CAMERAS = StationCameras((
    CameraConfig("top", "zed", "/dev/video_top_zed2i"),
    CameraConfig("left", "realsense", "/dev/video_left"),
    CameraConfig("right", "realsense", "/dev/video_right"),
    CameraConfig("bottom", "realsense", "/dev/video_bottom"),
))


def _default_linux() -> StationCAN:
    return StationCAN("can_follow_l", "can_follow_r", "can_leader_l", "can_leader_r")


def _default_darwin() -> StationCAN:
    return StationCAN(
        os.environ.get("YAM_LEFT_FOLLOWER_CAN_INTERFACE", ""),
        os.environ.get("YAM_RIGHT_FOLLOWER_CAN_INTERFACE", ""),
        os.environ.get("YAM_LEFT_LEADER_CAN_INTERFACE", ""),
        os.environ.get("YAM_RIGHT_LEADER_CAN_INTERFACE", ""),
    )


PROFILES: dict[str, tuple[StationCAN, StationCAN]] = {
    "default": (_default_linux(), _default_darwin()),
    "thanos": (_default_linux(), _default_darwin()),
    "thor": (_default_linux(), _default_darwin()),
    "tony": (_default_linux(), _default_darwin()),
}

CAMERA_PROFILES: dict[str, StationCameras] = {
    "default": _DEFAULT_CAMERAS,
    "thanos": _THANOS_CAMERAS,
    "thor": _THANOS_CAMERAS,
    "tony": _THANOS_CAMERAS,
}

_CAMERA_NAMES_ENV_KEYS = (
    "OPENFORGE_REAL_YAM_CAMERAS",
    "CAP_REAL_YAM_CAMERAS",
    "CAP_CAMERA_NAMES",
)

# Add local hostname mappings in an untracked station configuration rather than
# committing workstation identities here.
STATION_BY_HOSTNAME: dict[str, str] = {}

_resolved_key: str | None = None


def _load_station_from_local_toml() -> str | None:
    path = Path(__file__).resolve().parent / "local_station.toml"
    if not path.is_file():
        return None
    try:
        import tomllib

        with path.open("rb") as f:
            data = tomllib.load(f)
    except OSError:
        return None
    name = data.get("station")
    if name is None:
        return None
    s = str(name).strip()
    return s or None


def resolve_station_key() -> str:
    """Active profile name (cached for process lifetime)."""
    global _resolved_key
    if _resolved_key is not None:
        return _resolved_key

    env = os.environ.get("ASPIRE_STATION", "").strip()
    if env:
        _resolved_key = env
        return _resolved_key

    from_file = _load_station_from_local_toml()
    if from_file:
        _resolved_key = from_file
        return _resolved_key

    host = socket.gethostname().lower()
    mapped = STATION_BY_HOSTNAME.get(host)
    if mapped:
        _resolved_key = mapped
        return _resolved_key

    _resolved_key = "default"
    return _resolved_key


def active_station_can() -> StationCAN:
    key = resolve_station_key()
    if key not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"Unknown station profile {key!r}. "
            f"Known profiles: {known}. "
            f"Set ASPIRE_STATION, add robot/local_station.toml, or extend robot/station_profiles.py."
        )
    linux_can, darwin_can = PROFILES[key]
    return darwin_can if sys.platform == "darwin" else linux_can


def active_station_cameras() -> StationCameras:
    """Camera layout for the active station profile.

    Falls back to ``_DEFAULT_CAMERAS`` when the resolved key has no entry in
    ``CAMERA_PROFILES``.
    """
    key = resolve_station_key()
    cameras = CAMERA_PROFILES.get(key, _DEFAULT_CAMERAS)
    names = _camera_names_from_env()
    if not names:
        return cameras

    by_name = {cfg.name: cfg for cfg in cameras.cameras}
    return StationCameras(tuple(_camera_config_for_name(name, by_name) for name in names))


def _camera_names_from_env() -> tuple[str, ...] | None:
    raw = ""
    for key in _CAMERA_NAMES_ENV_KEYS:
        raw = os.environ.get(key, "").strip()
        if raw:
            break
    if not raw:
        return None

    names: list[str] = []
    for item in raw.replace(";", ",").split(","):
        name = item.strip().lower()
        if name and name not in names:
            names.append(name)
    return tuple(names) if names else None


def _camera_config_for_name(
    name: str, profile_configs: dict[str, CameraConfig]
) -> CameraConfig:
    cfg = profile_configs.get(name)
    if cfg is not None:
        return cfg
    camera_type: Literal["realsense", "zed"] = "zed" if name == "top" else "realsense"
    return CameraConfig(name, camera_type, f"/dev/video_{name}")
