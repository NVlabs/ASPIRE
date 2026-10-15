"""Real bimanual YAM embodiment for the CaP harness.

Layering, mirroring how a simulator adapter sits above its native env::

    CapApi                    embodiment-agnostic
      -> YamRealAdapter       adapter.py  -- normalize, delegate, never actuate
        -> RealYamEnv         env.py      -- owns the actuation chokepoint
          -> level-1 controller, arm servers, CAN, motor firmware

Imports here are lazy: the adapter pulls in mujoco, mink and portal, and the
harness must stay importable on a machine with none of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cap_harness.yam_real.config import (
    YamStationConfig,
    load_yam_station_config,
    station_config_path,
)
from cap_harness.yam_real.registry import (
    YAM_REAL_TASKS,
    YamRealRegistryError,
    YamRealTaskMetadata,
    YamRealTaskRegistry,
)

if TYPE_CHECKING:
    from cap_harness.yam_real.adapter import YamRealAdapter
    from cap_harness.yam_real.env import RealYamEnv
    from cap_harness.yam_real.station import build_real_station, build_sim_station

_LAZY = {
    "RealYamEnv": ("cap_harness.yam_real.env", "RealYamEnv"),
    "YamRealAdapter": ("cap_harness.yam_real.adapter", "YamRealAdapter"),
    "build_real_station": ("cap_harness.yam_real.station", "build_real_station"),
    "build_sim_station": ("cap_harness.yam_real.station", "build_sim_station"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    import importlib

    return getattr(importlib.import_module(module_name), attribute)


__all__ = [
    "YAM_REAL_TASKS",
    "RealYamEnv",
    "YamRealAdapter",
    "YamRealRegistryError",
    "YamRealTaskMetadata",
    "YamRealTaskRegistry",
    "YamStationConfig",
    "build_real_station",
    "build_sim_station",
    "load_yam_station_config",
    "station_config_path",
]
