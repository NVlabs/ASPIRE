"""One existing YAM adapter with serialized actuation and live camera metadata."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from cap_harness.yam_real.control.runtime import RuntimeControl

_CONTROL_METHODS = frozenset(
    {"step", "execute_trajectory", "set_gripper", "set_grippers", "go_home", "reset"}
)


class LiveYamAdapter:
    """Delegate reads; route every CAP motion to the runtime's owner."""

    def __init__(self, adapter: Any, control: RuntimeControl) -> None:
        self._adapter = adapter
        self._control = control
        self._monitor_lock = threading.Lock()
        self._identity = None
        self._sequence = 0

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._adapter, name)
        if name in _CONTROL_METHODS:
            return lambda *args, **kwargs: self._control.submit_cap(lambda: value(*args, **kwargs))
        return value

    def monitor_snapshot(self) -> dict[str, Any]:
        """Keep acquisition age even when polling the same cached camera frame."""
        with self._monitor_lock:
            now = time.monotonic()
            wall = time.time()
            env = self._adapter.native_env
            cameras, metadata = {}, {}
            identity = []
            timestamps = []
            for name in env.camera_aliases:
                frame = env.camera_frame(name)
                if frame is None:
                    metadata[name] = {"missing": True, "stale": True}
                    identity.append((name, None))
                    continue
                timestamp = float(frame.timestamp_s)
                age_ms = max(0, now - timestamp) * 1000
                cameras[name] = np.asarray(frame.rgb, dtype=np.uint8).copy()
                identity.append((name, timestamp))
                timestamps.append(timestamp)
                metadata[name] = {
                    "missing": False,
                    "stale": age_ms > 1000,
                    "age_ms": age_ms,
                    "stale_after_ms": 1000,
                    "sequence": timestamp,
                    "timestamp_s": wall - age_ms / 1000,
                }
            if tuple(identity) != self._identity:
                self._sequence += 1
                self._identity = tuple(identity)
            state = self._adapter.get_robot_state()
            return {
                "timestamp_s": None if not timestamps else wall - max(0, now - min(timestamps)),
                "refresh_timestamp_s": wall,
                "observation_seq": self._sequence,
                "cameras": cameras,
                "camera_metadata": metadata,
                "state": {
                    **{f"{arm}_joint_pos": value for arm, value in state.joint_positions.items()},
                    **{
                        f"{arm}_gripper_pos": np.array([value])
                        for arm, value in state.gripper_positions.items()
                    },
                },
            }

    def close(self) -> None:
        self._control.close()
        self._adapter.close()
