"""How a YAM plant is built, and what the adapter is allowed to assume about it.

The adapter talks to a *station*, not to hardware. Two implementations satisfy
the same port: portal clients to the real arm servers, and the MuJoCo station in
:mod:`cap_harness.yam_real.sim`. Both are wrapped by the same
:class:`~cap_harness.yam_real.env.RealYamEnv`, so the code under test off-robot
is the code that runs on the bench -- only the bottom of the stack is swapped.

Declaring the port as a Protocol rather than importing hardware here keeps this
package importable on a machine with no CAN bus and no camera SDK, which is what
lets the test suite run in CI.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from cap_harness.yam_real.config import YamStationConfig, load_yam_station_config


@runtime_checkable
class YamStation(Protocol):
    """What the adapter needs from a YAM plant, real or simulated."""

    config: YamStationConfig

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """Per-arm ``joint_pos``, ``gripper_pos``, ``ee_pos``, ``ee_quat``."""

    def execute_action_batch(self, batch: Any, **kwargs: Any) -> dict[str, Any]:
        """The single actuation chokepoint."""

    def set_gripper(self, side: str, position: float, **kwargs: Any) -> dict[str, Any]: ...

    def go_home(self, **kwargs: Any) -> dict[str, Any]: ...

    def read_camera(self) -> Any | None: ...

    def control_contract(self) -> dict[str, Any]:
        """Versioned plant description stamped into recorded episodes."""

    def close(self) -> None: ...


def build_real_station(
    station: str = "yam-example",
    *,
    config_root: str | None = None,
    enable_camera: bool = True,
    owner: str = "cap-harness",
) -> YamStation:
    """Connect to a physical station's arm servers and camera.

    Construction opens client connections and reads; it issues no command. The
    hardware stack is imported lazily so this module stays importable without it.

    Every camera the profile declares must open. ``DASHBOARD_PORT_ENV`` starts a
    live view of them; see :mod:`cap_harness.yam_real.dashboard`.
    """
    from cap_harness.yam_real.env import ARMS, RealYamEnv
    from cap_harness.yam_real.transport import FollowerArmClient

    config = load_yam_station_config(station, config_root=config_root)
    arms = {
        side: FollowerArmClient(config.arms[side].host, config.arms[side].port, label=side)
        for side in ARMS
    }
    marker, cameras = (
        open_station_cameras(config, station=station, owner=owner) if enable_camera else (None, {})
    )
    camera = cameras.pop(config.camera.role, None)
    env = RealYamEnv(config, arms, camera=camera, aux_cameras=cameras, camera_marker=marker)
    return env


def open_station_cameras(
    config: YamStationConfig, *, station: str, owner: str
) -> tuple[Any, dict[str, Any]]:
    """Claim `station`'s cameras and open every one its profile declares.

    Returns the ownership marker and the opened devices by role. Either all of
    them open or none do: half a station is worse than none, because two live
    pipelines lock those devices against the next attempt while the marker still
    claims all four.

    Separate from :func:`build_real_station` for ``scripts/yam_dashboard.py``,
    which hands the cameras back and forth while keeping one set of arm clients.
    """
    from cap_harness.yam_real.station_marker import claim

    marker = claim(station, owner=owner)
    specs = {config.camera.role: config.camera} | dict(config.aux_cameras)
    opened: dict[str, Any] = {}
    try:
        for role, spec in specs.items():
            opened[role] = _open_camera(spec, station=station, role=role)
    except BaseException:
        for source in opened.values():
            try:
                source.close()
            except Exception:
                pass
        if marker is not None:
            marker.release()
        raise
    return marker, opened


#: How long to keep retrying a camera that will not open. A courtesy viewer
#: releases the moment it sees a run's marker, but its own open may already be in
#: flight and closing four RealSense pipelines is not instant. Retrying the
#: *device* rather than waiting on the marker keeps a run's behaviour a function
#: of the hardware alone -- the marker stays advisory, and a run that has no
#: viewer to wait for is not delayed by one.
CAMERA_OPEN_RETRY_S = 6.0
CAMERA_OPEN_RETRY_DELAY_S = 0.75


def _open_camera(spec: Any, *, station: str, role: str) -> Any:
    """Open one of the station's cameras, or raise saying who is likely to have it.

    Every camera the profile declares is required. An aux camera used to be
    reported and skipped, which was defensible while these were a convenience and
    became wrong once they were recorded: a skipped camera vanishes from
    ``camera_aliases``, from the Observation, and from the episode's video set,
    so afterwards a three-camera episode is indistinguishable from a four-camera
    one. A dataset quietly missing a view is worse than a station that will not
    start.
    """
    import time

    from cap_harness.yam_real.cameras import open_station_camera

    deadline = time.monotonic() + CAMERA_OPEN_RETRY_S
    while True:
        try:
            return open_station_camera(spec)
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"station {station}: camera {role} (serial {spec.serial}) will not open "
                    f"after {CAMERA_OPEN_RETRY_S:.0f}s. A RealSense device belongs to one "
                    "process -- look for another run, a dashboard, or a stray script. "
                    f"Ownership marker: {_marker_hint(station)}."
                ) from exc
            time.sleep(CAMERA_OPEN_RETRY_DELAY_S)


def _marker_hint(station: str) -> str:
    """What the ownership marker says, phrased for an error message."""
    from cap_harness.yam_real.station_marker import read_marker

    held = read_marker(station)
    if held is None:
        return "nothing has claimed this station's cameras"
    return f"pid {held.pid} ({held.owner}) claims them"


def build_sim_station(
    station: str = "yam-example",
    *,
    config_root: str | None = None,
    home: np.ndarray | None = None,
    gravity_comp_factor: float = 1.0,
    realtime: bool = True,
    with_camera: bool = True,
) -> YamStation:
    """Build the MuJoCo station behind the same env, for runs with no robot.

    ``realtime`` steps physics on a background thread so the level-1 loop's
    wall-clock resampling is exercised honestly. Turn it off for a deterministic
    unit test that steps the plant itself.
    """
    from cap_harness.yam_real.cameras import SyntheticCamera
    from cap_harness.yam_real.env import ARMS, RealYamEnv
    from cap_harness.yam_real.sim import SimYamArm, SimYamStation

    config = load_yam_station_config(station, config_root=config_root)
    plant = SimYamStation(config.model_xml, gravity_comp_factor=gravity_comp_factor)
    start = {
        side: (
            config.arms[side].home_joints
            if home is None
            else np.asarray(home, dtype=np.float64).reshape(6)
        )
        for side in ARMS
    }
    plant.set_joint_positions(start["left"], start["right"])
    arms = {
        side: SimYamArm(
            plant,
            side,
            default_kp=config.controller.kp,
            default_kd=config.controller.kd,
        )
        for side in ARMS
    }
    if realtime:
        plant.start()
    # The profile's aux cameras get synthetic stand-ins too. The sim station
    # exists to be a mirror of the bench, and a one-camera mirror of a
    # four-camera station is a poor one: anything that iterates `camera_aliases`
    # -- the recorder's video set, the dashboard's tiles -- would go untested
    # off-robot precisely where it is most awkward to test on.
    aux_cameras = {role: SyntheticCamera() for role in config.aux_cameras} if with_camera else {}
    return RealYamEnv(
        config,
        arms,
        camera=SyntheticCamera() if with_camera else None,
        aux_cameras=aux_cameras,
        plant=plant,
    )


__all__ = ["YamStation", "build_real_station", "build_sim_station"]
