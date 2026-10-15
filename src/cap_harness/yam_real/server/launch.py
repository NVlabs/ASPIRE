"""`yam-servers`: start the follower arm servers described by a station profile.

Replaces the pre-migration launcher, which imported a `robot.*` package that does
not exist in this repository and shelled out to scripts by path. Everything here
comes from the profile, so starting a different bench is a `--station` change.

**Starting a server energizes motors.** Nothing else in the harness does that, so
it is deliberately a separate, explicit act rather than something a run performs
on your behalf. Stopping a server de-energizes them, and the arms drop under
gravity unless they are supported.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from cap_harness.yam_real.config import load_yam_station_config
from cap_harness.yam_real.server.arm_server import FollowerArmServer
from cap_harness.yam_real.server.motors import DaMiaoBus
from cap_harness.yam_real.server.yam_robot import YamRobot


def build_server(station: str, side: str, *, config_root: str | None = None) -> FollowerArmServer:
    """Construct one arm's server from the profile. Opens no bus yet."""
    config = load_yam_station_config(station, config_root=config_root)
    if side not in config.arms:
        raise ValueError(f"station {station!r} has no arm {side!r}")
    arm = config.arms[side]
    motors = config.motors
    bus = DaMiaoBus(
        arm.can_interface,
        motor_ids=(*motors.arm_motor_ids, motors.gripper_motor_id),
        motor_types=(*motors.arm_motor_types, motors.gripper_motor_type),
        bustype=motors.bustype,
    )
    return FollowerArmServer(YamRobot(config, side, bus), arm.port)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yam-servers",
        description="Start a YAM follower arm server (this energizes the motors)",
    )
    parser.add_argument("side", choices=("left", "right"))
    parser.add_argument("--station", default="yam-example")
    parser.add_argument("--config-root", default=None)
    parser.add_argument(
        "--calibrate-gripper",
        action="store_true",
        help="drive the gripper into both stops on startup to measure its travel",
    )
    args = parser.parse_args(argv)

    try:
        server = build_server(args.station, args.side, config_root=args.config_root)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    server.serve(calibrate_gripper=args.calibrate_gripper)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_server", "main"]
