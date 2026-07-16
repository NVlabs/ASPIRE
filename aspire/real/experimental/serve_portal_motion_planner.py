from __future__ import annotations

import argparse
import logging

from experimental.portal_motion_planner import (
    PortalMotionPlannerConfig,
    PortalMotionPlannerServer,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the Portal cuRobo motion planner."
    )
    parser.add_argument("--backend", default="curobo")
    parser.add_argument("--solver-speed", default="fast", choices=("fast", "slow"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--position-threshold", type=float, default=0.005)
    parser.add_argument("--rotation-threshold", type=float, default=0.05)
    parser.add_argument(
        "--enable-depth-collision",
        action="store_true",
        help="Enable depth-scene collision updates for the remote planner service.",
    )
    parser.add_argument(
        "--robot-type",
        default="yam",
        choices=("yam", "panda"),
        help="Robot type: yam (6-DOF bimanual) or panda (7-DOF single arm).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    config = PortalMotionPlannerConfig(
        backend=str(args.backend).strip().lower(),
        solver_speed=str(args.solver_speed).strip().lower(),
        port=int(args.port),
        position_threshold=float(args.position_threshold),
        rotation_threshold=float(args.rotation_threshold),
        enable_depth_collision=bool(args.enable_depth_collision),
        robot_type=str(args.robot_type).strip().lower(),
    )
    PortalMotionPlannerServer(config).serve()


if __name__ == "__main__":
    main()
