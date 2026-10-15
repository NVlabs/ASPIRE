"""Run standalone calibration with python -m cap_harness.yam_real.calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _board_options(parser):
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument(
        "--square-length", type=float, default=0.04, help="Measured square side in metres"
    )
    parser.add_argument(
        "--marker-length", type=float, default=0.03, help="Measured marker side in metres"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="Generate a printable ChArUco PNG")
    _board_options(board)
    board.add_argument("--output", type=Path, required=True)
    board.add_argument("--dpi", type=int, default=300)
    capture = commands.add_parser(
        "calibrate", help="Guide an arm, capture a sweep, solve, and export XML"
    )
    _board_options(capture)
    capture.add_argument("--serial", required=True, help="RealSense serial or /dev/video_* alias")
    capture.add_argument(
        "--camera-name", required=True, help="Your camera role, such as top or wrist_left"
    )
    capture.add_argument("--mode", choices=["fixed", "wrist"], required=True)
    capture.add_argument("--arm", choices=["left", "right"], required=True)
    capture.add_argument("--host", default="127.0.0.1")
    capture.add_argument("--port", type=int)
    capture.add_argument(
        "--model", type=Path, help="Station MuJoCo XML; defaults to the packaged model"
    )
    capture.add_argument(
        "--camera-body", help="XML body to update; defaults to the camera's standard body"
    )
    capture.add_argument(
        "--resolution", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT")
    )
    capture.add_argument(
        "--poses", type=Path, help="Optional JSON array of absolute six-joint targets"
    )
    capture.add_argument("--speed", type=float, default=0.2, help="Joint speed limit, rad/s")
    capture.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    capture.add_argument("--confirm-motion", action="store_true")
    solve = commands.add_parser("solve", help="Re-solve saved observations without hardware")
    solve.add_argument("samples", type=Path)
    solve.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "board":
        from .core import Board

        spec = Board(args.squares_x, args.squares_y, args.square_length, args.marker_length)
        spec.write_png(args.output, dpi=args.dpi)
        print(
            f"Print at actual size: pattern {args.squares_x * args.square_length * 1000:g} x "
            f"{args.squares_y * args.square_length * 1000:g} mm, plus white margin. Measure a square."
        )
    else:
        from .calibrator import MODEL, calibrate, solve_dataset

        if args.command == "calibrate":
            args.model = args.model or MODEL
            result = calibrate(args)
        else:
            result = solve_dataset(json.loads(args.samples.read_text()), args.output_dir)
        print(json.dumps(result["solution"], indent=2))
        print(f"Calibration JSON and station_calibrated.xml: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Calibration stopped. Saved samples remain on disk.")
        raise SystemExit(130) from None
