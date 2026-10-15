# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ENPIRE's guided nominal-pose, capture, hand-eye, and XML workflow."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np

from .arm_client import ArmClient
from .core import Board, Detector, solve_hand_eye
from .xml_writer import write_calibrated_xml

MODEL = Path(__file__).resolve().parents[1] / "description/station/station.xml"
# Reference pipeline offsets, applied to the operator's chosen nominal pose.
POSE_OFFSETS = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [-0.3, 0, 0, 0, 0, 0],
        [0.3, 0, 0, 0, 0, 0],
        [0, 0.15, 0, 0, 0, 0],
        [0, -0.15, 0, 0, 0, 0],
        [0, 0, 0.15, 0, 0, 0],
        [0, 0, -0.15, 0, 0, 0],
        [0, 0, 0, 0.3, 0, 0],
        [0, 0, 0, -0.3, 0, 0],
        [0, 0, 0, 0, 0.3, 0],
        [0, 0, 0, 0, -0.3, 0],
        [0, 0, 0, 0, 0, 0.6],
        [0, 0, 0, 0, 0, -0.6],
        [-0.25, 0, 0, 0, 0, 0.5],
        [0.25, 0, 0, 0, 0, -0.5],
    ]
)


def checked_poses(nominal, limits, supplied=None):
    poses = np.asarray(nominal + POSE_OFFSETS if supplied is None else supplied, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) < 12 or not np.all(np.isfinite(poses)):
        raise ValueError("calibration poses must contain at least 12 finite six-joint rows")
    if np.any(poses < limits[:, 0]) or np.any(poses > limits[:, 1]):
        raise ValueError("calibration sweep exceeds joint limits; choose another nominal pose")
    return poses


def solve_dataset(dataset: dict, output_dir: Path, *, translation_mm=10.0, rotation_deg=3.0):
    """Re-solve saved observations and write standalone JSON plus calibrated XML."""
    solution = solve_hand_eye(
        dataset["samples"],
        mode=dataset["mode"],
        max_translation_rms_mm=translation_mm,
        max_rotation_rms_deg=rotation_deg,
    )
    record = {key: value for key, value in dataset.items() if key != "samples"}
    record["solution"] = solution
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "calibration.json"
    if result_path.exists():
        raise ValueError("calibration.json already exists; choose a new output directory")
    write_calibrated_xml(Path(dataset["model_xml"]), output_dir / "station_calibrated.xml", record)
    result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def calibrate(args) -> dict:
    """Collect one camera's calibration using an already-running follower service."""
    if not args.confirm_motion:
        raise ValueError("live calibration requires --confirm-motion")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.camera_name) is None:
        raise ValueError("camera name must contain letters, digits, underscores or hyphens")
    if args.camera_body and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.camera_body) is None:
        raise ValueError("camera body must contain letters, digits, underscores or hyphens")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 0.3:
        raise ValueError("calibration speed must be in (0, 0.3] rad/s")
    import cv2
    import mujoco

    from .camera import RealSenseCamera, resolve_serial

    board = Board(args.squares_x, args.squares_y, args.square_length, args.marker_length)
    detector = Detector(board)
    model_path = args.model.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    names = [f"{args.arm}_joint{i}" for i in range(1, 7)]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{args.arm}_link_6")
    if min(*joint_ids, body_id) < 0:
        raise ValueError("station XML must contain the selected arm joints and link_6")
    limits = model.jnt_range[joint_ids]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    camera = RealSenseCamera(resolve_serial(args.serial), resolution=args.resolution)
    arm = None
    commanded = False
    dataset = {}
    try:
        # Claim the camera before any motor command; a running camera owner makes
        # this fail rather than starting calibration alongside an active session.
        camera.open()
        K, distortion, resolution = camera.get_intrinsics()
        cv2.namedWindow("YAM calibration", cv2.WINDOW_NORMAL)
        arm = ArmClient(args.host, args.port or (11333 if args.arm == "left" else 11334))
        dataset = {
            "schema_version": 1,
            "camera_name": args.camera_name,
            "camera_serial": camera.serial,
            "mode": args.mode,
            "arm": args.arm,
            "parent_frame": "world" if args.mode == "fixed" else f"{args.arm}_link_6",
            "camera_body": args.camera_body
            or (
                "top_camera_d405"
                if args.camera_name == "top"
                else f"{args.arm}_camera_d405"
                if args.mode == "wrist"
                else f"{args.camera_name}_camera"
            ),
            "model_xml": str(model_path),
            "resolution": list(resolution),
            "intrinsics": K.tolist(),
            "distortion": distortion.tolist(),
            "board": asdict(board),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "samples": [],
        }

        def preview(label):
            if not camera.grab():
                raise RuntimeError("camera stopped delivering frames; calibration stopped")
            image = camera.get_image()
            detection = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), K, distortion)
            annotated = image.copy()
            status = "BOARD OK" if detection else "NO BOARD"
            cv2.putText(
                annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1
            )
            cv2.putText(
                annotated,
                status,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if detection else (0, 0, 255),
                2,
            )
            cv2.imshow("YAM calibration", annotated)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (27, ord("q"))
                or cv2.getWindowProperty("YAM calibration", cv2.WND_PROP_VISIBLE) < 1
            ):
                raise KeyboardInterrupt
            return key, detection, image

        print("Guide the arm until the board is visible. SPACE locks nominal pose; Q/Esc stops.")
        while True:
            arm.command(arm.get_joint_pos(), gravity=True)
            commanded = True
            key, detection, _ = preview("Guide arm; SPACE: lock nominal; Q: stop")
            if key == ord(" ") and detection is not None:
                break
        arm.hold()
        nominal = arm.get_joint_pos()
        supplied = json.loads(args.poses.read_text()) if args.poses else None
        poses = checked_poses(nominal, limits, supplied)
        print("Sweep joint targets (radians):\n", poses)
        print("SPACE starts the sweep. Joint limits are checked; obstacles are not planned around.")
        while preview(f"{len(poses)} poses; SPACE: start sweep; Q: stop")[0] != ord(" "):
            pass
        for index, target in enumerate(poses):
            label = f"Pose {index + 1}/{len(poses)}; samples {len(dataset['samples'])}; Q: stop"
            arm.move(target, speed=args.speed, poll=lambda: preview(label))
            deadline = time.monotonic() + 0.8
            while time.monotonic() < deadline:
                preview(label)
            best = None
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                before = arm.get_joint_pos()
                _, detection, image = preview(label)
                after = arm.get_joint_pos()
                if detection is None or detection["reprojection_rms_px"] > 2.0:
                    continue
                if np.max(np.abs(after - before)) > 0.005:
                    continue
                if best is not None and detection["corners"] <= best["corners"]:
                    continue
                data.qpos[model.jnt_qposadr[joint_ids]] = (before + after) / 2
                mujoco.mj_forward(model, data)
                pose = np.eye(4)
                pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
                pose[:3, 3] = data.xpos[body_id]
                best = {**detection, "T_base_from_ee": pose.tolist(), "joints": after.tolist()}
                cv2.imwrite(str(args.output_dir / f"sample-{index:03d}.png"), image)
            if best is not None:
                dataset["samples"].append(best)
            (args.output_dir / "samples.json").write_text(
                json.dumps(dataset, indent=2, allow_nan=False)
            )
        arm.move(nominal, speed=args.speed, poll=lambda: preview("Returning to nominal pose"))
    finally:
        try:
            if arm is not None:
                try:
                    if commanded:
                        arm.hold()
                finally:
                    arm.close()
        finally:
            camera.close()
            cv2.destroyAllWindows()
    return solve_dataset(dataset, args.output_dir)
