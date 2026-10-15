# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ChArUco and hand-eye math adapted from ENPIRE's YAM calibration pipeline.

Fixed cameras use a board rigidly attached to the moving arm. Wrist cameras
observe a board rigidly fixed in the world. All transforms map right to left.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Board:
    squares_x: int = 5
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.03
    dictionary: str = "DICT_4X4_50"

    def create(self):
        import cv2

        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("board must have at least 3 squares in each direction")
        if not 0 < self.marker_length_m < self.square_length_m < 1:
            raise ValueError("board lengths must satisfy 0 < marker < square < 1 metre")
        if not self.dictionary.startswith("DICT_") or not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary: {self.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_length_m, self.marker_length_m, dictionary
        )

    def write_png(self, path: Path, *, dpi: int = 300) -> None:
        """Add a quiet margin outside the pattern, preserving physical square size."""
        from PIL import Image

        if dpi < 72:
            raise ValueError("board DPI must be at least 72")
        pixels = round(self.square_length_m / 0.0254 * dpi)
        pattern = self.create().generateImage((self.squares_x * pixels, self.squares_y * pixels))
        image = np.pad(pattern, pixels, constant_values=255)
        # Record the exact scale after rounding pixels per square.
        actual_dpi = pixels * 0.0254 / self.square_length_m
        Image.fromarray(image).save(path, dpi=(actual_dpi, actual_dpi))


def transform(value: object, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a rigid transform")
    return matrix


class Detector:
    def __init__(self, board: Board):
        import cv2

        self.board = board.create()
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.points = self.board.getChessboardCorners() - np.array(
            [
                board.squares_x * board.square_length_m / 2,
                board.squares_y * board.square_length_m / 2,
                0,
            ],
            dtype=np.float32,
        )

    def detect(
        self, rgb: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray
    ) -> dict | None:
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            return None
        points = self.points[ids.ravel()]
        try:
            ok, rvec, tvec = cv2.solvePnP(
                points, corners, intrinsics, distortion, flags=cv2.SOLVEPNP_SQPNP
            )
        except cv2.error:
            return None
        if not ok or tvec[2, 0] <= 0:
            return None
        projected, _ = cv2.projectPoints(points, rvec, tvec, intrinsics, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=-1))))
        result = np.eye(4)
        result[:3, :3] = cv2.Rodrigues(rvec)[0]
        result[:3, 3] = tvec.ravel()
        return {
            "T_camera_from_board": result.tolist(),
            "corners": len(ids),
            "reprojection_rms_px": error,
        }


def solve_hand_eye(
    samples: list[dict],
    *,
    mode: str,
    max_translation_rms_mm: float = 10.0,
    max_rotation_rms_deg: float = 3.0,
) -> dict:
    """Solve ENPIRE's fixed/wrist equations, rejecting invalid or degenerate fits."""
    import cv2

    if mode not in {"fixed", "wrist"}:
        raise ValueError("calibration mode must be fixed or wrist")
    if len(samples) < 12:
        raise ValueError("at least 12 stationary board observations are required")
    if not all(np.isfinite(v) and v > 0 for v in (max_translation_rms_mm, max_rotation_rms_deg)):
        raise ValueError("quality thresholds must be finite and positive")
    ee = [transform(s["T_base_from_ee"], "arm pose") for s in samples]
    board = [transform(s["T_camera_from_board"], "board pose") for s in samples]
    relative = [pose[:3, :3] @ ee[0][:3, :3].T for pose in ee]
    spread = np.linalg.svd(Rotation.from_matrix(relative).as_rotvec(), compute_uv=False)
    if spread[1] < 0.15:
        raise ValueError("insufficient rotation diversity: use rotations around at least two axes")
    gripper = ee if mode == "wrist" else [np.linalg.inv(pose) for pose in ee]
    methods = {
        name: getattr(cv2, f"CALIB_HAND_EYE_{name.upper()}")
        for name in ("tsai", "park", "horaud", "andreff", "daniilidis")
    }
    candidates = {}
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [m[:3, :3] for m in gripper],
                [m[:3, 3] for m in gripper],
                [m[:3, :3] for m in board],
                [m[:3, 3] for m in board],
                method=method,
            )
            result = np.eye(4)
            result[:3, :3], result[:3, 3] = rotation, translation.ravel()
            transform(result)
            # Fixed: ee_from_base @ base_from_camera @ camera_from_board is constant.
            # Wrist: base_from_ee @ ee_from_camera @ camera_from_board is constant.
            probes = [g @ result @ b for g, b in zip(gripper, board, strict=True)]
            positions = np.array([p[:3, 3] for p in probes])
            trans_rms = (
                float(np.sqrt(np.mean(np.sum((positions - positions.mean(0)) ** 2, 1)))) * 1000
            )
            rotations = Rotation.from_matrix([p[:3, :3] for p in probes])
            rot_rms = float(
                np.rad2deg(np.sqrt(np.mean((rotations * rotations.mean().inv()).magnitude() ** 2)))
            )
            if not np.isfinite(trans_rms + rot_rms):
                continue
            candidates[name] = {
                "T_parent_from_camera": result.tolist(),
                "translation_rms_mm": trans_rms,
                "rotation_rms_deg": rot_rms,
            }
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
    accepted = {
        name: r
        for name, r in candidates.items()
        if r["translation_rms_mm"] <= max_translation_rms_mm
        and r["rotation_rms_deg"] <= max_rotation_rms_deg
    }
    if not accepted:
        raise ValueError(f"no hand-eye solution passed quality thresholds: {candidates}")
    best = min(
        accepted,
        key=lambda n: (
            accepted[n]["translation_rms_mm"] / max_translation_rms_mm
            + accepted[n]["rotation_rms_deg"] / max_rotation_rms_deg
        ),
    )
    return {
        "method": best,
        **accepted[best],
        "sample_count": len(samples),
        "all_methods": candidates,
    }
