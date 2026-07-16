#!/usr/bin/env python3
"""Mock AnyGrasp HTTP server for integration tests.

This test double does not import or execute the licensed AnyGrasp SDK. It
implements the same small HTTP contract used by OpenForge so upper layers can
exercise request/response handling while a valid AnyGrasp license is unavailable.
The returned grasps are synthetic and must not be used for robot motion.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPO_ROOT, required_modules=["cv2", "fastapi", "numpy", "uvicorn"])

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock AnyGrasp Integration Test Server")

_SYNTHETIC_GRASPS = False


class PlanRequest(BaseModel):
    rgb_base64: str
    depth_base64: str
    cam_K_base64: str
    segmap_base64: str
    segmap_id: int = 1
    z_range: list[float] | None = None
    max_grasps: int = 20
    workspace_margin: float = 0.02
    collision_detection: bool = True
    object_input_mode: str = "segmented_object_cloud"


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    widths_base64: str
    n_grasps: int
    best_score: float | None = None


class PlanVizResponse(PlanResponse):
    overlay_jpeg_base64: str
    grasp_thumbnail_jpeg_base64: list[str] | None = None


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_np(s: str) -> np.ndarray:
    try:
        return np.load(io.BytesIO(base64.b64decode(s)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad numpy data: {exc}") from exc


def _depth_to_points(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])
    xmap = np.arange(depth.shape[1], dtype=np.float32)
    ymap = np.arange(depth.shape[0], dtype=np.float32)
    xmap, ymap = np.meshgrid(xmap, ymap)
    z = depth.astype(np.float32)
    x = (xmap - cx) / fx * z
    y = (ymap - cy) / fy * z
    return np.stack([x, y, z], axis=-1)


def _empty_result() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 4, 4), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
    )


def _mock_grasps(req: PlanRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = _b64_to_np(req.rgb_base64).astype(np.uint8)
    depth = _b64_to_np(req.depth_base64).astype(np.float32)
    cam_K = _b64_to_np(req.cam_K_base64).astype(np.float64)
    segmap = _b64_to_np(req.segmap_base64)

    if rgb.shape[:2] != depth.shape[:2] or segmap.shape[:2] != depth.shape[:2]:
        raise HTTPException(status_code=400, detail="RGB, depth, and segmap shapes must match")

    if not _SYNTHETIC_GRASPS or int(req.max_grasps) <= 0:
        grasps, scores, widths = _empty_result()
        return grasps, scores, widths, rgb, segmap == req.segmap_id

    z_min, z_max = req.z_range or [1e-6, 1.5]
    object_mask = (
        (segmap == req.segmap_id)
        & np.isfinite(depth)
        & (depth > float(z_min))
        & (depth < float(z_max))
    )
    if not np.any(object_mask):
        grasps, scores, widths = _empty_result()
        return grasps, scores, widths, rgb, object_mask

    points = _depth_to_points(depth, cam_K)[object_mask]
    center = np.median(points, axis=0).astype(np.float64)
    n = min(max(int(req.max_grasps), 0), 8)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    offsets = np.stack(
        [
            0.012 * np.cos(angles),
            0.012 * np.sin(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    )

    grasps = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    # Synthetic AnyGrasp-camera-frame orientation. It is only meant to exercise
    # client transforms and serialization, not to represent a physical grasp.
    base_rot = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    for i, theta in enumerate(angles):
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        yaw = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        grasps[i, :3, :3] = base_rot @ yaw
    grasps[:, :3, 3] = center + offsets
    scores = np.linspace(0.55, 0.45, n, dtype=np.float64)
    widths = np.full((n,), 0.06, dtype=np.float64)
    return grasps, scores, widths, rgb, object_mask


def _overlay_jpeg(rgb: np.ndarray, object_mask: np.ndarray, n_grasps: int) -> bytes:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if object_mask.any():
        green = np.zeros_like(img)
        green[:, :, 1] = 255
        img = np.where(object_mask[:, :, None], (0.65 * img + 0.35 * green).astype(np.uint8), img)
    cv2.putText(
        img,
        f"MOCK ANYGRASP - synthetic grasps: {n_grasps}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode mock overlay JPEG")
    return buf.tobytes()


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest) -> PlanResponse:
    grasps, scores, widths, _, _ = _mock_grasps(req)
    return PlanResponse(
        grasps_base64=_np_to_b64(grasps),
        scores_base64=_np_to_b64(scores),
        widths_base64=_np_to_b64(widths),
        n_grasps=int(len(scores)),
        best_score=float(scores[0]) if len(scores) else None,
    )


@app.post("/plan_viz", response_model=PlanVizResponse)
async def plan_viz_endpoint(req: PlanRequest) -> PlanVizResponse:
    grasps, scores, widths, rgb, object_mask = _mock_grasps(req)
    overlay = _overlay_jpeg(rgb, object_mask, int(len(scores)))
    return PlanVizResponse(
        grasps_base64=_np_to_b64(grasps),
        scores_base64=_np_to_b64(scores),
        widths_base64=_np_to_b64(widths),
        n_grasps=int(len(scores)),
        best_score=float(scores[0]) if len(scores) else None,
        overlay_jpeg_base64=base64.b64encode(overlay).decode(),
        grasp_thumbnail_jpeg_base64=[],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "mock": True,
        "synthetic_grasps": bool(_SYNTHETIC_GRASPS),
        "safe_for_robot_motion": False,
        "detection_only": True,
    }


@app.post("/reset_state")
def reset_state() -> dict[str, Any]:
    return {"ok": True, "mock": True}


def main() -> None:
    global _SYNTHETIC_GRASPS

    parser = argparse.ArgumentParser(description="Mock AnyGrasp server for integration tests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18122)
    parser.add_argument(
        "--synthetic-grasps",
        action="store_true",
        help="Return synthetic grasp matrices instead of an empty result.",
    )
    args = parser.parse_args()
    _SYNTHETIC_GRASPS = bool(args.synthetic_grasps)

    print(
        "[mock-anygrasp] starting test double; "
        f"synthetic_grasps={_SYNTHETIC_GRASPS}; safe_for_robot_motion=False"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
