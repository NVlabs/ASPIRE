"""Contact-GraspNet service matching :mod:`cap_harness.providers.graspnet.client`."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import functools
import io
import logging
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import uvicorn
import yaml

from cap_harness.providers.service_runtime import install_service_runtime

from .wire import PLAN_POINT_CLOUDS_PATH

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="cap-harness Contact-GraspNet provider")
# Bound concurrent GPU/inference requests so one instance safely serves
# multiple simulator clients (limit from CAP_HARNESS_SERVICE_CONCURRENCY).
install_service_runtime(app)
_ESTIMATOR: Any | None = None
_GPU_SEMAPHORE = asyncio.Semaphore(1)


class PlanPointCloudsRequest(BaseModel):
    pc_full_base64: str
    pc_segment_base64: str
    segmap_id: int = 1
    local_regions: bool = True
    filter_grasps: bool = True
    forward_passes: int = 1
    max_retries: int = 7


class PlanResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    contact_pts_base64: str


async def _run_serialized(function: Any, *args: Any) -> Any:
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(function, *args))


def _decode_numpy(payload: str) -> np.ndarray:
    try:
        raw = base64.b64decode(payload, validate=True)
        with io.BytesIO(raw) as buffer:
            return np.load(buffer, allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {exc}") from exc


def _encode_numpy(array: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        np.save(buffer, np.asarray(array), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _load_config(checkpoint_root: Path) -> dict[str, Any]:
    with (checkpoint_root / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["DATA"]["classes"] = None
    return config


def _camera_transform(target: np.ndarray, extent_m: float = 0.25) -> np.ndarray:
    position = np.random.uniform(-extent_m, extent_m, 3)
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        position[2] += extent_m
        forward = target - position
        norm = np.linalg.norm(forward)
    forward /= norm
    down = np.asarray([0.0, 1.0, 0.0])
    camera_y = down - np.dot(down, forward) * forward
    if np.linalg.norm(camera_y) < 1e-6:
        camera_y = np.asarray([1.0, 0.0, 0.0])
        camera_y -= np.dot(camera_y, forward) * forward
    camera_y /= np.linalg.norm(camera_y)
    camera_x = np.cross(camera_y, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([camera_x, camera_y, forward])
    transform[:3, 3] = position
    return transform


def _predict(
    full: np.ndarray,
    segment: np.ndarray,
    request: PlanPointCloudsRequest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_id = request.segmap_id

    def run_in_frame(
        full_points: np.ndarray, segment_points: np.ndarray
    ) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
        grasps, scores, contacts, _ = _ESTIMATOR.predict_scene_grasps(
            full_points,
            pc_segments={segment_id: segment_points},
            local_regions=request.local_regions,
            filter_grasps=request.filter_grasps,
            forward_passes=request.forward_passes,
        )
        return grasps, scores, contacts

    grasps, scores, contacts = run_in_frame(full, segment)
    if len(grasps.get(segment_id, [])):
        return (
            np.asarray(grasps[segment_id]),
            np.asarray(scores[segment_id]),
            np.asarray(contacts[segment_id]),
        )

    target = np.mean(segment if len(segment) else full, axis=0)
    full_h = np.column_stack([full, np.ones(len(full))])
    segment_h = np.column_stack([segment, np.ones(len(segment))])
    for _ in range(request.max_retries):
        camera_to_world = _camera_transform(target)
        world_to_camera = np.linalg.inv(camera_to_world)
        full_camera = (world_to_camera @ full_h.T).T[:, :3]
        segment_camera = (world_to_camera @ segment_h.T).T[:, :3]
        retry_grasps, retry_scores, retry_contacts = run_in_frame(full_camera, segment_camera)
        if not len(retry_grasps.get(segment_id, [])):
            continue
        grasp_array = camera_to_world @ np.asarray(retry_grasps[segment_id])
        contact_array = np.asarray(retry_contacts[segment_id])
        contact_h = np.column_stack([contact_array, np.ones(len(contact_array))])
        return (
            grasp_array,
            np.asarray(retry_scores[segment_id]),
            (camera_to_world @ contact_h.T).T[:, :3],
        )
    return np.asarray([]), np.asarray([]), np.asarray([])


def _plan(request: PlanPointCloudsRequest) -> PlanResponse:
    full = np.asarray(_decode_numpy(request.pc_full_base64), dtype=np.float32)
    segment = np.asarray(_decode_numpy(request.pc_segment_base64), dtype=np.float32)
    if full.ndim != 2 or full.shape[1] != 3 or not len(full):
        raise HTTPException(status_code=400, detail="pc_full must have non-empty shape (N, 3)")
    if segment.ndim != 2 or segment.shape[1] != 3 or not len(segment):
        raise HTTPException(status_code=400, detail="pc_segment must have non-empty shape (N, 3)")
    grasps, scores, contacts = _predict(full, segment, request)
    return PlanResponse(
        grasps_base64=_encode_numpy(grasps),
        scores_base64=_encode_numpy(scores),
        contact_pts_base64=_encode_numpy(contacts),
    )


@app.post(PLAN_POINT_CLOUDS_PATH, response_model=PlanResponse)
async def plan_point_clouds(request: PlanPointCloudsRequest) -> PlanResponse:
    if _ESTIMATOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _run_serialized(_plan, request)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Contact-GraspNet planning failed")
        raise HTTPException(status_code=500, detail=f"Grasp planning failed: {exc}") from exc


def _vendor_root() -> Path:
    configured = os.environ.get("CAP_HARNESS_VENDOR_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "third_party"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cap-harness Contact-GraspNet service")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8115)
    args = parser.parse_args()

    vendor = _vendor_root() / "contact_graspnet_pytorch"
    pointnet = vendor / "Pointnet_Pointnet2_pytorch"
    sys.path.extend([str(pointnet), str(vendor)])
    try:
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    except ImportError:
        LOGGER.exception("Cannot import Contact-GraspNet from %s", vendor)
        raise

    checkpoint_root = vendor / "checkpoints/contact_graspnet"
    global _ESTIMATOR
    _ESTIMATOR = GraspEstimator(_load_config(checkpoint_root))
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_root / "checkpoints", model=_ESTIMATOR.model
    )
    with contextlib.suppress(FileExistsError):
        checkpoint_io.load("model.pt")
    LOGGER.info("Contact-GraspNet loaded on %s", args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
