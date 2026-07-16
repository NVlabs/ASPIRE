#!/usr/bin/env python3
"""Detection-only AnyGrasp server.

This intentionally mirrors the upstream grasp_detection demo as closely as the
HTTP serving setup allows:

* build the scene point cloud from depth + intrinsics
* keep the full scene cloud inside a demo-style z-range
* derive ``lims`` from the selected object region
* call ``AnyGrasp.get_grasp(...)`` with the demo defaults
* run ``nms().sort_by_score()`` and display the returned grasps directly

Default behavior for this repo's SAM3→AnyGrasp pipeline is
``object_input_mode="segmented_object_cloud"``: the point cloud sent into
AnyGrasp contains only the SAM3-segmented object points. This reduces grasp pose
ambiguity versus feeding the broader ROI/workspace cloud. ``roi_workspace``
remains available only as an explicit override for debugging/comparison.

Deliberate remaining differences from ``third_party/anygrasp_sdk/grasp_detection/demo.py``:

* inputs arrive over HTTP as RGB/depth/K/segmap arrays instead of files
* the object selection crop comes from the provided segmap instead of a manual box
* visualization is rendered to a JPEG overlay for the web UI instead of an Open3D window

There is no tracking or remembered-grasp logic in this server anymore.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPO_ROOT, required_modules=["cv2", "uvicorn"])

_SITE_MODULE_PATH = REPO_ROOT / "sitecustomize.py"
_SITE_SPEC = importlib.util.spec_from_file_location("yam_repo_sitecustomize", _SITE_MODULE_PATH)
if _SITE_SPEC is None or _SITE_SPEC.loader is None:
    raise ImportError(f"Could not load repo sitecustomize from {_SITE_MODULE_PATH}")
_SITE_MODULE = importlib.util.module_from_spec(_SITE_SPEC)
_SITE_SPEC.loader.exec_module(_SITE_MODULE)
apply_numpy_compat_aliases = _SITE_MODULE.apply_numpy_compat_aliases

import argparse
import base64
import gc
import io
import logging
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cap.utils.anygrasp_runtime import configure_anygrasp_imports, prepare_anygrasp_runtime

apply_numpy_compat_aliases()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AnyGrasp Detection Server")

_ANYGRASP: Any | None = None
_CHECKPOINT_PATH: str | None = None
_TOP_DOWN_GRASP: bool = False


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


def _normalize_object_input_mode(value: str | None) -> str:
    mode = str(value or "segmented_object_cloud").strip().lower()
    aliases = {
        "segmented": "segmented_object_cloud",
        "segmentation": "segmented_object_cloud",
        "mask": "segmented_object_cloud",
        "object_mask": "segmented_object_cloud",
        "object_segmentation": "segmented_object_cloud",
        "segmented_object_cloud": "segmented_object_cloud",
        "roi": "roi_workspace",
        "bbox": "roi_workspace",
        "bounding_region": "roi_workspace",
        "roi_workspace": "roi_workspace",
    }
    if mode not in aliases:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported object_input_mode={value!r}. "
                "Use 'segmented_object_cloud' or 'roi_workspace'."
            ),
        )
    return aliases[mode]


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_np(s: str) -> np.ndarray:
    try:
        return np.load(io.BytesIO(base64.b64decode(s)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad numpy data: {e}")


def _depth_to_points(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    fx = float(cam_K[0, 0])
    fy = float(cam_K[1, 1])
    cx = float(cam_K[0, 2])
    cy = float(cam_K[1, 2])

    xmap = np.arange(depth.shape[1], dtype=np.float32)
    ymap = np.arange(depth.shape[0], dtype=np.float32)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth.astype(np.float32)
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    return np.stack([points_x, points_y, points_z], axis=-1)


def _workspace_lims(points: np.ndarray, margin: float) -> list[float]:
    lo = points.min(axis=0) - margin
    hi = points.max(axis=0) + margin
    return [
        float(lo[0]), float(hi[0]),
        float(lo[1]), float(hi[1]),
        float(lo[2]), float(hi[2]),
    ]


def _frame_to_scene(
    rgb: np.ndarray,
    depth: np.ndarray,
    cam_K: np.ndarray,
    *,
    z_range: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_all = _depth_to_points(depth, cam_K)
    mask = (
        np.isfinite(points_all).all(axis=-1)
        & (points_all[:, :, 2] > float(z_range[0]))
        & (points_all[:, :, 2] < float(z_range[1]))
    )
    points = points_all[mask].astype(np.float32)
    colors = (rgb[mask].astype(np.float32) / 255.0).astype(np.float32)
    return points_all, mask, points, colors


def _cloud_for_render(cloud: Any, fallback_points: np.ndarray, fallback_colors: np.ndarray):
    import open3d as o3d

    if cloud is not None:
        try:
            pts = np.asarray(cloud.points, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 0:
                return cloud
        except Exception:
            pass

    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(fallback_points, dtype=np.float64)
    colors = np.asarray(fallback_colors, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        pts = np.zeros((1, 3), dtype=np.float64)
        colors = np.full((1, 3), 0.6, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors.shape == pts.shape:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    else:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
    return pcd


def _extract_wireframe(geom: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(geom, 'vertices') and hasattr(geom, 'triangles'):
        pts = np.asarray(geom.vertices, dtype=np.float64)
        tris = np.asarray(geom.triangles, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
        edge_set: set[tuple[int, int]] = set()
        for tri in tris:
            a, b, c = [int(x) for x in tri]
            for u, v in ((a, b), (b, c), (c, a)):
                if u == v:
                    continue
                edge_set.add((u, v) if u < v else (v, u))
        edges = np.array(sorted(edge_set), dtype=np.int32) if edge_set else np.empty((0, 2), dtype=np.int32)
        return pts, edges

    if hasattr(geom, 'points') and hasattr(geom, 'lines'):
        pts = np.asarray(geom.points, dtype=np.float64)
        lines = np.asarray(geom.lines, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
        if lines.ndim != 2 or lines.shape[1] != 2:
            return pts, np.empty((0, 2), dtype=np.int32)
        return pts, lines

    return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)


def _project_cam_to_pixel(point_cam: np.ndarray, cam_K: np.ndarray) -> tuple[int, int] | None:
    z = float(point_cam[2])
    if z <= 0.0:
        return None
    u = int(round(float(cam_K[0, 0]) * float(point_cam[0]) / z + float(cam_K[0, 2])))
    v = int(round(float(cam_K[1, 1]) * float(point_cam[1]) / z + float(cam_K[1, 2])))
    return u, v


def _render_overlay_jpeg(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    cam_K: np.ndarray,
    grippers: list[Any],
) -> bytes:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]

    draw_order = list(range(len(grippers)))
    best_idx = 0 if draw_order else None
    second_idx = 1 if len(draw_order) > 1 else None
    highlight_idxs = [idx for idx in (best_idx, second_idx) if idx is not None]
    if highlight_idxs:
        draw_order = [idx for idx in draw_order if idx not in highlight_idxs] + highlight_idxs

    for draw_rank, grasp_idx in enumerate(draw_order):
        pts_cam, edges = _extract_wireframe(grippers[grasp_idx])
        px = [_project_cam_to_pixel(p, cam_K) for p in pts_cam]
        is_best = grasp_idx == best_idx
        is_second = grasp_idx == second_idx
        if is_best:
            color = (255, 0, 0)
            line_thickness = 2
        elif is_second:
            color = (0, 255, 0)
            line_thickness = 2
        else:
            alpha = 1.0 - 0.6 * (draw_rank / max(len(draw_order) - 1, 1))
            color = tuple(int(c * alpha) for c in (0, 220, 255))
            line_thickness = 2
        for a, b in edges:
            pa = px[int(a)]
            pb = px[int(b)]
            if pa is None or pb is None:
                continue
            if not (0 <= pa[0] < w and 0 <= pa[1] < h):
                continue
            if not (0 <= pb[0] < w and 0 <= pb[1] < h):
                continue
            cv2.line(img, pa, pb, color, line_thickness, cv2.LINE_AA)

    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('Failed to encode AnyGrasp overlay JPEG')
    return buf.tobytes()


def _compute_grasp_pixel_bounds(
    cam_K: np.ndarray,
    gripper: Any,
    image_hw: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    h, w = image_hw
    pts_cam, _ = _extract_wireframe(gripper)
    if len(pts_cam) == 0:
        return None
    px = [_project_cam_to_pixel(p, cam_K) for p in pts_cam]
    px = [p for p in px if p is not None]
    if not px:
        return None
    xs = [int(p[0]) for p in px]
    ys = [int(p[1]) for p in px]
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(w - 1, max(xs))
    y2 = min(h - 1, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _compute_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expand_and_clip_bbox(
    bbox: tuple[int, int, int, int],
    image_hw: tuple[int, int],
    *,
    pad_px: int = 18,
) -> tuple[int, int, int, int]:
    h, w = image_hw
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1) - pad_px),
        max(0, int(y1) - pad_px),
        min(w - 1, int(x2) + pad_px),
        min(h - 1, int(y2) + pad_px),
    )


def _union_bbox(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _crop_jpeg_bytes(
    jpeg_bytes: bytes,
    bbox: tuple[int, int, int, int] | None,
) -> bytes:
    if bbox is None:
        return jpeg_bytes
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return jpeg_bytes
    crop = img[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return jpeg_bytes
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jpeg_bytes
    return buf.tobytes()


def _render_single_grasp_overlays(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    cam_K: np.ndarray,
    grippers: list[Any],
) -> list[bytes]:
    overlays: list[bytes] = []
    image_hw = rgb.shape[:2]
    object_bbox = _compute_mask_bbox(object_mask)
    for grasp_idx in range(len(grippers)):
        overlay = _render_overlay_jpeg(rgb, object_mask, cam_K, [grippers[grasp_idx]])
        grasp_bbox = _compute_grasp_pixel_bounds(cam_K, grippers[grasp_idx], image_hw)
        crop_bbox = _union_bbox(object_bbox, grasp_bbox)
        crop_bbox = _expand_and_clip_bbox(crop_bbox, image_hw) if crop_bbox is not None else None
        overlays.append(_crop_jpeg_bytes(overlay, crop_bbox))
    return overlays



def _graspgroup_to_mats_and_widths(gg_pick: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if gg_pick is None or len(gg_pick) == 0:
        return np.empty((0, 4, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    rotations = np.asarray(gg_pick.rotation_matrices, dtype=np.float64)
    translations = np.asarray(gg_pick.translations, dtype=np.float64)
    widths = np.asarray(gg_pick.widths, dtype=np.float64)

    grasps = np.tile(np.eye(4, dtype=np.float64), (len(gg_pick), 1, 1))
    grasps[:, :3, :3] = rotations
    grasps[:, :3, 3] = translations
    return grasps, widths


def _run_demo_style_grasp_inference(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    cam_K: np.ndarray,
    segmap: np.ndarray,
    segmap_id: int,
    z_range: list[float],
    max_grasps: int,
    workspace_margin: float,
    collision_detection: bool,
    object_input_mode: str,
) -> tuple[Any | None, np.ndarray, np.ndarray, np.ndarray]:
    if _ANYGRASP is None:
        raise HTTPException(status_code=503, detail='Model not initialized')

    if rgb.shape[:2] != depth.shape[:2] or segmap.shape[:2] != depth.shape[:2]:
        raise HTTPException(status_code=400, detail='RGB, depth, and segmap must have matching height/width')

    points_all, mask, points, colors = _frame_to_scene(rgb, depth, cam_K, z_range=z_range)
    object_mask = (segmap == segmap_id) & mask
    if not np.any(object_mask):
        return None, object_mask, points, colors

    object_input_mode = _normalize_object_input_mode(object_input_mode)
    object_points = points_all[object_mask].astype(np.float32)
    object_colors = (rgb[object_mask].astype(np.float32) / 255.0).astype(np.float32)
    lims = _workspace_lims(object_points, workspace_margin)
    grasp_points = object_points if object_input_mode == "segmented_object_cloud" else points
    grasp_colors = object_colors if object_input_mode == "segmented_object_cloud" else colors
    apply_object_mask = object_input_mode != "segmented_object_cloud"
    logger.info(
        'AnyGrasp detection: mode=%s scene_points=%d object_points=%d sent_points=%d lims=%s',
        object_input_mode,
        int(len(points)),
        int(len(object_points)),
        int(len(grasp_points)),
        [round(float(v), 4) for v in lims],
    )

    gg, cloud = _ANYGRASP.get_grasp(
        grasp_points,
        grasp_colors,
        lims=lims,
        apply_object_mask=apply_object_mask,
        dense_grasp=False,
        collision_detection=collision_detection,
    )
    if gg is None or len(gg) == 0:
        logger.warning("AnyGrasp returned no grasps")
        from graspnetAPI.grasp import GraspGroup
        return GraspGroup(), object_mask, points, colors
    gg = gg.nms().sort_by_score()
    gg_pick = gg[:max_grasps]
    return gg_pick, object_mask, points, colors


def _run_demo_style_anygrasp(req: PlanRequest) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes, list[bytes]]:
    rgb = _b64_to_np(req.rgb_base64).astype(np.uint8)
    depth = _b64_to_np(req.depth_base64).astype(np.float32)
    cam_K = _b64_to_np(req.cam_K_base64).astype(np.float64)
    segmap = _b64_to_np(req.segmap_base64)
    z_range = req.z_range or [1e-6, 1.5]

    gg_pick, object_mask, points, colors = _run_demo_style_grasp_inference(
        rgb=rgb,
        depth=depth,
        cam_K=cam_K,
        segmap=segmap,
        segmap_id=req.segmap_id,
        z_range=z_range,
        max_grasps=req.max_grasps,
        workspace_margin=req.workspace_margin,
        collision_detection=req.collision_detection,
        object_input_mode=req.object_input_mode,
    )
    if gg_pick is None:
        overlay = _render_overlay_jpeg(rgb, object_mask, cam_K, [])
        return (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            overlay,
            [],
        )

    scores = np.asarray(gg_pick.scores, dtype=np.float64) if len(gg_pick) else np.empty((0,), dtype=np.float64)
    grasps, widths = _graspgroup_to_mats_and_widths(gg_pick)
    grippers = gg_pick.to_open3d_geometry_list() if len(gg_pick) else []
    overlay = _render_overlay_jpeg(rgb, object_mask, cam_K, grippers)
    thumbnails = _render_single_grasp_overlays(rgb, object_mask, cam_K, grippers)
    return grasps, scores, widths, overlay, thumbnails


@app.post('/plan', response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest) -> PlanResponse:
    grasps, scores, widths, _, _ = _run_demo_style_anygrasp(req)
    return PlanResponse(
        grasps_base64=_np_to_b64(grasps),
        scores_base64=_np_to_b64(scores),
        widths_base64=_np_to_b64(widths),
        n_grasps=int(len(scores)),
        best_score=float(scores[0]) if len(scores) else None,
    )


@app.post('/plan_viz', response_model=PlanVizResponse)
async def plan_viz_endpoint(req: PlanRequest) -> PlanVizResponse:
    grasps, scores, widths, overlay_bytes, thumbnail_bytes = _run_demo_style_anygrasp(req)
    return PlanVizResponse(
        grasps_base64=_np_to_b64(grasps),
        scores_base64=_np_to_b64(scores),
        widths_base64=_np_to_b64(widths),
        n_grasps=int(len(scores)),
        best_score=float(scores[0]) if len(scores) else None,
        overlay_jpeg_base64=base64.b64encode(overlay_bytes).decode(),
        grasp_thumbnail_jpeg_base64=[base64.b64encode(b).decode() for b in thumbnail_bytes],
    )


@app.get('/health')
def health() -> dict[str, Any]:
    return {
        'status': 'ok',
        'model_loaded': _ANYGRASP is not None,
        'checkpoint_path': _CHECKPOINT_PATH,
        'top_down_grasp': _TOP_DOWN_GRASP,
        'detection_only': True,
    }


@app.post('/reset_state')
def reset_state() -> dict[str, Any]:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        'ok': True,
        'model_loaded': _ANYGRASP is not None,
        'checkpoint_path': _CHECKPOINT_PATH,
        'top_down_grasp': _TOP_DOWN_GRASP,
    }


def main() -> None:
    global _ANYGRASP, _CHECKPOINT_PATH, _TOP_DOWN_GRASP

    parser = argparse.ArgumentParser(description='AnyGrasp demo-style detection server')
    parser.add_argument('--port', type=int, default=8122, help='HTTP port (default: 8122)')
    parser.add_argument('--host', default='0.0.0.0', help='Bind address')
    parser.add_argument(
        '--checkpoint-path',
        default=str(REPO_ROOT / 'checkpoint_detection.tar'),
        help='Path to the AnyGrasp detection checkpoint',
    )
    parser.add_argument(
        '--license-zip',
        default=os.environ.get('ANYGRASP_LICENSE_ZIP', ''),
        help='Path to the AnyGrasp license zip',
    )
    parser.add_argument('--max-gripper-width', type=float, default=0.1)
    parser.add_argument('--gripper-height', type=float, default=0.03)
    parser.add_argument(
        '--top-down-grasp',
        dest='top_down_grasp',
        action='store_true',
        default=True,
        help='Load AnyGrasp in top-down mode (default: enabled)',
    )
    parser.add_argument(
        '--no-top-down-grasp',
        dest='top_down_grasp',
        action='store_false',
        help='Disable top-down mode and use unrestricted AnyGrasp proposals',
    )
    args = parser.parse_args()

    if not args.license_zip:
        parser.error('--license-zip or ANYGRASP_LICENSE_ZIP is required')

    runtime = prepare_anygrasp_runtime(license_zip=args.license_zip)
    configure_anygrasp_imports(runtime)

    from gsnet import AnyGrasp

    cfg = SimpleNamespace(
        checkpoint_path=args.checkpoint_path,
        max_gripper_width=max(0.0, min(0.1, args.max_gripper_width)),
        gripper_height=args.gripper_height,
        top_down_grasp=args.top_down_grasp,
        debug=False,
    )

    logger.info('Loading AnyGrasp detection runtime...')
    _ANYGRASP = AnyGrasp(cfg)
    _ANYGRASP.load_net()
    _CHECKPOINT_PATH = args.checkpoint_path
    _TOP_DOWN_GRASP = bool(args.top_down_grasp)

    logger.info('AnyGrasp detection ready')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
