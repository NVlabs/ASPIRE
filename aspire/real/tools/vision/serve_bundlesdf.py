#!/usr/bin/env python3
"""BundleSDF multi-object tracking sidecar for CAP.

Pulls RGB+depth frames from cap_server via Portal RPC, feeds them into
SharedSam2Tracker + per-object BundleSdf, and exposes 6-DOF poses over HTTP.

Endpoints
---------
GET  /                          Multi-object management UI
GET  /preview                   MJPEG stream — raw camera frames (default camera)
GET  /preview?camera=<name>     MJPEG stream — raw camera frames for configured cameras
GET  /preview/<name>            MJPEG stream — raw camera frames for configured cameras
GET  /multiview                 Browser page showing configured camera previews
GET  /stream/<name>             MJPEG stream — per-object tracking visualization
GET  /stream_composite          MJPEG tiled view of all active streams
POST /add_detection             {"text": "...", "camera": opt, "name": opt, ...}
GET  /get_detection/<name>      Latest pose JSON for named session
POST /end_detection/<name>      Stop tracking named session, free GPU
GET  /list_detections           All active sessions with poses
POST /single_frame_pose         {"text": "..."} — one-shot pose, no session

Streaming design: one MJPEG stream per camera (composite of all tracked objects),
delivered via long-lived HTTP chunked response — no snapshot polling.

Usage
-----
    cd /path/to/aspire/real
    python tools/vision/serve_bundlesdf.py           # default port 8119
    python tools/vision/serve_bundlesdf.py --port 8120
    python tools/vision/serve_bundlesdf.py --cap_server_port 8300
"""

import argparse
import base64
import gc
import html
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(
    __file__,
    REPO_ROOT,
    required_modules=["cv2", "torch", "bundlesdf"],
    extras=["cap_tools"],
)

import cv2
import numpy as np
import torch
import uvicorn
import yaml
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from bundlesdf import BundleSdf
from bundlesdf.run_live_bundlesdf import (
    SharedSam2Tracker,
    build_configs,
    has_valid_depth,
)


# ── Remote SAM3 client (calls tools/vision/serve_sam3.py over HTTP) ───────────────────────

_sam3_url: str = "http://localhost:6767"


def _set_sam3_url(url: str) -> None:
    global _sam3_url
    _sam3_url = url.rstrip("/")


def _parse_camera_names(value: str) -> tuple[str, ...]:
    names: list[str] = []
    for item in value.replace(";", ",").split(","):
        name = item.strip().lower()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_preview_cameras(default_camera: str) -> tuple[str, ...]:
    raw = (
        os.environ.get("OPENFORGE_BUNDLESDF_PREVIEW_CAMERAS", "").strip()
        or os.environ.get("OPENFORGE_REAL_YAM_CAMERAS", "").strip()
        or os.environ.get("CAP_REAL_YAM_CAMERAS", "").strip()
        or os.environ.get("CAP_CAMERA_NAMES", "").strip()
    )
    if raw:
        cameras = list(_parse_camera_names(raw))
    else:
        try:
            from cap.config import CAMERA_NAMES

            cameras = list(CAMERA_NAMES)
        except Exception:
            cameras = ["top", "left", "right"]
    if default_camera and default_camera not in cameras:
        cameras.insert(0, default_camera)
    return tuple(cameras or ["top"])


def _camera_options_html(cameras: tuple[str, ...], default_camera: str) -> str:
    options = []
    for camera in cameras:
        selected = " selected" if camera == default_camera else ""
        safe_camera = html.escape(camera, quote=True)
        options.append(f'    <option value="{safe_camera}"{selected}>{safe_camera}</option>')
    return "\n".join(options)


def _preview_cards_html(cameras: tuple[str, ...]) -> str:
    cards = []
    for camera in cameras:
        safe_camera = html.escape(camera, quote=True)
        cards.append(
            "\n".join(
                [
                    (
                        f'      <div class="preview-card" data-camera="{safe_camera}" '
                        f'role="button" tabindex="0" onclick="selectCamera(\'{safe_camera}\')" '
                        f'onkeydown="onPreviewKey(event, \'{safe_camera}\')">'
                    ),
                    (
                        f'        <div class="preview-head"><span class="preview-name">{safe_camera}</span>'
                        "<span>click to target detections</span></div>"
                    ),
                    f'        <img src="/preview/{safe_camera}" alt="{safe_camera} camera preview">',
                    "      </div>",
                ]
            )
        )
    return "\n".join(cards)


def _render_index_html(cameras: tuple[str, ...], default_camera: str) -> str:
    option_block = """    <option value="top">top</option>
    <option value="left">left</option>
    <option value="right">right</option>"""
    preview_block = """      <div class="preview-card" data-camera="top" role="button" tabindex="0" onclick="selectCamera('top')" onkeydown="onPreviewKey(event, 'top')">
        <div class="preview-head"><span class="preview-name">top</span><span>click to target detections</span></div>
        <img src="/preview/top" alt="top camera preview">
      </div>
      <div class="preview-card" data-camera="left" role="button" tabindex="0" onclick="selectCamera('left')" onkeydown="onPreviewKey(event, 'left')">
        <div class="preview-head"><span class="preview-name">left</span><span>click to target detections</span></div>
        <img src="/preview/left" alt="left camera preview">
      </div>
      <div class="preview-card" data-camera="right" role="button" tabindex="0" onclick="selectCamera('right')" onkeydown="onPreviewKey(event, 'right')">
        <div class="preview-head"><span class="preview-name">right</span><span>click to target detections</span></div>
        <img src="/preview/right" alt="right camera preview">
      </div>"""
    return (
        _INDEX_HTML.replace(option_block, _camera_options_html(cameras, default_camera))
        .replace(preview_block, _preview_cards_html(cameras))
    )


def _render_multiview_html(cameras: tuple[str, ...]) -> str:
    section_block = """    <section class="view"><h2>top</h2><img src="/preview/top"></section>
    <section class="view"><h2>left</h2><img src="/preview/left"></section>
    <section class="view"><h2>right</h2><img src="/preview/right"></section>"""
    sections = "\n".join(
        f'    <section class="view"><h2>{html.escape(camera)}</h2><img src="/preview/{html.escape(camera, quote=True)}"></section>'
        for camera in cameras
    )
    return _MULTIVIEW_HTML.replace(section_block, sections)


def text_to_mask(
    rgb: np.ndarray, text: str, score_threshold: float = 0.2
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Call remote serve_sam3 for text-prompted segmentation."""
    import urllib.request

    buf = io.BytesIO()
    np.save(buf, rgb)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = json.dumps({"text": text, "image_b64": image_b64}).encode()
    req = urllib.request.Request(
        f"{_sam3_url}/segment",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 404:
            raise RuntimeError(body)
        raise RuntimeError(f"SAM3 server error ({e.code}): {body}")
    except Exception as e:
        raise RuntimeError(f"SAM3 server unreachable at {_sam3_url}: {e}")

    data = json.loads(resp.read())
    mask_bytes = base64.b64decode(data["mask_b64"])
    mask = np.load(io.BytesIO(mask_bytes))
    bbox_xywh = tuple(data["bbox_xywh"])
    score = float(data["score"])
    return mask, bbox_xywh, score


def _free_sam3_single_image() -> None:
    """No-op — SAM3 VRAM is managed by the external serve_sam3 process."""
    pass


# ── Pydantic models ───────────────────────────────────────────────────────────


class PushFrameRequest(BaseModel):
    camera: str = "top"
    image_base64: str  # base64 PNG/JPEG
    depth_base64: str | None = None  # base64 numpy float32 array
    intrinsics: list[float] | None = None  # [fx, fy, cx, cy]
    extrinsics: dict | None = None  # {position: [x,y,z], rotation: [9 floats]}


class PushFrameResponse(BaseModel):
    ok: bool
    camera: str
    seq: int


class AddDetectionRequest(BaseModel):
    text: str
    camera: str | None = None
    name: str | None = None
    out_folder: str | None = None
    debug_level: int = 0
    score_thresh: float = 0.3
    # Optional: provide image directly (no cap_server needed)
    image_base64: str | None = None
    intrinsics: list[float] | None = None  # [fx, fy, cx, cy]


class AddDetectionResponse(BaseModel):
    name: str
    bbox: list[int]
    first_score: float


class DetectionEntry(BaseModel):
    name: str
    text: str
    camera: str
    tracking: bool
    score: float
    frame_idx: int
    position_3d: list[float] | None
    pose_origin_3d: list[float] | None = None
    position_3d_source: str | None = None
    quaternion_xyzw: list[float] | None


class ListDetectionsResponse(BaseModel):
    detections: dict[str, DetectionEntry]


class SingleFramePoseRequest(BaseModel):
    text: str
    camera: str | None = None
    out_folder: str | None = None
    debug_level: int = 0
    # Optional: provide images directly (no cap_server needed)
    image_base64: str | None = None
    depth_base64: str | None = None
    intrinsics: list[float] | None = None  # [fx, fy, cx, cy]
    extrinsics: dict | None = None  # {position: [x,y,z], rotation: [9 floats]}


class SingleFramePoseResponse(BaseModel):
    bbox: list[int]
    score: float
    ob_in_cam: list[list[float]] | None = None
    ob_in_world: list[list[float]] | None = None
    position_3d: list[float] | None = None
    pose_origin_3d: list[float] | None = None
    position_3d_source: str | None = None
    quaternion_xyzw: list[float] | None = None


class PoseResponse(BaseModel):
    tracking: bool
    bbox: list[int] | None = None  # SAM2 2D bbox [x, y, w, h]
    ob_in_cam: list[list[float]] | None = None
    ob_in_world: list[list[float]] | None = None
    position_3d: list[float] | None = None
    pose_origin_3d: list[float] | None = None
    position_3d_source: str | None = None
    quaternion_xyzw: list[float] | None = None
    rpy: list[float] | None = None
    half_extents: list[float] | None = None
    score: float = 0.0
    frame_idx: int = 0


class EndDetectionResponse(BaseModel):
    ok: bool


class ResetStateResponse(BaseModel):
    ok: bool
    ended_detections: int
    retained_camera_loops: int


class SegmentRequest(BaseModel):
    text: str
    camera: str | None = None
    image_base64: str | None = None  # optional: provide image directly


class SegmentResponse(BaseModel):
    mask_b64: str  # base64-encoded uint8 HxW mask (0/1)
    bbox_xywh: list[int]
    score: float
    mask_area: int
    height: int
    width: int


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_name(text: str, name: str | None = None) -> str:
    """URL-safe session key — delegates to cap.config.make_bundlesdf_name."""
    from cap.config import make_bundlesdf_name

    return make_bundlesdf_name(text, name)


def _build_K(intrinsics: list[float]) -> np.ndarray:
    """Build 3×3 intrinsic matrix from [fx, fy, cx, cy]."""
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _build_SE3(extrinsics: dict, camera: str = "top") -> np.ndarray:
    """Build 4×4 SE3 from cap_server extrinsics dict (position + rotation).

    Extrinsics from cap_server are in Pinocchio convention.  When
    ``needs_optical_flip`` is set, a diag(-1,-1,1) correction converts
    to the OpenCV convention expected by BundleSDF.
    """
    R = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsics["position"], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    if extrinsics.get("needs_optical_flip", True):
        F = np.diag([-1.0, -1.0, 1.0])
        T[:3, :3] = R @ F
    else:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rotation_matrix_to_quaternion(R: np.ndarray) -> list[float]:
    """Convert 3×3 rotation matrix to quaternion [x, y, z, w]."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # scipy returns [x, y, z, w]
    return [float(x) for x in q]


def _rotation_matrix_to_rpy(R_mat: np.ndarray) -> list[float]:
    """Convert 3×3 rotation matrix to display [roll, pitch, yaw] in degrees.

    Display convention (matching Viser UI / scripted_policy):
        euler_xyz = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
        roll  =  euler_xyz[1]
        pitch = -euler_xyz[0]
        yaw   = -(euler_xyz[2] + 90)
    """
    from scipy.spatial.transform import Rotation

    e = Rotation.from_matrix(R_mat).as_euler("xyz", degrees=True)
    roll = float(e[1])
    pitch = float(-e[0])
    yaw = float(-(e[2] + 90.0))
    return [roll, pitch, yaw]


def _decode_rgb(b64: str) -> np.ndarray:
    """Decode base64 PNG/JPEG to RGB numpy array (HxWx3 uint8)."""
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    return np.asarray(img.convert("RGB"))


def _decode_depth(b64: str) -> np.ndarray:
    """Decode base64 numpy float32 depth array (HxW)."""
    data = base64.b64decode(b64)
    return np.load(io.BytesIO(data)).astype(np.float32)


# ── Visualization helpers ────────────────────────────────────────────────────


def _project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def _draw_xyz_axis(
    color,
    ob_in_cam,
    scale=0.1,
    K=np.eye(3),
    thickness=3,
    transparency=0.3,
    is_input_rgb=False,
):
    """Draw XYZ coordinate axes on a BGR image at the given pose."""
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(_project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
    xx = tuple(_project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(_project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(_project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.FILLED
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=(0, 0, 255),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=(0, 255, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=(255, 0, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    return tmp


def _draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.35,
    frame_idx: int = 0,
    score: float = 1.0,
    tracker_label: str = "SAM2",
    ob_in_cam: np.ndarray | None = None,
    K_vis: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay SAM2 mask + optional pose axis on an RGB image."""
    out = rgb.copy()
    colored = np.zeros_like(rgb)
    colored[mask > 0] = color
    cv2.addWeighted(colored, alpha, out, 1 - alpha, 0, out)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    label = f"Frame {frame_idx:04d}  {tracker_label}={score:.2f}"
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if ob_in_cam is not None and K_vis is not None:
        t = ob_in_cam[:3, 3]
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        bgr = _draw_xyz_axis(
            bgr,
            ob_in_cam,
            scale=0.05,
            K=K_vis,
            thickness=3,
            transparency=0,
            is_input_rgb=False,
        )
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cv2.putText(
            out,
            f"t: [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m",
            (8, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    return out


# ── Preview streamer ──────────────────────────────────────────────────────────


class _PreviewStreamer:
    """Live-only MJPEG stream for raw camera preview frames."""

    def __init__(self):
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._error: str | None = None
        self._version = 0

    def push(self, img_rgb: np.ndarray):
        _, buf = cv2.imencode(
            ".jpg", img_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 55]
        )
        with self._condition:
            self._frame = buf.tobytes()
            self._error = None
            self._version += 1
            self._condition.notify_all()

    def fail(self, detail: str):
        with self._condition:
            self._frame = None
            self._error = detail
            self._version += 1
            self._condition.notify_all()

    def get_frame(self) -> bytes | None:
        with self._condition:
            return self._frame

    def gen(self):
        seen_version = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._version != seen_version,
                    timeout=1.0,
                )
                if self._version == seen_version:
                    continue
                seen_version = self._version
                frame = self._frame
                error = self._error
            if error:
                raise RuntimeError(error)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )


# ── Per-camera frame buffer (push-based frame source) ────────────────────────


class _FrameBuffer:
    """Thread-safe per-camera latest-frame buffer for push-based frame sources.

    Allows serve_bundlesdf to operate without cap_server: external clients
    push RGB+depth frames via ``POST /push_frame``, and the tracking loop
    reads from this buffer instead of (or in addition to) Portal RPC.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._K: np.ndarray | None = None
        self._extrinsics: dict | None = None
        self._seq: int = 0

    def push(self, rgb: np.ndarray, depth: np.ndarray | None = None):
        with self._lock:
            self._rgb = rgb
            if depth is not None:
                self._depth = depth
            self._seq += 1

    def set_intrinsics(self, K: np.ndarray):
        with self._lock:
            self._K = K

    def set_extrinsics(self, extrinsics: dict):
        with self._lock:
            self._extrinsics = extrinsics

    def get(self) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        with self._lock:
            return self._rgb, self._depth, self._seq

    def get_rgb(self) -> np.ndarray | None:
        with self._lock:
            return self._rgb

    def get_K(self) -> np.ndarray | None:
        with self._lock:
            return self._K.copy() if self._K is not None else None

    def get_extrinsics(self) -> dict | None:
        with self._lock:
            return dict(self._extrinsics) if self._extrinsics is not None else None


# ── Per-object state ──────────────────────────────────────────────────────────


class _ObjectState:
    """Per-object state: BundleSdf instance, latest pose, and MJPEG frame buffer."""

    def __init__(
        self,
        text: str,
        camera: str,
        obj_id: int,
        tracker: BundleSdf,
        K: np.ndarray,
        erode_kernel,
        cfg_bt: dict,
        score_thresh: float,
    ):
        self.text = text
        self.camera = camera
        self.obj_id = obj_id
        self.tracker = tracker
        self.K = K
        self.erode_kernel = erode_kernel
        self.cfg_bt = cfg_bt
        self.score_thresh = score_thresh

        self._lock = threading.Lock()
        self._tracker_lock = threading.Lock()  # serialises tracker.run() vs stop()
        self._stopped = False
        self._frame: bytes | None = None
        self.ob_in_cam: np.ndarray | None = None
        self.ob_in_world: np.ndarray | None = None
        self.position_3d_world: np.ndarray | None = None
        self.pose_origin_3d_world: np.ndarray | None = None
        self.position_3d_source: str | None = None
        self.bbox: list[int] | None = None  # latest SAM2 2D bbox [x, y, w, h]
        self.score: float = 0.0
        self.frame_idx: int = 0

        # Occlusion recovery state
        self._bad_streak: int = 0  # consecutive frames with bad mask
        self._occluded: bool = False  # True while waiting for SAM3 to find the object
        self._sam3_check_frame: int = 0  # frame_idx at which next SAM3 check is due

    def push_frame(self, img_rgb: np.ndarray):
        _, buf = cv2.imencode(
            ".jpg", img_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 55]
        )
        with self._lock:
            self._frame = buf.tobytes()

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def get_pose(self) -> dict:
        with self._lock:
            ob_in_cam = self.ob_in_cam
            ob_in_world = self.ob_in_world
            position_3d_world = (
                self.position_3d_world.copy()
                if self.position_3d_world is not None
                else None
            )
            pose_origin_3d_world = (
                self.pose_origin_3d_world.copy()
                if self.pose_origin_3d_world is not None
                else None
            )
            position_3d_source = self.position_3d_source
            bbox = self.bbox
            score = self.score
            frame_idx = self.frame_idx
        if ob_in_cam is None:
            return {
                "tracking": True,
                "bbox": bbox,
                "ob_in_cam": None,
                "ob_in_world": None,
                "position_3d": None,
                "pose_origin_3d": None,
                "position_3d_source": None,
                "quaternion_xyzw": None,
                "half_extents": None,
                "score": score,
                "frame_idx": frame_idx,
            }
        ob_w = ob_in_world if ob_in_world is not None else np.eye(4)
        pose_origin = (
            pose_origin_3d_world.tolist()
            if pose_origin_3d_world is not None
            else ob_w[:3, 3].tolist()
        )
        pos = (
            position_3d_world.tolist() if position_3d_world is not None else pose_origin
        )
        quat = _rotation_matrix_to_quaternion(ob_w[:3, :3])
        rpy = _rotation_matrix_to_rpy(ob_w[:3, :3])
        he = (
            list(self.tracker.half_extents)
            if hasattr(self.tracker, "half_extents")
            else []
        )
        return {
            "tracking": True,
            "bbox": bbox,
            "ob_in_cam": ob_in_cam.tolist(),
            "ob_in_world": ob_w.tolist(),
            "position_3d": [round(x, 5) for x in pos],
            "pose_origin_3d": [round(x, 5) for x in pose_origin],
            "position_3d_source": position_3d_source or "pose_origin",
            "quaternion_xyzw": [round(x, 6) for x in quat],
            "rpy": [round(x, 6) for x in rpy],
            "half_extents": [round(x, 4) for x in he] if he else None,
            "score": score,
            "frame_idx": frame_idx,
        }

    def stop(self):
        with self._tracker_lock:
            self._stopped = True
            tracker = getattr(self, "tracker", None)
            self.tracker = None  # prevent future access
        if tracker is not None:
            try:
                tracker.on_finish()
            except Exception:
                pass
        torch.cuda.empty_cache()


# ── Per-camera tracking loop ──────────────────────────────────────────────────


class _CameraTrackingLoop:
    """Owns SharedSam2Tracker + per-object _ObjectState for one camera.

    Single background thread: grab → SAM2 propagate all → BundleSdf per object.
    """

    def __init__(
        self,
        portal_call,
        camera: str,
        shared_sam2: SharedSam2Tracker,
        K: np.ndarray,
        frame_buffer: _FrameBuffer | None = None,
    ):
        self._portal_call = portal_call
        self._camera = camera
        self._sam2 = shared_sam2
        self._K = K
        self._frame_buffer = frame_buffer
        self._objects: dict[str, _ObjectState] = {}
        self._objects_lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def add_object(
        self,
        name: str,
        text: str,
        rgb0: np.ndarray,
        bbox_xywh: tuple,
        tracker: BundleSdf,
        erode_kernel,
        cfg_bt: dict,
        score_thresh: float,
    ) -> None:
        """Anchor SAM2 on rgb0 with bbox_xywh and register the object."""
        obj_id = self._sam2.add_object(rgb0, bbox_xywh)
        state = _ObjectState(
            text=text,
            camera=self._camera,
            obj_id=obj_id,
            tracker=tracker,
            K=self._K,
            erode_kernel=erode_kernel,
            cfg_bt=cfg_bt,
            score_thresh=score_thresh,
        )
        with self._objects_lock:
            self._objects[name] = state

    def remove_object(self, name: str):
        """Deactivate from SAM2 and shut down the object's BundleSdf."""
        with self._objects_lock:
            state = self._objects.pop(name, None)
        if state is not None:
            self._sam2.deactivate_object(state.obj_id)
            state.stop()

    def get_objects(self) -> dict[str, _ObjectState]:
        with self._objects_lock:
            return dict(self._objects)

    _SAM2_REFRESH_INTERVAL = 100
    _VIS_THREAD: threading.Thread | None = None  # background vis thread
    _OCCLUSION_STREAK_THRESH = (
        999999  # set low (e.g. 10) to enable SAM3 occlusion recovery
    )
    _SAM3_CHECK_INTERVAL = 45  # frames between SAM3 re-detection attempts

    def _run_one_object(
        self, name, state, rgb, color_bgr, depth, mask, score, frame_idx, T_cam_world
    ):
        """Run BundleSdf for one object. Designed for ThreadPoolExecutor dispatch.

        Accepts pre-converted ``color_bgr`` (BGR uint8) to avoid redundant
        ``cv2.cvtColor`` per object.  ``rgb`` is kept for SAM2 re-anchor.
        """
        # ── Occlusion state machine ────────────────────────────────────────────
        is_bad = score < state.score_thresh or int((mask > 0).sum()) < 200

        if state._occluded:
            # Periodically ask SAM3 whether the object is visible again.
            if frame_idx >= state._sam3_check_frame:
                state._sam3_check_frame = frame_idx + self._SAM3_CHECK_INTERVAL
                try:
                    _, bbox_xywh, sam3_score = text_to_mask(rgb, state.text)
                    print(
                        f"[{self._camera}/{name}] frame {frame_idx:04d}: "
                        f"SAM3 found '{state.text}' (score={sam3_score:.3f}), re-anchoring"
                    )
                    self._sam2.re_anchor_object(rgb, state.obj_id, bbox_xywh)
                    state._occluded = False
                    state._bad_streak = 0
                except RuntimeError:
                    print(
                        f"[{self._camera}/{name}] frame {frame_idx:04d}: "
                        f"SAM3 still cannot find '{state.text}'"
                    )
            # Hold last known pose; skip BundleSDF while occluded.
            return name, 0.0

        if is_bad:
            state._bad_streak += 1
            if state._bad_streak >= self._OCCLUSION_STREAK_THRESH:
                state._occluded = True
                state._sam3_check_frame = frame_idx + self._SAM3_CHECK_INTERVAL
                print(
                    f"[{self._camera}/{name}] frame {frame_idx:04d}: "
                    f"occluded after {state._bad_streak} bad frames — "
                    f"SAM3 will check in {self._SAM3_CHECK_INTERVAL} frames"
                )
            # Skip BundleSDF during bad streak to avoid poisoning the pose graph.
            return name, 0.0

        state._bad_streak = 0
        # ── End occlusion state machine ───────────────────────────────────────

        mask_proc = (
            cv2.erode(mask, state.erode_kernel)
            if state.erode_kernel is not None
            else mask
        )

        ob_in_cam = None
        # Use uneroded mask for depth validity check — eroding a
        # small/fragmented SAM2 mask can eliminate valid depth pixels.
        if has_valid_depth(
            depth, mask, zfar=state.cfg_bt["depth_processing"]["zfar"], label=name[:15]
        ):
            # Each object needs its own depth copy — BundleSdf's percentile
            # denoise modifies depth in-place (depth[depth >= thres] = 0),
            # which would corrupt other objects running in parallel.
            depth_copy = depth.copy()
            with state._tracker_lock:
                if state._stopped:
                    return name, 0.0
                try:
                    ob_in_cam = state.tracker.run(
                        color_bgr,
                        depth_copy,
                        state.K,
                        id_str=f"{frame_idx:04d}",
                        mask=mask_proc,
                        occ_mask=None,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[{self._camera}/{name}] frame {frame_idx:04d}: CUDA OOM")
                except IndexError:
                    print(
                        f"[{self._camera}/{name}] frame {frame_idx:04d}: "
                        f"BundleTrack keyframes empty"
                    )

        # Compute 2D bbox from SAM2 mask
        ys, xs = np.where(mask > 0)
        bbox_xywh = None
        if len(xs) >= 5:
            bbox_xywh = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min()),
                int(ys.max() - ys.min()),
            ]

        with state._lock:
            if ob_in_cam is not None:
                state.ob_in_cam = ob_in_cam
                state.ob_in_world = T_cam_world @ ob_in_cam
                center_cam = state.tracker.estimate_center_cam(
                    ob_in_cam, depth=depth, mask=mask, K=state.K
                )
                center_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
                state.position_3d_world = (
                    T_cam_world[:3, :3] @ center_cam + T_cam_world[:3, 3]
                )
                state.pose_origin_3d_world = state.ob_in_world[:3, 3].copy()
                state.position_3d_source = (
                    "reference_model_obb"
                    if getattr(state.tracker, "center_local", None) is not None
                    else "masked_depth_median"
                )
            state.bbox = bbox_xywh
            state.score = score
            state.frame_idx = frame_idx
            cur_ob = state.ob_in_cam

        return name, time.time()

    @staticmethod
    def _push_vis(state, rgb, mask, frame_idx, score, cur_ob):
        """Generate and push visualization frame. Called from background thread."""
        try:
            vis = _draw_mask_overlay(
                rgb,
                mask,
                frame_idx=frame_idx,
                score=score,
                ob_in_cam=cur_ob,
                K_vis=state.K,
            )
            state.push_frame(vis)
        except Exception:
            pass

    def _loop(self):
        frame_idx = 1
        _last_rgb = None  # frame deduplication (portal mode — identity check)
        _last_seq = -1  # frame deduplication (push mode — sequence number)

        with ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="bundlesdf"
        ) as executor:
            while self._running:
                t0 = time.time()

                # ── Acquire frame: push buffer first, portal fallback ─────
                rgb = depth = None
                if self._frame_buffer is not None:
                    buf_rgb, buf_depth, seq = self._frame_buffer.get()
                    if buf_rgb is not None and buf_depth is not None:
                        if seq == _last_seq:
                            time.sleep(0.005)
                            continue
                        rgb, depth = buf_rgb, buf_depth
                        _last_seq = seq

                if rgb is None or depth is None:
                    rgb = self._portal_call(
                        lambda p: np.asarray(p.get_camera_image(self._camera).result())
                    )
                    depth = self._portal_call(
                        lambda p: np.asarray(p.get_camera_depth(self._camera).result())
                    )
                    if rgb is None or depth is None:
                        time.sleep(0.1)
                        continue
                    if rgb is _last_rgb:
                        time.sleep(0.005)
                        continue
                    _last_rgb = rgb

                with self._objects_lock:
                    snapshot = dict(self._objects)

                if not snapshot:
                    time.sleep(0.05)
                    continue

                # ONE SAM2 forward pass for ALL objects on this camera
                try:
                    all_masks = self._sam2.propagate(rgb)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(
                        f"[{self._camera}] frame {frame_idx:04d}: CUDA OOM in SAM2 propagate"
                    )
                    time.sleep(0.1)
                    frame_idx += 1
                    continue

                t_sam2 = time.time() - t0

                # ── Extrinsics: push buffer first, portal fallback ────
                extr = None
                if self._frame_buffer is not None:
                    extr = self._frame_buffer.get_extrinsics()
                if extr is None:
                    extr = self._portal_call(
                        lambda p: p.get_camera_extrinsics(self._camera).result()
                    )
                T_cam_world = (
                    _build_SE3(extr, self._camera) if extr is not None else np.eye(4)
                )

                # Pre-compute once for all objects (avoids per-object cvtColor)
                color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Collect tasks for objects that have a mask this frame
                obj_tasks = []
                for name, state in snapshot.items():
                    mask_score = all_masks.get(state.obj_id)
                    if mask_score is None:
                        continue
                    mask, score = mask_score
                    obj_tasks.append(
                        (
                            name,
                            state,
                            rgb,
                            color_bgr,
                            depth,
                            mask,
                            score,
                            frame_idx,
                            T_cam_world,
                        )
                    )

                t_bt: dict[str, float] = {}
                if len(obj_tasks) <= 1:
                    # Single object: avoid thread-pool overhead entirely
                    for args in obj_tasks:
                        n, _ = self._run_one_object(*args)
                        t_bt[n] = time.time() - t0 - t_sam2
                else:
                    # Multiple objects: run BundleSdf trackers in parallel
                    futures = [
                        executor.submit(self._run_one_object, *args)
                        for args in obj_tasks
                    ]
                    for fut in as_completed(futures):
                        n, _ = fut.result()
                        t_bt[n] = time.time() - t0 - t_sam2

                elapsed = time.time() - t0
                fps = 1.0 / elapsed if elapsed > 0 else float("inf")
                bt_str = "  ".join(
                    f"{n[:10]}={int(t * 1000)}ms" for n, t in t_bt.items()
                )
                print(
                    f"[{self._camera}] frame {frame_idx:04d}  "
                    f"{len(obj_tasks)}/{len(snapshot)} objs  "
                    f"sam2={int(t_sam2 * 1000)}ms  bt=[{bt_str}]  "
                    f"total={int(elapsed * 1000)}ms  ({fps:.1f} fps)"
                )

                # ── Update visualization (skip every other frame) ─────────
                # if frame_idx % 2 == 0:
                for args in obj_tasks:
                    _name, _state, _rgb, _bgr, _depth, _mask, _score, _fidx, _Tcw = args
                    with _state._lock:
                        _cur_ob = _state.ob_in_cam
                    self._push_vis(_state, _rgb, _mask, _fidx, _score, _cur_ob)

                # ── Periodic SAM2 session refresh ─────────────────────────
                if frame_idx > 0 and frame_idx % self._SAM2_REFRESH_INTERVAL == 0:
                    refresh_bboxes: dict[int, tuple] = {}
                    for name, state in snapshot.items():
                        ms = all_masks.get(state.obj_id)
                        if ms is not None:
                            m, _ = ms
                            ys, xs = np.where(m > 0)
                            if len(xs) >= 20:
                                refresh_bboxes[state.obj_id] = (
                                    int(xs.min()),
                                    int(ys.min()),
                                    int(xs.max() - xs.min()),
                                    int(ys.max() - ys.min()),
                                )
                        elif state.ob_in_cam is not None:
                            uvw = state.K @ state.ob_in_cam[:3, 3]
                            cx = int(uvw[0] / uvw[2])
                            cy = int(uvw[1] / uvw[2])
                            H, W = rgb.shape[:2]
                            r = 60
                            refresh_bboxes[state.obj_id] = (
                                max(0, cx - r),
                                max(0, cy - r),
                                min(2 * r, W - max(0, cx - r)),
                                min(2 * r, H - max(0, cy - r)),
                            )
                    if refresh_bboxes:
                        self._sam2.refresh_session(rgb, refresh_bboxes)
                        print(
                            f"[{self._camera}] SAM2 session refreshed at "
                            f"frame {frame_idx} "
                            f"({len(refresh_bboxes)}/{len(snapshot)} "
                            f"objects re-anchored)"
                        )

                frame_idx += 1

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_all(self):
        """Stop the tracking thread and clean up all remaining objects."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        with self._objects_lock:
            for state in self._objects.values():
                state.stop()
            self._objects.clear()
        torch.cuda.empty_cache()


# ── HTML UI ───────────────────────────────────────────────────────────────────

_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BundleSDF Live — Multi-Object</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #eee; font-family: sans-serif; }
  #add-bar {
    padding: 10px;
    background: #222;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .inp {
    padding: 7px 11px;
    background: #333;
    color: #eee;
    border: 1px solid #555;
    border-radius: 4px;
    font-size: 13px;
    outline: none;
  }
  .inp:focus { border-color: #0a84ff; }
  .inp::placeholder { color: #777; }
  #text-inp { flex: 2; min-width: 200px; }
  #name-inp { flex: 1; min-width: 120px; }
  #camera-sel { min-width: 80px; }
  #add-btn {
    padding: 7px 18px;
    background: #0a84ff;
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
  }
  #add-btn:disabled { background: #555; cursor: default; }
  #status { font-size: 12px; color: #aaa; flex: 1; min-width: 120px; }
  #page { padding: 10px; display: grid; gap: 12px; }
  .section {
    border: 1px solid #303030;
    border-radius: 6px;
    background: #181818;
    overflow: hidden;
  }
  .section-title {
    padding: 8px 10px;
    background: #202020;
    border-bottom: 1px solid #303030;
    color: #bbb;
    font-size: 13px;
  }
  #preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 10px;
    padding: 10px;
  }
  .preview-card {
    border: 1px solid #333;
    border-radius: 5px;
    background: #101010;
    overflow: hidden;
    cursor: pointer;
  }
  .preview-card.selected { border-color: #0a84ff; }
  .preview-card:focus {
    outline: 2px solid #0a84ff;
    outline-offset: 2px;
  }
  .preview-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 8px;
    background: #242424;
    font-size: 12px;
    color: #aaa;
  }
  .preview-name { color: #eee; font-weight: 600; }
  .preview-card img {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: contain;
    background: #000;
  }
  #detections-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  #detections-table th, #detections-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #333;
    text-align: left;
    vertical-align: middle;
  }
  #detections-table th { background: #1e1e1e; color: #aaa; font-weight: normal; }
  .end-btn {
    padding: 5px 12px;
    background: #ff4444;
    color: #fff;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
  }
  .end-btn:hover { background: #cc0000; }
  #stream-section { padding: 10px; }
  #composite-stream { max-width: 100%; border-radius: 4px; background: #000; display: block; }
  .pos-cell { font-family: monospace; font-size: 11px; }
  .score-cell { font-family: monospace; }
</style>
</head>
<body>

<div id="add-bar">
  <input id="text-inp" class="inp" type="text"
         placeholder="Object description (e.g. yellow mustard bottle)"
         oninput="onInput()" onkeydown="onKey(event)">
  <input id="name-inp" class="inp" type="text"
         placeholder="Short name (optional)">
  <select id="camera-sel" class="inp">
    <option value="top">top</option>
    <option value="left">left</option>
    <option value="right">right</option>
  </select>
  <button id="add-btn" disabled onclick="addDetection()">Add Detection</button>
  <span id="status">Describe an object and click Add Detection.</span>
</div>

<div id="page">
  <section class="section">
    <div class="section-title">Live camera views</div>
    <div id="preview-grid">
      <div class="preview-card" data-camera="top" role="button" tabindex="0" onclick="selectCamera('top')" onkeydown="onPreviewKey(event, 'top')">
        <div class="preview-head"><span class="preview-name">top</span><span>click to target detections</span></div>
        <img src="/preview/top" alt="top camera preview">
      </div>
      <div class="preview-card" data-camera="left" role="button" tabindex="0" onclick="selectCamera('left')" onkeydown="onPreviewKey(event, 'left')">
        <div class="preview-head"><span class="preview-name">left</span><span>click to target detections</span></div>
        <img src="/preview/left" alt="left camera preview">
      </div>
      <div class="preview-card" data-camera="right" role="button" tabindex="0" onclick="selectCamera('right')" onkeydown="onPreviewKey(event, 'right')">
        <div class="preview-head"><span class="preview-name">right</span><span>click to target detections</span></div>
        <img src="/preview/right" alt="right camera preview">
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-title">Active detections</div>
    <table id="detections-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Camera</th>
          <th>Frame</th>
          <th>Score</th>
          <th>Position (m)</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="detections-body"></tbody>
    </table>
  </section>

  <section class="section">
    <div class="section-title">Tracking composite</div>
    <div id="stream-section">
      <img id="composite-stream" src="/stream_composite">
    </div>
  </section>
</div>

<script>
const textInp   = document.getElementById('text-inp');
const nameInp   = document.getElementById('name-inp');
const cameraSel = document.getElementById('camera-sel');
const addBtn    = document.getElementById('add-btn');
const status    = document.getElementById('status');
const tbody     = document.getElementById('detections-body');

function onInput() {
  addBtn.disabled = textInp.value.trim().length === 0;
}

function onKey(e) {
  if (e.key === 'Enter' && !addBtn.disabled) addDetection();
}

function selectCamera(camera) {
  cameraSel.value = camera;
  syncSelectedPreview();
}

function onPreviewKey(e, camera) {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    selectCamera(camera);
  }
}

function syncSelectedPreview() {
  document.querySelectorAll('.preview-card').forEach(card => {
    card.classList.toggle('selected', card.dataset.camera === cameraSel.value);
  });
}

function addDetection() {
  const text   = textInp.value.trim();
  const name   = nameInp.value.trim() || undefined;
  const camera = cameraSel.value;
  if (!text) return;
  addBtn.disabled = true;
  status.textContent = 'Adding "' + text + '" on camera ' + camera + '\u2026';
  fetch('/add_detection', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, name, camera}),
  }).then(r => {
    if (r.ok) {
      textInp.value = '';
      nameInp.value = '';
      onInput();
      status.textContent = 'Detection queued \u2014 waiting for model to initialise\u2026';
    } else {
      r.json().then(d => { status.textContent = 'Error: ' + (d.detail || r.statusText); });
      addBtn.disabled = false;
    }
  }).catch(err => {
    status.textContent = 'Error: ' + err;
    addBtn.disabled = false;
  });
}

function endDetection(name) {
  fetch('/end_detection/' + encodeURIComponent(name), {method: 'POST'})
    .then(() => {
      const row = document.querySelector('tr[data-name="' + CSS.escape(name) + '"]');
      if (row) row.remove();
    });
}

function fmtPos(det) {
  if (!det || !det.position_3d) return '\u2014';
  const p = det.position_3d;
  return '[' + p.map(x => x.toFixed(3)).join(', ') + ']';
}

function fmtScore(det) {
  if (!det || det.score == null) return '\u2014';
  return det.score.toFixed(3);
}

function fmtFrame(det) {
  if (!det || det.frame_idx == null) return '\u2014';
  return det.frame_idx;
}

function buildRow(name, det) {
  const tr = document.createElement('tr');
  tr.dataset.name = name;
  tr.innerHTML = `
    <td>${escHtml(name)}</td>
    <td>${escHtml(det.camera || '')}</td>
    <td class="score-cell" data-frame>${fmtFrame(det)}</td>
    <td class="score-cell" data-score>${fmtScore(det)}</td>
    <td class="pos-cell" data-pos>${fmtPos(det)}</td>
    <td><button class="end-btn" onclick="endDetection('${escHtml(name)}')">End</button></td>
  `;
  return tr;
}

function updateRow(tr, det) {
  const poseTd  = tr.querySelector('[data-pos]');
  const scoreTd = tr.querySelector('[data-score]');
  const frameTd = tr.querySelector('[data-frame]');
  if (poseTd)  poseTd.textContent  = fmtPos(det);
  if (scoreTd) scoreTd.textContent = fmtScore(det);
  if (frameTd) frameTd.textContent = fmtFrame(det);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function refreshTable() {
  fetch('/list_detections').then(r => r.json()).then(data => {
    const dets  = data.detections || {};
    const names = Object.keys(dets);
    [...tbody.querySelectorAll('tr')].forEach(row => {
      if (!names.includes(row.dataset.name)) row.remove();
    });
    names.forEach(name => {
      let row = tbody.querySelector('tr[data-name="' + CSS.escape(name) + '"]');
      if (!row) {
        row = buildRow(name, dets[name]);
        tbody.appendChild(row);
      } else {
        updateRow(row, dets[name]);
      }
    });
  }).catch(() => {});
}

setInterval(refreshTable, 1000);
cameraSel.addEventListener('change', syncSelectedPreview);
syncSelectedPreview();
refreshTable();
</script>

</body>
</html>
"""


_MULTIVIEW_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BundleSDF Camera Multiview</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
  header {
    padding: 14px 18px;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: baseline;
    gap: 16px;
  }
  h1 { margin: 0; font-size: 20px; }
  a { color: #66aaff; text-decoration: none; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 12px;
    padding: 12px;
  }
  .view {
    background: #1b1b1b;
    border: 1px solid #333;
    border-radius: 6px;
    overflow: hidden;
  }
  .view h2 {
    margin: 0;
    padding: 8px 10px;
    font-size: 16px;
    background: #222;
  }
  img {
    display: block;
    width: 100%;
    height: auto;
    background: #000;
  }
</style>
</head>
<body>
  <header>
    <h1>BundleSDF Camera Multiview</h1>
    <a href="/">tracking UI</a>
    <a href="/preview">default preview</a>
  </header>
  <main class="grid">
    <section class="view"><h2>top</h2><img src="/preview/top"></section>
    <section class="view"><h2>left</h2><img src="/preview/left"></section>
    <section class="view"><h2>right</h2><img src="/preview/right"></section>
  </main>
</body>
</html>
"""


# ── FastAPI app ───────────────────────────────────────────────────────────────


def create_app(cap_server_host: str, cap_server_port: int, camera: str) -> FastAPI:
    app = FastAPI(title="serve_bundlesdf")
    default_camera = str(camera).strip().lower() or "top"
    valid_preview_cameras = _resolve_preview_cameras(default_camera)
    valid_preview_camera_set = set(valid_preview_cameras)
    if default_camera not in valid_preview_camera_set:
        default_camera = valid_preview_cameras[0]

    # Per-camera tracking loops and session registry
    _camera_loops: dict[str, _CameraTrackingLoop] = {}
    _name_to_camera: dict[str, str] = {}
    _frame_buffers: dict[str, _FrameBuffer] = {}  # push-based frame sources
    _registry_lock = threading.Lock()  # protects _camera_loops, _name_to_camera
    _detection_lock = threading.Lock()  # serializes add_detection (SAM3 + SAM2 init)

    _portal_addr = f"{cap_server_host}:{cap_server_port}"
    _portal_client = None
    _portal_lock = threading.Lock()
    _portal_next_retry = 0.0
    _portal_retry_interval = 1.0
    _preview_streamers: dict[str, _PreviewStreamer] = {}
    _preview_threads: set[str] = set()
    _preview_lock = threading.Lock()
    _preview_overlay_metadata = _env_flag(
        "OPENFORGE_BUNDLESDF_PREVIEW_OVERLAY_METADATA", False
    )
    try:
        _preview_duplicate_limit = max(
            1,
            int(os.environ.get("OPENFORGE_BUNDLESDF_PREVIEW_DUPLICATE_LIMIT", "150")),
        )
    except ValueError:
        _preview_duplicate_limit = 150

    def _reset_portal():
        nonlocal _portal_client
        nonlocal _portal_next_retry
        _portal_client = None
        _portal_next_retry = time.time() + _portal_retry_interval

    def _get_portal():
        nonlocal _portal_client
        if time.time() < _portal_next_retry:
            return None
        if _portal_client is None:
            with _portal_lock:
                if _portal_client is None:
                    import portal

                    _portal_client = portal.Client(_portal_addr)
        return _portal_client

    def _portal_call(
        call_fn, *, raise_http: bool = False, err_msg: str = "cap server unavailable"
    ):
        portal_client = _get_portal()
        if portal_client is None:
            if raise_http:
                raise HTTPException(status_code=503, detail=err_msg)
            return None
        try:
            return call_fn(portal_client)
        except Exception:
            _reset_portal()
            if raise_http:
                raise HTTPException(status_code=503, detail=err_msg)
            return None

    def _normalize_preview_camera(raw_camera: str | None) -> str:
        active_camera = str(raw_camera or default_camera).strip().lower()
        if active_camera not in valid_preview_camera_set:
            expected = ", ".join(valid_preview_cameras)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown camera {raw_camera!r}; "
                    f"expected one of {expected}"
                ),
            )
        return active_camera

    def _read_live_preview_rgb(
        active_camera: str, *, raise_http: bool = False
    ) -> np.ndarray | None:
        err_msg = f"live camera {active_camera!r} unavailable for preview"
        try:
            timeout_s = max(
                0.1,
                float(os.environ.get("OPENFORGE_BUNDLESDF_PREVIEW_PORTAL_TIMEOUT_S", "2.0")),
            )
        except ValueError:
            timeout_s = 2.0
        rgb = _portal_call(
            lambda p: p.get_camera_image(active_camera).result(timeout=timeout_s),
            raise_http=raise_http,
            err_msg=err_msg,
        )
        if rgb is None:
            if raise_http:
                raise HTTPException(status_code=503, detail=err_msg)
            return None
        arr = np.asarray(rgb)
        if arr.ndim < 2:
            if raise_http:
                raise HTTPException(status_code=503, detail=err_msg)
            return None
        return np.ascontiguousarray(arr)

    def _annotate_preview_rgb(
        active_camera: str, rgb: np.ndarray, frame_seq: int
    ) -> np.ndarray:
        if not _preview_overlay_metadata:
            return rgb
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return rgb
        out = np.ascontiguousarray(arr[..., :3].copy())
        now = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(now))
        millis = int((now % 1.0) * 1000)
        label = f"LIVE {active_camera} #{frame_seq:06d} {stamp}.{millis:03d}"
        cv2.rectangle(out, (6, 7), (420, 34), (0, 0, 0), thickness=-1)
        cv2.putText(
            out,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    def _preview_loop(active_camera: str, streamer: _PreviewStreamer):
        frame_seq = 0
        duplicate_streak = 0
        last_raw_rgb: np.ndarray | None = None
        while True:
            # Preview/recording must be live-only. Do not fall back to pushed
            # buffers: those can remain populated from old runs and create stale
            # MP4 evidence. If the Portal camera is unavailable, the route
            # preflight returns HTTP 503 and the loop simply stops publishing.
            rgb = _read_live_preview_rgb(active_camera)
            if rgb is None:
                streamer.fail(f"live camera {active_camera!r} unavailable for preview")
            else:
                if last_raw_rgb is not None and np.array_equal(rgb, last_raw_rgb):
                    duplicate_streak += 1
                else:
                    duplicate_streak = 0
                    last_raw_rgb = np.array(rgb, copy=True)
                if duplicate_streak >= _preview_duplicate_limit:
                    streamer.fail(
                        f"live camera {active_camera!r} repeated "
                        f"{duplicate_streak + 1} identical raw frames"
                    )
                else:
                    frame_seq += 1
                    streamer.push(
                        _annotate_preview_rgb(active_camera, rgb, frame_seq)
                    )
            time.sleep(0.05)

    def _get_preview_streamer(active_camera: str) -> _PreviewStreamer:
        rgb = _read_live_preview_rgb(active_camera, raise_http=True)
        with _preview_lock:
            streamer = _preview_streamers.get(active_camera)
            if streamer is None:
                streamer = _PreviewStreamer()
                _preview_streamers[active_camera] = streamer
            streamer.push(_annotate_preview_rgb(active_camera, rgb, 0))
            if active_camera not in _preview_threads:
                threading.Thread(
                    target=_preview_loop,
                    args=(active_camera, streamer),
                    daemon=True,
                    name=f"bundlesdf-preview-{active_camera}",
                ).start()
                _preview_threads.add(active_camera)
            return streamer

    _get_preview_streamer(default_camera)

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "default_camera": default_camera,
            "preview_cameras": list(valid_preview_cameras),
        }

    @app.get("/")
    def index():
        return HTMLResponse(_render_index_html(valid_preview_cameras, default_camera))

    @app.get("/preview")
    def preview(camera: str | None = None):
        active_camera = _normalize_preview_camera(camera)
        return StreamingResponse(
            _get_preview_streamer(active_camera).gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/preview/{camera_name}")
    def preview_by_camera(camera_name: str):
        active_camera = _normalize_preview_camera(camera_name)
        return StreamingResponse(
            _get_preview_streamer(active_camera).gen(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/multiview")
    def multiview():
        for active_camera in valid_preview_cameras:
            _get_preview_streamer(active_camera)
        return HTMLResponse(_render_multiview_html(valid_preview_cameras))

    @app.get("/stream/{name}")
    def stream_by_name(name: str):
        def _gen():
            while True:
                frame = None
                with _registry_lock:
                    cam = _name_to_camera.get(name)
                    loop = _camera_loops.get(cam) if cam else None
                if loop is not None:
                    state = loop.get_objects().get(name)
                    if state is not None:
                        frame = state.get_frame()
                if frame:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                time.sleep(0.03)

        return StreamingResponse(
            _gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/stream_composite")
    def stream_composite():
        def _gen():
            while True:
                tiles = []
                with _registry_lock:
                    loops = dict(_camera_loops)
                for _cam, loop in loops.items():
                    for obj_name, state in loop.get_objects().items():
                        raw = state.get_frame()
                        if raw is None:
                            continue
                        arr = cv2.imdecode(
                            np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
                        )
                        if arr is not None:
                            cv2.putText(
                                arr,
                                obj_name[:20],
                                (4, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            tiles.append(arr)
                if tiles:
                    h = min(t.shape[0] for t in tiles)
                    resized = [
                        cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
                        for t in tiles
                    ]
                    composite = np.concatenate(resized, axis=1)
                    _, buf = cv2.imencode(
                        ".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 55]
                    )
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                else:
                    # No active sessions — fall back to raw camera preview
                    frame = _get_preview_streamer(default_camera).get_frame()
                    if frame is not None:
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                time.sleep(0.05)

        return StreamingResponse(
            _gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # (Snapshot routes removed — use MJPEG streams directly for lower latency)

    @app.post("/push_frame", response_model=PushFrameResponse)
    def push_frame(req: PushFrameRequest):
        """Push an RGB (+optional depth) frame for a camera.

        This allows serve_bundlesdf to operate without cap_server: external
        clients push frames, and the tracking loop reads from the buffer.
        """
        cam = req.camera or camera
        if cam not in _frame_buffers:
            _frame_buffers[cam] = _FrameBuffer()
        fb = _frame_buffers[cam]

        rgb = _decode_rgb(req.image_base64)
        depth = _decode_depth(req.depth_base64) if req.depth_base64 else None
        fb.push(rgb, depth)

        if req.intrinsics:
            fb.set_intrinsics(_build_K(req.intrinsics))
        if req.extrinsics:
            fb.set_extrinsics(req.extrinsics)

        _, _, seq = fb.get()
        return PushFrameResponse(ok=True, camera=cam, seq=seq)

    @app.post("/add_detection", response_model=AddDetectionResponse)
    def add_detection(req: AddDetectionRequest):
        with _detection_lock:
            name = _make_name(req.text, req.name)
            active_camera = req.camera or camera

            with _registry_lock:
                if name in _name_to_camera:
                    raise HTTPException(
                        status_code=409, detail=f"Detection '{name}' already active"
                    )

            out_folder = req.out_folder or os.path.join(
                tempfile.gettempdir(), "bundlesdf", name
            )
            cfg_track_dir = build_configs(out_folder, req.debug_level)
            cfg_bt = yaml.load(open(cfg_track_dir), Loader=yaml.Loader)
            erode_size = cfg_bt.get("erode_mask", 0)
            erode_kernel = (
                np.ones((erode_size, erode_size), np.uint8) if erode_size > 0 else None
            )

            # ── Resolve initial frame: request body → frame buffer → portal
            fb = _frame_buffers.get(active_camera)
            if req.image_base64:
                rgb0 = _decode_rgb(req.image_base64)
            elif fb is not None and fb.get_rgb() is not None:
                rgb0 = fb.get_rgb()
            else:
                rgb0 = _portal_call(
                    lambda p: np.asarray(p.get_camera_image(active_camera).result()),
                    raise_http=True,
                    err_msg="No image available (provide image_base64, push frames, or start cap_server)",
                )

            # ── Resolve intrinsics: request body → frame buffer → portal
            if req.intrinsics:
                K = _build_K(req.intrinsics)
            elif fb is not None and fb.get_K() is not None:
                K = fb.get_K()
            else:
                intrinsics = _portal_call(
                    lambda p: p.get_camera_intrinsics(active_camera).result(),
                    raise_http=True,
                    err_msg="No intrinsics available (provide intrinsics, push frames with intrinsics, or start cap_server)",
                )
                K = _build_K([float(x) for x in intrinsics])

            # SAM3 single-image detection → bbox
            print(
                f"[serve_bundlesdf] SAM3 detecting '{req.text}' on camera '{active_camera}' …"
            )
            try:
                _mask_01, bbox_xywh, first_score = text_to_mask(rgb0, req.text)
            except RuntimeError as e:
                raise HTTPException(status_code=404, detail=str(e))
            print(
                f"[serve_bundlesdf] '{name}': initial bbox={bbox_xywh}  score={first_score:.3f}"
            )

            # SAM3 still loaded — grab fresh frame and re-detect (inference only, ~2-5 s).
            # rgb0's bbox is stale after SAM3's load time; re-detecting ensures the SAM2
            # anchor bbox matches where the object actually is in the current frame.
            try:
                if fb is not None and fb.get_rgb() is not None:
                    rgb_fresh = fb.get_rgb()
                else:
                    rgb_fresh = _portal_call(
                        lambda p: np.asarray(
                            p.get_camera_image(active_camera).result()
                        ),
                    )
                if rgb_fresh is None:
                    rgb_fresh = rgb0
                mask_fresh, bbox_fresh, score_fresh = text_to_mask(rgb_fresh, req.text)
                print(
                    f"[serve_bundlesdf] '{name}': fresh bbox={bbox_fresh}  score={score_fresh:.3f}"
                )
            except (RuntimeError, Exception) as e:
                print(
                    f"[serve_bundlesdf] '{name}': fresh re-detection failed ({e}), using initial"
                )
                rgb_fresh, mask_fresh, bbox_fresh, score_fresh = (
                    rgb0,
                    _mask_01,
                    bbox_xywh,
                    first_score,
                )

            # Free SAM3 VRAM before SAM2
            _free_sam3_single_image()

            # Load BundleSdf for this object
            print(f"[serve_bundlesdf] Loading BundleSdf for '{name}' …")
            tracker = BundleSdf(cfg_track_dir=cfg_track_dir)

            # Get or create the per-camera tracking loop
            with _registry_lock:
                if active_camera not in _camera_loops:
                    print(
                        f"[serve_bundlesdf] Creating SharedSam2Tracker for '{active_camera}' …"
                    )
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    shared_sam2 = SharedSam2Tracker(device=device)
                    loop = _CameraTrackingLoop(
                        portal_call=_portal_call,
                        camera=active_camera,
                        shared_sam2=shared_sam2,
                        K=K,
                        frame_buffer=fb,
                    )
                    _camera_loops[active_camera] = loop
                    loop.add_object(
                        name,
                        req.text,
                        rgb_fresh,
                        bbox_fresh,
                        tracker,
                        erode_kernel,
                        cfg_bt,
                        req.score_thresh,
                    )
                    loop.start()
                else:
                    loop = _camera_loops[active_camera]
                    loop.add_object(
                        name,
                        req.text,
                        rgb_fresh,
                        bbox_fresh,
                        tracker,
                        erode_kernel,
                        cfg_bt,
                        req.score_thresh,
                    )

                _name_to_camera[name] = active_camera

        return AddDetectionResponse(
            name=name,
            bbox=list(bbox_fresh),
            first_score=round(score_fresh, 4),
        )

    @app.get("/get_detection/{name}", response_model=PoseResponse)
    def get_detection(name: str):
        with _registry_lock:
            cam = _name_to_camera.get(name)
            loop = _camera_loops.get(cam) if cam else None
        if loop is None:
            raise HTTPException(status_code=404, detail=f"Detection '{name}' not found")
        state = loop.get_objects().get(name)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Detection '{name}' not found")
        return PoseResponse(**state.get_pose())

    @app.post("/end_detection/{name}", response_model=EndDetectionResponse)
    def end_detection(name: str):
        with _registry_lock:
            cam = _name_to_camera.pop(name, None)
            loop = _camera_loops.get(cam) if cam else None
        if loop is None:
            return EndDetectionResponse(ok=False)

        loop.remove_object(
            name
        )  # deactivates from SAM2, stops BundleSdf — slow, no lock

        # If this camera loop has no more objects, tear it down
        if not loop.get_objects():
            with _registry_lock:
                _camera_loops.pop(cam, None)
            loop.stop_all()  # stops thread, cleans up — slow, no lock

        return EndDetectionResponse(ok=True)

    @app.get("/list_detections", response_model=ListDetectionsResponse)
    def list_detections():
        with _registry_lock:
            loops_snap = dict(_camera_loops)
            names_snap = dict(_name_to_camera)

        result: dict[str, DetectionEntry] = {}
        for name, cam in names_snap.items():
            loop = loops_snap.get(cam)
            if loop is None:
                continue
            state = loop.get_objects().get(name)
            if state is None:
                continue
            pose = state.get_pose()
            result[name] = DetectionEntry(
                name=name,
                text=state.text,
                camera=cam,
                tracking=True,
                score=pose["score"],
                frame_idx=pose["frame_idx"],
                position_3d=pose.get("position_3d"),
                pose_origin_3d=pose.get("pose_origin_3d"),
                position_3d_source=pose.get("position_3d_source"),
                quaternion_xyzw=pose.get("quaternion_xyzw"),
            )

        return ListDetectionsResponse(detections=result)

    @app.post("/reset_state", response_model=ResetStateResponse)
    def reset_state():
        with _registry_lock:
            names_snap = dict(_name_to_camera)
            loops_snap = dict(_camera_loops)
            _name_to_camera.clear()
            _frame_buffers.clear()

        ended = 0
        for name, cam in names_snap.items():
            loop = loops_snap.get(cam)
            if loop is None:
                continue
            loop.remove_object(name)
            ended += 1

        _reset_portal()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ResetStateResponse(
            ok=True,
            ended_detections=ended,
            retained_camera_loops=len(loops_snap),
        )

    @app.post("/single_frame_pose", response_model=SingleFramePoseResponse)
    def single_frame_pose(req: SingleFramePoseRequest):
        active_camera = req.camera or camera

        out_folder = req.out_folder or os.path.join(
            tempfile.gettempdir(), "bundlesdf", "single_frame"
        )
        cfg_track_dir = build_configs(out_folder, req.debug_level)
        cfg_bt = yaml.load(open(cfg_track_dir), Loader=yaml.Loader)
        erode_size = cfg_bt.get("erode_mask", 0)
        erode_kernel = (
            np.ones((erode_size, erode_size), np.uint8) if erode_size > 0 else None
        )

        print("[serve_bundlesdf] Loading BundleSdf (single-frame) …")
        tracker = BundleSdf(cfg_track_dir=cfg_track_dir)

        try:
            fb = _frame_buffers.get(active_camera)
            no_source_msg = (
                "No {} available (provide in request, push frames, or start cap_server)"
            )

            # ── Resolve RGB
            if req.image_base64:
                rgb = _decode_rgb(req.image_base64)
            elif fb is not None and fb.get_rgb() is not None:
                rgb = fb.get_rgb()
            else:
                rgb = _portal_call(
                    lambda p: np.asarray(p.get_camera_image(active_camera).result()),
                    raise_http=True,
                    err_msg=no_source_msg.format("image"),
                )

            # ── Resolve depth
            if req.depth_base64:
                depth = _decode_depth(req.depth_base64)
            elif fb is not None:
                _, buf_depth, _ = fb.get()
                depth = (
                    buf_depth
                    if buf_depth is not None
                    else _portal_call(
                        lambda p: np.asarray(
                            p.get_camera_depth(active_camera).result()
                        ),
                        raise_http=True,
                        err_msg=no_source_msg.format("depth"),
                    )
                )
            else:
                depth = _portal_call(
                    lambda p: np.asarray(p.get_camera_depth(active_camera).result()),
                    raise_http=True,
                    err_msg=no_source_msg.format("depth"),
                )

            # ── Resolve intrinsics
            if req.intrinsics:
                K = _build_K(req.intrinsics)
            elif fb is not None and fb.get_K() is not None:
                K = fb.get_K()
            else:
                intrinsics = _portal_call(
                    lambda p: p.get_camera_intrinsics(active_camera).result(),
                    raise_http=True,
                    err_msg=no_source_msg.format("intrinsics"),
                )
                K = _build_K([float(x) for x in intrinsics])

            # ── Resolve extrinsics (optional — skip world-frame pose if absent)
            T_cam_world = None
            if req.extrinsics:
                T_cam_world = _build_SE3(req.extrinsics, active_camera)
            elif fb is not None and fb.get_extrinsics() is not None:
                T_cam_world = _build_SE3(fb.get_extrinsics(), active_camera)
            else:
                extr = _portal_call(
                    lambda p: p.get_camera_extrinsics(active_camera).result(),
                )
                if extr is not None:
                    T_cam_world = _build_SE3(extr, active_camera)

            print("[serve_bundlesdf] Running SAM3 detection (single-frame) …")
            mask_01, bbox_xywh, score = text_to_mask(rgb, req.text)
            _free_sam3_single_image()

            mask = mask_01.astype(np.uint8) * 255
            if erode_kernel is not None:
                mask = cv2.erode(mask, erode_kernel)

            ob_in_cam = None
            if has_valid_depth(depth, mask, zfar=cfg_bt["depth_processing"]["zfar"]):
                color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ob_in_cam = tracker.run(
                    color_bgr, depth, K, id_str="single", mask=mask, occ_mask=None
                )

            if ob_in_cam is None:
                return SingleFramePoseResponse(
                    bbox=list(bbox_xywh), score=round(score, 4)
                )

            resp_kwargs: dict = dict(
                bbox=list(bbox_xywh),
                score=round(score, 4),
                ob_in_cam=ob_in_cam.tolist(),
            )

            if T_cam_world is not None:
                ob_in_world = T_cam_world @ ob_in_cam
                resp_kwargs["ob_in_world"] = ob_in_world.tolist()
                center_cam = tracker.estimate_center_cam(
                    ob_in_cam, depth=depth, mask=mask, K=K
                )
                center_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
                center_world = T_cam_world[:3, :3] @ center_cam + T_cam_world[:3, 3]
                resp_kwargs["position_3d"] = [
                    round(float(x), 5) for x in center_world.tolist()
                ]
                resp_kwargs["pose_origin_3d"] = [
                    round(float(x), 5) for x in ob_in_world[:3, 3].tolist()
                ]
                resp_kwargs["position_3d_source"] = (
                    "reference_model_obb"
                    if getattr(tracker, "center_local", None) is not None
                    else "masked_depth_median"
                )
                resp_kwargs["quaternion_xyzw"] = [
                    round(float(x), 6)
                    for x in _rotation_matrix_to_quaternion(ob_in_world[:3, :3])
                ]

            return SingleFramePoseResponse(**resp_kwargs)
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))
        finally:
            try:
                tracker.on_finish()
            except Exception:
                pass
            del tracker
            torch.cuda.empty_cache()

    @app.post("/segment", response_model=SegmentResponse)
    def segment(req: SegmentRequest):
        """Run SAM3 text-prompted segmentation and return the binary mask."""
        active_camera = req.camera or camera

        # ── Resolve RGB: request body → frame buffer → portal
        if req.image_base64:
            rgb = _decode_rgb(req.image_base64)
        else:
            fb = _frame_buffers.get(active_camera)
            if fb is not None and fb.get_rgb() is not None:
                rgb = fb.get_rgb()
            else:
                rgb = _portal_call(
                    lambda p: np.asarray(p.get_camera_image(active_camera).result()),
                    raise_http=True,
                    err_msg="No image available (provide image_base64, push frames, or start cap_server)",
                )

        try:
            mask_01, bbox_xywh, score = text_to_mask(rgb, req.text)
            _free_sam3_single_image()
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))

        mask_uint8 = mask_01.astype(np.uint8)
        buf = io.BytesIO()
        np.save(buf, mask_uint8)
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        return SegmentResponse(
            mask_b64=mask_b64,
            bbox_xywh=[int(x) for x in bbox_xywh],
            score=round(float(score), 4),
            mask_area=int(mask_uint8.sum()),
            height=mask_uint8.shape[0],
            width=mask_uint8.shape[1],
        )

    return app


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="BundleSDF multi-object tracking sidecar for CAP"
    )
    parser.add_argument(
        "--port", type=int, default=8119, help="HTTP port (default: 8119)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--cap_server_host", default="localhost")
    parser.add_argument("--cap_server_port", type=int, default=8300)
    parser.add_argument("--camera", default="top", help="Default camera name")
    parser.add_argument(
        "--sam3_url",
        default="http://localhost:6767",
        help="URL of external serve_sam3 instance (default: http://localhost:6767)",
    )
    args = parser.parse_args()

    # Point the remote SAM3 client at the configured URL
    _set_sam3_url(args.sam3_url)

    # ── Preload all heavyweight models before accepting requests ──────────
    print("[serve_bundlesdf] Preloading models …")
    from bundlesdf.loftr_wrapper import LoftrRunner

    LoftrRunner()  # LoFTR weights → CUDA (~45 MB)
    SharedSam2Tracker.preload()  # SAM2 weights → CUDA (~900 MB)
    print(f"[serve_bundlesdf] SAM3 delegated to external server at {args.sam3_url}")

    # ── Warmup: run dummy inference to JIT-compile CUDA kernels ──────────
    # Blackwell (sm_120) and other new GPUs need the first real inference
    # to trigger kernel compilation; weight-loading alone is not enough.
    print("[serve_bundlesdf] Warming up CUDA kernels (LoFTR + SAM2) …")
    _dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup LoFTR
    _loftr = LoftrRunner()
    for _ in range(2):
        try:
            _loftr.predict(
                rgbAs=np.array([_dummy[:320, :320]]),
                rgbBs=np.array([_dummy[:320, :320]]),
            )
        except (RuntimeError, Exception):
            pass

    # Warmup SAM2
    try:
        _sam2_warmup = SharedSam2Tracker(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        _sam2_warmup.add_object(_dummy, (100, 100, 200, 200))
        _sam2_warmup.propagate(_dummy)
        del _sam2_warmup
    except (RuntimeError, Exception):
        pass
    torch.cuda.empty_cache()

    print("[serve_bundlesdf] All models loaded and warmed up.")

    app = create_app(args.cap_server_host, args.cap_server_port, args.camera)
    print(f"[serve_bundlesdf] Starting on {args.host}:{args.port}")
    print(
        f"[serve_bundlesdf] cap_server at {args.cap_server_host}:{args.cap_server_port}"
    )
    print(f"[serve_bundlesdf] sam3 at {args.sam3_url}")
    print(f"[serve_bundlesdf] Default camera: {args.camera}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
