"""AnyGrasp grasp planning tool for the CAP agent.

This tool exposes ``sample_grasp_pose_anygrasp`` with the world-frame +
display-RPY convention expected by ``freespace_move``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from cap.agent.tools.base import Tool, ToolParameter, ToolResult
from cap.config import (
    ANYGRASP_MIN_PLANNER_Z_M,
    ANYGRASP_SERVER_URL as ANYGRASP_DEFAULT_URL,
    CAP_SERVER_PORT,
    GRIPPER_DEFAULT_WIDTH_M,
    GRIPPER_TCP_OFFSET_Z_M,
    SAM3_SERVER_HOST,
    SAM3_SERVER_PORT,
)

logger = logging.getLogger(__name__)

ANYGRASP_WORKSPACE_MARGIN_M = 0.02
ANYGRASP_PLAN_TIMEOUT_S = float(os.environ.get("ANYGRASP_PLAN_TIMEOUT_S", "300"))
# AnyGrasp vendor gripper frame (from plot_gripper_pro_max):
#   +X = approach/depth
#   +Y = finger opening axis
#   +Z = gripper height
#
# Existing planner-facing frame in this repo:
#   +X = opening axis
#   +Y = lateral / gripper height
#   +Z = approach axis
#
# Remap AnyGrasp so the
# resulting pose can be fed directly into freespace_move using the same RPY
# convention as sample_grasp_pose.
_ANYGRASP_TO_GRIPPER = np.array(
    [
        [0.0, 0.0, 1.0],  # planner X = anygrasp Y
        [1.0, 0.0, 0.0],  # planner Y = anygrasp Z
        [0.0, 1.0, 0.0],  # planner Z = anygrasp X (approach)
    ],
    dtype=np.float64,
)
_ANYGRASP_TO_GRIPPER_T = np.eye(4, dtype=np.float64)
_ANYGRASP_TO_GRIPPER_T[:3, :3] = _ANYGRASP_TO_GRIPPER
_LOCAL_YAW_FLIP = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


def _quat_xyzw_to_display_rpy_deg(quat_xyzw: np.ndarray) -> list[float]:
    """Inverse of freespace_move._display_rpy_to_quat for planner-facing UI/tool output."""
    from scipy.spatial.transform import Rotation

    ex, ey, ez = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler(
        "xyz", degrees=True
    )
    disp = np.array([ey, -ex, -ez - 90.0], dtype=np.float64)
    disp = (disp + 180.0) % 360.0 - 180.0
    return disp.tolist()


def _round_list(values: np.ndarray | list[float], digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def _clip_planner_z_in_place(grasps_world: np.ndarray) -> int:
    """Clip planner-facing AnyGrasp Z to the configured table-safe floor."""
    if grasps_world.size == 0:
        return 0
    z = grasps_world[:, 2, 3]
    mask = z < float(ANYGRASP_MIN_PLANNER_Z_M)
    if not np.any(mask):
        return 0
    grasps_world[mask, 2, 3] = float(ANYGRASP_MIN_PLANNER_Z_M)
    return int(np.count_nonzero(mask))


def _filter_wrist_camera_y(
    grasps_world: np.ndarray,
    *,
    threshold: float = 0.0,
    allow_yaw_flip: bool = True,
) -> tuple[np.ndarray, list[int], list[str], list[float], list[float], int, int]:
    """Keep poses whose wrist-camera/local +Y axis points toward world +Y."""
    if grasps_world.size == 0:
        return grasps_world, [], [], [], [], 0, 0

    kept: list[np.ndarray] = []
    keep_indices: list[int] = []
    actions: list[str] = []
    before_dots: list[float] = []
    after_dots: list[float] = []
    n_flipped = 0
    n_filtered = 0
    threshold = float(threshold)
    for i, T in enumerate(grasps_world):
        y_dot = float(T[1, 1])
        if y_dot >= threshold:
            kept.append(T.copy())
            keep_indices.append(i)
            actions.append("kept")
            before_dots.append(y_dot)
            after_dots.append(y_dot)
            continue

        if allow_yaw_flip:
            flipped = T.copy()
            flipped[:3, :3] = flipped[:3, :3] @ _LOCAL_YAW_FLIP
            flipped_y_dot = float(flipped[1, 1])
            if flipped_y_dot >= threshold:
                kept.append(flipped)
                keep_indices.append(i)
                actions.append("yaw_flipped")
                before_dots.append(y_dot)
                after_dots.append(flipped_y_dot)
                n_flipped += 1
                continue

        n_filtered += 1

    if not kept:
        return (
            np.empty((0, 4, 4), dtype=grasps_world.dtype),
            [],
            [],
            [],
            [],
            n_flipped,
            n_filtered,
        )
    return (
        np.stack(kept, axis=0),
        keep_indices,
        actions,
        before_dots,
        after_dots,
        n_flipped,
        n_filtered,
    )


def _subset_list(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[i] for i in indices if i < len(values)]


@dataclass
class GraspCandidate:
    """Motion-planner-compatible grasp candidate."""

    position: list[float]  # [x, y, z] in world frame
    rpy: list[float]  # display RPY [roll, pitch, yaw] in degrees
    score: float
    width: float = GRIPPER_DEFAULT_WIDTH_M


def _np_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_np(s: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(s)))


class SampleGraspPoseAnyGraspTool(Tool):
    """Plan grasp poses for an object using AnyGrasp.

    Returned poses are converted to the same world-frame + display-RPY contract
    used by the planner in this repo, so they can be passed directly to
    ``freespace_move``.
    """

    name = "sample_grasp_pose_anygrasp"
    description = (
        "Generate 6-DOF grasp pose candidates for a named object using AnyGrasp. "
        "Returns a list of GraspCandidate(position, rpy, score, width) sorted "
        "by score (best first). The returned position/orientation is aligned to "
        "the same motion-planner convention expected by freespace_move, "
        "so it can be passed directly to freespace_move."
    )
    parameters = [
        ToolParameter(
            "object_name", "str", "Name of the object to grasp (e.g. 'red block')."
        ),
        ToolParameter(
            "camera",
            "str",
            "Camera to use (default: 'top').",
            required=False,
            default="top",
        ),
        ToolParameter(
            "max_grasps",
            "int",
            "Max number of grasp candidates to return.",
            required=False,
            default=10,
        ),
        ToolParameter(
            "top_down_only",
            "bool",
            "If true, filter for top-down grasps only (planner gripper Z aligned with world -Z).",
            required=False,
            default=False,
        ),
        ToolParameter(
            "vertical_threshold",
            "float",
            "Dot product threshold for top-down filtering (default: 0.8).",
            required=False,
            default=0.8,
        ),
        ToolParameter(
            "object_input_mode",
            "str",
            "How to feed the segmented object into AnyGrasp: 'segmented_object_cloud' (default) "
            "sends only the segmented object point cloud; 'roi_workspace' sends the scene cloud. THIS IS DEPRECATED! IT WILL CAUSE POSE AMBIGUITY."
            "cropped by the object's workspace bounds.",
            required=False,
            default="segmented_object_cloud",
        ),
        ToolParameter(
            "tcp_offset_z_m",
            "float",
            "Optional planner-frame TCP offset applied along local +Z before returning grasps.",
            required=False,
            default=GRIPPER_TCP_OFFSET_Z_M,
        ),
        ToolParameter(
            "disable_planner_z_clipping",
            "bool",
            "If true, do not clip planner-facing AnyGrasp Z to the configured safety floor.",
            required=False,
            default=False,
        ),
        ToolParameter(
            "filter_wrist_camera_y",
            "bool",
            "If true, keep or yaw-flip AnyGrasp poses so the wrist-camera/local +Y axis points toward world +Y.",
            required=False,
            default=True,
        ),
        ToolParameter(
            "wrist_camera_y_threshold",
            "float",
            "Minimum dot(local +Y, world +Y) accepted by the wrist-camera orientation filter.",
            required=False,
            default=0.0,
        ),
        ToolParameter(
            "allow_wrist_camera_yaw_flip",
            "bool",
            "If true, repair wrong-direction AnyGrasp poses by rotating 180 degrees around local +Z before filtering.",
            required=False,
            default=True,
        ),
    ]

    def __init__(
        self,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        sam3_url: str = f"http://{SAM3_SERVER_HOST}:{SAM3_SERVER_PORT}",
        anygrasp_url: str | None = None,
        env=None,
    ):
        self._env = env
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._sam3_url = sam3_url.rstrip("/")
        self._anygrasp_url = (
            anygrasp_url
            or os.environ.get("ANYGRASP_SERVICE_URL", str(ANYGRASP_DEFAULT_URL))
        ).rstrip("/")
        self._anygrasp_plan_timeout_s = float(
            os.environ.get("ANYGRASP_PLAN_TIMEOUT_S", str(ANYGRASP_PLAN_TIMEOUT_S))
        )
        self._portal_client = None
        self.last_rgb: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_overlay_jpeg: bytes | None = None
        self.last_native_viz_png: bytes | None = None
        self.last_grasp_debug: dict[str, Any] | None = None
        self._tcp_offset_z = GRIPPER_TCP_OFFSET_Z_M

    def _get_portal(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _get_rgb_depth_intrinsics_from_portal(
        self, camera: str
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        client = self._get_portal()
        rgb_raw = client.get_camera_image(camera).result()
        depth_raw = client.get_camera_depth(camera).result()
        intr_raw = client.get_camera_intrinsics(camera).result()
        return np.asarray(rgb_raw), np.asarray(depth_raw).astype(np.float32), intr_raw

    def _get_rgb_depth_intrinsics(
        self, camera: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._env is not None:
            try:
                rgb_raw = self._env.render_rgb(camera)
                depth_raw = self._env.render_depth(camera)
                intr_raw = self._env.get_camera_intrinsics(camera)
            except Exception as exc:
                logger.warning(
                    "[anygrasp] env camera %r unavailable (%s); falling back to portal %s:%s",
                    camera,
                    exc,
                    self._cap_host,
                    self._cap_port,
                )
                rgb_raw, depth_raw, intr_raw = self._get_rgb_depth_intrinsics_from_portal(
                    camera
                )
        else:
            rgb_raw, depth_raw, intr_raw = self._get_rgb_depth_intrinsics_from_portal(camera)
        rgb = np.asarray(rgb_raw)
        depth = np.asarray(depth_raw).astype(np.float32)
        if (rgb.size < 100 or depth.size < 100) and self._env is not None:
            logger.warning(
                "[anygrasp] env camera %r returned empty frame; falling back to portal %s:%s",
                camera,
                self._cap_host,
                self._cap_port,
            )
            rgb, depth, intr_raw = self._get_rgb_depth_intrinsics_from_portal(camera)
        if rgb.size < 100:
            raise ValueError(f"No image returned for camera {camera!r}")
        if depth.size < 100:
            raise ValueError(f"No depth returned for camera {camera!r}")
        fx, fy, cx, cy = [float(x) for x in intr_raw]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return rgb, depth, K

    def _get_extrinsics(self, camera: str) -> np.ndarray:
        if self._env is not None:
            try:
                extr = self._env.get_camera_extrinsics(camera)
            except Exception as exc:
                logger.warning(
                    "[anygrasp] env extrinsics for %r unavailable (%s); falling back to portal %s:%s",
                    camera,
                    exc,
                    self._cap_host,
                    self._cap_port,
                )
                extr = self._get_portal().get_camera_extrinsics(camera).result()
        else:
            extr = self._get_portal().get_camera_extrinsics(camera).result()
        if not extr and self._env is not None:
            logger.warning(
                "[anygrasp] env extrinsics for %r were empty; falling back to portal %s:%s",
                camera,
                self._cap_host,
                self._cap_port,
            )
            extr = self._get_portal().get_camera_extrinsics(camera).result()
        R = np.asarray(extr["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extr["position"], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        if extr.get("needs_optical_flip", True):
            F = np.diag([-1.0, -1.0, 1.0])
            T[:3, :3] = R @ F
        else:
            T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _segment_object(self, rgb: np.ndarray, object_name: str) -> np.ndarray:
        import urllib.error
        import urllib.request

        buf = io.BytesIO()
        np.save(buf, rgb)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = json.dumps({"text": object_name, "image_b64": image_b64}).encode()
        req = urllib.request.Request(
            f"{self._sam3_url}/segment",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except Exception:
                detail = body
            raise RuntimeError(f"SAM3 /segment failed ({e.code}): {detail}") from None
        data = json.loads(resp.read())
        mask_bytes = base64.b64decode(data["mask_b64"])
        return np.load(io.BytesIO(mask_bytes)).astype(np.int32)

    def _call_anygrasp_viz(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        segmap: np.ndarray,
        *,
        max_grasps: int,
        object_input_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes, list[str]]:
        payload = {
            "rgb_base64": _np_to_b64(rgb),
            "depth_base64": _np_to_b64(depth),
            "cam_K_base64": _np_to_b64(K),
            "segmap_base64": _np_to_b64(segmap),
            "segmap_id": 1,
            "z_range": [1e-6, 1.5],
            "max_grasps": max_grasps,
            "workspace_margin": ANYGRASP_WORKSPACE_MARGIN_M,
            "object_input_mode": object_input_mode,
        }
        start = time.perf_counter()
        resp = requests.post(
            f"{self._anygrasp_url}/plan_viz",
            json=payload,
            timeout=self._anygrasp_plan_timeout_s,
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
        logger.info(
            "[anygrasp] /plan_viz returned in %.1fs (timeout %.1fs)",
            elapsed,
            self._anygrasp_plan_timeout_s,
        )
        data = resp.json()
        grasps = (
            _b64_to_np(data["grasps_base64"])
            if data.get("grasps_base64")
            else np.empty((0, 4, 4), dtype=np.float64)
        )
        scores = (
            _b64_to_np(data["scores_base64"])
            if data.get("scores_base64")
            else np.empty((0,), dtype=np.float64)
        )
        widths = (
            _b64_to_np(data["widths_base64"])
            if data.get("widths_base64")
            else np.empty((0,), dtype=np.float64)
        )
        overlay = (
            base64.b64decode(data["overlay_jpeg_base64"])
            if data.get("overlay_jpeg_base64")
            else b""
        )
        thumbnails = [str(x) for x in (data.get("grasp_thumbnail_jpeg_base64") or [])]
        return grasps, scores, widths, overlay, thumbnails

    def execute(self, **kwargs: Any) -> ToolResult:
        from scipy.spatial.transform import Rotation

        object_name: str = kwargs["object_name"]
        camera: str = kwargs.get("camera", "top")
        max_grasps: int = kwargs.get("max_grasps", 10)
        top_down_only: bool = kwargs.get("top_down_only", False)
        vertical_threshold: float = kwargs.get("vertical_threshold", 0.8)
        object_input_mode: str = kwargs.get(
            "object_input_mode", "segmented_object_cloud"
        )
        tcp_offset_z_m: float = float(kwargs.get("tcp_offset_z_m", self._tcp_offset_z))
        disable_planner_z_clipping: bool = bool(
            kwargs.get("disable_planner_z_clipping", False)
        )
        filter_wrist_camera_y: bool = bool(kwargs.get("filter_wrist_camera_y", True))
        wrist_camera_y_threshold: float = float(
            kwargs.get("wrist_camera_y_threshold", 0.0)
        )
        allow_wrist_camera_yaw_flip: bool = bool(
            kwargs.get("allow_wrist_camera_yaw_flip", True)
        )

        self.last_rgb = None
        self.last_mask = None
        self.last_overlay_jpeg = None
        self.last_native_viz_png = None
        self.last_grasp_debug = {
            "camera": camera,
            "object_name": object_name,
            "n_grasps": 0,
            "status": "error",
            "backend": "anygrasp",
            "grasps": [],
            "object_input_mode": object_input_mode,
        }

        try:
            rgb, depth, K = self._get_rgb_depth_intrinsics(camera)
            T_cam_world = self._get_extrinsics(camera)

            logger.info(f"[anygrasp] Segmenting {object_name!r} on {camera}...")
            mask = self._segment_object(rgb, object_name)
            segmap = mask.astype(np.int32)
            self.last_rgb = rgb
            self.last_mask = mask

            logger.info(f"[anygrasp] Planning grasps for {object_name!r}...")
            grasps_cam_raw, scores, widths, overlay, thumbnails = (
                self._call_anygrasp_viz(
                    rgb,
                    depth,
                    K,
                    segmap,
                    max_grasps=max_grasps,
                    object_input_mode=object_input_mode,
                )
            )
            self.last_overlay_jpeg = overlay or None

            if grasps_cam_raw.size == 0:
                self.last_grasp_debug = {
                    "camera": camera,
                    "object_name": object_name,
                    "n_grasps": 0,
                    "status": "no_grasps",
                    "backend": "anygrasp",
                    "status_reason": f"No grasps found for {object_name!r} after SAM3 masking.",
                    "grasps": [],
                    "object_input_mode": object_input_mode,
                }
                return ToolResult(
                    success=False, error=f"AnyGrasp found no grasps for {object_name!r}"
                )

            # 1) Camera frame AnyGrasp → planner/gripper frame.
            grasps_cam = np.matmul(grasps_cam_raw, _ANYGRASP_TO_GRIPPER_T)

            # 2) Camera → world.
            grasps_world = np.matmul(T_cam_world, grasps_cam)

            # 3) Match the planner-facing output convention: translate from TCP-like
            # pose origin to the planner's grasp site along local +Z.
            translated_grasps_world = grasps_world.copy()
            translated_grasps_world[:, :3, 3] = (
                grasps_world[:, :3, 3] + tcp_offset_z_m * grasps_world[:, :3, 2]
            )
            n_z_clipped = 0
            if not disable_planner_z_clipping:
                n_z_clipped = _clip_planner_z_in_place(translated_grasps_world)

            wrist_actions = ["unchecked"] * len(translated_grasps_world)
            wrist_y_before = [float(T[1, 1]) for T in translated_grasps_world]
            wrist_y_after = list(wrist_y_before)
            n_wrist_yaw_flipped = 0
            n_wrist_filtered = 0
            if top_down_only:
                keep: list[int] = []
                for i in range(len(translated_grasps_world)):
                    approach = translated_grasps_world[i, :3, 2]
                    if float(
                        np.dot(approach, np.array([0.0, 0.0, -1.0], dtype=np.float64))
                    ) > float(vertical_threshold):
                        keep.append(i)
                if not keep:
                    self.last_grasp_debug = {
                        "camera": camera,
                        "object_name": object_name,
                        "n_grasps": 0,
                        "status": "no_top_down_grasps",
                        "backend": "anygrasp",
                        "status_reason": f"No top-down grasps remained for {object_name!r} after filtering.",
                        "grasps": [],
                        "object_input_mode": object_input_mode,
                    }
                    return ToolResult(
                        success=False,
                        error=f"No top-down grasps found for {object_name!r} (threshold={vertical_threshold})",
                    )
                translated_grasps_world = translated_grasps_world[keep]
                grasps_cam_raw = grasps_cam_raw[keep]
                scores = scores[keep]
                widths = widths[keep] if len(widths) >= len(keep) else widths
                thumbnails = _subset_list(thumbnails, keep)
                wrist_actions = [wrist_actions[i] for i in keep]
                wrist_y_before = [wrist_y_before[i] for i in keep]
                wrist_y_after = [wrist_y_after[i] for i in keep]

            if filter_wrist_camera_y:
                (
                    translated_grasps_world,
                    keep,
                    wrist_actions,
                    wrist_y_before,
                    wrist_y_after,
                    n_wrist_yaw_flipped,
                    n_wrist_filtered,
                ) = _filter_wrist_camera_y(
                    translated_grasps_world,
                    threshold=wrist_camera_y_threshold,
                    allow_yaw_flip=allow_wrist_camera_yaw_flip,
                )
                if not keep:
                    self.last_grasp_debug = {
                        "camera": camera,
                        "object_name": object_name,
                        "n_grasps": 0,
                        "status": "no_wrist_camera_y_grasps",
                        "backend": "anygrasp",
                        "status_reason": f"No grasps remained for {object_name!r} after wrist-camera +Y filtering.",
                        "grasps": [],
                        "object_input_mode": object_input_mode,
                        "wrist_camera_y_filter_enabled": True,
                        "wrist_camera_y_threshold": float(wrist_camera_y_threshold),
                        "allow_wrist_camera_yaw_flip": bool(allow_wrist_camera_yaw_flip),
                        "n_wrist_camera_yaw_flipped": int(n_wrist_yaw_flipped),
                        "n_wrist_camera_y_filtered": int(n_wrist_filtered),
                    }
                    return ToolResult(
                        success=False,
                        error=(
                            f"No wrist-camera +Y grasps found for {object_name!r} "
                            f"(threshold={wrist_camera_y_threshold})"
                        ),
                    )
                grasps_cam_raw = grasps_cam_raw[keep]
                scores = scores[keep]
                widths = widths[keep] if len(widths) >= len(keep) else widths
                thumbnails = _subset_list(thumbnails, keep)

            order = np.argsort(-scores)[:max_grasps]
            grasps_cam_raw = grasps_cam_raw[order]
            translated_grasps_world = translated_grasps_world[order]
            scores = scores[order]
            widths = widths[order] if len(widths) >= len(order) else widths
            wrist_actions = [wrist_actions[i] for i in order]
            wrist_y_before = [wrist_y_before[i] for i in order]
            wrist_y_after = [wrist_y_after[i] for i in order]
            thumbnails = _subset_list(thumbnails, [int(i) for i in order])
            raw_grasps_world = np.matmul(T_cam_world, grasps_cam_raw)

            candidates: list[GraspCandidate] = []
            grasp_rows: list[dict[str, Any]] = []
            for i, T in enumerate(translated_grasps_world):
                pos = T[:3, 3].tolist()
                quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
                rpy_deg = _quat_xyzw_to_display_rpy_deg(quat_xyzw)
                raw_T = raw_grasps_world[i]
                width = float(widths[i]) if i < len(widths) else GRIPPER_DEFAULT_WIDTH_M
                candidates.append(
                    GraspCandidate(
                        position=[round(float(x), 5) for x in pos],
                        rpy=[round(float(x), 4) for x in rpy_deg],
                        score=round(float(scores[i]), 4),
                        width=round(width, 5),
                    )
                )
                grasp_rows.append(
                    {
                        "rank": i + 1,
                        "score": round(float(scores[i]), 4),
                        "width": round(width, 5),
                        "raw_xyz": _round_list(raw_T[:3, 3], 5),
                        "raw_rpy": _round_list(
                            Rotation.from_matrix(raw_T[:3, :3]).as_euler(
                                "xyz", degrees=True
                            ),
                            3,
                        ),
                        "planner_xyz": _round_list(T[:3, 3], 5),
                        "planner_rpy": _round_list(rpy_deg, 3),
                        "wrist_camera_y_dot_before": round(float(wrist_y_before[i]), 4),
                        "wrist_camera_y_dot_after": round(float(wrist_y_after[i]), 4),
                        "wrist_camera_y_action": wrist_actions[i],
                        "status": "returned",
                        "thumbnail_b64": thumbnails[i] if i < len(thumbnails) else None,
                    }
                )

            self.last_grasp_debug = {
                "camera": camera,
                "object_name": object_name,
                "n_grasps": int(len(candidates)),
                "status": "ok",
                "backend": "anygrasp",
                "object_input_mode": object_input_mode,
                "best_score": float(candidates[0].score) if candidates else None,
                "tcp_offset_z_m": float(tcp_offset_z_m),
                "planner_z_floor_m": float(ANYGRASP_MIN_PLANNER_Z_M),
                "planner_z_clipping_enabled": not disable_planner_z_clipping,
                "n_planner_z_clipped": int(n_z_clipped),
                "wrist_camera_y_filter_enabled": bool(filter_wrist_camera_y),
                "wrist_camera_y_threshold": float(wrist_camera_y_threshold),
                "allow_wrist_camera_yaw_flip": bool(allow_wrist_camera_yaw_flip),
                "n_wrist_camera_yaw_flipped": int(n_wrist_yaw_flipped),
                "n_wrist_camera_y_filtered": int(n_wrist_filtered),
                "grasps": grasp_rows,
            }
            try:
                from cap.agent.tools._artifact_log import log_grasp

                log_grasp(
                    rgb,
                    mask,
                    candidates,
                    query=object_name,
                    tag="grasp_anygrasp",
                )
            except Exception:
                logger.debug("Failed to save AnyGrasp artifact", exc_info=True)
            logger.info(
                f"[anygrasp] {len(candidates)} planner-aligned grasps for {object_name!r} "
                f"(best score={candidates[0].score if candidates else 'n/a'})"
            )
            if n_z_clipped:
                logger.info(
                    f"[anygrasp] Clipped {n_z_clipped} planner-facing grasp z value(s) "
                    f"to floor {float(ANYGRASP_MIN_PLANNER_Z_M):.4f} m"
                )
            if n_wrist_yaw_flipped or n_wrist_filtered:
                logger.info(
                    "[anygrasp] Wrist-camera +Y filter: yaw_flipped=%d filtered=%d "
                    "threshold=%.3f",
                    n_wrist_yaw_flipped,
                    n_wrist_filtered,
                    wrist_camera_y_threshold,
                )
            return ToolResult(success=True, data=candidates)

        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach AnyGrasp server at {self._anygrasp_url}",
            )
        except requests.Timeout:
            return ToolResult(
                success=False,
                error=(
                    "Tool sample_grasp_pose_anygrasp failed: timed out while waiting "
                    f"for AnyGrasp /plan_viz after {self._anygrasp_plan_timeout_s:.0f}s. "
                    "This is often a cold-start warmup; retry once or increase "
                    "ANYGRASP_PLAN_TIMEOUT_S."
                ),
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
