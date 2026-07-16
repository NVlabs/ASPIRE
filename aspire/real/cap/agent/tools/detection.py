"""Detection tools for BundleSDF-based object localization and tracking."""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, Callable

import numpy as np
import requests
from PIL import Image

from cap.config import DETECTION_SERVER_PORT, make_bundlesdf_name
from cap.agent.tools.base import Detection3D, Tool, ToolParameter, ToolResult
from cap.agent.tools._artifact_log import log_detection


def _encode_rgb(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_depth(depth: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, depth.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _jsonify_extrinsics(extrinsics: dict[str, Any]) -> dict[str, Any]:
    R = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(extrinsics["position"], dtype=np.float64).reshape(-1)

    # Pre-apply optical flip so extrinsics are sent in OpenCV convention.
    # Remote BundleSDF servers may have stale needs_optical_flip() logic that
    # doesn't know about sim envs. By pre-flipping here, we ensure the
    # extrinsics arrive in OpenCV convention regardless.
    if extrinsics.get("needs_optical_flip", False):
        R = R @ np.diag([-1.0, -1.0, 1.0])

    return {
        "position": [float(x) for x in t.tolist()],
        "rotation": [float(x) for x in R.reshape(-1).tolist()],
        "needs_optical_flip": False,  # already flipped
    }


class DetectObjectTool(Tool):
    """Detect objects in a camera image via BundleSDF or simulation oracle.

    Supported backends:
      - ``bundlesdf``: 6-DOF pose tracking via BundleSDF. The first call
        starts a tracking session; subsequent calls poll the latest pose.
      - ``oracle``: ground-truth pose from simulation (sim-only).
    """

    name = "detect_object"
    description = (
        "Detect objects matching a text query. "
        "Returns Detection3D results with world-frame position and optional 6-DOF pose. "
        "Use backend='bundlesdf' for tracked 6-DOF pose, or backend='oracle' for "
        "ground-truth pose from simulation (sim-only)."
    )
    parameters = [
        ToolParameter("query", "str", "Text query describing the object to find."),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
        ToolParameter(
            "backend", "str",
            "'bundlesdf' (default, 6-DOF pose tracking) or 'oracle' (ground-truth from sim, sim-only).",
            required=False, default="bundlesdf",
        ),
        ToolParameter(
            "max_retries", "int",
            "Number of retry attempts if detection fails (default 3).",
            required=False, default="3",
        ),
    ]

    def __init__(
        self,
        detection_host: str = "localhost",
        detection_port: int = DETECTION_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int | None = None,
        bundlesdf_host: str = "localhost",
        bundlesdf_port: int | None = None,
        timeout: float = 30.0,
        env=None,
    ):
        self._env = env
        self._detection_host = detection_host
        self._detection_port = detection_port
        self._cap_server_host = cap_server_host
        self._cap_server_port = cap_server_port
        self._timeout = timeout
        self._portal_client = None

        # BundleSDF state
        from cap.config import BUNDLESDF_SERVER_PORT
        bsdf_port = bundlesdf_port or BUNDLESDF_SERVER_PORT
        self._bundlesdf_url = f"http://{bundlesdf_host}:{bsdf_port}"
        self._bundlesdf_sessions: dict[str, str] = {}  # name_key → camera
        self._bundlesdf_active_query: str | None = None
        self._bundlesdf_active_camera: str | None = None

    _KNOWN_HALF_EXTENTS = {
        "red block": [0.025, 0.025, 0.06],
        "red_block": [0.025, 0.025, 0.06],
        "red stick": [0.025, 0.025, 0.06],
        "red_stick": [0.025, 0.025, 0.06],
        "green block": [0.05, 0.05, 0.005],
        "green_block": [0.05, 0.05, 0.005],
        "blue plate": [0.05, 0.05, 0.005],
        "blue_plate": [0.05, 0.05, 0.005],
    }

    def _capture_snapshot(self, camera: str) -> dict[str, Any]:
        if self._env is not None:
            rgb_raw = self._env.render_rgb(camera)
            depth_raw = self._env.render_depth(camera)
            intrinsics_raw = self._env.get_camera_intrinsics(camera)
            extrinsics = self._env.get_camera_extrinsics(camera)
            if rgb_raw is None or depth_raw is None or intrinsics_raw is None:
                client = self._get_portal_client()
                rgb_raw = client.get_camera_image(camera).result()
                depth_raw = client.get_camera_depth(camera).result()
                intrinsics_raw = client.get_camera_intrinsics(camera).result()
                extrinsics = client.get_camera_extrinsics(camera).result()
        else:
            client = self._get_portal_client()
            rgb_raw = client.get_camera_image(camera).result()
            depth_raw = client.get_camera_depth(camera).result()
            intrinsics_raw = client.get_camera_intrinsics(camera).result()
            extrinsics = client.get_camera_extrinsics(camera).result()
        rgb = np.asarray(rgb_raw)
        depth = np.asarray(depth_raw)
        if rgb.size < 100:
            raise ValueError(f"No image returned for camera {camera!r}")
        if depth.size < 100:
            raise ValueError(f"No depth returned for camera {camera!r}")
        intrinsics = [float(x) for x in intrinsics_raw]
        return {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "intrinsics": intrinsics,
            "extrinsics": _jsonify_extrinsics(extrinsics),
        }

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox: list[float] | None) -> list[float]:
        if bbox and len(bbox) == 4:
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        return []

    def _bundlesdf_payload_to_detection(self, query: str, data: dict[str, Any]) -> Detection3D:
        he = data.get("half_extents") or self._KNOWN_HALF_EXTENTS.get(query, [])
        return Detection3D(
            label=query,
            score=data.get("score", 0.0),
            box_2d=self._bbox_xywh_to_xyxy(data.get("bbox")),
            position_3d=data["position_3d"],
            quaternion_xyzw=data.get("quaternion_xyzw", []),
            rpy=data.get("rpy", []),
            half_extents=he,
            position_3d_source=data.get("position_3d_source"),
            vis_b64=data.get("vis_b64"),
        )

    def _execute_bundlesdf_single_frame(
        self,
        *,
        query: str,
        camera: str = "top",
        snapshot: dict[str, Any] | None = None,
    ) -> ToolResult:
        try:
            snap = snapshot or self._capture_snapshot(camera)
            payload = {
                "text": query,
                "camera": camera,
                "image_base64": _encode_rgb(snap["rgb"]),
                "depth_base64": _encode_depth(snap["depth"]),
                "intrinsics": snap["intrinsics"],
                "extrinsics": snap["extrinsics"],
            }
            resp = requests.post(
                f"{self._bundlesdf_url}/single_frame_pose",
                json=payload,
                timeout=max(self._timeout, 120.0),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("position_3d") is None:
                score = float(data.get("score", 0.0))
                bbox = data.get("bbox")
                return ToolResult(
                    success=False,
                    error=(
                        f"BundleSDF single-frame pose unavailable for {query!r}. "
                        f"score={score:.3f}, bbox={bbox}"
                    ),
                )
            det = self._bundlesdf_payload_to_detection(query, data)

            # Save annotated snapshot to log
            try:
                rgb = snap["rgb"]
                log_detection(rgb, [det], tag="detect_oneshot")
            except Exception:
                pass  # best-effort

            return ToolResult(success=True, data=[det])
        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._bundlesdf_url}",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _get_portal_client(self):
        if self._portal_client is None:
            import portal
            from cap.config import CAP_SERVER_PORT

            port = self._cap_server_port or CAP_SERVER_PORT
            self._portal_client = portal.Client(f"{self._cap_server_host}:{port}")
        return self._portal_client

    @staticmethod
    def _cam_to_world(
        pos_cam: list[float],
        cam_pos: np.ndarray,
        cam_rot: np.ndarray,
    ) -> list[float]:
        """Transform a point from camera frame to world frame.

        RealSense camera convention: +x right, +y down, +z forward.
        The physical camera mount flips left/right relative to the world
        frame, so we negate x before applying the rotation.
        """
        p = np.asarray(pos_cam, dtype=np.float64)
        p[0] = -p[0]  # flip left/right to match world frame
        p[1] = -p[1]  # flip up/down to match world frame
        world_p = cam_pos + cam_rot @ p
        return [round(float(x), 4) for x in world_p]

    def execute(self, **kwargs: Any) -> ToolResult:
        import logging
        logger = logging.getLogger(__name__)

        backend: str = kwargs.get("backend", "bundlesdf")
        max_retries: int = int(kwargs.get("max_retries", 3))

        for attempt in range(1, max_retries + 1):
            if backend == "bundlesdf":
                result = self._execute_bundlesdf(**kwargs)
            elif backend == "oracle":
                result = self._execute_oracle(**kwargs)
            else:
                result = ToolResult(
                    success=False,
                    error=f"Unsupported detection backend: {backend}. Use 'bundlesdf' or 'oracle'.",
                )

            if result.success and result.data:
                return result

            if attempt < max_retries:
                logger.info(
                    f"[detect_object] Attempt {attempt}/{max_retries} failed: "
                    f"{result.error or 'no detections'}. Retrying in 1s..."
                )
                time.sleep(1.0)

        return result

    # -- Oracle backend (sim ground truth) -------------------------------------

    def _execute_oracle(self, **kwargs: Any) -> ToolResult:
        """Return the ground-truth pose of the queried object from simulation.

        Fuzzy-matches *query* against MuJoCo scene body names (case-insensitive
        substring match).  Only works when the cap_server is backed by a
        simulation (SimBackend / WarpSimBackend).
        """
        query: str = kwargs["query"]
        try:
            client = self._get_portal_client()
            result = client.get_object_positions().result()
            if not result.get("ok"):
                return ToolResult(success=False, error="get_object_positions failed (is sim running?)")

            objects: dict[str, dict] = result["objects"]
            if not objects:
                return ToolResult(success=False, error="No scene objects found in simulation")

            # Fuzzy match: case-insensitive substring
            query_lower = query.lower()
            matches: list[tuple[str, dict]] = [
                (name, data)
                for name, data in objects.items()
                if query_lower in name.lower() or name.lower() in query_lower
            ]

            if not matches:
                available = ", ".join(objects.keys())
                return ToolResult(
                    success=False,
                    error=f"No object matching '{query}'. Available: {available}",
                )

            detections: list[Detection3D] = []
            for name, data in matches:
                pos = data["pos"]  # [x, y, z]
                quat_wxyz = data["quat"]  # MuJoCo convention: [w, x, y, z]
                quat_xyzw = quat_wxyz[1:] + quat_wxyz[:1]  # -> [x, y, z, w]
                size = data.get("size", [])  # geom half-extents from MuJoCo
                detections.append(
                    Detection3D(
                        label=name,
                        score=1.0,
                        box_2d=[],
                        position_3d=[round(float(x), 4) for x in pos],
                        quaternion_xyzw=[round(float(x), 4) for x in quat_xyzw],
                        half_extents=[round(float(x), 4) for x in size],
                    )
                )

            # Save camera snapshot with detections to log
            try:
                camera = kwargs.get("camera", "top")
                rgb = self._capture_snapshot(camera)["rgb"]
                log_detection(rgb, detections, tag="detect_oracle")
            except Exception:
                pass  # best-effort

            return ToolResult(success=True, data=detections)

        except Exception as e:
            return ToolResult(success=False, error=f"Oracle backend error: {e}")

    # -- BundleSDF backend -----------------------------------------------------

    # BundleSDF cold-start takes 30-60s (model loading + first SAM3 inference).
    # Subsequent calls with the same query are fast (~1s).
    _BUNDLESDF_POLL_INTERVAL = 2.0  # seconds between pose polls
    _BUNDLESDF_POLL_TIMEOUT = 60.0  # max wait for pose after start_tracking

    def _execute_bundlesdf(self, **kwargs: Any) -> ToolResult:
        import logging
        import urllib.parse
        logger = logging.getLogger(__name__)

        query: str = kwargs["query"]
        camera: str = kwargs.get("camera", "top")
        name = make_bundlesdf_name(query)
        encoded_name = urllib.parse.quote(name, safe="")

        try:
            # Auto-start if new query or camera changed
            cold_start = False
            if self._bundlesdf_active_query != query or self._bundlesdf_active_camera != camera:
                cold_start = True
                if self._bundlesdf_active_query is not None:
                    old_name = make_bundlesdf_name(self._bundlesdf_active_query)
                    logger.info("[bundlesdf] Stopping previous tracking session...")
                    requests.post(
                        f"{self._bundlesdf_url}/end_detection"
                        f"/{urllib.parse.quote(old_name, safe='')}",
                        timeout=30,
                    )

                logger.info(
                    f"[bundlesdf] Starting tracking for '{query}' on camera '{camera}' "
                    f"(cold start — model loading may take 30-60s)..."
                )
                resp = requests.post(
                    f"{self._bundlesdf_url}/add_detection",
                    json={"text": query, "camera": camera},
                    timeout=120,
                )
                if resp.status_code not in (200, 204, 409):
                    resp.raise_for_status()
                start_data = resp.json()
                logger.info(
                    f"[bundlesdf] Tracking started — "
                    f"bbox={start_data.get('bbox')}, "
                    f"first_score={start_data.get('first_score', '?')}"
                )
                self._bundlesdf_active_query = query
                self._bundlesdf_active_camera = camera

            # Poll pose with retries — BundleSDF needs time to converge
            timeout = self._BUNDLESDF_POLL_TIMEOUT if cold_start else 10.0
            interval = self._BUNDLESDF_POLL_INTERVAL
            elapsed = 0.0
            data = None

            while elapsed < timeout:
                resp = requests.get(
                    f"{self._bundlesdf_url}/get_detection/{encoded_name}",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                tracking = data.get("tracking", False)
                has_pose = data.get("position_3d") is not None
                frame_idx = data.get("frame_idx", 0)
                score = data.get("score", 0.0)

                has_valid_depth = data.get("ob_in_cam") is not None

                if tracking and has_pose and has_valid_depth:
                    logger.info(
                        f"[bundlesdf] Pose ready — frame={frame_idx}, "
                        f"score={score:.3f}, elapsed={elapsed:.1f}s"
                    )
                    break

                if tracking and has_pose and not has_valid_depth:
                    logger.warning(
                        f"[bundlesdf] Depth invalid — SAM3 tracks the object "
                        f"(score={score:.3f}, frame={frame_idx}) but the depth "
                        f"sensor returned no valid data in the masked region. "
                        f"Position would be unreliable."
                    )

                logger.info(
                    f"[bundlesdf] Waiting for pose... "
                    f"tracking={tracking}, has_pose={has_pose}, "
                    f"frame={frame_idx}, score={score:.3f}, "
                    f"elapsed={elapsed:.1f}/{timeout:.0f}s"
                )
                time.sleep(interval)
                elapsed += interval
            else:
                self._bundlesdf_active_query = None  # reset so next call retries
                tracking = data.get("tracking", False) if data else False
                score = data.get("score", 0) if data else 0
                frame_idx = data.get("frame_idx", 0) if data else 0
                has_ob = (data.get("ob_in_cam") is not None) if data else False

                if tracking and score > 0.3 and not has_ob:
                    err = (
                        f"BundleSDF: object visually tracked (score={score:.3f}, "
                        f"frame={frame_idx}) but depth sensor returned no valid data "
                        f"in the masked region after {timeout:.0f}s. The object surface "
                        f"may be too reflective/shiny for the depth sensor. "
                        f"Try a different object or move it closer to the camera."
                    )
                else:
                    err = (
                        f"BundleSDF pose not available after {timeout:.0f}s. "
                        f"Last state: tracking={tracking}, "
                        f"frame={frame_idx}, score={score:.3f}"
                    )
                return ToolResult(success=False, error=err)

            # Convert bbox [x, y, w, h] to [x1, y1, x2, y2] for box_2d
            bbox = data.get("bbox")
            if bbox and len(bbox) == 4:
                box_2d = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            else:
                box_2d = []

            det = self._bundlesdf_payload_to_detection(query, data)
            det.box_2d = box_2d

            # Save annotated camera snapshot to log
            try:
                snap = self._capture_snapshot(camera)
                log_detection(snap["rgb"], [det], tag="detect_bundlesdf")
            except Exception:
                pass  # best-effort

            return ToolResult(success=True, data=[det])

        except requests.ConnectionError:
            return ToolResult(
                success=False,
                error=f"Cannot reach BundleSDF server at {self._bundlesdf_url}",
            )
        except requests.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            return ToolResult(success=False, error=detail or str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))

class DetectObjectsOneshotTool(Tool):
    """One-shot BundleSDF pose inference on a shared snapshot.

    This tool never starts a real-time tracking session. It captures one RGB-D
    snapshot, then runs BundleSDF single-frame pose estimation for one or more
    text queries against that same snapshot.
    """

    name = "detect_objects_oneshot"
    description = (
        "One-shot BundleSDF pose inference without real-time tracking. "
        "Accepts either a single query string or a list of query strings. "
        "Captures one shared RGB-D snapshot and returns a dict mapping each query "
        "to its Detection3D results. Best for quasistatic scenes and batch-style "
        "multi-object perception."
    )
    parameters = [
        ToolParameter(
            "query",
            "str | list[str]",
            "Single text query or list of text queries to localize in one shared snapshot.",
        ),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
        ToolParameter(
            "max_retries", "int",
            "Number of retry attempts using one shared snapshot per attempt (default 3).",
            required=False, default="3",
        ),
    ]

    def __init__(
        self,
        detect_tool: DetectObjectTool | None = None,
        *,
        env=None,
        detection_host: str = "localhost",
        detection_port: int = DETECTION_SERVER_PORT,
        cap_server_host: str = "localhost",
        cap_server_port: int | None = None,
        bundlesdf_host: str = "localhost",
        bundlesdf_port: int | None = None,
        timeout: float = 30.0,
    ):
        self._detect_tool = detect_tool or DetectObjectTool(
            env=env,
            detection_host=detection_host,
            detection_port=detection_port,
            cap_server_host=cap_server_host,
            cap_server_port=cap_server_port,
            bundlesdf_host=bundlesdf_host,
            bundlesdf_port=bundlesdf_port,
            timeout=timeout,
        )

    @staticmethod
    def _normalize_queries(raw: Any) -> list[str]:
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple)) and all(isinstance(q, str) for q in raw):
            return [q for q in raw if q.strip()]
        raise ValueError("query must be a string or a list of strings")

    def execute(self, **kwargs: Any) -> ToolResult:
        camera = kwargs.get("camera", "top")
        max_retries = max(1, int(kwargs.get("max_retries", 3)))
        try:
            queries = self._normalize_queries(kwargs["query"])
        except Exception as e:
            return ToolResult(success=False, error=str(e))
        if not queries:
            return ToolResult(success=False, error="query list is empty")

        results: dict[str, list[Detection3D]] = {query: [] for query in queries}
        last_errors: dict[str, str] = {}
        pending = list(queries)

        for attempt in range(1, max_retries + 1):
            if not pending:
                break
            try:
                snapshot = self._detect_tool._capture_snapshot(camera)
            except Exception as e:
                return ToolResult(success=False, error=f"Failed to capture shared snapshot: {e}")

            next_pending: list[str] = []
            for query in pending:
                result = self._detect_tool._execute_bundlesdf_single_frame(
                    query=query, camera=camera, snapshot=snapshot
                )
                if result.success and result.data is not None:
                    results[query] = result.data
                else:
                    last_errors[query] = result.error or "unknown error"
                    next_pending.append(query)
            pending = next_pending
            if pending and attempt < max_retries:
                time.sleep(0.1)

        errors = [f"{query!r}: {last_errors[query]}" for query in pending if query in last_errors]
        if errors and not any(results.values()):
            return ToolResult(success=False, error="; ".join(errors))
        return ToolResult(success=True, data=results)


class DetectObjectRealtimeTool(Tool):
    """Continuously detect objects and update visualization until stopped.

    Blocks the executor thread, running detection in a loop. Stops when
    the stop_event is set (by Home/Stop/E-Stop buttons).
    """

    name = "detect_object_realtime"
    description = (
        "Continuously detect objects and update 3D visualization in real-time. "
        "Blocks until stopped by Home/Stop/E-Stop. "
        "Returns the last set of detections when stopped."
    )
    parameters = [
        ToolParameter("query", "str", "Text query describing the object to find."),
        ToolParameter(
            "camera", "str", "Camera to use: 'top', 'left', or 'right'.",
            required=False, default="top",
        ),
    ]

    def __init__(
        self,
        detect_tool: DetectObjectTool,
        stop_event: threading.Event,
        on_detections: Callable[[list[Detection3D], str], None] | None = None,
    ):
        self._detect_tool = detect_tool
        self._stop_event = stop_event
        self._on_detections = on_detections

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        camera: str = kwargs.get("camera", "top")

        self._stop_event.clear()
        last_detections: list[Detection3D] = []
        frame_count = 0

        print(f"[DetectRealtime] Starting continuous detection: query='{query}', camera={camera}")
        print(f"[DetectRealtime] Press Home/Stop/E-Stop to end")

        while not self._stop_event.is_set():
            result = self._detect_tool.execute(query=query, camera=camera)
            if result.success and result.data:
                last_detections = result.data
                frame_count += 1

                # Notify callback (updates Viser + UI bounding boxes)
                if self._on_detections is not None:
                    self._on_detections(last_detections, camera)

            # Small sleep to avoid hammering detection server
            # (detection inference takes ~100-500ms anyway)
            time.sleep(0.05)

        print(f"[DetectRealtime] Stopped after {frame_count} frames")
        return ToolResult(success=True, data=last_detections)
