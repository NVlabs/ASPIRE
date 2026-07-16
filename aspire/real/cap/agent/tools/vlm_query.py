"""NVIDIA VLM query tool for the openforge runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cap.agent.tools.base import Tool, ToolParameter, ToolResult
from cap.agent.tools.vlm import query as _transport_query  # noqa: F401
from cap.agent.tools.vlm.backends.nvidia import (  # noqa: F401 (back-compat)
    _query_nvidia,
    list_nvidia_keys,
    pick_nvidia_key as _pick_nvidia_key,
)
from cap.config import (
    CAMERA_NAMES,
    CAP_SERVER_PORT,
)

logger = logging.getLogger(__name__)
_CAMERA_NAMES = CAMERA_NAMES
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# VlmQueryTool — agent-loop adapter
# ---------------------------------------------------------------------------
class VlmQueryTool(Tool):
    """Query a vision-language model with text and optional media sources.

    The ``media`` parameter accepts a list of image source strings:

    * ``"camera:top"`` / ``"camera:left"`` / ``"camera:right"`` — live camera
    * ``"local:~/path/to/img.png"`` — local file (supports ~ expansion)
    * ``"web:https://example.com/img.png"`` — download from URL

    Backward compatible: ``camera="top"`` and ``image=<array>`` still work
    but are deprecated in favor of ``media=[...]``.

    The pruned runtime supports the ``nvidia`` backend.
    """

    name = "vlm_query"
    description = (
        "Query a vision-language model with a text prompt and optional images. "
        'Backend: "nvidia" (NVIDIA gateway). '
        'Use media=["camera:top", "local:~/img.png", "web:https://..."] for '
        "flexible image sourcing. Returns the model's text response."
    )
    parameters = [
        ToolParameter("text", "str", "Text prompt or question for the VLM"),
        ToolParameter(
            "backend",
            "str",
            'VLM backend. This runtime supports "nvidia".',
            required=False,
            default="nvidia",
        ),
        ToolParameter(
            "media",
            "list[str]",
            'Image sources: "camera:<name>", "local:<path>", "web:<url>". '
            'Defaults to ["camera:top"].',
            required=False,
            default=None,
        ),
        ToolParameter(
            "camera",
            "str",
            '[DEPRECATED — use media instead] "top", "left", "right", or "all".',
            required=False,
            default=None,
        ),
        ToolParameter(
            "image",
            "Any",
            "[DEPRECATED — use media instead] Numpy RGB array to use directly.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "model",
            "str",
            "Override the default model name for the chosen backend.",
            required=False,
            default=None,
        ),
        ToolParameter(
            "temperature",
            "float",
            "Sampling temperature (0.0 = deterministic). Default 0.2.",
            required=False,
            default=0.2,
        ),
        ToolParameter(
            "reasoning_effort",
            "str",
            "Reserved for model backends that accept thinking effort.",
            required=False,
            default="high",
        ),
    ]

    def __init__(
        self,
        cap_server_host: str = "localhost",
        cap_server_port: int = CAP_SERVER_PORT,
        default_backend: str = "nvidia",
        env=None,
    ):
        self._env = env
        self._cap_host = cap_server_host
        self._cap_port = cap_server_port
        self._default_backend = default_backend
        self._portal_client = None

    # --- image resolution (agent-loop specific) ---------------------------

    def _get_cap_client(self):
        if self._portal_client is None:
            import portal

            self._portal_client = portal.Client(f"{self._cap_host}:{self._cap_port}")
        return self._portal_client

    def _capture_image(self, camera: str) -> np.ndarray | None:
        try:
            if self._env is not None:
                img = self._env.render_rgb(camera)
            else:
                client = self._get_cap_client()
                img = client.get_camera_image(camera).result()
            img = np.asarray(img)
            if img.size < 100:
                return None
            return img
        except Exception as e:
            logger.warning("vlm_query: failed to capture camera %s: %s", camera, e)
            return None

    @staticmethod
    def _load_image_from_file(path_str: str) -> np.ndarray:
        import cv2

        p = Path(path_str)
        if path_str.startswith("~"):
            path = p.expanduser().resolve()
        elif p.is_absolute():
            path = p.resolve()
        else:
            path = (_PROJECT_ROOT / p).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_image_from_url(url: str) -> np.ndarray:
        import cv2
        import urllib.request

        with urllib.request.urlopen(url, timeout=15) as resp:
            data = np.frombuffer(resp.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from URL: {url}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _resolve_media(
        self, media: list[str]
    ) -> tuple[list[np.ndarray], list[str], list[str]]:
        images: list[np.ndarray] = []
        labels: list[str] = []
        errors: list[str] = []
        for src in media:
            try:
                if src.startswith("camera:"):
                    cam_name = src[len("camera:") :]
                    img = self._capture_image(cam_name)
                    if img is not None:
                        images.append(img)
                        labels.append(src)
                    else:
                        errors.append(f"No image from camera:{cam_name}")
                elif src.startswith("local:"):
                    file_path = src[len("local:") :]
                    images.append(self._load_image_from_file(file_path))
                    labels.append(src)
                elif src.startswith("web:"):
                    url = src[len("web:") :]
                    images.append(self._load_image_from_url(url))
                    labels.append(src)
                else:
                    errors.append(
                        f"Unknown media prefix in {src!r}. Use camera:, local:, or web:"
                    )
            except Exception as e:
                errors.append(f"Failed to load {src!r}: {e}")
        return images, labels, errors

    # --- Tool interface ----------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        text: str = kwargs.get("text", "")
        backend: str = kwargs.get("backend", self._default_backend)
        media: list[str] | None = kwargs.get("media", None)
        camera: str | None = kwargs.get("camera", None)
        image_override = kwargs.get("image", None)
        model_override: str | None = kwargs.get("model", None)
        temperature: float = float(kwargs.get("temperature", 0.2))

        if not text:
            return ToolResult(success=False, error="text prompt is required")

        # Collect images from media list or legacy params.
        images: list[np.ndarray] = []
        media_labels: list[str] = []
        if media is not None:
            images, media_labels, media_errors = self._resolve_media(media)
            if media_errors:
                logger.warning("vlm_query media warnings: %s", media_errors)
        elif image_override is not None:
            if isinstance(image_override, list):
                image_labels = kwargs.get("image_labels", [])
                for i, img in enumerate(image_override):
                    images.append(np.asarray(img))
                    lbl = image_labels[i] if i < len(image_labels) else f"image_{i}"
                    media_labels.append(lbl)
            else:
                images = [np.asarray(image_override)]
                media_labels = ["image (provided directly)"]
        elif camera == "all":
            if self._env is not None:
                _cam_list = list(_CAMERA_NAMES)
            else:
                try:
                    _avail = self._get_cap_client().list_cameras().result()
                    _cam_list = _avail.get("cameras", list(_CAMERA_NAMES))
                except Exception:
                    _cam_list = list(_CAMERA_NAMES)
            for cam in _cam_list:
                img = self._capture_image(cam)
                if img is not None:
                    images.append(img)
                    media_labels.append(f"camera:{cam}")
        elif camera is not None:
            img = self._capture_image(camera)
            if img is not None:
                images = [img]
                media_labels = [f"camera:{camera}"]
        else:
            img = self._capture_image("top")
            if img is not None:
                images = [img]
                media_labels = ["camera:top"]

        if not images:
            sources = media if media else (camera or "top")
            return ToolResult(
                success=False, error=f"No images available (sources={sources!r})"
            )

        # Prepend image source labels so the model knows which image is which.
        if len(images) > 1 and media_labels:
            label_lines = ", ".join(
                f"Image {i + 1}: {lbl}" for i, lbl in enumerate(media_labels)
            )
            text = f"[Images: {label_lines}]\n{text}"

        # Forward everything else (api_key, key_index, reasoning_effort, …)
        # to the transport as backend-specific **kw.
        passthrough = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "text",
                "backend",
                "media",
                "camera",
                "image",
                "model",
                "temperature",
                "image_labels",
            }
        }

        try:
            response = _transport_query(
                backend=backend,
                text=text,
                images=images,
                model=model_override,
                temperature=temperature,
                **passthrough,
            )
        except ValueError as e:
            # Unknown backend — make the message readable to the agent.
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            logger.error("vlm_query [%s] failed: %s", backend, e)
            return ToolResult(
                success=False, error=f"VLM query failed ({backend}): {e}"
            )

        logger.info(
            "vlm_query [%s/%s] images=%d response=%s",
            backend,
            model_override or "<default>",
            len(images),
            response[:120] if response else "",
        )

        from cap.agent.tools._artifact_log import log_vlm_query

        log_vlm_query(
            prompt=text,
            response=response,
            images=images,
            media_labels=media_labels,
            backend=backend,
        )

        return ToolResult(success=True, data=response)
