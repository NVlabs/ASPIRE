"""Shared image encoding utilities for VLM backends and reward functions."""

from __future__ import annotations

import base64
import io

import numpy as np


def encode_image_b64(image: np.ndarray) -> str:
    """Encode a numpy RGB array to a base64 JPEG string."""
    from PIL import Image

    img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_image_jpeg(image: np.ndarray) -> bytes:
    """Encode a numpy RGB array to raw JPEG bytes."""
    from PIL import Image

    img = Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
