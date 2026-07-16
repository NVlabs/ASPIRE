"""NVIDIA inference gateway — OpenAI-compatible, routes to Gemini / Bedrock
Claude / Azure GPT / etc. based on the ``model`` string.

Auth supports two modes:
  - single key via ``NVIDIA_API_KEY``
  - rotated keys via ``NVIDIA_API_KEY_1..N`` (probed 1..100, gaps allowed)

Key rotation is handled by :mod:`cap.agent.providers.nvidia` using a
thread-safe atomic counter. The key list is shuffled once per process at
import time so concurrent experiments sharing the same N keys start at
different offsets and distribute load uniformly across all keys.

Pass an explicit ``api_key`` kwarg to override (e.g. the Phase-5 Queue-based
per-seed VLM reflection, which needs strict one-call-per-key semantics for
rate-limit control).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cap.agent.providers.nvidia import (
    _auto_pick_key,
    auto_pick_nvidia_key,
    list_nvidia_keys,
    nvidia_model_supports_temperature,
    pick_nvidia_key,
    post_chat_completions,
    response_text,
)
from cap.agent.tools.vlm.transport import register
from cap.config import NVIDIA_VL_BASE_URL, NVIDIA_VL_MODEL
from cap.utils.image import encode_image_b64

__all__ = [
    "NvidiaBackend",
    "_auto_pick_key",
    "_query_nvidia",
    "list_nvidia_keys",
    "nvidia_model_supports_temperature",
    "pick_nvidia_key",
]


def _query_nvidia(
    text: str,
    images: list[np.ndarray],
    model: str,
    api_key: str | None,
    temperature: float = 0.2,
    telemetry_source: str | None = None,
) -> str:
    """Low-level function used by legacy callers. Kept stable."""
    content: list[dict] = []
    for img in images:
        b64 = encode_image_b64(img)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    content.append({"type": "text", "text": text})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    if nvidia_model_supports_temperature(model):
        kwargs["temperature"] = temperature

    resp = post_chat_completions(
        kwargs,
        api_key=api_key,
        base_url=NVIDIA_VL_BASE_URL,
        telemetry_source=telemetry_source or "vlm",
    )
    return response_text(resp)


class NvidiaBackend:
    """NVIDIA inference gateway. Caller-supplied ``api_key`` preferred
    (for Queue-based pool control); otherwise rotate via ``key_index``.
    """

    name = "nvidia"

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
        key_index: int | None = None,
        telemetry_source: str | None = None,
        **_kw: Any,
    ) -> str:
        # Explicit api_key wins (Phase-5 Queue borrow/return, tests).
        # Explicit key_index is legacy — still honoured.
        # Default: automatic round-robin via module-level counter.
        if api_key:
            key = api_key
        elif key_index is not None:
            key = pick_nvidia_key(key_index) or auto_pick_nvidia_key()
        else:
            key = None
        return _query_nvidia(
            text,
            images,
            model or NVIDIA_VL_MODEL,
            key,
            temperature,
            telemetry_source=telemetry_source,
        )


register("nvidia", NvidiaBackend)
