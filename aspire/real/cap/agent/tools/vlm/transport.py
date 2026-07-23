# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM dispatch for the pruned NVIDIA backend."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np


class VLMBackend(Protocol):
    """A VLM provider. Implementations live in :mod:`cap.agent.tools.vlm.backends`."""

    name: str

    def generate(
        self,
        *,
        text: str,
        images: list[np.ndarray],
        model: str | None = None,
        temperature: float = 0.2,
        **kw: Any,
    ) -> str:
        """Return the model's text response for ``text`` + ``images``.

        Extra provider-specific kwargs (``api_key``, ``reasoning_effort``,
        ``thinking_budget``, ``api_url``, …) are passed through ``**kw``.
        Each backend consumes what it needs and ignores the rest.
        """


_BACKENDS: dict[str, Callable[[], VLMBackend]] = {}


def register(name: str, factory: Callable[[], VLMBackend]) -> None:
    """Register ``factory`` under ``name``. Idempotent (last caller wins)."""
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    """Return the list of registered backend names (sorted)."""
    return sorted(_BACKENDS)


def query(
    backend: str,
    *,
    text: str,
    images: list[np.ndarray],
    model: str | None = None,
    temperature: float = 0.2,
    **kw: Any,
) -> str:
    """Route a VLM call to the registered backend.

    Raises:
        ValueError: if ``backend`` is not registered. Error message lists
            all registered names so the fix is obvious.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown VLM backend {backend!r}. Available: {available_backends()}"
        )
    return _BACKENDS[backend]().generate(
        text=text, images=images, model=model, temperature=temperature, **kw
    )
