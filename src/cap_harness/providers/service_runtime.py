"""Shared provider-service runtime: bounded request concurrency and health.

A single long-lived provider instance is shared over HTTP by the LIBERO and
Robosuite simulator processes. On one GPU the safe design is one
provider instance with explicit, bounded, serialized request handling rather
than one model loaded per simulator process. This module supplies that bound as
a FastAPI middleware plus a uniform ``/healthz`` endpoint, so every provider
behaves the same way and the supervisor/doctor can reason about them uniformly.

The concurrency limit is read from ``CAP_HARNESS_SERVICE_CONCURRENCY`` (set by
the supervisor from the active profile). A limit of 1 serializes GPU access; CPU
providers (e.g. PyRoki) can run a higher bound.
"""

from __future__ import annotations

import asyncio
import os

_UNBOUNDED_PATHS = frozenset({"/openapi.json", "/docs", "/redoc", "/healthz"})


def request_limit(default: int = 1) -> int:
    """Resolve the in-flight request bound from the environment."""
    raw = os.environ.get("CAP_HARNESS_SERVICE_CONCURRENCY")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"CAP_HARNESS_SERVICE_CONCURRENCY must be an integer, got {raw!r}"
        ) from error
    if value < 1:
        raise ValueError("CAP_HARNESS_SERVICE_CONCURRENCY must be >= 1")
    return value


def install_service_runtime(app, *, limit: int | None = None):
    """Bound concurrent inference requests and expose a health endpoint.

    At most ``limit`` inference requests run concurrently; excess requests queue
    on the semaphore instead of oversubscribing the GPU. Health/metadata routes
    are never bounded so liveness checks stay responsive under load.
    """
    resolved = request_limit() if limit is None else limit
    semaphore = asyncio.Semaphore(resolved)
    app.state.request_limit = resolved

    @app.middleware("http")
    async def _bounded(request, call_next):  # pragma: no cover - exercised live
        if request.url.path in _UNBOUNDED_PATHS:
            return await call_next(request)
        async with semaphore:
            return await call_next(request)

    @app.get("/healthz")
    async def _healthz():  # pragma: no cover - exercised live
        return {"ok": True, "request_limit": app.state.request_limit}

    return app
