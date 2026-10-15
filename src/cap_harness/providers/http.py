"""Shared HTTP transport for model-service providers.

The transport is deliberately inert at construction time.  Provider health checks
and inference calls are the only operations that access the network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time
from typing import Any

import requests

from cap_harness.errors import ApiError, ErrorCode

_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ProviderHttpError(RuntimeError):
    """Internal exception carrying the public structured provider failure."""

    def __init__(self, error: ApiError) -> None:
        super().__init__(error.message)
        self.error = error


class HttpProviderClient:
    """Small, injectable JSON client with bounded transient retries."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float | tuple[float, float] = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.1,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if isinstance(timeout_s, tuple):
            if len(timeout_s) != 2 or any(value <= 0 for value in timeout_s):
                raise ValueError("timeout_s tuple must contain positive connect/read timeouts")
        elif timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be non-negative")

        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.session = session if session is not None else requests.Session()
        self._sleep = sleep

    def health(self, path: str = "/openapi.json") -> bool:
        """Return whether the service responds successfully to an explicit probe."""
        try:
            self.request_json("GET", path)
        except ProviderHttpError:
            return False
        return True

    def post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.request_json("POST", path, payload=payload)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue a request and decode a JSON object.

        ``max_retries`` counts retries after the initial attempt, so the total
        number of requests is always bounded by ``max_retries + 1``.
        """
        url = self._url(path)
        total_attempts = self.max_retries + 1
        last_error: BaseException | None = None

        for attempt in range(total_attempts):
            try:
                response = self.session.request(
                    method.upper(),
                    url,
                    json=dict(payload) if payload is not None else None,
                    timeout=self.timeout_s,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt + 1 < total_attempts:
                    self._backoff(attempt)
                    continue
                raise self._unavailable_error(url, total_attempts, exc) from exc
            except requests.RequestException as exc:
                raise ProviderHttpError(
                    ApiError(
                        code=ErrorCode.PROVIDER_UNAVAILABLE,
                        message=f"HTTP request to provider failed: {url}",
                        details={"url": url, "error": str(exc), "attempt": attempt + 1},
                    )
                ) from exc

            if response.status_code in _TRANSIENT_STATUS_CODES:
                last_error = requests.HTTPError(
                    f"transient HTTP {response.status_code}", response=response
                )
                if attempt + 1 < total_attempts:
                    self._backoff(attempt)
                    continue
                raise self._unavailable_error(url, total_attempts, last_error)

            if not 200 <= response.status_code < 300:
                code = (
                    ErrorCode.INVALID_REQUEST
                    if 400 <= response.status_code < 500
                    else ErrorCode.PROVIDER_UNAVAILABLE
                )
                raise ProviderHttpError(
                    ApiError(
                        code=code,
                        message=f"Provider returned HTTP {response.status_code}: {url}",
                        details={
                            "url": url,
                            "status_code": response.status_code,
                            "response": response.text[:512],
                            "attempt": attempt + 1,
                        },
                    )
                )

            try:
                decoded = response.json()
            except (ValueError, requests.RequestException) as exc:
                raise ProviderHttpError(
                    ApiError(
                        code=ErrorCode.INTERNAL,
                        message=f"Provider returned invalid JSON: {url}",
                        recoverable=False,
                        details={"url": url, "response": response.text[:512]},
                    )
                ) from exc

            if not isinstance(decoded, dict):
                raise ProviderHttpError(
                    ApiError(
                        code=ErrorCode.INTERNAL,
                        message=f"Provider JSON response must be an object: {url}",
                        recoverable=False,
                        details={"url": url, "response_type": type(decoded).__name__},
                    )
                )
            return decoded

        # The loop is exhaustive, but retaining an explicit typed failure keeps
        # static analyzers and future retry-policy edits honest.
        raise self._unavailable_error(url, total_attempts, last_error)

    def _url(self, path: str) -> str:
        if not path:
            return self.base_url
        return f"{self.base_url}/{path.lstrip('/')}"

    def _backoff(self, failed_attempt: int) -> None:
        delay = self.retry_backoff_s * (2**failed_attempt)
        if delay > 0:
            self._sleep(delay)

    @staticmethod
    def _unavailable_error(
        url: str, attempts: int, error: BaseException | None
    ) -> ProviderHttpError:
        return ProviderHttpError(
            ApiError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Provider unavailable after {attempts} attempts: {url}",
                details={"url": url, "attempts": attempts, "error": str(error)},
            )
        )


__all__ = ["HttpProviderClient", "ProviderHttpError"]
