"""Structured operational errors returned by harness APIs.

Invalid construction or misuse of a contract is a programming error and raises
``ValueError``.  ``ApiError`` is instead carried by result objects for expected,
recoverable failures such as an unreachable IK target or an unavailable provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class ErrorCode(str, Enum):
    """Stable machine-readable categories for recoverable API failures."""

    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    PERCEPTION_FAILED = "perception_failed"
    NO_SEGMENTATION = "no_segmentation"
    POINT_CLOUD_FAILED = "point_cloud_failed"
    GRASP_FAILED = "grasp_failed"
    NO_GRASP = "no_grasp"
    IK_FAILED = "ik_failed"
    IK_UNREACHABLE = "ik_unreachable"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"
    CONTROLLER_FAILED = "controller_failed"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    ADAPTER_FAILED = "adapter_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    STALE_PLAN = "stale_plan"
    SAFETY_INTERLOCK = "safety_interlock"
    UNSUPPORTED = "unsupported"
    INTERNAL = "internal"


def _details(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("details must be a mapping")
    copied: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("detail keys must be non-empty strings")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ApiError(Exception):
    """An operational failure suitable for returning across an API boundary."""

    code: ErrorCode
    message: str
    recoverable: bool = True
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        code = self.code
        if isinstance(code, str) and not isinstance(code, ErrorCode):
            try:
                code = ErrorCode(code)
            except ValueError as exc:
                raise ValueError(f"unknown error code: {self.code!r}") from exc
        if not isinstance(code, ErrorCode):
            raise ValueError("code must be an ErrorCode")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        if type(self.recoverable) is not bool:
            raise ValueError("recoverable must be a bool")

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "details", _details(self.details))
        object.__setattr__(self, "args", (self.message,))

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"


__all__ = ["ApiError", "ErrorCode"]
