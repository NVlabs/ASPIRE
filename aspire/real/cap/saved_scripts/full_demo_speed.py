"""Shared speed scaling for YAM saved-script demos."""

from __future__ import annotations

import math
import os


MIN_SPEED_SCALE = 0.25
MAX_SPEED_SCALE = 2.0


def speed_scale() -> float:
    raw = os.environ.get("YAM_FULL_DEMO_SPEED_SCALE", "1.0")
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    if not math.isfinite(value) or value <= 0.0:
        value = 1.0
    return min(MAX_SPEED_SCALE, max(MIN_SPEED_SCALE, value))


def scaled_speed(base: float) -> float:
    return float(base) * speed_scale()


def scaled_duration(base: float) -> float:
    return float(base) / speed_scale()


def scaled_speed_str(base: float) -> str:
    return f"{scaled_speed(base):.3f}"


def scaled_duration_str(base: float) -> str:
    return f"{scaled_duration(base):.3f}"


def set_scaled_default_env(name: str, base: float, *, kind: str = "speed") -> None:
    if name in os.environ:
        return
    if kind == "duration":
        os.environ[name] = scaled_duration_str(base)
    elif kind == "speed":
        os.environ[name] = scaled_speed_str(base)
    else:
        raise ValueError(f"unknown speed scale kind: {kind}")
