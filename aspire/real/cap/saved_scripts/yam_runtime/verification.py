"""Verification helpers for YAM saved scripts."""

from __future__ import annotations

import math
from typing import Any


def _xyz(det: dict[str, Any] | None) -> list[float] | None:
    if not det:
        return None
    value = det.get("position_3d") or det.get("position")
    if not value or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def compare_detection_motion(pre_detection: dict[str, Any] | None, post_detection: dict[str, Any] | None) -> dict[str, Any]:
    pre = _xyz(pre_detection)
    post = _xyz(post_detection)
    if pre is None or post is None:
        return {"ok": False, "reason": "missing pre or post detection", "pre": pre, "post": post}
    delta = [post[i] - pre[i] for i in range(3)]
    xy = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])
    return {
        "ok": True,
        "pre": pre,
        "post": post,
        "delta": [round(v, 5) for v in delta],
        "xy_delta_m": round(xy, 5),
        "z_delta_m": round(delta[2], 5),
    }


def verify_lift(
    pre_detection: dict[str, Any] | None,
    post_detection: dict[str, Any] | None,
    *,
    min_z_delta_m: float = 0.05,
    max_xy_delta_m: float = 0.20,
) -> dict[str, Any]:
    motion = compare_detection_motion(pre_detection, post_detection)
    if not motion.get("ok"):
        return {"success": False, "confidence": 0.0, "motion": motion, "reason": motion.get("reason")}
    z_ok = float(motion["z_delta_m"]) >= float(min_z_delta_m)
    xy_ok = float(motion["xy_delta_m"]) <= float(max_xy_delta_m)
    success = bool(z_ok and xy_ok)
    return {
        "success": success,
        "confidence": 0.8 if success else 0.2,
        "motion": motion,
        "reason": "lift_verified" if success else "post detection did not show a clean lift",
        "thresholds": {
            "min_z_delta_m": float(min_z_delta_m),
            "max_xy_delta_m": float(max_xy_delta_m),
        },
    }
