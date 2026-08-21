# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long read-only visualization run for real-YAM UI inspection.

This script is intentionally safe by default:

- no go_home
- no gripper commands
- no freespace_move
- no grasp execution
- no BundleSDF detections

It keeps run_script.py alive long enough to inspect:

- Real-YAM terminal dashboard
- browser debug UI on :8788
- stdout/profiling/debug events
- saved visual artifacts under logs/<run>/vis/

Run with runtime.no_cameras=true by default to avoid fighting the standalone
:8300 camera portal. Set OPENFORGE_VIS_USE_CAMERA=1 only when you intentionally
want this process to open the real cameras directly.
"""

from __future__ import annotations

import math
import os
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np

from cap.agent.tools._artifact_log import log_detection, log_grasp, log_vlm_query
from cap.agent.tools.base import Detection3D
from skill_library.namespace import display_rpy_to_quat, get_camera_image, get_robot_state


RUN_SECONDS = float(os.environ.get("OPENFORGE_VIS_RUN_SECONDS", "1800"))
STATUS_INTERVAL_S = float(os.environ.get("OPENFORGE_VIS_STATUS_INTERVAL_S", "2"))
ARTIFACT_INTERVAL_S = float(os.environ.get("OPENFORGE_VIS_ARTIFACT_INTERVAL_S", "10"))
SERVICE_INTERVAL_S = float(os.environ.get("OPENFORGE_VIS_SERVICE_INTERVAL_S", "10"))
USE_CAMERA = os.environ.get("OPENFORGE_VIS_USE_CAMERA", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def get_task_info():
    """Override run_script's default VLM reward for this visualization-only run."""
    return {
        "success": True,
        "reward": 0.0,
        "method": "visualization_long_run",
        "details": {
            "safe_read_only": True,
            "run_seconds": RUN_SECONDS,
            "use_camera": USE_CAMERA,
        },
    }


def _service_json(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return response.read(300).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"


def _synthetic_rgb(t: float, tick: int, height: int = 480, width: int = 640) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    wave = 0.5 + 0.5 * np.sin((x * 8.0) + (y * 5.0) + t)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((x * 255.0) + (tick * 7 % 80), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(wave * 255.0, 0, 255).astype(np.uint8)

    # Draw simple crosshair/box without depending on PIL here.
    cy = height // 2 + int(math.sin(t * 0.7) * height * 0.18)
    cx = width // 2 + int(math.cos(t * 0.5) * width * 0.18)
    rgb[max(0, cy - 2) : min(height, cy + 3), :, :] = [255, 255, 255]
    rgb[:, max(0, cx - 2) : min(width, cx + 3), :] = [255, 255, 255]
    rgb[max(0, cy - 60) : min(height, cy - 54), max(0, cx - 90) : min(width, cx + 90)] = [
        255,
        80,
        80,
    ]
    rgb[max(0, cy + 54) : min(height, cy + 60), max(0, cx - 90) : min(width, cx + 90)] = [
        255,
        80,
        80,
    ]
    rgb[max(0, cy - 60) : min(height, cy + 60), max(0, cx - 90) : min(width, cx - 84)] = [
        255,
        80,
        80,
    ]
    rgb[max(0, cy - 60) : min(height, cy + 60), max(0, cx + 84) : min(width, cx + 90)] = [
        255,
        80,
        80,
    ]
    return rgb


def _camera_or_synthetic(t: float, tick: int) -> np.ndarray:
    if not USE_CAMERA:
        return _synthetic_rgb(t, tick)
    try:
        image = get_camera_image(camera="top")
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[1] > 1:
            return arr[:, :, :3].astype(np.uint8, copy=False)
    except Exception as exc:
        print(f"[visualization_long_run] camera image unavailable, using synthetic: {exc}")
    return _synthetic_rgb(t, tick)


def _write_artifacts(t: float, tick: int) -> None:
    image = _camera_or_synthetic(t, tick)
    h, w = image.shape[:2]
    box = [w * 0.35, h * 0.30, w * 0.65, h * 0.62]
    detection = Detection3D(
        label=f"visualization tick {tick}",
        score=1.0,
        box_2d=box,
        position_3d=[0.50, 0.00, 0.20 + 0.02 * math.sin(t)],
        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        rpy=[0.0, 0.0, 0.0],
    )
    log_detection(image, [detection], tag="visualization_long_run")

    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - w * 0.5) ** 2 / (w * 0.18) ** 2 + (yy - h * 0.48) ** 2 / (h * 0.22) ** 2) <= 1.0
    grasps = [
        SimpleNamespace(
            position=[0.48 + 0.01 * i, -0.03 + 0.03 * i, 0.18],
            rpy=[180.0, 0.0, -30.0 + 30.0 * i],
            score=0.90 - 0.08 * i,
            width=0.055 + 0.004 * i,
        )
        for i in range(3)
    ]
    log_grasp(image, mask, grasps, query=f"visualization tick {tick}", tag="visualization_long_run")

    log_vlm_query(
        prompt=(
            "Synthetic visualization heartbeat. This file is produced locally so "
            "the browser debug UI has image artifacts to display; no VLM request "
            "was made."
        ),
        response=f"tick={tick}, elapsed_s={t:.1f}, use_camera={USE_CAMERA}",
        images=[image],
        media_labels=[f"heartbeat_tick_{tick}"],
        backend="local-no-vlm",
        tag="visualization_long_run",
    )


print("[visualization_long_run] Starting read-only visualization run.")
print(f"[visualization_long_run] duration={RUN_SECONDS}s")
print(f"[visualization_long_run] status_interval={STATUS_INTERVAL_S}s")
print(f"[visualization_long_run] artifact_interval={ARTIFACT_INTERVAL_S}s")
print(f"[visualization_long_run] use_camera={USE_CAMERA}")
print("[visualization_long_run] This script does not command robot motion.")
print("[visualization_long_run] Watch tmux dashboard, browser debug UI :8788, and BundleSDF :8119.")

start = time.monotonic()
next_status = 0.0
next_artifact = 0.0
next_service = 0.0
tick = 0

while True:
    elapsed = time.monotonic() - start
    if elapsed >= RUN_SECONDS:
        break

    if elapsed >= next_status:
        tick += 1
        print(f"[visualization_long_run] tick={tick} elapsed={elapsed:.1f}s")
        try:
            state = get_robot_state()
            left = state.arms.get("left")
            right = state.arms.get("right")
            if left is not None and right is not None:
                print(
                    "[visualization_long_run] arm summary "
                    f"L_j0={left.joint_pos[0]:+.4f} R_j0={right.joint_pos[0]:+.4f} "
                    f"L_grip={left.gripper_pos:.3f} R_grip={right.gripper_pos:.3f}"
                )
        except Exception as exc:
            print(f"[visualization_long_run] get_robot_state failed: {exc}")

        # Safe, non-motion tool call so dashboard/debug UI show a visible tool event.
        quat = display_rpy_to_quat([0.0, 0.0, float((tick * 15) % 360)])
        print(f"[visualization_long_run] display_rpy_to_quat -> {np.asarray(quat).round(4).tolist()}")
        next_status = elapsed + STATUS_INTERVAL_S

    if elapsed >= next_artifact:
        print("[visualization_long_run] writing visual artifacts")
        _write_artifacts(elapsed, tick)
        next_artifact = elapsed + ARTIFACT_INTERVAL_S

    if elapsed >= next_service:
        print("[visualization_long_run] service health snapshots:")
        print(f"  BundleSDF :8119/health -> {_service_json('http://127.0.0.1:8119/health')}")
        print(f"  SAM3      :6767/health -> {_service_json('http://127.0.0.1:6767/health')}")
        print(f"  AnyGrasp  :8122/health -> {_service_json('http://127.0.0.1:8122/health')}")
        next_service = elapsed + SERVICE_INTERVAL_S

    time.sleep(0.2)

print("[visualization_long_run] Finished.")
