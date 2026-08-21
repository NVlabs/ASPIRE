# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick a green apple from the basket and place it on the blue plate.

This script is a physical-motion script. It refuses to run unless:

- OPENFORGE_ALLOW_PHYSICAL_MOTION=1 is set by the operator
- real AnyGrasp health is OK when OPENFORGE_GRASP_MODE uses AnyGrasp
- BundleSDF/SAM3 localization is reachable when OPENFORGE_GRASP_MODE=topdown_geom
- the selected BundleSDF preview stream returns a usable frame

It intentionally uses the existing pick_place skill library instead of issuing
low-level joint commands here.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from skill_library.namespace import get_camera_image
from skill_library.pick_place import pick_and_place


OBJECT_NAME = os.environ.get("OPENFORGE_OBJECT_NAME", "green apple")
TARGET_NAME = os.environ.get("OPENFORGE_TARGET_NAME", "blue plate")
CAMERA = os.environ.get("OPENFORGE_CAMERA", "top")
GRASP_MODE = os.environ.get("OPENFORGE_GRASP_MODE", "topdown_geom").strip().lower()
TARGET_DROP_Z_OFFSET = float(os.environ.get("OPENFORGE_TARGET_DROP_Z_OFFSET", "0.10"))
MAX_ATTEMPTS = int(os.environ.get("OPENFORGE_MAX_GRASP_ATTEMPTS", "2"))
MAX_GRASPS = int(os.environ.get("OPENFORGE_MAX_GRASPS", "32"))
TOP_GRASP_TRY = int(os.environ.get("OPENFORGE_TOP_GRASP_TRY", "24"))
BATCH_TOP_K = int(os.environ.get("OPENFORGE_BATCH_TOP_K", str(TOP_GRASP_TRY)))
TOPDOWN_GRASP_Z_OFFSET_M = float(os.environ.get("OPENFORGE_TOPDOWN_GRASP_Z_OFFSET_M", "0.0"))
TOPDOWN_GRASP_WIDTH_M = float(os.environ.get("OPENFORGE_TOPDOWN_GRASP_WIDTH_M", "0.08"))
TOPDOWN_GRASP_YAWS = os.environ.get("OPENFORGE_TOPDOWN_GRASP_YAWS", "")
TOPDOWN_GRASP_PITCHES = os.environ.get("OPENFORGE_TOPDOWN_GRASP_PITCHES", "180,165,150,135")

TASK_RESULT = {
    "success": False,
    "reward": 0.0,
    "method": "pick_green_apple_from_basket_to_plate",
    "object": OBJECT_NAME,
    "target": TARGET_NAME,
    "grasp_mode": GRASP_MODE,
    "camera": CAMERA,
}


def get_task_info():
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _anygrasp_base_url() -> str:
    if os.environ.get("ANYGRASP_SERVICE_URL"):
        return os.environ["ANYGRASP_SERVICE_URL"].rstrip("/")
    host = os.environ.get("ANYGRASP_SERVER_HOST", "127.0.0.1")
    port = os.environ.get("ANYGRASP_SERVER_PORT", "8122")
    return f"http://{host}:{port}"


def _bundlesdf_base_url() -> str:
    if os.environ.get("BUNDLESDF_SERVICE_URL"):
        return os.environ["BUNDLESDF_SERVICE_URL"].rstrip("/")
    host = os.environ.get("BUNDLESDF_SERVER_HOST", "127.0.0.1")
    port = os.environ.get("BUNDLESDF_PORT", "8119")
    return f"http://{host}:{port}"


def _json_get(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _uses_anygrasp(grasp_mode: str) -> bool:
    return grasp_mode in {"anygrasp", "top_anygrasp"}


def _check_bundlesdf_preview(bundlesdf_url: str, camera: str) -> None:
    preview_url = f"{bundlesdf_url}/preview/{camera}"
    try:
        with urllib.request.urlopen(preview_url, timeout=5.0) as response:
            chunk = response.read(256)
            content_type = response.headers.get("content-type", "")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"BundleSDF preview check failed at {preview_url}: {exc}") from exc
    if "multipart/x-mixed-replace" not in content_type or b"\xff\xd8" not in chunk:
        raise RuntimeError(
            f"BundleSDF preview {preview_url} did not return an MJPEG frame: "
            f"content_type={content_type!r}, first_bytes={chunk[:32]!r}"
        )
    print(f"[green_apple_basket_to_plate] BundleSDF preview ok: {preview_url}")


def _preflight() -> None:
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError(
            "Refusing physical motion. Set OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
            "only after workspace is clear, hands are clear, and E-stop is reachable."
        )

    if _uses_anygrasp(GRASP_MODE):
        anygrasp_url = _anygrasp_base_url()
        try:
            health = _json_get(f"{anygrasp_url}/health")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"AnyGrasp health check failed at {anygrasp_url}: {exc}") from exc

        if health.get("mock") is True or health.get("safe_for_robot_motion") is False:
            raise RuntimeError(
                "Refusing to use mock/synthetic AnyGrasp for physical motion. "
                f"health={health}"
            )
        print(f"[green_apple_basket_to_plate] AnyGrasp health ok: {health}")

        image = np.asarray(get_camera_image(CAMERA))
        if image.ndim != 3 or image.shape[0] < 100 or image.shape[1] < 100:
            raise RuntimeError(
                f"Camera {CAMERA!r} did not return a usable RGB frame: {image.shape}"
            )
        print(f"[green_apple_basket_to_plate] Camera {CAMERA!r} frame ok: shape={image.shape}")
    else:
        bundlesdf_url = _bundlesdf_base_url()
        try:
            health = _json_get(f"{bundlesdf_url}/health")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"BundleSDF/SAM3 localization health check failed at {bundlesdf_url}: {exc}"
            ) from exc
        print(f"[green_apple_basket_to_plate] BundleSDF/SAM3 health ok: {health}")
        _check_bundlesdf_preview(bundlesdf_url, CAMERA)


print("[green_apple_basket_to_plate] Starting task.")
print(f"[green_apple_basket_to_plate] object={OBJECT_NAME!r}, target={TARGET_NAME!r}")
print(f"[green_apple_basket_to_plate] grasp_mode={GRASP_MODE!r}, camera={CAMERA!r}")
print("[green_apple_basket_to_plate] This script will command physical robot motion if preflight passes.")

_preflight()

ok = bool(
    pick_and_place(
        OBJECT_NAME,
        TARGET_NAME,
        grasp_mode=GRASP_MODE,
        camera=CAMERA,
        bundlesdf_camera=CAMERA,
        max_attempts=MAX_ATTEMPTS,
        max_grasps=MAX_GRASPS,
        top_grasp_try=TOP_GRASP_TRY,
        batch_top_k=BATCH_TOP_K,
        topdown_z_offset_m=TOPDOWN_GRASP_Z_OFFSET_M,
        topdown_width_m=TOPDOWN_GRASP_WIDTH_M,
        topdown_yaws=TOPDOWN_GRASP_YAWS,
        topdown_pitches=TOPDOWN_GRASP_PITCHES,
        target_drop_z_offset=TARGET_DROP_Z_OFFSET,
        target_drop_z_offsets={
            "blue plate": TARGET_DROP_Z_OFFSET,
            "plate": TARGET_DROP_Z_OFFSET,
        },
    )
)

TASK_RESULT.update(
    {
        "success": ok,
        "reward": 1.0 if ok else 0.0,
        "grasp_mode": GRASP_MODE,
        "camera": CAMERA,
        "target_drop_z_offset": TARGET_DROP_Z_OFFSET,
        "topdown_grasp_z_offset_m": TOPDOWN_GRASP_Z_OFFSET_M,
        "topdown_grasp_width_m": TOPDOWN_GRASP_WIDTH_M,
        "topdown_grasp_pitches": TOPDOWN_GRASP_PITCHES,
        "batch_top_k": BATCH_TOP_K,
    }
)

if not ok:
    raise RuntimeError(f"pick_and_place({OBJECT_NAME!r}, {TARGET_NAME!r}) returned False")

print("[green_apple_basket_to_plate] Done.")
