# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only scene capture helpers for YAM saved scripts."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import append_stage_summary, call_with_timeout, current_run_dir, json_safe, write_json


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        arr = list(values)
        if limit is not None:
            arr = arr[:limit]
        return [round(float(v), digits) for v in arr]
    except Exception:
        return []


def first_detection(det_map: Any, prompt: str) -> Any | None:
    """Return the first detection for a prompt from list or dict outputs."""
    if isinstance(det_map, dict):
        dets = det_map.get(prompt) or det_map.get(str(prompt)) or []
        if not dets and len(det_map) == 1:
            dets = next(iter(det_map.values())) or []
    else:
        dets = det_map or []
    if isinstance(dets, dict):
        return dets
    return dets[0] if dets else None


def serialize_detection(
    det: Any,
    prompt: str,
    *,
    camera: str | None = None,
    source: str = "detect_objects_oneshot",
) -> dict[str, Any] | None:
    if det is None:
        return None
    xyz = _round_list(_field(det, "position_3d", _field(det, "position", None)), limit=3)
    if len(xyz) != 3:
        return None
    out = {
        "prompt": prompt,
        "label": _field(det, "label", prompt),
        "score": _field(det, "score", None),
        "box_2d": json_safe(_field(det, "box_2d", _field(det, "bbox", []))),
        "position_3d": xyz,
        "position_3d_source": _field(det, "position_3d_source", None),
        "quaternion_xyzw": _round_list(_field(det, "quaternion_xyzw", []), limit=4),
        "rpy": _round_list(_field(det, "rpy", []), limit=3),
        "half_extents": _round_list(_field(det, "half_extents", []), limit=3),
        "source": source,
    }
    if camera:
        out["source_camera"] = camera
        out["camera"] = camera
    return out


def _detections_for_map(det_map: Any, prompts: list[str], camera: str) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for prompt in prompts:
        if isinstance(det_map, dict):
            dets = det_map.get(prompt) or []
        else:
            dets = det_map or []
        if isinstance(dets, dict):
            dets = [dets]
        for det in dets:
            item = serialize_detection(det, prompt, camera=camera)
            if item is not None:
                serialized.append(item)
    return serialized


def _select_best(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None
    return sorted(detections, key=lambda d: float(d.get("score") or 0.0), reverse=True)[0]


def _save_rgb(image: Any, path: Path) -> str | None:
    if image is None:
        return None
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[0] <= 1 or arr.shape[1] <= 1:
            return None
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)
        return str(path)
    except Exception:
        return None


def _save_overlay(image: Any, detections: list[dict[str, Any]], path: Path, title: str) -> str | None:
    if image is None:
        return None
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), title, fill=(0, 255, 0))
        for det in detections:
            box = det.get("box_2d") or []
            if len(box) >= 4:
                x0, y0, x1, y1 = [float(v) for v in box[:4]]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 80, 80), width=3)
                label = f"{det.get('prompt')} {float(det.get('score') or 0):.2f}"
                draw.text((x0, max(0, y0 - 16)), label, fill=(255, 80, 80))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return str(path)
    except Exception:
        return None


_CAMERA_PORTAL_CLIENT: Any | None = None


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _portal_camera_image(camera: str) -> Any | None:
    global _CAMERA_PORTAL_CLIENT
    if not _truthy_env("OPENFORGE_YAM_RUNTIME_CAMERA_PORTAL_FALLBACK", True):
        return None
    addr = (
        os.environ.get("OPENFORGE_YAM_RUNTIME_CAMERA_PORTAL")
        or os.environ.get("OPENFORGE_DEBUG_OBS_CAMERA_PORTAL")
        or "localhost:8300"
    ).strip()
    if not addr:
        return None
    try:
        import portal

        if _CAMERA_PORTAL_CLIENT is None:
            _CAMERA_PORTAL_CLIENT = portal.Client(addr)
        return _CAMERA_PORTAL_CLIENT.get_camera_image(str(camera)).result()
    except Exception:
        return None


def capture_scene(
    *,
    prompts: list[str],
    cameras: list[str] | None = None,
    detect_objects_oneshot: Callable[..., Any],
    get_camera_image: Callable[..., Any] | None = None,
    get_robot_state: Callable[..., Any] | None = None,
    run_in_background: Callable[..., Any] | None = None,
    run_dir: str | Path | None = None,
    stage: str = "observe",
    task_name: str = "yam_runtime",
    timeout_s: float = 20.0,
    max_retries: int = 2,
    save_overlays: bool = True,
    motion_cameras: tuple[str, ...] = ("top", "left", "right"),
    image_only_cameras: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Capture detections, optional overlays, and robot state without motion."""
    run_dir = Path(run_dir) if run_dir is not None else current_run_dir(task_name)
    cameras = cameras or ["top"]
    obs_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{stage}"
    obs_dir = run_dir / "observations" / obs_id
    vis_dir = run_dir / "vis" / "observations" / obs_id
    packet: dict[str, Any] = {
        "schema": "openforge.yam_runtime.scene.v1",
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prompts": list(prompts),
        "cameras": {},
        "robot_state": None,
        "best_detection": None,
        "all_detections": [],
        "errors": [],
    }
    all_detections: list[dict[str, Any]] = []
    image_only = {str(camera).strip() for camera in image_only_cameras if str(camera).strip()}
    for camera in cameras:
        cam_image_only = camera in image_only
        cam_packet: dict[str, Any] = {
            "detections": {},
            "serialized_detections": [],
            "rgb_path": None,
            "overlay_path": None,
            "image_source": None,
            "errors": [],
            "image_only": cam_image_only,
            "motion_source_allowed": bool(camera in motion_cameras and not cam_image_only),
        }
        if cam_image_only:
            serialized = []
            cam_packet["detections"] = {prompt: [] for prompt in list(prompts)}
        else:
            det_result = call_with_timeout(
                f"detect_objects_oneshot:{camera}",
                detect_objects_oneshot,
                timeout_s,
                list(prompts),
                camera=camera,
                max_retries=max_retries,
                run_in_background=run_in_background,
            )
            if det_result["ok"]:
                det_map = det_result["data"]
                serialized = _detections_for_map(det_map, list(prompts), camera)
                cam_packet["detections"] = json_safe(det_map)
                cam_packet["serialized_detections"] = serialized
                if camera in motion_cameras:
                    all_detections.extend(serialized)
            else:
                serialized = []
                cam_packet["errors"].append(det_result["error"])
                packet["errors"].append(f"{camera}: {det_result['error']}")
        if save_overlays and callable(get_camera_image):
            img_result = call_with_timeout(
                f"get_camera_image:{camera}",
                get_camera_image,
                timeout_s,
                camera,
                run_in_background=run_in_background,
            )
            if img_result["ok"]:
                image = img_result["data"]
                cam_packet["rgb_path"] = _save_rgb(image, obs_dir / f"{camera}.png")
                if cam_packet["rgb_path"] is not None:
                    cam_packet["image_source"] = "run_script_get_camera_image"
                else:
                    fallback = _portal_camera_image(camera)
                    if fallback is not None:
                        fallback_path = _save_rgb(fallback, obs_dir / f"{camera}.png")
                        if fallback_path is not None:
                            image = fallback
                            cam_packet["rgb_path"] = fallback_path
                            cam_packet["image_source"] = "camera_portal_fallback"
                cam_packet["overlay_path"] = _save_overlay(
                    image,
                    cam_packet["serialized_detections"],
                    vis_dir / f"{camera}_overlay.png",
                    f"{stage}:{camera}",
                )
                if cam_packet["rgb_path"] is None:
                    cam_packet["errors"].append("get_camera_image: returned no usable RGB image")
            else:
                cam_packet["errors"].append(f"get_camera_image: {img_result['error']}")
        packet["cameras"][camera] = cam_packet
    if callable(get_robot_state):
        state_result = call_with_timeout(
            "get_robot_state",
            get_robot_state,
            min(timeout_s, 8.0),
            run_in_background=run_in_background,
        )
        packet["robot_state"] = json_safe(state_result["data"]) if state_result["ok"] else {"error": state_result["error"]}
    packet["all_detections"] = all_detections
    packet["best_detection"] = _select_best(all_detections)
    packet["packet_path"] = write_json(obs_dir / "packet.json", packet)
    packet["latest_path"] = write_json(run_dir / "observations" / "latest.json", packet)
    append_stage_summary(
        run_dir,
        [
            f"## observation {stage}",
            f"- packet: {packet['packet_path']}",
            f"- best_detection: {packet['best_detection']}",
            f"- errors: {packet['errors']}",
        ],
    )
    return packet
