# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standard observation/debug artifacts for saved scripts.

This module is intentionally safe: it only calls perception/camera/state
readers passed by a script or exposed through ``skill_library.namespace``. It
does not call motion tools.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Union


PromptList = Union[str, list[str], tuple[str, ...]]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size <= 32:
                return value.tolist()
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return repr(value)


def current_run_dir(default: str | Path = "logs/debug_observation_manual") -> Path:
    """Return the active run directory when run under ``run_script.py``."""
    env_dir = os.environ.get("OPENFORGE_DEBUG_LOG_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        from cap.agent.tools import _artifact_log

        vis_dir = getattr(_artifact_log, "_artifact_dir", None)
        if vis_dir is not None:
            return Path(vis_dir).parent
    except Exception:
        pass
    return Path(default)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _tool(name: str) -> Callable[..., Any] | None:
    try:
        import skill_library.namespace as namespace

        fn = getattr(namespace, name, None)
        return fn if callable(fn) else None
    except Exception:
        return None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _call_with_timeout(
    fn: Callable[[], Any],
    *,
    timeout_s: float | None,
    label: str,
) -> tuple[Any | None, str | None]:
    if timeout_s is None or timeout_s <= 0:
        try:
            return fn(), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    out: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            out.put((True, fn()))
        except Exception as exc:
            out.put((False, exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        ok, value = out.get(timeout=timeout_s)
    except queue.Empty:
        return None, f"TimeoutError: {label} exceeded {timeout_s:.1f}s"
    if ok:
        return value, None
    exc = value
    return None, f"{type(exc).__name__}: {exc}"


def _as_prompt_list(prompts: PromptList) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    return [str(prompt) for prompt in prompts]


def _det_value(det: Any, name: str, default: Any = None) -> Any:
    if isinstance(det, dict):
        return det.get(name, default)
    return getattr(det, name, default)


def serialize_detection(det: Any, prompt: str | None = None) -> dict[str, Any] | None:
    """Convert a Detection3D-like object or dict to compact JSON."""
    if det is None:
        return None
    box = (
        _det_value(det, "box_2d")
        or _det_value(det, "bbox")
        or _det_value(det, "bbox_xyxy")
        or _det_value(det, "bbox_xywh")
    )
    position = _det_value(det, "position_3d")
    if position is None:
        position = _det_value(det, "position")
    mask = _det_value(det, "mask")
    mask_area = None
    if mask is not None:
        try:
            import numpy as np

            mask_area = int(np.asarray(mask).astype(bool).sum())
        except Exception:
            mask_area = None
    return {
        "prompt": prompt,
        "label": _det_value(det, "label", prompt),
        "score": _json_safe(_det_value(det, "score")),
        "box_2d": _json_safe(box),
        "position_3d": _json_safe(position),
        "quaternion_xyzw": _json_safe(_det_value(det, "quaternion_xyzw")),
        "rpy": _json_safe(_det_value(det, "rpy")),
        "half_extents": _json_safe(_det_value(det, "half_extents")),
        "mask_area_px": mask_area,
    }


def _strict_detection_map(raw: Any, prompts: list[str]) -> dict[str, list[Any]]:
    """Preserve prompt identity; never substitute another prompt's detection."""
    out = {prompt: [] for prompt in prompts}
    if raw is None:
        return out
    if isinstance(raw, dict):
        for prompt in prompts:
            value = raw.get(prompt) or []
            if isinstance(value, (list, tuple)):
                out[prompt] = list(value)
            elif value:
                out[prompt] = [value]
        return out
    if len(prompts) == 1:
        if isinstance(raw, (list, tuple)):
            out[prompts[0]] = list(raw)
        else:
            out[prompts[0]] = [raw]
    return out


def _capture_detections(
    *,
    camera: str,
    prompts: list[str],
    detect_fn: Callable[..., Any] | None,
    timeout_s: float | None,
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    if detect_fn is None:
        return {prompt: [] for prompt in prompts}, "detect_objects_oneshot unavailable"

    raw, error = _call_with_timeout(
        lambda: detect_fn(prompts, camera=camera),
        timeout_s=timeout_s,
        label=f"detect_objects_oneshot camera={camera}",
    )
    if error and error.startswith("TypeError:"):
        merged: dict[str, list[Any]] = {prompt: [] for prompt in prompts}
        errors: list[str] = []
        for prompt in prompts:
            raw_one, prompt_error = _call_with_timeout(
                lambda prompt=prompt: detect_fn(prompt, camera=camera),
                timeout_s=timeout_s,
                label=f"detect_objects_oneshot prompt={prompt} camera={camera}",
            )
            if prompt_error:
                errors.append(f"{prompt}: {prompt_error}")
            else:
                merged[prompt] = _strict_detection_map(raw_one, [prompt])[prompt]
        raw = merged
        if errors:
            return (
                {
                    prompt: [serialize_detection(det, prompt) for det in dets if det is not None]
                    for prompt, dets in merged.items()
                },
                "; ".join(errors),
            )
    elif error:
        return {prompt: [] for prompt in prompts}, error

    mapped = _strict_detection_map(raw, prompts)
    return (
        {
            prompt: [serialize_detection(det, prompt) for det in dets if det is not None]
            for prompt, dets in mapped.items()
        },
        None,
    )


def _capture_image(
    *,
    camera: str,
    get_camera_fn: Callable[..., Any] | None,
    timeout_s: float | None,
) -> tuple[Any | None, str | None]:
    if get_camera_fn is None:
        return None, "get_camera_image unavailable"

    def _call() -> Any:
        try:
            return get_camera_fn(camera=camera)
        except TypeError:
            return get_camera_fn(camera)

    return _call_with_timeout(
        _call,
        timeout_s=timeout_s,
        label=f"get_camera_image camera={camera}",
    )


def _image_to_uint8_rgb(image: Any) -> Any | None:
    try:
        import numpy as np

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[0] <= 1 or arr.shape[1] <= 1:
            return None
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        return None


def _box_xyxy(box: Any, image_size: tuple[int, int]) -> list[int] | None:
    if not box or len(box) != 4:
        return None
    width, height = image_size
    x0, y0, a, b = [float(v) for v in box]
    if a <= x0 or b <= y0:
        x1, y1 = x0 + a, y0 + b
    else:
        x1, y1 = a, b
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def _draw_overlay(
    *,
    image: Any | None,
    camera: str,
    stage: str,
    detections: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> Any | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    arr = _image_to_uint8_rgb(image)
    if arr is None:
        canvas = Image.new("RGB", (960, 540), color=(24, 28, 34))
    else:
        canvas = Image.fromarray(arr)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    colors = ["lime", "cyan", "yellow", "orange", "magenta", "white"]
    header = [f"stage={stage}", f"camera={camera}"]
    if errors:
        header.extend(errors[:3])
    y = 8
    for line in header:
        draw.text((9, y + 1), line, fill="black", font=font)
        draw.text((8, y), line, fill="white", font=font)
        y += 17

    for idx, (prompt, dets) in enumerate(detections.items()):
        color = colors[idx % len(colors)]
        if not dets:
            draw.text((8, y), f"{prompt}: missing", fill=color, font=small_font)
            y += 15
            continue
        for det_idx, det in enumerate(dets[:5]):
            box = _box_xyxy(det.get("box_2d"), canvas.size)
            if box is not None:
                draw.rectangle(box, outline=color, width=3)
                label_xy = (box[0], max(0, box[1] - 16))
            else:
                label_xy = (8, y)
                y += 15
            label = (
                f"{prompt}[{det_idx}] xyz={det.get('position_3d')} "
                f"score={det.get('score')} area={det.get('mask_area_px')}"
            )
            draw.text((label_xy[0] + 1, label_xy[1] + 1), label, fill="black", font=small_font)
            draw.text(label_xy, label, fill=color, font=small_font)
    return canvas


def save_observation_packet(packet: dict[str, Any], log_dir: str | Path | None = None) -> Path:
    run_dir = Path(log_dir) if log_dir is not None else current_run_dir()
    obs_dir = run_dir / "observations" / f"{packet['stamp']}_{packet['stage']}"
    obs_dir.mkdir(parents=True, exist_ok=True)
    packet_path = obs_dir / "packet.json"
    packet_path.write_text(json.dumps(_json_safe(packet), indent=2) + "\n", encoding="utf-8")
    latest_dir = run_dir / "observations"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "latest.json").write_text(
        json.dumps({"packet": str(packet_path.relative_to(run_dir))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return packet_path


def save_detection_overlays(
    packet: dict[str, Any],
    images: dict[str, Any],
    log_dir: str | Path | None = None,
) -> list[str]:
    run_dir = Path(log_dir) if log_dir is not None else current_run_dir()
    obs_name = f"{packet['stamp']}_{packet['stage']}"
    obs_dir = run_dir / "observations" / obs_name
    vis_dir = run_dir / "vis" / "observations" / obs_name
    obs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for camera, camera_data in packet.get("cameras", {}).items():
        arr = _image_to_uint8_rgb(images.get(camera))
        if arr is not None:
            try:
                from PIL import Image

                raw_path = obs_dir / f"{camera}.png"
                Image.fromarray(arr).save(raw_path)
                camera_data["rgb_path"] = str(raw_path.relative_to(run_dir))
            except Exception as exc:
                camera_data.setdefault("errors", []).append(f"save_rgb: {type(exc).__name__}: {exc}")
        overlay = _draw_overlay(
            image=images.get(camera),
            camera=camera,
            stage=str(packet.get("stage")),
            detections=camera_data.get("detections", {}),
            errors=camera_data.get("errors", []),
        )
        if overlay is None:
            continue
        path = vis_dir / f"{camera}_overlay.png"
        overlay.save(path)
        rel = str(path.relative_to(run_dir))
        camera_data["overlay_path"] = rel
        saved.append(rel)
    return saved


def write_stage_summary(
    *,
    stage: str,
    result: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
    append: bool = True,
) -> Path:
    run_dir = Path(log_dir) if log_dir is not None else current_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "stage_summary.md"
    lines = [
        f"## {time.strftime('%Y-%m-%d %H:%M:%S')} - {stage}",
        "",
    ]
    if result is not None:
        lines.append("### Result")
        lines.append("```json")
        lines.append(json.dumps(_json_safe(result), indent=2))
        lines.append("```")
    if observation is not None:
        lines.append("### Observation")
        for camera, data in observation.get("cameras", {}).items():
            lines.append(f"- `{camera}`: overlay `{data.get('overlay_path')}`")
            for prompt, dets in data.get("detections", {}).items():
                if dets:
                    first = dets[0]
                    lines.append(
                        f"  - `{prompt}`: xyz={first.get('position_3d')} "
                        f"score={first.get('score')} bbox={first.get('box_2d')}"
                    )
                else:
                    lines.append(f"  - `{prompt}`: missing")
            for error in data.get("errors", [])[:3]:
                lines.append(f"  - error: {error}")
    if plan is not None:
        lines.append("### Plan")
        lines.append("```json")
        lines.append(json.dumps(_json_safe(plan), indent=2))
        lines.append("```")
    lines.append("")
    text = "\n".join(lines)
    if append and path.exists():
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def capture_observation(
    *,
    stage: str,
    prompts: PromptList,
    cameras: list[str] | tuple[str, ...] = ("top",),
    image_only_cameras: list[str] | tuple[str, ...] = (),
    detect_fn: Callable[..., Any] | None = None,
    get_camera_fn: Callable[..., Any] | None = None,
    get_robot_state_fn: Callable[..., Any] | None = None,
    log_dir: str | Path | None = None,
    save: bool = True,
    capture_robot_state: bool = True,
    per_call_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Capture a standard observation packet and optional overlays.

    The returned packet is JSON-safe and stable enough for Codex to inspect.
    """
    prompt_list = _as_prompt_list(prompts)
    image_only = {str(camera).strip() for camera in image_only_cameras if str(camera).strip()}
    detect_fn = detect_fn or _tool("detect_objects_oneshot")
    get_camera_fn = get_camera_fn or _tool("get_camera_image")
    if capture_robot_state:
        get_robot_state_fn = get_robot_state_fn or _tool("get_robot_state")
    if per_call_timeout_s is None:
        per_call_timeout_s = _env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 12.0)
    stamp = _timestamp()
    packet: dict[str, Any] = {
        "schema": "openforge.observation.v1",
        "stage": stage,
        "stamp": stamp,
        "physical_motion_executed": False,
        "prompts": prompt_list,
        "cameras": {},
        "robot_state": None,
        "errors": [],
    }
    images: dict[str, Any] = {}
    for camera in cameras:
        cam = str(camera)
        cam_image_only = cam in image_only
        image, image_error = _capture_image(
            camera=cam,
            get_camera_fn=get_camera_fn,
            timeout_s=per_call_timeout_s,
        )
        if cam_image_only:
            detections = {prompt: [] for prompt in prompt_list}
            det_error = None
        else:
            detections, det_error = _capture_detections(
                camera=cam,
                prompts=prompt_list,
                detect_fn=detect_fn,
                timeout_s=per_call_timeout_s,
            )
        errors = [err for err in (image_error, det_error) if err]
        packet["cameras"][cam] = {
            "detections": detections,
            "errors": errors,
            "image_only": cam_image_only,
            "motion_source_allowed": not cam_image_only,
        }
        images[cam] = image
    if not capture_robot_state:
        packet["robot_state"] = "skipped"
    elif get_robot_state_fn is not None:
        try:
            packet["robot_state"] = _json_safe(get_robot_state_fn())
        except Exception as exc:
            packet["errors"].append(f"get_robot_state: {type(exc).__name__}: {exc}")
    else:
        packet["errors"].append("get_robot_state unavailable")
    if save:
        save_detection_overlays(packet, images, log_dir=log_dir)
        packet_path = save_observation_packet(packet, log_dir=log_dir)
        packet["packet_path"] = str(packet_path)
        write_stage_summary(stage=stage, observation=packet, log_dir=log_dir)
    return packet


def save_plan_packet(
    *,
    stage: str,
    selected: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    previews: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
) -> Path:
    run_dir = Path(log_dir) if log_dir is not None else current_run_dir()
    stamp = _timestamp()
    plan_dir = run_dir / "plans" / f"{stamp}_{stage}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    packet = {
        "schema": "openforge.plan.v1",
        "stage": stage,
        "stamp": stamp,
        "selected": selected,
        "candidates": candidates or [],
        "previews": previews or [],
        "observation_packet": observation.get("packet_path") if observation else None,
    }
    path = plan_dir / "plan.json"
    path.write_text(json.dumps(_json_safe(packet), indent=2) + "\n", encoding="utf-8")
    write_stage_summary(stage=stage, plan=packet, log_dir=log_dir)
    return path
