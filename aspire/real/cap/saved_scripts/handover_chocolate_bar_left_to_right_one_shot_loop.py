# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused KitKat/red-wrapper handover for real YAM.

This is intentionally a single-path script, not the long experiment notebook it
replaced. The sequence is:

1. observe KitKat/red wrapper and white trash can from live perception.
2. left arm grasps near one short endpoint with a top-down pinch.
3. left lifts, clears to the post-handover retreat lane, then presents the free end.
4. right arm closes on the free end.
5. left opens, both arms retreat apart; the right retreat can include a small -X
   pullback before the trash route.
6. right rotates to the configured outward-facing trash RPY, stages on the
   operator/front side of the live bin, moves over the bin from that side,
   lowers to the drop height, opens, and retreats back out through the same
   front-side corridor.

Physical motion still requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1 plus the
explicit handover ticket envs. Missing live detections or failed cuRobo previews
stop the run before the next motion.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from cap.saved_scripts.full_demo_speed import scaled_speed
from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    call_with_timeout,
    capture_scene,
    current_run_dir,
    json_safe,
    read_gripper_width,
    write_json,
)


TASK_NAME = "handover_chocolate_bar_left_to_right_one_shot_loop"
TASK_RESULT: dict[str, Any] = {}

DEFAULT_OBJECT_PROMPTS: tuple[str, ...] = (
    "kitkat bar",
    "kitkat",
    "KitKat",
    "KitKat wrapper",
    "kit kat",
    "kit kat bar",
    "kit kat wrapper",
    "red KitKat wrapper",
    "red wrapper",
    "small red wrapper",
    "red rectangular wrapper",
    "red rectangular candy wrapper",
    "small red rectangular candy wrapper",
    "red bar",
    "small red bar",
    "red candy bar",
    "red candy wrapper",
    "candy bar",
    "chocolate bar",
    "bar",
    "chocolate",
    "wrapped bar",
    "red chocolate wrapper",
    "red wafer bar",
    "red wafer bar wrapper",
    "red snack wrapper",
    "red package",
    "red packet",
    "small red package",
    "small red packet",
)
DEFAULT_OBJECT_PROMPTS_CSV = ",".join(DEFAULT_OBJECT_PROMPTS)
DEFAULT_CAPTURE_PROMPTS_CSV = ",".join(
    DEFAULT_OBJECT_PROMPTS
    + (
        "white trash can",
        "trash can",
        "bin",
        "white bowl",
        "bowl",
        "blue gripper fingers",
    )
)


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            continue
    return values or list(default)


def _env_xy(name: str, default: tuple[float, float]) -> list[float]:
    values = _env_float_list(name, [default[0], default[1]])
    if len(values) < 2:
        return [float(default[0]), float(default[1])]
    return [float(values[0]), float(values[1])]


def _env_xyz(name: str, default: tuple[float, float, float]) -> list[float]:
    values = _env_float_list(name, [default[0], default[1], default[2]])
    if len(values) < 3:
        return [float(default[0]), float(default[1]), float(default[2])]
    return [float(values[0]), float(values[1]), float(values[2])]


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        rows = list(values)
        if limit is not None:
            rows = rows[:limit]
        return [round(float(value), digits) for value in rows]
    except Exception:
        return []


def _xyz(value: Any) -> list[float] | None:
    rows = _round_list(value, limit=3)
    return rows if len(rows) == 3 else None


def _rpy_matrix_deg(rpy_deg: list[float]) -> list[list[float]]:
    roll, pitch, yaw = [math.radians(float(v)) for v in rpy_deg[:3]]
    yaw = -yaw
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(float(mat[i][j]) * float(vec[j]) for j in range(3)) for i in range(3)]


def _right_tcp_to_gripper_center_local() -> list[float]:
    return _env_xyz(
        "OPENFORGE_CHOCOLATE_HANDOVER_TRASH_RIGHT_TCP_TO_GRIPPER_CENTER_LOCAL_M",
        (0.0, 0.0, 0.045),
    )


def _tcp_pos_for_right_gripper_center(center_pos: list[float], rpy_deg: list[float]) -> list[float]:
    offset_world = _mat_vec(_rpy_matrix_deg(rpy_deg), _right_tcp_to_gripper_center_local())
    return [float(center_pos[i]) - float(offset_world[i]) for i in range(3)]


def _right_gripper_center_for_tcp_pos(tcp_pos: list[float], rpy_deg: list[float]) -> list[float]:
    offset_world = _mat_vec(_rpy_matrix_deg(rpy_deg), _right_tcp_to_gripper_center_local())
    return [float(tcp_pos[i]) + float(offset_world[i]) for i in range(3)]


def _pose_packet(pose: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pose:
        return None
    return {
        "position": _round_list(pose.get("position", []), limit=3),
        "rpy": _round_list(pose.get("rpy", []), digits=4, limit=3),
    }


def _unit_xy(value: Any, default: tuple[float, float]) -> list[float]:
    try:
        x = float(value[0])
        y = float(value[1])
        norm = math.hypot(x, y)
        if norm > 1e-6:
            return [x / norm, y / norm]
    except Exception:
        pass
    return [float(default[0]), float(default[1])]


def _axis_xy() -> list[float]:
    values = _env_xy("OPENFORGE_CHOCOLATE_HANDOVER_BAR_AXIS_XY", (0.0, 1.0))
    return _unit_xy(values, (0.0, 1.0))


def _angle_norm_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _clamp_delta(target: float, start: float, max_delta: float) -> float:
    delta = float(target) - float(start)
    limit = abs(float(max_delta))
    if abs(delta) <= limit:
        return float(target)
    return float(start) + math.copysign(limit, delta)


def _all_top_detections(scene: dict[str, Any]) -> list[dict[str, Any]]:
    top = (scene.get("cameras") or {}).get("top") or {}
    rows = top.get("serialized_detections") or []
    return [row for row in rows if isinstance(row, dict)]


def _best_detection(scene: dict[str, Any], prompts: tuple[str, ...]) -> dict[str, Any] | None:
    prompt_set = {prompt.lower() for prompt in prompts}
    candidates = [
        det for det in _all_top_detections(scene)
        if str(det.get("prompt") or det.get("label") or "").lower() in prompt_set
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda det: float(det.get("score") or 0.0), reverse=True)[0]


def _best_chocolate(scene: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    prompts = _csv(
        "OPENFORGE_CHOCOLATE_HANDOVER_OBJECT_PROMPTS",
        DEFAULT_OBJECT_PROMPTS_CSV,
    )
    rows: list[dict[str, Any]] = []
    for det in _all_top_detections(scene):
        prompt = str(det.get("prompt") or det.get("label") or "")
        if prompt.lower() not in {p.lower() for p in prompts}:
            continue
        xyz = _xyz(det.get("position_3d") or det.get("position"))
        reasons: list[str] = []
        if xyz is None:
            reasons.append("missing_position")
        rows.append(
            {
                "prompt": prompt,
                "score": round(float(det.get("score") or 0.0), 5),
                "xyz": xyz,
                "valid": not reasons,
                "reasons": reasons,
                "detection": det,
            }
        )
    valid = [row for row in rows if row["valid"]]
    selected = None
    if valid:
        selected = sorted(valid, key=lambda row: row["score"], reverse=True)[0]["detection"]
    return selected, {
        "schema": "openforge.chocolate_handover.chocolate_detection_check.v1",
        "object_prompts": prompts,
        "selection_policy": "highest-score object prompt with a 3D position; no workspace-bound veto",
        "rows": [{key: value for key, value in row.items() if key != "detection"} for row in rows],
        "selected": selected,
        "success": selected is not None,
    }


def _dummy_chocolate_detection_if_allowed() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return a saved-run dummy wrapper detection for plan checks or explicit dry motion."""
    if not _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_USE_DUMMY_OBJECT", False):
        return None, None
    physical_motion = _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION", False)
    plan_only = _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_PLAN_ONLY", False)
    dry_no_object = _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_DRY_MOTION_NO_OBJECT", False)
    dry_confirmed = _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_NO_OBJECT_DRY_RUN_CONFIRMED", False)
    if physical_motion:
        if not dry_no_object or not dry_confirmed:
            raise RuntimeError(
                "dummy chocolate detection with physical motion requires "
                "OPENFORGE_CHOCOLATE_HANDOVER_DRY_MOTION_NO_OBJECT=1 and "
                "OPENFORGE_CHOCOLATE_HANDOVER_NO_OBJECT_DRY_RUN_CONFIRMED=1"
            )
    elif not plan_only:
        raise RuntimeError(
            "dummy chocolate detection without physical motion requires "
            "OPENFORGE_CHOCOLATE_HANDOVER_PLAN_ONLY=1"
        )
    xyz = _env_xyz("OPENFORGE_CHOCOLATE_HANDOVER_DUMMY_OBJECT_XYZ_M", (0.67258, 0.25085, 0.72665))
    prompt = os.environ.get("OPENFORGE_CHOCOLATE_HANDOVER_DUMMY_OBJECT_PROMPT", "kitkat bar")
    source = "dummy_previous_run_for_dry_motion_no_object" if physical_motion else "dummy_previous_run_for_plan_only"
    policy = (
        "dummy previous-run wrapper pose for explicit no-object dry physical motion; not task success"
        if physical_motion
        else "dummy previous-run wrapper pose for plan-only cuRobo validation; physical motion forbidden"
    )
    det = {
        "prompt": prompt,
        "label": prompt,
        "score": 1.0,
        "position_3d": [float(v) for v in xyz],
        "position": [float(v) for v in xyz],
        "source": source,
    }
    packet = {
        "schema": "openforge.chocolate_handover.chocolate_detection_check.v1",
        "object_prompts": _csv(
            "OPENFORGE_CHOCOLATE_HANDOVER_OBJECT_PROMPTS",
            DEFAULT_OBJECT_PROMPTS_CSV,
        ),
        "selection_policy": policy,
        "dry_motion_no_object": bool(physical_motion),
        "rows": [{"prompt": prompt, "score": 1.0, "xyz": _round_list(xyz, limit=3), "valid": True, "reasons": []}],
        "selected": det,
        "success": True,
    }
    return det, packet


def _capture(stage: str, run_dir: Path) -> dict[str, Any]:
    return capture_scene(
        prompts=_csv("OPENFORGE_CHOCOLATE_HANDOVER_PROMPTS", DEFAULT_CAPTURE_PROMPTS_CSV),
        cameras=_csv("OPENFORGE_CHOCOLATE_HANDOVER_CAMERAS", "top,left,right,bottom"),
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage=stage,
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_CHOCOLATE_HANDOVER_OBS_TIMEOUT_S", 30.0),
        max_retries=1,
        image_only_cameras=_csv("OPENFORGE_CHOCOLATE_HANDOVER_IMAGE_ONLY_CAMERAS", "left,right,bottom"),
    )


def _top_rgb_path(scene: dict[str, Any]) -> Path | None:
    raw = (((scene.get("cameras") or {}).get("top") or {}).get("rgb_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.exists() else None


def _sam3_url() -> str:
    if os.environ.get("SAM3_SERVER_URL"):
        return os.environ["SAM3_SERVER_URL"].rstrip("/")
    host = os.environ.get("SAM3_SERVER_HOST", "localhost")
    port = os.environ.get("SAM3_SERVER_PORT", "9500")
    return f"http://{host}:{port}".rstrip("/")


def _sam3_mask_from_top_image(rgb_path: Path, prompt: str) -> Any:
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    buf = io.BytesIO()
    np.save(buf, rgb)
    payload = json.dumps({"text": prompt, "image_b64": base64.b64encode(buf.getvalue()).decode()}).encode()
    req = urllib.request.Request(
        f"{_sam3_url()}/segment",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=_env_float("OPENFORGE_CHOCOLATE_HANDOVER_SAM3_TIMEOUT_S", 30.0))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except Exception:
            detail = body
        raise RuntimeError(f"SAM3 /segment failed ({exc.code}): {detail}") from None
    data = json.loads(response.read())
    mask_bytes = base64.b64decode(data["mask_b64"])
    return np.load(io.BytesIO(mask_bytes)).astype(bool)


def _mask_axis_pixels(mask: Any) -> dict[str, Any]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size < _env_int("OPENFORGE_CHOCOLATE_HANDOVER_SAM3_MIN_MASK_PIXELS", 30):
        raise RuntimeError(f"SAM3 mask too small: {xs.size} px")
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    # Choose the image-left endpoint as the held endpoint. If the bar is nearly
    # vertical in the image, this still picks the endpoint with the smaller x
    # projection rather than relying on a fixed world-axis assumption.
    if axis[0] > 0.0:
        axis = -axis
    projections = centered @ axis
    span_px = float(np.percentile(projections, 95) - np.percentile(projections, 5))
    return {
        "source": "sam3_mask_pca",
        "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
        "axis_px_to_image_left_endpoint": [round(float(axis[0]), 6), round(float(axis[1]), 6)],
        "span_px_p05_p95": round(span_px, 3),
        "mask_pixels": int(xs.size),
    }


def _bbox_axis_pixels(det: dict[str, Any]) -> dict[str, Any]:
    box = det.get("box_2d") or []
    if len(box) < 4:
        raise RuntimeError("selected chocolate detection has no 2D box")
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    center = [(x0 + x1) * 0.5, (y0 + y1) * 0.5]
    if width >= height:
        axis = [-1.0, 0.0]
        span = width
    else:
        axis = [0.0, -1.0]
        span = height
    return {
        "source": "sam3_detection_box_long_axis",
        "center_px": [round(center[0], 3), round(center[1], 3)],
        "axis_px_to_image_left_endpoint": axis,
        "span_px_p05_p95": round(float(span), 3),
        "box_2d": _round_list(box, digits=3, limit=4),
    }


def _unique_pixel_world_pairs(scene: dict[str, Any]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for det in _all_top_detections(scene):
        box = det.get("box_2d") or []
        xyz = _xyz(det.get("position_3d") or det.get("position"))
        if len(box) < 4 or xyz is None:
            continue
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        u = (x0 + x1) * 0.5
        v = (y0 + y1) * 0.5
        key = (round(u), round(v), round(xyz[0] * 1000), round(xyz[1] * 1000))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "prompt": det.get("prompt") or det.get("label"),
                "pixel_uv": [u, v],
                "world_xy": [float(xyz[0]), float(xyz[1])],
            }
        )
    return pairs


def _fit_pixel_to_world_affine(scene: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    pairs = _unique_pixel_world_pairs(scene)
    if len(pairs) < 3:
        raise RuntimeError(f"need at least 3 unique top detections for pixel/world affine, got {len(pairs)}")
    uv1 = np.asarray([[row["pixel_uv"][0], row["pixel_uv"][1], 1.0] for row in pairs], dtype=np.float64)
    xy = np.asarray([row["world_xy"] for row in pairs], dtype=np.float64)
    coeff, residuals, rank, _ = np.linalg.lstsq(uv1, xy, rcond=None)
    if int(rank) < 3:
        raise RuntimeError(f"pixel/world affine rank deficient: rank={rank}")
    pred = uv1 @ coeff
    errors = np.linalg.norm(pred - xy, axis=1)
    return {
        "coeff_3x2": coeff.tolist(),
        "rank": int(rank),
        "fit_rmse_m": round(float(np.sqrt(np.mean(errors**2))), 5),
        "fit_max_error_m": round(float(np.max(errors)), 5),
        "pairs": [
            {
                "prompt": row["prompt"],
                "pixel_uv": _round_list(row["pixel_uv"], digits=3, limit=2),
                "world_xy": _round_list(row["world_xy"], digits=5, limit=2),
                "fit_error_m": round(float(errors[idx]), 5),
            }
            for idx, row in enumerate(pairs)
        ],
    }


def _axis_world_from_affine(axis_px: list[float], affine: dict[str, Any]) -> list[float]:
    import numpy as np

    coeff = np.asarray(affine["coeff_3x2"], dtype=np.float64)
    direction = np.asarray([float(axis_px[0]), float(axis_px[1]), 0.0], dtype=np.float64) @ coeff
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise RuntimeError("pixel axis mapped to near-zero world direction")
    return [float(direction[0] / norm), float(direction[1] / norm)]


def _estimate_bar_model(scene: dict[str, Any], chocolate: dict[str, Any], center: list[float]) -> dict[str, Any]:
    fallback_axis = _axis_xy()
    length_default = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_BAR_LENGTH_M", 0.140)
    model: dict[str, Any] = {
        "center_xyz": _round_list(center, limit=3),
        "axis_xy_positive_to_left_hold": _round_list(fallback_axis, digits=5, limit=2),
        "axis_source": "fallback_env_axis",
        "length_m": round(float(length_default), 5),
        "fallback_axis_xy": _round_list(fallback_axis, digits=5, limit=2),
        "sam3_axis_estimate": None,
        "affine_estimate": None,
        "errors": [],
    }
    if not _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_USE_SAM3_BAR_AXIS", False):
        return model

    prompt = str(chocolate.get("prompt") or chocolate.get("label") or "red wrapper")
    axis_packet: dict[str, Any] | None = None
    try:
        rgb_path = _top_rgb_path(scene)
        if rgb_path is None:
            raise RuntimeError("top RGB path unavailable")
        mask = _sam3_mask_from_top_image(rgb_path, prompt)
        axis_packet = _mask_axis_pixels(mask)
    except Exception as exc:
        model["errors"].append(f"sam3_mask_axis_failed: {type(exc).__name__}: {exc}")
        try:
            axis_packet = _bbox_axis_pixels(chocolate)
        except Exception as bbox_exc:
            model["errors"].append(f"bbox_axis_failed: {type(bbox_exc).__name__}: {bbox_exc}")
    if axis_packet is None:
        return model
    model["sam3_axis_estimate"] = axis_packet

    try:
        affine = _fit_pixel_to_world_affine(scene)
        world_axis = _axis_world_from_affine(axis_packet["axis_px_to_image_left_endpoint"], affine)
        model["affine_estimate"] = affine
        model["axis_xy_positive_to_left_hold"] = _round_list(world_axis, digits=5, limit=2)
        model["axis_source"] = f"{axis_packet['source']}_through_top_affine"
        span_px = float(axis_packet.get("span_px_p05_p95") or 0.0)
        if span_px > 1e-6:
            coeff = affine["coeff_3x2"]
            meters_per_px = math.hypot(
                axis_packet["axis_px_to_image_left_endpoint"][0] * coeff[0][0]
                + axis_packet["axis_px_to_image_left_endpoint"][1] * coeff[1][0],
                axis_packet["axis_px_to_image_left_endpoint"][0] * coeff[0][1]
                + axis_packet["axis_px_to_image_left_endpoint"][1] * coeff[1][1],
            )
            measured_length = span_px * meters_per_px
            min_len = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_BAR_MIN_LENGTH_M", 0.070)
            max_len = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_BAR_MAX_LENGTH_M", 0.180)
            if min_len <= measured_length <= max_len:
                model["length_m"] = round(float(measured_length), 5)
                model["length_source"] = "sam3_mask_span_affine"
            else:
                model["length_source"] = "env_default_clamped_by_plausibility"
                model["measured_length_rejected_m"] = round(float(measured_length), 5)
    except Exception as exc:
        model["errors"].append(f"world_axis_affine_failed: {type(exc).__name__}: {exc}")
    return model


def _read_eef_poses() -> dict[str, Any]:
    packet: dict[str, Any] = {"schema": "openforge.chocolate_handover.eef_pose_snapshot.v2", "success": False}
    try:
        state = _required_tool("get_robot_state")()
        arms = _field(state, "arms", {}) or {}
        for side in ("left", "right"):
            arm = arms.get(side) if isinstance(arms, dict) else _field(arms, side)
            grip = _field(arm, "gripper_pos")
            packet[side] = {
                "ee_pos": _round_list(_field(arm, "ee_pos", []), limit=3),
                "ee_quat_xyzw": _round_list(_field(arm, "ee_quat", []), digits=6, limit=4),
                "joint_pos": _round_list(_field(arm, "joint_pos", []), digits=6, limit=6),
                "gripper_pos": None if grip is None else round(float(grip), 6),
            }
        packet["success"] = True
    except Exception as exc:
        packet["error"] = f"{type(exc).__name__}: {exc}"
    return packet


def _eef_snapshot(stage: str, run_dir: Path) -> dict[str, Any]:
    packet = _read_eef_poses()
    packet["stage"] = stage
    packet["path"] = write_json(run_dir / "plans" / f"{stage}_eef_poses.json", packet)
    return packet


def _move_kwargs(side: str, pose: dict[str, Any], *, gripper: float | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        f"{side}_target_pos": [float(v) for v in pose["position"][:3]],
        f"{side}_target_rpy": [float(v) for v in pose["rpy"][:3]],
        "preview_only": True,
        "planner_backend": os.environ.get("OPENFORGE_CHOCOLATE_HANDOVER_PLANNER_BACKEND", "rrtconnect"),
        "planning_speed": _env_float("OPENFORGE_CHOCOLATE_HANDOVER_PLANNING_SPEED", scaled_speed(0.20)),
        "ik_error_threshold": _env_float("OPENFORGE_CHOCOLATE_HANDOVER_IK_ERROR_THRESHOLD_M", 0.025),
        "ik_rot_threshold_deg": _env_float("OPENFORGE_CHOCOLATE_HANDOVER_IK_ROT_THRESHOLD_DEG", 16.0),
    }
    if gripper is not None:
        kwargs[f"{side}_gripper"] = float(gripper)
    return kwargs


def _preview_summary(raw: Any) -> dict[str, Any]:
    safe = json_safe(raw)
    if isinstance(safe, dict) and isinstance(safe.get("data"), dict):
        safe = safe["data"]
    status = str(_field(safe, "status", "") or "")
    pos_error = _field(safe, "final_pos_error_m", _field(safe, "ik_error_m", None))
    rot_error = _field(safe, "final_rot_error_deg", _field(safe, "ik_rot_error_deg", None))
    status_ok = status.lower() in {"success", "succeeded", "ok"}
    return {
        "status": status or None,
        "reason": _field(safe, "reason", _field(safe, "error", None)),
        "executed": bool(_field(safe, "executed", False)),
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "trajectory_cache_key": _field(safe, "trajectory_cache_key", None),
        "success": bool(status_ok and not bool(_field(safe, "executed", False))),
        "acceptance_policy": "planner status success only; no extra script-level position/rotation error gate",
        "raw": safe,
    }


def _preview_pose(label: str, side: str, pose: dict[str, Any], *, gripper: float | None = None) -> dict[str, Any]:
    result = call_with_timeout(
        f"preview:{label}",
        _required_tool("freespace_move"),
        _env_float("OPENFORGE_CHOCOLATE_HANDOVER_PREVIEW_TIMEOUT_S", 25.0),
        run_in_background=_tool("run_in_background"),
        **_move_kwargs(side, pose, gripper=gripper),
    )
    if not result.get("ok"):
        return {"label": label, "side": side, "success": False, "reason": result.get("error")}
    packet = _preview_summary(result.get("data"))
    packet["label"] = label
    packet["side"] = side
    packet["target_pose"] = _pose_packet(pose)
    return packet


def _execute_pose(side: str, pose: dict[str, Any], label: str, run_dir: Path, *, gripper: float | None = None) -> dict[str, Any]:
    preview = _preview_pose(label, side, pose, gripper=gripper)
    if not preview.get("success") or not preview.get("trajectory_cache_key"):
        packet = {"stage": label, "side": side, "target_pose": _pose_packet(pose), "preview": preview, "execute": None}
        packet["why_stopped"] = f"{label} preview failed before execution"
        packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
        raise RuntimeError(f"{label} preview failed before execution: {preview.get('reason') or preview}")
    executed = _required_tool("freespace_move")(trajectory_cache_key=preview["trajectory_cache_key"])
    packet = {
        "stage": label,
        "side": side,
        "target_pose": _pose_packet(pose),
        "preview": preview,
        "execute": json_safe(executed),
        "post_exec_eef": _eef_snapshot(f"{label}_post_exec", run_dir),
    }
    packet["path"] = write_json(run_dir / "plans" / f"{label}.json", packet)
    return packet


def _set_gripper_checked(side: str, target: float, stage: str, run_dir: Path) -> dict[str, Any]:
    before = read_gripper_width(_required_tool("get_robot_state"), side)
    command = _required_tool("set_gripper")(side, float(target))
    after = read_gripper_width(_required_tool("get_robot_state"), side)
    packet = {
        "schema": "openforge.chocolate_handover.set_gripper.v2",
        "stage": stage,
        "side": side,
        "target": float(target),
        "before": before,
        "after": after,
        "command": json_safe(command),
    }
    packet["path"] = write_json(run_dir / "plans" / f"{stage}.json", packet)
    return packet


def _direct_close(side: str, stage: str, target: float, requested_steps: list[float], run_dir: Path) -> dict[str, Any]:
    packet = _set_gripper_checked(side, target, stage, run_dir)
    packet["schema"] = "openforge.chocolate_handover.direct_close.v1"
    packet["mode"] = "single_set_gripper"
    packet["requested_steps_ignored"] = [float(v) for v in requested_steps]
    write_json(run_dir / "plans" / f"{stage}.json", packet)
    return packet


def _physical_ticket_status() -> dict[str, Any]:
    required = {
        "OPENFORGE_CHOCOLATE_HANDOVER_PHYSICAL_TICKET": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_OPERATOR_PRESENT": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_ESTOP_REACHABLE": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_WORKSPACE_CLEAR": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_HANDS_TOOLS_CABLES_CLEAR": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_FOLLOWER_SERVERS_UP": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_NO_OTHER_MOTION_PRECHECK": "1",
        "OPENFORGE_CHOCOLATE_HANDOVER_RECORDING_DEBUG_CONFIRMED": "1",
    }
    dry_no_object = _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_DRY_MOTION_NO_OBJECT", False)
    if dry_no_object:
        required["OPENFORGE_CHOCOLATE_HANDOVER_NO_OBJECT_DRY_RUN_CONFIRMED"] = "1"
    else:
        required["OPENFORGE_CHOCOLATE_HANDOVER_OBJECT_VISIBLE"] = "1"
        required["OPENFORGE_CHOCOLATE_HANDOVER_NO_MANUAL_XYZ"] = "1"
    missing = [name for name, expected in required.items() if os.environ.get(name) != expected]
    return {"passed": not missing, "missing_or_invalid": missing, "required": required}


def _build_plan(scene: dict[str, Any], chocolate: dict[str, Any], trash: dict[str, Any] | None) -> dict[str, Any]:
    center = _xyz(chocolate.get("position_3d") or chocolate.get("position"))
    if center is None:
        raise RuntimeError("selected chocolate detection has no 3D position")
    bar_model = _estimate_bar_model(scene, chocolate, center)
    axis = _unit_xy(bar_model.get("axis_xy_positive_to_left_hold"), (0.0, 1.0))
    length = float(bar_model.get("length_m") or _env_float("OPENFORGE_CHOCOLATE_HANDOVER_BAR_LENGTH_M", 0.140))
    left_inset = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_PICK_INSET_M", 0.010)
    left_pick_xy = [
        center[0] + axis[0] * (length * 0.5 - left_inset),
        center[1] + axis[1] * (length * 0.5 - left_inset),
    ]
    right_free_xy = [center[0] - axis[0] * (length * 0.5), center[1] - axis[1] * (length * 0.5)]
    table_z_raw = os.environ.get("OPENFORGE_CHOCOLATE_HANDOVER_TABLE_Z_M")
    tcp_above_table_raw = os.environ.get("OPENFORGE_CHOCOLATE_HANDOVER_TOP_DOWN_TCP_ABOVE_TABLE_M")
    if table_z_raw is not None or tcp_above_table_raw is not None:
        table_z = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TABLE_Z_M", 0.760)
        tcp_above_table = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TOP_DOWN_TCP_ABOVE_TABLE_M", 0.030)
        left_grasp_z = table_z + tcp_above_table
        z_model = {
            "source": "table_relative",
            "table_z_m": round(float(table_z), 5),
            "tcp_above_table_m": round(float(tcp_above_table), 5),
            "left_grasp_z_m": round(float(left_grasp_z), 5),
            "detected_object_z_m": round(float(center[2]), 5),
            "tcp_to_detected_object_z_m": round(float(left_grasp_z - center[2]), 5),
        }
    else:
        z_offset = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TOP_DOWN_Z_OFFSET_M", 0.074)
        left_grasp_z = center[2] + z_offset
        z_model = {
            "source": "object_z_plus_offset",
            "top_down_z_offset_m": round(float(z_offset), 5),
            "left_grasp_z_m": round(float(left_grasp_z), 5),
            "detected_object_z_m": round(float(center[2]), 5),
            "tcp_to_detected_object_z_m": round(float(z_offset), 5),
        }
    left_yaw = _angle_norm_deg(
        math.degrees(math.atan2(axis[1], axis[0]))
        + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_YAW_OFFSET_DEG", 0.0)
    )
    left_grasp = {
        "position": [left_pick_xy[0], left_pick_xy[1], left_grasp_z],
        "rpy": [0.0, 180.0, left_yaw],
    }
    approach_z = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_APPROACH_Z_LIFT_M", 0.080)
    lift_z = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LIFT_Z_M", 0.090)
    left_approach = {
        "position": [left_grasp["position"][0], left_grasp["position"][1], left_grasp["position"][2] + approach_z],
        "rpy": left_grasp["rpy"],
    }
    left_lift_steps = [
        {
            "label": "left_lift_direct",
            "pose": {
                "position": [
                    left_grasp["position"][0],
                    left_grasp["position"][1],
                    left_grasp["position"][2] + lift_z,
                ],
                "rpy": left_grasp["rpy"],
            },
        }
    ]
    left_handoff_pos = _env_xyz(
        "OPENFORGE_CHOCOLATE_HANDOVER_LEFT_HANDOFF_POS_M",
        (0.560, 0.060, 0.940),
    )
    left_handoff = {
        "position": left_handoff_pos,
        "rpy": left_grasp["rpy"],
    }
    left_clearance_offset_default = (
        0.0,
        _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_RETREAT_ACTUAL_Y_M", 0.190),
        _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_RETREAT_ACTUAL_Z_M", 0.030),
    )
    left_clearance_offset = _env_xyz(
        "OPENFORGE_CHOCOLATE_HANDOVER_LEFT_POST_GRIP_CLEARANCE_OFFSET_FROM_HANDOFF_M",
        left_clearance_offset_default,
    )
    left_lift_steps.append(
        {
            "label": "left_post_grip_clearance_to_retreat_lane",
            "pose": {
                "position": [
                    left_handoff["position"][0] + left_clearance_offset[0],
                    left_handoff["position"][1] + left_clearance_offset[1],
                    left_handoff["position"][2] + left_clearance_offset[2],
                ],
                "rpy": left_grasp["rpy"],
            },
            "policy": "after closing on the wrapper, clear to the same fixed lane used for the left post-handover retreat before moving into handoff",
            "offset_from_left_handoff_m": _round_list(left_clearance_offset, limit=3),
        }
    )
    tcp_to_candy_z = float(left_grasp["position"][2]) - float(center[2])
    right_receive_offset = _env_xyz("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RECEIVE_OFFSET_FROM_LEFT_M", (0.0, -0.110, -0.030))
    right_receive = {
        "position": [
            left_handoff["position"][0] + right_receive_offset[0],
            left_handoff["position"][1] + right_receive_offset[1],
            left_handoff["position"][2] + right_receive_offset[2],
        ],
        "rpy": [0.0, 90.0, -90.0],
    }
    right_pre = {
        "position": [
            right_receive["position"][0],
            right_receive["position"][1] - _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_PRE_STANDOFF_M", 0.020),
            right_receive["position"][2] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_PRE_Z_LIFT_M", 0.020),
        ],
        "rpy": right_receive["rpy"],
    }
    plan = {
        "schema": "openforge.chocolate_handover.focused_plan.v1",
        "success": True,
        "chocolate_xyz": _round_list(center, limit=3),
        "bar_axis_xy": _round_list(axis, digits=5, limit=2),
        "bar_length_m": round(float(length), 5),
        "bar_model": {
            **bar_model,
            "axis_xy_positive_to_left_hold": _round_list(axis, digits=5, limit=2),
            "length_m": round(float(length), 5),
            "left_pick_xy": _round_list(left_pick_xy, limit=2),
            "right_free_xy": _round_list(right_free_xy, limit=2),
            "left_top_down_rpy": _round_list(left_grasp["rpy"], digits=4, limit=3),
            "left_yaw_policy": "yaw follows the selected long axis so the top-down pinch closes across the short wrapper width.",
            "policy": "endpoint top-down pinch: left grasps one short end; right receives the exposed end.",
        },
        "left_grasp_z_model": z_model,
        "left_approach_pose": left_approach,
        "left_grasp_pose": left_grasp,
        "left_lift_steps": left_lift_steps,
        "left_handoff_pose": left_handoff,
        "handoff_model": {
            "left_handoff_source": "fixed_env_or_default",
            "left_handoff_pos_m": _round_list(left_handoff["position"], limit=3),
            "left_post_grip_clearance_offset_from_handoff_m": _round_list(left_clearance_offset, limit=3),
            "right_receive_offset_from_left_m": _round_list(right_receive_offset, limit=3),
            "right_receive_policy": "right receive pose is a fixed offset from the fixed left handoff pose",
        },
        "right_receive_pre_pose": right_pre,
        "right_receive_pose": right_receive,
        "tcp_to_candy_z_at_grasp_m": round(tcp_to_candy_z, 5),
        "trash_detection": trash,
    }
    return plan


def _make_trash_plan(trash: dict[str, Any], right_start_pose: dict[str, Any]) -> dict[str, Any]:
    trash_xyz = _xyz(trash.get("position_3d") or trash.get("position"))
    if trash_xyz is None:
        return {"schema": "openforge.chocolate_handover.trash_plan.v2", "success": False, "why_stopped": "trash detection has no 3D position"}
    bias = _env_xy("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_XY_BIAS_M", (0.010, -0.035))
    drop_xy = [trash_xyz[0] + bias[0], trash_xyz[1] + bias[1]]
    fixed_drop_z_raw = os.environ.get("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_FIXED_DROP_Z_M")
    if fixed_drop_z_raw is not None:
        drop_z = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_FIXED_DROP_Z_M", 1.120)
        drop_z_source = "fixed_env"
    else:
        drop_z = _clamp(
            trash_xyz[2] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_DROP_Z_CLEARANCE_M", 0.080),
            _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_MIN_DROP_Z_M", 1.04),
            _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_MAX_DROP_Z_M", 1.10),
        )
        drop_z_source = "trash_detection_z_plus_clearance"
    start = [float(v) for v in right_start_pose["position"][:3]]
    fixed_transport_z_raw = os.environ.get("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_FIXED_TRANSPORT_Z_M")
    if fixed_transport_z_raw is not None:
        transport_z = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_FIXED_TRANSPORT_Z_M", 1.160)
        transport_z_source = "fixed_env"
    else:
        transport_z = _clamp(
            start[2] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_TRANSPORT_Z_LIFT_M", 0.300),
            _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_MIN_TRANSPORT_Z_M", 1.18),
            _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_MAX_TRANSPORT_Z_M", 1.24),
        )
        transport_z_source = "start_z_plus_lift_clamped"
    rpy = _env_xyz("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_RPY_DEG", (0.0, 90.0, 0.0))
    start_rpy = _round_list(right_start_pose.get("rpy") or rpy, limit=3)
    yz_stage_rpy = _env_xyz(
        "OPENFORGE_CHOCOLATE_HANDOVER_TRASH_YZ_STAGE_RPY_DEG",
        (float(start_rpy[0]), float(start_rpy[1]), float(start_rpy[2])),
    )
    approach_offset = _env_xy("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_APPROACH_OFFSET_M", (-0.120, 0.0))
    retreat_offset = _env_xy(
        "OPENFORGE_CHOCOLATE_HANDOVER_TRASH_RETREAT_OFFSET_M",
        (approach_offset[0], approach_offset[1]),
    )
    yz_stage_tcp = [drop_xy[0] + approach_offset[0], drop_xy[1] + approach_offset[1], transport_z]
    pose_items: list[dict[str, Any]] = [
        {
            "label": "right_trash_yz_stage_before_orient",
            "pose": {"position": yz_stage_tcp, "rpy": yz_stage_rpy},
            "target_gripper_center_m": _right_gripper_center_for_tcp_pos(yz_stage_tcp, yz_stage_rpy),
            "route_step_policy": "move to the selected trash-relative staging X/Y/Z using the configured pre-+X staging orientation",
        },
        {
            "label": "right_trash_orient_plus_x_at_yz_stage",
            "pose": {"position": yz_stage_tcp, "rpy": rpy},
            "target_gripper_center_m": _right_gripper_center_for_tcp_pos(yz_stage_tcp, rpy),
            "route_step_policy": "at high bin-Y staging pose, orient gripper so the modeled gripper-center axis points world +X",
        },
        {
            "label": "right_trash_over_bin_from_x_stage",
            "pose": {
                "position": _tcp_pos_for_right_gripper_center([drop_xy[0], drop_xy[1], transport_z], rpy),
                "rpy": rpy,
            },
            "target_gripper_center_m": [drop_xy[0], drop_xy[1], transport_z],
            "route_step_policy": "move along +X into the trash bin with Y and Z held fixed",
        },
        {
            "label": "right_trash_drop",
            "pose": {
                "position": _tcp_pos_for_right_gripper_center([drop_xy[0], drop_xy[1], drop_z], rpy),
                "rpy": rpy,
            },
            "target_gripper_center_m": [drop_xy[0], drop_xy[1], drop_z],
            "route_step_policy": "lower along -Z at fixed X/Y before release",
        },
    ]
    retreat_center = [
        drop_xy[0] + retreat_offset[0],
        drop_xy[1] + retreat_offset[1],
        min(max(drop_z, transport_z) + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_RETREAT_Z_M", 0.080), 1.28),
    ]
    retreat = {
        "position": _tcp_pos_for_right_gripper_center(retreat_center, rpy),
        "rpy": rpy,
    }
    return {
        "schema": "openforge.chocolate_handover.trash_plan.v2",
        "success": True,
        "trash_xyz_m": _round_list(trash_xyz, limit=3),
        "trash_xy_bias_m": _round_list(bias, limit=2),
        "drop_xy_m": _round_list(drop_xy, limit=2),
        "drop_z_m": round(float(drop_z), 5),
        "drop_z_source": drop_z_source,
        "detected_trash_z_m": round(float(trash_xyz[2]), 5),
        "trash_transport_z_m": round(float(transport_z), 5),
        "trash_transport_z_source": transport_z_source,
        "trash_transport_above_drop_z_m": round(float(transport_z - drop_z), 5),
        "trash_rpy_deg": _round_list(rpy, limit=3),
        "trash_approach_offset_m": _round_list(approach_offset, limit=2),
        "trash_retreat_offset_m": _round_list(retreat_offset, limit=2),
        "right_trash_yz_stage_tcp_m": _round_list(yz_stage_tcp, limit=3),
        "right_trash_start_rpy_deg": _round_list(start_rpy, digits=4, limit=3),
        "right_trash_yz_stage_rpy_deg": _round_list(yz_stage_rpy, digits=4, limit=3),
        "right_tcp_to_gripper_center_local_m": _round_list(_right_tcp_to_gripper_center_local(), limit=3),
        "trash_route_policy": "Robust fixed-offset trash approach: stage at the selected trash-relative X/Y/Z offset with a non-+X staging orientation, rotate high so the gripper points world +X away from the robot, move +X into the bin, lower -Z, release, then retreat back along -X/+Z.",
        "right_trash_transport_centers": [
            {
                "label": item["label"],
                "target_tcp_m": _round_list(item["pose"]["position"], limit=3),
                "target_gripper_center_m": _round_list(item["target_gripper_center_m"], limit=3),
                "route_step_policy": item["route_step_policy"],
            }
            for item in pose_items
        ],
        "right_trash_retreat_center_m": _round_list(retreat_center, limit=3),
        "right_trash_transport_poses": [{"label": item["label"], "pose": _pose_packet(item["pose"])} for item in pose_items],
        "right_trash_retreat_pose": _pose_packet(retreat),
        "_transport_pose_objects": [{"label": item["label"], "pose": item["pose"]} for item in pose_items],
        "_retreat_pose_object": retreat,
    }


def _planned_left_retreat_pose(left_handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": [
            float(left_handoff["position"][0]),
            float(left_handoff["position"][1]) + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_RETREAT_ACTUAL_Y_M", 0.190),
            float(left_handoff["position"][2]) + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_RETREAT_ACTUAL_Z_M", 0.030),
        ],
        "rpy": list(left_handoff["rpy"]),
    }


def _planned_right_retreat_pose(right_receive: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": [
            float(right_receive["position"][0]) + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_X_M", 0.0),
            float(right_receive["position"][1]) + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_Y_M", -0.045),
            float(right_receive["position"][2]) + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_Z_M", 0.015),
        ],
        "rpy": list(right_receive["rpy"]),
    }


def _preview_full_pipeline(plan: dict[str, Any], run_dir: Path, trash: dict[str, Any] | None) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.chocolate_handover.full_pipeline_preview.v1",
        "success": False,
        "steps": [],
        "why_stopped": None,
        "policy": "no-motion cuRobo preview of the original handover/trash waypoint chain; no gripper commands or cached executions",
    }
    if trash is None:
        packet["why_stopped"] = "missing white trash can detection"
        packet["path"] = write_json(run_dir / "plans" / "full_pipeline_preview.json", packet)
        return packet

    left_open = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_OPEN_GRIPPER", 1.0)
    right_open = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_OPEN_GRIPPER", 1.0)
    left_close_target = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_CLOSE_TARGET_POS", 0.04)
    right_close_target = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_CLOSE_TARGET_POS", 0.00)

    left_handoff = plan["left_handoff_pose"]
    right_receive = plan["right_receive_pose"]
    right_pre = plan["right_receive_pre_pose"]
    left_retreat = _planned_left_retreat_pose(left_handoff)
    right_retreat = _planned_right_retreat_pose(right_receive)

    stage_items: list[tuple[str, str, dict[str, Any], float | None]] = [
        ("left_approach", "left", plan["left_approach_pose"], left_open),
        ("left_grasp", "left", plan["left_grasp_pose"], left_open),
    ]
    for item in plan["left_lift_steps"]:
        stage_items.append((str(item["label"]), "left", item["pose"], left_close_target))
    stage_items.extend(
        [
            ("left_handoff_hold", "left", left_handoff, left_close_target),
            ("right_receive_pre_try_1", "right", right_pre, right_open),
            ("right_receive_close_pose_try_1", "right", right_receive, right_open),
            ("left_retreat_after_release_actual_small_y_clear_1", "left", left_retreat, 1.0),
            ("right_retreat_after_left_release", "right", right_retreat, right_close_target),
        ]
    )

    trash_plan = _make_trash_plan(trash, {"position": right_retreat["position"], "rpy": right_receive["rpy"]})
    transport_objects = trash_plan.pop("_transport_pose_objects", [])
    right_trash_retreat = trash_plan.pop("_retreat_pose_object", None)
    packet["trash_place_plan"] = trash_plan
    write_json(run_dir / "plans" / "full_preview_trash_place_plan.json", trash_plan)
    if not trash_plan.get("success") or right_trash_retreat is None:
        packet["why_stopped"] = f"trash plan failed: {trash_plan.get('why_stopped')}"
        packet["path"] = write_json(run_dir / "plans" / "full_pipeline_preview.json", packet)
        return packet
    for item in transport_objects:
        stage_items.append((str(item["label"]), "right", item["pose"], right_close_target))
    stage_items.append(("right_trash_retreat", "right", right_trash_retreat, 1.0))

    for label, side, pose, gripper in stage_items:
        preview = _preview_pose(label, side, pose, gripper=gripper)
        step = {
            "stage": label,
            "side": side,
            "target_pose": _pose_packet(pose),
            "gripper": gripper,
            "preview": preview,
            "success": bool(preview.get("success")),
        }
        packet["steps"].append(step)
        if not step["success"]:
            packet["why_stopped"] = f"{label} preview failed: {preview.get('reason') or preview.get('status') or preview}"
            packet["path"] = write_json(run_dir / "plans" / "full_pipeline_preview.json", packet)
            return packet

    packet["success"] = True
    packet["why_stopped"] = "full no-motion cuRobo preview succeeded"
    packet["path"] = write_json(run_dir / "plans" / "full_pipeline_preview.json", packet)
    return packet


def _post_trash_verify(run_dir: Path, pre_chocolate: dict[str, Any], trash: dict[str, Any], right_release: dict[str, Any]) -> dict[str, Any]:
    scene = _capture("post_trash_verify", run_dir)
    post_chocolate, _ = _best_chocolate(scene)
    post_trash = _best_detection(scene, _csv("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_PROMPTS", "white trash can,trash can,bin"))
    pre_xyz = _xyz(pre_chocolate.get("position_3d") or pre_chocolate.get("position"))
    post_xyz = _xyz((post_chocolate or {}).get("position_3d") or (post_chocolate or {}).get("position"))
    source_clearance = None if pre_xyz is None or post_xyz is None else _dist_xy(pre_xyz, post_xyz)
    right_open = None
    try:
        right_open = float(right_release.get("after"))
    except (TypeError, ValueError):
        pass
    packet = {
        "schema": "openforge.chocolate_handover.post_trash_verify.v2",
        "success": bool((source_clearance is None or source_clearance >= 0.06) and right_open is not None and right_open >= 0.55),
        "confidence": 0.45,
        "post_scene": scene,
        "pre_chocolate_xyz": pre_xyz,
        "post_chocolate_detection": post_chocolate,
        "post_chocolate_xyz": post_xyz,
        "post_trash_detection": post_trash or trash,
        "source_clearance_xy_m": source_clearance,
        "right_release": right_release,
        "why": "top scene source-clearance/debug check only; final in-bin proof is handled by handover_chocolate_success_checker.py using right.mp4",
    }
    packet["path"] = write_json(run_dir / "plans" / "post_trash_verify.json", packet)
    return packet


def _execute_physical(plan: dict[str, Any], run_dir: Path, chocolate: dict[str, Any], trash: dict[str, Any] | None) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "openforge.chocolate_handover.physical_attempt.v2",
        "success": False,
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "steps": [],
        "why_stopped": None,
    }
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        packet["why_stopped"] = "physical gate not set; no motion executed"
        packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
        return packet
    ticket = _physical_ticket_status()
    packet["physical_ticket_status"] = ticket
    if not ticket["passed"]:
        packet["why_stopped"] = f"missing physical ticket envs: {ticket['missing_or_invalid']}"
        packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
        return packet
    if trash is None:
        packet["why_stopped"] = "missing white trash can detection"
        packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
        return packet

    left_open = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_OPEN_GRIPPER", 1.0)
    right_open = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_OPEN_GRIPPER", 1.0)
    left_close_target = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_CLOSE_TARGET_POS", 0.04)
    right_close_target = _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_CLOSE_TARGET_POS", 0.00)

    left_handoff = plan["left_handoff_pose"]
    right_receive = plan["right_receive_pose"]
    right_pre = plan["right_receive_pre_pose"]
    try:
        for side, target, stage in (("left", left_open, "open_left_before_handover"), ("right", right_open, "open_right_before_handover")):
            packet["steps"].append(_set_gripper_checked(side, target, stage, run_dir))
            packet["movement_capable_calls"].append(f"set_gripper:{side}")
        packet["physical_motion_executed"] = True

        for label, side, pose, gripper in (
            ("left_approach", "left", plan["left_approach_pose"], left_open),
            ("left_grasp", "left", plan["left_grasp_pose"], left_open),
        ):
            packet["steps"].append(_execute_pose(side, pose, label, run_dir, gripper=gripper))
            packet["movement_capable_calls"].append(f"freespace_move:{label}")

        left_close = _direct_close(
            "left",
            "left_chocolate_close",
            left_close_target,
            _env_float_list("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_CLOSE_STEPS", [left_close_target]),
            run_dir,
        )
        packet["left_close"] = left_close
        packet["steps"].append(left_close)
        packet["movement_capable_calls"].append("set_gripper:left")

        for item in plan["left_lift_steps"]:
            packet["steps"].append(_execute_pose("left", item["pose"], item["label"], run_dir, gripper=left_close_target))
            packet["movement_capable_calls"].append(f"freespace_move:{item['label']}")

        packet["steps"].append(_execute_pose("left", left_handoff, "left_handoff_hold", run_dir, gripper=left_close_target))
        packet["movement_capable_calls"].append("freespace_move:left_handoff_hold")

        left_actual = _eef_snapshot("left_handoff_actual_for_right_receive", run_dir)
        left_ee = _xyz((left_actual.get("left") or {}).get("ee_pos"))
        if left_ee is not None:
            right_receive_offset = _env_xyz("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RECEIVE_OFFSET_FROM_LEFT_M", (0.0, -0.110, -0.030))
            right_receive["position"] = [
                left_ee[0] + right_receive_offset[0],
                left_ee[1] + right_receive_offset[1],
                left_ee[2] + right_receive_offset[2],
            ]
            right_pre["position"] = [
                right_receive["position"][0],
                right_receive["position"][1] - _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_PRE_STANDOFF_M", 0.020),
                right_receive["position"][2] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_PRE_Z_LIFT_M", 0.020),
            ]
        packet["dynamic_right_receive"] = {
            "left_handoff_actual": left_actual,
            "right_receive_offset_from_left_m": _round_list(
                _env_xyz("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RECEIVE_OFFSET_FROM_LEFT_M", (0.0, -0.110, -0.030)),
                limit=3,
            ),
            "right_receive_pre_pose": _pose_packet(right_pre),
            "right_receive_pose": _pose_packet(right_receive),
        }
        write_json(run_dir / "plans" / "dynamic_right_receive.json", packet["dynamic_right_receive"])

        packet["steps"].append(_execute_pose("right", right_pre, "right_receive_pre_try_1", run_dir, gripper=right_open))
        packet["movement_capable_calls"].append("freespace_move:right_receive_pre_try_1")
        packet["steps"].append(_execute_pose("right", right_receive, "right_receive_close_pose_try_1", run_dir, gripper=right_open))
        packet["movement_capable_calls"].append("freespace_move:right_receive_close_pose_try_1")
        right_close = _direct_close(
            "right",
            "right_chocolate_receive_close",
            right_close_target,
            _env_float_list("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_CLOSE_STEPS", [right_close_target]),
            run_dir,
        )
        packet["right_close"] = right_close
        packet["steps"].append(right_close)
        packet["movement_capable_calls"].append("set_gripper:right")
        write_json(
            run_dir / "plans" / "right_receive_physical_attempt_1.json",
            {
                "schema": "openforge.chocolate_handover.right_receive_physical_attempt.v2",
                "index": 1,
                "success": True,
                "right_receive_pre_pose": _pose_packet(right_pre),
                "right_receive_pose": _pose_packet(right_receive),
                "right_close": right_close,
                "policy": "Focused script proceeds after the right close command; final success is checked by right.mp4.",
            },
        )

        packet["steps"].append(_set_gripper_checked("left", 1.0, "left_release_after_right_hold", run_dir))
        packet["movement_capable_calls"].append("set_gripper:left")

        after_release = _eef_snapshot("post_left_release_actual", run_dir)
        left_pos = _xyz((after_release.get("left") or {}).get("ee_pos"))
        if left_pos is not None:
            left_retreat = {
                "position": [
                    left_pos[0],
                    left_pos[1] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_RETREAT_ACTUAL_Y_M", 0.190),
                    left_pos[2] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_LEFT_RETREAT_ACTUAL_Z_M", 0.030),
                ],
                "rpy": left_handoff["rpy"],
            }
        else:
            left_retreat = {
                "position": [left_handoff["position"][0], left_handoff["position"][1] + 0.190, left_handoff["position"][2] + 0.030],
                "rpy": left_handoff["rpy"],
            }
        packet["steps"].append(_execute_pose("left", left_retreat, "left_retreat_after_release_actual_small_y_clear_1", run_dir, gripper=1.0))
        packet["movement_capable_calls"].append("freespace_move:left_retreat_after_release_actual_small_y_clear_1")

        after_left_retreat = _eef_snapshot("post_left_retreat_actual", run_dir)
        right_pos = _xyz((after_left_retreat.get("right") or {}).get("ee_pos"))
        if right_pos is None:
            right_pos = right_receive["position"]
        right_retreat = {
            "position": [
                right_pos[0] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_X_M", 0.0),
                right_pos[1] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_Y_M", -0.045),
                right_pos[2] + _env_float("OPENFORGE_CHOCOLATE_HANDOVER_RIGHT_RETREAT_AFTER_LEFT_Z_M", 0.015),
            ],
            "rpy": right_receive["rpy"],
        }
        packet["steps"].append(_execute_pose("right", right_retreat, "right_retreat_after_left_release", run_dir, gripper=right_close_target))
        packet["movement_capable_calls"].append("freespace_move:right_retreat_after_left_release")

        trash_start = _eef_snapshot("right_current_after_left_retreat_for_trash_plan", run_dir)
        start_pos = _xyz((trash_start.get("right") or {}).get("ee_pos")) or right_retreat["position"]
        trash_plan = _make_trash_plan(trash, {"position": start_pos, "rpy": right_receive["rpy"]})
        transport_objects = trash_plan.pop("_transport_pose_objects", [])
        right_trash_retreat = trash_plan.pop("_retreat_pose_object", None)
        packet["trash_place_plan"] = trash_plan
        write_json(run_dir / "plans" / "trash_place_plan.json", trash_plan)
        if not trash_plan.get("success") or right_trash_retreat is None:
            raise RuntimeError(f"trash plan failed: {trash_plan.get('why_stopped')}")
        for item in transport_objects:
            label = str(item["label"])
            packet["steps"].append(_execute_pose("right", item["pose"], label, run_dir, gripper=right_close_target))
            packet["movement_capable_calls"].append(f"freespace_move:{label}")

        right_release = _set_gripper_checked("right", 1.0, "right_release_into_trash", run_dir)
        packet["steps"].append(right_release)
        packet["movement_capable_calls"].append("set_gripper:right")
        packet["steps"].append(_execute_pose("right", right_trash_retreat, "right_trash_retreat", run_dir, gripper=1.0))
        packet["movement_capable_calls"].append("freespace_move:right_trash_retreat")

        try:
            post_trash = _post_trash_verify(run_dir, chocolate, trash, right_release)
        except Exception as verify_exc:
            post_trash = {
                "schema": "openforge.chocolate_handover.post_trash_verify.v2",
                "success": None,
                "why": f"debug verification skipped/failed without gating sequence success: {type(verify_exc).__name__}: {verify_exc}",
            }
        packet["post_trash_verify"] = post_trash
        packet["success"] = True
        packet["confidence"] = 0.35
        packet["why_stopped"] = "scripted handover/trash motions executed; final completion still requires the frozen right-camera checker"
        packet["final_eef_poses"] = _eef_snapshot("final_after_trash_place", run_dir)
    except Exception as exc:
        packet["why_stopped"] = f"{type(exc).__name__}: {exc}"
        packet["final_eef_poses"] = _eef_snapshot("final_after_exception", run_dir)
    packet["path"] = write_json(run_dir / "plans" / "physical_attempt.json", packet)
    return packet


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result: dict[str, Any] = {
        "schema": "openforge.chocolate_handover.left_to_right.focused.v1",
        "success": False,
        "plan_success": False,
        "confidence": 0.0,
        "physical_motion_executed": False,
        "method": TASK_NAME,
        "run_dir": str(run_dir),
        "scene_observation": None,
        "selected_chocolate_detection": None,
        "chocolate_detection_check": None,
        "selected_trash_detection": None,
        "bar_model": None,
        "selected_pickup_candidate": None,
        "handover_plan": None,
        "full_pipeline_preview": None,
        "physical_attempt": None,
        "why_stopped": None,
        "next_recommendation": None,
    }
    try:
        result["source_snapshot"] = {
            "script": "cap/saved_scripts/handover_chocolate_bar_left_to_right_one_shot_loop.py",
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("OPENFORGE_CHOCOLATE_HANDOVER_")},
        }
        scene = _capture("pre_chocolate_handover", run_dir)
        result["scene_observation"] = scene
        chocolate, detection_check = _best_chocolate(scene)
        dummy_chocolate, dummy_check = _dummy_chocolate_detection_if_allowed()
        if dummy_chocolate is not None and dummy_check is not None:
            dummy_check["live_detection_ignored"] = chocolate
            chocolate = dummy_chocolate
            detection_check = dummy_check
        trash = _best_detection(scene, _csv("OPENFORGE_CHOCOLATE_HANDOVER_TRASH_PROMPTS", "white trash can,trash can,bin"))
        result["selected_chocolate_detection"] = chocolate
        result["chocolate_detection_check"] = detection_check
        result["selected_trash_detection"] = trash
        write_json(run_dir / "plans" / "chocolate_detection_check.json", detection_check)
        if chocolate is None:
            result["why_stopped"] = "missing valid live KitKat/red-wrapper detection"
            result["next_recommendation"] = "Inspect plans/chocolate_detection_check.json and the top overlay."
            return result
        if trash is None:
            result["why_stopped"] = "missing live white trash can/bin detection"
            result["next_recommendation"] = "Ensure the bin is visible to top camera and rerun."
            return result
        plan = _build_plan(scene, chocolate, trash)
        result["handover_plan"] = plan
        result["bar_model"] = plan["bar_model"]
        result["selected_pickup_candidate"] = {
            "index": 1,
            "source": "focused_top_down_endpoint",
            "position": plan["left_grasp_pose"]["position"],
            "rpy": plan["left_grasp_pose"]["rpy"],
            "bar_geometry": plan["bar_model"],
        }
        write_json(run_dir / "plans" / "bar_model.json", plan["bar_model"])
        write_json(run_dir / "plans" / "selected_handover_plan.json", plan)
        result["plan_success"] = True
        if _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_PLAN_ONLY", False):
            preview = _preview_full_pipeline(plan, run_dir, trash)
            result["full_pipeline_preview"] = preview
            result["plan_success"] = bool(preview.get("success"))
            result["success"] = False
            result["confidence"] = 0.0
            result["physical_motion_executed"] = False
            result["why_stopped"] = preview.get("why_stopped")
            result["next_recommendation"] = "Plan-only full pipeline preview completed; no robot motion was executed."
            return result
        physical = _execute_physical(plan, run_dir, chocolate, trash)
        result["physical_attempt"] = physical
        result["physical_motion_executed"] = bool(physical.get("physical_motion_executed"))
        if _truthy_env("OPENFORGE_CHOCOLATE_HANDOVER_DRY_MOTION_NO_OBJECT", False):
            result["success"] = False
            result["confidence"] = 0.0
            result["why_stopped"] = "dry no-object motion chain executed; no real wrapper was present, so this is not task success"
            result["next_recommendation"] = "Use this as motion-path evidence only; rerun normal handover with a real wrapper before using the frozen checker."
        else:
            result["success"] = bool(physical.get("success"))
            result["confidence"] = float(physical.get("confidence") or (0.2 if result["physical_motion_executed"] else 0.0))
            result["why_stopped"] = physical.get("why_stopped")
            result["next_recommendation"] = "Run handover_chocolate_success_checker.py on this log; if it fails, inspect the first failed motion artifact."
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Inspect result.json, stage_summary.md, and service logs before retrying."
        return result
    finally:
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## Focused KitKat Left-To-Right Handover",
                f"- success: {result.get('success')}",
                f"- plan_success: {result.get('plan_success')}",
                f"- physical_motion_executed: {result.get('physical_motion_executed')}",
                f"- chocolate_detection: {result.get('selected_chocolate_detection')}",
                f"- trash_detection: {result.get('selected_trash_detection')}",
                f"- bar_model: {result.get('bar_model')}",
                f"- full_pipeline_preview_success: {((result.get('full_pipeline_preview') or {}).get('success'))}",
                f"- physical_attempt_success: {((result.get('physical_attempt') or {}).get('success'))}",
                f"- why_stopped: {result.get('why_stopped')}",
                f"- next_recommendation: {result.get('next_recommendation')}",
            ],
        )
        print(json.dumps(json_safe(result), indent=2))


TASK_RESULT.update(_run())
