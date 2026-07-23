# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded no-motion planner for picking up a plate on YAM.

This script is intentionally observe/plan only.  It uses fresh live plate
detections, generates flat-object rim grasp candidates, previews them with the
planner, and writes compact artifacts.  It refuses to run if the physical gate
is set.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cap.saved_scripts.yam_runtime import (
    append_stage_summary,
    capture_scene,
    current_run_dir,
    json_safe,
    rank_motion_candidates,
    write_json,
)


TASK_NAME = os.environ.get("OPENFORGE_PLATE_TASK_NAME", "pick_plate_no_motion_plan")
OBJECT_NAME = os.environ.get("OPENFORGE_PLATE_OBJECT_NAME", "plate").strip() or "plate"
TASK_RESULT: dict[str, Any] = {}
BOWL_TOP_DOWN_RIM_Z_BIAS_DEFAULT_M = 0.080
BOWL_TOP_DOWN_TCP_ABOVE_RIM_DEFAULT_M = -0.040
BOWL_TOP_DOWN_SUPPORT_Z_DEFAULT_M = 0.760
BOWL_TOP_DOWN_HEIGHT_DEFAULT_M = 0.080
TOP_DOWN_RIM_STRATEGY_MODES = {"top_down_rim_straddle", "bowl_top_down", "top_down_bowl_rim"}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def _requested_top_down_rim_strategy() -> bool:
    return os.environ.get("OPENFORGE_PLATE_STRATEGY_MODE", "").strip().lower() in TOP_DOWN_RIM_STRATEGY_MODES


def _object_is_bowl() -> bool:
    object_name = OBJECT_NAME.lower()
    if "bowl" in object_name:
        return True
    return _requested_top_down_rim_strategy() and any(token in object_name for token in ("dish", "plate"))


def _strategy_mode() -> str:
    raw = os.environ.get("OPENFORGE_PLATE_STRATEGY_MODE", "").strip().lower()
    if raw:
        return raw
    return "top_down_bowl_rim" if _object_is_bowl() else "side_rim"


def _float_csv(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        try:
            values.append(float(part.strip()))
        except ValueError:
            pass
    return values or list(default)


def _rpy_offsets(name: str, default: list[list[float]]) -> list[list[float]]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return [list(item) for item in default]
    values: list[list[float]] = []
    for chunk in raw.replace("|", ";").split(";"):
        parts = [part.strip() for part in chunk.replace(":", ",").split(",") if part.strip()]
        if len(parts) != 3:
            continue
        try:
            values.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    return values or [list(item) for item in default]


def _tool(name: str) -> Callable[..., Any] | None:
    value = globals().get(name)
    return value if callable(value) else None


def _required_tool(name: str) -> Callable[..., Any]:
    value = _tool(name)
    if value is None:
        raise RuntimeError(f"required run_script tool is unavailable: {name}")
    return value


def _manual_pose_env_vars_present() -> list[str]:
    names = [
        "OPENFORGE_PLATE_XYZ",
        "OPENFORGE_PLATE_HALF_EXTENTS",
        "OPENFORGE_OBJECT_XYZ",
        "OPENFORGE_TARGET_XYZ",
        "OPENFORGE_GRASP_LIFT_OBJECT_XYZ",
        "OPENFORGE_GRASP_NO_MOTION_DEBUG_MANUAL_XYZ",
    ]
    return [name for name in names if os.environ.get(name)]


def _motion_cameras() -> tuple[str, ...]:
    cameras = _csv("OPENFORGE_PLATE_CAMERAS", ["top", "left", "right"])
    return tuple(camera for camera in cameras if camera in {"top", "left", "right"})


def _debug_observation_cameras() -> list[str]:
    raw = os.environ.get("OPENFORGE_PLATE_OBSERVATION_CAMERAS") or os.environ.get("OPENFORGE_DEBUG_OBS_CAMERAS")
    if raw and raw.strip():
        cameras = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    else:
        cameras = ["top", "left", "right", "bottom"]
    return cameras or ["top", "left", "right", "bottom"]


def _image_only_debug_cameras() -> tuple[str, ...]:
    return tuple(_csv("OPENFORGE_PLATE_IMAGE_ONLY_CAMERAS", ["bottom"]))


def _round_list(values: Any, digits: int = 5, limit: int | None = None) -> list[float]:
    try:
        items = list(values)
        if limit is not None:
            items = items[:limit]
        return [round(float(v), digits) for v in items]
    except Exception:
        return []


def _video_paths(run_dir: Path) -> list[str]:
    paths: list[str] = []
    for name in ("top.mp4", "left.mp4", "right.mp4", "bottom.mp4"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            paths.append(str(path))
    return paths


def _recording_evidence(run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, _env_float("OPENFORGE_PLATE_RECORDING_WAIT_S", 4.0))
    videos = _video_paths(run_dir)
    while not videos and time.monotonic() < deadline:
        time.sleep(0.2)
        videos = _video_paths(run_dir)
    evidence: dict[str, Any] = {"required": True, "passed": bool(videos), "video_paths": videos}
    if not videos:
        evidence.update(
            {
                "problem": (
                    "no mp4 files found; run live YAM probes with recording.enabled=true "
                    "and debug_ui.enabled=true"
                ),
                "ffmpeg_logs": sorted(str(path) for path in run_dir.glob("*.ffmpeg.log")),
                "blocker_before_physical": True,
            }
        )
    return evidence


def _detection_touches_image_edge(detection: dict[str, Any]) -> bool:
    box = detection.get("box_2d") or []
    if len(box) < 4:
        return False
    margin_px = _env_int("OPENFORGE_PLATE_IMAGE_EDGE_MARGIN_PX", 6)
    width_px = _env_int("OPENFORGE_PLATE_IMAGE_WIDTH_PX", 640)
    height_px = _env_int("OPENFORGE_PLATE_IMAGE_HEIGHT_PX", 480)
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return x0 <= margin_px or y0 <= margin_px or x1 >= width_px - margin_px or y1 >= height_px - margin_px


def _target_prompt_rank(detection: dict[str, Any]) -> int:
    prompt = str(detection.get("prompt") or "").strip().lower()
    label = str(detection.get("label") or "").strip().lower()
    object_name = OBJECT_NAME.lower()
    text = f"{prompt} {label}".strip()
    if object_name and object_name in text:
        return 0
    if _object_is_bowl() and prompt == "bowl":
        return 1
    if any(token in object_name for token in ("dish", "plate")) and prompt in {
        "dish",
        "plate",
        "white dish",
        "white plate",
        "dinner plate",
    }:
        return 1
    if object_name and prompt == object_name:
        return 0
    return 2


def _select_plate_detection(packet: dict[str, Any], min_score: float) -> dict[str, Any] | None:
    detections = [dict(det) for det in packet.get("all_detections") or []]
    allow_side_edge = _truthy_env("OPENFORGE_PLATE_ALLOW_SIDE_CAMERA_EDGE_DETECTIONS", False)
    target_y_sign = os.environ.get("OPENFORGE_PLATE_TARGET_Y_SIGN", "").strip().lower()
    filtered: list[dict[str, Any]] = []
    for det in detections:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        score = float(det.get("score") or 0.0)
        edge = _detection_touches_image_edge(det)
        det["touches_image_edge"] = edge
        if score < min_score or camera not in {"top", "left", "right"}:
            continue
        if edge and camera != "top" and not allow_side_edge:
            continue
        filtered.append(det)
    if not filtered:
        return None

    def _key(det: dict[str, Any]) -> tuple[int, int, int, float]:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        edge_penalty = 1 if det.get("touches_image_edge") else 0
        # Prefer top detections for plates because side views often crop the
        # rim and can turn background/fridge edges into high-score masks.
        camera_rank = 0 if camera == "top" else 1
        target_rank = 0
        if target_y_sign in {"negative", "neg", "right", "right_side"}:
            xyz = det.get("position_3d") or []
            target_rank = 0 if len(xyz) >= 2 and float(xyz[1]) < 0.0 else 1
        elif target_y_sign in {"positive", "pos", "left", "left_side"}:
            xyz = det.get("position_3d") or []
            target_rank = 0 if len(xyz) >= 2 and float(xyz[1]) > 0.0 else 1
        return (camera_rank, edge_penalty, target_rank, _target_prompt_rank(det), -float(det.get("score") or 0.0))

    return sorted(filtered, key=_key)[0]


def _detection_selection_details(
    packet: dict[str, Any],
    selected: dict[str, Any] | None,
    min_score: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected_xyz = (selected or {}).get("position_3d") or []
    for det in packet.get("all_detections") or []:
        item = dict(det)
        camera = str(item.get("source_camera") or item.get("camera") or "")
        score = float(item.get("score") or 0.0)
        edge = _detection_touches_image_edge(item)
        item["touches_image_edge"] = edge
        item["target_prompt_rank"] = _target_prompt_rank(item)
        item["eligible_for_motion_pose"] = bool(
            score >= min_score
            and camera in {"top", "left", "right"}
            and (camera == "top" or not edge or _truthy_env("OPENFORGE_PLATE_ALLOW_SIDE_CAMERA_EDGE_DETECTIONS", False))
        )
        xyz = item.get("position_3d") or []
        if len(xyz) >= 3 and len(selected_xyz) >= 3:
            dx = float(xyz[0]) - float(selected_xyz[0])
            dy = float(xyz[1]) - float(selected_xyz[1])
            dz = float(xyz[2]) - float(selected_xyz[2])
            item["delta_from_selected_m"] = [round(dx, 5), round(dy, 5), round(dz, 5)]
            item["xy_distance_from_selected_m"] = round(math.hypot(dx, dy), 5)
        rows.append(item)
    selected_camera = str((selected or {}).get("source_camera") or (selected or {}).get("camera") or "")
    return {
        "policy": (
            "prefer top-camera dish/plate masks; reject side-camera image-edge crops unless explicitly allowed; "
            "prefer the explicit object prompt over generic bowl/plate fallback prompts"
        ),
        "min_score": min_score,
        "selected_camera": selected_camera or None,
        "selected_prompt": (selected or {}).get("prompt"),
        "selected_score": (selected or {}).get("score"),
        "selected_position_3d": (selected or {}).get("position_3d"),
        "all_detections": rows,
        "rejected_side_edge_detections": [
            item for item in rows if item.get("touches_image_edge") and item.get("source_camera") != "top"
        ],
    }


def _prompt_matches_target_bowl(prompt: str, label: str) -> bool:
    if not _object_is_bowl():
        return False
    object_name = OBJECT_NAME.lower()
    text = f"{prompt} {label}".lower()
    if "orange" in object_name:
        return "orange bowl" in text or prompt.strip().lower() == "bowl"
    if any(token in object_name for token in ("dish", "plate")):
        return (
            "white dish" in text
            or "white plate" in text
            or prompt.strip().lower() in {"dish", "plate", "dinner plate"}
        )
    if "white" in object_name:
        return "white bowl" in text or prompt.strip().lower() == "bowl"
    return "bowl" in text


def _bowl_top_down_rim_z_model(center_z: float) -> tuple[float, dict[str, Any]]:
    mode = os.environ.get("OPENFORGE_BOWL_TOP_DOWN_RIM_Z_MODE", "detection_plus_bias").strip().lower()
    if mode in {"height_over_support", "support_height", "magic_height", "bowl_height"}:
        support_z = _env_float("OPENFORGE_BOWL_TOP_DOWN_SUPPORT_Z_M", BOWL_TOP_DOWN_SUPPORT_Z_DEFAULT_M)
        support_z += _env_float("OPENFORGE_BOWL_TOP_DOWN_SUPPORT_Z_OFFSET_M", 0.0)
        bowl_height = _env_float("OPENFORGE_BOWL_TOP_DOWN_HEIGHT_M", BOWL_TOP_DOWN_HEIGHT_DEFAULT_M)
        rim_z = support_z + bowl_height
        return rim_z, {
            "mode": mode,
            "selected_source": "support_z_plus_bowl_height",
            "support_z_m": round(support_z, 5),
            "bowl_height_m": round(bowl_height, 5),
            "center_z_m": round(center_z, 5),
            "selected_rim_z_m": round(rim_z, 5),
            "policy": "Use configured support/table Z plus configured bowl height; ignore masked bowl-depth Z for rim height.",
        }
    rim_z = center_z + _env_float(
        "OPENFORGE_BOWL_TOP_DOWN_RIM_Z_BIAS_M",
        BOWL_TOP_DOWN_RIM_Z_BIAS_DEFAULT_M,
    )
    return rim_z, {
        "mode": mode or "detection_plus_bias",
        "selected_source": "top_detection_plus_bias",
        "center_z_m": round(center_z, 5),
        "rim_z_bias_m": round(rim_z - center_z, 5),
        "selected_rim_z_m": round(rim_z, 5),
    }


def _attach_bowl_rim_z_evidence(
    selected: dict[str, Any] | None,
    detections: list[dict[str, Any]],
    min_score: float,
) -> None:
    if (
        not selected
        or not _object_is_bowl()
        or not _truthy_env("OPENFORGE_BOWL_TOP_DOWN_USE_MULTIVIEW_RIM_Z", True)
    ):
        return
    selected_xyz = selected.get("position_3d") or selected.get("position") or []
    if len(selected_xyz) < 3:
        return
    center_z = float(selected_xyz[2])
    default_rim_z, default_evidence = _bowl_top_down_rim_z_model(center_z)
    if default_evidence.get("selected_source") == "support_z_plus_bowl_height":
        selected["bowl_top_down_rim_z_m"] = round(default_rim_z, 5)
        selected["bowl_top_down_rim_z_evidence"] = {
            "enabled": True,
            "selected_source": "support_z_plus_bowl_height",
            "center_z_m": round(center_z, 5),
            "default_rim_z_m": round(default_rim_z, 5),
            "selected_rim_z_m": round(default_rim_z, 5),
            **default_evidence,
        }
        return
    target_y_sign = os.environ.get("OPENFORGE_PLATE_TARGET_Y_SIGN", "").strip().lower()
    allow_side_edge = _truthy_env("OPENFORGE_BOWL_TOP_DOWN_ALLOW_SIDE_EDGE_RIM_Z", False)
    max_raise = _env_float("OPENFORGE_BOWL_TOP_DOWN_MAX_RIM_Z_RAISE_M", 0.120)
    max_side_raise = _env_float("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_RAISE_M", 0.035)
    max_side_xy_disagreement = _env_float("OPENFORGE_BOWL_TOP_DOWN_MAX_SIDE_RIM_Z_XY_DISAGREEMENT_M", 0.080)
    max_top_xy_disagreement = _env_float("OPENFORGE_BOWL_TOP_DOWN_MAX_TOP_RIM_Z_XY_DISAGREEMENT_M", 0.080)
    side_bias = _env_float("OPENFORGE_BOWL_TOP_DOWN_SIDE_RIM_Z_BIAS_M", 0.0)
    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for det in detections:
        camera = str(det.get("source_camera") or det.get("camera") or "")
        if camera not in {"top", "left", "right"}:
            continue
        score = float(det.get("score") or 0.0)
        if score < min_score:
            continue
        xyz = det.get("position_3d") or det.get("position") or []
        if len(xyz) < 3:
            continue
        if not _prompt_matches_target_bowl(str(det.get("prompt") or ""), str(det.get("label") or "")):
            continue
        edge = _detection_touches_image_edge(det)
        if edge and camera != "top" and not allow_side_edge:
            continue
        y = float(xyz[1])
        if target_y_sign in {"negative", "neg", "right", "right_side"} and y > 0.0:
            continue
        if target_y_sign in {"positive", "pos", "left", "left_side"} and y < 0.0:
            continue
        xy_disagreement = ((float(xyz[0]) - float(selected_xyz[0])) ** 2 + (float(xyz[1]) - float(selected_xyz[1])) ** 2) ** 0.5
        if camera == "top" and xy_disagreement > max_top_xy_disagreement:
            rejected_rows.append(
                {
                    "prompt": det.get("prompt"),
                    "label": det.get("label"),
                    "score": round(score, 4),
                    "source_camera": camera,
                    "position_3d": [round(float(v), 5) for v in xyz[:3]],
                    "xy_disagreement_m": round(xy_disagreement, 5),
                    "reason": "top-camera bowl detection XY disagrees with selected top-camera source",
                }
            )
            continue
        if camera != "top" and xy_disagreement > max_side_xy_disagreement:
            rejected_rows.append(
                {
                    "prompt": det.get("prompt"),
                    "label": det.get("label"),
                    "score": round(score, 4),
                    "source_camera": camera,
                    "position_3d": [round(float(v), 5) for v in xyz[:3]],
                    "xy_disagreement_m": round(xy_disagreement, 5),
                    "reason": "side-camera bowl detection XY disagrees with selected top-camera source",
                }
            )
            continue
        z = float(xyz[2]) + (side_bias if camera != "top" else 0.0)
        if camera != "top":
            z = min(z, default_rim_z + max_side_raise)
        rows.append(
            {
                "prompt": det.get("prompt"),
                "label": det.get("label"),
                "score": round(score, 4),
                "source_camera": camera,
                "box_2d": det.get("box_2d"),
                "position_3d": [round(float(v), 5) for v in xyz[:3]],
                "touches_image_edge": edge,
                "xy_disagreement_m": round(xy_disagreement, 5),
                "candidate_rim_z_m": round(z, 5),
            }
        )
    if not rows:
        selected["bowl_top_down_rim_z_m"] = round(default_rim_z, 5)
        selected["bowl_top_down_rim_z_evidence"] = {
            "enabled": True,
            "selected_source": "top_detection_plus_bias",
            "center_z_m": round(center_z, 5),
            "default_rim_z_m": round(default_rim_z, 5),
            "selected_rim_z_m": round(default_rim_z, 5),
            "max_top_rim_z_xy_disagreement_m": round(max_top_xy_disagreement, 5),
            "max_side_rim_z_xy_disagreement_m": round(max_side_xy_disagreement, 5),
            "detections_considered": [],
            "detections_rejected": rejected_rows,
        }
        return
    max_detection_z = max(float(row["candidate_rim_z_m"]) for row in rows)
    selected_rim_z = max(default_rim_z, min(center_z + max_raise, max_detection_z))
    selected["bowl_top_down_rim_z_m"] = round(selected_rim_z, 5)
    selected["bowl_top_down_rim_z_evidence"] = {
        "enabled": True,
        "selected_source": "max_live_bowl_detection_z_for_rim",
        "center_z_m": round(center_z, 5),
        "default_rim_z_m": round(default_rim_z, 5),
        "max_rim_z_raise_m": round(max_raise, 5),
        "max_side_rim_z_raise_m": round(max_side_raise, 5),
        "max_side_rim_z_xy_disagreement_m": round(max_side_xy_disagreement, 5),
        "max_top_rim_z_xy_disagreement_m": round(max_top_xy_disagreement, 5),
        "selected_rim_z_m": round(selected_rim_z, 5),
        "detections_considered": sorted(
            rows,
            key=lambda row: (0 if row.get("source_camera") != "top" else 1, -float(row["candidate_rim_z_m"])),
        ),
        "detections_rejected": rejected_rows,
        "policy": (
            "Use top-camera XY/center as the motion source. Side-camera rim-height evidence is accepted only "
            "when its XY agrees with the selected source and its height raise is capped; otherwise use top-camera "
            "depth plus a small rim bias because inconsistent side detections can place the TCP above the actual rim."
        ),
    }


def _box_center_px(box: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return None
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]


def _box_size_px(box: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = [float(v) for v in list(box)[:4]]
    except Exception:
        return None
    return [max(0.0, x1 - x0), max(0.0, y1 - y0)]


def _box_gap_px(a: Any, b: Any) -> float | None:
    try:
        ax0, ay0, ax1, ay1 = [float(v) for v in list(a)[:4]]
        bx0, by0, bx1, by1 = [float(v) for v in list(b)[:4]]
    except Exception:
        return None
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return math.hypot(dx, dy)


def _gripper_plate_relation(det: dict[str, Any], plate_detection: dict[str, Any] | None) -> dict[str, Any]:
    relation: dict[str, Any] = {
        "same_camera_as_selected_plate": False,
        "plate_box_2d": None,
        "gripper_box_2d": det.get("box_2d"),
        "interpretation": "image-space diagnostic only; not used as a physical target",
    }
    if not plate_detection:
        return relation
    det_camera = str(det.get("source_camera") or det.get("camera") or "")
    plate_camera = str(plate_detection.get("source_camera") or plate_detection.get("camera") or "")
    plate_box = plate_detection.get("box_2d") or []
    grip_box = det.get("box_2d") or []
    relation["same_camera_as_selected_plate"] = bool(det_camera and det_camera == plate_camera)
    relation["plate_box_2d"] = plate_box
    plate_center = _box_center_px(plate_box)
    grip_center = _box_center_px(grip_box)
    plate_size = _box_size_px(plate_box)
    grip_size = _box_size_px(grip_box)
    if plate_center and grip_center and plate_size and grip_size:
        dx = grip_center[0] - plate_center[0]
        dy = grip_center[1] - plate_center[1]
        center_distance = math.hypot(dx, dy)
        plate_radius_px = max(plate_size) / 2.0
        relation.update(
            {
                "plate_center_px": _round_list(plate_center, digits=2, limit=2),
                "gripper_center_px": _round_list(grip_center, digits=2, limit=2),
                "plate_radius_px_estimate": round(plate_radius_px, 2),
                "gripper_size_px": _round_list(grip_size, digits=2, limit=2),
                "gripper_long_axis_px": round(max(grip_size), 2),
                "center_distance_px": round(center_distance, 2),
                "center_distance_minus_plate_radius_px": round(center_distance - plate_radius_px, 2),
                "box_gap_px": round(_box_gap_px(grip_box, plate_box) or 0.0, 2),
            }
        )
    xyz = det.get("position_3d") or []
    pxyz = (plate_detection or {}).get("position_3d") or []
    if len(xyz) >= 3 and len(pxyz) >= 3:
        delta = [float(xyz[i]) - float(pxyz[i]) for i in range(3)]
        relation.update(
            {
                "delta_from_selected_plate_m": _round_list(delta, limit=3),
                "xy_distance_from_selected_plate_m": round(math.hypot(delta[0], delta[1]), 5),
                "z_delta_from_selected_plate_m": round(delta[2], 5),
            }
        )
    return relation


def _capture_gripper_geometry_diagnostic(
    run_dir: Path,
    plate_detection: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _truthy_env("OPENFORGE_PLATE_GRIPPER_GEOMETRY_CAPTURE", True):
        return {"enabled": False, "reason": "OPENFORGE_PLATE_GRIPPER_GEOMETRY_CAPTURE disabled"}
    prompts = _csv(
        "OPENFORGE_PLATE_GRIPPER_PROMPTS",
        [
            "right gripper",
            "left gripper",
            "robot gripper",
            "gripper fingers",
            "blue gripper fingers",
            "black gripper",
        ],
    )
    cameras = _csv("OPENFORGE_PLATE_GRIPPER_CAMERAS", _debug_observation_cameras())
    if not cameras:
        cameras = ["top"]
    packet = capture_scene(
        prompts=prompts,
        cameras=cameras,
        detect_objects_oneshot=_required_tool("detect_objects_oneshot"),
        get_camera_image=_tool("get_camera_image"),
        get_robot_state=_tool("get_robot_state"),
        run_in_background=_tool("run_in_background"),
        run_dir=run_dir,
        stage="gripper_geometry_diagnostic",
        task_name=TASK_NAME,
        timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
        max_retries=max(1, _env_int("OPENFORGE_PLATE_GRIPPER_DETECT_RETRIES", 2)),
        save_overlays=True,
        motion_cameras=("top", "left", "right"),
        image_only_cameras=_image_only_debug_cameras(),
    )
    min_score = _env_float("OPENFORGE_PLATE_GRIPPER_MIN_SCORE", 0.18)
    rows: list[dict[str, Any]] = []
    for det in packet.get("all_detections") or []:
        item = dict(det)
        score = float(item.get("score") or 0.0)
        if score < min_score:
            continue
        size = _box_size_px(item.get("box_2d") or [])
        if size:
            item["box_size_px"] = _round_list(size, digits=2, limit=2)
            item["long_axis_px"] = round(max(size), 2)
            item["short_axis_px"] = round(min(size), 2)
        half_extents = item.get("half_extents") or []
        if len(half_extents) >= 2:
            item["detected_extent_long_axis_m"] = round(2.0 * max(abs(float(half_extents[0])), abs(float(half_extents[1]))), 5)
        item["relation_to_selected_plate"] = _gripper_plate_relation(item, plate_detection)
        rows.append(item)

    def _row_key(item: dict[str, Any]) -> tuple[int, float, float]:
        camera = str(item.get("source_camera") or item.get("camera") or "")
        relation = item.get("relation_to_selected_plate") or {}
        xy = relation.get("xy_distance_from_selected_plate_m")
        return (
            0 if camera == "top" else 1,
            float("inf") if xy is None else float(xy),
            -float(item.get("score") or 0.0),
        )

    rows = sorted(rows, key=_row_key)
    diagnostic = {
        "schema": "openforge.plate_gripper_geometry_diagnostic.v1",
        "enabled": True,
        "source": "fresh live SAM3/detect_objects_oneshot prompts; image/world diagnostic only",
        "prompts": prompts,
        "cameras": cameras,
        "min_score": min_score,
        "selected_plate_detection": plate_detection,
        "detections": rows,
        "selected_near_plate_gripper_detection": rows[0] if rows else None,
        "planner_use": (
            "Use this to check gripper length/shape relative to the plate edge. "
            "Do not use it as a replacement for fresh plate detection or motion preview."
        ),
        "packet_path": packet.get("packet_path"),
    }
    diagnostic["path"] = write_json(run_dir / "observations" / "gripper_geometry_diagnostic.json", diagnostic)
    append_stage_summary(
        run_dir,
        [
            "## gripper geometry diagnostic",
            f"- enabled: {diagnostic['enabled']}",
            f"- detections: {len(rows)}",
            f"- selected_near_plate: {diagnostic['selected_near_plate_gripper_detection']}",
            f"- path: {diagnostic['path']}",
        ],
    )
    return diagnostic


def _capture_prompt_fallback(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prompts = _csv(
        "OPENFORGE_PLATE_PROMPTS",
        ["plate", "dinner plate", "white plate"],
    )
    cameras = _debug_observation_cameras()
    min_score = _env_float("OPENFORGE_PLATE_MIN_SCORE", 0.25)
    detect_objects_oneshot = _required_tool("detect_objects_oneshot")
    get_camera_image = _tool("get_camera_image")
    get_robot_state = _tool("get_robot_state")
    run_in_background = _tool("run_in_background")
    packets: list[dict[str, Any]] = []
    selection_details: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    target_y_sign = os.environ.get("OPENFORGE_PLATE_TARGET_Y_SIGN", "").strip().lower()
    collect_all_prompts = _truthy_env("OPENFORGE_PLATE_COLLECT_ALL_PROMPTS", bool(target_y_sign))
    for prompt_index, prompt in enumerate(prompts):
        packet = capture_scene(
            prompts=[prompt],
            cameras=cameras,
            detect_objects_oneshot=detect_objects_oneshot,
            get_camera_image=get_camera_image,
            get_robot_state=get_robot_state,
            run_in_background=run_in_background,
            run_dir=run_dir,
            stage=f"pre_observe_plate_{prompt.replace(' ', '_')}",
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_DEBUG_OBS_TIMEOUT_S", 18.0),
            max_retries=max(1, _env_int("OPENFORGE_PLATE_DETECT_RETRIES", 2)),
            save_overlays=True,
            motion_cameras=("top", "left", "right"),
            image_only_cameras=_image_only_debug_cameras(),
        )
        packets.append(packet)
        detection = _select_plate_detection(packet, min_score)
        selection_details.append(_detection_selection_details(packet, detection, min_score))
        score = float((detection or {}).get("score") or 0.0)
        camera = str((detection or {}).get("source_camera") or "")
        if (
            not collect_all_prompts
            and detection is not None
            and score >= min_score
            and camera in {"top", "left", "right"}
        ):
            best = dict(detection)
            best["prompt_index"] = prompt_index
            best["prompt_fallback_used"] = bool(prompt_index > 0)
            best["raw_detection_source"] = best.get("source")
            best["source"] = "live_top_camera" if camera == "top" else "live_calibrated_camera"
            best["detection_source"] = best["source"]
            _attach_bowl_rim_z_evidence(best, [dict(det) for det in packet.get("all_detections") or []], min_score)
            break
    if collect_all_prompts and packets:
        all_detections: list[dict[str, Any]] = []
        for packet in packets:
            all_detections.extend(dict(det) for det in packet.get("all_detections") or [])
        detection = _select_plate_detection({"all_detections": all_detections}, min_score)
        selection_details.append(
            {
                "policy": "global selection after collecting all plate/dish prompts",
                "target_y_sign": target_y_sign or None,
                **_detection_selection_details({"all_detections": all_detections}, detection, min_score),
            }
        )
        score = float((detection or {}).get("score") or 0.0)
        camera = str((detection or {}).get("source_camera") or "")
        if detection is not None and score >= min_score and camera in {"top", "left", "right"}:
            best = dict(detection)
            try:
                best["prompt_index"] = prompts.index(str(best.get("prompt")))
            except ValueError:
                best["prompt_index"] = None
            best["prompt_fallback_used"] = bool(best.get("prompt_index") not in (None, 0))
            best["raw_detection_source"] = best.get("source")
            best["source"] = "live_top_camera" if camera == "top" else "live_calibrated_camera"
            best["detection_source"] = best["source"]
            _attach_bowl_rim_z_evidence(best, all_detections, min_score)
    combined = {
        "stage": "pre_observe_plate",
        "source": "live_perception",
        "prompts_tried": prompts[: len(packets)],
        "collect_all_prompts": collect_all_prompts,
        "target_y_sign": target_y_sign or None,
        "packets": [
            {
                "packet_path": packet.get("packet_path"),
                "best_detection": packet.get("best_detection"),
                "errors": packet.get("errors"),
            }
            for packet in packets
        ],
        "best_detection": best,
        "detection_selection_details": selection_details,
    }
    combined["path"] = write_json(run_dir / "observations" / "pre_observe_plate_prompt_fallback.json", combined)
    return combined, best


def _workspace_bounds() -> dict[str, tuple[float, float]]:
    return {
        "x": (
            _env_float("OPENFORGE_PLATE_WORKSPACE_X_MIN_M", 0.30),
            _env_float("OPENFORGE_PLATE_WORKSPACE_X_MAX_M", 1.05),
        ),
        "y": (
            _env_float("OPENFORGE_PLATE_WORKSPACE_Y_MIN_M", -0.65),
            _env_float("OPENFORGE_PLATE_WORKSPACE_Y_MAX_M", 0.65),
        ),
        "z": (
            _env_float("OPENFORGE_PLATE_WORKSPACE_Z_MIN_M", 0.55),
            _env_float("OPENFORGE_PLATE_WORKSPACE_Z_MAX_M", 1.20),
        ),
    }


def _validate_detection(detection: dict[str, Any] | None) -> dict[str, Any]:
    min_score = _env_float("OPENFORGE_PLATE_MIN_SCORE", 0.25)
    bounds = _workspace_bounds()
    result: dict[str, Any] = {
        "passed": False,
        "min_score": min_score,
        "workspace_bounds_m": bounds,
        "problems": [],
        "warnings": [],
    }
    if detection is None:
        result["problems"].append("missing plate detection")
        return result
    score = float(detection.get("score") or 0.0)
    result["score"] = score
    if score < min_score:
        result["problems"].append(f"score {score:.3f} below minimum {min_score:.3f}")
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        result["problems"].append("missing 3D position")
    else:
        rounded = _round_list(xyz, limit=3)
        result["position_3d"] = rounded
        for axis, value in zip(("x", "y", "z"), rounded):
            lo, hi = bounds[axis]
            if value < lo or value > hi:
                result["problems"].append(f"{axis}={value:.3f} outside [{lo:.3f}, {hi:.3f}]")
    box = detection.get("box_2d") or []
    if len(box) >= 4:
        result["box_2d"] = _round_list(box, digits=1, limit=4)
        camera = str(detection.get("source_camera") or detection.get("camera") or "")
        if _detection_touches_image_edge(detection):
            if camera == "top":
                result["warnings"].append("top-camera plate box touches image edge")
            elif not _truthy_env("OPENFORGE_PLATE_ALLOW_SIDE_CAMERA_EDGE_DETECTIONS", False):
                result["problems"].append("side-camera plate box touches image edge")
    result["passed"] = not result["problems"]
    return result


def _robot_plate_proximity(
    observation_packet: dict[str, Any],
    detection: dict[str, Any] | None,
) -> dict[str, Any]:
    xyz = (detection or {}).get("position_3d") or (detection or {}).get("position")
    result: dict[str, Any] = {
        "checked": False,
        "plate_position_3d": _round_list(xyz or [], limit=3),
        "arms": {},
        "blocking_risk": False,
        "warnings": [],
    }
    if not xyz or len(xyz) < 3:
        result["warnings"].append("missing plate position")
        return result
    state = observation_packet.get("packets", [{}])[0].get("robot_state")
    if state is None:
        # capture_scene stores robot_state in each raw packet, not in the combined
        # fallback packet.  Look up the packet JSON when available.
        packet_path = observation_packet.get("packets", [{}])[0].get("packet_path")
        if packet_path:
            try:
                import json

                state = json.loads(Path(packet_path).read_text(encoding="utf-8")).get("robot_state")
            except Exception:
                state = None
    arms = (state or {}).get("arms") if isinstance(state, dict) else None
    if not isinstance(arms, dict):
        result["warnings"].append("robot_state arms unavailable")
        return result
    result["checked"] = True
    plate = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    xy_warn = _env_float("OPENFORGE_PLATE_ARM_PROXIMITY_XY_WARN_M", 0.10)
    try:
        xy_warn = max(xy_warn, _plate_radius_m(detection or {}) + _env_float("OPENFORGE_PLATE_ARM_PROXIMITY_RADIUS_MARGIN_M", 0.08))
    except Exception:
        pass
    z_warn = _env_float("OPENFORGE_PLATE_ARM_PROXIMITY_Z_WARN_M", 0.10)
    for side in ("left", "right"):
        arm = arms.get(side) or {}
        ee = arm.get("ee_pos") or []
        if len(ee) < 3:
            continue
        ee_xyz = [float(ee[0]), float(ee[1]), float(ee[2])]
        delta = [ee_xyz[i] - plate[i] for i in range(3)]
        xy = math.hypot(delta[0], delta[1])
        z_abs = abs(delta[2])
        near = xy <= xy_warn and z_abs <= z_warn
        result["arms"][side] = {
            "ee_pos": _round_list(ee_xyz, limit=3),
            "delta_from_plate_m": _round_list(delta, limit=3),
            "xy_distance_m": round(xy, 5),
            "abs_z_delta_m": round(z_abs, 5),
            "near_plate": near,
        }
        if near:
            result["blocking_risk"] = True
            result["warnings"].append(
                f"{side} end effector is within {xy:.3f}m XY and {z_abs:.3f}m Z of the plate"
            )
    return result


def _plate_radius_estimate(detection: dict[str, Any]) -> dict[str, Any]:
    half_extents = detection.get("half_extents") or []
    values: list[float] = []
    override_radius = _env_float_or_none("OPENFORGE_PLATE_RADIUS_OVERRIDE_M")
    source = "env:OPENFORGE_PLATE_RADIUS_OVERRIDE_M"
    raw_radius = override_radius if override_radius is not None and override_radius > 0 else None
    if raw_radius is None:
        for value in list(half_extents)[:2]:
            try:
                fval = abs(float(value))
                if fval > 0:
                    values.append(fval)
            except Exception:
                pass
        source = "detection_half_extents"
        raw_radius = max(values) if values else None
    if raw_radius is None and _truthy_env("OPENFORGE_PLATE_USE_TOP_BOX_RADIUS", False):
        box = detection.get("box_2d") or []
        camera = str(detection.get("source_camera") or detection.get("camera") or "")
        if len(box) >= 4 and camera == "top":
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
            long_axis_px = max(abs(x1 - x0), abs(y1 - y0))
            reference_long_axis_px = _env_float("OPENFORGE_PLATE_REFERENCE_BOX_LONG_AXIS_PX", 56.0)
            reference_radius_m = _env_float("OPENFORGE_PLATE_REFERENCE_RADIUS_M", 0.095)
            if long_axis_px > 0 and reference_long_axis_px > 0 and reference_radius_m > 0:
                raw_radius = reference_radius_m * (long_axis_px / reference_long_axis_px)
                source = "top_box_scaled_from_reference_radius"
    if raw_radius is None:
        raw_radius = _env_float("OPENFORGE_PLATE_DEFAULT_RADIUS_M", 0.095)
        source = "default_radius"
    clipped = max(
        _env_float("OPENFORGE_PLATE_MIN_RADIUS_M", 0.045),
        min(_env_float("OPENFORGE_PLATE_MAX_RADIUS_M", 0.16), raw_radius),
    )
    return {
        "radius_m": clipped,
        "raw_radius_m": raw_radius,
        "source": source,
        "clipped": abs(clipped - raw_radius) > 1e-9,
        "half_extents": list(half_extents)[:3],
        "box_2d": detection.get("box_2d"),
        "source_camera": detection.get("source_camera") or detection.get("camera"),
        "reference_box_long_axis_px": _env_float("OPENFORGE_PLATE_REFERENCE_BOX_LONG_AXIS_PX", 56.0),
        "reference_radius_m": _env_float("OPENFORGE_PLATE_REFERENCE_RADIUS_M", 0.095),
    }


def _plate_radius_m(detection: dict[str, Any]) -> float:
    return float(_plate_radius_estimate(detection)["radius_m"])


def _candidate_arms(rim_y: float, requested: str) -> list[str]:
    if requested in {"left", "right"}:
        return [requested]
    deadband = _env_float("OPENFORGE_PLATE_ARM_DEADBAND_Y_M", 0.08)
    if rim_y > deadband:
        return ["left"]
    if rim_y < -deadband:
        return ["right"]
    return ["right", "left"]


def _side_grasp_rpy_from_approach(approach_xy: tuple[float, float]) -> list[float]:
    ax, ay = approach_xy
    yaw_deg = math.degrees(math.atan2(-ay, ax))
    return [0.0, 90.0, yaw_deg]


def _normalize_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _offset_rpy(rpy: list[float], offset: list[float]) -> list[float]:
    return [
        _normalize_angle_deg(float(rpy[0]) + float(offset[0])),
        _normalize_angle_deg(float(rpy[1]) + float(offset[1])),
        _normalize_angle_deg(float(rpy[2]) + float(offset[2])),
    ]


def _unit3(values: Any, fallback: tuple[float, float, float]) -> list[float]:
    try:
        raw = [float(v) for v in list(values)[:3]]
    except Exception:
        raw = list(fallback)
    if len(raw) < 3:
        raw = list(fallback)
    norm = math.sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2])
    if norm < 1e-8:
        raw = list(fallback)
        norm = math.sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2])
    return [raw[0] / norm, raw[1] / norm, raw[2] / norm]


def _dot3(a: Any, b: Any) -> float:
    av = [float(v) for v in list(a)[:3]]
    bv = [float(v) for v in list(b)[:3]]
    return av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2]


def _cross3(a: Any, b: Any) -> list[float]:
    av = [float(v) for v in list(a)[:3]]
    bv = [float(v) for v in list(b)[:3]]
    return [
        av[1] * bv[2] - av[2] * bv[1],
        av[2] * bv[0] - av[0] * bv[2],
        av[0] * bv[1] - av[1] * bv[0],
    ]


def _sub3(a: Any, b: Any) -> list[float]:
    av = [float(v) for v in list(a)[:3]]
    bv = [float(v) for v in list(b)[:3]]
    return [av[0] - bv[0], av[1] - bv[1], av[2] - bv[2]]


def _add3(a: Any, b: Any) -> list[float]:
    av = [float(v) for v in list(a)[:3]]
    bv = [float(v) for v in list(b)[:3]]
    return [av[0] + bv[0], av[1] + bv[1], av[2] + bv[2]]


def _scale3(a: Any, scalar: float) -> list[float]:
    av = [float(v) for v in list(a)[:3]]
    return [av[0] * scalar, av[1] * scalar, av[2] * scalar]


def _rotate3_about_axis(values: Any, axis: Any, angle_rad: float) -> list[float]:
    v = np.asarray([float(x) for x in list(values)[:3]], dtype=np.float64)
    k = np.asarray(_unit3(axis, (0.0, 0.0, 1.0)), dtype=np.float64)
    theta = float(angle_rad)
    rotated = v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + k * float(np.dot(k, v)) * (1.0 - math.cos(theta))
    return _unit3(rotated.tolist(), (float(v[0]), float(v[1]), float(v[2])))


def _display_rpy_from_axes(opening_axis: Any, approach_axis_hint: Any) -> list[float]:
    """Return display RPY whose local X is opening and local Z approaches rim."""
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation

        x_axis = np.asarray(_unit3(opening_axis, (0.0, 0.0, 1.0)), dtype=float)
        z_hint = np.asarray(_unit3(approach_axis_hint, (1.0, 0.0, 0.0)), dtype=float)
        z_axis = z_hint - x_axis * float(np.dot(z_hint, x_axis))
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-8:
            z_hint = np.asarray([1.0, 0.0, 0.0] if abs(x_axis[0]) < 0.9 else [0.0, 1.0, 0.0])
            z_axis = z_hint - x_axis * float(np.dot(z_hint, x_axis))
            z_norm = float(np.linalg.norm(z_axis))
        z_axis = z_axis / z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / float(np.linalg.norm(y_axis))
        matrix = np.column_stack([x_axis, y_axis, z_axis])
        euler_xyz = Rotation.from_matrix(matrix).as_euler("xyz", degrees=True)
        roll, pitch, yaw = [float(euler_xyz[1]), float(-euler_xyz[0]), float(-euler_xyz[2] - 90.0)]
        return [_normalize_angle_deg(roll), _normalize_angle_deg(pitch), _normalize_angle_deg(yaw)]
    except Exception:
        ax, ay = [float(v) for v in list(approach_axis_hint)[:2]]
        return _rim_pinch_rpy_from_approach((ax, ay))


def _display_rpy_world_axes(rpy: Any) -> dict[str, Any]:
    """Return world-frame local axes for the display-RPY convention used by freespace_move."""
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation

        display_roll, display_pitch, display_yaw = [float(v) for v in list(rpy)[:3]]
        euler_xyz = [-display_pitch, display_roll, -display_yaw - 90.0]
        matrix = Rotation.from_euler("xyz", euler_xyz, degrees=True).as_matrix()
        return {
            "source": "display_rpy_to_internal_euler_xyz",
            "display_rpy": _round_list([display_roll, display_pitch, display_yaw], limit=3),
            "internal_euler_xyz_deg": _round_list(euler_xyz, limit=3),
            "local_x_world": _round_list(np.asarray(matrix)[:, 0], limit=3),
            "local_y_world": _round_list(np.asarray(matrix)[:, 1], limit=3),
            "local_z_world": _round_list(np.asarray(matrix)[:, 2], limit=3),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "display_rpy": _round_list(rpy, limit=3)}


def _top_down_axis_gate(rpy: Any, opening_axis: Any, approach_axis: Any) -> dict[str, Any]:
    axes = _display_rpy_world_axes(rpy)
    if axes.get("error"):
        return {"passed": False, "reason": axes["error"], "ee_axes_world": axes}
    local_x = axes["local_x_world"]
    local_z = axes["local_z_world"]
    min_opening = _env_float("OPENFORGE_BOWL_TOP_DOWN_MIN_OPENING_AXIS_DOT", 0.75)
    min_approach = _env_float("OPENFORGE_BOWL_TOP_DOWN_MIN_APPROACH_AXIS_DOT", 0.75)
    opening_alignment = abs(_dot3(local_x, _unit3(opening_axis, (1.0, 0.0, 0.0))))
    approach_alignment = _dot3(local_z, _unit3(approach_axis, (0.0, 0.0, -1.0)))
    passed = opening_alignment >= min_opening and approach_alignment >= min_approach
    return {
        "passed": bool(passed),
        "opening_axis_alignment": round(float(opening_alignment), 5),
        "approach_axis_alignment": round(float(approach_alignment), 5),
        "min_opening_axis_alignment": min_opening,
        "min_approach_axis_alignment": min_approach,
        "expected_opening_axis_world": _round_list(opening_axis, limit=3),
        "expected_approach_axis_world": _round_list(approach_axis, limit=3),
        "ee_axes_world": axes,
        "reason": None if passed else "display RPY axes do not match top-down bowl strategy",
    }


def _rim_pinch_rpy_from_approach(approach_xy: tuple[float, float]) -> list[float]:
    ax, ay = approach_xy
    approach_yaw_deg = math.degrees(math.atan2(ay, ax))
    pitch_deg = _normalize_angle_deg(-approach_yaw_deg - 180.0)
    # Display RPY is converted internally as euler xyz [-pitch, roll, -yaw-90].
    # This combination makes local X, the modeled opening axis, vertical while
    # local Z approaches the chosen rim edge horizontally.
    return [-90.0, pitch_deg, -90.0]


def _tilted_rim_pinch_axes(
    *,
    radial: tuple[float, float],
    inward: tuple[float, float],
    tilt_deg: float,
) -> tuple[list[float], list[float], list[float]]:
    tilt_rad = math.radians(max(0.0, float(tilt_deg)))
    # Opening axis points from lower finger to upper finger.  Giving it an
    # outward component means the lower finger is modeled down-and-inward under
    # the lip, while the upper finger stays up-and-outside above the rim.
    opening = _unit3(
        [radial[0] * math.sin(tilt_rad), radial[1] * math.sin(tilt_rad), math.cos(tilt_rad)],
        (0.0, 0.0, 1.0),
    )
    approach_hint = _unit3([inward[0], inward[1], 0.0], (1.0, 0.0, 0.0))
    rpy = _display_rpy_from_axes(opening, approach_hint)
    return opening, approach_hint, rpy


def _build_side_rim_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        raise RuntimeError("plate detection has no 3D position")
    center = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    radius_estimate = _plate_radius_estimate(detection)
    radius = float(radius_estimate["radius_m"])
    rim_inset = _env_float("OPENFORGE_PLATE_RIM_INSET_M", 0.012)
    grasp_radius = max(0.01, radius - rim_inset)
    pregrasp_standoff = _env_float("OPENFORGE_PLATE_PREGRASP_STANDOFF_M", 0.085)
    lift_z = _env_float("OPENFORGE_PLATE_LIFT_Z_M", 0.07)
    z_offsets = _float_csv("OPENFORGE_PLATE_RIM_Z_OFFSETS_M", [0.025, 0.04, 0.06, 0.08])
    angles = _float_csv("OPENFORGE_PLATE_RIM_ANGLES_DEG", [90.0, -90.0, 0.0, 180.0, 45.0, 135.0, -45.0, -135.0])
    requested_arm = os.environ.get("OPENFORGE_PLATE_ARM", "auto").strip().lower()
    width = _env_float("OPENFORGE_PLATE_RIM_CLOSE_WIDTH_M", 0.035)
    max_candidates = max(1, _env_int("OPENFORGE_PLATE_MAX_CANDIDATES", 32))
    candidates: list[dict[str, Any]] = []
    for z_index, z_offset in enumerate(z_offsets):
        grasp_z = center[2] + float(z_offset)
        for angle_index, angle_deg in enumerate(angles):
            theta = math.radians(float(angle_deg))
            radial = (math.cos(theta), math.sin(theta))
            grasp = [
                center[0] + radial[0] * grasp_radius,
                center[1] + radial[1] * grasp_radius,
                grasp_z,
            ]
            inward = (-radial[0], -radial[1])
            pre = [
                grasp[0] - inward[0] * pregrasp_standoff,
                grasp[1] - inward[1] * pregrasp_standoff,
                grasp[2],
            ]
            lift = [grasp[0], grasp[1], grasp[2] + lift_z]
            rpy = _side_grasp_rpy_from_approach(inward)
            for arm in _candidate_arms(grasp[1], requested_arm):
                idx = len(candidates)
                candidates.append(
                    {
                        "label": f"plate_rim_side_{idx}",
                        "strategy_type": "flat_plate_rim_side_grasp",
                        "arm": arm,
                        "position": grasp,
                        "rpy": rpy,
                        "score": 1.0 - 0.04 * angle_index - 0.03 * z_index,
                        "width": width,
                        "pregrasp_pose": {"position": pre, "rpy": rpy},
                        "grasp_pose": {"position": grasp, "rpy": rpy},
                        "lift_pose": {"position": lift, "rpy": rpy},
                        "lift_policy": {
                            "first_lift_uses_grasp_orientation": True,
                            "grasp_rpy": rpy,
                            "lift_rpy": rpy,
                            "reorientation_before_verified_lift": False,
                            "source": "plate-libero-transfer high-level flat-object guidance",
                        },
                        "approach_direction_world": _round_list(inward, limit=2) + [0.0],
                        "rim_radial_direction_world": _round_list(radial, limit=2) + [0.0],
                        "rim_angle_deg": float(angle_deg),
                        "z_offset_m": float(z_offset),
                        "estimated_plate_center_m": _round_list(center, limit=3),
                        "estimated_plate_radius_m": radius,
                        "estimated_grasp_radius_m": grasp_radius,
                        "source_detection": detection,
                        "staged_close_plan": {
                            "policy": "future_physical_only",
                            "do_not_blind_full_close": True,
                            "target_normalized_position": _env_float("OPENFORGE_PLATE_CLOSE_TARGET_POS", 0.45),
                            "steps": _float_csv("OPENFORGE_PLATE_CLOSE_STEPS", [0.75, 0.60, 0.50, 0.45]),
                            "vel_limit": _env_float("OPENFORGE_PLATE_CLOSE_VEL_LIMIT", 0.08),
                            "torque_limit": _env_float("OPENFORGE_PLATE_CLOSE_TORQUE_LIMIT", 0.18),
                            "require_contact_before_lift": True,
                        },
                            "flat_object_cautions": [
                                "rim/side grasp candidate; not a blind top-down plate grasp",
                                "future physical execution must use staged closure and contact evidence",
                                "lift slightly before any reorientation",
                            ],
                    }
                )
                if len(candidates) >= max_candidates:
                    return candidates
    return candidates


def _build_top_down_rim_straddle_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        raise RuntimeError("bowl detection has no 3D position")
    center = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    radius_estimate = _plate_radius_estimate(detection)
    radius = float(radius_estimate["radius_m"])
    rim_inset = _env_float("OPENFORGE_BOWL_TOP_DOWN_RIM_INSET_M", 0.004)
    rim_inward_offset = _env_float("OPENFORGE_BOWL_TOP_DOWN_RIM_INWARD_OFFSET_M", 0.0)
    grasp_radius = max(0.01, radius - rim_inset)
    rim_z_default, rim_z_default_evidence = _bowl_top_down_rim_z_model(center[2])
    rim_z = float(detection.get("bowl_top_down_rim_z_m") or rim_z_default)
    tcp_above_rim = _env_float(
        "OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M",
        BOWL_TOP_DOWN_TCP_ABOVE_RIM_DEFAULT_M,
    )
    tcp_above_rim_values = _float_csv("OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_CANDIDATES_M", [tcp_above_rim])
    pregrasp_z_lift = _env_float("OPENFORGE_BOWL_TOP_DOWN_PREGRASP_Z_LIFT_M", 0.070)
    low_pregrasp_z_lift = _env_float("OPENFORGE_BOWL_TOP_DOWN_LOW_PREGRASP_Z_LIFT_M", 0.030)
    low_pregrasp_min_above_rim = _env_float("OPENFORGE_BOWL_TOP_DOWN_LOW_PREGRASP_MIN_ABOVE_RIM_M", 0.040)
    lift_z = _env_float("OPENFORGE_BOWL_TOP_DOWN_LIFT_Z_M", 0.055)
    xy_biases = _float_csv("OPENFORGE_BOWL_TOP_DOWN_RIM_XY_BIASES_M", [0.0, -0.010, 0.010])
    angles = _float_csv("OPENFORGE_BOWL_TOP_DOWN_RIM_ANGLES_DEG", [-90.0, 0.0, 180.0, 90.0, -45.0, 45.0, -135.0, 135.0])
    requested_arm = os.environ.get("OPENFORGE_PLATE_ARM", "auto").strip().lower()
    max_candidates = max(1, _env_int("OPENFORGE_PLATE_MAX_CANDIDATES", 48))
    preshape = _env_float("OPENFORGE_BOWL_TOP_DOWN_PRESHAPE_POS", 0.68)
    close_steps = _float_csv("OPENFORGE_BOWL_TOP_DOWN_CLOSE_STEPS", [0.68, 0.56, 0.48, 0.40])
    close_target = _env_float("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TARGET_POS", close_steps[-1] if close_steps else 0.40)
    close_vel = _env_float("OPENFORGE_BOWL_TOP_DOWN_CLOSE_VEL_LIMIT", 0.05)
    close_torque = _env_float("OPENFORGE_BOWL_TOP_DOWN_CLOSE_TORQUE_LIMIT", 0.12)
    rpy_offsets = _rpy_offsets("OPENFORGE_BOWL_TOP_DOWN_RPY_OFFSETS_DEG", [[0.0, 0.0, 0.0]])
    inward_tilts_deg = _float_csv("OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG", [0.0])
    tcp_to_contact_offsets = _float_csv("OPENFORGE_BOWL_TOP_DOWN_TCP_TO_CONTACT_OFFSETS_M", [0.060, 0.075, 0.045, 0.0])
    finger_forward_signs = _float_csv("OPENFORGE_BOWL_TOP_DOWN_FINGER_FORWARD_SIGNS", [-1.0, 1.0])
    reject_axis_mismatch = _truthy_env("OPENFORGE_BOWL_TOP_DOWN_REJECT_AXIS_MISMATCH", True)
    if len(inward_tilts_deg) > 1:
        original_tilt = os.environ.get("OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG")
        tilt_quota = max(1, math.ceil(max_candidates / len(inward_tilts_deg)))
        candidates: list[dict[str, Any]] = []
        try:
            for tilt_index, inward_tilt_deg in enumerate(inward_tilts_deg):
                os.environ["OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG"] = f"{float(inward_tilt_deg):.6g}"
                for candidate in _build_top_down_rim_straddle_candidates(detection)[:tilt_quota]:
                    item = dict(candidate)
                    item["label"] = f"bowl_top_down_rim_straddle_{len(candidates)}"
                    item["inward_tilt_candidate_index"] = tilt_index
                    item["score"] = float(item.get("score") or 0.0) - 0.002 * tilt_index
                    candidates.append(item)
                    if len(candidates) >= max_candidates:
                        return candidates
        finally:
            if original_tilt is None:
                os.environ.pop("OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG", None)
            else:
                os.environ["OPENFORGE_BOWL_TOP_DOWN_INWARD_TILT_DEG"] = original_tilt
        return candidates
    candidates: list[dict[str, Any]] = []
    for depth_index, tcp_above_rim_value in enumerate(tcp_above_rim_values):
        for angle_index, angle_deg in enumerate(angles):
            theta = math.radians(float(angle_deg))
            radial = (math.cos(theta), math.sin(theta))
            horizontal_opening_axis = _unit3([radial[0], radial[1], 0.0], (1.0, 0.0, 0.0))
            vertical_approach_axis = [0.0, 0.0, -1.0]
            # The visible fingertip/contact patch is offset from the abstract TCP.
            # Physical evidence showed the sign can flip with wrist orientation, so
            # sweep both signs and rank the currently supported flipped model first.
            nominal_finger_forward_axis = _unit3(
                _cross3(vertical_approach_axis, horizontal_opening_axis),
                (1.0, 0.0, 0.0),
            )
            inward_tilt_deg = float(inward_tilts_deg[0]) if inward_tilts_deg else 0.0
            # Positive configured tilt means lean the tool's down approach
            # toward the bowl center.  The rotation axis is finger-forward, and
            # the rim radial points outward, so the applied axis rotation is
            # negative under the right-hand rule.
            applied_tilt_rad = -math.radians(inward_tilt_deg)
            opening_axis = _rotate3_about_axis(horizontal_opening_axis, nominal_finger_forward_axis, applied_tilt_rad)
            approach_axis = _rotate3_about_axis(vertical_approach_axis, nominal_finger_forward_axis, applied_tilt_rad)
            inward_axis = _unit3([-radial[0], -radial[1], 0.0], (1.0, 0.0, 0.0))
            tilt_model = {
                "configured_inward_tilt_deg": round(inward_tilt_deg, 5),
                "applied_axis_rotation_deg": round(math.degrees(applied_tilt_rad), 5),
                "inward_axis_world": _round_list(inward_axis, limit=3),
                "approach_inward_alignment": round(float(_dot3(approach_axis, inward_axis) or 0.0), 5),
                "policy": "Positive inward tilt leans the local-Z approach axis toward the bowl center.",
            }
            base_rpy = _display_rpy_from_axes(opening_axis, approach_axis)
            rim_point = [
                center[0] + radial[0] * radius,
                center[1] + radial[1] * radius,
                rim_z,
            ]
            for bias_index, xy_bias in enumerate(xy_biases):
                # The inward offset moves the physical gripper center toward the
                # bowl center while keeping the descent and lift vertically aligned.
                candidate_grasp_radius = max(0.01, grasp_radius - rim_inward_offset + float(xy_bias))
                gap_center = [
                    center[0] + radial[0] * candidate_grasp_radius,
                    center[1] + radial[1] * candidate_grasp_radius,
                    rim_z,
                ]
                for sign_index, finger_forward_sign in enumerate(finger_forward_signs):
                    finger_forward_axis = _unit3(
                        _scale3(nominal_finger_forward_axis, float(finger_forward_sign)),
                        nominal_finger_forward_axis,
                    )
                    for tcp_offset_index, tcp_to_contact_offset in enumerate(tcp_to_contact_offsets):
                        tcp_center = _sub3(gap_center, _scale3(finger_forward_axis, float(tcp_to_contact_offset)))
                        grasp = [tcp_center[0], tcp_center[1], rim_z + float(tcp_above_rim_value)]
                        low_pre_z = max(grasp[2] + low_pregrasp_z_lift, rim_z + low_pregrasp_min_above_rim)
                        high_pre_z = max(grasp[2] + pregrasp_z_lift, low_pre_z)
                        low_pre = [grasp[0], grasp[1], low_pre_z]
                        high_pre = [grasp[0], grasp[1], high_pre_z]
                        lift = [grasp[0], grasp[1], grasp[2] + lift_z]
                        for offset_index, rpy_offset in enumerate(rpy_offsets):
                            rpy = _offset_rpy(base_rpy, rpy_offset)
                            axis_gate = _top_down_axis_gate(rpy, opening_axis, approach_axis)
                            if reject_axis_mismatch and not axis_gate.get("passed"):
                                continue
                            for arm in _candidate_arms(gap_center[1], requested_arm):
                                idx = len(candidates)
                                candidates.append(
                                    {
                                        "label": f"bowl_top_down_rim_straddle_{idx}",
                                        "strategy_type": "top_down_bowl_rim_straddle",
                                        "arm": arm,
                                        "position": grasp,
                                        "rpy": rpy,
                                        "score": (
                                            1.15
                                            - 0.025 * depth_index
                                            - 0.04 * angle_index
                                            - 0.015 * bias_index
                                            - 0.003 * sign_index
                                            - 0.004 * tcp_offset_index
                                            - 0.006 * offset_index
                                        ),
                                        "width": _env_float("OPENFORGE_BOWL_TOP_DOWN_WALL_THICKNESS_M", 0.010),
                                        "approach_pose": {"position": high_pre, "rpy": rpy},
                                        "low_pregrasp_pose": {"position": low_pre, "rpy": rpy},
                                        "pregrasp_pose": {"position": low_pre, "rpy": rpy},
                                        "grasp_pose": {"position": grasp, "rpy": rpy},
                                        "lift_pose": {"position": lift, "rpy": rpy},
                                        "lift_policy": {
                                            "first_lift_uses_grasp_orientation": True,
                                            "grasp_rpy": rpy,
                                            "lift_rpy": rpy,
                                            "reorientation_before_verified_lift": False,
                                            "source": "bowl vertical-wall rim straddle; lift straight up before any transport",
                                        },
                                        "approach_direction_world": approach_axis,
                                        "rim_radial_direction_world": _round_list(radial, limit=2) + [0.0],
                                        "opening_axis_world": _round_list(opening_axis, limit=3),
                                        "finger_forward_axis_world": _round_list(finger_forward_axis, limit=3),
                                        "finger_forward_sign": round(float(finger_forward_sign), 5),
                                        "inward_tilt_deg": round(inward_tilt_deg, 5),
                                        "inward_tilt_model": tilt_model,
                                        "tcp_to_contact_offset_m": round(float(tcp_to_contact_offset), 5),
                                        "tcp_above_rim_m": round(float(tcp_above_rim_value), 5),
                                        "rim_angle_deg": float(angle_deg),
                                        "rim_xy_bias_m": float(xy_bias),
                                        "rim_inward_offset_m": round(float(rim_inward_offset), 5),
                                        "base_rpy": base_rpy,
                                        "rpy_offset_deg": rpy_offset,
                                        "rpy_offset_index": offset_index,
                                        "axis_gate": axis_gate,
                                        "ee_axes_world": axis_gate.get("ee_axes_world"),
                                        "estimated_plate_center_m": _round_list(center, limit=3),
                                        "estimated_plate_radius_m": radius,
                                        "estimated_grasp_radius_m": round(candidate_grasp_radius, 5),
                                        "radius_estimate": radius_estimate,
                                        "source_detection": detection,
                                        "selected_rim_edge": {
                                            "edge_point_m": _round_list(rim_point, limit=3),
                                            "gap_center_m": _round_list(gap_center, limit=3),
                                            "contact_center_m": _round_list(gap_center, limit=3),
                                            "tcp_center_m": _round_list(grasp, limit=3),
                                            "approach_side": {
                                                "rim_angle_deg": float(angle_deg),
                                                "policy": "top-down straddle: one finger inside bowl, one outside bowl wall",
                                                "inward_tilt_deg": round(inward_tilt_deg, 5),
                                                "inward_tilt_model": tilt_model,
                                                "horizontal_opening_axis_world": _round_list(
                                                    horizontal_opening_axis,
                                                    limit=3,
                                                ),
                                                "opening_axis_world": _round_list(opening_axis, limit=3),
                                                "approach_direction_world": approach_axis,
                                                "finger_forward_axis_world": _round_list(finger_forward_axis, limit=3),
                                                "finger_forward_sign": round(float(finger_forward_sign), 5),
                                            },
                                        },
                                        "rim_geometry_estimate": {
                                            "estimated_rim_z_m": round(rim_z, 5),
                                            "default_center_plus_bias_rim_z_m": round(rim_z_default, 5),
                                            "default_rim_z_model": rim_z_default_evidence,
                                            "rim_z_evidence": detection.get("bowl_top_down_rim_z_evidence"),
                                            "tcp_above_rim_m": round(float(tcp_above_rim_value), 5),
                                            "tcp_to_contact_offset_m": round(float(tcp_to_contact_offset), 5),
                                            "rim_inward_offset_m": round(float(rim_inward_offset), 5),
                                            "finger_forward_sign": round(float(finger_forward_sign), 5),
                                            "inward_tilt_deg": round(inward_tilt_deg, 5),
                                            "inward_tilt_model": tilt_model,
                                            "radius_source": radius_estimate.get("source"),
                                            "radius_estimate": radius_estimate,
                                            "top_camera_detection_required": True,
                                            "orientation_model": "axis-gated display RPY: local Z points down and leans toward bowl center for positive inward tilt; local X spans inside/outside across rim; local Y/finger-forward is offset from TCP with sign swept from physical evidence",
                                        },
                                        "gripper_length_model": {
                                            "source": "2026-05-29 bowl top-camera diagnostic plus blue-gripper geometry reference",
                                            "tcp_to_visible_blue_contact_offset_m": round(float(tcp_to_contact_offset), 5),
                                            "finger_forward_axis_world": _round_list(finger_forward_axis, limit=3),
                                            "finger_forward_sign": round(float(finger_forward_sign), 5),
                                            "expected_contact_center_m": _round_list(gap_center, limit=3),
                                            "commanded_tcp_center_m": _round_list(grasp, limit=3),
                                            "purpose": "place the visible fingertip/contact patch over the bowl wall rather than the abstract TCP",
                                        },
                                        "preshape_plan": {
                                            "expected_gripper_open_before_close_normalized": preshape,
                                            "purpose": "descend with jaws open around the bowl rim before staged close",
                                        },
                                        "staged_close_plan": {
                                            "policy": "future_physical_only",
                                            "do_not_blind_full_close": True,
                                            "target_normalized_position": close_target,
                                            "steps": close_steps,
                                            "vel_limit": close_vel,
                                            "torque_limit": close_torque,
                                            "require_contact_before_lift": _truthy_env(
                                                "OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True
                                            ),
                                            "expected_contact_delta_min": _env_float(
                                                "OPENFORGE_BOWL_TOP_DOWN_CONTACT_MIN_DELTA_POS", 0.010
                                            ),
                                            "stop_if_no_contact": _truthy_env(
                                                "OPENFORGE_BOWL_RACK_REQUIRE_CONTACT_BEFORE_LIFT", True
                                            ),
                                        },
                                        "flat_object_cautions": [
                                            "top-down rim straddle for a bowl vertical wall; not a top-surface grasp",
                                            "one blue fingertip should be inside the bowl and the other outside the wall before close",
                                            "contact center is intentionally offset from the TCP using the visible blue fingertip model",
                                            "finger-forward sign is swept because physical camera evidence showed the nominal sign can miss free-space",
                                            "close gently with contact evidence before lift",
                                            "lift straight up only a few centimeters before any transport or rack placement",
                                        ],
                                    }
                                )
                                if len(candidates) >= max_candidates:
                                    return candidates
    return candidates


def _build_asymmetric_rim_pinch_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    xyz = detection.get("position_3d") or detection.get("position")
    if not xyz or len(xyz) < 3:
        raise RuntimeError("plate detection has no 3D position")
    center = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    radius = _plate_radius_m(detection)
    rim_inset = _env_float("OPENFORGE_PLATE_RIM_PINCH_INSET_M", 0.006)
    grasp_radius = max(0.01, radius - rim_inset)
    pregrasp_standoff = _env_float("OPENFORGE_PLATE_RIM_PINCH_PREGRASP_STANDOFF_M", 0.080)
    pregrasp_z_lift = _env_float("OPENFORGE_PLATE_RIM_PINCH_PREGRASP_Z_LIFT_M", 0.0)
    low_entry_enabled = _truthy_env("OPENFORGE_PLATE_RIM_PINCH_LOW_ENTRY", True)
    high_approach_z_lift = _env_float("OPENFORGE_PLATE_RIM_PINCH_HIGH_APPROACH_Z_LIFT_M", 0.050)
    lift_z = _env_float("OPENFORGE_PLATE_RIM_PINCH_LIFT_Z_M", 0.050)
    plate_thickness = _env_float("OPENFORGE_PLATE_ESTIMATED_THICKNESS_M", 0.012)
    rim_top_z = center[2] + _env_float("OPENFORGE_PLATE_RIM_TOP_Z_BIAS_M", 0.0)
    rim_bottom_z = rim_top_z - plate_thickness
    configured_tcp_offset = _env_float_or_none("OPENFORGE_PLATE_RIM_PINCH_TCP_ABOVE_FINGER_GAP_M")
    tcp_above_finger_gap_offsets = (
        [configured_tcp_offset]
        if configured_tcp_offset is not None
        else _float_csv("OPENFORGE_PLATE_RIM_PINCH_TCP_ABOVE_FINGER_GAP_OFFSETS_M", [0.040, 0.035, 0.050, 0.060])
    )
    lower_clearance = _env_float("OPENFORGE_PLATE_LOWER_FINGER_UNDER_EDGE_CLEARANCE_M", 0.008)
    upper_clearance = _env_float("OPENFORGE_PLATE_UPPER_FINGER_ABOVE_RIM_CLEARANCE_M", 0.010)
    min_lower_clearance = _env_float("OPENFORGE_PLATE_MIN_LOWER_UNDER_EDGE_CLEARANCE_M", 0.002)
    min_upper_clearance = _env_float("OPENFORGE_PLATE_MIN_UPPER_ABOVE_RIM_CLEARANCE_M", 0.004)
    finger_forward_reach = _env_float("OPENFORGE_PLATE_GRIPPER_FINGER_FORWARD_REACH_M", 0.045)
    min_lower_tip_inset = _env_float("OPENFORGE_PLATE_MIN_LOWER_FINGERTIP_UNDER_LIP_M", 0.016)
    desired_gap = max(
        _env_float("OPENFORGE_PLATE_MIN_ASYMMETRIC_PINCH_GAP_M", 0.018),
        plate_thickness + lower_clearance + upper_clearance,
    )
    nominal_center_z = ((rim_bottom_z - lower_clearance) + (rim_top_z + upper_clearance)) / 2.0
    center_z_biases = _float_csv("OPENFORGE_PLATE_RIM_PINCH_CENTER_Z_BIASES_M", [-0.006, -0.004, 0.0, 0.003])
    angles = _float_csv("OPENFORGE_PLATE_RIM_PINCH_ANGLES_DEG", [90.0, -90.0, 0.0, 180.0, 45.0, 135.0, -45.0, -135.0])
    tilt_angles = _float_csv("OPENFORGE_PLATE_RIM_PINCH_TILT_DEG", [10.0, 14.0, 7.0, 0.0])
    min_lower_lip_inset = _env_float("OPENFORGE_PLATE_MIN_LOWER_FINGER_INWARD_UNDER_LIP_M", 0.007)
    requested_arm = os.environ.get("OPENFORGE_PLATE_ARM", "auto").strip().lower()
    max_candidates = max(1, _env_int("OPENFORGE_PLATE_MAX_CANDIDATES", 64))
    preshape = _env_float("OPENFORGE_PLATE_RIM_PINCH_PRESHAPE_POS", 0.68)
    close_steps = _float_csv("OPENFORGE_PLATE_RIM_PINCH_CLOSE_STEPS", [0.68, 0.62, 0.58, 0.54])
    close_target = _env_float("OPENFORGE_PLATE_RIM_PINCH_CLOSE_TARGET_POS", close_steps[-1] if close_steps else 0.54)
    close_vel = _env_float("OPENFORGE_PLATE_RIM_PINCH_CLOSE_VEL_LIMIT", 0.06)
    close_torque = _env_float("OPENFORGE_PLATE_RIM_PINCH_CLOSE_TORQUE_LIMIT", 0.14)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for z_index, center_z_bias in enumerate(center_z_biases):
        finger_gap_center_z = nominal_center_z + float(center_z_bias)
        lower_finger_z = finger_gap_center_z - desired_gap / 2.0
        upper_finger_z = finger_gap_center_z + desired_gap / 2.0
        lower_under_edge = rim_bottom_z - lower_finger_z
        upper_above_rim = upper_finger_z - rim_top_z
        plausible_z = lower_under_edge >= min_lower_clearance and upper_above_rim >= min_upper_clearance
        for tilt_index, tilt_deg in enumerate(tilt_angles):
            for tcp_offset_index, tcp_above_finger_gap in enumerate(tcp_above_finger_gap_offsets):
                tcp_center_z = finger_gap_center_z + float(tcp_above_finger_gap)
                for angle_index, angle_deg in enumerate(angles):
                    theta = math.radians(float(angle_deg))
                    radial = (math.cos(theta), math.sin(theta))
                    inward = (-radial[0], -radial[1])
                    rim_edge = [center[0] + radial[0] * radius, center[1] + radial[1] * radius, rim_top_z]
                    pinch = [
                        center[0] + radial[0] * grasp_radius,
                        center[1] + radial[1] * grasp_radius,
                        tcp_center_z,
                    ]
                    opening_axis, approach_axis, rpy = _tilted_rim_pinch_axes(
                        radial=radial,
                        inward=inward,
                        tilt_deg=float(tilt_deg),
                    )
                    finger_gap_center = [
                        center[0] + radial[0] * grasp_radius,
                        center[1] + radial[1] * grasp_radius,
                        finger_gap_center_z,
                    ]
                    lower_finger_center = _sub3(finger_gap_center, _scale3(opening_axis, desired_gap / 2.0))
                    upper_finger_center = _add3(finger_gap_center, _scale3(opening_axis, desired_gap / 2.0))
                    lower_fingertip_estimate = _add3(lower_finger_center, _scale3(approach_axis, finger_forward_reach))
                    upper_fingertip_estimate = _add3(upper_finger_center, _scale3(approach_axis, finger_forward_reach))
                    tilted_lower_under_edge = rim_bottom_z - lower_finger_center[2]
                    tilted_upper_above_rim = upper_finger_center[2] - rim_top_z
                    lower_inward_under_lip = _dot3(_sub3(lower_finger_center, rim_edge), [inward[0], inward[1], 0.0])
                    lower_tip_inward_under_lip = _dot3(
                        _sub3(lower_fingertip_estimate, rim_edge),
                        [inward[0], inward[1], 0.0],
                    )
                    upper_outward_from_edge = _dot3(_sub3(upper_finger_center, rim_edge), [radial[0], radial[1], 0.0])
                    plausible_tilted_z = (
                        tilted_lower_under_edge >= min_lower_clearance
                        and tilted_upper_above_rim >= min_upper_clearance
                        and lower_inward_under_lip >= min_lower_lip_inset
                        and lower_tip_inward_under_lip >= min_lower_tip_inset
                    )
                    if not (plausible_z and plausible_tilted_z):
                        rejected.append(
                            {
                                "rim_angle_deg": float(angle_deg),
                                "tilt_deg": float(tilt_deg),
                                "center_z_bias_m": float(center_z_bias),
                                "tcp_above_finger_gap_center_m": round(float(tcp_above_finger_gap), 5),
                                "reason": "under/over tilted finger clearances below configured minimum",
                                "lower_finger_under_edge_clearance_m": round(tilted_lower_under_edge, 5),
                                "upper_finger_above_rim_clearance_m": round(tilted_upper_above_rim, 5),
                                "lower_finger_inward_under_lip_m": round(lower_inward_under_lip, 5),
                                "lower_fingertip_inward_under_lip_m": round(lower_tip_inward_under_lip, 5),
                            }
                        )
                        continue
                    pre = [
                        pinch[0] - approach_axis[0] * pregrasp_standoff,
                        pinch[1] - approach_axis[1] * pregrasp_standoff,
                        pinch[2] + pregrasp_z_lift,
                    ]
                    high_pre = [pre[0], pre[1], pre[2] + high_approach_z_lift]
                    lift = [pinch[0], pinch[1], pinch[2] + lift_z]
                    for arm in _candidate_arms(pinch[1], requested_arm):
                        idx = len(candidates)
                        candidates.append(
                            {
                            "label": f"plate_rim_pinch_{idx}",
                            "strategy_type": "asymmetric_plate_rim_under_over_pinch",
                            "arm": arm,
                            "position": pinch,
                            "rpy": rpy,
                            "score": 1.25 - 0.04 * angle_index - 0.03 * z_index - 0.015 * tcp_offset_index - 0.02 * tilt_index,
                            "width": desired_gap,
                            "approach_pose": {"position": high_pre, "rpy": rpy} if low_entry_enabled else None,
                            "low_pregrasp_pose": {"position": pre, "rpy": rpy} if low_entry_enabled else None,
                            "pregrasp_pose": {"position": pre, "rpy": rpy},
                            "grasp_pose": {"position": pinch, "rpy": rpy},
                            "lift_pose": {"position": lift, "rpy": rpy},
                            "insertion_path": {
                                "enabled": low_entry_enabled,
                                "policy": "high approach, descend to low near-table pregrasp, then tilted horizontal insertion to rim",
                                "high_approach_pose": {"position": high_pre, "rpy": rpy} if low_entry_enabled else None,
                                "low_pregrasp_pose": {"position": pre, "rpy": rpy} if low_entry_enabled else None,
                                "final_grasp_pose": {"position": pinch, "rpy": rpy},
                                "approach_standoff_m": round(pregrasp_standoff, 5),
                                "high_approach_z_lift_m": round(high_approach_z_lift, 5),
                                "pregrasp_z_lift_m": round(pregrasp_z_lift, 5),
                                "reason": "operator image review showed previous direct approach left the actual fingers too high above the plate lip",
                            },
                            "lift_policy": {
                                "first_lift_uses_grasp_orientation": True,
                                "grasp_rpy": rpy,
                                "lift_rpy": rpy,
                                "reorientation_before_verified_lift": False,
                                "source": "plate-libero-transfer high-level flat-object guidance",
                            },
                            "approach_direction_world": _round_list(approach_axis, limit=3),
                            "rim_radial_direction_world": _round_list(radial, limit=2) + [0.0],
                            "opening_axis_world": _round_list(opening_axis, limit=3),
                            "lower_finger_axis_world": _round_list(_scale3(opening_axis, -1.0), limit=3),
                            "upper_finger_axis_world": _round_list(opening_axis, limit=3),
                            "rim_angle_deg": float(angle_deg),
                            "tilt_toward_lower_edge_deg": float(tilt_deg),
                            "z_offset_m": round(tcp_center_z - center[2], 5),
                            "center_z_bias_m": float(center_z_bias),
                            "tilt_index": tilt_index,
                            "tcp_offset_index": tcp_offset_index,
                            "estimated_plate_center_m": _round_list(center, limit=3),
                            "estimated_plate_radius_m": radius,
                            "estimated_grasp_radius_m": grasp_radius,
                            "source_detection": detection,
                            "selected_rim_edge": {
                                "edge_point_m": _round_list(rim_edge, limit=3),
                                "pinch_tcp_center_m": _round_list(pinch, limit=3),
                                "approach_side": {
                                    "rim_angle_deg": float(angle_deg),
                                    "tilt_toward_lower_edge_deg": float(tilt_deg),
                                    "approach_direction_world": _round_list(approach_axis, limit=3),
                                    "rim_radial_direction_world": _round_list(radial, limit=2) + [0.0],
                                },
                            },
                            "rim_geometry_estimate": {
                                "estimated_surface_z_m": round(rim_top_z, 5),
                                "estimated_bottom_z_m": round(rim_bottom_z, 5),
                                "estimated_thickness_m": round(plate_thickness, 5),
                                "radius_source": "detection_half_extents_or_default",
                                "top_camera_detection_required": True,
                            },
                            "tool_offset_model": {
                                "tcp_above_finger_gap_center_m": round(tcp_above_finger_gap, 5),
                                "finger_gap_center_z_m": round(finger_gap_center_z, 5),
                                "source": "calibrated from prior successful side-rim preview height; keeps live object XYZ unchanged",
                            },
                            "gripper_length_model": {
                                "finger_forward_reach_m": round(finger_forward_reach, 5),
                                "lower_fingertip_estimate_m": _round_list(lower_fingertip_estimate, digits=5, limit=3),
                                "upper_fingertip_estimate_m": _round_list(upper_fingertip_estimate, digits=5, limit=3),
                                "lower_fingertip_inward_under_lip_m": round(lower_tip_inward_under_lip, 5),
                                "min_lower_fingertip_under_lip_m": round(min_lower_tip_inset, 5),
                                "source": "script parameter plus live SAM3 gripper-geometry diagnostic; verify against observations/gripper_geometry_diagnostic.json",
                            },
                            "finger_clearance_model": {
                                "lower_finger_center_m": _round_list(lower_finger_center, digits=5, limit=3),
                                "upper_finger_center_m": _round_list(upper_finger_center, digits=5, limit=3),
                                "lower_finger_center_z_m": round(lower_finger_center[2], 5),
                                "upper_finger_center_z_m": round(upper_finger_center[2], 5),
                                "lower_finger_under_edge_clearance_m": round(tilted_lower_under_edge, 5),
                                "upper_finger_above_rim_clearance_m": round(tilted_upper_above_rim, 5),
                                "lower_finger_inward_under_lip_m": round(lower_inward_under_lip, 5),
                                "lower_fingertip_inward_under_lip_m": round(lower_tip_inward_under_lip, 5),
                                "upper_finger_outward_from_edge_m": round(upper_outward_from_edge, 5),
                                "desired_gap_m": round(desired_gap, 5),
                                "opening_axis_world": _round_list(opening_axis, limit=3),
                                "tilt_toward_lower_edge_deg": float(tilt_deg),
                                "passed": True,
                            },
                            "preshape_plan": {
                                "expected_gripper_open_before_close_normalized": preshape,
                                "purpose": "enter final rim-pinch pose with one finger modeled below the edge and one above the rim",
                            },
                            "staged_close_plan": {
                                "policy": "future_physical_only",
                                "do_not_blind_full_close": True,
                                "target_normalized_position": close_target,
                                "steps": close_steps,
                                "vel_limit": close_vel,
                                "torque_limit": close_torque,
                                "require_contact_before_lift": True,
                                "expected_contact_delta_min": _env_float(
                                    "OPENFORGE_PLATE_RIM_PINCH_CONTACT_MIN_DELTA_POS", 0.012
                                ),
                                "stop_if_no_contact": True,
                            },
                            "flat_object_cautions": [
                                "asymmetric under/over rim pinch; not side-close around empty rim space",
                                "candidate includes a low-entry path so the fingers approach the rim close to table height before insertion",
                                "lower finger must plausibly be tilted inward/down under the edge before any lift",
                                "upper finger stays above the rim/top surface",
                                "future physical execution must use conservative staged close and contact evidence",
                                "lift slightly before any reorientation",
                            ],
                        }
                    )
                        if len(candidates) >= max_candidates:
                            return candidates
    if not candidates:
        raise RuntimeError(f"no plausible asymmetric rim-pinch candidates generated: {rejected[:8]}")
    return candidates


def _build_rim_candidates(detection: dict[str, Any]) -> list[dict[str, Any]]:
    mode = _strategy_mode()
    top_down_modes = TOP_DOWN_RIM_STRATEGY_MODES
    if _object_is_bowl() and mode not in top_down_modes:
        raise RuntimeError(
            f"{OBJECT_NAME!r} requires a top-down rim-straddle strategy; refusing mode {mode!r}"
        )
    if mode in {"top_down_rim_straddle", "bowl_top_down", "top_down_bowl_rim"}:
        return _build_top_down_rim_straddle_candidates(detection)
    if mode in {"asymmetric_rim_pinch", "rim_pinch", "under_over_rim_pinch"}:
        return _build_asymmetric_rim_pinch_candidates(detection)
    if mode in {"combined", "pinch_then_side"}:
        pinch = _build_asymmetric_rim_pinch_candidates(detection)
        if len(pinch) >= _env_int("OPENFORGE_PLATE_MAX_CANDIDATES", 32):
            return pinch
        return pinch + _build_side_rim_candidates(detection)[: max(0, _env_int("OPENFORGE_PLATE_MAX_CANDIDATES", 32) - len(pinch))]
    if mode == "side_rim":
        return _build_side_rim_candidates(detection)
    raise RuntimeError(f"unknown plate pickup strategy mode {mode!r}; refusing silent side-rim fallback")


def _guarded_preview_freespace(freespace_move: Callable[..., Any]) -> Callable[..., Any]:
    def _wrapped(**kwargs: Any) -> Any:
        if not bool(kwargs.get("preview_only", False)):
            raise RuntimeError("plate no-motion planner attempted freespace_move without preview_only=True")
        result = freespace_move(**kwargs)
        safe = json_safe(result)
        if isinstance(safe, dict) and bool(safe.get("executed", False)):
            raise RuntimeError("preview_only freespace_move reported executed=True")
        return result

    return _wrapped


def _preview_wrapper_data(preview: dict[str, Any]) -> dict[str, Any]:
    result = json_safe(preview.get("result"))
    if not isinstance(result, dict):
        return {}
    data = result.get("data")
    if isinstance(data, dict):
        return data
    return result


def _preview_numeric(data: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = data.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _preview_status_text(data: dict[str, Any]) -> str:
    return str(data.get("status") or "").strip().lower()


def _candidate_allows_axis_gated_rotation_residual(candidate: dict[str, Any]) -> bool:
    if not _truthy_env("OPENFORGE_PLATE_ALLOW_AXIS_GATED_ROTATION_RESIDUAL", _object_is_bowl()):
        return False
    if candidate.get("strategy_type") != "top_down_bowl_rim_straddle":
        return False
    axis_gate = candidate.get("axis_gate") or {}
    return bool(axis_gate) and axis_gate.get("passed") is True


def _preview_stage_residual_gate(preview: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    data = _preview_wrapper_data(preview)
    stage = str(preview.get("stage") or "unknown")
    pos_error = _preview_numeric(data, "final_pos_error_m", "ik_error_m")
    rot_error = _preview_numeric(data, "final_rot_error_deg", "ik_rot_error_deg")
    max_pos = _env_float("OPENFORGE_PLATE_PREVIEW_MAX_POS_ERROR_M", _env_float("OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M", 0.02))
    max_rot = _env_float("OPENFORGE_PLATE_PREVIEW_MAX_ROT_ERROR_DEG", _env_float("OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG", 12.0))
    status = _preview_status_text(data)
    wrapper_ok = bool(preview.get("ok"))
    status_ok = status in {"success", "succeeded", "ok"} if status else wrapper_ok
    executed = bool(data.get("executed", False))
    axis_gated_rotation = _candidate_allows_axis_gated_rotation_residual(candidate or {})
    missing_residuals = pos_error is None or (rot_error is None and not axis_gated_rotation)
    allow_missing = _truthy_env("OPENFORGE_PLATE_ALLOW_MISSING_PREVIEW_RESIDUALS", False)
    rotation_ok = axis_gated_rotation or rot_error is None or rot_error <= max_rot
    residual_ok = bool(
        (allow_missing or not missing_residuals)
        and (pos_error is None or pos_error <= max_pos)
        and rotation_ok
    )
    passed = bool(wrapper_ok and status_ok and not executed and residual_ok)
    reason = None
    if not wrapper_ok:
        reason = str((preview.get("result") or {}).get("error") or "preview wrapper failed")
    elif not status_ok:
        reason = f"tool status is not success: {status or None}"
    elif executed:
        reason = "preview unexpectedly reported executed=True"
    elif missing_residuals and not allow_missing:
        reason = "preview residuals missing"
    elif pos_error is not None and pos_error > max_pos:
        reason = f"position residual {pos_error:.6f}m exceeds {max_pos:.6f}m"
    elif rot_error is not None and rot_error > max_rot and not axis_gated_rotation:
        reason = f"rotation residual {rot_error:.4f}deg exceeds {max_rot:.4f}deg"
    return {
        "stage": stage,
        "passed": passed,
        "reason": reason,
        "status": status or None,
        "executed": executed,
        "final_pos_error_m": pos_error,
        "final_rot_error_deg": rot_error,
        "max_pos_error_m": max_pos,
        "max_rot_error_deg": max_rot,
        "axis_gated_rotation_residual": axis_gated_rotation,
        "rotation_residual_policy": (
            "ignored because top-down bowl axis_gate independently validates local-Z down and rim-opening axis"
            if axis_gated_rotation
            else "raw motion-tool rotation residual must be within limit"
        ),
    }


def _apply_preview_residual_gates(plan: dict[str, Any]) -> dict[str, Any]:
    """Reject planner candidates whose reported final pose residual is unsafe."""
    candidates = []
    rejected: list[dict[str, Any]] = []
    for candidate in plan.get("candidates") or []:
        item = dict(candidate)
        checks = [_preview_stage_residual_gate(dict(preview), item) for preview in item.get("previews") or []]
        passed = bool(checks and all(check.get("passed") for check in checks))
        item["preview_success_raw"] = bool(item.get("preview_success"))
        item["preview_residual_gate"] = {
            "passed": passed,
            "checks": checks,
            "policy": (
                "motion-tool Success is accepted only when final position residuals are within limits; "
                "for axis-gated top-down bowl rim candidates, the known display-RPY rotation residual is "
                "accepted only when axis_gate independently validates local-Z down and rim-opening axis"
            ),
        }
        if not passed:
            item["preview_success"] = False
            first_bad = next((check for check in checks if not check.get("passed")), None)
            rejected.append(
                {
                    "label": item.get("label"),
                    "arm": item.get("arm"),
                    "strategy_type": item.get("strategy_type"),
                    "stage": (first_bad or {}).get("stage"),
                    "reason": (first_bad or {}).get("reason"),
                    "final_pos_error_m": (first_bad or {}).get("final_pos_error_m"),
                    "final_rot_error_deg": (first_bad or {}).get("final_rot_error_deg"),
                }
            )
        candidates.append(item)
    selected = next((item for item in candidates if item.get("preview_success")), None)
    plan = dict(plan)
    plan["candidates"] = candidates
    plan["selected_raw"] = plan.get("selected")
    plan["selected"] = selected
    plan["preview_residual_gate"] = {
        "selected_label": selected.get("label") if selected else None,
        "rejected_count": len(rejected),
        "rejected": rejected[:24],
    }
    return plan


def _source_snapshot(run_dir: Path) -> dict[str, Any]:
    script_path = Path(globals().get("__file__", "cap/saved_scripts/pick_plate_no_motion_plan.py")).resolve()
    snapshot: dict[str, Any] = {
        "script_path": str(script_path),
        "entrypoint": os.environ.get("OPENFORGE_PLATE_ENTRYPOINT"),
        "task_name": TASK_NAME,
        "object_name": OBJECT_NAME,
        "script_sha256": None,
        "git_head": None,
        "git_status_short": None,
        "env_knobs": {
            name: os.environ.get(name)
            for name in sorted(os.environ)
            if name.startswith("OPENFORGE_PLATE_")
            or name in {
                "YAM_STATION_CALIBRATED_XML",
                "CAP_TOP_CAMERA_BACKEND",
                "CAP_TOP_CAMERA_FRAME",
                "CAP_TOP_CAMERA_NEEDS_OPTICAL_FLIP",
            }
        },
    }
    try:
        import hashlib

        snapshot["script_sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    except Exception as exc:
        snapshot["script_sha256_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import subprocess

        snapshot["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        snapshot["git_status_short"] = subprocess.check_output(
            ["git", "status", "--short", "--", str(script_path.relative_to(Path.cwd()))],
            text=True,
        ).strip()
    except Exception as exc:
        snapshot["git_error"] = f"{type(exc).__name__}: {exc}"
    snapshot["path"] = write_json(run_dir / "source_snapshot.json", snapshot)
    return snapshot


def _classify_preview_error(error: str) -> str:
    if "no collision-free path" in error:
        return "no_collision_free_path"
    if "Invalid Problem" in error:
        return "planner_invalid_problem"
    if "IK failed" in error or "IK_Failed" in error:
        return "ik_failed"
    if "timeout" in error.lower():
        return "timeout"
    return "other"


def _preview_residuals(error: str) -> dict[str, float | None]:
    import re

    pos = None
    rot = None
    match = re.search(r"pos_err=([0-9.]+) m", error)
    if match:
        pos = float(match.group(1))
    match = re.search(r"rot_err=([0-9.]+) deg", error)
    if match:
        rot = float(match.group(1))
    return {"pos_err_m": pos, "rot_err_deg": rot}


def _analyze_plan_failures(
    plan: dict[str, Any],
    *,
    proximity: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    by_stage: dict[str, int] = {}
    by_error_class: dict[str, int] = {}
    near_misses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in plan.get("candidates") or []:
        bad = next((preview for preview in candidate.get("previews") or [] if not preview.get("ok")), None)
        residual_bad = next(
            (
                check
                for check in (candidate.get("preview_residual_gate") or {}).get("checks", [])
                if not check.get("passed")
            ),
            None,
        )
        if not bad and not residual_bad:
            continue
        stage = str((bad or residual_bad or {}).get("stage") or "unknown")
        if bad:
            error = str(((bad.get("result") or {}).get("error") or ""))
            error_class = _classify_preview_error(error)
            residuals = _preview_residuals(error)
        else:
            error = str((residual_bad or {}).get("reason") or "preview residual gate failed")
            error_class = "pose_residual_exceeded"
            residuals = {
                "pos_err_m": (residual_bad or {}).get("final_pos_error_m"),
                "rot_err_deg": (residual_bad or {}).get("final_rot_error_deg"),
            }
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_error_class[error_class] = by_error_class.get(error_class, 0) + 1
        item = {
            "label": candidate.get("label"),
            "arm": candidate.get("arm"),
            "rim_angle_deg": candidate.get("rim_angle_deg"),
            "z_offset_m": candidate.get("z_offset_m"),
            "failed_stage": stage,
            "error_class": error_class,
            "error": error,
            **residuals,
            "approach_pose": candidate.get("approach_pose"),
            "low_pregrasp_pose": candidate.get("low_pregrasp_pose"),
            "pregrasp_pose": candidate.get("pregrasp_pose"),
            "grasp_pose": candidate.get("grasp_pose"),
            "insertion_path": candidate.get("insertion_path"),
            "gripper_length_model": candidate.get("gripper_length_model"),
        }
        failures.append(item)
        if residuals["pos_err_m"] is not None or residuals["rot_err_deg"] is not None:
            near_misses.append(item)
    near_misses = sorted(
        near_misses,
        key=lambda item: (
            float("inf") if item.get("pos_err_m") is None else float(item["pos_err_m"]),
            float("inf") if item.get("rot_err_deg") is None else float(item["rot_err_deg"]),
        ),
    )
    likely_blockers: list[str] = []
    selected_label = (plan.get("selected") or {}).get("label")
    if proximity.get("blocking_risk"):
        likely_blockers.append("arm_pose_or_gripper_near_plate")
    if selected_label is None and by_stage and set(by_stage) == {"pregrasp_pose"}:
        likely_blockers.append("all_candidates_fail_before_reaching_pregrasp")
    if by_error_class.get("no_collision_free_path"):
        likely_blockers.append("collision_constraints")
    if by_error_class.get("ik_failed"):
        likely_blockers.append("ik_or_start_state_constraints")
    if by_error_class.get("pose_residual_exceeded"):
        likely_blockers.append("preview_pose_residuals_exceed_limits")
    best = near_misses[:8]
    analysis = {
        "schema": "openforge.plate_pickup_plan_analysis.v1",
        "candidate_count": len(plan.get("candidates") or []),
        "selected_label": selected_label,
        "rejected_candidate_count": len(failures),
        "by_failed_stage": by_stage,
        "by_error_class": by_error_class,
        "near_misses": best,
        "likely_blockers": likely_blockers,
        "interpretation": (
            "A candidate passed planner preview; rejected alternatives should be treated as strategy ranking evidence."
            if selected_label is not None
            else (
                "Current evidence points to a robot-state/arm-pose blocker before a plate-pickup physical attempt."
                if proximity.get("blocking_risk")
                else (
                    "Planner returned trajectories, but their final pose residuals exceeded limits; inspect orientation axes "
                    "and tune top-down/rim-pinch candidates."
                    if by_error_class.get("pose_residual_exceeded")
                    else "Current evidence is primarily planner/candidate failure; inspect near_misses and tune candidates."
                )
            )
        ),
        "next_no_motion_actions": [
            "Rerun after recovery/home or operator reset so the right gripper is not near the plate rim.",
            "Keep top-camera plate detection as the motion pose source; side-camera edge crops are diagnostic only.",
            "If arm pose is clear but previews still fail, probe near_miss rim angles with relaxed backend/thresholds.",
        ],
        "next_physical_action_requires_ticket": True,
    }
    analysis["path"] = write_json(run_dir / "plans" / "plate_preview_failure_analysis.json", analysis)
    return analysis


def _initial_result(run_dir: Path) -> dict[str, Any]:
    return {
        "success": False,
        "reward": 0.0,
        "confidence": 0.0,
        "method": TASK_NAME,
        "task": f"no_motion_{OBJECT_NAME.replace('/', '_').replace(' ', '_')}_pickup_plan",
        "object_name": OBJECT_NAME,
        "run_dir": str(run_dir),
        "physical_motion_executed": False,
        "movement_capable_calls": [],
        "pre_observation": None,
        "post_observation": None,
        "selected_strategy": None,
        "candidate_artifacts": {},
        "planner_result": None,
        "planner_failure_analysis": None,
        "gripper_geometry_diagnostic": None,
        "source_snapshot": None,
        "recording_evidence": None,
        "video_paths": [],
        "planner_config": {
            "strategy_mode": _strategy_mode(),
            "planner_backend": os.environ.get("OPENFORGE_PLATE_PLANNER_BACKEND", "curobo"),
            "solver_speed": os.environ.get("OPENFORGE_PLATE_SOLVER_SPEED", "fast"),
            "planning_speed": _env_float("OPENFORGE_PLATE_PLANNING_SPEED", 0.25),
            "ik_error_threshold_m": _env_float("OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M", 0.02),
            "ik_rot_threshold_deg": _env_float("OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG", 12.0),
        },
        "why_stopped": "not_started",
        "next_recommendation": None,
        "transfer_guidance_used": {
            "source": ".agents/skills/yam-simulation-transfer/references/plate-libero-transfer.md",
            "scope": "high_level_only",
            "borrowed": [
                "detect plate before possible occlusion",
                "prefer early top-camera plate pose over later side-view occlusions",
                "reason about fingertip/TCP offset instead of moving the live object pose",
                "inspect visible gripper geometry relative to the plate edge before retrying",
                "flat-object rim-grasp caution",
                "staged gripper closure metadata",
                "lift-before-reorient policy",
                "post-action verification requirement for future physical runs",
            ],
            "not_copied": [
                "LIBERO coordinates",
                "LIBERO quaternions",
                "simulator constants",
                "simulator APIs",
            ],
        },
    }


def _run() -> dict[str, Any]:
    run_dir = current_run_dir(TASK_NAME)
    result = _initial_result(run_dir)
    try:
        result["source_snapshot"] = _source_snapshot(run_dir)
        if _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
            result["why_stopped"] = "refusing: OPENFORGE_ALLOW_PHYSICAL_MOTION is set but this script is no-motion only"
            result["next_recommendation"] = "Unset OPENFORGE_ALLOW_PHYSICAL_MOTION and run recorded no-motion validation first."
            return result
        manual_vars = _manual_pose_env_vars_present()
        if manual_vars:
            result["why_stopped"] = f"refusing manual pose env vars for plate planning: {manual_vars}"
            result["next_recommendation"] = "Use fresh live plate detections; do not provide hardcoded plate XYZ."
            return result

        freespace_move = _required_tool("freespace_move")
        prompt_packet, plate_detection = _capture_prompt_fallback(run_dir)
        result["pre_observation"] = prompt_packet
        detection_sanity = _validate_detection(plate_detection)
        result["detection_sanity"] = detection_sanity
        proximity = _robot_plate_proximity(prompt_packet, plate_detection)
        result["robot_plate_proximity"] = proximity
        gripper_geometry = _capture_gripper_geometry_diagnostic(run_dir, plate_detection)
        result["gripper_geometry_diagnostic"] = gripper_geometry
        append_stage_summary(
            run_dir,
            [
                "## plate detection sanity",
                f"- passed: {detection_sanity['passed']}",
                f"- problems: {detection_sanity['problems']}",
                f"- selected_detection: {plate_detection}",
                "## robot/plate proximity",
                f"- checked: {proximity['checked']}",
                f"- blocking_risk: {proximity['blocking_risk']}",
                f"- warnings: {proximity['warnings']}",
                "## gripper/plate image geometry",
                f"- diagnostic_path: {gripper_geometry.get('path')}",
                f"- selected_near_plate_gripper: {gripper_geometry.get('selected_near_plate_gripper_detection')}",
            ],
        )
        if not detection_sanity.get("passed"):
            result["why_stopped"] = f"fresh live {OBJECT_NAME} detection sanity failed"
            result["next_recommendation"] = (
                f"Make the {OBJECT_NAME} visible to top/side cameras, then rerun recorded no-motion validation."
            )
            return result

        candidates = _build_rim_candidates(plate_detection or {})
        result["candidate_artifacts"]["plate_rim_candidates"] = write_json(
            run_dir / "plans" / "plate_rim_candidates.json",
            {
                "schema": "openforge.plate_pickup_candidates.v1",
                "strategy_mode": _strategy_mode(),
                "plate_detection": plate_detection,
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
        )
        preview_stage = (
            "plate_rim_pinch_preview"
            if "pinch" in _strategy_mode()
            else "plate_rim_pickup_preview"
        )
        plan = rank_motion_candidates(
            candidates=candidates,
            freespace_move=_guarded_preview_freespace(freespace_move),
            run_in_background=_tool("run_in_background"),
            run_dir=run_dir,
            stage=preview_stage,
            task_name=TASK_NAME,
            timeout_s=_env_float("OPENFORGE_PLATE_PREVIEW_TIMEOUT_S", 20.0),
            planner_backend=os.environ.get("OPENFORGE_PLATE_PLANNER_BACKEND", "curobo"),
            solver_speed=os.environ.get("OPENFORGE_PLATE_SOLVER_SPEED", "fast"),
            planning_speed=_env_float("OPENFORGE_PLATE_PLANNING_SPEED", 0.25),
            ik_error_threshold=_env_float("OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M", 0.02),
            ik_rot_threshold_deg=_env_float("OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG", 12.0),
            ik_xyz_weight=_env_float("OPENFORGE_PLATE_IK_XYZ_WEIGHT", 1.0),
            ik_rpy_weight=_env_float("OPENFORGE_PLATE_IK_RPY_WEIGHT", 0.3),
            stop_after_successes=_env_int("OPENFORGE_PLATE_PREVIEW_STOP_AFTER_SUCCESSES", 3),
        )
        plan = _apply_preview_residual_gates(plan)
        write_json(Path(str(plan["plan_path"])), plan)
        residual_gate = plan.get("preview_residual_gate") or {}
        append_stage_summary(
            run_dir,
            [
                "## preview residual gate",
                f"- selected_after_gate: {residual_gate.get('selected_label')}",
                f"- rejected_count: {residual_gate.get('rejected_count')}",
                f"- limits: pos<={_env_float('OPENFORGE_PLATE_PREVIEW_MAX_POS_ERROR_M', _env_float('OPENFORGE_PLATE_IK_ERROR_THRESHOLD_M', 0.02))}m, rot<={_env_float('OPENFORGE_PLATE_PREVIEW_MAX_ROT_ERROR_DEG', _env_float('OPENFORGE_PLATE_IK_ROT_THRESHOLD_DEG', 12.0))}deg",
                f"- first_rejections: {(residual_gate.get('rejected') or [])[:6]}",
            ],
        )
        result["planner_result"] = {
            "plan_path": plan.get("plan_path"),
            "candidate_count": len(plan.get("candidates") or []),
            "selected_label": (plan.get("selected") or {}).get("label"),
            "preview_residual_gate": residual_gate,
        }
        failure_analysis = _analyze_plan_failures(plan, proximity=proximity, run_dir=run_dir)
        result["planner_failure_analysis"] = failure_analysis
        result["planner_result"]["failure_analysis_path"] = failure_analysis.get("path")
        append_stage_summary(
            run_dir,
            [
                "## planner failure analysis",
                f"- by_failed_stage: {failure_analysis['by_failed_stage']}",
                f"- by_error_class: {failure_analysis['by_error_class']}",
                f"- likely_blockers: {failure_analysis['likely_blockers']}",
                f"- analysis: {failure_analysis['path']}",
            ],
        )
        selected = plan.get("selected")
        if selected:
            result["success"] = True
            result["reward"] = 1.0
            result["confidence"] = 0.55
            result["selected_strategy"] = selected
            if selected.get("strategy_type") in {
                "asymmetric_plate_rim_under_over_pinch",
                "top_down_bowl_rim_straddle",
            }:
                append_stage_summary(
                    run_dir,
                    [
                        "## selected rim-pinch evidence",
                        f"- selected_rim_edge: {selected.get('selected_rim_edge')}",
                        f"- insertion_path: {selected.get('insertion_path')}",
                        f"- finger_clearance_model: {selected.get('finger_clearance_model')}",
                        f"- rim_geometry_estimate: {selected.get('rim_geometry_estimate')}",
                        f"- approach_direction_world: {selected.get('approach_direction_world')}",
                        f"- opening_axis_world: {selected.get('opening_axis_world')}",
                        f"- gripper_length_model: {selected.get('gripper_length_model')}",
                        f"- preshape_plan: {selected.get('preshape_plan')}",
                        f"- staged_close_plan: {selected.get('staged_close_plan')}",
                        f"- preview_residual_gate: {selected.get('preview_residual_gate')}",
                    ],
                )
            result["why_stopped"] = f"recorded no-motion {OBJECT_NAME} pickup preview selected a rim-grasp candidate"
            result["next_recommendation"] = (
                f"Before any physical run, require a fresh {OBJECT_NAME} pickup ticket, enabled follower arm servers, "
                "recording/debug UI, staged close/contact gate, lift-only verification, and post-action observation."
            )
        else:
            result["why_stopped"] = f"no {OBJECT_NAME} rim candidate passed planner preview"
            if result.get("robot_plate_proximity", {}).get("blocking_risk"):
                result["next_recommendation"] = (
                    "A gripper is already close to the plate and may be blocking collision-free plans. "
                    "Do not move under this no-motion note; a director-approved recovery/home ticket or "
                    "operator scene reset is needed before physical work."
                )
            else:
                result["next_recommendation"] = (
                    "Inspect plans/plate_rim_candidates.json and the preview plan failures, then tune rim angles/z offsets."
                )
        return result
    except Exception as exc:
        result["why_stopped"] = f"{type(exc).__name__}: {exc}"
        result["next_recommendation"] = "Fix the no-motion observe/plan blocker and rerun without the physical gate."
        return result
    finally:
        recording = _recording_evidence(run_dir)
        result["recording_evidence"] = recording
        result["video_paths"] = recording.get("video_paths", [])
        if not recording.get("passed"):
            result["success"] = False
            result["reward"] = 0.0
            result["confidence"] = 0.0
            result["why_stopped"] = f"{result.get('why_stopped')}; recording evidence missing"
        write_json(run_dir / "result.json", result)
        write_json(run_dir / "task_result.json", result)
        append_stage_summary(
            run_dir,
            [
                "## result",
                f"- success: {result['success']}",
                f"- physical_motion_executed: {result['physical_motion_executed']}",
                f"- selected_strategy: {(result.get('selected_strategy') or {}).get('label')}",
                f"- recording_evidence: {recording}",
                f"- why_stopped: {result['why_stopped']}",
                f"- next_recommendation: {result['next_recommendation']}",
            ],
        )


TASK_RESULT.update(_run())
print(f"[{TASK_NAME}] success={TASK_RESULT.get('success')} why={TASK_RESULT.get('why_stopped')}")
