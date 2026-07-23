# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dry bottle-to-cup planning path for real YAM.

Current implementation scope:

- Stage 1: dry bottle side-grasp planning and gated dry physical validation.
- Stage 2: dry pour-path planning only.
- No Stage 2 physical motion.
- No liquid, water, or pouring.

Any movement-capable stage must keep the explicit
OPENFORGE_ALLOW_PHYSICAL_MOTION=1 gate before the first movement-capable call.

No-motion examples:

    OPENFORGE_BOTTLE_XYZ=0.55,-0.18,0.86 \
    OPENFORGE_BOTTLE_HALF_EXTENTS=0.04,0.04,0.15 \
    OPENFORGE_CUP_XYZ=0.62,0.08,0.79 \
    python3 cap/saved_scripts/pour_bottle_to_green_cup.py

    OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 \
    OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http \
    python3 cap/saved_scripts/pour_bottle_to_green_cup.py
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


BOTTLE_PROMPT = os.environ.get("OPENFORGE_BOTTLE_PROMPT", "bottle")
CUP_PROMPT = os.environ.get("OPENFORGE_CUP_PROMPT", "cup")
CUP_LABEL = os.environ.get("OPENFORGE_CUP_LABEL", "green cup")
CAMERA = os.environ.get("OPENFORGE_CAMERA", "top")

REQUESTED_STAGE = os.environ.get(
    "OPENFORGE_WINE_CUP_STAGE", "stage1_dry_side_grasp_only"
).strip()
VALIDATION_MODE = os.environ.get("OPENFORGE_VALIDATION_MODE", "no_motion_dry_plan")
ENABLE_READ_ONLY_PERCEPTION = os.environ.get(
    "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION", ""
).strip().lower() in {"1", "true", "yes", "on"}
PERCEPTION_BACKEND = os.environ.get("OPENFORGE_PERCEPTION_BACKEND", "bundlesdf_http").strip().lower()
BUNDLESDF_SERVICE_URL = os.environ.get("BUNDLESDF_SERVICE_URL", "http://127.0.0.1:8119").rstrip("/")
READ_ONLY_PERCEPTION_TIMEOUT_S = float(os.environ.get("OPENFORGE_READ_ONLY_PERCEPTION_TIMEOUT_S", "180"))

BODY_GRASP_FRACTION = float(os.environ.get("OPENFORGE_BOTTLE_BODY_GRASP_FRACTION", "0.45"))
GRASP_Z_OFFSET_M = float(os.environ.get("OPENFORGE_BOTTLE_GRASP_Z_OFFSET_M", "0.0"))
DEFAULT_BOTTLE_RADIUS_M = float(os.environ.get("OPENFORGE_BOTTLE_RADIUS_M", "0.04"))
GRASP_CLEARANCE_M = float(os.environ.get("OPENFORGE_BOTTLE_GRASP_CLEARANCE_M", "0.018"))
PREGRASP_STANDOFF_M = float(os.environ.get("OPENFORGE_BOTTLE_PREGRASP_STANDOFF_M", "0.12"))
LIFT_Z_M = float(os.environ.get("OPENFORGE_DRY_LIFT_Z_M", "0.08"))
PHYSICAL_LIFT_Z_M = float(os.environ.get("OPENFORGE_STAGE1_PHYSICAL_LIFT_Z_M", "0.05"))
POUR_STANDOFF_M = float(os.environ.get("OPENFORGE_DRY_POUR_STANDOFF_M", "0.12"))
PHYSICAL_HOLD_S = float(os.environ.get("OPENFORGE_STAGE1_HOLD_S", "1.0"))
PHYSICAL_PLANNING_SPEED = float(os.environ.get("OPENFORGE_STAGE1_PLANNING_SPEED", "0.45"))
PHYSICAL_IK_ERROR_THRESHOLD_M = float(
    os.environ.get("OPENFORGE_STAGE1_IK_ERROR_THRESHOLD_M", "0.015")
)
PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG = float(
    os.environ.get("OPENFORGE_STAGE1_IK_ROT_ERROR_THRESHOLD_DEG", "5.0")
)
PHYSICAL_GRIPPER_MIN_WIDTH_M = float(
    os.environ.get("OPENFORGE_STAGE1_GRIPPER_MIN_WIDTH_M", "0.004")
)
PHYSICAL_GRASP_ADVANCE_M = float(os.environ.get("OPENFORGE_STAGE1_GRASP_ADVANCE_M", "0.0"))
PHYSICAL_GRASP_Z_OFFSET_M = float(os.environ.get("OPENFORGE_STAGE1_GRASP_Z_OFFSET_M", "0.0"))
PHYSICAL_GRASP_ADVANCE_LIMIT_M = float(
    os.environ.get("OPENFORGE_STAGE1_GRASP_ADVANCE_LIMIT_M", "0.06")
)
PHYSICAL_GRASP_Z_OFFSET_LIMIT_M = float(
    os.environ.get("OPENFORGE_STAGE1_GRASP_Z_OFFSET_LIMIT_M", "0.05")
)
ANYGRASP_SERVICE_URL = os.environ.get("ANYGRASP_SERVICE_URL", "http://127.0.0.1:8122").rstrip("/")
ANYGRASP_MAX_GRASPS = int(os.environ.get("OPENFORGE_STAGE1_ANYGRASP_MAX_GRASPS", "32"))
ANYGRASP_BATCH_TOP_K = int(os.environ.get("OPENFORGE_STAGE1_ANYGRASP_BATCH_TOP_K", "16"))
ANYGRASP_OBJECT_INPUT_MODE = os.environ.get(
    "OPENFORGE_STAGE1_ANYGRASP_OBJECT_INPUT_MODE", "segmented_object_cloud"
)
ANYGRASP_STAGE1_MAX_CENTER_Z_DELTA_M = float(
    os.environ.get("OPENFORGE_STAGE1_ANYGRASP_MAX_CENTER_Z_DELTA_M", "0.09")
)
CAMERA_PORTAL_ADDR = os.environ.get("OPENFORGE_CAMERA_PORTAL_ADDR", "127.0.0.1:8300")
GEOM_DEPTH_MAX_M = float(os.environ.get("OPENFORGE_GEOM_DEPTH_MAX_M", "2.0"))
GEOM_DEPTH_MIN_POINTS = int(os.environ.get("OPENFORGE_GEOM_DEPTH_MIN_POINTS", "300"))
GEOM_BODY_BAND_LOW = float(os.environ.get("OPENFORGE_GEOM_BODY_BAND_LOW", "0.32"))
GEOM_BODY_BAND_HIGH = float(os.environ.get("OPENFORGE_GEOM_BODY_BAND_HIGH", "0.62"))
GEOM_BODY_GRASP_FRACTION = float(
    os.environ.get("OPENFORGE_GEOM_BODY_GRASP_FRACTION", "0.48")
)
GEOM_RADIUS_PERCENTILE = float(os.environ.get("OPENFORGE_GEOM_RADIUS_PERCENTILE", "72"))
GEOM_RADIUS_MIN_M = float(os.environ.get("OPENFORGE_GEOM_RADIUS_MIN_M", "0.025"))
GEOM_RADIUS_MAX_M = float(os.environ.get("OPENFORGE_GEOM_RADIUS_MAX_M", "0.055"))
GEOM_GRASP_WIDTH_CLEARANCE_M = float(
    os.environ.get("OPENFORGE_GEOM_GRASP_WIDTH_CLEARANCE_M", "0.03")
)
GEOM_GRASP_WIDTH_MIN_M = float(os.environ.get("OPENFORGE_GEOM_GRASP_WIDTH_MIN_M", "0.06"))
GEOM_GRASP_WIDTH_MAX_M = float(os.environ.get("OPENFORGE_GEOM_GRASP_WIDTH_MAX_M", "0.10"))
GEOM_MAX_DETECTION_CENTER_DELTA_M = float(
    os.environ.get("OPENFORGE_GEOM_MAX_DETECTION_CENTER_DELTA_M", "0.18")
)
GEOM_APPROACH_OFFSETS_DEG = os.environ.get(
    "OPENFORGE_GEOM_APPROACH_OFFSETS_DEG", "0,-15,15,-30,30,-60,60,-90,90"
)
GEOM_GRASP_Z_OFFSETS_M = os.environ.get(
    "OPENFORGE_GEOM_GRASP_Z_OFFSETS_M", "0,-0.012,0.012,-0.024,0.024"
)
GEOM_GRASP_X_OFFSETS_M = os.environ.get(
    "OPENFORGE_GEOM_GRASP_X_OFFSETS_M", "0,-0.012,-0.024"
)
GEOM_WRIST_ROLL_OFFSETS_DEG = os.environ.get(
    "OPENFORGE_GEOM_WRIST_ROLL_OFFSETS_DEG", "0,-20,20,-35,35"
)
GEOM_SINGLE_PREVIEW_FALLBACK_TOP_K = int(
    os.environ.get("OPENFORGE_GEOM_SINGLE_PREVIEW_FALLBACK_TOP_K", "3")
)
GEOM_BATCH_CANDIDATE_LIMIT = int(
    os.environ.get("OPENFORGE_GEOM_BATCH_CANDIDATE_LIMIT", "120")
)
GEOM_GUARDED_PREGRASP_STANDOFF_M = float(
    os.environ.get("OPENFORGE_GEOM_GUARDED_PREGRASP_STANDOFF_M", "0.06")
)
GEOM_GUARDED_PREGRASP_STANDOFFS_M = os.environ.get(
    "OPENFORGE_GEOM_GUARDED_PREGRASP_STANDOFFS_M", "0.025,0.04,0.06,0.08"
)
GEOM_GUARDED_PREVIEW_CANDIDATE_LIMIT = int(
    os.environ.get("OPENFORGE_GEOM_GUARDED_PREVIEW_CANDIDATE_LIMIT", "30")
)
GEOM_GUARDED_PREVIEW_COMBINATION_LIMIT = int(
    os.environ.get("OPENFORGE_GEOM_GUARDED_PREVIEW_COMBINATION_LIMIT", "80")
)
GEOM_GUARDED_USE_CANDIDATE_PREGRASP = os.environ.get(
    "OPENFORGE_GEOM_GUARDED_USE_CANDIDATE_PREGRASP", ""
).strip().lower() in {"1", "true", "yes", "on"}
GEOM_REJECT_CUP_SIDE_APPROACH = os.environ.get(
    "OPENFORGE_GEOM_REJECT_CUP_SIDE_APPROACH", "1"
).strip().lower() in {"1", "true", "yes", "on"}
GEOM_CUP_SIDE_DOT_MAX = float(os.environ.get("OPENFORGE_GEOM_CUP_SIDE_DOT_MAX", "0.15"))


STAGED_PLAN = [
    {
        "stage": 1,
        "name": "dry bottle side-grasp only",
        "status": "implemented_no_motion_planning_and_gated_physical_stage1",
    },
    {
        "stage": 2,
        "name": "dry pour motion with empty/sealed bottle",
        "status": "dry_planning_supported_motion_future_requires_physical_gate",
    },
    {
        "stage": 3,
        "name": "tiny-water stationary-cup pour",
        "status": "future_requires_stage2_success_and_liquid_safety_review",
    },
    {
        "stage": 4,
        "name": "optional bimanual cup-hold after stationary-cup works",
        "status": "future_after_stationary_cup_pour_is_reliable",
    },
]

TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "pour_bottle_to_green_cup",
    "validation_mode": VALIDATION_MODE,
    "requested_stage": REQUESTED_STAGE,
    "implemented_stage": "stage1_dry_side_grasp_only",
    "staged_plan": STAGED_PLAN,
    "camera": CAMERA,
    "bottle_prompt": BOTTLE_PROMPT,
    "cup_prompt": CUP_PROMPT,
    "cup_label": CUP_LABEL,
    "dry_run": True,
    "liquid_used": False,
    "physical_motion_executed": False,
    "movement_capable_calls": [],
    "physical_attempt": None,
    "detected_bottle": None,
    "detected_cup": None,
    "perception_backend": PERCEPTION_BACKEND,
    "perception_errors": [],
    "selected_arm": None,
    "selected_arm_reason": None,
    "grasp_candidates": [],
    "selected_grasp_pose": None,
    "geometry_notes": [],
    "target_pose_estimates": {},
    "visual_artifacts": [],
    "observation_freshness": (
        "Each run_script invocation re-observes bottle/cup poses at script "
        "start when read-only perception is enabled; do not reuse coordinates "
        "across physical attempts."
    ),
    "next_no_motion_commands": {
        "offline_manual_xyz_stage1": (
            "OPENFORGE_BOTTLE_XYZ=0.55,-0.18,0.86 "
            "OPENFORGE_BOTTLE_HALF_EXTENTS=0.04,0.04,0.15 "
            "OPENFORGE_CUP_XYZ=0.62,0.08,0.79 "
            "python3 cap/saved_scripts/pour_bottle_to_green_cup.py"
        ),
        "read_only_perception_stage1": (
            "source .forge_env && "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage1 "
            "python3 cap/saved_scripts/pour_bottle_to_green_cup.py"
        ),
        "read_only_perception_stage2_planning": (
            "source .forge_env && "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage2 "
            "python3 cap/saved_scripts/pour_bottle_to_green_cup.py"
        ),
        "anygrasp_stage1_preview_only": (
            "source .forge_env && "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage1_anygrasp_preview "
            "OPENFORGE_STAGE1_ANYGRASP_MAX_GRASPS=32 "
            "OPENFORGE_STAGE1_ANYGRASP_BATCH_TOP_K=16 "
            "OPENFORGE_STAGE1_ANYGRASP_MAX_CENTER_Z_DELTA_M=0.09 "
            "uv run python run_script.py "
            "script_file=cap/saved_scripts/pour_bottle_to_green_cup.py "
            "skill_library_path=cap/saved_scripts/skill_library "
            "env.name=yam-real robot=real_yam robot.dashboard=false "
            "robot.await_exit=false robot.go_home_on_exit=false "
            "runtime.no_cameras=false recording.enabled=true "
            "debug_ui.enabled=true debug_ui.auto_open=false "
            "debug_ui.auto_exit_on_run_end=false debug_ui.host=0.0.0.0 "
            "debug_ui.port=8788"
        ),
        "geom_depth_stage1_preview_only": (
            "source .forge_env && "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage1_geom_depth_preview "
            "uv run python run_script.py "
            "script_file=cap/saved_scripts/pour_bottle_to_green_cup.py "
            "skill_library_path=cap/saved_scripts/skill_library "
            "env.name=yam-real robot=real_yam robot.dashboard=false "
            "robot.await_exit=false robot.go_home_on_exit=false "
            "runtime.no_cameras=false recording.enabled=true "
            "debug_ui.enabled=true debug_ui.auto_open=false "
            "debug_ui.auto_exit_on_run_end=false debug_ui.host=0.0.0.0 "
            "debug_ui.port=8788"
        ),
        "bounded_stage1_physical_dry_grasp": (
            "source .forge_env && "
            "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage1_geom_depth_physical "
            "uv run python run_script.py "
            "script_file=cap/saved_scripts/pour_bottle_to_green_cup.py "
            "skill_library_path=cap/saved_scripts/skill_library "
            "env.name=yam-real robot=real_yam robot.dashboard=true "
            "robot.await_exit=false robot.go_home_on_exit=false "
            "runtime.no_cameras=false recording.enabled=true "
            "debug_ui.enabled=true debug_ui.auto_open=false "
            "debug_ui.auto_exit_on_run_end=false debug_ui.host=0.0.0.0 "
            "debug_ui.port=8788"
        ),
        "bounded_stage1_geometric_physical_dry_grasp": (
            "source .forge_env && "
            "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage1_geom_depth_physical "
            "uv run python run_script.py "
            "script_file=cap/saved_scripts/pour_bottle_to_green_cup.py "
            "skill_library_path=cap/saved_scripts/skill_library "
            "env.name=yam-real robot=real_yam robot.dashboard=true "
            "robot.await_exit=false robot.go_home_on_exit=false "
            "runtime.no_cameras=false recording.enabled=true "
            "debug_ui.enabled=true debug_ui.auto_open=false "
            "debug_ui.auto_exit_on_run_end=false debug_ui.host=0.0.0.0 "
            "debug_ui.port=8788"
        ),
        "bounded_stage1_anygrasp_physical_dry_grasp": (
            "source .forge_env && "
            "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 "
            "OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http "
            "OPENFORGE_WINE_CUP_STAGE=stage1_anygrasp_physical "
            "OPENFORGE_STAGE1_ANYGRASP_MAX_CENTER_Z_DELTA_M=0.09 "
            "OPENFORGE_STAGE1_ANYGRASP_MAX_GRASPS=32 "
            "OPENFORGE_STAGE1_ANYGRASP_BATCH_TOP_K=16 "
            "uv run python run_script.py "
            "script_file=cap/saved_scripts/pour_bottle_to_green_cup.py "
            "skill_library_path=cap/saved_scripts/skill_library "
            "env.name=yam-real robot=real_yam robot.dashboard=true "
            "robot.await_exit=false robot.go_home_on_exit=false "
            "runtime.no_cameras=false recording.enabled=true "
            "debug_ui.enabled=true debug_ui.auto_open=false "
            "debug_ui.auto_exit_on_run_end=false debug_ui.host=0.0.0.0 "
            "debug_ui.port=8788"
        ),
        "offline_manual_xyz_stage2_planning": (
            "OPENFORGE_WINE_CUP_STAGE=stage2 "
            "OPENFORGE_BOTTLE_XYZ=0.55,-0.18,0.86 "
            "OPENFORGE_BOTTLE_HALF_EXTENTS=0.04,0.04,0.15 "
            "OPENFORGE_CUP_XYZ=0.62,0.08,0.79 "
            "python3 cap/saved_scripts/pour_bottle_to_green_cup.py"
        ),
    },
    "physical_run_preflight_required": [
        "bounded task and exact run count",
        "operator present at robot",
        "E-stop reachable and operator knows how to use it",
        "hands, tools, cables, loose objects, and workspace clear",
        "exact bottle and cup target unambiguous",
        "no other motion-capable script active",
        "dry/no-liquid scope confirmed",
        "OPENFORGE_ALLOW_PHYSICAL_MOTION=1 set only for the bounded physical run",
    ],
    "risk_notes": [],
    "why_stopped": "not_started",
}

STAGE_ALIASES = {
    "1": "stage1_dry_side_grasp_only",
    "stage1": "stage1_dry_side_grasp_only",
    "stage1_dry_side_grasp_only": "stage1_dry_side_grasp_only",
    "2": "stage2_dry_pour_planning_only",
    "stage2": "stage2_dry_pour_planning_only",
    "stage2_dry_pour_planning_only": "stage2_dry_pour_planning_only",
    "stage1_preview": "stage1_planner_preview_only",
    "stage1_planner_preview": "stage1_planner_preview_only",
    "stage1_planner_preview_only": "stage1_planner_preview_only",
    "stage1_geom_depth_preview": "stage1_geom_depth_preview_only",
    "stage1_geom_depth_preview_only": "stage1_geom_depth_preview_only",
    "stage1_geom_preview": "stage1_geom_depth_preview_only",
    "stage1_geometric_preview": "stage1_geom_depth_preview_only",
    "stage1_depth_geom_preview": "stage1_geom_depth_preview_only",
    "stage1_geom_depth_physical": "stage1_geom_depth_physical_dry_grasp_lift",
    "stage1_geom_depth_physical_dry_grasp_lift": "stage1_geom_depth_physical_dry_grasp_lift",
    "stage1_geom_physical": "stage1_geom_depth_physical_dry_grasp_lift",
    "stage1_geometric_physical": "stage1_geom_depth_physical_dry_grasp_lift",
    "stage1_depth_geom_physical": "stage1_geom_depth_physical_dry_grasp_lift",
    "stage1_anygrasp_preview": "stage1_anygrasp_preview_only",
    "stage1_anygrasp_preview_only": "stage1_anygrasp_preview_only",
    "stage1_anygrasp_physical": "stage1_anygrasp_physical_dry_grasp_lift",
    "stage1_anygrasp_physical_dry_grasp_lift": "stage1_anygrasp_physical_dry_grasp_lift",
    "stage1_physical": "stage1_physical_dry_grasp_lift",
    "stage1_dry_physical": "stage1_physical_dry_grasp_lift",
    "stage1_physical_dry_grasp_lift": "stage1_physical_dry_grasp_lift",
}

MOVEMENT_CAPABLE_STAGES = {
    "stage1_physical_dry_grasp_lift",
    "stage1_geom_depth_physical_dry_grasp_lift",
    "stage1_anygrasp_physical_dry_grasp_lift",
}

GEOMETRIC_DEPTH_STAGES = {
    "stage1_geom_depth_preview_only",
    "stage1_geom_depth_physical_dry_grasp_lift",
}


def get_task_info() -> dict[str, Any]:
    return dict(TASK_RESULT)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_list(value: str | None, expected: int, name: str) -> list[float] | None:
    if value is None or not value.strip():
        return None
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated floats, got {value!r}")
    return [float(part) for part in parts]


def _manual_detection(label: str, prompt: str, xyz_env: str, half_extents_env: str) -> dict[str, Any] | None:
    xyz = _parse_float_list(os.environ.get(xyz_env), 3, xyz_env)
    if xyz is None:
        return None
    half_extents = _parse_float_list(os.environ.get(half_extents_env), 3, half_extents_env) or []
    return {
        "label": label,
        "prompt": prompt,
        "score": 1.0,
        "position_3d": xyz,
        "half_extents": half_extents,
        "rpy": [],
        "quaternion_xyzw": [],
        "source": "manual_env",
    }


def _normalize_stage(stage: str) -> str | None:
    return STAGE_ALIASES.get(str(stage).strip().lower())


def _namespace_tool(name: str):
    try:
        import skill_library.namespace as namespace  # type: ignore
    except Exception:
        return None
    return getattr(namespace, name, None)


def _required_tool(name: str):
    fn = _namespace_tool(name)
    if not callable(fn):
        raise RuntimeError(
            f"Required YAM tool {name!r} is unavailable. Run through run_script.py "
            "with skill_library_path=cap/saved_scripts/skill_library."
        )
    return fn


def _det_value(det: Any, name: str, default: Any = None) -> Any:
    if isinstance(det, dict):
        return det.get(name, default)
    return getattr(det, name, default)


def _first_detection(det_map: Any, prompt: str) -> Any | None:
    if isinstance(det_map, dict):
        dets = det_map.get(prompt) or []
        if not dets:
            for maybe_dets in det_map.values():
                if maybe_dets:
                    dets = maybe_dets
                    break
    else:
        dets = det_map or []
    return dets[0] if dets else None


def _serialize_detection(det: Any, prompt: str, source: str) -> dict[str, Any] | None:
    if det is None:
        return None
    position = _det_value(det, "position_3d")
    if position is None:
        position = _det_value(det, "position")
    if position is None:
        return None
    return {
        "label": _det_value(det, "label", prompt),
        "prompt": prompt,
        "score": _det_value(det, "score"),
        "box_2d": _det_value(det, "box_2d", _det_value(det, "bbox", [])),
        "position_3d": [float(x) for x in list(position)[:3]],
        "half_extents": [float(x) for x in (_det_value(det, "half_extents", []) or [])[:3]],
        "rpy": [float(x) for x in (_det_value(det, "rpy", []) or [])[:3]],
        "quaternion_xyzw": [
            float(x) for x in (_det_value(det, "quaternion_xyzw", []) or [])[:4]
        ],
        "source": source,
    }


def _json_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _depth_to_npy_base64(depth: Any) -> str:
    import base64
    import io

    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.asarray(depth, dtype=np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jsonify_extrinsics_for_bundlesdf(extrinsics: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    rotation = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    position = np.asarray(extrinsics["position"], dtype=np.float64).reshape(-1)
    if extrinsics.get("needs_optical_flip", False):
        rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
    return {
        "position": [float(x) for x in position.tolist()],
        "rotation": [float(x) for x in rotation.reshape(-1).tolist()],
        "needs_optical_flip": False,
    }


def _bundlesdf_single_frame_payload(
    prompt: str,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": prompt, "camera": CAMERA, "debug_level": 0}
    if snapshot is not None:
        payload.update(
            {
                "image_base64": _rgb_to_png_base64(snapshot["rgb"]),
                "depth_base64": _depth_to_npy_base64(snapshot["depth"]),
                "intrinsics": snapshot["intrinsics"],
                "extrinsics": _jsonify_extrinsics_for_bundlesdf(snapshot["extrinsics"]),
            }
        )
    return payload


def _detect_one_bundlesdf_http(
    prompt: str,
    snapshot: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    url = f"{BUNDLESDF_SERVICE_URL}/single_frame_pose"
    payload = _bundlesdf_single_frame_payload(prompt, snapshot)
    data = _json_post(url, payload, READ_ONLY_PERCEPTION_TIMEOUT_S)
    source = (
        "bundlesdf_http_single_frame_snapshot"
        if snapshot is not None
        else "bundlesdf_http_single_frame"
    )
    det = _serialize_detection(data, prompt, source)
    if det is not None:
        return det, None
    score = data.get("score")
    bbox = data.get("bbox")
    return None, f"{prompt!r}: no position_3d from {url}; score={score}, bbox={bbox}"


def _detect_bundlesdf_http() -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors = []
    bottle = None
    cup = None
    snapshot = None
    try:
        snapshot = _camera_portal_snapshot(CAMERA)
    except Exception as exc:
        errors.append(f"camera portal snapshot unavailable: {type(exc).__name__}: {exc}")
    try:
        bottle, err = _detect_one_bundlesdf_http(BOTTLE_PROMPT, snapshot)
        if err:
            errors.append(err)
    except Exception as exc:
        errors.append(f"{BOTTLE_PROMPT!r}: {type(exc).__name__}: {exc}")
    try:
        cup, err = _detect_one_bundlesdf_http(CUP_PROMPT, snapshot)
        if err:
            errors.append(err)
    except Exception as exc:
        errors.append(f"{CUP_PROMPT!r}: {type(exc).__name__}: {exc}")
    return bottle, cup, errors


def _detect_namespace() -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors = []
    detect_objects_oneshot = _namespace_tool("detect_objects_oneshot")
    if not callable(detect_objects_oneshot):
        return None, None, ["detect_objects_oneshot_unavailable"]

    try:
        det_map = detect_objects_oneshot([BOTTLE_PROMPT, CUP_PROMPT], camera=CAMERA)
    except TypeError:
        bottle_map = detect_objects_oneshot(BOTTLE_PROMPT, camera=CAMERA)
        cup_map = detect_objects_oneshot(CUP_PROMPT, camera=CAMERA)
        bottle = _serialize_detection(
            _first_detection(bottle_map, BOTTLE_PROMPT), BOTTLE_PROMPT, "perception"
        )
        cup = _serialize_detection(_first_detection(cup_map, CUP_PROMPT), CUP_PROMPT, "perception")
        return bottle, cup, errors
    except Exception as exc:
        return None, None, [f"namespace perception failed: {type(exc).__name__}: {exc}"]
    bottle = _serialize_detection(_first_detection(det_map, BOTTLE_PROMPT), BOTTLE_PROMPT, "perception")
    cup = _serialize_detection(_first_detection(det_map, CUP_PROMPT), CUP_PROMPT, "perception")
    return bottle, cup, errors


def _detect_read_only() -> tuple[dict[str, Any] | None, dict[str, Any] | None, str, list[str]]:
    manual_bottle = _manual_detection(
        "bottle", BOTTLE_PROMPT, "OPENFORGE_BOTTLE_XYZ", "OPENFORGE_BOTTLE_HALF_EXTENTS"
    )
    manual_cup = _manual_detection(
        CUP_LABEL, CUP_PROMPT, "OPENFORGE_CUP_XYZ", "OPENFORGE_CUP_HALF_EXTENTS"
    )
    if manual_bottle or manual_cup:
        return manual_bottle, manual_cup, "manual_env", []

    if not ENABLE_READ_ONLY_PERCEPTION:
        return None, None, "read_only_perception_disabled", []

    if PERCEPTION_BACKEND in {"bundlesdf_http", "http", "single_frame_pose"}:
        bottle, cup, errors = _detect_bundlesdf_http()
        return bottle, cup, "bundlesdf_http_single_frame", errors
    if PERCEPTION_BACKEND in {"namespace", "detect_objects_oneshot"}:
        bottle, cup, errors = _detect_namespace()
        return bottle, cup, "namespace_detect_objects_oneshot", errors
    return None, None, f"unsupported_perception_backend:{PERCEPTION_BACKEND}", []


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_angle_deg(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def _round_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def _artifact_vis_dir() -> Path | None:
    try:
        from cap.agent.tools import _artifact_log

        artifact_dir = getattr(_artifact_log, "_artifact_dir", None)
    except Exception:
        return None
    if artifact_dir is None:
        return None
    path = Path(artifact_dir) / "wine_cup"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _first_bundlesdf_preview_frame(camera: str) -> Any | None:
    import io
    import urllib.request

    try:
        from PIL import Image
    except Exception:
        return None

    url = f"{BUNDLESDF_SERVICE_URL}/preview/{camera}"
    chunks: list[bytes] = []
    deadline = time.monotonic() + 3.0
    try:
        with urllib.request.urlopen(url, timeout=3.0) as response:
            while time.monotonic() < deadline:
                chunk = response.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                data = b"".join(chunks)
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9", start + 2)
                if start >= 0 and end >= 0:
                    return Image.open(io.BytesIO(data[start : end + 2])).convert("RGB")
                if len(data) > 3_000_000:
                    break
    except Exception:
        return None
    return None


def _box_xyxy(det: dict[str, Any] | None, image_size: tuple[int, int]) -> list[int] | None:
    if not det:
        return None
    box = det.get("box_2d") or det.get("bbox")
    if not box or len(box) != 4:
        return None
    width, height = image_size
    x0, y0, a, b = [float(v) for v in box]
    if a <= x0 or b <= y0:
        x1 = x0 + a
        y1 = y0 + b
    else:
        x1 = a
        y1 = b
    x0 = _clip(x0, 0, width - 1)
    y0 = _clip(y0, 0, height - 1)
    x1 = _clip(x1, 0, width - 1)
    y1 = _clip(y1, 0, height - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def _draw_text_block(draw: Any, xy: tuple[int, int], lines: list[str], fill: str, font: Any) -> None:
    x, y = xy
    for line in lines:
        draw.text((x + 1, y + 1), line, fill="black", font=font)
        draw.text((x, y), line, fill=fill, font=font)
        y += 16


def _save_detection_plan_overlay(
    *,
    stage: str,
    bottle: dict[str, Any] | None,
    cup: dict[str, Any] | None,
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    estimates: dict[str, Any] | None,
) -> list[str]:
    vis_dir = _artifact_vis_dir()
    if vis_dir is None:
        return []
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        TASK_RESULT["risk_notes"].append(
            f"visual artifact skipped: PIL unavailable ({type(exc).__name__})"
        )
        return []

    image = _first_bundlesdf_preview_frame(CAMERA)
    preview_note = "BundleSDF preview frame"
    if image is None:
        image = Image.new("RGB", (960, 540), color=(24, 28, 34))
        preview_note = "BundleSDF preview unavailable; text overlay only"

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
        )
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for det, color, name in (
        (bottle, "lime", "bottle"),
        (cup, "cyan", CUP_LABEL),
    ):
        box = _box_xyxy(det, image.size)
        if box is not None:
            draw.rectangle(box, outline=color, width=3)
            label_xy = (box[0], max(0, box[1] - 18))
        else:
            label_xy = (10, 54 if name == "bottle" else 76)
        pos = det.get("position_3d") if det else None
        score = det.get("score") if det else None
        label = f"{name}: xyz={_round_list(pos, 3) if pos else None} score={score}"
        draw.text((label_xy[0] + 1, label_xy[1] + 1), label, fill="black", font=small_font)
        draw.text(label_xy, label, fill=color, font=small_font)

    header_lines = [
        f"wine/cup {stage}",
        f"{preview_note}; camera={CAMERA}",
        f"detection_source={TASK_RESULT.get('detection_source')}",
        "fresh observation at run start; no stale coordinates reused",
    ]
    for index, error in enumerate(TASK_RESULT.get("perception_errors", [])[:4], start=1):
        header_lines.append(f"perception_error_{index}={error}")
    if selected:
        header_lines.extend(
            [
                f"selected_arm={selected.get('arm')} type={selected.get('type')}",
                f"selected_position={_round_list(selected.get('position', []), 4)}",
                f"selected_rpy={_round_list(selected.get('rpy', []), 2)}",
            ]
        )
    if estimates:
        grasp = estimates.get("stage1_physical_commanded_grasp_pose") or estimates.get(
            "stage1_nominal_grasp_pose"
        )
        lift = estimates.get("stage1_lift_pose")
        if grasp:
            header_lines.append(f"commanded_grasp={_round_list(grasp[:3], 4)}")
        if lift:
            header_lines.append(f"lift_pose={_round_list(lift[:3], 4)}")
    header_lines.append(f"candidate_count={len(candidates)}")
    _draw_text_block(draw, (10, 10), header_lines, "white", font)

    stamp = time.strftime("%H%M%S")
    path = vis_dir / f"{stamp}_{stage}_detection_plan_overlay.png"
    image.save(path)
    rel = path
    try:
        artifact_root = Path(vis_dir).parents[1]
        rel = path.relative_to(artifact_root)
    except Exception:
        pass
    print(f"[pour_bottle_to_green_cup] Saved visual artifact: {rel}")
    return [str(rel)]


def _rgb_to_png_base64(rgb: Any) -> str:
    import base64
    import io

    from PIL import Image

    image = Image.fromarray(rgb.astype("uint8"), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_np_array_base64(value: str) -> Any:
    import base64
    import io

    import numpy as np

    return np.load(io.BytesIO(base64.b64decode(value)))


def _static_top_camera_extrinsics() -> dict[str, Any]:
    import os

    import numpy as np

    from robot.models.station.paths import get_top_camera_frame, needs_optical_flip
    from robot.yam.kinematics import YamKinematics

    frame_name = os.environ.get("CAP_TOP_CAMERA_FRAME", get_top_camera_frame())
    kin = YamKinematics()
    T = kin.configuration.get_transform_frame_to_world(frame_name, "body")
    rot = T.rotation()
    if hasattr(rot, "as_matrix"):
        rot_mat = rot.as_matrix()
    else:
        maybe_matrix = getattr(rot, "matrix", rot)
        rot_mat = maybe_matrix() if callable(maybe_matrix) else maybe_matrix
    return {
        "position": [float(x) for x in T.translation().tolist()],
        "rotation": np.asarray(rot_mat, dtype=np.float64).reshape(3, 3).tolist(),
        "needs_optical_flip": needs_optical_flip("top"),
        "source": "station_model_static_top_camera",
    }


def _segment_bundlesdf_http(prompt: str, *, image_base64: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": prompt, "camera": CAMERA}
    if image_base64:
        payload["image_base64"] = image_base64
    return _json_post(
        f"{BUNDLESDF_SERVICE_URL}/segment",
        payload,
        READ_ONLY_PERCEPTION_TIMEOUT_S,
    )


def _camera_portal_snapshot(camera: str) -> dict[str, Any]:
    import numpy as np
    import portal

    client = portal.Client(CAMERA_PORTAL_ADDR)
    rgb_raw = client.get_camera_image(camera).result()
    depth_raw = client.get_camera_depth(camera).result()
    intrinsics_raw = client.get_camera_intrinsics(camera).result()
    try:
        extrinsics = client.get_camera_extrinsics(camera).result()
    except Exception as exc:
        if str(camera).strip().lower() != "top":
            raise
        extrinsics = _static_top_camera_extrinsics()
        extrinsics["portal_extrinsics_error"] = f"{type(exc).__name__}: {exc}"

    rgb = np.asarray(rgb_raw)
    depth = np.asarray(depth_raw, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size < 100:
        raise RuntimeError(f"Camera portal returned invalid RGB frame for {camera!r}: {rgb.shape}")
    if depth.ndim != 2 or depth.size < 100:
        raise RuntimeError(
            f"Camera portal returned invalid depth frame for {camera!r}: {depth.shape}"
        )
    if float(np.nanmedian(depth)) > 10.0:
        depth = depth / 1000.0
    intrinsics = [float(x) for x in intrinsics_raw]
    if len(intrinsics) != 4 or all(abs(x) < 1e-9 for x in intrinsics):
        raise RuntimeError(f"Camera portal returned invalid intrinsics for {camera!r}: {intrinsics}")
    return {
        "rgb": np.ascontiguousarray(rgb[:, :, :3]),
        "depth": np.ascontiguousarray(depth),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }


def _camera_to_world_from_extrinsics(extrinsics: dict[str, Any]) -> Any:
    import numpy as np

    rotation = np.asarray(extrinsics["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(extrinsics["position"], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    if extrinsics.get("needs_optical_flip", True):
        transform[:3, :3] = rotation @ np.diag([-1.0, -1.0, 1.0])
    else:
        transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _world_points_from_mask_depth(
    *,
    mask: Any,
    depth: Any,
    intrinsics: list[float],
    extrinsics: dict[str, Any],
) -> tuple[Any, Any, Any]:
    import numpy as np

    if mask.shape[:2] != depth.shape[:2]:
        raise RuntimeError(
            "SAM3 mask and portal depth shape mismatch: "
            f"mask={mask.shape[:2]} depth={depth.shape[:2]}"
        )

    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) < GEOM_DEPTH_MIN_POINTS:
        raise RuntimeError(
            f"SAM3 mask has only {len(xs)} pixels; need at least {GEOM_DEPTH_MIN_POINTS}"
        )

    z = np.asarray(depth[ys, xs], dtype=np.float64)
    valid = np.isfinite(z) & (z > 0.05) & (z < GEOM_DEPTH_MAX_M)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid]
    if len(z) < GEOM_DEPTH_MIN_POINTS:
        raise RuntimeError(
            f"Only {len(z)} masked depth pixels survived depth filtering; "
            f"need at least {GEOM_DEPTH_MIN_POINTS}"
        )

    fx, fy, cx, cy = [float(v) for v in intrinsics]
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    points_cam = np.stack([x_cam, y_cam, z], axis=1)
    T_cam_world = _camera_to_world_from_extrinsics(extrinsics)
    points_world = (T_cam_world[:3, :3] @ points_cam.T).T + T_cam_world[:3, 3]
    return points_world, xs, ys


def _estimate_bottle_depth_geometry(
    bottle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    snapshot = _camera_portal_snapshot(CAMERA)
    segment = _segment_bundlesdf_http(
        BOTTLE_PROMPT,
        image_base64=_rgb_to_png_base64(snapshot["rgb"]),
    )
    mask = _decode_np_array_base64(segment["mask_b64"]).astype(np.uint8)
    points_world, mask_xs, mask_ys = _world_points_from_mask_depth(
        mask=mask,
        depth=snapshot["depth"],
        intrinsics=snapshot["intrinsics"],
        extrinsics=snapshot["extrinsics"],
    )

    z_low, z_high = np.percentile(points_world[:, 2], [5.0, 95.0])
    z_keep = (points_world[:, 2] >= z_low) & (points_world[:, 2] <= z_high)
    trimmed = points_world[z_keep]
    if len(trimmed) < GEOM_DEPTH_MIN_POINTS:
        trimmed = points_world
    bottom_z = float(np.percentile(trimmed[:, 2], 5.0))
    top_z = float(np.percentile(trimmed[:, 2], 95.0))
    height_m = max(0.001, top_z - bottom_z)
    body_low_fraction = _clip(GEOM_BODY_BAND_LOW, 0.05, 0.90)
    body_high_fraction = _clip(GEOM_BODY_BAND_HIGH, body_low_fraction + 0.05, 0.95)
    body_z_low = bottom_z + height_m * body_low_fraction
    body_z_high = bottom_z + height_m * body_high_fraction
    body_mask = (trimmed[:, 2] >= body_z_low) & (trimmed[:, 2] <= body_z_high)
    body_points = trimmed[body_mask]
    if len(body_points) < max(80, GEOM_DEPTH_MIN_POINTS // 3):
        body_points = trimmed
        body_z_low = float(np.percentile(body_points[:, 2], 35.0))
        body_z_high = float(np.percentile(body_points[:, 2], 65.0))

    center_xy = np.median(body_points[:, :2], axis=0)
    center_z = float((body_z_low + body_z_high) / 2.0)
    center_world = [float(center_xy[0]), float(center_xy[1]), center_z]
    grasp_fraction = _clip(GEOM_BODY_GRASP_FRACTION, body_low_fraction, body_high_fraction)
    grasp_z = bottom_z + height_m * grasp_fraction + GRASP_Z_OFFSET_M
    grasp_z = _clip(grasp_z, body_z_low, body_z_high)
    radial = np.linalg.norm(body_points[:, :2] - center_xy, axis=1)
    radius_raw = float(np.percentile(radial, GEOM_RADIUS_PERCENTILE))
    radius_m = _clip(radius_raw, GEOM_RADIUS_MIN_M, GEOM_RADIUS_MAX_M)
    gripper_width_m = _clip(
        2.0 * radius_m + GEOM_GRASP_WIDTH_CLEARANCE_M,
        GEOM_GRASP_WIDTH_MIN_M,
        GEOM_GRASP_WIDTH_MAX_M,
    )

    detection_xyz = [float(x) for x in bottle["position_3d"][:3]]
    detection_delta_m = float(
        math.sqrt(sum((center_world[i] - detection_xyz[i]) ** 2 for i in range(3)))
    )
    if detection_delta_m > GEOM_MAX_DETECTION_CENTER_DELTA_M:
        raise RuntimeError(
            "Depth-derived bottle center is too far from BundleSDF pose: "
            f"{detection_delta_m:.3f}m > {GEOM_MAX_DETECTION_CENTER_DELTA_M:.3f}m"
        )

    bbox_xywh = [int(x) for x in segment.get("bbox_xywh", [])[:4]]
    center_pixel_uv = [
        float(np.median(mask_xs)),
        float(np.median(mask_ys)),
    ]
    geometry = {
        "source": "bundlesdf_sam3_depth_geometry",
        "camera": CAMERA,
        "camera_portal_addr": CAMERA_PORTAL_ADDR,
        "segment_score": segment.get("score"),
        "mask_area_px": int(segment.get("mask_area", int(mask.sum()))),
        "bbox_xywh": bbox_xywh,
        "valid_depth_points": int(len(points_world)),
        "trimmed_depth_points": int(len(trimmed)),
        "body_band_points": int(len(body_points)),
        "axis_world": [0.0, 0.0, 1.0],
        "axis_status": "upright_world_z_assumption_for_stage1_side_grasp",
        "bottom_z_m": round(bottom_z, 4),
        "top_z_m": round(top_z, 4),
        "height_m": round(height_m, 4),
        "body_band_z_m": [round(body_z_low, 4), round(body_z_high, 4)],
        "body_band_fraction": [round(body_low_fraction, 3), round(body_high_fraction, 3)],
        "center_world": _round_list(center_world, 4),
        "grasp_position_world": _round_list([center_world[0], center_world[1], grasp_z], 4),
        "radius_raw_m": round(radius_raw, 4),
        "radius_m": round(radius_m, 4),
        "gripper_width_m": round(gripper_width_m, 4),
        "detection_center_world": _round_list(detection_xyz, 4),
        "detection_center_delta_m": round(detection_delta_m, 4),
        "center_pixel_uv": _round_list(center_pixel_uv, 1),
        "depth_units": "meters",
    }
    overlay_source = {
        "rgb": snapshot["rgb"],
        "mask": mask,
    }
    return geometry, overlay_source


def _parse_float_offsets(raw: str, fallback: list[float]) -> list[float]:
    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values or list(fallback)


def _parse_angle_offsets(raw: str) -> list[float]:
    return _parse_float_offsets(raw, [0.0, -15.0, 15.0, -30.0, 30.0])


def _display_rpy_from_rotation_matrix(rotation_matrix: Any) -> list[float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    euler_xyz = Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64)).as_euler(
        "xyz",
        degrees=True,
    )
    display = np.array(
        [euler_xyz[1], -euler_xyz[0], -euler_xyz[2] - 90.0],
        dtype=np.float64,
    )
    display = (display + 180.0) % 360.0 - 180.0
    return [float(x) for x in display]


def _side_grasp_display_rpy_from_approach(
    approach_dir: list[float],
    wrist_roll_deg: float = 0.0,
) -> tuple[list[float], dict[str, list[float]]]:
    import numpy as np

    approach = np.asarray(approach_dir, dtype=np.float64)
    norm = float(np.linalg.norm(approach))
    if norm < 1e-6:
        raise RuntimeError(f"Invalid side-grasp approach direction: {approach_dir!r}")
    z_axis = approach / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(world_up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    if abs(float(wrist_roll_deg)) > 1e-6:
        roll_rad = math.radians(float(wrist_roll_deg))
        cos_t = math.cos(roll_rad)
        sin_t = math.sin(roll_rad)
        x_base = x_axis
        y_base = y_axis
        x_axis = cos_t * x_base + sin_t * y_base
        y_axis = -sin_t * x_base + cos_t * y_base
        x_axis = x_axis / float(np.linalg.norm(x_axis))
        y_axis = y_axis / float(np.linalg.norm(y_axis))
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return (
        _display_rpy_from_rotation_matrix(rotation_matrix),
        {
            "local_x_opening_axis": _round_list(x_axis.tolist(), 4),
            "local_y_height_axis": _round_list(y_axis.tolist(), 4),
            "local_z_approach_axis": _round_list(z_axis.tolist(), 4),
        },
    )


def _geometric_depth_candidate_for_angle(
    geometry: dict[str, Any],
    approach_angle_deg: float,
    arm: str,
    score: float,
    z_offset_m: float,
    x_offset_m: float,
    wrist_roll_deg: float,
) -> dict[str, Any]:
    grasp_xyz = [float(x) for x in geometry["grasp_position_world"]]
    radius_m = float(geometry.get("radius_m", GEOM_RADIUS_MIN_M) or GEOM_RADIUS_MIN_M)
    max_center_offset = max(0.0, min(radius_m * 0.65, 0.026))
    requested_x_offset = float(x_offset_m)
    applied_x_offset = _clip(requested_x_offset, -max_center_offset, max_center_offset)
    grasp_xyz[0] += applied_x_offset
    body_band_z = geometry.get("body_band_z_m") or []
    if len(body_band_z) == 2:
        requested_z = grasp_xyz[2] + float(z_offset_m)
        grasp_xyz[2] = _clip(requested_z, float(body_band_z[0]), float(body_band_z[1]))
    actual_z_offset_m = grasp_xyz[2] - float(geometry["grasp_position_world"][2])
    angle_rad = math.radians(approach_angle_deg)
    approach_dir = [math.cos(angle_rad), math.sin(angle_rad), 0.0]
    cup_side_dot: float | None = None
    cup_clearance_status = "cup_position_unavailable"
    cup_position = geometry.get("cup_position_world")
    if isinstance(cup_position, list) and len(cup_position) >= 2:
        cup_vec = [
            float(cup_position[0]) - float(grasp_xyz[0]),
            float(cup_position[1]) - float(grasp_xyz[1]),
        ]
        cup_norm = math.sqrt(cup_vec[0] * cup_vec[0] + cup_vec[1] * cup_vec[1])
        if cup_norm > 1e-6:
            pregrasp_side = [-approach_dir[0], -approach_dir[1]]
            cup_side_dot = (
                pregrasp_side[0] * cup_vec[0] + pregrasp_side[1] * cup_vec[1]
            ) / cup_norm
            if cup_side_dot > GEOM_CUP_SIDE_DOT_MAX:
                cup_clearance_status = "reject_pregrasp_side_points_toward_cup"
            else:
                cup_clearance_status = "ok_pregrasp_side_away_from_cup"
    pregrasp = [
        grasp_xyz[0] - approach_dir[0] * PREGRASP_STANDOFF_M,
        grasp_xyz[1] - approach_dir[1] * PREGRASP_STANDOFF_M,
        grasp_xyz[2],
    ]
    rpy, axes = _side_grasp_display_rpy_from_approach(
        approach_dir,
        wrist_roll_deg=wrist_roll_deg,
    )
    return {
        "arm": arm,
        "type": "depth_body_midline_side_grasp",
        "source": "bundlesdf_sam3_depth_geometry",
        "position": _round_list(grasp_xyz, 4),
        "rpy": _round_list(rpy, 3),
        "score": round(float(score), 3),
        "width": round(float(geometry["gripper_width_m"]), 4),
        "estimated_radius_m": geometry.get("radius_m"),
        "nominal_position": _round_list(geometry["grasp_position_world"], 4),
        "z_offset_m": round(float(actual_z_offset_m), 4),
        "x_offset_m": round(float(applied_x_offset), 4),
        "x_offset_requested_m": round(float(requested_x_offset), 4),
        "x_offset_limit_m": round(float(max_center_offset), 4),
        "approach_direction_world": _round_list(approach_dir, 4),
        "gripper_local_axes_world": axes,
        "pregrasp_position": _round_list(pregrasp, 4),
        "approach_angle_deg": round(float(approach_angle_deg), 3),
        "wrist_roll_deg": round(float(wrist_roll_deg), 3),
        "cup_side_dot": round(float(cup_side_dot), 4) if cup_side_dot is not None else None,
        "cup_clearance_status": cup_clearance_status,
        "orientation_status": (
            "geometric_axes: local +Z follows approach, local +X spans bottle body, "
            "local +Y stays near upright with bounded wrist-roll variants"
        ),
    }


def _geometric_depth_side_grasp_candidates(
    geometry: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    preferred_arm, reason = _select_arm([float(x) for x in geometry["center_world"]])
    alternate_arm = "left" if preferred_arm == "right" else "right"
    base_angles = {"right": 90.0, "left": -90.0}
    offsets = _parse_angle_offsets(GEOM_APPROACH_OFFSETS_DEG)
    z_offsets = _parse_float_offsets(GEOM_GRASP_Z_OFFSETS_M, [0.0, -0.012, 0.012])
    x_offsets = _parse_float_offsets(GEOM_GRASP_X_OFFSETS_M, [0.0, -0.012])
    wrist_roll_offsets = _parse_angle_offsets(GEOM_WRIST_ROLL_OFFSETS_DEG)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, float, tuple[float, float, float]]] = set()
    for arm, base_score in ((preferred_arm, 1.0), (alternate_arm, 0.82)):
        for offset in offsets:
            angle = base_angles[arm] + offset
            angle_penalty = min(abs(offset) * 0.004, 0.18)
            for wrist_roll in wrist_roll_offsets:
                wrist_penalty = min(abs(wrist_roll) * 0.002, 0.10)
                for x_offset in x_offsets:
                    x_penalty = min(abs(x_offset) * 3.0, 0.10)
                    for z_offset in z_offsets:
                        z_penalty = min(abs(z_offset) * 4.0, 0.12)
                        score = (
                            base_score
                            - angle_penalty
                            - wrist_penalty
                            - x_penalty
                            - z_penalty
                        )
                        candidate = _geometric_depth_candidate_for_angle(
                            geometry,
                            angle,
                            arm,
                            score,
                            z_offset,
                            x_offset,
                            wrist_roll,
                        )
                        if (
                            GEOM_REJECT_CUP_SIDE_APPROACH
                            and candidate.get("cup_clearance_status")
                            == "reject_pregrasp_side_points_toward_cup"
                        ):
                            continue
                        dedupe_key = (
                            arm,
                            round(float(angle), 3),
                            round(float(wrist_roll), 3),
                            round(float(candidate.get("x_offset_m", 0.0)), 4),
                            tuple(float(x) for x in candidate["position"]),
                        )
                        if dedupe_key in seen:
                            continue
                        seen.add(dedupe_key)
                        candidates.append(candidate)
    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return preferred_arm, f"{reason}; geometric candidates generated for both arms", candidates


def _save_geometric_depth_overlay(
    *,
    stage: str,
    geometry: dict[str, Any],
    overlay_source: dict[str, Any] | None,
    bottle: dict[str, Any] | None,
    cup: dict[str, Any] | None,
    selected: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    preview: dict[str, Any] | None,
) -> list[str]:
    vis_dir = _artifact_vis_dir()
    if vis_dir is None:
        return []
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        TASK_RESULT["risk_notes"].append(
            f"geometric visual artifact skipped: image deps unavailable ({type(exc).__name__})"
        )
        return []

    image = None
    mask = None
    if overlay_source:
        try:
            image = Image.fromarray(overlay_source["rgb"].astype("uint8"), mode="RGB")
            mask = np.asarray(overlay_source.get("mask"), dtype=np.uint8)
        except Exception:
            image = None
            mask = None
    if image is None:
        image = _first_bundlesdf_preview_frame(CAMERA)
    if image is None:
        image = Image.new("RGB", (960, 540), color=(24, 28, 34))

    draw = ImageDraw.Draw(image)
    if mask is not None and mask.shape[:2] == (image.height, image.width):
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_arr = np.asarray(overlay).copy()
        overlay_arr[mask.astype(bool)] = [0, 255, 80, 75]
        image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay_arr)).convert("RGB")
        draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
        )
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    bbox = geometry.get("bbox_xywh") or []
    if len(bbox) == 4:
        x, y, w, h = [int(v) for v in bbox]
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
    uv = geometry.get("center_pixel_uv") or []
    if len(uv) == 2:
        u, v = [float(x) for x in uv]
        draw.ellipse([u - 6, v - 6, u + 6, v + 6], outline="yellow", width=3)

    for det, color, name in (
        (bottle, "lime", "bottle_pose"),
        (cup, "cyan", CUP_LABEL),
    ):
        box = _box_xyxy(det, image.size)
        if box is not None:
            draw.rectangle(box, outline=color, width=2)
            label_xy = (box[0], max(0, box[1] - 16))
            draw.text(label_xy, name, fill=color, font=small_font)

    header_lines = [
        f"wine/cup {stage}",
        "primary grasp source: BundleSDF/SAM3/depth geometry",
        f"center={geometry.get('center_world')} radius={geometry.get('radius_m')}m width={geometry.get('gripper_width_m')}m",
        f"body_z={geometry.get('body_band_z_m')} points={geometry.get('body_band_points')}",
        f"detect_delta={geometry.get('detection_center_delta_m')}m candidates={len(candidates)}",
        "fresh portal depth + SAM3 mask in this run; no stale coordinates reused",
    ]
    if preview:
        header_lines.append(f"curobo_preview_success={preview.get('success')}")
    if selected:
        header_lines.extend(
            [
                f"selected_arm={selected.get('arm')} angle={selected.get('approach_angle_deg')}",
                f"selected_position={selected.get('position')} z_offset={selected.get('z_offset_m')}",
                f"selected_rpy={selected.get('rpy')}",
            ]
        )
    _draw_text_block(draw, (10, 10), header_lines, "white", font)

    stamp = time.strftime("%H%M%S")
    path = vis_dir / f"{stamp}_{stage}_geometric_depth_overlay.png"
    image.save(path)
    rel = path
    try:
        artifact_root = Path(vis_dir).parents[1]
        rel = path.relative_to(artifact_root)
    except Exception:
        pass
    print(f"[pour_bottle_to_green_cup] Saved visual artifact: {rel}")
    return [str(rel)]


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "status": getattr(result, "status", None),
        "executed": bool(getattr(result, "executed", False)),
        "side": getattr(result, "side", None),
        "trajectory_steps": int(getattr(result, "trajectory_steps", 0) or 0),
        "trajectory_cache_key": getattr(result, "trajectory_cache_key", None),
        "final_pos_error_m": getattr(result, "final_pos_error_m", None),
        "final_rot_error_deg": getattr(result, "final_rot_error_deg", None),
        "reason": getattr(result, "reason", ""),
    }


def _obj_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _gripper_pos(state: Any, side: str) -> float:
    value = getattr(state, f"{side}_gripper_pos", 0.0)
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else 0.0
    return float(value)


def _select_arm(bottle_xyz: list[float]) -> tuple[str, str]:
    y = float(bottle_xyz[1])
    if y > 0.04:
        return "left", "bottle y is on YAM left side"
    if y < -0.04:
        return "right", "bottle y is on YAM right side"
    return "right", "bottle near centerline; choose right arm as first dry-plan default"


def _bottle_radius(half_extents: list[float]) -> float:
    if len(half_extents) >= 2:
        lateral = sorted(abs(float(x)) for x in half_extents[:2])
        if lateral[0] > 0.0:
            return lateral[0]
    return DEFAULT_BOTTLE_RADIUS_M


def _grasp_z(center_z: float, half_extents: list[float]) -> float:
    if len(half_extents) >= 3 and abs(float(half_extents[2])) > 0.0:
        half_height = abs(float(half_extents[2]))
        bottom_z = float(center_z) - half_height
        body_fraction = _clip(BODY_GRASP_FRACTION, 0.25, 0.65)
        return bottom_z + 2.0 * half_height * body_fraction + GRASP_Z_OFFSET_M
    return float(center_z) + GRASP_Z_OFFSET_M


def _candidate_for_angle(
    grasp_xyz: list[float],
    approach_angle_deg: float,
    arm: str,
    radius_m: float,
    score: float,
) -> dict[str, Any]:
    angle_rad = math.radians(approach_angle_deg)
    approach_dir = [math.cos(angle_rad), math.sin(angle_rad), 0.0]
    pregrasp = [
        float(grasp_xyz[0]) - approach_dir[0] * PREGRASP_STANDOFF_M,
        float(grasp_xyz[1]) - approach_dir[1] * PREGRASP_STANDOFF_M,
        float(grasp_xyz[2]),
    ]
    jaw_axis_yaw = _normalize_angle_deg(approach_angle_deg + 90.0)
    width = _clip(2.0 * radius_m + GRASP_CLEARANCE_M, 0.045, 0.085)
    return {
        "arm": arm,
        "type": "mid_body_side_grasp",
        "position": _round_list(grasp_xyz),
        "rpy": [0.0, 90.0, round(jaw_axis_yaw, 3)],
        "score": round(float(score), 3),
        "width": round(width, 4),
        "approach_direction_world": _round_list(approach_dir),
        "pregrasp_position": _round_list(pregrasp),
        "approach_angle_deg": round(float(approach_angle_deg), 3),
        "orientation_status": "tentative_requires_no_motion_planner_check",
    }


def _side_grasp_candidates(bottle: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    bottle_xyz = [float(x) for x in bottle["position_3d"][:3]]
    half_extents = [float(x) for x in bottle.get("half_extents", [])[:3]]
    arm, reason = _select_arm(bottle_xyz)
    radius_m = _bottle_radius(half_extents)
    grasp_xyz = [bottle_xyz[0], bottle_xyz[1], _grasp_z(bottle_xyz[2], half_extents)]

    primary_angle = 90.0 if arm == "right" else -90.0
    alternate_arm = "left" if arm == "right" else "right"
    alternate_angle = -90.0 if arm == "right" else 90.0
    candidates = [
        _candidate_for_angle(grasp_xyz, primary_angle, arm, radius_m, 1.0),
        _candidate_for_angle(grasp_xyz, primary_angle - 20.0, arm, radius_m, 0.92),
        _candidate_for_angle(grasp_xyz, primary_angle + 20.0, arm, radius_m, 0.92),
        _candidate_for_angle(grasp_xyz, alternate_angle, alternate_arm, radius_m, 0.72),
        _candidate_for_angle(grasp_xyz, alternate_angle - 20.0, alternate_arm, radius_m, 0.66),
        _candidate_for_angle(grasp_xyz, alternate_angle + 20.0, alternate_arm, radius_m, 0.66),
    ]
    return arm, reason, candidates


def _pose_estimates(
    bottle: dict[str, Any],
    cup: dict[str, Any] | None,
    selected_grasp: dict[str, Any],
) -> dict[str, Any]:
    grasp_pos = [float(x) for x in selected_grasp["position"]]
    estimates: dict[str, Any] = {
        "dry_lift_pose": {
            "position": _round_list([grasp_pos[0], grasp_pos[1], grasp_pos[2] + LIFT_Z_M]),
            "rpy": selected_grasp["rpy"],
            "lift_z_m": LIFT_Z_M,
            "stage": 1,
        },
        "bottle_mid_body_grasp_z_source": (
            "half_extents_body_fraction" if bottle.get("half_extents") else "detection_center_z"
        ),
    }
    if cup is None:
        estimates["cup_target"] = None
        estimates["future_pour_pose"] = {
            "status": "not_estimated_without_cup_detection",
            "stage": 2,
        }
        return estimates

    cup_xyz = [float(x) for x in cup["position_3d"][:3]]
    dx = cup_xyz[0] - grasp_pos[0]
    dy = cup_xyz[1] - grasp_pos[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        unit = [1.0, 0.0]
    else:
        unit = [dx / norm, dy / norm]
    pre_pour_xy = [cup_xyz[0] - unit[0] * POUR_STANDOFF_M, cup_xyz[1] - unit[1] * POUR_STANDOFF_M]
    estimates["cup_target"] = {
        "label": cup.get("label", CUP_LABEL),
        "position": _round_list(cup_xyz),
    }
    estimates["future_pour_pose"] = {
        "position": _round_list([pre_pour_xy[0], pre_pour_xy[1], cup_xyz[2] + 0.18]),
        "aim_direction_xy": _round_list(unit[:2]),
        "nominal_tilt_deg": 65.0,
        "stage": 2,
        "status": "dry_empty_or_sealed_bottle_future_only",
        "limitation": "bottle mouth pose is not estimated in Stage 1",
    }
    estimates["stage3_stationary_cup_pour_target"] = {
        "cup_position": _round_list(cup_xyz),
        "status": "future_tiny_water_only_after_stage2_success",
    }
    return estimates


def _stage2_dry_pour_plan(estimates: dict[str, Any], selected_grasp: dict[str, Any]) -> dict[str, Any]:
    dry_lift = estimates.get("dry_lift_pose") or {}
    future_pour = estimates.get("future_pour_pose") or {}
    return {
        "stage": 2,
        "status": "planning_only_no_motion_no_liquid",
        "object_requirement": "empty_or_sealed_bottle",
        "stationary_cup_only": True,
        "nominal_tilt_deg": future_pour.get("nominal_tilt_deg", 65.0),
        "waypoints": [
            {
                "name": "post_grasp_lift",
                "position": dry_lift.get("position"),
                "rpy": dry_lift.get("rpy"),
                "note": "Stage 1 dry lift estimate; not executed.",
            },
            {
                "name": "pre_pour_standoff",
                "position": future_pour.get("position"),
                "rpy": selected_grasp.get("rpy"),
                "note": "Move near stationary cup in future dry run; not executed.",
            },
            {
                "name": "nominal_dry_tilt",
                "position": future_pour.get("position"),
                "tilt_deg": future_pour.get("nominal_tilt_deg", 65.0),
                "note": "Orientation placeholder; bottle-mouth pose is not estimated yet.",
            },
            {
                "name": "untilt_and_retreat",
                "position": dry_lift.get("position"),
                "rpy": dry_lift.get("rpy"),
                "note": "Future retreat after dry tilt; not executed.",
            },
        ],
        "blocked_before_execution": [
            "requires bounded physical-run confirmation",
            "requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1",
            "requires no-motion planner validation of side-grasp and dry-pour poses",
            "requires empty or sealed bottle; no liquid",
        ],
    }


def _require_physical_gate() -> None:
    if not _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        raise RuntimeError(
            "Refusing physical motion. This stage requires a valid physical-run "
            "ticket and OPENFORGE_ALLOW_PHYSICAL_MOTION=1."
        )


def _real_anygrasp_health() -> dict[str, Any]:
    import urllib.request

    with urllib.request.urlopen(f"{ANYGRASP_SERVICE_URL}/health", timeout=5) as response:
        data = json.loads(response.read().decode())
    if data.get("status") != "ok":
        raise RuntimeError(f"AnyGrasp health is not ok: {data}")
    if data.get("mock") is True or data.get("safe_for_robot_motion") is False:
        raise RuntimeError(f"Refusing mock or unsafe AnyGrasp service for YAM work: {data}")
    return data


def _anygrasp_candidate_to_dict(candidate: Any, index: int) -> dict[str, Any]:
    position = _obj_value(candidate, "position")
    rpy = _obj_value(candidate, "rpy")
    if position is None or rpy is None:
        raise RuntimeError(f"AnyGrasp candidate {index} lacks position/rpy: {candidate!r}")
    width = _obj_value(candidate, "width", DEFAULT_BOTTLE_RADIUS_M * 2.0 + GRASP_CLEARANCE_M)
    return {
        "type": "anygrasp",
        "source": "real_anygrasp",
        "candidate_index": int(index),
        "position": _round_list([float(x) for x in position]),
        "rpy": _round_list([float(x) for x in rpy], 3),
        "score": round(float(_obj_value(candidate, "score", 1.0)), 4),
        "width": round(float(width), 5),
        "orientation_status": "planner_aligned_anygrasp",
    }


def _sample_real_anygrasp_candidates(*, require_real: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    health = _real_anygrasp_health() if require_real else {}
    sample_grasp_pose_anygrasp = _required_tool("sample_grasp_pose_anygrasp")
    candidates_raw = sample_grasp_pose_anygrasp(
        object_name=BOTTLE_PROMPT,
        camera=CAMERA,
        max_grasps=ANYGRASP_MAX_GRASPS,
        object_input_mode=ANYGRASP_OBJECT_INPUT_MODE,
        disable_planner_z_clipping=True,
        filter_wrist_camera_y=True,
        allow_wrist_camera_yaw_flip=True,
    )
    candidates = [
        _anygrasp_candidate_to_dict(candidate, index)
        for index, candidate in enumerate(candidates_raw or [], start=1)
    ]
    if not candidates:
        raise RuntimeError(f"Real AnyGrasp returned no candidates for {BOTTLE_PROMPT!r}")
    return health, candidates


def _filter_stage1_anygrasp_side_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bottle = TASK_RESULT.get("detected_bottle") or {}
    position = bottle.get("position_3d") if isinstance(bottle, dict) else None
    info: dict[str, Any] = {
        "enabled": ANYGRASP_STAGE1_MAX_CENTER_Z_DELTA_M > 0.0,
        "max_center_z_delta_m": round(float(ANYGRASP_STAGE1_MAX_CENTER_Z_DELTA_M), 4),
        "input_count": len(candidates),
        "kept_count": len(candidates),
        "bottle_center_z": None,
    }
    if not info["enabled"] or not position:
        info["reason"] = "disabled_or_missing_bottle_center"
        return list(candidates), info

    center_z = float(position[2])
    max_delta = float(ANYGRASP_STAGE1_MAX_CENTER_Z_DELTA_M)
    filtered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate = dict(candidate)
        delta = float(candidate["position"][2]) - center_z
        candidate["stage1_center_z_delta_m"] = round(delta, 4)
        if abs(delta) <= max_delta:
            filtered.append(candidate)
        else:
            rejected.append(
                {
                    "candidate_index": candidate.get("candidate_index"),
                    "position": candidate.get("position"),
                    "stage1_center_z_delta_m": round(delta, 4),
                    "score": candidate.get("score"),
                }
            )

    info.update(
        {
            "kept_count": len(filtered),
            "rejected_count": len(rejected),
            "bottle_center_z": round(center_z, 4),
            "rejected_first": rejected[:5],
        }
    )
    if not filtered:
        info["reason"] = "no_anygrasp_candidates_within_stage1_side_grasp_z_band"
    return filtered, info


def _batch_result_summary(result: Any) -> dict[str, Any]:
    summary = _result_summary(result)
    summary.update(
        {
            "input_candidate_count": int(getattr(result, "input_candidate_count", 0) or 0),
            "evaluated_candidate_count": int(
                getattr(result, "evaluated_candidate_count", 0) or 0
            ),
            "truncated_input_count": int(getattr(result, "truncated_input_count", 0) or 0),
            "planning_mode": getattr(result, "planning_mode", None),
            "curobo_solve_time_ms": getattr(result, "curobo_solve_time_ms", None),
        }
    )
    return summary


def _geometric_batch_contact_rank_key(
    *,
    row: Any,
    candidate: dict[str, Any],
    side: str,
) -> tuple[float, float, float, float, float, float, float]:
    base_angle = -90.0 if side == "left" else 90.0
    angle_error = abs(
        _normalize_angle_deg(float(candidate.get("approach_angle_deg", base_angle)) - base_angle)
    )
    wrist_error = abs(float(candidate.get("wrist_roll_deg", 0.0) or 0.0))
    z_error = abs(float(candidate.get("z_offset_m", 0.0) or 0.0))
    x_error = abs(float(candidate.get("x_offset_m", 0.0) or 0.0))
    pos_err = float(_obj_value(row, "ik_error_m", float("inf")) or float("inf"))
    rot_err = float(_obj_value(row, "ik_rot_error_deg", float("inf")) or float("inf"))
    # For physical bottle contact, prefer the grasp geometry we intentionally
    # generated before tiny IK residual differences. Attempt 10 selected a
    # lower-score oblique candidate because its IK residual was a few mm lower,
    # then pushed the bottle instead of enclosing it.
    return (
        -float(candidate.get("score", 0.0) or 0.0),
        angle_error,
        wrist_error,
        z_error,
        x_error,
        pos_err,
        rot_err,
    )


def _select_geometric_batch_candidate(
    result: Any,
    side_indexed: list[tuple[int, dict[str, Any]]],
    *,
    side: str,
) -> tuple[Any | None, int | None, dict[str, Any] | None, list[dict[str, Any]]]:
    feasible: list[tuple[tuple[float, float, float, float, float, float, float], Any, int, dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in list(getattr(result, "batch_candidates", []) or []):
        if getattr(row, "motion_plan_error", True) is not False:
            continue
        source_index = int(_obj_value(row, "source_index", 0) or 0)
        if source_index < 0 or source_index >= len(side_indexed):
            continue
        original_index, candidate = side_indexed[source_index]
        pos_err = _obj_value(row, "ik_error_m")
        rot_err = _obj_value(row, "ik_rot_error_deg")
        try:
            pos_err_f = float(pos_err)
            rot_err_f = float(rot_err)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(pos_err_f)
            or not math.isfinite(rot_err_f)
            or pos_err_f > PHYSICAL_IK_ERROR_THRESHOLD_M
            or rot_err_f > PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG
        ):
            continue
        key = _geometric_batch_contact_rank_key(row=row, candidate=candidate, side=side)
        feasible.append((key, row, original_index, candidate))
        diagnostics.append(
            {
                "rank": int(_obj_value(row, "rank", 0) or 0),
                "candidate_index": int(original_index + 1),
                "score": candidate.get("score"),
                "approach_angle_deg": candidate.get("approach_angle_deg"),
                "wrist_roll_deg": candidate.get("wrist_roll_deg"),
                "z_offset_m": candidate.get("z_offset_m"),
                "x_offset_m": candidate.get("x_offset_m"),
                "ik_error_m": pos_err,
                "ik_rot_error_deg": rot_err,
            }
        )
    if not feasible:
        return None, None, None, diagnostics
    feasible.sort(key=lambda item: item[0])
    _, row, original_index, candidate = feasible[0]
    return row, int(original_index), candidate, diagnostics[:8]


def _rank_anygrasp_candidates_preview(
    freespace_move,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    preferred_side, reason = _select_arm([float(x) for x in candidates[0]["position"]])
    sides = [preferred_side, "left" if preferred_side == "right" else "right"]
    ranking: dict[str, Any] = {
        "stage": "stage1_anygrasp_candidate_preview",
        "preferred_side": preferred_side,
        "preferred_side_reason": reason,
        "candidate_count": len(candidates),
        "side_results": [],
        "success": False,
    }
    for side in sides:
        result = freespace_move(
            grasp_candidates=candidates,
            batch_side=side,
            batch_top_k=ANYGRASP_BATCH_TOP_K,
            solver_speed="fast",
            batch_validate_trajectory=False,
            preview_only=True,
            planning_speed=PHYSICAL_PLANNING_SPEED,
            ik_error_threshold=PHYSICAL_IK_ERROR_THRESHOLD_M,
            planner_backend="curobo",
        )
        side_summary = _batch_result_summary(result)
        side_summary["side"] = side
        best = getattr(result, "best_candidate", None)
        if best is not None and getattr(best, "motion_plan_error", True) is False:
            selected = _anygrasp_candidate_to_dict(best, int(_obj_value(best, "rank", 0) or 0))
            selected["arm"] = side
            selected["trajectory_cache_key"] = _obj_value(best, "trajectory_cache_key")
            selected["rank"] = int(_obj_value(best, "rank", 0) or 0)
            side_summary["selected"] = selected
            ranking["side_results"].append(side_summary)
            ranking["selected_grasp"] = selected
            ranking["success"] = True
            return ranking

        failures = []
        for candidate in list(getattr(result, "batch_candidates", []) or [])[:5]:
            if getattr(candidate, "motion_plan_error", False):
                failures.append(
                    {
                        "rank": int(getattr(candidate, "rank", 0) or 0),
                        "reason": getattr(candidate, "motion_plan_reason", None),
                    }
                )
        side_summary["failures"] = failures
        ranking["side_results"].append(side_summary)

    ranking["error"] = "No real AnyGrasp candidate passed preview ranking on either arm."
    return ranking


def _validate_physical_grasp_adjustments() -> None:
    if abs(PHYSICAL_GRASP_ADVANCE_M) > PHYSICAL_GRASP_ADVANCE_LIMIT_M:
        raise RuntimeError(
            "OPENFORGE_STAGE1_GRASP_ADVANCE_M exceeds the bounded limit: "
            f"{PHYSICAL_GRASP_ADVANCE_M:.3f}m > {PHYSICAL_GRASP_ADVANCE_LIMIT_M:.3f}m"
        )
    if abs(PHYSICAL_GRASP_Z_OFFSET_M) > PHYSICAL_GRASP_Z_OFFSET_LIMIT_M:
        raise RuntimeError(
            "OPENFORGE_STAGE1_GRASP_Z_OFFSET_M exceeds the bounded limit: "
            f"{PHYSICAL_GRASP_Z_OFFSET_M:.3f}m > {PHYSICAL_GRASP_Z_OFFSET_LIMIT_M:.3f}m"
        )


def _commanded_physical_grasp_pose(selected_grasp: dict[str, Any]) -> dict[str, list[float]]:
    _validate_physical_grasp_adjustments()
    nominal = [float(x) for x in selected_grasp["position"]]
    approach = [float(x) for x in selected_grasp.get("approach_direction_world", [0.0, 0.0, 0.0])]
    if len(approach) != 3 or math.sqrt(sum(component * component for component in approach)) < 0.5:
        raise RuntimeError(f"Invalid Stage 1 approach direction: {approach!r}")
    norm = math.sqrt(sum(component * component for component in approach))
    approach = [component / norm for component in approach]
    commanded = [
        nominal[0] + approach[0] * PHYSICAL_GRASP_ADVANCE_M,
        nominal[1] + approach[1] * PHYSICAL_GRASP_ADVANCE_M,
        nominal[2] + approach[2] * PHYSICAL_GRASP_ADVANCE_M + PHYSICAL_GRASP_Z_OFFSET_M,
    ]
    pregrasp = [
        commanded[0] - approach[0] * PREGRASP_STANDOFF_M,
        commanded[1] - approach[1] * PREGRASP_STANDOFF_M,
        commanded[2] - approach[2] * PREGRASP_STANDOFF_M,
    ]
    return {
        "nominal_grasp": nominal,
        "approach_direction": approach,
        "commanded_grasp": commanded,
        "commanded_pregrasp": pregrasp,
    }


def _side_target_kwargs(
    side: str,
    position: list[float],
    rpy: list[float],
    *,
    preview_only: bool,
    gripper_collision_width: float,
) -> dict[str, Any]:
    prefix = "left" if side == "left" else "right"
    return {
        f"{prefix}_target_pos": [float(x) for x in position],
        f"{prefix}_target_rpy": [float(x) for x in rpy],
        f"{prefix}_gripper": float(gripper_collision_width),
        "preview_only": bool(preview_only),
        "planning_speed": PHYSICAL_PLANNING_SPEED,
        "ik_error_threshold": PHYSICAL_IK_ERROR_THRESHOLD_M,
        "ik_rot_threshold_deg": PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG,
        "planner_backend": "curobo",
        "solver_speed": "fast",
    }


def _move_pose(
    freespace_move,
    side: str,
    position: list[float],
    rpy: list[float],
    label: str,
    *,
    preview_only: bool,
    gripper_collision_width: float,
) -> dict[str, Any]:
    kwargs = _side_target_kwargs(
        side,
        position,
        rpy,
        preview_only=preview_only,
        gripper_collision_width=gripper_collision_width,
    )
    result = freespace_move(**kwargs)
    summary = _result_summary(result)
    summary.update(
        {
            "label": label,
            "position": _round_list([float(x) for x in position]),
            "rpy": _round_list([float(x) for x in rpy], digits=3),
            "preview_only": bool(preview_only),
        }
    )
    if summary["status"] != "Success":
        raise RuntimeError(f"{label} freespace_move failed: {summary}")
    if preview_only and summary["executed"]:
        raise RuntimeError(f"{label} preview unexpectedly executed: {summary}")
    if not preview_only and not summary["executed"]:
        raise RuntimeError(f"{label} did not execute: {summary}")
    return summary


def _build_stage1_physical_attempt(selected_grasp: dict[str, Any]) -> dict[str, Any]:
    side = str(selected_grasp["arm"])
    rpy = [float(x) for x in selected_grasp["rpy"]]
    pose_adjustment = _commanded_physical_grasp_pose(selected_grasp)
    nominal_grasp = pose_adjustment["nominal_grasp"]
    approach_direction = pose_adjustment["approach_direction"]
    grasp = pose_adjustment["commanded_grasp"]
    pregrasp = pose_adjustment["commanded_pregrasp"]
    width = float(selected_grasp["width"])
    lift = [grasp[0], grasp[1], grasp[2] + PHYSICAL_LIFT_Z_M]
    release = list(grasp)

    return {
        "stage": "stage1_physical_dry_grasp_lift",
        "side": side,
        "candidate_score": selected_grasp.get("score"),
        "candidate_approach_angle_deg": selected_grasp.get("approach_angle_deg"),
        "nominal_grasp_pose": {
            "position": _round_list(nominal_grasp),
            "rpy": _round_list(rpy, 3),
        },
        "physical_adjustments": {
            "advance_along_approach_m": round(PHYSICAL_GRASP_ADVANCE_M, 4),
            "z_offset_m": round(PHYSICAL_GRASP_Z_OFFSET_M, 4),
            "advance_limit_m": round(PHYSICAL_GRASP_ADVANCE_LIMIT_M, 4),
            "z_offset_limit_m": round(PHYSICAL_GRASP_Z_OFFSET_LIMIT_M, 4),
            "ik_rot_threshold_deg": round(PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG, 3),
        },
        "approach_direction_world": _round_list(approach_direction),
        "pregrasp_pose": {"position": _round_list(pregrasp), "rpy": _round_list(rpy, 3)},
        "grasp_pose": {"position": _round_list(grasp), "rpy": _round_list(rpy, 3)},
        "lift_pose": {"position": _round_list(lift), "rpy": _round_list(rpy, 3)},
        "release_pose": {"position": _round_list(release), "rpy": _round_list(rpy, 3)},
        "gripper_width_m": width,
        "hold_s": PHYSICAL_HOLD_S,
        "moves": [],
        "gripper": {},
        "recovery": [],
        "success": False,
    }


def _preview_stage1_attempt(freespace_move, attempt: dict[str, Any]) -> None:
    side = str(attempt["side"])
    rpy = [float(x) for x in attempt["grasp_pose"]["rpy"]]
    width = float(attempt["gripper_width_m"])
    for label, pose_key in (
        ("preview_pregrasp", "pregrasp_pose"),
        ("preview_grasp", "grasp_pose"),
        ("preview_lift", "lift_pose"),
    ):
        attempt["moves"].append(
            _move_pose(
                freespace_move,
                side,
                [float(x) for x in attempt[pose_key]["position"]],
                rpy,
                label,
                preview_only=True,
                gripper_collision_width=width,
            )
        )


def _select_preview_feasible_stage1_attempt(
    freespace_move,
    selected_grasp: dict[str, Any],
    grasp_candidates: list[dict[str, Any]] | None,
    *,
    record_failure: bool = True,
) -> dict[str, Any]:
    indexed_candidates = list(enumerate(list(grasp_candidates or []), start=1))
    selected_source_index = 1
    for source_index, candidate in indexed_candidates:
        if candidate == selected_grasp:
            selected_source_index = source_index
            break
    candidate_pool: list[tuple[int, dict[str, Any]]] = [
        (selected_source_index, selected_grasp)
    ]
    for source_index, candidate in indexed_candidates:
        if candidate != selected_grasp:
            candidate_pool.append((source_index, candidate))

    preview_failures: list[dict[str, Any]] = []
    for preview_order, (source_index, candidate) in enumerate(candidate_pool, start=1):
        attempt = _build_stage1_physical_attempt(candidate)
        attempt["candidate_index"] = source_index
        attempt["candidate_preview_order"] = preview_order
        try:
            _preview_stage1_attempt(freespace_move, attempt)
            attempt["candidate_preview_failures"] = preview_failures
            return attempt
        except Exception as exc:
            attempt["preview_error"] = f"{type(exc).__name__}: {exc}"
            preview_failures.append(
                {
                    "candidate_index": source_index,
                    "candidate_preview_order": preview_order,
                    "side": attempt["side"],
                    "score": attempt.get("candidate_score"),
                    "approach_angle_deg": attempt.get("candidate_approach_angle_deg"),
                    "pregrasp_pose": attempt["pregrasp_pose"],
                    "grasp_pose": attempt["grasp_pose"],
                    "error": attempt["preview_error"],
                }
            )

    if record_failure:
        failed_attempt = {
            "stage": "stage1_physical_dry_grasp_lift",
            "success": False,
            "movement_capable_calls": [],
            "candidate_preview_failures": preview_failures,
            "error": "No Stage 1 side-grasp candidate passed preview planning.",
        }
        TASK_RESULT["physical_attempt"] = failed_attempt
        TASK_RESULT["physical_motion_executed"] = False
        TASK_RESULT["movement_capable_calls"] = []
        TASK_RESULT["why_stopped"] = (
            "Stage 1 physical dry attempt stopped before gripper/arm motion because "
            "no candidate passed preview planning."
        )
    raise RuntimeError(f"No Stage 1 side-grasp candidate passed preview planning: {preview_failures}")


def _run_stage1_planner_preview_only(
    selected_grasp: dict[str, Any],
    grasp_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    try:
        attempt = _select_preview_feasible_stage1_attempt(
            freespace_move,
            selected_grasp,
            grasp_candidates,
            record_failure=False,
        )
        return {
            "stage": "stage1_planner_preview_only",
            "success": True,
            "physical_motion_executed": False,
            "selected_preview_attempt": attempt,
        }
    except Exception as exc:
        return {
            "stage": "stage1_planner_preview_only",
            "success": False,
            "physical_motion_executed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _rank_geometric_candidates_batch_preview(
    freespace_move,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        return {
            "stage": "stage1_geom_depth_batch_preview",
            "success": False,
            "candidate_count": 0,
            "error": "No geometric candidates provided.",
        }
    preferred_side, reason = _select_arm([float(x) for x in candidates[0]["position"]])
    sides = [preferred_side, "left" if preferred_side == "right" else "right"]
    ranking: dict[str, Any] = {
        "stage": "stage1_geom_depth_batch_preview",
        "preferred_side": preferred_side,
        "preferred_side_reason": reason,
        "candidate_count": len(candidates),
        "side_results": [],
        "success": False,
    }
    for side in sides:
        side_indexed = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if str(candidate.get("arm")) == side
        ]
        side_candidates = [candidate for _, candidate in side_indexed]
        if not side_candidates:
            ranking["side_results"].append(
                {"side": side, "success": False, "error": "no_candidates_for_side"}
            )
            continue
        truncated_input_count = 0
        if GEOM_BATCH_CANDIDATE_LIMIT > 0 and len(side_candidates) > GEOM_BATCH_CANDIDATE_LIMIT:
            truncated_input_count = len(side_candidates) - GEOM_BATCH_CANDIDATE_LIMIT
            side_indexed = side_indexed[:GEOM_BATCH_CANDIDATE_LIMIT]
            side_candidates = side_candidates[:GEOM_BATCH_CANDIDATE_LIMIT]
        result = freespace_move(
            grasp_candidates=side_candidates,
            batch_side=side,
            batch_top_k=len(side_candidates),
            solver_speed="fast",
            batch_validate_trajectory=False,
            preview_only=True,
            planning_speed=PHYSICAL_PLANNING_SPEED,
            ik_error_threshold=PHYSICAL_IK_ERROR_THRESHOLD_M,
            ik_rot_threshold_deg=PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG,
            planner_backend="curobo",
        )
        side_summary = _batch_result_summary(result)
        side_summary["side"] = side
        side_summary["geometric_input_candidate_count"] = len(side_indexed) + truncated_input_count
        side_summary["geometric_batch_candidate_limit"] = int(GEOM_BATCH_CANDIDATE_LIMIT)
        side_summary["geometric_truncated_before_batch_count"] = int(truncated_input_count)
        best_row, original_index, contact_candidate, contact_diagnostics = (
            _select_geometric_batch_candidate(result, side_indexed, side=side)
        )
        if contact_diagnostics:
            side_summary["contact_prioritized_feasible_candidates"] = contact_diagnostics
        if best_row is not None and original_index is not None and contact_candidate is not None:
            selected = dict(candidates[original_index])
            selected["arm"] = side
            selected["trajectory_cache_key"] = _obj_value(best_row, "trajectory_cache_key")
            selected["rank"] = int(_obj_value(best_row, "rank", 0) or 0)
            selected["ik_error_m"] = _obj_value(best_row, "ik_error_m")
            selected["ik_rot_error_deg"] = _obj_value(best_row, "ik_rot_error_deg")
            selected["planner_status"] = _obj_value(best_row, "planner_status")
            selected["batch_source_index"] = int(_obj_value(best_row, "source_index", 0) or 0)
            selected["candidate_index"] = original_index + 1
            selected["selection_mode"] = "contact_prioritized_feasible_batch_candidate"
            selected_candidate_index = original_index + 1
            side_summary["selection_method"] = selected["selection_mode"]
            side_summary["selected"] = selected
            ranking["side_results"].append(side_summary)
            ranking["selected_grasp"] = selected
            ranking["selected_candidate_index"] = selected_candidate_index
            ranking["selection_mode"] = selected["selection_mode"]
            ranking["success"] = True
            return ranking

        single_fallbacks: list[dict[str, Any]] = []
        low_residual_rows = []
        for row in list(getattr(result, "batch_candidates", []) or []):
            pos_err = _obj_value(row, "ik_error_m")
            rot_err = _obj_value(row, "ik_rot_error_deg")
            if pos_err is None or rot_err is None:
                continue
            try:
                pos_err_f = float(pos_err)
                rot_err_f = float(rot_err)
            except (TypeError, ValueError):
                continue
            if (
                math.isfinite(pos_err_f)
                and math.isfinite(rot_err_f)
                and pos_err_f <= PHYSICAL_IK_ERROR_THRESHOLD_M
                and rot_err_f <= PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG
            ):
                low_residual_rows.append(row)

        for row in low_residual_rows[: max(0, GEOM_SINGLE_PREVIEW_FALLBACK_TOP_K)]:
            source_index = int(_obj_value(row, "source_index", 0) or 0)
            if source_index < 0 or source_index >= len(side_indexed):
                continue
            original_index = side_indexed[source_index][0]
            candidate = dict(candidates[original_index])
            try:
                preview_result = freespace_move(
                    **_side_target_kwargs(
                        side,
                        [float(x) for x in candidate["position"]],
                        [float(x) for x in candidate["rpy"]],
                        preview_only=True,
                        gripper_collision_width=float(candidate.get("width", 0.09)),
                    )
                )
                preview_summary = _result_summary(preview_result)
            except Exception as exc:
                preview_summary = {
                    "status": "Error",
                    "executed": False,
                    "side": side,
                    "trajectory_steps": 0,
                    "trajectory_cache_key": None,
                    "final_pos_error_m": None,
                    "final_rot_error_deg": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            preview_summary.update(
                {
                    "source_index": source_index,
                    "candidate_index": original_index + 1,
                    "batch_planner_status": _obj_value(row, "planner_status"),
                    "batch_ik_error_m": _obj_value(row, "ik_error_m"),
                    "batch_ik_rot_error_deg": _obj_value(row, "ik_rot_error_deg"),
                }
            )
            single_fallbacks.append(preview_summary)
            if (
                preview_summary["status"] == "Success"
                and not preview_summary["executed"]
                and int(preview_summary.get("trajectory_steps") or 0) > 0
                and preview_summary.get("trajectory_cache_key")
            ):
                selected = dict(candidate)
                selected["arm"] = side
                selected["trajectory_cache_key"] = preview_summary.get(
                    "trajectory_cache_key"
                )
                selected["rank"] = int(_obj_value(row, "rank", 0) or 0)
                selected["ik_error_m"] = preview_summary.get("final_pos_error_m")
                selected["ik_rot_error_deg"] = preview_summary.get(
                    "final_rot_error_deg"
                )
                selected["planner_status"] = "SinglePreviewFallbackSuccess"
                selected["batch_source_index"] = source_index
                selected["candidate_index"] = original_index + 1
                selected["selection_mode"] = (
                    "single_direct_preview_fallback_after_batch_low_residual"
                )
                selected_candidate_index = original_index + 1
                side_summary["single_preview_fallbacks"] = single_fallbacks
                side_summary["selected"] = selected
                ranking["side_results"].append(side_summary)
                ranking["selected_grasp"] = selected
                ranking["selected_candidate_index"] = selected_candidate_index
                ranking["selection_mode"] = selected["selection_mode"]
                ranking["success"] = True
                return ranking

        if single_fallbacks:
            side_summary["single_preview_fallbacks"] = single_fallbacks

        failures = []
        for candidate in list(getattr(result, "batch_candidates", []) or [])[:5]:
            failures.append(
                {
                    "rank": int(getattr(candidate, "rank", 0) or 0),
                    "source_index": int(getattr(candidate, "source_index", 0) or 0),
                    "planner_status": getattr(candidate, "planner_status", None),
                    "motion_plan_error": getattr(candidate, "motion_plan_error", None),
                    "reason": getattr(candidate, "motion_plan_reason", None),
                    "ik_error_m": getattr(candidate, "ik_error_m", None),
                    "ik_rot_error_deg": getattr(candidate, "ik_rot_error_deg", None),
                }
            )
        side_summary["failures"] = failures
        ranking["side_results"].append(side_summary)

    ranking["error"] = "No geometric depth candidate passed batch cuRobo preview on either arm."
    return ranking


def _geometric_candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(candidate.get("arm")),
        tuple(round(float(x), 4) for x in candidate.get("position", [])[:3]),
        tuple(round(float(x), 3) for x in candidate.get("rpy", [])[:3]),
    )


def _ordered_geometric_guarded_candidate_pool(
    ranking: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = ranking.get("selected_grasp")
    selected_arm = str(selected.get("arm")) if isinstance(selected, dict) else None
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(candidate: dict[str, Any], candidate_index: int | None) -> None:
        enriched = dict(candidate)
        if candidate_index is not None:
            enriched["candidate_index"] = int(candidate_index)
        elif enriched.get("candidate_index") is None:
            enriched["candidate_index"] = len(ordered) + 1
        key = _geometric_candidate_key(enriched)
        if key in seen:
            return
        seen.add(key)
        ordered.append(enriched)

    if isinstance(selected, dict):
        selected_index = ranking.get("selected_candidate_index") or selected.get("candidate_index")
        add(selected, int(selected_index or 1))

    indexed = list(enumerate(candidates, start=1))
    side_order = []
    if selected_arm:
        side_order.append(selected_arm)
    preferred_side = ranking.get("preferred_side")
    if preferred_side and str(preferred_side) not in side_order:
        side_order.append(str(preferred_side))
    for side in ("left", "right"):
        if side not in side_order:
            side_order.append(side)

    for side in side_order:
        same_side = [
            (index, candidate)
            for index, candidate in indexed
            if str(candidate.get("arm")) == side
        ]
        same_side.sort(key=lambda pair: float(pair[1].get("score", 0.0)), reverse=True)
        for index, candidate in same_side:
            add(candidate, index)
            if len(ordered) >= max(1, GEOM_GUARDED_PREVIEW_CANDIDATE_LIMIT):
                return ordered

    return ordered


def _geometric_guarded_failure_summary(
    attempt: dict[str, Any],
    preview: dict[str, Any],
    preview_order: int,
) -> dict[str, Any]:
    selected = attempt.get("selected_grasp") or {}
    return {
        "candidate_index": attempt.get("candidate_index"),
        "preview_order": int(preview_order),
        "side": attempt.get("side"),
        "score": selected.get("score"),
        "approach_angle_deg": selected.get("approach_angle_deg"),
        "z_offset_m": selected.get("z_offset_m"),
        "wrist_roll_deg": selected.get("wrist_roll_deg"),
        "guarded_pregrasp_standoff_m": selected.get("guarded_pregrasp_standoff_m"),
        "pregrasp_pose": attempt.get("pregrasp_pose"),
        "grasp_pose": attempt.get("grasp_pose"),
        "error": preview.get("error") or preview.get("reason"),
    }


def _select_geometric_guarded_stage1_attempt(
    freespace_move,
    ranking: dict[str, Any],
    candidates: list[dict[str, Any]],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    pool = _ordered_geometric_guarded_candidate_pool(ranking, candidates)
    failures: list[dict[str, Any]] = []
    preview_order = 0
    standoffs = _guarded_pregrasp_standoff_values()
    for candidate in pool:
        candidate = dict(candidate)
        candidate_index = int(candidate.get("candidate_index", len(failures) + 1) or 1)
        for standoff in standoffs:
            preview_order += 1
            if preview_order > max(1, GEOM_GUARDED_PREVIEW_COMBINATION_LIMIT):
                return {
                    "success": False,
                    "selection_mode": "guarded_pregrasp_candidate_preview",
                    "candidate_count": len(pool),
                    "combination_limit": GEOM_GUARDED_PREVIEW_COMBINATION_LIMIT,
                    "failures": failures,
                    "preview": {
                        "stage": "stage1_geometric_guarded_pregrasp_preview",
                        "success": False,
                        "physical_motion_executed": False,
                        "execution_mode": "guarded_pregrasp_then_grasp",
                        "previews": [],
                        "error": (
                            "No geometric candidate passed guarded pregrasp/grasp/lift "
                            f"preview within {GEOM_GUARDED_PREVIEW_COMBINATION_LIMIT} "
                            "candidate/standoff combination(s)."
                        ),
                    },
                    "error": (
                        "No geometric candidate passed guarded pregrasp/grasp/lift "
                        f"preview within {GEOM_GUARDED_PREVIEW_COMBINATION_LIMIT} "
                        "candidate/standoff combination(s)."
                    ),
                }
            candidate_variant = dict(candidate)
            candidate_variant["candidate_index"] = candidate_index
            candidate_variant["guarded_pregrasp_standoff_m"] = float(standoff)
            candidate_ranking = dict(ranking)
            candidate_ranking["selected_candidate_index"] = candidate_index
            candidate_ranking["selected_grasp"] = candidate_variant
            candidate_ranking["selection_mode"] = "guarded_pregrasp_candidate_preview"
            attempt = _build_geometric_guarded_stage1_attempt(
                candidate_variant,
                candidate_ranking,
                geometry,
            )
            attempt["guarded_candidate_preview_order"] = preview_order
            preview = _preview_geometric_guarded_stage1_attempt(
                freespace_move,
                attempt,
            )
            if preview.get("success"):
                attempt["previews"] = preview.get("previews", [])
                attempt["candidate_preview_failures"] = failures
                return {
                    "success": True,
                    "selected_candidate_index": candidate_index,
                    "selected_grasp": candidate_variant,
                    "selection_mode": "guarded_pregrasp_candidate_preview",
                    "preview_order": preview_order,
                    "attempt": attempt,
                    "preview": preview,
                    "failures": failures,
                }
            failures.append(
                _geometric_guarded_failure_summary(attempt, preview, preview_order)
            )

    return {
        "success": False,
        "selection_mode": "guarded_pregrasp_candidate_preview",
        "candidate_count": len(pool),
        "failures": failures,
        "preview": {
            "stage": "stage1_geometric_guarded_pregrasp_preview",
            "success": False,
            "physical_motion_executed": False,
            "execution_mode": "guarded_pregrasp_then_grasp",
            "previews": [],
            "error": (
                "No geometric candidate passed guarded pregrasp/grasp/lift "
                f"preview across {len(pool)} candidate(s)."
            ),
        },
        "error": (
            "No geometric candidate passed guarded pregrasp/grasp/lift "
            f"preview across {len(pool)} candidate(s)."
        ),
    }


def _run_stage1_geometric_depth_preview_only(
    selected_grasp: dict[str, Any],
    grasp_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    ranking = _rank_geometric_candidates_batch_preview(freespace_move, grasp_candidates)
    guarded_preview: dict[str, Any] | None = None
    guarded_selection: dict[str, Any] | None = None
    if ranking.get("success") and ranking.get("selected_grasp"):
        guarded_selection = _select_geometric_guarded_stage1_attempt(
            freespace_move,
            ranking,
            grasp_candidates,
            geometry={},
        )
        guarded_preview = dict(guarded_selection.get("preview") or {})
        if guarded_selection.get("success"):
            ranking["selected_grasp"] = guarded_selection["selected_grasp"]
            ranking["selected_candidate_index"] = guarded_selection[
                "selected_candidate_index"
            ]
            ranking["selection_mode"] = guarded_selection["selection_mode"]
        ranking["guarded_selection"] = {
            "success": guarded_selection.get("success"),
            "selection_mode": guarded_selection.get("selection_mode"),
            "selected_candidate_index": guarded_selection.get("selected_candidate_index"),
            "preview_order": guarded_selection.get("preview_order"),
            "failure_count": len(guarded_selection.get("failures") or []),
            "failures": (guarded_selection.get("failures") or [])[:8],
            "error": guarded_selection.get("error"),
        }
    return {
        "stage": "stage1_geom_depth_preview_only",
        "source": "bundlesdf_sam3_depth_geometry",
        "success": bool(ranking.get("success"))
        and (guarded_preview is None or bool(guarded_preview.get("success"))),
        "physical_motion_executed": False,
        "ranking_method": "batch_curobo_rank_then_guarded_pregrasp_preview",
        "candidate_count": len(grasp_candidates),
        "selected_candidate_index": ranking.get("selected_candidate_index"),
        "selected_grasp": ranking.get("selected_grasp"),
        "ranking": ranking,
        "guarded_motion_preview": guarded_preview,
        **(
            {}
            if ranking.get("success") and (guarded_preview is None or guarded_preview.get("success"))
            else {
                "error": (
                    ranking.get("error")
                    or (guarded_preview or {}).get("error")
                    or "Guarded pregrasp/grasp/lift preview failed."
                )
            }
        ),
    }


def _run_geometric_depth_stage(
    *,
    stage: str,
    bottle: dict[str, Any],
    cup: dict[str, Any] | None,
) -> None:
    geometry, overlay_source = _estimate_bottle_depth_geometry(bottle)
    if cup and isinstance(cup.get("position_3d"), list) and len(cup["position_3d"]) >= 3:
        geometry["cup_position_world"] = _round_list(
            [float(x) for x in cup["position_3d"][:3]],
            4,
        )
        geometry["cup_side_rejection"] = {
            "enabled": bool(GEOM_REJECT_CUP_SIDE_APPROACH),
            "dot_max": round(float(GEOM_CUP_SIDE_DOT_MAX), 4),
            "reason": "avoid pregrasp/approach paths from the stationary cup side",
        }
    TASK_RESULT["geometry_notes"].extend(
        [
            "Using synchronized camera-portal RGB/depth plus BundleSDF /segment "
            "SAM3 mask for the primary Stage 1 side-grasp geometry.",
            "Bottle axis is treated as upright world Z for this Stage 1 side-grasp; "
            "no pour-mouth pose is estimated.",
            "AnyGrasp is not used as the primary grasp source for this stage.",
        ]
    )
    preferred_arm, arm_reason, candidates = _geometric_depth_side_grasp_candidates(geometry)
    selected = candidates[0] if candidates else None
    if selected is None:
        TASK_RESULT["why_stopped"] = "No geometric depth side-grasp candidates generated."
        return

    estimates = _pose_estimates(bottle, cup, selected)
    estimates["stage1_depth_geometry"] = geometry
    TASK_RESULT.update(
        {
            "selected_arm": preferred_arm,
            "selected_arm_reason": arm_reason,
            "grasp_candidates": candidates,
            "selected_grasp_pose": selected,
            "target_pose_estimates": estimates,
            "geometric_depth": geometry,
        }
    )

    preview: dict[str, Any] | None = None
    if stage == "stage1_geom_depth_preview_only":
        try:
            preview = _run_stage1_geometric_depth_preview_only(selected, candidates)
        except Exception as exc:
            preview = {
                "stage": "stage1_geom_depth_preview_only",
                "source": "bundlesdf_sam3_depth_geometry",
                "success": False,
                "physical_motion_executed": False,
                "candidate_count": len(candidates),
                "selected_candidate_index": None,
                "selected_grasp": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            TASK_RESULT["geometric_depth_preview"] = preview
            TASK_RESULT["physical_motion_executed"] = False
            TASK_RESULT["movement_capable_calls"] = []
            TASK_RESULT["visual_artifacts"].extend(
                _save_geometric_depth_overlay(
                    stage=f"{stage}_planner_failed",
                    geometry=geometry,
                    overlay_source=overlay_source,
                    bottle=bottle,
                    cup=cup,
                    selected=selected,
                    candidates=candidates,
                    preview=preview,
                )
            )
            TASK_RESULT["why_stopped"] = (
                "Stage 1 geometric depth preview stopped before planner "
                f"selection: {type(exc).__name__}: {exc}"
            )
            return
        selected_index = int(preview.get("selected_candidate_index", 1) or 1)
        if 1 <= selected_index <= len(candidates):
            selected = dict(preview.get("selected_grasp") or candidates[selected_index - 1])
            TASK_RESULT["selected_grasp_pose"] = selected
            TASK_RESULT["target_pose_estimates"] = _pose_estimates(bottle, cup, selected)
            TASK_RESULT["target_pose_estimates"]["stage1_depth_geometry"] = geometry
        TASK_RESULT["geometric_depth_preview"] = preview
        TASK_RESULT["physical_motion_executed"] = False
        TASK_RESULT["movement_capable_calls"] = []
        TASK_RESULT["visual_artifacts"].extend(
            _save_geometric_depth_overlay(
                stage=stage,
                geometry=geometry,
                overlay_source=overlay_source,
                bottle=bottle,
                cup=cup,
                selected=selected,
                candidates=candidates,
                preview=preview,
            )
        )
        if not preview.get("success"):
            TASK_RESULT["why_stopped"] = (
                "Stage 1 geometric depth preview found no cuRobo-feasible "
                "candidate; no gripper, arm motion, lift, pour, or liquid "
                "actions were run."
            )
            return
        TASK_RESULT["success"] = True
        TASK_RESULT["reward"] = 1.0
        TASK_RESULT["why_stopped"] = (
            "Stage 1 geometric depth preview completed; no gripper, arm motion, "
            "lift, pour, or liquid actions were run."
        )
        return

    if stage == "stage1_geom_depth_physical_dry_grasp_lift":
        freespace_move = _required_tool("freespace_move")
        try:
            ranking = _rank_geometric_candidates_batch_preview(freespace_move, candidates)
        except Exception as exc:
            ranking = {
                "stage": "stage1_geom_depth_physical_precheck",
                "source": "bundlesdf_sam3_depth_geometry",
                "success": False,
                "physical_motion_executed": False,
                "candidate_count": len(candidates),
                "selected_candidate_index": None,
                "selected_grasp": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            TASK_RESULT["geometric_depth_preview"] = ranking
            TASK_RESULT["visual_artifacts"].extend(
                _save_geometric_depth_overlay(
                    stage=f"{stage}_planner_failed",
                    geometry=geometry,
                    overlay_source=overlay_source,
                    bottle=bottle,
                    cup=cup,
                    selected=selected,
                    candidates=candidates,
                    preview=ranking,
                )
            )
            raise RuntimeError(
                f"Geometric candidate planner precheck failed before motion: {ranking}"
            ) from exc
        if not ranking.get("success"):
            TASK_RESULT["geometric_depth_preview"] = ranking
            TASK_RESULT["visual_artifacts"].extend(
                _save_geometric_depth_overlay(
                    stage=f"{stage}_precheck_failed",
                    geometry=geometry,
                    overlay_source=overlay_source,
                    bottle=bottle,
                    cup=cup,
                    selected=selected,
                    candidates=candidates,
                    preview=ranking,
                )
            )
            raise RuntimeError(f"No batch-preview-feasible geometric candidate: {ranking}")
        selected_index = int(ranking.get("selected_candidate_index", 1) or 1)
        if 1 <= selected_index <= len(candidates):
            selected = dict(ranking.get("selected_grasp") or candidates[selected_index - 1])
            TASK_RESULT["selected_grasp_pose"] = selected
            estimates = _pose_estimates(bottle, cup, selected)
            estimates["stage1_depth_geometry"] = geometry
            TASK_RESULT["target_pose_estimates"] = estimates
        guarded_selection = _select_geometric_guarded_stage1_attempt(
            freespace_move,
            ranking,
            candidates,
            geometry,
        )
        if guarded_selection.get("success"):
            selected = dict(guarded_selection["selected_grasp"])
            selected_index = int(guarded_selection["selected_candidate_index"])
            ranking["selected_grasp"] = selected
            ranking["selected_candidate_index"] = selected_index
            ranking["selection_mode"] = guarded_selection["selection_mode"]
            TASK_RESULT["selected_grasp_pose"] = selected
            estimates = _pose_estimates(bottle, cup, selected)
            estimates["stage1_depth_geometry"] = geometry
            TASK_RESULT["target_pose_estimates"] = estimates
        preview = {
            "stage": "stage1_geom_depth_physical_precheck",
            "source": "bundlesdf_sam3_depth_geometry",
            "success": bool(guarded_selection.get("success")),
            "physical_motion_executed": False,
            "ranking_method": "batch_curobo_rank_then_guarded_pregrasp_preview",
            "candidate_count": len(candidates),
            "selected_candidate_index": selected_index,
            "selected_grasp": selected,
            "ranking": ranking,
            "guarded_selection": {
                "success": guarded_selection.get("success"),
                "selection_mode": guarded_selection.get("selection_mode"),
                "selected_candidate_index": guarded_selection.get("selected_candidate_index"),
                "preview_order": guarded_selection.get("preview_order"),
                "failure_count": len(guarded_selection.get("failures") or []),
                "failures": (guarded_selection.get("failures") or [])[:8],
                "error": guarded_selection.get("error"),
            },
        }
        guarded_preview = dict(guarded_selection.get("preview") or {})
        preview["guarded_motion_preview"] = guarded_preview
        if not guarded_preview.get("success"):
            TASK_RESULT["geometric_depth_preview"] = preview
            TASK_RESULT["visual_artifacts"].extend(
                _save_geometric_depth_overlay(
                    stage=f"{stage}_guarded_precheck_failed",
                    geometry=geometry,
                    overlay_source=overlay_source,
                    bottle=bottle,
                    cup=cup,
                    selected=selected,
                    candidates=candidates,
                    preview=preview,
                )
            )
            raise RuntimeError(
                f"Guarded pregrasp/grasp/lift preview failed: {guarded_preview}"
            )
        TASK_RESULT["geometric_depth_preview"] = preview
        TASK_RESULT["visual_artifacts"].extend(
            _save_geometric_depth_overlay(
                stage=f"{stage}_before_motion",
                geometry=geometry,
                overlay_source=overlay_source,
                bottle=bottle,
                cup=cup,
                selected=selected,
                candidates=candidates,
                preview=preview,
            )
        )
        try:
            attempt = _run_stage1_geometric_depth_batch_physical_dry_grasp_lift(
                selected,
                ranking,
                geometry,
            )
        except Exception:
            attempt = TASK_RESULT.get("physical_attempt")
            if isinstance(attempt, dict):
                attempt["stage"] = "stage1_geom_depth_physical_dry_grasp_lift"
                attempt["source"] = "bundlesdf_sam3_depth_geometry"
                attempt["geometry"] = geometry
                attempt_index = int(attempt.get("candidate_index", selected_index) or selected_index)
                if 1 <= attempt_index <= len(candidates):
                    TASK_RESULT["selected_grasp_pose"] = candidates[attempt_index - 1]
                TASK_RESULT["physical_attempt"] = attempt
            raise
        attempt["stage"] = "stage1_geom_depth_physical_dry_grasp_lift"
        attempt["source"] = "bundlesdf_sam3_depth_geometry"
        attempt["geometry"] = geometry
        selected_index = int(attempt.get("candidate_index", 1) or 1)
        if 1 <= selected_index <= len(candidates):
            selected = candidates[selected_index - 1]
            TASK_RESULT["selected_grasp_pose"] = selected
        TASK_RESULT["physical_attempt"] = attempt
        TASK_RESULT["physical_motion_executed"] = True
        TASK_RESULT["movement_capable_calls"] = list(attempt.get("movement_capable_calls", []))
        TASK_RESULT["success"] = True
        TASK_RESULT["reward"] = 1.0
        TASK_RESULT["why_stopped"] = (
            "Stage 1 geometric depth physical dry grasp/lift/release/home "
            "attempt completed within the ticket scope."
        )
        return

    raise RuntimeError(f"Unsupported geometric depth stage: {stage}")


def _run_stage1_anygrasp_preview_only() -> dict[str, Any]:
    freespace_move = _required_tool("freespace_move")
    health, candidates = _sample_real_anygrasp_candidates(require_real=True)
    filtered_candidates, filter_info = _filter_stage1_anygrasp_side_candidates(candidates)
    if not filtered_candidates:
        return {
            "stage": "stage1_anygrasp_preview_only",
            "success": False,
            "physical_motion_executed": False,
            "anygrasp_health": health,
            "anygrasp_candidates": candidates,
            "stage1_side_grasp_filter": filter_info,
            "ranking": {"success": False, "error": filter_info.get("reason")},
        }
    ranking = _rank_anygrasp_candidates_preview(freespace_move, filtered_candidates)
    return {
        "stage": "stage1_anygrasp_preview_only",
        "success": bool(ranking.get("success")),
        "physical_motion_executed": False,
        "anygrasp_health": health,
        "anygrasp_candidates": candidates,
        "filtered_anygrasp_candidates": filtered_candidates,
        "stage1_side_grasp_filter": filter_info,
        "ranking": ranking,
    }


def _run_stage1_anygrasp_physical_dry_grasp_lift() -> dict[str, Any]:
    _require_physical_gate()
    freespace_move = _required_tool("freespace_move")
    open_gripper = _required_tool("open_gripper")
    close_gripper = _required_tool("close_gripper")
    get_robot_state = _required_tool("get_robot_state")
    go_home = _required_tool("go_home")

    health, candidates = _sample_real_anygrasp_candidates(require_real=True)
    filtered_candidates, filter_info = _filter_stage1_anygrasp_side_candidates(candidates)
    if not filtered_candidates:
        TASK_RESULT["physical_motion_executed"] = False
        TASK_RESULT["movement_capable_calls"] = []
        TASK_RESULT["physical_attempt"] = {
            "stage": "stage1_anygrasp_physical_dry_grasp_lift",
            "source": "real_anygrasp",
            "success": False,
            "physical_motion_executed": False,
            "movement_capable_calls": [],
            "anygrasp_health": health,
            "anygrasp_candidate_count": len(candidates),
            "filtered_anygrasp_candidate_count": 0,
            "stage1_side_grasp_filter": filter_info,
            "error": "No Stage 1 side-grasp AnyGrasp candidates passed the Z-band filter.",
        }
        TASK_RESULT["why_stopped"] = (
            "Stage 1 AnyGrasp physical attempt stopped before planner, arm, or "
            "gripper commands because no candidate passed the side-grasp Z-band "
            "filter."
        )
        raise RuntimeError(f"No Stage 1 side-grasp AnyGrasp candidates: {filter_info}")
    ranking = _rank_anygrasp_candidates_preview(freespace_move, filtered_candidates)
    if not ranking.get("success"):
        TASK_RESULT["physical_motion_executed"] = False
        TASK_RESULT["movement_capable_calls"] = []
        raise RuntimeError(f"No preview-feasible AnyGrasp candidate: {ranking}")

    selected = dict(ranking["selected_grasp"])
    side = str(selected["arm"])
    grasp = [float(x) for x in selected["position"]]
    rpy = [float(x) for x in selected["rpy"]]
    width = float(selected.get("width", 0.08))
    lift = [grasp[0], grasp[1], grasp[2] + PHYSICAL_LIFT_Z_M]
    release = list(grasp)

    attempt: dict[str, Any] = {
        "stage": "stage1_anygrasp_physical_dry_grasp_lift",
        "source": "real_anygrasp",
        "anygrasp_health": health,
        "anygrasp_candidate_count": len(candidates),
        "filtered_anygrasp_candidate_count": len(filtered_candidates),
        "stage1_side_grasp_filter": filter_info,
        "ranking": ranking,
        "side": side,
        "grasp_pose": {"position": _round_list(grasp), "rpy": _round_list(rpy, 3)},
        "lift_pose": {"position": _round_list(lift), "rpy": _round_list(rpy, 3)},
        "release_pose": {"position": _round_list(release), "rpy": _round_list(rpy, 3)},
        "gripper_width_m": round(width, 5),
        "hold_s": PHYSICAL_HOLD_S,
        "moves": [],
        "gripper": {},
        "recovery": [],
        "success": False,
        "movement_capable_calls": [],
    }

    try:
        open_gripper(side)
        attempt["movement_capable_calls"].append("open_gripper")
        attempt["gripper"]["opened"] = True

        cache_key = selected.get("trajectory_cache_key")
        if cache_key:
            result = freespace_move(preview_only=False, trajectory_cache_key=cache_key)
            move_summary = _result_summary(result)
            move_summary.update(
                {
                    "label": "move_anygrasp_cached_grasp",
                    "position": _round_list(grasp),
                    "rpy": _round_list(rpy, 3),
                    "preview_only": False,
                    "trajectory_cache_key": cache_key,
                }
            )
            if move_summary["status"] != "Success" or not move_summary["executed"]:
                raise RuntimeError(f"cached AnyGrasp grasp move failed: {move_summary}")
            attempt["moves"].append(move_summary)
        else:
            attempt["moves"].append(
                _move_pose(
                    freespace_move,
                    side,
                    grasp,
                    rpy,
                    "move_anygrasp_grasp",
                    preview_only=False,
                    gripper_collision_width=width,
                )
            )
        attempt["movement_capable_calls"].append("freespace_move")

        close_gripper(side)
        attempt["movement_capable_calls"].append("close_gripper")
        time.sleep(0.4)
        closed_width = _gripper_pos(get_robot_state(), side)
        attempt["gripper"]["closed_width_m"] = round(closed_width, 5)
        if closed_width < PHYSICAL_GRIPPER_MIN_WIDTH_M:
            raise RuntimeError(
                f"Gripper width {closed_width:.5f}m after close is below "
                f"minimum {PHYSICAL_GRIPPER_MIN_WIDTH_M:.5f}m; likely no object."
            )

        attempt["moves"].append(
            _move_pose(
                freespace_move,
                side,
                lift,
                rpy,
                "move_lift",
                preview_only=False,
                gripper_collision_width=width,
            )
        )
        attempt["movement_capable_calls"].append("freespace_move")
        time.sleep(max(0.0, PHYSICAL_HOLD_S))
        attempt["gripper"]["hold_width_m"] = round(_gripper_pos(get_robot_state(), side), 5)

        attempt["moves"].append(
            _move_pose(
                freespace_move,
                side,
                release,
                rpy,
                "move_release_pose",
                preview_only=False,
                gripper_collision_width=width,
            )
        )
        attempt["movement_capable_calls"].append("freespace_move")
        open_gripper(side)
        attempt["movement_capable_calls"].append("open_gripper")
        attempt["gripper"]["released"] = True
        go_home()
        attempt["movement_capable_calls"].append("go_home")
        attempt["recovery"].append("go_home_after_release")
        attempt["success"] = True
        return attempt
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        try:
            open_gripper(side)
            attempt["movement_capable_calls"].append("open_gripper")
            attempt["recovery"].append("open_gripper_after_error")
        except Exception as open_exc:
            attempt["recovery"].append(f"open_gripper_failed:{type(open_exc).__name__}:{open_exc}")
        try:
            go_home()
            attempt["movement_capable_calls"].append("go_home")
            attempt["recovery"].append("go_home_after_error")
        except Exception as home_exc:
            attempt["recovery"].append(f"go_home_failed:{type(home_exc).__name__}:{home_exc}")
        TASK_RESULT["physical_attempt"] = attempt
        TASK_RESULT["physical_motion_executed"] = bool(attempt.get("movement_capable_calls"))
        TASK_RESULT["movement_capable_calls"] = list(attempt.get("movement_capable_calls", []))
        TASK_RESULT["why_stopped"] = "Stage 1 AnyGrasp physical dry attempt failed; recovery was attempted."
        raise RuntimeError(f"Stage 1 AnyGrasp physical attempt failed: {attempt}") from exc


def _geometric_pregrasp_position(selected: dict[str, Any], grasp: list[float]) -> list[float]:
    pregrasp = selected.get("pregrasp_position")
    if GEOM_GUARDED_USE_CANDIDATE_PREGRASP and isinstance(pregrasp, list) and len(pregrasp) == 3:
        return [float(x) for x in pregrasp]
    approach = selected.get("approach_direction_world") or [0.0, -1.0, 0.0]
    norm = math.sqrt(sum(float(component) * float(component) for component in approach))
    if norm < 1e-6:
        raise RuntimeError(f"Invalid geometric approach direction: {approach!r}")
    approach = [float(component) / norm for component in approach]
    standoff = _clip(
        float(selected.get("guarded_pregrasp_standoff_m", GEOM_GUARDED_PREGRASP_STANDOFF_M)),
        0.0,
        0.12,
    )
    return [
        float(grasp[0]) - float(approach[0]) * standoff,
        float(grasp[1]) - float(approach[1]) * standoff,
        float(grasp[2]) - float(approach[2]) * standoff,
    ]


def _guarded_pregrasp_standoff_values() -> list[float]:
    values = _parse_float_offsets(
        GEOM_GUARDED_PREGRASP_STANDOFFS_M,
        [GEOM_GUARDED_PREGRASP_STANDOFF_M],
    )
    values.append(float(GEOM_GUARDED_PREGRASP_STANDOFF_M))
    cleaned: list[float] = []
    seen: set[float] = set()
    for value in values:
        clipped = round(_clip(float(value), 0.0, 0.12), 4)
        if clipped in seen:
            continue
        seen.add(clipped)
        cleaned.append(clipped)
    cleaned.sort()
    return cleaned or [round(_clip(float(GEOM_GUARDED_PREGRASP_STANDOFF_M), 0.0, 0.12), 4)]


def _build_geometric_guarded_stage1_attempt(
    selected: dict[str, Any],
    ranking: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    _validate_physical_grasp_adjustments()
    side = str(selected["arm"])
    nominal_grasp = [float(x) for x in selected["position"]]
    rpy = [float(x) for x in selected["rpy"]]
    width = float(selected.get("width", geometry.get("gripper_width_m", 0.09)))
    approach = selected.get("approach_direction_world") or [0.0, -1.0, 0.0]
    norm = math.sqrt(sum(float(component) * float(component) for component in approach))
    if norm < 1e-6:
        raise RuntimeError(f"Invalid geometric approach direction: {approach!r}")
    approach = [float(component) / norm for component in approach]
    grasp = [
        nominal_grasp[0] + approach[0] * PHYSICAL_GRASP_ADVANCE_M,
        nominal_grasp[1] + approach[1] * PHYSICAL_GRASP_ADVANCE_M,
        nominal_grasp[2] + approach[2] * PHYSICAL_GRASP_ADVANCE_M + PHYSICAL_GRASP_Z_OFFSET_M,
    ]
    pregrasp = _geometric_pregrasp_position(selected, grasp)
    lift = [grasp[0], grasp[1], grasp[2] + PHYSICAL_LIFT_Z_M]
    release = list(grasp)
    return {
        "stage": "stage1_geom_depth_physical_dry_grasp_lift",
        "source": "bundlesdf_sam3_depth_geometry",
        "execution_mode": "guarded_pregrasp_then_grasp",
        "geometry": geometry,
        "ranking": ranking,
        "side": side,
        "selected_grasp": selected,
        "candidate_index": ranking.get("selected_candidate_index")
        or selected.get("candidate_index"),
        "nominal_grasp_pose": {
            "position": _round_list(nominal_grasp),
            "rpy": _round_list(rpy, 3),
        },
        "approach_direction_world": _round_list(approach),
        "pregrasp_pose": {"position": _round_list(pregrasp), "rpy": _round_list(rpy, 3)},
        "grasp_pose": {"position": _round_list(grasp), "rpy": _round_list(rpy, 3)},
        "lift_pose": {"position": _round_list(lift), "rpy": _round_list(rpy, 3)},
        "release_pose": {"position": _round_list(release), "rpy": _round_list(rpy, 3)},
        "gripper_width_m": round(width, 5),
        "hold_s": PHYSICAL_HOLD_S,
        "physical_adjustments": {
            "lift_z_m": round(PHYSICAL_LIFT_Z_M, 4),
            "advance_along_approach_m": round(PHYSICAL_GRASP_ADVANCE_M, 4),
            "z_offset_m": round(PHYSICAL_GRASP_Z_OFFSET_M, 4),
            "advance_limit_m": round(PHYSICAL_GRASP_ADVANCE_LIMIT_M, 4),
            "z_offset_limit_m": round(PHYSICAL_GRASP_Z_OFFSET_LIMIT_M, 4),
            "ik_position_threshold_m": round(PHYSICAL_IK_ERROR_THRESHOLD_M, 4),
            "ik_rot_threshold_deg": round(PHYSICAL_IK_ROT_ERROR_THRESHOLD_DEG, 3),
            "direct_batch_cache_execution": False,
            "direct_batch_cache_disabled_reason": (
                "2026-05-25 attempt 8 knocked the bottle down after direct cached execution"
            ),
            "guarded_pregrasp_standoff_m": round(
                _clip(
                    float(
                        selected.get(
                            "guarded_pregrasp_standoff_m",
                            GEOM_GUARDED_PREGRASP_STANDOFF_M,
                        )
                    ),
                    0.0,
                    0.12,
                ),
                4,
            ),
            "guarded_uses_candidate_pregrasp": bool(GEOM_GUARDED_USE_CANDIDATE_PREGRASP),
        },
        "moves": [],
        "previews": [],
        "gripper": {},
        "recovery": [],
        "success": False,
        "movement_capable_calls": [],
    }


def _preview_geometric_guarded_stage1_attempt(
    freespace_move,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    side = str(attempt["side"])
    rpy = [float(x) for x in attempt["grasp_pose"]["rpy"]]
    width = float(attempt["gripper_width_m"])
    previews: list[dict[str, Any]] = []
    try:
        for label, pose_key in (
            ("preview_guarded_pregrasp", "pregrasp_pose"),
            ("preview_guarded_grasp", "grasp_pose"),
            ("preview_guarded_lift", "lift_pose"),
        ):
            previews.append(
                _move_pose(
                    freespace_move,
                    side,
                    [float(x) for x in attempt[pose_key]["position"]],
                    rpy,
                    label,
                    preview_only=True,
                    gripper_collision_width=width,
                )
            )
        return {
            "stage": "stage1_geometric_guarded_pregrasp_preview",
            "success": True,
            "physical_motion_executed": False,
            "execution_mode": attempt.get("execution_mode"),
            "side": side,
            "previews": previews,
        }
    except Exception as exc:
        return {
            "stage": "stage1_geometric_guarded_pregrasp_preview",
            "success": False,
            "physical_motion_executed": False,
            "execution_mode": attempt.get("execution_mode"),
            "side": side,
            "previews": previews,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_stage1_geometric_depth_batch_physical_dry_grasp_lift(
    selected: dict[str, Any],
    ranking: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    _require_physical_gate()
    freespace_move = _required_tool("freespace_move")
    open_gripper = _required_tool("open_gripper")
    close_gripper = _required_tool("close_gripper")
    get_robot_state = _required_tool("get_robot_state")
    go_home = _required_tool("go_home")

    attempt = _build_geometric_guarded_stage1_attempt(selected, ranking, geometry)
    side = str(attempt["side"])
    rpy = [float(x) for x in attempt["grasp_pose"]["rpy"]]
    width = float(attempt["gripper_width_m"])

    try:
        guarded_preview = _preview_geometric_guarded_stage1_attempt(
            freespace_move,
            attempt,
        )
        attempt["previews"] = guarded_preview.get("previews", [])
        if not guarded_preview.get("success"):
            raise RuntimeError(f"guarded pregrasp/grasp/lift preview failed: {guarded_preview}")

        open_gripper(side)
        attempt["movement_capable_calls"].append("open_gripper")
        attempt["gripper"]["opened"] = True

        for label, pose_key in (
            ("move_geometric_pregrasp", "pregrasp_pose"),
            ("move_geometric_grasp", "grasp_pose"),
        ):
            attempt["moves"].append(
                _move_pose(
                    freespace_move,
                    side,
                    [float(x) for x in attempt[pose_key]["position"]],
                    rpy,
                    label,
                    preview_only=False,
                    gripper_collision_width=width,
                )
            )
            attempt["movement_capable_calls"].append("freespace_move")

        close_gripper(side)
        attempt["movement_capable_calls"].append("close_gripper")
        time.sleep(0.4)
        closed_width = _gripper_pos(get_robot_state(), side)
        attempt["gripper"]["closed_width_m"] = round(closed_width, 5)
        if closed_width < PHYSICAL_GRIPPER_MIN_WIDTH_M:
            raise RuntimeError(
                f"Gripper width {closed_width:.5f}m after close is below "
                f"minimum {PHYSICAL_GRIPPER_MIN_WIDTH_M:.5f}m; likely no object."
            )

        attempt["moves"].append(
            _move_pose(
                freespace_move,
                side,
                [float(x) for x in attempt["lift_pose"]["position"]],
                rpy,
                "move_lift",
                preview_only=False,
                gripper_collision_width=width,
            )
        )
        attempt["movement_capable_calls"].append("freespace_move")
        time.sleep(max(0.0, PHYSICAL_HOLD_S))
        attempt["gripper"]["hold_width_m"] = round(_gripper_pos(get_robot_state(), side), 5)

        attempt["moves"].append(
            _move_pose(
                freespace_move,
                side,
                [float(x) for x in attempt["release_pose"]["position"]],
                rpy,
                "move_release_pose",
                preview_only=False,
                gripper_collision_width=width,
            )
        )
        attempt["movement_capable_calls"].append("freespace_move")
        open_gripper(side)
        attempt["movement_capable_calls"].append("open_gripper")
        attempt["gripper"]["released"] = True
        go_home()
        attempt["movement_capable_calls"].append("go_home")
        attempt["recovery"].append("go_home_after_release")
        attempt["success"] = True
        return attempt
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        try:
            open_gripper(side)
            attempt["movement_capable_calls"].append("open_gripper")
            attempt["recovery"].append("open_gripper_after_error")
        except Exception as open_exc:
            attempt["recovery"].append(f"open_gripper_failed:{type(open_exc).__name__}:{open_exc}")
        try:
            go_home()
            attempt["movement_capable_calls"].append("go_home")
            attempt["recovery"].append("go_home_after_error")
        except Exception as home_exc:
            attempt["recovery"].append(f"go_home_failed:{type(home_exc).__name__}:{home_exc}")
        TASK_RESULT["physical_attempt"] = attempt
        TASK_RESULT["physical_motion_executed"] = bool(attempt.get("movement_capable_calls"))
        TASK_RESULT["movement_capable_calls"] = list(attempt.get("movement_capable_calls", []))
        TASK_RESULT["why_stopped"] = "Stage 1 geometric depth physical dry attempt failed; recovery was attempted."
        raise RuntimeError(f"Stage 1 geometric depth physical attempt failed: {attempt}") from exc


def _run_stage1_physical_dry_grasp_lift(
    selected_grasp: dict[str, Any],
    estimates: dict[str, Any],
    grasp_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _require_physical_gate()
    freespace_move = _required_tool("freespace_move")
    open_gripper = _required_tool("open_gripper")
    close_gripper = _required_tool("close_gripper")
    get_robot_state = _required_tool("get_robot_state")
    go_home = _required_tool("go_home")

    attempt = _select_preview_feasible_stage1_attempt(
        freespace_move,
        selected_grasp,
        grasp_candidates,
    )
    side = str(attempt["side"])
    rpy = [float(x) for x in attempt["grasp_pose"]["rpy"]]
    pregrasp = [float(x) for x in attempt["pregrasp_pose"]["position"]]
    grasp = [float(x) for x in attempt["grasp_pose"]["position"]]
    lift = [float(x) for x in attempt["lift_pose"]["position"]]
    release = [float(x) for x in attempt["release_pose"]["position"]]
    width = float(attempt["gripper_width_m"])

    try:
        open_gripper(side)
        attempt["movement_capable_calls"] = ["open_gripper"]
        attempt["gripper"]["opened"] = True

        for label, pos in (
            ("move_pregrasp", pregrasp),
            ("move_grasp", grasp),
        ):
            attempt["moves"].append(
                _move_pose(
                    freespace_move,
                    side,
                    pos,
                    rpy,
                    label,
                    preview_only=False,
                    gripper_collision_width=width,
                )
            )
            attempt["movement_capable_calls"].append("freespace_move")

        close_gripper(side)
        attempt["movement_capable_calls"].append("close_gripper")
        time.sleep(0.4)
        closed_width = _gripper_pos(get_robot_state(), side)
        attempt["gripper"]["closed_width_m"] = round(closed_width, 5)
        if closed_width < PHYSICAL_GRIPPER_MIN_WIDTH_M:
            raise RuntimeError(
                f"Gripper width {closed_width:.5f}m after close is below "
                f"minimum {PHYSICAL_GRIPPER_MIN_WIDTH_M:.5f}m; likely no object."
            )

        attempt["moves"].append(
            _move_pose(
                freespace_move,
                side,
                lift,
                rpy,
                "move_lift",
                preview_only=False,
                gripper_collision_width=width,
            )
        )
        attempt["movement_capable_calls"].append("freespace_move")

        time.sleep(max(0.0, PHYSICAL_HOLD_S))
        hold_width = _gripper_pos(get_robot_state(), side)
        attempt["gripper"]["hold_width_m"] = round(hold_width, 5)

        for label, pos in (
            ("move_release_pose", release),
            ("move_retreat", pregrasp),
        ):
            if label == "move_release_pose":
                attempt["moves"].append(
                    _move_pose(
                        freespace_move,
                        side,
                        pos,
                        rpy,
                        label,
                        preview_only=False,
                        gripper_collision_width=width,
                    )
                )
                attempt["movement_capable_calls"].append("freespace_move")
                open_gripper(side)
                attempt["movement_capable_calls"].append("open_gripper")
                attempt["gripper"]["released"] = True
            else:
                attempt["moves"].append(
                    _move_pose(
                        freespace_move,
                        side,
                        pos,
                        rpy,
                        label,
                        preview_only=False,
                        gripper_collision_width=width,
                    )
                )
                attempt["movement_capable_calls"].append("freespace_move")

        go_home()
        attempt["movement_capable_calls"].append("go_home")
        attempt["recovery"].append("go_home_after_release")
        attempt["success"] = True
        return attempt
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        try:
            open_gripper(side)
            attempt.setdefault("movement_capable_calls", []).append("open_gripper")
            attempt["recovery"].append("open_gripper_after_error")
        except Exception as open_exc:
            attempt["recovery"].append(f"open_gripper_failed:{type(open_exc).__name__}:{open_exc}")
        try:
            go_home()
            attempt.setdefault("movement_capable_calls", []).append("go_home")
            attempt["recovery"].append("go_home_after_error")
        except Exception as home_exc:
            attempt["recovery"].append(f"go_home_failed:{type(home_exc).__name__}:{home_exc}")
        TASK_RESULT["physical_attempt"] = attempt
        TASK_RESULT["physical_motion_executed"] = bool(attempt.get("movement_capable_calls"))
        TASK_RESULT["movement_capable_calls"] = list(attempt.get("movement_capable_calls", []))
        TASK_RESULT["why_stopped"] = "Stage 1 physical dry attempt failed; recovery was attempted."
        raise RuntimeError(f"Stage 1 physical attempt failed: {attempt}") from exc


def _main() -> None:
    print("[pour_bottle_to_green_cup] Starting no-motion dry planning.")
    print(f"[pour_bottle_to_green_cup] stage={REQUESTED_STAGE!r}, camera={CAMERA!r}")
    print(f"[pour_bottle_to_green_cup] prompts: bottle={BOTTLE_PROMPT!r}, cup={CUP_PROMPT!r}")

    stage = _normalize_stage(REQUESTED_STAGE)
    TASK_RESULT["implemented_stage"] = stage or "unsupported"
    if _truthy_env("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
        if stage in MOVEMENT_CAPABLE_STAGES:
            TASK_RESULT["risk_notes"].append(
                "OPENFORGE_ALLOW_PHYSICAL_MOTION is set for a movement-capable "
                "Stage 1 dry physical validation path."
            )
        else:
            TASK_RESULT["risk_notes"].append(
                "OPENFORGE_ALLOW_PHYSICAL_MOTION is set, but the requested stage "
                "does not execute movement-capable calls."
            )

    if stage is None:
        TASK_RESULT["why_stopped"] = (
            "Only Stage 1 dry side-grasp planning and Stage 2 dry pour-path "
            "planning are implemented as no-motion outputs. Stage 3 tiny-water "
            "stationary-cup pour and Stage 4 optional bimanual cup-hold are "
            "locked future stages."
        )
        TASK_RESULT["risk_notes"].append("Requested stage is not implemented in this no-motion iteration.")
        print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
        print(json.dumps(TASK_RESULT, indent=2, default=str))
        return

    bottle, cup, detection_source, perception_errors = _detect_read_only()
    TASK_RESULT["detection_source"] = detection_source
    TASK_RESULT["detected_bottle"] = bottle
    TASK_RESULT["detected_cup"] = cup
    TASK_RESULT["perception_errors"] = perception_errors

    if bottle is None:
        TASK_RESULT["why_stopped"] = (
            "No bottle pose available. Provide OPENFORGE_BOTTLE_XYZ for offline "
            "planning or run with OPENFORGE_ENABLE_READ_ONLY_PERCEPTION=1 and "
            "OPENFORGE_PERCEPTION_BACKEND=bundlesdf_http against the read-only "
            "BundleSDF service."
        )
        TASK_RESULT["risk_notes"].append("Stage 1 needs bottle localization before any physical retry.")
        TASK_RESULT["visual_artifacts"].extend(
            _save_detection_plan_overlay(
                stage=f"{stage}_perception_failed",
                bottle=bottle,
                cup=cup,
                selected=None,
                candidates=[],
                estimates=None,
            )
        )
        print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
        print(json.dumps(TASK_RESULT, indent=2, default=str))
        return

    if stage in GEOMETRIC_DEPTH_STAGES:
        try:
            _run_geometric_depth_stage(stage=stage, bottle=bottle, cup=cup)
        except Exception as exc:
            TASK_RESULT["success"] = False
            TASK_RESULT["reward"] = 0.0
            TASK_RESULT["why_stopped"] = (
                "Stage 1 geometric depth path stopped before completing the "
                f"requested stage: {type(exc).__name__}: {exc}"
            )
            TASK_RESULT["risk_notes"].append(
                "Do not consume another physical attempt until the geometric "
                "depth failure is understood."
            )
        TASK_RESULT["risk_notes"].extend(
            [
                "Geometric side-grasp RPY values are tentative and must pass "
                "no-motion planner checks before any motion.",
                "AnyGrasp may be used only as optional debug/comparison for this task.",
                "Do not run Stage 2 tilt, liquid, Stage 3, or Stage 4 from this Stage 1 path.",
            ]
        )
        print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
        print(json.dumps(TASK_RESULT, indent=2, default=str))
        return

    if bottle.get("half_extents"):
        TASK_RESULT["geometry_notes"].append(
            "Using detection/manual half_extents for mid-body grasp height and bottle radius."
        )
    else:
        TASK_RESULT["geometry_notes"].append(
            "No OBB, point cloud, or half_extents available; using detection center and "
            "default bottle radius."
        )
    TASK_RESULT["geometry_notes"].append(
        "Stage 1 does not estimate bottle mouth pose or compute a point-cloud OBB."
    )

    selected_arm, selected_arm_reason, candidates = _side_grasp_candidates(bottle)
    selected = candidates[0] if candidates else None
    TASK_RESULT.update(
        {
            "selected_arm": selected_arm,
            "selected_arm_reason": selected_arm_reason,
            "grasp_candidates": candidates,
            "selected_grasp_pose": selected,
        }
    )
    if selected is not None:
        estimates = _pose_estimates(bottle, cup, selected)
        if stage == "stage2_dry_pour_planning_only":
            if cup is None:
                TASK_RESULT["why_stopped"] = (
                    "Stage 2 dry pour-path planning needs cup localization. "
                    "Stopped before planner, arm, gripper, lift, pour, or liquid actions."
                )
                TASK_RESULT["risk_notes"].append("Stage 2 needs cup localization before any dry motion trial.")
                TASK_RESULT["target_pose_estimates"] = estimates
                print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
                print(json.dumps(TASK_RESULT, indent=2, default=str))
                return
            estimates["stage2_dry_pour_motion_plan"] = _stage2_dry_pour_plan(estimates, selected)
        TASK_RESULT["target_pose_estimates"] = estimates
        TASK_RESULT["visual_artifacts"].extend(
            _save_detection_plan_overlay(
                stage=stage,
                bottle=bottle,
                cup=cup,
                selected=selected,
                candidates=candidates,
                estimates=estimates,
            )
        )
        if stage == "stage1_anygrasp_preview_only":
            preview = _run_stage1_anygrasp_preview_only()
            TASK_RESULT["anygrasp_preview"] = preview
            TASK_RESULT["physical_motion_executed"] = False
            TASK_RESULT["movement_capable_calls"] = []
            if not preview.get("success"):
                TASK_RESULT["why_stopped"] = (
                    "Stage 1 real-AnyGrasp preview found no feasible candidate; "
                    "no gripper, arm motion, lift, pour, or liquid actions were run."
                )
                print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
                print(json.dumps(TASK_RESULT, indent=2, default=str))
                return
        if stage == "stage1_planner_preview_only":
            preview = _run_stage1_planner_preview_only(selected, candidates)
            TASK_RESULT["planner_preview"] = preview
            TASK_RESULT["physical_motion_executed"] = False
            TASK_RESULT["movement_capable_calls"] = []
            if not preview.get("success"):
                TASK_RESULT["why_stopped"] = (
                    "Stage 1 planner preview found no feasible side-grasp "
                    "candidate; no gripper, arm motion, lift, pour, or liquid "
                    "actions were run."
                )
                print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
                print(json.dumps(TASK_RESULT, indent=2, default=str))
                return
        if stage == "stage1_anygrasp_physical_dry_grasp_lift":
            attempt = _run_stage1_anygrasp_physical_dry_grasp_lift()
            TASK_RESULT["physical_attempt"] = attempt
            TASK_RESULT["physical_motion_executed"] = True
            TASK_RESULT["movement_capable_calls"] = list(
                attempt.get("movement_capable_calls", [])
            )
        if stage == "stage1_physical_dry_grasp_lift":
            attempt = _run_stage1_physical_dry_grasp_lift(selected, estimates, candidates)
            TASK_RESULT["physical_attempt"] = attempt
            TASK_RESULT["physical_motion_executed"] = True
            TASK_RESULT["movement_capable_calls"] = list(
                attempt.get("movement_capable_calls", [])
            )
        TASK_RESULT["success"] = True
        TASK_RESULT["reward"] = 1.0
        if stage == "stage1_physical_dry_grasp_lift":
            TASK_RESULT["why_stopped"] = (
                "Stage 1 physical dry grasp/lift/release/home attempt completed "
                "within the ticket scope."
            )
        elif stage == "stage1_anygrasp_physical_dry_grasp_lift":
            TASK_RESULT["why_stopped"] = (
                "Stage 1 real-AnyGrasp physical dry grasp/lift/release/home "
                "attempt completed within the ticket scope."
            )
        elif stage == "stage1_anygrasp_preview_only":
            TASK_RESULT["why_stopped"] = (
                "Stage 1 real-AnyGrasp preview completed; no gripper, arm "
                "motion, lift, pour, or liquid actions were run."
            )
        elif stage == "stage1_planner_preview_only":
            TASK_RESULT["why_stopped"] = (
                "Stage 1 planner preview completed; no gripper, arm motion, "
                "lift, pour, or liquid actions were run."
            )
        elif stage == "stage2_dry_pour_planning_only":
            TASK_RESULT["why_stopped"] = (
                "Stage 2 dry pour-path plan generated; stopped before planner, arm, "
                "gripper, lift, pour, or liquid actions."
            )
        else:
            TASK_RESULT["why_stopped"] = (
                "Stage 1 dry side-grasp plan generated; stopped before planner, arm, "
                "gripper, lift, pour, or liquid actions."
            )
    else:
        TASK_RESULT["why_stopped"] = "No side-grasp candidates generated from bottle pose."

    TASK_RESULT["risk_notes"].extend(
        [
            "Side-grasp RPY values are tentative and must pass no-motion planner checks before any motion.",
            "Use an empty or sealed bottle for Stage 2 dry pour motion.",
            "Do not introduce tiny water until stationary-cup dry behavior is reliable.",
            "Do not attempt bimanual cup holding until stationary-cup pouring works.",
        ]
    )
    print(f"[pour_bottle_to_green_cup] {TASK_RESULT['why_stopped']}")
    print(json.dumps(TASK_RESULT, indent=2, default=str))


_main()
