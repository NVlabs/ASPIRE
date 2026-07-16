"""Safe smoke test for standardized observation/debug artifacts.

Default mode is synthetic and read-only. Set
``OPENFORGE_DEBUG_OBS_USE_TOOLS=1`` only when the station's read-only camera and
perception services are intentionally available.
"""

from __future__ import annotations

import os
from typing import Any

from skill_library.debug_observation import capture_observation, current_run_dir


PROMPTS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_DEBUG_OBS_PROMPTS", "green apple,basket").split(",")
    if part.strip()
]
CAMERAS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_DEBUG_OBS_CAMERAS", "top,left,right,bottom").split(",")
    if part.strip()
]
IMAGE_ONLY_CAMERAS = [
    part.strip()
    for part in os.environ.get("OPENFORGE_DEBUG_OBS_IMAGE_ONLY_CAMERAS", "bottom").split(",")
    if part.strip()
]
USE_TOOLS = os.environ.get("OPENFORGE_DEBUG_OBS_USE_TOOLS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CAPTURE_STATE = os.environ.get("OPENFORGE_DEBUG_OBS_CAPTURE_STATE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CAMERA_PORTAL = os.environ.get("OPENFORGE_DEBUG_OBS_CAMERA_PORTAL", "").strip()
TIMEOUT_S = os.environ.get("OPENFORGE_DEBUG_OBS_TIMEOUT_S", "12").strip()

TASK_RESULT: dict[str, Any] = {
    "success": False,
    "reward": 0.0,
    "method": "debug_observation_smoke",
    "safe_read_only": True,
    "use_tools": USE_TOOLS,
    "capture_state": CAPTURE_STATE,
    "camera_portal": CAMERA_PORTAL or None,
    "timeout_s": TIMEOUT_S,
    "prompts": PROMPTS,
    "cameras": CAMERAS,
    "image_only_cameras": IMAGE_ONLY_CAMERAS,
}


def get_task_info() -> dict[str, Any]:
    return TASK_RESULT


def _synthetic_rgb(camera: str, height: int = 480, width: int = 640) -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = 80
    if camera == "left":
        rgb[..., 2] = np.clip((1.0 - x) * 255.0, 0, 255).astype(np.uint8)
    elif camera == "right":
        rgb[..., 0] = 70
        rgb[..., 2] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    elif camera == "bottom":
        rgb[..., 0] = np.clip((1.0 - y) * 255.0, 0, 255).astype(np.uint8)
        rgb[..., 1] = 90
    rgb[210:270, 270:370] = [60, 210, 80]
    rgb[120:220, 390:520] = [210, 180, 90]
    return rgb


def _synthetic_camera(camera: str = "top") -> Any | None:
    return _synthetic_rgb(camera)


def _synthetic_detect(prompts: Any, camera: str = "top") -> dict[str, list[dict[str, Any]]]:
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = [str(prompt) for prompt in prompts]
    offset = {"top": 0.0, "left": -0.03, "right": 0.03, "bottom": 0.015}.get(
        camera, 0.0
    )
    detections: dict[str, list[dict[str, Any]]] = {}
    for prompt in prompt_list:
        lower = prompt.lower()
        if "apple" in lower:
            detections[prompt] = [
                {
                    "label": prompt,
                    "score": 0.91,
                    "box_2d": [270, 210, 370, 270],
                    "position_3d": [0.53 + offset, 0.19, 0.82],
                }
            ]
        elif "basket" in lower:
            detections[prompt] = [
                {
                    "label": prompt,
                    "score": 0.88,
                    "box_2d": [390, 120, 520, 220],
                    "position_3d": [0.84 + offset, -0.14, 0.87],
                }
            ]
        else:
            detections[prompt] = []
    return detections


def _synthetic_state() -> dict[str, Any]:
    return {
        "source": "synthetic",
        "left_gripper_pos": 1.0,
        "right_gripper_pos": 1.0,
        "left_ee_pos": [0.35, 0.28, 0.9],
        "right_ee_pos": [0.35, -0.28, 0.9],
    }


def _portal_camera(camera: str = "top") -> Any:
    import portal

    return portal.Client(CAMERA_PORTAL).get_camera_image(camera).result()


print("[debug_observation_smoke] Starting safe observation artifact smoke test.")
print(f"[debug_observation_smoke] use_tools={USE_TOOLS}")
print(f"[debug_observation_smoke] capture_state={CAPTURE_STATE}")
print(f"[debug_observation_smoke] camera_portal={CAMERA_PORTAL or None}")
print(f"[debug_observation_smoke] timeout_s={TIMEOUT_S}")
print(f"[debug_observation_smoke] prompts={PROMPTS}")
print(f"[debug_observation_smoke] cameras={CAMERAS}")
print(f"[debug_observation_smoke] image_only_cameras={IMAGE_ONLY_CAMERAS}")
print(f"[debug_observation_smoke] run_dir={current_run_dir()}")
if os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION"):
    print("[debug_observation_smoke] OPENFORGE_ALLOW_PHYSICAL_MOTION ignored; this script has no motion calls.")

if USE_TOOLS:
    packet = capture_observation(
        stage="debug_smoke",
        prompts=PROMPTS,
        cameras=CAMERAS,
        image_only_cameras=IMAGE_ONLY_CAMERAS,
        get_camera_fn=_portal_camera if CAMERA_PORTAL else None,
        capture_robot_state=CAPTURE_STATE,
        per_call_timeout_s=float(TIMEOUT_S),
    )
else:
    packet = capture_observation(
        stage="debug_smoke",
        prompts=PROMPTS,
        cameras=CAMERAS,
        image_only_cameras=IMAGE_ONLY_CAMERAS,
        detect_fn=_synthetic_detect,
        get_camera_fn=_synthetic_camera,
        get_robot_state_fn=_synthetic_state,
        capture_robot_state=True,
        per_call_timeout_s=float(TIMEOUT_S),
    )

TASK_RESULT.update(
    {
        "success": True,
        "details": {
            "packet_path": packet.get("packet_path"),
            "observation_schema": packet.get("schema"),
            "camera_count": len(packet.get("cameras", {})),
            "stage_summary": str(current_run_dir() / "stage_summary.md"),
        },
    }
)
print("[debug_observation_smoke] Done.")
