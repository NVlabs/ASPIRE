"""No-motion smoke test for cap.saved_scripts.yam_runtime."""

from __future__ import annotations

from pathlib import Path
import sys

_THIS_FILE = globals().get("__file__")
_ROOT = Path(_THIS_FILE).resolve().parents[2] if _THIS_FILE else Path.cwd()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cap.saved_scripts.yam_runtime import (
    capture_scene,
    execute_grasp_lift_attempt,
    generate_side_grasp_candidates,
    rank_motion_candidates,
    verify_lift,
)


TASK_RESULT = {
    "success": False,
    "reward": 0.0,
    "physical_motion_executed": False,
    "method": "yam_runtime_smoke",
}


def get_task_info() -> dict:
    return dict(TASK_RESULT)


def _fake_detect(query, camera="top", max_retries=1):
    prompts = query if isinstance(query, list) else [query]
    return {
        prompt: [
            {
                "label": prompt,
                "score": 0.9,
                "box_2d": [100, 100, 140, 180],
                "position_3d": [0.70, 0.08, 0.94],
                "half_extents": [0.035, 0.035, 0.13],
            }
        ]
        for prompt in prompts
    }


def _fake_state():
    return {
        "arms": {
            "left": {"ee_pos": [0.5, 0.31, 0.91], "gripper_pos": _GRIPPER_STATE["left"]},
            "right": {"ee_pos": [0.5, -0.31, 0.91], "gripper_pos": 1.0},
        }
    }


_GRIPPER_STATE = {"left": 1.0}
_MOVE_CALLS: list[dict] = []


def _fake_freespace_move(**kwargs):
    _MOVE_CALLS.append(dict(kwargs))
    return {
        "status": "Success",
        "executed": bool(not kwargs.get("preview_only", False)),
        "final_pos_error_m": 0.0,
        "final_rot_error_deg": 0.0,
        "request": kwargs,
    }


def _fake_open_gripper(side):
    _GRIPPER_STATE[side] = 1.0
    return {"success": True, "side": side, "target": 1.0}


def _fake_set_gripper(side, target, **kwargs):
    _GRIPPER_STATE[side] = float(target)
    return {"success": True, "side": side, "target": float(target), "kwargs": kwargs}


run_dir = Path("logs") / "yam_runtime_smoke"
scene = capture_scene(
    prompts=["bottle"],
    cameras=["top"],
    detect_objects_oneshot=_fake_detect,
    get_robot_state=_fake_state,
    run_dir=run_dir,
    task_name="yam_runtime_smoke",
    save_overlays=False,
)
det = scene["best_detection"]
candidates = generate_side_grasp_candidates(det, object_kind="bottle")
plan = rank_motion_candidates(
    candidates=candidates,
    freespace_move=_fake_freespace_move,
    run_dir=run_dir,
    task_name="yam_runtime_smoke",
)
verification = verify_lift(det, {**det, "position_3d": [0.70, 0.08, 1.01]})
execution = execute_grasp_lift_attempt(
    plan=plan["selected"],
    freespace_move=_fake_freespace_move,
    open_gripper=_fake_open_gripper,
    set_gripper=_fake_set_gripper,
    get_robot_state=_fake_state,
    physical_gate=True,
    run_dir=run_dir,
    task_name="yam_runtime_smoke",
    gripper_target=0.48,
    gripper_steps=(0.78, 0.48),
    require_contact_for_lift=False,
    preview_before_each_move=False,
)
lift_move = next((call for call in reversed(_MOVE_CALLS) if call.get("left_target_pos")), {})
lift_preserved_close = "left_gripper" not in lift_move

TASK_RESULT.update(
    {
        "success": bool(
            scene["best_detection"]
            and plan["selected"]
            and verification["success"]
            and execution["success"]
            and lift_preserved_close
        ),
        "reward": 1.0,
        "scene_packet": scene.get("packet_path"),
        "plan_packet": plan.get("plan_path"),
        "verification": verification,
        "execution_success": execution["success"],
        "lift_preserved_close": lift_preserved_close,
    }
)

print("[yam_runtime_smoke] success=", TASK_RESULT["success"])
