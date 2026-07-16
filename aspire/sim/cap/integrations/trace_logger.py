"""Execution Trace Logger for ASPIRE API calls.

Wraps any ApiBase subclass to log every API function call with inputs, outputs,
timing, and optional keyframe captures. Zero changes to core logic — just swap
the API class in the YAML config or register the traced variant.

Suite-specific traced API wrappers live in:
    - aspire.sim.cap.integrations.libero_trace_logger
    - aspire.sim.cap.integrations.robosuite_trace_logger

To wrap another API class dynamically:
    from aspire.sim.cap.integrations.trace_logger import make_traced_api
    TracedApi = make_traced_api(SomeApiClass)
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aspire.sim.cap.envs.base import BaseEnv
from aspire.sim.cap.integrations.base_api import ApiBase


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if obj.size > 50:
                return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Image.Image):
            return f"<PIL.Image size={obj.size} mode={obj.mode}>"
        return super().default(obj)


def _summarize_args(func_name: str, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Create a serializable summary of function arguments."""
    summary: dict[str, Any] = {}

    if func_name in ("get_observation", "get_env_observation"):
        pass  # No args

    elif func_name == "segment_sam3_text_prompt":
        if len(args) >= 1:
            rgb = args[0]
            summary["rgb_shape"] = list(rgb.shape) if isinstance(rgb, np.ndarray) else str(type(rgb))
        if len(args) >= 2:
            summary["text_prompt"] = args[1]
        summary.update({k: v for k, v in kwargs.items() if k != "rgb"})

    elif func_name == "segment_sam3_point_prompt":
        if len(args) >= 1:
            rgb = args[0]
            summary["rgb_shape"] = list(rgb.shape) if isinstance(rgb, np.ndarray) else str(type(rgb))
        if len(args) >= 2:
            summary["point_coords"] = list(args[1]) if hasattr(args[1], "__iter__") else args[1]

    elif func_name == "plan_grasp":
        for i, name in enumerate(["depth", "intrinsics", "segmentation"]):
            if i < len(args) and isinstance(args[i], np.ndarray):
                summary[f"{name}_shape"] = list(args[i].shape)

    elif func_name in ("solve_ik", "solve_ik_arm0", "solve_ik_arm1"):
        if len(args) >= 1:
            summary["position"] = args[0].tolist() if isinstance(args[0], np.ndarray) else args[0]
        if len(args) >= 2:
            summary["quaternion_wxyz"] = args[1].tolist() if isinstance(args[1], np.ndarray) else args[1]

    elif func_name in ("move_to_joints", "move_to_joints_arm0", "move_to_joints_arm1"):
        if len(args) >= 1:
            summary["joints"] = args[0].tolist() if isinstance(args[0], np.ndarray) else args[0]

    elif func_name == "move_to_joints_both":
        if len(args) >= 1:
            summary["joints0"] = args[0].tolist() if isinstance(args[0], np.ndarray) else args[0]
        if len(args) >= 2:
            summary["joints1"] = args[1].tolist() if isinstance(args[1], np.ndarray) else args[1]

    elif func_name in (
        "open_gripper", "close_gripper",
        "open_gripper_arm0", "close_gripper_arm0",
        "open_gripper_arm1", "close_gripper_arm1",
    ):
        pass  # No args

    elif func_name == "point_prompt_molmo":
        if len(args) >= 2:
            summary["text_prompt"] = args[1]

    elif func_name in (
        "rotation_matrix_to_quaternion",
        "decompose_transform",
        "depth_to_point_cloud",
        "mask_to_world_points",
        "pixel_to_world_point",
        "transform_points",
        "interpolate_segment",
        "normalize_vector",
        "select_top_down_grasp",
        "get_oriented_bounding_box_from_3d_points",
    ):
        # Skill library functions — log shapes for arrays, values for scalars
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray):
                summary[f"arg{i}_shape"] = list(a.shape)
            else:
                try:
                    json.dumps(a)
                    summary[f"arg{i}"] = a
                except (TypeError, ValueError):
                    summary[f"arg{i}"] = str(type(a))
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                summary[f"{k}_shape"] = list(v.shape)
            else:
                try:
                    json.dumps(v)
                    summary[k] = v
                except (TypeError, ValueError):
                    summary[k] = str(type(v))

    else:
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray):
                summary[f"arg{i}_shape"] = list(a.shape)
            else:
                try:
                    json.dumps(a)
                    summary[f"arg{i}"] = a
                except (TypeError, ValueError):
                    summary[f"arg{i}"] = str(type(a))
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                summary[f"{k}_shape"] = list(v.shape)
            else:
                try:
                    json.dumps(v)
                    summary[k] = v
                except (TypeError, ValueError):
                    summary[k] = str(type(v))

    return summary


def _summarize_result(func_name: str, result: Any) -> dict[str, Any]:
    """Create a serializable summary of function return value."""
    summary: dict[str, Any] = {}

    if func_name == "get_env_observation":
        if isinstance(result, tuple) and len(result) >= 2:
            rgb, depth = result[:2]
            if isinstance(rgb, np.ndarray):
                summary["rgb_shape"] = list(rgb.shape)
            if isinstance(depth, np.ndarray):
                summary["depth_shape"] = list(depth.shape)

    elif func_name == "get_observation":
        if isinstance(result, dict):
            for cam_name, cam_data in result.items():
                if isinstance(cam_data, dict) and "images" in cam_data:
                    imgs = cam_data["images"]
                    if "rgb" in imgs:
                        summary[f"{cam_name}_rgb_shape"] = list(imgs["rgb"].shape)
                    if "depth" in imgs:
                        summary[f"{cam_name}_depth_shape"] = list(imgs["depth"].shape)
                    if "pose_mat" in cam_data:
                        summary[f"{cam_name}_cam_pose"] = cam_data["pose_mat"][:3, 3].tolist()
                elif cam_name in ("robot_joint_pos", "robot_cartesian_pos"):
                    if isinstance(cam_data, np.ndarray):
                        summary[cam_name] = cam_data.tolist()
                    else:
                        summary[cam_name] = cam_data
                elif cam_name == "full_prompt":
                    pass  # Skip prompt data
                else:
                    summary[cam_name] = str(type(cam_data))

    elif func_name in ("segment_sam3_text_prompt", "segment_sam3_point_prompt"):
        if isinstance(result, list):
            summary["num_masks"] = len(result)
            for i, mask_info in enumerate(result[:3]):  # Top 3
                prefix = f"mask_{i}"
                if isinstance(mask_info, dict):
                    if "score" in mask_info:
                        summary[f"{prefix}_score"] = float(mask_info["score"])
                    if "box" in mask_info:
                        summary[f"{prefix}_bbox"] = [float(x) for x in mask_info["box"]]
                    if "mask" in mask_info and isinstance(mask_info["mask"], np.ndarray):
                        summary[f"{prefix}_area_pct"] = round(
                            float(mask_info["mask"].sum()) / mask_info["mask"].size * 100, 2
                        )

    elif func_name == "plan_grasp":
        if isinstance(result, tuple) and len(result) == 2:
            poses, scores = result
            if isinstance(poses, np.ndarray):
                summary["num_grasps"] = poses.shape[0]
            if isinstance(scores, np.ndarray) and scores.size > 0:
                summary["best_score"] = float(scores.max())
                summary["best_idx"] = int(scores.argmax())
                best_pose = poses[scores.argmax()]
                summary["best_grasp_position"] = best_pose[:3, 3].tolist()

    elif func_name in ("solve_ik", "solve_ik_arm0", "solve_ik_arm1"):
        if isinstance(result, np.ndarray):
            summary["joints"] = result.tolist()

    elif func_name in ("move_to_joints", "move_to_joints_arm0", "move_to_joints_arm1", "move_to_joints_both"):
        summary["completed"] = True

    elif func_name in (
        "open_gripper", "close_gripper",
        "open_gripper_arm0", "close_gripper_arm0",
        "open_gripper_arm1", "close_gripper_arm1",
    ):
        summary["completed"] = True

    elif func_name == "point_prompt_molmo":
        summary["result"] = str(result)

    elif func_name == "get_oriented_bounding_box_from_3d_points":
        if isinstance(result, dict):
            if "center" in result:
                c = result["center"]
                summary["center"] = c.tolist() if isinstance(c, np.ndarray) else c
            if "extent" in result:
                e = result["extent"]
                summary["extent"] = e.tolist() if isinstance(e, np.ndarray) else e

    elif func_name == "select_top_down_grasp":
        if isinstance(result, tuple) and len(result) == 2:
            grasp, score = result
            summary["found_grasp"] = grasp is not None
            if grasp is not None:
                summary["grasp_position"] = grasp[:3, 3].tolist()
                summary["score"] = float(score)

    else:
        # Generic: try to serialize
        try:
            json.dumps(result, cls=_NumpyEncoder)
            summary["result"] = result
        except (TypeError, ValueError):
            summary["result_type"] = str(type(result))

    return summary


def _create_sam3_overlay(
    rgb: np.ndarray,
    masks: list[dict],
) -> Image.Image:
    """Create RGB image with SAM3 mask/bbox overlays and return as PIL Image."""
    img = Image.fromarray(rgb.copy()).convert("RGBA")
    draw = ImageDraw.Draw(img)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for i, mask_info in enumerate(masks[:5]):
        color = colors[i % len(colors)]
        if "box" in mask_info:
            box = mask_info["box"]
            draw.rectangle(box, outline=color, width=2)
            score = mask_info.get("score", 0)
            draw.text((box[0], box[1] - 12), f"{score:.2f}", fill=color)
        if "mask" in mask_info and isinstance(mask_info["mask"], np.ndarray):
            mask = mask_info["mask"]
            mask_img = Image.fromarray((mask * 80).astype(np.uint8), mode="L")
            colored = Image.new("RGBA", img.size, (*color, 0))
            colored.putalpha(mask_img)
            img = Image.alpha_composite(img, colored)
            draw = ImageDraw.Draw(img)
            if "box" in mask_info:
                box = mask_info["box"]
                draw.rectangle(box, outline=color, width=2)

    return img.convert("RGB")


class TraceLogger:
    """Logs API calls with timing, args, results, and keyframes."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._step = 0
        self._start_time = time.time()
        self._last_rgb: np.ndarray | None = None
        self._observation_count = 0
        self._buffered_keyframes: list[tuple[str, Image.Image]] = []
        self._buffered_arrays: list[tuple[str, Any]] = []
        self._env: BaseEnv | None = None

    def set_env(self, env: BaseEnv) -> None:
        """Store env reference for querying robot state after gripper actions."""
        self._env = env

    def set_output_dir(self, output_dir: str | Path) -> None:
        """Deprecated — keyframes are now buffered in memory and saved at trace.save() time."""
        pass

    def log_call(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Record a single API call."""
        entry: dict[str, Any] = {
            "step": self._step,
            "timestamp": round(time.time() - self._start_time, 3),
            "function": func_name,
            "args": _summarize_args(func_name, args, kwargs),
            "duration_ms": round(duration_ms, 1),
        }

        if error:
            entry["error"] = error
        else:
            entry["result"] = _summarize_result(func_name, result)

        # After gripper actions, query actual gripper width from env
        _gripper_fns = (
            "close_gripper", "open_gripper",
            "close_gripper_arm0", "open_gripper_arm0",
            "close_gripper_arm1", "open_gripper_arm1",
        )
        if func_name in _gripper_fns and self._env is not None:
            try:
                obs = self._env.get_observation()
                if "result" not in entry:
                    entry["result"] = {}
                if func_name.endswith("_arm0"):
                    qpos = obs.get("robot0_gripper_qpos")
                elif func_name.endswith("_arm1"):
                    qpos = obs.get("robot1_gripper_qpos")
                else:
                    qpos = obs.get("robot0_gripper_qpos")
                if qpos is not None and isinstance(qpos, np.ndarray) and len(qpos) > 0:
                    entry["result"]["gripper_width"] = float(qpos[0])
                else:
                    joint_pos = obs.get("robot_joint_pos")
                    if joint_pos is not None and isinstance(joint_pos, np.ndarray) and len(joint_pos) > 0:
                        entry["result"]["gripper_width"] = float(joint_pos[-1])
            except Exception:
                pass  # Don't break tracing if env query fails

        # Buffer keyframes for key functions
        keyframe_saved = False
        if func_name == "get_env_observation" and isinstance(result, tuple) and len(result) >= 2:
            rgb, depth = result[:2]
            if isinstance(rgb, np.ndarray):
                self._last_rgb = rgb.copy()
                filename = f"step_{self._step:03d}_obs_env.jpg"
                self._buffered_keyframes.append((filename, Image.fromarray(rgb)))
                keyframe_saved = True
            if isinstance(depth, np.ndarray):
                self._buffered_arrays.append(
                    (f"step_{self._step:03d}_depth_env.npy", depth.copy()))

        elif func_name == "get_observation" and isinstance(result, dict):
            self._observation_count += 1
            for cam_name in ("agentview", "robot0_robotview", "robot0_eye_in_hand"):
                if cam_name in result and "images" in result[cam_name]:
                    rgb = result[cam_name]["images"].get("rgb")
                    if rgb is not None:
                        self._last_rgb = rgb.copy()
                        filename = f"step_{self._step:03d}_obs_{cam_name}.jpg"
                        self._buffered_keyframes.append((filename, Image.fromarray(rgb)))
                        keyframe_saved = True
                    # Save depth, intrinsics, extrinsics as numpy arrays
                    depth = result[cam_name]["images"].get("depth")
                    if depth is not None:
                        self._buffered_arrays.append(
                            (f"step_{self._step:03d}_depth_{cam_name}.npy", depth.copy()))
                    intrinsics = result[cam_name].get("intrinsics")
                    if intrinsics is not None:
                        self._buffered_arrays.append(
                            (f"step_{self._step:03d}_intrinsics_{cam_name}.npy", intrinsics.copy()))
                    extrinsics = result[cam_name].get("pose_mat")
                    if extrinsics is not None:
                        self._buffered_arrays.append(
                            (f"step_{self._step:03d}_extrinsics_{cam_name}.npy", extrinsics.copy()))

        elif func_name in ("segment_sam3_text_prompt", "segment_sam3_point_prompt"):
            rgb = args[0] if len(args) >= 1 and isinstance(args[0], np.ndarray) else self._last_rgb
            if rgb is not None and isinstance(result, list) and len(result) > 0:
                try:
                    filename = f"step_{self._step:03d}_sam3.jpg"
                    overlay_img = _create_sam3_overlay(rgb, result)
                    self._buffered_keyframes.append((filename, overlay_img))
                    keyframe_saved = True
                except Exception as e:
                    entry["keyframe_error"] = str(e)
            # Save top mask as numpy array
            if isinstance(result, list) and len(result) > 0:
                top_mask = result[0].get("mask") if isinstance(result[0], dict) else None
                if top_mask is not None and isinstance(top_mask, np.ndarray):
                    self._buffered_arrays.append(
                        (f"step_{self._step:03d}_mask_0.npy", top_mask.copy()))

        elif func_name == "plan_grasp":
            if isinstance(result, tuple) and len(result) == 2:
                poses, scores = result
                if isinstance(poses, np.ndarray) and isinstance(scores, np.ndarray):
                    self._buffered_arrays.append(
                        (f"step_{self._step:03d}_grasps.npz", {"poses": poses, "scores": scores}))

        entry["keyframe_saved"] = keyframe_saved
        self._entries.append(entry)
        self._step += 1

    def save(self, output_dir: str | Path) -> Path:
        """Write trace.json and buffered keyframes to the given directory."""
        out_dir = Path(output_dir)

        # Save buffered keyframes
        keyframes_dir = out_dir / "keyframes"
        if self._buffered_keyframes:
            keyframes_dir.mkdir(parents=True, exist_ok=True)
            for filename, img in self._buffered_keyframes:
                try:
                    img.save(keyframes_dir / filename, quality=85)
                except Exception:
                    pass

        # Save buffered numpy arrays
        if self._buffered_arrays:
            keyframes_dir.mkdir(parents=True, exist_ok=True)
            for filename, data in self._buffered_arrays:
                try:
                    filepath = keyframes_dir / filename
                    if isinstance(data, dict):
                        np.savez_compressed(filepath, **data)
                    else:
                        np.save(filepath, data)
                except Exception:
                    pass

        out = out_dir / "trace.json"
        out.write_text(json.dumps(self._entries, indent=2, cls=_NumpyEncoder))
        return out

    def reset(self) -> None:
        """Reset for a new trial."""
        self._entries.clear()
        self._step = 0
        self._start_time = time.time()
        self._last_rgb = None
        self._observation_count = 0
        self._buffered_keyframes.clear()
        self._buffered_arrays.clear()

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self._entries


def _wrap_function(fn: Any, name: str, logger: TraceLogger) -> Any:
    """Wrap a single API function with trace logging."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        error = None
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            dt = (time.perf_counter() - t0) * 1000
            logger.log_call(name, args, kwargs, result, dt, error)

    return wrapper


class TracedApiMixin:
    """Mixin that wraps all functions() with trace logging."""

    _trace_logger: TraceLogger

    def __init_trace__(self) -> None:
        self._trace_logger = TraceLogger()

    def functions(self) -> dict[str, Any]:
        """Override to wrap all API functions with tracing."""
        raw_fns = super().functions()  # type: ignore[misc]
        return {
            name: _wrap_function(fn, name, self._trace_logger)
            for name, fn in raw_fns.items()
        }

    def get_trace_logger(self) -> TraceLogger:
        return self._trace_logger


def make_traced_api(api_class: type) -> type:
    """Dynamically create a traced version of any ApiBase subclass."""

    class TracedApi(TracedApiMixin, api_class):  # type: ignore[valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            api_class.__init__(self, *args, **kwargs)
            self.__init_trace__()
            # Wire env reference if available (ApiBase stores self._env)
            if hasattr(self, "_env"):
                self._trace_logger.set_env(self._env)

    TracedApi.__name__ = f"Traced{api_class.__name__}"
    TracedApi.__qualname__ = TracedApi.__name__
    return TracedApi
