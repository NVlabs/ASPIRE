# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a saved script against a RoboCasa env (direct mode, no CapServer).

Uses the same Hydra config as run_agent.py (AgentConfig + experiments/config.yaml).

Usage::

    # Standalone — uses experiments/config.yaml defaults
    uv run python run_script.py script_file=cap/saved_scripts/test.py

    # With an experiment config (gets env, ports, seed, etc.)
    uv run python run_script.py experiment=pick_place_sink_to_counter \\
        script_file=cap/saved_scripts/robocasa_test_planning.py

    # Override fields
    uv run python run_script.py script_file=test.py \\
        env.name=robocasa:PickPlaceSinkToCounter env.seed=42 \\
        runtime.curobo_port=9400 recording.enabled=true
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import traceback
import builtins
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

_ROOT_PATH = Path(__file__).resolve().parent
_ROOT = str(_ROOT_PATH)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cap.agent.agent_config import register_configs

register_configs()


def _resolve_file(path_str: str, saved_scripts_dir: Path) -> Path:
    """Resolve a script path: direct → repo-relative → saved_scripts lookup."""
    raw = path_str.strip().replace("\\", "/")
    if not raw:
        raise FileNotFoundError(f"Script not found: '{path_str}'")

    saved_scripts_dir = saved_scripts_dir.resolve()
    attempts: list[Path] = []

    def _try(p: Path) -> Path | None:
        attempts.append(p)
        return p.resolve() if p.exists() else None

    # 1. Direct / absolute / ~/
    result = _try(Path(raw).expanduser())
    if result:
        return result

    # 2. Repo-relative
    if not Path(raw).is_absolute():
        result = _try(_ROOT_PATH / raw)
        if result:
            return result

    # 3. Paths copied with a repo prefix, e.g. aspire/real/cap/saved_scripts/foo.py.
    parts = Path(raw).parts
    for idx in range(len(parts) - 1):
        if parts[idx] == "cap" and parts[idx + 1] == "saved_scripts":
            suffix = Path(*parts[idx + 2 :])
            result = _try(saved_scripts_dir / suffix)
            if result and result.is_relative_to(saved_scripts_dir):
                return result

    # 4. Bare name in saved_scripts/
    result = _try(saved_scripts_dir / raw)
    if result and result.is_relative_to(saved_scripts_dir):
        return result

    tried = "\n".join(f"  Tried: {c}" for c in attempts)
    raise FileNotFoundError(f"Script not found: '{path_str}'\n{tried}")


def resolve_file(path_str: str) -> Path:
    """Resolve a script path using this repo's default saved_scripts directory."""
    return _resolve_file(path_str, _ROOT_PATH / "cap" / "saved_scripts")


def _save_result(output_dir: Path, get_task_info, exec_error: str | None) -> None:
    """Save result.json in the format expected by the agent pipeline."""
    info: dict = {}
    success = False
    reward = 0.0
    if get_task_info is not None:
        try:
            info = get_task_info()
            success = bool(info.get("success", False))
            reward = float(info.get("reward", 0.0))
        except Exception as exc:
            info = {"error": str(exc)}

    feedback_parts = [f"reward={reward:.3f}, success={success}"]
    if info.get("obj_pos"):
        feedback_parts.append(f"obj_pos={[round(x, 4) for x in info['obj_pos']]}")
    if exec_error:
        feedback_parts.append(f"error={exec_error[:200]}")

    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "success": success,
                "score": reward,
                "feedback": "\n".join(feedback_parts),
                "method": "reward",
                "details": info,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


def _save_script_snapshot(
    output_dir: Path,
    *,
    script_path: Path,
    requested_script_file: str,
    code: str,
) -> None:
    """Copy the exact executed saved script into the run artifact directory."""
    try:
        (output_dir / "code.py").write_text(code, encoding="utf-8")
        (output_dir / "code_snapshot.json").write_text(
            json.dumps(
                {
                    "source_path": str(script_path),
                    "requested_script_file": requested_script_file,
                    "snapshot_file": "code.py",
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "argv": sys.argv,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            f"[run_script] WARNING: failed to save script snapshot: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _capture_frames(env, cameras: list[str]) -> dict:
    """Capture RGB frames from all cameras. Returns {cam_name: np.ndarray}."""
    frames = {}
    for cam in cameras:
        try:
            img = env.render_rgb(cam)
            if img is not None and img.shape[0] > 1 and img.shape[1] > 1:
                frames[cam] = img
        except Exception:
            pass
    return frames


def _setup_skill_reflection(cfg, sr_cfg, namespace: dict, log_dir: Path):
    """Register a before/after hook on the SkillRegistry that fires an async
    Gemini query after each @skill call, capturing debug keyframes."""
    from concurrent.futures import ThreadPoolExecutor
    from cap.agent.skill_registry import SkillRegistry

    cameras: list[str] = list(sr_cfg.cameras) if sr_cfg.cameras else ["top", "wrist"]
    vlm_backend: str = str(sr_cfg.vlm_backend or "nvidia")
    vlm_model: str | None = sr_cfg.vlm_model or None
    max_workers: int = int(sr_cfg.max_workers or 4)
    try:
        task: str = str(cfg.task)
    except Exception:
        task = ""

    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="skill_reflect")
    _pending_frames: dict[str, dict] = {}  # "{call_id}_{name}" -> frames

    def _hook(skill_name: str, call_id: int, phase: str) -> None:
        key = f"{call_id:03d}_{skill_name}"
        cam_fn = namespace.get("get_camera_image")
        if cam_fn is None:
            return
        frames: dict[str, Any] = {}
        for cam in cameras:
            try:
                img = cam_fn(cam)
                if img is not None:
                    frames[cam] = img
            except Exception:
                pass

        if phase == "before":
            _pending_frames[key] = {"before": frames}
        elif phase == "after":
            entry = _pending_frames.pop(key, {})
            entry["after"] = frames
            pool.submit(_reflect_skill, key, skill_name, entry, task,
                        vlm_backend, vlm_model, log_dir)

    # Do not replace other hooks (e.g. live debug UI event logging).
    registry = SkillRegistry.get()
    add_hook = getattr(registry, "add_capture_hook", None)
    if callable(add_hook):
        add_hook(_hook)
    else:
        registry.set_capture_hook(_hook)
    print(f"[run_script] Skill reflection enabled "
          f"(backend={vlm_backend}, cameras={cameras}, workers={max_workers})")
    return pool


def _launch_debug_ui(log_dir: Path, debug_cfg: Any) -> subprocess.Popen | None:
    """Launch the standalone log-watching debug UI.

    The UI is intentionally a separate process that only reads artifacts from
    ``log_dir``.  run_script continues to own robot execution.
    """
    host = str(getattr(debug_cfg, "host", "0.0.0.0"))
    port = int(getattr(debug_cfg, "port", 8788))
    event_log_name = str(getattr(debug_cfg, "event_log_name", "debug_events.jsonl"))
    poll_interval_s = float(getattr(debug_cfg, "poll_interval_s", 0.25))
    auto_exit_on_run_end = bool(getattr(debug_cfg, "auto_exit_on_run_end", True))
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    if bool(getattr(debug_cfg, "replace_existing_on_port", True)):
        _terminate_existing_debug_ui_on_port(port)
    cmd = [
        sys.executable,
        "-m",
        "cap.debug_ui.app",
        "--log-dir",
        str(log_dir.resolve()),
        "--host",
        host,
        "--port",
        str(port),
        "--event-log-name",
        event_log_name,
        "--poll-interval",
        str(poll_interval_s),
        "--parent-pid",
        str(os.getpid() if auto_exit_on_run_end else 0),
        "--auto-exit-on-run-end",
        "true" if auto_exit_on_run_end else "false",
    ]
    try:
        ui_log = (log_dir / "debug_ui.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT_PATH),
            stdout=ui_log,
            stderr=ui_log,
            start_new_session=True,
        )
        ui_log.close()
        (log_dir / "debug_ui.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        (log_dir / "debug_ui.url").write_text(url + "\n", encoding="utf-8")
        print(f"[run_script] Debug UI: {url} (pid={proc.pid})")
        if bool(getattr(debug_cfg, "auto_open", False)):
            try:
                import webbrowser

                webbrowser.open(url)
            except Exception:
                pass
        return proc
    except Exception as exc:
        print(f"[run_script] Debug UI launch failed: {type(exc).__name__}: {exc}")
        return None


def _iter_debug_ui_pids_on_port(port: int) -> list[int]:
    """Return existing cap.debug_ui.app process IDs for a specific port."""
    pids: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return pids
    current_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        if not raw:
            continue
        parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        joined = " ".join(parts)
        if "cap.debug_ui.app" not in joined:
            continue
        port_match = False
        for idx, part in enumerate(parts):
            if part == "--port" and idx + 1 < len(parts) and parts[idx + 1] == str(port):
                port_match = True
                break
            if part == f"--port={port}" or part == f"debug_ui.port={port}" or part == f"port={port}":
                port_match = True
                break
        if port_match:
            pids.append(pid)
    return sorted(set(pids), reverse=True)


def _terminate_existing_debug_ui_on_port(port: int) -> None:
    """Make a run's debug UI own the configured port.

    The YAM workflow expects :8788 to show the newest run. Previous debug UI
    processes intentionally stay alive when auto_exit_on_run_end=false, so a new
    run needs to replace only older cap.debug_ui.app processes on that port.
    """
    pids = _iter_debug_ui_pids_on_port(port)
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"[run_script] Could not stop existing debug UI pid={pid}: {exc}")
    import time

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        remaining = _iter_debug_ui_pids_on_port(port)
        if not remaining:
            break
        time.sleep(0.05)
    for pid in _iter_debug_ui_pids_on_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"[run_script] Could not kill existing debug UI pid={pid}: {exc}")
    print(f"[run_script] Replaced existing debug UI on port {port}")


def _terminate_debug_ui(proc: subprocess.Popen | None, debug_cfg: Any) -> None:
    """Stop the launched debug UI unless configured to stay alive."""
    if proc is None:
        return
    if not bool(getattr(debug_cfg, "auto_exit_on_run_end", True)):
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
    except Exception:
        pass


def _reflect_skill(
    key: str,
    skill_name: str,
    frames: dict,
    task: str,
    vlm_backend: str,
    vlm_model: str | None,
    log_dir: Path,
) -> None:
    """Run one async Gemini query for a single @skill before/after pair."""
    try:
        # Use the transport-layer query (same path as skills.py uses internally).
        from cap.agent.tools.vlm import query as _vlm_query

        before = frames.get("before", {})
        after = frames.get("after", {})
        saved_paths = _save_skill_reflection_frames(key, before, after, log_dir)

        # Build image list in consistent before/after order per camera.
        images: list[Any] = []
        image_desc_parts: list[str] = []
        for cam in sorted(set(before) | set(after)):
            if cam in before:
                images.append(before[cam])
                image_desc_parts.append(f"{cam} BEFORE")
            if cam in after:
                images.append(after[cam])
                image_desc_parts.append(f"{cam} AFTER")

        if not images:
            return

        image_order = ", ".join(f"image {i+1}={d}" for i, d in enumerate(image_desc_parts))
        prompt = (
            f"Task: {task}\n"
            f"Skill \"{skill_name}\" just completed.\n"
            f"Image order: {image_order}.\n\n"
            f"Describe what visually changed between BEFORE and AFTER, "
            f"and whether it represents progress toward the task. "
            f"Note any discrepancy between the reported outcome and what the images show. "
            f"Under 50 words."
        )

        result = _vlm_query(
            vlm_backend,
            text=prompt,
            images=images,
            model=vlm_model or None,
            telemetry_source="skill_reflection",
        )

        # Write to exec_dir/skill_steps/NNN_skillname.md
        steps_dir = log_dir / "skill_steps"
        steps_dir.mkdir(exist_ok=True)
        out = steps_dir / f"{key}.md"
        image_note = ""
        if saved_paths:
            rel_paths = [
                str(path.relative_to(log_dir)) if path.is_relative_to(log_dir) else str(path)
                for path in saved_paths
            ]
            image_note = "\n\n## Saved keyframes\n\n" + "\n".join(
                f"- `{path}`" for path in rel_paths
            )
        out.write_text(
            f"# {key}\n\n{result}{image_note}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        try:
            steps_dir = log_dir / "skill_steps"
            steps_dir.mkdir(exist_ok=True)
            (steps_dir / f"{key}.error").write_text(str(exc), encoding="utf-8")
        except Exception:
            pass


def _save_skill_reflection_frames(
    key: str,
    before: dict,
    after: dict,
    output_dir: Path,
) -> list[Path]:
    """Save per-skill debug keyframes under ``vis/skill_steps/<key>/``.

    Video recording and skill-reflection keyframes are intentionally separate:
    recording writes continuous ``*.mp4`` files, while this helper writes the
    exact before/after RGB frames that Gemini saw for one @skill call.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def _as_uint8_rgb(img: Any) -> np.ndarray | None:
        if img is None:
            return None
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[0] <= 1 or arr.shape[1] <= 1:
            return None
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _same_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = min(left.shape[0], right.shape[0])
        if left.shape[0] != h:
            pil = Image.fromarray(left).resize((int(left.shape[1] * h / left.shape[0]), h))
            left = np.asarray(pil)
        if right.shape[0] != h:
            pil = Image.fromarray(right).resize((int(right.shape[1] * h / right.shape[0]), h))
            right = np.asarray(pil)
        return left, right

    def _label_pair(left: np.ndarray, right: np.ndarray) -> Image.Image:
        left, right = _same_height(left, right)
        concat = np.concatenate([left, right], axis=1)
        pil_img = Image.fromarray(concat)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except Exception:
            font = ImageFont.load_default()
        pad = 10
        w_left = left.shape[1]
        draw.rectangle([pad - 2, pad - 2, pad + 90, pad + 24], fill=(0, 0, 0))
        draw.text((pad, pad), "BEFORE", fill=(255, 255, 255), font=font)
        draw.rectangle(
            [w_left + pad - 2, pad - 2, w_left + pad + 72, pad + 24],
            fill=(0, 0, 0),
        )
        draw.text((w_left + pad, pad), "AFTER", fill=(255, 255, 255), font=font)
        return pil_img

    vis_dir = output_dir / "vis" / "skill_steps" / key
    vis_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for cam in sorted(set(before) | set(after)):
        before_img = _as_uint8_rgb(before.get(cam))
        after_img = _as_uint8_rgb(after.get(cam))
        if before_img is not None:
            path = vis_dir / f"{cam}_before.png"
            Image.fromarray(before_img).save(path)
            saved.append(path)
        if after_img is not None:
            path = vis_dir / f"{cam}_after.png"
            Image.fromarray(after_img).save(path)
            saved.append(path)
        if before_img is not None and after_img is not None:
            path = vis_dir / f"{cam}_before_after.png"
            _label_pair(before_img, after_img).save(path)
            saved.append(path)
    if saved:
        print(f"[run_script] Saved skill keyframes: vis/skill_steps/{key}/")
    return saved


def _save_before_after(
    before: dict,
    after: dict,
    output_dir: Path,
) -> None:
    """Save side-by-side before/after images for each camera to vis/."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    for cam in before:
        if cam not in after:
            continue
        img_before = before[cam]
        img_after = after[cam]

        # Resize to same height if needed
        h = min(img_before.shape[0], img_after.shape[0])
        if img_before.shape[0] != h:
            pil_b = Image.fromarray(img_before).resize(
                (int(img_before.shape[1] * h / img_before.shape[0]), h)
            )
            img_before = np.array(pil_b)
        if img_after.shape[0] != h:
            pil_a = Image.fromarray(img_after).resize(
                (int(img_after.shape[1] * h / img_after.shape[0]), h)
            )
            img_after = np.array(pil_a)

        # Concat side by side
        concat = np.concatenate([img_before, img_after], axis=1)
        pil_img = Image.fromarray(concat)
        draw = ImageDraw.Draw(pil_img)

        # Add labels
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except Exception:
            font = ImageFont.load_default()

        pad = 10
        w_before = img_before.shape[1]
        # "BEFORE" on left
        draw.rectangle([pad - 2, pad - 2, pad + 90, pad + 24], fill=(0, 0, 0))
        draw.text((pad, pad), "BEFORE", fill=(255, 255, 255), font=font)
        # "AFTER" on right
        draw.rectangle(
            [w_before + pad - 2, pad - 2, w_before + pad + 72, pad + 24], fill=(0, 0, 0)
        )
        draw.text((w_before + pad, pad), "AFTER", fill=(255, 255, 255), font=font)

        pil_img.save(vis_dir / f"{cam}_before_after.png")
        print(f"[run_script] Saved before/after: vis/{cam}_before_after.png")


@hydra.main(version_base="1.3", config_path="experiments", config_name="config")
def main(cfg: DictConfig) -> None:
    from cap.agent.recorder import ScriptRecorder
    from cap.agent.tools._artifact_log import set_artifact_dir

    # --- Resolve script file ---
    if not cfg.script_file:
        sys.exit(
            "script_file is required. Usage:\n"
            "  uv run python run_script.py script_file=cap/saved_scripts/test.py"
        )
    saved_scripts_dir = (_ROOT_PATH / cfg.runtime.saved_scripts_dir).resolve()
    try:
        script_path = _resolve_file(cfg.script_file, saved_scripts_dir)
    except FileNotFoundError as e:
        print(f"[run_script] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    code = script_path.read_text()
    print(f"[run_script] Script : {script_path}")

    # --- Seed ---
    if cfg.env.seed is not None:
        os.environ["ROBOCASA_SEED"] = str(cfg.env.seed)
        print(f"[run_script] Seed   : {cfg.env.seed}")

    # --- Plumb remote-service config into env vars ---
    # Clients (cap/config.py, cap/env/base/skill_library.py, etc.) read from
    # ANYGRASP_SERVER_*/SAM3_SERVER_* env. Make experiments/infra/ports.yaml
    # authoritative by exporting the Hydra values here.
    os.environ["ANYGRASP_SERVER_HOST"] = str(cfg.runtime.anygrasp_host)
    os.environ["ANYGRASP_SERVER_PORT"] = str(cfg.runtime.anygrasp_port)
    os.environ["ANYGRASP_SERVICE_URL"] = (
        f"http://{os.environ['ANYGRASP_SERVER_HOST']}:"
        f"{os.environ['ANYGRASP_SERVER_PORT']}"
    )
    os.environ["SAM3_SERVER_HOST"] = str(cfg.runtime.sam3_host)
    os.environ["SAM3_SERVER_PORT"] = str(cfg.runtime.sam3_port)
    os.environ["BUNDLESDF_SERVER_HOST"] = str(cfg.runtime.bundlesdf_host)
    os.environ["BUNDLESDF_SERVER_PORT"] = str(cfg.runtime.bundlesdf_port)
    os.environ["CAP_CUROBO_HOST"] = str(cfg.runtime.curobo_host)
    os.environ["CAP_CUROBO_PORT"] = str(cfg.runtime.curobo_port)
    os.environ["CAP_CUROBO_START_SERVER"] = (
        "0" if int(cfg.runtime.curobo_port or 0) > 0 else "1"
    )

    # --- Output directory (computed before env so tools can take artifact_dir) ---
    if cfg.script_output_dir:
        log_dir = Path(cfg.script_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_dir = Path(cfg.runtime.log_dir) / f"{script_path.stem}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    _save_script_snapshot(
        log_dir,
        script_path=script_path,
        requested_script_file=str(cfg.script_file),
        code=code,
    )

    # --- Optional live debug UI + structured event stream ---
    debug_cfg = getattr(cfg, "debug_ui", None)
    debug_writer = None
    debug_ui_proc = None
    if debug_cfg is not None and bool(getattr(debug_cfg, "enabled", False)):
        try:
            from cap.agent.debug_events import DebugEventWriter

            debug_writer = DebugEventWriter(
                log_dir,
                filename=str(getattr(debug_cfg, "event_log_name", "debug_events.jsonl")),
            )
            debug_writer.emit(
                "run_start",
                script=str(script_path),
                log_dir=str(log_dir.resolve()),
                pid=os.getpid(),
            )
            if bool(getattr(debug_cfg, "launch", True)):
                debug_ui_proc = _launch_debug_ui(log_dir, debug_cfg)
        except Exception as exc:
            debug_writer = None
            print(
                f"[run_script] Debug event setup failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # Save seed/layout/style so it's easy to verify which config ran in this exec dir.
    import json as _json
    (log_dir / "episode_config.json").write_text(
        _json.dumps({
            "env": cfg.env.name,
            "seed": cfg.env.seed,
            "layout_id": cfg.env.layout_id,
            "style_id": cfg.env.style_id,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- Create env + namespace ---
    env_name = cfg.env.name or "robocasa:PickPlaceSinkToCounter"
    from cap.agent.robot_adapters import get_robot_adapter

    print(f"[run_script] Creating {env_name} env (direct mode)")
    adapter = get_robot_adapter(cfg)
    _env, namespace = adapter.create_runtime(cfg=cfg, runtime_role="script")
    print(f"[run_script] Namespace: {len(namespace)} tools\n")
    if debug_writer is not None:
        debug_writer.emit("namespace_ready", env=env_name, tools=len(namespace))
        debug_writer.register_tool_sources(namespace)

    # --- Robot backend (replaces all robot-specific if/else) ---
    from cap.agent.script_backends import get_backend
    backend = get_backend(env_name, cfg, _env)

    # --- Profiler / file logging ---
    from cap.agent.profiler import (
        close_file_logging,
        disable_stdout_tee,
        enable_file_logging,
        enable_stdout_tee,
        reset_call_records,
        save_call_records,
        set_quiet,
        set_state_fn,
        set_tool_event_hooks,
        wrap_callables_with_timing,
    )

    set_quiet(cfg.runtime.quiet or backend.is_quiet)

    log_path = None
    original_stdout = None
    if not cfg.script_no_log:
        log_name = "profiling" if cfg.script_output_dir else script_path.stem
        log_path = enable_file_logging(str(log_dir), log_name)
        _state_fn = namespace.get("get_robot_state")
        if _state_fn is not None:
            set_state_fn(_state_fn)
        original_stdout = enable_stdout_tee()
        print(f"[run_script] Logging to {log_path}")

    if debug_writer is not None:
        set_tool_event_hooks(
            on_start=debug_writer.tool_start,
            on_end=debug_writer.tool_end,
        )

    namespace.update(wrap_callables_with_timing(namespace))

    # Force script-level ``print(...)`` to resolve through the current
    # ``sys.stdout`` during exec. This keeps script/skill prints aligned with
    # the same tee/capture path used for terminal output and exec.log.
    _original_builtin_print = builtins.print

    def _script_print(*args, sep=" ", end="\n", file=None, flush=False):
        target = sys.stdout if file is None else file
        _original_builtin_print(
            *args,
            sep=sep,
            end=end,
            file=target,
            flush=flush,
        )

    namespace["print"] = _script_print
    _script_builtins = dict(builtins.__dict__)
    _script_builtins["print"] = _script_print
    namespace["__builtins__"] = _script_builtins

    # --- Per-skill async VLM reflection (optional) ---
    _skill_reflect_pool = None
    if debug_writer is not None:
        try:
            from cap.agent.skill_registry import SkillRegistry

            SkillRegistry.get().add_capture_hook(debug_writer.skill_event)
        except Exception as exc:
            debug_writer.emit(
                "debug_warning",
                message=f"failed to register skill hook: {type(exc).__name__}: {exc}",
            )
    sr_cfg = getattr(cfg, "skill_reflection", None)
    if sr_cfg is not None and getattr(sr_cfg, "enabled", False):
        _skill_reflect_pool = _setup_skill_reflection(cfg, sr_cfg, namespace, log_dir)

    # --- Skill library: prepend parent of skill_library/ to sys.path ---
    # skill_library_path points to the skill_library/ dir itself.
    # We insert its parent so that `from skill_library.X import Y` resolves.
    if cfg.skill_library_path:
        _sl_path = Path(cfg.skill_library_path).resolve()
        _sl_parent = str(_sl_path.parent)
        if _sl_parent not in sys.path:
            sys.path.insert(0, _sl_parent)
        print(f"[run_script] Skill library parent on sys.path: {_sl_parent}")

        # Inject a live `skill_library.namespace` module into sys.modules so that
        # saved skill files (which start with `from skill_library.namespace import *`)
        # resolve to the real callable tools instead of hitting AttributeError
        # on names declared in __all__ but never defined in the generated
        # namespace.py stub.
        import types as _types

        # The seed directory does not have to be physically named
        # ``skill_library``.  For example, a human-curated seed dir may be
        # ``cap/saved_scripts/yam_autorl/pin_insert`` but scripts should still
        # deterministically import ``from skill_library.<skill_file> ...``.
        # Register a package alias whose __path__ points at cfg.skill_library_path.
        _pkg_mod = _types.ModuleType("skill_library")
        _pkg_mod.__path__ = [str(_sl_path)]  # type: ignore[attr-defined]
        _pkg_mod.__package__ = "skill_library"
        _init_file = _sl_path / "__init__.py"
        if _init_file.exists():
            _pkg_mod.__file__ = str(_init_file)  # type: ignore[attr-defined]
        sys.modules["skill_library"] = _pkg_mod

        _ns_mod = _types.ModuleType("skill_library.namespace")
        _ns_names: list[str] = []
        for _k, _v in namespace.items():
            if _k.startswith("_") or not callable(_v):
                continue
            setattr(_ns_mod, _k, _v)
            _ns_names.append(_k)
        _ns_mod.__all__ = sorted(_ns_names)  # type: ignore[attr-defined]
        sys.modules["skill_library.namespace"] = _ns_mod
        setattr(_pkg_mod, "namespace", _ns_mod)
        print(f"[run_script] Injected skill_library.namespace with {len(_ns_names)} tools")

    # --- cuRobo motion planner ---
    _planner_instance = None
    _curobo_host = cfg.runtime.curobo_host
    _curobo_port = cfg.runtime.curobo_port

    def create_motion_planner(
        solver_speed="slow", position_threshold=0.01, rotation_threshold=0.1
    ):
        nonlocal _planner_instance
        if _planner_instance is not None:
            return _planner_instance
        from experimental.portal_motion_planner import PortalMotionPlanner

        start_server = _curobo_port == 0
        port = _curobo_port if _curobo_port != 0 else None
        robot_type = os.environ.get("CAP_ROBOT_TYPE", "panda").strip().lower()
        _planner_instance = PortalMotionPlanner(
            backend="curobo",
            solver_speed=solver_speed,
            host=_curobo_host,
            port=port,
            start_server=start_server,
            robot_type=robot_type,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
        )
        return _planner_instance

    namespace["create_motion_planner"] = create_motion_planner

    # --- Recorder ---
    recorder: ScriptRecorder | None = backend.setup_recorder(_env, log_dir, cfg)

    set_artifact_dir(log_dir)
    reset_call_records()

    # --- Capture first frames (before execution) ---
    _vis_cameras = backend.vis_cameras(_env)
    _frames_before = _capture_frames(_env, _vis_cameras)

    print(f"{'─' * 60}")
    print(f"Running: {script_path.name}")
    print(f"{'─' * 60}\n")
    sys.stdout.flush()
    if debug_writer is not None:
        debug_writer.emit("exec_start", script=str(script_path))

    # --- Execute ---
    # sys.stdout may be the profiler's StdoutTee (writes to terminal + log file).
    # In quiet mode we want: log file YES, terminal NO, capture buffer YES.
    # So we still write to the profiler tee (for the log file) but skip the
    # real terminal fd.
    stdout_capture = io.StringIO()
    profiler_tee = sys.stdout  # may be StdoutTee or raw stdout
    # StdoutTee stores the real terminal as _original and log file as _log
    real_terminal = getattr(profiler_tee, "_original", profiler_tee)
    profiler_log = getattr(profiler_tee, "_log", None)
    # Dashboard owns the terminal — force log-only mode so script prints
    # don't fight the live display for stdout.
    quiet = cfg.runtime.quiet or backend.is_quiet

    class _TeeWriter:
        def write(self, s):
            # Always capture for exec.log
            stdout_capture.write(s)
            dashboard_stdout = getattr(_env, "_dashboard", None)
            if dashboard_stdout is not None and hasattr(dashboard_stdout, "on_stdout"):
                try:
                    dashboard_stdout.on_stdout(s)
                except Exception:
                    pass
            if quiet:
                # Write to profiler log file only (skip terminal)
                if (
                    profiler_log is not None
                    and not getattr(profiler_log, "closed", False)
                    and s.strip()
                ):
                    profiler_log.write(s)
                    profiler_log.flush()
            else:
                # Write to profiler tee (terminal + log file)
                profiler_tee.write(s)

        def flush(self):
            try:
                profiler_tee.flush()
            except ValueError:
                pass

        def isatty(self):
            return real_terminal.isatty()

        def fileno(self):
            return real_terminal.fileno()

    tee = _TeeWriter()
    old_stdout = sys.stdout

    exec_error: str | None = None

    old_builtin_print = builtins.print
    try:
        sys.stdout = tee
        builtins.print = _script_print
        with backend.exec_context(namespace, script_path):
            exec(compile(code, str(script_path), "exec"), namespace)  # noqa: S102
    except KeyboardInterrupt:
        print("\n[run_script] Interrupted")
        exec_error = "KeyboardInterrupt"
    except Exception:
        exec_error = traceback.format_exc()
        traceback.print_exc()
    finally:
        builtins.print = old_builtin_print
        sys.stdout = old_stdout
        if recorder is not None:
            _env.set_recorder(None)
            recorder.stop()
        if original_stdout is not None:
            disable_stdout_tee(original_stdout)
        close_file_logging()
        # Be defensive: if any wrapper restored stdout to the script tee, do
        # not let post-exec artifact prints write into a closed profiler log.
        sys.stdout = original_stdout or real_terminal
        if debug_writer is not None:
            debug_writer.emit("exec_end", error=exec_error)

    # --- Post-exec (await_exit, go_home — delegated to backend) ---
    backend.post_exec(exec_error, namespace)

    # --- Capture last frames + save before/after ---
    # Wait for all async skill-reflection Gemini calls to finish before
    # the process exits — otherwise skill_steps/*.md files are never written.
    if _skill_reflect_pool is not None:
        _skill_reflect_pool.shutdown(wait=True)

    _frames_after = _capture_frames(_env, _vis_cameras)
    if _frames_before or _frames_after:
        _save_before_after(_frames_before, _frames_after, log_dir)

    # --- Save artifacts ---
    profiling_path = save_call_records(log_dir / "profiling.json")
    print(f"[run_script] Profiling: {profiling_path}")

    # --- Flush skill execution logs ---
    if cfg.skill_library_path:
        try:
            from cap.agent.skill_registry import SkillRegistry

            registry = SkillRegistry.get()
            if registry.profiles():
                skill_log_data = registry.to_json()
                (log_dir / "skill_log.json").write_text(
                    json.dumps(skill_log_data, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[run_script] Skill log: {len(skill_log_data)} skill(s) recorded"
                )
        except Exception as _e:
            print(f"[run_script] Skill log flush failed: {_e}")

    _save_result(log_dir, namespace.get("get_task_info"), exec_error)

    # exec.log
    stdout_text = stdout_capture.getvalue()
    if stdout_text or exec_error:
        parts = []
        if stdout_text:
            parts.append("--- stdout ---\n" + stdout_text)
        if exec_error:
            parts.append("\n--- error ---\n" + exec_error)
        (log_dir / "exec.log").write_text("\n".join(parts), encoding="utf-8")

    # --- Summary ---
    result_path = log_dir / "result.json"
    if result_path.exists():
        result_data = json.loads(result_path.read_text())
        print(
            f"[run_script] reward={result_data.get('score', 0.0)}  "
            f"success={result_data.get('success', False)}"
        )

    print(f"\n{'─' * 60}")
    if log_path is not None:
        print(f"[run_script] Log saved to {log_path}")
    print(f"[run_script] Output: {log_dir}")
    print("[run_script] Done")
    if debug_writer is not None:
        debug_writer.emit("run_end", output=str(log_dir), error=exec_error)
        debug_writer.close()
    _terminate_debug_ui(debug_ui_proc, debug_cfg)


if __name__ == "__main__":
    main()
