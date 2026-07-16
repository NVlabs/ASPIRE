"""Artifact and timeout utilities for YAM saved scripts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def json_safe(value: Any) -> Any:
    """Convert dataclasses, numpy values, and small objects to JSON-safe data."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return repr(value)


def current_run_dir(task_name: str = "yam_runtime") -> Path:
    """Return the active run-script log dir, or a deterministic fallback."""
    try:
        from cap.agent.tools import _artifact_log

        artifact_dir = getattr(_artifact_log, "_artifact_dir", None)
        if artifact_dir is not None:
            return Path(artifact_dir).parent
    except Exception:
        pass
    raw = os.environ.get("OPENFORGE_RUN_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path("logs") / f"{task_name}_{stamp()}"


def write_json(path: str | Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return str(path)


def append_stage_summary(run_dir: str | Path, lines: list[str]) -> str:
    path = Path(run_dir) / "stage_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(
        existing + ("\n" if existing and not existing.endswith("\n") else "") + text,
        encoding="utf-8",
    )
    return str(path)


def call_with_timeout(
    label: str,
    fn: Callable[..., Any],
    timeout_s: float,
    *args: Any,
    run_in_background: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call a tool with optional run_in_background timeout support."""
    if callable(run_in_background) and timeout_s and timeout_s > 0:
        handle = run_in_background(fn, *args, **kwargs)
        try:
            return {"ok": True, "label": label, "data": handle.result(timeout=timeout_s)}
        except TimeoutError:
            try:
                handle.stop()
            except Exception:
                pass
            return {"ok": False, "label": label, "error": f"timeout after {timeout_s}s"}
        except Exception as exc:
            return {"ok": False, "label": label, "error": f"{type(exc).__name__}: {exc}"}
    try:
        return {"ok": True, "label": label, "data": fn(*args, **kwargs)}
    except Exception as exc:
        return {"ok": False, "label": label, "error": f"{type(exc).__name__}: {exc}"}
