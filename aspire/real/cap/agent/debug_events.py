# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured live debug event logging for run_script.py.

The writer appends one JSON object per line to ``debug_events.jsonl`` inside a
run log directory.  It is intentionally dependency-free so robot execution does
not depend on the web UI process.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def _json_safe(value: Any, max_repr: int = 500) -> Any:
    """Best-effort conversion to JSON-safe compact values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, max_repr=max_repr) for v in value[:20]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= 40:
                out["..."] = f"+{len(value) - idx} more"
                break
            out[str(k)] = _json_safe(v, max_repr=max_repr)
        return out
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            if value.size <= 20:
                return value.tolist()
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    except Exception:
        pass
    text = repr(value)
    if len(text) > max_repr:
        text = text[:max_repr] + "..."
    return text


def _callable_source(fn: Callable[..., Any] | None) -> dict[str, Any]:
    if fn is None:
        return {}
    try:
        unwrapped = inspect.unwrap(fn)
    except Exception:
        unwrapped = fn
    try:
        source_file = inspect.getsourcefile(unwrapped) or inspect.getfile(unwrapped)
    except Exception:
        source_file = None
    try:
        line = int(getattr(unwrapped, "__code__", None).co_firstlineno)  # type: ignore[union-attr]
    except Exception:
        line = None
    data: dict[str, Any] = {}
    if source_file:
        data["source_file"] = os.path.abspath(source_file)
    if line is not None:
        data["source_line"] = line
    return data


class DebugEventWriter:
    """Append-only JSONL event writer used by run_script and profiler hooks."""

    def __init__(self, log_dir: str | Path, filename: str = "debug_events.jsonl") -> None:
        self.log_dir = Path(log_dir)
        self.path = self.log_dir / filename
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")
        self._tool_sources: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def register_tool_sources(self, namespace: dict[str, Any]) -> None:
        """Capture best-effort source file/line for callable namespace tools."""
        for name, fn in namespace.items():
            if name.startswith("_") or not callable(fn):
                continue
            source = _callable_source(fn)
            if source:
                self._tool_sources[name] = source

    def emit(self, event_type: str, **data: Any) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": event_type,
            **{k: _json_safe(v) for k, v in data.items()},
        }
        line = json.dumps(payload, default=str, ensure_ascii=False)
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self._fh.flush()

    # Profiler hook signatures.
    def tool_start(self, *, name: str, call_id: int, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.emit(
            "tool_start",
            name=name,
            call_id=call_id,
            args=_json_safe(args, max_repr=250),
            kwargs=_json_safe(kwargs, max_repr=250),
            **self._tool_sources.get(name, {}),
        )

    def tool_end(
        self,
        *,
        name: str,
        call_id: int,
        result: Any,
        error: BaseException | None,
        elapsed_ms: float,
    ) -> None:
        self.emit(
            "tool_end",
            name=name,
            call_id=call_id,
            elapsed_ms=round(float(elapsed_ms), 1),
            result=_json_safe(result, max_repr=350),
            error=None if error is None else f"{type(error).__name__}: {error}",
            **self._tool_sources.get(name, {}),
        )

    # SkillRegistry hook signature.
    def skill_event(self, skill_name: str, call_id: int, phase: str) -> None:
        source: dict[str, Any] = {}
        try:
            from cap.agent.skill_registry import SkillRegistry

            profile = SkillRegistry.get().profiles().get(skill_name)
            if profile is not None:
                source_file = getattr(profile, "source_file", None)
                source_line = getattr(profile, "source_line", None)
                if source_file:
                    source["source_file"] = source_file
                if source_line is not None:
                    source["source_line"] = source_line
        except Exception:
            pass
        if phase == "before":
            event_type = "skill_start"
        elif phase == "after":
            event_type = "skill_end"
        else:
            event_type = "skill_error"
        self.emit(
            event_type,
            name=skill_name,
            call_id=call_id,
            phase=phase,
            **source,
        )
