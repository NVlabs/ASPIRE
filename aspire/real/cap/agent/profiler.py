# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure-level profiling for CAP tool callables.

Wraps each callable with timing that prints duration and idle gaps
between consecutive API calls.  Optionally writes comprehensive
structured logs to a text file in real-time.
"""

from __future__ import annotations

import dataclasses
import functools
import io
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_call_counter = [0]
_last_call_end: list[float | None] = [None]  # timestamp of previous call's end
_call_records: list[dict[str, Any]] = []  # structured per-call profiling records

# File logger — set via `enable_file_logging()`
_log_file: io.TextIOBase | None = None
_log_path: Path | None = None
_get_state_fn: Callable[..., Any] | None = None  # bound later to get_robot_state
_on_tool_start: Callable[..., Any] | None = None
_on_tool_end: Callable[..., Any] | None = None
_quiet: bool = False  # when True, skip console prints (still log to file)


def set_quiet(quiet: bool = True) -> None:
    """Suppress profiler console output (file logging is unaffected)."""
    global _quiet
    _quiet = quiet


def enable_file_logging(log_dir: str | Path, script_name: str = "") -> Path:
    """Open a log file in *log_dir* and redirect all profile output there.

    Returns the log file path.
    """
    global _log_file, _log_path
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    tag = f"_{script_name}" if script_name else ""
    _log_path = d / f"run{tag}_{stamp}.txt"
    _log_file = open(
        _log_path,
        "w",
        buffering=1,
        encoding="utf-8",
        errors="replace",
    )  # line-buffered
    _log("=" * 72)
    _log(f"Log started: {datetime.now().isoformat()}")
    _log(f"Script: {script_name}")
    _log(f"PID: {os.getpid()}")
    _log("=" * 72)
    return _log_path


def set_state_fn(fn: Callable[..., Any]) -> None:
    """Register get_robot_state so the profiler can snapshot state."""
    global _get_state_fn
    _get_state_fn = fn


def set_tool_event_hooks(
    on_start: Callable[..., Any] | None = None,
    on_end: Callable[..., Any] | None = None,
) -> None:
    """Register callbacks for tool-call lifecycle events."""
    global _on_tool_start, _on_tool_end
    _on_tool_start = on_start
    _on_tool_end = on_end


def close_file_logging() -> None:
    global _log_file
    if _log_file is not None:
        _log("=" * 72)
        _log(f"Log ended: {datetime.now().isoformat()}")
        _log("=" * 72)
        _log_file.close()
        _log_file = None


def get_call_records() -> list[dict[str, Any]]:
    """Return accumulated per-call profiling records."""
    return list(_call_records)


def reset_call_records() -> None:
    """Clear accumulated call records."""
    _call_records.clear()


def save_call_records(path: str | Path) -> Path:
    """Dump accumulated call records to a JSON file. Returns the path."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    total_ms = sum(r["duration_ms"] for r in _call_records)
    total_idle_ms = sum(r.get("idle_gap_ms", 0) for r in _call_records)
    errors = [r for r in _call_records if r.get("error")]
    tool_stats: dict[str, dict[str, Any]] = {}
    for r in _call_records:
        name = r["tool"]
        if name not in tool_stats:
            tool_stats[name] = {"count": 0, "total_ms": 0.0, "errors": 0}
        tool_stats[name]["count"] += 1
        tool_stats[name]["total_ms"] += r["duration_ms"]
        if r.get("error"):
            tool_stats[name]["errors"] += 1

    payload = {
        "summary": {
            "total_calls": len(_call_records),
            "total_duration_ms": round(total_ms, 1),
            "total_idle_ms": round(total_idle_ms, 1),
            "errors": len(errors),
        },
        "per_tool": {
            k: {
                "count": v["count"],
                "total_ms": round(v["total_ms"], 1),
                "errors": v["errors"],
            }
            for k, v in sorted(tool_stats.items(), key=lambda x: -x[1]["total_ms"])
        },
        "calls": _call_records,
    }
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p


def _log(msg: str) -> None:
    """Write a line to the log file (if open)."""
    if _log_file is not None:
        _log_file.write(msg + "\n")


def _fmt_arg(v: Any, max_len: int = 200) -> str:
    """Format a single argument for logging."""
    if isinstance(v, np.ndarray):
        if v.size <= 20:
            return f"array({np.array2string(v, precision=4, separator=', ')})"
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    s = repr(v)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _fmt_args(args: tuple, kwargs: dict) -> str:
    """Format call arguments for logging."""
    parts = [_fmt_arg(a) for a in args]
    parts += [f"{k}={_fmt_arg(v)}" for k, v in kwargs.items()]
    return ", ".join(parts)


def _fmt_result(result: Any, max_len: int = 400) -> str:
    """Format a return value for logging."""
    if result is None:
        return "None"
    if isinstance(result, np.ndarray):
        if result.size <= 20:
            return f"array({np.array2string(result, precision=4, separator=', ')})"
        return f"ndarray(shape={result.shape}, dtype={result.dtype})"
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        fields = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
        return _fmt_arg(fields, max_len)
    if isinstance(result, list) and len(result) > 0:
        items = [_fmt_arg(r, 120) for r in result[:5]]
        suffix = f" ... +{len(result) - 5}" if len(result) > 5 else ""
        return f"[{', '.join(items)}{suffix}]"
    if isinstance(result, dict):
        return _fmt_arg(result, max_len)
    s = repr(result)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _snapshot_state() -> str | None:
    """Capture a compact robot state snapshot."""
    if _get_state_fn is None:
        return None
    try:
        st = _get_state_fn()

        def _f(lst: list[float]) -> str:
            return "[" + ", ".join(f"{v:.4f}" for v in lst) + "]"

        lines = [
            f"    L_ee_pos={_f(st.left_ee_pos)}  L_grip={st.left_gripper_pos:.3f}",
            f"    R_ee_pos={_f(st.right_ee_pos)}  R_grip={st.right_gripper_pos:.3f}",
            f"    L_jp={_f(st.left_joint_pos)}",
            f"    R_jp={_f(st.right_joint_pos)}",
        ]
        return "\n".join(lines)
    except Exception:
        return None


# Tools after which we snapshot robot state (state-changing tools)
_STATE_SNAPSHOT_TOOLS = {
    "_ik_servo",
    "go_home",
    "open_gripper",
    "close_gripper",
    "set_gripper",
    "learn_skill",
}


def wrap_callables_with_timing(
    callables: dict[str, Callable[..., Any]],
) -> dict[str, Callable[..., Any]]:
    """Wrap each callable with timing/profiling that prints duration.

    Returns a new dict with the same keys and wrapped values.
    """
    wrapped: dict[str, Callable[..., Any]] = {}
    for name, fn in callables.items():
        if not callable(fn):
            continue
        wrapped[name] = _wrap_with_timing(name, fn)
    return wrapped


def _wrap_with_timing(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _call_counter[0] += 1
        call_id = _call_counter[0]
        t_now = time.time()
        if _on_tool_start is not None:
            try:
                _on_tool_start(name=name, call_id=call_id, args=args, kwargs=kwargs)
            except Exception:
                logger.warning(
                    "Profiler on_start hook failed for %s", name, exc_info=True
                )

        # Compute wait (idle) time since the last API call ended
        wait_ms = 0.0
        wait_str = ""
        if _last_call_end[0] is not None:
            wait_ms = (t_now - _last_call_end[0]) * 1000
            wait_str = f"  (wait {wait_ms:.1f}ms)"

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        console_msg = f"[profile {ts}] #{call_id} {name}{wait_str}"
        if not _quiet:
            print(console_msg)

        # File log: START with full arguments
        args_str = _fmt_args(args, kwargs)
        _log(f"[{ts}] #{call_id} CALL {name}({args_str})")
        if wait_ms > 0:
            _log(f"  idle_gap={wait_ms:.1f}ms")

        try:
            result = fn(*args, **kwargs)
            elapsed = (time.time() - t_now) * 1000
            ts2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            console_end = f"[profile {ts2}] #{call_id} {name} -> {elapsed:.1f}ms"
            if not _quiet:
                print(console_end)

            # File log: result + timing
            _log(f"[{ts2}] #{call_id} OK {name} -> {elapsed:.1f}ms")
            _log(f"  result={_fmt_result(result)}")

            # Structured record — keep a JSON-safe copy of dict results so
            # downstream consumers (wandb, result.json parsers) can read
            # fields back without parsing the truncated repr in ``result``.
            record = {
                "call_id": call_id,
                "tool": name,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": round(elapsed, 1),
                "idle_gap_ms": round(wait_ms, 1),
                "args": args_str,
                "result": _fmt_result(result),
                "error": None,
            }
            if isinstance(result, dict):
                try:
                    import json as _json

                    record["result_data"] = _json.loads(
                        _json.dumps(result, default=str)
                    )
                except Exception:
                    pass
            _call_records.append(record)

            # State snapshot after state-changing tools
            if name in _STATE_SNAPSHOT_TOOLS:
                snap = _snapshot_state()
                if snap:
                    _log("  [state_after]")
                    _log(snap)

            _last_call_end[0] = time.time()
            if _on_tool_end is not None:
                try:
                    _on_tool_end(
                        name=name,
                        call_id=call_id,
                        result=result,
                        error=None,
                        elapsed_ms=elapsed,
                    )
                except Exception:
                    logger.warning(
                        "Profiler on_end hook failed for %s", name, exc_info=True
                    )
            return result
        except Exception as e:
            elapsed = (time.time() - t_now) * 1000
            ts2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if not _quiet:
                print(f"[profile {ts2}] #{call_id} {name} ERROR {elapsed:.1f}ms: {e}")

            _log(f"[{ts2}] #{call_id} ERROR {name} -> {elapsed:.1f}ms")
            _log(f"  exception={type(e).__name__}: {e}")

            # Structured record (error case)
            _call_records.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": round(elapsed, 1),
                    "idle_gap_ms": round(wait_ms, 1),
                    "args": args_str,
                    "result": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

            _last_call_end[0] = time.time()
            if _on_tool_end is not None:
                try:
                    _on_tool_end(
                        name=name,
                        call_id=call_id,
                        result=None,
                        error=e,
                        elapsed_ms=elapsed,
                    )
                except Exception:
                    logger.warning(
                        "Profiler on_end hook failed for %s", name, exc_info=True
                    )
            raise

    return wrapper


class StdoutTee:
    """Tee stdout to both the terminal and a log file."""

    def __init__(self, log_file: io.TextIOBase, original: Any) -> None:
        self._log = log_file
        self._original = original

    def write(self, s: str) -> int:
        self._original.write(s)
        if s.strip():
            self._log.write(s)
            self._log.flush()
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        self._log.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return False


def enable_stdout_tee() -> Any:
    """Start teeing stdout to the log file. Returns the original stdout."""
    global _log_file
    if _log_file is None:
        return sys.stdout
    original = sys.stdout
    sys.stdout = StdoutTee(_log_file, original)  # type: ignore[assignment]
    return original


def disable_stdout_tee(original: Any) -> None:
    """Restore original stdout."""
    sys.stdout = original
