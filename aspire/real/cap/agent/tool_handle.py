# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Background tool execution with controlled stop.

Provides ToolHandle and ToolRunner so generated code can run any tool
concurrently and stop it at any time — including slow HTTP calls and
Portal RPC calls — via PyThreadState_SetAsyncExc (same mechanism as
Executor.force_cancel()).

Injected into the execution namespace as:
    run_in_background(fn, *args, **kwargs) -> ToolHandle
    stop_all_tools()
"""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolStopped(Exception):
    """Raised inside a tool's thread when ToolHandle.stop() is called."""


@dataclass
class ToolHandle:
    """Handle for a tool call running in a background daemon thread.

    Usage::

        handle = run_in_background(_ik_servo, "right", pos, quat)
        # ... monitor ...
        handle.stop()   # stops the tool immediately
        handle.wait()   # optional: wait for thread to exit
    """

    call_id: int
    tool_name: str
    _thread: threading.Thread = field(repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _result: Any = field(default=None, repr=False)
    _error: Exception | None = field(default=None, repr=False)
    _stopped: bool = False

    def done(self) -> bool:
        """Return True if the tool call has finished (success, error, or stopped)."""
        return self._done.is_set()

    def result(self, timeout: float | None = None) -> Any:
        """Block until the tool finishes and return its result.

        Raises:
            TimeoutError: if *timeout* expires before the tool finishes.
            ToolStopped: if stop() was called.
            Exception: any exception raised by the tool itself.
        """
        if not self._done.wait(timeout=timeout):
            raise TimeoutError(f"{self.tool_name} did not finish within {timeout}s")
        if self._stopped:
            raise ToolStopped(f"{self.tool_name} was stopped")
        if self._error is not None:
            raise self._error
        return self._result

    def stop(self) -> None:
        """Stop this tool call NOW.

        Injects ToolStopped into the tool's thread via PyThreadState_SetAsyncExc.
        This interrupts: time.sleep(), requests/httpx HTTP calls, Portal RPC,
        Anthropic/OpenAI SDK calls — any Python-level blocking operation.
        Has no effect if the tool has already finished.
        """
        if self._done.is_set():
            return
        self._stopped = True
        tid = self._thread.ident
        if tid is not None and self._thread.is_alive():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(ToolStopped),
            )

    def wait(self, timeout: float | None = None) -> None:
        """Wait for the thread to exit. Does not raise."""
        self._thread.join(timeout=timeout)


class ToolRunner:
    """Runs tool calls in background threads and tracks active handles.

    Thread-safe. Injected into the execution namespace as ``run_in_background``
    and ``stop_all_tools``.
    """

    def __init__(self) -> None:
        self._active: dict[int, ToolHandle] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def start(self, fn: Callable, *args: Any, **kwargs: Any) -> ToolHandle:
        """Run *fn* in a background daemon thread. Returns a ToolHandle immediately."""
        with self._lock:
            self._counter += 1
            call_id = self._counter

        name = getattr(fn, "__name__", str(fn))
        # placeholder thread — replaced below before start()
        handle = ToolHandle(
            call_id=call_id,
            tool_name=name,
            _thread=threading.current_thread(),  # temp placeholder
        )

        def _run() -> None:
            try:
                handle._result = fn(*args, **kwargs)
            except ToolStopped:
                handle._stopped = True
            except Exception as exc:
                handle._error = exc
            finally:
                handle._done.set()
                with self._lock:
                    self._active.pop(call_id, None)

        t = threading.Thread(
            target=_run,
            daemon=True,
            name=f"tool-{name}-{call_id}",
        )
        handle._thread = t
        with self._lock:
            self._active[call_id] = handle
        t.start()
        return handle

    def stop_all(self) -> None:
        """Stop all currently running tool calls."""
        with self._lock:
            handles = list(self._active.values())
        for h in handles:
            h.stop()

    def get_running(self) -> list[ToolHandle]:
        """Return handles for tool calls still in progress."""
        with self._lock:
            return [h for h in self._active.values() if not h.done()]


def make_tool_runner_namespace(runner: ToolRunner | None = None) -> dict[str, Any]:
    """Return namespace entries for background tool execution.

    Inject into the script/agent execution namespace::

        namespace.update(make_tool_runner_namespace())
    """
    r = runner or ToolRunner()
    return {
        "run_in_background": r.start,
        "stop_all_tools": r.stop_all,
        "ToolStopped": ToolStopped,
        "_tool_runner": r,
    }
