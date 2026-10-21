"""Synchronous client for ``cerulion-rcpd``, the Cerulion MCP daemon.

``cerulion-rcpd`` is an MCP server over a running Cerulion graph. It lists only
the tools its manifest declares plus a built-in ``estop``. It answers a call
with one text block holding JSON; a refused call carries ``isError`` and
``{"error": {"code", "message", ...}, "receipt": ...}``.

The adapter is synchronous and the ``mcp`` client is asyncio, so this module
runs one event loop on a private thread. One task on that loop owns the whole
stdio session (the anyio scopes inside ``stdio_client`` must exit in the task
that entered them); calls are submitted to the loop from the caller's thread.
The daemon is a child process; closing the client ends it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import concurrent.futures
from datetime import timedelta
import json
import os
from pathlib import Path
import shlex
import threading
from typing import Any

RCPD_CMD_ENV = "CAP_HARNESS_CERULION_RCPD_CMD"
"""Full daemon command line, shell-quoted. Overrides the manifest arguments."""

REQUIRED_TOOLS = ("arm.reset", "arm.step")
DEFAULT_CALL_TIMEOUT_S = 10.0
#: How long close() lets the session end by itself before cancelling it.
CLOSE_GRACE_S = 2.0


class RcpError(RuntimeError):
    """A call the daemon refused (``isError``). ``body`` is the full refusal object."""

    def __init__(self, tool: str, body: Mapping[str, Any]) -> None:
        error = body.get("error") if isinstance(body.get("error"), Mapping) else {}
        self.tool = tool
        self.body = dict(body)
        self.error = dict(error)
        self.code = str(error.get("code", "error"))
        self.message = str(error.get("message", ""))
        receipt = body.get("receipt")
        self.receipt = dict(receipt) if isinstance(receipt, Mapping) else None
        super().__init__(f"{tool}: {self.code}: {self.message}")

    @property
    def may_have_executed(self) -> bool:
        value = self.error.get("may_have_executed", self.body.get("may_have_executed", False))
        return bool(value)


class RcpTransportError(RuntimeError):
    """The daemon could not be started, or its session ended."""


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the cerulion runtime target needs the `mcp` package: "
            "install ASPIRE with the `cerulion` extra (pip install -e '.[cerulion]')"
        ) from exc


def rcpd_command(
    manifest: str | Path | None = None,
    receipts: str | Path | None = None,
    *,
    executable: str = "cerulion-rcpd",
) -> list[str]:
    """The daemon command: ``$CAP_HARNESS_CERULION_RCPD_CMD`` when set, else built from arguments."""
    override = os.environ.get(RCPD_CMD_ENV)
    if override:
        command = shlex.split(override)
        if not command:
            raise ValueError(f"{RCPD_CMD_ENV} is set but empty")
        return command
    if manifest is None:
        raise ValueError(f"an rcpd manifest is required (or set {RCPD_CMD_ENV})")
    command = [executable, "--manifest", str(Path(manifest).expanduser()), "--stdio"]
    if receipts is not None:
        command += ["--receipts", str(Path(receipts).expanduser())]
    return command


def _decode_text(result: Any) -> dict[str, Any]:
    texts = [
        item.text for item in getattr(result, "content", ()) if getattr(item, "type", "") == "text"
    ]
    if len(texts) != 1:
        raise RcpTransportError(f"expected one text content in the tool result, got {len(texts)}")
    payload = json.loads(texts[0])
    if not isinstance(payload, dict):
        raise RcpTransportError("tool result text is not a JSON object")
    return payload


class RcpClient:
    """One stdio MCP session to ``cerulion-rcpd``; ``call`` blocks until the reply."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        startup_timeout_s: float = 30.0,
        required_tools: Sequence[str] = REQUIRED_TOOLS,
    ) -> None:
        _require_mcp()
        if not command:
            raise ValueError("rcpd command must not be empty")
        self._command = [str(part) for part in command]
        self._env = dict(os.environ if env is None else env)
        self._cwd = None if cwd is None else str(cwd)
        self._call_timeout_s = float(call_timeout_s)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="cerulion-rcpd-client", daemon=True
        )
        self._thread.start()
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._runner: concurrent.futures.Future[None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.tool_names: tuple[str, ...] = ()
        try:
            self._runner = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
            self._submit(self._wait_ready(), startup_timeout_s)
            missing = [name for name in required_tools if name not in self.tool_names]
            if missing:
                raise RcpTransportError(
                    f"{self._command[0]} does not list {missing}; listed: {list(self.tool_names)}"
                )
        except BaseException:
            self.close()
            raise

    def _submit(self, coroutine: Any, timeout_s: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout_s)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise RcpTransportError(f"rcpd did not answer within {timeout_s:.1f} s") from exc

    async def _wait_ready(self) -> None:
        while self._session is None:
            runner = self._runner
            if runner is not None and runner.done():
                exc = runner.exception()
                raise RcpTransportError(f"cannot start {self._command[0]}: {exc}") from exc
            await asyncio.sleep(0.01)

    async def _run(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._task = asyncio.current_task()
        self._stop = asyncio.Event()
        if self._closed:
            self._stop.set()
        params = StdioServerParameters(
            command=self._command[0], args=self._command[1:], env=self._env, cwd=self._cwd
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()
            self.tool_names = tuple(tool.name for tool in listed.tools)
            self._session = session
            try:
                await self._stop.wait()
            finally:
                self._session = None

    async def _call(self, tool: str, arguments: Mapping[str, Any]) -> Any:
        return await self._session.call_tool(
            tool,
            dict(arguments),
            read_timeout_seconds=timedelta(seconds=self._call_timeout_s),
        )

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Call one tool; returns the decoded success payload or raises :class:`RcpError`."""
        if self._closed or self._session is None:
            raise RcpTransportError("rcp client is closed or its session ended")
        try:
            result = self._submit(self._call(tool, arguments or {}), self._call_timeout_s + 5.0)
        except RcpTransportError:
            raise
        except Exception as exc:
            raise RcpTransportError(f"{tool}: MCP call failed: {exc}") from exc
        payload = _decode_text(result)
        if getattr(result, "isError", False):
            raise RcpError(tool, payload)
        return payload

    def _signal_stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def _cancel_session(self) -> None:
        """Cancel the session task and wait until it has actually finished."""
        task = self._task

        async def cancel_and_wait() -> None:
            if task is None or task.done():
                return
            task.cancel()
            await asyncio.wait({task})

        try:
            asyncio.run_coroutine_threadsafe(cancel_and_wait(), self._loop).result(10.0)
        except Exception:
            pass

    def close(self) -> None:
        """End the session and the daemon. Idempotent.

        The session task normally ends by itself once the stop flag is set. A
        task stuck before that point (in ``initialize``, say) is cancelled
        after :data:`CLOSE_GRACE_S`, and the loop is stopped only after the
        task has finished, so ``stdio_client`` always terminates the child.
        """
        if self._closed:
            return
        self._closed = True
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._signal_stop)
        runner = self._runner
        if runner is not None and self._loop.is_running():
            try:
                runner.result(CLOSE_GRACE_S)
            except concurrent.futures.TimeoutError:
                self._cancel_session()
            except Exception:
                pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        if not self._thread.is_alive():
            self._loop.close()


__all__ = [
    "RCPD_CMD_ENV",
    "RcpClient",
    "RcpError",
    "RcpTransportError",
    "rcpd_command",
]
