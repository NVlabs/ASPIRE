"""Attach a persistent agent session to a caller-owned CAP API."""

from typing import Any

from .session import LiveAgentSession


class EmbeddedAgentRuntime:
    """Own session jobs and monitoring; the caller owns the environment."""

    def __init__(
        self,
        api: Any,
        registry: Any,
        *,
        operation_controller: Any = None,
        recorder: Any = None,
        event_sink: Any = None,
        tool_result_sink: Any = None,
    ) -> None:
        self.session = LiveAgentSession(
            api,
            registry,
            operation_controller=operation_controller,
            recorder=recorder,
            event_sink=event_sink,
            tool_result_sink=tool_result_sink,
        )

    def close(self) -> None:
        self.session.close()
