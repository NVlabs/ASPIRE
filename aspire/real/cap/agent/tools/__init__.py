"""Minimal CAP tool registry for the openforge real-YAM runtime."""

from __future__ import annotations

from typing import Any

from cap.agent.tools.base import Tool, ToolResult
from cap.agent.tools.base import (  # noqa: F401
    Detection3D,
    FreespaceResult,
    MoveResult,
    RobotState,
    SkillResult,
    ToolParameter,
)


class ToolRegistry:
    """Container for registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(**kwargs)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def callable_dict(self) -> dict[str, Any]:
        def _make_fn(tool: Tool):
            param_names = [p.name for p in tool.parameters]
            allowed = set(param_names)

            def fn(*args: Any, **kw: Any) -> Any:
                for i, val in enumerate(args):
                    if i < len(param_names):
                        kw[param_names[i]] = val
                unknown = set(kw) - allowed
                if unknown:
                    raise TypeError(
                        f"{tool.name}() got unexpected keyword argument(s) "
                        f"{sorted(unknown)}. Known: {sorted(allowed)}."
                    )
                result = tool.execute(**kw)
                if not result.success:
                    raise RuntimeError(f"Tool {tool.name} failed: {result.error}")
                return result.data

            fn.__name__ = tool.name
            fn.__doc__ = tool.description
            return fn

        return {name: _make_fn(tool) for name, tool in self._tools.items()}


def create_default_registry(
    cap_server_host: str = "localhost",
    cap_server_port: int | None = None,
    detection_host: str = "localhost",
    detection_port: int | None = None,
    bundlesdf_host: str | None = None,
    bundlesdf_port: int | None = None,
    sam3_host: str | None = None,
    sam3_port: int | None = None,
) -> ToolRegistry:
    """Build the reduced registry needed by the table-bussing runtime."""
    from cap.config import (
        BUNDLESDF_SERVER_HOST,
        BUNDLESDF_SERVER_PORT,
        CAP_SERVER_PORT,
        DETECTION_SERVER_PORT,
        SAM3_SERVER_HOST,
        SAM3_SERVER_PORT,
    )
    from cap.agent.tools.detection import DetectObjectTool, DetectObjectsOneshotTool
    from cap.agent.tools.freespace_move import FreespaceMoveTool
    from cap.agent.tools.grasp_anygrasp import SampleGraspPoseAnyGraspTool
    from cap.agent.tools.native import (
        CloseGripperTool,
        GetCameraImageTool,
        GetRobotStateTool,
        GoHomeTool,
        OpenGripperTool,
        SetGripperTool,
    )
    from cap.agent.tools.vlm_query import VlmQueryTool

    srv_port = cap_server_port or CAP_SERVER_PORT
    det_port = detection_port or DETECTION_SERVER_PORT
    bsdf_host = bundlesdf_host or BUNDLESDF_SERVER_HOST
    bsdf_port = bundlesdf_port or BUNDLESDF_SERVER_PORT
    s3_host = sam3_host or SAM3_SERVER_HOST
    s3_port = sam3_port or SAM3_SERVER_PORT

    registry = ToolRegistry()
    registry.register(GetRobotStateTool(host=cap_server_host, port=srv_port))
    registry.register(SetGripperTool(host=cap_server_host, port=srv_port))
    registry.register(OpenGripperTool(host=cap_server_host, port=srv_port))
    registry.register(CloseGripperTool(host=cap_server_host, port=srv_port))
    registry.register(GoHomeTool(host=cap_server_host, port=srv_port))
    registry.register(GetCameraImageTool(host=cap_server_host, port=srv_port))

    detect_tool = DetectObjectTool(
        detection_host=detection_host,
        detection_port=det_port,
        cap_server_host=cap_server_host,
        cap_server_port=srv_port,
        bundlesdf_host=bsdf_host,
        bundlesdf_port=bsdf_port,
    )
    registry.register(DetectObjectsOneshotTool(detect_tool=detect_tool))
    registry.register(
        VlmQueryTool(cap_server_host=cap_server_host, cap_server_port=srv_port)
    )
    registry.register(FreespaceMoveTool(host=cap_server_host, port=srv_port))
    registry.register(
        SampleGraspPoseAnyGraspTool(
            cap_server_host=cap_server_host,
            cap_server_port=srv_port,
            sam3_url=f"http://{s3_host}:{s3_port}",
        )
    )
    return registry
