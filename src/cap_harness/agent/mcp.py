"""Allowlisted MCP tools over a persistent robot session."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping, Sequence
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from contextlib import asynccontextmanager
import inspect
from io import BytesIO
import json
import types as python_types
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpy as np

from cap_harness.contracts import MotionStrategy
from cap_harness.geometry import pose_from_mapping

from .session import LiveAgentSession, _compact_summary


def _object_schema(
    properties: Mapping[str, Any] | None = None, *, required: Sequence[str] = ()
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties or {}),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


_CAMERAS = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Camera names to return; defaults to the station's primary camera.",
}


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation in (inspect.Parameter.empty, Any, object):
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is np.ndarray:
        return {"type": "array"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (list, tuple, set, Sequence, SequenceABC):
        item = _annotation_schema(arguments[0]) if arguments else {}
        return {"type": "array", **({"items": item} if item else {})}
    if origin in (dict, Mapping, MappingABC):
        return {"type": "object"}
    if origin in (Union, python_types.UnionType):
        schemas = [
            {"type": "null"} if item is type(None) else _annotation_schema(item)
            for item in arguments
        ]
        schemas = [schema for schema in schemas if schema]
        if len(schemas) == 1:
            return schemas[0]
        if schemas:
            return {"anyOf": schemas}
    return {}


def _cap_input_schema(spec: Any) -> dict[str, Any]:
    """Derive the JSON-callable portion of a registered Python signature."""
    try:
        hints = get_type_hints(spec.function)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    additional = False
    for parameter in spec.signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            additional = True
            continue
        schema = _annotation_schema(hints.get(parameter.name, parameter.annotation))
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        elif isinstance(parameter.default, (str, int, float, bool)) or parameter.default is None:
            schema = {**schema, "default": parameter.default}
        properties[parameter.name] = schema
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional,
    }
    if required:
        result["required"] = required
    return result


def _pose_input_schema() -> dict[str, Any]:
    schema = _object_schema(
        {
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
            "quaternion_wxyz": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Absolute unit quaternion [w, x, y, z] in the pose frame.",
            },
            "rpy_deg": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Absolute [roll, pitch, yaw] in degrees in the pose frame. "
                    "Fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."
                ),
            },
            "frame": {"type": "string", "minLength": 1},
        },
        required=("position", "frame"),
    )
    schema["oneOf"] = [{"required": ["quaternion_wxyz"]}, {"required": ["rpy_deg"]}]
    schema["description"] = "Absolute pose; provide exactly one of quaternion_wxyz or rpy_deg."
    return schema


def _motion_strategy_input_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "ik_solver": {
                "type": "string",
                "enum": ["pyroki", "curobo", "mink"],
                "default": "pyroki",
            },
            "trajectory_planner": {
                "type": "string",
                "enum": ["interpolation", "curobo"],
                "default": "interpolation",
            },
            "pose_planner": {
                "type": "string",
                "enum": ["composed", "curobo-integrated"],
                "default": "composed",
            },
            "time_dilation_factor": {
                "type": ["number", "null"],
                "exclusiveMinimum": 0.0,
                "maximum": 1.0,
            },
            "interpolation_dt_s": {
                "type": ["number", "null"],
                "exclusiveMinimum": 0.0,
            },
            "maximum_trajectory_dt_s": {
                "type": ["number", "null"],
                "exclusiveMinimum": 0.0,
            },
        }
    )


def _encode_jpeg(frame: np.ndarray, *, max_side: int = 960) -> str:
    from PIL import Image

    array = np.asarray(frame)
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    image.thumbnail((max_side, max_side))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_DIRECT = {
    "get_robot_state",
    "localize_object",
    "move_to_joints",
    "move_to_pose",
    "move_synchronized",
    "set_gripper",
    "yam_real.get_controller_metadata",
}
_READ_ONLY = {"get_robot_state", "localize_object", "yam_real.get_controller_metadata"}


def _jsonable(value: Any, *, _active: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim <= 1 and value.size <= 32:
            return np.round(value.astype(float), 6).tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (Mapping, list, tuple)):
        active = set() if _active is None else _active
        identity = id(value)
        if identity in active:
            return "<recursive-reference>"
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                return {str(key): _jsonable(item, _active=active) for key, item in value.items()}
            return [_jsonable(item, _active=active) for item in value]
        finally:
            active.remove(identity)
    return str(value)


class CapMcpBridge:
    """Keep the remote tool surface small and describe the live capabilities."""

    def __init__(self, session: LiveAgentSession) -> None:
        self.session = session
        configured = session.api.configured_providers
        allowed = set(_DIRECT)
        if not configured["segmentation"]:
            allowed.discard("localize_object")
        if not configured["ik"] and not configured["pose_planning"]:
            allowed.discard("move_to_pose")
        self._default_ik = next(iter(configured["ik"]), "pyroki")
        self._direct = {
            "cap_" + spec.name.replace(".", "__"): spec
            for spec in session.registry.all_specs(public_only=True)
            if spec.name in allowed
        }

    def tool_specs(self) -> list[Any]:
        from mcp.types import Tool

        specs = [
            (
                "get_capability_graph",
                "Read available tools and program provenance.",
                _object_schema(),
            ),
            (
                "observe",
                "Read fresh station state and camera images during any job.",
                _object_schema({"cameras": _CAMERAS}),
            ),
            (
                "get_job",
                "Read job status and terminal state. Cameras are omitted unless requested.",
                _object_schema({"job_id": {"type": "string"}, "cameras": _CAMERAS}),
            ),
            ("read_operator_messages", "Drain pending operator instructions.", _object_schema()),
        ]
        for name, spec in self._direct.items():
            schema = _cap_input_schema(spec)
            if "target_pose" in schema["properties"]:
                schema["properties"]["target_pose"] = _pose_input_schema()
            if "targets" in schema["properties"]:
                schema["properties"]["targets"] = {
                    "type": "object",
                    "additionalProperties": {
                        "oneOf": [
                            _pose_input_schema(),
                            {"type": "array", "items": {"type": "number"}},
                        ]
                    },
                }
            if "strategy" in schema["properties"]:
                strategy_schema = _motion_strategy_input_schema()
                strategy_schema["properties"]["ik_solver"]["default"] = self._default_ik
                schema["properties"]["strategy"] = {"anyOf": [strategy_schema, {"type": "null"}]}
            specs.append((name, spec.documented_signature, schema))
        specs.extend(
            (p.definition.tool_name, p.definition.summary, _object_schema())
            for p in self.session.program_catalog.all()
        )
        return [
            Tool(name=name, description=description, inputSchema=schema)
            for name, description, schema in specs
        ]

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        from jsonschema import validate
        from mcp.types import CallToolResult, ImageContent, TextContent

        try:
            args = dict(arguments or {})
            spec = next((s for s in self.tool_specs() if s.name == name), None)
            if spec is None:
                raise ValueError(f"unknown tool {name!r}")
            validate(args, spec.inputSchema)
            value, frames = self._dispatch(name, args)
            value = _jsonable(value)
            content = [TextContent(type="text", text=json.dumps(value))]
            for camera, frame in frames.items():
                content.append(TextContent(type="text", text=f"camera: {camera}"))
                content.append(
                    ImageContent(type="image", data=_encode_jpeg(frame), mimeType="image/jpeg")
                )
            return CallToolResult(
                content=content, structuredContent=value, isError=bool(value.get("ok") is False)
            )
        except Exception as exc:
            value = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(value))],
                structuredContent=value,
                isError=True,
            )

    def _snapshot(
        self, value: dict[str, Any], cameras: Sequence[str] | None = None
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        snapshot = self.session.visual_snapshot()
        frames = {}
        for name in snapshot["cameras"] if cameras is None else cameras:
            if name not in snapshot["cameras"]:
                raise ValueError(f"camera {name!r} is unavailable")
            frames[name] = snapshot["cameras"][name]
        return {
            **value,
            "observation": {
                key: snapshot[key]
                for key in (
                    "observation_seq",
                    "timestamp_s",
                    "stale",
                    "monitor_error",
                    "camera_metadata",
                    "state",
                )
            },
            "image_cameras": list(frames),
        }, frames

    def _terminal(self, job_id: str) -> dict[str, Any]:
        job = self.session.wait(job_id, timeout_s=None)
        result = job.get("result") if job else None
        return {
            "ok": bool(
                job
                and not job["error"]
                and not (isinstance(result, dict) and result.get("ok") is False)
            ),
            "job": job,
        }

    def _dispatch(
        self, name: str, args: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if name == "get_capability_graph":
            return {
                "ok": True,
                "tools": [
                    {"name": s.name, "description": s.description} for s in self.tool_specs()
                ],
                "programs": [p.metadata() for p in self.session.program_catalog.all()],
                "configured_providers": self.session.api.configured_providers,
            }, {}
        if name == "observe":
            return self._snapshot(
                {"ok": True}, args.get("cameras", [self.session.api.default_camera])
            )
        if name == "get_job":
            job = self.session.job(args.get("job_id"))
            value = {"ok": job is not None, "job": job}
            return self._snapshot(value, args["cameras"]) if args.get("cameras") else (value, {})
        if name == "read_operator_messages":
            return {"ok": True, "messages": self.session.drain_steering()}, {}
        if name.startswith("cap_program_"):
            program = self.session.program_catalog.get_by_tool(name)
            value = self._terminal(self.session.start_named_program(program.definition.name))
            return self._snapshot(value) if program.definition.moves_robot else (value, {})
        spec = self._direct[name]
        if "target_pose" in args:
            args["target_pose"] = pose_from_mapping(args["target_pose"])
        if "target" in args:
            args["target"] = np.asarray(args["target"], dtype=float)
        if "targets" in args:
            args["targets"] = {
                arm: pose_from_mapping(target)
                if isinstance(target, dict)
                else np.asarray(target, dtype=float)
                for arm, target in args["targets"].items()
            }
        if spec.name in {"move_to_pose", "move_synchronized"}:
            args["strategy"] = MotionStrategy(
                **{"ik_solver": self._default_ik, **(args.get("strategy") or {})}
            )
        if spec.name in _READ_ONLY:
            result = spec.function(**args)
            return {"ok": bool(getattr(result, "ok", True)), "result": _compact_summary(result)}, {}
        value = self._terminal(self.session.start_tool(spec.name, args))
        return self._snapshot(value)


def create_mcp_transport(session: LiveAgentSession) -> tuple[Any, Any]:
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    bridge = CapMcpBridge(session)
    server = Server("cap", version="0.1.0")

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return bridge.tool_specs()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        return await asyncio.to_thread(bridge.call, name, arguments)

    manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

    async def endpoint(scope: Any, receive: Any, send: Any) -> None:
        await manager.handle_request(scope, receive, send)

    return manager, endpoint


def create_lifespan(manager: Any) -> Any:
    @asynccontextmanager
    async def lifespan(_app: Any):
        async with manager.run():
            yield

    return lifespan
