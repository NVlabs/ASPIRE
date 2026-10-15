"""Allowlisted registry for APIs exposed to generated programs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
import inspect
import re
from types import MappingProxyType

SHARED_PUBLIC_TOOL_NAMES = frozenset(
    {
        "close_gripper",
        "crop_point_cloud",
        "estimate_geometry",
        "execute_trajectory",
        "generate_grasps",
        "get_observation",
        "get_robot_state",
        "get_task_context",
        "go_home",
        "localize_object",
        "mask_to_point_cloud",
        "move_to_joints",
        "move_to_pose",
        "move_synchronized",
        "open_gripper",
        "plan_motion",
        "plan_synchronized_motion",
        "segment_points",
        "segment_text",
        "select_grasp",
        "set_gripper",
        "set_grippers",
        "solve_ik",
        "step",
    }
)
"""Shared APIs that may be made public without an embodiment prefix."""

LIBERO_PUBLIC_TOOL_NAMES = frozenset(
    {
        "libero.get_controller_metadata",
        "libero.get_task_metadata",
    }
)
"""Nonprivileged LIBERO metadata extensions approved for V1."""

DEFAULT_LIBERO_PUBLIC_ALLOWLIST = LIBERO_PUBLIC_TOOL_NAMES

ROBOSUITE_PUBLIC_TOOL_NAMES = frozenset(
    {
        "robosuite.get_controller_metadata",
        "robosuite.get_task_metadata",
    }
)
"""Nonprivileged Robosuite metadata extensions."""

BEHAVIOR_PUBLIC_TOOL_NAMES = frozenset(
    {
        "behavior.get_base_pose",
        "behavior.get_controller_metadata",
        "behavior.get_task_metadata",
        "behavior.move_torso",
        "behavior.navigate_to_pose",
        "behavior.plan_approach_pose",
        "behavior.base_pose_is_free",
        "behavior.plan_standoff_pose",
        "behavior.reset_torso",
    }
)
"""Nonprivileged BEHAVIOR-1K (R1 Pro) metadata, base, torso and standoff-geometry extensions."""

YAM_REAL_PUBLIC_TOOL_NAMES = frozenset(
    {"yam_real.get_task_metadata", "yam_real.get_controller_metadata"}
)

_APPROVED_EMBODIMENT_PREFIXES = (
    "behavior.",
    "yam_real.",
    "libero.",
    "robosuite.",
)

_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_MEMORY_ADDRESS_PATTERN = re.compile(r"0x[0-9A-Fa-f]+")
_PRIVILEGED_EXTENSION_TERMS = (
    "ground_truth",
    "groundtruth",
    "mujoco",
    "object_pose",
    "privileged",
    "qpos",
    "qvel",
    "raw_observation",
    "raw_state",
    "reward",
    "sim_state",
    "success",
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _tool_name(value: object, name: str = "name") -> str:
    result = _text(value, name)
    if _NAME_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{name} must be a dotted Python identifier")
    return result


def _is_privileged_extension_name(name: str) -> bool:
    normalized = name.lower()
    return any(term in normalized for term in _PRIVILEGED_EXTENSION_TERMS)


def _stable_signature(signature: inspect.Signature) -> str:
    """Remove process-specific addresses from otherwise stable signature text."""
    return _MEMORY_ADDRESS_PATTERN.sub("0xADDR", str(signature))


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One registered callable and the metadata used to expose and document it."""

    name: str
    function: Callable[..., object]
    layer: str
    capability: str
    public: bool
    signature: inspect.Signature
    docstring: str

    @property
    def func(self) -> Callable[..., object]:
        """Short alias for integrations that refer to a registered function."""
        return self.function

    @property
    def documented_signature(self) -> str:
        return f"{self.name}{_stable_signature(self.signature)}"


class ToolRegistry:
    """Register runtime tools while strictly constraining the public namespace."""

    def __init__(
        self,
        *,
        shared_public_names: Iterable[str] = SHARED_PUBLIC_TOOL_NAMES,
        libero_public_allowlist: Iterable[str] | None = None,
        public_extension_allowlist: Iterable[str] | None = None,
    ) -> None:
        if libero_public_allowlist is not None and public_extension_allowlist is not None:
            raise ValueError(
                "pass only one of libero_public_allowlist and public_extension_allowlist"
            )
        shared = frozenset(_tool_name(name, "shared public name") for name in shared_public_names)
        if any("." in name for name in shared):
            raise ValueError("shared public names must be unqualified identifiers")

        requested_extensions = (
            public_extension_allowlist
            if public_extension_allowlist is not None
            else libero_public_allowlist
        )
        if requested_extensions is None:
            requested_extensions = LIBERO_PUBLIC_TOOL_NAMES
        extensions = frozenset(
            _tool_name(name, "public extension name") for name in requested_extensions
        )
        for name in extensions:
            if not name.startswith(_APPROVED_EMBODIMENT_PREFIXES):
                raise ValueError("public extension names must begin with an approved embodiment")
            if _is_privileged_extension_name(name):
                raise ValueError(f"privileged extension cannot be public: {name!r}")

        self._shared_public_names = shared
        self._public_extension_allowlist = extensions
        self._specs: dict[str, ToolSpec] = {}

    @property
    def shared_public_names(self) -> frozenset[str]:
        return self._shared_public_names

    @property
    def public_extension_allowlist(self) -> frozenset[str]:
        return self._public_extension_allowlist

    def register(
        self,
        function: Callable[..., object] | None = None,
        *,
        name: str | None = None,
        public_name: str | None = None,
        layer: str,
        capability: str,
        public: bool = False,
    ) -> ToolSpec | Callable[[Callable[..., object]], Callable[..., object]]:
        """Register a callable directly or return a registration decorator.

        ``public_name`` is accepted as an explicit alias for ``name``.  Direct
        registration returns the resulting :class:`ToolSpec`; decorator use
        returns the original function after registering it.
        """
        if name is not None and public_name is not None:
            raise ValueError("pass only one of name and public_name")
        resolved_name = public_name if public_name is not None else name

        if function is None:
            if resolved_name is None:
                raise ValueError("decorator registration requires an explicit name")

            def decorator(target: Callable[..., object]) -> Callable[..., object]:
                self._register(
                    target,
                    name=resolved_name,
                    layer=layer,
                    capability=capability,
                    public=public,
                )
                return target

            return decorator

        if not callable(function):
            raise ValueError("function must be callable")
        if resolved_name is None:
            resolved_name = getattr(function, "__name__", None)
        return self._register(
            function,
            name=resolved_name,
            layer=layer,
            capability=capability,
            public=public,
        )

    def add(
        self,
        name: str,
        function: Callable[..., object],
        *,
        layer: str,
        capability: str,
        public: bool = False,
    ) -> ToolSpec:
        """Name-first registration form for callers assembling tools dynamically."""
        return self._register(
            function,
            name=name,
            layer=layer,
            capability=capability,
            public=public,
        )

    def _register(
        self,
        function: Callable[..., object],
        *,
        name: object,
        layer: object,
        capability: object,
        public: object,
    ) -> ToolSpec:
        if not callable(function):
            raise ValueError("function must be callable")
        normalized_name = _tool_name(name)
        normalized_layer = _text(layer, "layer")
        normalized_capability = _text(capability, "capability")
        if type(public) is not bool:
            raise ValueError("public must be a bool")
        if normalized_name in self._specs:
            raise ValueError(f"tool already registered: {normalized_name!r}")
        if public:
            self._validate_public_name(normalized_name)
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot inspect signature for tool {normalized_name!r}") from exc
        docstring = inspect.cleandoc(inspect.getdoc(function) or "")
        spec = ToolSpec(
            name=normalized_name,
            function=function,
            layer=normalized_layer,
            capability=normalized_capability,
            public=public,
            signature=signature,
            docstring=docstring,
        )
        self._specs[normalized_name] = spec
        return spec

    def _validate_public_name(self, name: str) -> None:
        if "." not in name:
            if name not in self._shared_public_names:
                raise ValueError(f"unallowlisted shared public tool: {name!r}")
            return
        if not name.startswith(_APPROVED_EMBODIMENT_PREFIXES):
            raise ValueError("public extensions must begin with an approved embodiment")
        if name not in self._public_extension_allowlist:
            raise ValueError(f"unallowlisted public extension: {name!r}")
        if _is_privileged_extension_name(name):
            raise ValueError(f"privileged extension cannot be public: {name!r}")

    def get_spec(self, name: str) -> ToolSpec:
        """Return registration metadata or raise ``KeyError``."""
        return self._specs[name]

    def get(self, name: str) -> Callable[..., object]:
        """Resolve a registered callable or raise ``KeyError``."""
        return self.get_spec(name).function

    def all_specs(self, *, public_only: bool = False) -> tuple[ToolSpec, ...]:
        """Return specs in deterministic public-name order."""
        specs = (spec for spec in self._specs.values() if not public_only or spec.public)
        return tuple(sorted(specs, key=lambda spec: spec.name))

    def public_tools(self) -> Mapping[str, Callable[..., object]]:
        """Return a read-only, sorted generated-program namespace."""
        return MappingProxyType(
            {spec.name: spec.function for spec in self.all_specs(public_only=True)}
        )

    def public_namespace(self) -> Mapping[str, Callable[..., object]]:
        """Alias emphasizing that the mapping is suitable for code exposure."""
        return self.public_tools()

    def render_docs(self, *, public_only: bool = True) -> str:
        """Generate stable Markdown from registered signatures and docstrings."""
        specs = self.all_specs(public_only=public_only)
        lines = ["# Tool API"]
        for spec in specs:
            lines.extend(
                (
                    "",
                    f"## `{spec.documented_signature}`",
                    f"Layer: `{spec.layer}`",
                    f"Capability: `{spec.capability}`",
                    "",
                    spec.docstring or "No documentation provided.",
                )
            )
        return "\n".join(lines).rstrip() + "\n"

    def documentation(self, *, public_only: bool = True) -> str:
        """Alias for :meth:`render_docs`."""
        return self.render_docs(public_only=public_only)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._specs))

    def __getitem__(self, name: str) -> Callable[..., object]:
        return self.get(name)


__all__ = [
    "BEHAVIOR_PUBLIC_TOOL_NAMES",
    "DEFAULT_LIBERO_PUBLIC_ALLOWLIST",
    "LIBERO_PUBLIC_TOOL_NAMES",
    "ROBOSUITE_PUBLIC_TOOL_NAMES",
    "SHARED_PUBLIC_TOOL_NAMES",
    "YAM_REAL_PUBLIC_TOOL_NAMES",
    "ToolRegistry",
    "ToolSpec",
]
