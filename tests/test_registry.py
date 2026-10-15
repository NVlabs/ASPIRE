from __future__ import annotations

import re

import pytest

from cap_harness.registry import ToolRegistry


def segment_text(query: str, *, threshold: float = 0.5) -> list[str]:
    """Return masks matching a language query."""
    return [query] if threshold <= 1.0 else []


def test_registry_exposes_only_registered_public_allowlisted_tools() -> None:
    registry = ToolRegistry()
    public_spec = registry.register(
        segment_text,
        name="segment_text",
        layer="atomic",
        capability="perception",
        public=True,
    )
    registry.add(
        "libero.raw_state",
        lambda: {"secret": True},
        layer="runtime",
        capability="privileged_state",
        public=False,
    )

    namespace = registry.public_tools()

    assert public_spec.name == "segment_text"
    assert public_spec.layer == "atomic"
    assert tuple(namespace) == ("segment_text",)
    assert namespace["segment_text"]("cup") == ["cup"]
    assert "libero.raw_state" in registry
    with pytest.raises(TypeError):
        namespace["other"] = lambda: None  # type: ignore[index]


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.add("step", lambda action: action, layer="atomic", capability="control")

    with pytest.raises(ValueError, match="already registered"):
        registry.add("step", lambda action: action, layer="atomic", capability="control")


@pytest.mark.parametrize(
    "name",
    [
        "unknown_shared_tool",
        "libero.unapproved_metadata",
        "libero.raw_state",
        "libero.success_predicate",
    ],
)
def test_registry_rejects_public_unallowlisted_or_privileged_names(name: str) -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        registry.add(name, lambda: None, layer="atomic", capability="test", public=True)


def test_registry_allows_only_approved_libero_metadata_extensions() -> None:
    registry = ToolRegistry()
    registry.add(
        "libero.get_task_metadata",
        lambda: {"suite": "libero_goal_task"},
        layer="embodiment",
        capability="task_metadata",
        public=True,
    )
    registry.add(
        "libero.get_controller_metadata",
        lambda: {"frequency_hz": 20},
        layer="embodiment",
        capability="controller_metadata",
        public=True,
    )

    assert tuple(registry.public_tools()) == (
        "libero.get_controller_metadata",
        "libero.get_task_metadata",
    )


def test_registry_rejects_invalid_custom_public_extension_allowlists() -> None:
    with pytest.raises(ValueError, match="approved embodiment"):
        ToolRegistry(public_extension_allowlist={"unknown_embodiment.get_metadata"})
    with pytest.raises(ValueError, match="privileged extension"):
        ToolRegistry(public_extension_allowlist={"libero.get_reward"})


def test_registry_allows_robosuite_metadata_extensions() -> None:
    registry = ToolRegistry(
        public_extension_allowlist={
            "robosuite.get_controller_metadata",
            "robosuite.get_task_metadata",
        }
    )
    registry.add(
        "robosuite.get_task_metadata",
        lambda: {"task": "cube_lifting"},
        layer="embodiment",
        capability="task_metadata",
        public=True,
    )

    assert tuple(registry.public_tools()) == ("robosuite.get_task_metadata",)


def test_registration_decorator_preserves_the_original_callable() -> None:
    registry = ToolRegistry()

    @registry.register(
        name="get_observation",
        layer="atomic",
        capability="observation",
        public=True,
    )
    def get_observation() -> str:
        """Get the current public observation."""
        return "observation"

    assert get_observation() == "observation"
    assert registry["get_observation"] is get_observation


def test_registry_documentation_is_sorted_and_deterministic() -> None:
    sentinel = object()

    def go_home(*, marker=sentinel) -> object:
        """Move the selected arm to its configured home state."""
        return marker

    registry = ToolRegistry()
    registry.add(
        "step",
        lambda action: action,
        layer="atomic",
        capability="control",
        public=False,
    )
    registry.add(
        "segment_text",
        segment_text,
        layer="atomic",
        capability="perception",
        public=True,
    )
    registry.add(
        "go_home",
        go_home,
        layer="composed",
        capability="motion",
        public=True,
    )

    first = registry.render_docs()
    second = registry.documentation()

    assert first == second
    assert first.index("go_home") < first.index("segment_text")
    assert "step" not in first
    assert "Return masks matching a language query." in first
    assert "0xADDR" in first
    assert re.search(r"0x[0-9a-fA-F]{6,}", first) is None
    all_docs = registry.render_docs(public_only=False)
    assert all_docs.index("segment_text") < all_docs.index("step")


def test_registry_validates_registration_metadata() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="callable"):
        registry.register(3, name="step", layer="atomic", capability="control")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dotted Python identifier"):
        registry.add("bad name", lambda: None, layer="atomic", capability="control")
    with pytest.raises(ValueError, match="public must be a bool"):
        registry.add(
            "step",
            lambda: None,
            layer="atomic",
            capability="control",
            public=1,  # type: ignore[arg-type]
        )
