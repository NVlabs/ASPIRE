"""BEHAVIOR-1K (OmniGibson on Isaac Sim) support for the R1 Pro pickup tasks.

The adapter and planner import OmniGibson, which imports Isaac Sim, so they are exposed lazily:
importing this package for the registry, codec or config never touches the simulator.
"""

from __future__ import annotations

from typing import Any

from cap_harness.behavior.codec import NATIVE_ARMS, BehaviorActionCodec, R1ProJointLayout
from cap_harness.behavior.config import build_environment_config
from cap_harness.behavior.registry import (
    BEHAVIOR_ARMS,
    BEHAVIOR_CAMERA_NAMES,
    BEHAVIOR_TASKS,
    BehaviorRegistryError,
    BehaviorTaskMetadata,
    BehaviorTaskRegistry,
)

_LAZY = {
    "BASE_FRAME": ("cap_harness.behavior.adapter", "BASE_FRAME"),
    "BehaviorAdapter": ("cap_harness.behavior.adapter", "BehaviorAdapter"),
    "OmniGibsonCuroboPlanner": ("cap_harness.behavior.planning", "OmniGibsonCuroboPlanner"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BASE_FRAME",
    "BEHAVIOR_ARMS",
    "BEHAVIOR_CAMERA_NAMES",
    "BEHAVIOR_TASKS",
    "NATIVE_ARMS",
    "BehaviorActionCodec",
    "BehaviorAdapter",
    "BehaviorRegistryError",
    "BehaviorTaskMetadata",
    "BehaviorTaskRegistry",
    "OmniGibsonCuroboPlanner",
    "R1ProJointLayout",
    "build_environment_config",
]
