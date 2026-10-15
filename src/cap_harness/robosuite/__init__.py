"""First-class Robosuite embodiment surfaces."""

from cap_harness.robosuite.adapter import RobosuiteAdapter
from cap_harness.robosuite.codec import RobosuiteActionCodec
from cap_harness.robosuite.registry import (
    ROBOSUITE_TASKS,
    RobosuiteRegistryError,
    RobosuiteTaskMetadata,
    RobosuiteTaskRegistry,
)

__all__ = [
    "ROBOSUITE_TASKS",
    "RobosuiteActionCodec",
    "RobosuiteAdapter",
    "RobosuiteRegistryError",
    "RobosuiteTaskMetadata",
    "RobosuiteTaskRegistry",
]
