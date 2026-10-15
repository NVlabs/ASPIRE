"""Allowlisted LIBERO metadata extensions for generated programs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cap_harness.libero.adapter import LiberoAdapter


class LiberoRuntimeExtensions:
    """Public metadata-only extension surface backed by a ``LiberoAdapter``."""

    __slots__ = ("_adapter",)

    def __init__(self, adapter: LiberoAdapter) -> None:
        self._adapter = adapter

    def get_task_metadata(self) -> Mapping[str, object]:
        task = self._adapter.task_metadata
        return MappingProxyType(
            {
                "family": task.family,
                "init_mode": self._adapter.init_mode,
                "init_state_count": task.init_state_count,
                "init_state_index": self._adapter.selected_init_state_index,
                "language": task.language,
                "seed": self._adapter.seed,
                "suite": task.suite_name,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_ref": task.task_ref,
            }
        )

    def get_controller_metadata(self) -> Mapping[str, object]:
        low, high = self._adapter.action_bounds
        return MappingProxyType(
            {
                "action_dimension": int(low.size),
                "action_lower_bounds": tuple(float(value) for value in low),
                "action_upper_bounds": tuple(float(value) for value in high),
                "arm_names": ("primary",),
                "camera_names": ("agentview", "robot0_eye_in_hand"),
                "control_frequency_hz": self._adapter.control_frequency,
                "control_period_s": self._adapter.control_period_s,
                "controller": self._adapter.controller,
                "gripper_semantics": {
                    "normalized": "0=closed, 1=open",
                    "native": "1=closed, -1=open",
                },
                "joint_command_semantics": (
                    "absolute seven-joint target encoded as "
                    "(target-current)*control_frequency and clamped to native bounds"
                ),
                "supported_action_modes": ("joint_position",),
            }
        )


LiberoExtensions = LiberoRuntimeExtensions


def get_task_metadata(adapter: LiberoAdapter) -> Mapping[str, object]:
    return LiberoRuntimeExtensions(adapter).get_task_metadata()


def get_controller_metadata(adapter: LiberoAdapter) -> Mapping[str, object]:
    return LiberoRuntimeExtensions(adapter).get_controller_metadata()


__all__ = [
    "LiberoExtensions",
    "LiberoRuntimeExtensions",
    "get_controller_metadata",
    "get_task_metadata",
]
