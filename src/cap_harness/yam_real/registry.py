"""Typed discovery for the supported real-YAM tasks.

A hardware task is a language goal plus a physical scene, not a seeded simulator
state. ``scene_reset`` therefore names how the scene is restored between trials,
so a recorded run can state whether a human reset it or nothing did, instead of
leaving it implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class YamRealRegistryError(ValueError):
    """Raised when a real-YAM task reference is invalid."""


@dataclass(frozen=True, slots=True)
class YamRealTaskMetadata:
    """Static metadata identifying one real-robot task."""

    task_name: str
    language: str
    task_id: int = 0
    family: str = "yam_real"
    scene_reset: str = "manual"

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "language": self.language,
            "scene_reset": self.scene_reset,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


YAM_REAL_TASKS: tuple[YamRealTaskMetadata, ...] = (
    YamRealTaskMetadata("observe_station", "Observe the YAM station.", scene_reset="none"),
    YamRealTaskMetadata(
        "reach_home", "Return both arms to the station home pose.", scene_reset="none"
    ),
    YamRealTaskMetadata("joint_motion", "Execute a bounded joint motion.", scene_reset="none"),
)


class YamRealTaskRegistry:
    """Resolve only the explicitly supported real-YAM task set."""

    def __init__(self, tasks: tuple[YamRealTaskMetadata, ...] = YAM_REAL_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise YamRealRegistryError("real-YAM task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[YamRealTaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self,
        task_ref: str | tuple[str, int] | YamRealTaskMetadata,
        task_id: int | None = None,
    ) -> YamRealTaskMetadata:
        if isinstance(task_ref, YamRealTaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise YamRealRegistryError("task tuple must contain (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and task_id != tuple_id:
                raise YamRealRegistryError("conflicting real-YAM task ids")
            task_id = int(tuple_id)
        else:
            task_name = task_ref
            if ":" in task_name:
                task_name, encoded_id = task_name.rsplit(":", 1)
                if task_id is not None and task_id != int(encoded_id):
                    raise YamRealRegistryError("conflicting real-YAM task ids")
                task_id = int(encoded_id)
        if task_id is None:
            task_id = 0
        if task_id != 0:
            raise YamRealRegistryError("each real-YAM task currently has task_id 0")
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise YamRealRegistryError(
                f"unknown real-YAM task {task_name!r}; available: {', '.join(self.available_tasks)}"
            ) from exc


__all__ = [
    "YAM_REAL_TASKS",
    "YamRealRegistryError",
    "YamRealTaskMetadata",
    "YamRealTaskRegistry",
]
