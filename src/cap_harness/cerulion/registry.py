"""Typed discovery for the tasks of the Cerulion simulated arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CerulionRegistryError(ValueError):
    """Raised when a Cerulion task reference is invalid."""


SUITE = "cerulion"


@dataclass(frozen=True, slots=True)
class CerulionTaskMetadata:
    """Static metadata identifying one task on the Cerulion simulated arm."""

    task_name: str
    language: str
    task_id: int = 0
    family: str = "cerulion"
    horizon: int = 600

    @property
    def suite_name(self) -> str:
        return SUITE

    @property
    def task_ref(self) -> str:
        return f"{SUITE}:{self.task_name}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "horizon": self.horizon,
            "language": self.language,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


CERULION_TASKS: tuple[CerulionTaskMetadata, ...] = (
    CerulionTaskMetadata(
        "pick_cube",
        "Pick up the cube and lift it at least 15 cm above the table. This arm has no "
        "perception: the cube's position is supplied in the task metadata as "
        "task_supplied_cube_position.",
        task_id=0,
    ),
)


class CerulionTaskRegistry:
    """Resolve ``(suite, task_id)`` where suite is ``cerulion`` and ids index the task list."""

    def __init__(self, tasks: tuple[CerulionTaskMetadata, ...] = CERULION_TASKS) -> None:
        self._by_id = {task.task_id: task for task in tasks}
        self._by_name = {task.task_name: task for task in tasks}
        if len(self._by_id) != len(tasks) or len(self._by_name) != len(tasks):
            raise CerulionRegistryError("Cerulion task ids and names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def enumerate_tasks(self) -> tuple[CerulionTaskMetadata, ...]:
        return tuple(self._by_id.values())

    def resolve(self, task_ref: object, task_id: int | None = None) -> CerulionTaskMetadata:
        if isinstance(task_ref, CerulionTaskMetadata):
            return task_ref
        name = str(task_ref)
        if name.startswith(f"{SUITE}:"):
            name = name.split(":", 1)[1]
        if name == SUITE:
            index = 0 if task_id is None else int(task_id)
            try:
                return self._by_id[index]
            except KeyError as exc:
                raise CerulionRegistryError(
                    f"unknown Cerulion task id {index}; available: {sorted(self._by_id)}"
                ) from exc
        try:
            task = self._by_name[name]
        except KeyError as exc:
            raise CerulionRegistryError(
                f"unknown Cerulion task {name!r}; available: {', '.join(self.available_tasks)}"
            ) from exc
        if task_id is not None and int(task_id) != task.task_id:
            raise CerulionRegistryError(f"task {name!r} has task_id {task.task_id}, not {task_id}")
        return task


__all__ = [
    "CERULION_TASKS",
    "SUITE",
    "CerulionRegistryError",
    "CerulionTaskMetadata",
    "CerulionTaskRegistry",
]
