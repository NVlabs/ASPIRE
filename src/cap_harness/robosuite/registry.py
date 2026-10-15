"""Typed discovery for the seven supported Robosuite CaP tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RobosuiteRegistryError(ValueError):
    """Raised when a Robosuite task reference is invalid."""


@dataclass(frozen=True, slots=True)
class RobosuiteTaskMetadata:
    task_name: str
    environment: str
    language: str
    robots: tuple[str, ...]
    camera_names: tuple[str, ...]
    env_configuration: str | None = None
    task_id: int = 0
    family: str = "robosuite"
    horizon: int | None = None
    """Episode length the task's own authors specify, in control ticks.

    ``None`` means the upstream benchmark states no horizon for this task and the
    caller's default stands. Where a horizon IS stated it belongs here rather than
    in a caller's flag: a task that is scored against half the ticks it was
    written for fails for a reason that has nothing to do with the policy.
    """

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",) if len(self.robots) == 1 else ("primary", "secondary")

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "camera_names": list(self.camera_names),
            "environment": self.environment,
            "family": self.family,
            "language": self.language,
            "robots": list(self.robots),
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


_SINGLE_CAMERAS = ("agentview", "robot0_eye_in_hand")
_BIMANUAL_CAMERAS = ("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")

ROBOSUITE_TASKS: tuple[RobosuiteTaskMetadata, ...] = (
    RobosuiteTaskMetadata(
        "cube_lifting",
        "Lift",
        "Pick up the cube and lift it clear of the table.",
        ("Panda",),
        _SINGLE_CAMERAS,
    ),
    RobosuiteTaskMetadata(
        "cube_restack",
        "Stack",
        "Move the upper cube off the lower cube and place the lower cube on the upper cube.",
        ("Panda",),
        _SINGLE_CAMERAS,
    ),
    RobosuiteTaskMetadata(
        "cube_stack",
        "Stack",
        "Pick up the red cube and stack it on the green cube.",
        ("Panda",),
        _SINGLE_CAMERAS,
    ),
    RobosuiteTaskMetadata(
        "nut_assembly",
        "NutAssemblySquare",
        "Pick up the square nut and place it onto the square peg.",
        ("Panda",),
        _SINGLE_CAMERAS,
    ),
    RobosuiteTaskMetadata(
        "spill_wipe",
        "Wipe",
        "Use the wiping tool to clean all marked spill regions from the table.",
        ("Panda",),
        _SINGLE_CAMERAS,
    ),
    RobosuiteTaskMetadata(
        "two_arm_lift",
        "TwoArmLift",
        "Use both arms simultaneously to grasp both pot handles and lift the pot.",
        ("Panda", "Panda"),
        _BIMANUAL_CAMERAS,
        "single-arm-opposed",
    ),
    RobosuiteTaskMetadata(
        "two_arm_handover",
        "TwoArmHandover",
        "Primary should lift the hammer and hand its handle to secondary.",
        ("Panda", "Panda"),
        _BIMANUAL_CAMERAS,
        "single-arm-opposed",
    ),
)


class RobosuiteTaskRegistry:
    """Resolve only the audited Robosuite task set."""

    def __init__(self, tasks: tuple[RobosuiteTaskMetadata, ...] = ROBOSUITE_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise RobosuiteRegistryError("Robosuite task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[RobosuiteTaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self, task_ref: str | tuple[str, int] | RobosuiteTaskMetadata, task_id: int | None = None
    ) -> RobosuiteTaskMetadata:
        if isinstance(task_ref, RobosuiteTaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise RobosuiteRegistryError("task tuple must contain (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and task_id != tuple_id:
                raise RobosuiteRegistryError("conflicting Robosuite task ids")
            task_id = int(tuple_id)
        else:
            task_name = task_ref
            if ":" in task_name:
                task_name, encoded_id = task_name.rsplit(":", 1)
                if task_id is not None and task_id != int(encoded_id):
                    raise RobosuiteRegistryError("conflicting Robosuite task ids")
                task_id = int(encoded_id)
        if task_id is None:
            task_id = 0
        if task_id != 0:
            raise RobosuiteRegistryError("each Robosuite task currently has task_id 0")
        try:
            return self._tasks[str(task_name)]
        except KeyError as exc:
            raise RobosuiteRegistryError(
                f"unknown Robosuite task {task_name!r}; available: {', '.join(self.available_tasks)}"
            ) from exc


__all__ = [
    "ROBOSUITE_TASKS",
    "RobosuiteRegistryError",
    "RobosuiteTaskMetadata",
    "RobosuiteTaskRegistry",
]
