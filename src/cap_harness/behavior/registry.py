"""Audited BEHAVIOR-1K (R1 Pro) task registry: two pickup sub-goals, nothing else.

The suites are the BEHAVIOR activities the pickup targets live in. Success is the host-side
pickup witness (``validation/evaluators/behavior_pickup.py``), not the BDDL goal, exactly as the
ASPIRE reference defines the two tasks. Seeds are challenge task-instance ids, one to one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re

DEFAULT_SCENE_MODEL = "house_double_floor_lower"
TASK_INSTANCE_DATASET = "2026-challenge-task-instances"
BEHAVIOR_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
BEHAVIOR_ARMS = ("primary", "secondary")


class BehaviorRegistryError(KeyError):
    """Raised for unknown BEHAVIOR task references."""


@dataclass(frozen=True, slots=True)
class BehaviorTaskMetadata:
    """One supported BEHAVIOR pickup task.

    ``target_scope`` is the BDDL object-scope name of the pickup target and is consumed only by
    the host-side witness; it never reaches generated programs (see ``to_manifest_record``).
    """

    task_name: str
    activity_name: str
    language: str
    target_scope: str
    target_prompt: str
    support_prompt: str
    scene_model: str = DEFAULT_SCENE_MODEL
    load_room_types: tuple[str, ...] = ("living_room",)
    robots: tuple[str, ...] = ("r1pro",)
    camera_names: tuple[str, ...] = BEHAVIOR_CAMERA_NAMES
    instance_mode: str = "train"
    task_id: int = 0
    family: str = "behavior"

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    @property
    def arms(self) -> tuple[str, ...]:
        return BEHAVIOR_ARMS

    def to_manifest_record(self) -> dict[str, object]:
        """Program-visible task record; excludes the privileged target scope name."""
        return {
            "activity_name": self.activity_name,
            "arms": self.arms,
            "camera_names": self.camera_names,
            "family": self.family,
            "instance_mode": self.instance_mode,
            "language": self.language,
            "robots": self.robots,
            "scene_model": self.scene_model,
            "suite_name": self.suite_name,
            "support_prompt": self.support_prompt,
            "target_prompt": self.target_prompt,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


BEHAVIOR_TASKS: tuple[BehaviorTaskMetadata, ...] = (
    BehaviorTaskMetadata(
        task_name="turning_on_radio",
        activity_name="turning_on_radio",
        language="pick up the red radio",
        target_scope="radio_receiver.n.01_1",
        target_prompt="red radio",
        support_prompt="table",
        load_room_types=("living_room",),
    ),
    BehaviorTaskMetadata(
        task_name="picking_up_trash",
        activity_name="picking_up_trash",
        language="pick up the blue can of soda",
        target_scope="can__of__soda.n.01_3",
        target_prompt="blue can of soda",
        support_prompt="floor",
        load_room_types=("living_room", "kitchen"),
    ),
)


class BehaviorTaskRegistry:
    """Resolve task references and enumerate the task instances present on disk."""

    def __init__(self, tasks: Iterable[BehaviorTaskMetadata] = BEHAVIOR_TASKS) -> None:
        self._tasks = {task.task_name: task for task in tasks}
        if not self._tasks:
            raise ValueError("registry needs at least one task")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def enumerate_tasks(self) -> tuple[BehaviorTaskMetadata, ...]:
        return tuple(self._tasks.values())

    def resolve(
        self,
        task_ref: str | tuple[str, int] | BehaviorTaskMetadata,
        task_id: int | None = None,
    ) -> BehaviorTaskMetadata:
        if isinstance(task_ref, BehaviorTaskMetadata):
            if task_ref.task_name not in self._tasks:
                raise BehaviorRegistryError(task_ref.task_name)
            return task_ref
        if isinstance(task_ref, tuple):
            name, task_id = task_ref
        elif isinstance(task_ref, str) and ":" in task_ref:
            name, _, suffix = task_ref.partition(":")
            task_id = int(suffix)
        else:
            name = str(task_ref)
        if name not in self._tasks:
            raise BehaviorRegistryError(name)
        if task_id not in (None, 0):
            raise BehaviorRegistryError(
                f"each BEHAVIOR task currently has task_id 0, got {task_id}"
            )
        return self._tasks[name]

    @staticmethod
    def instance_directory(metadata: BehaviorTaskMetadata, data_root: Path | str) -> Path:
        mode_dir = {
            "train": "scenes",
            "public_test": "scene_test/public",
            "hidden_test": "scene_test/private",
        }[metadata.instance_mode]
        return (
            Path(data_root)
            / TASK_INSTANCE_DATASET
            / mode_dir
            / metadata.scene_model
            / "json"
            / f"{metadata.scene_model}_task_{metadata.activity_name}_instances"
        )

    def instance_ids(
        self, metadata: BehaviorTaskMetadata, data_root: Path | str
    ) -> tuple[int, ...]:
        """Task-instance ids present on disk for this task; seeds map to these one to one."""
        directory = self.instance_directory(metadata, data_root)
        pattern = re.compile(
            rf"^{re.escape(metadata.scene_model)}_task_{re.escape(metadata.activity_name)}"
            r"_0_(\d+)_template-tro_state\.json$"
        )
        ids = sorted(
            int(match.group(1))
            for path in directory.glob("*.json")
            if (match := pattern.match(path.name)) is not None
        )
        return tuple(ids)


__all__ = [
    "BEHAVIOR_ARMS",
    "BEHAVIOR_CAMERA_NAMES",
    "BEHAVIOR_TASKS",
    "DEFAULT_SCENE_MODEL",
    "TASK_INSTANCE_DATASET",
    "BehaviorRegistryError",
    "BehaviorTaskMetadata",
    "BehaviorTaskRegistry",
]
