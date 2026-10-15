from __future__ import annotations

from pathlib import Path

import pytest

from cap_harness.behavior.registry import (
    BEHAVIOR_TASKS,
    BehaviorRegistryError,
    BehaviorTaskMetadata,
    BehaviorTaskRegistry,
)


def test_registry_ships_exactly_the_two_aspire_pickup_tasks() -> None:
    registry = BehaviorTaskRegistry()
    assert registry.available_tasks == ("turning_on_radio", "picking_up_trash")
    radio = registry.resolve("turning_on_radio")
    trash = registry.resolve(("picking_up_trash", 0))
    assert radio.language == "pick up the red radio"
    assert trash.language == "pick up the blue can of soda"
    assert radio.target_scope == "radio_receiver.n.01_1"
    assert trash.target_scope == "can__of__soda.n.01_3"
    assert radio.load_room_types == ("living_room",)
    assert trash.load_room_types == ("living_room", "kitchen")
    assert all(task.scene_model == "house_double_floor_lower" for task in BEHAVIOR_TASKS)
    assert all(task.arms == ("primary", "secondary") for task in BEHAVIOR_TASKS)


def test_registry_resolves_every_reference_form_and_rejects_unknowns() -> None:
    registry = BehaviorTaskRegistry()
    radio = registry.resolve("turning_on_radio:0")
    assert registry.resolve(radio) is radio
    assert registry.resolve("turning_on_radio", 0) is radio
    with pytest.raises(BehaviorRegistryError):
        registry.resolve("folding_laundry")
    with pytest.raises(BehaviorRegistryError):
        registry.resolve("turning_on_radio:3")


def test_manifest_record_never_names_the_privileged_target_scope() -> None:
    record = BehaviorTaskRegistry().resolve("picking_up_trash").to_manifest_record()
    assert "target_scope" not in record
    assert record["target_prompt"] == "blue can of soda"
    assert record["support_prompt"] == "floor"
    assert record["task_ref"] == "picking_up_trash:0"
    assert record["camera_names"] == ("head", "left_wrist", "right_wrist")


def test_instance_ids_are_parsed_from_the_challenge_dataset_layout(tmp_path: Path) -> None:
    registry = BehaviorTaskRegistry()
    radio = registry.resolve("turning_on_radio")
    directory = registry.instance_directory(radio, tmp_path)
    assert directory == (
        tmp_path
        / "2026-challenge-task-instances/scenes/house_double_floor_lower/json"
        / "house_double_floor_lower_task_turning_on_radio_instances"
    )
    directory.mkdir(parents=True)
    for instance in (3, 1, 12):
        (
            directory
            / f"house_double_floor_lower_task_turning_on_radio_0_{instance}_template-tro_state.json"
        ).write_text("{}")
    (
        directory / "house_double_floor_lower_task_turning_on_radio_1_7_template-tro_state.json"
    ).write_text("{}")
    (
        directory / "house_double_floor_lower_task_picking_up_trash_0_2_template-tro_state.json"
    ).write_text("{}")
    assert registry.instance_ids(radio, tmp_path) == (1, 3, 12)
    assert registry.instance_ids(registry.resolve("picking_up_trash"), tmp_path) == ()


def test_custom_metadata_keeps_defaults_consistent() -> None:
    task = BehaviorTaskMetadata(
        task_name="custom",
        activity_name="custom",
        language="pick up the thing",
        target_scope="thing.n.01_1",
        target_prompt="thing",
        support_prompt="table",
    )
    assert task.suite_name == "custom"
    assert task.task_ref == "custom:0"
    assert task.instance_mode == "train"
    with pytest.raises(ValueError):
        BehaviorTaskRegistry(tasks=())
