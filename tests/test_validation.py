from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pytest

from cap_harness.validation import (
    EXPECTED_PAIR_COUNT,
    ManifestEntry,
    ManifestError,
    ValidationManifest,
    ValidationRunner,
    build_required_manifest,
    load_validation_cases,
    select_representatives,
)

SUITES = (
    ("libero_goal_swap", "goal", "swap"),
    ("libero_goal_task", "goal", "task"),
    ("libero_object_swap", "object", "swap"),
    ("libero_object_task", "object", "task"),
    ("libero_spatial_swap", "spatial", "swap"),
    ("libero_spatial_task", "spatial", "task"),
    ("libero_10_swap", "long", "swap"),
    ("libero_10_task", "long", "task"),
)


@dataclass(frozen=True)
class FakeTask:
    suite_name: str
    task_id: int
    task_name: str
    language: str
    bddl_path: Path
    init_state_count: int = 3


class FakeRegistry:
    def __init__(self, *, tasks_per_suite: int = 10) -> None:
        self.tasks_per_suite = tasks_per_suite

    def enumerate_tasks(self, suite_names):
        tasks = []
        for suite_name in suite_names:
            family = next(family for name, family, _ in SUITES if name == suite_name)
            for task_id in range(self.tasks_per_suite):
                if family == "goal":
                    language = f"open drawer {task_id} and put away the bowl"
                    task_name = f"goal_task_{task_id}"
                elif family == "object":
                    language = (
                        f"pick the alphabet soup and place it in the basket {task_id}"
                        if task_id == 0
                        else f"complete object task {task_id}"
                    )
                    task_name = f"object_task_{task_id}"
                elif family == "spatial":
                    language = f"put bowl {task_id} to the left of the plate"
                    task_name = f"spatial_task_{task_id}"
                else:
                    language = f"complete {family} task {task_id}"
                    task_name = (
                        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
                        "of_the_cabinet_and_close_it"
                        if task_id == 4
                        else f"long_task_{task_id}"
                    )
                tasks.append(
                    FakeTask(
                        suite_name=suite_name,
                        task_id=task_id,
                        task_name=task_name,
                        language=language,
                        bddl_path=Path(f"/fake/{suite_name}/{task_id}.bddl"),
                    )
                )
        return tuple(tasks)


def _manifest() -> ValidationManifest:
    return build_required_manifest(FakeRegistry())


def _observation(joints: np.ndarray | None = None):
    camera = {
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 3), dtype=np.float64),
        "intrinsics": np.eye(3),
        "extrinsics": np.eye(4),
    }
    return {
        "cameras": {
            "agentview": camera,
            "robot0_eye_in_hand": camera,
        },
        "robot_state": {
            "joint_positions": {
                "primary": np.zeros(7) if joints is None else np.asarray(joints).copy()
            },
            "gripper_positions": {"primary": 1.0},
        },
    }


class FakeAdapter:
    def __init__(self, entry: ManifestEntry, seed: int, *, wrong_language: bool = False) -> None:
        self.entry = entry
        self.seed = seed
        self.current_time_s = 0.0
        self.wrong_language = wrong_language
        self.closed = False
        self.runtime_steps = 0
        self.public_steps = 0
        self.joints = np.zeros(7)

    def reset(self, *, seed: int):
        assert seed == self.seed
        language = "wrong instruction" if self.wrong_language else self.entry.task_language
        self.joints = np.zeros(7)
        return _observation(self.joints), {"task_language": language, "sim_time_s": 0.0}

    def hold_action(self, observation):
        assert observation is not None
        return {"kind": "hold"}

    def runtime_step(self, action):
        if action != {"kind": "hold"}:
            self.joints = np.asarray(action.arms["primary"].target, dtype=np.float64).copy()
        self.runtime_steps += 1
        self.current_time_s += 0.05
        return _observation(self.joints), 0.25, False, False, {"sim_time_s": self.current_time_s}

    def step(self, action):  # pragma: no cover - a call is an immediate test failure
        self.public_steps += 1
        raise AssertionError("validation must use the runtime-only step")

    def close(self):
        self.closed = True


def test_manifest_is_exactly_80_and_representatives_are_deterministic():
    manifest = _manifest()

    assert len(manifest) == EXPECTED_PAIR_COUNT
    assert len({entry.key for entry in manifest}) == EXPECTED_PAIR_COUNT
    assert manifest.representatives["object"].suite == "libero_object_swap"
    assert manifest.representatives["object"].task_id == 0
    assert manifest.representatives["spatial"].suite == "libero_spatial_swap"
    assert manifest.representatives["goal"].suite == "libero_goal_swap"
    assert manifest.representatives["long"].suite == "libero_10_swap"
    assert manifest.representatives["long"].task_id == 4


def test_manifest_rejects_any_count_other_than_80():
    with pytest.raises(ManifestError, match="must have 10 tasks"):
        build_required_manifest(FakeRegistry(tasks_per_suite=9))


def test_representative_selection_uses_task_id_not_registry_order():
    entries = list(_manifest().entries)
    config = {
        "representatives": {
            "chosen": {
                "family": "object",
                "suite_priority": ["libero_object_task", "libero_object_swap"],
                "task_id_priority": [7, 2],
                "language_match_any": ["object|wine bottle"],
            }
        }
    }

    selected = select_representatives(reversed(entries), config)

    assert selected["chosen"].suite == "libero_object_task"
    assert selected["chosen"].task_id == 7


def test_long_representative_has_no_fallback_when_exact_task_is_missing():
    entries = [
        entry
        for entry in _manifest().entries
        if not (
            entry.family == "long"
            and entry.task_name
            == "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
        )
    ]

    with pytest.raises(ManifestError, match="representative 'long' matched no"):
        select_representatives(entries, load_validation_cases())


def test_matrix_resumes_by_suite_task_seed_and_writes_summary(tmp_path):
    manifest = _manifest()
    adapters: list[FakeAdapter] = []

    def factory(entry, seed):
        adapter = FakeAdapter(entry, seed)
        adapters.append(adapter)
        return adapter

    runner = ValidationRunner(
        tmp_path,
        adapter_factory=factory,
        environment={"kind": "unit-test"},
    )
    summary = runner.run(manifest=manifest)

    assert summary["success"] is True
    assert summary["passed"] == 80
    assert summary["expected"] == 80
    assert len(adapters) == 80
    assert all(adapter.runtime_steps == 2 for adapter in adapters)
    assert all(adapter.public_steps == 0 and adapter.closed for adapter in adapters)
    assert len((tmp_path / "matrix.jsonl").read_text().splitlines()) == 80
    assert json.loads((tmp_path / "environment.json").read_text()) == {"kind": "unit-test"}
    assert json.loads((tmp_path / "summary.json").read_text())["success"] is True
    fingerprints = json.loads((tmp_path / "fingerprints.json").read_text())
    assert fingerprints["combined_sha256"] == summary["validation_fingerprint"]

    resumed = runner.run(manifest=manifest)

    assert resumed["skipped_resume"] == 80
    assert len(adapters) == 80
    assert len((tmp_path / "matrix.jsonl").read_text().splitlines()) == 80


def test_matrix_does_not_resume_records_from_a_different_environment_fingerprint(tmp_path):
    manifest = _manifest()
    adapters: list[FakeAdapter] = []

    def factory(entry, seed):
        adapter = FakeAdapter(entry, seed)
        adapters.append(adapter)
        return adapter

    runner = ValidationRunner(
        tmp_path,
        adapter_factory=factory,
        environment={"kind": "unit-test"},
    )
    first = runner.run(manifest=manifest)
    runner.environment = {"kind": "unit-test", "python": "changed"}

    second = runner.run(manifest=manifest)

    assert first["validation_fingerprint"] != second["validation_fingerprint"]
    assert second["skipped_resume"] == 0
    assert len(adapters) == 160
    assert len((tmp_path / "matrix.jsonl").read_text().splitlines()) == 160


def test_matrix_failure_is_summarized_and_gets_artifact(tmp_path):
    manifest = _manifest()
    first_key = manifest.entries[0].key

    def factory(entry, seed):
        return FakeAdapter(entry, seed, wrong_language=entry.key == first_key)

    summary = ValidationRunner(
        tmp_path,
        adapter_factory=factory,
        environment={"kind": "unit-test"},
    ).run(manifest=manifest)

    assert summary["success"] is False
    assert summary["passed"] == 79
    assert summary["failed"] == 1
    records = [json.loads(line) for line in (tmp_path / "matrix.jsonl").read_text().splitlines()]
    failure = next(record for record in records if record["status"] == "failed")
    artifact = tmp_path / failure["failure_artifact"]
    assert artifact.is_file()
    assert "authoritative manifest language" in json.loads(artifact.read_text())["error"]["message"]
