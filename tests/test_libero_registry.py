from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from cap_harness.libero.registry import (
    EXPECTED_REQUIRED_TASK_COUNT,
    REQUIRED_SUITE_NAMES,
    LiberoRegistryError,
    LiberoSuiteRegistry,
    extract_language_from_bddl_text,
)


class FakeSuite:
    def __init__(self, tasks: list[SimpleNamespace], *, init_state_count: int = 3) -> None:
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self._init_states = [object()] * init_state_count

    def get_task(self, task_id: int) -> SimpleNamespace:
        return self.tasks[task_id]

    def get_task_init_states(self, task_id: int) -> list[object]:
        del task_id
        return self._init_states


def _fake_benchmarks(tmp_path, *, short_suite: str | None = None):
    bddl_root = tmp_path / "bddl_files"
    benchmark_dict = {}
    for suite_name in REQUIRED_SUITE_NAMES:
        task_count = 9 if suite_name == short_suite else 10
        tasks = []
        suite_dir = bddl_root / suite_name
        suite_dir.mkdir(parents=True)
        for task_id in range(task_count):
            task_name = f"task_{task_id}"
            language = f"authoritative instruction {suite_name} {task_id}"
            (suite_dir / f"{task_name}.bddl").write_text(
                f"""
                (define (problem {task_name})
                    ; (:language this comment must be ignored)
                    (:language
                        {language}
                    )
                )
                """,
                encoding="utf-8",
            )
            tasks.append(
                SimpleNamespace(
                    name=task_name,
                    language="stale filename-derived language",
                    problem_folder=suite_name,
                    bddl_file=f"{task_name}.bddl",
                )
            )
        suite = FakeSuite(tasks)
        benchmark_dict[suite_name] = lambda suite=suite: suite
    return benchmark_dict, lambda key: bddl_root if key == "bddl_files" else tmp_path


def test_extract_language_balances_parentheses_and_quotes() -> None:
    assert (
        extract_language_from_bddl_text(
            '(define (problem x) (:language "put the (small) bowl on the plate"))'
        )
        == "put the (small) bowl on the plate"
    )


def test_required_manifest_uses_authoritative_bddl_language(tmp_path) -> None:
    benchmark_dict, path_resolver = _fake_benchmarks(tmp_path)
    registry = LiberoSuiteRegistry(
        benchmark_dict=benchmark_dict,
        path_resolver=path_resolver,
    )

    manifest = registry.required_manifest()

    assert manifest["task_count"] == EXPECTED_REQUIRED_TASK_COUNT
    assert manifest["suites"] == list(REQUIRED_SUITE_NAMES)
    first = manifest["tasks"][0]
    assert first["language"].startswith("authoritative instruction")
    assert first["language"] != "stale filename-derived language"
    assert first["init_state_count"] == 3
    assert first["family"] == "goal"


def test_required_manifest_fails_unless_all_80_pairs_exist(tmp_path) -> None:
    benchmark_dict, path_resolver = _fake_benchmarks(tmp_path, short_suite="libero_10_task")
    registry = LiberoSuiteRegistry(
        benchmark_dict=benchmark_dict,
        path_resolver=path_resolver,
    )

    with pytest.raises(LiberoRegistryError, match="must each contain 10 tasks"):
        registry.required_manifest()


def test_manifest_write_is_deterministic(tmp_path) -> None:
    benchmark_dict, path_resolver = _fake_benchmarks(tmp_path)
    registry = LiberoSuiteRegistry(
        benchmark_dict=benchmark_dict,
        path_resolver=path_resolver,
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    registry.write_manifest(first_path)
    registry.write_manifest(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()


def test_init_state_loader_temporarily_enables_trusted_legacy_torch_load(
    tmp_path, monkeypatch
) -> None:
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(
        "(define (problem task) (:language move the mug))",
        encoding="utf-8",
    )
    observed: list[str | None] = []

    class TrackingSuite(FakeSuite):
        def get_task_init_states(self, task_id: int) -> list[object]:
            observed.append(os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"))
            return super().get_task_init_states(task_id)

    suite = TrackingSuite([SimpleNamespace(name="task", bddl_path=bddl_path)])
    registry = LiberoSuiteRegistry(benchmark_dict={"libero_goal_swap": suite})
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "previous")

    tasks = registry.enumerate_tasks(("libero_goal_swap",))

    assert len(tasks) == 1
    assert observed == ["1"]
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "previous"
