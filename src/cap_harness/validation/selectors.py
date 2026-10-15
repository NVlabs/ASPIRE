"""Per-embodiment structural selectors behind one interface.

A selector knows how to enumerate an embodiment's tasks and run the generic
structural check (reset -> authoritative language -> observation schema -> one
hold tick -> one bounded probe tick -> finite reward, one native control period,
state change, clean close). RoboSuite uses the generic implementation,
parameterized by control period and language-check mode (``strict`` asserts the
runtime language equals the registry language; ``present`` only requires a
non-empty string, for an embodiment that draws its language from episode
metadata). LIBERO keeps its rich
80-pair/representative matrix and is wrapped by its own selector (added with the
LIBERO migration).

All embodiment imports are lazy (inside methods), so importing this module pulls
in no simulator or MuJoCo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import time
import traceback
from typing import Any, Protocol

import numpy as np

from .model import CaseResult, SeedResult, ValidationCase


class StructuralSelector(Protocol):
    """Runs one structural case and returns its aggregated result."""

    def run(
        self, case: ValidationCase, output_dir: Path, *, retry_failures: bool
    ) -> CaseResult: ...


def _generic_action(
    observation: Any, *, movement: bool, controllable_gripper_arms: tuple[str, ...]
):
    from cap_harness.contracts import ArmCommand, RobotAction

    state = observation.robot_state
    commands: dict[str, Any] = {}
    for arm, joints in state.joint_positions.items():
        target = np.array(joints, copy=True)
        if movement and arm == "primary":
            target[0] += 0.01
        commands[arm] = ArmCommand(
            "joint_position",
            target,
            state.gripper_positions[arm] if arm in controllable_gripper_arms else None,
            embodiment=state.embodiment,
        )
    return RobotAction(commands)


@dataclass(frozen=True, slots=True)
class GenericStructuralSelector:
    """Generic structural check, used by RoboSuite.

    ``language_mode`` is ``strict`` (runtime language must equal the registry
    language) or ``present`` (runtime language must merely be a non-empty string,
    for an embodiment whose episode language differs from the registry placeholder).
    """

    benchmark: str
    control_period_s: float
    enumerate_tasks: Callable[[], list[Any]]
    make_adapter: Callable[[int], Any]
    language_mode: str = "strict"
    camera_size: int = 64

    def _tasks_for_case(self, case: ValidationCase) -> list[Any]:
        tasks = self.enumerate_tasks()
        if case.select == "all" or case.suite is None:
            return tasks
        return [t for t in tasks if getattr(t, "task_name", None) == case.suite]

    def _check(self, task: Any, seed: int) -> dict[str, Any]:
        started = time.monotonic()
        record: dict[str, Any] = {
            "key": {"suite": task.task_name, "task_id": 0, "seed": seed},
            "status": "failed",
            "checks": {},
            "started_at": _utc_now(),
        }
        adapter = self.make_adapter(self.camera_size)
        try:
            from cap_harness.validation import validate_observation_schema

            observation = adapter.reset(task, seed)
            record["checks"]["reset"] = True
            runtime_language = adapter.get_task_context().language
            if self.language_mode == "strict":
                if runtime_language != task.language:
                    raise ValueError("runtime language differs from authoritative manifest")
            elif not (isinstance(runtime_language, str) and runtime_language.strip()):
                raise ValueError("runtime language is empty")
            record["checks"]["authoritative_language"] = True
            record["observation_schema"] = validate_observation_schema(observation)
            record["checks"]["observation_schema"] = True
            gripper_arms = tuple(adapter.get_controller_metadata()["controllable_gripper_arms"])

            before = adapter.current_time_s
            hold = adapter.native_step(
                _generic_action(observation, movement=False, controllable_gripper_arms=gripper_arms)
            )
            if not hold.ok or hold.observation is None or hold.reward is None:
                raise ValueError(f"hold step failed: {hold.error}")
            if not math.isfinite(hold.reward):
                raise ValueError("hold reward is not finite")
            if not math.isclose(
                adapter.current_time_s - before, self.control_period_s, abs_tol=1e-9
            ):
                raise ValueError("hold did not advance exactly one native control period")
            validate_observation_schema(hold.observation)
            record["checks"].update(
                hold_step=True, finite_hold_reward=True, control_period=True, post_hold_schema=True
            )

            before_joints = np.array(
                hold.observation.robot_state.joint_positions["primary"], copy=True
            )
            movement = adapter.native_step(
                _generic_action(
                    hold.observation, movement=True, controllable_gripper_arms=gripper_arms
                )
            )
            if not movement.ok or movement.observation is None or movement.reward is None:
                raise ValueError(f"movement step failed: {movement.error}")
            if not math.isfinite(movement.reward):
                raise ValueError("movement reward is not finite")
            delta = float(
                np.max(
                    np.abs(
                        movement.observation.robot_state.joint_positions["primary"] - before_joints
                    )
                )
            )
            if delta <= 1e-6:
                raise ValueError("movement tick did not change primary joint state")
            record["checks"].update(
                movement_step=True, finite_movement_reward=True, movement_changed_joint_state=True
            )
            record["movement_max_joint_delta_rad"] = delta
            record["status"] = "passed"
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["traceback"] = "".join(traceback.format_exception(exc))
        finally:
            try:
                adapter.close()
                record["checks"]["clean_close"] = True
            except Exception as exc:
                record["checks"]["clean_close"] = False
                record["status"] = "failed"
                record["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["duration_s"] = round(time.monotonic() - started, 6)
            record["finished_at"] = _utc_now()
        return record

    def run(self, case: ValidationCase, output_dir: Path, *, retry_failures: bool) -> CaseResult:
        from cap_harness.validation import AtomicJsonlStore, atomic_write_json

        case_dir = output_dir / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        store = AtomicJsonlStore(case_dir / "matrix.jsonl")
        latest = store.latest_by_key()
        by_key = {(k.suite, k.task_id, k.seed): rec for k, rec in latest.items()}
        tasks = self._tasks_for_case(case)
        expected = len(tasks) * len(case.seeds)
        seed_results: list[SeedResult] = []
        for task in tasks:
            for seed in case.seeds:
                key = (task.task_name, 0, seed)
                existing = by_key.get(key)
                if existing is not None and (
                    existing.get("status") == "passed" or not retry_failures
                ):
                    record = existing
                else:
                    record = self._check(task, seed)
                    store.append(record)
                seed_results.append(
                    SeedResult(
                        seed=seed,
                        passed=record.get("status") == "passed",
                        detail={"suite": task.task_name},
                    )
                )
        passed = sum(1 for r in seed_results if r.passed)
        result = CaseResult(
            id=case.id,
            kind="structural",
            passed=passed == expected,
            passed_count=passed,
            required_count=expected,
            total=expected,
            seed_results=tuple(seed_results),
        )
        atomic_write_json(case_dir / "summary.json", result.to_dict())
        return result


def _utc_now() -> str:
    from cap_harness.validation import utc_now

    return utc_now()


def _robosuite_selector() -> GenericStructuralSelector:
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from cap_harness.robosuite.registry import RobosuiteTaskRegistry

    registry = RobosuiteTaskRegistry()
    return GenericStructuralSelector(
        benchmark="robosuite",
        control_period_s=0.05,
        enumerate_tasks=lambda: list(registry.enumerate_tasks()),
        make_adapter=lambda size: RobosuiteAdapter(
            camera_height=size, camera_width=size, horizon=100
        ),
        language_mode="strict",
    )


def get_structural_selector(benchmark: str) -> StructuralSelector:
    if benchmark == "robosuite":
        return _robosuite_selector()
    if benchmark == "libero-pro":
        from .selectors_libero import libero_selector  # added in the LIBERO migration

        return libero_selector()
    if benchmark == "behavior":
        from .selectors_behavior import behavior_selector

        return behavior_selector()
    raise ValueError(f"no structural selector for benchmark {benchmark!r}")


__all__ = [
    "GenericStructuralSelector",
    "StructuralSelector",
    "get_structural_selector",
]
