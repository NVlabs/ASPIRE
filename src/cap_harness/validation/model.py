"""Typed plan/case/result contracts for the unified validation engine.

Import-light by construction: this module pulls in no simulator, provider, or
LIBERO/RoboSuite code, so ``cap-harness --help`` and plan parsing work
on a machine without MuJoCo, CUDA, or any embodiment installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

BENCHMARKS = ("behavior", "libero-pro", "robosuite")
CASE_KINDS = ("structural", "program")
PROGRAM_REQUIRE_KEYS = ("program_ok", "task_success", "evaluator_success")
RENDERERS = ("cpu", "egl")


@dataclass(frozen=True, slots=True)
class ValidationCase:
    """One declared validation case: a structural sweep or a program run.

    A ``program`` case runs ``program`` once per seed in a fresh process and
    checks ``require`` against the run's ``outcome.json``; it passes when at
    least ``min_success`` of its seeds pass (default: all of them). A
    ``structural`` case resets every selected task/seed and checks the generic
    reset/observation/step invariants.
    """

    id: str
    kind: str
    seeds: tuple[int, ...]
    suite: str | None = None
    task_id: int = 0
    select: str | None = None
    program: str | None = None
    max_steps: int | None = None
    evaluator: str | None = None
    min_success: int | None = None
    require: Mapping[str, bool] = field(default_factory=dict)

    @property
    def required_success(self) -> int:
        """Seeds that must pass for the case to pass (floor); default all."""
        return len(self.seeds) if self.min_success is None else self.min_success


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    """A declarative validation plan: examples + tasks + seeds + gates."""

    schema_version: int
    name: str
    benchmark: str
    cases: tuple[ValidationCase, ...]
    renderer: str | None = None
    sealed: bool = False
    required_providers: tuple[str, ...] = ()

    @property
    def program_cases(self) -> tuple[ValidationCase, ...]:
        return tuple(case for case in self.cases if case.kind == "program")

    @property
    def structural_cases(self) -> tuple[ValidationCase, ...]:
        return tuple(case for case in self.cases if case.kind == "structural")


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Outcome of a single (case, seed) attempt."""

    seed: int
    passed: bool
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Aggregated outcome of one case across its seeds/tasks."""

    id: str
    kind: str
    passed: bool
    passed_count: int
    required_count: int
    total: int
    seed_results: tuple[SeedResult, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "passed": self.passed,
            "passed_count": self.passed_count,
            "required_count": self.required_count,
            "total": self.total,
            "seeds": [
                {
                    "seed": s.seed,
                    "passed": s.passed,
                    **({"detail": dict(s.detail)} if s.detail else {}),
                }
                for s in self.seed_results
            ],
            **({"detail": dict(self.detail)} if self.detail else {}),
        }


def as_int_tuple(values: Sequence[Any], *, field_name: str) -> tuple[int, ...]:
    """Coerce a YAML sequence to a tuple of ints, raising on non-ints."""
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must contain integers, got {value!r}")
        result.append(int(value))
    return tuple(result)


__all__ = [
    "BENCHMARKS",
    "CASE_KINDS",
    "PROGRAM_REQUIRE_KEYS",
    "RENDERERS",
    "CaseResult",
    "SeedResult",
    "ValidationCase",
    "ValidationPlan",
    "as_int_tuple",
]
