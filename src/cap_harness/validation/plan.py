"""Strict loader for declarative validation plans.

Validates a plan's benchmark, case kinds, task selectors, seed lists, program
paths, evaluator names, and required-outcome keys *before* any execution, so a
malformed plan fails fast with a precise message rather than mid-run. Import-light:
only PyYAML + stdlib; no simulator, provider, or embodiment import.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .evaluators import EVALUATOR_ALLOWLIST
from .model import (
    BENCHMARKS,
    CASE_KINDS,
    PROGRAM_REQUIRE_KEYS,
    RENDERERS,
    ValidationCase,
    ValidationPlan,
    as_int_tuple,
)

KNOWN_PROVIDERS = ("sam3", "contact_graspnet", "pyroki", "curobo")


class PlanError(ValueError):
    """Raised when a validation plan is malformed or references missing assets."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def _repo_root(plan_path: Path) -> Path:
    """Resolve the repo root (the cap package root) for program-path checks."""
    # Plans live under <cap>/configs/validation/<plan>.yaml; the cap root holds
    # examples/ and configs/. Walk up until an examples/ dir is found.
    for parent in plan_path.resolve().parents:
        if (parent / "examples").is_dir() and (parent / "configs").is_dir():
            return parent
    return plan_path.resolve().parent


def _load_case(raw: Any, *, index: int, benchmark: str, repo_root: Path) -> ValidationCase:
    _require(isinstance(raw, Mapping), f"case #{index} must be a mapping")
    cid = raw.get("id")
    _require(
        isinstance(cid, str) and cid.strip() != "", f"case #{index} needs a non-empty string id"
    )
    kind = raw.get("kind")
    _require(kind in CASE_KINDS, f"case {cid!r} kind must be one of {CASE_KINDS}, got {kind!r}")

    seeds_raw = raw.get("seeds")
    _require(
        isinstance(seeds_raw, list) and len(seeds_raw) > 0,
        f"case {cid!r} needs a non-empty seeds list",
    )
    seeds = as_int_tuple(seeds_raw, field_name=f"case {cid!r} seeds")
    _require(all(s >= 0 for s in seeds), f"case {cid!r} seeds must be non-negative")

    task_id = raw.get("task_id", 0)
    _require(
        isinstance(task_id, int) and not isinstance(task_id, bool),
        f"case {cid!r} task_id must be an int",
    )

    require_raw = raw.get("require", {}) or {}
    _require(isinstance(require_raw, Mapping), f"case {cid!r} require must be a mapping")

    suite = raw.get("suite")
    select = raw.get("select")
    program = raw.get("program")
    max_steps = raw.get("max_steps")
    evaluator = raw.get("evaluator")
    min_success = raw.get("min_success")

    if kind == "program":
        _require(
            isinstance(program, str) and program != "",
            f"case {cid!r} (program) needs a program path",
        )
        program_path = (repo_root / program).resolve()
        _require(program_path.is_file(), f"case {cid!r} program not found: {program}")
        _require(
            isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps > 0,
            f"case {cid!r} (program) needs a positive integer max_steps",
        )
        _require(isinstance(suite, str) and suite != "", f"case {cid!r} (program) needs a suite")
        _require(
            all(key in PROGRAM_REQUIRE_KEYS for key in require_raw),
            f"case {cid!r} require keys must be a subset of {PROGRAM_REQUIRE_KEYS}",
        )
        _require(
            all(isinstance(v, bool) for v in require_raw.values()),
            f"case {cid!r} require values must be booleans",
        )
        if evaluator is not None:
            _require(
                evaluator in EVALUATOR_ALLOWLIST,
                f"case {cid!r} evaluator {evaluator!r} is not allowlisted {tuple(sorted(EVALUATOR_ALLOWLIST))}",
            )
            _require(
                bool(require_raw.get("evaluator_success")) is True,
                f"case {cid!r} declares an evaluator but does not require evaluator_success",
            )
        elif "evaluator_success" in require_raw:
            raise PlanError(f"case {cid!r} requires evaluator_success but declares no evaluator")
        if min_success is not None:
            _require(
                isinstance(min_success, int)
                and not isinstance(min_success, bool)
                and 0 <= min_success <= len(seeds),
                f"case {cid!r} min_success must be an int in [0, {len(seeds)}]",
            )
    else:  # structural
        _require(
            select in (None, "all", "representatives") or (isinstance(suite, str) and suite != ""),
            f"case {cid!r} (structural) needs select: all|representatives or a suite",
        )
        _require(program is None, f"case {cid!r} (structural) must not set program")
        _require(evaluator is None, f"case {cid!r} (structural) must not set evaluator")
        _require(min_success is None, f"case {cid!r} (structural) must not set min_success")

    return ValidationCase(
        id=cid,
        kind=kind,
        seeds=seeds,
        suite=suite,
        task_id=task_id,
        select=select,
        program=program,
        max_steps=max_steps,
        evaluator=evaluator,
        min_success=min_success,
        require={str(k): bool(v) for k, v in require_raw.items()},
    )


def load_plan(path: str | Path) -> ValidationPlan:
    """Parse and fully validate a plan file; raise PlanError on any problem."""
    plan_path = Path(path)
    _require(plan_path.is_file(), f"plan file not found: {plan_path}")
    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlanError(f"plan {plan_path} is not valid YAML: {exc}") from exc
    _require(isinstance(raw, Mapping), f"plan {plan_path} must be a top-level mapping")

    _require(raw.get("schema_version") == 1, "plan schema_version must be 1")
    name = raw.get("name")
    _require(isinstance(name, str) and name.strip() != "", "plan needs a non-empty name")
    benchmark = raw.get("benchmark")
    _require(
        benchmark in BENCHMARKS, f"plan benchmark must be one of {BENCHMARKS}, got {benchmark!r}"
    )

    renderer = raw.get("renderer")
    _require(renderer is None or renderer in RENDERERS, f"plan renderer must be one of {RENDERERS}")

    sealed = raw.get("sealed", False)
    _require(isinstance(sealed, bool), "plan sealed must be a boolean")

    providers_raw = raw.get("required_providers", []) or []
    _require(isinstance(providers_raw, list), "required_providers must be a list")
    for provider in providers_raw:
        _require(
            provider in KNOWN_PROVIDERS,
            f"required provider {provider!r} is unknown; known: {KNOWN_PROVIDERS}",
        )

    cases_raw = raw.get("cases")
    _require(
        isinstance(cases_raw, list) and len(cases_raw) > 0, "plan needs a non-empty cases list"
    )
    repo_root = _repo_root(plan_path)
    cases = tuple(
        _load_case(case, index=i, benchmark=benchmark, repo_root=repo_root)
        for i, case in enumerate(cases_raw)
    )
    seen: set[str] = set()
    for case in cases:
        _require(case.id not in seen, f"duplicate case id {case.id!r}")
        seen.add(case.id)

    return ValidationPlan(
        schema_version=1,
        name=name,
        benchmark=benchmark,
        cases=cases,
        renderer=renderer,
        sealed=sealed,
        required_providers=tuple(providers_raw),
    )


__all__ = ["KNOWN_PROVIDERS", "PlanError", "load_plan"]
