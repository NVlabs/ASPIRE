"""Allowlisted, evaluator-only semantic checks for validation plans.

Plans may reference a semantic evaluator only by an allowlisted name; a plan can
never import an arbitrary Python evaluator. Evaluators observe privileged
simulator state and are never registered with the generated-program tool
registry. Program-kind runs bind the evaluator inside the shared run path and
surface its verdict as ``outcome.protocol_success``; the validation runner reads
that field, so it does not instantiate evaluators itself.

Currently the only allowlisted evaluator is ``robosuite_bimanual`` (ordered
lift/handover witnesses). Its implementation lives in
``cap_harness.validation.evaluators.robosuite_bimanual`` and is lazy-imported by
the run path so that plan loading stays free of any simulator import.
"""

from __future__ import annotations

EVALUATOR_ALLOWLIST = frozenset({"behavior_pickup", "robosuite_bimanual"})


def is_allowed(name: str) -> bool:
    return name in EVALUATOR_ALLOWLIST


__all__ = ["EVALUATOR_ALLOWLIST", "is_allowed"]
