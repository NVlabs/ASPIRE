"""LIBERO structural selector: delegates to the resumable release matrix.

LIBERO's structural gate is richer than the generic reset/step check (dynamic
registry enumeration, BDDL/init-state checks, authoritative ``_task`` language,
representative selection, one-tick timing, resumability, and validation
fingerprints). Rather than reimplement it, the LIBERO plan case delegates to the
existing ``validate_matrix`` runner and maps its summary onto a ``CaseResult``.
A single-seed case runs the 80-pair smoke gate; a multi-seed case runs the
240-case nightly gate. All LIBERO imports are lazy (inside ``run``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import CaseResult, ValidationCase


@dataclass(frozen=True, slots=True)
class LiberoMatrixSelector:
    def run(self, case: ValidationCase, output_dir: Path, *, retry_failures: bool) -> CaseResult:
        from cap_harness.validation import atomic_write_json, validate_matrix

        nightly = len(case.seeds) > 1
        case_dir = output_dir / case.id
        summary = validate_matrix(
            case_dir,
            config_path=None,
            nightly=nightly,
            retry_failures=retry_failures,
        )
        passed = int(summary["passed"])
        expected = int(summary["expected"])
        result = CaseResult(
            id=case.id,
            kind="structural",
            passed=bool(summary["success"]),
            passed_count=passed,
            required_count=expected,
            total=expected,
            seed_results=(),
            detail={
                "mode": "nightly" if nightly else "smoke",
                "failed": int(summary["failed"]),
                "remaining": int(summary["remaining"]),
                "matrix_summary": str(case_dir / "summary.json"),
            },
        )
        atomic_write_json(case_dir / "case-summary.json", result.to_dict())
        return result


def libero_selector() -> LiberoMatrixSelector:
    return LiberoMatrixSelector()


__all__ = ["LiberoMatrixSelector", "libero_selector"]
