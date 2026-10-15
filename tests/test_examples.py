from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cap_harness.runtime import _ProgramValidator

EXAMPLE_PROGRAMS = tuple(str(path) for path in sorted(Path("examples").glob("**/*.py")))


@pytest.mark.parametrize("relative_path", EXAMPLE_PROGRAMS)
def test_example_is_accepted_by_generated_program_validator(relative_path: str) -> None:
    source = Path(relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path, mode="exec")

    _ProgramValidator().visit(tree)
