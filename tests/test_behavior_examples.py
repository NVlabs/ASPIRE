"""The BEHAVIOR example programs share one prelude and stay inside the program sandbox."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from cap_harness.registry import BEHAVIOR_PUBLIC_TOOL_NAMES, ToolRegistry
from cap_harness.runtime import ProgramExecutor, _ProgramValidator

EXAMPLES = Path("examples/behavior")
PUBLIC_BEHAVIOR_ATTRIBUTES = {name.split(".", 1)[1] for name in BEHAVIOR_PUBLIC_TOOL_NAMES}
BEGIN, END = "# --- prelude begin ---\n", "# --- prelude end ---\n"


def _block(text: str) -> str:
    return text[text.index(BEGIN) : text.index(END) + len(END)]


def test_examples_embed_the_shared_prelude_verbatim() -> None:
    prelude = _block((EXAMPLES / "_prelude.py").read_text(encoding="utf-8"))
    assert "EEF_TO_FINGERTIP_M" in prelude and "def attempt_grasps" in prelude
    assert "def trim_cloud" in prelude and "def choose_arm" in prelude
    assert "def top_slab_grasp_pose" in prelude and "def approach_object" in prelude
    programs = sorted(path for path in EXAMPLES.glob("*.py") if not path.name.startswith("_"))
    assert [path.name for path in programs] == [
        "picking_up_trash_seed1.py",
        "turning_on_radio_seed1.py",
    ]
    for path in programs:
        text = path.read_text(encoding="utf-8")
        assert "# prelude: behavior-r1pro-v10" in text
        assert _block(text) == prelude, path.name
        assert text.rstrip().endswith("result = report")


def test_examples_only_use_public_names() -> None:
    allowed_namespaces = {"behavior"}
    for path in EXAMPLES.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _ProgramValidator().visit(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in allowed_namespaces:
                    assert node.attr in PUBLIC_BEHAVIOR_ATTRIBUTES, (path.name, node.attr)


def test_programs_may_import_math_and_nothing_else() -> None:
    _ProgramValidator().visit(ast.parse("import math\nresult = math.cos(0.0)"))
    for source in ("import os", "import math as m", "from math import cos", "import numpy"):
        with pytest.raises(ValueError):
            _ProgramValidator().visit(ast.parse(source))


def test_executor_serves_math_and_blocks_other_imports() -> None:
    executor = ProgramExecutor(ToolRegistry())
    ok = executor.execute_program("import math\nresult = math.atan2(1.0, 1.0)")
    assert ok.ok and abs(ok.result - math.pi / 4) < 1e-12
    blocked = executor.execute_program("result = __import__('os')")
    assert not blocked.ok
