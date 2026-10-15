"""CPU-only tests for the unified validation engine (model, plan, provenance, runner).

No simulator, provider, or MuJoCo import: the runner is exercised with a fake
run command that writes outcome.json, and structural selection is not invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from cap_harness.validation import provenance
from cap_harness.validation.model import ValidationCase, ValidationPlan, as_int_tuple
from cap_harness.validation.plan import PlanError, load_plan
from cap_harness.validation.runner import run_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
EXISTING_PROGRAM = "examples/robosuite/two_arm_lift_seed1.py"


def _write_plan(tmp_path: Path, body: str) -> Path:
    # Plans resolve repo_root by walking up to a dir with examples/ + configs/;
    # place the plan inside the real repo's configs/validation tree via symlink-free copy.
    plan_dir = REPO_ROOT / "configs" / "validation"
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = plan_dir / f"_test_{tmp_path.name}.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# ---- model ---------------------------------------------------------------


def test_required_success_defaults_to_all_seeds():
    case = ValidationCase(id="c", kind="program", seeds=(1, 2, 3))
    assert case.required_success == 3
    floored = ValidationCase(id="c", kind="program", seeds=(1, 2, 3), min_success=1)
    assert floored.required_success == 1


def test_as_int_tuple_rejects_bool_and_non_int():
    assert as_int_tuple([1, 2], field_name="s") == (1, 2)
    with pytest.raises(ValueError):
        as_int_tuple([True], field_name="s")
    with pytest.raises(ValueError):
        as_int_tuple(["x"], field_name="s")


# ---- plan loader ---------------------------------------------------------


def _valid_program_plan_body() -> str:
    return f"""
    schema_version: 1
    name: robosuite-bimanual
    benchmark: robosuite
    renderer: egl
    sealed: true
    required_providers: [sam3, pyroki, curobo]
    cases:
      - id: lift
        kind: program
        suite: two_arm_lift
        task_id: 0
        seeds: [1, 2, 3]
        program: {EXISTING_PROGRAM}
        max_steps: 3000
        evaluator: robosuite_bimanual
        min_success: 3
        require:
          program_ok: true
          task_success: true
          evaluator_success: true
      - id: handover
        kind: program
        suite: two_arm_handover
        seeds: [1, 2, 3]
        program: {EXISTING_PROGRAM}
        max_steps: 7000
        evaluator: robosuite_bimanual
        min_success: 1
        require:
          program_ok: true
          task_success: true
          evaluator_success: true
    """


def test_load_valid_program_plan(tmp_path):
    path = _write_plan(tmp_path, _valid_program_plan_body())
    try:
        plan = load_plan(path)
    finally:
        path.unlink()
    assert plan.benchmark == "robosuite"
    assert plan.sealed is True
    assert plan.required_providers == ("sam3", "pyroki", "curobo")
    assert [c.id for c in plan.cases] == ["lift", "handover"]
    assert plan.cases[0].required_success == 3
    assert plan.cases[1].required_success == 1  # accepted Handover 1/3 floor


def test_load_valid_structural_plan(tmp_path):
    path = _write_plan(
        tmp_path,
        """
    schema_version: 1
    name: robosuite-structural
    benchmark: robosuite
    cases:
      - id: all
        kind: structural
        select: all
        seeds: [1, 2, 3]
    """,
    )
    try:
        plan = load_plan(path)
    finally:
        path.unlink()
    assert plan.structural_cases[0].select == "all"


@pytest.mark.parametrize(
    "header,message",
    [
        ("schema_version: 2\n    name: p\n    benchmark: robosuite", "schema_version"),
        ("schema_version: 1\n    name: p\n    benchmark: nope", "benchmark"),
    ],
)
def test_bad_header_fields(tmp_path, header, message):
    path = _write_plan(
        tmp_path,
        f"""
    {header}
    cases:
      - id: c
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: {EXISTING_PROGRAM}
        max_steps: 10
        require: {{program_ok: true}}
    """,
    )
    try:
        with pytest.raises(PlanError) as exc:
            load_plan(path)
    finally:
        path.unlink()
    assert message in str(exc.value)


def test_duplicate_case_id_rejected(tmp_path):
    path = _write_plan(
        tmp_path,
        f"""
    schema_version: 1
    name: dup
    benchmark: robosuite
    cases:
      - id: same
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: {EXISTING_PROGRAM}
        max_steps: 10
        require: {{program_ok: true}}
      - id: same
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: {EXISTING_PROGRAM}
        max_steps: 10
        require: {{program_ok: true}}
    """,
    )
    try:
        with pytest.raises(PlanError, match="duplicate case id"):
            load_plan(path)
    finally:
        path.unlink()


def test_unknown_evaluator_rejected(tmp_path):
    path = _write_plan(
        tmp_path,
        f"""
    schema_version: 1
    name: badeval
    benchmark: robosuite
    cases:
      - id: c
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: {EXISTING_PROGRAM}
        max_steps: 10
        evaluator: totally_not_allowed
        require: {{program_ok: true, evaluator_success: true}}
    """,
    )
    try:
        with pytest.raises(PlanError, match="allowlist"):
            load_plan(path)
    finally:
        path.unlink()


def test_missing_program_path_rejected(tmp_path):
    path = _write_plan(
        tmp_path,
        """
    schema_version: 1
    name: missingprog
    benchmark: robosuite
    cases:
      - id: c
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: examples/robosuite/does_not_exist.py
        max_steps: 10
        require: {program_ok: true}
    """,
    )
    try:
        with pytest.raises(PlanError, match="program not found"):
            load_plan(path)
    finally:
        path.unlink()


def test_evaluator_success_without_evaluator_rejected(tmp_path):
    path = _write_plan(
        tmp_path,
        f"""
    schema_version: 1
    name: eval-mismatch
    benchmark: robosuite
    cases:
      - id: c
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: {EXISTING_PROGRAM}
        max_steps: 10
        require: {{program_ok: true, evaluator_success: true}}
    """,
    )
    try:
        with pytest.raises(PlanError, match="no evaluator"):
            load_plan(path)
    finally:
        path.unlink()


def test_unknown_provider_rejected(tmp_path):
    path = _write_plan(
        tmp_path,
        f"""
    schema_version: 1
    name: badprov
    benchmark: robosuite
    required_providers: [sam3, nope]
    cases:
      - id: c
        kind: program
        suite: two_arm_lift
        seeds: [1]
        program: {EXISTING_PROGRAM}
        max_steps: 10
        require: {{program_ok: true}}
    """,
    )
    try:
        with pytest.raises(PlanError, match="unknown"):
            load_plan(path)
    finally:
        path.unlink()


def test_structural_with_program_field_rejected(tmp_path):
    path = _write_plan(
        tmp_path,
        f"""
    schema_version: 1
    name: bad-structural
    benchmark: robosuite
    cases:
      - id: c
        kind: structural
        seeds: [1]
        program: {EXISTING_PROGRAM}
    """,
    )
    try:
        with pytest.raises(PlanError, match="must not set program"):
            load_plan(path)
    finally:
        path.unlink()


# ---- provenance ----------------------------------------------------------


def test_sha256_file_and_source_tree(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    assert len(provenance.sha256_file(f)) == 64
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        cwd=tmp_path,
        check=True,
    )
    digest = provenance.source_tree_sha256(tmp_path)
    assert len(digest) == 64


def test_seal_environment_refuses_dirty_worktree(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "dirty.txt").write_text("uncommitted", encoding="utf-8")
    with pytest.raises(provenance.DirtyWorktreeError):
        provenance.seal_environment(tmp_path / "out", tmp_path, renderer="egl")


def test_check_program_provenance_flags_mismatches():
    issues = provenance.check_program_provenance(
        {
            "sha256": "wrong",
            "harness_git": {"commit": "abc", "dirty": True},
            "dependency_lock_sha256": "x",
        },
        program_sha256="right",
        commit="abc",
        dependency_lock_sha256="y",
    )
    assert any("program provenance hash" in i for i in issues)
    assert any("clean sealed commit" in i for i in issues)
    assert any("dependency lock" in i for i in issues)


# ---- runner (fake run command) -------------------------------------------


def _fake_run(outcome: dict):
    def _cmd(argv, env):
        oidx = argv.index("--output-root")
        root = Path(argv[oidx + 1])
        run_dir = root / "run-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
        return 0

    return _cmd


def test_program_case_all_pass(tmp_path):
    plan = ValidationPlan(
        schema_version=1,
        name="p",
        benchmark="robosuite",
        cases=(
            ValidationCase(
                id="lift",
                kind="program",
                suite="two_arm_lift",
                seeds=(1, 2, 3),
                program=EXISTING_PROGRAM,
                max_steps=10,
                min_success=3,
                require={"program_ok": True, "task_success": True},
            ),
        ),
    )
    summary = run_plan(
        plan,
        tmp_path,
        repo_root=REPO_ROOT,
        run_command=_fake_run({"program_ok": True, "task_success": True, "protocol_success": None}),
    )
    assert summary["success"] is True
    assert summary["cases"][0]["passed_count"] == 3


def test_program_case_floor_handover_one_of_three(tmp_path):
    # Only seed 1 "succeeds"; a min_success=1 floor still passes the case.
    def _cmd(argv, env):
        seed = argv[argv.index("--seed") + 1]
        root = Path(argv[argv.index("--output-root") + 1])
        run_dir = root / "run-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        ok = seed == "1"
        (run_dir / "outcome.json").write_text(
            json.dumps({"program_ok": ok, "task_success": ok, "protocol_success": ok}),
            encoding="utf-8",
        )
        return 0

    plan = ValidationPlan(
        schema_version=1,
        name="p",
        benchmark="robosuite",
        cases=(
            ValidationCase(
                id="handover",
                kind="program",
                suite="two_arm_handover",
                seeds=(1, 2, 3),
                program=EXISTING_PROGRAM,
                max_steps=10,
                min_success=1,
                require={"program_ok": True, "task_success": True, "evaluator_success": True},
            ),
        ),
    )
    summary = run_plan(plan, tmp_path, repo_root=REPO_ROOT, run_command=_cmd)
    assert summary["success"] is True
    assert summary["cases"][0]["passed_count"] == 1
    assert summary["cases"][0]["required_count"] == 1


def test_program_case_fails_below_floor(tmp_path):
    plan = ValidationPlan(
        schema_version=1,
        name="p",
        benchmark="robosuite",
        cases=(
            ValidationCase(
                id="lift",
                kind="program",
                suite="two_arm_lift",
                seeds=(1, 2, 3),
                program=EXISTING_PROGRAM,
                max_steps=10,
                min_success=3,
                require={"program_ok": True, "task_success": True},
            ),
        ),
    )
    summary = run_plan(
        plan,
        tmp_path,
        repo_root=REPO_ROOT,
        run_command=_fake_run(
            {"program_ok": True, "task_success": False, "protocol_success": None}
        ),
    )
    assert summary["success"] is False
    assert summary["cases"][0]["passed_count"] == 0


# ---- every shipped plan parses -------------------------------------------


def test_all_shipped_plans_load():
    plan_dir = REPO_ROOT / "configs" / "validation"
    plans = sorted(p for p in plan_dir.glob("*.yaml") if not p.name.startswith("_test_"))
    assert {p.name for p in plans} >= {
        "libero-smoke.yaml",
        "libero-nightly.yaml",
        "robosuite-structural.yaml",
        "robosuite-bimanual.yaml",
        "robosuite-control.yaml",
    }
    for path in plans:
        plan = load_plan(path)
        assert plan.name and plan.cases


# ---- import-light invariant ----------------------------------------------


def test_engine_import_is_simulator_free():
    code = (
        "import cap_harness.validation.model, cap_harness.validation.plan, "
        "cap_harness.validation.provenance, cap_harness.validation.runner, "
        "cap_harness.validation.selectors, cap_harness.validation.evaluators; "
        "import sys; "
        "bad={'mujoco','robosuite','libero','torch','omnigibson','cap_harness.robosuite.adapter',"
        "'cap_harness.behavior.adapter'} & set(sys.modules); "
        "assert not bad, bad; print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout
