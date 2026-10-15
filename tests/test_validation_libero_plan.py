"""CPU-only tests for the LIBERO plan migration (no LIBERO/MuJoCo import).

The LIBERO structural selector delegates to ``validate_matrix``; here that call is
monkeypatched to a canned summary so the plan->selector->CaseResult wiring and the
CLI ``validate --plan`` dispatch are verified without any simulator.
"""

from __future__ import annotations

from pathlib import Path

from cap_harness import validation
from cap_harness.validation.plan import load_plan
from cap_harness.validation.runner import run_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "configs/validation/libero-smoke.yaml"
NIGHTLY = REPO_ROOT / "configs/validation/libero-nightly.yaml"


def test_real_libero_plans_load():
    smoke = load_plan(SMOKE)
    assert smoke.benchmark == "libero-pro"
    assert smoke.cases[0].seeds == (1,)
    nightly = load_plan(NIGHTLY)
    assert nightly.cases[0].seeds == (1, 2, 3)


def test_libero_smoke_plan_maps_matrix_summary(tmp_path, monkeypatch):
    captured = {}

    def fake_matrix(output_dir, *, config_path=None, nightly=False, retry_failures=False, **kw):
        captured["nightly"] = nightly
        return {"success": True, "passed": 80, "expected": 80, "failed": 0, "remaining": 0}

    monkeypatch.setattr(validation, "validate_matrix", fake_matrix)
    summary = run_plan(SMOKE, tmp_path, repo_root=REPO_ROOT)
    assert captured["nightly"] is False  # smoke
    assert summary["success"] is True
    case = summary["cases"][0]
    assert case["passed_count"] == 80
    assert case["required_count"] == 80


def test_libero_nightly_plan_runs_three_seeds(tmp_path, monkeypatch):
    captured = {}

    def fake_matrix(output_dir, *, config_path=None, nightly=False, retry_failures=False, **kw):
        captured["nightly"] = nightly
        return {"success": True, "passed": 240, "expected": 240, "failed": 0, "remaining": 0}

    monkeypatch.setattr(validation, "validate_matrix", fake_matrix)
    summary = run_plan(NIGHTLY, tmp_path, repo_root=REPO_ROOT)
    assert captured["nightly"] is True
    assert summary["cases"][0]["passed_count"] == 240


def test_cli_validate_plan_dispatches(tmp_path, monkeypatch, capsys):
    from cap_harness.cli import main

    def fake_matrix(output_dir, *, config_path=None, nightly=False, retry_failures=False, **kw):
        return {"success": True, "passed": 80, "expected": 80, "failed": 0, "remaining": 0}

    monkeypatch.setattr(validation, "validate_matrix", fake_matrix)
    code = main(["validate", "--plan", str(SMOKE), "--output-dir", str(tmp_path)])
    assert code == 0
    assert "plan libero-smoke" in capsys.readouterr().out
