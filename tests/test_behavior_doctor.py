from __future__ import annotations

from pathlib import Path

from cap_harness import doctor
from cap_harness.cli import build_parser


def test_doctor_environment_check_reads_the_omnigibson_variables(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMNIGIBSON_HEADLESS", "1")
    monkeypatch.setenv("OMNIGIBSON_GPU_ID", "0")
    monkeypatch.setenv("OMNIGIBSON_APPDATA_PATH", str(tmp_path))
    monkeypatch.delenv("EXP_PATH", raising=False)
    monkeypatch.delenv("CARB_APP_PATH", raising=False)
    monkeypatch.delenv("ISAAC_PATH", raising=False)
    result = doctor.check_behavior_environment(doctor.DoctorConfig(unit_mode=True))
    assert result.status == "pass", result.details
    monkeypatch.setenv("OMNIGIBSON_HEADLESS", "0")
    monkeypatch.setenv("ISAAC_PATH", "/opt/isaac")
    result = doctor.check_behavior_environment(doctor.DoctorConfig(unit_mode=False))
    assert result.status == "fail"
    assert len(result.details["problems"]) == 2


def test_doctor_dataset_check_wants_real_payload_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", str(tmp_path))
    result = doctor.check_behavior_datasets(doctor.DoctorConfig(unit_mode=False))
    assert result.status == "fail"
    assert set(result.details["missing"]) == set(doctor.BEHAVIOR_DATASET_FILES)
    for name in doctor.BEHAVIOR_DATASET_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert doctor.check_behavior_datasets(doctor.DoctorConfig(unit_mode=False)).status == "pass"
    monkeypatch.delenv("OMNIGIBSON_DATA_PATH")
    monkeypatch.setenv("CAP_HARNESS_BEHAVIOR_DATA", str(tmp_path / "other"))
    assert doctor.behavior_data_root() == tmp_path / "other"


def test_doctor_runs_the_behavior_checks_in_unit_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMNIGIBSON_DATA_PATH", str(tmp_path))
    report = doctor.run_doctor(
        doctor.DoctorConfig(embodiment="behavior", unit_mode=True, check_services=False),
        output_path=tmp_path / "doctor.json",
    )
    names = [check.name for check in report.checks]
    assert names[:7] == [
        "python",
        "platform",
        "nvidia",
        "egl",
        "source_checkout",
        "python_environment",
        "core_package_imports",
    ]
    assert names[7:12] == [
        "behavior_package_imports",
        "behavior_curobo_arch",
        "behavior_source",
        "behavior_datasets",
        "behavior_environment",
    ]
    by_name = {check.name: check for check in report.checks}
    assert by_name["behavior_source"].status == "pass", by_name["behavior_source"].details
    assert by_name["behavior_datasets"].status == "warn"
    assert by_name["behavior_datasets"].required is False


def test_cli_accepts_the_behavior_benchmark_and_runtime() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "run",
            "--benchmark",
            "behavior",
            "--suite",
            "turning_on_radio",
            "--task-id",
            "0",
            "--program",
            "p.py",
        ]
    )
    assert run.benchmark == "behavior"
    doc = parser.parse_args(["doctor", "--runtime", "behavior", "--reset-embodiment", "behavior"])
    assert doc.runtime == "behavior" and doc.reset_embodiment == "behavior"
