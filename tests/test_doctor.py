from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cap_harness.doctor import (
    DoctorConfig,
    ResetTarget,
    _default_reset_probe,
    check_nvidia,
    check_services,
    redact_secrets,
    run_doctor,
)


def _doctor_tree(tmp_path: Path):
    shared = tmp_path / "shared"
    source = shared / "cap-harness"
    build = source / ".venv-libero"
    submodule = source / "third_party/LIBERO-PRO"
    config = source / "configs/libero/validation_cases.yaml"
    build.mkdir(parents=True)
    submodule.mkdir(parents=True)
    (submodule / "README.md").write_text("fixture\n")
    libero_root = submodule / "libero/libero"
    configured = {
        "benchmark_root": libero_root,
        "bddl_files": libero_root / "bddl_files",
        "init_states": libero_root / "init_files",
        "datasets": submodule / "libero/datasets",
        "assets": libero_root / "assets",
    }
    for path in configured.values():
        path.mkdir(parents=True, exist_ok=True)
    path_config = tmp_path / ".libero/config.yaml"
    path_config.parent.mkdir()
    path_config.write_text(json.dumps({key: str(value) for key, value in configured.items()}))
    config.parent.mkdir(parents=True)
    config.write_text("required_count: 80\n")
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    (source / "uv.lock").write_text("version = 1\n")
    (source / "configs/dependency-lock.json").write_text("{}\n")
    return shared, source, build, submodule, config, path_config


def test_redaction_removes_secret_keys_and_inline_values():
    secret = "sk-test-value-123"
    payload = {
        "NVIDIA_API_KEY": secret,
        "nested": {
            "message": f"token={secret} Authorization: Bearer {secret}",
            "url": f"https://user:{secret}@example.test/path",
        },
        "safe": "visible",
    }

    encoded = json.dumps(redact_secrets(payload))

    assert secret not in encoded
    assert "user:" not in encoded
    assert "visible" in encoded
    assert encoded.count("[REDACTED]") >= 3


def test_cpu_unit_mode_report_is_structured_and_non_gpu_is_nonfatal(tmp_path, monkeypatch):
    shared, source, build, submodule, validation_config, path_config = _doctor_tree(tmp_path)
    monkeypatch.setattr("cap_harness.doctor.shutil.which", lambda _: None)
    reset_calls = []

    def reset_probe(target):
        reset_calls.append(target)
        return {"suite": "fake", "task_id": 0, "seed": target.seed}

    output = tmp_path / "environment.json"
    report = run_doctor(
        DoctorConfig(
            repository_root=source,
            environment_root=build,
            unit_mode=True,
            check_services=False,
            core_packages=("json",),
            libero_packages=(),
            validation_config=validation_config,
            libero_submodule=submodule,
            libero_path_config=path_config,
            reset=ResetTarget(suite="fake", task_id=0, seed=1),
        ),
        output_path=output,
        reset_probe=reset_probe,
    )

    assert report.success is True
    assert report.exit_code == 0
    assert reset_calls == [ResetTarget(suite="fake", task_id=0, seed=1)]
    checks = {check.name: check for check in report.checks}
    assert checks["nvidia"].status == "warn"
    assert checks["nvidia"].required is False
    assert checks["python_environment"].status == "pass"
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["mode"] == "unit"
    assert payload["success"] is True
    assert {check["name"] for check in payload["checks"]} >= {
        "python",
        "platform",
        "nvidia",
        "egl",
        "source_checkout",
        "python_environment",
        "core_package_imports",
        "libero_package_imports",
        "libero_paths",
        "services",
        "single_reset",
    }


def test_required_environment_path_failure_produces_nonzero_exit(tmp_path):
    shared, source, _, submodule, validation_config, path_config = _doctor_tree(tmp_path)
    missing_build = tmp_path / "does-not-exist"
    report = run_doctor(
        DoctorConfig(
            repository_root=source,
            environment_root=missing_build,
            unit_mode=True,
            check_services=False,
            core_packages=("json",),
            libero_packages=(),
            validation_config=validation_config,
            libero_submodule=submodule,
            libero_path_config=path_config,
        )
    )

    assert report.success is False
    assert report.exit_code == 1
    build_check = next(check for check in report.checks if check.name == "python_environment")
    assert build_check.status == "fail"
    assert build_check.required is True


def test_nvidia_absence_is_warning_in_cpu_only_mode(monkeypatch):
    monkeypatch.setattr("cap_harness.doctor.shutil.which", lambda _: None)

    check = check_nvidia(DoctorConfig(unit_mode=True))

    assert check.status == "warn"
    assert check.required is False


def test_service_health_checks_exact_required_ports(monkeypatch):
    calls = []

    def http_status(host, port, timeout):
        calls.append((host, port, timeout))
        return True, 200, None

    monkeypatch.setattr("cap_harness.doctor._http_status", http_status)

    check = check_services(DoctorConfig(service_timeout_s=0.25))

    assert check.status == "pass"
    assert calls == [
        ("127.0.0.1", 8114, 0.25),
        ("127.0.0.1", 8115, 0.25),
        ("127.0.0.1", 8116, 0.25),
    ]


def test_default_reset_probe_passes_task_ref_to_adapter(monkeypatch):
    entry = SimpleNamespace(
        suite="libero_goal_swap",
        task_id=3,
        task_language="open the drawer",
    )

    class Manifest:
        representatives = {"goal": entry}

        def __iter__(self):
            return iter((entry,))

    class Adapter:
        def __init__(self):
            self.reset_calls = []
            self.closed = False

        def reset(self, task_ref, seed):
            self.reset_calls.append((task_ref, seed))
            return {"observation": True}

        def close(self):
            self.closed = True

    adapter = Adapter()
    monkeypatch.setattr("cap_harness.validation.load_libero_registry", lambda: object())
    monkeypatch.setattr("cap_harness.validation.build_required_manifest", lambda _: Manifest())
    monkeypatch.setattr(
        "cap_harness.validation._default_adapter_factory",
        lambda _: lambda selected, seed: adapter,
    )
    monkeypatch.setattr("cap_harness.validation._unpack_reset", lambda result: (result, {}))
    monkeypatch.setattr(
        "cap_harness.validation._runtime_language", lambda selected, info: "open the drawer"
    )
    monkeypatch.setattr("cap_harness.validation.validate_observation_schema", lambda obs: {})

    result = _default_reset_probe(ResetTarget(seed=1))

    assert adapter.reset_calls == [(("libero_goal_swap", 3), 1)]
    assert adapter.closed is True
    assert result["suite"] == "libero_goal_swap"


def test_behavior_curobo_arch_check_compares_compiled_kernels_with_the_gpu(tmp_path) -> None:
    from cap_harness.doctor import check_behavior_curobo_arch, compiled_cuda_arches

    blob = tmp_path / "geom_cu.so"
    cubin = bytearray(b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(44))
    cubin[18:20] = (190).to_bytes(2, "little")  # EM_CUDA
    cubin[48:52] = (0x59).to_bytes(4, "little")  # e_flags: sm_89
    blob.write_bytes(b"host code..." + bytes(cubin) + b"...compute_120 ptx text...")
    assert compiled_cuda_arches([blob]) == {"sm_89", "sm_120"}
    config = DoctorConfig(unit_mode=True)
    bad = check_behavior_curobo_arch(config, arches={"sm_120"}, device_arch="sm_89")
    assert bad.status != "pass" and "sm_89" in bad.summary and "bootstrap_behavior" in bad.summary
    good = check_behavior_curobo_arch(config, arches={"sm_89", "sm_120"}, device_arch="sm_89")
    assert good.status == "pass"
    unknown = check_behavior_curobo_arch(config, arches=set(), device_arch="sm_89")
    assert unknown.status == "pass" and "note" in unknown.details
