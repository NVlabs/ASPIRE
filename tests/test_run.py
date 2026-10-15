from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from cap_harness.artifacts import RunTiming, StepLimitReached
import cap_harness.run as run_module
from cap_harness.runtime import ProgramExecutionResult


class _Recorder:
    instances: ClassVar[list[_Recorder]] = []

    def __init__(self, *, output_root: Path, **kwargs) -> None:
        del kwargs
        self.root = output_root / "run"
        (self.root / "logs").mkdir(parents=True)
        self.timing = RunTiming()
        self.terminal_reason = None
        self.terminated = False
        self.truncated = False
        self.events: list[str] = []
        self.finalize_kwargs = None
        self.instances.append(self)

    def save_program(self, path: Path) -> None:
        del path

    def import_model_trace(self, path: Path | None) -> None:
        del path

    def save_protocol_evidence(self, evidence) -> None:
        assert evidence["protocol_success"] is True
        self.events.append("protocol")

    def finalize(self, **kwargs) -> None:
        self.events.append("finalize")
        self.finalize_kwargs = kwargs

    def close_incomplete(self, error: BaseException, **kwargs) -> None:
        raise AssertionError(f"unexpected incomplete run: {error!r}, {kwargs!r}")


class _Adapter:
    instances: ClassVar[list[_Adapter]] = []
    close_error: Exception | None = None

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.instances.append(self)

    def reset(self, metadata, seed: int) -> None:
        del metadata, seed

    def check_success(self) -> bool:
        return False

    def close(self) -> None:
        _Recorder.instances[-1].events.append("close")
        if self.close_error is not None:
            raise self.close_error


def _patch_run(monkeypatch, execution) -> None:
    metadata = SimpleNamespace(suite_name="libero_object_swap", task_id=0, task_name="task")
    monkeypatch.setattr(
        run_module, "LiberoSuiteRegistry", lambda: SimpleNamespace(resolve=lambda *args: metadata)
    )
    monkeypatch.setattr(run_module, "RunRecorder", _Recorder)
    monkeypatch.setattr(run_module, "LiberoAdapter", _Adapter)
    monkeypatch.setattr(run_module, "Sam3Provider", lambda **kwargs: object())
    monkeypatch.setattr(run_module, "ContactGraspNetProvider", lambda: object())
    monkeypatch.setattr(run_module, "PyRokiProvider", lambda: object())
    monkeypatch.setattr(run_module, "TracedProvider", lambda provider, *args: provider)
    monkeypatch.setattr(
        run_module,
        "CapApi",
        lambda *args, **kwargs: SimpleNamespace(register_tools=lambda registry: registry),
    )
    monkeypatch.setattr(run_module, "ToolRegistry", lambda: object())
    monkeypatch.setattr(
        run_module,
        "ProgramExecutor",
        lambda *args, **kwargs: SimpleNamespace(execute_program=lambda source: execution(source)),
    )


def test_run_closes_before_finalizing_records_cleanup_failure_and_redacts_stdout(
    tmp_path, monkeypatch
) -> None:
    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = RuntimeError("cleanup failed")
    _patch_run(
        monkeypatch,
        lambda source: ProgramExecutionResult(
            ok=True,
            result=None,
            stdout=f"ran {source.strip()} with Bearer sensitive-value",
            calls=(),
        ),
    )
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )

    recorder = _Recorder.instances[-1]
    assert outcome.program_ok
    assert recorder.events == ["close", "finalize"]
    assert recorder.finalize_kwargs["cleanup_errors"] == (
        "adapter close: RuntimeError: cleanup failed",
    )
    stdout = (recorder.root / "logs/program.stdout").read_text(encoding="utf-8")
    assert stdout == "ran result = 1 with [REDACTED]"
    assert "sensitive-value" not in stdout


def test_run_records_user_interrupt_after_cleanup(tmp_path, monkeypatch) -> None:
    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = None

    def interrupt(source: str) -> ProgramExecutionResult:
        del source
        raise KeyboardInterrupt

    _patch_run(monkeypatch, interrupt)
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    with pytest.raises(KeyboardInterrupt):
        run_module.run_program(
            benchmark="libero-pro",
            suite="libero_object_swap",
            task_id=0,
            seed=1,
            program_path=program,
            output_root=tmp_path / "outputs",
            max_steps=10,
        )

    recorder = _Recorder.instances[-1]
    assert recorder.events == ["close", "finalize"]
    assert recorder.finalize_kwargs["termination_reason"] == "user_interrupt"


def test_run_registers_all_program_selectable_motion_backends(tmp_path, monkeypatch) -> None:
    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = None
    _patch_run(
        monkeypatch,
        lambda source: ProgramExecutionResult(ok=True, result=source, stdout="", calls=()),
    )
    curobo = object()
    captured: dict[str, object] = {}
    curobo_init: dict[str, object] = {}
    monkeypatch.setenv("CAP_HARNESS_CUROBO_URL", "http://127.0.0.1:8128")

    def curobo_factory(base_url, **kwargs):
        curobo_init.update(base_url=base_url, **kwargs)
        return curobo

    monkeypatch.setattr(run_module, "CuRoboProvider", curobo_factory)

    def api_factory(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(register_tools=lambda registry: registry)

    monkeypatch.setattr(run_module, "CapApi", api_factory)
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )

    assert outcome.program_ok
    assert set(captured["grasp_providers"]) == {"contact-graspnet"}
    ik_providers = captured["ik_providers"]
    assert set(ik_providers) == {"pyroki", "curobo"}
    assert ik_providers["curobo"] is curobo
    assert captured["trajectory_planning_providers"] == {"curobo": curobo}
    assert captured["integrated_pose_planning_providers"] == {"curobo-integrated": curobo}
    assert curobo_init == {"base_url": "http://127.0.0.1:8128"}


def test_step_limit_still_asks_the_adapter_whether_the_task_succeeded(
    tmp_path, monkeypatch
) -> None:
    """Running out of step budget is not evidence of failure.

    A program that completed the task and kept stepping used to be recorded as
    task_success=False by the StepLimitReached branch alone; the interrupt branch
    already consulted the adapter, and this one must too.
    """
    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = None

    def exhausted(source):
        del source
        raise StepLimitReached("run reached max_steps=10")

    _patch_run(monkeypatch, exhausted)
    monkeypatch.setattr(_Adapter, "check_success", lambda self: True)
    monkeypatch.setattr(run_module, "CuRoboProvider", lambda **kwargs: object())
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    run_module.run_program(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )

    finalize = _Recorder.instances[-1].finalize_kwargs
    assert finalize is not None
    assert finalize["termination_reason"] == "step_limit"
    assert finalize["program_ok"] is False
    assert finalize["task_success"] is True


def test_run_points_providers_at_the_urls_in_the_environment(tmp_path, monkeypatch) -> None:
    """CAP_HARNESS_CUROBO_URL/CAP_HARNESS_SAM3_URL must reach the provider constructors.

    The drawer README's private-planner procedure depended on this and it was a
    no-op: run.py built CuRoboProvider() with the default URL for every benchmark
    so the "fresh" planner was never contacted.
    """
    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = None
    _patch_run(
        monkeypatch,
        lambda source: ProgramExecutionResult(ok=True, result=source, stdout="", calls=()),
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        run_module, "Sam3Provider", lambda **kwargs: seen.setdefault("sam3", kwargs) and object()
    )
    monkeypatch.setattr(
        run_module,
        "CuRoboProvider",
        lambda **kwargs: seen.setdefault("curobo", kwargs) and object(),
    )
    monkeypatch.setenv("CAP_HARNESS_CUROBO_URL", "http://127.0.0.1:8120")
    monkeypatch.setenv("CAP_HARNESS_SAM3_URL", "http://127.0.0.1:8130")
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )

    assert outcome.program_ok
    assert seen["curobo"]["base_url"] == "http://127.0.0.1:8120"
    assert seen["sam3"]["base_url"] == "http://127.0.0.1:8130"


def test_bimanual_protocol_is_captured_before_adapter_close(tmp_path, monkeypatch) -> None:
    import cap_harness.robosuite.adapter as adapter_module
    import cap_harness.validation.evaluators.robosuite_bimanual as protocol_module

    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = None
    _patch_run(
        monkeypatch,
        lambda source: ProgramExecutionResult(ok=True, result=source, stdout="", calls=()),
    )

    class ProtocolAdapter(_Adapter):
        native_env = object()

        def bind_protocol_evaluator(self, evaluator) -> None:
            self.evaluator = evaluator

        def check_success(self) -> bool:
            return True

    class ProtocolEvaluator:
        def __init__(self, task_name, native_env) -> None:
            assert task_name == "two_arm_lift"
            assert native_env is ProtocolAdapter.native_env

        def evidence(self, *, final_native_success: bool):
            assert final_native_success is True
            return {"protocol_success": True}

    monkeypatch.setattr(adapter_module, "RobosuiteAdapter", ProtocolAdapter)
    monkeypatch.setattr(
        protocol_module,
        "RobosuiteBimanualProtocolEvaluator",
        ProtocolEvaluator,
    )
    monkeypatch.setattr(run_module, "ToolRegistry", lambda **_kwargs: object())
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="robosuite",
        suite="two_arm_lift",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )

    recorder = _Recorder.instances[-1]
    assert outcome.protocol_success is True
    assert recorder.events == ["protocol", "close", "finalize"]
    assert recorder.finalize_kwargs["protocol_success"] is True


def test_bimanual_protocol_evaluator_failure_is_recorded_and_cleanup_continues(
    tmp_path, monkeypatch
) -> None:
    import cap_harness.robosuite.adapter as adapter_module
    import cap_harness.validation.evaluators.robosuite_bimanual as protocol_module

    _Recorder.instances.clear()
    _Adapter.instances.clear()
    _Adapter.close_error = None
    _patch_run(
        monkeypatch,
        lambda source: ProgramExecutionResult(ok=True, result=source, stdout="", calls=()),
    )

    class ProtocolAdapter(_Adapter):
        native_env = object()

        def bind_protocol_evaluator(self, evaluator) -> None:
            self.evaluator = evaluator

        def check_success(self) -> bool:
            return True

    class FailingProtocolEvaluator:
        def __init__(self, task_name, native_env) -> None:
            del task_name, native_env

        def evidence(self, *, final_native_success: bool):
            del final_native_success
            raise RuntimeError("evaluator failed")

    recorded: dict[str, object] = {}

    def save_protocol_evidence(self, evidence) -> None:
        recorded.update(evidence)
        self.events.append("protocol")

    monkeypatch.setattr(_Recorder, "save_protocol_evidence", save_protocol_evidence)
    monkeypatch.setattr(adapter_module, "RobosuiteAdapter", ProtocolAdapter)
    monkeypatch.setattr(
        protocol_module,
        "RobosuiteBimanualProtocolEvaluator",
        FailingProtocolEvaluator,
    )
    monkeypatch.setattr(run_module, "ToolRegistry", lambda **_kwargs: object())
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")

    outcome = run_module.run_program(
        benchmark="robosuite",
        suite="two_arm_lift",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=10,
    )

    recorder = _Recorder.instances[-1]
    assert outcome.protocol_success is False
    assert recorded["protocol_success"] is False
    assert recorded["evaluator_errors"] == ("RuntimeError",)
    assert recorder.events == ["protocol", "close", "finalize"]
