"""run.py wiring for the behavior benchmark, with the simulator replaced by fakes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cap_harness.artifacts import RunTiming
import cap_harness.run as run_module
from cap_harness.runtime import ProgramExecutionResult


class _Recorder:
    instances: list[_Recorder] = []

    def __init__(self, *, output_root: Path, **kwargs) -> None:
        self.timing = RunTiming()
        self.kwargs = kwargs
        self.root = output_root / "run"
        (self.root / "logs").mkdir(parents=True)
        self.terminal_reason = None
        self.terminated = False
        self.truncated = False
        self.frequency = 20.0
        self.events: list[str] = []
        self.finalize_kwargs = None
        self.instances.append(self)

    def save_program(self, path: Path) -> None:
        del path

    def import_model_trace(self, path: Path | None) -> None:
        del path

    def save_protocol_evidence(self, evidence) -> None:
        self.evidence = evidence
        self.events.append("protocol")

    def finalize(self, **kwargs) -> None:
        self.events.append("finalize")
        self.finalize_kwargs = kwargs

    def close_incomplete(self, error: BaseException, **kwargs) -> None:
        raise AssertionError(f"unexpected incomplete run: {error!r}, {kwargs!r}")


class _Planner:
    pass


class _Adapter:
    instances: list[_Adapter] = []
    native_env = object()
    control_frequency = 30.0
    success_observed_step = 7

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.evaluator = None
        self.closed = False
        self.instances.append(self)

    def reset(self, metadata, seed: int) -> None:
        self.reset_args = (metadata, seed)

    def planner(self) -> _Planner:
        return _Planner()

    def bind_protocol_evaluator(self, evaluator) -> None:
        self.evaluator = evaluator

    def check_success(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True
        _Recorder.instances[-1].events.append("close")


class _Witness:
    def __init__(self, task_name, native_env, *, target_scope) -> None:
        assert task_name == "turning_on_radio"
        assert native_env is _Adapter.native_env
        assert target_scope == "radio_receiver.n.01_1"

    def evidence(self, *, final_native_success: bool):
        return {"protocol_success": bool(final_native_success)}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    import cap_harness.behavior.adapter as adapter_module
    import cap_harness.validation.evaluators.behavior_pickup as witness_module

    _Recorder.instances.clear()
    _Adapter.instances.clear()
    captured: dict[str, object] = {}
    monkeypatch.setattr(run_module, "RunRecorder", _Recorder)
    monkeypatch.setattr(adapter_module, "BehaviorAdapter", _Adapter)
    monkeypatch.setattr(witness_module, "BehaviorPickupWitness", _Witness)
    monkeypatch.setattr(run_module, "Sam3Provider", lambda **kwargs: object())
    monkeypatch.setattr(run_module, "ContactGraspNetProvider", lambda: object())
    monkeypatch.setattr(run_module, "PyRokiProvider", lambda: pytest.fail("PyRoKi is not used"))
    monkeypatch.setattr(
        run_module, "CuRoboProvider", lambda **kwargs: pytest.fail("HTTP cuRobo is not used")
    )
    monkeypatch.setattr(run_module, "TracedProvider", lambda provider, *args: provider)

    def api_factory(adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return SimpleNamespace(register_tools=lambda registry: registry)

    monkeypatch.setattr(run_module, "CapApi", api_factory)
    monkeypatch.setattr(run_module, "ToolRegistry", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        run_module,
        "ProgramExecutor",
        lambda *args, **kwargs: SimpleNamespace(
            execute_program=lambda source: ProgramExecutionResult(
                ok=True, result=source, stdout="", calls=()
            )
        ),
    )
    program = tmp_path / "policy.py"
    program.write_text("result = 1", encoding="utf-8")
    return captured, program, tmp_path


def test_behavior_run_wires_the_in_process_planner_and_keeps_isaac_alive_until_finalized(
    wired,
) -> None:
    captured, program, tmp_path = wired
    outcome = run_module.run_program(
        benchmark="behavior",
        suite="turning_on_radio",
        task_id=0,
        seed=3,
        program_path=program,
        output_root=tmp_path / "outputs",
        max_steps=60,
    )
    recorder = _Recorder.instances[-1]
    adapter = _Adapter.instances[-1]
    assert outcome.task_success is True and outcome.protocol_success is True
    assert outcome.termination_reason == "task_succeeded"
    # Artifacts are finalized before the simulator is torn down.
    assert recorder.events == ["protocol", "finalize", "close"]
    assert adapter.closed
    assert recorder.kwargs["video_frame_stride"] == 3
    assert recorder.kwargs["providers"]["ik"] == ("curobo",)
    assert recorder.kwargs["providers"]["pose_planning"] == ("curobo-integrated",)
    assert recorder.frequency == 30.0
    assert adapter.reset_args[1] == 3 and adapter.reset_args[0].task_name == "turning_on_radio"
    assert adapter.kwargs["horizon"] == 60 and "init_mode" not in adapter.kwargs
    assert isinstance(adapter.evaluator, _Witness)
    assert captured["default_camera"] == "head"
    strategy = captured["default_motion_strategy"]
    assert (strategy.ik_solver, strategy.trajectory_planner, strategy.pose_planner) == (
        "curobo",
        "curobo",
        "curobo-integrated",
    )
    assert isinstance(captured["ik_providers"]["curobo"], _Planner)
    assert set(captured["trajectory_planning_providers"]) == {"curobo"}
    assert set(captured["integrated_pose_planning_providers"]) == {"curobo-integrated"}
    assert recorder.finalize_kwargs["success_observed_step"] == 7


def test_behavior_run_rejects_seeded_init_mode(wired) -> None:
    _, program, tmp_path = wired
    with pytest.raises(ValueError, match="libero-pro"):
        run_module.run_program(
            benchmark="behavior",
            suite="turning_on_radio",
            task_id=0,
            seed=1,
            program_path=program,
            output_root=tmp_path / "outputs",
            max_steps=10,
            init_mode="seeded",
        )
