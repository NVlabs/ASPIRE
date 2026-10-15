from __future__ import annotations

import numpy as np
import pytest

from cap_harness.contracts import Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor


def _registry() -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry()
    calls: list[str] = []

    def get_task_context() -> dict[str, object]:
        calls.append("task")
        return {"language": "put cup away", "reward": 9.0, "success": True}

    def get_task_metadata() -> dict[str, object]:
        calls.append("metadata")
        return {"suite": "suite", "task_id": 1}

    def internal_secret() -> str:
        calls.append("secret")
        return "secret"

    registry.register(
        get_task_context,
        name="get_task_context",
        layer="shared",
        capability="task",
        public=True,
    )
    registry.register(
        get_task_metadata,
        name="libero.get_task_metadata",
        layer="extension",
        capability="metadata",
        public=True,
    )
    registry.register(
        internal_secret,
        name="internal_secret",
        layer="runtime",
        capability="private",
        public=False,
    )
    return registry, calls


def test_runtime_builds_hierarchical_allowlisted_namespace_and_trace() -> None:
    registry, calls = _registry()
    executor = ProgramExecutor(registry)

    result = executor.execute_program(
        "task = get_task_context()\nmeta = libero.get_task_metadata()\nresult = (task, meta)\n"
    )

    assert result.ok
    assert calls == ["task", "metadata"]
    assert [entry.tool for entry in result.calls] == [
        "get_task_context",
        "libero.get_task_metadata",
    ]
    task, metadata = result.result
    assert "reward" not in task
    assert "success" not in task
    assert metadata["suite"] == "suite"


def test_runtime_does_not_expose_private_tools_or_adapter_names() -> None:
    registry, calls = _registry()
    executor = ProgramExecutor(registry)

    for code in (
        "result = internal_secret()",
        "result = adapter",
        "result = reward",
        "result = success",
        "result = mujoco",
    ):
        result = executor.execute_program(code)
        assert not result.ok
    assert "secret" not in calls


def test_runtime_rejects_import_and_callable_introspection() -> None:
    registry, _ = _registry()
    executor = ProgramExecutor(registry)

    imported = executor.execute_program("import os\nresult = os.environ")
    introspected = executor.execute_program("result = get_task_context.__closure__")

    assert not imported.ok
    assert "Import" in imported.error.message
    assert not introspected.ok
    assert "private" in introspected.error.message


def test_runtime_allows_a_lambda_but_still_guards_its_body() -> None:
    """A lambda is a ``def`` with less syntax, and ``def`` was always allowed.

    Rejecting it bought no safety and forced sort keys into index-tuple shapes.
    The guards that do the work are on names and attributes, and those recurse
    into the lambda body -- which is what the second half pins.
    """
    registry, _ = _registry()
    executor = ProgramExecutor(registry)

    allowed = executor.execute_program(
        "result = sorted([(2, 'b'), (1, 'a')], key=lambda row: row[0])[0][1]"
    )
    guarded = executor.execute_program("result = (lambda: get_task_context.__closure__)()")

    assert allowed.ok, allowed.error
    assert allowed.result == "a"
    assert not guarded.ok
    assert "private" in guarded.error.message


def test_execute_program_does_not_implicitly_step_robot() -> None:
    registry = ToolRegistry()
    steps: list[int] = []

    def step(action: object) -> object:
        steps.append(1)
        return action

    registry.register(
        step,
        name="step",
        layer="shared",
        capability="control",
        public=True,
    )
    result = ProgramExecutor(registry).execute_program("result = 42")

    assert result.ok
    assert result.result == 42
    assert steps == []


def test_keyboard_interrupt_propagates_out_of_generated_tool_call() -> None:
    registry = ToolRegistry()

    def step(action: object) -> object:
        del action
        raise KeyboardInterrupt

    registry.register(
        step,
        name="step",
        layer="shared",
        capability="control",
        public=True,
    )

    with pytest.raises(KeyboardInterrupt):
        ProgramExecutor(registry).execute_program("step(None)")


def test_runtime_exposes_safe_contract_constructors() -> None:
    executor = ProgramExecutor(ToolRegistry())

    result = executor.execute_program(
        "pose = Pose(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0], frame='base')\n"
        "command = ArmCommand(mode='joint_position', target=[0, 0, 0, 0, 0, 0, 0])\n"
        "strategy = MotionStrategy(ik_solver='pyroki', trajectory_planner='curobo')\n"
        "result = (RobotAction(arms={'primary': command}), strategy)\n"
    )

    assert result.ok
    action, strategy = result.result
    assert tuple(action.arms) == ("primary",)
    assert strategy.ik_solver == "pyroki"
    assert strategy.trajectory_planner == "curobo"


def test_trace_summarizes_arrays_provider_frame_and_failure_code() -> None:
    registry = ToolRegistry()

    def segment_points(camera_name: str, points: np.ndarray) -> SegmentationSet:
        del camera_name, points
        return SegmentationSet(
            ok=True,
            segmentations=(
                Segmentation(
                    mask=np.ones((2, 2), dtype=bool),
                    label="object",
                    score=1.0,
                    camera_name="agentview",
                    frame="camera/agentview",
                ),
            ),
            diagnostics={"provider": "sam3"},
        )

    def generate_grasps(camera_name: str, mask: object) -> object:
        del camera_name, mask
        from cap_harness.contracts import GraspSet

        return GraspSet(
            ok=False,
            error=ApiError(code=ErrorCode.NO_GRASP, message="none"),
            diagnostics={"provider": "contact_graspnet"},
        )

    registry.register(
        segment_points,
        name="segment_points",
        layer="shared",
        capability="perception",
        public=True,
    )
    registry.register(
        generate_grasps,
        name="generate_grasps",
        layer="shared",
        capability="grasping",
        public=True,
    )
    executor = ProgramExecutor(registry)

    successful = executor.execute_program("result = segment_points('agentview', [[10, 20]])")
    assert successful.ok
    trace = successful.calls[0]
    assert trace.provider == "sam3"
    assert trace.frame == "camera/agentview"
    assert trace.outputs_summary["count"] == 1
    assert "mask" not in repr(trace.outputs_summary).lower()

    failed = executor.execute_program("result = generate_grasps('agentview', [[1]])")
    assert failed.ok
    trace = failed.calls[0]
    assert trace.ok is False
    assert trace.provider == "contact_graspnet"
    assert trace.failure_code == ErrorCode.NO_GRASP.value
    assert trace.inputs_summary["positional"][1]["type"] == "list"
