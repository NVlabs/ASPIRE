from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

import cap_harness.artifacts as artifacts_module
from cap_harness.artifacts import (
    TERMINATION_REASONS,
    RunRecorder,
    Serializer,
    StepLimitReached,
    redact,
)
from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    Observation,
    PointCloud,
    Pose,
    RobotAction,
    RobotState,
    Segmentation,
    SegmentationSet,
    StepResult,
    TaskContext,
)


def _observation(timestamp: float = 0.0) -> Observation:
    pose = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), "robot_base")
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=pose,
        gripper_positions=1.0,
        base_frame="robot_base",
        timestamp_s=timestamp,
    )
    cameras = {
        name: CameraObservation(
            rgb=np.full((16, 16, 3), index * 30, dtype=np.uint8),
            depth_m=np.ones((16, 16)),
            intrinsics=np.eye(3),
            frame=f"camera/{name}",
            camera_pose=pose,
            timestamp_s=timestamp,
        )
        for index, name in enumerate(("agentview", "robot0_eye_in_hand"), 1)
    }
    task = TaskContext("suite", 0, "Easy Task", "do the task", "object")
    return Observation(cameras=cameras, robot_state=state, task_context=task, timestamp_s=timestamp)


def _action() -> RobotAction:
    return RobotAction(arms={"primary": ArmCommand("joint_position", np.zeros(7))})


def test_recorded_run_has_compact_hierarchy_exact_payloads_and_video(tmp_path) -> None:
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        task_name="Easy Task",
        seed=1,
        max_steps=2,
    )
    policy = tmp_path / "policy.py"
    policy.write_bytes(b"result = 1\r\n")
    recorder.save_program(policy)
    recorder.on_reset(_observation(), {"language": "do the task"})
    recorder.keyframe("before-step")
    recorder.before_step(_action())
    recorder.after_step(_action(), StepResult(ok=True, observation=_observation(0.05), reward=0.5))
    recorder.keyframe("after-step")
    with recorder.span(
        "localize_object", category="program", inputs={"mask": np.ones((2, 2), bool)}
    ) as span:
        span.output({"points": np.ones((3, 3))})
    recorder.finalize(
        program_ok=True,
        task_success=False,
        termination_reason="program_completed",
        program_result={"ok": True},
        cleanup_errors=("adapter close: RuntimeError: cleanup failed",),
    )

    assert recorder.root.relative_to(tmp_path).parts[:4] == (
        "libero-pro",
        "libero_object_swap",
        "00-easy-task",
        "0001",
    )
    assert json.loads((recorder.root / "run.json").read_text())["status"] == "finalized"
    assert json.loads((recorder.root / "outcome.json").read_text())["cumulative_reward"] == 0.5
    assert json.loads((recorder.root / "outcome.json").read_text())["finalization_errors"] == [
        "adapter close: RuntimeError: cleanup failed"
    ]
    assert len(list((recorder.root / "media/keyframes").rglob("*-depth.npy"))) == 1
    assert len(list((recorder.root / "media/keyframes").glob("*"))) == 2
    assert not (recorder.root / "payloads").exists()
    assert len(json.loads((recorder.root / "manifest.json").read_text())["files"]) > 10
    assert (recorder.root / "source/program.py").read_bytes() == b"result = 1\r\n"
    for camera in ("agentview", "robot0_eye_in_hand"):
        frames = imageio.mimread(recorder.root / f"media/videos/{camera}.mp4")
        assert len(frames) == 2


def test_git_provenance_is_anchored_to_harness_checkout(monkeypatch) -> None:
    calls: list[Path] = []

    def run(command, **kwargs):
        calls.append(kwargs["cwd"])
        stdout = "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(artifacts_module.subprocess, "run", run)

    assert RunRecorder._git() == {"commit": "abc123", "dirty": False}
    assert calls == [
        Path(artifacts_module.__file__).resolve().parents[2],
        Path(artifacts_module.__file__).resolve().parents[2],
    ]


def test_step_limit_and_secret_redaction(tmp_path) -> None:
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="libero-pro",
        suite="suite",
        task_id=0,
        task_name="task",
        seed=1,
        max_steps=1,
    )
    recorder.on_reset(_observation(), {})
    recorder.before_step(_action())
    recorder.after_step(_action(), StepResult(ok=True, observation=_observation(0.05)))
    with pytest.raises(StepLimitReached):
        recorder.before_step(_action())
    recorder.finalize(
        program_ok=False,
        task_success=False,
        termination_reason="step_limit",
        program_result=None,
    )

    assert recorder.terminal_reason == "step_limit"
    assert redact({"Authorization": "Bearer abc", "text": "hf_abcdefghijklmnop"}) == {
        "Authorization": "[REDACTED]",
        "text": "[REDACTED]",
    }
    assert set(TERMINATION_REASONS) == {
        "program_completed",
        "task_succeeded",
        "environment_terminated",
        "environment_truncated",
        "step_limit",
        "program_error",
        "provider_error",
        "harness_error",
        "user_interrupt",
    }


def test_point_cloud_payloads_are_summarized_not_saved(tmp_path) -> None:
    cloud = PointCloud(points=np.ones((100, 3)), colors=np.zeros((100, 3)), frame="robot_base")
    encoded = Serializer(tmp_path).encode(cloud, tmp_path / "unused")

    assert encoded == {
        "$type": "PointCloud",
        "frame": "robot_base",
        "point_count": 100,
        "has_colors": True,
    }
    assert not (tmp_path / "payloads").exists()


def test_protocol_evidence_is_separate_and_manifested(tmp_path) -> None:
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="robosuite",
        suite="two_arm_lift",
        task_id=0,
        task_name="two_arm_lift",
        seed=1,
        max_steps=1,
        capture_videos=False,
    )
    recorder.on_reset(_observation(), {})
    recorder.save_protocol_evidence(
        {
            "schema_version": 1,
            "protocol_success": True,
            "checks": {"same_tick_dual_close": True},
            "witness_steps": {"dual_close": 1},
        }
    )
    recorder.finalize(
        program_ok=True,
        task_success=True,
        protocol_success=True,
        termination_reason="task_succeeded",
        program_result={"ok": True},
        success_observed_step=17,
    )

    protocol = json.loads((recorder.root / "evaluation/protocol.json").read_text())
    outcome = json.loads((recorder.root / "outcome.json").read_text())
    manifest = json.loads((recorder.root / "manifest.json").read_text())
    program_result = json.loads((recorder.root / "source/program-result.json").read_text())

    assert protocol["protocol_success"] is True
    assert outcome["protocol_success"] is True
    # task_succeeded is assigned from the latched success; the outcome must still
    # say whether the environment itself ever ended, and when success was seen.
    assert outcome["environment_terminated"] is False
    assert outcome["environment_truncated"] is False
    assert outcome["success_observed_step"] == 17
    assert program_result == {"ok": True}
    assert "protocol" not in (recorder.root / "episode/steps.jsonl").read_text()
    assert any(item["path"] == "evaluation/protocol.json" for item in manifest["files"])


def test_raw_masks_are_summarized_not_saved(tmp_path) -> None:
    encoded = Serializer(tmp_path).encode(np.eye(800, dtype=bool), tmp_path / "unused")

    assert encoded == {"$type": "MaskSummary", "shape": [800, 800], "true_pixels": 800}
    assert not (tmp_path / "payloads").exists()


def test_segmentation_trace_keeps_boxes_and_annotated_overlay_without_masks(tmp_path) -> None:
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="libero-pro",
        suite="suite",
        task_id=0,
        task_name="task",
        seed=1,
        max_steps=1,
    )
    recorder.on_reset(_observation(), {})
    segmentations = SegmentationSet(
        ok=True,
        segmentations=tuple(
            Segmentation(
                mask=np.eye(16, dtype=bool),
                label=f"candidate-{index}",
                score=score,
                camera_name="agentview",
                frame="camera/agentview",
                box_xyxy=np.array([1, 2, 12, 13]),
            )
            for index, score in enumerate((0.9, 0.8, 0.7, 0.6, 0.5, 0.4))
        ),
    )
    with recorder.span(
        "sam3.segment_text",
        category="provider",
        inputs={"camera": _observation().cameras["agentview"]},
    ) as span:
        span.output(segmentations)
    recorder.finalize(
        program_ok=True,
        task_success=False,
        termination_reason="program_completed",
        program_result=None,
    )

    output = json.loads(next((recorder.root / "trace/calls").glob("*/output.json")).read_text())
    assert len(output["segmentations"]) == 5
    assert output["segmentations"][0]["box_xyxy"] == [1.0, 2.0, 12.0, 13.0]
    assert "mask" not in output["segmentations"][0]
    assert len(list((recorder.root / "media/overlays").glob("*.png"))) == 1
    overlay_metadata = json.loads(
        next((recorder.root / "media/overlays").glob("*.json")).read_text()
    )
    assert [item["label"] for item in overlay_metadata["candidates"]] == [
        f"candidate-{index}" for index in range(5)
    ]
    assert overlay_metadata["candidates"][0]["score"] == 0.9
    assert overlay_metadata["candidates"][0]["mask_pixels"] == 16
    assert not (recorder.root / "payloads").exists()


def test_timing_buckets_are_disjoint_and_nesting_never_double_counts() -> None:
    """A bucket is charged only what nothing inside it already claimed.

    Without this, a region that encloses instrumented work (reset writing a
    keyframe, finalize flushing a video) reports its own cost plus its
    children's, the totals sum past the wall clock, and the profile points at
    whichever bucket happens to sit outermost.
    """
    timing = artifacts_module.RunTiming()
    with timing.measure("outer"):
        with timing.measure("inner"):
            time.sleep(0.02)
        time.sleep(0.02)

    assert timing.buckets["inner"] >= 0.02
    assert timing.buckets["outer"] >= 0.015
    snapshot = timing.snapshot(wall_s=timing.buckets["outer"] + timing.buckets["inner"] + 0.05)
    assert snapshot["attributed_s"] == pytest.approx(
        timing.buckets["outer"] + timing.buckets["inner"], abs=1e-6
    )
    assert snapshot["unattributed_s"] == pytest.approx(0.05, abs=1e-3)


def test_outcome_attributes_the_wall_clock_to_the_work_that_spent_it(tmp_path) -> None:
    """The pipeline is optimized against this block, so it must name real costs.

    ``adapter_step`` is the gap between asking the adapter to step and its
    answer -- the simulator step and its observation normalization -- and it is
    charged separately from the recorder's own write of that same step.
    """
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="libero-pro",
        suite="libero_object_swap",
        task_id=0,
        task_name="Easy Task",
        seed=1,
        max_steps=4,
    )
    recorder.on_reset(_observation(), {"language": "do the task"})
    for index in range(2):
        recorder.before_step(_action())
        time.sleep(0.01)  # stand in for the simulator
        recorder.after_step(
            _action(), StepResult(ok=True, observation=_observation(0.05 * index), reward=0.0)
        )
    recorder.finalize(
        program_ok=True,
        task_success=True,
        termination_reason="task_succeeded",
        program_result=None,
    )

    timing = json.loads((recorder.root / "outcome.json").read_text())["timing"]
    buckets = timing["buckets"]
    assert buckets["adapter_step"]["calls"] == 2
    assert buckets["adapter_step"]["seconds"] >= 0.02
    # The recorder's own per-step work is separate from the simulator's.
    assert "record_step" in buckets and "video_encode" in buckets
    assert buckets["record_step"]["seconds"] < buckets["adapter_step"]["seconds"]
    assert timing["attributed_s"] <= timing["wall_s"] + 1e-6
    assert sum(bucket["seconds"] for bucket in buckets.values()) == pytest.approx(
        timing["attributed_s"], abs=1e-3
    )
