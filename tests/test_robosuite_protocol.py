from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from cap_harness.artifacts import atomic_json
from cap_harness.contracts import ArmCommand, RobotAction
from cap_harness.registry import SHARED_PUBLIC_TOOL_NAMES
from cap_harness.validation.evaluators.robosuite_bimanual import (
    BimanualProtocolTracker,
    NativeProtocolState,
    RobosuiteBimanualProtocolEvaluator,
)


def _action(
    *,
    primary_gripper: float | None = None,
    secondary_gripper: float | None = None,
    primary_target: float = 0.0,
    secondary_target: float = 0.0,
) -> RobotAction:
    arms = {}
    if primary_gripper is not None or primary_target != 0.0:
        arms["primary"] = ArmCommand("joint_position", np.full(7, primary_target), primary_gripper)
    if secondary_gripper is not None or secondary_target != 0.0:
        arms["secondary"] = ArmCommand(
            "joint_position", np.full(7, secondary_target), secondary_gripper
        )
    return RobotAction(arms=arms)


def _native(
    primary: bool,
    secondary: bool,
    clearance: float,
    success: bool = False,
    *,
    handle: tuple[float, float, float] | None = None,
    receiver_distance: float | None = None,
    receiver_offset: tuple[float, float, float] | None = None,
    long_axis: tuple[float, float, float] | None = None,
    pad_contacts: int | None = None,
    full_finger_contacts: int | None = None,
    hammer_finger_contacts: int | None = None,
    contact_pairs: tuple[tuple[str, str], ...] | None = None,
) -> NativeProtocolState:
    return NativeProtocolState(
        primary,
        secondary,
        clearance,
        success,
        None if handle is None else np.asarray(handle),
        receiver_distance,
        None if receiver_offset is None else np.asarray(receiver_offset),
        None if long_axis is None else np.asarray(long_axis),
        pad_contacts,
        full_finger_contacts,
        hammer_finger_contacts,
        contact_pairs,
    )


def test_lift_protocol_requires_same_tick_close_coupled_motion_and_dual_grasp() -> None:
    tracker = BimanualProtocolTracker("two_arm_lift", success_clearance_m=0.10)
    tracker.observe(
        1,
        _action(primary_gripper=0.0, secondary_gripper=0.0),
        _native(False, False, 0.0),
    )
    tracker.observe(
        2,
        _action(
            primary_gripper=0.0,
            secondary_gripper=0.0,
            primary_target=0.1,
            secondary_target=-0.1,
        ),
        _native(True, True, 0.05),
    )
    tracker.observe(
        3,
        _action(
            primary_gripper=0.0,
            secondary_gripper=0.0,
            primary_target=0.2,
            secondary_target=-0.2,
        ),
        _native(True, True, 0.11, True),
    )

    evidence = tracker.evidence(final_native_success=True)

    assert evidence["protocol_success"] is True
    assert evidence["witness_steps"] == {
        "coupled_motion": 2,
        "dual_close": 1,
        "dual_grasp": 2,
        "lifted_with_dual_grasp": 3,
    }


def test_lift_protocol_rejects_sequential_closes_even_with_native_success() -> None:
    tracker = BimanualProtocolTracker("two_arm_lift", success_clearance_m=0.10)
    tracker.observe(1, _action(primary_gripper=0.0), _native(True, False, 0.0))
    tracker.observe(2, _action(secondary_gripper=0.0), _native(True, True, 0.0))
    tracker.observe(
        3,
        _action(primary_target=0.2, secondary_target=-0.2),
        _native(True, True, 0.11, True),
    )

    evidence = tracker.evidence(final_native_success=True)

    assert evidence["protocol_success"] is False
    assert evidence["checks"]["same_tick_dual_close"] is False


def test_lift_protocol_rejects_single_arm_motion_after_dual_close() -> None:
    tracker = BimanualProtocolTracker("two_arm_lift", success_clearance_m=0.10)
    tracker.observe(
        1,
        _action(primary_gripper=0.0, secondary_gripper=0.0),
        _native(False, False, 0.0),
    )
    tracker.observe(
        2,
        _action(
            primary_gripper=0.0,
            secondary_gripper=0.0,
            primary_target=0.2,
        ),
        _native(True, True, 0.11, True),
    )

    evidence = tracker.evidence(final_native_success=True)

    assert evidence["protocol_success"] is False
    assert evidence["checks"]["coupled_motion_after_close"] is False


def test_handover_protocol_requires_overlap_before_giver_release() -> None:
    tracker = BimanualProtocolTracker("two_arm_handover", success_clearance_m=0.10)
    tracker.observe(1, _action(primary_gripper=0.0), _native(True, False, 0.20))
    tracker.observe(2, _action(secondary_gripper=0.0), _native(True, True, 0.20))
    tracker.observe(3, _action(primary_gripper=1.0), _native(False, True, 0.20, True))

    evidence = tracker.evidence(final_native_success=True)

    assert evidence["protocol_success"] is True
    assert evidence["witness_steps"] == {
        "giver_grasp_elevated": 1,
        "giver_open_command": 3,
        "overlap_grasp": 2,
        "receiver_close_command": 2,
        "receiver_only_elevated": 3,
    }


def test_handover_protocol_rejects_receiver_acquisition_after_release() -> None:
    tracker = BimanualProtocolTracker("two_arm_handover", success_clearance_m=0.10)
    tracker.observe(1, _action(primary_gripper=0.0), _native(True, False, 0.20))
    tracker.observe(2, _action(primary_gripper=1.0), _native(False, False, 0.20))
    tracker.observe(3, _action(secondary_gripper=0.0), _native(False, True, 0.20, True))

    evidence = tracker.evidence(final_native_success=True)

    assert evidence["protocol_success"] is False
    assert evidence["checks"]["overlap_before_release"] is False
    assert evidence["checks"]["giver_open_after_receiver_grasp"] is False


def test_handover_stage_evidence_is_cumulative_and_evaluator_only() -> None:
    tracker = BimanualProtocolTracker("two_arm_handover", success_clearance_m=0.10)
    tracker.observe(
        1,
        _action(primary_gripper=0.0),
        _native(
            True,
            False,
            0.20,
            handle=(0.40, 0.0, 0.20),
            receiver_distance=0.50,
            long_axis=(0.0, 1.0, 0.0),
        ),
    )
    tracker.observe(
        2,
        _action(primary_target=0.1),
        _native(
            True,
            False,
            0.20,
            handle=(0.75, 0.0, 0.20),
            receiver_distance=0.30,
            long_axis=(1.0, 0.0, 0.0),
        ),
    )
    tracker.observe(
        3,
        _action(secondary_target=0.1),
        _native(
            True,
            False,
            0.20,
            handle=(0.75, 0.0, 0.20),
            receiver_distance=0.08,
            long_axis=(1.0, 0.0, 0.0),
        ),
    )
    tracker.observe(
        4,
        _action(secondary_gripper=0.0),
        _native(
            True,
            True,
            0.20,
            handle=(0.75, 0.0, 0.20),
            receiver_distance=0.02,
            receiver_offset=(-0.02, 0.0, 0.0),
            long_axis=(1.0, 0.0, 0.0),
            pad_contacts=2,
            full_finger_contacts=2,
            hammer_finger_contacts=2,
            contact_pairs=(("receiver_pad", "hammer_handle"),),
        ),
    )
    tracker.observe(
        5,
        _action(primary_gripper=1.0),
        _native(
            False,
            True,
            0.20,
            True,
            handle=(0.75, 0.0, 0.20),
            receiver_distance=0.02,
            long_axis=(1.0, 0.0, 0.0),
        ),
    )

    evidence = tracker.evidence(final_native_success=True)

    assert {name: stage["passed"] for name, stage in evidence["stages"].items()} == {
        "pickup": True,
        "presentation": True,
        "receiver_approach": True,
        "receiver_close": True,
        "giver_release": True,
        "native_success": True,
        "protocol_witnesses": True,
    }
    assert evidence["stages"]["receiver_close"]["command_step"] == 4
    assert evidence["stage_metrics"]["min_receiver_to_handle_m"] == 0.02
    assert evidence["stage_metrics"]["receiver_to_handle_at_pickup_m"] == 0.50
    assert evidence["stage_metrics"]["max_object_long_axis_x_alignment"] == 1.0
    assert evidence["stage_metrics"]["max_secondary_handle_finger_contacts"] == 2
    assert evidence["stage_metrics"]["max_secondary_handle_full_finger_contacts"] == 2
    assert evidence["stage_metrics"]["max_secondary_hammer_full_finger_contacts"] == 2
    assert evidence["stage_metrics"]["closest_receiver_to_handle_xyz"] == (
        -0.02,
        0.0,
        0.0,
    )
    assert evidence["stage_metrics"]["final_receiver_to_handle_xyz"] == (
        -0.02,
        0.0,
        0.0,
    )
    assert evidence["stage_metrics"]["secondary_hammer_contact_pairs"] == (
        ("receiver_pad", "hammer_handle"),
    )


def test_handover_presentation_can_be_witnessed_by_relative_receiver_progress() -> None:
    tracker = BimanualProtocolTracker("two_arm_handover", success_clearance_m=0.10)
    tracker.observe(
        1,
        _action(primary_gripper=0.0),
        _native(
            True,
            False,
            0.20,
            handle=(0.30, 0.0, 0.20),
            receiver_distance=0.60,
        ),
    )
    tracker.observe(
        2,
        _action(primary_target=0.1),
        _native(
            True,
            False,
            0.20,
            handle=(0.40, 0.0, 0.20),
            receiver_distance=0.44,
        ),
    )

    evidence = tracker.evidence(final_native_success=False)

    assert evidence["stages"]["presentation"]["passed"] is True
    assert evidence["stages"]["presentation"]["witness_step"] == 2


def test_handover_stage_close_command_does_not_claim_native_acquisition() -> None:
    tracker = BimanualProtocolTracker("two_arm_handover", success_clearance_m=0.10)
    tracker.observe(
        1,
        _action(primary_gripper=0.0),
        _native(
            True,
            False,
            0.20,
            handle=(0.75, 0.0, 0.20),
            receiver_distance=0.20,
            long_axis=(1.0, 0.0, 0.0),
        ),
    )
    tracker.observe(
        2,
        _action(secondary_gripper=0.0),
        _native(
            True,
            False,
            0.20,
            handle=(0.75, 0.0, 0.20),
            receiver_distance=0.05,
            long_axis=(1.0, 0.0, 0.0),
        ),
    )

    evidence = tracker.evidence(final_native_success=False)

    assert evidence["stages"]["receiver_approach"]["passed"] is True
    assert evidence["stages"]["receiver_close"]["passed"] is False
    assert evidence["stages"]["receiver_close"]["command_step"] == 2
    assert evidence["stages"]["receiver_close"]["native_overlap"] is False


def test_handover_native_geometry_uses_hammer_local_z_in_public_base() -> None:
    primary_base_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    primary_base_position = np.array([1.0, 2.0, 3.0])
    hammer_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    data = SimpleNamespace(
        body_xmat=np.array([hammer_rotation.reshape(-1)]),
        contact=[],
        ncon=0,
        get_body_xmat=lambda name: primary_base_rotation,
        get_body_xpos=lambda name: primary_base_position,
    )
    env = SimpleNamespace(
        height_threshold=0.10,
        sim=SimpleNamespace(data=data),
        hammer_body_id=0,
        _get_task_info=lambda: (True, False, 0.30, 0.10),
        _check_success=lambda: False,
        _handle_xpos=np.array([2.0, 2.0, 3.0]),
        _gripper_1_to_handle=np.array([0.2, 0.0, 0.0]),
        robots=[
            SimpleNamespace(gripper=None),
            SimpleNamespace(
                gripper=SimpleNamespace(
                    contact_geoms=["left", "right"],
                    important_geoms={
                        "left_finger": ["left"],
                        "right_finger": ["right"],
                        "left_fingerpad": ["left_pad"],
                        "right_fingerpad": ["right_pad"],
                    },
                )
            ),
        ],
        hammer=SimpleNamespace(
            handle_geoms=["handle"],
            contact_geoms=["handle", "head"],
        ),
        check_contact=lambda finger, hammer: finger in (["left"], ["left_pad"]),
    )

    native = RobosuiteBimanualProtocolEvaluator(
        "two_arm_handover",
        env,
    )._native_state()

    np.testing.assert_allclose(native.handle_position_xyz, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(native.receiver_to_handle_xyz, [0.0, -0.2, 0.0])
    np.testing.assert_allclose(native.object_long_axis_xyz, [0.0, -1.0, 0.0])
    assert native.secondary_handle_finger_contacts == 1
    assert native.secondary_handle_full_finger_contacts == 1
    assert native.secondary_hammer_full_finger_contacts == 1
    assert native.secondary_hammer_contact_pairs == ()


def test_evaluator_only_names_are_absent_from_public_registry() -> None:
    assert "set_grippers" in SHARED_PUBLIC_TOOL_NAMES
    assert all(
        "protocol" not in name and "native" not in name and "evaluator" not in name
        for name in SHARED_PUBLIC_TOOL_NAMES
    )


def test_immutable_protocol_evidence_serializes_as_plain_json(tmp_path) -> None:
    tracker = BimanualProtocolTracker("two_arm_lift", success_clearance_m=0.10)
    path = tmp_path / "protocol.json"

    atomic_json(path, tracker.evidence(final_native_success=False))

    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["task"] == "two_arm_lift"
    assert evidence["protocol_success"] is False
    assert evidence["checks"]["native_task_success"] is False
