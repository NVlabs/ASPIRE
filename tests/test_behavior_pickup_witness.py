from __future__ import annotations

from types import SimpleNamespace

from behavior_fakes import FakeOmniGibsonEnv
import numpy as np
import pytest

from cap_harness.behavior.config import build_environment_config
from cap_harness.behavior.registry import BehaviorTaskRegistry
from cap_harness.contracts import ArmCommand, RobotAction
from cap_harness.validation.evaluators import EVALUATOR_ALLOWLIST
from cap_harness.validation.evaluators.behavior_pickup import (
    DEBOUNCE_STEPS,
    LIFT_THRESHOLD_M,
    BehaviorPickupWitness,
)


def _env() -> FakeOmniGibsonEnv:
    metadata = BehaviorTaskRegistry().resolve("turning_on_radio")
    config = build_environment_config(metadata, camera_width=32, camera_height=32, horizon=50)
    return FakeOmniGibsonEnv(config)


def _step(witness: BehaviorPickupWitness, ok: bool = True) -> None:
    action = RobotAction(
        {"primary": ArmCommand("joint_position", np.zeros(7), embodiment="behavior")}
    )
    witness.after_step(action, SimpleNamespace(ok=ok))


def test_rule_constants_match_the_aspire_comparable_decision() -> None:
    assert LIFT_THRESHOLD_M == 0.005
    assert DEBOUNCE_STEPS == 0
    assert "behavior_pickup" in EVALUATOR_ALLOWLIST


def test_baseline_is_the_height_after_the_instance_load() -> None:
    env = _env()
    env.target.position[2] = 0.75  # instance load moved the target before the witness exists
    witness = BehaviorPickupWitness("turning_on_radio", env, target_scope="radio_receiver.n.01_1")
    _step(witness)
    assert witness.clearance_m() == pytest.approx(0.0)
    assert not witness.success


def test_lift_without_a_grasp_and_grasp_without_a_lift_do_not_count() -> None:
    env = _env()
    witness = BehaviorPickupWitness("turning_on_radio", env, target_scope="radio_receiver.n.01_1")
    env.target.position[2] += 0.05
    _step(witness)
    assert not witness.success
    env.target.position[2] -= 0.05
    env.robots[0].grasping["left"] = env.target
    _step(witness)
    assert not witness.success
    evidence = witness.evidence()
    assert evidence["checks"] == {"held": True, "lifted": True, "held_and_lifted": False}


def test_success_latches_on_the_first_step_both_hold_and_survives_release() -> None:
    env = _env()
    witness = BehaviorPickupWitness("turning_on_radio", env, target_scope="radio_receiver.n.01_1")
    _step(witness)
    env.robots[0].grasping["right"] = env.target
    env.target.position[2] += 0.0049
    _step(witness)
    assert not witness.success
    env.target.position[2] += 0.0002
    _step(witness)
    assert witness.success and witness.success_step == 3
    env.robots[0].grasping["right"] = None
    env.target.position[2] -= 0.2
    _step(witness)
    assert witness.success
    evidence = witness.evidence(final_native_success=False)
    assert evidence["holding_arm"] == "right"
    assert evidence["success_step"] == 3
    assert evidence["max_clearance_m"] == pytest.approx(0.0051)
    assert evidence["protocol_success"] is True


def test_failed_steps_are_ignored_and_unknown_targets_are_rejected() -> None:
    env = _env()
    witness = BehaviorPickupWitness("turning_on_radio", env, target_scope="radio_receiver.n.01_1")
    env.robots[0].grasping["left"] = env.target
    env.target.position[2] += 1.0
    _step(witness, ok=False)
    assert not witness.success
    with pytest.raises(KeyError):
        BehaviorPickupWitness("turning_on_radio", env, target_scope="can__of__soda.n.01_3")
    with pytest.raises(ValueError):
        BehaviorPickupWitness(
            "turning_on_radio", env, target_scope="radio_receiver.n.01_1", lift_threshold_m=0.0
        )
