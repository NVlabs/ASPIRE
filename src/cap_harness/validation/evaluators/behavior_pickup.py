"""Host-side pickup witness for the BEHAVIOR-1K R1 Pro tasks.

Success is the ASPIRE-comparable rule: the robot's grasp holds the target object AND the object
sits at least ``LIFT_THRESHOLD_M`` above the height it had when the task instance was loaded.
Unlike the reference, the baseline is captured after the instance load, and success is latched
on the first step both conditions hold (``DEBOUNCE_STEPS`` consecutive steps, zero today).

Only booleans, step indices and clearances leave this module; generated programs never see it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cap_harness.contracts import RobotAction, StepResult

LIFT_THRESHOLD_M = 0.005
"""Minimum rise above the instance-start height, ASPIRE's 5 mm."""
DEBOUNCE_STEPS = 0
"""Extra consecutive steps both predicates must hold before success latches (ASPIRE: none)."""


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _object_height(entity: Any) -> float:
    position, _ = entity.get_position_orientation()
    if hasattr(position, "detach"):
        position = position.detach().cpu()
    return float(np.asarray(position, dtype=np.float64).reshape(-1)[2])


def _grasp_state_is_true(state: Any) -> bool:
    """Interpret OmniGibson's ``IsGraspingState`` without importing it."""
    if state is True:
        return True
    if isinstance(state, int | float | np.integer) and not isinstance(state, bool):
        return int(state) == 1
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name.upper() == "TRUE"
    return str(state).upper().endswith("TRUE")


class BehaviorPickupWitness:
    """Latch ``held and lifted`` for the task's target object on the first episode only."""

    def __init__(
        self,
        task_name: str,
        native_env: Any,
        *,
        target_scope: str,
        lift_threshold_m: float = LIFT_THRESHOLD_M,
        debounce_steps: int = DEBOUNCE_STEPS,
    ) -> None:
        self.task_name = str(task_name)
        self.target_scope = str(target_scope)
        self.lift_threshold_m = float(lift_threshold_m)
        self.debounce_steps = int(debounce_steps)
        if self.lift_threshold_m <= 0.0 or self.debounce_steps < 0:
            raise ValueError("lift threshold must be positive and debounce non-negative")
        self._env = native_env
        self._robot = native_env.robots[0]
        scope = native_env.task.object_scope
        if self.target_scope not in scope or scope[self.target_scope] is None:
            raise KeyError(f"target {self.target_scope!r} is not in the task object scope")
        self._target = scope[self.target_scope]
        self._baseline_z = _object_height(self._target)
        self._step_index = 0
        self._consecutive = 0
        self._success = False
        self._success_step: int | None = None
        self._first_held_step: int | None = None
        self._first_lifted_step: int | None = None
        self._max_clearance_m = 0.0
        self._holding_arm: str | None = None

    @property
    def success(self) -> bool:
        return self._success

    @property
    def success_step(self) -> int | None:
        return self._success_step

    def held_by(self) -> str | None:
        """Return the arm whose grasp holds the target, or None."""
        for arm in self._robot.arm_names:
            try:
                state = self._robot.is_grasping(arm=arm, candidate_obj=self._target)
            except Exception:  # pragma: no cover - defensive against native API drift
                state = False
            if _grasp_state_is_true(state):
                return str(arm)
        return None

    def clearance_m(self) -> float:
        return _object_height(self._target) - self._baseline_z

    def after_step(self, action: RobotAction, result: StepResult) -> None:
        if not result.ok:
            return
        self._step_index += 1
        clearance = self.clearance_m()
        self._max_clearance_m = max(self._max_clearance_m, clearance)
        arm = self.held_by()
        lifted = clearance >= self.lift_threshold_m
        if arm is not None and self._first_held_step is None:
            self._first_held_step = self._step_index
        if lifted and self._first_lifted_step is None:
            self._first_lifted_step = self._step_index
        if self._success:
            return
        if arm is not None and lifted:
            self._consecutive += 1
            if self._consecutive > self.debounce_steps:
                self._success = True
                self._success_step = self._step_index
                self._holding_arm = arm
        else:
            self._consecutive = 0

    def evidence(self, *, final_native_success: bool | None = None) -> Mapping[str, object]:
        return {
            "schema_version": 1,
            "task_name": self.task_name,
            "rule": {
                "lift_threshold_m": self.lift_threshold_m,
                "debounce_steps": self.debounce_steps,
                "baseline": "instance-start height, captured after the task instance was loaded",
            },
            "checks": {
                "held": self._first_held_step is not None,
                "lifted": self._first_lifted_step is not None,
                "held_and_lifted": self._success,
            },
            "first_held_step": self._first_held_step,
            "first_lifted_step": self._first_lifted_step,
            "success_step": self._success_step,
            "holding_arm": self._holding_arm,
            "max_clearance_m": float(self._max_clearance_m),
            "steps_observed": self._step_index,
            "final_native_success": final_native_success,
            "protocol_success": self._success,
        }


__all__ = ["DEBOUNCE_STEPS", "LIFT_THRESHOLD_M", "BehaviorPickupWitness"]
