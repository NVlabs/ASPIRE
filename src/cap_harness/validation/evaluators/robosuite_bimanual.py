"""Evaluator-only bimanual protocol witnesses for Robosuite acceptance runs.

This module intentionally consumes privileged simulator predicates. Nothing in
it is registered with generated programs, serialized into public call traces,
or returned through :class:`cap_harness.api.CapApi`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import RobotAction, StepResult

_CLOSED = 0.05
_OPEN = 0.95
_MOTION_EPSILON_RAD = 1e-4
_HANDOVER_PRESENTATION_X_RANGE_M = (0.55, 1.10)
_HANDOVER_PRESENTATION_ABS_Y_M = 0.25
_HANDOVER_PRESENTATION_PROGRESS_M = 0.15
_HANDOVER_RECEIVER_APPROACH_M = 0.12


@dataclass(frozen=True, slots=True)
class NativeProtocolState:
    """Minimal privileged state consumed by the evaluator state machine."""

    primary_grasp: bool
    secondary_grasp: bool
    object_clearance_m: float
    native_success: bool
    handle_position_xyz: np.ndarray | None = None
    receiver_to_handle_m: float | None = None
    receiver_to_handle_xyz: np.ndarray | None = None
    object_long_axis_xyz: np.ndarray | None = None
    secondary_handle_finger_contacts: int | None = None
    secondary_handle_full_finger_contacts: int | None = None
    secondary_hammer_full_finger_contacts: int | None = None
    secondary_hammer_contact_pairs: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if type(self.primary_grasp) is not bool or type(self.secondary_grasp) is not bool:
            raise ValueError("grasp predicates must be bools")
        if type(self.native_success) is not bool:
            raise ValueError("native_success must be a bool")
        if not np.isfinite(self.object_clearance_m):
            raise ValueError("object_clearance_m must be finite")
        if self.handle_position_xyz is not None:
            handle = np.asarray(self.handle_position_xyz, dtype=np.float64)
            if handle.shape != (3,) or not bool(np.all(np.isfinite(handle))):
                raise ValueError("handle_position_xyz must be a finite three-vector or None")
            object.__setattr__(self, "handle_position_xyz", np.array(handle, copy=True))
        if self.receiver_to_handle_m is not None:
            distance = float(self.receiver_to_handle_m)
            if not np.isfinite(distance) or distance < 0.0:
                raise ValueError("receiver_to_handle_m must be finite and non-negative or None")
            object.__setattr__(self, "receiver_to_handle_m", distance)
        if self.receiver_to_handle_xyz is not None:
            offset = np.asarray(self.receiver_to_handle_xyz, dtype=np.float64)
            if offset.shape != (3,) or not bool(np.all(np.isfinite(offset))):
                raise ValueError("receiver_to_handle_xyz must be a finite three-vector or None")
            object.__setattr__(self, "receiver_to_handle_xyz", np.array(offset, copy=True))
        if self.object_long_axis_xyz is not None:
            axis = np.asarray(self.object_long_axis_xyz, dtype=np.float64)
            if axis.shape != (3,) or not bool(np.all(np.isfinite(axis))):
                raise ValueError("object_long_axis_xyz must be a finite three-vector or None")
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                raise ValueError("object_long_axis_xyz must be non-zero or None")
            object.__setattr__(self, "object_long_axis_xyz", axis / norm)
        if self.secondary_handle_finger_contacts is not None and (
            isinstance(self.secondary_handle_finger_contacts, bool)
            or self.secondary_handle_finger_contacts not in {0, 1, 2}
        ):
            raise ValueError("secondary_handle_finger_contacts must be 0, 1, 2, or None")
        for name in (
            "secondary_handle_full_finger_contacts",
            "secondary_hammer_full_finger_contacts",
        ):
            count = getattr(self, name)
            if count is not None and (isinstance(count, bool) or count not in {0, 1, 2}):
                raise ValueError(f"{name} must be 0, 1, 2, or None")
        if self.secondary_hammer_contact_pairs is not None:
            pairs = tuple(self.secondary_hammer_contact_pairs)
            if any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(not isinstance(geom, str) or not geom for geom in pair)
                for pair in pairs
            ):
                raise ValueError(
                    "secondary_hammer_contact_pairs must contain non-empty string pairs"
                )
            object.__setattr__(self, "secondary_hammer_contact_pairs", pairs)


class BimanualProtocolTracker:
    """Pure state machine producing ordered protocol witnesses."""

    def __init__(self, task_name: str, *, success_clearance_m: float) -> None:
        if task_name not in {"two_arm_lift", "two_arm_handover"}:
            raise ValueError("unsupported bimanual protocol task")
        if not np.isfinite(success_clearance_m) or success_clearance_m < 0.0:
            raise ValueError("success_clearance_m must be finite and non-negative")
        self.task_name = task_name
        self.success_clearance_m = float(success_clearance_m)
        self.steps_observed = 0
        self.max_clearance_m = float("-inf")
        self.last_native_success = False
        self.witnesses: dict[str, int] = {}
        self._dual_close_targets: dict[str, np.ndarray] | None = None
        self.min_receiver_to_handle_m = float("inf")
        self.receiver_to_handle_at_pickup_m: float | None = None
        self.max_object_long_axis_x_alignment = 0.0
        self.max_secondary_handle_finger_contacts = 0
        self.max_secondary_handle_full_finger_contacts = 0
        self.max_secondary_hammer_full_finger_contacts = 0
        self.closest_receiver_to_handle_xyz: np.ndarray | None = None
        self.final_receiver_to_handle_xyz: np.ndarray | None = None
        self.secondary_hammer_contact_pairs: set[tuple[str, str]] = set()
        self.errors: list[str] = []

    def observe(
        self,
        step_index: int,
        action: RobotAction,
        native: NativeProtocolState,
    ) -> None:
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index <= 0:
            raise ValueError("step_index must be a positive integer")
        if not isinstance(action, RobotAction) or not isinstance(native, NativeProtocolState):
            raise ValueError("action and native state must use typed protocol values")
        self.steps_observed = max(self.steps_observed, step_index)
        self.max_clearance_m = max(self.max_clearance_m, native.object_clearance_m)
        self.last_native_success = native.native_success
        if native.receiver_to_handle_m is not None:
            if native.receiver_to_handle_m < self.min_receiver_to_handle_m:
                self.min_receiver_to_handle_m = native.receiver_to_handle_m
                self.closest_receiver_to_handle_xyz = (
                    None
                    if native.receiver_to_handle_xyz is None
                    else np.array(native.receiver_to_handle_xyz, copy=True)
                )
        if native.receiver_to_handle_xyz is not None:
            self.final_receiver_to_handle_xyz = np.array(
                native.receiver_to_handle_xyz,
                copy=True,
            )
        if (
            native.object_long_axis_xyz is not None
            and native.primary_grasp
            and native.object_clearance_m > self.success_clearance_m
        ):
            self.max_object_long_axis_x_alignment = max(
                self.max_object_long_axis_x_alignment,
                abs(float(native.object_long_axis_xyz[0])),
            )
        if native.secondary_handle_finger_contacts is not None:
            self.max_secondary_handle_finger_contacts = max(
                self.max_secondary_handle_finger_contacts,
                native.secondary_handle_finger_contacts,
            )
        if native.secondary_handle_full_finger_contacts is not None:
            self.max_secondary_handle_full_finger_contacts = max(
                self.max_secondary_handle_full_finger_contacts,
                native.secondary_handle_full_finger_contacts,
            )
        if native.secondary_hammer_full_finger_contacts is not None:
            self.max_secondary_hammer_full_finger_contacts = max(
                self.max_secondary_hammer_full_finger_contacts,
                native.secondary_hammer_full_finger_contacts,
            )
        if native.secondary_hammer_contact_pairs is not None:
            self.secondary_hammer_contact_pairs.update(native.secondary_hammer_contact_pairs)
        if self.task_name == "two_arm_lift":
            self._observe_lift(step_index, action, native)
        else:
            self._observe_handover(step_index, action, native)

    def record_error(self, error: BaseException) -> None:
        if not self.errors:
            self.errors.append(type(error).__name__)

    def evidence(self, *, final_native_success: bool) -> Mapping[str, object]:
        checks = (
            self._lift_checks(final_native_success)
            if self.task_name == "two_arm_lift"
            else self._handover_checks(final_native_success)
        )
        result: dict[str, object] = {
            "schema_version": 1,
            "task": self.task_name,
            "protocol_success": all(checks.values()) and not self.errors,
            "checks": MappingProxyType(checks),
            "witness_steps": MappingProxyType(dict(sorted(self.witnesses.items()))),
            "steps_observed": self.steps_observed,
            "success_clearance_m": self.success_clearance_m,
            "max_object_clearance_m": (
                None if self.max_clearance_m == float("-inf") else self.max_clearance_m
            ),
            "final_native_success": bool(final_native_success),
            "evaluator_errors": tuple(self.errors),
        }
        if self.task_name == "two_arm_handover":
            result["stages"] = MappingProxyType(self._handover_stages(final_native_success))
            result["stage_metrics"] = MappingProxyType(
                {
                    "min_receiver_to_handle_m": (
                        None
                        if self.min_receiver_to_handle_m == float("inf")
                        else self.min_receiver_to_handle_m
                    ),
                    "max_object_long_axis_x_alignment": (self.max_object_long_axis_x_alignment),
                    "receiver_to_handle_at_pickup_m": self.receiver_to_handle_at_pickup_m,
                    "presentation_x_range_m": _HANDOVER_PRESENTATION_X_RANGE_M,
                    "presentation_abs_y_m": _HANDOVER_PRESENTATION_ABS_Y_M,
                    "presentation_progress_m": _HANDOVER_PRESENTATION_PROGRESS_M,
                    "receiver_approach_m": _HANDOVER_RECEIVER_APPROACH_M,
                    "max_secondary_handle_finger_contacts": (
                        self.max_secondary_handle_finger_contacts
                    ),
                    "max_secondary_handle_full_finger_contacts": (
                        self.max_secondary_handle_full_finger_contacts
                    ),
                    "max_secondary_hammer_full_finger_contacts": (
                        self.max_secondary_hammer_full_finger_contacts
                    ),
                    "closest_receiver_to_handle_xyz": (
                        None
                        if self.closest_receiver_to_handle_xyz is None
                        else tuple(float(value) for value in self.closest_receiver_to_handle_xyz)
                    ),
                    "final_receiver_to_handle_xyz": (
                        None
                        if self.final_receiver_to_handle_xyz is None
                        else tuple(float(value) for value in self.final_receiver_to_handle_xyz)
                    ),
                    "secondary_hammer_contact_pairs": tuple(
                        sorted(self.secondary_hammer_contact_pairs)
                    ),
                }
            )
        return MappingProxyType(result)

    def _observe_lift(
        self,
        step: int,
        action: RobotAction,
        native: NativeProtocolState,
    ) -> None:
        commands = action.arms
        if (
            "dual_close" not in self.witnesses
            and set(commands) == {"primary", "secondary"}
            and all(
                commands[arm].gripper_position is not None
                and commands[arm].gripper_position <= _CLOSED
                for arm in ("primary", "secondary")
            )
        ):
            self.witnesses["dual_close"] = step
            self._dual_close_targets = {
                arm: np.array(commands[arm].target, copy=True) for arm in ("primary", "secondary")
            }

        close_step = self.witnesses.get("dual_close")
        if close_step is None or step < close_step:
            return
        if "dual_grasp" not in self.witnesses and native.primary_grasp and native.secondary_grasp:
            self.witnesses["dual_grasp"] = step
        if (
            "coupled_motion" not in self.witnesses
            and step > close_step
            and set(commands) == {"primary", "secondary"}
            and self._dual_close_targets is not None
            and all(
                float(np.max(np.abs(commands[arm].target - self._dual_close_targets[arm])))
                > _MOTION_EPSILON_RAD
                for arm in ("primary", "secondary")
            )
        ):
            self.witnesses["coupled_motion"] = step
        coupled_step = self.witnesses.get("coupled_motion")
        if (
            "lifted_with_dual_grasp" not in self.witnesses
            and coupled_step is not None
            and step >= coupled_step
            and native.primary_grasp
            and native.secondary_grasp
            and native.object_clearance_m > self.success_clearance_m
        ):
            self.witnesses["lifted_with_dual_grasp"] = step

    def _observe_handover(
        self,
        step: int,
        action: RobotAction,
        native: NativeProtocolState,
    ) -> None:
        primary = action.arms.get("primary")
        secondary = action.arms.get("secondary")
        if (
            "giver_grasp_elevated" not in self.witnesses
            and native.primary_grasp
            and native.object_clearance_m > self.success_clearance_m
        ):
            self.witnesses["giver_grasp_elevated"] = step
            self.receiver_to_handle_at_pickup_m = native.receiver_to_handle_m
        giver_step = self.witnesses.get("giver_grasp_elevated")
        handle = native.handle_position_xyz
        entered_shared_workspace = (
            handle is not None
            and _HANDOVER_PRESENTATION_X_RANGE_M[0]
            <= float(handle[0])
            <= _HANDOVER_PRESENTATION_X_RANGE_M[1]
            and abs(float(handle[1])) <= _HANDOVER_PRESENTATION_ABS_Y_M
        )
        progressed_toward_receiver = (
            self.receiver_to_handle_at_pickup_m is not None
            and native.receiver_to_handle_m is not None
            and native.receiver_to_handle_m
            <= self.receiver_to_handle_at_pickup_m - _HANDOVER_PRESENTATION_PROGRESS_M
        )
        if (
            giver_step is not None
            and "presentation" not in self.witnesses
            and native.primary_grasp
            and native.object_clearance_m > self.success_clearance_m
            and (entered_shared_workspace or progressed_toward_receiver)
        ):
            self.witnesses["presentation"] = step
        presentation = self.witnesses.get("presentation")
        if (
            presentation is not None
            and "receiver_approach" not in self.witnesses
            and native.primary_grasp
            and native.object_clearance_m > self.success_clearance_m
            and native.receiver_to_handle_m is not None
            and native.receiver_to_handle_m <= _HANDOVER_RECEIVER_APPROACH_M
        ):
            self.witnesses["receiver_approach"] = step
        if (
            giver_step is not None
            and "receiver_close_command" not in self.witnesses
            and secondary is not None
            and secondary.gripper_position is not None
            and secondary.gripper_position <= _CLOSED
        ):
            self.witnesses["receiver_close_command"] = step
        receiver_close = self.witnesses.get("receiver_close_command")
        if (
            receiver_close is not None
            and "overlap_grasp" not in self.witnesses
            and step >= receiver_close
            and native.primary_grasp
            and native.secondary_grasp
        ):
            self.witnesses["overlap_grasp"] = step
        overlap = self.witnesses.get("overlap_grasp")
        if (
            overlap is not None
            and "giver_open_command" not in self.witnesses
            and primary is not None
            and primary.gripper_position is not None
            and primary.gripper_position >= _OPEN
        ):
            self.witnesses["giver_open_command"] = step
        giver_open = self.witnesses.get("giver_open_command")
        if (
            giver_open is not None
            and "receiver_only_elevated" not in self.witnesses
            and step >= giver_open
            and not native.primary_grasp
            and native.secondary_grasp
            and native.object_clearance_m > self.success_clearance_m
        ):
            self.witnesses["receiver_only_elevated"] = step

    def _lift_checks(self, final_native_success: bool) -> dict[str, bool]:
        close = self.witnesses.get("dual_close")
        grasp = self.witnesses.get("dual_grasp")
        coupled = self.witnesses.get("coupled_motion")
        lifted = self.witnesses.get("lifted_with_dual_grasp")
        return {
            "same_tick_dual_close": close is not None,
            "both_correct_grasps": grasp is not None and close is not None and grasp >= close,
            "coupled_motion_after_close": (
                coupled is not None and close is not None and coupled > close
            ),
            "lifted_while_both_grasp": (
                lifted is not None and coupled is not None and lifted >= coupled
            ),
            "native_task_success": bool(final_native_success and self.last_native_success),
        }

    def _handover_checks(self, final_native_success: bool) -> dict[str, bool]:
        giver = self.witnesses.get("giver_grasp_elevated")
        receiver_close = self.witnesses.get("receiver_close_command")
        overlap = self.witnesses.get("overlap_grasp")
        giver_open = self.witnesses.get("giver_open_command")
        receiver_only = self.witnesses.get("receiver_only_elevated")
        return {
            "giver_grasp_elevated": giver is not None,
            "receiver_close_after_giver_grasp": (
                receiver_close is not None and giver is not None and receiver_close >= giver
            ),
            "overlap_before_release": (
                overlap is not None and receiver_close is not None and overlap >= receiver_close
            ),
            "giver_open_after_receiver_grasp": (
                giver_open is not None and overlap is not None and giver_open >= overlap
            ),
            "receiver_only_elevated": (
                receiver_only is not None and giver_open is not None and receiver_only >= giver_open
            ),
            "native_task_success": bool(final_native_success and self.last_native_success),
        }

    def _handover_stages(self, final_native_success: bool) -> dict[str, Mapping[str, object]]:
        """Return host-only cumulative stage gates for targeted experiments."""
        receiver_close = self.witnesses.get("receiver_close_command")
        overlap = self.witnesses.get("overlap_grasp")
        receiver_only = self.witnesses.get("receiver_only_elevated")
        native_success = bool(final_native_success and self.last_native_success)

        def stage(
            passed: bool,
            witness: str | None,
            **details: object,
        ) -> Mapping[str, object]:
            payload: dict[str, object] = {
                "passed": passed,
                "witness_step": None if witness is None else self.witnesses.get(witness),
            }
            payload.update(details)
            return MappingProxyType(payload)

        protocol_checks = self._handover_checks(final_native_success)
        return {
            "pickup": stage(
                "giver_grasp_elevated" in self.witnesses,
                "giver_grasp_elevated",
            ),
            "presentation": stage(
                "presentation" in self.witnesses,
                "presentation",
            ),
            "receiver_approach": stage(
                "receiver_approach" in self.witnesses,
                "receiver_approach",
            ),
            "receiver_close": stage(
                overlap is not None,
                "overlap_grasp",
                command_step=receiver_close,
                native_overlap=overlap is not None,
            ),
            "giver_release": stage(
                receiver_only is not None,
                "receiver_only_elevated",
                command_step=self.witnesses.get("giver_open_command"),
                receiver_only_elevated=receiver_only is not None,
            ),
            "native_success": stage(native_success, None),
            "protocol_witnesses": stage(
                all(protocol_checks.values()) and not self.errors,
                None,
                checks=MappingProxyType(protocol_checks),
            ),
        }


class RobosuiteBimanualProtocolEvaluator:
    """Adapter-bound privileged sampler feeding the pure protocol tracker."""

    def __init__(self, task_name: str, native_env: Any) -> None:
        self.task_name = task_name
        self.native_env = native_env
        threshold = 0.10 if task_name == "two_arm_lift" else float(native_env.height_threshold)
        self.tracker = BimanualProtocolTracker(
            task_name,
            success_clearance_m=threshold,
        )
        self.step_index = 0

    def after_step(self, action: RobotAction, result: StepResult) -> None:
        if not result.ok:
            return
        self.step_index += 1
        try:
            self.tracker.observe(self.step_index, action, self._native_state())
        except Exception as exc:  # Evaluation failure must not mutate simulation control.
            self.tracker.record_error(exc)

    def evidence(self, *, final_native_success: bool) -> Mapping[str, object]:
        return self.tracker.evidence(final_native_success=final_native_success)

    def _native_state(self) -> NativeProtocolState:
        env = self.native_env
        if self.task_name == "two_arm_handover":
            primary, secondary, object_height, table_height = env._get_task_info()
            primary_base_rotation = np.asarray(
                env.sim.data.get_body_xmat("robot0_base"),
                dtype=np.float64,
            ).reshape(3, 3)
            primary_base_position = np.asarray(
                env.sim.data.get_body_xpos("robot0_base"),
                dtype=np.float64,
            ).reshape(3)
            primary_from_world_rotation = primary_base_rotation.T
            hammer_rotation = np.asarray(
                env.sim.data.body_xmat[env.hammer_body_id],
                dtype=np.float64,
            ).reshape(3, 3)
            receiver = env.robots[1].gripper
            handle_geoms = env.hammer.handle_geoms
            hammer_geoms = env.hammer.contact_geoms
            finger_pad_contacts = sum(
                bool(env.check_contact(receiver.important_geoms[name], handle_geoms))
                for name in ("left_fingerpad", "right_fingerpad")
            )
            handle_full_finger_contacts = sum(
                bool(env.check_contact(receiver.important_geoms[name], handle_geoms))
                for name in ("left_finger", "right_finger")
            )
            hammer_full_finger_contacts = sum(
                bool(env.check_contact(receiver.important_geoms[name], hammer_geoms))
                for name in ("left_finger", "right_finger")
            )
            receiver_geoms = set(receiver.contact_geoms)
            hammer_geom_set = set(hammer_geoms)
            contact_pairs: set[tuple[str, str]] = set()
            for contact in env.sim.data.contact[: env.sim.data.ncon]:
                geom1 = env.sim.model.geom_id2name(contact.geom1)
                geom2 = env.sim.model.geom_id2name(contact.geom2)
                if geom1 in receiver_geoms and geom2 in hammer_geom_set:
                    contact_pairs.add((geom1, geom2))
                elif geom2 in receiver_geoms and geom1 in hammer_geom_set:
                    contact_pairs.add((geom2, geom1))
            handle_position = primary_from_world_rotation @ (
                np.asarray(env._handle_xpos, dtype=np.float64) - primary_base_position
            )
            receiver_to_handle = primary_from_world_rotation @ np.asarray(
                env._gripper_1_to_handle,
                dtype=np.float64,
            )
            return NativeProtocolState(
                primary_grasp=bool(primary),
                secondary_grasp=bool(secondary),
                object_clearance_m=float(object_height - table_height),
                native_success=bool(env._check_success()),
                handle_position_xyz=handle_position,
                receiver_to_handle_m=float(np.linalg.norm(env._gripper_1_to_handle)),
                receiver_to_handle_xyz=receiver_to_handle,
                # HammerObject builds its handle cylinder along local Z.
                object_long_axis_xyz=primary_from_world_rotation @ hammer_rotation[:, 2],
                secondary_handle_finger_contacts=finger_pad_contacts,
                secondary_handle_full_finger_contacts=handle_full_finger_contacts,
                secondary_hammer_full_finger_contacts=hammer_full_finger_contacts,
                secondary_hammer_contact_pairs=tuple(sorted(contact_pairs)),
            )
        grippers = (
            (env.robots[0].gripper["right"], env.robots[0].gripper["left"])
            if env.env_configuration == "bimanual"
            else (env.robots[0].gripper, env.robots[1].gripper)
        )
        primary = env._check_grasp(gripper=grippers[0], object_geoms=env.pot.handle0_geoms)
        secondary = env._check_grasp(gripper=grippers[1], object_geoms=env.pot.handle1_geoms)
        pot_bottom = env.sim.data.site_xpos[env.pot_center_id][2] - env.pot.top_offset[2]
        table_height = env.sim.data.site_xpos[env.table_top_id][2]
        return NativeProtocolState(
            primary_grasp=bool(primary),
            secondary_grasp=bool(secondary),
            object_clearance_m=float(pot_bottom - table_height),
            native_success=bool(env._check_success()),
        )


__all__ = [
    "BimanualProtocolTracker",
    "NativeProtocolState",
    "RobosuiteBimanualProtocolEvaluator",
]
