# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gripper helpers with contact-oriented evidence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .artifacts import append_stage_summary, current_run_dir, write_json


def read_gripper_width(get_robot_state: Callable[..., Any], side: str) -> float | None:
    state = get_robot_state()
    value = getattr(state, f"{side}_gripper_pos", None)
    if value is None and isinstance(state, dict):
        arm = (state.get("arms") or {}).get(side) or {}
        value = arm.get("gripper_pos")
    return None if value is None else float(value)


def staged_close_with_contact(
    *,
    side: str,
    set_gripper: Callable[..., Any],
    get_robot_state: Callable[..., Any],
    target: float = 0.0,
    steps: list[float] | tuple[float, ...] = (0.65, 0.45, 0.25, 0.0),
    vel_limit: float | None = 0.08,
    torque_limit: float | None = 0.35,
    min_contact_delta: float = 0.02,
    hold_min: float = 0.025,
    hold_max: float = 0.95,
    target_tolerance: float = 0.04,
    confirm_timeout_s: float = 0.0,
    state_poll_s: float = 0.1,
    run_dir: str | Path | None = None,
    task_name: str = "yam_runtime",
    stage: str = "grasp_close",
) -> dict[str, Any]:
    """Close in stages and record both contact and target-reached evidence."""
    run_dir = Path(run_dir) if run_dir is not None else current_run_dir(task_name)
    before = read_gripper_width(get_robot_state, side)
    records: list[dict[str, Any]] = []
    for step in steps:
        goal = max(float(target), min(1.0, float(step)))
        try:
            command_kwargs: dict[str, float] = {}
            if vel_limit is not None:
                command_kwargs["vel_limit"] = float(vel_limit)
            if torque_limit is not None:
                command_kwargs["torque_limit"] = float(torque_limit)
            result = set_gripper(side, goal, **command_kwargs)
            ok = True
            err = None
        except Exception as exc:
            result = None
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        after_step = read_gripper_width(get_robot_state, side)
        records.append({"target": goal, "ok": ok, "error": err, "after": after_step, "result": result})
        if not ok:
            break
    after = records[-1]["after"] if records else read_gripper_width(get_robot_state, side)
    samples: list[float] = []
    deadline = time.monotonic() + max(0.0, float(confirm_timeout_s))
    while after is not None and time.monotonic() < deadline:
        samples.append(float(after))
        if float(after) <= float(target) + float(target_tolerance):
            break
        time.sleep(max(0.02, float(state_poll_s)))
        after = read_gripper_width(get_robot_state, side)
    if after is not None and (not samples or samples[-1] != float(after)):
        samples.append(float(after))
    contact_delta = None if after is None else float(after) - float(target)
    all_commands_ok = bool(records) and all(bool(record.get("ok")) for record in records)
    target_reached = (
        after is not None
        and all_commands_ok
        and float(after) <= float(target) + float(target_tolerance)
    )
    in_hold_range = after is not None and float(hold_min) <= float(after) <= float(hold_max)
    contact_detected = (
        all_commands_ok
        and contact_delta is not None
        and contact_delta >= float(min_contact_delta)
        and bool(in_hold_range)
    )
    plausible_for_lift = bool(all_commands_ok and in_hold_range and (contact_detected or target_reached))
    packet = {
        "schema": "openforge.yam_runtime.gripper_close.v1",
        "side": side,
        "before": before,
        "after": after,
        "target": float(target),
        "contact_delta": contact_delta,
        "contact_detected": contact_detected,
        "target_reached": target_reached,
        "target_tolerance": float(target_tolerance),
        "all_commands_ok": all_commands_ok,
        "in_hold_range": bool(in_hold_range),
        "hold_min": float(hold_min),
        "hold_max": float(hold_max),
        "confirm_timeout_s": float(confirm_timeout_s),
        "state_poll_s": float(state_poll_s),
        "plausible_for_lift": plausible_for_lift,
        "evidence_note": (
            "contact_detected means final width stopped above target; "
            "target_reached can still be valid for bottle grasps where the normalized "
            "gripper encoder reaches the commanded mid-body close value."
        ),
        "steps": records,
        "samples": samples,
    }
    packet["path"] = write_json(run_dir / "plans" / f"{stage}.json", packet)
    append_stage_summary(
        run_dir,
        [
            f"## gripper {stage}",
            f"- side: {side}",
            f"- before: {before}",
            f"- after: {after}",
            f"- contact_detected: {contact_detected}",
            f"- target_reached: {target_reached}",
            f"- plausible_for_lift: {plausible_for_lift}",
        ],
    )
    return packet
