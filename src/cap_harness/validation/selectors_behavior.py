"""Structural selector for the BEHAVIOR-1K R1 Pro tasks."""

from __future__ import annotations

from cap_harness.validation.selectors import GenericStructuralSelector


def behavior_selector() -> GenericStructuralSelector:
    from cap_harness.behavior.adapter import BehaviorAdapter
    from cap_harness.behavior.registry import BehaviorTaskRegistry

    registry = BehaviorTaskRegistry()
    return GenericStructuralSelector(
        benchmark="behavior",
        control_period_s=1.0 / 30.0,
        enumerate_tasks=lambda: list(registry.enumerate_tasks()),
        make_adapter=lambda size: BehaviorAdapter(
            camera_height=size, camera_width=size, horizon=300
        ),
        language_mode="strict",
        camera_size=128,
    )


__all__ = ["behavior_selector"]
