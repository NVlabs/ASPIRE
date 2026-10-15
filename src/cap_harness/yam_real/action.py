"""The typed action every YAM motion is expressed as.

One batch type, one entry point. Motion tools, homing, gripper commands and any
future policy all build a :class:`YamActionBatch` and hand it to
``RealYamEnv.execute_action_batch``; nothing commands the arms another way. That
is what lets a recorded episode name the action abstraction it ran under instead
of leaving it implied by whichever function happened to be called.

This module carries the *joint* action space only: absolute joint targets plus a
normalized gripper. Cartesian goals reach the arms through the shared IK and
planning providers, which return joint trajectories -- so an end-effector action
space would be a second way to say the same thing, and would need the policy
frame conventions that belong with the policy path.

Gripper convention throughout: ``0.0`` closed, ``1.0`` open. The gripper is a
separate field, never a seventh joint; it is only concatenated into the joint7
vector inside the level-1 controller, which is the layer the arm server's wire
format belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ARMS = ("left", "right")
ARM_DOF = 6


@dataclass(slots=True)
class YamActionBatch:
    """A timestamped bimanual joint trajectory, submitted as one unit."""

    space: str
    timestamps: Any
    left: Any
    right: Any
    left_gripper: Any | None = None
    right_gripper: Any | None = None
    source: str = "unknown"
    chunk_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def joint_abs(
        cls,
        timestamps: Any,
        left_joint_positions: Any,
        right_joint_positions: Any,
        left_gripper_positions: Any | None = None,
        right_gripper_positions: Any | None = None,
        *,
        source: str = "unknown",
        chunk_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> YamActionBatch:
        """Absolute joint targets, shape ``(N, 6)`` or ``(N, 7)`` per arm."""
        return cls(
            space="joint_abs",
            timestamps=timestamps,
            left=left_joint_positions,
            right=right_joint_positions,
            left_gripper=left_gripper_positions,
            right_gripper=right_gripper_positions,
            source=source,
            chunk_id=chunk_id,
            meta=dict(meta or {}),
        )


@dataclass(slots=True)
class ResolvedYamAction:
    """A batch checked and expanded into the arrays the controller streams."""

    timestamps: np.ndarray
    left_joint_positions: np.ndarray
    right_joint_positions: np.ndarray
    left_gripper_positions: np.ndarray
    right_gripper_positions: np.ndarray
    source: str
    input_space: str
    chunk_id: str | None
    meta: dict[str, Any]


def _as_timestamps(values: Any) -> np.ndarray:
    ts = np.asarray(values, dtype=np.float64).reshape(-1)
    if ts.size < 1:
        raise ValueError("timestamps must contain at least one value")
    if not np.all(np.isfinite(ts)):
        raise ValueError("timestamps contain non-finite values")
    if np.any(np.diff(ts) < -1e-9):
        raise ValueError("timestamps must be monotonically increasing")
    return ts


def _as_rows(values: Any, n: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < ARM_DOF:
        raise ValueError(f"{name} must have shape (N,{ARM_DOF}+) with N={n}; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _gripper(values: Any | None, held: Any, n: int) -> np.ndarray:
    """Resolve a gripper column, holding the measured position when unspecified."""
    if values is None:
        return np.full((n, 1), float(np.asarray(held).reshape(-1)[0]), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full((n, 1), float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(n, -1)[:, :1]
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def resolve_action_batch(env: Any, batch: YamActionBatch | dict[str, Any]) -> ResolvedYamAction:
    """Validate a batch against the plant and expand it to per-arm arrays.

    An unspecified gripper holds its **measured** position rather than defaulting
    to open or closed, so a joint-only motion cannot drop what the robot is
    carrying.
    """
    if isinstance(batch, dict):
        batch = YamActionBatch(**batch)
    space = str(batch.space).strip().lower()
    if space not in {"joint", "joint_abs", "joint_position"}:
        raise ValueError(
            f"unsupported YAM action space {batch.space!r}; this embodiment accepts joint_abs"
        )

    ts = _as_timestamps(batch.timestamps)
    n = int(ts.size)
    left = _as_rows(batch.left, n, name="left joint action")
    right = _as_rows(batch.right, n, name="right joint action")

    observed = {side: env.get_observations(side) for side in ARMS}
    # A joint7 row carries its own gripper at index 6; an explicit argument wins.
    left_grip_source = batch.left_gripper
    if left_grip_source is None and left.shape[1] > ARM_DOF:
        left_grip_source = left[:, ARM_DOF : ARM_DOF + 1]
    right_grip_source = batch.right_gripper
    if right_grip_source is None and right.shape[1] > ARM_DOF:
        right_grip_source = right[:, ARM_DOF : ARM_DOF + 1]

    return ResolvedYamAction(
        timestamps=ts,
        left_joint_positions=left[:, :ARM_DOF],
        right_joint_positions=right[:, :ARM_DOF],
        left_gripper_positions=_gripper(left_grip_source, observed["left"]["gripper_pos"], n),
        right_gripper_positions=_gripper(right_grip_source, observed["right"]["gripper_pos"], n),
        source=str(batch.source),
        input_space="joint_abs",
        chunk_id=batch.chunk_id,
        meta=dict(batch.meta),
    )


__all__ = ["ARMS", "ARM_DOF", "ResolvedYamAction", "YamActionBatch", "resolve_action_batch"]
