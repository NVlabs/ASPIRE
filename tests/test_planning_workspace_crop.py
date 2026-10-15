"""The planning-scene crop removes only what the robot cannot reach.

Two properties, and the second is the one that matters. The crop is an
optimization: a depth camera sees the whole room, and returns beyond the arm's
reach cannot be collided with, so carrying them into the planner only inflates
the voxel grid the obstacle mesh is built over. Measured on a real bimanual station that is 27% of
the cloud and 3x the planning latency.

It is emphatically *not* a safety mechanism, and a test suite that only checked
"points went down" would pass on a crop tight enough to delete the object being
grasped. So the tests below pin the boundary from both sides: reachable geometry
survives, unreachable geometry does not, and an embodiment that declares no
workspace is left exactly as it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.api import CapApi


class _Adapter:
    """Minimal adapter stub: the crop only needs the hook and an embodiment."""

    embodiment = "robosuite"

    def __init__(self, bounds: object | None) -> None:
        self._bounds = bounds

    def get_robot_state(self):  # pragma: no cover - CapApi probes for home joints
        raise RuntimeError("no state")

    if True:  # keep the attribute conditional on construction, see _without_hook

        def planning_workspace(self):
            return self._bounds


class _NoHookAdapter:
    """An embodiment that never heard of a planning workspace -- e.g. LIBERO."""

    embodiment = "libero"

    def get_robot_state(self):  # pragma: no cover
        raise RuntimeError("no state")


LOWER = np.array([-0.61, -1.17, -0.11])
UPPER = np.array([1.11, 1.17, 1.61])

#: A believable slice of a live station frame: table returns, the cube, a point on
#: the arm itself, and the room outliers the raw cloud actually contains.
REACHABLE = np.array(
    [
        [0.60, 0.00, 0.745],  # table surface
        [0.57, -0.12, 0.79],  # the cube's top face
        [0.50, -0.31, 0.91],  # a point on the right arm
        [0.25, 0.31, 0.75],  # the left arm mount
    ]
)
OUT_OF_REACH = np.array(
    [
        [6.43, 0.00, 0.90],  # far wall
        [0.50, -6.45, 0.90],  # off to the side of the room
        [0.50, 0.00, -5.63],  # a degenerate return below the floor
        [0.50, 0.00, 1.80],  # above the gantry, beyond any reach
    ]
)


def _crop(points: np.ndarray, bounds: object | None) -> np.ndarray:
    return CapApi(_Adapter(bounds))._crop_to_planning_workspace(points, "world")


def test_reachable_points_all_survive() -> None:
    """Nothing inside the box is ever discarded. This is the safety-relevant half."""
    kept = _crop(REACHABLE, (LOWER, UPPER))
    assert len(kept) == len(REACHABLE)
    np.testing.assert_allclose(np.sort(kept, axis=0), np.sort(REACHABLE, axis=0))


def test_out_of_reach_points_are_dropped() -> None:
    mixed = np.concatenate([REACHABLE, OUT_OF_REACH])
    kept = _crop(mixed, (LOWER, UPPER))
    assert len(kept) == len(REACHABLE)


def test_the_cube_survives_the_crop() -> None:
    """Explicit regression: the object being grasped must reach the planner.

    A crop drawn round the table rather than round the arm's reach could delete
    an object placed at the workspace edge, and the failure would look like a
    planning failure rather than a missing obstacle.
    """
    cube = np.array([[0.6927, -0.246, 0.7843], [0.5593, -0.127, 0.79]])
    kept = _crop(cube, (LOWER, UPPER))
    assert len(kept) == len(cube)


def test_an_adapter_without_the_hook_is_untouched() -> None:
    """Every simulator embodiment must behave exactly as before."""
    mixed = np.concatenate([REACHABLE, OUT_OF_REACH])
    api = CapApi(_NoHookAdapter())
    kept = api._crop_to_planning_workspace(mixed, "world")
    np.testing.assert_allclose(kept, mixed)


def test_a_none_workspace_disables_the_crop() -> None:
    mixed = np.concatenate([REACHABLE, OUT_OF_REACH])
    np.testing.assert_allclose(_crop(mixed, None), mixed)


def test_inverted_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="lower bounds must be below"):
        _crop(REACHABLE, (UPPER, LOWER))


def test_non_finite_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _crop(REACHABLE, (LOWER, np.array([np.inf, 1.17, 1.61])))


def test_cropping_everything_away_is_an_error_not_an_empty_scene() -> None:
    """An empty scene would plan happily against nothing at all.

    That is the most dangerous possible outcome -- a collision-free plan through
    geometry the planner never saw -- so it has to fail loudly. Mis-set bounds
    and wrong camera extrinsics both land here.
    """
    with pytest.raises(ValueError, match="contains no observed points"):
        _crop(OUT_OF_REACH, (LOWER, UPPER))
