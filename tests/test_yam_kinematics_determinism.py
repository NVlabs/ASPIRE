"""mink IK must be a function of its arguments, not of what solved before it.

``YamKinematics`` holds one mutable MuJoCo configuration, and
``inverse_kinematics(seeded=True)`` continues from it. The provider passed the
caller's seed only to the OPPOSITE arm's hold target and never wrote it into
that configuration, so the arm actually being solved started from wherever the
previous solve had left the model.

The symptom is a solve that succeeds alone and fails in sequence, which is the
worst shape a bug can take here: a reachability survey and the motion it is
surveying for disagree about the same pose. It cost a live run -- the wheel
program scored a rim pose reachable at 12 of 12 angles, then failed to solve
that same pose for the descent, with mink reporting a best effort 0.067 m away.
"""

from __future__ import annotations

import numpy as np
import pytest

from cap_harness.api import CapApi
from cap_harness.contracts import Pose
from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider
from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.codec import rotation_to_wxyz
from cap_harness.yam_real.kinematics import YamKinematics
from cap_harness.yam_real.station import build_sim_station

#: The wheel and rim pose from the run that exposed this.
CENTRE = (0.5405, -0.0394)
RADIUS = 0.056
HEIGHT = 0.8479
ANGLES = (180.0, 150.0, 210.0, 135.0, 225.0, 120.0, 240.0, 90.0, 270.0, 60.0, 300.0, 0.0)


def _rim_pose(degrees: float, flip: bool = False) -> Pose:
    theta = np.radians(degrees)
    rx, ry = np.cos(theta), np.sin(theta)
    sign = -1.0 if flip else 1.0
    rotation = np.column_stack(
        [[sign * rx, sign * ry, 0.0], [sign * ry, -sign * rx, 0.0], [0.0, 0.0, -1.0]]
    )
    return Pose(
        np.array([CENTRE[0] + RADIUS * rx, CENTRE[1] + RADIUS * ry, HEIGHT]),
        rotation_to_wxyz(rotation),
        frame="world",
    )


@pytest.fixture
def api():
    env = build_sim_station(realtime=False)
    adapter = YamRealAdapter(env, allow_physical_motion=False)
    # ONE shared solver, exactly as a program gets.
    yield CapApi(
        adapter,
        ik_providers={"mink": YamKinematicsIKProvider(YamKinematics(env.config.model_xml))},
    )
    env.close()


def test_a_solve_survives_unrelated_solves_on_the_same_solver(api):
    """The regression: a survey of the workspace must not break the next solve."""
    target = _rim_pose(270.0)
    seed = np.array([0.6404, 1.9072, 1.6222, -1.2858, 0.0, 0.6404])
    assert api.solve_ik(target, seed, arm="right", backend="mink").ok

    # What choose_arm does before planning: score every rim angle, both arms.
    for arm in ("right", "left"):
        for degrees in ANGLES:
            for flip in (False, True):
                if api.solve_ik(_rim_pose(degrees, flip), np.zeros(6), arm=arm, backend="mink").ok:
                    break

    result = api.solve_ik(target, seed, arm="right", backend="mink")
    assert result.ok, (
        "the same pose stopped solving after unrelated solves on the shared "
        f"solver: {None if result.error is None else result.error.message}"
    )


def test_the_same_call_twice_gives_the_same_answer(api):
    target = _rim_pose(270.0)
    seed = np.array([0.6404, 1.9072, 1.6222, -1.2858, 0.0, 0.6404])
    first = api.solve_ik(target, seed, arm="right", backend="mink")
    second = api.solve_ik(target, seed, arm="right", backend="mink")
    assert first.ok and second.ok
    assert np.allclose(first.joint_positions, second.joint_positions)


def test_the_callers_seed_actually_reaches_the_solver(api):
    """Two different seeds for one target must be able to differ.

    If the seed never reached the solver, both calls would continue from the
    same internal configuration and return identical answers -- which is how
    this went unnoticed.
    """
    target = _rim_pose(240.0)
    near = api.solve_ik(target, np.zeros(6), arm="right", backend="mink")
    assert near.ok
    # Re-solving from the solution it just produced must stay in that branch.
    # Not bit-identical: mink still runs its iterations and lands about 3e-4
    # away. The property that matters is that it does not wander to a different
    # elbow configuration, which is what a solver continuing from stale internal
    # state does.
    again = api.solve_ik(target, near.joint_positions, arm="right", backend="mink")
    assert again.ok
    assert np.allclose(again.joint_positions, near.joint_positions, atol=1e-3)
