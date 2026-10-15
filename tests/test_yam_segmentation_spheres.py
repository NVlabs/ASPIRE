"""The segmentation sphere set is a *second* model, and it covers the whole arm.

Two separate claims are pinned here, because the bug this fixes came from
conflating them:

* the planner's self-collision skeleton stays sparse and untouched -- it is
  tuned for "can link 3 hit link 6" and making it dense would slow every plan;
* the segmentation set is dense and covers every link the camera can see,
  because anything it misses stays in the depth cloud as a phantom obstacle and
  the planner then refuses to move the arm into its own reflection.

A test that only checked "spheres exist" would pass on the broken configuration,
which had 30 spheres, one finger link out of four, and links 4 and 5 represented
by a single ball each.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

STATION = (
    Path(__file__).resolve().parents[1] / "src/cap_harness/yam_real/description/station/curobo"
)
SELF_COLLISION_PATH = STATION / "yam_dual_isaacsim_physics_fixed_fingers.yml"
SEGMENTATION_PATH = STATION / "yam_dual_segmentation_spheres.yml"

ARM_LINKS = ("arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6")
FINGER_LINKS = ("left_link_finger", "right_link_finger")

#: Must exceed the generator's reported worst surface gap, or uncovered slivers
#: of the arm survive the carve. Mirrors ``_SEGMENTATION_TOLERANCE_M``.
SEGMENTATION_TOLERANCE_M = 0.02


@pytest.fixture(scope="module")
def self_collision_spheres() -> dict[str, list]:
    config = yaml.safe_load(SELF_COLLISION_PATH.read_text(encoding="utf-8"))
    return config["robot_cfg"]["kinematics"]["collision_spheres"]


@pytest.fixture(scope="module")
def segmentation_document() -> dict:
    return yaml.safe_load(SEGMENTATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def segmentation_spheres(segmentation_document: dict) -> dict[str, list]:
    return segmentation_document["collision_spheres"]


def test_self_collision_model_stays_sparse(self_collision_spheres: dict[str, list]) -> None:
    """The planning model is unchanged: coarse on purpose, and cheap."""
    total = sum(len(entries) for entries in self_collision_spheres.values())
    assert len(self_collision_spheres) == 15
    assert total == 30
    radii = [entry["radius"] for entries in self_collision_spheres.values() for entry in entries]
    assert max(radii) <= 0.05


def test_the_two_sphere_sets_are_different_models(
    self_collision_spheres: dict[str, list],
    segmentation_spheres: dict[str, list],
) -> None:
    """Separate files, separate geometry. Neither is derived from the other."""
    assert SELF_COLLISION_PATH != SEGMENTATION_PATH
    assert segmentation_spheres != self_collision_spheres
    segmentation_total = sum(len(entries) for entries in segmentation_spheres.values())
    self_collision_total = sum(len(entries) for entries in self_collision_spheres.values())
    # An order of magnitude denser, which is the entire point.
    assert segmentation_total > 10 * self_collision_total


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("suffix", ARM_LINKS)
def test_segmentation_covers_every_arm_link(
    segmentation_spheres: dict[str, list], side: str, suffix: str
) -> None:
    name = f"{side}_{suffix}"
    assert name in segmentation_spheres, f"segmentation set is missing {name}"
    assert len(segmentation_spheres[name]) >= 10


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("finger", FINGER_LINKS)
def test_segmentation_covers_every_finger(
    segmentation_spheres: dict[str, list], side: str, finger: str
) -> None:
    """All four finger links, not the one the self-collision model happens to name.

    The fingers are what hovers directly over the object, so a finger left in the
    cloud lands exactly where the pregrasp goal is.
    """
    name = f"{side}_{finger}"
    assert name in segmentation_spheres, f"segmentation set is missing {name}"
    assert len(segmentation_spheres[name]) >= 10


def test_segmentation_covers_the_finger_links_self_collision_omits(
    self_collision_spheres: dict[str, list],
    segmentation_spheres: dict[str, list],
) -> None:
    """Regression: the shipped self-collision model names one finger link of four."""
    self_fingers = {name for name in self_collision_spheres if "finger" in name}
    segmentation_fingers = {name for name in segmentation_spheres if "finger" in name}
    assert len(self_fingers) < len(segmentation_fingers)
    assert len(segmentation_fingers) == 4


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("suffix", ("link_4", "link_5"))
def test_links_four_and_five_are_no_longer_a_single_ball(
    self_collision_spheres: dict[str, list],
    segmentation_spheres: dict[str, list],
    side: str,
    suffix: str,
) -> None:
    """Regression: these carried exactly one sphere each in the self-collision model."""
    name = f"{side}_{suffix}"
    assert len(self_collision_spheres[name]) == 1
    assert len(segmentation_spheres[name]) >= 20


def test_segmentation_spheres_follow_the_surface(segmentation_spheres: dict[str, list]) -> None:
    """Small radii, not inflated balls.

    Coverage is bought with sphere count, not radius. A set of few large spheres
    would carve a hole around the arm big enough to swallow a nearby object,
    which is the failure mode this whole approach exists to avoid.
    """
    radii = [entry["radius"] for entries in segmentation_spheres.values() for entry in entries]
    assert max(radii) <= 0.02
    assert min(radii) > 0.0


def test_carve_tolerance_exceeds_the_measured_surface_gap(segmentation_document: dict) -> None:
    """The tolerance has to cover whatever the fit failed to reach, plus calibration."""
    worst_gap = float(segmentation_document["worst_surface_gap_m"])
    assert worst_gap < SEGMENTATION_TOLERANCE_M


def test_segmentation_set_excludes_static_station_geometry(
    segmentation_spheres: dict[str, list],
) -> None:
    """The table and frame are real obstacles; carving them would erase the world."""
    for static_link in ("base_link", "play_table", "gate_collision", "top_bar_center"):
        assert static_link not in segmentation_spheres


def test_service_constants_match_this_configuration() -> None:
    """The service applies the small tolerance and leaves the global threshold alone."""
    pytest.importorskip("uvicorn")
    pytest.importorskip("curobo")
    from cap_harness.providers.curobo import service

    assert service._SEGMENTATION_TOLERANCE_M == SEGMENTATION_TOLERANCE_M
    # Raising this to compensate for sparse geometry is the fix that erases the
    # target cube near contact; it must stay where it was.
    assert service._ROBOT_SEGMENTATION_DISTANCE_M == 0.05
