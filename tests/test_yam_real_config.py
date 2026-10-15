from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cap_harness.yam_real.config import load_yam_station_config, station_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "src/cap_harness/configs/yam_real"
#: One bench is one directory, named for its --station identifier.
STATION_DIR = CONFIG_ROOT / "yam-example"


def test_yam_example_profile_loads_with_logical_arm_roles() -> None:
    cfg = load_yam_station_config("yam-example")
    assert cfg.station == "yam-example"
    assert cfg.base_frame == "world"
    # Public roles are logical; this bench maps primary -> right, secondary -> left.
    # Both are now DECLARED rather than implied: the loader used to hardcode
    # {"primary": "right"} and discard whatever the profile said, so the field it
    # validated on load was never the field anything read.
    assert cfg.default_arm == "right"
    assert dict(cfg.arm_aliases) == {"primary": "right", "secondary": "left"}
    assert set(cfg.arms) == {"left", "right"}
    # YAM arms are 6-DoF and bound to the external follower Portal ports.
    assert cfg.arms["left"].port == 11333
    assert cfg.arms["right"].port == 11334
    for arm in cfg.arms.values():
        assert len(arm.joint_names) == 6
        assert arm.home_joints.shape == (6,)
        assert not arm.home_joints.flags.writeable  # immutable
    assert cfg.joint_limits_lower.shape == (6,) and cfg.joint_limits_upper.shape == (6,)
    assert np.all(cfg.joint_limits_lower < cfg.joint_limits_upper)
    assert cfg.controller.control_frequency_hz == 30.0
    assert cfg.controller.follower_hold_frequency_hz == 100.0


def test_runtime_and_mujoco_joint_limits_match() -> None:
    """Keep the example runtime guard and station model on one command envelope."""
    cfg = load_yam_station_config("yam-example")
    expected = np.column_stack((cfg.joint_limits_lower, cfg.joint_limits_upper))

    station = ET.parse(cfg.model_xml).getroot()
    for side in ("left", "right"):
        for joint_index, expected_range in enumerate(expected, start=1):
            joint = station.find(f".//joint[@name='{side}_joint{joint_index}']")
            assert joint is not None
            np.testing.assert_allclose(
                np.fromstring(joint.attrib["range"], sep=" "), expected_range
            )

            actuator = station.find(f".//actuator/*[@name='{side}_joint{joint_index}']")
            assert actuator is not None
            np.testing.assert_allclose(
                np.fromstring(actuator.attrib["ctrlrange"], sep=" "), expected_range
            )


def test_reference_models_share_joint_limits() -> None:
    """The generic mechanical models retain their own reference joint limits."""
    cfg = load_yam_station_config("yam-example")
    station = ET.parse(cfg.model_xml.parent / "station.xml").getroot()
    arm = ET.parse(cfg.arm_model_xml).getroot()
    for joint_index in range(1, 7):
        reference_joint = station.find(f".//joint[@name='left_joint{joint_index}']")
        assert reference_joint is not None
        expected_range = np.fromstring(reference_joint.attrib["range"], sep=" ")
        joint = arm.find(f".//joint[@name='joint{joint_index}']")
        assert joint is not None
        np.testing.assert_allclose(np.fromstring(joint.attrib["range"], sep=" "), expected_range)


def test_yam_example_calibration_bundle_is_verified_and_immutable() -> None:
    cfg = load_yam_station_config("yam-example")
    assert cfg.calibration.bundle_id == "yam-example-v1"
    assert (
        cfg.calibration.manifest_sha256
        == "ee0090ef55578327e3501869901c4938b6ff5db32d5c49d5605d85dcf6da489b"
    )
    assert (
        cfg.calibration.calibration_sha256
        == "418ff7bba9b3a820bacc4ff2a6fd9d054b01fd6a7d34f2d9f9705e3dd126886e"
    )
    # Top-camera intrinsics/extrinsics come from the bundle and are immutable.
    assert cfg.camera.role == "top"
    assert cfg.camera.resolution == (1280, 720)
    assert cfg.camera.intrinsics.shape == (3, 3)
    assert not cfg.camera.intrinsics.flags.writeable
    assert cfg.camera.base_from_camera.shape == (4, 4)


def test_camera_is_addressed_by_serial_and_declares_no_service_endpoint() -> None:
    """The camera is a USB device, not a service, and the profile must say so.

    ``open_station_camera`` reads ``serial`` and ``resolution`` and hands them to
    ``RealSenseCamera``, which selects the device from pyrealsense2's enumeration.
    The profile used to carry a ``host``/``port``/``rpc_timeout_s`` triple from an
    earlier camera-owner Portal design. Nothing read them, but their presence
    said a listener had to be running, and a closed port was read as a live-gate
    blocker for work that never needed one.
    """
    camera = load_yam_station_config("yam-example").camera

    assert camera.serial == "335122273143"
    assert camera.resolution == (1280, 720)
    for addressing in ("host", "port", "rpc_timeout_s"):
        assert not hasattr(camera, addressing), f"camera should not carry {addressing}"
    # The arm servers *are* reached over TCP; only the camera lost its endpoint.
    assert load_yam_station_config("yam-example").arms["right"].port == 11334


def test_station_profile_does_not_declare_provider_endpoints() -> None:
    """Provider endpoints belong to the harness, not to a station.

    ``configs/environments.json`` is the single registry of each provider's venv,
    module and port, and it is what the clients and the supervisor resolve
    against. The station profile used to carry its own ``providers`` block naming
    different ports (SAM3 on 6767, cuRobo on portal 8611) and a provider absent
    from the registry entirely. Nothing read it, so it could only ever drift.
    """
    cfg = load_yam_station_config("yam-example")
    assert not hasattr(cfg, "providers")
    assert "providers:" not in (STATION_DIR / "station.yaml").read_text(encoding="utf-8")

    registry = json.loads((REPO_ROOT / "configs/environments.json").read_text(encoding="utf-8"))
    assert registry["providers"]["sam3"]["port"] == 8114
    assert registry["providers"]["curobo"]["port"] == 8118


def test_unknown_station_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_yam_station_config("does-not-exist")
    with pytest.raises(ValueError):
        # Path traversal outside the config root is refused.
        station_config_path("../secret")


def test_tampered_calibration_bundle_is_refused(tmp_path: Path) -> None:
    # Copy the profile + bundle into a scratch root, corrupt the calibration
    # artifact, and require that the SHA-256 verification fails closed.
    scratch = tmp_path / "yam"
    shutil.copytree(CONFIG_ROOT, scratch)
    cal = scratch / "yam-example/calibration/yam-example-v1/calibration.json"
    data = json.loads(cal.read_text())
    data["_tamper"] = True
    cal.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_yam_station_config("yam-example", config_root=scratch)


def test_profile_config_sha_is_recorded() -> None:
    cfg = load_yam_station_config("yam-example")
    assert len(cfg.config_sha256) == 64
    assert cfg.source_path.name == "station.yaml"
    assert cfg.source_path.parent.name == "yam-example"


def test_logical_arm_roles_come_from_the_profile_not_the_loader() -> None:
    """Audit finding A3: the aliases must be data, not a constant in the loader.

    ``load_yam_station_config`` built its result with a literal
    ``{"primary": "right"}`` while validating the profile's own field two
    hundred lines earlier, so a bench that declared a different mapping was
    silently overridden and the validation guarded a value nothing used.
    """
    import yaml

    declared = yaml.safe_load((STATION_DIR / "station.yaml").read_text(encoding="utf-8"))
    cfg = load_yam_station_config("yam-example")
    assert dict(cfg.arm_aliases) == declared["arm_aliases"]
    assert cfg.default_arm == declared["default_arm"]


def test_the_shared_apis_arm_names_reach_a_real_arm() -> None:
    """Audit finding A3: every shared default failed on this embodiment.

    ``arm="primary"`` is what every signature in the shared API defaults to, and
    it reported ``execution_failed`` here while the profile had declared the
    mapping that would have resolved it all along.
    """
    from cap_harness.yam_real.adapter import YamRealAdapter
    from cap_harness.yam_real.station import build_sim_station

    station = build_sim_station(realtime=False)
    try:
        adapter = YamRealAdapter(station, allow_physical_motion=False)
        assert adapter.resolve_arm("primary") == "right"
        assert adapter.resolve_arm("secondary") == "left"
        # The physical names are not aliases and must survive untouched.
        assert adapter.resolve_arm("left") == "left"
        assert adapter.resolve_arm("right") == "right"
        # An unknown name is passed through so it fails in the caller's terms,
        # as "no arm 'wrist'" rather than as a resolver error.
        assert adapter.resolve_arm("wrist") == "wrist"
    finally:
        station.close()
