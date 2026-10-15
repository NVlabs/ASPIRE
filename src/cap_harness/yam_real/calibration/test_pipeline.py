"""Offline calibration checks; no camera, network connection, or motor access."""

from __future__ import annotations

from concurrent.futures import Future
import json
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from .__main__ import main
from .arm_client import KD, KP, ArmClient
from .calibrator import MODEL, calibrate, checked_poses
from .core import Board, Detector, solve_hand_eye


def pose(position, rotvec):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = position
    return value


def dataset(mode):
    rng = np.random.default_rng(42)
    expected = pose([0.2, -0.1, 0.4], [0.2, -0.3, 0.1])
    mount = pose([0.1, 0.2, 0.6], [-0.2, 0.1, 0.3])
    samples = []
    for _ in range(20):
        ee = pose(rng.normal(0, 0.1, 3), rng.normal(0, 0.5, 3))
        board = np.linalg.inv(expected) @ (ee if mode == "fixed" else np.linalg.inv(ee)) @ mount
        samples.append({"T_base_from_ee": ee.tolist(), "T_camera_from_board": board.tolist()})
    return {
        "mode": mode,
        "samples": samples,
        "model_xml": str(MODEL),
        "parent_frame": "world" if mode == "fixed" else "left_link_6",
        "camera_body": "top_camera_d405" if mode == "fixed" else "left_camera_d405",
        "camera_name": "top" if mode == "fixed" else "wrist_left",
        "intrinsics": [[600, 0, 320], [0, 610, 240], [0, 0, 1]],
        "resolution": [640, 480],
    }, expected


@pytest.mark.parametrize("mode", ["fixed", "wrist"])
def test_solve_cli_exports_correct_camera_frame_and_loadable_model(tmp_path, mode):
    record, expected = dataset(mode)
    source = MODEL.read_bytes()
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps(record))
    output = tmp_path / "result"
    assert main(["solve", str(samples), "--output-dir", str(output)]) == 0
    solved = json.loads((output / "calibration.json").read_text())
    np.testing.assert_allclose(solved["solution"]["T_parent_from_camera"], expected, atol=1e-8)
    model = mujoco.MjModel.from_xml_path(str(output / "station_calibrated.xml"))
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    camera = model.camera(record["camera_name"]).id
    parent = model.body(record["parent_frame"]).id
    expected_world = np.eye(4)
    expected_world[:3, :3] = state.xmat[parent].reshape(3, 3)
    expected_world[:3, 3] = state.xpos[parent]
    expected_world = expected_world @ expected
    np.testing.assert_allclose(state.cam_xpos[camera], expected_world[:3, 3], atol=1e-8)
    np.testing.assert_allclose(
        state.cam_xmat[camera].reshape(3, 3),
        expected_world[:3, :3] @ np.diag([1, -1, -1]),
        atol=1e-8,
    )
    assert model.ncam == mujoco.MjModel.from_xml_path(str(MODEL)).ncam
    assert MODEL.read_bytes() == source
    with pytest.raises(ValueError, match="already exists"):
        main(["solve", str(samples), "--output-dir", str(output)])


@pytest.mark.parametrize("failure", ["few", "degenerate", "nonfinite", "inconsistent"])
def test_bad_samples_are_rejected(failure):
    record, _ = dataset("fixed")
    samples = record["samples"]
    if failure == "few":
        samples = samples[:11]
    elif failure == "degenerate":
        samples = [samples[0]] * 20
    elif failure == "nonfinite":
        samples[0]["T_base_from_ee"][0][3] = float("nan")
    else:
        for index, sample in enumerate(samples):
            sample["T_camera_from_board"][0][3] += index % 3
    with pytest.raises(ValueError):
        solve_hand_eye(samples, mode="fixed")


def test_printed_board_has_correct_scale_and_can_be_detected(tmp_path):
    path = tmp_path / "board.png"
    board = Board()
    board.write_png(path, dpi=100)
    with Image.open(path) as image:
        pixels = round(0.04 / 0.0254 * 100)
        assert image.size == (pixels * 7, pixels * 7)
        assert pixels / image.info["dpi"][0] * 0.0254 == pytest.approx(0.04, abs=1e-5)
        rgb = np.array(image.convert("RGB"))
    center = (rgb.shape[0] - 1) / 2
    intrinsics = np.array([[1000, 0, center], [0, 1000, center], [0, 0, 1]], dtype=float)
    result = Detector(board).detect(rgb, intrinsics, np.zeros(5))
    assert result is not None
    assert result["corners"] == 16
    assert result["reprojection_rms_px"] < 0.1
    assert result["T_camera_from_board"][2][3] == pytest.approx(1000 * 0.04 / pixels, abs=1e-3)


class FakeRPC:
    def __init__(self):
        self.commands = []
        self.gripper = 0.37
        self.accepted = True

    @staticmethod
    def reply(value):
        future = Future()
        future.set_result(value)
        return future

    def get_observations(self):
        return self.reply({"joint_pos": np.zeros(6), "gripper_pos": [self.gripper]})

    def command_joint_state(self, command):
        self.commands.append(command)
        return self.reply({"accepted": self.accepted})


def test_follower_contract_preserves_gripper_and_restores_stiffness():
    rpc = FakeRPC()
    arm = ArmClient("unused", 0, client=rpc)
    arm.command(arm.get_joint_pos(), gravity=True)
    np.testing.assert_array_equal(rpc.commands[-1]["kp"], np.r_[np.zeros(6), KP[-1]])
    rpc.gripper = 0.5
    arm.hold()
    for command in rpc.commands:
        np.testing.assert_array_equal(command["pos"], np.r_[np.zeros(6), 0.37])
        np.testing.assert_array_equal(command["kd"], KD)
    np.testing.assert_array_equal(rpc.commands[-1]["kp"], KP)
    rpc.accepted = False
    with pytest.raises(RuntimeError, match="refused"):
        arm.hold()


def test_invalid_sweep_and_unconfirmed_motion_are_rejected():
    limits = np.tile([-0.1, 0.1], (6, 1))
    with pytest.raises(ValueError, match="joint limits"):
        checked_poses(np.zeros(6), limits)
    with pytest.raises(ValueError, match="confirm-motion"):
        calibrate(SimpleNamespace(confirm_motion=False))


def test_camera_failure_closes_resources_and_requests_hold(tmp_path, monkeypatch):
    from . import calibrator, camera

    events = []

    class Camera:
        serial = "test"

        def __init__(self, *_args, **_kwargs):
            pass

        def open(self):
            events.append("open")

        def get_intrinsics(self):
            return np.eye(3), np.zeros(5), (640, 480)

        def grab(self):
            return False

        def close(self):
            events.append("camera_close")

    class Arm:
        def __init__(self, *_args):
            pass

        def get_joint_pos(self):
            return np.zeros(6)

        def command(self, _joints, *, gravity):
            assert gravity
            events.append("gravity")

        def hold(self):
            events.append("hold")

        def close(self):
            events.append("arm_close")

    monkeypatch.setattr(camera, "RealSenseCamera", Camera)
    monkeypatch.setattr(calibrator, "ArmClient", Arm)
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: events.append("window_close"))
    with pytest.raises(RuntimeError, match="camera stopped"):
        main(
            [
                "calibrate",
                "--serial",
                "test",
                "--camera-name",
                "top",
                "--mode",
                "fixed",
                "--arm",
                "left",
                "--output-dir",
                str(tmp_path / "capture"),
                "--confirm-motion",
            ]
        )
    assert events == ["open", "gravity", "hold", "arm_close", "camera_close", "window_close"]
