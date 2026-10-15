"""Exercise Viser with synthetic frames and no robot or camera connections."""

import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.request

import msgspec
import numpy as np
import pytest
import viser
from websockets.sync.client import connect

from cap_harness.agent.session import SessionBusyError
from cap_harness.agent.visualizer import System2Visualizer


@pytest.fixture
def viewer():
    snapshot = {"cameras": {}, "camera_metadata": {}, "observation_seq": 1}
    session = SimpleNamespace(
        cached_visual_snapshot=Mock(return_value=snapshot),
        job=Mock(return_value=None),
        start_named_program=Mock(return_value="home-job"),
        steer=Mock(),
    )
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    visualizer = System2Visualizer(session, allow_motion=True, port=port)
    yield visualizer, session, snapshot
    visualizer.close()


@pytest.mark.parametrize("count", [2, 3, 4])
def test_camera_views_follow_station_names_and_count(viewer, count):
    visualizer, _session, snapshot = viewer
    names = [f"camera_{index}" for index in range(count)]
    with visualizer._lock:
        snapshot["cameras"] = {name: np.full((30, 40, 3), 80, dtype=np.uint8) for name in names}
        snapshot["camera_metadata"] = {name: {"sequence": 1} for name in names}
        visualizer._refresh()
        assert list(visualizer._images) == names
        for handle in visualizer._images.values():
            assert np.all(handle.image == 80)

        # A disconnected frame is labeled rather than displayed as live.
        snapshot["camera_metadata"][names[0]]["missing"] = True
        del snapshot["cameras"][names[0]]
        snapshot["camera_metadata"][names[1]]["stale"] = True
        visualizer._refresh()
        assert "missing" in visualizer._images[names[0]].label
        assert np.all(visualizer._images[names[0]].image == 0)
        assert "stale" in visualizer._images[names[1]].label

        # Remove a camera from the station and recover the other feed.
        del snapshot["camera_metadata"][names[0]]
        snapshot["cameras"][names[1]] = np.full((30, 40, 3), 120, dtype=np.uint8)
        snapshot["camera_metadata"][names[1]] = {"sequence": 2}
        visualizer._refresh()
        assert names[0] not in visualizer._images
        assert visualizer._images[names[1]].label == names[1]
        assert np.all(visualizer._images[names[1]].image == 120)


def test_viser_serves_browser_page_and_dispatches_home_to_shared_session(viewer):
    visualizer, session, _snapshot = viewer
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{visualizer.server.get_port()}/", timeout=5) as response:
        assert response.status == 200
        assert b"<html" in response.read()
    session.start_named_program.assert_not_called()

    visualizer._on_home(None)

    session.start_named_program.assert_called_once_with("go_home")
    session.steer.assert_called_once()
    assert visualizer._home.disabled


def test_browser_home_event_reaches_session_over_websocket(viewer):
    visualizer, session, _snapshot = viewer
    connected, called = threading.Event(), threading.Event()
    visualizer.server.on_client_connect(lambda _client: connected.set())

    def home(_name):
        called.set()
        return "home-job"

    session.start_named_program.side_effect = home
    with connect(
        f"ws://127.0.0.1:{visualizer.server.get_port()}",
        proxy=None,
        subprotocols=[f"viser-v{viser.__version__}"],
        max_size=None,
    ) as browser:
        browser.send(
            msgspec.msgpack.encode(
                {
                    "type": "ViewerCameraMessage",
                    "wxyz": [1.0, 0.0, 0.0, 0.0],
                    "position": [0.0, 0.0, 1.0],
                    "fov": 1.0,
                    "near": 0.01,
                    "far": 100.0,
                    "image_height": 600,
                    "image_width": 800,
                    "look_at": [0.0, 0.0, 0.0],
                    "up_direction": [0.0, 1.0, 0.0],
                }
            )
        )
        assert connected.wait(3)
        client = next(iter(visualizer.server.get_clients().values()))
        # Connect callbacks run concurrently; wait for the initial view callback.
        deadline = time.monotonic() + 3
        while (
            not np.allclose(client.camera.look_at, [0.4, 0.0, 0.8]) and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert np.all(np.isfinite(client.camera.wxyz))
        np.testing.assert_allclose(client.camera.look_at, [0.4, 0.0, 0.8])
        browser.send(
            msgspec.msgpack.encode(
                {
                    "type": "GuiUpdateMessage",
                    "uuid": visualizer._home._impl.uuid,
                    "updates": {"value": True},
                }
            )
        )
        assert called.wait(3)
    session.start_named_program.assert_called_once_with("go_home")


def test_home_disabled_in_observation_mode_and_rejects_busy_session(viewer):
    visualizer, session, _snapshot = viewer
    visualizer._allow_motion = False
    visualizer._refresh()
    assert visualizer._home.disabled
    visualizer._on_home(None)
    session.start_named_program.assert_not_called()

    visualizer._allow_motion = True
    session.job.return_value = {"id": "motion-job", "active": True}
    session.start_named_program.side_effect = SessionBusyError("An arm motion is running")
    visualizer._refresh()
    assert visualizer._home.disabled
    # A click that races with a new MCP action is also refused by the session.
    visualizer._on_home(None)
    visualizer._refresh()
    assert "An arm motion is running" in visualizer._status.content
    session.steer.assert_not_called()


def test_home_failure_is_visible_and_button_recovers(viewer):
    visualizer, session, _snapshot = viewer
    visualizer._on_home(None)
    session.job.return_value = {
        "id": "home-job",
        "active": False,
        "error": None,
        "result": {"ok": False},
    }
    visualizer._refresh()
    assert "Home failed" in visualizer._status.content
    assert not visualizer._home.disabled


def test_robot_model_tracks_cached_joints_and_gripper_width(viewer):
    visualizer, session, snapshot = viewer
    with visualizer._lock:
        assert not visualizer._robot_root.visible
        snapshot["state"] = {
            "left_joint_pos": np.zeros(6),
            "right_joint_pos": np.zeros(6),
            "left_gripper_pos": np.array([0.0]),
            "right_gripper_pos": np.array([0.5]),
        }
        visualizer._refresh()
        assert visualizer._robot_root.visible
        model = visualizer._robot._urdf
        before = model.get_transform("left_link_6").copy()
        snapshot["state"]["left_joint_pos"][0] = 0.25
        visualizer._refresh()
        assert not np.allclose(model.get_transform("left_link_6"), before)
        cfg = dict(zip(model.actuated_joint_names, model.cfg))
        assert cfg["left_joint1"] == 0.25
        assert cfg["right_joint1"] == 0.0
        assert cfg["left_left_finger_joint"] == 0.0
        assert cfg["right_left_finger_joint"] == pytest.approx(0.5 * 0.037524)
        assert cfg["right_right_finger_joint"] == pytest.approx(-0.5 * 0.037524)
        assert len(visualizer._robot._meshes) > 0
    session.start_named_program.assert_not_called()


@pytest.mark.parametrize("unavailable", ["missing", "invalid", "stale", "error"])
def test_robot_model_hides_when_telemetry_is_unavailable(viewer, unavailable):
    visualizer, _session, snapshot = viewer
    with visualizer._lock:
        snapshot["state"] = {
            "left_joint_pos": np.zeros(6),
            "right_joint_pos": np.zeros(6),
            "left_gripper_pos": np.array([1.0]),
            "right_gripper_pos": np.array([1.0]),
        }
        visualizer._refresh()
        assert visualizer._robot_root.visible
        if unavailable == "missing":
            del snapshot["state"]["right_joint_pos"]
        elif unavailable == "invalid":
            snapshot["state"]["left_joint_pos"][0] = np.nan
        elif unavailable == "stale":
            snapshot["age_s"] = 2.0
        else:
            snapshot["monitor_error"] = "monitor disconnected"
        visualizer._refresh()
        assert not visualizer._robot_root.visible
        assert "Robot model hidden" in visualizer._status.content
