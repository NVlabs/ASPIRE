"""Exercise standalone control ownership and MCP calls without hardware."""

import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from cap_harness.agent.mcp import CapMcpBridge
from cap_harness.agent.server import build_yam_runtime, create_app
from cap_harness.yam_real.action import resolve_action_batch
from cap_harness.yam_real.sim import FINGER_QSLICE
from cap_harness.yam_real.station import build_sim_station


@pytest.fixture
def rig(monkeypatch):
    env = build_sim_station(realtime=False)
    commands = []

    def execute(batch, **_kwargs):
        resolved = resolve_action_batch(env, batch)
        commands.append((threading.get_ident(), resolved))
        env._note_action_input(resolved)
        env._plant.set_joint_positions(
            resolved.left_joint_positions[-1], resolved.right_joint_positions[-1]
        )
        with env._plant.lock:
            for side, positions in (
                ("left", resolved.left_gripper_positions),
                ("right", resolved.right_gripper_positions),
            ):
                env._plant.data.qpos[FINGER_QSLICE[side]] = env._plant._finger_targets(
                    side, float(positions[-1, 0])
                )
        return {"success": True}

    monkeypatch.setattr(env, "execute_action_batch", execute)
    adapter, runtime = build_yam_runtime(env, allow_motion=True)
    value = SimpleNamespace(
        env=env,
        adapter=adapter,
        runtime=runtime,
        session=runtime.session,
        bridge=CapMcpBridge(runtime.session),
        commands=commands,
    )
    yield value
    runtime.close()
    adapter.close()


def test_mcp_gripper_returns_terminal_state_and_labeled_images(rig):
    result = rig.bridge.call("cap_set_gripper", {"position": 0.6, "arm": "left"})
    assert not result.isError
    assert result.structuredContent["job"]["terminal_state"]["embodiment"] == "yam_real"
    assert any(block.type == "image" for block in result.content)
    assert any(block.type == "text" and block.text == "camera: top" for block in result.content)
    assert len({thread for thread, _ in rig.commands}) == 1
    assert rig.commands[0][0] != threading.get_ident()


def test_observation_and_steering_remain_available_during_motion(rig, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = rig.env.execute_action_batch

    def delayed(batch, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(batch, **kwargs)

    monkeypatch.setattr(rig.env, "execute_action_batch", delayed)
    job_id = rig.session.start_tool("set_gripper", {"position": 0.5, "arm": "left"})
    try:
        assert entered.wait(1)
        assert not rig.bridge.call("observe").isError
        busy = rig.bridge.call("cap_set_gripper", {"position": 0.7, "arm": "right"})
        assert busy.isError and "SessionBusyError" in busy.structuredContent["error"]
        rig.session.steer("hold position")
        assert rig.bridge.call("read_operator_messages").structuredContent["messages"] == [
            "hold position"
        ]
    finally:
        release.set()
    assert not rig.session.wait(job_id, timeout_s=3)["error"]


def test_unauthorized_motion_never_reaches_plant(rig):
    rig.adapter._adapter._allow_physical_motion = False
    result = rig.bridge.call("cap_set_gripper", {"position": 0.5, "arm": "left"})
    assert result.isError
    assert not rig.commands


def test_home_shortcut_opens_grippers_and_homes_once_on_control_worker(rig):
    start = np.array([0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    rig.env._plant.set_joint_positions(start, start)
    assert not rig.commands  # Building the MCP runtime must not home by itself.

    result = rig.bridge.call("cap_program_go_home", {})

    assert not result.isError
    assert len(rig.commands) == 3  # Two single-arm opens and one whole-robot Home.
    assert len({thread for thread, _ in rig.commands}) == 1
    assert rig.commands[0][0] != threading.get_ident()
    home = rig.commands[-1][1]
    for side in ("left", "right"):
        np.testing.assert_allclose(
            getattr(home, f"{side}_joint_positions")[-1], rig.env.config.arms[side].home_joints
        )
        assert getattr(home, f"{side}_gripper_positions")[-1, 0] == pytest.approx(1.0)
    summary = result.structuredContent["job"]["result"]
    diagnostics = summary["result"]["diagnostics"]
    assert diagnostics["home_verified"] is True
    assert diagnostics["joint_residual_rad"] == {"left": 0.0, "right": 0.0}
    assert diagnostics["grippers_open"] == {"left": True, "right": True}
    assert any(block.type == "image" for block in result.content)


def test_home_shortcut_obeys_server_motion_interlock(rig):
    rig.adapter._adapter._allow_physical_motion = False

    result = rig.bridge.call("cap_program_go_home", {})

    assert result.isError
    assert not rig.commands


def test_home_shortcut_stops_after_gripper_failure(rig, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("gripper unavailable")

    monkeypatch.setattr(rig.env, "set_gripper", fail)
    result = rig.bridge.call("cap_program_go_home", {})

    assert result.isError
    assert not rig.commands
    assert "gripper unavailable" in str(result.structuredContent)


def test_home_reports_measured_residual_even_when_command_succeeds(rig, monkeypatch):
    start = np.array([0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
    rig.env._plant.set_joint_positions(start, start)
    monkeypatch.setattr(rig.env, "go_home", lambda **kwargs: {"success": True})

    result = rig.bridge.call("cap_program_go_home", {})

    assert not result.isError
    diagnostics = result.structuredContent["job"]["result"]["result"]["diagnostics"]
    assert diagnostics["home_verified"] is False
    assert diagnostics["joint_residual_rad"]["left"] == pytest.approx(0.4)
    assert diagnostics["joint_residual_rad"]["right"] == pytest.approx(0.4)


@pytest.mark.parametrize("count", [2, 3, 4])
def test_mcp_motion_returns_every_active_camera_without_fixed_names(rig, monkeypatch, count):
    snapshot = rig.session.visual_snapshot()
    names = [f"camera_{index}" for index in range(count)]
    snapshot["cameras"] = {name: np.zeros((10, 10, 3), dtype=np.uint8) for name in names}
    snapshot["camera_metadata"] = {name: {"missing": False, "stale": False} for name in names}
    monkeypatch.setattr(rig.session, "visual_snapshot", lambda: snapshot)

    result = rig.bridge.call("cap_set_gripper", {"position": 0.6, "arm": "left"})

    assert not result.isError
    assert result.structuredContent["image_cameras"] == names
    assert sum(block.type == "image" for block in result.content) == count


def test_mcp_http_transport_exposes_only_configured_tools(rig):
    with TestClient(create_app(rig.session)) as client:
        headers = {
            "accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-11-25",
        }
        response = client.post(
            "/mcp/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert response.status_code == 200
        response = client.post(
            "/mcp/",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        names = {t["name"] for t in response.json()["result"]["tools"]}
        assert names == {
            "get_capability_graph",
            "observe",
            "get_job",
            "read_operator_messages",
            "cap_get_robot_state",
            "cap_move_to_joints",
            "cap_move_synchronized",
            "cap_set_gripper",
            "cap_yam_real__get_controller_metadata",
            "cap_program_observe",
            "cap_program_go_home",
        }
        response = client.post(
            "/mcp/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "cap_get_robot_state", "arguments": {}},
            },
        )
        assert response.json()["result"]["structuredContent"]["ok"]
        assert not rig.commands
