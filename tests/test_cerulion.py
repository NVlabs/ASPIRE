from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import textwrap

import numpy as np
import pytest

from cap_harness.api import CapApi
from cap_harness.cerulion import kinematics as kin
from cap_harness.cerulion.adapter import CAMERA_NAME, CerulionAdapter
from cap_harness.cerulion.rcp_client import RcpClient, RcpError, rcpd_command
from cap_harness.cerulion.registry import CerulionRegistryError, CerulionTaskRegistry
from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.errors import ApiError
from cap_harness.registry import CERULION_PUBLIC_TOOL_NAMES
from cap_harness.run import run_program_with_api

PICK_CUBE = Path(__file__).resolve().parents[1] / "examples" / "cerulion" / "pick_cube.py"
MASK64 = (1 << 64) - 1


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


class FakePlant:
    """The contract's plant law in Python, so the adapter runs with no Rust."""

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed: int) -> None:
        state, a = _splitmix64(seed)
        _, b = _splitmix64(state)
        ux, uy = (a >> 11) * 2.0**-53, (b >> 11) * 2.0**-53
        self.q = list(kin.HOME_JOINTS)
        self.qd = [0.0] * 6
        self.gripper = 0.0
        self.cube = [0.30 + 0.12 * ux, -0.12 + 0.24 * uy, kin.CUBE_HALF_SIZE_M]
        self.grasped = False
        self.offset = [0.0, 0.0, 0.0]
        self.estopped = False
        self.steps = 0

    def ee(self) -> list[float]:
        return [float(v) for v in kin.end_effector_position(np.asarray(self.q))]

    def quaternion(self) -> list[float]:
        j0, a3, roll = self.q[0], self.q[1] + self.q[2] + self.q[3], self.q[4] + self.q[5]
        qz = [math.cos(j0 / 2), 0.0, 0.0, math.sin(j0 / 2)]
        qy = [math.cos(-a3 / 2), 0.0, math.sin(-a3 / 2), 0.0]
        qx = [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]
        return _quat_mul(_quat_mul(qz, qy), qx)

    def state(self) -> dict:
        return {
            "step": self.steps,
            "dt_s": kin.CONTROL_PERIOD_S,
            "q": list(self.q),
            "qd": list(self.qd),
            "gripper": self.gripper,
            "ee_position": self.ee(),
            "ee_quaternion_wxyz": self.quaternion(),
            "cube_position": list(self.cube),
            "grasped": self.grasped,
            "estopped": self.estopped,
            "task_success": self.grasped and self.cube[2] >= kin.SUCCESS_LIFT_M,
        }

    def step(self, q_target, gripper) -> dict:
        if self.estopped:
            return {"code": "estopped", "message": "e-stop latched"}
        if any(abs(v) > kin.JOINT_LIMIT_RAD for v in q_target):
            return {"code": "joint_limit", "message": "target outside [-3, 3]"}
        previous = list(self.q)
        self.q = [
            q + max(-kin.MAX_JOINT_STEP_RAD, min(kin.MAX_JOINT_STEP_RAD, t - q))
            for q, t in zip(self.q, q_target, strict=True)
        ]
        self.qd = [(n - p) / kin.CONTROL_PERIOD_S for n, p in zip(self.q, previous, strict=True)]
        self.gripper += max(
            -kin.MAX_GRIPPER_STEP, min(kin.MAX_GRIPPER_STEP, gripper - self.gripper)
        )
        ee = self.ee()
        if self.grasped and self.gripper < 0.5:
            self.grasped = False
            self.cube[2] = kin.CUBE_HALF_SIZE_M
        if not self.grasped and self.gripper >= 0.9 and math.dist(ee, self.cube) <= 0.03:
            self.grasped = True
            self.offset = [c - e for c, e in zip(self.cube, ee, strict=True)]
        if self.grasped:
            self.cube = [e + o for e, o in zip(ee, self.offset, strict=True)]
        self.steps += 1
        return {}


class FakeRcpClient:
    """Models cerulion-rcpd: manifest schemas, its e-stop latch, and refusals as RcpError."""

    def __init__(self) -> None:
        self.plant = FakePlant()
        self.calls: list[tuple[str, dict]] = []
        self.latched = False
        self.timeout_next: str | None = None
        self.closed = False

    def _refuse(self, tool: str, code: str, message: str, **extra) -> RcpError:
        return RcpError(
            tool, {"error": {"code": code, "message": message, **extra}, "receipt": None}
        )

    def _validate(self, tool: str, arguments: dict) -> str | None:
        if tool == "arm.reset":
            seed = arguments.get("seed")
            ok = isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
            return None if ok and set(arguments) == {"seed"} else "seed must be an integer >= 0"
        q, g = arguments.get("q_target"), arguments.get("gripper")
        if not isinstance(q, list) or len(q) != 6:
            return "q_target must hold 6 numbers"
        if any(not isinstance(v, int | float) or not -3.0 <= v <= 3.0 for v in q):
            return "q_target entries must lie in [-3, 3]"
        if not isinstance(g, int | float) or not 0.0 <= g <= 1.0:
            return "gripper must lie in [0, 1]"
        return None

    def call(self, tool: str, arguments: dict) -> dict:
        self.calls.append((tool, dict(arguments)))
        seq = len(self.calls)
        receipt = {"seq": seq, "hash": f"{seq:064x}", "prev": f"{seq - 1:064x}"}
        if tool == "estop":
            self.latched = True
            self.plant.estopped = True
            return {
                "estopped": True,
                "graph_acknowledged": True,
                "graph_error": None,
                "receipt": receipt,
                "result": {"ok": True, "error": None, "state": self.plant.state()},
            }
        if tool not in {"arm.reset", "arm.step"}:
            raise self._refuse(tool, "not_in_manifest", "tool is not in the manifest")
        problem = self._validate(tool, arguments)
        if problem is not None:
            raise self._refuse(tool, "invalid_args", problem)
        if self.latched:
            raise self._refuse(tool, "estopped", "the e-stop is engaged")
        error = None
        if tool == "arm.reset":
            self.plant.reset(int(arguments["seed"]))
        else:
            error = (
                self.plant.step(list(arguments["q_target"]), float(arguments["gripper"])) or None
            )
        if self.timeout_next == tool:
            self.timeout_next = None
            raise self._refuse(tool, "timeout", "no answer", may_have_executed=True)
        return {
            "call_id": seq,
            "receipt": receipt,
            "result": {"ok": error is None, "error": error, "state": self.plant.state()},
        }

    def close(self) -> None:
        self.closed = True


def _api(seed: int = 0) -> tuple[CerulionAdapter, CapApi, FakeRcpClient]:
    client = FakeRcpClient()
    adapter = CerulionAdapter(client)
    adapter.reset("cerulion", seed)
    return adapter, CapApi(adapter, default_camera=CAMERA_NAME), client


def test_registry_resolves_suite_and_name() -> None:
    registry = CerulionTaskRegistry()
    assert registry.resolve("cerulion", 0).task_name == "pick_cube"
    assert registry.resolve("pick_cube").task_id == 0
    with pytest.raises(CerulionRegistryError):
        registry.resolve("cerulion", 3)


def test_reset_publishes_task_metadata_and_a_valid_observation() -> None:
    from cap_harness.protocols import EnvironmentAdapter

    adapter, api, client = _api(seed=4)
    assert isinstance(adapter, EnvironmentAdapter)
    assert client.calls == [("arm.reset", {"seed": 4})]
    context = adapter.get_task_context()
    assert context.metadata["task_supplied_cube_position"] == client.plant.cube
    assert "task_supplied_cube_position" in context.language
    assert set(context.metadata["kinematics"]) == {"H", "L1", "L2", "L3"}
    observation = adapter.get_observation()
    camera = observation.cameras[CAMERA_NAME]
    assert camera.rgb.shape == (128, 256, 3) and camera.rgb.dtype == np.uint8
    assert observation.robot_state.embodiment == "cerulion"
    # Home is open in ASPIRE's convention (plant 0.0 = open).
    assert observation.robot_state.gripper_positions["primary"] == 1.0
    assert api.get_robot_state().joint_positions["primary"].shape == (6,)


def test_step_is_one_rcp_call_and_follows_the_plant_rate() -> None:
    adapter, api, client = _api()
    target = np.asarray(kin.HOME_JOINTS) + np.array([0.2, 0, 0, 0, 0, 0])
    result = api.move_to_joints(target)
    assert result.ok and result.steps_executed == 4
    assert [tool for tool, _ in client.calls[1:]] == ["arm.step"] * 4
    # ASPIRE's open (1.0) reaches the plant as its open (0.0).
    assert all(args["gripper"] == 0.0 for _, args in client.calls[1:])


def test_rcpd_refusals_become_typed_failures() -> None:
    adapter, api, client = _api()
    bad = np.asarray(kin.HOME_JOINTS) + np.array([4.0, 0, 0, 0, 0, 0])
    result = api.move_to_joints(bad)
    assert not result.ok and result.error.code.value == "invalid_request"
    assert result.error.details["rcpd_code"] == "invalid_args"


def test_estop_latches_steps_and_reset() -> None:
    adapter, api, client = _api()
    reply = adapter.estop("operator")
    assert reply["estopped"] and reply["graph_acknowledged"]
    result = api.move_to_joints(np.asarray(kin.HOME_JOINTS))
    assert not result.ok and result.error.code.value == "safety_interlock"
    assert result.error.recoverable is False
    with pytest.raises(ApiError) as excinfo:
        adapter.reset("cerulion", 0)
    assert excinfo.value.code.value == "safety_interlock" and not excinfo.value.recoverable
    with pytest.raises(RuntimeError):
        adapter.get_robot_state()


def test_estop_reply_without_a_graph_answer() -> None:
    adapter, _, client = _api()
    client.call = lambda tool, arguments: {  # type: ignore[method-assign]
        "estopped": True,
        "graph_acknowledged": False,
        "graph_error": "timeout",
        "receipt": {"seq": 9},
        "result": None,
    }
    reply = adapter.estop("operator")
    assert reply == {
        "estopped": True,
        "graph_acknowledged": False,
        "graph_error": "timeout",
        "receipt": {"seq": 9},
    }


def test_leftover_latch_file_refuses_reset(tmp_path) -> None:
    receipts = tmp_path / "receipts.jsonl"
    (tmp_path / "receipts.jsonl.estop").write_text("", encoding="utf-8")
    client = FakeRcpClient()
    adapter = CerulionAdapter(client, receipts=receipts)
    with pytest.raises(ApiError) as excinfo:
        adapter.reset("cerulion", 0)
    assert excinfo.value.code.value == "safety_interlock"
    assert "receipts.jsonl.estop" in excinfo.value.message
    assert client.calls == []


def test_timeout_is_not_recoverable_and_marks_state_stale() -> None:
    adapter, api, client = _api()
    client.timeout_next = "arm.step"
    result = adapter.step(
        RobotAction(
            arms={
                "primary": ArmCommand(
                    mode="joint_position",
                    target=np.asarray(kin.HOME_JOINTS),
                    embodiment="cerulion",
                )
            }
        )
    )
    assert not result.ok and result.error.code.value == "timeout"
    assert result.error.details["may_have_executed"] is True
    assert result.error.recoverable is False
    with pytest.raises(RuntimeError, match="stale"):
        adapter.get_robot_state()


def test_program_visible_results_carry_no_success_or_grasp_signal() -> None:
    adapter, api, client = _api()
    result = api.move_to_joints(np.asarray(kin.HOME_JOINTS) + np.array([0.1, 0, 0, 0, 0, 0]))
    assert result.ok
    text = repr(dict(result.diagnostics))
    assert "success" not in text and "grasp" not in text
    snapshot = adapter.monitor_snapshot()
    assert set(snapshot["state"]) == {"joint_positions", "gripper_positions", "estopped"}
    assert snapshot["camera_metadata"][CAMERA_NAME]["kind"] == "synthetic_render"


def test_trajectory_checks_embodiment_period_and_start() -> None:
    adapter, api, client = _api()
    start = adapter.get_robot_state()
    goal = np.asarray(kin.HOME_JOINTS) + np.array([0.1, 0, 0, 0, 0, 0])
    waypoints = np.linspace(np.asarray(kin.HOME_JOINTS), goal, 3)

    def trajectory(**overrides):
        fields = {
            "joint_positions": waypoints,
            "dt_s": kin.CONTROL_PERIOD_S,
            "joint_names": kin.JOINT_NAMES,
            "planner": "test",
            "collision_aware": False,
            "expected_start": start,
            "embodiment": "cerulion",
        }
        fields.update(overrides)
        return Trajectory(**fields)

    assert adapter.execute_trajectory(trajectory(dt_s=0.1)).error.code.value == "invalid_request"
    assert adapter.execute_trajectory(trajectory()).ok
    stale = adapter.execute_trajectory(trajectory())
    assert not stale.ok and stale.error.code.value == "stale_plan"


def test_controller_metadata_matches_the_plant_rate() -> None:
    from cap_harness.registry import ToolRegistry

    adapter, api, _ = _api()
    tools = api.register_tools(ToolRegistry(public_extension_allowlist=CERULION_PUBLIC_TOOL_NAMES))
    names = {spec.name for spec in tools.all_specs(public_only=True)}
    assert names >= CERULION_PUBLIC_TOOL_NAMES
    metadata = adapter.get_controller_metadata()
    assert metadata["control_frequency_hz"] == kin.CONTROL_FREQUENCY_HZ
    assert metadata["control_period_s"] == kin.CONTROL_PERIOD_S
    assert metadata["max_joint_step_rad"] == kin.MAX_JOINT_STEP_RAD
    # A one-joint move of 0.2 rad takes 0.2 / 0.05 = 4 ticks through the public API.
    result = api.plan_motion(np.asarray(kin.HOME_JOINTS) + np.array([0.2, 0, 0, 0, 0, 0]))
    assert result.ok and len(result.trajectory.joint_positions) <= 5


def _ik(x: float, y: float, z: float) -> np.ndarray:
    """The pick_cube program's IK, run outside the sandbox for an FK check."""
    namespace: dict = {"LINKS": kin.kinematics_metadata()}
    source = PICK_CUBE.read_text(encoding="utf-8")
    header = source.split("stages = {}")[0]
    header = header.replace(
        'context = get_task_context()\ncube = context.metadata["task_supplied_cube_position"]\n'
        'links = context.metadata["kinematics"]\n',
        "links = LINKS\n",
    )
    exec(header, namespace)
    return np.asarray(namespace["tool_down_ik"](x, y, z))


@pytest.mark.parametrize(
    "target", [(0.30, -0.12, 0.025), (0.42, 0.12, 0.025), (0.36, 0.0, 0.105), (0.33, 0.05, 0.25)]
)
def test_pick_cube_ik_inverts_the_plant_fk(target) -> None:
    q = _ik(*target)
    assert np.allclose(kin.end_effector_position(q), target, atol=1e-9)
    assert q[1] + q[2] + q[3] == pytest.approx(-math.pi / 2)
    assert np.all(np.abs(q) <= kin.JOINT_LIMIT_RAD)


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_pick_cube_program_succeeds_end_to_end_on_the_fake_plant(seed: int) -> None:
    adapter, api, client = _api(seed)
    from cap_harness.registry import ToolRegistry
    from cap_harness.runtime import ProgramExecutor

    tools = api.register_tools(ToolRegistry(public_extension_allowlist=CERULION_PUBLIC_TOOL_NAMES))
    execution = ProgramExecutor(tools).execute_program(PICK_CUBE.read_text(encoding="utf-8"))
    assert execution.ok, execution
    assert not execution.result["failed_stages"]
    assert adapter.check_success()
    steps = [tool for tool, _ in client.calls if tool == "arm.step"]
    assert 20 <= len(steps) <= 200


def test_pick_cube_through_run_program_with_api() -> None:
    adapter, api, client = _api(0)
    execution = run_program_with_api(
        api=api, program_path=PICK_CUBE, extension_allowlist=CERULION_PUBLIC_TOOL_NAMES
    )
    assert execution.ok and adapter.check_success()


def test_rcpd_command_builds_from_manifest_or_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CAP_HARNESS_CERULION_RCPD_CMD", raising=False)
    assert rcpd_command(tmp_path / "rcp.yaml", tmp_path / "r.jsonl") == [
        "cerulion-rcpd",
        "--manifest",
        str(tmp_path / "rcp.yaml"),
        "--stdio",
        "--receipts",
        str(tmp_path / "r.jsonl"),
    ]
    monkeypatch.setenv(
        "CAP_HARNESS_CERULION_RCPD_CMD", "/opt/bin/cerulion-rcpd --manifest 'a b.yaml' --stdio"
    )
    assert rcpd_command() == ["/opt/bin/cerulion-rcpd", "--manifest", "a b.yaml", "--stdio"]
    monkeypatch.delenv("CAP_HARNESS_CERULION_RCPD_CMD")
    with pytest.raises(ValueError):
        rcpd_command()


_FAKE_SERVER = textwrap.dedent(
    """
    import json
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent

    server = FastMCP("fake-rcpd")

    @server.tool(name="arm.reset")
    def reset(seed: int) -> CallToolResult:
        body = {"call_id": 1, "receipt": {"seq": 1, "hash": "h", "prev": "p"},
                "result": {"ok": True, "error": None, "state": {"step": 0, "seed": seed}}}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(body))])

    @server.tool(name="arm.step")
    def step(q_target: list[float], gripper: float) -> CallToolResult:
        body = {"error": {"code": "not_in_manifest", "message": "denied"}, "receipt": None}
        return CallToolResult(isError=True,
                              content=[TextContent(type="text", text=json.dumps(body))])

    server.run("stdio")
    """
)


def test_rcp_client_speaks_mcp_over_stdio(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")
    script = tmp_path / "fake_rcpd.py"
    script.write_text(_FAKE_SERVER, encoding="utf-8")
    client = RcpClient([sys.executable, str(script)], call_timeout_s=10.0)
    try:
        assert {"arm.reset", "arm.step"} <= set(client.tool_names)
        payload = client.call("arm.reset", {"seed": 9})
        assert payload["result"]["state"] == {"step": 0, "seed": 9}
        with pytest.raises(RcpError) as excinfo:
            client.call("arm.step", {"q_target": [0.0] * 6, "gripper": 0.0})
        assert excinfo.value.code == "not_in_manifest"
    finally:
        client.close()
    client.close()
    assert json.dumps(payload)


_FAKE_PLANT_SERVER = textwrap.dedent(
    """
    import json
    import sys

    sys.path.insert(0, sys.argv[1])
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent
    from cap_harness.cerulion.rcp_client import RcpError
    from test_cerulion import FakeRcpClient

    client = FakeRcpClient()
    server = FastMCP("fake-rcpd")

    def reply(tool, arguments):
        try:
            body, error = client.call(tool, arguments), False
        except RcpError as exc:
            body, error = exc.body, True
        return CallToolResult(
            isError=error, content=[TextContent(type="text", text=json.dumps(body))]
        )

    @server.tool(name="arm.reset")
    def reset(seed: int) -> CallToolResult:
        return reply("arm.reset", {"seed": seed})

    @server.tool(name="arm.step")
    def step(q_target: list[float], gripper: float) -> CallToolResult:
        return reply("arm.step", {"q_target": q_target, "gripper": gripper})

    @server.tool(name="estop")
    def estop(reason: str) -> CallToolResult:
        return reply("estop", {"reason": reason})

    server.run("stdio")
    """
)


def test_cli_run_records_a_successful_episode_over_mcp(tmp_path, monkeypatch) -> None:
    pytest.importorskip("mcp.server.fastmcp")
    pytest.importorskip("imageio")
    import shlex

    from cap_harness.cli import main

    script = tmp_path / "fake_plant_rcpd.py"
    script.write_text(_FAKE_PLANT_SERVER, encoding="utf-8")
    tests_dir = str(Path(__file__).resolve().parent)
    monkeypatch.setenv(
        "CAP_HARNESS_CERULION_RCPD_CMD", shlex.join([sys.executable, str(script), tests_dir])
    )
    code = main(
        [
            "run",
            "--benchmark",
            "cerulion",
            "--suite",
            "cerulion",
            "--task-id",
            "0",
            "--seed",
            "0",
            "--program",
            str(PICK_CUBE),
            "--output-root",
            str(tmp_path / "out"),
            "--camera-width",
            "256",
            "--camera-height",
            "128",
        ]
    )
    assert code == 0
    manifests = list((tmp_path / "out").rglob("manifest.json"))
    assert manifests, "the run wrote no manifest"
    steps = list((tmp_path / "out").rglob("steps.jsonl"))
    assert steps and 20 <= len(steps[0].read_text(encoding="utf-8").splitlines()) <= 200


def test_agent_mcp_bridge_exposes_the_cerulion_surface() -> None:
    pytest.importorskip("jsonschema")
    from cap_harness.agent.mcp import CapMcpBridge
    from cap_harness.agent.server import build_cerulion_runtime

    client = FakeRcpClient()
    adapter = CerulionAdapter(client)
    runtime = build_cerulion_runtime(adapter, seed=0)
    try:
        bridge = CapMcpBridge(runtime.session)
        names = {tool.name for tool in bridge.tool_specs()}
        assert {
            "cap_get_robot_state",
            "cap_move_to_joints",
            "cap_set_gripper",
            "cap_cerulion__get_task_metadata",
            "cap_cerulion__get_controller_metadata",
        } <= names
        assert "cap_move_to_pose" not in names
        metadata = bridge.call("cap_cerulion__get_task_metadata", {})
        text = metadata.content[0].text if hasattr(metadata, "content") else str(metadata)
        assert "task_supplied_cube_position" in text
    finally:
        runtime.close()


def test_rcp_client_close_ends_a_daemon_stuck_before_initialize(tmp_path) -> None:
    import os
    import time

    from cap_harness.cerulion.rcp_client import RcpTransportError

    pid_file = tmp_path / "pid"
    script = tmp_path / "silent.py"
    script.write_text(
        "import os, sys, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "sys.stdin.read()\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    with pytest.raises(RcpTransportError):
        RcpClient([sys.executable, str(script)], startup_timeout_s=1.0)
    assert time.monotonic() - started < 30.0
    pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        pytest.fail("the daemon outlived RcpClient.close()")


def test_rcp_client_refuses_a_daemon_missing_the_arm_tools(tmp_path) -> None:
    pytest.importorskip("mcp.server.fastmcp")
    from cap_harness.cerulion.rcp_client import RcpTransportError

    script = tmp_path / "fake_rcpd.py"
    script.write_text(_FAKE_SERVER, encoding="utf-8")
    with pytest.raises(RcpTransportError, match=r"arm\.missing"):
        RcpClient([sys.executable, str(script)], required_tools=("arm.reset", "arm.missing"))


def test_cli_requires_a_manifest_for_cerulion(monkeypatch, capsys) -> None:
    from cap_harness.cli import main

    monkeypatch.delenv("CAP_HARNESS_CERULION_RCPD_CMD", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "run",
                "--benchmark",
                "cerulion",
                "--suite",
                "cerulion",
                "--task-id",
                "0",
                "--program",
                str(PICK_CUBE),
            ]
        )
    assert excinfo.value.code == 2
    assert "--rcpd-manifest" in capsys.readouterr().err
