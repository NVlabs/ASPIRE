from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cap_harness.cli import build_parser
from cap_harness.run import run_program
from cap_harness.yam_real import station


def test_passive_yam_run_seals_episode_without_video(tmp_path, monkeypatch):
    env = station.build_sim_station(realtime=False)
    commands = []
    monkeypatch.setattr(env, "execute_action_batch", lambda *a, **kw: commands.append(a))
    monkeypatch.setattr(station, "build_real_station", lambda *a, **kw: env)
    program = tmp_path / "observe.py"
    program.write_text("result = get_robot_state()\n")
    outcome = run_program(
        benchmark="yam_real",
        suite="observe_station",
        task_id=0,
        seed=1,
        program_path=program,
        output_root=tmp_path / "runs",
        max_steps=10,
        capture_videos=False,
    )
    assert outcome.program_ok
    assert not commands
    assert not list(outcome.run_dir.rglob("*.mp4"))
    raw = outcome.run_dir / "episode/raw"
    metadata = json.loads((raw / "metadata.json").read_text())
    assert metadata["fps"] == 30
    assert metadata["physical_motion_authorized"] is False
    assert np.load(raw / "left-joint_pos.npy").shape[1] == 6
    # The sealed manifest must account for arrays created during sampler teardown.
    manifests = list(outcome.run_dir.rglob("manifest.json"))
    assert manifests
    text = "\n".join(path.read_text() for path in manifests)
    assert "episode/raw/left-joint_pos.npy" in text
    assert hashlib.sha256((raw / "left-joint_pos.npy").read_bytes()).hexdigest() in text


def test_yam_cli_accepts_explicit_station_and_motion_options():
    args = build_parser().parse_args(
        [
            "run",
            "--benchmark",
            "yam_real",
            "--suite",
            "reach_home",
            "--task-id",
            "0",
            "--program",
            "home.py",
            "--station",
            "my-yam",
            "--station-config-root",
            "/profiles",
            "--allow-motion",
        ]
    )
    assert args.station == "my-yam"
    assert args.station_config_root == Path("/profiles")
    assert args.allow_motion


def test_curobo_yam_model_is_cached_bimanual_and_collision_checked(monkeypatch):
    service = pytest.importorskip("cap_harness.providers.curobo.service")
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)
    assert not service._collision_disabled("yam_real")
    path = service._yam_real_robot_config()
    assert service._yam_real_robot_config() == path
    kin = yaml.safe_load(Path(path).read_text())["robot_cfg"]["kinematics"]
    assert Path(kin["urdf_path"]).is_file()
    assert {"left_grasp", "right_grasp"} <= set(kin["tool_frames"])
    assert len(kin["cspace"]["joint_names"]) == 12
    request = service.BaseRequest(
        base_frame="world",
        model="yam_real",
        joint_positions={side: [0.0] * 6 for side in ("left", "right")},
        joint_names={side: [f"{side}_joint{i}" for i in range(1, 7)] for side in ("left", "right")},
        base_transforms={side: np.eye(4).tolist() for side in ("left", "right")},
        end_effector_links={side: f"{side}_grasp" for side in ("left", "right")},
    )
    assert service._validate(request) == ("left", "right")
    assert service._robot_config(request) == path
