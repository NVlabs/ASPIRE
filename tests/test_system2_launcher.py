"""Check launch permissions and startup instructions without starting an agent."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "arguments",
    [[], ["--allow-motion"], ["--allow-motion", "exec", "Pick up the cube."]],
)
def test_launcher_selects_startup_mode_and_preserves_task(tmp_path, arguments):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "start_system2_agent.sh"
    shutil.copyfile(root / "scripts/start_system2_agent.sh", launcher)
    docs = tmp_path / "docs"
    docs.mkdir()
    guide = (root / "docs/system2-agent-guide.md").read_text().rstrip("\n")
    (docs / "system2-agent-guide.md").write_text(guide)
    binaries = tmp_path / ".venv-system2/bin"
    binaries.mkdir(parents=True)
    # Stub the launcher's HTTP health check and capture Codex argv. Neither a
    # real service nor a real agent runs in this test.
    python = binaries / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    codex = binaries / "codex"
    codex.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    codex.chmod(0o755)

    launched = subprocess.run(
        ["bash", str(launcher), *arguments],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )
    argv = json.loads(launched.stdout)
    prompt = argv[-1]
    assert argv[argv.index("-C") + 1] == str(tmp_path)
    assert 'approval_policy="never"' in argv
    assert 'sandbox_mode="workspace-write"' in argv
    assert 'sandbox_workspace_write.network_access=true' in argv
    assert prompt.startswith(f"{guide}\n\nSession startup:")
    motion_approvals = {
        f'mcp_servers.cap.tools.{tool}.approval_mode="approve"'
        for tool in (
            "cap_program_go_home",
            "cap_move_to_pose",
            "cap_move_synchronized",
            "cap_move_to_joints",
            "cap_set_gripper",
        )
    }
    assert "Session startup:" in prompt
    if "--allow-motion" in arguments:
        assert motion_approvals <= set(argv)
        assert "call cap_program_go_home exactly once before the operator task" in prompt
        assert "Do not move the robot during readiness checks" not in prompt
    else:
        assert motion_approvals.isdisjoint(argv)
        assert "observation-only startup" in prompt
    if "exec" in arguments:
        assert argv[0] == "exec"
        assert prompt.endswith("Operator task:\nPick up the cube.")
    else:
        assert prompt.endswith(
            "Operator task:\nAfter startup completes, wait for the operator task."
        )
