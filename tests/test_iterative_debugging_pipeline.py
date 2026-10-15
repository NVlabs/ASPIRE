from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".claude" / "skills" / "iterative-debugging" / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_linter_reports_invalid_programs(tmp_path, monkeypatch, capsys):
    linter = load_script("lint_skills")
    proposal = tmp_path / "proposal.md"
    proposal.write_text("```python\nimport os\n```\n")
    monkeypatch.setattr(linter, "SKILLS", tmp_path / "empty-library")
    monkeypatch.setattr(sys, "argv", ["lint_skills", "--proposal", str(proposal)])
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert linter.main() == 1
    assert "Import of 'os' is not allowed" in capsys.readouterr().out


def test_skill_linter_does_not_disguise_validator_bugs(tmp_path, monkeypatch):
    from cap_harness import runtime

    linter = load_script("lint_skills")
    proposal = tmp_path / "proposal.md"
    proposal.write_text("```python\nresult = 1\n```\n")
    monkeypatch.setattr(linter, "SKILLS", tmp_path / "empty-library")
    monkeypatch.setattr(sys, "argv", ["lint_skills", "--proposal", str(proposal)])
    monkeypatch.setattr(sys, "path", list(sys.path))

    def broken_visit(self, node):
        del self, node
        raise RuntimeError("validator malfunction")

    monkeypatch.setattr(runtime._ProgramValidator, "visit", broken_visit)
    with pytest.raises(RuntimeError, match="validator malfunction"):
        linter.main()


def test_validation_finish_respects_the_campaign_heldout_count(tmp_path, monkeypatch):
    validation = load_script("run_validation")
    (tmp_path / "campaign.json").write_text(json.dumps({"heldout_count": 2}))
    manifest = {
        "run_id": "test-run",
        "identity": {"seeds": [1, 2]},
        "results": {"1": {"task_success": True}, "2": {"task_success": False}},
    }
    promoted = []
    monkeypatch.setattr(validation, "update_stage1_validation", lambda *args: promoted.append(args))
    monkeypatch.setattr(validation.subprocess, "run", lambda *args, **kwargs: None)
    assert (
        validation.finish(
            SimpleNamespace(),
            manifest,
            tmp_path / "manifest.json",
            tmp_path / "fix_code.py",
            [1, 2],
            0,
            tmp_path,
        )
        == 0
    )
    assert len(promoted) == 1
    assert manifest["trials"] == 2 and manifest["passes"] == 1


def test_validation_retains_simulator_log_diagnostics(tmp_path):
    validation = load_script("run_validation")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "outcome.json").write_text(
        json.dumps(
            {
                "program_ok": False,
                "task_success": False,
                "termination_reason": "program_error",
            }
        )
    )
    log_path = tmp_path / "seed_01.log"
    log_path.write_text("[Error] first\n[Error] device-side assert\n")
    manifest = {"results": {}}
    assert validation.record_trial(
        manifest,
        tmp_path / "manifest.json",
        seed=1,
        run_dir=run_dir,
        exit_code=1,
        log_path=log_path,
    )
    trial = manifest["results"]["1"]
    assert trial["kit_error_lines"] == 2
    assert trial["device_assert"] is True
    assert trial["process_exit_code"] == 1


def test_generated_code_prompts_enforce_non_privileged_provenance_gate():
    skills = ROOT / ".claude" / "skills"
    for skill in ("iterative-debugging", "evosearch"):
        subagent = (skills / skill / "subagent-prompt.md").read_text()
        coordinator = (skills / skill / "main-agent-prompt.md").read_text()

        assert "Every task-dependent object choice" in subagent
        assert "`BENCHMARK` is `libero-pro` or `robosuite`" in subagent
        assert "LIBERO-specific examples" in subagent
        assert "Robosuite-specific examples" in subagent
        assert "`env.sim.*`" in subagent
        assert "`_check_success()`" in subagent
        assert "generic relative offsets, tolerances, quantiles, and step limits" in subagent
        assert "never rewrite it to hide its provenance" in subagent
        assert "Non-privileged API audit: PASS/QUARANTINED" in subagent
        assert "Non-privileged API audit: PASS" in coordinator
        assert "report the task as invalid" in coordinator


def test_campaign_heldout_partition_comes_from_campaign_json(tmp_path: Path):
    """A campaign may validate on 1..20; the default stays the historical 1..50."""
    campaign = load_script("campaign")
    assert campaign.heldout_seeds(None) == list(range(1, 51))
    assert campaign.heldout_seeds(tmp_path) == list(range(1, 51))
    (tmp_path / "campaign.json").write_text(json.dumps({"heldout_count": 20}))
    assert campaign.heldout_seeds(tmp_path) == list(range(1, 21))

    validation = load_script("run_validation")
    twenty = {"seeds": list(range(1, 21))}
    assert validation.is_full_heldout(twenty, list(range(1, 21)))
    assert not validation.is_full_heldout(twenty)  # module default is still 1..50
    assert not validation.is_full_heldout({"seeds": list(range(1, 51))}, list(range(1, 21)))


def test_init_run_writes_campaign_settings(tmp_path: Path, monkeypatch):
    init_run = load_script("init_run")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"benchmark": "robosuite", "suite": "cube_lifting", "task_id": 0}])
    )
    monkeypatch.setattr(init_run, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(init_run, "LATEST_LINK", tmp_path / "runs" / "LATEST")
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr("sys.argv", ["init_run.py", "--tasks", str(tasks), "--heldout-count", "20"])

    init_run.main()

    root = (tmp_path / "runs" / "LATEST").resolve()
    settings = json.loads((root / "campaign.json").read_text())
    assert settings["heldout_count"] == 20
    assert settings["dev_seeds"] == list(range(51, 66))


def test_validation_identity_changes_with_code_or_settings(tmp_path: Path):
    validation = load_script("run_validation")
    code = tmp_path / "fix.py"
    code.write_text("result = {}\n")

    identity = validation.build_identity(
        benchmark="libero-pro",
        suite="suite",
        task_id=3,
        program=code,
        seeds=[2, 1, 2],
        init_mode="seeded",
        max_steps=1000,
        camera_width=800,
        camera_height=512,
    )
    first = validation.run_id_for_identity(identity)
    assert identity["seeds"] == [1, 2]

    code.write_text("observation = get_observation()\nresult = {}\n")
    changed_code = validation.build_identity(
        benchmark="libero-pro",
        suite="suite",
        task_id=3,
        program=code,
        seeds=[1, 2],
        init_mode="seeded",
        max_steps=1000,
        camera_width=800,
        camera_height=512,
    )
    assert validation.run_id_for_identity(changed_code) != first

    code.write_text("result = {}\n")
    changed_mode = validation.build_identity(
        benchmark="libero-pro",
        suite="suite",
        task_id=3,
        program=code,
        seeds=[1, 2],
        init_mode="saved",
        max_steps=1000,
        camera_width=800,
        camera_height=512,
    )
    assert validation.run_id_for_identity(changed_mode) != first


def test_progress_counts_one_manifest_only(tmp_path: Path):
    progress = load_script("gen_progress")
    old = tmp_path / "runs" / "old"
    new = tmp_path / "runs" / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "old",
                "updated_at": "2026-01-01",
                "results": {
                    "1": {"task_success": True, "program_ok": True},
                    "2": {"task_success": True, "program_ok": True},
                },
            }
        )
    )
    (new / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "new",
                "updated_at": "2026-02-01",
                "results": {
                    "1": {"task_success": False, "program_ok": True},
                    "2": {"task_success": True, "program_ok": True},
                    "51": {"task_success": True, "program_ok": True},
                },
            }
        )
    )
    manifest = progress.latest_validation_manifest(tmp_path)
    assert manifest and manifest["run_id"] == "new"
    assert progress.manifest_counts(manifest) == (2, 1)


def test_progress_pass_is_task_success_alone(tmp_path: Path):
    # Matches the original pipeline: a crashed-but-completed trial still passes.
    progress = load_script("gen_progress")
    run = tmp_path / "runs" / "only"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "only",
                "updated_at": "2026-02-01",
                "results": {
                    "1": {"task_success": True, "program_ok": False},
                    "2": {"task_success": False, "program_ok": True},
                },
            }
        )
    )
    manifest = progress.latest_validation_manifest(tmp_path)
    assert manifest and progress.manifest_counts(manifest) == (2, 1)


def test_progress_filters_identity_and_prefers_full_run(tmp_path: Path):
    progress = load_script("gen_progress")
    subset = tmp_path / "runs" / "subset"
    full = tmp_path / "runs" / "full"
    for path in (subset, full):
        path.mkdir(parents=True)
    (subset / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "subset",
                "updated_at": "2026-03-01",
                "identity": {"code_sha256": "code", "seeds": [1, 2]},
                "results": {},
            }
        )
    )
    (full / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "full",
                "updated_at": "2026-02-01",
                "identity": {"code_sha256": "code", "seeds": list(range(1, 51))},
                "results": {},
            }
        )
    )
    manifest = progress.latest_validation_manifest(tmp_path, code_sha256="code")
    assert manifest and manifest["run_id"] == "full"

    mismatched = progress.latest_validation_manifest(tmp_path, code_sha256="other")
    assert mismatched is None


def test_only_full_partition_validates_stage1_evidence():
    validation = load_script("run_validation")
    assert validation.is_full_heldout({"seeds": list(range(1, 51))})
    assert not validation.is_full_heldout({"seeds": [1, 2]})


def test_selected_fix_requires_all_50_results(tmp_path: Path):
    progress = load_script("gen_progress")
    fix = tmp_path / "fix_code.py"
    fix.write_text("result = {}")
    # A finished Stage 1: Step 6 wrote a report that VALIDATES, so the only thing
    # left to decide is whether the held-out partition is complete.
    (tmp_path / "skill_report.json").write_text(json.dumps(_report()))
    assert progress.get_status(None, 0) == "pending"
    assert progress.get_status(fix, 49) == "stage1-done"
    assert progress.get_status(fix, 50) == "done"


def test_find_run_dir_prefers_cli_line_then_newest(tmp_path: Path):
    validation = load_script("run_validation")
    results = tmp_path / "results"
    printed = results / "0007" / "run-a"
    newest = results / "0007" / "run-b"
    printed.mkdir(parents=True)
    newest.mkdir(parents=True)

    found = validation.find_run_dir(f"run: {printed}\n", results, 7)
    assert found == printed

    found = validation.find_run_dir("no run line", results, 7)
    assert found in (printed, newest)

    assert validation.find_run_dir("", results, 8) is None


@pytest.mark.parametrize(
    ("arguments", "dev_seeds", "heldout_count", "unseen_seeds"),
    [
        ([], list(range(51, 66)), 50, list(range(66, 81))),
        (
            [
                "--dev-seeds",
                *map(str, range(101, 126)),
                "--heldout-count",
                "100",
                "--unseen-seeds",
                *map(str, range(126, 131)),
            ],
            list(range(101, 126)),
            100,
            list(range(126, 131)),
        ),
    ],
)
def test_init_run_seed_partitions(
    tmp_path, monkeypatch, arguments, dev_seeds, heldout_count, unseen_seeds
):
    init_run = load_script("init_run")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"benchmark": "robosuite", "suite": "cube_lifting", "task_id": 0}])
    )
    runs = tmp_path / "runs"
    monkeypatch.setattr(init_run, "RUNS_ROOT", runs)
    monkeypatch.setattr(init_run, "LATEST_LINK", runs / "LATEST")
    monkeypatch.setattr("sys.argv", ["init_run.py", "--tasks", str(tasks), *arguments])

    init_run.main()

    root = (runs / "LATEST").resolve()
    assert json.loads((root / "campaign.json").read_text()) == {
        "heldout_count": heldout_count,
        "dev_seeds": dev_seeds,
        "unseen_seeds": unseen_seeds,
    }
    campaign = load_script("campaign")
    assert campaign.heldout_seeds(root) == list(range(1, heldout_count + 1))


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dev-seeds", "0"],
        ["--dev-seeds", "-1"],
        ["--dev-seeds", "101", "101"],
        ["--dev-seeds", "125", "100", "--heldout-count", "100"],
        ["--dev-seeds", "101", "--heldout-count", "101"],
        ["--heldout-count", "0"],
        ["--unseen-seeds", "0"],
        ["--unseen-seeds", "66", "66"],
        ["--dev-seeds", "66", "--unseen-seeds", "66"],
        ["--heldout-count", "100", "--dev-seeds", "101", "--unseen-seeds", "50"],
    ],
)
def test_init_run_rejects_invalid_or_overlapping_seeds(tmp_path, monkeypatch, arguments):
    init_run = load_script("init_run")
    runs = tmp_path / "runs"
    monkeypatch.setattr(init_run, "RUNS_ROOT", runs)
    monkeypatch.setattr(init_run, "LATEST_LINK", runs / "LATEST")
    monkeypatch.setattr("sys.argv", ["init_run.py", "--tasks", "unused.json", *arguments])
    with pytest.raises(SystemExit, match=r"--dev-seeds|--heldout-count|--unseen-seeds"):
        init_run.main()
    assert not runs.exists()


def test_existing_validation_and_progress_support_100_heldout_seeds(tmp_path, monkeypatch):
    validation = load_script("run_validation")
    progress = load_script("gen_progress")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "campaign.json").write_text(
        json.dumps({"dev_seeds": list(range(101, 126)), "heldout_count": 100})
    )
    (tmp_path / "tasks.json").write_text(
        json.dumps([{"benchmark": "robosuite", "suite": "cube_lifting", "task_id": 0}])
    )
    task_dir = tmp_path / "robosuite" / "cube_lifting" / "task_0"
    task_dir.mkdir(parents=True)
    program = task_dir / "fix_code.py"
    program.write_text("result = {}\n")
    (task_dir / "skill_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "robosuite",
                "suite": "cube_lifting",
                "task_id": 0,
                "skill_library_sha": "test-sha",
                "consulted": [],
                "proposed_edits": [],
                "proposed_new": [],
            }
        )
    )
    commands = []

    def fake_run(command, **kwargs):
        if command[0] == ".venv-robosuite/bin/cap-harness":
            commands.append(command)
            seed = int(command[command.index("--seed") + 1])
            run_dir = Path(command[command.index("--output-root") + 1]) / str(seed) / "trial"
            run_dir.mkdir(parents=True)
            (run_dir / "outcome.json").write_text(
                json.dumps({"task_success": True, "program_ok": True, "steps_executed": 5})
            )
            kwargs["stdout"].write(f"run: {run_dir}\n")
        return SimpleNamespace(returncode=0, stdout="test-commit\n")

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_validation.py",
            "--run-root",
            str(tmp_path),
            "--benchmark",
            "robosuite",
            "--suite",
            "cube_lifting",
            "--task-id",
            "0",
            "--gpu",
            "2",
            "--program",
            str(program),
            "--cap-harness",
            ".venv-robosuite/bin/cap-harness",
            "--init-mode",
            "saved",
            "--max-steps",
            "1000",
        ],
    )
    assert validation.main() == 0
    assert [int(command[command.index("--seed") + 1]) for command in commands] == list(
        range(1, 101)
    )
    for command in commands:
        for flag, value in (
            ("--init-mode", "saved"),
            ("--max-steps", "1000"),
            ("--camera-width", "800"),
            ("--camera-height", "512"),
        ):
            assert command[command.index(flag) + 1] == value

    manifest_path = next((task_dir / "validation" / "runs").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["evidence_scope"] == "heldout_full"
    assert manifest["trials"] == 100
    complete_results = manifest["results"]
    monkeypatch.setattr("sys.argv", ["gen_progress.py", "--run-root", str(tmp_path)])
    for count, status in [(50, "stage1-done"), (100, "done")]:
        manifest["results"] = {
            str(seed): complete_results[str(seed)] for seed in range(1, count + 1)
        }
        manifest_path.write_text(json.dumps(manifest))
        progress.main()
        assert (
            f"| 00 task_0 | {status} | {count}/{count} (100%)"
            in (tmp_path / "progress.md").read_text()
        )


@pytest.mark.parametrize(
    ("benchmark", "grippers"),
    [("robosuite", []), ("robosuite", ["secondary"]), ("libero", ["primary"])],
)
def test_scene_snapshot_gripper_or_fixed_tool_in_sandbox(benchmark, grippers):
    from cap_harness.contracts import TaskContext
    from cap_harness.registry import ToolRegistry
    from cap_harness.runtime import ProgramExecutor

    registry = ToolRegistry(public_extension_allowlist={"robosuite.get_controller_metadata"})
    calls = []
    state = SimpleNamespace(
        base_frame="robot0_base",
        joint_positions={"primary": [0.1] * 7},
    )
    methods = {
        "get_task_context": lambda: TaskContext("suite", 0, "task", "observe", benchmark),
        "get_robot_state": lambda: state,
        "get_observation": lambda: SimpleNamespace(cameras={"agentview": None}),
        "open_gripper": lambda **kwargs: calls.append(kwargs),
        "step": lambda action: calls.append(action),
    }
    if benchmark == "robosuite":
        methods["robosuite.get_controller_metadata"] = lambda: {
            "controllable_gripper_arms": grippers
        }
    for name, method in methods.items():
        registry.register(
            method,
            name=name,
            layer="extension" if "." in name else "shared",
            capability="test",
            public=True,
        )
    result = ProgramExecutor(registry).execute_program((SCRIPTS / "scene_snapshot.py").read_text())
    assert result.ok, result.error
    assert len(calls) == 1
    if benchmark == "libero":
        assert calls == [{}]
    elif grippers:
        assert calls == [{"arm": grippers[0]}]
    else:
        command = calls[0].arms["primary"]
        assert command.mode == "joint_position"
        assert list(command.target) == state.joint_positions["primary"]
        assert command.gripper_position is None
        assert calls[0].embodiment == "robosuite"
def test_codex_entry_points_are_pointers_not_copies() -> None:
    """`.codex` registers the skills for another harness; `.claude` owns the pipeline.

    Nothing has ever checked this pair, which is how the Codex summaries drifted out of
    step with the canonical files. A pointer cannot drift; a summary can.
    """
    root = Path(__file__).resolve().parents[1]
    codex = root / ".codex" / "skills"
    claude = root / ".claude" / "skills"
    assert codex.is_dir() and claude.is_dir()

    names = sorted(p.name for p in codex.iterdir() if p.is_dir())
    assert names, "no Codex skills found"

    for name in names:
        entry = codex / name / "SKILL.md"
        canonical = claude / name / "SKILL.md"
        assert entry.is_file(), f"{name}: missing Codex entry point"
        assert canonical.is_file(), f"{name}: Codex entry point names no canonical skill"
        text = entry.read_text(encoding="utf-8")

        # The Codex manifest is the reason this folder exists; it has no `.claude` twin.
        manifest = codex / name / "agents" / "openai.yaml"
        assert manifest.is_file(), f"{name}: missing agents/openai.yaml"
        assert not (claude / name / "agents").exists(), (
            f"{name}: the Codex manifest must not be duplicated under .claude"
        )

        # Front matter must agree, or the two harnesses register different skills.
        for field in ("name", "description"):
            want = _front_matter_field(canonical.read_text(encoding="utf-8"), field)
            got = _front_matter_field(text, field)
            assert got == want, f"{name}: {field} differs between .codex and .claude"

        # It must point at the canonical file and say it is not a copy.
        assert f".claude/skills/{name}/SKILL.md" in text, (
            f"{name}: no pointer to the canonical file"
        )
        assert "pointer, not a copy" in text, f"{name}: does not declare itself a pointer"

        # It must not restate the pipeline: that is what drifted before.
        for heading in ("## Setup", "## Run Order"):
            assert heading not in text, (
                f"{name}: restates {heading}; point at the canonical file instead"
            )

    # The behavior protocol is a different one, and the entry point must say so.
    debug = (codex / "iterative-debugging" / "SKILL.md").read_text(encoding="utf-8")
    assert "behavior/SKILL.md" in debug and "different protocol" in debug


def _front_matter_field(text: str, field: str) -> str:
    """The value of one key in the leading `---` block."""
    lines = text.splitlines()
    assert lines and lines[0].strip() == "---", "file does not open with front matter"
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip().strip('"')
    raise AssertionError(f"no {field} in front matter")


def _stage2_argv(program: Path, output_root: Path, extra: list[str]) -> list[str]:
    return [
        "--suite",
        "libero_goal_swap",
        "--task-id",
        "0",
        "--gpu",
        "3",
        "--program",
        str(program),
        "--output-root",
        str(output_root),
        "--seeds",
        "1",
        "2",
        "3",
        "--init-mode",
        "seeded",
        *extra,
    ]


def _passthrough(command) -> SimpleNamespace | None:
    """Let the script's own housekeeping calls (git, gen_progress) run as no-ops."""
    if not command or Path(str(command[0])).name not in {"cap-harness", "cap-harness.exe"}:
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return None


def _finished_run(results_base: Path, seed: int, *, task_success: bool) -> Path:
    run_dir = results_base / f"{seed:04d}" / f"run-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outcome.json").write_text(
        json.dumps(
            {
                "program_ok": True,
                "task_success": task_success,
                "termination_reason": "task_succeeded" if task_success else "program_completed",
                "steps_executed": 120,
            }
        )
    )
    return run_dir


def test_stage2_defaults_to_one_process_per_seed(tmp_path, monkeypatch):
    """Keep one process per seed as the default validation mode.

    The unbatched path is the one every existing campaign was run on; adding
    a worker pool must not change what happens when nobody asks for one.
    """
    validation = load_script("run_validation")
    program = tmp_path / "fix_code.py"
    program.write_text("result = {}\n")
    output_root = tmp_path / "validation"
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        passthrough = _passthrough(command)
        if passthrough is not None:
            return passthrough
        commands.append(command)
        # Mirror what cap-harness would leave behind for this seed.
        seed = int(command[command.index("--seed") + 1])
        results_base = Path(command[command.index("--output-root") + 1])
        run_dir = _finished_run(results_base, seed, task_success=seed != 2)
        log = kwargs.get("stdout")
        if log is not None:
            log.write(f"run: {run_dir}\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validation.sys, "argv", ["run_validation.py", *_stage2_argv(program, output_root, [])]
    )
    assert validation.main() == 0

    assert [command[1] for command in commands] == ["run", "run", "run"]
    assert "run-batch" not in [command[1] for command in commands]
    manifest = json.loads(next(output_root.rglob("manifest.json")).read_text())
    assert manifest["trials"] == 3
    assert manifest["passes"] == 2


def test_stage2_workers_run_one_batch_and_score_from_the_recorded_outcomes(tmp_path, monkeypatch):
    """Score batched seeds from their individual recorded outcomes.

    With workers, the sweep is one child process -- but a seed is still
    scored from its own outcome.json, so the two paths cannot disagree.
    """
    validation = load_script("run_validation")
    program = tmp_path / "fix_code.py"
    program.write_text("result = {}\n")
    output_root = tmp_path / "validation"
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        passthrough = _passthrough(command)
        if passthrough is not None:
            return passthrough
        commands.append(command)
        results_base = Path(command[command.index("--output-root") + 1])
        results_jsonl = Path(command[command.index("--results-jsonl") + 1])
        seeds = [
            int(value)
            for value in command[command.index("--seeds") + 1 : command.index("--program")]
        ]
        results_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with results_jsonl.open("a") as stream:
            for seed in seeds:
                run_dir = _finished_run(results_base, seed, task_success=seed != 2)
                stream.write(json.dumps({"seed": seed, "run_dir": str(run_dir)}) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validation.sys,
        "argv",
        ["run_validation.py", *_stage2_argv(program, output_root, ["--workers", "4"])],
    )
    assert validation.main() == 0

    assert len(commands) == 1
    batch = commands[0]
    assert batch[1] == "run-batch"
    assert batch[batch.index("--workers") + 1] == "4"
    assert "--flat-run-dir" in batch
    assert batch[batch.index("--init-mode") + 1] == "seeded"
    manifest = json.loads(next(output_root.rglob("manifest.json")).read_text())
    assert manifest["status"] == "complete"
    assert manifest["trials"] == 3
    assert manifest["passes"] == 2
    assert manifest["results"]["2"]["task_success"] is False


def test_stage2_resume_only_asks_the_batch_for_the_seeds_it_is_missing(tmp_path, monkeypatch):
    """Resume only the seeds missing from the manifest.

    A resumed sweep must not re-run a finished seed: that is a duplicate
    trial in an immutable manifest.
    """
    validation = load_script("run_validation")
    program = tmp_path / "fix_code.py"
    program.write_text("result = {}\n")
    output_root = tmp_path / "validation"
    requested: list[list[int]] = []

    def fake_run(command, **kwargs):
        passthrough = _passthrough(command)
        if passthrough is not None:
            return passthrough
        results_base = Path(command[command.index("--output-root") + 1])
        results_jsonl = Path(command[command.index("--results-jsonl") + 1])
        seeds = [
            int(value)
            for value in command[command.index("--seeds") + 1 : command.index("--program")]
        ]
        requested.append(seeds)
        with results_jsonl.open("a") as stream:
            for seed in seeds:
                run_dir = _finished_run(results_base, seed, task_success=True)
                stream.write(json.dumps({"seed": seed, "run_dir": str(run_dir)}) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    argv = ["run_validation.py", *_stage2_argv(program, output_root, ["--workers", "2"])]
    monkeypatch.setattr(validation.sys, "argv", argv)
    assert validation.main() == 0

    monkeypatch.setattr(validation.sys, "argv", [*argv, "--resume"])
    assert validation.main() == 0

    assert requested == [[1, 2, 3]]  # the resumed pass asked for nothing


def test_stage2_reports_a_seed_the_batch_never_produced(tmp_path, monkeypatch):
    """A silently missing seed would leave a 'complete' manifest with a hole."""
    validation = load_script("run_validation")
    program = tmp_path / "fix_code.py"
    program.write_text("result = {}\n")
    output_root = tmp_path / "validation"

    def fake_run(command, **kwargs):
        passthrough = _passthrough(command)
        if passthrough is not None:
            return passthrough
        results_base = Path(command[command.index("--output-root") + 1])
        results_jsonl = Path(command[command.index("--results-jsonl") + 1])
        with results_jsonl.open("a") as stream:
            for seed in (1, 3):
                run_dir = _finished_run(results_base, seed, task_success=True)
                stream.write(json.dumps({"seed": seed, "run_dir": str(run_dir)}) + "\n")
            stream.write(json.dumps({"seed": 2, "run_dir": None}) + "\n")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(validation.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validation.sys,
        "argv",
        ["run_validation.py", *_stage2_argv(program, output_root, ["--workers", "2"])],
    )
    assert validation.main() == 1

    manifest = json.loads(next(output_root.rglob("manifest.json")).read_text())
    assert manifest["status"] == "partial"
    assert set(manifest["results"]) == {"1", "3"}


# --- skill grading, verification, and the curriculum -------------------------


def prose(path: Path) -> str:
    """Read a prompt with its hard wrapping collapsed.

    These files are wrapped at 100 columns, so a sentence a test cares about is
    routinely split across lines -- and inside a blockquote each continuation
    line carries its own "> " marker. Asserting on the raw text makes the test
    fail on rewrapping rather than on meaning.
    """
    lines = [line.lstrip().removeprefix(">").strip() for line in path.read_text().splitlines()]
    return " ".join(" ".join(lines).split())


def _task(run_root: Path, task_id: int = 0) -> Path:
    d = run_root / "libero-pro" / "libero_goal_swap" / f"task_{task_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _campaign(tmp_path: Path, task_ids=(0,)) -> Path:
    run_root = tmp_path / "campaign"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "tasks.json").write_text(
        json.dumps(
            [
                {"benchmark": "libero-pro", "suite": "libero_goal_swap", "task_id": t}
                for t in task_ids
            ]
        )
    )
    return run_root


def test_skill_library_hash_tracks_content_and_names(tmp_path: Path):
    """Change the library hash when its content or names change.

    A task records which library it ran under; that only means something if
    the hash moves when the library does.
    """
    campaign = load_script("campaign")
    library = tmp_path / "skills"
    library.mkdir(exist_ok=True)
    (library / "grasp.md").write_text("top-down grasp")
    first = campaign.skill_library_sha(library)

    assert first == campaign.skill_library_sha(library)  # stable

    (library / "grasp.md").write_text("top-down grasp, revised")
    assert campaign.skill_library_sha(library) != first

    (library / "grasp.md").write_text("top-down grasp")
    assert campaign.skill_library_sha(library) == first  # content-addressed

    # A rename changes nothing textual, but it is a different library.
    (library / "grasp.md").rename(library / "grasping.md")
    assert campaign.skill_library_sha(library) != first


def test_a_missing_library_hashes_without_crashing(tmp_path: Path):
    campaign = load_script("campaign")
    assert campaign.skill_library_sha(tmp_path / "nope")


def test_skill_state_reports_where_a_proposal_stands(tmp_path: Path):
    progress = load_script("gen_progress")
    task = _task(tmp_path)
    verification = tmp_path / "verification" / "libero-pro__libero_goal_swap__task_0"
    verification.mkdir(parents=True)

    assert progress.skill_state(task, verification) == ("missing", 0)

    # A valid report with no proposals is a real result, not a pending one.
    (task / "skill_report.json").write_text(
        json.dumps(
            {
                "skill_library_sha": "abc123def4567890",
                "consulted": [
                    {"skill": "grasp.md", "verdict": "useful", "evidence": "seeds 53, 57"}
                ],
            }
        )
    )
    assert progress.skill_state(task, verification) == ("none", 0)

    (task / "skill_report.json").write_text(
        json.dumps(
            {
                "skill_library_sha": "abc123def4567890",
                "proposed_new": [
                    {
                        "skill": "transport.md",
                        "title": "probe",
                        "trigger": "t",
                        "code": "pass",
                        "evidence": "seeds 51",
                    }
                ],
                "proposed_edits": [
                    {
                        "skill": "localize.md",
                        "section": "Disambiguation",
                        "change": "narrow the trigger",
                        "why": "it fired on the wrong scenes",
                    }
                ],
            }
        )
    )
    assert progress.skill_state(task, verification) == ("proposed", 2)

    (verification / "verification_report.json").write_text(json.dumps({"verdict": "refuted"}))
    assert progress.skill_state(task, verification) == ("refuted", 2)

    (verification / "verification_report.json").write_text(json.dumps({"verdict": "verified"}))
    assert progress.skill_state(task, verification) == ("verified", 2)


def test_a_void_verification_leaves_the_proposal_waiting(tmp_path: Path):
    """Leave a proposal undecided when verification is void.

    `void` says the run was broken, not that the skill is good or bad. It
    must not read as a decision either way.
    """
    progress = load_script("gen_progress")
    task = _task(tmp_path)
    verification = tmp_path / "verification" / "libero-pro__libero_goal_swap__task_0"
    verification.mkdir(parents=True)
    (task / "skill_report.json").write_text(
        json.dumps(
            {
                "skill_library_sha": "abc123def4567890",
                "proposed_new": [
                    {
                        "skill": "grasp.md",
                        "title": "x",
                        "trigger": "t",
                        "code": "pass",
                        "evidence": "seeds 51",
                    }
                ],
            }
        )
    )
    (verification / "verification_report.json").write_text(
        json.dumps({"verdict": "void", "notes": "sam3 was down"})
    )
    assert progress.skill_state(task, verification) == ("proposed", 1)


def test_progress_lists_tasks_whose_skills_need_a_verifier(tmp_path: Path, monkeypatch, capsys):
    progress = load_script("gen_progress")
    run_root = _campaign(tmp_path)
    task = _task(run_root)
    (task / "skill_report.json").write_text(
        json.dumps(
            {
                "skill_library_sha": "abc123def4567890",
                "proposed_new": [
                    {
                        "skill": "grasp.md",
                        "title": "top-down",
                        "trigger": "t",
                        "code": "pass",
                        "evidence": "seeds 51",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(progress.sys, "argv", ["gen_progress.py", "--run-root", str(run_root)])
    progress.main()

    written = (run_root / "progress.md").read_text()
    assert "Skill verification needed" in written
    assert "libero-pro/libero_goal_swap/task_0 (1 proposal(s))" in written
    assert "need skill verification" in capsys.readouterr().out


def _done_task(run_root: Path, task_id: int, *, passes: int, library_sha: str | None) -> Path:
    """A task with a full 50-seed held-out manifest, at a given pass count."""
    task = _task(run_root, task_id)
    (task / "fix_code.py").write_text("result = {}\n")
    code_sha = __import__("hashlib").sha256(b"result = {}\n").hexdigest()
    runs = task / "validation" / "runs" / "abc123"
    runs.mkdir(parents=True)
    (runs / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "updated_at": "2026-03-01",
                "identity": {"code_sha256": code_sha, "seeds": list(range(1, 51))},
                "results": {
                    str(seed): {"task_success": seed <= passes, "program_ok": True}
                    for seed in range(1, 51)
                },
            }
        )
    )
    if library_sha is not None:
        (task / "skill_report.json").write_text(json.dumps({"skill_library_sha": library_sha}))
    return task


def test_rerun_sweep_names_only_weak_tasks_left_behind_by_the_library(tmp_path: Path, monkeypatch):
    """Rerun weak tasks only after the skill library changes.

    Rerunning a weak task against the SAME library just reproduces it. Only
    a task whose library has since moved on is worth the GPU time.
    """
    progress = load_script("gen_progress")
    run_root = _campaign(tmp_path, task_ids=(0, 1, 2))
    current = progress.skill_library_sha()
    _done_task(run_root, 0, passes=20, library_sha="stale00000000000")  # weak, old library
    _done_task(run_root, 1, passes=48, library_sha="stale00000000000")  # strong, old library
    _done_task(run_root, 2, passes=20, library_sha=current)  # weak, current library

    monkeypatch.setattr(progress.sys, "argv", ["gen_progress.py", "--run-root", str(run_root)])
    progress.main()
    written = (run_root / "progress.md").read_text()

    assert "## Rerun sweep" in written
    assert "task_0" in written.split("## Rerun sweep")[1]
    assert "task_1" not in written.split("## Rerun sweep")[1]
    assert "task_2" not in written.split("## Rerun sweep")[1]
    # Every task is done, so the sweep is cleared to run.
    assert "Every task is done" in written


def test_the_rerun_sweep_is_withheld_until_the_campaign_finishes(tmp_path: Path, monkeypatch):
    """The sweep's whole value is that each rerun sees the final library."""
    progress = load_script("gen_progress")
    run_root = _campaign(tmp_path, task_ids=(0, 1))
    _done_task(run_root, 0, passes=20, library_sha="stale00000000000")
    _task(run_root, 1)  # still pending

    monkeypatch.setattr(progress.sys, "argv", ["gen_progress.py", "--run-root", str(run_root)])
    progress.main()
    written = (run_root / "progress.md").read_text()

    assert "Do not rerun yet" in written
    assert "Every task is done" not in written


def test_rerun_threshold_is_tunable(tmp_path: Path, monkeypatch):
    progress = load_script("gen_progress")
    run_root = _campaign(tmp_path)
    _done_task(run_root, 0, passes=45, library_sha="stale00000000000")  # 90%

    monkeypatch.setattr(
        progress.sys,
        "argv",
        ["gen_progress.py", "--run-root", str(run_root), "--rerun-threshold", "0.5"],
    )
    progress.main()
    assert "## Rerun sweep" not in (run_root / "progress.md").read_text()

    monkeypatch.setattr(
        progress.sys,
        "argv",
        ["gen_progress.py", "--run-root", str(run_root), "--rerun-threshold", "0.95"],
    )
    progress.main()
    assert "## Rerun sweep" in (run_root / "progress.md").read_text()


def test_verifier_prompt_denies_the_proposers_answer():
    """Keep the verifier isolated from the proposer's answer.

    The verifier's isolation IS the experiment. If the template stops naming
    what is off-limits, the verification silently becomes a copy.
    """
    verifier = prose(ROOT / ".claude/skills/iterative-debugging/verifier-prompt.md")

    for forbidden in ("fix_code.py", "findings.md", "task_analysis.md", "skill_report.json"):
        assert forbidden in verifier
    assert "MUST NOT" in verifier
    assert "$VERIFY_DIR/proposed_skills.md" in verifier
    # The budget is the measurement; it has to be stated.
    assert "51, 52, 53, 54, 55" in verifier
    assert "2 maximum" in verifier
    for verdict in ("verified", "refuted", "void"):
        assert f"`{verdict}`" in verifier
    # It must not overclaim what a same-task pass proves.
    assert "does **not** mean the skill generalizes" in verifier


def test_subagent_must_grade_the_skills_it_used():
    subagent = prose(ROOT / ".claude/skills/iterative-debugging/subagent-prompt.md")

    assert "skill_report.json" in subagent
    for verdict in ("useful", "misleading", "wrong", "not-applicable"):
        assert f"`{verdict}`" in subagent
    assert "--skill-library-sha" in subagent
    # A proposal is tested by an agent that has none of this task's context.
    assert "not your program" in subagent


def test_coordinator_curates_and_never_promotes_unverified_skills():
    coordinator = prose(ROOT / ".claude/skills/iterative-debugging/main-agent-prompt.md")

    assert "Only verified skills enter the library" in coordinator
    assert "curriculum.md" in coordinator
    assert "verifier-prompt.md" in coordinator
    # Verification gates promotion, not the task's own benchmark result.
    assert "gates promotion into the library, not the task's own Stage 2" in coordinator
    # A rerun must never clobber the original evidence.
    assert "rerun_02" in coordinator
    assert "never overwrites the first attempt" in coordinator


# --- reruns are scored, and verifiers cannot see the answer ------------------


def test_attempts_are_ordered_and_require_a_selected_fix(tmp_path: Path):
    progress = load_script("gen_progress")
    task = _task(tmp_path)

    assert progress.attempts(task) == []
    assert progress.latest_attempt(task) == (task, None)

    (task / "fix_code.py").write_text("result = {}\n")
    assert progress.latest_attempt(task) == (task, None)

    # A rerun directory without a selected fix is in flight, not an attempt.
    (task / "rerun_02").mkdir()
    assert progress.latest_attempt(task) == (task, None)

    (task / "rerun_02" / "fix_code.py").write_text("result = {}\n")
    assert progress.latest_attempt(task) == (task / "rerun_02", "rerun_02")

    (task / "rerun_03").mkdir()
    (task / "rerun_03" / "fix_code.py").write_text("result = {}\n")
    assert progress.latest_attempt(task) == (task / "rerun_03", "rerun_03")


def _attempt_result(attempt: Path, *, passes: int, library_sha: str) -> None:
    """Give one attempt directory a full 50-seed manifest and a skill report."""
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "fix_code.py").write_text(f"result = {{'v': {passes}}}\n")
    code_sha = __import__("hashlib").sha256((attempt / "fix_code.py").read_bytes()).hexdigest()
    runs = attempt / "validation" / "runs" / f"run{passes}"
    runs.mkdir(parents=True)
    (runs / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"run{passes}",
                "updated_at": "2026-03-01",
                "identity": {"code_sha256": code_sha, "seeds": list(range(1, 51))},
                "results": {
                    str(seed): {"task_success": seed <= passes, "program_ok": True}
                    for seed in range(1, 51)
                },
            }
        )
    )
    (attempt / "skill_report.json").write_text(json.dumps({"skill_library_sha": library_sha}))


def test_a_rerun_is_what_the_task_is_judged_by(tmp_path: Path, monkeypatch):
    """Score the task's latest rerun rather than its original attempt.

    Without this the rerun is invisible: progress keeps reporting the first
    attempt, so a sweep that improved a task looks like it did nothing.
    """
    progress = load_script("gen_progress")
    run_root = _campaign(tmp_path)
    task = _task(run_root)
    _attempt_result(task, passes=18, library_sha="stale00000000000")
    _attempt_result(task / "rerun_02", passes=46, library_sha=progress.skill_library_sha())

    monkeypatch.setattr(progress.sys, "argv", ["gen_progress.py", "--run-root", str(run_root)])
    progress.main()
    written = (run_root / "progress.md").read_text()

    assert "46/50" in written and "18/50" not in written
    assert "[rerun_02]" in written


def test_a_completed_rerun_stops_being_a_rerun_candidate(tmp_path: Path, monkeypatch):
    """Stop proposing a completed rerun against the same library.

    The stale attempt carries the old library hash forever. Scoring it would
    re-list the task on every regeneration, so the sweep never terminates.
    """
    progress = load_script("gen_progress")
    run_root = _campaign(tmp_path)
    task = _task(run_root)
    _attempt_result(task, passes=18, library_sha="stale00000000000")
    # The rerun is still weak, but it ran under the CURRENT library.
    _attempt_result(task / "rerun_02", passes=20, library_sha=progress.skill_library_sha())

    monkeypatch.setattr(progress.sys, "argv", ["gen_progress.py", "--run-root", str(run_root)])
    progress.main()

    assert "## Rerun sweep" not in (run_root / "progress.md").read_text()


def test_a_verifier_workspace_is_outside_the_task_directory(tmp_path: Path):
    """Keep verifier workspaces outside the solution directory.

    A verifier that can reach the task directory can reach the solution it
    exists to rediscover.
    """
    campaign = load_script("campaign")
    run_root = tmp_path / "campaign"
    task = campaign.task_dir(run_root, "libero-pro", "libero_goal_swap", 0)
    verify = campaign.verification_dir(run_root, "libero-pro", "libero_goal_swap", 0)

    assert task not in verify.parents and verify != task
    assert run_root in verify.parents
    # A rerun's verification does not overwrite the original's.
    rerun = campaign.verification_dir(run_root, "libero-pro", "libero_goal_swap", 0, "rerun_02")
    assert rerun != verify


def test_the_verifier_is_never_handed_the_path_it_must_not_read():
    """Naming TASK_DIR to explain what to avoid hands over the index instead."""
    verifier = prose(ROOT / ".claude/skills/iterative-debugging/verifier-prompt.md")

    assert "TASK_DIR" not in verifier
    assert "VERIFY_DIR" in verifier
    assert "outside" in verifier
    coordinator = prose(ROOT / ".claude/skills/iterative-debugging/main-agent-prompt.md")
    assert "Do not put `TASK_DIR` in the verifier's prompt" in coordinator


def test_a_worker_midway_through_stage1_is_not_mistaken_for_a_finished_one(tmp_path: Path):
    """A worker writes fix_code.py at Step 3 and skill_report.json at Step 6.

    In between, the task looked `stage1-done` with a missing report — two
    readings that are both actionable and both wrong: the coordinator would
    start a 50-seed eval on a GPU the worker still owns, and would write off
    findings that are merely not written yet.
    """
    gen = load_script("gen_progress")
    task = tmp_path / "libero-pro" / "libero_object_swap" / "task_7"
    task.mkdir(parents=True)
    (task / "fix_code.py").write_text("result = {}\n")

    assert gen.get_status(task / "fix_code.py", 0) == "stage1-running"

    # A report that does not validate is still the middle of Step 6, not the end
    # of it: the worker writes the file, runs the validator, and rewrites it until
    # it passes. Reading this as `stage1-done` started a 50-seed eval on a GPU a
    # live worker still owned.
    (task / "skill_report.json").write_text("{}")
    assert gen.get_status(task / "fix_code.py", 0) == "stage1-running"

    # Step 6 actually lands -> the task is genuinely ready for Stage 2.
    (task / "skill_report.json").write_text(json.dumps(_report()))
    assert gen.get_status(task / "fix_code.py", 0) == "stage1-done"


def test_a_finished_task_is_still_done_and_a_bare_dir_still_pending(tmp_path: Path):
    """The new state must not swallow the two it sits between."""
    gen = load_script("gen_progress")
    task = tmp_path / "task_0"
    task.mkdir()
    assert gen.get_status(None, 0) == "pending"

    (task / "fix_code.py").write_text("result = {}\n")
    # Held-out trials on disk outrank a missing report: the eval already ran.
    assert gen.get_status(task / "fix_code.py", 50) == "done"


# --- the skill-report validator ---------------------------------------------


def _report(**overrides) -> dict:
    report = {
        "schema_version": 1,
        "skill_library_sha": "abc123def4567890",
        "consulted": [
            {"skill": "grasp.md", "verdict": "useful", "evidence": "seeds 53, 57 flipped"}
        ],
        "proposed_new": [],
        "proposed_edits": [],
    }
    report.update(overrides)
    return report


def _skills(tmp_path: Path) -> Path:
    library = tmp_path / "skills"
    library.mkdir(exist_ok=True)
    for name in ("grasp.md", "localize.md", "transport.md", "manipulation.md"):
        (library / name).write_text("# skill\n")
    return library


def test_a_well_formed_report_has_no_problems(tmp_path: Path):
    campaign = load_script("campaign")
    assert campaign.validate_skill_report(_report(), _skills(tmp_path)) == []


def test_a_report_that_is_not_an_object_is_rejected(tmp_path: Path):
    campaign = load_script("campaign")
    assert campaign.validate_skill_report(["not", "a", "dict"], _skills(tmp_path))


def test_a_missing_library_hash_is_a_problem(tmp_path: Path):
    """Without it the rerun sweep cannot tell which library the task ran under."""
    campaign = load_script("campaign")
    problems = campaign.validate_skill_report(_report(skill_library_sha=""), _skills(tmp_path))
    assert any("skill_library_sha" in p for p in problems)


def test_a_grade_against_a_skill_that_does_not_exist_is_caught(tmp_path: Path):
    campaign = load_script("campaign")
    report = _report(
        consulted=[{"skill": "grasping.md", "verdict": "useful", "evidence": "x" * 50}]
    )
    problems = campaign.validate_skill_report(report, _skills(tmp_path))
    assert any("no such skill file" in p for p in problems)


def test_an_unknown_verdict_is_caught(tmp_path: Path):
    campaign = load_script("campaign")
    report = _report(consulted=[{"skill": "grasp.md", "verdict": "great", "evidence": "x" * 50}])
    problems = campaign.validate_skill_report(report, _skills(tmp_path))
    assert any("not one of" in p for p in problems)


def test_a_complaint_too_vague_to_act_on_is_caught(tmp_path: Path):
    """'grasp.md was confusing' is unactionable; the point of grading is repair."""
    campaign = load_script("campaign")
    vague = _report(consulted=[{"skill": "grasp.md", "verdict": "wrong", "evidence": "it's wrong"}])
    assert any("specifics" in p for p in campaign.validate_skill_report(vague, _skills(tmp_path)))

    specific = _report(
        consulted=[
            {
                "skill": "grasp.md",
                "verdict": "wrong",
                "evidence": "grasp.md says approach from +Z, but this handle is vertical and the "
                "grasp always slipped on seeds 52, 55, 61",
            }
        ]
    )
    assert campaign.validate_skill_report(specific, _skills(tmp_path)) == []


def test_a_proposal_missing_the_code_a_verifier_needs_is_caught(tmp_path: Path):
    campaign = load_script("campaign")
    report = _report(proposed_new=[{"skill": "transport.md", "title": "probe", "trigger": "t"}])
    problems = campaign.validate_skill_report(report, _skills(tmp_path))
    assert any("code is empty" in p for p in problems)
    assert any("evidence is empty" in p for p in problems)


def test_a_snippet_raising_an_exception_programs_cannot_use_is_caught(tmp_path: Path):
    """Reject snippets using exceptions absent from the runtime.

    Six library snippets shipped `raise RuntimeError`, which is not in the
    program builtin allowlist, so every program that copied one died with
    NameError. The validator refuses to let that in again.
    """
    campaign = load_script("campaign")
    report = _report(
        proposed_new=[
            {
                "skill": "grasp.md",
                "title": "t",
                "trigger": "t",
                "evidence": "e",
                "code": "if not found.ok:\n    raise RuntimeError('nope')",
            }
        ]
    )
    problems = campaign.validate_skill_report(report, _skills(tmp_path))
    assert any("RuntimeError" in p and "cannot use" in p for p in problems)

    report["proposed_new"][0]["code"] = "if not found.ok:\n    raise ValueError('nope')"
    assert campaign.validate_skill_report(report, _skills(tmp_path)) == []


def _proposal(**overrides) -> dict:
    entry = {
        "skill": "manipulation.md",
        "title": "t",
        "trigger": "t",
        "evidence": "e",
        "code": "z = max(float(p[2]) for p in knob_points)\nmove_to_pose(pose)",
    }
    entry.update(overrides)
    return entry


def test_a_snippet_that_computes_with_a_number_it_never_derived_is_caught(tmp_path: Path):
    """Reject snippets using scalar values they never derived.

    The gap a verifier reported as the one that would have cost it an attempt:
    `fixture_z` shipped undefined, the obvious reading (max(z)) landed on edge
    noise, and the entry then selected nothing.
    """
    campaign = load_script("campaign")
    report = _report(
        proposed_new=[
            _proposal(
                code="top = [p for p in pts if float(p[2]) >= fixture_z - 0.006]\n"
                "keep = fixture_z + 0.010 < z_top",
            )
        ]
    )
    problems = campaign.validate_skill_report(report, _skills(tmp_path))
    assert any("fixture_z" in p and "guess the derivation" in p for p in problems)


def test_a_container_the_snippet_only_iterates_is_not_a_placeholder(tmp_path: Path):
    """Allow placeholder containers that are only iterated.

    A fragment is allowed to be a fragment. `knob_points` says what it is;
    the check is for scalars whose *derivation* is the content.
    """
    campaign = load_script("campaign")
    assert (
        campaign.validate_skill_report(_report(proposed_new=[_proposal()]), _skills(tmp_path)) == []
    )


def test_naming_the_placeholder_in_the_trigger_clears_it(tmp_path: Path):
    campaign = load_script("campaign")
    report = _report(
        proposed_new=[
            _proposal(
                code="z_top = max(float(p[2]) for p in cand)\nkeep = fixture_z + 0.010 < z_top",
                trigger="fixture_z is the fixture's robust top, quantile(z, 0.9) — never max(z).",
            )
        ]
    )
    assert campaign.validate_skill_report(report, _skills(tmp_path)) == []


def test_an_attribute_read_is_not_a_derived_scalar(tmp_path: Path):
    """`not found.ok` depends on the attribute, not on deriving `found`."""
    campaign = load_script("campaign")
    report = _report(proposed_new=[_proposal(code="if not found.ok:\n    result = {}")])
    assert campaign.validate_skill_report(report, _skills(tmp_path)) == []


def test_a_localize_proposal_that_names_no_prompt_is_caught(tmp_path: Path):
    """Require localization proposals to identify their prompts.

    The entry this comes from existed *because* a prompt was unreliable, and
    named no prompt at all — so the library's registry stayed empty and a
    verifier had to measure the prompts itself.
    """
    campaign = load_script("campaign")
    report = _report(
        proposed_new=[
            _proposal(
                skill="localize.md",
                code="for cloud in candidates:\n    keep = cloud",
            )
        ]
    )
    assert any(
        "names no prompt" in p for p in campaign.validate_skill_report(report, _skills(tmp_path))
    )


def test_a_prose_quote_is_not_a_prompt(tmp_path: Path):
    """Distinguish prose quotations from executable prompts.

    Prose quotes English phrases constantly; only a literal in the code is
    something a reader can copy into segment_text.
    """
    campaign = load_script("campaign")
    quoted = _proposal(
        skill="localize.md",
        code="for cloud in candidates:\n    keep = cloud",
        trigger="The tempting rule, 'the candidate nearest the fixture centre', picks wrong.",
    )
    assert any(
        "names no prompt" in p
        for p in campaign.validate_skill_report(_report(proposed_new=[quoted]), _skills(tmp_path))
    )

    named = dict(quoted, code='found = segment_text("agentview", "black knob")\nkeep = found')
    assert campaign.validate_skill_report(_report(proposed_new=[named]), _skills(tmp_path)) == []


def test_a_camera_name_alone_does_not_count_as_a_prompt(tmp_path: Path):
    campaign = load_script("campaign")
    report = _report(
        proposed_new=[
            _proposal(
                skill="localize.md",
                code='cloud = mask_to_point_cloud(mask, "agentview")',
            )
        ]
    )
    assert any(
        "names no prompt" in p for p in campaign.validate_skill_report(report, _skills(tmp_path))
    )


def test_a_snippet_that_does_not_parse_is_caught(tmp_path: Path):
    campaign = load_script("campaign")
    report = _report(proposed_new=[_proposal(code="| moka pot | 'coffee maker' (0.42) |")])
    assert any(
        "does not parse" in p for p in campaign.validate_skill_report(report, _skills(tmp_path))
    )


def test_the_duplicated_builtin_allowlist_matches_the_runtime(tmp_path: Path):
    """Keep the duplicated builtin allowlist aligned with the runtime.

    Same duplication, same reason, as the exception allowlist below: a
    snippet using a builtin the runtime does not inject dies with NameError.
    """
    from cap_harness.runtime import _SAFE_BUILTINS, _SAFE_CONSTRUCTORS

    campaign = load_script("campaign")
    non_exceptions = {
        name
        for name, value in _SAFE_BUILTINS.items()
        if not (isinstance(value, type) and issubclass(value, BaseException))
        # __import__ is interpreter machinery for the allowlisted `import math`,
        # not a public callable that skill snippets may invoke directly.
        and name != "__import__"
    }
    assert set(campaign.PROGRAM_BUILTINS) == non_exceptions
    assert set(campaign.PROGRAM_CONSTRUCTORS) == set(_SAFE_CONSTRUCTORS)


def test_the_duplicated_exception_allowlist_matches_the_runtime(tmp_path: Path):
    """Keep the duplicated exception allowlist aligned with the runtime.

    campaign.py runs under a bare python3 that cannot import cap_harness, so
    the allowlist is duplicated. This fails if the two ever drift apart.
    """
    from cap_harness.runtime import _SAFE_BUILTINS

    campaign = load_script("campaign")
    runtime_exceptions = {
        name
        for name, value in _SAFE_BUILTINS.items()
        if isinstance(value, type) and issubclass(value, BaseException)
    }
    assert set(campaign.PROGRAM_EXCEPTIONS) == runtime_exceptions


def test_an_unusable_report_is_visible_rather_than_reading_as_nothing_proposed(tmp_path: Path):
    """Report unusable findings instead of treating them as no proposal.

    The silent failure this exists to stop: a malformed report degrading to
    'this task had nothing to say'.
    """
    progress = load_script("gen_progress")
    task = _task(tmp_path)
    verification = tmp_path / "verification" / "v"
    verification.mkdir(parents=True)

    assert progress.skill_state(task, verification) == ("missing", 0)

    (task / "skill_report.json").write_text("{not json")
    assert progress.skill_state(task, verification) == ("invalid", 0)

    (task / "skill_report.json").write_text(json.dumps({"consulted": [], "proposed_new": []}))
    assert progress.skill_state(task, verification) == ("invalid", 0)  # no library hash

    (task / "skill_report.json").write_text(
        json.dumps({"skill_library_sha": "abc123", "consulted": [], "proposed_new": []})
    )
    assert progress.skill_state(task, verification) == ("none", 0)
