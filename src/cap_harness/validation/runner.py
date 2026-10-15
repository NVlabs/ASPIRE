"""Unified validation runner: executes a plan's structural and program cases.

- structural cases run through the per-embodiment selector (generic reset/step
  invariants, or the LIBERO matrix).
- program cases run each seed in a fresh ``cap-harness run`` process, read the
  run's ``outcome.json``, and pass when at least ``min_success`` seeds satisfy the
  case's ``require`` gates (this is what encodes the accepted floors, e.g.
  Handover 1/3).
- when ``plan.sealed`` is set, the runner writes a sealed ``environment.json``
  (refusing a dirty worktree), injects its hash into every run, and verifies each
  run's provenance, capture profile, termination, and required videos — the same
  guarantees the RoboSuite acceptance campaign provided.

Renderer/GL/GPU environment and any parallel per-GPU fan-out are the wrapper's
job (they must be set before the interpreter starts); the runner inherits the
ambient environment and only adds ``CAP_HARNESS_ENVIRONMENT_SHA256`` when sealed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .model import CaseResult, SeedResult, ValidationCase, ValidationPlan
from .plan import load_plan
from .provenance import check_program_provenance, seal_environment, sha256_file

RunCommand = Callable[[list[str], Mapping[str, str]], int]

REQUIRED_VIDEOS: dict[str, set[str]] = {
    "robosuite": {"agentview.mp4", "robot0_eye_in_hand.mp4", "robot1_eye_in_hand.mp4"},
    "behavior": {"head.mp4", "left_wrist.mp4", "right_wrist.mp4"},
}
_SEALED_CAPTURE = {
    "profile": "balanced_evidence",
    "camera_width": 512,
    "camera_height": 512,
    "videos": True,
}


def _default_run_command(argv: list[str], env: Mapping[str, str]) -> int:
    return subprocess.run(argv, env=dict(env), check=False).returncode


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _find_run_dir(root: Path) -> Path | None:
    outcomes = sorted(root.rglob("outcome.json"))
    if not outcomes:
        return None
    return outcomes[-1].parent


def _evaluate_require(outcome: Mapping[str, Any], require: Mapping[str, bool]) -> list[str]:
    values = {
        "program_ok": outcome.get("program_ok"),
        "task_success": outcome.get("task_success"),
        "evaluator_success": outcome.get("protocol_success"),
    }
    issues: list[str] = []
    for key, expected in require.items():
        if bool(values.get(key)) is not bool(expected):
            issues.append(f"{key}={values.get(key)!r} expected {expected!r}")
    return issues


def _verify_sealed_run(
    run_dir: Path,
    case: ValidationCase,
    *,
    benchmark: str = "robosuite",
    program_sha256: str,
    commit: str,
    dependency_lock_sha256: str | None,
    environment_sha256: str,
) -> list[str]:
    issues: list[str] = []
    run = _read_json(run_dir / "run.json")
    if run is not None:
        if run.get("environment_sha256") != environment_sha256:
            issues.append("run environment_sha256 does not match the sealed environment")
        capture = run.get("capture")
        if not isinstance(capture, Mapping) or any(
            capture.get(k) != v for k, v in {**_SEALED_CAPTURE, "max_steps": case.max_steps}.items()
        ):
            issues.append("run capture profile does not match the sealed plan")
    outcome = _read_json(run_dir / "outcome.json") or {}
    if outcome.get("termination_reason") != "task_succeeded":
        issues.append("termination_reason is not task_succeeded")
    if outcome.get("finalization_errors") not in ([], None):
        issues.append("run reported finalization errors")
    provenance = _read_json(run_dir / "source/provenance.json")
    if provenance is not None:
        issues.extend(
            check_program_provenance(
                provenance,
                program_sha256=program_sha256,
                commit=commit,
                dependency_lock_sha256=dependency_lock_sha256,
            )
        )
    required_videos = REQUIRED_VIDEOS.get(
        benchmark, {"agentview.mp4", "robot0_eye_in_hand.mp4", "robot1_eye_in_hand.mp4"}
    )
    videos_dir = run_dir / "media/videos"
    actual_videos = {path.name for path in videos_dir.glob("*.mp4")}
    if actual_videos != required_videos:
        issues.append(f"videos {sorted(actual_videos)} != required {sorted(required_videos)}")
    elif any((videos_dir / video).stat().st_size <= 0 for video in actual_videos):
        issues.append("a required video is empty")
    video_index = _read_json(run_dir / "media/video-index.json")
    if video_index is not None:
        frames = video_index.get("frames")
        expected_frame_keys = {name.removesuffix(".mp4") for name in required_videos}
        if (
            not isinstance(frames, Mapping)
            or set(frames) != expected_frame_keys
            or any(not isinstance(c, int) or isinstance(c, bool) or c <= 0 for c in frames.values())
        ):
            issues.append("video-index frames do not cover the required cameras")
    return issues


def _run_program_case(
    plan: ValidationPlan,
    case: ValidationCase,
    output_dir: Path,
    *,
    repo_root: Path,
    run_command: RunCommand,
    base_env: dict[str, str],
    seal: Mapping[str, Any] | None,
) -> CaseResult:
    case_dir = output_dir / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    program_sha256 = sha256_file(repo_root / case.program) if case.program else ""
    seed_results: list[SeedResult] = []
    for seed in case.seeds:
        seed_root = case_dir / "runs" / str(seed)
        seed_root.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            "-m",
            "cap_harness.cli",
            "run",
            "--benchmark",
            plan.benchmark,
            "--suite",
            str(case.suite),
            "--task-id",
            str(case.task_id),
            "--seed",
            str(seed),
            "--program",
            str(repo_root / case.program),
            "--output-root",
            str(seed_root),
            "--flat-run-dir",
            "--max-steps",
            str(case.max_steps),
        ]
        if seal is not None:
            argv += ["--camera-width", "512", "--camera-height", "512"]
        env = dict(base_env)
        if seal is not None:
            env["CAP_HARNESS_ENVIRONMENT_SHA256"] = seal["environment_sha256"]
        code = run_command(argv, env)
        run_dir = _find_run_dir(seed_root)
        detail: dict[str, Any] = {"exit_code": code}
        if run_dir is None:
            detail["error"] = "no outcome.json produced"
            seed_results.append(SeedResult(seed=seed, passed=False, detail=detail))
            continue
        outcome = _read_json(run_dir / "outcome.json") or {}
        issues = _evaluate_require(outcome, case.require)
        if seal is not None:
            issues += _verify_sealed_run(
                run_dir,
                case,
                benchmark=plan.benchmark,
                program_sha256=program_sha256,
                commit=seal["commit"],
                dependency_lock_sha256=seal["dependency_lock_sha256"],
                environment_sha256=seal["environment_sha256"],
            )
        detail["run_dir"] = str(run_dir)
        if issues:
            detail["issues"] = issues
        seed_results.append(SeedResult(seed=seed, passed=not issues, detail=detail))
    passed = sum(1 for r in seed_results if r.passed)
    result = CaseResult(
        id=case.id,
        kind="program",
        passed=passed >= case.required_success,
        passed_count=passed,
        required_count=case.required_success,
        total=len(case.seeds),
        seed_results=tuple(seed_results),
    )
    from cap_harness.validation import atomic_write_json

    atomic_write_json(case_dir / "summary.json", result.to_dict())
    return result


def run_plan(
    plan: ValidationPlan | str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    retry_failures: bool = False,
    run_command: RunCommand | None = None,
) -> dict[str, Any]:
    """Execute every case in a plan and write a normalized ``summary.json``."""
    if not isinstance(plan, ValidationPlan):
        plan = load_plan(plan)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root).resolve()
    run_command = run_command or _default_run_command
    base_env = dict(os.environ)

    seal: dict[str, Any] | None = None
    if plan.sealed:
        renderer = plan.renderer or "egl"
        env_path, environment_sha256, commit = seal_environment(
            output_dir, repo_root, renderer=renderer
        )
        dependency_lock = repo_root / "configs/dependency-lock.json"
        seal = {
            "environment_sha256": environment_sha256,
            "commit": commit,
            "dependency_lock_sha256": (
                sha256_file(dependency_lock) if dependency_lock.is_file() else None
            ),
            "environment_path": str(env_path),
        }

    results: list[CaseResult] = []
    for case in plan.cases:
        if case.kind == "structural":
            from .selectors import get_structural_selector

            selector = get_structural_selector(plan.benchmark)
            results.append(selector.run(case, output_dir, retry_failures=retry_failures))
        else:
            results.append(
                _run_program_case(
                    plan,
                    case,
                    output_dir,
                    repo_root=repo_root,
                    run_command=run_command,
                    base_env=base_env,
                    seal=seal,
                )
            )

    success = all(r.passed for r in results)
    summary = {
        "schema_version": 1,
        "plan": plan.name,
        "benchmark": plan.benchmark,
        "sealed": plan.sealed,
        "success": success,
        "cases": [r.to_dict() for r in results],
        "required_providers": list(plan.required_providers),
    }
    from cap_harness.validation import atomic_write_json

    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = ["RunCommand", "run_plan"]
