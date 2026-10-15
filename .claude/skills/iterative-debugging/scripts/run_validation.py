#!/usr/bin/env python3
"""Run an immutable, resumable held-out validation for one debugged task.

A run identity hashes the program, benchmark, suite, task id, seeds, init mode,
and capture settings. Results from different identities are never mixed.
``--resume`` continues only the matching manifest.

Each held-out seed is executed in a fresh process with ``cap-harness run`` and
scored from the finalized ``outcome.json``: a seed passes on ``task_success``
alone (matching the original pipeline, where a crashed-but-completed trial
still counted); ``program_ok`` is recorded for triage.

``--workers N`` (N > 1) runs the sweep through ``cap-harness run-batch``
instead, where seeds share worker processes but each still builds and closes
its own environment. Scoring is unchanged: both paths read the same
``outcome.json`` from disk, so they cannot disagree about what a seed did.
Confirm the two agree on the target machine first --
``scripts/check_batch_equivalence.py``.
"""

# This is a standalone CLI script; progress/status is reported to stdout by design.

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import ROOT, heldout_seeds, resolve_run_root
from campaign import task_dir as campaign_task_dir

# Historical default partition; the effective one comes from the campaign's
# campaign.json (init_run.py --heldout-count), see main().
HELDOUT_SEEDS = list(range(1, 51))
RUN_LINE_RE = re.compile(r"^run: (.+)$", re.MULTILINE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def benchmark_environment(benchmark: str, gpu: int | str, *, root: Path) -> dict[str, str]:
    """Per-benchmark process environment for one episode on one GPU.

    MuJoCo benchmarks render through EGL and must be told the EGL device explicitly (EGL
    enumeration ignores CUDA_VISIBLE_DEVICES). BEHAVIOR runs Isaac Sim, which picks its GPU
    from OMNIGIBSON_GPU_ID and needs the dataset and appdata roots; MuJoCo variables are
    irrelevant there and are not set.
    """
    if benchmark == "behavior":
        home = Path(os.environ.get("HOME", "~")).expanduser()
        return {
            "OMNIGIBSON_HEADLESS": "1",
            "OMNIGIBSON_GPU_ID": str(gpu),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMNIGIBSON_DATA_PATH": os.environ.get(
                "OMNIGIBSON_DATA_PATH",
                os.environ.get("CAP_HARNESS_BEHAVIOR_DATA", str(home / "behavior-data")),
            ),
            "OMNIGIBSON_APPDATA_PATH": os.environ.get(
                "OMNIGIBSON_APPDATA_PATH", str(root / ".behavior-appdata")
            ),
            "PYTHONHASHSEED": "0",
        }
    return {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "EGL_PLATFORM": "device",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # EGL device enumeration ignores CUDA_VISIBLE_DEVICES, so without this MuJoCo
        # renders every episode on physical GPU 0 no matter which GPU was assigned.
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }


def build_identity(
    *,
    benchmark: str,
    suite: str,
    task_id: int,
    program: Path,
    seeds: list[int],
    init_mode: str,
    max_steps: int,
    camera_width: int,
    camera_height: int,
) -> dict:
    return {
        "benchmark": benchmark,
        "suite": suite,
        "task_id": task_id,
        "code_sha256": sha256_file(program),
        "seeds": sorted(set(seeds)),
        "init_mode": init_mode,
        "max_steps": max_steps,
        "camera_width": camera_width,
        "camera_height": camera_height,
    }


def run_id_for_identity(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def is_full_heldout(identity: dict, heldout: list[int] | None = None) -> bool:
    return identity.get("seeds") == (heldout if heldout is not None else HELDOUT_SEEDS)


def _stated_horizon(benchmark: str, suite: str, task_id: int) -> int:
    """The episode length the benchmark itself states, else the flat default.

    Mirrors `cap_harness.batch._task_horizon`; imported lazily so this script
    keeps working where that registry is not importable.
    """
    try:
        if benchmark == "robosuite":
            from cap_harness.robosuite.registry import RobosuiteTaskRegistry

            stated = RobosuiteTaskRegistry().resolve(suite, task_id).horizon
        else:
            stated = None
    except ImportError as exc:
        # Do NOT swallow this. Run under the benchmark's own interpreter
        # (e.g. .venv-robosuite/bin/python); under a bare `python3` the registry is
        # not importable, the horizon silently falls back to 1000, and Stage 2 --
        # the measurement -- may measure a task at the wrong episode length.
        print(
            f"WARNING: cannot import the {benchmark} registry ({exc}); falling back to 1000 "
            "ticks. Re-run with the benchmark venv's python if this task states a horizon.",
            file=sys.stderr,
        )
        stated = None
    # Optional horizon lookup has a documented fallback and warning.
    except Exception as exc:
        print(f"WARNING: horizon lookup failed for {benchmark}/{suite}: {exc}", file=sys.stderr)
        stated = None
    return int(stated) if stated else 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="libero-pro")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--program", type=Path, required=True, help="frozen fix_code.py")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="campaign root (default: outputs/runs/LATEST)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="override validation output dir (default: <run-root>/<task>/validation)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="held-out seeds (default: the campaign's 1..N partition)",
    )
    parser.add_argument("--resume", action="store_true", help="Continue this exact run identity")
    parser.add_argument(
        "--init-mode",
        default=None,
        choices=("saved", "seeded"),
        help="default: seeded for libero-pro, saved for every other benchmark",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "episode length in ticks; defaults to the horizon the task's own benchmark "
            "states, falling back to 1000 where none is stated"
        ),
    )
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=512)
    parser.add_argument(
        "--cap-harness",
        default="cap-harness",
        help="cap-harness executable (e.g. .venv-libero/bin/cap-harness)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "seeds to run concurrently on this GPU (default: 1). 1 keeps the "
            "original one-process-per-seed loop unchanged; above 1 runs the sweep "
            "through `cap-harness run-batch`, where each seed still builds its own "
            "environment but workers are reused across seeds"
        ),
    )
    parser.add_argument(
        "--recycle-after",
        type=int,
        default=10,
        metavar="N",
        help="retire a batch worker after N seeds (only with --workers > 1)",
    )
    args = parser.parse_args()
    if args.max_steps is None:
        # Stage 2 IS the measurement, so it must not re-introduce the flat budget
        # the registry exists to replace: a hardcoded 1000 hands `close_drawer`
        # (450) more than twice its stated episode, and that difference reads as
        # a result rather than as a setting.
        args.max_steps = _stated_horizon(args.benchmark, args.suite, args.task_id)
        print(f"--max-steps defaulted to the stated horizon: {args.max_steps}", file=sys.stderr)
    return args


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def read_seed_outcome(run_dir: Path) -> dict | None:
    outcome_path = run_dir / "outcome.json"
    if not outcome_path.is_file():
        return None
    try:
        outcome = json.loads(outcome_path.read_text())
    except (OSError, ValueError):
        return None
    return {
        "program_ok": bool(outcome.get("program_ok")),
        "task_success": bool(outcome.get("task_success")),
        "termination_reason": outcome.get("termination_reason"),
        "steps_executed": outcome.get("steps_executed"),
        "run_dir": str(run_dir),
    }


def find_run_dir(log_text: str, results_base: Path, seed: int) -> Path | None:
    """Locate the run directory: prefer the CLI's ``run: <dir>`` line, else newest on disk."""
    matches = RUN_LINE_RE.findall(log_text)
    if matches:
        candidate = Path(matches[-1].strip())
        if candidate.is_dir():
            return candidate
    seed_dirs = [path for path in results_base.glob(f"{seed:04d}/*") if path.is_dir()]
    if not seed_dirs:
        return None
    return max(seed_dirs, key=lambda path: path.stat().st_mtime)


def update_stage1_validation(program: Path, manifest: dict, manifest_path: Path) -> None:
    stage1_path = program.parent / "stage1_result.json"
    validation_path = program.parent / "validation_result.json"
    summary = {
        "schema_version": 1,
        "state": "validated",
        "run_id": manifest["run_id"],
        "manifest": str(manifest_path),
        "code_sha256": manifest["identity"]["code_sha256"],
        "seeds": manifest["identity"]["seeds"],
        "passes": manifest["passes"],
        "trials": manifest["trials"],
        "pass_rate": manifest["pass_rate"],
        "validated_at": manifest["updated_at"],
    }
    # Do not let a different-identity run (e.g. a diagnostic at another camera
    # size) silently clobber the canonical sidecar. Same run_id updates are
    # idempotent (resume); a differing run_id is surfaced and left untouched.
    if validation_path.exists():
        existing = json.loads(validation_path.read_text())
        if existing.get("run_id") != summary["run_id"]:
            print(
                f"validation_result.json already records run {existing.get('run_id')} "
                f"(different identity than {summary['run_id']}); leaving it untouched. "
                "Remove it to replace with this run.",
                flush=True,
            )
            return
    write_json_atomic(validation_path, summary)
    if not stage1_path.exists():
        return
    stage1 = json.loads(stage1_path.read_text())
    if stage1.get("code_sha256") != summary["code_sha256"]:
        return
    stage1["validation_state"] = "validated"
    stage1["validation"] = summary
    stage1["updated_at"] = now()
    write_json_atomic(stage1_path, stage1)


def record_trial(
    manifest: dict,
    manifest_path: Path,
    *,
    seed: int,
    run_dir: Path | None,
    exit_code: int,
    log_path: Path | None = None,
) -> bool:
    """Score one finished seed from its own ``outcome.json`` and persist it.

    Both paths score here, from the artifact on disk rather than from whatever
    the runner reported, so a batched sweep and a per-seed loop cannot disagree
    about what a seed did.
    """
    trial = read_seed_outcome(run_dir) if run_dir else None
    if trial is None:
        print(f"seed {seed:02d}: NO ARTIFACT exit={exit_code}", flush=True)
        return False
    trial["seed"] = seed
    trial["process_exit_code"] = exit_code
    if log_path is not None and log_path.is_file():
        log_text = log_path.read_text(errors="replace")
        trial["kit_error_lines"] = log_text.count("[Error]")
        trial["device_assert"] = "device-side assert" in log_text
    trial["recorded_at"] = now()
    manifest["results"][str(seed)] = trial
    manifest["trials"] = len(manifest["results"])
    manifest["passes"] = sum(int(bool(row["task_success"])) for row in manifest["results"].values())
    manifest["pass_rate"] = round(manifest["passes"] / manifest["trials"], 6)
    manifest["updated_at"] = now()
    write_json_atomic(manifest_path, manifest)
    print(
        f"seed {seed:02d}: task_success={trial['task_success']} "
        f"program_ok={trial['program_ok']} exit={exit_code}",
        flush=True,
    )
    return True


def run_batched(
    args: argparse.Namespace,
    *,
    manifest: dict,
    manifest_path: Path,
    program: Path,
    pending: list[int],
    results_base: Path,
    logs: Path,
    env: dict,
) -> int:
    """Run the pending seeds through ``cap-harness run-batch`` and fold in the results.

    The sweep is one child process that reuses its workers across seeds; each
    seed still constructs and tears down its own environment there, and each
    still lands in its own ``<results_base>/<seed>/<run-id>`` directory with its
    own log. Only the interpreter start is shared.
    """
    if not pending:
        return 0
    results_jsonl = manifest_path.parent / "batch-results.jsonl"
    command = [
        args.cap_harness,
        "run-batch",
        "--benchmark",
        args.benchmark,
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--seeds",
        *[str(seed) for seed in pending],
        "--program",
        str(program),
        "--output-root",
        str(results_base),
        "--max-steps",
        str(args.max_steps),
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--init-mode",
        args.init_mode,
        "--flat-run-dir",
        "--workers",
        str(args.workers),
        "--recycle-after",
        str(args.recycle_after),
        "--log-dir",
        str(logs),
        "--results-jsonl",
        str(results_jsonl),
        "--summary",
        str(manifest_path.parent / "batch.json"),
    ]
    print(
        f"batch: {len(pending)} seed(s) on {args.workers} worker(s) -> {logs}",
        flush=True,
    )
    subprocess.run(command, env=env, check=False)

    # Read what landed rather than what was asked for: a sweep killed partway
    # still has every seed it finished, and resume needs exactly those.
    recorded: dict[int, Path | None] = {}
    if results_jsonl.is_file():
        for line in results_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            run_dir = record.get("run_dir")
            recorded[int(record["seed"])] = (
                Path(run_dir) if run_dir else None,
                # The batch reports what actually happened; hardcoding 0 made
                # every manifest row read as a clean exit, so triage could not
                # tell a crashed seed from a clean one.
                0 if record.get("ok") else 1,
            )

    missing = 0
    for seed in pending:
        run_dir, exit_code = recorded.get(seed, (None, 1))
        if run_dir is None:
            # The batch reported nothing usable; fall back to what is on disk.
            run_dir = find_run_dir("", results_base, seed)
        if not record_trial(
            manifest,
            manifest_path,
            seed=seed,
            run_dir=run_dir,
            exit_code=exit_code,
            log_path=logs / f"seed_{seed:02d}.log",
        ):
            missing += 1
    return missing


def finish(
    args: argparse.Namespace,
    manifest: dict,
    manifest_path: Path,
    program: Path,
    seeds: list[int],
    missing_artifacts: int,
    run_root: Path | None,
) -> int:
    """Seal the manifest, refresh campaign progress, and report."""
    del args  # Retain the shared stage-completion calling convention.
    manifest["trials"] = len(manifest["results"])
    manifest["passes"] = sum(int(bool(row["task_success"])) for row in manifest["results"].values())
    manifest["pass_rate"] = (
        round(manifest["passes"] / manifest["trials"], 6) if manifest["trials"] else 0.0
    )
    manifest["status"] = "complete" if manifest["trials"] == len(seeds) else "partial"
    manifest["updated_at"] = now()
    write_json_atomic(manifest_path, manifest)

    if manifest["status"] == "complete" and is_full_heldout(
        manifest["identity"], heldout_seeds(run_root)
    ):
        update_stage1_validation(program, manifest, manifest_path)
    # Only regenerate progress for a known campaign. With a bare --output-root
    # (e.g. evosearch's shared eval dir) there is no campaign to update, and
    # spawning gen_progress with no root would rewrite an unrelated LATEST.
    if run_root is not None:
        subprocess.run(
            [
                "python3",
                str(Path(__file__).resolve().parent / "gen_progress.py"),
                "--run-root",
                str(run_root),
            ],
            check=False,
        )
    print(
        f"run={manifest['run_id']} status={manifest['status']} passes={manifest['passes']}/"
        f"{manifest['trials']} manifest={manifest_path}"
    )
    return 1 if missing_artifacts or manifest["status"] != "complete" else 0


def main() -> int:
    args = parse_args()
    if args.init_mode is None:
        args.init_mode = "seeded" if args.benchmark == "libero-pro" else "saved"
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.workers > 1 and args.benchmark not in {"libero-pro", "robosuite"}:
        raise SystemExit("worker reuse only supports libero-pro and robosuite; use --workers 1")
    if args.init_mode != "saved" and args.benchmark != "libero-pro":
        raise SystemExit(
            "init_mode 'seeded' is only supported for the libero-pro benchmark; "
            "pass --init-mode saved for other benchmarks"
        )
    # Resolve caller-relative paths before chdir, so relative --program /
    # --output-root are interpreted against the invoking cwd, not ROOT.
    program = args.program.resolve()
    run_root = resolve_run_root(args.run_root) if args.run_root is not None else None
    if args.output_root is not None:
        output_root = args.output_root.resolve()
    else:
        if run_root is None:
            run_root = resolve_run_root(None)
        output_root = (
            campaign_task_dir(run_root, args.benchmark, args.suite, args.task_id) / "validation"
        )
    os.chdir(ROOT)
    heldout = heldout_seeds(run_root)
    seeds = sorted(set(args.seeds)) if args.seeds else list(heldout)

    if not program.is_file():
        raise SystemExit(f"program does not exist: {program}")
    invalid = [seed for seed in seeds if seed not in heldout]
    if invalid:
        raise SystemExit(f"held-out seeds must be in 1..{heldout[-1]}: {invalid}")

    identity = build_identity(
        benchmark=args.benchmark,
        suite=args.suite,
        task_id=args.task_id,
        program=program,
        seeds=seeds,
        init_mode=args.init_mode,
        max_steps=args.max_steps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    run_id = run_id_for_identity(identity)
    if args.output_root is not None:
        # Explicit output roots (e.g. evosearch's shared eval dir) keep the
        # legacy per-task nesting so manifests from different tasks stay apart.
        run_dir = (
            output_root / args.benchmark / args.suite / f"task_{args.task_id}" / "runs" / run_id
        )
    else:
        run_dir = output_root / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    results_base = run_dir / "results"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise SystemExit(f"manifest identity mismatch: {manifest_path}")
        if not args.resume:
            raise SystemExit(f"immutable run already exists; pass --resume: {manifest_path}")
    else:
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "identity": identity,
            "program_path": str(program),
            "git_commit": git_commit(),
            "created_at": now(),
            "updated_at": now(),
            "status": "running",
            "evidence_scope": (
                "heldout_full" if is_full_heldout(identity, heldout) else "heldout_subset"
            ),
            "results": {},
            "passes": 0,
            "trials": 0,
            "pass_rate": 0.0,
        }
        write_json_atomic(manifest_path, manifest)

    env = os.environ.copy()
    env.update(benchmark_environment(args.benchmark, args.gpu, root=ROOT))
    # LIBERO prompts on stdin at import when this is unset, and a child with no
    # terminal dies with a bare `EOFError: EOF when reading a line`. The
    # documented flow exports it, but an eval launched from a bare shell did not
    # and lost every seed to an error naming nothing.
    env.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
    pending = [seed for seed in seeds if str(seed) not in manifest["results"]]
    for seed in seeds:
        if seed not in pending:
            print(f"seed {seed:02d}: SKIP matching manifest")

    if args.workers > 1:
        missing_artifacts = run_batched(
            args,
            manifest=manifest,
            manifest_path=manifest_path,
            program=program,
            pending=pending,
            results_base=results_base,
            logs=logs,
            env=env,
        )
        return finish(args, manifest, manifest_path, program, seeds, missing_artifacts, run_root)

    missing_artifacts = 0
    for seed in pending:
        log_path = logs / f"seed_{seed:02d}.log"
        command = [
            args.cap_harness,
            "run",
            "--benchmark",
            args.benchmark,
            "--suite",
            args.suite,
            "--task-id",
            str(args.task_id),
            "--seed",
            str(seed),
            "--program",
            str(program),
            "--output-root",
            str(results_base),
            "--max-steps",
            str(args.max_steps),
            "--camera-width",
            str(args.camera_width),
            "--camera-height",
            str(args.camera_height),
            "--init-mode",
            args.init_mode,
            "--flat-run-dir",
        ]
        print(f"seed {seed:02d}: RUN -> {log_path}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
            )
        seed_run_dir = find_run_dir(log_path.read_text(), results_base, seed)
        if not record_trial(
            manifest,
            manifest_path,
            seed=seed,
            run_dir=seed_run_dir,
            exit_code=process.returncode,
            log_path=log_path,
        ):
            missing_artifacts += 1

    return finish(args, manifest, manifest_path, program, seeds, missing_artifacts, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
