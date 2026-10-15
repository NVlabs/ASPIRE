"""Command-line entry point for environment and LIBERO release validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

# Stdlib-only at import time; the simulator stack is loaded inside batch workers.
from cap_harness.batch import DEFAULT_RECYCLE_AFTER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cap-harness",
        description="Diagnose and validate the LIBERO-Pro CaP harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="check the host and a simulator/provider environment"
    )
    doctor.add_argument(
        "--output",
        type=Path,
        default=Path("environment.json"),
        help="structured report path (default: environment.json)",
    )
    doctor.add_argument(
        "--libero-submodule",
        type=Path,
        help="pinned LIBERO-Pro checkout",
    )
    doctor.add_argument(
        "--libero-config",
        type=Path,
        help="LIBERO path config (default: <repo>/.libero/config.yaml)",
    )
    doctor.add_argument(
        "--environment-root",
        type=Path,
        help="Python environment that must be active and writable",
    )
    doctor.add_argument(
        "--expected-environment-name",
        help="required basename of the active Python environment",
    )
    doctor.add_argument(
        "--runtime",
        choices=("behavior", "libero", "robosuite"),
        help="dependency/runtime checks to perform (defaults to reset embodiment)",
    )
    doctor.add_argument(
        "--unit",
        "--cpu-only",
        action="store_true",
        dest="unit_mode",
        help="make GPU, EGL, platform, services, and LIBERO extras nonfatal",
    )
    doctor.add_argument(
        "--skip-services",
        action="store_true",
        help="do not probe baseline or explicitly requested optional services",
    )
    doctor.add_argument("--service-host", default="127.0.0.1")
    doctor.add_argument(
        "--providers",
        help=(
            "comma-separated provider names to probe (from configs/environments.json); "
            "e.g. sam3,contact_graspnet,pyroki,curobo for a full stack"
        ),
    )
    doctor.add_argument(
        "--check-curobo",
        action="store_true",
        help="deprecated: also probe curobo (use --providers ...,curobo)",
    )
    doctor.add_argument(
        "--reset",
        action="store_true",
        help="perform one reset-only simulator smoke check and close it",
    )
    doctor.add_argument("--reset-suite", help="suite for --reset (default: representative)")
    doctor.add_argument(
        "--reset-embodiment",
        choices=("behavior", "libero", "robosuite"),
        default="libero",
        help="simulation embodiment for --reset (default: libero)",
    )
    doctor.add_argument("--reset-task-id", type=int, help="task id for --reset")
    doctor.add_argument("--seed", type=int, default=1, help="reset seed (default: 1)")
    doctor.add_argument("--quiet", action="store_true", help="write JSON without check lines")
    doctor.set_defaults(handler=_doctor_command)

    manifest = subparsers.add_parser(
        "manifest", help="discover and emit the required 80-pair manifest"
    )
    manifest.add_argument("--config", type=Path, help="validation_cases.yaml override")
    manifest.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write JSON to this path (default: stdout)",
    )
    manifest.set_defaults(handler=_manifest_command)

    validate = subparsers.add_parser(
        "validate", help="run the resumable one-tick LIBERO validation matrix"
    )
    validate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation-output"),
        help="artifact directory (default: validation-output)",
    )
    validate.add_argument(
        "--plan",
        type=Path,
        help="run a declarative validation plan (configs/validation/*.yaml) across "
        "all embodiments; when set, --config/--nightly are ignored",
    )
    validate.add_argument("--config", type=Path, help="validation_cases.yaml override (legacy)")
    validate.add_argument(
        "--nightly",
        action="store_true",
        help="run seeds 1, 2, and 3 instead of smoke seed 1 (legacy)",
    )
    validate.add_argument(
        "--retry-failures",
        action="store_true",
        help="append new attempts for prior failed keys; passed keys remain resumed",
    )
    validate.set_defaults(handler=_validate_command)

    # RoboSuite structural, control, and bimanual acceptance are now declarative
    # plans run via `validate --plan configs/validation/robosuite-*.yaml`.

    run = subparsers.add_parser("run", help="execute and record one Code-as-Policy episode")
    run.add_argument(
        "--benchmark",
        choices=("behavior", "libero-pro", "robosuite", "yam_real"),
        default="libero-pro",
    )
    run.add_argument("--suite", required=True)
    run.add_argument("--station", default="yam-example", help="YAM station profile identifier")
    run.add_argument(
        "--station-config-root", type=Path, help="Custom YAM station profiles directory"
    )
    run.add_argument("--allow-motion", action="store_true", help="Enable physical YAM commands")
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--program", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=Path("outputs"))
    run.add_argument(
        "--flat-run-dir",
        action="store_true",
        help=(
            "write runs as <output-root>/<seed>/<run-id> instead of nesting "
            "<benchmark>/<suite>/<task> — use when --output-root is already task-scoped"
        ),
    )
    run.add_argument(
        "--no-videos",
        action="store_true",
        help="skip per-camera mp4 capture (keyframes, overlays, and traces are still recorded)",
    )
    run.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="episode length in control ticks; defaults to the horizon the task's own "
        "benchmark states, or 1000 where none is stated",
    )
    run.add_argument("--camera-width", type=int, default=800)
    run.add_argument("--camera-height", type=int, default=512)
    run.add_argument(
        "--model-trace",
        type=Path,
        help="optional canonical JSONL model request/response trace",
    )
    run.add_argument(
        "--init-mode",
        choices=("saved", "seeded"),
        default="saved",
        help=(
            "LIBERO-Pro initial-state selection: 'saved' restores saved state "
            "(seed-1) %% N; 'seeded' seeds procedural placement randomization "
            "so every integer seed is a distinct scene"
        ),
    )
    run.set_defaults(handler=_run_command)

    batch = subparsers.add_parser(
        "run-batch",
        help="execute one program across many seeds, reusing worker processes",
        description=(
            "Run one program on several seeds without starting an interpreter per "
            "seed. Every seed still builds and tears down its own environment; only "
            "the imports are shared. BEHAVIOR requires a fresh process per episode."
        ),
    )
    batch.add_argument(
        "--benchmark",
        choices=("libero-pro", "robosuite"),
        default="libero-pro",
    )
    batch.add_argument("--suite", required=True)
    batch.add_argument("--task-id", type=int, required=True)
    batch.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        metavar="SEED",
        help="seeds to run, as numbers or inclusive ranges: --seeds 51-65, --seeds 1 2 7-9",
    )
    batch.add_argument("--program", type=Path, required=True)
    batch.add_argument("--output-root", type=Path, default=Path("outputs"))
    batch.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "seeds to run concurrently (default: 1). Each worker is its own spawned "
            "process; they share the assigned GPU and the perception services"
        ),
    )
    batch.add_argument(
        "--recycle-after",
        type=int,
        default=DEFAULT_RECYCLE_AFTER,
        metavar="N",
        help=(
            f"retire a worker after N seeds, bounding any driver-side leak from "
            f"repeated environment construction (default: {DEFAULT_RECYCLE_AFTER}; "
            "0 keeps workers for the whole sweep)"
        ),
    )
    batch.add_argument(
        "--log-dir",
        type=Path,
        help="per-seed logs (default: <output-root>/logs/seed_<NN>.log)",
    )
    batch.add_argument(
        "--results-jsonl",
        type=Path,
        help=(
            "append one JSON record per finished seed here as it lands, so an "
            "interrupted sweep keeps what it completed "
            "(default: <output-root>/batch-results.jsonl)"
        ),
    )
    batch.add_argument(
        "--summary",
        type=Path,
        help="write the aggregate summary here (default: <output-root>/batch.json)",
    )
    batch.add_argument(
        "--flat-run-dir",
        action="store_true",
        help="write runs as <output-root>/<seed>/<run-id>, as `run --flat-run-dir` does",
    )
    batch.add_argument(
        "--no-videos",
        action="store_true",
        help="skip per-camera mp4 capture for every seed in the sweep",
    )
    batch.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="episode length in control ticks; defaults to the horizon the task's own "
        "benchmark states, or 1000 where none is stated",
    )
    batch.add_argument("--camera-width", type=int, default=800)
    batch.add_argument("--camera-height", type=int, default=512)
    batch.add_argument(
        "--init-mode",
        choices=("saved", "seeded"),
        default="saved",
        help="initial-state selection, as `run --init-mode` (default: saved)",
    )
    batch.set_defaults(handler=_run_batch_command)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _doctor_command(args: argparse.Namespace) -> int:
    from cap_harness.doctor import DoctorConfig, ResetTarget, run_doctor

    reset = None
    if args.reset:
        reset = ResetTarget(
            suite=args.reset_suite,
            task_id=args.reset_task_id,
            seed=args.seed,
            embodiment=args.reset_embodiment,
        )
    from cap_harness.doctor import topology_service_targets

    if getattr(args, "providers", None):
        names = [p.strip() for p in args.providers.split(",") if p.strip()]
        services = list(topology_service_targets(names))
    else:
        services = list(topology_service_targets())
        if args.check_curobo:
            print(
                "warning: --check-curobo is deprecated; use --providers ...,curobo",
                file=sys.stderr,
            )
            services.extend(topology_service_targets(["curobo"]))
    config = DoctorConfig(
        environment_root=args.environment_root,
        expected_environment_basename=args.expected_environment_name,
        embodiment=args.runtime or args.reset_embodiment,
        libero_submodule=args.libero_submodule,
        libero_path_config=args.libero_config,
        unit_mode=args.unit_mode,
        check_services=not args.skip_services,
        service_host=args.service_host,
        services=tuple(services),
        reset=reset,
    )
    report = run_doctor(config, output_path=args.output)
    if not args.quiet:
        for check in report.checks:
            print(f"{check.status.upper():7} {check.name}: {check.summary}")
        print(f"report: {args.output}", flush=True)
    return report.exit_code


def _manifest_command(args: argparse.Namespace) -> int:
    from cap_harness.validation import build_required_manifest, write_manifest

    if args.output is not None:
        manifest = write_manifest(args.output, config_path=args.config)
        print(f"wrote {len(manifest)} pairs to {args.output}")
    else:
        manifest = build_required_manifest(config_path=args.config)
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    if getattr(args, "plan", None) is not None:
        from cap_harness.validation.runner import run_plan

        repo_root = Path(__file__).resolve().parents[2]
        summary = run_plan(
            args.plan,
            args.output_dir,
            repo_root=repo_root,
            retry_failures=args.retry_failures,
        )
        passed = sum(1 for c in summary["cases"] if c["passed"])
        print(
            f"plan {summary['plan']} ({summary['benchmark']}): "
            f"{passed}/{len(summary['cases'])} cases passed"
            + ("" if summary["success"] else " — FAILED")
        )
        print(f"summary: {args.output_dir / 'summary.json'}")
        return 0 if summary["success"] else 1

    from cap_harness.validation import validate_matrix

    summary = validate_matrix(
        args.output_dir,
        config_path=args.config,
        nightly=args.nightly,
        retry_failures=args.retry_failures,
    )
    print(
        "validation: "
        f"{summary['passed']}/{summary['expected']} passed, "
        f"{summary['failed']} failed, {summary['remaining']} remaining"
    )
    print(f"summary: {args.output_dir / 'summary.json'}")
    return 0 if summary["success"] else 1


def _run_command(args: argparse.Namespace) -> int:
    from cap_harness.run import run_program

    outcome = run_program(
        benchmark=args.benchmark,
        suite=args.suite,
        task_id=args.task_id,
        seed=args.seed,
        program_path=args.program,
        output_root=args.output_root,
        max_steps=args.max_steps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        model_trace=args.model_trace,
        init_mode=args.init_mode,
        flat_layout=args.flat_run_dir,
        capture_videos=not args.no_videos,
        station=args.station,
        station_config_root=args.station_config_root,
        allow_motion=args.allow_motion,
    )
    # Flushed explicitly: a simulator that segfaults on interpreter teardown (Isaac Sim does)
    # would otherwise discard these lines when stdout is a file.
    print(f"run: {outcome.run_dir}", flush=True)
    print(
        f"program_ok={outcome.program_ok} task_success={outcome.task_success} "
        f"protocol_success={outcome.protocol_success} "
        f"termination_reason={outcome.termination_reason}",
        flush=True,
    )
    return 0 if outcome.program_ok else 1


def _run_batch_command(args: argparse.Namespace) -> int:
    import time

    from cap_harness.batch import parse_seed_arguments, run_seeds, summarize

    seeds = parse_seed_arguments(args.seeds)
    results_jsonl = args.results_jsonl or args.output_root / "batch-results.jsonl"
    summary_path = args.summary or args.output_root / "batch.json"

    def report(result: Any) -> None:
        if result.ok:
            print(
                f"seed {result.seed:02d}: task_success={result.task_success} "
                f"program_ok={result.program_ok} {result.wall_s:.1f}s "
                f"pid={result.worker_pid} rss={result.max_rss_mb:.0f}MB",
                flush=True,
            )
        else:
            print(f"seed {result.seed:02d}: NO ARTIFACT ({result.error})", flush=True)

    started = time.monotonic()
    results = run_seeds(
        benchmark=args.benchmark,
        suite=args.suite,
        task_id=args.task_id,
        program_path=args.program,
        output_root=args.output_root,
        seeds=seeds,
        workers=args.workers,
        recycle_after=args.recycle_after,
        max_steps=args.max_steps,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        init_mode=args.init_mode,
        flat_layout=args.flat_run_dir,
        capture_videos=not args.no_videos,
        log_dir=args.log_dir,
        results_jsonl=results_jsonl,
        on_result=report,
    )
    summary = summarize(results)
    summary["elapsed_s"] = round(time.monotonic() - started, 3)
    summary["workers"] = args.workers
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"batch: {summary['runs']}/{summary['seeds']} runs, "
        f"{summary['passes']} passed, {summary['elapsed_s']:.1f}s elapsed "
        f"({summary['seed_wall_s']:.1f}s of seed time on {args.workers} worker(s))"
    )
    print(f"summary: {summary_path}")
    return 0 if summary["runs"] == summary["seeds"] else 1


def _safe_error_message(exc: BaseException) -> str:
    try:
        from cap_harness.doctor import redact_secrets

        return str(redact_secrets(str(exc)))
    # Fail closed if formatting or redacting an exception itself fails.
    except BaseException:
        return type(exc).__name__


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = args.handler
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    # Fail closed if formatting or redacting an exception itself fails.
    except Exception as exc:
        print(f"cap-harness {args.command}: {_safe_error_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
