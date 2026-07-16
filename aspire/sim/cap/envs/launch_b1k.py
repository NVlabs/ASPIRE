"""BEHAVIOR-1K launcher with exact-trial, replay, and REPL modes.

This module is a thin B1K-specific wrapper around the standard
``aspire.sim.cap.envs.launch`` / ``aspire.sim.cap.envs.runner`` path. Normal batch execution still
uses the same trial engine as ``launch.py``. The extra modes are for debugging:

- ``--trial-ids``: run an explicit set of seeds.
- ``--replay-code``: execute a saved ``code.py`` block-by-block without LLM calls.
- ``--interactive``: reset one B1K scene and expose the R1Pro API in a Python REPL.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

_gpu_id = os.environ.get("OMNIGIBSON_GPU_ID")
if _gpu_id is not None and "OMNIGIBSON_APPDATA_PATH" not in os.environ:
    os.environ["OMNIGIBSON_APPDATA_PATH"] = f"/tmp/og_appdata_gpu{_gpu_id}"


@dataclass
class B1KLaunchArgs:
    """Launch args for BEHAVIOR-1K debugging."""

    config_path: str
    """Path to the YAML configuration file defining the environment and task."""

    server_url: str = "http://127.0.0.1:8110/chat/completions"
    """URL of the model server's chat completions endpoint."""

    model: str = "google/gemini-3.1-pro-preview"
    """Name of the model to query from the server_url."""

    temperature: float = 1.0
    """Sampling temperature for code generation."""

    max_tokens: int = 2048 * 10
    """Maximum number of tokens to generate in the model response."""

    reasoning_effort: str = "medium"
    """Effort level for reasoning models. Options: minimal, low, medium, high."""

    api_key: str | None = None
    """Optional API key for authentication with the model server."""

    use_visual_feedback: bool | None = None
    """Whether to provide visual feedback images to the model."""

    use_img_differencing: bool | None = None
    """Whether to provide image differencing feedback to the model."""

    use_video_differencing: bool | None = None
    """Whether to provide video differencing feedback to the model."""

    use_wrist_camera: bool | None = None
    """Also record and pass wrist camera video to the VDM."""

    use_legacy_multi_turn_decision_prompt: bool | None = None
    """Whether to use the legacy multi-turn decision prompt."""

    visual_differencing_model: str | None = "gcp/google/gemini-3.1-pro-preview"
    """Model to use for visual differencing."""

    visual_differencing_model_server_url: str | None = "http://127.0.0.1:8110/chat/completions"
    """Server URL of the image/video differencing model."""

    visual_differencing_model_api_key: str | None = None
    """API key for the image/video differencing model."""

    total_trials: int | None = None
    """Total number of trials to run. Overrides the YAML config."""

    num_workers: int | None = None
    """Number of parallel worker processes. Overrides the YAML config."""

    record_video: bool | None = None
    """Whether to record and save videos."""

    output_dir: str | None = None
    """Directory to save trial outputs."""

    debug: bool | None = False
    """Enable debug logging."""

    use_oracle_code: bool | None = None
    """Use pre-defined oracle code instead of querying the model."""

    use_parallel_ensemble: bool | None = None
    """Whether to use parallel ensemble for the coding agent."""

    use_multimodel: bool | None = None
    """Whether to use multimodel parallel ensembling."""

    trial_ids: list[int] | None = None
    """Explicit trial IDs / seeds to run. Overrides YAML resume_idx and total_trials."""

    replay_code: str | None = None
    """Saved code.py to replay block-by-block without any LLM calls."""

    interactive: bool = False
    """Drop into a Python REPL after resetting one B1K trial."""


def _parse_code_blocks(code_path: str) -> list[str]:
    """Parse saved code into execution blocks.

    The normal launcher writes generated code as ``# Code block N`` sections.
    If a file has no section markers, replay it as one block.
    """
    code_text = Path(code_path).read_text()
    parts = re.split(r"^# Code block \d+\s*\n", code_text, flags=re.MULTILINE)
    blocks = [part.strip() for part in parts if part.strip()]
    return blocks or [code_text.strip()]


def _find_apis(obj: Any, depth: int = 0) -> dict[str, Any]:
    """Walk common env wrapper links to find an ``_apis`` dict."""
    if depth > 6:
        return {}
    if hasattr(obj, "_apis"):
        return obj._apis
    for attr in ("env", "low_level_env", "_executor", "executor"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            result = _find_apis(child, depth + 1)
            if result:
                return result
    return {}


def _instantiate_env(env_factory: dict[str, Any]) -> Any:
    """Instantiate a B1K env while hiding CLI args from Isaac Sim."""
    from aspire.sim.cap.envs.configs.instantiate import instantiate

    original_sys_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        return instantiate(env_factory)
    finally:
        sys.argv = original_sys_argv


def _reset_env(env: Any, trial: int) -> tuple[Any, Any]:
    """Reset using the same seed convention as the standard trial runner."""
    try:
        return env.reset(options={"trial": trial}, seed=trial)
    except TypeError:
        return env.reset(seed=trial)


def _save_traces(env: Any, output_dir: str | Path) -> None:
    """Save trace loggers from traced APIs, if the config uses them."""
    for api in _find_apis(env).values():
        if hasattr(api, "get_trace_logger"):
            trace_logger = api.get_trace_logger()
            trace_path = trace_logger.save(output_dir)
            trace_logger.reset()
            print(f"Saved trace to {trace_path}")


def _save_videos(env: Any, output_dir: str | Path, reward: float) -> None:
    """Save replay videos from env frame buffers."""
    from aspire.sim.cap.utils.video_utils import _write_video

    if not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames:
        return
    if isinstance(frames, dict):
        for name, frame_list in frames.items():
            if frame_list:
                _write_video(frame_list, str(output_dir), suffix=f"{reward:.3f}_{name}")
    elif isinstance(frames, list):
        _write_video(frames, str(output_dir), suffix=f"{reward:.3f}")


def _run_interactive_repl(env: Any, obs: Any, args: B1KLaunchArgs) -> None:
    """Drop into a Python REPL with R1Pro API functions in scope."""
    import code as code_module
    import numpy as np

    repl_ns: dict[str, Any] = {
        "env": env,
        "obs": obs,
        "np": np,
        "args": args,
    }

    fn_names: list[str] = []
    trace_logger = None
    for api in _find_apis(env).values():
        for fn_name, fn in api.functions().items():
            repl_ns[fn_name] = fn
            fn_names.append(fn_name)
        if hasattr(api, "get_trace_logger"):
            trace_logger = api.get_trace_logger()
            repl_ns["trace_logger"] = trace_logger

    def step(code_str: str):
        return env.step(code_str)

    repl_ns["step"] = step

    trial = args.trial_ids[0] if args.trial_ids else 1
    print("\n" + "=" * 72)
    print("  BEHAVIOR-1K Interactive Trial Debugger")
    print("=" * 72)
    print(f"  Config: {args.config_path}")
    print(f"  Trial:  {trial}")
    print("\n  Available API functions:")
    for fn_name in sorted(set(fn_names)):
        fn = repl_ns[fn_name]
        doc = (getattr(fn, "__doc__", "") or "").strip().split("\n")
        first = doc[0][:72] if doc and doc[0] else ""
        print(f"    {fn_name}()" + (f"  - {first}" if first else ""))
    print("\n  Other objects:")
    print("    env, obs, np, args")
    if trace_logger is not None:
        print("    trace_logger")
    print("    step(code_str)  # execute code through env.step")
    print("\n  Example:")
    print("    >>> rgb, depth = get_env_observation()")
    print("    >>> save_current_observation('debug_start')")
    print("=" * 72 + "\n")

    code_module.interact(banner="", local=repl_ns, exitmsg="REPL exited.")


def _run_interactive(args: B1KLaunchArgs, env_factory: dict[str, Any], config: dict[str, Any]) -> None:
    """Instantiate one env and open the REPL after reset."""
    trial = args.trial_ids[0] if args.trial_ids else config.get("resume_idx") or 1
    print(f"[launch_b1k] Interactive mode: resetting trial {trial}")
    env = _instantiate_env(env_factory)
    obs, _ = _reset_env(env, trial)
    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=config.get("use_wrist_camera", False))
    _run_interactive_repl(env, obs, args)


def _run_replay(
    args: B1KLaunchArgs,
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> list[Any]:
    """Replay saved code blocks without calling an LLM."""
    from aspire.sim.cap.utils.launch_utils import TrialSummary

    if args.replay_code is None:
        raise ValueError("--replay-code is required for replay mode")
    if not config["output_dir"]:
        config["output_dir"] = "outputs/b1k_replay"

    code_blocks = _parse_code_blocks(args.replay_code)
    trial_ids = sorted(set(args.trial_ids or [config.get("resume_idx") or 1]))
    print(f"[launch_b1k] Replay mode: {len(code_blocks)} block(s) from {args.replay_code}")
    print(f"[launch_b1k] Trial IDs: {trial_ids}")

    env = _instantiate_env(env_factory)
    summaries: list[TrialSummary] = []

    for trial in trial_ids:
        print("\n" + "=" * 80)
        print(f"[launch_b1k] Replay trial {trial}")
        print("=" * 80)

        _reset_env(env, trial)
        if config["record_video"] and hasattr(env, "enable_video_capture"):
            env.enable_video_capture(True, clear=True, wrist_camera=config.get("use_wrist_camera", False))

        reward = 0.0
        terminated = False
        truncated = False
        info_step: dict[str, Any] = {
            "sandbox_rc": 0,
            "stdout": "",
            "stderr": "",
            "task_completed": False,
        }

        executed_blocks: list[str] = []
        for i, code in enumerate(code_blocks):
            print(f"\n--- Executing code block {i} ---")
            executed_blocks.append(code)
            _, reward, terminated, truncated, info_step = env.step(code)
            print(
                "  "
                f"reward={reward}, terminated={terminated}, "
                f"task_completed={info_step.get('task_completed')}"
            )
            stderr = info_step.get("stderr")
            if stderr:
                print(f"  stderr: {stderr[:300]}")
            if terminated or truncated:
                break

        sandbox_rc = int(info_step.get("sandbox_rc", 0))
        task_completed = bool(info_step.get("task_completed", False))
        final_code = "\n\n".join(
            f"# Code block {i}\n{block}" for i, block in enumerate(executed_blocks)
        )
        log_lines = [
            "-" * 100,
            f"Trial {trial} - B1K replay from {args.replay_code}",
            "Generated program:",
            final_code,
            "",
            "Environment response:",
            f"  Sandbox failed: {sandbox_rc}",
            f"  Stdout: {info_step.get('stdout', '')}",
            f"  Stderr: {info_step.get('stderr', '')}",
            f"  Reward: {reward}",
            f"  Task Completed: {task_completed}",
            f"  Terminated: {terminated}, Truncated: {truncated}",
            f"  Num Code Blocks: {len(executed_blocks)}",
            "-" * 100,
        ]

        trial_dir = (
            Path(config["output_dir"])
            / f"trial_{trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed)}"
        )
        trial_dir.mkdir(parents=True, exist_ok=True)
        code_path = trial_dir / "code.py"
        code_path.write_text(final_code)
        (trial_dir / "summary.txt").write_text("\n".join(log_lines))

        _save_traces(env, trial_dir)
        if config["record_video"]:
            _save_videos(env, trial_dir, reward)

        summaries.append(
            TrialSummary(
                trial=trial,
                success=sandbox_rc == 0,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                sandbox_rc=sandbox_rc,
                log="\n".join(log_lines),
                task_completed=task_completed,
                code_path=str(code_path),
                num_code_blocks=len(executed_blocks),
            )
        )

    return summaries


def main(args: B1KLaunchArgs) -> None:
    """Run B1K normal, exact-trial, replay, or interactive mode."""
    from aspire.sim.cap.envs.runner import (
        _run_headless_trials,
        _run_trial_batch,
        _setup_output_dir,
        _start_api_servers,
        _stop_api_servers,
    )
    from aspire.sim.cap.utils.launch_utils import _load_config, _print_and_save_summary

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    server_procs = _start_api_servers(api_servers)

    try:
        if args.interactive:
            _run_interactive(args, env_factory, config)
            return

        if args.replay_code:
            _setup_output_dir(args, config)
            summaries = _run_replay(args, env_factory, config)
            summaries.sort(key=lambda summary: summary.trial)
            _print_and_save_summary(summaries, args, config, start_time)
            return

        if args.trial_ids is not None:
            trial_ids = sorted(set(args.trial_ids))
            if not trial_ids:
                print("[launch_b1k] No trial IDs specified; exiting.")
                return
            print(f"[launch_b1k] Running specific trial IDs: {trial_ids}")

            if config["record_video"] and not config["output_dir"]:
                raise ValueError("record_video requires --output-dir")
            if config["total_trials"] < max(trial_ids):
                config["total_trials"] = max(trial_ids)

            _setup_output_dir(args, config)
            summaries = _run_trial_batch(
                trial_ids,
                args=args,
                env_factory=env_factory,
                config=config,
            )
            summaries.sort(key=lambda summary: summary.trial)
            _print_and_save_summary(summaries, args, config, start_time)

            if config["output_dir"]:
                flag_dir = Path(config["output_dir"]) / "aaa_done_flag"
                flag_dir.mkdir(parents=True, exist_ok=True)
                (flag_dir / "aaa_done_flag.txt").write_text("1")
            return

        _run_headless_trials(args, env_factory, config, start_time)
    finally:
        try:
            _stop_api_servers(server_procs)
        except KeyboardInterrupt:
            sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(B1KLaunchArgs))
