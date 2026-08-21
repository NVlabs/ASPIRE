# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from dataclasses import dataclass, field

import tyro
from aspire.sim.cap.envs.launch import LaunchArgs
from aspire.sim.cap.envs.launch import main as launch_main

# Set default environment variable for MuJoCo, similar to launch.py
os.environ.setdefault("MUJOCO_GL", "egl")


@dataclass
class BatchLaunchArgs:
    """Command-line arguments for batch execution of ASPIRE environments."""

    # List of YAML config paths to run
    config_paths: list[str] = field(
        default_factory=lambda: [
            "env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml",
            "env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml",
        ]
    )
    """List of paths to the YAML configuration files to run sequentially."""

    # Overrides (mirrored from LaunchArgs to allow global overrides)
    # server_url: str = "http://0.0.0.0:8009/v1/responses"  # local server for running codex models
    server_url: str = "http://127.0.0.1:8110/chat/completions"  # local server 


    models: list[str] = field(
        default_factory=lambda: [
            # "nvidia/openai/gpt-oss-120b", # Open source models
            # "Qwen/Qwen2.5-Coder-7B-Instruct",
            # "openai/gpt-oss-20b",
            # "deepseek/deepseek-v3.2",
            # "deepseek/deepseek-r1-0528",
            # "deepseek/deepseek-r1",
            # "qwen/qwen3.5-122b-a10b",
            # "moonshotai/kimi-k2",
            # "google/gemini-3.1-pro-preview", # Closed source models
            # "google/gemini-2.5-flash-lite",
            # "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-4-5",
            # "openai/gpt-5.4",
            # "openai/o1",
            # "openai/o4-mini",
        ]
    )
    """Names of the models to query on the vLLM server."""

    temperature: float = 1.0
    """Sampling temperature for code generation (higher = more random)."""

    max_tokens: int = 2048 * 10
    """Maximum number of tokens to generate in the model response."""

    reasoning_effort: str = "medium"
    """Effort level for reasoning models (if applicable)."""

    api_key: str | None = None
    """Optional API key for authentication with the model server."""

    visual_differencing_model: str | None = None
    """Override the visual-differencing (VDM) model from the YAML config."""

    visual_differencing_model_server_url: str | None = None
    """Override the VDM server URL from the YAML config."""

    visual_differencing_model_api_key: str | None = None
    """Override the VDM API key from the YAML config."""

    use_visual_feedback: bool | None = None
    """Whether to provide visual feedback (images) to the model during generation."""

    use_img_differencing: bool | None = None
    """Whether to use image differencing."""

    use_legacy_multi_turn_decision_prompt: bool | None = None
    """Whether to use the legacy multi-turn decision prompt."""

    total_trials: int | None = None
    """Total number of trials to run per yaml config file. Overrides the value in the YAML config."""

    num_workers: int | None = None
    """Number of parallel worker processes to use. Overrides the value in the YAML config."""

    record_video: bool | None = None
    """Whether to record and save videos of the environment execution."""

    output_dir: str | None = None
    """Directory to save trial outputs (code, logs, videos)."""

    debug: bool = False
    """Enable debug logging (prints full model responses)."""

    use_oracle_code: bool | None = None
    """If True, uses pre-defined oracle code instead of querying the model."""


def main(args: BatchLaunchArgs) -> None:
    """Run multiple experiments sequentially."""

    total_runs = len(args.models) * len(args.config_paths)
    print(
        f"Found {len(args.models)} models x {len(args.config_paths)} configurations "
        f"to run ({total_runs} total)."
    )

    failed_runs = []
    experiment_idx = 1

    for model in args.models:
        print(f"\n{'=' * 80}")
        print(f"Running model: {model}")
        print(f"{'=' * 80}")

        for config_path in args.config_paths:
            print(f"\n{'-' * 80}")
            print(f"Running Experiment {experiment_idx}/{total_runs}")
            print(f"Model: {model}")
            print(f"Config: {config_path}")
            print(f"{'-' * 80}\n")

            try:
                # If output_dir is overridden, append model and config name to avoid
                # collisions and ensure unique paths for each config/model in the batch.
                current_output_dir = args.output_dir
                if current_output_dir:
                    import pathlib

                    config_stem = pathlib.Path(config_path).stem
                    model_dir = model.replace("/", "_")
                    current_output_dir = os.path.join(current_output_dir, model_dir, config_stem)

                # Create LaunchArgs with the current config_path and global overrides
                launch_args = LaunchArgs(
                    config_path=config_path,
                    server_url=args.server_url,
                    model=model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    api_key=args.api_key,
                    use_visual_feedback=args.use_visual_feedback,
                    use_img_differencing=args.use_img_differencing,
                    use_legacy_multi_turn_decision_prompt=args.use_legacy_multi_turn_decision_prompt,
                    total_trials=args.total_trials,
                    num_workers=args.num_workers,
                    record_video=args.record_video,
                    output_dir=current_output_dir,
                    debug=args.debug,
                    use_oracle_code=args.use_oracle_code,
                    **({"visual_differencing_model": args.visual_differencing_model} if args.visual_differencing_model else {}),
                    **({"visual_differencing_model_server_url": args.visual_differencing_model_server_url} if args.visual_differencing_model_server_url else {}),
                    **({"visual_differencing_model_api_key": args.visual_differencing_model_api_key} if args.visual_differencing_model_api_key else {}),
                )

                launch_main(launch_args)

            except Exception as e:
                print(f"\nERROR running config {config_path} with model {model}: {e}")
                import traceback

                traceback.print_exc()
                failed_runs.append((model, config_path))

            experiment_idx += 1

    if failed_runs:
        print(f"\n{'=' * 80}")
        print(f"Batch execution completed with {len(failed_runs)} failures:")
        for model, cfg in failed_runs:
            print(f"  - model={model}, config={cfg}")
        print(f"{'=' * 80}")
        sys.exit(1)
    else:
        print(f"\n{'=' * 80}")
        print("Batch execution completed successfully.")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
