# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASPIRE evaluation entry point.

Usage::

    uv run --no-sync --active python -m aspire.sim.cap.envs.launch \\
        --config-path env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml

Execution flow::

    main()
      └─ _run_headless_trials()   (CLI batch mode)  [in aspire.sim.cap.envs.runner]
           ├─ _run_trial_batch()  (sequential)
           └─ run_parallel_*()    (multi-worker)
                └─ _run_single_trial()  [in aspire.sim.cap.envs.trial]
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import tyro

from aspire.sim.cap.utils.launch_utils import _load_config

os.environ.setdefault("MUJOCO_GL", "egl")


# ---------------------------------------------------------------------------
# CLI argument dataclass
# ---------------------------------------------------------------------------

@dataclass
class LaunchArgs:
    """Command-line arguments for ASPIRE evaluation.

    Defines configuration options for model querying, execution, and output.
    """

    # YAML config path (required)
    config_path: str
    """Path to the YAML configuration file defining the environment and task."""

    # Model server configuration
    server_url: str = "http://127.0.0.1:8110/chat/completions"
    """URL of the vLLM server's chat completions endpoint."""

    model: str = "google/gemini-3.1-pro-preview"
    """Name of the model to query on from the server_url."""

    temperature: float = 1.0
    """Sampling temperature for code generation (higher = more random)."""

    max_tokens: int = 2048 * 10
    """Maximum number of tokens to generate in the model response."""

    reasoning_effort: str = "medium"
    """Effort level for reasoning models (if applicable). Options: minimal, low, medium, high."""

    api_key: str | None = None
    """Optional API key for authentication with the model server."""

    # Execution configuration (can override YAML values)
    use_visual_feedback: bool | None = None
    """Whether to provide visual feedback (images) to the model during generation."""

    use_img_differencing: bool | None = None
    """Whether to provide image differencing to the model during generation."""

    use_video_differencing: bool | None = None
    """Use video-based VDM: pass a video of each turn's execution to the differencing model."""

    use_wrist_camera: bool | None = None
    """Also record and pass wrist camera video to the VDM alongside the main camera."""

    use_legacy_multi_turn_decision_prompt: bool | None = None
    """Whether to use the legacy multi-turn decision prompt."""

    visual_differencing_model: str | None = "gcp/google/gemini-3.1-pro-preview"
    """Model to use for visual differencing."""

    visual_differencing_model_server_url: str | None = (
        "http://127.0.0.1:8110/chat/completions"
    )
    """Server URL of the image differencing model."""

    visual_differencing_model_api_key: str | None = None
    """API key for authentication with the image differencing model."""

    total_trials: int | None = None
    """Total number of trials to run. Overrides the value in the YAML config."""

    num_workers: int | None = None
    """Number of parallel worker processes to use. Overrides the value in the YAML config."""

    record_video: bool | None = None
    """Whether to record and save videos of the environment execution."""

    output_dir: str | None = None
    """Directory to save trial outputs (code, logs, videos)."""

    debug: bool | None = False
    """Enable debug logging (prints full model responses)."""

    use_oracle_code: bool | None = None
    """If True, uses pre-defined oracle code instead of querying the model."""

    use_parallel_ensemble: bool | None = None
    """Whether to use parallel ensemble for the coding agent."""

    use_multimodel: bool | None = None
    """Whether to use multimodel for parallel ensembling."""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: LaunchArgs) -> None:
    """Load config and run headless trial execution."""
    from aspire.sim.cap.envs.runner import _run_headless_trials, _start_api_servers, _stop_api_servers

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    server_procs = _start_api_servers(api_servers)

    try:
        _run_headless_trials(args, env_factory, config, start_time)
    finally:
        try:
            _stop_api_servers(server_procs)
        except KeyboardInterrupt:
            # Force exit if user interrupts during cleanup
            import sys
            sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(LaunchArgs))
