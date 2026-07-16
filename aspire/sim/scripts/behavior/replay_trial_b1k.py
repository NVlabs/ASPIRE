#!/usr/bin/env python3
"""Replay or interactively debug one BEHAVIOR-1K trial.

This is a small CLI facade over ``aspire.sim.cap.envs.launch_b1k``. It avoids the LIBERO
task injection in ``scripts/libero/replay_trial.py`` and uses B1K configs directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

@dataclass
class ReplayB1KArgs:
    """Arguments for a single B1K replay or interactive debug run."""

    config_path: str
    """B1K YAML config path."""

    replay_code: str | None = None
    """Saved code.py to replay. Omit when using --interactive."""

    trial: int = 1
    """Trial ID / seed to reset and replay."""

    output_dir: str = "outputs/b1k_replay"
    """Output directory for replay artifacts."""

    interactive: bool = False
    """Open a Python REPL instead of replaying code."""

    record_video: bool = True
    """Record rollout video when the environment supports video capture."""

    model: str = "replay"
    """Model label used only for output directory naming."""

    extra_trial_ids: list[int] = field(default_factory=list)
    """Additional trial IDs to replay with the same code."""


def main(args: ReplayB1KArgs) -> None:
    from aspire.sim.cap.envs.launch_b1k import B1KLaunchArgs, main as launch_b1k_main

    trial_ids = [args.trial, *args.extra_trial_ids]
    if not args.interactive and not args.replay_code:
        raise ValueError("--replay-code is required unless --interactive is set")

    launch_b1k_main(
        B1KLaunchArgs(
            config_path=args.config_path,
            replay_code=args.replay_code,
            interactive=args.interactive,
            trial_ids=trial_ids,
            output_dir=args.output_dir,
            record_video=args.record_video,
            model=args.model,
            total_trials=max(trial_ids),
        )
    )


if __name__ == "__main__":
    main(tyro.cli(ReplayB1KArgs))
