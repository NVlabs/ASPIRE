from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tyro
from tyro.conf import Positional


# Capture everything AFTER the trial prefix
TRIAL_SUFFIX_PATTERN = re.compile(
    r"^trial_.*?(_sandboxrc_\d+_reward_-?\d+(?:\.\d+)?_taskcompleted_\d+)$"
)


@dataclass
class Config:
    output_dir: Positional[Path]
    dry_run: bool = False
    start_index: int = 1  # trials start at 01


def collect_trial_dirs(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir()
        if p.is_dir() and p.name.startswith("trial_")
    )


def rename_trials(cfg: Config) -> None:
    output_dir = cfg.output_dir

    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Invalid directory: {output_dir}")

    trial_dirs = collect_trial_dirs(output_dir)

    if not trial_dirs:
        print("No trial directories found.")
        return

    print(f"Found {len(trial_dirs)} trial folders\n")

    rename_plan: list[tuple[Path, Path]] = []

    for i, trial_dir in enumerate(trial_dirs):
        m = TRIAL_SUFFIX_PATTERN.match(trial_dir.name)
        if not m:
            raise ValueError(
                f"Unrecognized trial folder format: {trial_dir.name}"
            )

        suffix = m.group(1)

        trial_num = cfg.start_index + i
        trial_str = f"{trial_num:02d}"  # zero-padded

        new_name = f"trial_{trial_str}{suffix}"
        new_path = trial_dir.parent / new_name

        rename_plan.append((trial_dir, new_path))

    # Safety: ensure no collisions
    new_names = [dst.name for _, dst in rename_plan]
    if len(new_names) != len(set(new_names)):
        raise RuntimeError("Name collision detected in planned renames")

    print("Planned renames:")
    for src, dst in rename_plan:
        print(f"  {src.name}  →  {dst.name}")

    if cfg.dry_run:
        print("\nDry run enabled — no changes made.")
        return

    # Phase 1: rename everything to temporary names
    temp_paths: list[tuple[Path, Path]] = []
    for src, dst in rename_plan:
        tmp = src.with_name(src.name + "__tmp__")
        src.rename(tmp)
        temp_paths.append((tmp, dst))

    # Phase 2: rename to final names
    for tmp, final in temp_paths:
        tmp.rename(final)

    print("\n✓ Trial folders successfully renamed.")


def main(cfg: Config) -> None:
    rename_trials(cfg)


if __name__ == "__main__":
    tyro.cli(main)
