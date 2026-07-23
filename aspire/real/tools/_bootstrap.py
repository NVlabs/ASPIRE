# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence


def ensure_repo_root(file_path: str | Path, *, parents_up: int = 2) -> Path:
    repo_root = Path(file_path).resolve().parents[parents_up]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def maybe_reexec_with_uv(
    script_path: str | Path,
    repo_root: str | Path,
    *,
    required_modules: Iterable[str],
    mode: str = "project",
    extras: Sequence[str] = (),
    env_flag: str = "YAM_UV_ABS_BOOTSTRAPPED",
) -> None:
    if os.environ.get(env_flag) == "1":
        return

    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if not missing:
        return

    uv = shutil.which("uv")
    if uv is None:
        return

    script = str(Path(script_path).resolve())
    repo = str(Path(repo_root).resolve())
    env = os.environ.copy()
    env[env_flag] = "1"

    if mode == "project":
        argv = [uv, "run", "--project", repo]
        for extra in extras:
            argv.extend(["--extra", extra])
        argv.extend(["python", script, *sys.argv[1:]])
    elif mode == "script":
        argv = [uv, "run", script, *sys.argv[1:]]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    os.execvpe(uv, argv, env)
