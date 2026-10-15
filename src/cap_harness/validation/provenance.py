"""Generic sealed-evidence helpers for acceptance-grade validation plans.

Extracted from the RoboSuite campaign so any plan marked ``sealed`` reproduces
the same guarantees: a clean-committed-worktree requirement, a full source-tree
hash, package/render provenance, program hashes, and a sealed environment file
whose hash is injected into every fresh program run. Import-light (stdlib only).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

_SEALED_PACKAGES = ("cap-harness", "mujoco", "numpy", "requests", "robosuite", "scipy")
_RENDER_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "EGL_PLATFORM",
    "MUJOCO_EGL_DEVICE_ID",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
)


class DirtyWorktreeError(RuntimeError):
    """Raised when a sealed plan is run from a dirty or uncommitted worktree."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | os.PathLike[str], value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(repo_root: str | os.PathLike[str], *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_tree_sha256(repo_root: str | os.PathLike[str]) -> str:
    """Hash every tracked file under the repo top, in sorted path order."""
    git_top = Path(git(repo_root, "rev-parse", "--show-toplevel"))
    tracked = git(repo_root, "ls-files", "--", ".").splitlines()
    digest = hashlib.sha256()
    for relative in sorted(tracked):
        path = git_top / relative
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def require_clean_worktree(repo_root: str | os.PathLike[str]) -> str:
    """Return HEAD commit; raise DirtyWorktreeError if the tree is not clean."""
    if git(repo_root, "status", "--porcelain"):
        raise DirtyWorktreeError("sealed validation plans require a clean committed worktree")
    return git(repo_root, "rev-parse", "HEAD")


def seal_environment(
    output_root: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    *,
    renderer: str,
    schema_version: int = 1,
) -> tuple[Path, str, str]:
    """Write ``environment.json`` and return (path, environment_sha256, commit).

    Records python/platform/package/render provenance, the HEAD commit, the full
    source-tree hash, and the dependency-lock hash. The returned environment hash
    is injected into each fresh program run for later verification.
    """
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root)
    commit = require_clean_worktree(repo_root)
    dependency_lock = repo_root / "configs/dependency-lock.json"
    environment = {
        "schema_version": schema_version,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "packages": package_versions(_SEALED_PACKAGES),
        "renderer": renderer,
        "render_environment": {name: os.environ.get(name) for name in _RENDER_ENV_KEYS},
        "harness_git_commit": commit,
        "harness_source_tree_sha256": source_tree_sha256(repo_root),
        "dependency_lock_sha256": (
            sha256_file(dependency_lock) if dependency_lock.is_file() else None
        ),
    }
    environment_path = output_root / "environment.json"
    write_json(environment_path, environment)
    return environment_path, sha256_file(environment_path), commit


def check_program_provenance(
    provenance: Mapping[str, object],
    *,
    program_sha256: str,
    commit: str,
    dependency_lock_sha256: str | None,
) -> list[str]:
    """Return a list of issue strings for a single run's ``provenance.json``."""
    issues: list[str] = []
    if provenance.get("sha256") != program_sha256:
        issues.append("program provenance hash mismatch")
    git_info = provenance.get("harness_git")
    if (
        not isinstance(git_info, Mapping)
        or git_info.get("commit") != commit
        or git_info.get("dirty") is not False
    ):
        issues.append("run was not recorded from the clean sealed commit")
    if provenance.get("dependency_lock_sha256") != dependency_lock_sha256:
        issues.append("dependency lock hash mismatch")
    return issues


__all__ = [
    "DirtyWorktreeError",
    "check_program_provenance",
    "git",
    "package_versions",
    "require_clean_worktree",
    "seal_environment",
    "sha256_file",
    "source_tree_sha256",
    "write_json",
]
