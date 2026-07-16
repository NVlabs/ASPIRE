"""Helpers for preparing the vendored AnyGrasp SDK runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "third_party" / "anygrasp_sdk"
RUNTIME_ROOT = Path(os.environ.get("ANYGRASP_RUNTIME_DIR", "/tmp/anygrasp_sdk_runtime"))


def _ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)


def _python_ext_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError(
            "Could not determine Python extension suffix for AnyGrasp runtime"
        )
    return suffix


@lru_cache(maxsize=1)
def _git_dir() -> Path:
    git_dir = subprocess.check_output(
        ["git", "rev-parse", "--git-dir"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    path = Path(git_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _git_lfs_object_path(oid: str) -> Path:
    return _git_dir() / "lfs" / "objects" / oid[:2] / oid[2:4] / oid


def _parse_lfs_oid(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.removeprefix("oid sha256:").strip()
            return oid or None
    return None


def _parse_lfs_object_target(text: str) -> Path | None:
    candidate = text.strip()
    if not candidate:
        return None

    path = Path(candidate)
    if path.exists():
        return path

    basename = path.name
    if len(basename) == 64 and all(ch in "0123456789abcdef" for ch in basename.lower()):
        obj = _git_lfs_object_path(basename)
        if obj.exists():
            return obj
    return None


def _materialize_lfs_pointer(path: Path) -> Path:
    if not path.exists() or path.is_symlink():
        return path

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return path

    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return path

    oid = _parse_lfs_oid(text)
    if not oid:
        raise FileNotFoundError(
            f"Could not parse Git LFS oid from pointer file: {path}"
        )

    obj = _git_lfs_object_path(oid)
    if not obj.exists():
        raise FileNotFoundError(f"Missing local Git LFS object for {path}: {obj}")

    path.unlink()
    path.symlink_to(obj)
    return path


def _is_elf_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def _binary_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_value = os.environ.get("ANYGRASP_BINARY_SEARCH_ROOTS", "")
    for raw in env_value.split(os.pathsep):
        raw = raw.strip()
        if raw:
            roots.append(Path(raw).expanduser())

    roots.extend(
        [
            REPO_ROOT,
            REPO_ROOT.parent,
            REPO_ROOT.parent.parent,
            Path.home(),
            Path.home().parent,
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if not resolved.exists() or resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _find_nearby_binary_copy(path: Path) -> Path | None:
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None

    try:
        original = path.resolve()
    except OSError:
        original = path

    for search_root in _binary_search_roots():
        try:
            output = subprocess.check_output(
                [
                    "find",
                    str(search_root),
                    "-maxdepth",
                    "8",
                    "-path",
                    f"*/{rel}",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue

        for line in output.splitlines():
            candidate = Path(line.strip())
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved == original:
                continue
            if _is_elf_binary(resolved):
                return resolved

    return None


def _historical_lfs_object_for_repo_path(path: Path) -> Path | None:
    rel = path.relative_to(REPO_ROOT).as_posix()
    try:
        revs = subprocess.check_output(
            ["git", "log", "--follow", "--format=%H", "--", rel],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        return None

    for rev in revs:
        try:
            ls_tree = subprocess.check_output(
                ["git", "ls-tree", "-r", rev, "--", rel],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue

        if not ls_tree:
            continue

        parts = ls_tree.split(maxsplit=3)
        if len(parts) < 3:
            continue
        blob_sha = parts[2]

        try:
            blob_bytes = subprocess.check_output(
                ["git", "cat-file", "-p", blob_sha],
                cwd=REPO_ROOT,
            )
        except subprocess.CalledProcessError:
            continue

        try:
            blob_text = blob_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        oid = _parse_lfs_oid(blob_text)
        if oid:
            obj = _git_lfs_object_path(oid)
            if obj.exists():
                return obj

        target = _parse_lfs_object_target(blob_text)
        if target is not None and target.exists():
            return target

    return None


def _resolve_binary_source(path: Path, *, allow_history: bool = False) -> Path:
    if path.is_symlink():
        target = _parse_lfs_object_target(os.readlink(path))
        if target is not None and target.exists():
            return target.resolve()

    if path.exists():
        try:
            materialized = _materialize_lfs_pointer(path)
        except FileNotFoundError:
            materialized = path
        if materialized.exists() and _is_elf_binary(materialized):
            return materialized.resolve()

    if allow_history:
        historical = _historical_lfs_object_for_repo_path(path)
        if historical is not None and _is_elf_binary(historical):
            return historical.resolve()

    nearby = _find_nearby_binary_copy(path)
    if nearby is not None:
        return nearby.resolve()

    raise FileNotFoundError(f"Could not resolve native binary at {path}")


def _overlay_python_root(
    kind: str, *, package_dir: Path, backend_name: str, backend_source: Path
) -> Path:
    root = RUNTIME_ROOT / "python_roots" / kind
    _ensure_symlink(package_dir, root / package_dir.name)

    backend_dir = root / "MinkowskiEngineBackend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(backend_source, backend_dir / backend_name)

    init_py = backend_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")

    return root


def _prepare_minkowski_python_root() -> Path:
    suffix = _python_ext_suffix()
    build_root = VENDOR_ROOT / "dependencies" / "MinkowskiEngine" / "build"
    source_pkg = VENDOR_ROOT / "dependencies" / "MinkowskiEngine" / "MinkowskiEngine"
    backend_rel = Path("MinkowskiEngineBackend") / f"_C{suffix}"
    init_rel = Path("MinkowskiEngine") / "__init__.py"

    candidates = sorted(build_root.glob("lib.*"))
    for candidate in candidates:
        backend_path = candidate / backend_rel
        if backend_path.exists():
            _materialize_lfs_pointer(backend_path)
        if (candidate / init_rel).exists() and backend_path.exists():
            return candidate

    for candidate in candidates:
        backend_path = candidate / backend_rel
        try:
            backend_source = _resolve_binary_source(backend_path, allow_history=True)
        except FileNotFoundError:
            continue
        return _overlay_python_root(
            "minkowski",
            package_dir=source_pkg,
            backend_name=f"_C{suffix}",
            backend_source=backend_source,
        )

    raise FileNotFoundError(
        "Could not prepare MinkowskiEngine runtime: no usable build/lib.* root was found, "
        f"and no local Git LFS backend object was available under {build_root}."
    )


def _binary_path(kind: str) -> Path:
    suffix = _python_ext_suffix()
    if kind == "gsnet":
        root = VENDOR_ROOT / "grasp_detection" / "gsnet_versions"
    elif kind == "tracker":
        root = VENDOR_ROOT / "grasp_tracking" / "tracker_versions"
    elif kind == "lib_cxx":
        root = VENDOR_ROOT / "license_registration" / "lib_cxx_versions"
    else:
        raise ValueError(f"Unknown AnyGrasp binary kind: {kind}")

    path = root / f"{kind}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Missing AnyGrasp binary for this Python: {path}")
    return _resolve_binary_source(path, allow_history=True)


def _prebuilt_python_root(kind: str) -> Path:
    if kind == "minkowski":
        return _prepare_minkowski_python_root()
    elif kind == "pointnet2":
        build_root = VENDOR_ROOT / "pointnet2" / "build"
        source_pkg = VENDOR_ROOT / "pointnet2" / "pointnet2"
        required = [
            "pointnet2/__init__.py",
            f"pointnet2/_ext{_python_ext_suffix()}",
        ]
    else:
        raise ValueError(f"Unknown AnyGrasp prebuilt package kind: {kind}")

    candidates = sorted(build_root.glob("lib.*"))
    for candidate in candidates:
        for rel in required:
            full = candidate / rel
            if full.suffix.startswith(".so") or ".so" in full.name:
                _materialize_lfs_pointer(full)
        if all((candidate / rel).exists() for rel in required):
            return candidate

    if kind == "pointnet2":
        backend_rel = Path(required[1])
        for candidate in candidates:
            backend_path = candidate / backend_rel
            try:
                backend_source = _resolve_binary_source(
                    backend_path, allow_history=True
                )
            except FileNotFoundError:
                continue
            return _overlay_python_root(
                "pointnet2",
                package_dir=source_pkg,
                backend_name=backend_path.name,
                backend_source=backend_source,
            )

    required_display = ", ".join(required)
    raise FileNotFoundError(
        f"Could not find prebuilt {kind} package root under {build_root} "
        f"with required files: {required_display}"
    )


def _extract_license(license_zip: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(license_zip) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            out_path = dest_dir / Path(info.filename).name
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_anygrasp_runtime(
    *,
    license_zip: str | Path,
) -> dict[str, Path]:
    """Prepare an importable AnyGrasp runtime tree under ``/tmp``."""

    license_zip = Path(license_zip)
    if not license_zip.exists():
        raise FileNotFoundError(f"AnyGrasp license zip not found: {license_zip}")

    detect_root = RUNTIME_ROOT / "grasp_detection"
    track_root = RUNTIME_ROOT / "grasp_tracking"
    _ensure_symlink(_binary_path("gsnet"), detect_root / "gsnet.so")
    _ensure_symlink(_binary_path("lib_cxx"), detect_root / "lib_cxx.so")
    _ensure_symlink(_binary_path("tracker"), track_root / "tracker.so")
    _ensure_symlink(_binary_path("lib_cxx"), track_root / "lib_cxx.so")

    _extract_license(license_zip, detect_root / "license")
    _extract_license(license_zip, track_root / "license")

    return {
        "runtime_root": RUNTIME_ROOT,
        "detect_root": detect_root,
        "track_root": track_root,
        "vendor_root": VENDOR_ROOT,
        "pointnet2_root": VENDOR_ROOT / "pointnet2",
        "minkowski_prebuilt_root": _prebuilt_python_root("minkowski"),
        "pointnet2_prebuilt_root": _prebuilt_python_root("pointnet2"),
    }


def configure_anygrasp_imports(runtime: dict[str, Path]) -> None:
    """Prepend import paths required for AnyGrasp detection and tracking."""

    paths = [
        str(runtime["minkowski_prebuilt_root"]),
        str(runtime["pointnet2_prebuilt_root"]),
        str(runtime["detect_root"]),
        str(runtime["track_root"]),
    ]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
