"""Declarative environment topology and hardware profile overlays.

This module is the single source of truth for provider names, ports, virtual
environment paths, GPU placement, build architecture, and bounded request
concurrency. The supervisor, doctor, bootstrap orchestration, and structural
tests all resolve their configuration through here instead of hard-coding the
values in several places.

Topology (`configs/environments.json`) is hardware-neutral: it declares which
simulators and providers exist, their venv basenames, service modules, and
canonical ports. A profile overlay (`configs/profiles/<name>.json`) binds a
topology to hardware: GPU placement, the build architecture (resolved against
`gpu_runtime.arches` in the dependency lock), request concurrency, optional port
overrides, and an optional node-local venv backing root.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re


def repository_root() -> Path:
    """Repository root (the checkout that owns configs/ and src/)."""
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_topology(repo_root: Path | None = None) -> dict:
    root = repo_root or repository_root()
    topology = _load_json(root / "configs/environments.json")
    if topology.get("schema") != "cap-harness/environments/1":
        raise ValueError(f"unexpected environments schema: {topology.get('schema')!r}")
    return topology


def load_profile(name: str, repo_root: Path | None = None) -> dict:
    root = repo_root or repository_root()
    path = root / "configs/profiles" / f"{name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in (root / "configs/profiles").glob("*.json"))
        raise ValueError(f"unknown profile {name!r}; available: {', '.join(available)}")
    profile = _load_json(path)
    if profile.get("schema") != "cap-harness/profile/1":
        raise ValueError(f"unexpected profile schema: {profile.get('schema')!r}")
    if profile.get("name") != name:
        raise ValueError(f"profile name mismatch: {profile.get('name')!r} != {name!r}")
    return profile


def gpu_runtime(repo_root: Path | None = None) -> dict:
    root = repo_root or repository_root()
    return _load_json(root / "configs/dependency-lock.json")["gpu_runtime"]


EM_CUDA = 190
"""ELF e_machine value of NVIDIA cubins; their e_flags low byte is the SM version (0x59 = sm_89)."""


def compiled_cuda_arches(paths) -> set[str]:
    """The ``sm_XY`` architectures a set of CUDA extension modules were compiled for.

    Cubins are ELF objects embedded in the shared library (what ``cuobjdump --list-elf`` prints);
    PTX, when embedded, names its target in text. A module compiled only for another card reports
    only that card's architecture.
    """
    found: set[str] = set()
    for path in paths:
        try:
            data = Path(path).read_bytes()
        except OSError:
            continue
        offset = data.find(b"\x7fELF")
        while offset != -1:
            header = data[offset : offset + 52]
            if len(header) == 52 and header[4] in (1, 2) and header[5] == 1:
                machine = int.from_bytes(header[18:20], "little")
                flags_at = 48 if header[4] == 2 else 36
                if machine == EM_CUDA:
                    flags = int.from_bytes(header[flags_at : flags_at + 4], "little")
                    found.add(f"sm_{flags & 0xFF}")
            offset = data.find(b"\x7fELF", offset + 4)
        for match in re.findall(rb"(?:sm|compute)_(\d{2,3})", data):
            found.add(f"sm_{int(match)}")
    return found


def arch_for_compute_capability(capability: str) -> str:
    """Map an NVIDIA compute capability such as ``8.9`` to the profile architecture ``sm_89``."""
    major, _, minor = capability.strip().partition(".")
    minor = minor or "0"
    if not major.isdigit() or not minor.isdigit():
        raise ValueError(f"unrecognised compute capability {capability!r}")
    return f"sm_{int(major)}{int(minor)}"


def profile_for_gpu(arch: str, gpu_name: str = "", repo_root: Path | None = None) -> str | None:
    """The hardware profile to compile for a GPU of architecture ``arch`` named ``gpu_name``.

    Several profiles share an architecture (l40, rtx4090 and rtx5880 are all sm_89), so the
    profile whose name appears in the GPU's product name wins; otherwise a single-GPU
    workstation profile (every provider on GPU 0, no node-local venv backing) is preferred over a
    cluster-node one.
    """
    root = repo_root or repository_root()
    candidates = []
    for path in sorted((root / "configs/profiles").glob("*.json")):
        profile = _load_json(path)
        if profile.get("arch") == arch:
            candidates.append(profile)
    if not candidates:
        return None
    normalised_gpu = re.sub(r"[^a-z0-9]", "", gpu_name.lower())
    for profile in candidates:
        if re.sub(r"[^a-z0-9]", "", str(profile["name"]).lower()) in normalised_gpu:
            return str(profile["name"])
    for profile in candidates:
        gpus = profile.get("gpus") or {}
        if all(int(index) == 0 for index in gpus.values()) and not profile.get("venv_root"):
            return str(profile["name"])
    return str(candidates[0]["name"])


def cuda_arch_list(arch: str, repo_root: Path | None = None) -> str:
    """Map a profile architecture (e.g. sm_120) to its TORCH_CUDA_ARCH_LIST value."""
    arches = gpu_runtime(repo_root)["arches"]
    if arch not in arches:
        raise ValueError(f"architecture {arch!r} not in gpu_runtime.arches: {sorted(arches)}")
    return arches[arch]["cuda_arch_list"]


def venv_backing_root(profile: Mapping[str, object] | None = None) -> str | None:
    """Resolve the node-local backing root for `.venv-*` directories.

    Precedence: CAP_HARNESS_VENV_ROOT env var, then the profile's `venv_root`,
    then None (repository-local real directories). When set, bootstraps create
    the real venv under this root and leave a `<repo>/.venv-*` symlink so that
    user-facing paths stay canonical on shared node-local storage.
    """
    env_value = os.environ.get("CAP_HARNESS_VENV_ROOT")
    if env_value:
        return env_value
    if profile is not None:
        value = profile.get("venv_root")
        if value:
            return str(value)
    return None


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    name: str
    venv: str
    module: str | None
    port: int
    gpu: bool
    gpu_index: int | None
    concurrency: int
    optional: bool
    source_build: bool
    arch: str
    cuda_arch_list: str | None


def resolve_provider(
    name: str,
    profile: Mapping[str, object],
    repo_root: Path | None = None,
    topology: Mapping[str, object] | None = None,
) -> ResolvedProvider:
    topo = topology or load_topology(repo_root)
    providers = topo["providers"]
    if name not in providers:
        raise ValueError(f"unknown provider {name!r}; known: {sorted(providers)}")
    spec = providers[name]
    port = int(profile.get("ports", {}).get(name, spec["port"]))
    gpu = bool(spec.get("gpu", False))
    gpu_index = profile.get("gpus", {}).get(name) if gpu else None
    concurrency = int(profile.get("concurrency", {}).get(name, 1))
    arch = str(profile["arch"])
    return ResolvedProvider(
        name=name,
        venv=spec["venv"],
        module=spec.get("module"),
        port=port,
        gpu=gpu,
        gpu_index=None if gpu_index is None else int(gpu_index),
        concurrency=concurrency,
        optional=bool(spec.get("optional", False)),
        source_build=bool(spec.get("source_build", False)),
        arch=arch,
        cuda_arch_list=cuda_arch_list(arch, repo_root) if gpu else None,
    )


def provider_names(
    repo_root: Path | None = None, topology: Mapping[str, object] | None = None
) -> list[str]:
    topo = topology or load_topology(repo_root)
    return list(topo["providers"].keys())


def simulator_names(
    repo_root: Path | None = None, topology: Mapping[str, object] | None = None
) -> list[str]:
    topo = topology or load_topology(repo_root)
    return list(topo["simulators"].keys())


def required_provider_ports(
    repo_root: Path | None = None, topology: Mapping[str, object] | None = None
) -> tuple[int, ...]:
    topo = topology or load_topology(repo_root)
    return tuple(int(p) for p in topo["required_provider_ports"])
