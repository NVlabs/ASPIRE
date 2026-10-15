"""Structured environment diagnostics for the LIBERO harness."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ctypes.util
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import http.client
import importlib
from importlib import metadata as importlib_metadata
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit

from cap_harness import environments

ENVIRONMENT_SCHEMA_VERSION = 1
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 13)
DEFAULT_SERVICES = (
    ("sam3", 8114),
    ("contact_graspnet", 8115),
    ("pyroki", 8116),
)


def topology_service_targets(
    names: Sequence[str] | None = None,
) -> tuple[tuple[str, int], ...]:
    """Resolve (name, port) service targets from the declarative topology.

    With no names, returns the always-on providers (those not compiled from
    source). This keeps doctor embodiment-neutral and topology-driven: ports
    live in configs/environments.json, not hard-coded here or in the CLI. Falls
    back to the static DEFAULT_SERVICES if the topology cannot be read.
    """
    try:
        topology = environments.load_topology()
        providers = topology["providers"]
    except Exception:
        if names is None:
            return DEFAULT_SERVICES
        raise
    if names is None:
        selected = [n for n, spec in providers.items() if not spec.get("source_build")]
    else:
        selected = list(names)
    targets: list[tuple[str, int]] = []
    for name in selected:
        if name not in providers:
            raise ValueError(f"unknown provider {name!r}; known: {sorted(providers)}")
        targets.append((name, int(providers[name]["port"])))
    return tuple(targets)


CORE_PACKAGES = ("cap_harness", "numpy", "scipy", "requests", "yaml")
LIBERO_PACKAGES = ("libero", "robosuite", "mujoco", "gymnasium", "torch")
RUNTIME_DISTRIBUTIONS = (
    "cap-harness",
    "isaacsim",
    "jax",
    "libero",
    "mujoco",
    "numpy",
    "nvidia-curobo",
    "omnigibson",
    "pyroki",
    "robosuite",
    "torch",
)
BEHAVIOR_PACKAGES = ("omnigibson", "bddl", "curobo", "torch")
"""Importable BEHAVIOR packages; Isaac Sim is checked by distribution metadata because importing
it boots Kit (and prompts for the EULA when stdin is not a terminal)."""
BEHAVIOR_DATASET_FILES = (
    "omnigibson-robot-assets/models/r1pro/r1pro.yaml",
    "omnigibson-robot-assets/models/r1pro/curobo/r1pro_description_curobo_arm.yaml",
    "omnigibson-robot-assets/models/r1pro/curobo/r1pro_description_curobo_base.yaml",
    "behavior-1k-assets/VERSION",
    "2026-challenge-task-instances/metadata/task.jsonl",
    "omnigibson.key",
)
SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|authorization|"
    r"credential|cookie|private[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password|passwd|authorization|"
    r"credential|cookie|private[_-]?key|client[_-]?secret)\b(\s*[:=]\s*)"
    r"([^\s,;]+|\"[^\"]*\"|'[^']*')"
)
BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
URL_CREDENTIAL_RE = re.compile(r"(?P<scheme>https?://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_secrets(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact credential-shaped keys and common inline secret forms."""
    if key is not None and SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_secrets(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        redacted = BEARER_RE.sub("Bearer [REDACTED]", value)
        redacted = SECRET_ASSIGNMENT_RE.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted
        )
        return URL_CREDENTIAL_RE.sub(r"\g<scheme>[REDACTED]@", redacted)
    return value


@dataclass(frozen=True, slots=True)
class ResetTarget:
    suite: str | None = None
    task_id: int | None = None
    seed: int = 1
    embodiment: str = "libero"


@dataclass(frozen=True, slots=True)
class DoctorConfig:
    """Configuration for a full L40 check or a non-GPU unit-mode check."""

    repository_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    environment_root: Path | None = None
    expected_environment_basename: str | None = None
    embodiment: str = "libero"
    unit_mode: bool = False
    check_services: bool = True
    service_host: str = "127.0.0.1"
    services: tuple[tuple[str, int], ...] = DEFAULT_SERVICES
    service_timeout_s: float = 1.0
    core_packages: tuple[str, ...] = CORE_PACKAGES
    libero_packages: tuple[str, ...] = LIBERO_PACKAGES
    validation_config: Path | None = None
    libero_submodule: Path | None = None
    libero_path_config: Path | None = None
    egl_vendor_paths: tuple[Path, ...] = (
        Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json"),
        Path("/etc/glvnd/egl_vendor.d/10_nvidia.json"),
    )
    reset: ResetTarget | None = None

    def resolved_validation_config(self) -> Path:
        return (
            self.validation_config
            or Path(__file__).resolve().parent / "configs/libero/validation_cases.yaml"
        )

    def resolved_environment_root(self) -> Path:
        return self.environment_root or self.repository_root / ".venv-libero"

    def resolved_libero_submodule(self) -> Path:
        if self.libero_submodule is not None:
            return self.libero_submodule
        configured = os.environ.get("CAP_HARNESS_LIBERO_SUBMODULE")
        if configured:
            return Path(configured)
        return self.repository_root / "third_party/LIBERO-PRO"

    def resolved_libero_path_config(self) -> Path:
        if self.libero_path_config is not None:
            return self.libero_path_config
        config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", self.repository_root / ".libero"))
        return config_root / "config.yaml"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    required: bool
    summary: str
    details: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "warn", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(asdict(self))


@dataclass(frozen=True, slots=True)
class DoctorReport:
    generated_at: str
    mode: str
    environment: Mapping[str, Any]
    checks: tuple[CheckResult, ...]
    schema_version: int = ENVIRONMENT_SCHEMA_VERSION

    @property
    def success(self) -> bool:
        return not any(check.required and check.status == "fail" for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.success else 1

    def to_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "mode": self.mode,
                "success": self.success,
                "environment": self.environment,
                "checks": [check.to_dict() for check in self.checks],
            }
        )


def environment_snapshot(config: DoctorConfig | None = None) -> dict[str, Any]:
    """Return an allowlisted, secret-free environment description."""
    config = config or DoctorConfig()
    env_allowlist = {
        name: os.environ[name]
        for name in (
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
            "EGL_DEVICE_ID",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        )
        if name in os.environ
    }
    snapshot = {
        "captured_at": utc_now(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "paths": {
            "repository": str(config.repository_root),
            "environment": str(config.resolved_environment_root()),
            "validation_config": str(config.resolved_validation_config()),
            "libero_submodule": str(config.resolved_libero_submodule()),
            "libero_path_config": str(config.resolved_libero_path_config()),
        },
        "graphics_and_devices": env_allowlist,
        "packages": {name: _distribution_version(name) for name in RUNTIME_DISTRIBUTIONS},
        "unit_mode": config.unit_mode,
    }
    return redact_secrets(snapshot)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(redact_secrets(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _writable_directory(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "path does not exist"
    if not path.is_dir():
        return False, "path is not a directory"
    try:
        fd, probe = tempfile.mkstemp(prefix=".cap-harness-doctor-", dir=path)
        os.close(fd)
        Path(probe).unlink()
    except OSError as exc:
        return False, f"write probe failed: {type(exc).__name__}: {exc}"
    return True, None


def _result(
    name: str,
    ok: bool,
    required: bool,
    success_summary: str,
    failure_summary: str,
    details: Mapping[str, Any] | None = None,
    *,
    warning_when_optional: bool = True,
) -> CheckResult:
    if ok:
        status = "pass"
        summary = success_summary
    elif not required and warning_when_optional:
        status = "warn"
        summary = failure_summary
    else:
        status = "fail"
        summary = failure_summary
    return CheckResult(name, status, required, summary, redact_secrets(details or {}))


def check_python(config: DoctorConfig) -> CheckResult:
    del config
    version = sys.version_info[:3]
    ok = SUPPORTED_PYTHON_MIN <= version[:2] < SUPPORTED_PYTHON_MAX
    return _result(
        "python",
        ok,
        True,
        f"Python {platform.python_version()} is supported",
        (
            f"Python {platform.python_version()} is unsupported; expected "
            f">={SUPPORTED_PYTHON_MIN[0]}.{SUPPORTED_PYTHON_MIN[1]} and "
            f"<{SUPPORTED_PYTHON_MAX[0]}.{SUPPORTED_PYTHON_MAX[1]}"
        ),
        {"version": platform.python_version(), "executable": sys.executable},
    )


def check_platform(config: DoctorConfig) -> CheckResult:
    system = platform.system()
    machine = platform.machine().lower()
    ok = system == "Linux" and machine in {"x86_64", "amd64"}
    return _result(
        "platform",
        ok,
        not config.unit_mode,
        "Linux/x86_64 platform detected",
        f"expected Linux/x86_64, found {system}/{machine}",
        {"system": system, "machine": machine, "release": platform.release()},
    )


def check_nvidia(config: DoctorConfig) -> CheckResult:
    required = not config.unit_mode
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return _result(
            "nvidia",
            False,
            required,
            "NVIDIA GPUs are visible",
            "nvidia-smi is unavailable (allowed in CPU-only unit mode)",
            {"gpu_count": 0},
        )
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _result(
            "nvidia",
            False,
            required,
            "NVIDIA GPUs are visible",
            f"nvidia-smi probe failed: {type(exc).__name__}",
        )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    ok = completed.returncode == 0 and bool(rows)
    gpu_models: list[str] = []
    drivers: set[str] = set()
    for row in rows:
        columns = [column.strip() for column in row.split(",")]
        if len(columns) >= 4:
            gpu_models.append(columns[1])
            drivers.add(columns[2])
    return _result(
        "nvidia",
        ok,
        required,
        f"{len(rows)} NVIDIA GPU(s) visible",
        "nvidia-smi returned no visible GPUs (allowed in CPU-only unit mode)",
        {
            "gpu_count": len(rows),
            "models": gpu_models,
            "driver_versions": sorted(drivers),
            "return_code": completed.returncode,
        },
    )


def check_egl(config: DoctorConfig) -> CheckResult:
    required = not config.unit_mode
    mujoco_gl = os.environ.get("MUJOCO_GL")
    pyopengl = os.environ.get("PYOPENGL_PLATFORM")
    vendor_files = [str(path) for path in config.egl_vendor_paths if path.is_file()]
    library = ctypes.util.find_library("EGL")
    # Isaac Sim renders through Vulkan and ignores the MuJoCo EGL switches; only the MuJoCo
    # runtimes need MUJOCO_GL/PYOPENGL_PLATFORM to be set.
    env_ok = config.embodiment == "behavior" or (mujoco_gl == "egl" and pyopengl == "egl")
    config_ok = bool(vendor_files) and bool(library)
    return _result(
        "egl",
        env_ok and config_ok,
        required,
        "EGL environment and NVIDIA vendor configuration are ready",
        "EGL environment/configuration is incomplete (allowed in CPU-only unit mode)",
        {
            "MUJOCO_GL": mujoco_gl,
            "PYOPENGL_PLATFORM": pyopengl,
            "egl_library": library,
            "vendor_files": vendor_files,
        },
    )


def check_source_checkout(config: DoctorConfig) -> CheckResult:
    writable, error = _writable_directory(config.repository_root)
    project_files = all(
        (config.repository_root / relative).is_file()
        for relative in ("pyproject.toml", "uv.lock", "configs/dependency-lock.json")
    )
    ok = writable and project_files
    return _result(
        "source_checkout",
        ok,
        True,
        "repository checkout is writable and contains the project locks",
        "repository checkout is missing, unwritable, or incomplete",
        {
            "path": str(config.repository_root),
            "writable": writable,
            "project_files": project_files,
            "error": error,
        },
    )


def check_build_root(config: DoctorConfig) -> CheckResult:
    environment_root = config.resolved_environment_root()
    writable, error = _writable_directory(environment_root)
    inside_source = _is_relative_to(environment_root, config.repository_root)
    active_environment = Path(sys.prefix).resolve()
    expected_name = config.expected_environment_basename
    name_ok = expected_name is None or active_environment.name == expected_name
    active_ok = expected_name is None or active_environment == environment_root.resolve()
    ok = writable and name_ok and active_ok
    return _result(
        "python_environment",
        ok,
        True,
        "requested Python environment is active and writable",
        "requested Python environment must be active, writable, and correctly named",
        {
            "path": str(environment_root),
            "active_path": str(active_environment),
            "expected_basename": expected_name,
            "active_matches_requested": active_ok,
            "basename_matches": name_ok,
            "writable": writable,
            "inside_source": inside_source,
            "error": error,
        },
    )


def _import_packages(packages: Sequence[str]) -> tuple[dict[str, str], dict[str, str]]:
    imported: dict[str, str] = {}
    failures: dict[str, str] = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
        except BaseException as exc:
            failures[package] = f"{type(exc).__name__}: {redact_secrets(str(exc))}"
            continue
        version = getattr(module, "__version__", None)
        imported[package] = str(version) if version is not None else "imported"
    return imported, failures


def check_core_packages(config: DoctorConfig) -> CheckResult:
    imported, failures = _import_packages(config.core_packages)
    return _result(
        "core_package_imports",
        not failures,
        True,
        "all required core packages import",
        f"{len(failures)} required core package import(s) failed",
        {"imported": imported, "failures": failures},
    )


def check_libero_packages(config: DoctorConfig) -> CheckResult:
    imported, failures = _import_packages(config.libero_packages)
    return _result(
        "libero_package_imports",
        not failures,
        not config.unit_mode,
        "all required LIBERO packages import",
        f"{len(failures)} LIBERO package import(s) failed (allowed in CPU-only unit mode)",
        {"imported": imported, "failures": failures},
    )


def check_robosuite_packages(config: DoctorConfig) -> CheckResult:
    imported, failures = _import_packages(("robosuite", "mujoco"))
    return _result(
        "robosuite_package_imports",
        not failures,
        not config.unit_mode,
        "all required Robosuite packages import",
        f"{len(failures)} Robosuite package import(s) failed",
        {"imported": imported, "failures": failures},
    )


def check_robosuite_source(config: DoctorConfig) -> CheckResult:
    checkout = config.repository_root / "third_party/robosuite"
    lock = config.repository_root / "configs/dependency-lock.json"
    expected = None
    actual = None
    error = None
    try:
        expected = json.loads(lock.read_text(encoding="utf-8"))["dependencies"]["robosuite"][
            "commit"
        ]
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _result(
        "robosuite_source",
        expected is not None and actual == expected,
        True,
        "Robosuite checkout matches the dependency lock",
        "Robosuite checkout is missing or does not match the dependency lock",
        {"path": str(checkout), "expected": expected, "actual": actual, "error": error},
    )


def check_behavior_packages(config: DoctorConfig) -> CheckResult:
    imported, failures = _import_packages(BEHAVIOR_PACKAGES)
    try:
        imported["isaacsim"] = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        failures["isaacsim"] = f"{type(exc).__name__}: {exc}"
    details: dict[str, Any] = {"imported": imported, "failures": failures}
    compiled = False
    if "curobo" in imported:
        try:
            importlib.import_module("curobo.curobolib.geom_cu")
            compiled = True
        except Exception as exc:  # pragma: no cover - only reachable with a broken build
            details["curobo_extension_error"] = f"{type(exc).__name__}: {exc}"
    details["curobo_extension_compiled"] = compiled
    details["python"] = ".".join(str(part) for part in sys.version_info[:3])
    ok = not failures and compiled and sys.version_info[:2] == (3, 11)
    return _result(
        "behavior_package_imports",
        ok,
        not config.unit_mode,
        "Isaac Sim, OmniGibson, BDDL and compiled cuRobo import on Python 3.11",
        "BEHAVIOR runtime is incomplete (Isaac Sim wheels need Python 3.11; cuRobo must be compiled)",
        details,
    )


compiled_cuda_arches = environments.compiled_cuda_arches


def _device_arch(gpu_index: int) -> str | None:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability(gpu_index)
    except Exception:
        return None
    return f"sm_{major}{minor}"


def check_behavior_curobo_arch(
    config: DoctorConfig,
    *,
    arches: set[str] | None = None,
    device_arch: str | None = None,
) -> CheckResult:
    """CuRobo's compiled kernels must include the architecture of the GPU that runs them.

    The bootstrap compiles for one profile's architecture; a venv built for another card (or
    copied from one) imports fine and fails at the first plan, so the mismatch is caught here.
    """
    details: dict[str, Any] = {}
    if arches is None:
        try:
            module = importlib.import_module("curobo.curobolib")
            paths = sorted(Path(module.__file__).parent.glob("*.so"))
            details["extension_modules"] = [path.name for path in paths]
            arches = compiled_cuda_arches(paths)
        except Exception as exc:
            details["error"] = f"{type(exc).__name__}: {exc}"
            arches = set()
    if device_arch is None:
        gpu = os.environ.get("OMNIGIBSON_GPU_ID", "0")
        device_arch = _device_arch(int(gpu) if gpu.isdigit() else 0)
    details["compiled_arches"] = sorted(arches)
    details["device_arch"] = device_arch
    undetermined = not arches or device_arch is None
    ok = undetermined or device_arch in arches
    if undetermined:
        details["note"] = "compiled or device architecture could not be determined"
    return _result(
        "behavior_curobo_arch",
        ok,
        not config.unit_mode,
        f"cuRobo kernels compiled for {sorted(arches) or 'unknown'} include this GPU ({device_arch})",
        f"cuRobo kernels were compiled for {sorted(arches)} but GPU is {device_arch}; rebuild with "
        "scripts/bootstrap_behavior.sh (it detects the GPU) or --profile <matching profile>",
        details,
    )


def check_behavior_source(config: DoctorConfig) -> CheckResult:
    checkout = config.repository_root / "third_party/BEHAVIOR-1K"
    lock = config.repository_root / "configs/dependency-lock.json"
    expected = actual = error = None
    try:
        expected = json.loads(lock.read_text(encoding="utf-8"))["dependencies"]["behavior_1k"][
            "commit"
        ]
        actual = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return _result(
        "behavior_source",
        expected is not None and actual == expected,
        True,
        "BEHAVIOR-1K checkout matches the dependency lock",
        "BEHAVIOR-1K checkout is missing or does not match the dependency lock",
        {"path": str(checkout), "expected": expected, "actual": actual, "error": error},
    )


def behavior_data_root() -> Path:
    """Dataset root: OMNIGIBSON_DATA_PATH, else CAP_HARNESS_BEHAVIOR_DATA, else ~/behavior-data."""
    for name in ("OMNIGIBSON_DATA_PATH", "CAP_HARNESS_BEHAVIOR_DATA"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return Path.home() / "behavior-data"


def check_behavior_datasets(config: DoctorConfig) -> CheckResult:
    root = behavior_data_root()
    present = {name: (root / name).is_file() for name in BEHAVIOR_DATASET_FILES}
    missing = [name for name, ok in present.items() if not ok]
    return _result(
        "behavior_datasets",
        not missing,
        not config.unit_mode,
        "BEHAVIOR assets, robot assets, task instances and the decryption key are present",
        f"{len(missing)} BEHAVIOR dataset payload file(s) missing under {root}",
        {"data_root": str(root), "present": present, "missing": missing},
    )


def check_behavior_environment(config: DoctorConfig) -> CheckResult:
    appdata = os.environ.get("OMNIGIBSON_APPDATA_PATH")
    gpu_id = os.environ.get("OMNIGIBSON_GPU_ID")
    headless = os.environ.get("OMNIGIBSON_HEADLESS", "")
    problems: list[str] = []
    if headless.lower() not in {"1", "true", "t"}:
        problems.append("OMNIGIBSON_HEADLESS is not set to 1")
    if gpu_id is not None and not gpu_id.isdigit():
        problems.append("OMNIGIBSON_GPU_ID must be an integer GPU index")
    if appdata is not None and not os.access(appdata, os.W_OK):
        problems.append("OMNIGIBSON_APPDATA_PATH is not writable")
    for name in ("EXP_PATH", "CARB_APP_PATH", "ISAAC_PATH"):
        if os.environ.get(name):
            problems.append(f"{name} is set; a foreign Isaac Sim install would shadow the wheels")
    return _result(
        "behavior_environment",
        not problems,
        not config.unit_mode,
        "OmniGibson process environment is headless, GPU-pinned and writable",
        "; ".join(problems) or "OmniGibson process environment is misconfigured",
        {
            "OMNIGIBSON_HEADLESS": headless,
            "OMNIGIBSON_GPU_ID": gpu_id,
            "OMNIGIBSON_APPDATA_PATH": appdata,
            "OMNIGIBSON_DATA_PATH": str(behavior_data_root()),
            "problems": problems,
        },
    )


def check_libero_paths(config: DoctorConfig) -> CheckResult:
    validation_config = config.resolved_validation_config()
    submodule = config.resolved_libero_submodule()
    path_config = config.resolved_libero_path_config()
    config_ok = validation_config.is_file()
    submodule_ok = submodule.is_dir() and any(submodule.iterdir())
    required_path_keys = ("benchmark_root", "bddl_files", "init_states", "datasets", "assets")
    configured_paths: dict[str, str] = {}
    path_errors: list[str] = []
    if path_config.is_file():
        try:
            import yaml
        except ImportError:
            path_errors.append("PyYAML is unavailable")
        else:
            try:
                loaded = yaml.safe_load(path_config.read_text(encoding="utf-8"))
                if not isinstance(loaded, Mapping):
                    raise ValueError("config root is not a mapping")
                for key in required_path_keys:
                    raw_path = loaded.get(key)
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        path_errors.append(f"{key} is missing")
                        continue
                    resolved = Path(raw_path).expanduser().resolve()
                    configured_paths[key] = str(resolved)
                    if not resolved.exists():
                        path_errors.append(f"{key} does not exist")
            except (OSError, ValueError, yaml.YAMLError) as exc:
                path_errors.append(f"cannot parse path config: {type(exc).__name__}: {exc}")
    else:
        path_errors.append("path config is missing")
    path_config_ok = not path_errors
    return _result(
        "libero_paths",
        config_ok and submodule_ok and path_config_ok,
        not config.unit_mode,
        "LIBERO validation config, path config, and submodule are valid",
        "LIBERO validation config, upstream path config, or populated submodule is invalid",
        {
            "validation_config": str(validation_config),
            "validation_config_present": config_ok,
            "submodule": str(submodule),
            "submodule_populated": submodule_ok,
            "path_config": str(path_config),
            "configured_paths": configured_paths,
            "path_errors": path_errors,
        },
    )


def _http_status(host: str, port: int, timeout_s: float) -> tuple[bool, int | None, str | None]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request("GET", "/openapi.json", headers={"Accept": "application/json"})
        response = connection.getresponse()
        response.read(4096)
        responding = 200 <= response.status < 300
        error = None if responding else f"unexpected HTTP {response.status}"
        return responding, response.status, error
    except (OSError, http.client.HTTPException) as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()


def check_services(config: DoctorConfig) -> CheckResult:
    if not config.check_services:
        return CheckResult(
            "services",
            "skipped",
            False,
            "service checks explicitly skipped",
            {"services": [{"name": name, "port": port} for name, port in config.services]},
        )
    results: list[dict[str, Any]] = []
    for name, port in config.services:
        responding, status, error = _http_status(
            config.service_host, port, config.service_timeout_s
        )
        results.append(
            {
                "name": name,
                "host": config.service_host,
                "port": port,
                "responding": responding,
                "http_status": status,
                "error": error,
            }
        )
    ok = all(result["responding"] for result in results)
    return _result(
        "services",
        ok,
        not config.unit_mode,
        "all requested model services are responding",
        "one or more model services are unreachable (allowed in CPU-only unit mode)",
        {"services": results},
    )


def _default_reset_probe(target: ResetTarget) -> Mapping[str, Any]:
    """Reset one task through the registry without taking any action."""
    if target.embodiment == "robosuite":
        from cap_harness.robosuite.adapter import RobosuiteAdapter
        from cap_harness.robosuite.registry import RobosuiteTaskRegistry
        from cap_harness.validation import validate_observation_schema

        registry = RobosuiteTaskRegistry()
        task_name = target.suite or registry.available_tasks[0]
        metadata = registry.resolve(task_name, target.task_id)
        adapter = RobosuiteAdapter(registry=registry)
        try:
            observation = adapter.reset(metadata, target.seed)
            language = adapter.get_task_context().language
            validate_observation_schema(observation)
        finally:
            adapter.close()
        return {
            "embodiment": "robosuite",
            "suite": metadata.suite_name,
            "task_id": metadata.task_id,
            "seed": target.seed,
            "task_language_present": bool(language),
        }
    if target.embodiment == "behavior":
        from cap_harness.behavior.adapter import BehaviorAdapter
        from cap_harness.behavior.registry import BehaviorTaskRegistry
        from cap_harness.validation import validate_observation_schema

        registry = BehaviorTaskRegistry()
        task_name = target.suite or registry.available_tasks[0]
        metadata = registry.resolve(task_name, target.task_id)
        adapter = BehaviorAdapter(
            registry=registry, camera_width=256, camera_height=256, horizon=300
        )
        try:
            observation = adapter.reset(metadata, target.seed)
            language = adapter.get_task_context().language
            validate_observation_schema(observation)
        finally:
            adapter.close()
        return {
            "embodiment": "behavior",
            "suite": metadata.suite_name,
            "task_id": metadata.task_id,
            "seed": target.seed,
            "task_language_present": bool(language),
        }
    if target.embodiment != "libero":
        raise ValueError("reset embodiment must be 'behavior', 'libero' or 'robosuite'")

    from cap_harness.validation import (
        _default_adapter_factory,
        _reset_adapter,
        _runtime_language,
        _unpack_reset,
        build_required_manifest,
        load_libero_registry,
        validate_observation_schema,
    )

    registry = load_libero_registry()
    manifest = build_required_manifest(registry)
    if target.suite is None:
        entry = manifest.representatives[sorted(manifest.representatives)[0]]
    else:
        matches = [
            item
            for item in manifest
            if item.suite == target.suite
            and (target.task_id is None or item.task_id == target.task_id)
        ]
        if not matches:
            raise RuntimeError(f"reset target not found: {target.suite}/{target.task_id}")
        entry = sorted(matches, key=lambda item: item.task_id)[0]
    adapter = _default_adapter_factory(registry)(entry, target.seed)
    try:
        reset_result = _reset_adapter(adapter, entry, target.seed)
        observation, info = _unpack_reset(reset_result)
        language = _runtime_language(adapter, info)
        validate_observation_schema(observation)
    finally:
        adapter.close()
    return {
        "embodiment": "libero",
        "suite": entry.suite,
        "task_id": entry.task_id,
        "seed": target.seed,
        "task_language_present": bool(language),
    }


def check_reset(
    config: DoctorConfig,
    reset_probe: Callable[[ResetTarget], Mapping[str, Any]] | None,
) -> CheckResult:
    if config.reset is None:
        return CheckResult(
            "single_reset",
            "skipped",
            False,
            "single reset not requested",
            {},
        )
    probe = reset_probe or _default_reset_probe
    try:
        details = probe(config.reset)
    except BaseException as exc:
        return _result(
            "single_reset",
            False,
            True,
            "one simulation reset completed",
            f"simulation reset failed: {type(exc).__name__}: {redact_secrets(str(exc))}",
            {"target": asdict(config.reset)},
        )
    return _result(
        "single_reset",
        True,
        True,
        "one simulation reset completed and closed cleanly",
        "simulation reset failed",
        {"target": asdict(config.reset), "result": details},
    )


def _timed_check(name: str, operation: Callable[[], CheckResult], *, required: bool) -> CheckResult:
    started = time.monotonic()
    try:
        result = operation()
    except BaseException as exc:
        result = CheckResult(
            name,
            "fail" if required else "warn",
            required,
            f"check raised {type(exc).__name__}: {redact_secrets(str(exc))}",
            {},
        )
    duration_ms = round((time.monotonic() - started) * 1000)
    return CheckResult(
        result.name,
        result.status,
        result.required,
        result.summary,
        result.details,
        duration_ms,
    )


def run_doctor(
    config: DoctorConfig | None = None,
    *,
    output_path: str | os.PathLike[str] | None = None,
    reset_probe: Callable[[ResetTarget], Mapping[str, Any]] | None = None,
) -> DoctorReport:
    """Run all checks, write ``environment.json`` if requested, and return a report."""
    config = config or DoctorConfig()
    embodiment_checks: tuple[tuple[str, bool, Callable[[], CheckResult]], ...]
    if config.embodiment == "robosuite":
        embodiment_checks = (
            (
                "robosuite_package_imports",
                not config.unit_mode,
                lambda: check_robosuite_packages(config),
            ),
            ("robosuite_source", True, lambda: check_robosuite_source(config)),
        )
    elif config.embodiment == "libero":
        embodiment_checks = (
            (
                "libero_package_imports",
                not config.unit_mode,
                lambda: check_libero_packages(config),
            ),
            ("libero_paths", not config.unit_mode, lambda: check_libero_paths(config)),
        )
    elif config.embodiment == "behavior":
        embodiment_checks = (
            (
                "behavior_package_imports",
                not config.unit_mode,
                lambda: check_behavior_packages(config),
            ),
            (
                "behavior_curobo_arch",
                not config.unit_mode,
                lambda: check_behavior_curobo_arch(config),
            ),
            ("behavior_source", True, lambda: check_behavior_source(config)),
            ("behavior_datasets", not config.unit_mode, lambda: check_behavior_datasets(config)),
            (
                "behavior_environment",
                not config.unit_mode,
                lambda: check_behavior_environment(config),
            ),
        )
    else:
        raise ValueError("doctor embodiment must be 'behavior', 'libero' or 'robosuite'")
    operations: tuple[tuple[str, bool, Callable[[], CheckResult]], ...] = (
        ("python", True, lambda: check_python(config)),
        ("platform", not config.unit_mode, lambda: check_platform(config)),
        ("nvidia", not config.unit_mode, lambda: check_nvidia(config)),
        ("egl", not config.unit_mode, lambda: check_egl(config)),
        ("source_checkout", True, lambda: check_source_checkout(config)),
        ("python_environment", True, lambda: check_build_root(config)),
        ("core_package_imports", True, lambda: check_core_packages(config)),
        *embodiment_checks,
        ("services", not config.unit_mode, lambda: check_services(config)),
        ("single_reset", config.reset is not None, lambda: check_reset(config, reset_probe)),
    )
    checks = tuple(
        _timed_check(name, operation, required=required) for name, required, operation in operations
    )
    report = DoctorReport(
        generated_at=utc_now(),
        mode="unit" if config.unit_mode else "full",
        environment=environment_snapshot(config),
        checks=checks,
    )
    if output_path is not None:
        _atomic_write_json(Path(output_path), report.to_dict())
    return report


def parse_service_url(url: str) -> tuple[str, int]:
    """Small validated helper for callers that override local service addresses."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ValueError("service URL must include http(s), host, and port")
    return parsed.hostname, parsed.port
