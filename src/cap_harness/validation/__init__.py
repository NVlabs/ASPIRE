"""LIBERO-Pro manifest construction and resumable validation.

The module deliberately has no LIBERO imports at module import time.  Unit tests and
``cap-harness --help`` must work on machines without MuJoCo, CUDA, or LIBERO.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any

import numpy as np

EXPECTED_PAIR_COUNT = 80
SMOKE_SEEDS = (1,)
NIGHTLY_SEEDS = (1, 2, 3)
EXPECTED_CONTROL_HZ = 20.0
EXPECTED_CONTROL_DT_S = 1.0 / EXPECTED_CONTROL_HZ
# This module lives at src/cap_harness/validation/__init__.py, so the cap_harness
# package root (which holds configs/) is two parents up from this file.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = _PACKAGE_ROOT / "configs/libero/validation_cases.yaml"


class ManifestError(ValueError):
    """Raised when the installed LIBERO registry does not match the release gate."""


class ValidationError(RuntimeError):
    """Raised when a validation invariant fails for one matrix case."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One authoritative suite/task pair discovered through ``LiberoSuiteRegistry``."""

    suite: str
    task_id: int
    task_name: str
    task_language: str
    bddl_path: str
    init_state_count: int
    family: str
    variant: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.suite, self.task_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationManifest:
    """Validated release-gate manifest and deterministic representative cases."""

    entries: tuple[ManifestEntry, ...]
    representatives: Mapping[str, ManifestEntry]
    required_count: int = EXPECTED_PAIR_COUNT
    schema_version: int = 1

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "required_count": self.required_count,
            "count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "representatives": {
                name: entry.to_dict() for name, entry in sorted(self.representatives.items())
            },
        }


@dataclass(frozen=True, slots=True)
class MatrixKey:
    suite: str
    task_id: int
    seed: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> MatrixKey:
        key = record.get("key", record)
        if not isinstance(key, Mapping):
            # Preserve the invalid-data ValueError contract.
            raise ValueError("matrix record key must be an object")
        return cls(str(key["suite"]), int(key["task_id"]), int(key["seed"]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by installation doctor
        raise ManifestError("PyYAML is required to load validation cases") from exc

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read validation config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid validation config: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ManifestError(f"validation config must contain a mapping: {path}")
    return loaded


def load_validation_cases(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and minimally validate the release-gate YAML."""
    config_path = Path(path) if path is not None else DEFAULT_CASES_PATH
    config = _load_yaml(config_path)
    suites = config.get("required_suites")
    representatives = config.get("representatives")
    if not isinstance(suites, list) or not suites:
        raise ManifestError("validation config requires a nonempty required_suites list")
    if not isinstance(representatives, Mapping) or not representatives:
        raise ManifestError("validation config requires representative selection rules")
    names = [item.get("name") if isinstance(item, Mapping) else None for item in suites]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ManifestError("each required suite needs a nonempty name")
    if len(names) != len(set(names)):
        raise ManifestError("required suite names must be unique")
    return config


def load_libero_registry() -> Any:
    """Instantiate the production registry without importing LIBERO eagerly.

    The first path is the canonical location.  The alternatives make the validation
    tooling tolerant of a package split while the adapter remains independently owned.
    """
    candidates = (
        "cap_harness.libero",
        "cap_harness.adapters.libero",
        "cap_harness.libero.registry",
    )
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(f"{module_name}: {type(exc).__name__}")
            continue
        registry_type = getattr(module, "LiberoSuiteRegistry", None)
        if registry_type is not None:
            return registry_type()
        errors.append(f"{module_name}: LiberoSuiteRegistry missing")
    joined = "; ".join(errors)
    raise ManifestError(f"LiberoSuiteRegistry is unavailable ({joined})")


def _read_field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _call_first(target: Any, names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        method = getattr(target, name, None)
        if callable(method):
            return method(*args, **kwargs)
    raise AttributeError(f"none of {', '.join(names)} are available")


def _tasks_for_suite(registry: Any, suite_name: str) -> list[Any]:
    enumerate_tasks = getattr(registry, "enumerate_tasks", None)
    if callable(enumerate_tasks):
        return list(enumerate_tasks((suite_name,)))
    for method_name in ("tasks_for_suite", "list_tasks", "iter_tasks", "get_tasks"):
        method = getattr(registry, method_name, None)
        if callable(method):
            return list(method(suite_name))

    suite = None
    for method_name in ("get_suite", "suite", "load_suite"):
        method = getattr(registry, method_name, None)
        if callable(method):
            suite = method(suite_name)
            break
    if suite is None:
        suites = _read_field(registry, "suites")
        if isinstance(suites, Mapping):
            suite = suites.get(suite_name)
    if suite is None:
        raise ManifestError(
            f"registry cannot enumerate suite {suite_name!r}; expected tasks_for_suite()"
        )

    tasks = _read_field(suite, "tasks")
    if tasks is not None:
        return list(tasks)
    count = _read_field(suite, "n_tasks", "num_tasks")
    if count is None:
        getter = getattr(suite, "get_num_tasks", None)
        count = getter() if callable(getter) else None
    get_task = getattr(suite, "get_task", None)
    if count is not None and callable(get_task):
        return [get_task(task_id) for task_id in range(int(count))]
    raise ManifestError(f"registry suite {suite_name!r} does not expose tasks")


def _normalize_entry(
    registry: Any,
    suite_name: str,
    task: Any,
    position: int,
    *,
    family: str,
    variant: str,
) -> ManifestEntry:
    task_id = int(_read_field(task, "task_id", "id", "index", default=position))
    task_name = str(_read_field(task, "task_name", "name", default="")).strip()
    language = str(
        _read_field(task, "task_language", "language", "description", default="")
    ).strip()
    bddl_path_value = _read_field(
        task,
        "bddl_path",
        "bddl_file_path",
        "bddl_file_name",
        "bddl_file",
        default="",
    )
    bddl_path = str(bddl_path_value).strip()
    init_state_count = _read_field(task, "init_state_count", "num_init_states")
    if init_state_count is None:
        init_states = _read_field(task, "init_states")
        if init_states is not None:
            try:
                init_state_count = len(init_states)
            except TypeError:
                init_state_count = None
    if init_state_count is None:
        for method_name in ("get_task_init_states", "task_init_states"):
            method = getattr(registry, method_name, None)
            if not callable(method):
                continue
            try:
                states = method(suite_name, task_id)
            except TypeError:
                states = method(task_id)
            init_state_count = len(states)
            break

    if not task_name:
        raise ManifestError(f"{suite_name}[{task_id}] has an empty task name")
    if not language:
        raise ManifestError(
            f"{suite_name}/{task_name} has no authoritative benchmark/BDDL language"
        )
    if not bddl_path:
        raise ManifestError(f"{suite_name}/{task_name} has no BDDL path")
    if init_state_count is None or int(init_state_count) <= 0:
        raise ManifestError(f"{suite_name}/{task_name} has no initial states")

    return ManifestEntry(
        suite=suite_name,
        task_id=task_id,
        task_name=task_name,
        task_language=language,
        bddl_path=bddl_path,
        init_state_count=int(init_state_count),
        family=family,
        variant=variant,
    )


def select_representatives(
    entries: Iterable[ManifestEntry], config: Mapping[str, Any]
) -> dict[str, ManifestEntry]:
    """Select representatives using only explicit, deterministic YAML rules.

    Candidates are ordered by configured suite priority, configured task-id priority,
    numeric task id, then task name. Field-specific exact, substring, and regex rules
    have no random or registry-order fallback.
    """
    all_entries = tuple(entries)
    rules = config.get("representatives")
    if not isinstance(rules, Mapping):
        raise ManifestError("representatives must be a mapping")

    selected: dict[str, ManifestEntry] = {}
    for representative_name, raw_rule in rules.items():
        if not isinstance(representative_name, str) or not isinstance(raw_rule, Mapping):
            raise ManifestError("representative rules must map names to objects")
        family = str(raw_rule.get("family", "")).strip()
        suite_priority = tuple(str(item) for item in raw_rule.get("suite_priority", ()))
        task_id_priority = tuple(int(item) for item in raw_rule.get("task_id_priority", ()))
        match_any = tuple(str(item) for item in raw_rule.get("match_any", ()))
        match_all = tuple(str(item) for item in raw_rule.get("match_all", ()))
        language_match_any = tuple(str(item) for item in raw_rule.get("language_match_any", ()))
        language_contains_any = tuple(
            str(item).casefold() for item in raw_rule.get("language_contains_any", ())
        )
        task_name_exact = raw_rule.get("task_name_exact")
        if task_name_exact is not None:
            task_name_exact = str(task_name_exact)

        if not family or not suite_priority:
            raise ManifestError(
                f"representative {representative_name!r} needs family and suite_priority"
            )
        try:
            any_patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in match_any)
            all_patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in match_all)
            language_patterns = tuple(
                re.compile(pattern, re.IGNORECASE) for pattern in language_match_any
            )
        except re.error as exc:
            raise ManifestError(
                f"invalid regex for representative {representative_name!r}: {exc}"
            ) from exc

        def matches(
            entry: ManifestEntry,
            *,
            family: str = family,
            suite_priority: tuple[str, ...] = suite_priority,
            task_name_exact: str | None = task_name_exact,
            language_contains_any: tuple[str, ...] = language_contains_any,
            language_patterns: tuple[re.Pattern[str], ...] = language_patterns,
            any_patterns: tuple[re.Pattern[str], ...] = any_patterns,
            all_patterns: tuple[re.Pattern[str], ...] = all_patterns,
        ) -> bool:
            if entry.family != family or entry.suite not in suite_priority:
                return False
            if task_name_exact is not None and entry.task_name != task_name_exact:
                return False
            language_folded = entry.task_language.casefold()
            if language_contains_any and not any(
                text in language_folded for text in language_contains_any
            ):
                return False
            if language_patterns and not any(
                pattern.search(entry.task_language) for pattern in language_patterns
            ):
                return False
            searchable = f"{entry.task_name}\n{entry.task_language}"
            if any_patterns and not any(pattern.search(searchable) for pattern in any_patterns):
                return False
            return all(pattern.search(searchable) for pattern in all_patterns)

        candidates = [entry for entry in all_entries if matches(entry)]
        if not candidates:
            raise ManifestError(f"representative {representative_name!r} matched no manifest entry")
        suite_rank = {suite: index for index, suite in enumerate(suite_priority)}
        task_rank = {task_id: index for index, task_id in enumerate(task_id_priority)}
        candidates.sort(
            key=lambda entry: (
                suite_rank[entry.suite],
                task_rank.get(entry.task_id, len(task_rank)),
                entry.task_id,
                entry.task_name,
            )
        )
        selected[representative_name] = candidates[0]
    return selected


def build_required_manifest(
    registry: Any | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
) -> ValidationManifest:
    """Discover and validate exactly 80 required LIBERO-Pro pairs."""
    registry = registry if registry is not None else load_libero_registry()
    config = load_validation_cases(config_path)
    expected_total = int(config.get("required_count", EXPECTED_PAIR_COUNT))
    if expected_total != EXPECTED_PAIR_COUNT:
        raise ManifestError(
            f"release config required_count must be exactly {EXPECTED_PAIR_COUNT}, got {expected_total}"
        )

    entries: list[ManifestEntry] = []
    for suite_config in config["required_suites"]:
        if not isinstance(suite_config, Mapping):
            raise ManifestError("required suite entries must be objects")
        suite_name = str(suite_config["name"])
        family = str(suite_config.get("family", "")).strip()
        variant = str(suite_config.get("variant", "")).strip()
        expected_tasks = int(suite_config.get("expected_tasks", 10))
        tasks = _tasks_for_suite(registry, suite_name)
        if len(tasks) != expected_tasks:
            raise ManifestError(
                f"suite {suite_name!r} must have {expected_tasks} tasks, found {len(tasks)}"
            )
        normalized = [
            _normalize_entry(
                registry,
                suite_name,
                task,
                position,
                family=family,
                variant=variant,
            )
            for position, task in enumerate(tasks)
        ]
        normalized.sort(key=lambda entry: (entry.task_id, entry.task_name))
        entries.extend(normalized)

    if len(entries) != EXPECTED_PAIR_COUNT:
        raise ManifestError(
            f"required manifest must contain exactly {EXPECTED_PAIR_COUNT} pairs, found {len(entries)}"
        )
    keys = [entry.key for entry in entries]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise ManifestError(f"manifest contains duplicate suite/task keys: {duplicates}")
    representatives = select_representatives(entries, config)
    return ValidationManifest(tuple(entries), representatives)


# Backwards-friendly, concise public alias used by the CLI.
build_manifest = build_required_manifest


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Write JSON with fsync + replace so readers never observe a partial document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class AtomicJsonlStore:
    """Append-only JSONL store with process-safe records and resumable keys."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                is_partial_tail = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
                if is_partial_tail:
                    break
                raise ValidationError(
                    f"invalid complete JSONL record at {self.path}:{index + 1}"
                ) from exc
            if not isinstance(value, dict):
                raise ValidationError(f"matrix record at line {index + 1} is not an object")
            records.append(value)
        return records

    def latest_by_key(
        self, *, validation_fingerprint: str | None = None
    ) -> dict[MatrixKey, dict[str, Any]]:
        latest: dict[MatrixKey, dict[str, Any]] = {}
        for record in self.records():
            if (
                validation_fingerprint is not None
                and record.get("validation_fingerprint") != validation_fingerprint
            ):
                continue
            latest[MatrixKey.from_record(record)] = record
        return latest

    def append(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - Windows is not a supported runtime target
                pass
            size = os.fstat(fd).st_size
            if size:
                tail_start = max(0, size - 1_048_576)
                tail = os.pread(fd, size - tail_start, tail_start)
                if tail and not tail.endswith(b"\n"):
                    boundary = tail.rfind(b"\n")
                    candidate = tail[boundary + 1 :]
                    try:
                        json.loads(candidate)
                    except json.JSONDecodeError:
                        os.ftruncate(fd, tail_start + boundary + 1)
                    else:
                        separator_written = os.write(fd, b"\n")
                        if separator_written != 1:
                            raise OSError("short JSONL separator append")
            written = os.write(fd, encoded)
            if written != len(encoded):
                raise OSError(f"short atomic append: wrote {written} of {len(encoded)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return repr(value)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_environment_identity(value: Any) -> Any:
    volatile_keys = {
        "captured_at",
        "duration_ms",
        "duration_s",
        "finished_at",
        "generated_at",
        "started_at",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_environment_identity(item)
            for key, item in value.items()
            if str(key) not in volatile_keys
        }
    if isinstance(value, list | tuple):
        return [_stable_environment_identity(item) for item in value]
    return _jsonable(value)


def _source_fingerprint(package_root: Path) -> str:
    hasher = hashlib.sha256()
    included = 0
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".json", ".yaml"}:
            continue
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        hasher.update(relative)
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
        included += 1
    if included == 0:
        raise ManifestError(f"no validation source files found under {package_root}")
    return hasher.hexdigest()


def build_validation_fingerprints(
    manifest: ValidationManifest,
    *,
    environment: Mapping[str, Any],
    repository_root: str | os.PathLike[str] | None = None,
) -> dict[str, str | int]:
    """Fingerprint code, manifest, and the pinned standalone dependency lock."""
    package_root = _PACKAGE_ROOT
    repository_path = Path(repository_root or _PACKAGE_ROOT.parent.parent).expanduser()
    dependency_lock = repository_path / "configs/dependency-lock.json"
    if not dependency_lock.is_file():
        raise ManifestError(f"pinned dependency lock is missing: {dependency_lock}")
    try:
        dependency_payload = json.loads(dependency_lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load pinned dependency lock: {dependency_lock}") from exc

    components: dict[str, str | int] = {
        "schema_version": 2,
        "source_sha256": _source_fingerprint(package_root),
        "manifest_sha256": _canonical_sha256(manifest.to_dict()),
        "dependency_lock_sha256": _canonical_sha256(dependency_payload),
        "environment_sha256": _canonical_sha256(_stable_environment_identity(environment)),
    }
    components["combined_sha256"] = _canonical_sha256(components)
    return components


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(item) for item in shape)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if not value:
            return (0,)
        return (len(value), *_shape(value[0]))
    return ()


def _flatten_numeric(value: Any) -> list[float]:
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        value = value.reshape(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    try:
        return [float(value)]
    except (TypeError, ValueError) as exc:
        raise ValidationError("schema contains a nonnumeric value") from exc


def _require_finite(value: Any, label: str) -> None:
    flattened = _flatten_numeric(value)
    if not flattened or not all(math.isfinite(item) for item in flattened):
        raise ValidationError(f"{label} must be nonempty and finite")


def validate_observation_schema(observation: Any) -> dict[str, Any]:
    """Validate two calibrated cameras and a seven-joint robot state."""
    cameras = _read_field(observation, "cameras", "camera_observations")
    if cameras is None and isinstance(observation, Mapping):
        camera_names = (
            "agent",
            "agentview",
            "external",
            "wrist",
            "robot0_eye_in_hand",
            "eye_in_hand",
        )
        cameras = {name: observation[name] for name in camera_names if name in observation}
    if not isinstance(cameras, Mapping) or len(cameras) < 2:
        raise ValidationError("observation must contain at least two named cameras")

    lowered = {str(name).lower(): camera for name, camera in cameras.items()}
    has_wrist = any("wrist" in name or "eye_in_hand" in name for name in lowered)
    has_agent = any(
        "agent" in name or "external" in name or "head" in name or name in {"front", "main"}
        for name in lowered
    )
    if not has_wrist or not has_agent:
        raise ValidationError("observation requires distinct agent/external and wrist cameras")

    camera_summary: dict[str, Any] = {}
    for name, camera in sorted(cameras.items(), key=lambda item: str(item[0])):
        rgb = _read_field(camera, "rgb", "color", "image")
        depth = _read_field(camera, "depth_m", "depth", "depth_image")
        intrinsics = _read_field(camera, "intrinsics", "camera_matrix", "K")
        extrinsics = _read_field(
            camera, "extrinsics", "camera_to_world", "world_from_camera", "transform"
        )
        if extrinsics is None:
            camera_pose = _read_field(camera, "camera_pose", "pose")
            as_matrix = getattr(camera_pose, "as_matrix", None)
            if callable(as_matrix):
                extrinsics = as_matrix()
        rgb_shape = _shape(rgb)
        depth_shape = _shape(depth)
        if len(rgb_shape) != 3 or rgb_shape[-1] not in (3, 4):
            raise ValidationError(f"camera {name!r} RGB must have shape HxWx3/4")
        if len(depth_shape) not in (2, 3) or depth_shape[:2] != rgb_shape[:2]:
            raise ValidationError(f"camera {name!r} depth must align with RGB")
        if _shape(intrinsics) != (3, 3):
            raise ValidationError(f"camera {name!r} intrinsics must be 3x3")
        if _shape(extrinsics) != (4, 4):
            raise ValidationError(f"camera {name!r} extrinsics must be 4x4")
        _require_finite(depth, f"camera {name!r} depth")
        _require_finite(intrinsics, f"camera {name!r} intrinsics")
        _require_finite(extrinsics, f"camera {name!r} extrinsics")
        camera_summary[str(name)] = {"rgb_shape": rgb_shape, "depth_shape": depth_shape}

    robot = _read_field(observation, "robot", "robot_state", "state")
    if robot is None:
        raise ValidationError("observation has no robot state")
    joints = _read_field(robot, "joint_positions", "q", "qpos", "arm_joint_positions")
    if isinstance(joints, Mapping):
        joints = joints.get("primary", next(iter(joints.values()), None))
    joint_values = _flatten_numeric(joints)
    if len(joint_values) != 7 or not all(math.isfinite(item) for item in joint_values):
        raise ValidationError("robot state must contain seven finite arm joints")
    gripper = _read_field(
        robot,
        "gripper_position",
        "gripper_positions",
        "gripper",
        "gripper_state",
    )
    if isinstance(gripper, Mapping):
        gripper = gripper.get("primary", next(iter(gripper.values()), None))
    _require_finite(gripper, "robot gripper state")
    return {"cameras": camera_summary, "arm_joint_count": 7}


def _unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], result[1]
    observation = _read_field(result, "observation", "obs", default=result)
    info = _read_field(result, "info", default={})
    return observation, info if isinstance(info, Mapping) else {}


def _unpack_step(result: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    if isinstance(result, tuple):
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, done, info = result
            terminated, truncated = done, False
        else:
            raise ValidationError("runtime step returned an unsupported tuple")
    else:
        ok = _read_field(result, "ok", default=True)
        if ok is False:
            error = _read_field(result, "error")
            message = _read_field(error, "message", default="runtime step failed")
            raise ValidationError(str(message))
        observation = _read_field(result, "observation", "obs")
        reward = _read_field(result, "reward", "runtime_reward")
        terminated = bool(_read_field(result, "terminated", "done", default=False))
        truncated = bool(_read_field(result, "truncated", default=False))
        info = _read_field(result, "info", "runtime_info", "diagnostics", default={})
    try:
        reward_value = float(reward)
    except (TypeError, ValueError) as exc:
        raise ValidationError("runtime-only step did not return a numeric reward") from exc
    if not math.isfinite(reward_value):
        raise ValidationError("runtime-only reward must be finite")
    return (
        observation,
        reward_value,
        bool(terminated),
        bool(truncated),
        info if isinstance(info, Mapping) else {},
    )


def _runtime_language(adapter: Any, reset_info: Mapping[str, Any]) -> str:
    context = None
    getter = getattr(adapter, "get_task_context", None)
    if callable(getter):
        context = getter()
    language = _read_field(context, "language", "task_language", "description")
    if language is None:
        language = _read_field(
            reset_info, "task_language", "language", "task_prompt", "description"
        )
    language_text = str(language or "").strip()
    if not language_text:
        raise ValidationError("reset did not expose authoritative nonempty task language")
    return language_text


def _sim_time(adapter: Any, info: Mapping[str, Any] | None = None) -> float | None:
    sources = (info or {}, adapter)
    for source in sources:
        value = _read_field(
            source,
            "sim_time_s",
            "simulation_time_s",
            "sim_time",
            "simulation_time",
            "time_s",
            "current_time_s",
        )
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    getter = getattr(adapter, "get_current_time_s", None)
    if callable(getter):
        try:
            return float(getter())
        except (TypeError, ValueError):
            pass
    return None


def _control_dt(
    adapter: Any,
    before_s: float | None,
    after_s: float | None,
    info: Mapping[str, Any],
) -> float:
    if before_s is not None and after_s is not None:
        return after_s - before_s
    value = _read_field(
        info,
        "control_dt_s",
        "sim_dt_s",
        "elapsed_sim_time_s",
        "dt_s",
        "dt",
    )
    if value is None:
        value = _read_field(adapter, "control_dt_s", "dt_s")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("cannot prove that one real simulator tick advanced") from exc


def _hold_action(adapter: Any, observation: Any) -> Any:
    for target in (adapter, _read_field(adapter, "runtime")):
        if target is None:
            continue
        for name in ("hold_action", "make_hold_action"):
            factory = getattr(target, name, None)
            if callable(factory):
                try:
                    return factory(observation)
                except TypeError:
                    return factory()

    robot = _read_field(observation, "robot_state", "robot", "state")
    joint_positions = _read_field(robot, "joint_positions")
    gripper_positions = _read_field(robot, "gripper_positions", "gripper_position")
    if isinstance(joint_positions, Mapping) and joint_positions:
        from cap_harness.contracts import DEFAULT_EMBODIMENT, ArmCommand, RobotAction

        embodiment = _read_field(robot, "embodiment", default=DEFAULT_EMBODIMENT)
        commands = {}
        for arm, joints in joint_positions.items():
            gripper = (
                gripper_positions.get(arm)
                if isinstance(gripper_positions, Mapping)
                else gripper_positions
            )
            commands[str(arm)] = ArmCommand(
                mode="joint_position",
                target=joints,
                gripper_position=gripper,
                embodiment=embodiment,
            )
        return RobotAction(arms=commands)
    raise ValidationError("adapter cannot construct a typed hold action from robot state")


def _primary_joint_positions(observation: Any) -> np.ndarray:
    robot = _read_field(observation, "robot_state", "robot", "state")
    joints = _read_field(robot, "joint_positions", "q", "qpos", "arm_joint_positions")
    if isinstance(joints, Mapping):
        joints = joints.get("primary")
    try:
        vector = np.asarray(joints, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError("primary arm joints are not numeric") from exc
    if vector.shape != (7,) or not bool(np.all(np.isfinite(vector))):
        raise ValidationError("primary arm joints must be a finite seven-value vector")
    return vector


def _movement_action(observation: Any, *, delta_rad: float = 0.01) -> Any:
    from cap_harness.contracts import DEFAULT_EMBODIMENT, ArmCommand, RobotAction

    robot = _read_field(observation, "robot_state", "robot", "state")
    grippers = _read_field(robot, "gripper_positions", "gripper_position")
    gripper = grippers.get("primary") if isinstance(grippers, Mapping) else grippers
    embodiment = _read_field(robot, "embodiment", default=DEFAULT_EMBODIMENT)
    target = _primary_joint_positions(observation).copy()
    target[0] += delta_rad
    return RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=target,
                gripper_position=gripper,
                embodiment=embodiment,
            )
        }
    )


def _runtime_step(adapter: Any, action: Any) -> Any:
    runtime_step = getattr(adapter, "runtime_step", None)
    if callable(runtime_step):
        return runtime_step(action)
    native_step = getattr(adapter, "native_step", None)
    if callable(native_step):
        return native_step(action)
    runtime = _read_field(adapter, "runtime", "runtime_adapter")
    if runtime is None:
        getter = getattr(adapter, "get_runtime_adapter", None)
        runtime = getter() if callable(getter) else None
    step = getattr(runtime, "step", None)
    if callable(step):
        return step(action)
    raise ValidationError(
        "validation requires runtime_step() or a runtime-only adapter; public step is insufficient"
    )


def _default_adapter_factory(registry: Any) -> Callable[[ManifestEntry, int], Any]:
    def factory(entry: ManifestEntry, seed: int) -> Any:
        for name in (
            "create_validation_adapter",
            "create_runtime_adapter",
            "make_runtime_adapter",
            "make_adapter",
        ):
            method = getattr(registry, name, None)
            if not callable(method):
                continue
            try:
                return method(entry.suite, entry.task_id, seed=seed)
            except TypeError:
                return method(suite_name=entry.suite, task_id=entry.task_id, seed=seed)
        from cap_harness.libero.adapter import LiberoAdapter

        return LiberoAdapter(registry=registry)

    return factory


def _reset_adapter(adapter: Any, entry: ManifestEntry, seed: int) -> Any:
    reset = getattr(adapter, "reset", None)
    if not callable(reset):
        raise ValidationError("adapter does not expose reset()")
    try:
        import inspect

        parameters = inspect.signature(reset).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "task_ref" in parameters:
        return reset((entry.suite, entry.task_id), seed=seed)
    if "suite" in parameters or "suite_name" in parameters:
        return reset(suite_name=entry.suite, task_id=entry.task_id, seed=seed)
    return reset(seed=seed)


def _safe_error(exc: BaseException) -> dict[str, str]:
    # Importing doctor is safe: it has no simulator imports and centralizes redaction.
    from cap_harness.doctor import redact_secrets

    return {
        "type": type(exc).__name__,
        "message": str(redact_secrets(str(exc))),
    }


def _case_slug(key: MatrixKey) -> str:
    suite = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.suite)
    return f"{suite}__task-{key.task_id:02d}__seed-{key.seed}.json"


class ValidationRunner:
    """Run hold-and-movement LIBERO checks with fingerprinted append-only resume."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        registry: Any | None = None,
        adapter_factory: Callable[[ManifestEntry, int], Any] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        expected_dt_s: float = EXPECTED_CONTROL_DT_S,
        dt_tolerance_s: float = 1e-6,
        environment: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.registry = registry
        self.adapter_factory = adapter_factory
        self.config_path = config_path
        self.expected_dt_s = float(expected_dt_s)
        self.dt_tolerance_s = float(dt_tolerance_s)
        self.environment = environment
        self.store = AtomicJsonlStore(self.output_dir / "matrix.jsonl")
        self._fingerprints: Mapping[str, str | int] | None = None

    def _write_environment(self) -> Mapping[str, Any]:
        if self.environment is None:
            from cap_harness.doctor import environment_snapshot

            payload: Mapping[str, Any] = environment_snapshot()
        else:
            payload = self.environment
        normalized = _jsonable(payload)
        if not isinstance(normalized, Mapping):
            raise TypeError("validation environment must be a mapping")
        atomic_write_json(self.output_dir / "environment.json", normalized)
        return normalized

    def _run_case(
        self, entry: ManifestEntry, seed: int
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        key = MatrixKey(entry.suite, entry.task_id, seed)
        started_wall = time.monotonic()
        record: dict[str, Any] = {
            "schema_version": 1,
            "validation_fingerprint": (
                self._fingerprints["combined_sha256"] if self._fingerprints is not None else None
            ),
            "key": key.to_dict(),
            "family": entry.family,
            "variant": entry.variant,
            "task_name": entry.task_name,
            "started_at": utc_now(),
            "status": "failed",
            "checks": {},
        }
        failure: dict[str, Any] | None = None
        adapter: Any = None
        active_error: BaseException | None = None
        try:
            if self.adapter_factory is None:
                registry = self.registry if self.registry is not None else load_libero_registry()
                factory = _default_adapter_factory(registry)
            else:
                factory = self.adapter_factory
            adapter = factory(entry, seed)
            reset_result = _reset_adapter(adapter, entry, seed)
            observation, reset_info = _unpack_reset(reset_result)
            record["checks"]["reset"] = True

            language = _runtime_language(adapter, reset_info)
            if language != entry.task_language:
                raise ValidationError(
                    "runtime task language differs from the authoritative manifest language"
                )
            record["checks"]["authoritative_language"] = True
            record["task_language_sha256"] = hashlib.sha256(language.encode("utf-8")).hexdigest()

            schema = validate_observation_schema(observation)
            record["checks"]["observation_schema"] = True
            record["observation_schema"] = _jsonable(schema)

            before_s = _sim_time(adapter, reset_info)
            action = _hold_action(adapter, observation)
            step_result = _runtime_step(adapter, action)
            next_observation, reward, terminated, truncated, step_info = _unpack_step(step_result)
            record["checks"]["one_runtime_hold_tick"] = True
            validate_observation_schema(next_observation)
            record["checks"]["post_step_schema"] = True

            after_s = _sim_time(adapter, step_info)
            control_dt_s = _control_dt(adapter, before_s, after_s, step_info)
            if not math.isclose(
                control_dt_s,
                self.expected_dt_s,
                rel_tol=0.0,
                abs_tol=self.dt_tolerance_s,
            ):
                raise ValidationError(
                    f"one hold tick advanced {control_dt_s:.9g}s; expected {self.expected_dt_s:.9g}s"
                )
            record["checks"]["control_period"] = True
            record["control_dt_s"] = control_dt_s
            record["runtime_reward"] = reward
            record["checks"]["finite_runtime_reward"] = True
            if terminated or truncated:
                raise ValidationError(
                    "episode ended during the hold tick before movement validation"
                )

            joints_before = _primary_joint_positions(next_observation)
            movement_action = _movement_action(next_observation)
            movement_result = _runtime_step(adapter, movement_action)
            (
                movement_observation,
                movement_reward,
                movement_terminated,
                movement_truncated,
                movement_info,
            ) = _unpack_step(movement_result)
            record["checks"]["one_runtime_movement_tick"] = True
            validate_observation_schema(movement_observation)
            joints_after = _primary_joint_positions(movement_observation)
            movement_delta = float(np.max(np.abs(joints_after - joints_before)))
            if movement_delta <= 1e-6:
                raise ValidationError("movement action did not change robot joint state")
            movement_dt_s = _control_dt(
                adapter,
                after_s,
                _sim_time(adapter, movement_info),
                movement_info,
            )
            if not math.isclose(
                movement_dt_s,
                self.expected_dt_s,
                rel_tol=0.0,
                abs_tol=self.dt_tolerance_s,
            ):
                raise ValidationError(
                    f"one movement tick advanced {movement_dt_s:.9g}s; "
                    f"expected {self.expected_dt_s:.9g}s"
                )
            record["checks"]["movement_changed_robot_state"] = True
            record["checks"]["movement_control_period"] = True
            record["movement_max_joint_delta_rad"] = movement_delta
            record["movement_runtime_reward"] = movement_reward
            record["terminated"] = movement_terminated
            record["truncated"] = movement_truncated
            record["status"] = "passed"
        # Persist a failed matrix record and keep the sweep resumable.
        except Exception as exc:
            active_error = exc
            record["error"] = _safe_error(exc)
            failure = {
                "schema_version": 1,
                "validation_fingerprint": record["validation_fingerprint"],
                "key": key.to_dict(),
                "entry": entry.to_dict(),
                "checks": dict(record["checks"]),
                "error": record["error"],
                "traceback": str(
                    __import__("cap_harness.doctor", fromlist=["redact_secrets"]).redact_secrets(
                        "".join(traceback.format_exception(exc))
                    )
                ),
            }
        finally:
            try:
                close = getattr(adapter, "close", None)
                if adapter is None or not callable(close):
                    raise ValidationError("adapter does not expose close()")
                close()
                record["checks"]["clean_close"] = True
            # Persist a failed matrix record and keep the sweep resumable.
            except Exception as close_exc:
                record["checks"]["clean_close"] = False
                record["status"] = "failed"
                close_error = _safe_error(close_exc)
                record["close_error"] = close_error
                if failure is None:
                    record["error"] = close_error
                    failure = {
                        "schema_version": 1,
                        "validation_fingerprint": record["validation_fingerprint"],
                        "key": key.to_dict(),
                        "entry": entry.to_dict(),
                        "checks": dict(record["checks"]),
                        "error": close_error,
                        "traceback": str(
                            __import__(
                                "cap_harness.doctor", fromlist=["redact_secrets"]
                            ).redact_secrets("".join(traceback.format_exception(close_exc)))
                        ),
                    }
                else:
                    failure["close_error"] = close_error
            record["duration_s"] = round(time.monotonic() - started_wall, 6)
            record["finished_at"] = utc_now()
            if active_error is None and record["checks"].get("clean_close"):
                record["status"] = "passed"
        return record, failure

    def run(
        self,
        *,
        manifest: ValidationManifest | None = None,
        seeds: Sequence[int] = SMOKE_SEEDS,
        retry_failures: bool = False,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = manifest or build_required_manifest(self.registry, config_path=self.config_path)
        if len(manifest) != EXPECTED_PAIR_COUNT:
            raise ManifestError(
                f"validation requires exactly {EXPECTED_PAIR_COUNT} pairs, found {len(manifest)}"
            )
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if not normalized_seeds or any(seed <= 0 for seed in normalized_seeds):
            raise ValueError("validation seeds must be positive integers")
        if len(normalized_seeds) != len(set(normalized_seeds)):
            raise ValueError("validation seeds must be unique")

        environment = self._write_environment()
        atomic_write_json(self.output_dir / "manifest.json", manifest.to_dict())
        self._fingerprints = build_validation_fingerprints(
            manifest,
            environment=environment,
        )
        atomic_write_json(self.output_dir / "fingerprints.json", self._fingerprints)
        validation_fingerprint = str(self._fingerprints["combined_sha256"])
        existing = self.store.latest_by_key(validation_fingerprint=validation_fingerprint)
        skipped_resume = 0
        for entry in manifest:
            for seed in normalized_seeds:
                key = MatrixKey(entry.suite, entry.task_id, seed)
                prior = existing.get(key)
                if prior is not None and not (retry_failures and prior.get("status") != "passed"):
                    skipped_resume += 1
                    continue
                record, failure = self._run_case(entry, seed)
                if failure is not None:
                    artifact_path = self.output_dir / "failures" / _case_slug(key)
                    atomic_write_json(artifact_path, failure)
                    record["failure_artifact"] = str(artifact_path.relative_to(self.output_dir))
                self.store.append(record)
                existing[key] = record

        summary = self._summarize(manifest, normalized_seeds, skipped_resume)
        atomic_write_json(self.output_dir / "summary.json", summary)
        return summary

    def _summarize(
        self,
        manifest: ValidationManifest,
        seeds: Sequence[int],
        skipped_resume: int,
    ) -> dict[str, Any]:
        if self._fingerprints is None:
            raise RuntimeError("validation fingerprints were not initialized")
        validation_fingerprint = str(self._fingerprints["combined_sha256"])
        latest = self.store.latest_by_key(validation_fingerprint=validation_fingerprint)
        planned = {
            MatrixKey(entry.suite, entry.task_id, seed) for entry in manifest for seed in seeds
        }
        relevant = {key: value for key, value in latest.items() if key in planned}
        passed = sum(record.get("status") == "passed" for record in relevant.values())
        failed = sum(record.get("status") != "passed" for record in relevant.values())
        completed = len(relevant)
        by_family: dict[str, dict[str, int]] = {}
        entry_by_key = {entry.key: entry for entry in manifest}
        for key, record in relevant.items():
            family = entry_by_key[(key.suite, key.task_id)].family
            counts = by_family.setdefault(family, {"passed": 0, "failed": 0})
            counts["passed" if record.get("status") == "passed" else "failed"] += 1
        expected = len(manifest) * len(seeds)
        return {
            "schema_version": 1,
            "validation_fingerprint": validation_fingerprint,
            "fingerprints": dict(self._fingerprints),
            "generated_at": utc_now(),
            "manifest_pair_count": len(manifest),
            "required_pair_count": EXPECTED_PAIR_COUNT,
            "seeds": list(seeds),
            "mode": "nightly" if tuple(seeds) == NIGHTLY_SEEDS else "smoke",
            "expected": expected,
            "completed": completed,
            "passed": passed,
            "failed": failed,
            "remaining": expected - completed,
            "skipped_resume": skipped_resume,
            "by_family": dict(sorted(by_family.items())),
            "success": completed == expected and failed == 0,
        }


def validate_matrix(
    output_dir: str | os.PathLike[str],
    *,
    registry: Any | None = None,
    adapter_factory: Callable[[ManifestEntry, int], Any] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    nightly: bool = False,
    retry_failures: bool = False,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience entry point used by the CLI and integration tests."""
    runner = ValidationRunner(
        output_dir,
        registry=registry,
        adapter_factory=adapter_factory,
        config_path=config_path,
        environment=environment,
    )
    return runner.run(
        seeds=NIGHTLY_SEEDS if nightly else SMOKE_SEEDS,
        retry_failures=retry_failures,
    )


def write_manifest(
    path: str | os.PathLike[str],
    *,
    registry: Any | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> ValidationManifest:
    manifest = build_required_manifest(registry, config_path=config_path)
    atomic_write_json(path, manifest.to_dict())
    return manifest
