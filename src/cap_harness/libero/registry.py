"""Discovery and validation for installed LIBERO benchmark suites."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any

REQUIRED_SUITE_NAMES: tuple[str, ...] = (
    "libero_goal_swap",
    "libero_goal_task",
    "libero_object_swap",
    "libero_object_task",
    "libero_spatial_swap",
    "libero_spatial_task",
    "libero_10_swap",
    "libero_10_task",
)
"""LIBERO-Pro suites that gate the first cap-harness release."""

REQUIRED_LIBERO_SUITES = REQUIRED_SUITE_NAMES
EXPECTED_REQUIRED_TASK_COUNT = 80
EXPECTED_REQUIRED_TASKS = EXPECTED_REQUIRED_TASK_COUNT


class LiberoRegistryError(ValueError):
    """Raised when an installed LIBERO benchmark cannot form a valid manifest."""


@dataclass(frozen=True, slots=True)
class LiberoTaskMetadata:
    """Resolved, public metadata for one installed LIBERO task."""

    suite_name: str
    task_id: int
    task_name: str
    family: str
    bddl_path: Path
    language: str
    init_state_count: int

    @property
    def task_ref(self) -> str:
        return f"{self.suite_name}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "bddl_path": str(self.bddl_path),
            "family": self.family,
            "init_state_count": self.init_state_count,
            "language": self.language,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


LiberoTaskSpec = LiberoTaskMetadata


def _without_bddl_comments(text: str) -> str:
    """Remove semicolon comments without touching quoted language text."""
    output: list[str] = []
    quoted = False
    escaped = False
    in_comment = False
    for char in text:
        if in_comment:
            if char == "\n":
                output.append(char)
                in_comment = False
            continue
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            in_comment = True
        else:
            output.append(char)
            if char == '"':
                quoted = True
    return "".join(output)


def extract_language_from_bddl_text(text: str) -> str:
    """Return the authoritative ``(:language ...)`` form from BDDL text.

    The scanner balances nested parentheses and respects quoted strings. This is
    deliberately more precise than deriving an instruction from a task filename,
    which is wrong for LIBERO-Pro task-perturbation suites.
    """
    content = _without_bddl_comments(text)
    match = re.search(r"\(\s*:language\b", content, flags=re.IGNORECASE)
    if match is None:
        raise LiberoRegistryError("BDDL file has no (:language ...) form")

    start = match.end()
    depth = 1
    quoted = False
    escaped = False
    end: int | None = None
    for index in range(start, len(content)):
        char = content[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end is None:
        raise LiberoRegistryError("unterminated (:language ...) form in BDDL file")

    language = " ".join(content[start:end].split()).strip()
    if len(language) >= 2 and language[0] == language[-1] == '"':
        try:
            decoded = ast.literal_eval(language)
        except (SyntaxError, ValueError):
            decoded = language[1:-1]
        if isinstance(decoded, str):
            language = decoded.strip()
    if not language:
        raise LiberoRegistryError("BDDL (:language ...) form is empty")
    return language


def extract_language_from_bddl(path: str | Path) -> str:
    bddl_path = Path(path).expanduser().resolve()
    try:
        text = bddl_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LiberoRegistryError(f"cannot read BDDL file {bddl_path}: {exc}") from exc
    try:
        return extract_language_from_bddl_text(text)
    except LiberoRegistryError as exc:
        raise LiberoRegistryError(f"invalid BDDL file {bddl_path}: {exc}") from exc


def _suite_family(suite_name: str) -> str:
    if suite_name.startswith("libero_goal_"):
        return "goal"
    if suite_name.startswith("libero_object_"):
        return "object"
    if suite_name.startswith("libero_spatial_"):
        return "spatial"
    if suite_name.startswith("libero_10_"):
        return "long"
    tokens = suite_name.split("_")
    return tokens[1] if len(tokens) > 1 else suite_name


class LiberoSuiteRegistry:
    """Enumerate LIBERO suites without importing LIBERO at module import time."""

    def __init__(
        self,
        *,
        benchmark_module: Any | None = None,
        benchmark_dict: Mapping[str, Callable[[], Any] | Any] | None = None,
        path_resolver: Callable[[str], str | Path] | None = None,
        get_libero_path: Callable[[str], str | Path] | None = None,
    ) -> None:
        if path_resolver is not None and get_libero_path is not None:
            raise TypeError("pass only one of path_resolver and get_libero_path")
        self._benchmark_module = benchmark_module
        self._provided_benchmark_dict = benchmark_dict
        self._path_resolver = path_resolver or get_libero_path
        self._benchmark_dict_cache: Mapping[str, Callable[[], Any] | Any] | None = None
        self._suite_cache: dict[str, Any] = {}
        self._metadata_cache: dict[tuple[str, int], LiberoTaskMetadata] = {}
        self._init_states_cache: dict[tuple[str, int], Sequence[Any]] = {}

    @staticmethod
    def _import_first(module_names: Sequence[str]) -> Any:
        failures: list[BaseException] = []
        for module_name in module_names:
            try:
                return importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError) as exc:
                failures.append(exc)
        raise ModuleNotFoundError(
            "LIBERO is not installed; install the LIBERO-Pro runtime to discover suites"
        ) from failures[-1]

    def _benchmark_dict(self) -> Mapping[str, Callable[[], Any] | Any]:
        if self._benchmark_dict_cache is not None:
            return self._benchmark_dict_cache
        if self._provided_benchmark_dict is not None:
            mapping = self._provided_benchmark_dict
        else:
            module = self._benchmark_module or self._import_first(
                ("libero.benchmark", "libero.libero.benchmark")
            )
            get_benchmark_dict = getattr(module, "get_benchmark_dict", None)
            if get_benchmark_dict is None:
                raise LiberoRegistryError("LIBERO benchmark module has no get_benchmark_dict()")
            try:
                mapping = get_benchmark_dict(help=False)
            except TypeError:
                mapping = get_benchmark_dict()
        if not isinstance(mapping, Mapping):
            raise LiberoRegistryError("LIBERO get_benchmark_dict() did not return a mapping")
        self._benchmark_dict_cache = mapping
        return mapping

    def _get_path_resolver(self) -> Callable[[str], str | Path]:
        if self._path_resolver is not None:
            return self._path_resolver
        for module_name in ("libero.libero", "libero.utils", "libero.libero.utils"):
            try:
                module = importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError):
                continue
            resolver = getattr(module, "get_libero_path", None)
            if callable(resolver):
                self._path_resolver = resolver
                return resolver
        raise ModuleNotFoundError(
            "LIBERO is installed without a discoverable get_libero_path() helper"
        )

    @property
    def available_suites(self) -> tuple[str, ...]:
        return tuple(sorted(str(name) for name in self._benchmark_dict()))

    def _get_suite(self, suite_name: str) -> Any:
        if suite_name in self._suite_cache:
            return self._suite_cache[suite_name]
        benchmark_dict = self._benchmark_dict()
        if suite_name not in benchmark_dict:
            raise LiberoRegistryError(
                f"LIBERO suite {suite_name!r} is not installed; available: "
                f"{', '.join(self.available_suites)}"
            )
        entry = benchmark_dict[suite_name]
        suite = entry() if callable(entry) else entry
        self._suite_cache[suite_name] = suite
        return suite

    @staticmethod
    def _task_count(suite: Any) -> int:
        count = getattr(suite, "n_tasks", None)
        if count is None and callable(getattr(suite, "get_num_tasks", None)):
            count = suite.get_num_tasks()
        if count is None and hasattr(suite, "tasks"):
            count = len(suite.tasks)
        try:
            count = int(count)
        except (TypeError, ValueError) as exc:
            raise LiberoRegistryError("LIBERO suite does not report a valid task count") from exc
        if count < 0:
            raise LiberoRegistryError(f"LIBERO suite reports negative task count {count}")
        return count

    def _resolve_bddl_path(self, task: Any) -> Path:
        direct_path = getattr(task, "bddl_path", None)
        if direct_path is not None:
            path = Path(direct_path).expanduser()
        else:
            filename = getattr(task, "bddl_file", None)
            problem_folder = getattr(task, "problem_folder", None)
            if not filename or not problem_folder:
                raise LiberoRegistryError("LIBERO task is missing problem_folder or bddl_file")
            filename_path = Path(str(filename)).expanduser()
            if filename_path.is_absolute():
                path = filename_path
            else:
                root = Path(self._get_path_resolver()("bddl_files")).expanduser()
                path = root / str(problem_folder) / filename_path
        path = path.resolve()
        if not path.is_file():
            raise LiberoRegistryError(f"BDDL file does not exist: {path}")
        return path

    def get_init_states(self, task_ref: Any, task_id: int | None = None) -> Sequence[Any]:
        metadata = self.resolve(task_ref, task_id)
        key = (metadata.suite_name, metadata.task_id)
        if key in self._init_states_cache:
            return self._init_states_cache[key]
        suite = self._get_suite(metadata.suite_name)
        loader = getattr(suite, "get_task_init_states", None)
        if not callable(loader):
            raise LiberoRegistryError(
                f"suite {metadata.suite_name!r} has no get_task_init_states()"
            )
        previous_torch_load_mode = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            # The pinned LIBERO-Pro init-state files are trusted project assets
            # created before Torch changed the default weights_only behavior.
            states = loader(metadata.task_id)
        except Exception as exc:
            raise LiberoRegistryError(
                f"cannot load init states for {metadata.task_ref}: {exc}"
            ) from exc
        finally:
            if previous_torch_load_mode is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_torch_load_mode
        if states is None:
            raise LiberoRegistryError(f"init states are missing for {metadata.task_ref}")
        try:
            count = len(states)
        except TypeError as exc:
            raise LiberoRegistryError(
                f"init states for {metadata.task_ref} are not a sized sequence"
            ) from exc
        if count <= 0:
            raise LiberoRegistryError(f"init states are empty for {metadata.task_ref}")
        self._init_states_cache[key] = states
        return states

    def _metadata(self, suite_name: str, task_id: int) -> LiberoTaskMetadata:
        key = (suite_name, task_id)
        if key in self._metadata_cache:
            return self._metadata_cache[key]
        suite = self._get_suite(suite_name)
        count = self._task_count(suite)
        if task_id < 0 or task_id >= count:
            raise LiberoRegistryError(
                f"task id {task_id} is outside suite {suite_name!r} range [0, {count})"
            )
        get_task = getattr(suite, "get_task", None)
        if not callable(get_task):
            raise LiberoRegistryError(f"suite {suite_name!r} has no get_task()")
        task = get_task(task_id)
        bddl_path = self._resolve_bddl_path(task)
        language = extract_language_from_bddl(bddl_path)
        task_name = str(getattr(task, "name", bddl_path.stem)).strip()
        if not task_name:
            raise LiberoRegistryError(f"task {suite_name}:{task_id} has an empty name")

        # Cache a provisional record so get_init_states() can resolve this task
        # without recursing through _metadata().
        provisional = LiberoTaskMetadata(
            suite_name=suite_name,
            task_id=task_id,
            task_name=task_name,
            family=_suite_family(suite_name),
            bddl_path=bddl_path,
            language=language,
            init_state_count=1,
        )
        self._metadata_cache[key] = provisional
        try:
            init_state_count = len(self.get_init_states(provisional))
        except Exception:
            self._metadata_cache.pop(key, None)
            raise
        metadata = LiberoTaskMetadata(
            suite_name=suite_name,
            task_id=task_id,
            task_name=task_name,
            family=_suite_family(suite_name),
            bddl_path=bddl_path,
            language=language,
            init_state_count=init_state_count,
        )
        self._metadata_cache[key] = metadata
        return metadata

    def enumerate_tasks(
        self, suite_names: Sequence[str] | None = None
    ) -> tuple[LiberoTaskMetadata, ...]:
        """Enumerate installed suites, resolving BDDL and init-state data."""
        names = tuple(sorted(self.available_suites)) if suite_names is None else tuple(suite_names)
        tasks: list[LiberoTaskMetadata] = []
        for suite_name in names:
            suite = self._get_suite(suite_name)
            tasks.extend(
                self._metadata(suite_name, task_id) for task_id in range(self._task_count(suite))
            )
        return tuple(tasks)

    discover = enumerate_tasks

    def resolve(self, task_ref: Any, task_id: int | None = None) -> LiberoTaskMetadata:
        """Resolve a task metadata object, ``(suite, id)``, mapping, or ref string."""
        if isinstance(task_ref, LiberoTaskMetadata):
            return task_ref
        suite_name: Any
        resolved_task_id: Any
        if task_id is not None:
            suite_name, resolved_task_id = task_ref, task_id
        elif isinstance(task_ref, Mapping):
            suite_name = task_ref.get("suite_name", task_ref.get("suite"))
            resolved_task_id = task_ref.get("task_id", task_ref.get("id"))
        elif isinstance(task_ref, tuple | list) and len(task_ref) == 2:
            suite_name, resolved_task_id = task_ref
        elif isinstance(task_ref, str):
            separator = ":" if ":" in task_ref else "/"
            if separator not in task_ref:
                raise LiberoRegistryError("string task references must use '<suite>:<task_id>'")
            suite_name, raw_task_id = task_ref.rsplit(separator, 1)
            resolved_task_id = raw_task_id
        else:
            suite_name = getattr(task_ref, "suite_name", None)
            resolved_task_id = getattr(task_ref, "task_id", None)
        if not isinstance(suite_name, str) or not suite_name:
            raise LiberoRegistryError(f"invalid LIBERO task reference: {task_ref!r}")
        try:
            resolved_task_id = int(resolved_task_id)
        except (TypeError, ValueError) as exc:
            raise LiberoRegistryError(f"invalid LIBERO task id: {resolved_task_id!r}") from exc
        return self._metadata(suite_name, resolved_task_id)

    def required_tasks(self) -> tuple[LiberoTaskMetadata, ...]:
        missing = [name for name in REQUIRED_SUITE_NAMES if name not in self._benchmark_dict()]
        if missing:
            raise LiberoRegistryError("required LIBERO suites are missing: " + ", ".join(missing))
        tasks = self.enumerate_tasks(REQUIRED_SUITE_NAMES)
        suite_counts = {
            suite_name: sum(task.suite_name == suite_name for task in tasks)
            for suite_name in REQUIRED_SUITE_NAMES
        }
        invalid_counts = {name: count for name, count in suite_counts.items() if count != 10}
        if invalid_counts:
            details = ", ".join(f"{name}={count}" for name, count in invalid_counts.items())
            raise LiberoRegistryError(f"required suites must each contain 10 tasks: {details}")
        if len(tasks) != EXPECTED_REQUIRED_TASK_COUNT:
            raise LiberoRegistryError(
                f"required manifest has {len(tasks)} task pairs; "
                f"expected exactly {EXPECTED_REQUIRED_TASK_COUNT}"
            )
        pairs = {(task.suite_name, task.task_id) for task in tasks}
        if len(pairs) != EXPECTED_REQUIRED_TASK_COUNT:
            raise LiberoRegistryError("required manifest contains duplicate task pairs")
        for task in tasks:
            if not task.bddl_path.is_file() or not task.language or task.init_state_count <= 0:
                raise LiberoRegistryError(f"invalid required task metadata: {task.task_ref}")
        return tasks

    def build_manifest(self, *, required_only: bool = True) -> dict[str, Any]:
        tasks = self.required_tasks() if required_only else self.enumerate_tasks()
        suites = (
            REQUIRED_SUITE_NAMES
            if required_only
            else tuple(sorted({task.suite_name for task in tasks}))
        )
        return {
            "expected_task_count": EXPECTED_REQUIRED_TASK_COUNT if required_only else None,
            "required": required_only,
            "schema_version": 1,
            "suites": list(suites),
            "task_count": len(tasks),
            "tasks": [task.to_manifest_record() for task in tasks],
        }

    def required_manifest(self) -> dict[str, Any]:
        """Build the release-gating manifest, failing unless all 80 pairs are valid."""
        return self.build_manifest(required_only=True)

    def write_manifest(self, path: str | Path, *, required_only: bool = True) -> dict[str, Any]:
        """Write a byte-stable JSON manifest and return its payload."""
        payload = self.build_manifest(required_only=required_only)
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return payload
