"""One persistent CAP session shared by an agent, operator UI, and robot."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import logging
import threading
import time
from typing import Any
from uuid import uuid4

import numpy as np

from ..contracts import ExecutionResult
from ..registry import ToolRegistry
from ..runtime import ProgramExecutionResult, ProgramExecutor
from .program_catalog import CapProgramCatalog

PHASES = ("idle", "perceive", "reason", "act", "verify", "done", "error")
_READ_ONLY_TOOL_NAMES = frozenset({"get_task_context", "get_observation", "get_robot_state"})
logger = logging.getLogger(__name__)


class SessionBusyError(RuntimeError):
    """Raised when a second physical operation is submitted concurrently."""


@dataclass(frozen=True, slots=True)
class SessionEvent:
    seq: int
    timestamp_s: float
    role: str
    text: str
    phase: str
    source_id: str = ""
    turn_id: str = ""
    message_phase: str = ""


@dataclass(slots=True)
class _Job:
    id: str
    kind: str
    label: str
    phase: str
    started_s: float
    thread: threading.Thread | None = None
    finished_s: float | None = None
    result: Any = None
    error: str | None = None
    terminal_state: Any = None

    @property
    def active(self) -> bool:
        return self.finished_s is None


@dataclass(frozen=True, slots=True)
class _ReviewedProgramResult:
    """One checked-in program result plus evidence captured by the live session."""

    execution: ProgramExecutionResult
    verification: Mapping[str, Any]
    images: Mapping[str, np.ndarray]
    terminal_state: Any = None

    @property
    def ok(self) -> bool:
        return self.execution.ok and bool(getattr(self.execution.result, "ok", True))


def _summary(value: Any, *, depth: int = 0) -> Any:
    """Make an API-safe result summary without copying images or action arrays."""
    if depth > 6:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        # Small state vectors are the typed numeric payload of contracts such as
        # RobotState.  Images and action tensors remain bounded
        # metadata, but joint/EEF/gripper values must reach System-2 intact.
        if value.ndim <= 1 and value.size <= 32:
            return np.round(value.astype(float), 6).tolist()
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {
            str(key): _summary(item, depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [_summary(item, depth=depth + 1) for item in value]
    if is_dataclass(value):
        return {
            field.name: _summary(getattr(value, field.name), depth=depth + 1)
            for field in fields(value)
        }
    return str(value)


def _compact_robot_state(value: Any) -> dict[str, Any] | None:
    """Return the complete numeric state needed for the next metric action."""
    if value is None:
        return None
    return {
        "base_frame": getattr(value, "base_frame", None),
        "embodiment": getattr(value, "embodiment", None),
        "joint_positions": _summary(getattr(value, "joint_positions", {})),
        "joint_velocities": _summary(getattr(value, "joint_velocities", {})),
        "gripper_positions": _summary(getattr(value, "gripper_positions", {})),
        "end_effector_poses": _summary(getattr(value, "end_effector_poses", {})),
        "timestamp_s": getattr(value, "timestamp_s", None),
    }


def _compact_execution_result(value: Any) -> dict[str, Any]:
    """Expose a motion's terminal state without nesting camera observations.

    ``ExecutionResult.final_observation`` contains the exact state the motion
    used for its terminal tolerance check, but it also contains every RGB-D
    camera.  Putting that full observation in compact MCP feedback buried the
    useful joint/gripper values behind image metadata and encouraged a redundant
    ``get_robot_state`` call.  Keep the full observation in the recorder while
    presenting the already-verified terminal state directly to System-2. The MCP bridge attaches
    fresh post-action camera frames as separate image content blocks.
    """
    observation = getattr(value, "final_observation", None)
    state = getattr(observation, "robot_state", None)
    terminal_state = _compact_robot_state(state)
    ok = bool(getattr(value, "ok", False))
    status = getattr(value, "status", None)
    return {
        "ok": ok,
        "status": _summary(status),
        "postcondition_satisfied": ok,
        "steps_executed": int(getattr(value, "steps_executed", 0)),
        "terminated": bool(getattr(value, "terminated", False)),
        "truncated": bool(getattr(value, "truncated", False)),
        "final_errors": _summary(getattr(value, "final_errors", {})),
        "diagnostics": _summary(getattr(value, "diagnostics", {})),
        "error": _summary(getattr(value, "error", None)),
        "terminal_state": terminal_state,
        "terminal_observation_timestamp_s": getattr(observation, "timestamp_s", None),
        "state_refresh_required": terminal_state is None,
    }


def _compact_summary(value: Any, *, depth: int = 0) -> Any:
    """Bound decision feedback while retaining its semantic fields."""
    if depth > 6:
        return f"<{type(value).__name__}>"
    if isinstance(value, ExecutionResult):
        return _compact_execution_result(value)
    if isinstance(value, str):
        return value if len(value) <= 512 else value[:509] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _compact_summary(item, depth=depth + 1)
            for key, item in list(value.items())[:24]
            if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [_compact_summary(item, depth=depth + 1) for item in value[:12]]
    return _compact_summary(_summary(value), depth=depth + 1)


def _reviewed_program_summary(
    value: _ReviewedProgramResult, *, detail: str = "compact"
) -> dict[str, Any]:
    execution = value.execution
    terminal_state = _compact_robot_state(value.terminal_state)
    summary = {
        "ok": value.ok,
        "result": _compact_summary(execution.result),
        "error": _compact_summary(execution.error),
        "verification": _compact_summary(value.verification),
        "evidence_images": sorted(value.images),
        "terminal_state": terminal_state,
        "state_refresh_required": terminal_state is None,
    }
    if detail == "full":
        summary.update(stdout=execution.stdout, calls=_summary(execution.calls))
    return summary


class LiveAgentSession:
    """Serialize physical work while leaving monitoring and steering responsive.

    Programs use the existing restricted :class:`ProgramExecutor`; individual
    calls resolve through the same public :class:`ToolRegistry`. Every physical
    operation is a background job so a UI or agent can monitor
    execution while it is running.
    """

    def __init__(
        self,
        api: Any,
        registry: ToolRegistry,
        *,
        recorder: Any = None,
        event_limit: int = 100,
        monitor_hz: float = 2.0,
        monitor_active_hz: float = 5.0,
        program_catalog: CapProgramCatalog | None = None,
        operation_controller: Any = None,
        event_sink: Any = None,
        tool_result_sink: Any = None,
    ) -> None:
        self.api = api
        self.registry = registry
        self.executor = ProgramExecutor(registry, recorder=recorder)
        self.program_catalog = program_catalog or CapProgramCatalog()
        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)
        self._operation_lock = threading.Lock()
        self._events: deque[SessionEvent] = deque(maxlen=event_limit)
        self._steering: deque[str] = deque()
        self._operation_controller = operation_controller
        self._event_sink = event_sink
        self._tool_result_sink = tool_result_sink
        self._job: _Job | None = None
        self._phase = "idle"
        self._seq = 0
        self._closed = False
        self._cameras: dict[str, np.ndarray] = {}
        self._robot_state: dict[str, np.ndarray] = {}
        self._observation_timestamp_s: float | None = None
        self._observation_seq: int | None = None
        self._refresh_timestamp_s: float | None = None
        self._refresh_monotonic_s: float | None = None
        self._camera_metadata: dict[str, dict[str, Any]] = {}
        self._monitor_error: str | None = None
        if float(monitor_hz) <= 0 or float(monitor_active_hz) <= 0:
            raise ValueError("monitor rates must be positive")
        self._monitor_idle_period_s = 1.0 / float(monitor_hz)
        self._monitor_active_period_s = 1.0 / float(monitor_active_hz)
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="cap-agent-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        self.publish("system", "Live CAP session ready", phase="perceive")

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def publish(
        self,
        role: str,
        text: str,
        *,
        phase: str | None = None,
        timestamp_s: float | None = None,
        source_id: str = "",
        turn_id: str = "",
        message_phase: str = "",
    ) -> SessionEvent:
        message = str(text).strip()
        if not message:
            raise ValueError("event text must be non-empty")
        if phase is not None and phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        with self._lock:
            if phase is not None:
                self._phase = phase
            self._seq += 1
            event = SessionEvent(
                seq=self._seq,
                timestamp_s=time.time() if timestamp_s is None else float(timestamp_s),
                role=str(role).strip() or "system",
                text=message,
                phase=self._phase,
                source_id=str(source_id),
                turn_id=str(turn_id),
                message_phase=str(message_phase),
            )
            self._events.append(event)
        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception:
                logger.exception("System-2 event recorder failed")
        return event

    def _record_tool_result(self, job: _Job, result: Any = None) -> None:
        if self._tool_result_sink is None:
            return
        try:
            self._tool_result_sink(
                job_id=job.id,
                label=job.label,
                timestamp_s=time.time(),
                result=_summary(result),
                error=job.error,
            )
        except Exception:
            logger.exception("System-2 tool-result recorder failed")

    def steer(self, message: str) -> SessionEvent:
        text = str(message).strip()
        if not text:
            raise ValueError("steering message must be non-empty")
        event = self.publish("human", text)
        with self._lock:
            self._steering.append(text)
            self._state_changed.notify_all()
        self.publish("system", "Operator message queued")
        return event

    def drain_steering(self) -> list[str]:
        """Return queued operator messages exactly once, in arrival order."""
        with self._lock:
            messages = list(self._steering)
            self._steering.clear()
            return messages

    def terminal_robot_state(self) -> Any:
        """Read the canonical state from the shared environment observation cache."""
        try:
            return self.api.get_robot_state()
        except Exception:
            return None

    def terminal_robot_state_summary(self) -> dict[str, Any] | None:
        """Return that state in the bounded numeric form used by MCP feedback."""
        return _compact_robot_state(self.terminal_robot_state())

    def start_program(self, code: str, *, label: str = "CaP program") -> str:
        source = str(code)
        if not source.strip():
            raise ValueError("program must be non-empty")
        return self._start_job(
            kind="program",
            label=label,
            phase="act",
            operation=lambda: self.executor.execute_program(source),
        )

    def start_named_program(self, name: str) -> str:
        """Execute one immutable, human-reviewed program from the catalog."""
        program = self.program_catalog.get(name)
        return self._start_reviewed_program(
            program.definition.name,
            program.source,
            label=f"CaP program: {program.definition.name}",
            controls_runtime=program.definition.moves_robot,
        )

    def _start_reviewed_program(
        self,
        name: str,
        source: str,
        *,
        label: str,
        controls_runtime: bool = True,
    ) -> str:
        return self._start_job(
            kind="program",
            label=label,
            phase="act",
            operation=lambda: self._run_reviewed_program(name, source),
            controls_runtime=controls_runtime,
        )

    def _run_reviewed_program(self, name: str, source: str) -> _ReviewedProgramResult:
        """Bracket a reviewed program with station evidence."""
        before = self.visual_snapshot()
        execution = self.executor.execute_program(source)
        after = self.visual_snapshot()
        images: dict[str, np.ndarray] = {}
        for phase, snapshot in (("before", before), ("after", after)):
            for camera, frame in snapshot.get("cameras", {}).items():
                images[f"{phase}.{camera}"] = np.asarray(frame, dtype=np.uint8).copy()
        verification = {
            "program": name,
            "ok": execution.ok and bool(getattr(execution.result, "ok", True)),
        }
        return _ReviewedProgramResult(
            execution=execution,
            verification=verification,
            images=images,
            terminal_state=self.terminal_robot_state(),
        )

    def _start_job(
        self,
        *,
        kind: str,
        label: str,
        phase: str,
        operation: Any,
        on_result: Any = None,
        controls_runtime: bool = True,
    ) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("session is closed")
            if self._job is not None and self._job.active:
                raise SessionBusyError(f"job {self._job.id} ({self._job.label}) is still active")
            job = _Job(
                id=uuid4().hex[:12],
                kind=kind,
                label=str(label),
                phase=phase,
                started_s=time.time(),
            )
            self._job = job
        self.publish("tool", f"started {job.label} [{job.id}]", phase=phase)

        def run() -> None:
            try:
                with self._operation_lock:
                    controller = self._operation_controller
                    try:
                        if controller is not None and controls_runtime:
                            controller.begin_operation(kind=kind, label=label)
                        result = operation()
                    finally:
                        if controller is not None and controls_runtime:
                            controller.end_operation(kind=kind, label=label)
                    if on_result is not None:
                        on_result(result)
                    terminal_state = self.terminal_robot_state()
                    with self._lock:
                        job.result = result
                        job.terminal_state = terminal_state
                    self._capture_result(result)
                    self._record_tool_result(job, result)
                    ok = bool(getattr(result, "ok", True))
                    if isinstance(result, ProgramExecutionResult):
                        ok = result.ok
                    next_phase = "verify" if phase == "act" else "reason"
                    self.publish(
                        "tool",
                        f"finished {job.label} [{job.id}] ({'ok' if ok else 'failed'})",
                        phase=next_phase,
                    )
            except Exception as exc:
                terminal_state = self.terminal_robot_state()
                with self._lock:
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.terminal_state = terminal_state
                self._record_tool_result(job)
                self.publish("tool", f"{job.label} failed: {job.error}", phase="error")
            finally:
                with self._lock:
                    job.finished_s = time.time()
                    self._state_changed.notify_all()

        job.thread = threading.Thread(target=run, name=f"cap-agent-{kind}", daemon=True)
        job.thread.start()
        return job.id

    def job(self, job_id: str | None = None, *, detail: str = "compact") -> dict[str, Any] | None:
        if detail not in {"compact", "full"}:
            raise ValueError("detail must be 'compact' or 'full'")
        with self._lock:
            job = self._job
            if job is None or (job_id is not None and job.id != job_id):
                return None
            result = None
            if not job.active:
                result = (
                    _reviewed_program_summary(job.result, detail=detail)
                    if isinstance(job.result, _ReviewedProgramResult)
                    else _compact_summary(job.result)
                    if detail == "compact"
                    else _summary(job.result)
                )
            return {
                "id": job.id,
                "kind": job.kind,
                "label": job.label,
                "phase": job.phase,
                "active": job.active,
                "elapsed_s": round(
                    (job.finished_s or time.time()) - job.started_s,
                    3,
                ),
                "error": job.error,
                "result": result,
                "terminal_state": _compact_robot_state(job.terminal_state),
            }

    def job_evidence_images(
        self, job_id: str | None, *, cameras: tuple[str, ...] = ("top",)
    ) -> dict[str, np.ndarray]:
        """Return immutable before/after evidence for a finished reviewed program."""
        with self._lock:
            job = self._job
            if job is None or (job_id is not None and job.id != job_id):
                return {}
            result = job.result
            if not isinstance(result, _ReviewedProgramResult):
                return {}
            return {
                label: np.asarray(frame, dtype=np.uint8).copy()
                for label, frame in result.images.items()
                if label.rsplit(".", 1)[-1] in cameras
            }

    def wait(
        self, job_id: str, *, timeout_s: float | None = 0.0, detail: str = "compact"
    ) -> dict[str, Any] | None:
        """Wait for the current job, or until an optional timeout expires."""
        with self._lock:
            job = self._job
            thread = job.thread if job is not None and job.id == job_id else None
        if thread is None:
            return None
        thread.join(timeout=None if timeout_s is None else max(0.0, float(timeout_s)))
        return self.job(job_id, detail=detail)

    def finish(self, *, success: bool, summary: str) -> SessionEvent:
        """Mark the agent task complete; this does not shut down robot services."""
        with self._lock:
            if self._job is not None and self._job.active:
                raise SessionBusyError("cannot finish while a physical job is active")
        return self.publish("agent", summary, phase="done" if success else "error")

    def state(self, *, after: int = 0) -> dict[str, Any]:
        with self._lock:
            events = [
                {
                    "seq": item.seq,
                    "timestamp_s": item.timestamp_s,
                    "role": item.role,
                    "text": item.text,
                    "phase": item.phase,
                }
                for item in self._events
                if item.seq > int(after)
            ]
            steering_pending = len(self._steering)
            phase = self._phase
        return {
            "phase": phase,
            "job": self.job(),
            "events": events,
            "steering_pending": steering_pending,
            "tools": [
                spec.documented_signature for spec in self.registry.all_specs(public_only=True)
            ],
        }

    def decision_context(self, *, after_seq: int = 0) -> dict[str, Any]:
        """Project only live execution and sensor state for System 2."""
        with self._lock:
            context_seq = self._seq
            job = self._job
            execution = None
            if job is not None:
                execution = {
                    "job_id": job.id,
                    "actor": "cap",
                    "action": job.label,
                    "status": "running" if job.active else "finished",
                    "elapsed_s": round((job.finished_s or time.time()) - job.started_s, 2),
                    "error": job.error,
                }
            elapsed_ms = (
                0.0
                if self._refresh_monotonic_s is None
                else max(0.0, time.monotonic() - self._refresh_monotonic_s) * 1000.0
            )
            camera_health = {}
            for name, metadata in self._camera_metadata.items():
                item = {
                    key: metadata.get(key)
                    for key in ("sequence", "age_ms", "stale", "missing", "frame_hash")
                }
                if isinstance(item.get("age_ms"), (int, float)):
                    item["age_ms"] = round(float(item["age_ms"]) + elapsed_ms, 1)
                    stale_after_ms = metadata.get("stale_after_ms")
                    if isinstance(stale_after_ms, (int, float)):
                        item["stale"] = item["age_ms"] > float(stale_after_ms)
                camera_health[name] = item
            observation = {
                "seq": self._observation_seq,
                "stale": bool(self._monitor_error)
                or any(
                    bool(item.get("stale") or item.get("missing"))
                    for item in camera_health.values()
                ),
                "monitor_error": self._monitor_error,
                "cameras": camera_health,
            }
        return {
            "context_seq": context_seq,
            "execution": execution,
            "observation": observation,
        }

    def visual_snapshot(self) -> dict[str, Any]:
        """Refresh and return the station monitoring cache."""
        station_monitor = getattr(self.api, "monitor_snapshot", None)
        if callable(station_monitor):
            try:
                station_live = station_monitor()
            except Exception as exc:
                station_live = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if isinstance(station_live, Mapping) and station_live.get("ok"):
                self._capture_station_snapshot(station_live)
            else:
                error = (
                    station_live.get("error", "station monitor unavailable")
                    if isinstance(station_live, Mapping)
                    else "station monitor unavailable"
                )
                self._capture_station_failure(str(error))

        return self.cached_visual_snapshot()

    def cached_visual_snapshot(self) -> dict[str, Any]:
        """Return the monitor cache without polling the station."""
        with self._lock:
            timestamp_s = self._observation_timestamp_s
            elapsed_ms = (
                0.0
                if self._refresh_monotonic_s is None
                else max(0.0, time.monotonic() - self._refresh_monotonic_s) * 1000.0
            )
            camera_metadata = {}
            for name, raw in self._camera_metadata.items():
                metadata = dict(raw)
                age_ms = metadata.get("age_ms")
                if isinstance(age_ms, (int, float)):
                    metadata["age_ms"] = float(age_ms) + elapsed_ms
                    stale_after_ms = metadata.get("stale_after_ms")
                    if isinstance(stale_after_ms, (int, float)):
                        metadata["stale"] = metadata["age_ms"] > float(stale_after_ms)
                camera_metadata[name] = metadata
            stale = bool(self._monitor_error) or any(
                bool(item.get("stale") or item.get("missing")) for item in camera_metadata.values()
            )
            active_job = self._job if self._job is not None and self._job.active else None
            job_id = None if active_job is None else active_job.id
            return {
                "timestamp_s": timestamp_s,
                "age_s": None if timestamp_s is None else max(0.0, time.time() - timestamp_s),
                "refresh_timestamp_s": self._refresh_timestamp_s,
                "observation_seq": self._observation_seq,
                "job_id": job_id,
                "job_label": None if active_job is None else active_job.label,
                "stale": stale,
                "monitor_error": self._monitor_error,
                "camera_metadata": camera_metadata,
                "cameras": {key: value.copy() for key, value in self._cameras.items()},
                "state": {key: value.copy() for key, value in self._robot_state.items()},
            }

    def _capture_result(self, result: Any) -> None:
        if isinstance(result, _ReviewedProgramResult):
            result = result.execution
        if isinstance(result, ProgramExecutionResult):
            result = result.result
        observation = getattr(result, "final_observation", None) or result
        cameras = getattr(observation, "cameras", None)
        robot_state = getattr(observation, "robot_state", None)
        with self._lock:
            if isinstance(cameras, Mapping) and cameras:
                self._cameras = {
                    str(name): np.asarray(camera.rgb, dtype=np.uint8).copy()
                    for name, camera in cameras.items()
                    if hasattr(camera, "rgb")
                }
            joints = getattr(robot_state, "joint_positions", None)
            grippers = getattr(robot_state, "gripper_positions", None)
            if isinstance(joints, Mapping):
                state = {
                    f"{side}_joint_pos": np.asarray(value, dtype=float).copy()
                    for side, value in joints.items()
                }
                if isinstance(grippers, Mapping):
                    state.update(
                        {
                            f"{side}_gripper_pos": np.asarray([value], dtype=float)
                            for side, value in grippers.items()
                        }
                    )
                self._robot_state = state

    def _capture_station_snapshot(self, value: Mapping[str, Any]) -> None:
        cameras = value.get("cameras")
        camera_metadata = value.get("camera_metadata")
        state = value.get("state")
        timestamp_s = value.get("timestamp_s")
        refresh_timestamp_s = value.get("refresh_timestamp_s")
        observation_seq = value.get("observation_seq")
        copied_cameras = (
            {
                str(name): np.asarray(frame, dtype=np.uint8).copy()
                for name, frame in cameras.items()
                if isinstance(frame, np.ndarray)
            }
            if isinstance(cameras, Mapping)
            else {}
        )
        copied_metadata = (
            {
                str(name): dict(item)
                for name, item in camera_metadata.items()
                if isinstance(item, Mapping)
            }
            if isinstance(camera_metadata, Mapping)
            else {}
        )
        with self._lock:
            self._cameras.update(copied_cameras)
            self._camera_metadata = copied_metadata
            if isinstance(state, Mapping) and state:
                next_state = {
                    str(name): np.asarray(item, dtype=float).copy()
                    for name, item in state.items()
                    if isinstance(item, np.ndarray)
                }
                self._robot_state = next_state
            if isinstance(timestamp_s, (int, float)):
                self._observation_timestamp_s = float(timestamp_s)
            if isinstance(refresh_timestamp_s, (int, float)):
                self._refresh_timestamp_s = float(refresh_timestamp_s)
            else:
                self._refresh_timestamp_s = time.time()
            if isinstance(observation_seq, int):
                self._observation_seq = observation_seq
            self._refresh_monotonic_s = time.monotonic()
            self._monitor_error = None
            self._state_changed.notify_all()

    def _capture_station_failure(self, error: str) -> None:
        with self._lock:
            self._monitor_error = str(error)
            self._refresh_timestamp_s = time.time()
            self._state_changed.notify_all()

    def start_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        spec = self.registry.get_spec(str(name))
        if not spec.public:
            raise ValueError(f"tool {name!r} is not public")
        kwargs = dict(arguments or {})
        readonly = name in _READ_ONLY_TOOL_NAMES or spec.capability == "metadata"
        return self._start_job(
            kind="tool",
            label=name,
            phase="perceive" if readonly else "act",
            operation=lambda: spec.function(**kwargs),
            controls_runtime=not readonly,
        )

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self._next_monitor_period_s()):
            try:
                self.visual_snapshot()
            except Exception as exc:
                self._capture_station_failure(str(exc))

    def _next_monitor_period_s(self) -> float:
        with self._lock:
            active = self._job is not None and self._job.active
        return self._monitor_active_period_s if active else self._monitor_idle_period_s

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._monitor_stop.set()
        with self._state_changed:
            self._state_changed.notify_all()
        if self._monitor_thread is not threading.current_thread():
            self._monitor_thread.join(timeout=1.0)
        controller_close = getattr(self._operation_controller, "close", None)
        if callable(controller_close):
            controller_close()
        with self._lock:
            thread = None if self._job is None else self._job.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()


__all__ = [
    "PHASES",
    "LiveAgentSession",
    "SessionBusyError",
    "SessionEvent",
]
