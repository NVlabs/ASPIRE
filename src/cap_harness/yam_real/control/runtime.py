"""One serialized control worker for a live YAM session."""

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any


class RuntimeControl:
    """Run CAP actuation on one worker while observations remain available."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self.episode_recorder = None
        self._lock = threading.RLock()
        self._owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yam-control-owner")
        self._closed = False
        self._operation: tuple[str, str] | None = None

    def begin_operation(self, *, kind: str, label: str) -> None:
        with self._lock:
            if self._closed or self._operation is not None:
                raise RuntimeError("control owner is closed or busy")
            self._operation = (kind, label)
            if self.episode_recorder is not None:
                self.episode_recorder.set_source("cap")

    def end_operation(self, *, kind: str, label: str) -> None:
        with self._lock:
            if self._operation == (kind, label):
                self._operation = None
                if self.episode_recorder is not None:
                    self.episode_recorder.set_source("idle")

    def submit_cap(self, operation: Any) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("control owner is closed")
            future = self._owner.submit(operation)
        return future.result()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._owner.shutdown(wait=True)
