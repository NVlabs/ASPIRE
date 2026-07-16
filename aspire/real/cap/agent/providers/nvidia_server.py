"""Node-level NVIDIA provider daemon and dashboard.

Run one daemon per node or egress IP, then point CAP jobs at it with
``CAP_NVIDIA_PROVIDER_URL=http://127.0.0.1:8765``. The daemon owns key
selection, scheduler state, telemetry, and the provider-level dashboard.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import tyro

from cap.agent.providers import nvidia


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
RPM_WINDOW_S = 60.0
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_HEALTH_MODELS = (
    "aws/anthropic/bedrock-claude-opus-4-6",
    "azure/anthropic/claude-opus-4-7",
    "gcp/google/gemini-3.1-pro-preview",
)
_global_request_lock = threading.Lock()
_global_last_request_at = 0.0


class ProviderRequestError(RuntimeError):
    """Final daemon-side request failure with an HTTP status for the client."""

    def __init__(self, message: str, *, http_status: int = 500) -> None:
        super().__init__(message)
        self.http_status = http_status


@dataclass
class ServeCommand:
    """Run the node-level NVIDIA provider daemon."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    base_url: str = nvidia.DEFAULT_BASE_URL
    telemetry_file: str | None = None
    scheduler_db: str | None = None
    fresh_start: bool = False
    request_delay_s: float | None = None
    global_request_delay_s: float | None = None
    max_concurrent_per_key: int | None = None
    health_models: list[str] | None = None
    dashboard: bool = False


@dataclass
class DashboardCommand:
    """Render local daemon telemetry."""

    pass


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _event_lines(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return events


def _peak_window_rate(
    timestamps: list[float],
    *,
    values: list[int] | None = None,
    window_s: float = RPM_WINDOW_S,
) -> float:
    """Return the highest rolling-window per-minute rate."""
    if not timestamps:
        return 0.0
    if values is None:
        values = [1] * len(timestamps)
    paired = sorted(zip(timestamps, values), key=lambda item: item[0])
    window: deque[tuple[float, int]] = deque()
    window_total = 0
    peak_total = 0
    for ts, value in paired:
        window.append((ts, value))
        window_total += value
        while window and ts - window[0][0] > window_s:
            _old_ts, old_value = window.popleft()
            window_total -= old_value
        peak_total = max(peak_total, window_total)
    return peak_total * (60.0 / window_s)


def build_stats() -> dict[str, Any]:
    """Build provider-wide stats from scheduler rows and telemetry events."""
    telemetry_raw = os.environ.get(nvidia.TELEMETRY_FILE_ENV)
    telemetry_path = Path(telemetry_raw) if telemetry_raw else None
    events = _event_lines(telemetry_path)
    now = time.time()
    cutoff = now - RPM_WINDOW_S

    key_stats: dict[str, dict[str, Any]] = {}
    active_requests: dict[str, str] = {}
    recent_request_ts: deque[float] = deque()
    recent_token_events: deque[tuple[float, int]] = deque()
    all_request_ts: list[float] = []
    all_token_events: list[tuple[float, int]] = []
    recent_errors: deque[dict[str, Any]] = deque(maxlen=8)
    source_stats: dict[str, dict[str, Any]] = {}

    def stats_for_key(key: str) -> dict[str, Any]:
        return key_stats.setdefault(
            key,
            {
                "key": key,
                "state": "unknown",
                "active": 0,
                "rpm": 0.0,
                "tpm": 0.0,
                "ok": 0,
                "err": 0,
                "last_ms": None,
                "source": "",
                "last_error": "",
                "_recent_request_ts": deque(),
                "_recent_token_events": deque(),
            },
        )

    def stats_for_source(source: str) -> dict[str, Any]:
        return source_stats.setdefault(
            source,
            {
                "source": source,
                "active": 0,
                "rpm": 0.0,
                "tpm": 0.0,
                "ok": 0,
                "err": 0,
                "_recent_request_ts": deque(),
                "_recent_token_events": deque(),
            },
        )

    def is_benign_health_probe_error(event: dict[str, Any]) -> bool:
        if event.get("source") != "health_check" or event.get("http_status") != 400:
            return False
        text = str(event.get("error") or "").lower()
        return "max_tokens" in text and "model output limit" in text

    for key in nvidia.list_nvidia_keys():
        stats_for_key(nvidia.nvidia_key_label(key))

    for event in events:
        key = str(event.get("key") or "key?:unknown")
        request_id = str(event.get("request_id") or "")
        source = str(event.get("source") or "nvidia")
        key_row = stats_for_key(key)
        source_row = stats_for_source(source)
        if source:
            key_row["source"] = source

        event_name = event.get("event")
        if event_name == "start":
            if request_id:
                active_requests[request_id] = key
            key_row["active"] = int(key_row.get("active", 0) or 0) + 1
            source_row["active"] = int(source_row.get("active", 0) or 0) + 1
            continue
        if event_name != "end":
            continue

        start_key = active_requests.pop(request_id, None)
        if start_key is not None:
            start_row = stats_for_key(start_key)
            start_row["active"] = max(int(start_row.get("active", 0) or 0) - 1, 0)
        else:
            key_row["active"] = max(int(key_row.get("active", 0) or 0) - 1, 0)
        source_row["active"] = max(int(source_row.get("active", 0) or 0) - 1, 0)

        ts_raw = event.get("ts")
        ts = float(ts_raw) if isinstance(ts_raw, (int, float)) else now
        if is_benign_health_probe_error(event):
            continue
        all_request_ts.append(ts)
        total_tokens = event.get("total_tokens")
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            all_token_events.append((ts, total_tokens))
        if ts >= cutoff:
            recent_request_ts.append(ts)
            key_row["_recent_request_ts"].append(ts)
            source_row["_recent_request_ts"].append(ts)
            if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
                recent_token_events.append((ts, total_tokens))
                key_row["_recent_token_events"].append((ts, total_tokens))
                source_row["_recent_token_events"].append((ts, total_tokens))

        latency_ms = event.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            key_row["last_ms"] = float(latency_ms)
        if event.get("status") == "ok":
            key_row["ok"] = int(key_row.get("ok", 0) or 0) + 1
            source_row["ok"] = int(source_row.get("ok", 0) or 0) + 1
            continue

        key_row["err"] = int(key_row.get("err", 0) or 0) + 1
        source_row["err"] = int(source_row.get("err", 0) or 0) + 1
        error_text = str(event.get("error") or "").strip()
        http_status = event.get("http_status")
        prefix = f"HTTP {http_status}: " if http_status else ""
        key_row["last_error"] = (prefix + error_text).strip()
        recent_errors.appendleft(
            {
                "ts": ts,
                "key": key,
                "source": source,
                "http_status": http_status,
                "error": error_text,
            }
        )

    scheduler_rows = nvidia.read_nvidia_scheduler_stats()
    for row in scheduler_rows:
        key = str(row.get("key") or "")
        if not key:
            continue
        key_row = stats_for_key(key)
        key_row["state"] = str(row.get("status") or "unknown")
        key_row["active"] = int(row.get("active", 0) or 0)
        key_row["ok"] = max(int(key_row.get("ok", 0) or 0), int(row.get("ok_count", 0) or 0))
        key_row["err"] = max(
            int(key_row.get("err", 0) or 0), int(row.get("error_count", 0) or 0)
        )
        last_ms = row.get("last_latency_ms")
        if isinstance(last_ms, (int, float)):
            key_row["last_ms"] = float(last_ms)
        last_error = str(row.get("last_error") or "").strip()
        if last_error:
            http_status = row.get("last_http_status")
            key_row["last_error"] = (f"HTTP {http_status}: " if http_status else "") + last_error

    for row in key_stats.values():
        req_ts = row.pop("_recent_request_ts")
        token_events = row.pop("_recent_token_events")
        row["rpm"] = len(req_ts) * (60.0 / RPM_WINDOW_S)
        row["tpm"] = sum(tokens for _ts, tokens in token_events) * (60.0 / RPM_WINDOW_S)

    for row in source_stats.values():
        req_ts = row.pop("_recent_request_ts")
        token_events = row.pop("_recent_token_events")
        row["rpm"] = len(req_ts) * (60.0 / RPM_WINDOW_S)
        row["tpm"] = sum(tokens for _ts, tokens in token_events) * (60.0 / RPM_WINDOW_S)

    keys = sorted(key_stats.values(), key=lambda row: str(row.get("key") or ""))
    healthy = sum(1 for row in keys if row.get("state") == "healthy")
    summary = {
        "active": sum(int(row.get("active", 0) or 0) for row in keys),
        "rpm": len(recent_request_ts) * (60.0 / RPM_WINDOW_S),
        "tpm": sum(tokens for _ts, tokens in recent_token_events) * (60.0 / RPM_WINDOW_S),
        "peak_rpm": _peak_window_rate(all_request_ts),
        "peak_tpm": _peak_window_rate(
            [ts for ts, _tokens in all_token_events],
            values=[tokens for _ts, tokens in all_token_events],
        ),
        "ok": sum(int(row.get("ok", 0) or 0) for row in keys),
        "err": sum(int(row.get("err", 0) or 0) for row in keys),
        "keys": len(keys),
        "healthy": healthy,
        "telemetry_file": str(telemetry_path) if telemetry_path else "",
        "scheduler_db": os.environ.get(nvidia.SCHEDULER_DB_ENV, ""),
        "global_request_delay_s": float(
            os.environ.get(nvidia.GLOBAL_REQUEST_DELAY_ENV, "0") or 0.0
        ),
    }
    return {
        "summary": summary,
        "keys": keys,
        "sources": sorted(source_stats.values(), key=lambda row: str(row.get("source") or "")),
        "recent_errors": list(recent_errors),
    }


class NvidiaProviderHandler(BaseHTTPRequestHandler):
    server_version = "CapNvidiaProvider/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            _json_response(self, 200, {"ok": True})
            return
        if parsed.path == "/stats":
            _json_response(self, 200, build_stats())
            return
        _json_response(self, 404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/v1/chat/completions":
            _json_response(self, 404, {"error": {"message": "not found"}})
            return
        try:
            payload = _read_json_body(self)
        except Exception as exc:
            _json_response(self, 400, {"error": {"message": str(exc)}})
            return

        source = self.headers.get("X-CAP-NVIDIA-Source") or "provider_server"
        try:
            response = post_chat_completions_with_key_retry(
                payload,
                base_url=getattr(
                    self.server, "nvidia_base_url", nvidia.DEFAULT_BASE_URL
                ),
                telemetry_source=source,
            )
            _json_response(self, 200, response)
        except nvidia.NvidiaRateLimitError as exc:
            _json_response(self, 429, {"error": {"message": nvidia.sanitize_nvidia_error(str(exc))}})
        except ProviderRequestError as exc:
            _json_response(
                self,
                exc.http_status,
                {"error": {"message": nvidia.sanitize_nvidia_error(str(exc))}},
            )
        except Exception as exc:
            _json_response(
                self,
                500,
                {"error": {"message": nvidia.sanitize_nvidia_error(str(exc))}},
            )

    def log_message(self, _format: str, *_args: object) -> None:
        return


def init_provider_state(
    *,
    telemetry_file: str | None,
    scheduler_db: str | None,
    fresh_start: bool,
    request_delay_s: float | None,
    global_request_delay_s: float | None,
    max_concurrent_per_key: int | None,
    health_models: list[str] | None,
    base_url: str,
) -> None:
    if telemetry_file:
        os.environ[nvidia.TELEMETRY_FILE_ENV] = telemetry_file
    elif not os.environ.get(nvidia.TELEMETRY_FILE_ENV):
        os.environ[nvidia.TELEMETRY_FILE_ENV] = "/tmp/cap_nvidia_provider_requests.jsonl"

    if scheduler_db:
        os.environ[nvidia.SCHEDULER_DB_ENV] = scheduler_db
    elif not os.environ.get(nvidia.SCHEDULER_DB_ENV):
        os.environ[nvidia.SCHEDULER_DB_ENV] = "/tmp/cap_nvidia_provider.sqlite"

    if fresh_start:
        _fresh_start_provider_files(
            telemetry_path=Path(os.environ[nvidia.TELEMETRY_FILE_ENV]),
            scheduler_path=Path(os.environ[nvidia.SCHEDULER_DB_ENV]),
        )

    if request_delay_s is not None:
        os.environ[nvidia.REQUEST_DELAY_ENV] = str(request_delay_s)
    if global_request_delay_s is not None:
        os.environ[nvidia.GLOBAL_REQUEST_DELAY_ENV] = str(global_request_delay_s)
    if max_concurrent_per_key is not None:
        os.environ[nvidia.MAX_CONCURRENT_ENV] = str(max_concurrent_per_key)

    keys = nvidia.list_nvidia_keys()
    nvidia.init_nvidia_scheduler(
        os.environ[nvidia.SCHEDULER_DB_ENV],
        keys=keys,
        min_interval_s=float(os.environ.get(nvidia.REQUEST_DELAY_ENV, "0") or 0.0),
        max_concurrent_per_key=int(os.environ.get(nvidia.MAX_CONCURRENT_ENV, "1") or 1),
    )
    if keys:
        health = health_check_predefined_models(
            keys=keys,
            models=health_models or list(DEFAULT_HEALTH_MODELS),
            base_url=base_url,
        )
        nvidia.apply_nvidia_health_results(health)


def _fresh_start_provider_files(*, telemetry_path: Path, scheduler_path: Path) -> None:
    """Delete persisted provider telemetry/scheduler state before startup."""
    paths = [
        telemetry_path,
        scheduler_path,
        Path(str(scheduler_path) + "-wal"),
        Path(str(scheduler_path) + "-shm"),
        Path(str(scheduler_path) + "-journal"),
    ]
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[nvidia_provider] warning: failed to remove {path}: {exc}")


def _rank_health_status(status: str) -> int:
    order = {
        "healthy": 0,
        "rate_limited": 1,
        "timeout": 2,
        "error": 3,
        "invalid_auth": 4,
    }
    return order.get(status, 3)


def health_check_predefined_models(
    *,
    keys: list[str],
    models: list[str],
    base_url: str,
) -> list[dict[str, Any]]:
    """Health-check keys against predefined model routes and keep best status."""
    best: dict[str, dict[str, Any]] = {}
    for model in models:
        rows = nvidia.health_check_nvidia_keys(
            keys=keys,
            model=model,
            base_url=base_url,
            max_workers=min(4, len(keys)),
        )
        for row in rows:
            key = str(row.get("key") or "")
            if not key:
                continue
            existing = best.get(key)
            if existing is None or _rank_health_status(str(row.get("status") or "")) < _rank_health_status(
                str(existing.get("status") or "")
            ):
                best[key] = dict(row)
                best[key]["model"] = model
    return [best.get(nvidia.nvidia_key_label(key), {"key": nvidia.nvidia_key_label(key), "status": "error", "http_status": None, "error": "not checked"}) for key in keys]


def _http_status_from_error(exc: BaseException) -> int | None:
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(exc), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_invalid_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "http 401" in text or "authentication error" in text or "invalid_auth" in text


def classify_error(exc: BaseException) -> str:
    """Classify transport/provider failures for daemon retry policy."""
    if isinstance(exc, nvidia.NvidiaRateLimitError):
        return "rate_limit"
    status = _http_status_from_error(exc)
    text = str(exc).lower()
    if _is_invalid_auth_error(exc):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status == 409:
        return "conflict"
    if status is not None and 500 <= status <= 599:
        return "server"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if status is not None and 400 <= status <= 499:
        return "request"
    return "transient"


def _retry_delay_s(error_kind: str, attempt: int) -> float:
    if error_kind == "auth":
        return 0.0
    if error_kind == "rate_limit":
        return min(5.0, 1.0 + attempt * 0.5) + random.uniform(0.0, 0.25)
    if error_kind in {"conflict", "server", "timeout", "transient"}:
        return min(3.0, 0.25 * (2**attempt)) + random.uniform(0.0, 0.2)
    return 0.0


def post_chat_completions_with_key_retry(
    payload: dict[str, Any],
    *,
    base_url: str,
    telemetry_source: str,
) -> dict[str, Any]:
    """Send one request through daemon-managed key/error retry policy."""
    key_count = max(len(nvidia.list_nvidia_keys()), 1)
    max_attempts = max(1, _int_env("CAP_NVIDIA_PROVIDER_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    attempts = max(max_attempts, key_count)
    last_exc: Exception | None = None
    last_kind = "error"
    transient_attempts = 0
    for attempt in range(attempts):
        try:
            wait_for_global_request_slot()
            return nvidia.post_chat_completions_direct(
                payload,
                api_key=None,
                base_url=base_url,
                telemetry_source=telemetry_source,
                use_scheduler=True,
            )
        except nvidia.NvidiaRateLimitError as exc:
            last_exc = exc
            last_kind = classify_error(exc)
            # 429s usually indicate shared model/IP pressure. Do not burn the
            # whole key pool; give scheduler cooldown time to take effect.
            if attempt >= min(attempts, 2) - 1:
                raise
            time.sleep(_retry_delay_s(last_kind, attempt))
        except RuntimeError as exc:
            last_exc = exc
            last_kind = classify_error(exc)
            if last_kind == "request":
                raise ProviderRequestError(str(exc), http_status=400) from exc
            if last_kind not in {"auth", "conflict", "server", "timeout", "transient"}:
                raise
            if last_kind != "auth":
                transient_attempts += 1
                if transient_attempts >= max_attempts:
                    break
            if attempt >= attempts - 1:
                break
            time.sleep(_retry_delay_s(last_kind, attempt))

    status = 401 if last_kind == "auth" else 503
    raise ProviderRequestError(
        f"NVIDIA provider daemon exhausted retry budget after {attempts} attempt(s); "
        f"last_error={last_kind}: {last_exc}",
        http_status=status,
    ) from last_exc


def wait_for_global_request_slot() -> None:
    """Apply daemon-wide pacing before sending any NVIDIA request."""
    global _global_last_request_at
    delay = float(os.environ.get(nvidia.GLOBAL_REQUEST_DELAY_ENV, "0") or 0.0)
    if delay <= 0:
        return
    with _global_request_lock:
        now = time.time()
        wait_s = _global_last_request_at + delay - now
        if wait_s > 0:
            time.sleep(wait_s)
            now = time.time()
        _global_last_request_at = now


def make_server(host: str, port: int, *, base_url: str) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), NvidiaProviderHandler)
    server.nvidia_base_url = base_url  # type: ignore[attr-defined]
    return server


def _render_dashboard(stats: dict[str, Any]):
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    summary = stats.get("summary", {})
    line = Text.assemble(
        ("active ", "dim"),
        (str(int(summary.get("active", 0) or 0)), "cyan"),
        ("  rpm ", "dim"),
        (f"{float(summary.get('rpm', 0.0) or 0.0):.0f}", "cyan"),
        ("  tpm ", "dim"),
        (f"{float(summary.get('tpm', 0.0) or 0.0):.0f}", "cyan"),
        ("  peak rpm ", "dim"),
        (f"{float(summary.get('peak_rpm', 0.0) or 0.0):.0f}", "cyan"),
        ("  peak tpm ", "dim"),
        (f"{float(summary.get('peak_tpm', 0.0) or 0.0):.0f}", "cyan"),
        ("  ok ", "dim"),
        (str(int(summary.get("ok", 0) or 0)), "green"),
        ("  err ", "dim"),
        (str(int(summary.get("err", 0) or 0)), "red" if summary.get("err") else "white"),
        ("  keys ", "dim"),
        (str(int(summary.get("keys", 0) or 0)), "yellow"),
        ("  healthy ", "dim"),
        (str(int(summary.get("healthy", 0) or 0)), "green"),
        ("  delay ", "dim"),
        (f"{float(summary.get('global_request_delay_s', 0.0) or 0.0):.2f}s", "yellow"),
    )

    keys = Table(expand=True, box=None, show_header=True, header_style="dim")
    keys.add_column("key", width=12)
    keys.add_column("state", width=12)
    keys.add_column("act", width=4, justify="right")
    keys.add_column("rpm", width=5, justify="right")
    keys.add_column("tpm", width=7, justify="right")
    keys.add_column("ok", width=5, justify="right")
    keys.add_column("err", width=5, justify="right")
    keys.add_column("last", width=8, justify="right")
    keys.add_column("source", width=18)
    keys.add_column("last error", overflow="fold")
    for row in stats.get("keys", [])[:32]:
        state = str(row.get("state") or "unknown")
        state_style = "green" if state == "healthy" else "red"
        last_ms = row.get("last_ms")
        keys.add_row(
            Text(str(row.get("key") or ""), style="yellow"),
            Text(state, style=state_style),
            Text(str(int(row.get("active", 0) or 0)), style="cyan" if row.get("active") else "dim"),
            Text(f"{float(row.get('rpm', 0.0) or 0.0):.0f}", style="cyan"),
            Text(f"{float(row.get('tpm', 0.0) or 0.0):.0f}", style="cyan"),
            Text(str(int(row.get("ok", 0) or 0)), style="green"),
            Text(str(int(row.get("err", 0) or 0)), style="red" if row.get("err") else "dim"),
            f"{last_ms:.0f}ms" if isinstance(last_ms, (int, float)) else "-",
            str(row.get("source") or "-"),
            str(row.get("last_error") or "-"),
        )

    sources = Table(expand=True, box=None, show_header=True, header_style="dim")
    sources.add_column("source", width=20)
    sources.add_column("act", width=4, justify="right")
    sources.add_column("rpm", width=6, justify="right")
    sources.add_column("tpm", width=8, justify="right")
    sources.add_column("ok", width=6, justify="right")
    sources.add_column("err", width=6, justify="right")
    for row in stats.get("sources", [])[:8]:
        sources.add_row(
            str(row.get("source") or "-"),
            str(int(row.get("active", 0) or 0)),
            f"{float(row.get('rpm', 0.0) or 0.0):.0f}",
            f"{float(row.get('tpm', 0.0) or 0.0):.0f}",
            str(int(row.get("ok", 0) or 0)),
            str(int(row.get("err", 0) or 0)),
        )

    body = Table.grid(expand=True)
    body.add_column()
    body.add_row(line)
    body.add_row(keys)
    body.add_row(Panel(sources, title="sources", border_style="blue", padding=(0, 1)))
    return Panel(body, title="nvidia provider daemon", border_style="cyan", padding=(0, 1))


def run_dashboard(stop: threading.Event | None = None) -> None:
    from rich.live import Live

    stop = stop or threading.Event()
    with Live(_render_dashboard(build_stats()), refresh_per_second=2.0, transient=False) as live:
        while not stop.is_set():
            live.update(_render_dashboard(build_stats()))
            time.sleep(0.5)


def serve(args: ServeCommand) -> None:
    init_provider_state(
        telemetry_file=args.telemetry_file,
        scheduler_db=args.scheduler_db,
        fresh_start=args.fresh_start,
        request_delay_s=args.request_delay_s,
        global_request_delay_s=args.global_request_delay_s,
        max_concurrent_per_key=args.max_concurrent_per_key,
        health_models=args.health_models,
        base_url=args.base_url,
    )
    server = make_server(args.host, args.port, base_url=args.base_url)
    if args.dashboard:
        stop = threading.Event()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            run_dashboard(stop)
        finally:
            stop.set()
            server.shutdown()
            thread.join(timeout=2.0)
        return
    print(f"CAP NVIDIA provider server listening on http://{args.host}:{args.port}")
    print(f"Set CAP_NVIDIA_PROVIDER_URL=http://{args.host}:{args.port}")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    command_type = tyro.extras.subcommand_type_from_defaults(
        {
            "serve": ServeCommand(),
            "dashboard": DashboardCommand(),
        }
    )
    args = tyro.cli(
        command_type,
        args=argv,
        description="CAP NVIDIA provider daemon",
    )
    if isinstance(args, ServeCommand):
        serve(args)
        return 0
    if isinstance(args, DashboardCommand):
        run_dashboard()
        return 0
    raise AssertionError(f"Unhandled command type: {type(args)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
