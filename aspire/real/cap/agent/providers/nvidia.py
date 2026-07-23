# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NVIDIA inference-gateway helpers.

This module owns provider-level behavior common to both the agent LLM backend
and the VLM tool backend: key discovery/rotation, model quirks, direct
chat-completions HTTP calls, and OpenAI-compatible response parsing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1/"
REQUEST_TIMEOUT_S = 60.0
TELEMETRY_FILE_ENV = "CAP_NVIDIA_TELEMETRY_FILE"
SCHEDULER_DB_ENV = "CAP_NVIDIA_SCHEDULER_DB"
REQUEST_DELAY_ENV = "CAP_NVIDIA_REQUEST_DELAY_S"
MAX_CONCURRENT_ENV = "CAP_NVIDIA_MAX_CONCURRENT_PER_KEY"
ACQUIRE_TIMEOUT_ENV = "CAP_NVIDIA_ACQUIRE_TIMEOUT_S"
COOLDOWN_429_ENV = "CAP_NVIDIA_429_COOLDOWN_S"
PROVIDER_URL_ENV = "CAP_NVIDIA_PROVIDER_URL"
GLOBAL_REQUEST_DELAY_ENV = "CAP_NVIDIA_GLOBAL_REQUEST_DELAY_S"
DEFAULT_ACQUIRE_TIMEOUT_S = 300.0
DEFAULT_429_COOLDOWN_S = 30.0
HEALTH_CHECK_MAX_TOKENS = 16

log = logging.getLogger(__name__)


class NvidiaRateLimitError(RuntimeError):
    """429 from NVIDIA chat completions."""


@dataclass(frozen=True)
class NvidiaKeyLease:
    """One scheduler lease for a selected NVIDIA key."""

    api_key: str
    key: str
    request_id: str
    scheduler_path: Path | None


def list_nvidia_keys() -> list[str]:
    """Return NVIDIA keys from env, collecting ``NVIDIA_API_KEY_1..100``.

    Indexed keys win over the single fallback key. Gaps are allowed.
    """
    keys: list[str] = []
    for i in range(1, 101):
        key = os.environ.get(f"NVIDIA_API_KEY_{i}")
        if key:
            keys.append(key)
    if not keys:
        fallback = os.environ.get("NVIDIA_API_KEY")
        if fallback:
            keys.append(fallback)
    return keys


_rr_keys: list[str] = list_nvidia_keys()
random.shuffle(_rr_keys)
_rr_counter: itertools.count = itertools.count()
_request_counter: itertools.count = itertools.count()


def pick_nvidia_key(key_index: int | None) -> str | None:
    """Pick an NVIDIA key by explicit index, or the first configured key."""
    keys = list_nvidia_keys()
    if not keys:
        return None
    if key_index is None:
        return keys[0]
    return keys[key_index % len(keys)]


def auto_pick_nvidia_key() -> str:
    """Pick the next NVIDIA key from the process-local round-robin snapshot."""
    if not _rr_keys:
        raise RuntimeError(
            "No NVIDIA key found. Set NVIDIA_API_KEY or NVIDIA_API_KEY_1..N."
        )
    return _rr_keys[next(_rr_counter) % len(_rr_keys)]


# Backward-compatible private name used by older imports.
_auto_pick_key = auto_pick_nvidia_key


def nvidia_key_label(api_key: str) -> str:
    """Return a stable redacted key label suitable for telemetry/UI."""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:6]
    for idx, key in enumerate(list_nvidia_keys(), start=1):
        if key == api_key:
            return f"key{idx:02d}:{digest}"
    return f"key?:{digest}"


def sanitize_nvidia_error(message: str | None) -> str:
    """Redact API-key material from provider/gateway error strings."""
    if not message:
        return ""
    text = str(message)
    text = re.sub(
        r"(Received API Key\s*=\s*)([^,\s]+)",
        r"\1[redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Key Hash \(Token\)\s*=\s*[0-9a-f]{12,}",
        "Key Hash (Token) = [redacted]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_\-]{6,}\b", "[redacted-api-key]", text)
    return text


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _scheduler_db_path() -> Path | None:
    raw = os.environ.get(SCHEDULER_DB_ENV)
    if not raw:
        return None
    return Path(raw)


def _connect_scheduler(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _create_scheduler_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nvidia_keys (
            key TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0,
            next_allowed_at REAL NOT NULL DEFAULT 0,
            ok_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            last_http_status INTEGER,
            last_error TEXT,
            last_latency_ms REAL,
            min_interval_s REAL NOT NULL DEFAULT 0,
            max_concurrent INTEGER NOT NULL DEFAULT 1,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nvidia_leases (
            request_id TEXT PRIMARY KEY,
            key TEXT NOT NULL,
            pid INTEGER NOT NULL,
            thread_id INTEGER NOT NULL,
            acquired_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def init_nvidia_scheduler(
    db_path: str | Path,
    *,
    keys: list[str] | None = None,
    min_interval_s: float | None = None,
    max_concurrent_per_key: int | None = None,
) -> Path:
    """Create/reset the local SQLite scheduler state for one agent run."""
    path = Path(db_path)
    os.environ[SCHEDULER_DB_ENV] = str(path)
    keys = keys if keys is not None else list_nvidia_keys()
    min_interval = (
        _float_env(REQUEST_DELAY_ENV, 0.0)
        if min_interval_s is None
        else float(min_interval_s)
    )
    max_concurrent = (
        _int_env(MAX_CONCURRENT_ENV, 1)
        if max_concurrent_per_key is None
        else int(max_concurrent_per_key)
    )
    now = time.time()
    with _connect_scheduler(path) as conn:
        _create_scheduler_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM nvidia_leases")
        conn.execute("DELETE FROM nvidia_keys")
        for key in keys:
            conn.execute(
                """
                INSERT INTO nvidia_keys (
                    key, status, active, next_allowed_at, min_interval_s,
                    max_concurrent, updated_at
                )
                VALUES (?, 'healthy', 0, 0, ?, ?, ?)
                """,
                (nvidia_key_label(key), min_interval, max(max_concurrent, 1), now),
            )
        conn.commit()
    return path


def _local_key_map(extra_api_key: str | None = None) -> dict[str, str]:
    mapping = {nvidia_key_label(key): key for key in list_nvidia_keys()}
    if extra_api_key:
        mapping.setdefault(nvidia_key_label(extra_api_key), extra_api_key)
    return mapping


def _release_stale_leases(
    conn: sqlite3.Connection, *, stale_after_s: float = 900.0
) -> None:
    cutoff = time.time() - stale_after_s
    stale = conn.execute(
        "SELECT request_id, key FROM nvidia_leases WHERE acquired_at < ?", (cutoff,)
    ).fetchall()
    for row in stale:
        conn.execute(
            """
            UPDATE nvidia_keys
            SET active = MAX(active - 1, 0), updated_at = ?
            WHERE key = ?
            """,
            (time.time(), row["key"]),
        )
        conn.execute(
            "DELETE FROM nvidia_leases WHERE request_id = ?", (row["request_id"],)
        )


def acquire_nvidia_key(
    *,
    request_id: str,
    api_key: str | None = None,
    timeout_s: float | None = None,
) -> NvidiaKeyLease:
    """Acquire one healthy key lease from the process-shared scheduler."""
    path = _scheduler_db_path()
    if path is None:
        key = api_key or auto_pick_nvidia_key()
        return NvidiaKeyLease(
            api_key=key,
            key=nvidia_key_label(key),
            request_id=request_id,
            scheduler_path=None,
        )

    key_map = _local_key_map(api_key)
    target_label = nvidia_key_label(api_key) if api_key else None
    timeout = (
        _float_env(ACQUIRE_TIMEOUT_ENV, DEFAULT_ACQUIRE_TIMEOUT_S)
        if timeout_s is None
        else timeout_s
    )
    deadline = time.time() + max(timeout, 0.0)
    while True:
        now = time.time()
        if not key_map:
            raise RuntimeError(
                "No NVIDIA key found. Set NVIDIA_API_KEY or NVIDIA_API_KEY_1..N."
            )
        with _connect_scheduler(path) as conn:
            _create_scheduler_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            _release_stale_leases(conn)
            labels = [target_label] if target_label else list(key_map)
            placeholders = ",".join("?" for _ in labels)
            row = conn.execute(
                f"""
                SELECT key FROM nvidia_keys
                WHERE key IN ({placeholders})
                  AND status = 'healthy'
                  AND active < max_concurrent
                  AND next_allowed_at <= ?
                ORDER BY active ASC, next_allowed_at ASC, updated_at ASC, key ASC
                LIMIT 1
                """,
                [*labels, now],
            ).fetchone()
            if row is not None:
                label = str(row["key"])
                conn.execute(
                    """
                    UPDATE nvidia_keys
                    SET active = active + 1, updated_at = ?
                    WHERE key = ?
                    """,
                    (now, label),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO nvidia_leases (
                        request_id, key, pid, thread_id, acquired_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (request_id, label, os.getpid(), threading.get_ident(), now),
                )
                conn.commit()
                return NvidiaKeyLease(
                    api_key=key_map[label],
                    key=label,
                    request_id=request_id,
                    scheduler_path=path,
                )
            conn.commit()

        if now >= deadline:
            raise RuntimeError("No healthy NVIDIA key available from scheduler")
        time.sleep(min(0.1, max(deadline - now, 0.0)))


def release_nvidia_key(
    lease: NvidiaKeyLease,
    *,
    status: str,
    http_status: int | None,
    latency_ms: float,
    error: str | None,
) -> None:
    """Release a scheduler key lease and update key health/cooldown state."""
    if lease.scheduler_path is None:
        return
    now = time.time()
    clean_error = sanitize_nvidia_error(error)
    key_status = "healthy"
    next_allowed_at = now
    if http_status == 401:
        key_status = "invalid_auth"
    elif http_status == 429:
        next_allowed_at = now + _float_env(COOLDOWN_429_ENV, DEFAULT_429_COOLDOWN_S)

    with _connect_scheduler(lease.scheduler_path) as conn:
        _create_scheduler_schema(conn)
        row = conn.execute(
            "SELECT min_interval_s FROM nvidia_keys WHERE key = ?", (lease.key,)
        ).fetchone()
        min_interval = float(row["min_interval_s"]) if row is not None else 0.0
        if http_status != 429:
            next_allowed_at = max(next_allowed_at, now + min_interval)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM nvidia_leases WHERE request_id = ?", (lease.request_id,)
        )
        conn.execute(
            """
            UPDATE nvidia_keys
            SET active = MAX(active - 1, 0),
                status = ?,
                next_allowed_at = ?,
                ok_count = ok_count + ?,
                error_count = error_count + ?,
                last_http_status = ?,
                last_error = ?,
                last_latency_ms = ?,
                updated_at = ?
            WHERE key = ?
            """,
            (
                key_status,
                next_allowed_at,
                1 if status == "ok" else 0,
                1 if status != "ok" else 0,
                http_status,
                clean_error,
                latency_ms,
                now,
                lease.key,
            ),
        )
        conn.commit()


def read_nvidia_scheduler_stats() -> list[dict[str, Any]]:
    """Read current scheduler key rows for dashboard/tests."""
    path = _scheduler_db_path()
    if path is None or not path.exists():
        return []
    with _connect_scheduler(path) as conn:
        _create_scheduler_schema(conn)
        rows = conn.execute(
            """
            SELECT key, status, active, next_allowed_at, ok_count, error_count,
                   last_http_status, last_error, last_latency_ms, min_interval_s,
                   max_concurrent, updated_at
            FROM nvidia_keys
            ORDER BY key ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def apply_nvidia_health_results(results: list[dict[str, Any]]) -> None:
    """Apply health-check statuses to the active scheduler DB."""
    path = _scheduler_db_path()
    if path is None:
        return
    now = time.time()
    with _connect_scheduler(path) as conn:
        _create_scheduler_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        for result in results:
            label = str(result.get("key") or "")
            status = str(result.get("status") or "error")
            scheduler_status = "healthy" if status in ("healthy", "rate_limited") else status
            next_allowed_at = 0.0
            if status == "rate_limited":
                next_allowed_at = now + _float_env(
                    COOLDOWN_429_ENV, DEFAULT_429_COOLDOWN_S
                )
            conn.execute(
                """
                UPDATE nvidia_keys
                SET status = ?, next_allowed_at = ?, last_http_status = ?,
                    last_error = ?, updated_at = ?
                WHERE key = ?
                """,
                (
                    scheduler_status,
                    next_allowed_at,
                    result.get("http_status"),
                    sanitize_nvidia_error(str(result.get("error") or "")),
                    now,
                    label,
                ),
            )
        conn.commit()


def _telemetry_path() -> Path | None:
    raw = os.environ.get(TELEMETRY_FILE_ENV)
    if not raw:
        return None
    return Path(raw)


def _write_telemetry_event(event: dict[str, Any]) -> None:
    path = _telemetry_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        log.debug("failed to write NVIDIA telemetry event", exc_info=True)


def _request_id() -> str:
    return f"{os.getpid()}-{threading.get_ident()}-{next(_request_counter)}"


def _telemetry_base(
    *,
    request_id: str,
    api_key: str,
    payload: dict[str, Any],
    source: str | None,
) -> dict[str, Any]:
    model = payload.get("model")
    return {
        "request_id": request_id,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "source": source or "nvidia",
        "model": model if isinstance(model, str) else None,
        "key": nvidia_key_label(api_key),
    }


def _emit_request_start(
    *,
    request_id: str,
    api_key: str,
    payload: dict[str, Any],
    source: str | None,
) -> None:
    _write_telemetry_event(
        {
            **_telemetry_base(
                request_id=request_id,
                api_key=api_key,
                payload=payload,
                source=source,
            ),
            "event": "start",
            "ts": time.time(),
        }
    )


def _emit_request_end(
    *,
    request_id: str,
    api_key: str,
    payload: dict[str, Any],
    source: str | None,
    status: str,
    latency_ms: float,
    http_status: int | None,
    error: str | None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    _write_telemetry_event(
        {
            **_telemetry_base(
                request_id=request_id,
                api_key=api_key,
                payload=payload,
                source=source,
            ),
            "event": "end",
            "ts": time.time(),
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "http_status": http_status,
            "error": error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )


def _usage_tokens(response: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    def as_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None

    prompt_tokens = as_int(usage.get("prompt_tokens"))
    completion_tokens = as_int(usage.get("completion_tokens"))
    total_tokens = as_int(usage.get("total_tokens"))
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return prompt_tokens, completion_tokens, total_tokens


def nvidia_model_supports_temperature(model: str) -> bool:
    """Return whether the NVIDIA gateway model accepts explicit temperature."""
    model_lower = model.lower()
    claude4_suffixes = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")
    return not any(suffix in model_lower for suffix in claude4_suffixes)


def chat_completions_url(base_url: str = DEFAULT_BASE_URL) -> str:
    """Normalize a gateway base URL to the chat-completions endpoint."""
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _provider_server_chat_url() -> str | None:
    raw = os.environ.get(PROVIDER_URL_ENV, "").strip()
    if not raw:
        return None
    base = raw.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_error_message(payload: Any) -> str:
    """Extract a human-readable error string from gateway-style payloads."""
    message = ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("message", "detail", "code", "type"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    message = value.strip()
                    break
        for key in ("message", "detail", "error"):
            if message:
                break
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break
    elif isinstance(payload, str) and payload.strip():
        message = payload.strip()
    return sanitize_nvidia_error(message)


def _health_status_from_error(exc: Exception, http_status: int | None = None) -> str:
    text = str(exc).lower()
    if http_status == 401 or "http 401" in text or "authentication error" in text:
        return "invalid_auth"
    if http_status == 429 or "http 429" in text or "rate" in text:
        return "rate_limited"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    return "error"


def _http_status_from_error_text(text: str) -> int | None:
    match = re.search(r"\bHTTP\s+(\d{3})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_health_probe_output_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "max_tokens" in text
        and "model output limit" in text
        and _http_status_from_error_text(str(exc)) == 400
    )


def health_check_nvidia_keys(
    *,
    keys: list[str] | None = None,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 10.0,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """Probe keys with a tiny chat request and return per-key health rows."""
    keys = keys if keys is not None else list_nvidia_keys()
    if not keys:
        return []

    def check_one(key: str) -> dict[str, Any]:
        label = nvidia_key_label(key)
        try:
            post_chat_completions(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply only: ok"}],
                    "max_tokens": HEALTH_CHECK_MAX_TOKENS,
                },
                api_key=key,
                base_url=base_url,
                timeout=timeout,
                telemetry_source="health_check",
                use_scheduler=False,
            )
            return {"key": label, "status": "healthy", "http_status": 200, "error": ""}
        except NvidiaRateLimitError as exc:
            return {
                "key": label,
                "status": "rate_limited",
                "http_status": 429,
                "error": sanitize_nvidia_error(str(exc)),
            }
        except Exception as exc:
            if _is_health_probe_output_limit_error(exc):
                return {"key": label, "status": "healthy", "http_status": 200, "error": ""}
            http_status = _http_status_from_error_text(str(exc))
            return {
                "key": label,
                "status": _health_status_from_error(exc, http_status=http_status),
                "http_status": http_status,
                "error": sanitize_nvidia_error(str(exc)),
            }

    workers = max(1, min(max_workers, len(keys)))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(check_one, key): idx for idx, key in enumerate(keys)}
        for fut in as_completed(future_to_idx):
            results.append((future_to_idx[fut], fut.result()))
    return [row for _, row in sorted(results, key=lambda item: item[0])]


def post_chat_completions(
    payload: dict[str, Any],
    *,
    api_key: str | None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = REQUEST_TIMEOUT_S,
    telemetry_source: str | None = None,
    use_scheduler: bool = True,
) -> dict[str, Any]:
    """POST a raw OpenAI-compatible chat-completions request to NVIDIA."""
    provider_url = _provider_server_chat_url()
    if provider_url and use_scheduler:
        return _post_chat_completions_via_provider_server(
            provider_url,
            payload,
            timeout=timeout,
            telemetry_source=telemetry_source,
        )
    return post_chat_completions_direct(
        payload,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        telemetry_source=telemetry_source,
        use_scheduler=use_scheduler,
    )


def _post_chat_completions_via_provider_server(
    provider_url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    telemetry_source: str | None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        provider_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-CAP-NVIDIA-Source": telemetry_source or "nvidia",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(error_body)
        except json.JSONDecodeError:
            error_payload = error_body
        message = extract_error_message(error_payload) or str(exc.reason)
        if exc.code == 429:
            raise NvidiaRateLimitError(message) from exc
        raise RuntimeError(
            f"NVIDIA provider server request failed with HTTP {exc.code}: {message}"
        ) from exc
    except urllib.error.URLError as exc:
        message = sanitize_nvidia_error(str(exc.reason))
        raise RuntimeError(f"NVIDIA provider server request failed: {message}") from exc

    try:
        response_payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("NVIDIA provider server returned invalid JSON") from exc
    if not isinstance(response_payload, dict):
        raise RuntimeError("NVIDIA provider server returned a non-object response")
    response_error_message = extract_error_message(response_payload)
    if response_error_message:
        raise RuntimeError(
            f"NVIDIA provider server returned an error: {response_error_message}"
        )
    return response_payload


def post_chat_completions_direct(
    payload: dict[str, Any],
    *,
    api_key: str | None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = REQUEST_TIMEOUT_S,
    telemetry_source: str | None = None,
    use_scheduler: bool = True,
) -> dict[str, Any]:
    """Direct NVIDIA chat-completions call used by the node provider daemon."""
    request_id = _request_id()
    lease: NvidiaKeyLease | None = None
    if use_scheduler:
        lease = acquire_nvidia_key(request_id=request_id, api_key=api_key)
        request_key = lease.api_key
    else:
        request_key = api_key or auto_pick_nvidia_key()

    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {request_key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    status = "ok"
    http_status: int | None = None
    error_message_for_telemetry: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    _emit_request_start(
        request_id=request_id,
        api_key=request_key,
        payload=payload,
        source=telemetry_source,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_status = getattr(response, "status", None) or getattr(response, "code", None)
            if isinstance(raw_status, int):
                http_status = raw_status
            raw_body = response.read().decode("utf-8")
        try:
            response_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("NVIDIA chat completions returned invalid JSON") from exc
        if not isinstance(response_payload, dict):
            raise RuntimeError("NVIDIA chat completions returned a non-object response")
        response_error_message = extract_error_message(response_payload)
        if response_error_message:
            raise RuntimeError(
                f"NVIDIA chat completions returned an error: {response_error_message}"
            )
        prompt_tokens, completion_tokens, total_tokens = _usage_tokens(response_payload)
        return response_payload
    except urllib.error.HTTPError as exc:
        status = "error"
        http_status = exc.code
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(error_body)
        except json.JSONDecodeError:
            error_payload = error_body
        message = extract_error_message(error_payload) or str(exc.reason)
        error_message_for_telemetry = message
        if exc.code == 429:
            raise NvidiaRateLimitError(message) from exc
        raise RuntimeError(
            f"NVIDIA chat completions request failed with HTTP {exc.code}: {message}"
        ) from exc
    except urllib.error.URLError as exc:
        status = "error"
        error_message_for_telemetry = sanitize_nvidia_error(str(exc.reason))
        raise RuntimeError(
            f"NVIDIA chat completions request failed: {error_message_for_telemetry}"
        ) from exc
    except Exception as exc:
        status = "error"
        error_message_for_telemetry = sanitize_nvidia_error(str(exc))
        raise
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if lease is not None:
            release_nvidia_key(
                lease,
                status=status,
                http_status=http_status,
                latency_ms=latency_ms,
                error=error_message_for_telemetry,
            )
        _emit_request_end(
            request_id=request_id,
            api_key=request_key,
            payload=payload,
            source=telemetry_source,
            status=status,
            latency_ms=latency_ms,
            http_status=http_status,
            error=error_message_for_telemetry,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def coerce_message_text(content: Any) -> str:
    """Coerce OpenAI-compatible message content into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            if item.get("type") == "text" and text is not None:
                parts.append(str(text))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def response_text(response: dict[str, Any]) -> str:
    """Extract the first chat-completions message text."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("NVIDIA chat completions returned no choices")
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise RuntimeError("NVIDIA chat completions returned an invalid choice")
    message = choice0.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("NVIDIA chat completions returned no message payload")
    return coerce_message_text(message.get("content"))


def usage_dict(response: dict[str, Any], *, model: str) -> dict[str, int | str]:
    """Return the normalized usage dict used by agent conversation logs."""
    usage = response.get("usage")
    raw_usage = usage if isinstance(usage, dict) else {}
    return {
        "model": model,
        "input_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
    }
