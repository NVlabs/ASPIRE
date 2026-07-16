"""Standalone real-time debug UI for run_script log directories.

Run manually:

    uv run python -m cap.debug_ui.app log_dir=logs/nail_handover_...

run_script.py can also launch this module as a separate process.  The UI never
commands the robot; it only watches files inside a run log directory.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a run_script log dir with a web UI")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--script-name", default=None)
    parser.add_argument("--auto-latest", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--event-log-name", default="debug_events.jsonl")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--auto-exit-on-run-end", default="true")

    # Also accept Hydra-style key=value args so this can be launched from a
    # Hydra-configured process without needing another config system here.
    raw, unknown = parser.parse_known_args()
    data = vars(raw)
    for item in unknown:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().replace("debug_ui.", "").replace("-", "_")
        if key in {"log_dir", "logs_root", "script_name", "host", "event_log_name"}:
            data[key] = value
        elif key == "port":
            data[key] = int(value)
        elif key == "parent_pid":
            data["parent_pid"] = int(value)
        elif key == "poll_interval_s" or key == "poll_interval":
            data["poll_interval"] = float(value)
        elif key in {"auto_latest", "auto_exit_on_run_end"}:
            data[key] = _as_bool(value)
    data["auto_exit_on_run_end"] = _as_bool(data.get("auto_exit_on_run_end", True))
    return argparse.Namespace(**data)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _find_latest_log_dir(logs_root: Path, script_name: str | None) -> Path:
    if not logs_root.exists():
        raise FileNotFoundError(f"logs_root does not exist: {logs_root}")
    candidates = [p for p in logs_root.iterdir() if p.is_dir()]
    if script_name:
        stem = Path(script_name).stem
        candidates = [p for p in candidates if p.name.startswith(f"{stem}_")]
    if not candidates:
        raise FileNotFoundError(f"no log directories found under {logs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


class LogMonitor:
    def __init__(
        self,
        log_dir: Path,
        event_log_name: str,
        poll_interval: float,
        *,
        parent_pid: int = 0,
        auto_exit_on_run_end: bool = True,
    ) -> None:
        self.log_dir = log_dir.resolve()
        self.event_log_name = event_log_name
        self.poll_interval = poll_interval
        self.parent_pid = int(parent_pid or 0)
        self.auto_exit_on_run_end = bool(auto_exit_on_run_end)
        self._exit_timer_started = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._offsets: dict[Path, int] = {}
        self._seen_images: dict[Path, tuple[int, float]] = {}
        self._pending_images: dict[Path, int] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.stdout_tail: deque[str] = deque(maxlen=300)
        self.images: deque[dict[str, Any]] = deque(maxlen=300)
        self.state: dict[str, Any] = {
            "log_dir": str(self.log_dir),
            "status": "starting",
            "current_skill": None,
            "current_tool": None,
            "last_event": None,
            "last_image": None,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="debug-ui-log-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": dict(self.state),
                "events": list(self.events),
                "stdout_tail": list(self.stdout_tail),
                "images": list(self.images),
            }

    def _publish(self, event: dict[str, Any]) -> None:
        event = {k: _json_safe(v) for k, v in event.items()}
        with self._lock:
            self.events.append(event)
            self.state["last_event"] = event
            if event.get("type") == "skill_start":
                self.state["current_skill"] = event
                self.state["status"] = f"skill: {event.get('name')}"
            elif event.get("type") == "skill_end":
                if self.state.get("current_skill", {}).get("call_id") == event.get("call_id"):
                    self.state["current_skill"] = None
                self.state["status"] = f"finished skill: {event.get('name')}"
            elif event.get("type") == "skill_error":
                if self.state.get("current_skill", {}).get("call_id") == event.get("call_id"):
                    self.state["current_skill"] = None
                self.state["status"] = f"skill error: {event.get('name')}"
            elif event.get("type") == "tool_start":
                self.state["current_tool"] = event
                self.state["status"] = f"tool: {event.get('name')}"
            elif event.get("type") == "tool_end":
                if self.state.get("current_tool", {}).get("call_id") == event.get("call_id"):
                    self.state["current_tool"] = None
                self.state["status"] = f"finished tool: {event.get('name')}"
            elif event.get("type") == "image":
                self.images.appendleft(event)
                self.state["last_image"] = event
            elif event.get("type") == "stdout":
                text = str(event.get("text", ""))
                if text:
                    self.stdout_tail.append(text)
            elif event.get("type") == "run_end":
                self.state["status"] = "run ended"
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass
        if event.get("type") == "run_end" and self.auto_exit_on_run_end:
            self._exit_soon("run_end")

    def _run(self) -> None:
        self._publish({"type": "ui_started", "log_dir": str(self.log_dir)})
        while not self._stop.is_set():
            try:
                if self._parent_is_gone():
                    self._publish({"type": "ui_parent_gone", "parent_pid": self.parent_pid})
                    self._exit_soon("parent_gone")
                    return
                self._scan_event_log()
                self._scan_text_logs()
                self._scan_images()
            except Exception as exc:
                self._publish({"type": "ui_error", "error": f"{type(exc).__name__}: {exc}"})
            time.sleep(self.poll_interval)

    def _parent_is_gone(self) -> bool:
        if self.parent_pid <= 0:
            return False
        if self.parent_pid == os.getpid():
            return False
        try:
            os.kill(self.parent_pid, 0)
            return False
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except Exception:
            return False

    def _exit_soon(self, reason: str) -> None:
        if self._exit_timer_started:
            return
        self._exit_timer_started = True

        def _exit() -> None:
            time.sleep(0.5)
            os._exit(0)

        threading.Thread(target=_exit, name=f"debug-ui-exit-{reason}", daemon=True).start()

    def _read_new_text(self, path: Path) -> str:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return ""
        old = self._offsets.get(path, 0)
        if size < old:
            old = 0
        if size == old:
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(old)
            chunk = fh.read()
            self._offsets[path] = fh.tell()
        return chunk

    def _scan_event_log(self) -> None:
        path = self.log_dir / self.event_log_name
        if not path.exists():
            return
        chunk = self._read_new_text(path)
        if not chunk:
            return
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._publish(event)

    def _scan_text_logs(self) -> None:
        paths = sorted(self.log_dir.glob("run_*.txt")) + [self.log_dir / "exec.log"]
        for path in paths:
            if not path.exists() or path.name == self.event_log_name:
                continue
            chunk = self._read_new_text(path)
            if not chunk:
                continue
            for line in chunk.splitlines():
                if line.strip():
                    self._publish({"type": "stdout", "source": path.name, "text": line})

    def _scan_images(self) -> None:
        vis_dir = self.log_dir / "vis"
        if not vis_dir.exists():
            return
        paths = list(vis_dir.rglob("*"))
        for path in sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if path in self._seen_images:
                continue
            previous_size = self._pending_images.get(path)
            if previous_size != stat.st_size:
                self._pending_images[path] = stat.st_size
                continue
            self._pending_images.pop(path, None)
            rel = path.relative_to(self.log_dir).as_posix()
            self._seen_images[path] = (stat.st_size, stat.st_mtime)
            self._publish(
                {
                    "type": "image",
                    "path": rel,
                    "url": f"/artifact/{rel}",
                    "name": path.name,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            )


def _html() -> str:
    return r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>run_script debug UI</title>
  <style>
    body { margin:0; font-family: Arial, sans-serif; background:#111; color:#eee; }
    header { height:44px; display:flex; align-items:center; gap:20px; padding:0 14px; background:#181818; border-bottom:1px solid #333; }
    #main { display:grid; grid-template-columns: 42% 58%; height: calc(100vh - 45px); }
    #left { border-right:1px solid #333; overflow:auto; padding:12px; }
    #right { overflow:auto; padding:12px; background:#0d0d0d; }
    .card { background:#1b1b1b; border:1px solid #333; border-radius:8px; margin-bottom:12px; padding:10px; }
    .muted { color:#aaa; font-size:12px; }
    .name { color:#79b8ff; font-weight:bold; }
    pre { white-space:pre-wrap; word-break:break-word; background:#080808; padding:10px; border-radius:6px; border:1px solid #333; }
    img { max-width:100%; border-radius:4px; border:1px solid #444; background:#000; }
    button { background:#2c2c2c; color:#eee; border:1px solid #555; border-radius:5px; padding:5px 8px; }
    .row { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    #events .event { border-bottom:1px solid #292929; padding:5px 0; font-size:13px; }
    #source { max-height:320px; overflow:auto; }
  </style>
</head>
<body>
<header>
  <strong>run_script debug UI</strong>
  <span id="status" class="muted">connecting...</span>
  <button onclick="toggleAutoScroll()">toggle autoscroll</button>
</header>
<div id="main">
  <section id="left">
    <div class="card">
      <div class="muted">log dir</div>
      <div id="logdir"></div>
    </div>
    <div class="card">
      <div class="row"><h3>Current execution</h3><span class="muted" id="lastts"></span></div>
      <div>Skill: <span id="skill" class="name">none</span></div>
      <div>Tool: <span id="tool" class="name">none</span></div>
    </div>
    <div class="card">
      <h3>Current source</h3>
      <pre id="source">No source event yet.</pre>
    </div>
    <div class="card">
      <h3>Recent events</h3>
      <div id="events"></div>
    </div>
    <div class="card">
      <h3>Stdout tail</h3>
      <pre id="stdout"></pre>
    </div>
  </section>
  <section id="right">
    <h3>Debug visualization feed</h3>
    <div id="feed"></div>
  </section>
</div>
<script>
let autoScroll = true;
let lastSourceKey = null;
function toggleAutoScroll(){ autoScroll = !autoScroll; }
function esc(s){ return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function summarize(ev){
  const type = ev.type || 'event';
  const name = ev.name ? ' ' + ev.name : '';
  const err = ev.error ? ' ERROR=' + ev.error : '';
  return `${ev.ts || ''} ${type}${name}${err}`;
}
async function loadSource(ev){
  if (!ev || !ev.source_file) return;
  const key = ev.source_file + ':' + (ev.source_line || '');
  if (key === lastSourceKey) return;
  lastSourceKey = key;
  try {
    const res = await fetch(`/source?path=${encodeURIComponent(ev.source_file)}&line=${encodeURIComponent(ev.source_line || 1)}`);
    const data = await res.json();
    document.getElementById('source').textContent = data.text || data.error || 'No source';
  } catch(e) {}
}
function applyState(data){
  document.getElementById('logdir').textContent = data.state.log_dir || '';
  document.getElementById('status').textContent = data.state.status || '';
  document.getElementById('skill').textContent = data.state.current_skill?.name || 'none';
  document.getElementById('tool').textContent = data.state.current_tool?.name || 'none';
  const srcEv = data.state.current_skill || data.state.current_tool || data.state.last_event;
  loadSource(srcEv);
  document.getElementById('stdout').textContent = (data.stdout_tail || []).slice(-120).join('\n');
  const events = document.getElementById('events');
  events.innerHTML = (data.events || []).slice(-80).reverse().map(ev => `<div class="event">${esc(summarize(ev))}</div>`).join('');
  const feed = document.getElementById('feed');
  feed.innerHTML = (data.images || []).map(img => imageCard(img)).join('');
}
function imageCard(img){
  return `<div class="card"><div class="row"><div class="name">${esc(img.name || img.path)}</div><div class="muted">${esc(img.ts || '')}</div></div><div class="muted">${esc(img.path || '')}</div><a href="${esc(img.url)}" target="_blank"><img src="${esc(img.url)}" loading="lazy" /></a></div>`;
}
function onEvent(ev){
  document.getElementById('lastts').textContent = ev.ts || '';
  fetch('/state').then(r=>r.json()).then(applyState).catch(()=>{});
  if (autoScroll && ev.type === 'image') document.getElementById('right').scrollTop = 0;
}
fetch('/state').then(r=>r.json()).then(applyState);
const es = new EventSource('/events/stream');
es.onmessage = (msg) => { try { onEvent(JSON.parse(msg.data)); } catch(e) {} };
es.onerror = () => { document.getElementById('status').textContent = 'disconnected/retrying'; };
</script>
</body>
</html>
"""


def create_app(
    log_dir: Path,
    event_log_name: str,
    poll_interval: float,
    *,
    parent_pid: int = 0,
    auto_exit_on_run_end: bool = True,
) -> FastAPI:
    monitor = LogMonitor(
        log_dir,
        event_log_name,
        poll_interval,
        parent_pid=parent_pid,
        auto_exit_on_run_end=auto_exit_on_run_end,
    )
    app = FastAPI(title="run_script debug UI")
    app.state.monitor = monitor

    @app.on_event("startup")
    def _startup() -> None:
        monitor.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        monitor.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html()

    @app.get("/state")
    def state() -> JSONResponse:
        return JSONResponse(monitor.snapshot())

    @app.get("/events/stream")
    def events_stream(request: Request) -> StreamingResponse:
        q = monitor.subscribe()

        def gen() -> Iterable[str]:
            try:
                snap = monitor.snapshot()
                yield "data: " + json.dumps({"type": "snapshot", "state": snap["state"]}) + "\n\n"
                while True:
                    try:
                        event = q.get(timeout=10.0)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield "data: " + json.dumps(event, default=str) + "\n\n"
            finally:
                monitor.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/source")
    def source(path: str = Query(...), line: int = Query(1)) -> JSONResponse:
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail=f"source not found: {p}")
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        start = max(1, int(line) - 20)
        end = min(len(lines), int(line) + 80)
        width = len(str(end))
        snippet = "\n".join(f"{i:>{width}}  {lines[i-1]}" for i in range(start, end + 1))
        return JSONResponse({"path": str(p), "line": line, "text": snippet})

    # Restrict static serving to the selected log directory.
    app.mount("/artifact", StaticFiles(directory=str(log_dir.resolve())), name="artifact")
    return app


def main() -> None:
    args = _parse_args()
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve()
    else:
        log_dir = _find_latest_log_dir(Path(args.logs_root).expanduser().resolve(), args.script_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    import uvicorn

    app = create_app(
        log_dir,
        args.event_log_name,
        args.poll_interval,
        parent_pid=args.parent_pid,
        auto_exit_on_run_end=args.auto_exit_on_run_end,
    )
    print(f"[debug_ui] Watching {log_dir}")
    print(f"[debug_ui] http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
