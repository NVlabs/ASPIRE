"""Interactive TUI dashboard for real YAM deployment.

Keyboard shortcuts while RUNNING:
  SPACE  — pause / resume motion (robot holds current position)
  X / Q  — emergency stop & go home

Keyboard shortcuts after script DONE:
  ENTER  — go home then exit
  S      — skip go home, exit immediately
  X      — emergency stop (no graceful home)

Usage — automatic via run_script.py:
    uv run python run_script.py robot=real_yam env.name=yam-real script_file=...
    # RealYamBackend starts the dashboard only for the script-execution process.

Usage — manual context manager:
    from cap.env.real_bimanual_yam.dashboard import YamDashboard
    dash = YamDashboard(env, ns["get_robot_state"])
    with dash:
        run_script(ns)
    should_home = dash.await_exit()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cap.env.real_bimanual_yam.skills import (
    _pause_requested,
    _stop_requested,
)

_REFRESH_HZ = 5.0
_REFRESH_DT = 1.0 / _REFRESH_HZ
_TOOL_HISTORY_LEN = 20
_SLOW_MS = 1000.0
_HANG_MS = 30000.0
_DASHBOARD_NOISY_TOOLS = {"get_robot_state"}


def _gripper_bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _fmt_vec(vals, decimals: int = 3) -> str:
    return "  ".join(f"{v:+.{decimals}f}" for v in vals)


def _arm_panel(title: str, state: Any | None) -> Panel:
    if state is None:
        return Panel("[dim]no data[/]", title=title, border_style="dim")
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=7)
    table.add_column()
    ee = getattr(state, "ee_pos", [])
    rpy = getattr(state, "ee_rpy", [])
    jp = getattr(state, "joint_pos", [])
    gp = float(getattr(state, "gripper_pos", 0.0))
    if ee:
        table.add_row("EE pos", _fmt_vec(ee))
    if rpy:
        table.add_row("RPY °", "  ".join(f"{v:+.1f}" for v in rpy))
    if jp:
        table.add_row("joints", "  ".join(f"{v:.2f}" for v in jp))
    table.add_row("gripper", f"[cyan]{_gripper_bar(gp)}[/] {gp:.2f}")
    return Panel(table, title=f"[bold]{title}[/]", border_style="blue")


def _tool_panel(current: dict[str, Any] | None, history, totals: dict[str, Any]) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column()

    # In-flight row mirrors RoboCasa's dashboard so slow/hung calls are visible.
    if current is None:
        live_line = Text("(idle)", style="dim")
        current_title = "in-flight"
        current_border = "blue"
    else:
        live_ms = (time.time() - float(current["t0"])) * 1000.0
        if live_ms > _HANG_MS:
            style = "bold red"
            current_border = "red"
            current_title = "in-flight ⚠ HANGING"
        elif live_ms > _SLOW_MS:
            style = "yellow"
            current_border = "yellow"
            current_title = "in-flight"
        else:
            style = "cyan"
            current_border = "blue"
            current_title = "in-flight"
        live_line = Text.assemble(
            ("▶ ", style),
            (f"#{current['call_id']} ", "dim"),
            (str(current["tool"]), f"bold {style}"),
            (f"({current.get('args', '')})", "dim"),
            ("  ", ""),
            (f"{live_ms:.0f}ms", style),
        )

    grid.add_row(Panel(live_line, title=current_title, border_style=current_border, padding=(0, 1)))

    tbl = Table(expand=True, box=None, show_header=True, header_style="dim")
    tbl.add_column("#", justify="right", width=4, style="dim")
    tbl.add_column("tool", overflow="crop", width=19)
    tbl.add_column("args", overflow="ellipsis", ratio=1)
    tbl.add_column("ms", justify="right", width=7)
    tbl.add_column("", justify="right", width=2)
    for h in reversed(list(history)):
        if h.get("error"):
            status = Text("✗", style="bold red")
            tool_style = "red"
        elif float(h.get("duration_ms", 0.0)) > _SLOW_MS:
            status = Text("✓", style="yellow")
            tool_style = "yellow"
        else:
            status = Text("✓", style="green")
            tool_style = "green"
        tbl.add_row(
            str(h.get("call_id", "")),
            Text(str(h.get("tool", "?")), style=tool_style),
            str(h.get("args", "")),
            f"{float(h.get('duration_ms', 0.0)):.0f}",
            status,
        )
    grid.add_row(tbl)

    footer = Text.assemble(
        ("calls=", "dim"),
        (str(int(totals.get("calls", 0))), "white"),
        ("  errors=", "dim"),
        (
            str(int(totals.get("errors", 0))),
            "red" if int(totals.get("errors", 0)) else "green",
        ),
        ("  total_tool_ms=", "dim"),
        (f"{float(totals.get('total_ms', 0.0)):.0f}", "white"),
    )
    grid.add_row(footer)
    return Panel(grid, title="[bold]Script tools[/]", border_style="magenta")


def _stdout_panel(lines: list[str]) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1, overflow="ellipsis")
    if not lines:
        table.add_row("[dim](no script output yet)[/]")
    else:
        for line in lines[-12:]:
            table.add_row(str(line))
    return Panel(table, title="[bold]Script output[/]", border_style="dim")


def _header_panel(paused: bool, stopping: bool, done: bool) -> Panel:
    # Status indicator
    if stopping:
        status = Text("⛔ STOPPING", style="bold red")
    elif done:
        status = Text("⏹  DONE", style="bold green")
    elif paused:
        status = Text("⏸  PAUSED", style="bold yellow")
    else:
        status = Text("▶  RUNNING", style="bold green")

    # Key hints change after script completes
    hint = Text()
    if done:
        hint.append("  [ENTER] ", style="bold white on green")
        hint.append(" go home & exit  ", style="dim")
        hint.append("  [S] ", style="bold white on dark_orange")
        hint.append(" skip home & exit  ", style="dim")
        hint.append("  [X] ", style="bold white on red")
        hint.append(" emergency stop", style="dim")
    else:
        hint.append("  [SPACE] ", style="bold white on dark_orange")
        hint.append(" pause/resume  ", style="dim")
        hint.append("  [X] ", style="bold white on red")
        hint.append(" stop & go home", style="dim")

    row = Table.grid(expand=True)
    row.add_column(ratio=1)
    row.add_column(justify="right")
    row.add_row(Text.assemble("YAM Real Robot  ", status), hint)
    return Panel(row, border_style="bold blue")


def _build_display(
    state: Any | None,
    paused: bool,
    stopping: bool,
    done: bool,
    current_tool: dict[str, Any] | None,
    tool_history,
    tool_totals: dict[str, Any],
    stdout_lines: list[str],
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="arms",   size=6),
        Layout(name="log",    size=16),
        Layout(name="stdout", size=12),
    )
    layout["arms"].split_row(
        Layout(name="left",  ratio=1),
        Layout(name="right", ratio=1),
    )
    layout["header"].update(_header_panel(paused, stopping, done))
    left  = getattr(state, "arms", {}).get("left")  if state else None
    right = getattr(state, "arms", {}).get("right") if state else None
    layout["left"].update(_arm_panel("LEFT ARM",  left))
    layout["right"].update(_arm_panel("RIGHT ARM", right))
    layout["log"].update(_tool_panel(current_tool, tool_history, tool_totals))
    layout["stdout"].update(_stdout_panel(stdout_lines))
    return layout


def _short_args(args: tuple, kwargs: dict, max_len: int = 70) -> str:
    parts = [_short_repr(a) for a in args]
    parts += [f"{k}={_short_repr(v)}" for k, v in kwargs.items()]
    s = ", ".join(parts)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _short_repr(value: Any) -> str:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return f"ndarray{tuple(value.shape)}"
    except ImportError:
        pass
    r = repr(value)
    return r if len(r) <= 40 else r[:39] + "…"


class YamDashboard:
    """Rich TUI + pynput keyboard listener for real YAM scripts.

    Lifecycle:
      start()       — launch render + keyboard threads
      await_exit()  — call after script finishes; blocks until user presses
                      ENTER (returns True → caller should go_home) or
                      S (returns False → skip home)
      stop()        — clean up threads
    """

    def __init__(self, env, get_robot_state: Callable):
        self._env = env
        self._get_state = get_robot_state
        self._render_done = threading.Event()   # signals render loop to stop
        self._done_flag = threading.Event()     # set when script finishes
        self._exit_event = threading.Event()    # set when user selects exit action
        self._should_go_home = True             # default: go home on exit
        self._last_state: Any = None
        self._render_thread: threading.Thread | None = None
        self._kb_listener = None
        self._pynput_ready = threading.Event()
        self._tty_thread: threading.Thread | None = None
        self._tty_stop = threading.Event()
        self._tty_ready = threading.Event()
        self._tty_fd: int | None = None
        self._tty_old_attrs = None
        self._tool_lock = threading.Lock()
        self._current_tool: dict[str, Any] | None = None
        self._tool_history: deque[dict[str, Any]] = deque(maxlen=_TOOL_HISTORY_LEN)
        self._tool_totals: dict[str, float | int] = {
            "calls": 0,
            "errors": 0,
            "total_ms": 0.0,
        }
        self._stdout_lock = threading.Lock()
        self._stdout_lines: deque[str] = deque(maxlen=8)
        self._stdout_partial = ""

    # ------------------------------------------------------------------
    # Profiler hooks — script tools only, not dashboard state polling
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        name: str,
        call_id: int,
        args: tuple,
        kwargs: dict,
    ) -> None:
        if name in _DASHBOARD_NOISY_TOOLS:
            return
        with self._tool_lock:
            self._current_tool = {
                "call_id": call_id,
                "tool": name,
                "args": _short_args(args, kwargs),
                "t0": time.time(),
            }

    def on_tool_end(
        self,
        name: str,
        call_id: int,
        result: Any,
        error: Exception | None,
        elapsed_ms: float,
    ) -> None:
        _ = result
        if name in _DASHBOARD_NOISY_TOOLS:
            return
        with self._tool_lock:
            args_str = ""
            if self._current_tool and self._current_tool.get("call_id") == call_id:
                args_str = str(self._current_tool.get("args", ""))
            self._current_tool = None
            self._tool_history.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "args": args_str,
                    "duration_ms": float(elapsed_ms),
                    "error": None
                    if error is None
                    else f"{type(error).__name__}: {error}",
                }
            )
            self._tool_totals["calls"] = int(self._tool_totals["calls"]) + 1
            self._tool_totals["total_ms"] = (
                float(self._tool_totals["total_ms"]) + float(elapsed_ms)
            )
            if error is not None:
                self._tool_totals["errors"] = int(self._tool_totals["errors"]) + 1

    def on_stdout(self, text: str) -> None:
        """Receive script stdout so quiet-mode messages are still visible."""
        if not text:
            return
        with self._stdout_lock:
            data = self._stdout_partial + str(text).replace("\r", "\n")
            parts = data.split("\n")
            self._stdout_partial = parts[-1]
            for line in parts[:-1]:
                line = line.rstrip()
                if line:
                    self._stdout_lines.append(line)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _handle_key(self, key_name: str) -> None:
        """Handle either a pynput key name or one char from terminal stdin."""
        key = key_name or ""
        key_lower = key.lower()

        if self._done_flag.is_set():
            # Post-script keys.
            if key in ("\n", "\r", "enter"):
                self._should_go_home = True
                self._exit_event.set()
            elif key_lower == "s":
                self._should_go_home = False
                self._exit_event.set()
            elif key_lower in ("x", "q", "\x03"):
                _stop_requested.set()
                self._should_go_home = False
                self._exit_event.set()
            return

        # Mid-script keys.
        if key in (" ", "space"):
            if _pause_requested.is_set():
                _pause_requested.clear()
            else:
                _pause_requested.set()
        elif key_lower in ("x", "q", "\x03"):
            if not _stop_requested.is_set():
                _stop_requested.set()

    def _on_key_press(self, key):
        from pynput import keyboard as _kb

        if key == _kb.Key.enter:
            self._handle_key("enter")
        elif key == _kb.Key.space:
            self._handle_key("space")
        elif hasattr(key, "char") and key.char:
            self._handle_key(str(key.char))

    def _start_terminal_input_listener(self) -> bool:
        """Start a tmux/SSH-friendly stdin key reader when pynput is unavailable."""
        import sys as _sys

        try:
            stream = _sys.__stdin__
            fd = stream.fileno()
            if not stream.isatty():
                return False
        except Exception:
            return False

        self._tty_fd = fd
        self._tty_stop.clear()
        self._tty_thread = threading.Thread(
            target=self._terminal_input_loop,
            daemon=True,
            name="yam-dashboard-tty-input",
        )
        self._tty_thread.start()
        return True

    def _terminal_input_loop(self) -> None:
        import os
        import select
        import sys as _sys
        import termios
        import tty

        fd = self._tty_fd
        if fd is None:
            return

        old_attrs = None
        try:
            old_attrs = termios.tcgetattr(fd)
            self._tty_old_attrs = old_attrs
            tty.setcbreak(fd)
            self._tty_ready.set()

            while not self._tty_stop.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if fd not in readable:
                    continue
                try:
                    ch = os.read(fd, 1).decode(errors="ignore")
                except OSError:
                    break
                if not ch:
                    continue
                # Ignore terminal escape-sequence prefix from arrow/function keys.
                if ch == "\x1b":
                    continue
                self._handle_key(ch)
        except Exception as exc:
            print(f"[YamDashboard] terminal key listener failed: {exc}", file=_sys.stderr)
        finally:
            self._tty_ready.clear()
            if old_attrs is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        import sys as _sys
        try:
            # force_terminal=True so Rich renders even in SSH / tmux sessions
            # where isatty() returns False.
            # Bind to the real terminal. run_script.py may replace
            # sys.stdout with a profiler tee shortly after this thread starts;
            # if Rich binds to that tee, the dashboard disappears into the log
            # instead of rendering in tmux.
            console = Console(
                file=_sys.__stdout__,
                force_terminal=True,
                force_jupyter=False,
            )
            with Live(console=console, refresh_per_second=_REFRESH_HZ,
                      screen=False, transient=False) as live:
                while not self._render_done.is_set():
                    try:
                        self._last_state = self._get_state()
                    except BaseException:
                        pass
                    with self._tool_lock:
                        current_tool = (
                            None
                            if self._current_tool is None
                            else dict(self._current_tool)
                        )
                        tool_history = list(self._tool_history)
                        tool_totals = dict(self._tool_totals)
                    with self._stdout_lock:
                        stdout_lines = list(self._stdout_lines)
                        if self._stdout_partial.strip():
                            stdout_lines.append(self._stdout_partial.strip())
                    live.update(_build_display(
                        self._last_state,
                        paused=_pause_requested.is_set(),
                        stopping=_stop_requested.is_set(),
                        done=self._done_flag.is_set(),
                        current_tool=current_tool,
                        tool_history=tool_history,
                        tool_totals=tool_totals,
                        stdout_lines=stdout_lines,
                    ))
                    time.sleep(_REFRESH_DT)
        except Exception as exc:
            print(f"[YamDashboard] render loop failed: {exc}", file=_sys.stderr)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        import sys as _sys
        print(
            "[YAM] Dashboard starting — SPACE pause/resume | X stop & go home",
            file=_sys.stderr,
        )
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="yam-dashboard"
        )
        self._render_thread.start()

        pynput_ok = False
        try:
            from pynput import keyboard as _kb
            self._kb_listener = _kb.Listener(on_press=self._on_key_press)
            self._kb_listener.start()
            pynput_ok = True
            self._pynput_ready.set()
        except Exception as exc:
            print(
                f"[YamDashboard] keyboard listener unavailable ({exc}); "
                "falling back to terminal stdin keys.",
                file=_sys.stderr,
            )
        if not pynput_ok:
            if self._start_terminal_input_listener():
                print(
                    "[YamDashboard] terminal keys active — SPACE pause/resume | "
                    "X stop | ENTER/S exit after done",
                    file=_sys.stderr,
                )
            else:
                print(
                    "[YamDashboard] terminal keys unavailable — use Ctrl+C to stop; "
                    "post-run prompt will read a line from stdin.",
                    file=_sys.stderr,
                )

    def await_exit(self) -> bool:
        """Mark script done, update display, block until user picks an exit action.

        Returns True if caller should call go_home() before exiting,
        False to skip go_home (user pressed S or emergency X).
        """
        import sys as _sys
        self._done_flag.set()
        print(
            "\n[YAM] Script done — ENTER go home & exit | S skip home | X emergency stop",
            file=_sys.stderr,
        )
        # If a keyboard listener (pynput or terminal stdin) is active, wait for
        # ENTER/S/X through that listener.  Only fall back to input() when no
        # listener could be started.
        if self._tty_ready.is_set() or self._pynput_ready.is_set():
            self._exit_event.wait()
        elif not self._exit_event.wait(timeout=0.1):
            _sys.stderr.write("[YAM] Waiting for input (pynput unavailable): ")
            _sys.stderr.flush()
            try:
                line = input().strip().lower()
            except (EOFError, OSError):
                line = ""
            if line in ("x", "q"):
                _stop_requested.set()
                self._should_go_home = False
            elif line == "s":
                self._should_go_home = False
            else:  # ENTER or anything else → go home
                self._should_go_home = True
            self._exit_event.set()
        return self._should_go_home

    def stop(self) -> None:
        self._render_done.set()
        self._tty_stop.set()
        if self._kb_listener is not None:
            self._kb_listener.stop()
            self._pynput_ready.clear()
        if self._tty_thread is not None:
            self._tty_thread.join(timeout=1.0)
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)

    def __enter__(self) -> "YamDashboard":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
