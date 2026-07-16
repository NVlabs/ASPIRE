"""Live motor temperature monitor for robot arm servers.

Queries arm servers over Portal RPC (`get_motor_temperatures`) and prints a
continuously updating table suitable for a dedicated tmux window.
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import portal
import tyro

from robot.constants import (
    LEFT_FOLLOWER_PORT,
    LEFT_LEADER_PORT,
    RIGHT_FOLLOWER_PORT,
    RIGHT_LEADER_PORT,
)


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    refresh_hz: float = 2.0
    rpc_timeout_s: float = 0.5


def _format_temp_row(values: np.ndarray | None) -> str:
    if values is None or values.size == 0:
        return "-"
    parts: list[str] = []
    for v in values:
        if np.isnan(v):
            parts.append("  nan")
        else:
            parts.append(f"{float(v):5.1f}")
    return " ".join(parts)


def _max_temp(values: np.ndarray | None) -> str:
    if values is None or values.size == 0:
        return "-"
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return "-"
    return f"{float(np.max(valid)):.1f}"


def main(args: Args) -> None:
    endpoints = [
        ("follower_left", LEFT_FOLLOWER_PORT),
        ("follower_right", RIGHT_FOLLOWER_PORT),
        ("leader_left", LEFT_LEADER_PORT),
        ("leader_right", RIGHT_LEADER_PORT),
    ]
    clients: dict[str, portal.Client] = {
        name: portal.Client(f"{args.host}:{port}") for name, port in endpoints
    }
    period = 1.0 / max(args.refresh_hz, 0.2)

    while True:
        rows: list[tuple[str, str, str, str]] = []
        for name, _ in endpoints:
            try:
                data = clients[name].get_motor_temperatures().result(timeout=args.rpc_timeout_s)
                arr = np.asarray(data, dtype=np.float32).ravel()
                rows.append((name, "ok", _format_temp_row(arr), _max_temp(arr)))
            except Exception as e:
                rows.append((name, f"down ({type(e).__name__})", "-", "-"))

        # Clear and redraw
        os.system("clear")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[motor_temps] {now}  host={args.host}\n")
        print(f"{'arm':<16} {'status':<24} {'temps (C)':<56} {'max':>6}")
        print("-" * 108)
        for name, status, temps, max_t in rows:
            print(f"{name:<16} {status:<24} {temps:<56} {max_t:>6}")
        print("\nCtrl+C to stop.")
        time.sleep(period)


if __name__ == "__main__":
    main(tyro.cli(Args))
