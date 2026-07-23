# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive, whitelisted real-YAM command console.

This is for supervised bench-side checks. It intentionally does not expose
arbitrary Python exec/eval; every motion command is explicit and requires
typing ``arm`` first.
"""

from __future__ import annotations

import shlex
import time
import builtins
from functools import partial
from typing import Iterable

from skill_library.namespace import (
    close_gripper,
    get_robot_state,
    go_home,
    open_gripper,
    set_gripper,
)


print = partial(builtins.print, flush=True)


HELP = """
commands:
  help                         show this help
  state                        print current joints/grippers
  arm                          enable motion commands
  disarm                       disable motion commands
  home                         move both arms to home/zero pose
  open left|right|both          open gripper(s)
  close left|right|both         close gripper(s)
  grip left|right|both POS      set gripper position, POS in [0, 1]
  demo-grippers                 open/close both grippers once
  quit                         exit

notes:
  - motion commands are blocked until you type: arm
  - use the physical E-stop for any unsafe motion
  - this console does not run arbitrary Python
""".strip()


def get_task_info():
    return {
        "success": True,
        "reward": 0.0,
        "method": "manual_realtime_console",
    }


def _sides(token: str) -> list[str]:
    side = token.strip().lower()
    if side == "both":
        return ["left", "right"]
    if side in {"left", "right"}:
        return [side]
    raise ValueError("side must be left, right, or both")


def _require_armed(armed: bool) -> None:
    if not armed:
        raise RuntimeError("motion blocked; type 'arm' first")


def _print_state() -> None:
    state = get_robot_state()
    print("[manual_console] robot state:")
    for side in ("left", "right"):
        arm = state.arms.get(side)
        if arm is None:
            print(f"  {side}: unavailable")
            continue
        joints = " ".join(f"{float(x):+.4f}" for x in arm.joint_pos[:6])
        ee = " ".join(f"{float(x):+.3f}" for x in arm.ee_pos)
        print(
            f"  {side}: joints=[{joints}] "
            f"gripper={float(arm.gripper_pos):.3f} ee=[{ee}]"
        )


def _for_each_side(command: str, sides: Iterable[str], fn) -> None:
    for side in sides:
        print(f"[manual_console] {command} {side} ...")
        result = fn(side)
        print(f"[manual_console] {command} {side} -> {result}")


print("[manual_console] Real-time YAM console started.")
print(HELP)
print("")
_print_state()

armed = False

while True:
    try:
        print("yam> ", end="")
        line = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[manual_console] exiting")
        break

    if not line:
        continue

    try:
        parts = shlex.split(line)
    except ValueError as exc:
        print(f"[manual_console] parse error: {exc}")
        continue

    cmd = parts[0].lower()

    try:
        if cmd in {"help", "h", "?"}:
            print(HELP)
        elif cmd in {"quit", "exit", "q"}:
            print("[manual_console] exiting")
            break
        elif cmd == "state":
            _print_state()
        elif cmd == "arm":
            armed = True
            print("[manual_console] armed: motion commands enabled")
        elif cmd == "disarm":
            armed = False
            print("[manual_console] disarmed: motion commands blocked")
        elif cmd == "home":
            _require_armed(armed)
            print("[manual_console] go_home ...")
            result = go_home()
            print(f"[manual_console] go_home -> {result}")
            _print_state()
        elif cmd == "open":
            _require_armed(armed)
            if len(parts) != 2:
                raise ValueError("usage: open left|right|both")
            _for_each_side("open", _sides(parts[1]), open_gripper)
        elif cmd == "close":
            _require_armed(armed)
            if len(parts) != 2:
                raise ValueError("usage: close left|right|both")
            _for_each_side("close", _sides(parts[1]), close_gripper)
        elif cmd == "grip":
            _require_armed(armed)
            if len(parts) != 3:
                raise ValueError("usage: grip left|right|both POS")
            pos = float(parts[2])
            if not 0.0 <= pos <= 1.0:
                raise ValueError("POS must be in [0, 1]")
            _for_each_side(
                f"grip {pos:.3f}",
                _sides(parts[1]),
                lambda side: set_gripper(side, pos),
            )
        elif cmd == "demo-grippers":
            _require_armed(armed)
            for fn, name in [
                (open_gripper, "open"),
                (close_gripper, "close"),
                (open_gripper, "open"),
            ]:
                _for_each_side(name, ["left", "right"], fn)
                time.sleep(0.5)
        else:
            print(f"[manual_console] unknown command: {cmd!r}. Type 'help'.")
    except Exception as exc:
        print(f"[manual_console] ERROR: {type(exc).__name__}: {exc}")

print("[manual_console] done.")
