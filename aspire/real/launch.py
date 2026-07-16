from dataclasses import dataclass
import subprocess
from typing import Literal
import time

import tyro


class TmuxSession:
    def __init__(self, session: str):
        self.session = session
        self._first_window_done = False

        # Kill old session if it exists
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Start new session
        subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True)

        # Bind Ctrl-\ to kill tmux server
        subprocess.run(["tmux", "bind-key", "-n", "C-\\", "kill-server"], check=True)

        # Show exit instructions in status bar
        subprocess.run(
            [
                "tmux",
                "set-option",
                "-t",
                session,
                "status-right",
                "Press Ctrl+\\ to exit",
            ],
            check=True,
        )

    def new_window(self, name=None, command=None):
        # Create window
        args = ["tmux", "new-window", "-P", "-F", "#{window_id}", "-t", self.session]
        if name:
            args += ["-n", name]
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        window_id = result.stdout.strip()

        # Replace the default :0 window
        if not self._first_window_done:
            subprocess.run(
                ["tmux", "kill-window", "-t", f"{self.session}:0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["tmux", "move-window", "-s", window_id, "-t", f"{self.session}:0"],
                check=True,
            )
            self._first_window_done = True

        # If a command is given, send it to the window
        if command:
            send_args = ["tmux", "send-keys", "-t", window_id, command, "C-m"]
            last_err = ""
            for attempt in range(5):
                proc = subprocess.run(send_args, capture_output=True, text=True)
                if proc.returncode == 0:
                    break
                last_err = (proc.stderr or proc.stdout).strip()
                if attempt < 4:
                    time.sleep(0.05)
                    continue
                raise RuntimeError(f"tmux send-keys failed for window '{name}': {last_err}")

    def attach(self):
        subprocess.run(["tmux", "attach", "-t", self.session])


@dataclass
class Args:
    mode: Literal["evaluation"] = "evaluation"

    attach: bool = True
    """Whether to attach to the main session."""


def main(args: Args):
    robots_session = TmuxSession("robots")
    robots_session.new_window("follow_l", "uv run robot/yam/arm_server.py --mode follower --side left")
    robots_session.new_window("follow_r", "uv run robot/yam/arm_server.py --mode follower --side right")

    # Live motor temperature table (followers + leaders)
    robots_session.new_window("motor_temps", "uv run robot/monitor_motor_temps.py")

    # Start camera servers
    cameras_session = TmuxSession("cameras")
    cameras_session.new_window("top", "")
    cameras_session.new_window("left", "")
    cameras_session.new_window("right", "")

    # Start main loop
    main_session = TmuxSession("main")
    main_session.new_window("main", "echo 'Run your evaluation script here:'")

    # Attach to main session
    if args.attach:
        main_session.attach()


if __name__ == "__main__":
    main(tyro.cli(Args))
