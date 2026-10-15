"""Background sampler that records a real-YAM episode in the raw deploy format.

Records measured states, commanded actions, camera frames, and timing.

**Why a thread and not a hook.** CAP control happens in many places -- programs
call ``step``, ``execute_trajectory``, ``set_gripper`` and ``go_home``, and the
planner and any takeover path command the arms too -- so there is no single
choke point to record at. Recording at command boundaries is what the harness
recorder already does, and on a real run it produces a slideshow: a measured
episode logged 9 frames across several minutes, because a 2.8 s descent is
*one* command and therefore one frame. This samples the plant on its own clock
instead, so a trajectory is captured as a trajectory.

Raw files written into ``out_dir``:

    {left,right}-joint_pos.npy      (N, 6)    measured
    {left,right}-gripper_pos.npy    (N, 1)    measured
    action-{left,right}-pos.npy     (N, 7)    commanded: joints(6) + gripper(1)
    action-fresh.npy                (N,)      bool: a NEW command this frame
    action-source.npy / .json       (N,)      what was driving
    timestamp.npy                   (N,)      seconds since start
    metadata.json
    {alias}-images-rgb.mp4                    one per camera

Sampling is best-effort by construction. Every read is wrapped, a failure is
counted and skipped rather than raised, and the thread is a daemon: recording
must never be able to stall or kill the robot loop.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np

from .action import ARM_DOF, ARMS

#: Sampling rate. Matches the station's 30 Hz caller tick, so a recorded frame
#: lines up with roughly one control decision and the video plays at the rate
#: the motion actually happened.
DEFAULT_FPS = 30

#: Optional per-arm channels, recorded when the arm server reports them and
#: NaN-padded when it does not. Padding rather than omitting keeps every channel
#: the same length as the episode, so a consumer can tell "not measured" from
#: "measured zero" instead of having to reconcile ragged arrays.
FORCE_CHANNELS: dict[str, int] = {
    "joint_vel": ARM_DOF,
    "joint_eff": ARM_DOF,
    "force_feedback_torque": ARM_DOF,
    "eef_force": 3,
}


class YamEpisodeRecorder:
    """Threaded sampler writing one real-YAM episode in the raw deploy layout."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        fps: int = DEFAULT_FPS,
        source: str = "cap_program",
        cameras: tuple[str, ...] | None = None,
        task: str = "",
        video_dir: str | Path | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        #: Where the mp4s go. Split from ``out_dir`` so a harness run can put the
        #: video where every other embodiment puts it -- media/videos -- while
        #: the arrays stay in the raw episode layout the converters expect.
        self.video_dir = Path(video_dir) if video_dir is not None else self.out_dir
        self.fps = int(fps)
        self.task = task
        #: None means "ask the env at start()", so a station with one camera
        #: records one and a station with three records three, without callers
        #: restating the station's own layout.
        self.cameras = cameras
        self.current_source = source

        self._env: Any = None
        self._observations: dict[str, list[np.ndarray]] = {
            f"{arm}-{channel}": [] for arm in ARMS for channel in ("joint_pos", "gripper_pos")
        }
        self._force: dict[str, list[np.ndarray]] = {
            f"{arm}-{channel}": [] for arm in ARMS for channel in FORCE_CHANNELS
        }
        self._actions: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
        self._source: list[str] = []
        self._fresh: list[bool] = []
        self._timestamps: list[float] = []
        self._previous_seq: dict[str, int] = dict.fromkeys(ARMS, -1)

        self._writers: dict[str, Any] = {}
        self._frames: dict[str, int] = {}
        self._started_at: float | None = None
        self._metadata: dict[str, Any] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_errors = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, env: Any, *, meta: dict[str, Any] | None = None) -> None:
        """Begin sampling ``env`` on a background thread."""
        self._env = env
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if self.cameras is None:
            self.cameras = tuple(getattr(env, "camera_aliases", ()) or ())
        self._reset_action_baseline(env)
        self._metadata = {
            "robot": "yam_bimanual",
            "control_mode": "joint_position",
            "task": self.task,
            "fps": self.fps,
            "cameras": list(self.cameras),
        }
        if meta:
            self._metadata.update(meta)
        # The plant the actions were executed under. Actions only transfer to a
        # deployment that reproduces it (same gains, same streaming rate), so
        # stamping it per episode turns an implicit convention into data.
        contract = getattr(env, "control_contract", None)
        if callable(contract):
            try:
                self._metadata["control_contract"] = contract()
            except Exception:
                pass

        self._started_at = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="yam-episode-recorder", daemon=True)
        self._thread.start()

    def _reset_action_baseline(self, env: Any) -> None:
        """Make frame 0 a hold at the current pose, not a leftover command.

        Without this the first frames carry whatever the previous episode last
        commanded, which reads as the policy driving somewhere before the
        episode began.
        """
        try:
            last_action = getattr(env, "_last_action_input", None)
            for arm in ARMS:
                observation = env.get_observations(arm)
                joints = np.asarray(observation["joint_pos"], dtype=np.float64).reshape(ARM_DOF)
                gripper = float(
                    np.asarray(observation["gripper_pos"], dtype=np.float64).reshape(-1)[0]
                )
                if isinstance(last_action, dict):
                    last_action[arm] = (joints.copy(), gripper)
            sequence = getattr(env, "_action_input_seq", None) or {}
            self._previous_seq = {arm: int(sequence.get(arm, 0)) for arm in ARMS}
        except Exception:
            pass

    def set_source(self, source: str) -> None:
        """Label subsequent frames, e.g. when a human takes over."""
        with self._lock:
            self.current_source = str(source)

    # -- sampling ----------------------------------------------------------

    def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.fps))
        while self._running:
            began = time.monotonic()
            try:
                self._sample()
            except Exception:
                self._sample_errors += 1
            remaining = period - (time.monotonic() - began)
            if remaining > 0:
                time.sleep(remaining)

    def _sample(self) -> None:
        env = self._env
        last_action = getattr(env, "_last_action_input", {}) or {}
        for arm in ARMS:
            observation = env.get_observations(arm)
            joints = np.asarray(observation["joint_pos"], dtype=np.float64).reshape(ARM_DOF)
            gripper = np.asarray(observation["gripper_pos"], dtype=np.float64).reshape(1)
            self._observations[f"{arm}-joint_pos"].append(joints)
            self._observations[f"{arm}-gripper_pos"].append(gripper)
            for channel, width in FORCE_CHANNELS.items():
                self._force[f"{arm}-{channel}"].append(
                    _fixed_width(observation.get(channel), width)
                )
            # The trainable action stream: what was commanded, falling back to a
            # hold at the measured pose when nothing has been commanded yet.
            commanded = last_action.get(arm)
            if commanded is None:
                self._actions[arm].append(np.concatenate([joints, gripper]))
            else:
                self._actions[arm].append(
                    np.concatenate(
                        [
                            np.asarray(commanded[0], dtype=np.float64).reshape(-1)[:ARM_DOF],
                            np.asarray([commanded[1]], dtype=np.float64).reshape(1),
                        ]
                    )
                )

        with self._lock:
            source = self.current_source
        self._source.append(source)

        sequence = getattr(env, "_action_input_seq", None) or {}
        fresh = False
        for arm in ARMS:
            current = int(sequence.get(arm, 0))
            if current != self._previous_seq.get(arm):
                fresh = True
            self._previous_seq[arm] = current
        self._fresh.append(fresh)
        self._timestamps.append(
            0.0 if self._started_at is None else time.monotonic() - self._started_at
        )

        for alias in self.cameras or ():
            frame = env.render_rgb(alias)
            if frame is not None:
                self._write_frame(alias, np.asarray(frame, dtype=np.uint8))

    def _write_frame(self, alias: str, frame: np.ndarray) -> None:
        if alias not in self._writers:
            self._writers[alias] = imageio.get_writer(
                self.video_dir / f"{alias}-images-rgb.mp4",
                fps=self.fps,
                codec="libx264",
                # Frames are whatever the camera produces (1280x720 here); do not
                # let the writer silently pad to a macroblock multiple.
                macro_block_size=None,
            )
            self._frames[alias] = 0
        self._writers[alias].append_data(frame)
        self._frames[alias] += 1

    # -- finish ------------------------------------------------------------

    def finalize(self, *, terminal_event: str | None = None, success: bool = False) -> Path:
        """Stop sampling and write the episode. Safe to call twice."""
        self._running = False
        if self._thread is not None:
            # Bounded: a sampler wedged on a plant read must not hold up the run.
            self._thread.join(timeout=2.0)
            self._thread = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._writers.clear()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        count = len(self._timestamps)
        # Every channel is truncated to the shortest, so the arrays are
        # rectangular even if the thread was stopped mid-sample.
        for name, rows in list(self._observations.items()) + list(self._force.items()):
            np.save(self.out_dir / f"{name}.npy", _stack(rows, count))
        for arm in ARMS:
            np.save(self.out_dir / f"action-{arm}-pos.npy", _stack(self._actions[arm], count))
        np.save(
            self.out_dir / "timestamp.npy", np.asarray(self._timestamps[:count], dtype=np.float64)
        )
        np.save(self.out_dir / "action-fresh.npy", np.asarray(self._fresh[:count], dtype=bool))
        np.save(self.out_dir / "action-source.npy", np.asarray(self._source[:count], dtype=object))

        self._metadata.update(
            {
                "frames": count,
                "duration_s": float(self._timestamps[count - 1]) if count else 0.0,
                "video_frames": dict(self._frames),
                "terminal_event": terminal_event,
                "success": bool(success),
                "sample_errors": self._sample_errors,
            }
        )
        (self.out_dir / "metadata.json").write_text(
            json.dumps(self._metadata, indent=2, default=str), encoding="utf-8"
        )
        (self.out_dir / "action-source.json").write_text(
            json.dumps(self._source[:count]), encoding="utf-8"
        )
        return self.out_dir


def _fixed_width(value: Any, width: int) -> np.ndarray:
    """``value`` as exactly ``width`` floats, NaN-padded when short or absent."""
    if value is None:
        return np.full(width, np.nan, dtype=np.float64)
    row = np.asarray(value, dtype=np.float64).reshape(-1)[:width]
    if row.size < width:
        return np.concatenate([row, np.full(width - row.size, np.nan, dtype=np.float64)])
    return row


def _stack(rows: list[np.ndarray], count: int) -> np.ndarray:
    """``rows`` truncated to ``count`` as a 2-D array, empty-safe."""
    if not rows or count <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(rows[:count]).astype(np.float64)
