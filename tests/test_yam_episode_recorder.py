"""The episode recorder samples continuously, not once per command.

The bug these cover: the harness recorder writes one frame per commanded
action, so a real run produced a 10-frame video across several minutes. A
recorder that only sampled at command boundaries would pass a "did it write a
file" test and still be useless, so these assert on frame COUNT against elapsed
time rather than on existence.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cap_harness.yam_real.recorder import YamEpisodeRecorder
from cap_harness.yam_real.station import build_sim_station


def _station():
    try:
        return build_sim_station(with_camera=True, realtime=True)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        pytest.skip(f"synthetic station unavailable: {exc}")


def test_samples_continuously_while_nothing_is_commanded(tmp_path):
    """Frames accrue from elapsed time alone, with no commands at all.

    This is the whole point of the port. Under the old command-boundary
    recording, an idle second produced zero frames.
    """
    env = _station()
    recorder = YamEpisodeRecorder(tmp_path / "raw", fps=30)
    try:
        recorder.start(env)
        time.sleep(0.5)
        recorder.finalize()
    finally:
        env.close()

    joints = np.load(tmp_path / "raw" / "left-joint_pos.npy")
    # 0.5 s at 30 fps is ~15 frames; allow a wide band for scheduling jitter and
    # slow CI, but insist it is many-per-second rather than one-per-command.
    assert joints.shape[0] >= 5, f"only {joints.shape[0]} frames in 0.5 s"
    assert joints.shape[1] == 6


def test_every_channel_has_the_same_length(tmp_path):
    """Ragged channels would break the LeRobot conversion downstream."""
    env = _station()
    recorder = YamEpisodeRecorder(tmp_path / "raw", fps=30)
    try:
        recorder.start(env)
        time.sleep(0.3)
        recorder.finalize()
    finally:
        env.close()

    raw = tmp_path / "raw"
    count = np.load(raw / "timestamp.npy").shape[0]
    for name in (
        "left-joint_pos",
        "left-gripper_pos",
        "right-joint_pos",
        "right-gripper_pos",
        "action-left-pos",
        "action-right-pos",
        "action-fresh",
        "action-source",
    ):
        assert np.load(raw / f"{name}.npy", allow_pickle=True).shape[0] == count, name


def test_action_stream_follows_commands_not_measurements(tmp_path):
    """A commanded target must show up in the action stream.

    Without ``_note_action_input`` every recorded action is the arm's own
    measured pose -- an episode asserting the policy commanded exactly where the
    arm already was, which trains nothing.
    """
    env = _station()
    recorder = YamEpisodeRecorder(tmp_path / "raw", fps=30)
    try:
        recorder.start(env)
        time.sleep(0.1)
        target = np.full(6, 0.25)
        env._last_action_input["left"] = (target, 0.75)
        env._action_input_seq["left"] = env._action_input_seq.get("left", 0) + 1
        time.sleep(0.2)
        recorder.finalize()
    finally:
        env.close()

    actions = np.load(tmp_path / "raw" / "action-left-pos.npy")
    assert actions.shape[1] == 7
    assert np.isclose(actions[-1][:6], 0.25).all(), actions[-1]
    assert np.isclose(actions[-1][6], 0.75)
    # The freshness mask must have flagged the new command.
    assert np.load(tmp_path / "raw" / "action-fresh.npy").any()


def test_finalize_is_idempotent_and_survives_a_dead_env(tmp_path):
    """Recording must never be able to fail a run."""
    env = _station()
    recorder = YamEpisodeRecorder(tmp_path / "raw", fps=30)
    recorder.start(env)
    time.sleep(0.1)
    env.close()  # sampler now reads a closed plant
    time.sleep(0.1)
    recorder.finalize()
    recorder.finalize()  # second call must not raise
    assert (tmp_path / "raw" / "metadata.json").is_file()


def test_unknown_camera_alias_does_not_kill_the_sampler(tmp_path):
    env = _station()
    recorder = YamEpisodeRecorder(tmp_path / "raw", fps=30, cameras=("nonexistent",))
    try:
        recorder.start(env)
        time.sleep(0.2)
        recorder.finalize()
    finally:
        env.close()
    assert np.load(tmp_path / "raw" / "timestamp.npy").shape[0] > 0
    assert not list((tmp_path / "raw").glob("*.mp4"))
