"""Camera acquisition timestamps share the runtime's monotonic clock."""

from types import SimpleNamespace

import numpy as np

from cap_harness.yam_real.cameras import RealSenseCamera


def test_device_clock_does_not_make_cached_images_look_fresh(monkeypatch):
    intrinsics = SimpleNamespace(width=2, height=2, fx=1, fy=1, ppx=1, ppy=1)
    colour = SimpleNamespace(
        profile=SimpleNamespace(
            as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intrinsics)
        ),
        get_data=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    frames = SimpleNamespace(
        get_color_frame=lambda: colour,
        get_depth_frame=lambda: None,
        get_timestamp=lambda: 1_789_706_248_000.0,
    )
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera._pipeline = SimpleNamespace(wait_for_frames=lambda **kwargs: frames)
    camera._align = SimpleNamespace(process=lambda value: value)
    monkeypatch.setattr("cap_harness.yam_real.cameras.time.monotonic", lambda: 100.0)
    frame = camera.read()
    assert frame.timestamp_s == 100.0
    # A two-second-old cached frame is stale, even if the sensor uses epoch time.
    assert (102.0 - frame.timestamp_s) * 1000 > 1000
