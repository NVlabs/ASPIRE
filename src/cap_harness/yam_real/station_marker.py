"""A note saying which process currently holds a station's cameras.

A RealSense device belongs to one process: ``pipeline.start`` claims it and the
next process to try fails. That is fine when only runs open cameras, but the
dashboard makes it likely that something is already watching when a run starts.

So a station owner leaves a note. It is **advisory in one direction only**: a run
writes the marker and never reads one, never waits, and never fails because of
one. Only a courtesy viewer reads it, and yields.

That asymmetry is the whole design. It means this file's absence is
indistinguishable from success -- delete it, fill the disk, mount ``/tmp``
read-only, and every run behaves exactly as it does today. Nothing has to be
running for the harness to work, which is the property that a camera-owner
service would have taken away. ``YamCameraConfig`` in
:mod:`cap_harness.yam_real.config` records why one was removed.

A crashed owner leaves its note behind, so a reader treats a marker whose process
is gone as no marker at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time


def marker_dir(root: str | Path | None = None) -> Path:
    """Where markers live: the user's runtime dir, or a temp dir beside it.

    Never inside a run's output tree -- ``artifacts`` seals a manifest over
    whatever it finds there, and a runtime file appearing mid-run would land in
    it.
    """
    if root is not None:
        return Path(root)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(tempfile.gettempdir())
    return base / "cap-harness"


def marker_path(station: str, *, root: str | Path | None = None) -> Path:
    return marker_dir(root) / f"yam-{station}-cameras.json"


@dataclass(frozen=True, slots=True)
class MarkerInfo:
    """Who says they hold the cameras."""

    pid: int
    station: str
    owner: str
    since_epoch_s: float


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, but a live one.
        return True
    except OSError:
        return False
    return True


def read_marker(station: str, *, root: str | Path | None = None) -> MarkerInfo | None:
    """Who holds `station`'s cameras, or None -- including when nobody really does.

    A marker naming a process that has exited reads as None. That is what makes
    a crashed run self-healing rather than something a human has to clean up.
    """
    path = marker_path(station, root=root)
    try:
        payload = json.loads(path.read_text())
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not _process_is_alive(pid):
        return None
    return MarkerInfo(
        pid=pid,
        station=str(payload.get("station", station)),
        owner=str(payload.get("owner", "unknown")),
        since_epoch_s=float(payload.get("since_epoch_s", 0.0)),
    )


class CameraMarker:
    """A claim this process has written. Release it when the cameras go down."""

    def __init__(self, path: Path, pid: int) -> None:
        self.path = path
        self.pid = pid

    def release(self) -> None:
        """Remove the marker, if it is still ours. Idempotent, never raises.

        The ownership check matters because a run overwrites whatever it finds:
        without it, a run's release would delete the marker of whoever came
        next.
        """
        try:
            payload = json.loads(self.path.read_text())
            if int(payload.get("pid", -1)) == self.pid:
                self.path.unlink()
        except (OSError, ValueError, TypeError):
            return


def claim(station: str, *, owner: str, root: str | Path | None = None) -> CameraMarker | None:
    """Announce that this process is taking `station`'s cameras.

    Always succeeds if it can write at all, overwriting any existing marker: the
    caller is a station owner, and station owners do not queue. Returns None when
    the note cannot be written, which is not a failure -- it only means nobody
    will be told.
    """
    path = marker_path(station, root=root)
    payload = {
        "pid": os.getpid(),
        "station": station,
        "owner": owner,
        "since_epoch_s": time.time(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(json.dumps(payload))
    except OSError:
        return None
    return CameraMarker(path, os.getpid())


__all__ = [
    "CameraMarker",
    "MarkerInfo",
    "claim",
    "marker_dir",
    "marker_path",
    "read_marker",
]
