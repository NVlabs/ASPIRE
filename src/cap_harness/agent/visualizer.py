"""Live robot model, camera views, and Home control for the existing CAP session."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image
import viser
from viser.extras import ViserUrdf

from .session import LiveAgentSession, SessionBusyError

logger = logging.getLogger(__name__)


class System2Visualizer:
    """Read the session cache and submit Home through its serialized job queue."""

    def __init__(
        self,
        session: LiveAgentSession,
        *,
        allow_motion: bool = False,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self._session = session
        self._allow_motion = allow_motion
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._images: dict[str, Any] = {}
        self._image_sequences: dict[str, Any] = {}
        self._home_job_id: str | None = None
        self._notice = ""
        self.server = viser.ViserServer(host=host, port=port, label="ASPIRE System 2")
        if self.server.get_port() != port:
            self.server.stop()
            raise RuntimeError(f"Visualizer port {port} is occupied; choose --viser-port.")
        self.server.scene.set_up_direction("+z")
        self._robot_root = self.server.scene.add_frame("/robot", show_axes=False, visible=False)
        model_path = (
            Path(__file__).resolve().parents[1] / "yam_real/description/station/station.urdf"
        )
        try:
            self._robot = ViserUrdf(self.server, model_path, root_node_name="/robot")
        except BaseException:
            self.server.stop()
            raise
        self._joint_limits = self._robot.get_actuated_joint_limits()
        self.server.on_client_connect(self._set_initial_view)
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            show_logo=False,
            show_share_button=False,
        )
        self._home = self.server.gui.add_button(
            "Go home",
            disabled=not allow_motion,
            hint="Open both grippers and return both arms to their configured home pose.",
        )
        self._status = self.server.gui.add_markdown("Waiting for camera frames.")
        self._home.on_click(self._on_home)
        self._thread = threading.Thread(target=self._run, name="cap-viser", daemon=True)
        self._thread.start()

    @staticmethod
    def _set_initial_view(client: viser.ClientHandle) -> None:
        client.camera.position = (1.8, -1.6, 1.6)
        client.camera.look_at = (0.4, 0.0, 0.8)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def _refresh_robot(self, snapshot: dict[str, Any]) -> bool:
        """Map measured telemetry onto the packaged model without querying hardware."""
        state = snapshot.get("state", {})
        values: dict[str, float] = {}
        age_s = snapshot.get("age_s")
        if snapshot.get("monitor_error") or (age_s is not None and age_s > 1.0):
            self._robot_root.visible = False
            return False
        for side in ("left", "right"):
            joints = np.asarray(state.get(f"{side}_joint_pos", []), dtype=float).reshape(-1)
            gripper = np.asarray(state.get(f"{side}_gripper_pos", []), dtype=float).reshape(-1)
            if (
                joints.size != 6
                or gripper.size != 1
                or not np.all(np.isfinite(joints))
                or not np.all(np.isfinite(gripper))
            ):
                self._robot_root.visible = False
                return False
            values.update({f"{side}_joint{i}": float(q) for i, q in enumerate(joints, 1)})
            opening = float(np.clip(gripper[0], 0.0, 1.0))
            for finger, bound in (("left", 1), ("right", 0)):
                name = f"{side}_{finger}_finger_joint"
                values[name] = opening * self._joint_limits[name][bound]
        with self.server.atomic():
            self._robot.update_cfg(np.asarray([values[name] for name in self._joint_limits]))
            self._robot_root.visible = True
        return True

    def _on_home(self, _event: Any) -> None:
        with self._lock:
            if self._stop.is_set() or not self._allow_motion:
                return
            try:
                self._home_job_id = self._session.start_named_program("go_home")
            except (SessionBusyError, RuntimeError) as exc:
                self._notice = str(exc)
            else:
                self._notice = ""
                self._home.disabled = True
                self._session.steer(
                    "The operator requested Home from the visualizer. "
                    "The Home job ends the current task; wait for a new operator task."
                )

    def _refresh(self) -> None:
        snapshot = self._session.cached_visual_snapshot()
        cameras = snapshot["cameras"]
        metadata = snapshot["camera_metadata"]
        names = dict.fromkeys((*metadata, *cameras))
        job = self._session.job()
        with self._lock:
            robot_live = self._refresh_robot(snapshot)
            self._home.disabled = (
                not self._allow_motion or self._stop.is_set() or bool(job and job["active"])
            )
            for name in self._images.keys() - names.keys():
                self._images.pop(name).remove()
                self._image_sequences.pop(name, None)
            for name in names:
                health = metadata.get(name, {})
                missing = bool(health.get("missing")) or name not in cameras
                stale = bool(health.get("stale")) or bool(snapshot.get("monitor_error"))
                label = f"{name} — missing" if missing else f"{name} — stale" if stale else name
                if name not in self._images:
                    self._images[name] = self.server.gui.add_image(
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        label=label,
                        format="jpeg",
                        jpeg_quality=80,
                    )
                handle = self._images[name]
                handle.label = label
                sequence = (health.get("sequence", snapshot.get("observation_seq")), missing)
                if sequence != self._image_sequences.get(name):
                    if missing:
                        handle.image = np.zeros((1, 1, 3), dtype=np.uint8)
                    else:
                        image = Image.fromarray(cameras[name])
                        image.thumbnail((640, 480))
                        handle.image = np.asarray(image)
                    self._image_sequences[name] = sequence
            status = "Live camera views." if cameras else "Waiting for camera frames."
            if not robot_live:
                status += " Robot model hidden: waiting for fresh joint state."
            if not self._allow_motion:
                status += " Home is disabled: observation-only server."
            elif job and job["active"]:
                status += " Motion in progress; Home is available when it finishes."
            elif job and job["id"] == self._home_job_id:
                summary = job.get("result") or {}
                if job["error"] or not summary.get("ok"):
                    status += " Home failed."
                else:
                    diagnostics = summary.get("result", {}).get("diagnostics", {})
                    verified = diagnostics.get("home_verified")
                    opened = diagnostics.get("grippers_open", {})
                    status += f" Home complete. Home verified: {verified}."
                    if not all(opened.values()) or not opened:
                        status += " A gripper is not open."
            if snapshot.get("monitor_error"):
                status += " Camera monitoring is unavailable; displayed frames are stale."
            if self._notice:
                status += f" {self._notice}"
            self._status.content = status

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            try:
                self._refresh()
            except Exception:
                logger.exception("System 2 visualizer update failed")

    def close(self) -> None:
        """Stop the viewer without closing the shared robot session."""
        self._stop.set()
        self._thread.join()
        self.server.stop()
