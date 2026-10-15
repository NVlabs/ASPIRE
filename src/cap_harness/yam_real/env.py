"""The YAM plant: arms, camera, kinematics, and the one path that actuates.

``RealYamEnv`` is the native environment behind :class:`~cap_harness.yam_real.adapter.YamRealAdapter`,
the same way a MuJoCo env sits behind a simulator adapter. It owns
:meth:`execute_action_batch`, and that method is the **only** way anything in
this package commands the arms -- ``go_home`` and ``set_gripper`` are built on
it rather than beside it.

Keeping one chokepoint is what makes recorded episodes comparable. Two code
paths to the motors would be two plants, and data collected under one would not
transfer to the other even if both looked correct.

The arm clients are injected rather than constructed here, so the same env drives
either portal clients or the MuJoCo station in :mod:`cap_harness.yam_real.sim`.
The control path under test off-robot is then the real one.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from cap_harness.yam_real.action import YamActionBatch, resolve_action_batch
from cap_harness.yam_real.config import YamStationConfig
from cap_harness.yam_real.kinematics import ARM_DOF, YamKinematics

ARMS = ("left", "right")

#: Per-joint channels the arm server reports at full MOTOR width -- the six arm
#: joints plus the gripper. Only ``joint_pos`` arrives pre-sliced, so the env
#: splits the rest rather than exposing a ``joint_pos`` of length 6 beside a
#: ``joint_vel`` of length 7.
MOTOR_WIDTH_CHANNELS = ("joint_vel", "joint_eff")

#: Where each motor-width channel's gripper entry is republished.
GRIPPER_CHANNELS = {"joint_vel": "gripper_vel", "joint_eff": "gripper_eff"}


class RealYamEnv:
    """Bimanual YAM plant, driven through a single action chokepoint."""

    def __init__(
        self,
        config: YamStationConfig,
        arms: dict[str, Any],
        *,
        camera: Any | None = None,
        aux_cameras: dict[str, Any] | None = None,
        plant: Any | None = None,
        camera_marker: Any | None = None,
    ) -> None:
        """``plant`` is an owned physics backend, if any, stopped on close.

        Hardware has none -- the arm servers outlive this process, and closing a
        client must never stop them. A simulated station does, and would leak a
        stepping thread per env without this.

        ``camera_marker`` is an advisory note that this process holds the
        cameras, released last of all on close; see
        :mod:`cap_harness.yam_real.station_marker`.
        """
        if set(arms) != set(ARMS):
            raise ValueError("arms must contain exactly 'left' and 'right'")
        self.config = config
        self._arms = dict(arms)
        self._camera = camera
        self._aux_cameras = dict(aux_cameras or {})
        self._plant = plant
        self._camera_marker = camera_marker
        # Attachments that read this env and must stop before it does -- today
        # only the dashboard. Kept as callables so env.py needs no import of, and
        # no opinion about, whatever attached itself.
        self._close_hooks: list[Any] = []
        self.kin = YamKinematics(config.model_xml)
        # YamKinematics carries a mutable configuration, and both observation and
        # planning write to it. One lock, held across write-then-read.
        self._kin_lock = threading.Lock()
        self._closed = False
        # Most recent commanded target per arm, and a per-arm counter of how many
        # batches have been issued. Written by _note_action_input, read by the
        # episode recorder's sampler thread. Plain dicts: the writer replaces a
        # whole value per arm and the reader tolerates a stale one, so a lock
        # would only add a way for recording to stall actuation.
        self._last_action_input: dict[str, tuple[np.ndarray, float]] = {}
        self._action_input_seq: dict[str, int] = dict.fromkeys(ARMS, 0)

    # -- observation -------------------------------------------------------

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        """One arm's state: joints, gripper, and the end-effector pose from FK.

        FK is bimanual -- the model holds both arms -- so both arms are read even
        when only one side is asked for. Doing otherwise would evaluate one arm's
        pose against a stale configuration for the other.
        """
        if side not in ARMS:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        raw = {name: self._arms[name].get_observations() for name in ARMS}
        with self._kin_lock:
            left_pos, left_quat, right_pos, right_quat = self.kin.forward_kinematics(
                raw["left"]["joint_pos"], raw["right"]["joint_pos"]
            )
        poses = {"left": (left_pos, left_quat), "right": (right_pos, right_quat)}

        own = raw[side]
        out: dict[str, np.ndarray] = {
            "joint_pos": np.asarray(own["joint_pos"], dtype=np.float64).reshape(ARM_DOF),
            "gripper_pos": np.asarray(own["gripper_pos"], dtype=np.float64).reshape(1),
            "ee_pos": poses[side][0],
            "ee_quat": poses[side][1],
        }
        for key in MOTOR_WIDTH_CHANNELS:
            value = own.get(key)
            if value is None:
                continue
            channel = np.asarray(value, dtype=np.float64).reshape(-1)
            out[key] = channel[:ARM_DOF]
            if channel.size > ARM_DOF:
                out[GRIPPER_CHANNELS[key]] = channel[ARM_DOF : ARM_DOF + 1]
        return out

    def read_camera(self) -> Any | None:
        """Most recent frame from the station camera, or None when there is none."""
        return None if self._camera is None else self._camera.read()

    def read_aux_camera(self, role: str) -> Any | None:
        """Most recent frame from one image-only camera, or None."""
        source = self._aux_cameras.get(role)
        return None if source is None else source.read()

    @property
    def aux_camera_roles(self) -> tuple[str, ...]:
        """Roles of the cameras that are opened but not calibrated."""
        return tuple(sorted(self._aux_cameras))

    @property
    def camera_aliases(self) -> tuple[str, ...]:
        """Names ``render_rgb`` will answer to. Empty when the station has none.

        The calibrated camera comes first, then the image-only ones in profile
        order, so a caller that wants "the" camera and takes the first alias
        still gets the one that can be projected into metres.
        """
        names: list[str] = []
        if self._camera is not None:
            names.append(str(self.config.camera.role))
        names.extend(sorted(self._aux_cameras))
        return tuple(names)

    def attach_cameras(self, cameras: dict[str, Any], *, marker: Any | None = None) -> None:
        """Take ownership of already-opened cameras, keyed by role.

        The inverse of :meth:`release_cameras`, for a viewer that gives the
        devices up to a run and takes them back afterwards without dropping its
        arm connections. Refuses to overwrite cameras it already holds, since
        that would silently leak the pipelines.
        """
        if self._camera is not None or self._aux_cameras:
            raise RuntimeError("this station already holds cameras; release them first")
        remaining = dict(cameras)
        self._camera = remaining.pop(str(self.config.camera.role), None)
        self._aux_cameras = remaining
        self._camera_marker = marker

    def release_cameras(self) -> None:
        """Give up the cameras, keeping the arm connections.

        A RealSense device belongs to one process, so a courtesy viewer holding
        them has to let go when a run wants them -- but the arms are a
        client/server split and need no handover at all. Dropping the whole
        station to release a USB handle would blind the viewer to joints and
        poses for no reason.

        Afterwards ``camera_aliases`` is empty and ``render_rgb`` answers None,
        which is the same shape as a station built with ``enable_camera=False``.
        """
        sources = ([self._camera] if self._camera is not None else []) + list(
            self._aux_cameras.values()
        )
        self._camera = None
        self._aux_cameras = {}
        errors: list[str] = []
        for source in sources:
            try:
                source.close()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if self._camera_marker is not None:
            self._camera_marker.release()
            self._camera_marker = None
        if errors:
            raise RuntimeError("YAM camera release failed: " + "; ".join(errors))

    def add_close_hook(self, hook: Any) -> None:
        """Call `hook` when this station closes, before the devices go down.

        For things that read the env and must stop first. Hooks run in reverse
        registration order and their failures are printed, not raised.
        """
        self._close_hooks.append(hook)

    def camera_frame(self, alias: str) -> Any | None:
        """One camera's most recent frame by alias, or None.

        The frame-level twin of :meth:`render_rgb`, and total in the same way.
        A caller that only wants pixels should use ``render_rgb``; this exists
        for callers that also need ``timestamp_s`` -- notably
        :mod:`cap_harness.yam_real.dashboard`, which watches that field for
        *change* to tell a live camera from a wedged one still serving its last
        good frame.
        """
        if self._camera is not None and alias == str(self.config.camera.role):
            return self._camera.read()
        source = self._aux_cameras.get(alias)
        return None if source is None else source.read()

    def render_rgb(self, alias: str) -> np.ndarray | None:
        """One camera's RGB frame as ``(H, W, 3)`` uint8, or None.

        Exists for :class:`~cap_harness.yam_real.recorder.YamEpisodeRecorder`,
        which samples cameras by name on a background thread and must not raise:
        a station with no camera, an unknown alias, or a frame the driver has not
        produced yet all return None rather than killing the sampler.
        """
        frame = self.camera_frame(alias)
        if frame is None:
            return None
        return np.asarray(frame.rgb, dtype=np.uint8)

    # -- actuation ---------------------------------------------------------

    def command_joint_state(self, side: str, state: dict[str, Any]) -> None:
        """Forward one already-resolved tick to an arm server. Level-1 only."""
        self._arms[side].command_joint_state(state)

    def execute_action_batch(
        self,
        batch: YamActionBatch | dict[str, Any],
        *,
        command_hz: float | None = None,
        start_interp_s: float = 0.0,
        settle_s: float = 0.2,
        playback_speed: float = 1.0,
    ) -> dict[str, Any]:
        """Execute one typed action batch. The only path to the motors."""
        from cap_harness.yam_real.control.controller import execute_joint_trajectory

        resolved = resolve_action_batch(self, batch)
        self._note_action_input(resolved)
        rate = float(command_hz or self.config.controller.command_stream_hz)
        result = execute_joint_trajectory(
            self,
            resolved.timestamps,
            resolved.left_joint_positions,
            resolved.right_joint_positions,
            resolved.left_gripper_positions,
            resolved.right_gripper_positions,
            command_hz=rate,
            start_interp_s=start_interp_s,
            settle_s=settle_s,
            playback_speed=playback_speed,
        )
        result.setdefault("source", resolved.source)
        result.setdefault("input_space", resolved.input_space)
        return result

    def _note_action_input(self, resolved: Any) -> None:
        """Publish the batch's final target so a recorder can sample it.

        The episode recorder runs on its own thread and cannot see inside this
        call, so it reads the most recent commanded target from here. Without
        it, every recorded action is the arm's measured position -- an episode
        that says "the policy commanded exactly where the arm already was",
        which trains nothing.

        The batch's LAST waypoint is published, not each one in turn: the
        recorder samples at its own rate and only ever asks "what is the current
        target", and a batch is commanded as a unit.

        ``_action_input_seq`` increments per arm so the recorder can tell a genuinely
        new command from a zero-order hold. Perception and planning gaps are long
        here -- seven segmentation reads and a cuRobo round trip between motions
        -- so without it an episode looks like continuous actuation when most of
        it is the arm standing still.

        Deliberately total: a recorder attaching mid-run sees whatever the last
        batch left, and nothing here can fail a motion.
        """
        try:
            positions = {
                "left": resolved.left_joint_positions,
                "right": resolved.right_joint_positions,
            }
            grippers = {
                "left": resolved.left_gripper_positions,
                "right": resolved.right_gripper_positions,
            }
            for side in ARMS:
                rows = np.asarray(positions[side], dtype=np.float64)
                grip = np.asarray(grippers[side], dtype=np.float64).reshape(-1)
                if rows.size == 0:
                    continue
                self._last_action_input[side] = (
                    rows.reshape(-1, ARM_DOF)[-1].copy(),
                    float(grip[-1]) if grip.size else 0.0,
                )
                self._action_input_seq[side] = self._action_input_seq.get(side, 0) + 1
        except Exception:
            pass

    def set_gripper(
        self, side: str, position: float, *, duration_s: float | None = None
    ) -> dict[str, Any]:
        """Move one gripper while both arms hold their measured joint positions.

        The command is time-boxed, so the window has to cover the travel: a fixed
        short duration stops a large move partway. Measured on the bench, a 0.25 s
        window closed only 0.55 of a commanded 0.75 span. When ``duration_s`` is
        not given it is scaled from how far the jaws actually have to go.
        """
        if side not in ARMS:
            return {"success": False, "reason": f"invalid side: {side!r}"}
        observed = {name: self.get_observations(name) for name in ARMS}
        # Clamped for the same reason as the adapter's hold targets: a measured
        # pose can sit outside the commandable range, and commanding it verbatim
        # is refused, which would make a drifted arm impossible to command.
        joints = {
            name: np.clip(
                np.asarray(observed[name]["joint_pos"], dtype=np.float64).reshape(ARM_DOF),
                self.config.joint_limits_lower,
                self.config.joint_limits_upper,
            )
            for name in ARMS
        }
        start = {name: float(observed[name]["gripper_pos"][0]) for name in ARMS}
        goal = dict(start)
        goal[side] = float(np.clip(position, 0.0, 1.0))

        travel = abs(goal[side] - start[side])
        duration = (
            max(0.05, travel * self.config.controller.gripper_full_travel_s)
            if duration_s is None
            else max(0.02, float(duration_s))
        )

        batch = YamActionBatch.joint_abs(
            [0.0, duration],
            [joints["left"], joints["left"]],
            [joints["right"], joints["right"]],
            [[start["left"]], [goal["left"]]],
            [[start["right"]], [goal["right"]]],
            source="cap_gripper",
            meta={"side": side},
        )
        result = self.execute_action_batch(batch, settle_s=0.0)
        result.update({"side": side, "gripper": goal[side]})
        return result

    def go_ready(self, *, duration_s: float = 4.0, keep_grippers: bool = True) -> dict[str, Any]:
        """Drive both arms to the profile's ready pose.

        Kept as an explicit primitive, but no longer what ``reset`` uses.

        "Ready" is a misnomer worth knowing about before you call this: measured
        against home, it moves the grasp frame *forward* 0.11 m and *inward*
        0.16 m, parking the arm over the table rather than clear of it, with the
        forearm at z 1.23. That height is inside the band where the station
        camera returns arm hardware the URDF does not model, which makes the
        start state read as in-collision to a scene-aware planner.

        Home plans fine too. An earlier version of this docstring claimed home was
        a singularity IK could not solve from; that was wrong -- the failure was a
        stale self-collision ignore map, since fixed. See BUGS.md.
        """
        return self._drive_to(
            {name: self.config.arms[name].ready_joints for name in ARMS},
            duration_s=duration_s,
            keep_grippers=keep_grippers,
            source="cap_ready",
        )

    def go_home(self, *, duration_s: float = 3.0, keep_grippers: bool = True) -> dict[str, Any]:
        """Drive both arms to the profile's home pose -- the mechanical zero.

        Home is the parking and calibration reference, and what ``reset`` drives
        to. cuRobo IK solves both arms from here.

        Two properties to know before commanding it. Joints 2 and 3 bottom out at
        exactly 0.0, so at home they sit *on* their lower limits -- a planner
        cannot seed below them, only away from them. And the arm is fully
        extended, putting the grasp frame at roughly x 0.50, z 0.91: reaching
        over the table about 13 cm above its surface, so the path there sweeps
        the workspace and anything left on the table is in it.
        """
        return self._drive_to(
            {name: self.config.arms[name].home_joints for name in ARMS},
            duration_s=duration_s,
            keep_grippers=keep_grippers,
            source="cap_home",
        )

    def _drive_to(
        self,
        targets: dict[str, np.ndarray],
        *,
        duration_s: float,
        keep_grippers: bool,
        source: str,
    ) -> dict[str, Any]:
        """One command with a start interpolation, not a settling loop.

        The controller drives smoothly to the target and arrives; re-commanding
        the goal one control period at a time creeps instead of converging.
        """
        observed = {name: self.get_observations(name) for name in ARMS}
        grippers = {
            name: (float(observed[name]["gripper_pos"][0]) if keep_grippers else 0.0)
            for name in ARMS
        }
        batch = YamActionBatch.joint_abs(
            [0.0],
            [targets["left"]],
            [targets["right"]],
            [[grippers["left"]]],
            [[grippers["right"]]],
            source=source,
        )
        return self.execute_action_batch(
            batch, start_interp_s=max(0.0, float(duration_s)), settle_s=0.2
        )

    # -- lifecycle ---------------------------------------------------------

    def control_contract(self) -> dict[str, Any]:
        """The versioned plant description recorded actions were executed under."""
        from cap_harness.yam_real.control.controller import build_control_contract

        return build_control_contract(self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        # Hooks first: an attachment that samples cameras must stop before the
        # devices do. Their failures are not the station's -- a web server that
        # will not shut down cleanly is no reason to report that the robot did
        # not -- so they are printed, not collected into `errors`.
        for hook in reversed(self._close_hooks):
            try:
                hook()
            except Exception as exc:
                print(f"[yam-station] close hook failed: {type(exc).__name__}: {exc}")
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception as exc:
                errors.append(f"camera: {type(exc).__name__}: {exc}")
        # Every aux camera gets a close attempt even if an earlier one raises:
        # one wedged device must not leak the rest of the USB handles.
        for role, aux in self._aux_cameras.items():
            try:
                aux.close()
            except Exception as exc:
                errors.append(f"camera {role}: {type(exc).__name__}: {exc}")
        for name, arm in self._arms.items():
            close = getattr(arm, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if self._plant is not None:
            try:
                self._plant.stop()
            except Exception as exc:
                errors.append(f"plant: {type(exc).__name__}: {exc}")
        # Last of all, and after the devices are actually free: the marker says
        # the cameras are held, so clearing it before they are released would
        # invite a waiting viewer in while they are still ours.
        if self._camera_marker is not None:
            self._camera_marker.release()
        if errors:
            raise RuntimeError("YAM station cleanup failed: " + "; ".join(errors))


__all__ = ["ARMS", "RealYamEnv"]
