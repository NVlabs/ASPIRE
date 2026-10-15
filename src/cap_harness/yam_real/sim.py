"""MuJoCo stand-in for the YAM arm servers, so the real path runs off-robot.

:class:`SimYamArm` duck-types the portal follower client: same
``get_observations`` / ``command_joint_state`` / ``command_joint_pos`` surface,
same observation keys. Substituting it into :class:`~cap_harness.yam_real.env.RealYamEnv`
exercises the *entire* control path -- action batch, level-1 streaming loop,
settle, residual check -- against a physical plant rather than a stub that agrees
with whatever it is told.

That distinction is the point. A fake that echoes commanded positions back as
measured positions can never fail the trajectory residual check, so testing
against one would prove nothing about the check that exists to catch an arm
falling behind. Here the arms have mass and finite gain, and they lag exactly as
hardware does.

Fidelity is deliberate about one thing: the inner loop reproduces the arm
server's actual law rather than a convenient approximation --

    tau = kp * (q_d - q) + kd * (0 - qdot) + tau_gravity

the DaMiao MIT command, with gravity compensation solved by inverse dynamics on
a collision-free copy of the model, as ``YamRobot`` does on hardware.

What this is NOT: mass, inertia and friction come from the station XML and will
not match the real arm quantitatively. Use it to verify law, plumbing, direction
and stability -- never to tune gains for hardware.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from cap_harness.yam_real.kinematics import ARM_QSLICE

#: Finger joints per arm, alongside ``ARM_QSLICE`` for the arm joints.
FINGER_QSLICE: dict[str, slice] = {"left": slice(6, 8), "right": slice(14, 16)}

GRIPPER_FINGER_KP = 200.0
GRIPPER_FINGER_KD = 5.0
GRIPPER_TORQUE_LIMIT_NM = 0.5


class SimYamStation:
    """Shared MuJoCo station, optionally stepped in real time by a thread."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        gravity: bool = True,
        gravity_comp_factor: float = 1.0,
    ) -> None:
        """``gravity_comp_factor`` scales the gravity *estimate* the controller
        uses, leaving the plant's true gravity untouched.

        This reproduces the dominant real-world error source: hardware applies an
        empirical factor to a model that omits friction and gearing losses, so the
        estimate never exactly equals true gravity and a stationary arm sags a
        little. At exactly 1.0 compensation is perfect, which is convenient for
        tests about tracking but unrepresentative of the bench.
        """
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if not gravity:
            self.model.opt.gravity[:] = 0.0
        # Neutralize the XML's built-in position servos: the plant is driven by
        # explicit joint torques so the executed law is ours, not the model's.
        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_biasprm[:] = 0.0
        self.data = mujoco.MjData(self.model)

        # Collision-free scratch copy for the gravity-compensation solve,
        # mirroring the hardware controller's setup.
        self._grav_model = mujoco.MjModel.from_xml_path(str(model_path))
        self._grav_model.geom_contype[:] = 0
        self._grav_model.geom_conaffinity[:] = 0
        self._grav_data = mujoco.MjData(self._grav_model)

        self.lock = threading.RLock()
        self._targets = {side: np.zeros(7) for side in ARM_QSLICE}
        self._gains = {side: (np.zeros(7), np.zeros(7)) for side in ARM_QSLICE}
        self._last_torque = {side: np.zeros(6) for side in ARM_QSLICE}

        # The two fingers of a gripper travel in OPPOSITE directions: on this
        # model one has range [-0.002, 0.0375] and the other [-0.0375, 0.002].
        # Driving both toward the same signed target jams one against its limit
        # and leaves the mean of the pair near zero, so a normalized command maps
        # through each joint's own range instead of a shared travel constant.
        self._finger_closed: dict[str, np.ndarray] = {}
        self._finger_open: dict[str, np.ndarray] = {}
        for side in FINGER_QSLICE:
            ranges = []
            for finger in ("left", "right"):
                name = f"{side}_{finger}_finger"
                joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint < 0:
                    raise RuntimeError(f"finger joint {name!r} not found in {model_path}")
                ranges.append(self.model.jnt_range[joint])
            lo = np.array([r[0] for r in ranges], dtype=np.float64)
            hi = np.array([r[1] for r in ranges], dtype=np.float64)
            # The open end is whichever limit is further from zero.
            wider = np.abs(hi) >= np.abs(lo)
            self._finger_open[side] = np.where(wider, hi, lo)
            self._finger_closed[side] = np.where(wider, lo, hi)

        self.gravity_comp_factor = float(gravity_comp_factor)
        self._running = False
        self._thread: threading.Thread | None = None
        self.steps = 0

    def _finger_targets(self, side: str, normalized: float) -> np.ndarray:
        """Map a normalized gripper command onto each finger's own travel."""
        closed = self._finger_closed[side]
        opened = self._finger_open[side]
        return closed + float(np.clip(normalized, 0.0, 1.0)) * (opened - closed)

    def _finger_normalized(self, side: str, positions: np.ndarray) -> float:
        """Invert :meth:`_finger_targets`, averaging the mirrored pair."""
        closed = self._finger_closed[side]
        opened = self._finger_open[side]
        span = opened - closed
        span = np.where(np.abs(span) < 1e-12, 1.0, span)
        return float(np.clip(np.mean((positions - closed) / span), 0.0, 1.0))

    # -- setup -------------------------------------------------------------

    def set_joint_positions(self, left: np.ndarray, right: np.ndarray) -> None:
        """Teleport the arms (test setup only), zero velocity, and hold there."""
        with self.lock:
            for side, value in (("left", left), ("right", right)):
                self.data.qpos[ARM_QSLICE[side]] = np.asarray(value, dtype=np.float64).reshape(6)
            self.data.qvel[:] = 0.0
            for side in ARM_QSLICE:
                target = np.zeros(7)
                target[:6] = self.data.qpos[ARM_QSLICE[side]]
                self._targets[side] = target
            mujoco.mj_forward(self.model, self.data)

    # -- control -----------------------------------------------------------

    def command(self, side: str, pos7: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        with self.lock:
            self._targets[side] = np.asarray(pos7, dtype=np.float64).reshape(7)
            self._gains[side] = (
                np.asarray(kp, dtype=np.float64).reshape(7),
                np.asarray(kd, dtype=np.float64).reshape(7),
            )

    def _gravity_torque(self, side: str) -> np.ndarray:
        """Gravity-only inverse dynamics for one arm, as the arm server computes it."""
        gd = self._grav_data
        gd.qpos[:] = self.data.qpos
        gd.qvel[:] = 0.0
        gd.qacc[:] = 0.0
        mujoco.mj_inverse(self._grav_model, gd)
        raw = np.asarray(gd.qfrc_inverse[ARM_QSLICE[side]], dtype=np.float64).copy()
        return raw * self.gravity_comp_factor

    def _apply_control(self) -> None:
        """One physics step's worth of joint torque, per the MIT law."""
        self.data.qfrc_applied[:] = 0.0
        for side, qslice in ARM_QSLICE.items():
            target = self._targets[side]
            kp, kd = self._gains[side]
            q = self.data.qpos[qslice]
            qd = self.data.qvel[qslice]
            tau = kp[:6] * (target[:6] - q) + kd[:6] * (0.0 - qd) + self._gravity_torque(side)
            self.data.qfrc_applied[qslice] = tau
            self._last_torque[side] = tau.copy()

            # Gripper fingers: a simple PD onto the normalized target. The real
            # gripper takes a FORCE_POS command, not MIT, so it deliberately does
            # not use the arm gains.
            fslice = FINGER_QSLICE[side]
            finger_target = self._finger_targets(side, float(target[6]))
            fq = self.data.qpos[fslice]
            fqd = self.data.qvel[fslice]
            per_finger_limit = GRIPPER_TORQUE_LIMIT_NM / max(1, len(fq))
            self.data.qfrc_applied[fslice] = np.clip(
                GRIPPER_FINGER_KP * (finger_target - fq) - GRIPPER_FINGER_KD * fqd,
                -per_finger_limit,
                per_finger_limit,
            )

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(int(n)):
                self._apply_control()
                mujoco.mj_step(self.model, self.data)
                self.steps += 1

    # -- observation -------------------------------------------------------

    def observe(self, side: str) -> dict[str, np.ndarray]:
        """Mirror the arm server's observation dict, including one hardware quirk.

        Every per-joint channel except ``joint_pos`` is returned at full MOTOR
        width -- six arm joints plus the gripper -- because the hardware server
        slices only ``joint_pos`` down to the arm. Returning tidy 6-vectors here
        would hide a shape mismatch that then only appears on the robot.
        """
        with self.lock:
            qslice = ARM_QSLICE[side]
            fslice = FINGER_QSLICE[side]
            q = np.asarray(self.data.qpos[qslice], dtype=np.float64).copy()
            qd = np.asarray(self.data.qvel[qslice], dtype=np.float64).copy()
            fingers = np.asarray(self.data.qpos[fslice], dtype=np.float64).copy()
            # Mirrored fingers: average the SIGNED opening rates, not the raw
            # positions, which would cancel to zero however far the jaws travel.
            fqd = float(np.mean(np.abs(self.data.qvel[fslice])))
            ftau = float(np.sum(np.abs(self.data.qfrc_applied[fslice])))
            tau = self._last_torque[side].copy()

        return {
            "joint_pos": q,
            "joint_vel": np.concatenate([qd, [fqd]]),
            "joint_eff": np.concatenate([tau, [ftau]]),
            "gripper_pos": np.array([self._finger_normalized(side, fingers)]),
        }

    def site_position(self, side: str) -> np.ndarray:
        with self.lock:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_site")
            return np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Step physics in real time, so wall-clock level-1 timing stays honest."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        dt = float(self.model.opt.timestep)
        t_start = time.time()
        simulated = 0.0
        while self._running:
            behind = (time.time() - t_start) - simulated
            if behind < dt:
                time.sleep(max(0.0, dt - behind) * 0.5)
                continue
            # Bound catch-up so a scheduling hiccup cannot make the sim sprint.
            for _ in range(min(int(behind / dt), 20)):
                self.step()
                simulated += dt

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class SimYamArm:
    """Per-arm client facade over a shared :class:`SimYamStation`."""

    def __init__(
        self,
        station: SimYamStation,
        side: str,
        *,
        default_kp: np.ndarray,
        default_kd: np.ndarray,
    ) -> None:
        self.station = station
        self.side = side
        self._default_kp = np.asarray(default_kp, dtype=np.float64).reshape(7)
        self._default_kd = np.asarray(default_kd, dtype=np.float64).reshape(7)

    def get_observations(self) -> dict[str, np.ndarray]:
        return self.station.observe(self.side)

    def get_joint_pos(self) -> np.ndarray:
        obs = self.station.observe(self.side)
        return np.concatenate([obs["joint_pos"], obs["gripper_pos"]])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.station.command(self.side, joint_pos, self._default_kp, self._default_kd)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        self.station.command(
            self.side,
            joint_state["pos"],
            joint_state.get("kp", self._default_kp),
            joint_state.get("kd", self._default_kd),
        )

    def close(self) -> None:
        """Client-side close; the shared station outlives any one arm facade."""


__all__ = [
    "FINGER_QSLICE",
    "SimYamArm",
    "SimYamStation",
]
