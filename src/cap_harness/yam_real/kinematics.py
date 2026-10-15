"""FK, IK and grasp-site Jacobians for the bimanual YAM station.

Solved locally with mink over the station's own calibrated MuJoCo model rather
than through the shared PyRoki provider: that provider is built for a 7-joint
Panda and rejects a 6-DOF seed, and YAM's two arms sit 620 mm apart in a single
model, so a per-arm solver would have to be told where the other arm is anyway.

The model is bimanual, so the qpos layout is shared and fixed: the left arm
occupies ``qpos[0:6]``, its fingers ``[6:8]``, the right arm ``[8:14]`` and its
fingers ``[14:16]``. Callers pass and receive per-arm 6-vectors; the interleaving
is this module's business.

Quaternions here are **xyzw**, matching mink and scipy. The harness contracts use
wxyz, and that conversion lives in :mod:`cap_harness.yam_real.codec` -- one
boundary, one place.

This class is not thread safe. It carries a mutable ``mink.Configuration``, so a
caller sharing one instance across threads must serialize access (``RealYamEnv``
holds a lock for exactly this reason).
"""

from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np

#: Where each arm's six joints live in the bimanual model's qpos vector.
ARM_QSLICE: dict[str, slice] = {"left": slice(0, 6), "right": slice(8, 14)}

ARM_DOF = 6


class YamKinematics:
    """FK/IK over the calibrated bimanual station model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 1.0,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"YAM station model XML not found: {path}")
        self.model_path = path
        self._model = mujoco.MjModel.from_xml_path(str(path))
        self.configuration = mink.Configuration(self._model)
        self.tasks = [
            mink.FrameTask(
                frame_name=f"{side}_grasp_site",
                frame_type="site",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=lm_damping,
            )
            for side in ("left", "right")
        ]
        self.left_end_effector_task, self.right_end_effector_task = self.tasks

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    def seed(self, left_joint_pos: np.ndarray, right_joint_pos: np.ndarray) -> None:
        """Write both arms' measured positions into the shared configuration."""
        self.configuration.data.qpos[ARM_QSLICE["left"]] = np.asarray(
            left_joint_pos, dtype=np.float64
        ).reshape(ARM_DOF)
        self.configuration.data.qpos[ARM_QSLICE["right"]] = np.asarray(
            right_joint_pos, dtype=np.float64
        ).reshape(ARM_DOF)
        self.configuration.update()

    def forward_kinematics(
        self, left_joint_pos: np.ndarray, right_joint_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(left_pos, left_quat_xyzw, right_pos, right_quat_xyzw)``."""
        self.seed(left_joint_pos, right_joint_pos)
        poses = {
            side: self.configuration.get_transform_frame_to_world(f"{side}_grasp_site", "site")
            for side in ("left", "right")
        }
        return (
            poses["left"].translation(),
            poses["left"].rotation().wxyz[[1, 2, 3, 0]],
            poses["right"].translation(),
            poses["right"].rotation().wxyz[[1, 2, 3, 0]],
        )

    def grasp_site_jacobian(
        self,
        side: str,
        left_joint_pos: np.ndarray | None = None,
        right_joint_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(J_pos, J_rot)``, each ``(3, 6)``, for one arm's grasp site.

        Columns are restricted to that arm's own six joints, so the result maps
        arm joint velocity to grasp-site linear/angular velocity in the world
        frame. Pass both arms' positions to evaluate at a specific configuration,
        or omit them to reuse whatever was last written (for example by a
        preceding ``forward_kinematics`` under the same lock).
        """
        if side not in ARM_QSLICE:
            raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
        if left_joint_pos is not None and right_joint_pos is not None:
            self.seed(left_joint_pos, right_joint_pos)
        elif left_joint_pos is not None or right_joint_pos is not None:
            raise ValueError("pass both arms' joint positions, or neither")

        site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp_site")
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site {side}_grasp_site not found in {self.model_path}")
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self._model, self.configuration.data, jacp, jacr, site_id)
        cols = ARM_QSLICE[side]
        return jacp[:, cols].copy(), jacr[:, cols].copy()

    def inverse_kinematics(
        self,
        left_ee_pos: np.ndarray,
        left_ee_quat_xyzw: np.ndarray,
        right_ee_pos: np.ndarray,
        right_ee_quat_xyzw: np.ndarray,
        *,
        seeded: bool = False,
        dt: float = 0.01,
        solver: str = "daqp",
        damping: float = 1e-3,
        err_threshold: float = 1e-4,
        max_iters: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve both arms to their targets, returning per-arm 6-vectors.

        ``seeded=True`` continues from the current configuration, which is what a
        caller stepping along a path wants: successive solutions stay in the same
        branch instead of jumping between elbow configurations. ``seeded=False``
        restarts from zero and is only appropriate for a one-shot solve.

        Returns the best configuration reached within ``max_iters``; convergence
        is not guaranteed, so a caller that needs a guarantee must check the
        resulting pose with :meth:`forward_kinematics`.
        """
        if not seeded:
            self.configuration.update(np.zeros_like(self.configuration.data.qpos))

        targets = {
            "left": (left_ee_pos, left_ee_quat_xyzw),
            "right": (right_ee_pos, right_ee_quat_xyzw),
        }
        for task, side in zip(self.tasks, ("left", "right")):
            position, quat_xyzw = targets[side]
            quat_wxyz = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)[[3, 0, 1, 2]]
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(wxyz=quat_wxyz),
                    np.asarray(position, dtype=np.float64).reshape(3),
                )
            )

        for _ in range(max_iters):
            velocity = mink.solve_ik(self.configuration, self.tasks, dt, solver, damping)
            self.configuration.integrate_inplace(velocity, dt)
            errors = [np.linalg.norm(task.compute_error(self.configuration)) for task in self.tasks]
            if max(errors) <= err_threshold:
                break

        qpos = self.configuration.q
        return qpos[ARM_QSLICE["left"]].copy(), qpos[ARM_QSLICE["right"]].copy()


__all__ = ["ARM_DOF", "ARM_QSLICE", "YamKinematics"]
