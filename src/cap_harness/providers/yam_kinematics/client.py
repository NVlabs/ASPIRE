"""YAM's own mink IK behind the harness ``IKProvider`` seam.

The shipped PyRoki provider solves a seven-joint Panda and rejects a YAM seed
outright ("seed joints must be a finite seven-joint vector"). Rather than teach
that service a YAM description, this exposes the solver the station already
uses -- :class:`~cap_harness.yam_real.kinematics.YamKinematics`, mink over the
calibrated station model -- at the same seam. The protocol is the harness's;
only the solver is ours.

It earns its place next to the cuRobo planner because the two answer different
questions. cuRobo returns a configuration reached by a **collision-free plan**,
which is what you want before moving. This returns the configuration that
**satisfies the pose**, which is what you want when asking whether a pose is
reachable at all -- and it runs in-process, with no service on the path.

That distinction is what makes the contact segment of a pick work. A descent
onto an object is, by construction, a move into the observed point cloud, so a
collision-aware planner is right to refuse it. Solving those waypoints here and
executing them as an explicitly ``collision_aware=False`` trajectory keeps the
refusal meaningful everywhere else instead of switching collision checking off
globally to get one motion through.
"""

from __future__ import annotations

import numpy as np

from ...contracts import IKResult, Pose, RobotState
from ...errors import ApiError, ErrorCode

_ARMS = ("left", "right")

#: How close the forward kinematics of a returned configuration must land to the
#: requested position for the solve to count as converged. Matches the default
#: execution tolerance in ``CapApi.move_to_pose``: a solution the executor would
#: reject on arrival must not be handed out as a success beforehand.
DEFAULT_IK_TOLERANCE_M = 0.01

ARM_DOF = 6


class YamKinematicsIKProvider:
    """Local IK for the 6-DOF YAM arms."""

    def __init__(
        self,
        kinematics: object,
        *,
        tolerance_m: float = DEFAULT_IK_TOLERANCE_M,
    ) -> None:
        self._kin = kinematics
        self._tolerance_m = float(tolerance_m)

    def solve_ik(
        self,
        target_pose: Pose,
        robot_state: RobotState,
        *,
        arm: str = "primary",
    ) -> IKResult:
        side = "left" if arm not in _ARMS else arm

        try:
            current = {
                name: np.asarray(robot_state.joint_positions[name], dtype=float).reshape(ARM_DOF)
                for name in _ARMS
            }
            poses = {name: robot_state.end_effector_poses[name] for name in _ARMS}
        except (KeyError, TypeError, ValueError) as exc:
            return IKResult(
                ok=False,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"robot state does not carry both YAM arms: {exc}",
                ),
            )

        # The solver is bimanual and moves whatever it is given a target for, so
        # the arm we are not solving is pinned to its measured pose. Passing that
        # arm's current pose as its target is what "hold still" means here.
        targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in _ARMS:
            pose = target_pose if name == side else poses[name]
            quaternion = np.asarray(pose.quaternion_wxyz, dtype=float).reshape(4)
            targets[name] = (
                np.asarray(pose.position, dtype=float).reshape(3),
                # contracts carry wxyz; YamKinematics takes xyzw
                np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]),
            )

        try:
            # Seed the solver from the caller's state before solving. Without
            # this the call is not a function of its arguments: ``seeded=True``
            # means "continue from the CURRENT configuration", and that
            # configuration is shared mutable state left behind by whatever
            # solved last. The seed passed in here was reaching only the
            # opposite arm's hold target, never the arm actually being solved.
            #
            # The symptom is a solve that succeeds alone and fails in sequence.
            # Measured: a rim pose that solves from a clean solver fails with
            # "best configuration is 0.067 m from the target" once two dozen
            # unrelated reachability probes have run on the same instance --
            # which is exactly what a program does when it surveys the workspace
            # before planning into it.
            self._kin.seed(current["left"], current["right"])
            left_solution, right_solution = self._kin.inverse_kinematics(
                targets["left"][0],
                targets["left"][1],
                targets["right"][0],
                targets["right"][1],
                seeded=True,
            )
        except Exception as exc:
            return IKResult(
                ok=False,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    code=ErrorCode.IK_FAILED,
                    message=f"mink IK failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )

        joints = np.asarray(
            left_solution if side == "left" else right_solution, dtype=float
        ).reshape(-1)
        if joints.shape[0] != ARM_DOF or not bool(np.all(np.isfinite(joints))):
            return IKResult(
                ok=False,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    code=ErrorCode.IK_FAILED,
                    message=f"solver returned an unusable configuration of shape {joints.shape}",
                ),
            )

        # mink returns its best effort whether or not it converged, so the
        # solution is checked by forward kinematics before it is called a
        # success. Without this an unreachable grasp comes back ok, and
        # ``move_to_pose`` drives the arm at it and fails its own tolerance check
        # afterwards -- after moving. An IK solver that cannot reach the pose has
        # to say so beforehand.
        try:
            left_position, _, right_position, _ = self._kin.forward_kinematics(
                joints if side == "left" else current["left"],
                joints if side == "right" else current["right"],
            )
            achieved = left_position if side == "left" else right_position
            position_error_m = float(
                np.linalg.norm(
                    np.asarray(achieved, dtype=float).reshape(3)
                    - np.asarray(target_pose.position, dtype=float).reshape(3)
                )
            )
        except Exception as exc:
            return IKResult(
                ok=False,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    code=ErrorCode.IK_FAILED,
                    message=f"could not verify the mink solution by forward kinematics: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )

        if not np.isfinite(position_error_m) or position_error_m > self._tolerance_m:
            return IKResult(
                ok=False,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    code=ErrorCode.IK_UNREACHABLE,
                    message=(
                        f"mink did not converge: best configuration is "
                        f"{position_error_m:.3f} m from the target "
                        f"(tolerance {self._tolerance_m:.3f} m)"
                    ),
                    details={"position_error_m": position_error_m, "arm": side},
                ),
            )

        return IKResult(
            ok=True,
            embodiment=robot_state.embodiment,
            joint_positions=joints,
            arm=side,
            diagnostics={"solver": "mink", "position_error_m": position_error_m},
        )


__all__ = ["ARM_DOF", "DEFAULT_IK_TOLERANCE_M", "YamKinematicsIKProvider"]
