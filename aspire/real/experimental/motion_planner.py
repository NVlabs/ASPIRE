"""RRTConnect EE-pose motion planner for the YAM bimanual robot.

Takes end-effector pose targets and uses MuJoCo's built-in collision
detection to produce collision-free joint trajectories. Supports
single-arm and dual-arm planning.

Algorithm: Bidirectional RRT (RRT-Connect) with random shortcutting.

Usage:
    from experimental.motion_planner import YamMotionPlanner

    planner = YamMotionPlanner()
    result = planner.plan_to_pose(
        target_left_pos=np.array([0.5, 0.2, 1.0]),
        target_left_quat_xyzw=np.array([0, 0.707, 0, 0.707]),
        current_left_jp=np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25]),
        current_right_jp=np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25]),
        side="left",
    )
    if result["status"] == "Success":
        for jp in result["position"]:
            ...  # execute joint positions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from robot.models.station.paths import get_station_xml

_MODEL_XML = get_station_xml()

# qpos layout: [left_j1..j6 (0-5), left_lf (6), left_rf (7),
#               right_j1..j6 (8-13), right_rf (14), right_rf (15)]
_LEFT_ARM_QIDX = list(range(0, 6))
_RIGHT_ARM_QIDX = list(range(8, 14))
_ALL_ARM_QIDX = _LEFT_ARM_QIDX + _RIGHT_ARM_QIDX

# Default planning parameters
_MAX_ITERS = 5000
_STEP_SIZE = 0.15  # radians
_COLLISION_CHECKS_PER_EDGE = 10
_SHORTCUT_ATTEMPTS = 100
_RESAMPLE_DT = 1.0 / 30.0  # 30 Hz control
_MAX_JOINT_VEL = 1.0  # rad/s per joint (conservative)
_IK_POSITION_COST = 1.0
_IK_ORIENTATION_COST = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class YamMotionPlanner:
    """Sampling-based motion planner for the YAM station using MuJoCo."""

    def __init__(
        self,
        model_xml: str | Path | None = None,
        *,
        position_cost: float = _IK_POSITION_COST,
        orientation_cost: float = _IK_ORIENTATION_COST,
    ) -> None:
        xml_path = Path(model_xml or _MODEL_XML).resolve()
        self._model_xml_path = xml_path
        self._model = self._load_model_with_large_buffers(xml_path)
        self._data = mujoco.MjData(self._model)
        self.ik_position_cost = float(position_cost)
        self.ik_orientation_cost = float(orientation_cost)

        # Joint limits for each arm (6 revolute joints)
        self._left_lo = self._model.jnt_range[_LEFT_ARM_QIDX, 0].copy()
        self._left_hi = self._model.jnt_range[_LEFT_ARM_QIDX, 1].copy()
        self._right_lo = self._model.jnt_range[_RIGHT_ARM_QIDX, 0].copy()
        self._right_hi = self._model.jnt_range[_RIGHT_ARM_QIDX, 1].copy()

        from robot.yam.kinematics import YamKinematics

        self._kin = YamKinematics(
            position_cost=self.ik_position_cost,
            orientation_cost=self.ik_orientation_cost,
        )

        # Build set of body IDs for each arm's full kinematic chain.
        # Contacts between geoms on the same arm should be ignored
        # (self-collisions within one arm are not meaningful for planning).
        self._left_arm_bodies = self._get_body_ids(
            [
                "left_arm",
                "left_link_1",
                "left_link_2",
                "left_link_3",
                "left_link_4",
                "left_link_5",
                "left_link_6",
                "left_tcp",
                "left_left_link_finger",
                "left_right_link_finger",
                "left_lf_rot",
                "left_lf_down",
                "left_rf_rot",
                "left_rf_down",
                "left_camera_d405",
                "left_camera_frame",
            ]
        )
        self._right_arm_bodies = self._get_body_ids(
            [
                "right_arm",
                "right_link_1",
                "right_link_2",
                "right_link_3",
                "right_link_4",
                "right_link_5",
                "right_link_6",
                "right_tcp",
                "right_left_link_finger",
                "right_right_link_finger",
                "right_lf_rot",
                "right_lf_down",
                "right_rf_rot",
                "right_rf_down",
                "right_camera_d405",
                "right_camera_frame",
            ]
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model_with_large_buffers(xml_path: Path) -> mujoco.MjModel:
        """Load MuJoCo model with enlarged constraint buffers.

        Patches the XML ``<size>`` element to set ``nconmax`` and ``njmax``
        so that ``mj_forward`` does not crash with "nefc under-allocation"
        when gripper contacts create many constraints.
        """
        import re

        xml_text = xml_path.read_text()
        size_match = re.search(r"<size\b([^/]*)/?>", xml_text)
        if size_match:
            old_tag = size_match.group(0)
            attrs = size_match.group(1)
            if "nconmax" not in attrs:
                attrs += ' nconmax="500"'
            if "njmax" not in attrs:
                attrs += ' njmax="500"'
            xml_text = xml_text.replace(old_tag, f"<size{attrs}/>", 1)
        # Write patched XML next to the original so mesh paths resolve.
        import tempfile

        fd, tmp_name = tempfile.mkstemp(
            prefix=".station_planner_tmp_",
            suffix=".xml",
            dir=xml_path.parent,
            text=True,
        )
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(xml_text)
        try:
            return mujoco.MjModel.from_xml_path(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _get_body_ids(self, body_names: list[str]) -> set[int]:
        ids = set()
        for name in body_names:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids.add(bid)
        return ids

    def _is_same_arm_contact(self, geom1: int, geom2: int) -> bool:
        """Return True if both geoms belong to the same arm's kinematic chain."""
        b1 = int(self._model.geom_bodyid[geom1])
        b2 = int(self._model.geom_bodyid[geom2])
        if b1 in self._left_arm_bodies and b2 in self._left_arm_bodies:
            return True
        if b1 in self._right_arm_bodies and b2 in self._right_arm_bodies:
            return True
        return False

    def set_gripper_qpos(
        self,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> None:
        """Set gripper finger qpos used during collision checks.

        The gripper value is in [0, 1] (0 = closed, 1 = open).
        Internally maps to the slide-joint ranges for each finger.
        Must be called before planning if grippers are not fully closed.
        """
        if left_gripper is not None:
            g = float(np.clip(left_gripper, 0.0, 1.0))
            # left_left_finger: range [-0.002, 0.038], left_right_finger: range [-0.038, 0.002]
            self._left_gripper_qpos = (
                -0.002 + g * 0.040,  # left finger opens positive
                0.002 - g * 0.040,  # right finger opens negative
            )
        if right_gripper is not None:
            g = float(np.clip(right_gripper, 0.0, 1.0))
            self._right_gripper_qpos = (
                -0.002 + g * 0.040,
                0.002 - g * 0.040,
            )

    def check_collision(
        self,
        left_jp: np.ndarray,
        right_jp: np.ndarray,
    ) -> bool:
        """Return True if the configuration has unexpected collisions.

        Ignores contacts between geoms on the same arm (self-collisions).
        Only reports inter-arm or arm-environment collisions.
        """
        self._data.qpos[:6] = left_jp
        self._data.qpos[8:14] = right_jp
        # Set gripper finger positions (avoids false collisions from default-zero fingers)
        if hasattr(self, "_left_gripper_qpos"):
            self._data.qpos[6], self._data.qpos[7] = self._left_gripper_qpos
        if hasattr(self, "_right_gripper_qpos"):
            self._data.qpos[14], self._data.qpos[15] = self._right_gripper_qpos
        mujoco.mj_forward(self._model, self._data)
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            if not self._is_same_arm_contact(c.geom1, c.geom2):
                return True
        return False

    def check_collision_verbose(
        self,
        left_jp: np.ndarray,
        right_jp: np.ndarray,
    ) -> tuple[bool, list[str]]:
        """Like check_collision but returns details of offending contacts."""
        self._data.qpos[:6] = left_jp
        self._data.qpos[8:14] = right_jp
        if hasattr(self, "_left_gripper_qpos"):
            self._data.qpos[6], self._data.qpos[7] = self._left_gripper_qpos
        if hasattr(self, "_right_gripper_qpos"):
            self._data.qpos[14], self._data.qpos[15] = self._right_gripper_qpos
        mujoco.mj_forward(self._model, self._data)
        details: list[str] = []
        has_collision = False
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            if not self._is_same_arm_contact(c.geom1, c.geom2):
                has_collision = True
                b1 = int(self._model.geom_bodyid[c.geom1])
                b2 = int(self._model.geom_bodyid[c.geom2])
                b1_name = (
                    mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, b1)
                    or f"body_{b1}"
                )
                b2_name = (
                    mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, b2)
                    or f"body_{b2}"
                )
                details.append(f"{b1_name} <-> {b2_name} (dist={c.dist:.4f})")
        return has_collision, details

    # ------------------------------------------------------------------
    # Internal configuration-space helpers
    # ------------------------------------------------------------------

    def _get_qidx(self, side: str) -> list[int]:
        if side == "left":
            return _LEFT_ARM_QIDX
        elif side == "right":
            return _RIGHT_ARM_QIDX
        else:
            return _ALL_ARM_QIDX

    def _get_limits(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        if side == "left":
            return self._left_lo, self._left_hi
        elif side == "right":
            return self._right_lo, self._right_hi
        else:
            return (
                np.concatenate([self._left_lo, self._right_lo]),
                np.concatenate([self._left_hi, self._right_hi]),
            )

    def _to_config(
        self,
        left_jp: np.ndarray,
        right_jp: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """Extract the planning configuration from full joint positions."""
        if side == "left":
            return left_jp.copy()
        elif side == "right":
            return right_jp.copy()
        else:
            return np.concatenate([left_jp, right_jp])

    def _split_config(
        self,
        q: np.ndarray,
        side: str,
        ref_left: np.ndarray,
        ref_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split a planning config back to (left_jp, right_jp)."""
        if side == "left":
            return q.copy(), ref_right.copy()
        elif side == "right":
            return ref_left.copy(), q.copy()
        else:
            return q[:6].copy(), q[6:].copy()

    def _sample_random(self, side: str) -> np.ndarray:
        lo, hi = self._get_limits(side)
        return np.random.uniform(lo, hi)

    def _steer(
        self,
        q_near: np.ndarray,
        q_rand: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        diff = q_rand - q_near
        dist = np.linalg.norm(diff)
        if dist <= step_size:
            return q_rand.copy()
        return q_near + diff * (step_size / dist)

    def _is_edge_collision_free(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        side: str,
        ref_left: np.ndarray,
        ref_right: np.ndarray,
        n_checks: int = _COLLISION_CHECKS_PER_EDGE,
    ) -> bool:
        """Check collision at n_checks points along the edge (including endpoints)."""
        for i in range(n_checks + 1):
            t = i / max(n_checks, 1)
            q = q_start + t * (q_end - q_start)
            ljp, rjp = self._split_config(q, side, ref_left, ref_right)
            if self.check_collision(ljp, rjp):
                return False
        return True

    def _is_config_valid(
        self,
        q: np.ndarray,
        side: str,
        ref_left: np.ndarray,
        ref_right: np.ndarray,
    ) -> bool:
        """Check if a configuration is within limits and collision-free."""
        lo, hi = self._get_limits(side)
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            return False
        ljp, rjp = self._split_config(q, side, ref_left, ref_right)
        return not self.check_collision(ljp, rjp)

    def _diagnose_config(
        self,
        q: np.ndarray,
        side: str,
        ref_left: np.ndarray,
        ref_right: np.ndarray,
        label: str = "config",
    ) -> str:
        """Return a human-readable diagnosis of why a config is invalid."""
        lo, hi = self._get_limits(side)
        oob_lo = q < lo - 1e-6
        oob_hi = q > hi + 1e-6
        reasons: list[str] = []
        if np.any(oob_lo) or np.any(oob_hi):
            violations = []
            for i in range(len(q)):
                if oob_lo[i]:
                    violations.append(f"j{i}={q[i]:.4f}<{lo[i]:.4f}")
                elif oob_hi[i]:
                    violations.append(f"j{i}={q[i]:.4f}>{hi[i]:.4f}")
            reasons.append(f"joint limits violated: {', '.join(violations)}")
        ljp, rjp = self._split_config(q, side, ref_left, ref_right)
        has_col, details = self.check_collision_verbose(ljp, rjp)
        if has_col:
            reasons.append(f"collision: {'; '.join(details[:3])}")
        if not reasons:
            return f"{label}: valid"
        return f"{label}: INVALID — {'; '.join(reasons)}"

    # ------------------------------------------------------------------
    # RRT-Connect
    # ------------------------------------------------------------------

    def rrt_connect(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        side: str,
        ref_left: np.ndarray,
        ref_right: np.ndarray,
        max_iters: int = _MAX_ITERS,
        step_size: float = _STEP_SIZE,
        verbose: bool = True,
    ) -> list[np.ndarray] | None:
        """Bidirectional RRT-Connect.

        Returns a list of waypoints (joint configs) from q_start to q_goal,
        or None if no path was found within max_iters.
        """
        # Clamp start/goal to joint limits so tiny floating-point drift
        # (e.g. sensor noise, interpolation) doesn't cause spurious failures.
        lo, hi = self._get_limits(side)
        q_start = np.clip(q_start, lo, hi)
        q_goal = np.clip(q_goal, lo, hi)

        # Validate start and goal
        if not self._is_config_valid(q_start, side, ref_left, ref_right):
            diag = self._diagnose_config(q_start, side, ref_left, ref_right, "Start")
            if verbose:
                print(f"[MotionPlanner] {diag}")
            return None
        if not self._is_config_valid(q_goal, side, ref_left, ref_right):
            diag = self._diagnose_config(q_goal, side, ref_left, ref_right, "Goal")
            if verbose:
                print(f"[MotionPlanner] {diag}")
            return None

        # Tree data structures: node list + parent index
        tree_a_nodes = [q_start.copy()]
        tree_a_parent = [-1]
        tree_b_nodes = [q_goal.copy()]
        tree_b_parent = [-1]

        def _nearest(tree_nodes: list[np.ndarray], q: np.ndarray) -> int:
            dists = [np.linalg.norm(n - q) for n in tree_nodes]
            return int(np.argmin(dists))

        def _extend(
            tree_nodes: list[np.ndarray],
            tree_parent: list[int],
            q_target: np.ndarray,
        ) -> tuple[str, int]:
            """Extend tree toward q_target. Returns ("reached"|"advanced"|"trapped", node_idx)."""
            near_idx = _nearest(tree_nodes, q_target)
            q_near = tree_nodes[near_idx]
            q_new = self._steer(q_near, q_target, step_size)
            # Clamp to limits
            lo, hi = self._get_limits(side)
            q_new = np.clip(q_new, lo, hi)

            if self._is_edge_collision_free(q_near, q_new, side, ref_left, ref_right):
                new_idx = len(tree_nodes)
                tree_nodes.append(q_new)
                tree_parent.append(near_idx)
                if np.linalg.norm(q_new - q_target) < 1e-4:
                    return "reached", new_idx
                return "advanced", new_idx
            return "trapped", near_idx

        def _connect(
            tree_nodes: list[np.ndarray],
            tree_parent: list[int],
            q_target: np.ndarray,
        ) -> tuple[str, int]:
            """Repeatedly extend toward q_target until reached or trapped."""
            while True:
                status, idx = _extend(tree_nodes, tree_parent, q_target)
                if status != "advanced":
                    return status, idx

        def _extract_path(
            tree_nodes: list[np.ndarray],
            tree_parent: list[int],
            leaf_idx: int,
        ) -> list[np.ndarray]:
            path = []
            idx = leaf_idx
            while idx != -1:
                path.append(tree_nodes[idx])
                idx = tree_parent[idx]
            path.reverse()
            return path

        swap = False
        for _ in range(max_iters):
            if swap:
                ta_nodes, ta_parent = tree_b_nodes, tree_b_parent
                tb_nodes, tb_parent = tree_a_nodes, tree_a_parent
            else:
                ta_nodes, ta_parent = tree_a_nodes, tree_a_parent
                tb_nodes, tb_parent = tree_b_nodes, tree_b_parent

            q_rand = self._sample_random(side)
            status_a, idx_a = _extend(ta_nodes, ta_parent, q_rand)
            if status_a != "trapped":
                q_new = ta_nodes[idx_a]
                status_b, idx_b = _connect(tb_nodes, tb_parent, q_new)
                if status_b == "reached":
                    # Extract and join paths
                    path_a = _extract_path(ta_nodes, ta_parent, idx_a)
                    path_b = _extract_path(tb_nodes, tb_parent, idx_b)
                    path_b.reverse()
                    if swap:
                        # path_a is from goal side, path_b is from start side
                        path = path_b + path_a
                    else:
                        path = path_a + path_b
                    return path
            swap = not swap

        if verbose:
            print(f"[MotionPlanner] RRT-Connect failed after {max_iters} iterations")
        return None

    # ------------------------------------------------------------------
    # Path post-processing
    # ------------------------------------------------------------------

    def shortcut_path(
        self,
        path: list[np.ndarray],
        side: str,
        ref_left: np.ndarray,
        ref_right: np.ndarray,
        n_attempts: int = _SHORTCUT_ATTEMPTS,
    ) -> list[np.ndarray]:
        """Random shortcutting: try to skip intermediate waypoints."""
        if len(path) <= 2:
            return path
        path = [q.copy() for q in path]
        for _ in range(n_attempts):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if self._is_edge_collision_free(
                path[i],
                path[j],
                side,
                ref_left,
                ref_right,
                n_checks=max(
                    _COLLISION_CHECKS_PER_EDGE,
                    int(np.linalg.norm(path[j] - path[i]) / 0.02),
                ),
            ):
                path = path[: i + 1] + path[j:]
        return path

    def resample_path(
        self,
        path: list[np.ndarray],
        max_joint_vel: float = _MAX_JOINT_VEL,
        dt: float = _RESAMPLE_DT,
    ) -> np.ndarray:
        """Resample waypoints at uniform time steps respecting velocity limits.

        Returns an (T, ndof) array of joint positions.
        """
        if len(path) <= 1:
            return np.array(path)

        # Compute segment lengths and time per segment
        positions = []
        for k in range(len(path) - 1):
            diff = path[k + 1] - path[k]
            max_delta = float(np.max(np.abs(diff)))
            seg_time = max(max_delta / max_joint_vel, dt)
            n_steps = max(1, int(np.ceil(seg_time / dt)))
            for s in range(n_steps):
                t = s / n_steps
                positions.append(path[k] + t * diff)
        positions.append(path[-1].copy())
        return np.array(positions)

    # ------------------------------------------------------------------
    # IK with multiple seeds
    # ------------------------------------------------------------------

    def _solve_ik(
        self,
        tgt_l_pos: np.ndarray,
        tgt_l_q: np.ndarray,
        tgt_r_pos: np.ndarray,
        tgt_r_q: np.ndarray,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        ik_error_threshold: float,
        n_random_seeds: int = 8,
        ik_max_iters: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Try IK with seeded attempt first, then random seeds.

        Returns (goal_left_jp, goal_right_jp, max_err) or None.
        """
        best = None
        best_err = float("inf")

        def _try_ik(seeded: bool) -> tuple[np.ndarray, np.ndarray, float] | None:
            if seeded:
                # Seed from current configuration (FK sets it)
                self._kin.forward_kinematics(current_left_jp, current_right_jp)
            else:
                # Random seed within joint limits
                q_rand = np.random.uniform(
                    np.concatenate([self._left_lo, self._right_lo]),
                    np.concatenate([self._left_hi, self._right_hi]),
                )
                self._kin.configuration.data.qpos[:6] = q_rand[:6]
                self._kin.configuration.data.qpos[8:14] = q_rand[6:]
                self._kin.configuration.update()

            gl, gr = self._kin.inverse_kinematics(
                tgt_l_pos,
                tgt_l_q,
                tgt_r_pos,
                tgt_r_q,
                seeded=True,
                max_iters=ik_max_iters,
            )
            got_l, _, got_r, _ = self._kin.forward_kinematics(gl, gr)
            l_err = float(np.linalg.norm(got_l - tgt_l_pos))
            r_err = float(np.linalg.norm(got_r - tgt_r_pos))
            max_err = max(l_err, r_err)
            if max_err <= ik_error_threshold:
                return gl, gr, max_err
            return None

        # Attempt 1: seeded from current config (best for nearby targets)
        result = _try_ik(seeded=True)
        if result is not None:
            return result

        # Attempt 2+: random seeds
        for _ in range(n_random_seeds):
            result = _try_ik(seeded=False)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # High-level: plan to an EE pose
    # ------------------------------------------------------------------

    def plan_to_pose(
        self,
        current_left_jp: np.ndarray,
        current_right_jp: np.ndarray,
        target_left_pos: np.ndarray | None = None,
        target_left_quat_xyzw: np.ndarray | None = None,
        target_right_pos: np.ndarray | None = None,
        target_right_quat_xyzw: np.ndarray | None = None,
        side: Literal["left", "right", "both"] = "both",
        max_iters: int = _MAX_ITERS,
        step_size: float = _STEP_SIZE,
        max_joint_vel: float = _MAX_JOINT_VEL,
        dt: float = _RESAMPLE_DT,
        ik_error_threshold: float = 0.005,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
        verbose: bool = True,
    ) -> dict:
        """Plan a collision-free trajectory to a target EE pose.

        Returns:
            dict with keys:
                "status": "Success" | "IK_Failed" | "Planning_Failed"
                "position": np.ndarray (T, 6) or (T, 12) — joint trajectory
                "left_positions": np.ndarray (T, 6) — left arm joints
                "right_positions": np.ndarray (T, 6) — right arm joints
        """
        # Set gripper positions for accurate collision checking
        self.set_gripper_qpos(left_gripper, right_gripper)

        # Resolve target joint config via IK
        cur_l_pos, cur_l_q, cur_r_pos, cur_r_q = self._kin.forward_kinematics(
            current_left_jp, current_right_jp
        )

        tgt_l_pos = (
            np.asarray(target_left_pos) if target_left_pos is not None else cur_l_pos
        )
        tgt_l_q = (
            np.asarray(target_left_quat_xyzw)
            if target_left_quat_xyzw is not None
            else cur_l_q
        )
        tgt_r_pos = (
            np.asarray(target_right_pos) if target_right_pos is not None else cur_r_pos
        )
        tgt_r_q = (
            np.asarray(target_right_quat_xyzw)
            if target_right_quat_xyzw is not None
            else cur_r_q
        )

        ik_result = self._solve_ik(
            tgt_l_pos,
            tgt_l_q,
            tgt_r_pos,
            tgt_r_q,
            current_left_jp,
            current_right_jp,
            ik_error_threshold,
        )
        if ik_result is None:
            # Report the error from the last seeded attempt for diagnostics
            self._kin.forward_kinematics(current_left_jp, current_right_jp)
            gl, gr = self._kin.inverse_kinematics(
                tgt_l_pos,
                tgt_l_q,
                tgt_r_pos,
                tgt_r_q,
                seeded=True,
            )
            got_l, _, got_r, _ = self._kin.forward_kinematics(gl, gr)
            l_err = float(np.linalg.norm(got_l - tgt_l_pos))
            r_err = float(np.linalg.norm(got_r - tgt_r_pos))
            if verbose:
                print(
                    f"[MotionPlanner] IK failed after multi-seed: left_err={l_err:.4f}m, "
                    f"right_err={r_err:.4f}m (threshold={ik_error_threshold:.4f}m)"
                )
            return {"status": "IK_Failed", "position": np.empty((0, 6))}

        goal_left_jp, goal_right_jp, _ = ik_result

        q_start = self._to_config(current_left_jp, current_right_jp, side)
        q_goal = self._to_config(goal_left_jp, goal_right_jp, side)

        # RRT-Connect
        raw_path = self.rrt_connect(
            q_start,
            q_goal,
            side,
            ref_left=current_left_jp,
            ref_right=current_right_jp,
            max_iters=max_iters,
            step_size=step_size,
            verbose=verbose,
        )
        if raw_path is None:
            return {"status": "Planning_Failed", "position": np.empty((0, 6))}

        # Post-process
        smooth_path = self.shortcut_path(
            raw_path,
            side,
            ref_left=current_left_jp,
            ref_right=current_right_jp,
        )
        positions = self.resample_path(smooth_path, max_joint_vel, dt)

        # Split the shortcut path itself into left/right waypoints so callers can
        # apply their own timing without changing the collision-free geometry.
        if side == "left":
            smooth_left_waypoints = np.asarray(smooth_path, dtype=np.float64)
            smooth_right_waypoints = np.tile(
                np.asarray(current_right_jp, dtype=np.float64),
                (len(smooth_left_waypoints), 1),
            )
        elif side == "right":
            smooth_left_waypoints = np.tile(
                np.asarray(current_left_jp, dtype=np.float64),
                (len(smooth_path), 1),
            )
            smooth_right_waypoints = np.asarray(smooth_path, dtype=np.float64)
        else:
            smooth_path_arr = np.asarray(smooth_path, dtype=np.float64)
            smooth_left_waypoints = smooth_path_arr[:, :6]
            smooth_right_waypoints = smooth_path_arr[:, 6:]

        # Split into left/right for convenience
        if side == "left":
            left_positions = positions
            right_positions = np.tile(current_right_jp, (len(positions), 1))
        elif side == "right":
            left_positions = np.tile(current_left_jp, (len(positions), 1))
            right_positions = positions
        else:
            left_positions = positions[:, :6]
            right_positions = positions[:, 6:]

        return {
            "status": "Success",
            "position": positions,
            "left_positions": left_positions,
            "right_positions": right_positions,
            "left_waypoints": smooth_left_waypoints,
            "right_waypoints": smooth_right_waypoints,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def main() -> None:
    import time

    planner = YamMotionPlanner()

    home_left = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25])
    home_right = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25])

    # Get current EE pose at home
    l_pos, l_q, r_pos, r_q = planner._kin.forward_kinematics(home_left, home_right)
    print(f"Home left  EE: pos={l_pos}, quat={l_q}")
    print(f"Home right EE: pos={r_pos}, quat={r_q}")

    # Plan: move left arm forward by 5 cm
    target_l_pos = l_pos.copy()
    target_l_pos[0] += 0.05

    print("\nPlanning left arm forward 5 cm...")
    t0 = time.time()
    result = planner.plan_to_pose(
        current_left_jp=home_left,
        current_right_jp=home_right,
        target_left_pos=target_l_pos,
        target_left_quat_xyzw=l_q,
        side="left",
    )
    dt = time.time() - t0
    print(f"Status: {result['status']}, Time: {dt:.3f}s")
    if result["status"] == "Success":
        print(f"Trajectory length: {len(result['position'])} steps")

    # Plan: dual-arm — both arms move 3 cm up
    target_l_pos2 = l_pos.copy()
    target_l_pos2[2] += 0.03
    target_r_pos2 = r_pos.copy()
    target_r_pos2[2] += 0.03

    print("\nPlanning both arms up 3 cm...")
    t0 = time.time()
    result2 = planner.plan_to_pose(
        current_left_jp=home_left,
        current_right_jp=home_right,
        target_left_pos=target_l_pos2,
        target_left_quat_xyzw=l_q,
        target_right_pos=target_r_pos2,
        target_right_quat_xyzw=r_q,
        side="both",
    )
    dt = time.time() - t0
    print(f"Status: {result2['status']}, Time: {dt:.3f}s")
    if result2["status"] == "Success":
        print(f"Trajectory length: {len(result2['position'])} steps")

    # Verify collision-free
    if result["status"] == "Success":
        collisions = 0
        for i, q in enumerate(result["left_positions"]):
            if planner.check_collision(q, home_right):
                collisions += 1
        print(
            f"\nCollision check on left-arm trajectory: {collisions} collisions in {len(result['left_positions'])} steps"
        )


if __name__ == "__main__":
    main()
