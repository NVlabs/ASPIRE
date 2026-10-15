"""In-process cuRobo planning for R1 Pro through OmniGibson's ``CuRoboMotionGenerator``.

Implements the harness provider protocols (IK, joint-space trajectories, integrated pose planning)
on the simulator's own robot model and world geometry instead of the HTTP cuRobo service, which
has no R1 Pro model and only a point-cloud view of the scene. Poses arrive in ``odom`` and are
converted to the world frame the OmniGibson wrapper plans in; results come back as 30 Hz
trajectories bound to the arm's measured start state.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.behavior.codec import BehaviorActionCodec
from cap_harness.contracts import (
    IKResult,
    PlanningScene,
    PlanResult,
    Pose,
    RobotPlanningContext,
    RobotState,
    SynchronizedPlanResult,
    SynchronizedTrajectory,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import pose_to_matrix

PLANNER_NAME = "curobo_omnigibson"
DEFAULT_BATCH_SIZE = 1
"""One cuRobo seed batch: three seeds triple the warm-up memory next to Isaac on a 24 GB GPU."""
DEFAULT_USE_CUDA_GRAPH = False
"""CUDA graphs save milliseconds per plan but a failed capture (e.g. under memory pressure) poisons
every later allocation with "Global alloc not supported yet"; planning is not the bottleneck."""
COLLISION_ACTIVATION_DISTANCE_M = 0.02
PLAN_TIMEOUT_S = 60.0
MAX_PLANNING_ATTEMPTS = 100
MAX_IK_FAILURES_BEFORE_RETURN = 50
WAYPOINT_INTERPOLATION_RAD = 0.01
FULL_PATH_MEMORY = 8
GOAL_CONTAINMENT_MARGIN_M = 0.02
"""A scene object whose bounding box contains a pose goal (plus this margin) is the object the
arm is reaching into, so it is left out of the collision world for that plan, as OmniGibson's own
grasp primitive does with its target. Tables and floors never contain a goal above them.
"""
BASE_PRISMATIC_LIMIT_M = 12.0
"""cuRobo bound on the base's virtual x/y joints, measured from the articulation root.

OmniGibson leaves that root at the world origin and loads a task instance into the joints, so the
upstream ±5 m bound cannot even reach the robot's own start pose in a house scene; 12 m covers the
scenes of the supported tasks. Override with CAP_HARNESS_BEHAVIOR_BASE_LIMIT_M.
"""


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def locked_torso_config(
    source: str | os.PathLike[str], output_dir: str | os.PathLike[str], torso_joints: Any
) -> Path:
    """Write a copy of an embodiment yaml with the torso joints added to ``lock_joints``.

    cuRobo's R1 Pro arm embodiment leaves the four torso joints free, so an "arm" plan may also
    lean the torso; our ``Trajectory`` carries only the arm, so that motion would be lost on
    execution. Locking the torso makes arm plans pure arm motion; the wrapper re-syncs locked
    joints with the live state before every plan, so the torso stays where the program put it.
    """
    import yaml

    source = Path(source)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    kinematics = data["robot_cfg"]["kinematics"]
    locks = dict(kinematics.get("lock_joints") or {})
    for name in torso_joints:
        locks.setdefault(str(name), None)
    kinematics["lock_joints"] = locks
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}_locked_torso{source.suffix}"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def _tensor(values: np.ndarray) -> Any:
    """float32 torch tensor when torch is importable, else the numpy array (unit tests)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only without torch
        return np.asarray(values, dtype=np.float32)
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def _batched(values: np.ndarray, batch_size: int) -> Any:
    """Stack one target ``batch_size`` times, the layout OmniGibson's primitives send to cuRobo."""
    array = np.asarray(values, dtype=np.float32)
    return _tensor(np.repeat(array[None, :], max(1, int(batch_size)), axis=0))


def _full_result_outcome(results: Any) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    """Success flags, per-target paths and a failure summary from cuRobo MotionGenResult objects."""
    flags: list[np.ndarray] = []
    paths: list[Any] = []
    statuses: list[str] = []
    attempts = 0
    solve_time = 0.0
    for result in list(results):
        success = _to_numpy(result.success).astype(bool).reshape(-1)
        batch_paths = (
            list(result.get_paths())
            if getattr(result, "interpolated_plan", None) is not None
            else [None] * len(success)
        )
        flags.append(success)
        paths.extend(batch_paths[: len(success)])
        status = getattr(result, "status", None)
        if status is not None:
            statuses.append(str(getattr(status, "value", status)))
        attempts += int(getattr(result, "attempts", 0) or 0)
        solve_time += float(getattr(result, "solve_time", 0.0) or 0.0)
    merged = np.concatenate(flags) if flags else np.zeros(0, dtype=bool)
    outcome = {
        "status": statuses[0] if statuses else ("ok" if merged.any() else "unknown"),
        "attempts": attempts,
        "solve_time_s": round(solve_time, 3),
    }
    return merged, paths, outcome


def _link_position(link: Any) -> np.ndarray | None:
    """World position of a simulator link, or None when it cannot be read."""
    if link is None:
        return None
    try:
        return _to_numpy(link.get_position_orientation()[0]).astype(np.float64)[:3]
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def _object_bounds(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame axis-aligned bounds of a scene object, or None when it has none."""
    aabb = getattr(obj, "aabb", None)
    if aabb is None:
        return None
    try:
        lower = _to_numpy(aabb[0]).astype(np.float64).reshape(3)
        upper = _to_numpy(aabb[1]).astype(np.float64).reshape(3)
    except (TypeError, ValueError, IndexError):
        return None
    return lower, upper


def _to_cpu_float(trajectory: Any) -> Any:
    """Move a cuRobo path to the CPU before interpolating it.

    OmniGibson does the same (``.cpu().float()``): interpolating on the GPU allocates while cuRobo
    may still hold a CUDA-graph capture open, which fails with "Global alloc not supported yet".
    """
    if hasattr(trajectory, "detach"):
        return trajectory.detach().cpu().float()
    return np.asarray(trajectory, dtype=np.float32)


def _matrix_to_xyzw(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from cap_harness.geometry import matrix_to_quaternion_wxyz

    wxyz = matrix_to_quaternion_wxyz(transform[:3, :3])
    return transform[:3, 3].copy(), np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


class OmniGibsonCuroboPlanner:
    """Bind cuRobo to the adapter's live robot; construction is lazy and warm-up happens once."""

    def __init__(
        self,
        adapter: Any,
        *,
        batch_size: int | None = None,
        use_cuda_graph: bool | None = None,
    ) -> None:
        self._adapter = adapter
        env_batch = os.environ.get("CAP_HARNESS_BEHAVIOR_CUROBO_BATCH")
        env_graph = os.environ.get("CAP_HARNESS_BEHAVIOR_CUROBO_CUDA_GRAPH")
        self._batch_size = int(
            batch_size if batch_size is not None else (env_batch or DEFAULT_BATCH_SIZE)
        )
        if use_cuda_graph is None:
            use_cuda_graph = (
                env_graph.strip().lower() in {"1", "true", "yes"}
                if env_graph
                else DEFAULT_USE_CUDA_GRAPH
            )
        self._use_cuda_graph = bool(use_cuda_graph)
        # Upstream's arm embodiment plans with the four torso joints active (only the base and
        # the fingers are locked); the torso is what brings a table top or the floor within reach
        # of the hanging arms. The planned torso motion is executed alongside the arm (the adapter
        # asks ``full_path_for``); CAP_HARNESS_BEHAVIOR_LOCK_TORSO=1 restores arm-only plans.
        env_lock = os.environ.get("CAP_HARNESS_BEHAVIOR_LOCK_TORSO", "0").strip().lower()
        self._lock_torso = env_lock in {"1", "true", "yes"}
        self._generator: Any | None = None
        self._embodiment: Any | None = None
        self.last_ignored_objects: list[str] = []
        self.last_attached_objects: list[str] = []
        self._full_paths: collections.OrderedDict[bytes, np.ndarray] = collections.OrderedDict()

    # ------------------------------------------------------------------ native access
    def _generator_for(self) -> tuple[Any, Any]:
        if self._generator is None:
            import omnigibson.action_primitives.curobo as curobo_module
            from omnigibson.action_primitives.curobo import (
                CuRoboEmbodimentSelection,
                CuRoboMotionGenerator,
            )

            # The planar base joints are clamped to ±5 m around the articulation root upstream,
            # too little for a room-to-room task; the macro is read when the generator is built.
            try:
                curobo_module.m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT = float(
                    os.environ.get("CAP_HARNESS_BEHAVIOR_BASE_LIMIT_M", BASE_PRISMATIC_LIMIT_M)
                )
            except AttributeError:  # pragma: no cover - macro already read and locked
                pass

            robot = self._adapter.robot
            config_paths = None
            if self._lock_torso:
                paths = dict(robot.curobo_path)
                joint_names = list(robot.joints.keys())
                trunk_names = [joint_names[int(i)] for i in _to_numpy(robot.trunk_control_idx)]
                output_dir = (
                    Path(os.environ.get("OMNIGIBSON_APPDATA_PATH", os.path.expanduser("~/.cache")))
                    / "cap-harness-curobo"
                )
                paths[CuRoboEmbodimentSelection.ARM] = str(
                    locked_torso_config(
                        paths[CuRoboEmbodimentSelection.ARM], output_dir, trunk_names
                    )
                )
                config_paths = paths
            self._generator = CuRoboMotionGenerator(
                robot=robot,
                robot_cfg_path=config_paths,
                batch_size=self._batch_size,
                use_cuda_graph=self._use_cuda_graph,
                collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
            )
            self._embodiment = CuRoboEmbodimentSelection
        return self._generator, self._embodiment

    def _attached_object(self, native_arm: str) -> Mapping[str, Any] | None:
        """The object held by ``native_arm``, keyed by its eef link, if its mesh is available."""
        robot = self._adapter.robot
        held = getattr(robot, "_ag_obj_in_hand", {}).get(native_arm)
        if held is None:
            return None
        try:
            held.root_link.get_trimesh_mesh()
        except Exception:
            return None
        return {str(robot.eef_link_names[native_arm]): held.root_link}

    def _start_in_collision(self, generator: Any, embodiment: Any) -> bool:
        """Whether the robot's current configuration is itself in collision.

        A plan refused from such a state looks exactly like an unreachable goal, and no goal is
        reachable until the arm is servoed out of it.
        """
        check = getattr(generator, "check_collisions", None)
        if check is None:
            return False
        try:
            q = _to_numpy(self._adapter.robot.get_joint_positions()).astype(np.float64)
            verdict = _to_numpy(check(_batched(q, 1), skip_obstacle_update=True))
            return bool(np.asarray(verdict).reshape(-1)[0])
        except Exception:
            return False

    def _held_objects(self) -> list[Any]:
        """Objects assisted-grasped by either hand (they ride with the hand, never obstacles)."""
        robot = self._adapter.robot
        held = getattr(robot, "_ag_obj_in_hand", None) or {}
        return [obj for obj in held.values() if obj is not None]

    def _eef_world_positions(self, native_arms: list[str]) -> list[np.ndarray]:
        robot = self._adapter.robot
        links = getattr(robot, "eef_links", None) or {}
        points = [_link_position(links.get(native)) for native in native_arms]
        return [point for point in points if point is not None]

    def _plan_exclusions(self, native_arms: list[str], goal_points: list[np.ndarray]) -> list[Any]:
        """Objects left out of the collision world for one plan: the object a goal reaches into,
        the object the hand is currently inside (a grasp just made or just missed), and anything
        held by either hand.
        """
        candidates = self._objects_containing(goal_points + self._eef_world_positions(native_arms))
        candidates.extend(self._held_objects())
        unique: list[Any] = []
        for obj in candidates:
            if not any(obj is seen for seen in unique):
                unique.append(obj)
        return unique

    def _objects_containing(self, world_points: list[np.ndarray]) -> list[Any]:
        """Scene objects whose axis-aligned bounds contain any of ``world_points``."""
        robot = self._adapter.robot
        scene = getattr(robot, "scene", None)
        objects = list(getattr(scene, "objects", []) or [])
        found: list[Any] = []
        for obj in objects:
            if obj is robot or getattr(obj, "visual_only", False):
                continue
            bounds = _object_bounds(obj)
            if bounds is None:
                continue
            lower = bounds[0] - GOAL_CONTAINMENT_MARGIN_M
            upper = bounds[1] + GOAL_CONTAINMENT_MARGIN_M
            for point in world_points:
                if bool(np.all(point >= lower) and np.all(point <= upper)):
                    found.append(obj)
                    break
        return found

    def _world_targets(self, targets: Mapping[str, Pose]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Map public arm → odom pose into eef-link → world position / xyzw quaternion tensors."""
        codec: BehaviorActionCodec = self._adapter._require_codec()
        robot = self._adapter.robot
        world_from_odom = self._adapter.world_from_odom
        positions: dict[str, Any] = {}
        quaternions: dict[str, Any] = {}
        for arm, pose in targets.items():
            native = codec.native_arm(arm)
            link = str(robot.eef_link_names[native])
            world = world_from_odom @ pose_to_matrix(pose)
            position, xyzw = _matrix_to_xyzw(world)
            positions[link] = _batched(position, self._batch_size)
            quaternions[link] = _batched(xyzw, self._batch_size)
        return positions, quaternions

    def _compute(
        self,
        targets: Mapping[str, Pose],
        *,
        embodiment: Any,
        attached: Mapping[str, Any] | None,
        ik_only: bool,
        initial_joint_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, str | None]:
        """Run cuRobo once; return the full joint-space path (T, D) or a failure message."""
        generator, _ = self._generator_for()
        positions, quaternions = self._world_targets(targets)
        world_from_odom = self._adapter.world_from_odom
        goal_points = [
            (world_from_odom @ np.append(np.asarray(pose.position, dtype=np.float64), 1.0))[:3]
            for pose in targets.values()
        ]
        codec_for_arms: BehaviorActionCodec = self._adapter._require_codec()
        ignored = self._plan_exclusions(
            [codec_for_arms.native_arm(arm) for arm in targets], goal_points
        )
        try:
            generator.update_obstacles(ignore_objects=ignored or None)
            response = generator.compute_trajectories(
                target_pos=positions,
                target_quat=quaternions,
                initial_joint_pos=initial_joint_pos,
                is_local=False,
                max_attempts=math.ceil(MAX_PLANNING_ATTEMPTS / self._batch_size),
                timeout=PLAN_TIMEOUT_S,
                ik_fail_return=MAX_IK_FAILURES_BEFORE_RETURN,
                enable_finetune_trajopt=True,
                finetune_attempts=1,
                # IK-only answers are cuRobo IKResults without a status; full plans carry
                # MotionGenStatus, which tells reach (IK Fail) from a blocked path (TrajOpt Fail).
                return_full_result=not ik_only,
                success_ratio=1.0 / max(1, self._batch_size),
                attached_obj=attached,
                skip_obstacle_update=True,
                ik_only=ik_only,
                ik_world_collision_check=True,
                emb_sel=embodiment,
            )
        except Exception as exc:
            return None, f"cuRobo raised {type(exc).__name__}: {exc}"
        if ik_only:
            successes, paths = response
            flags = _to_numpy(successes).astype(bool).reshape(-1)
            outcome = {"status": "IK Fail"}
        else:
            flags, paths, outcome = _full_result_outcome(response)
        if not flags.any():
            ignored_names = ", ".join(str(getattr(obj, "name", obj)) for obj in ignored) or "none"
            status = str(outcome["status"])
            if "Start State" in status:
                # cuRobo says so itself: nothing plans from a configuration already in collision,
                # including a straight retreat, so the caller must servo out rather than replan.
                return None, (
                    "the arm's current configuration is in collision, so no plan is possible from "
                    "it; servo out with go_home(), or move_to_joints to a configuration you stored "
                    f"earlier, before planning again ({status}; ignored objects: {ignored_names})"
                )
            if self._start_in_collision(generator, embodiment):
                # Advisory only. The probe runs on cuRobo's default embodiment and does not see
                # this plan's exclusion list, so a correct pregrasp over the target can read as a
                # collision; seed 26 of picking_up_trash measured exactly that false positive.
                return None, (
                    f"cuRobo found no collision-free solution ({status}; ignored objects: "
                    f"{ignored_names}). A collision probe also flags the arm's current "
                    "configuration, which may be a false positive over the target: if a retreat "
                    "also fails to plan, servo out with go_home() rather than replanning"
                )
            return None, (
                f"cuRobo found no collision-free solution ({status}; "
                f"ignored objects: {ignored_names})"
            )
        index = int(np.argmax(flags))
        path = paths[index]
        try:
            try:
                trajectory = _to_cpu_float(
                    generator.path_to_joint_trajectory(path, get_full_js=True, emb_sel=embodiment)
                )
            except ValueError as exc:
                # cuRobo's IK solutions already carry the locked (torso) joints, so appending
                # them again is refused; the solution is then complete as it is.
                if not ik_only or "lock_joints" not in str(exc):
                    raise
                trajectory = _to_cpu_float(
                    generator.path_to_joint_trajectory(path, get_full_js=False, emb_sel=embodiment)
                )
            if not ik_only:
                trajectory = generator.add_linearly_interpolated_waypoints(
                    trajectory, max_inter_dist=WAYPOINT_INTERPOLATION_RAD
                )
        except Exception as exc:
            return None, f"cuRobo path conversion failed: {type(exc).__name__}: {exc}"
        array = _to_numpy(trajectory).astype(np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
            return None, "cuRobo returned an empty or non-finite trajectory"
        self.last_ignored_objects = [str(getattr(obj, "name", obj)) for obj in ignored]
        self.last_attached_objects = (
            [str(getattr(obj, "name", obj)) for obj in self._held_objects()] if attached else []
        )
        return array, None

    # ------------------------------------------------------------------ base planning
    def plan_base(self, x: float, y: float, yaw: float) -> tuple[list[np.ndarray], ApiError | None]:
        """Plan the holonomic base to a root-frame planar pose; waypoints are full joint vectors."""
        generator, embodiment = self._generator_for()
        from cap_harness.behavior.adapter import _planar_transform

        robot = self._adapter.robot
        world = self._adapter.world_from_root @ _planar_transform(x, y, yaw)
        # A planar base cannot change its height: the target keeps the footprint's current world
        # z (OmniGibson's own navigation does the same), or IK fails and no path is ever found.
        from cap_harness.behavior.adapter import _pose_matrix

        world[2, 3] = float(_pose_matrix(*robot.get_position_orientation())[2, 3])
        position, xyzw = _matrix_to_xyzw(world)
        link = str(robot.base_footprint_link_name)
        held = self._held_objects()
        attached: dict[str, Any] = {}
        for native in getattr(robot, "arm_names", ()):
            bundle = self._attached_object(str(native))
            if bundle:
                attached.update(bundle)
        try:
            generator.update_obstacles(ignore_objects=held or None)
            results = generator.compute_trajectories(
                target_pos={link: _batched(position, self._batch_size)},
                target_quat={link: _batched(xyzw, self._batch_size)},
                attached_obj=attached or None,
                is_local=False,
                max_attempts=math.ceil(MAX_PLANNING_ATTEMPTS / self._batch_size),
                timeout=PLAN_TIMEOUT_S,
                ik_fail_return=MAX_IK_FAILURES_BEFORE_RETURN,
                enable_finetune_trajopt=True,
                finetune_attempts=1,
                return_full_result=True,
                success_ratio=1.0 / max(1, self._batch_size),
                skip_obstacle_update=True,
                emb_sel=embodiment.BASE,
            )
            flags, paths, outcome = _full_result_outcome(results)
            if not flags.any():
                return [], ApiError(
                    ErrorCode.PLANNING_FAILED,
                    f"cuRobo found no base path ({outcome['status']})",
                    details=outcome,
                )
            trajectory = _to_cpu_float(
                generator.path_to_joint_trajectory(
                    paths[int(np.argmax(flags))], get_full_js=True, emb_sel=embodiment.BASE
                )
            )
            trajectory = generator.add_linearly_interpolated_waypoints(
                trajectory, max_inter_dist=WAYPOINT_INTERPOLATION_RAD
            )
        except Exception as exc:
            return [], ApiError(
                ErrorCode.PLANNING_FAILED,
                f"cuRobo base planning failed: {exc}",
                details={"exception_type": type(exc).__name__},
            )
        array = _to_numpy(trajectory).astype(np.float64)
        if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
            return [], ApiError(ErrorCode.PLANNING_FAILED, "cuRobo returned an invalid base path")
        return [row.copy() for row in array], None

    # ------------------------------------------------------------------ provider protocols
    def solve_ik(
        self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary"
    ) -> IKResult:
        arm = self._adapter.resolve_arm(arm)
        error = self._validate(robot_state, target_pose, arm)
        if error is not None:
            return IKResult(ok=False, arm=arm, error=error, embodiment=robot_state.embodiment)
        _, embodiment = self._generator_for()
        codec: BehaviorActionCodec = self._adapter._require_codec()
        native = codec.native_arm(arm)
        # `CapApi.solve_ik` substitutes any caller-supplied seed into the robot state's arm
        # joints, so the seed arrives here as `robot_state`. Build cuRobo's start configuration
        # from the live robot and overwrite this arm with it; without this the seed is ignored and
        # the returned solution can sit a radian away on a different IK branch.
        seed = _to_numpy(self._adapter.robot.get_joint_positions()).astype(np.float64).copy()
        seed = codec.apply_arm(
            seed, arm, np.asarray(robot_state.joint_positions[arm], dtype=np.float64)
        )
        path, failure = self._compute(
            {arm: target_pose},
            embodiment=embodiment.ARM,
            attached=self._attached_object(native),
            ik_only=True,
            initial_joint_pos=_batched(seed, self._batch_size),
        )
        if path is None:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=robot_state.embodiment,
                error=ApiError(
                    ErrorCode.IK_FAILED, failure or "IK failed", details={"provider": PLANNER_NAME}
                ),
            )
        joints = codec.arm_joints(path[-1], arm)
        return IKResult(
            ok=True,
            joint_positions=joints,
            arm=arm,
            embodiment=robot_state.embodiment,
            diagnostics={"provider": PLANNER_NAME, "collision_aware": True},
        )

    def plan_to_joints(
        self,
        robot_state: RobotState,
        target_joints: np.ndarray,
        *,
        arm: str = "primary",
        gripper_position: float | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> PlanResult:
        arm = self._adapter.resolve_arm(arm)
        error = self._validate(robot_state, np.asarray(target_joints, dtype=np.float64), arm)
        if error is not None:
            return PlanResult(ok=False, error=error)
        # Joint goals are planned as a pose goal at the target configuration's forward kinematics
        # is not available here, so drive a straight joint-space interpolation that cuRobo checks
        # for collisions; the sim executes it at the control rate.
        start = robot_state.joint_positions[arm]
        goal = np.asarray(target_joints, dtype=np.float64)
        steps = max(2, int(np.ceil(np.max(np.abs(goal - start)) / WAYPOINT_INTERPOLATION_RAD)) + 1)
        waypoints = np.linspace(start, goal, steps)
        collision_free, message = self._check_arm_path(waypoints, arm)
        if not collision_free:
            return PlanResult(
                ok=False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED, message, details={"provider": PLANNER_NAME}
                ),
            )
        return PlanResult(
            ok=True,
            trajectory=self._trajectory(robot_state, waypoints, arm, gripper_position, "cspace"),
            diagnostics={"provider": PLANNER_NAME, "planning_mode": "cspace", "waypoints": steps},
        )

    def _check_arm_path(self, waypoints: np.ndarray, arm: str) -> tuple[bool, str]:
        generator, _ = self._generator_for()
        codec: BehaviorActionCodec = self._adapter._require_codec()
        try:
            current = _to_numpy(self._adapter.robot.get_joint_positions()).astype(np.float64)
            full = np.repeat(current[None, :], len(waypoints), axis=0)
            full[:, self._adapter.layout.arms[codec.native_arm(arm)]] = waypoints
            generator.update_obstacles()
            collisions = generator.check_collisions(_tensor(full), skip_obstacle_update=True)
            flags = _to_numpy(collisions).astype(bool).reshape(-1)
        except Exception as exc:
            return False, f"cuRobo collision check failed: {type(exc).__name__}: {exc}"
        if flags.any():
            return (
                False,
                f"joint-space path collides at {int(flags.sum())} of {len(flags)} waypoints",
            )
        return True, ""

    def plan_to_pose(
        self,
        robot_state: RobotState,
        target_pose: Pose,
        *,
        arm: str = "primary",
        gripper_position: float | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> PlanResult:
        arm = self._adapter.resolve_arm(arm)
        error = self._validate(robot_state, target_pose, arm)
        if error is not None:
            return PlanResult(ok=False, error=error)
        _, embodiment = self._generator_for()
        codec: BehaviorActionCodec = self._adapter._require_codec()
        native = codec.native_arm(arm)
        path, failure = self._compute(
            {arm: target_pose},
            embodiment=embodiment.ARM,
            attached=self._attached_object(native),
            ik_only=False,
        )
        if path is None:
            return PlanResult(
                ok=False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED,
                    failure or "planning failed",
                    details={"provider": PLANNER_NAME},
                ),
            )
        waypoints = np.stack([codec.arm_joints(row, arm) for row in path])
        return PlanResult(
            ok=True,
            trajectory=self._trajectory(
                robot_state, waypoints, arm, gripper_position, "pose", full_path=path
            ),
            diagnostics={
                "provider": PLANNER_NAME,
                "planning_mode": "pose",
                "waypoints": int(len(waypoints)),
                "time_dilation_factor": time_dilation_factor,
                "ignored_objects": tuple(self.last_ignored_objects),
                "attached_objects": tuple(self.last_attached_objects),
            },
        )

    def plan_synchronized_motion(
        self,
        robot_state: RobotState,
        targets: Mapping[str, Pose],
        *,
        gripper_positions: Mapping[str, float] | None = None,
        scene: PlanningScene | None = None,
        context: RobotPlanningContext | None = None,
        time_dilation_factor: float | None = None,
        interpolation_dt_s: float | None = None,
        maximum_trajectory_dt_s: float | None = None,
    ) -> SynchronizedPlanResult:
        resolved = {self._adapter.resolve_arm(arm): pose for arm, pose in targets.items()}
        if set(resolved) != set(robot_state.arms):
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(ErrorCode.INVALID_REQUEST, "targets must name every robot arm"),
            )
        for arm, pose in resolved.items():
            if not isinstance(pose, Pose):
                return SynchronizedPlanResult(
                    ok=False,
                    error=ApiError(
                        ErrorCode.UNSUPPORTED, "synchronized planning requires pose targets"
                    ),
                )
            error = self._validate(robot_state, pose, arm)
            if error is not None:
                return SynchronizedPlanResult(ok=False, error=error)
        _, embodiment = self._generator_for()
        codec: BehaviorActionCodec = self._adapter._require_codec()
        attached: dict[str, Any] = {}
        for arm in resolved:
            held = self._attached_object(codec.native_arm(arm))
            if held:
                attached.update(held)
        path, failure = self._compute(
            resolved, embodiment=embodiment.ARM, attached=attached or None, ik_only=False
        )
        if path is None:
            return SynchronizedPlanResult(
                ok=False,
                error=ApiError(
                    ErrorCode.PLANNING_FAILED,
                    failure or "planning failed",
                    details={"provider": PLANNER_NAME},
                ),
            )
        positions = {
            arm: np.stack([codec.arm_joints(row, arm) for row in path]) for arm in resolved
        }
        grippers = gripper_positions or robot_state.gripper_positions
        count = len(path)
        self._remember_full_path(positions, path)
        trajectory = SynchronizedTrajectory(
            joint_positions=positions,
            dt_s=self._adapter.control_period_s,
            joint_names={arm: robot_state.joint_names[arm] for arm in resolved},
            planner=PLANNER_NAME,
            collision_aware=True,
            expected_start=robot_state,
            gripper_positions={arm: np.full(count, float(grippers[arm])) for arm in resolved},
            embodiment=robot_state.embodiment,
        )
        return SynchronizedPlanResult(
            ok=True,
            trajectory=trajectory,
            diagnostics={
                "provider": PLANNER_NAME,
                "waypoints": count,
                "ignored_objects": tuple(self.last_ignored_objects),
                "attached_objects": tuple(self.last_attached_objects),
            },
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _path_key(positions: Mapping[str, np.ndarray]) -> bytes:
        return b"".join(
            np.ascontiguousarray(positions[arm], dtype=np.float64).tobytes()
            for arm in sorted(positions)
        )

    def _remember_full_path(self, positions: Mapping[str, np.ndarray], path: np.ndarray) -> None:
        """Keep the full 28-DoF path behind an arm-only trajectory so execution can move the torso."""
        self._full_paths[self._path_key(positions)] = np.asarray(path, dtype=np.float64).copy()
        while len(self._full_paths) > FULL_PATH_MEMORY:
            self._full_paths.popitem(last=False)

    def full_path_for(self, trajectory: Trajectory | SynchronizedTrajectory) -> np.ndarray | None:
        """The full joint-space path this planner produced for ``trajectory``, if it is one of ours."""
        if isinstance(trajectory, Trajectory):
            positions = {trajectory.arm: trajectory.joint_positions}
        else:
            positions = dict(trajectory.joint_positions)
        return self._full_paths.get(self._path_key(positions))

    def _trajectory(
        self,
        robot_state: RobotState,
        waypoints: np.ndarray,
        arm: str,
        gripper_position: float | None,
        mode: str,
        full_path: np.ndarray | None = None,
    ) -> Trajectory:
        gripper = robot_state.gripper_positions[arm]
        if gripper_position is not None:
            gripper = float(gripper_position)
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if full_path is not None:
            self._remember_full_path({arm: waypoints}, full_path)
        return Trajectory(
            joint_positions=np.asarray(waypoints, dtype=np.float64),
            dt_s=self._adapter.control_period_s,
            joint_names=robot_state.joint_names[arm],
            planner=f"{PLANNER_NAME}_{mode}",
            collision_aware=True,
            expected_start=robot_state,
            arm=arm,
            gripper_positions=np.full(len(waypoints), gripper),
            embodiment=robot_state.embodiment,
        )

    @staticmethod
    def _validate(state: RobotState, target: Pose | np.ndarray, arm: str) -> ApiError | None:
        if not isinstance(state, RobotState):
            return ApiError(ErrorCode.INVALID_REQUEST, "robot_state is invalid")
        if arm not in state.joint_positions:
            return ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}")
        if isinstance(target, Pose):
            if target.frame != state.base_frame:
                return ApiError(
                    ErrorCode.INVALID_REQUEST, "target pose and robot state frames do not match"
                )
            return None
        joints = np.asarray(target, dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            return ApiError(ErrorCode.INVALID_REQUEST, "joint target must contain 7 finite values")
        return None


__all__ = ["PLANNER_NAME", "OmniGibsonCuroboPlanner"]
