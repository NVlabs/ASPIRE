from __future__ import annotations

from types import SimpleNamespace

from behavior_fakes import ARM_IDX, BASE_IDX, JOINT_COUNT, FakeOmniGibsonEnv
import numpy as np

from cap_harness.behavior.adapter import BehaviorAdapter
from cap_harness.behavior.planning import OmniGibsonCuroboPlanner
from cap_harness.contracts import Pose, Trajectory


class _FakeGenerator:
    """Records requests and answers with a straight joint-space path to a synthetic goal."""

    def __init__(self, robot, *, succeed: bool = True) -> None:
        self.robot = robot
        self.succeed = succeed
        self.requests: list[dict] = []
        self.obstacle_updates = 0

    def update_obstacles(self, ignore_objects=None) -> None:
        self.ignored = ignore_objects
        self.obstacle_updates += 1

    def compute_trajectories(self, **kwargs):
        self.requests.append(kwargs)
        start = self.robot.get_joint_positions()
        goal = start.copy()
        for link in kwargs["target_pos"]:
            if link.endswith("_eef_link"):
                arm = link.split("_")[0]
                goal[ARM_IDX[arm]] = 0.7
            else:
                goal[BASE_IDX[[0, 1, 5]]] = [1.0, 0.5, 0.25]
        path = np.linspace(start, goal, 5)
        flags = np.array([self.succeed, False, False])
        paths = [path if self.succeed else None, None, None]
        if kwargs.get("return_full_result"):
            return [
                SimpleNamespace(
                    success=flags,
                    status=None if self.succeed else "TrajOpt Fail",
                    attempts=3,
                    solve_time=0.25,
                    interpolated_plan=object() if self.succeed else None,
                    get_paths=lambda: paths,
                )
            ]
        return flags, paths

    def path_to_joint_trajectory(self, path, get_full_js=True, emb_sel=None):
        return np.asarray(path)

    def add_linearly_interpolated_waypoints(self, trajectory, max_inter_dist=0.01):
        return np.asarray(trajectory)

    def check_collisions(self, q, skip_obstacle_update=False):
        return np.zeros(len(q), dtype=bool)


def _planner(
    succeed: bool = True,
) -> tuple[BehaviorAdapter, OmniGibsonCuroboPlanner, _FakeGenerator]:
    factory = lambda config: FakeOmniGibsonEnv(config)
    adapter = BehaviorAdapter(env_factory=factory, camera_width=32, camera_height=32, horizon=100)
    adapter.reset("turning_on_radio", 1)
    planner = OmniGibsonCuroboPlanner(adapter)
    generator = _FakeGenerator(adapter.robot, succeed=succeed)
    planner._generator = generator
    planner._embodiment = SimpleNamespace(ARM="arm", BASE="base", DEFAULT="default")
    adapter._planner = planner
    return adapter, planner, generator


def _pose(adapter: BehaviorAdapter) -> Pose:
    state = adapter.get_robot_state()
    eef = state.end_effector_poses["primary"]
    return Pose(
        position=eef.position + (0.0, 0.0, 0.15), quaternion_wxyz=eef.quaternion_wxyz, frame="odom"
    )


def test_plan_to_pose_returns_an_arm_trajectory_at_the_control_rate() -> None:
    adapter, planner, generator = _planner()
    state = adapter.get_robot_state()
    result = planner.plan_to_pose(state, _pose(adapter), arm="left", gripper_position=1.0)
    assert result.ok, result.error
    trajectory = result.trajectory
    assert trajectory.arm == "primary" and trajectory.dt_s == adapter.control_period_s
    assert trajectory.joint_positions.shape == (5, 7)
    np.testing.assert_allclose(trajectory.joint_positions[-1], 0.7)
    np.testing.assert_allclose(trajectory.gripper_positions, 1.0)
    request = generator.requests[-1]
    assert list(request["target_pos"]) == ["left_eef_link"]
    assert request["emb_sel"] == "arm" and request["ik_only"] is False
    assert generator.obstacle_updates == 1
    executed = adapter.execute_trajectory(trajectory)
    assert executed.ok


def test_targets_are_expressed_in_the_world_frame_for_curobo() -> None:
    adapter, planner, generator = _planner()
    pose = _pose(adapter)
    planner.plan_to_pose(adapter.get_robot_state(), pose, arm="primary")
    sent = np.asarray(generator.requests[-1]["target_pos"]["left_eef_link"], dtype=np.float64)
    assert sent.shape == (1, 3), "targets are stacked to the cuRobo batch size"
    sent = sent[0]
    expected = (adapter.world_from_odom @ np.append(pose.position, 1.0))[:3]
    np.testing.assert_allclose(sent, expected, atol=1e-5)


def test_solve_ik_uses_the_final_configuration_and_reports_failures() -> None:
    adapter, planner, generator = _planner()
    result = planner.solve_ik(_pose(adapter), adapter.get_robot_state(), arm="primary")
    assert result.ok and result.joint_positions.shape == (7,)
    assert generator.requests[-1]["ik_only"] is True
    _, failing, _ = _planner(succeed=False)
    failed = failing.solve_ik(_pose(adapter), adapter.get_robot_state())
    assert not failed.ok and failed.error.code.value == "ik_failed"
    wrong_frame = Pose(
        position=np.zeros(3), quaternion_wxyz=np.array([1.0, 0, 0, 0]), frame="world"
    )
    assert not planner.plan_to_pose(adapter.get_robot_state(), wrong_frame).ok


def test_plan_base_returns_full_joint_waypoints_in_the_root_frame() -> None:
    adapter, planner, generator = _planner()
    waypoints, error = planner.plan_base(1.0, 0.5, 0.25)
    assert error is None and len(waypoints) == 5
    assert all(row.shape == (JOINT_COUNT,) for row in waypoints)
    assert "base_link" in generator.requests[-1]["target_pos"]
    assert generator.requests[-1]["emb_sel"] == "base"
    result = adapter.navigate_to_pose(
        0.0, 0.0, 0.0, planner="curobo", tolerance_m=2.0, tolerance_rad=3.0
    )
    assert result.ok and result.diagnostics["planner"] == "curobo"


def test_plan_to_joints_checks_collisions_along_the_interpolation() -> None:
    adapter, planner, generator = _planner()
    state = adapter.get_robot_state()
    result = planner.plan_to_joints(state, np.full(7, 0.05), arm="secondary")
    assert result.ok and result.trajectory.joint_positions.shape[1] == 7
    assert result.trajectory.joint_positions.shape[0] >= 6
    generator.check_collisions = lambda q, skip_obstacle_update=False: np.ones(len(q), dtype=bool)
    assert not planner.plan_to_joints(state, np.full(7, 0.05), arm="secondary").ok


def test_synchronized_plans_name_both_arms() -> None:
    adapter, planner, generator = _planner()
    state = adapter.get_robot_state()
    targets = {arm: _pose(adapter) for arm in ("primary", "secondary")}
    result = planner.plan_synchronized_motion(state, targets)
    assert result.ok and result.trajectory.waypoint_count == 5
    assert set(generator.requests[-1]["target_pos"]) == {"left_eef_link", "right_eef_link"}
    partial = planner.plan_synchronized_motion(state, {"primary": targets["primary"]})
    assert not partial.ok


def test_locked_torso_config_adds_the_torso_joints_to_lock_joints(tmp_path) -> None:
    import yaml

    from cap_harness.behavior.planning import locked_torso_config

    source = tmp_path / "r1pro_description_curobo_arm.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "cspace": {"joint_names": ["torso_joint1", "left_arm_joint1"]},
                        "lock_joints": {"base_footprint_x_joint": None},
                        "ee_link": "left_eef_link",
                    }
                }
            }
        )
    )
    written = locked_torso_config(source, tmp_path / "out", ["torso_joint1", "torso_joint2"])
    data = yaml.safe_load(written.read_text())
    locks = data["robot_cfg"]["kinematics"]["lock_joints"]
    assert written.name == "r1pro_description_curobo_arm_locked_torso.yaml"
    assert set(locks) == {"base_footprint_x_joint", "torso_joint1", "torso_joint2"}
    assert all(value is None for value in locks.values())
    assert data["robot_cfg"]["kinematics"]["ee_link"] == "left_eef_link"


def test_pose_plans_ignore_the_object_that_contains_the_goal() -> None:
    adapter, planner, generator = _planner()
    robot = adapter.robot

    class _Obj:
        def __init__(self, name, lower, upper, visual_only=False):
            self.name = name
            self.aabb = (np.array(lower), np.array(upper))
            self.visual_only = visual_only

    state = adapter.get_robot_state()
    goal = _pose(adapter)
    world_goal = (adapter.world_from_odom @ np.append(goal.position, 1.0))[:3]
    inside = _Obj("radio", world_goal - 0.05, world_goal + 0.05)
    below = _Obj(
        "table", world_goal - np.array([0.5, 0.5, 0.6]), world_goal - np.array([-0.5, -0.5, 0.2])
    )
    ghost = _Obj("ghost", world_goal - 0.05, world_goal + 0.05, visual_only=True)
    robot.scene = SimpleNamespace(objects=[robot, inside, below, ghost])
    result = planner.plan_to_pose(state, goal, arm="primary")
    assert result.ok
    assert [o.name for o in generator.ignored] == ["radio"]
    assert planner.last_ignored_objects == ["radio"]
    assert result.diagnostics["ignored_objects"] == ("radio",)


def test_base_plan_failures_carry_curobo_status() -> None:
    adapter, planner, generator = _planner(succeed=False)
    waypoints, error = planner.plan_base(1.0, 0.5, 0.25)
    assert waypoints == [] and error is not None
    assert "TrajOpt Fail" in error.message
    assert error.details["status"] == "TrajOpt Fail" and error.details["attempts"] == 3


def test_base_plans_come_from_the_full_curobo_result() -> None:
    adapter, planner, generator = _planner()
    waypoints, error = planner.plan_base(1.0, 0.5, 0.25)
    assert error is None and len(waypoints) == 5
    assert generator.requests[-1]["return_full_result"] is True


def test_arm_plan_failures_name_the_curobo_status_and_ignored_objects() -> None:
    adapter, planner, generator = _planner(succeed=False)
    result = planner.plan_to_pose(adapter.get_robot_state(), _pose(adapter), arm="primary")
    assert not result.ok
    assert (
        "TrajOpt Fail" in result.error.message and "ignored objects: none" in result.error.message
    )


def test_ik_solutions_that_already_carry_locked_joints_are_accepted() -> None:
    adapter, planner, generator = _planner()
    seen: list[bool] = []

    def convert(path, get_full_js=True, emb_sel=None):
        seen.append(get_full_js)
        if get_full_js:
            raise ValueError("lock_joints is also listed in self.joint_names")
        return np.asarray(path)

    generator.path_to_joint_trajectory = convert
    result = planner.solve_ik(_pose(adapter), adapter.get_robot_state(), arm="primary")
    assert result.ok and seen == [True, False]


def test_planner_remembers_the_full_path_behind_each_trajectory() -> None:
    adapter, planner, generator = _planner()
    state = adapter.get_robot_state()
    result = planner.plan_to_pose(state, _pose(adapter), arm="primary")
    assert result.ok
    full = planner.full_path_for(result.trajectory)
    assert full is not None and full.shape == (5, adapter.layout.joint_count)
    np.testing.assert_allclose(
        full[:, adapter.layout.arms["left"]], result.trajectory.joint_positions
    )
    other = Trajectory(
        joint_positions=result.trajectory.joint_positions + 0.01,
        dt_s=result.trajectory.dt_s,
        joint_names=result.trajectory.joint_names,
        planner="other",
        collision_aware=False,
        expected_start=state,
        arm="primary",
        embodiment=state.embodiment,
    )
    assert planner.full_path_for(other) is None


def _held(name: str, centre: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        aabb=(centre - 0.05, centre + 0.05),
        visual_only=False,
        root_link=SimpleNamespace(get_trimesh_mesh=lambda: object()),
    )


def test_held_objects_ride_with_the_hand_and_leave_the_collision_world() -> None:
    adapter, planner, generator = _planner()
    robot = adapter.robot
    far = np.array([9.0, 9.0, 9.0])
    radio = _held("radio", far)
    robot._ag_obj_in_hand = {"left": radio, "right": None}
    robot.scene = SimpleNamespace(objects=[robot, radio])
    result = planner.plan_to_pose(adapter.get_robot_state(), _pose(adapter), arm="primary")
    assert result.ok
    assert [o.name for o in generator.ignored] == ["radio"]
    assert list(generator.requests[-1]["attached_obj"]) == [str(robot.eef_link_names["left"])]
    assert result.diagnostics["attached_objects"] == ("radio",)
    assert result.diagnostics["ignored_objects"] == ("radio",)


def test_the_object_around_the_hand_is_not_an_obstacle_for_the_next_plan() -> None:
    adapter, planner, generator = _planner()
    robot = adapter.robot
    eef_world = np.asarray(robot.eef_links["right"].get_position_orientation()[0], dtype=float)
    box = _held("box", eef_world)
    robot._ag_obj_in_hand = {}
    robot.scene = SimpleNamespace(objects=[robot, box])
    result = planner.plan_to_pose(adapter.get_robot_state(), _pose(adapter), arm="secondary")
    assert result.ok
    assert [o.name for o in generator.ignored] == ["box"]
    assert result.diagnostics["attached_objects"] == ()


def test_base_plans_carry_the_held_object() -> None:
    adapter, planner, generator = _planner()
    robot = adapter.robot
    can = _held("can", np.array([9.0, 9.0, 9.0]))
    robot._ag_obj_in_hand = {"left": can}
    robot.scene = SimpleNamespace(objects=[robot, can])
    waypoints, error = planner.plan_base(1.0, 0.5, 0.25)
    assert error is None and len(waypoints) == 5
    assert [o.name for o in generator.ignored] == ["can"]
    assert list(generator.requests[-1]["attached_obj"]) == [str(robot.eef_link_names["left"])]
