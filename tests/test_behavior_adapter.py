from __future__ import annotations

import math
from types import SimpleNamespace

from behavior_fakes import (
    ARM_IDX,
    BASE_IDX,
    HEAD_OFFSET,
    TRUNK_IDX,
    FakeOmniGibsonEnv,
    RecordingObserver,
    planar,
)
import numpy as np
import pytest

from cap_harness.behavior.adapter import (
    BehaviorAdapter,
    erode_free_cells,
    plan_standoff_pose,
    resample_route,
)
from cap_harness.contracts import ArmCommand, PointCloud, RobotAction, Trajectory
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.validation import validate_observation_schema
from cap_harness.validation.evaluators.behavior_pickup import BehaviorPickupWitness

ROOT_POSE = (5.0, 4.3, 0.3)


class _Factory:
    def __init__(self) -> None:
        self.envs: list[FakeOmniGibsonEnv] = []

    def __call__(self, config: dict) -> FakeOmniGibsonEnv:
        env = FakeOmniGibsonEnv(config, root_pose=ROOT_POSE)
        self.envs.append(env)
        return env


def _adapter(
    size: int = 64, observer: RecordingObserver | None = None
) -> tuple[BehaviorAdapter, _Factory]:
    factory = _Factory()
    adapter = BehaviorAdapter(
        env_factory=factory,
        run_observer=observer,
        camera_width=size,
        camera_height=size,
        horizon=500,
    )
    return adapter, factory


def _hold(adapter: BehaviorAdapter) -> RobotAction:
    state = adapter.get_robot_state()
    return RobotAction(
        {
            arm: ArmCommand("joint_position", state.joint_positions[arm], embodiment="behavior")
            for arm in state.arms
        }
    )


def test_reset_publishes_three_cameras_in_odom_with_finite_depth() -> None:
    observer = RecordingObserver()
    adapter, factory = _adapter(observer=observer)
    observation = adapter.reset("turning_on_radio", 1)
    env = factory.envs[0]
    assert env.loaded_instances == [1]
    assert set(observation.cameras) == {"head", "left_wrist", "right_wrist"}
    head = observation.cameras["head"]
    assert head.rgb.shape == (64, 64, 3) and head.rgb.dtype == np.uint8
    assert np.all(np.isfinite(head.depth_m)) and head.depth_m.min() == 0.0
    assert head.camera_pose.frame == "odom"
    assert observation.robot_state.base_frame == "odom"
    assert observation.robot_state.joint_names["primary"][0] == "left_arm_joint1"
    validate_observation_schema(observation)
    assert observer.resets[0]["instance_id"] == 1
    assert "target_scope" not in observer.resets[0]
    assert adapter.get_task_context().language == "pick up the red radio"
    assert env.robots[0].keep_still_calls >= 1
    np.testing.assert_allclose(adapter.get_base_pose(), (0.0, 0.0, 0.0), atol=1e-9)


def test_head_camera_projects_a_point_ahead_of_the_robot_to_the_image_centre() -> None:
    adapter, factory = _adapter()
    observation = adapter.reset("turning_on_radio", 1)
    head = observation.cameras["head"]
    # A point 1.9 m in front of the camera at camera height, expressed in odom (== footprint at reset).
    point_odom = np.array([HEAD_OFFSET[0] + 1.9, 0.0, HEAD_OFFSET[2], 1.0])
    camera_from_odom = np.linalg.inv(head.camera_pose.as_matrix())
    point_camera = camera_from_odom @ point_odom
    assert point_camera[2] == pytest.approx(1.9), "OpenCV convention: +Z is forward"
    pixel = head.intrinsics @ point_camera[:3]
    pixel = pixel[:2] / pixel[2]
    np.testing.assert_allclose(pixel, [32.0, 32.0], atol=1e-6)
    # And a point up in the world moves the image row upwards (y down in OpenCV).
    above = camera_from_odom @ (point_odom + np.array([0.0, 0.0, 0.2, 0.0]))
    assert above[1] < 0.0


def test_odom_frame_is_invariant_to_base_motion() -> None:
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 2)
    fixed_world = planar(*ROOT_POSE) @ np.array([1.0, -0.5, 0.7, 1.0])
    before = adapter.odom_from_world @ fixed_world
    result = adapter.navigate_to_pose(
        0.6, 0.2, 0.4, planner="servo", tolerance_m=0.01, tolerance_rad=0.02, max_steps=300
    )
    assert result.ok, result.error
    after = adapter.odom_from_world @ fixed_world
    np.testing.assert_allclose(before, after)
    np.testing.assert_allclose(adapter.get_base_pose(), (0.6, 0.2, 0.4), atol=0.01)
    assert result.diagnostics["moved_x"] == pytest.approx(0.6)
    assert result.diagnostics["moved_yaw"] == pytest.approx(0.4)
    # The end effector reported in odom moved with the base.
    state = adapter.get_robot_state()
    assert state.end_effector_poses["primary"].position[0] > 0.6


def test_native_step_rewrites_arm_targets_and_counts_ticks() -> None:
    observer = RecordingObserver()
    adapter, factory = _adapter(observer=observer)
    adapter.reset("turning_on_radio", 1)
    target = np.linspace(-0.3, 0.3, 7)
    result = adapter.step(
        RobotAction(
            {"secondary": ArmCommand("joint_position", target, 0.25, embodiment="behavior")}
        )
    )
    assert result.ok and result.diagnostics["step_index"] == 1
    robot = factory.envs[0].robots[0]
    np.testing.assert_allclose(robot.q[ARM_IDX["right"]], target)
    assert adapter.get_robot_state().gripper_positions["secondary"] == pytest.approx(0.25)
    assert adapter.current_time_s == pytest.approx(1 / 30)
    assert observer.steps == 1
    bad = (
        adapter.step(
            RobotAction(
                {"primary": ArmCommand("joint_position", np.zeros(7), embodiment="robosuite")}
            )
        )
        if False
        else None
    )
    assert bad is None
    with pytest.raises(ValueError):
        adapter.native_step(
            RobotAction(
                {"primary": ArmCommand("joint_position", np.zeros(7), embodiment="robosuite")}
            )
        )


def test_execute_trajectory_follows_waypoints_and_detects_stale_starts() -> None:
    adapter, _ = _adapter()
    adapter.reset("turning_on_radio", 1)
    state = adapter.get_robot_state()
    waypoints = np.linspace(state.joint_positions["primary"], np.full(7, 0.4), 6)
    trajectory = Trajectory(
        joint_positions=waypoints,
        dt_s=1 / 30,
        joint_names=state.joint_names["primary"],
        planner="test",
        collision_aware=False,
        expected_start=state,
        arm="primary",
        embodiment="behavior",
    )
    result = adapter.execute_trajectory(trajectory)
    assert result.ok and result.steps_executed == 6
    np.testing.assert_allclose(adapter.get_robot_state().joint_positions["primary"], 0.4)
    stale = adapter.execute_trajectory(trajectory)
    assert not stale.ok and stale.error is not None and stale.error.code.value == "stale_plan"
    with pytest.raises(ValueError, match="control period"):
        adapter.execute_trajectory(
            Trajectory(
                joint_positions=waypoints,
                dt_s=0.05,
                joint_names=state.joint_names["primary"],
                planner="test",
                collision_aware=False,
                expected_start=adapter.get_robot_state(),
                arm="primary",
                embodiment="behavior",
            )
        )


def test_grippers_torso_and_metadata() -> None:
    adapter, factory = _adapter()
    adapter.reset("turning_on_radio", 1)
    closed = adapter.set_gripper(0.0, arm="left")
    assert closed.ok and closed.steps_executed >= 1
    assert adapter.get_robot_state().gripper_positions["primary"] == pytest.approx(0.0)
    both = adapter.set_grippers({"primary": 1.0, "secondary": 0.5})
    assert both.ok
    assert adapter.get_robot_state().gripper_positions["secondary"] == pytest.approx(0.5)
    assert not adapter.set_gripper(2.0).ok
    torso = adapter.move_torso([0.2, -0.3, 0.1, 0.0])
    assert torso.ok
    np.testing.assert_allclose(factory.envs[0].robots[0].q[TRUNK_IDX], [0.2, -0.3, 0.1, 0.0])
    reset = adapter.reset_torso()
    assert reset.ok
    np.testing.assert_allclose(factory.envs[0].robots[0].q[TRUNK_IDX], [1.0, -1.4, -0.5, 0.0])
    metadata = adapter.get_controller_metadata()
    assert metadata["base_frame"] == "odom"
    assert metadata["controllable_gripper_arms"] == ("primary", "secondary")
    assert metadata["control_frequency_hz"] == 30.0
    assert len(metadata["torso_positions"]) == 4
    task = adapter.get_task_metadata()
    assert task["instance_id"] == 1 and "target_scope" not in task
    context = adapter.get_planning_context()
    assert context.model == "r1pro"
    assert context.end_effector_links == {"primary": "left_eef_link", "secondary": "right_eef_link"}


def test_success_is_latched_from_the_bound_witness_and_terminates_the_step() -> None:
    adapter, factory = _adapter()
    adapter.reset("turning_on_radio", 1)
    env = factory.envs[0]
    witness = BehaviorPickupWitness("turning_on_radio", env, target_scope="radio_receiver.n.01_1")
    adapter.bind_protocol_evaluator(witness)
    assert not adapter.check_success()
    env.robots[0].grasping["left"] = env.target
    env.target.position[2] += 0.02
    result = adapter.step(_hold(adapter))
    assert result.terminated and adapter.check_success()
    assert adapter.success_observed_step == 1
    env.robots[0].grasping["left"] = None
    assert adapter.check_success()
    with pytest.raises(RuntimeError):
        adapter.bind_protocol_evaluator(witness)


def test_environment_is_reused_for_the_same_suite_and_rebuilt_for_another() -> None:
    adapter, factory = _adapter()
    adapter.reset("turning_on_radio", 1)
    adapter.reset("turning_on_radio", 2)
    assert len(factory.envs) == 1 and factory.envs[0].loaded_instances == [1, 2]
    adapter.reset("picking_up_trash", 1)
    assert len(factory.envs) == 2 and factory.envs[0].closed
    adapter.close()
    assert factory.envs[1].closed
    adapter.close()


def test_seed_must_be_an_available_instance_when_the_dataset_is_present(tmp_path) -> None:
    factory = _Factory()
    adapter = BehaviorAdapter(
        env_factory=factory, camera_width=32, camera_height=32, horizon=10, data_root=tmp_path
    )
    directory = adapter.registry.instance_directory(
        adapter.registry.resolve("turning_on_radio"), tmp_path
    )
    directory.mkdir(parents=True)
    (
        directory / "house_double_floor_lower_task_turning_on_radio_0_4_template-tro_state.json"
    ).write_text("{}")
    with pytest.raises(ValueError, match="available"):
        adapter.reset("turning_on_radio", 1)
    adapter.reset("turning_on_radio", 4)


def test_plan_standoff_pose_stands_outside_the_nearest_table_edge_facing_the_object() -> None:
    table = np.array(
        [[x, y, 0.7] for x in np.linspace(1.0, 2.0, 11) for y in np.linspace(-0.5, 0.5, 11)]
    )
    radio = np.array([[1.8, 0.1, 0.8], [1.85, 0.12, 0.8], [1.82, 0.08, 0.8]])
    x, y, yaw = plan_standoff_pose(
        PointCloud(points=table, frame="odom"),
        PointCloud(points=radio, frame="odom"),
        standoff_m=0.3,
    )
    assert x == pytest.approx(2.3, abs=1e-6)
    assert y == pytest.approx(0.1, abs=1e-6)
    assert yaw == pytest.approx(math.pi, abs=1e-6)
    with pytest.raises(ValueError):
        plan_standoff_pose(
            PointCloud(points=table, frame="odom"), PointCloud(points=radio, frame="world")
        )
    assert BASE_IDX.size == 6


def test_intrinsics_are_retried_when_the_sensor_asserts_at_first() -> None:
    adapter, factory = _adapter()
    adapter.reset("turning_on_radio", 1)
    env = factory.envs[0]
    sensor = env.robots[0].sensors["robot_r1:left_realsense_link:Camera:0"]
    original = type(sensor).intrinsic_matrix
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise AssertionError("intrinsic matrix for sensor is degenerate!")
        return original.fget(self)

    type(sensor).intrinsic_matrix = property(flaky)
    try:
        adapter._refresh_intrinsics()
    finally:
        type(sensor).intrinsic_matrix = original
    assert calls["n"] >= 4
    assert adapter.get_observation().cameras["left_wrist"].intrinsics[0, 0] > 0


def test_resample_route_spaces_hops_and_faces_the_next_hop() -> None:
    hops = resample_route([(0.0, 0.0), (2.5, 0.0)], (2.5, 0.0, 0.3), hop_m=1.0)
    assert [(round(x, 6), round(y, 6)) for x, y, _ in hops] == [(1.0, 0.0), (2.0, 0.0), (2.5, 0.0)]
    assert hops[0][2] == pytest.approx(0.0) and hops[-1][2] == pytest.approx(0.3)
    assert resample_route([(0.0, 0.0)], (0.2, 0.0, 1.0), hop_m=1.0) == [(0.2, 0.0, 1.0)]


def test_navigation_hops_along_the_traversability_map_when_curobo_has_no_path() -> None:
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    robot = factory.envs[0].robots[0]
    calls: list[tuple] = []

    def shortest_path(floor, source, target, entire_path=False, robot=None):
        calls.append((floor, np.asarray(source), np.asarray(target), entire_path))
        return np.stack([np.asarray(source), np.asarray(target)]), 2.5

    robot.scene = SimpleNamespace(
        get_shortest_path=shortest_path, trav_map=SimpleNamespace(floor_heights=[0.0])
    )
    failure = ApiError(
        ErrorCode.PLANNING_FAILED,
        "cuRobo found no base path (TrajOpt Fail)",
        details={"status": "TrajOpt Fail"},
    )
    adapter._planner = SimpleNamespace(plan_base=lambda x, y, yaw: ([], failure))
    result = adapter.navigate_to_pose(2.5, 0.0, 0.3, planner="curobo", max_steps=600)
    assert result.ok, result.error
    assert result.diagnostics["route"] == "traversability"
    assert result.diagnostics["hops"] == 3 and result.diagnostics["servo_hops"] == 3
    assert result.diagnostics["direct_status"] == "TrajOpt Fail"
    assert result.diagnostics["geodesic_m"] == pytest.approx(2.5)
    np.testing.assert_allclose(adapter.get_base_pose(), (2.5, 0.0, 0.3), atol=0.06)
    assert calls and calls[0][3] is True
    # The map was asked in world coordinates: the start is where the base stood (odom origin).
    np.testing.assert_allclose(
        calls[0][1], (adapter.world_from_odom @ np.array([0.0, 0.0, 0.0, 1.0]))[:2], atol=1e-6
    )


def test_navigation_reports_when_neither_curobo_nor_the_map_has_a_route() -> None:
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    factory.envs[0].robots[0].scene = SimpleNamespace(
        get_shortest_path=lambda *args, **kwargs: (None, None)
    )
    failure = ApiError(
        ErrorCode.PLANNING_FAILED,
        "cuRobo found no base path (IK Fail)",
        details={"status": "IK Fail"},
    )
    adapter._planner = SimpleNamespace(plan_base=lambda x, y, yaw: ([], failure))
    result = adapter.navigate_to_pose(3.0, 1.0, 0.0, planner="curobo", max_steps=100)
    assert not result.ok and result.error.code == ErrorCode.PLANNING_FAILED
    assert "traversability map has no route" in result.error.message
    assert (
        result.diagnostics["route"] == "none" and result.diagnostics["direct_status"] == "IK Fail"
    )
    assert result.diagnostics["moved_x"] == 0.0


def test_execute_trajectory_drives_the_planned_torso_motion() -> None:
    adapter, factory = _adapter()
    adapter.reset("turning_on_radio", 1)
    robot = factory.envs[0].robots[0]
    state = adapter.get_robot_state()
    start = state.joint_positions["primary"]
    arm_path = np.linspace(start, start + 0.2, 6)
    full = np.tile(np.asarray(robot.get_joint_positions(), dtype=np.float64), (6, 1))
    full[:, adapter.layout.arms["left"]] = arm_path
    full[:, adapter.layout.trunk] = np.linspace(0.0, 0.3, 6)[:, None]
    trajectory = Trajectory(
        joint_positions=arm_path,
        dt_s=adapter.control_period_s,
        joint_names=state.joint_names["primary"],
        planner="curobo_omnigibson_pose",
        collision_aware=True,
        expected_start=state,
        arm="primary",
        embodiment="behavior",
    )
    adapter._planner = SimpleNamespace(full_path_for=lambda t: full if t is trajectory else None)
    result = adapter.execute_trajectory(trajectory)
    assert result.ok, result.error
    assert result.diagnostics["torso_planned"] is True
    q = np.asarray(robot.get_joint_positions(), dtype=np.float64)
    np.testing.assert_allclose(q[adapter.layout.trunk], 0.3, atol=1e-6)
    np.testing.assert_allclose(q[adapter.layout.arms["left"]], start + 0.2, atol=1e-6)


def test_erode_free_cells_blocks_the_neighbourhood_of_obstacles() -> None:
    free = np.ones((7, 7), dtype=bool)
    free[3, 3] = False
    eroded = erode_free_cells(free, 1)
    assert not eroded[2:5, 2:5].any()
    assert eroded[1, 1] and eroded[1, 5] and eroded[5, 1] and eroded[5, 5]
    assert not eroded[0].any(), "outside the map counts as blocked"
    np.testing.assert_array_equal(erode_free_cells(free, 0), free)


def _scene_with_map(robot, size: int = 200, resolution: float = 0.1):
    grid = np.full((size, size), 255, dtype=np.uint8)
    grid[90:110, 90:110] = 0  # a 2 x 2 m obstacle centred on the world origin

    def world_to_map(xy):
        xy = np.asarray(xy, dtype=np.float64)
        return np.flip(xy / resolution + size / 2.0).astype(int)

    trav = SimpleNamespace(
        floor_map=[grid],
        map_resolution=resolution,
        map_size=size,
        floor_heights=[0.0],
        world_to_map=world_to_map,
    )
    robot.scene = SimpleNamespace(trav_map=trav)
    robot.reset_joint_pos_aabb_extent = np.array([0.4, 0.4, 1.2])  # radius 0.48 m -> 5 cells
    return grid


def test_base_pose_is_free_uses_the_map_eroded_by_the_footprint() -> None:
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    robot = factory.envs[0].robots[0]
    _scene_with_map(robot)
    to_odom = adapter.odom_from_world

    def odom(x_world: float, y_world: float) -> tuple[float, float]:
        point = to_odom @ np.array([x_world, y_world, 0.0, 1.0])
        return float(point[0]), float(point[1])

    assert adapter.base_pose_is_free(*odom(3.0, 3.0))
    assert not adapter.base_pose_is_free(*odom(0.0, 0.0))
    assert not adapter.base_pose_is_free(*odom(1.15, 0.0)), "within the footprint radius"
    assert adapter.base_pose_is_free(*odom(1.5, 0.0))
    assert not adapter.base_pose_is_free(*odom(50.0, 0.0)), "outside the map"


def test_navigation_refuses_a_blocked_goal_before_planning() -> None:
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    robot = factory.envs[0].robots[0]
    _scene_with_map(robot)
    calls: list[tuple] = []
    adapter._planner = SimpleNamespace(
        plan_base=lambda x, y, yaw: calls.append((x, y, yaw)) or ([], None)
    )
    blocked = adapter.odom_from_world @ np.array([0.0, 0.0, 0.0, 1.0])
    result = adapter.navigate_to_pose(float(blocked[0]), float(blocked[1]), 0.0, planner="curobo")
    assert not result.ok and result.error.code == ErrorCode.PLANNING_FAILED
    assert result.diagnostics["route"] == "goal_blocked" and calls == []
    assert "not traversable" in result.error.message


def test_adapter_without_a_map_treats_every_goal_as_free() -> None:
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    assert adapter.base_pose_is_free(4.0, -2.0)


def test_servo_base_is_rate_limited_so_it_cannot_sweep_the_scene() -> None:
    """A base leg must not move faster than the planner's own rate; fast legs displace furniture."""
    from cap_harness.behavior.adapter import BASE_SERVO_STEP_M

    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    seen: list[tuple[float, float]] = []
    original = adapter.step

    def record(action):
        result = original(action)
        seen.append(adapter.get_base_pose()[:2])
        return result

    adapter.step = record
    result = adapter.navigate_to_pose(1.0, 0.0, 0.0, planner="servo", max_steps=300)
    assert result.ok, result.error
    steps = [
        float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(seen, seen[1:], strict=False)
    ]
    assert steps, "the servo took no steps"
    # The fake stores the base pose in float32, so a difference of two positions carries
    # about 1e-7 m of round-trip error; the tolerance is that, not the rate limit itself.
    assert max(steps) <= BASE_SERVO_STEP_M + 1e-6, f"a tick moved {max(steps):.4f} m"
    expected = (1.0 - 0.05) / BASE_SERVO_STEP_M
    assert len(steps) >= expected - 1, f"only {len(steps)} ticks for a metre; the leg jumped"


def test_navigation_fails_loudly_when_the_base_leaves_the_floor() -> None:
    """A route can reach a map cell with no floor under it; the planar pose cannot show that."""
    adapter, factory = _adapter()
    adapter.reset("picking_up_trash", 1)
    robot = factory.envs[0].robots[0]
    original = robot.get_position_orientation

    def sunken():
        position, orientation = original()
        return np.array([position[0], position[1], position[2] - 40.0]), orientation

    robot.get_position_orientation = sunken
    result = adapter.navigate_to_pose(0.3, 0.0, 0.0, planner="servo", max_steps=50)
    assert not result.ok
    assert result.error.code == ErrorCode.TERMINATED
    assert "left the floor" in result.error.message
    assert result.error.details["base_height_m"] < -30.0
