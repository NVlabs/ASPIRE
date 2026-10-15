from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.libero.adapter import LiberoAdapter
from cap_harness.libero.registry import LiberoTaskMetadata


class FakeModel:
    def __init__(self) -> None:
        self.cam_fovy = np.array([60.0, 45.0])
        self.stat = SimpleNamespace(extent=1.0)
        self.vis = SimpleNamespace(map=SimpleNamespace(znear=0.1, zfar=10.0))

    def body_name2id(self, name: str) -> int:
        return {"robot0_base": 0, "gripper0_eef": 1}[name]

    def camera_name2id(self, name: str) -> int:
        return {"agentview": 0, "robot0_eye_in_hand": 1}[name]


class FakeData:
    def __init__(self) -> None:
        self.xmat = np.stack([np.eye(3), np.eye(3)])
        self.xpos = np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.5]])

    def get_camera_xmat(self, name: str) -> np.ndarray:
        del name
        return np.eye(3)

    def get_camera_xpos(self, name: str) -> np.ndarray:
        return {
            "agentview": np.array([2.0, 0.0, 1.0]),
            "robot0_eye_in_hand": np.array([1.5, 0.0, 0.5]),
        }[name]


class FakeEnv:
    def __init__(self) -> None:
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData())
        self.action_spec = (
            np.array([-0.25] * 7 + [-0.75]),
            np.array([0.25] * 7 + [0.75]),
        )
        self.seed_value: int | None = None
        self.selected_init_state: float | None = None
        self.step_actions: list[np.ndarray] = []
        self.close_calls = 0
        self.success_calls = 0
        self.joints = np.zeros(7)
        self.gripper_m = 0.03
        self.freeze_gripper = False

    def _observation(self) -> dict[str, np.ndarray]:
        agent_rgb = np.array(
            [
                [[10, 0, 0], [11, 0, 0]],
                [[20, 0, 0], [21, 0, 0]],
            ],
            dtype=np.uint8,
        )
        wrist_rgb = agent_rgb + np.array([0, 10, 0], dtype=np.uint8)
        depth = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        return {
            "agentview_image": agent_rgb,
            "agentview_depth": depth[:, :, None],
            "robot0_eye_in_hand_image": wrist_rgb,
            "robot0_eye_in_hand_depth": depth[:, :, None],
            "robot0_joint_pos": self.joints.copy(),
            "robot0_joint_vel": np.full(7, 0.01),
            "robot0_gripper_qpos": np.array([self.gripper_m, -self.gripper_m]),
            # LIBERO / robosuite observations publish this quaternion as XYZW.
            "robot0_eef_pos": np.array([1.2, 0.0, 0.5]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self) -> dict[str, np.ndarray]:
        self.joints = np.zeros(7)
        self.gripper_m = 0.03
        return self._observation()

    def set_init_state(self, state: float) -> dict[str, np.ndarray]:
        self.selected_init_state = state
        self.joints = np.full(7, state)
        return self._observation()

    def step(self, action: np.ndarray):
        native = np.asarray(action, dtype=np.float64).copy()
        self.step_actions.append(native)
        self.joints = self.joints + native[:7] / 20.0
        if not self.freeze_gripper:
            self.gripper_m = float(np.clip(self.gripper_m - native[-1] * 0.01, 0.0, 0.04))
        return (
            self._observation(),
            7.5,
            False,
            {
                "reward": 7.5,
                "success": True,
                "raw_state": [1, 2, 3],
            },
        )

    def check_success(self) -> bool:
        self.success_calls += 1
        return True

    def close(self) -> None:
        self.close_calls += 1


class FakeRegistry:
    def __init__(self, metadata: LiberoTaskMetadata) -> None:
        self.metadata = metadata
        self.init_states = [0.1, 0.2, 0.3]

    def resolve(self, task_ref) -> LiberoTaskMetadata:
        assert task_ref in (
            (self.metadata.suite_name, self.metadata.task_id),
            self.metadata.task_ref,
        )
        return self.metadata

    def get_init_states(self, task_ref) -> list[float]:
        assert task_ref == self.metadata
        return self.init_states


def _build_adapter(
    tmp_path,
    *,
    post_settle_steps: int = 0,
    init_mode: str = "saved",
    init_states: list[float] | None = None,
    env: FakeEnv | None = None,
):
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text("(define (problem task) (:language move the mug))", encoding="utf-8")
    metadata = LiberoTaskMetadata(
        suite_name="libero_goal_task",
        task_id=2,
        task_name="move_mug",
        family="goal",
        bddl_path=bddl_path,
        language="move the blue mug into the drawer",
        init_state_count=3 if init_states is None else len(init_states),
    )
    registry = FakeRegistry(metadata)
    if init_states is not None:
        registry.init_states = init_states
    if env is None:
        env = FakeEnv()
    factory_kwargs = {}

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return env

    adapter = LiberoAdapter(
        registry=registry,
        env_factory=factory,
        control_frequency=20.0,
        camera_height=2,
        camera_width=2,
        post_settle_steps=post_settle_steps,
        depth_converter=lambda sim, depth: depth + 1.0,
        init_mode=init_mode,
    )
    return adapter, env, factory_kwargs


def test_seeded_init_mode_applies_no_saved_state_and_records_mode(tmp_path) -> None:
    """Seeded mode must leave the procedural placement from reset() intact.

    It previously restored init_states[0] after reset(), overwriting the very
    randomization the seed produced, so all 50 held-out seeds rendered one scene.
    """
    adapter, env, _ = _build_adapter(tmp_path, init_mode="seeded")

    observation = adapter.reset(("libero_goal_task", 2), seed=53)

    assert env.seed_value == 53
    assert env.selected_init_state is None, "no saved state may be applied in seeded mode"
    assert observation.task_context.metadata["init_mode"] == "seeded"
    assert observation.task_context.metadata["init_state_index"] is None


def test_seeded_init_mode_works_without_any_saved_states(tmp_path) -> None:
    """Three LIBERO-Pro tasks ship zero saved states; seeded is their only option."""
    adapter, env, _ = _build_adapter(tmp_path, init_mode="seeded", init_states=[])

    observation = adapter.reset(("libero_goal_task", 2), seed=7)

    assert env.seed_value == 7
    assert observation.task_context.metadata["init_state_index"] is None


def test_seeded_layouts_differ_and_repeat_across_fresh_and_reused_envs(tmp_path) -> None:
    """Model native NumPy placement sampling without requiring MuJoCo or a GPU.

    Joint observations stand in for movable-object positions: both reset() and
    set_init_state() write them, so restoring a saved state destroys the layout.
    This checks adapter sequencing, not native simulator/rendering equivalence.
    """

    class PlacementEnv(FakeEnv):
        def seed(self, seed: int) -> None:
            super().seed(seed)
            self.rng = np.random.RandomState(seed)

        def reset(self) -> dict[str, np.ndarray]:
            super().reset()
            self.joints = self.rng.uniform(-0.2, 0.2, size=7)
            return self._observation()

    env = PlacementEnv()
    adapter, _, _ = _build_adapter(tmp_path, init_mode="seeded", env=env)
    layouts = {}
    for seed in (1, 2, 53, 2, 1):
        adapter.reset(("libero_goal_task", 2), seed=seed)
        expected = np.random.RandomState(seed).uniform(-0.2, 0.2, size=7)
        np.testing.assert_array_equal(env.joints, expected)
        if seed in layouts:
            np.testing.assert_array_equal(env.joints, layouts[seed])
        layouts[seed] = env.joints.copy()
        assert env.selected_init_state is None
    assert len({layout.tobytes() for layout in layouts.values()}) == 3
    adapter.close()

    for seed, layout in layouts.items():
        fresh_env = PlacementEnv()
        fresh_adapter, _, _ = _build_adapter(tmp_path, init_mode="seeded", env=fresh_env)
        fresh_adapter.reset(("libero_goal_task", 2), seed=seed)
        np.testing.assert_array_equal(fresh_env.joints, layout)
        fresh_adapter.close()


def test_saved_init_mode_rejects_a_task_with_no_saved_states(tmp_path) -> None:
    """Better a clear error than (seed - 1) %% 0."""
    adapter, _, _ = _build_adapter(tmp_path, init_mode="saved", init_states=[])

    with pytest.raises(ValueError, match="no saved init states"):
        adapter.reset(("libero_goal_task", 2), seed=1)


def test_init_mode_rejects_unknown_value(tmp_path) -> None:
    with pytest.raises(ValueError, match="init_mode"):
        _build_adapter(tmp_path, init_mode="procedural")


def _assert_no_privileged_keys(value) -> None:
    if isinstance(value, dict) or hasattr(value, "items"):
        for key, item in value.items():
            normalized = str(key).lower()
            assert normalized not in {"reward", "success", "raw_state", "sim_state", "qpos"}
            _assert_no_privileged_keys(item)
    elif isinstance(value, tuple | list):
        for item in value:
            _assert_no_privileged_keys(item)


def test_reset_selects_init_state_settles_and_normalizes_observation(tmp_path) -> None:
    adapter, env, factory_kwargs = _build_adapter(tmp_path, post_settle_steps=2)

    observation = adapter.reset(("libero_goal_task", 2), seed=5)

    assert env.seed_value == 5
    assert env.selected_init_state == pytest.approx(0.2)  # (5 - 1) % 3 == 1
    assert len(env.step_actions) == 2
    np.testing.assert_allclose(env.step_actions[0][:-1], 0.0)
    assert env.step_actions[0][-1] == pytest.approx(-0.5)
    assert adapter.current_time_s == 0.0
    assert observation.timestamp_s == 0.0
    assert observation.task_context is not None
    assert observation.task_context.language == "move the blue mug into the drawer"
    assert factory_kwargs["controller"] == "JOINT_POSITION"
    assert factory_kwargs["bddl_file_name"].endswith("task.bddl")

    agent = observation.cameras["agentview"]
    assert agent.rgb[0, 0, 0] == 20
    np.testing.assert_allclose(agent.depth_m, np.array([[1.3, 1.4], [1.1, 1.2]]))
    np.testing.assert_allclose(agent.camera_pose.position, np.array([1.0, 0.0, 1.0]))
    np.testing.assert_allclose(
        np.abs(agent.camera_pose.quaternion_wxyz), np.array([0.0, 1.0, 0.0, 0.0])
    )
    assert agent.camera_pose.frame == "robot_base"
    assert agent.frame == "camera/agentview"

    state = observation.robot_state
    assert state.joint_positions["primary"].shape == (7,)
    assert state.gripper_positions["primary"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        state.end_effector_poses["primary"].quaternion_wxyz,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_public_step_calls_native_once_clamps_and_advances_005_seconds(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    observation = adapter.reset(("libero_goal_task", 2), seed=1)
    current = observation.robot_state.joint_positions["primary"]
    target = current + np.array([0.10, -0.10, 0.01, -0.01, 0.0, 0.02, -0.02])
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=target,
                embodiment="libero",
            ),
        }
    )

    result = adapter.step(action)

    assert len(env.step_actions) == 1
    np.testing.assert_allclose(
        env.step_actions[0],
        np.array([0.25, -0.25, 0.2, -0.2, 0.0, 0.25, -0.25, -0.5]),
    )
    assert adapter.current_time_s == pytest.approx(0.05)
    assert result.observation is not None
    assert result.observation.timestamp_s == pytest.approx(0.05)
    assert result.reward is None
    assert env.success_calls == 0
    _assert_no_privileged_keys(result.diagnostics)
    assert adapter.compute_reward() == pytest.approx(7.5)
    assert adapter.check_success() is True
    assert env.success_calls == 1


def test_native_step_retains_runtime_reward_only(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    observation = adapter.reset(("libero_goal_task", 2), seed=1)
    action = RobotAction(
        arms={
            "primary": ArmCommand(
                mode="joint_position",
                target=observation.robot_state.joint_positions["primary"],
                embodiment="libero",
            )
        }
    )

    runtime_result = adapter.native_step(action)

    assert len(env.step_actions) == 1
    assert runtime_result.reward == pytest.approx(7.5)
    assert runtime_result.diagnostics["native_info"]["success"] is True
    public_result = runtime_result.public_view()
    assert public_result.reward is None
    _assert_no_privileged_keys(public_result.diagnostics)


def test_set_gripper_steps_until_observed_position_converges(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    adapter.reset(("libero_goal_task", 2), seed=1)

    result = adapter.set_gripper(1.0)

    assert result.ok
    assert result.steps_executed == 2
    assert len(env.step_actions) == 2
    np.testing.assert_allclose(env.step_actions[0][:-1], 0.0)
    assert env.step_actions[0][-1] == pytest.approx(-0.75)
    assert result.final_observation is not None
    assert result.final_observation.robot_state.gripper_positions["primary"] == pytest.approx(1.0)
    assert adapter.current_time_s == pytest.approx(0.10)


def test_close_gripper_treats_stable_nonzero_position_as_object_contact(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    adapter.reset(("libero_goal_task", 2), seed=1)
    env.freeze_gripper = True

    result = adapter.close_gripper()

    assert result.ok
    assert result.steps_executed == 5
    assert result.diagnostics["completion"] == "contact_stall"
    assert result.final_observation is not None
    assert result.final_observation.robot_state.gripper_positions["primary"] == pytest.approx(0.75)


def test_execute_trajectory_reissues_each_waypoint_until_converged(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    observation = adapter.reset(("libero_goal_task", 2), seed=1)
    start = observation.robot_state.joint_positions["primary"]
    waypoints = np.stack((start + 0.05, start + 0.10))
    trajectory = Trajectory(
        joint_positions=waypoints,
        dt_s=0.05,
        joint_names=observation.robot_state.joint_names["primary"],
        planner="test",
        collision_aware=False,
        expected_start=observation.robot_state,
        embodiment="libero",
        gripper_positions=np.zeros(2),
    )

    result = adapter.execute_trajectory(trajectory)

    assert result.ok
    assert result.steps_executed > len(waypoints)
    assert result.final_observation is not None
    final_joints = result.final_observation.robot_state.joint_positions["primary"]
    assert np.max(np.abs(final_joints - waypoints[-1])) <= 0.01
    assert all(action[-1] == pytest.approx(0.75) for action in env.step_actions)


def test_execute_trajectory_stops_at_bounded_convergence_timeout(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    env.action_spec = (np.zeros(8), np.zeros(8))
    observation = adapter.reset(("libero_goal_task", 2), seed=1)
    trajectory = Trajectory(
        joint_positions=np.full((1, 7), 0.2),
        dt_s=0.05,
        joint_names=observation.robot_state.joint_names["primary"],
        planner="test",
        collision_aware=False,
        expected_start=observation.robot_state,
        embodiment="libero",
    )

    result = adapter.execute_trajectory(trajectory)

    assert not result.ok
    assert result.steps_executed == 120
    assert result.error is not None
    assert result.error.code.value == "timeout"
    assert result.diagnostics["failed_waypoint"] == 0


def test_metadata_extensions_are_allowlisted_and_close_is_idempotent(tmp_path) -> None:
    adapter, env, _ = _build_adapter(tmp_path)
    adapter.reset(("libero_goal_task", 2), seed=3)

    task_metadata = adapter.get_task_metadata()
    controller_metadata = adapter.get_controller_metadata()

    assert task_metadata["language"] == "move the blue mug into the drawer"
    assert task_metadata["init_state_index"] == 2
    assert task_metadata["seed"] == 3
    assert controller_metadata["control_frequency_hz"] == pytest.approx(20.0)
    assert controller_metadata["action_dimension"] == 8
    assert controller_metadata["camera_names"] == (
        "agentview",
        "robot0_eye_in_hand",
    )
    assert controller_metadata["gripper_semantics"]["normalized"] == "0=closed, 1=open"
    assert adapter.embodiment == "libero"
    _assert_no_privileged_keys(task_metadata)
    _assert_no_privileged_keys(controller_metadata)
    assert "bddl_path" not in task_metadata

    adapter.close()
    adapter.close()
    assert env.close_calls == 1
