from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.contracts import (
    ArmCommand,
    Pose,
    RobotAction,
    RobotState,
    SynchronizedTrajectory,
    Trajectory,
)
from cap_harness.errors import ErrorCode
from cap_harness.robosuite.adapter import BASE_FRAME, RobosuiteAdapter
from cap_harness.robosuite.codec import RobosuiteActionCodec
from cap_harness.robosuite.registry import (
    ROBOSUITE_TASKS,
    RobosuiteRegistryError,
    RobosuiteTaskRegistry,
)
from cap_harness.validation.selectors import _generic_action as _action

EXPECTED_TASKS = {
    "cube_lifting",
    "cube_restack",
    "cube_stack",
    "nut_assembly",
    "spill_wipe",
    "two_arm_lift",
    "two_arm_handover",
}


class FakeModel:
    cam_fovy = np.array([45.0, 45.0, 45.0])
    stat = SimpleNamespace(extent=1.0)
    vis = SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=10.0))

    @staticmethod
    def camera_name2id(name):
        return {"agentview": 0, "robot0_eye_in_hand": 1, "robot1_eye_in_hand": 2}[name]


class FakeData:
    def __init__(self, arm_count):
        self.qpos = np.full(4 * arm_count, 0.04)

    @staticmethod
    def get_body_xmat(name):
        if name == "robot1_base":
            return np.diag([-1.0, -1.0, 1.0])
        return np.eye(3)

    @staticmethod
    def get_body_xpos(name):
        return np.array([1.0, 0.0, 0.0]) if name == "robot1_base" else np.zeros(3)

    @staticmethod
    def get_site_xmat(name):
        return np.eye(3)

    @staticmethod
    def get_site_xpos(name):
        return np.array([0.7, 0.0, 0.3]) if "gripper1" in name else np.array([0.4, 0.0, 0.3])

    @staticmethod
    def get_camera_xmat(name):
        return np.eye(3)

    @staticmethod
    def get_camera_xpos(name):
        return np.array([0.5, 0.0, 1.0])


class FakeRobot:
    def __init__(self, index):
        self.action_dim = 8
        self._joint_positions = np.zeros(7)
        self._joint_velocities = np.zeros(7)
        self._ref_gripper_joint_pos_indexes = np.arange(index * 4, index * 4 + 2)
        self.gripper = SimpleNamespace(important_sites={"grip_site": f"gripper{index}_grip_site"})


class FakeEnv:
    def __init__(self, metadata):
        self.metadata = metadata
        self.robots = [FakeRobot(index) for index in range(len(metadata.arms))]
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData(len(self.robots)))
        self.action_spec = (
            -np.ones(8 * len(self.robots)),
            np.ones(8 * len(self.robots)),
        )
        self.actions = []
        self.closed = False
        self.done = False
        self.success = False
        self.freeze = False

    def _observation(self):
        result = {}
        for name in self.metadata.camera_names:
            result[f"{name}_image"] = np.zeros((4, 5, 3), dtype=np.uint8)
            result[f"{name}_depth"] = np.full((4, 5, 1), 0.5)
        return result

    def reset(self):
        return self._observation()

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        if not self.freeze:
            for index, robot in enumerate(self.robots):
                start = 8 * index
                robot._joint_positions += action[start : start + 7] / 20.0
                native_gripper = action[start + 7]
                self.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = (
                    (1.0 - native_gripper) * 0.5 * 0.04
                )
        return self._observation(), 0.25, self.done, {}

    def _check_success(self):
        return self.success

    def close(self):
        self.closed = True


def _adapter(task="two_arm_lift"):
    envs = []

    def factory(metadata):
        env = FakeEnv(metadata)
        envs.append(env)
        return env

    adapter = RobosuiteAdapter(env_factory=factory, camera_height=4, camera_width=5)
    observation = adapter.reset(task, 1)
    return adapter, envs[-1], observation


def test_registry_exposes_exact_audited_tasks_and_arm_mapping():
    registry = RobosuiteTaskRegistry()

    assert set(registry.available_tasks) == EXPECTED_TASKS
    assert len(ROBOSUITE_TASKS) == 7
    assert registry.resolve("cube_lifting").arms == ("primary",)
    assert registry.resolve(("two_arm_handover", 0)).arms == ("primary", "secondary")
    with pytest.raises(RobosuiteRegistryError):
        registry.resolve("unknown")
    with pytest.raises(RobosuiteRegistryError):
        registry.resolve("cube_lifting", 1)


def test_codec_combines_independent_commands_in_native_robot_order(robot_state):
    state = robot_state(arms=("primary", "secondary"))
    codec = RobosuiteActionCodec(
        ("primary", "secondary"),
        action_spec=(-np.ones(16), np.ones(16)),
    )
    action = RobotAction(
        {
            "primary": ArmCommand("joint_position", np.full(7, 0.01), 0.0),
            "secondary": ArmCommand("joint_position", np.full(7, -0.02), 1.0),
        }
    )

    native = codec.encode(action, state)

    np.testing.assert_allclose(native[:7], 0.2)
    assert native[7] == 1.0
    np.testing.assert_allclose(native[8:15], -0.4)
    assert native[15] == -1.0


def test_codec_supports_fixed_tool_seven_dimension_action(robot_state):
    state = robot_state(arms=("primary",))
    codec = RobosuiteActionCodec(
        ("primary",),
        action_spec=(-np.ones(7), np.ones(7)),
        arm_action_dimensions=(7,),
    )

    native = codec.encode(
        RobotAction({"primary": ArmCommand("joint_position", np.full(7, 0.01))}),
        state,
    )

    np.testing.assert_allclose(native, 0.2)


def test_adapter_normalizes_frames_and_executes_both_arms_in_one_tick():
    adapter, env, observation = _adapter()

    assert observation.robot_state.base_frame == BASE_FRAME
    assert observation.robot_state.arms == ("primary", "secondary")
    np.testing.assert_allclose(
        observation.robot_state.end_effector_poses["secondary"].position,
        [0.7, 0.0, 0.3],
    )
    assert all(camera.camera_pose.frame == BASE_FRAME for camera in observation.cameras.values())
    action = RobotAction(
        {
            arm: ArmCommand("joint_position", joints + 0.01, 0.5)
            for arm, joints in observation.robot_state.joint_positions.items()
        }
    )

    result = adapter.native_step(action)

    assert result.ok and result.reward == 0.25
    assert len(env.actions) == 1
    np.testing.assert_allclose(result.observation.robot_state.joint_positions["primary"], 0.01)
    np.testing.assert_allclose(result.observation.robot_state.joint_positions["secondary"], 0.01)
    assert not hasattr(result.observation, "reward")


def test_adapter_rejects_unknown_arm_before_mutating_or_stepping():
    adapter, env, _ = _adapter()

    result = adapter.native_step(
        RobotAction({"tertiary": ArmCommand("joint_position", np.zeros(7), 0.5)})
    )

    assert not result.ok
    assert result.error.code == ErrorCode.NOT_FOUND
    assert not env.actions


def test_single_arm_control_is_exactly_one_native_tick():
    adapter, env, observation = _adapter("cube_lifting")
    target = observation.robot_state.joint_positions["primary"] + 0.01

    result = adapter.native_step(
        RobotAction({"primary": ArmCommand("joint_position", target, 0.5)})
    )

    assert result.ok
    assert len(env.actions) == 1
    assert adapter.current_time_s == pytest.approx(0.05)
    np.testing.assert_allclose(result.observation.robot_state.joint_positions["primary"], target)


def test_structural_validation_omits_gripper_command_for_fixed_tool():
    _, _, observation = _adapter("spill_wipe")

    action = _action(
        observation,
        movement=False,
        controllable_gripper_arms=(),
    )

    assert action.arms["primary"].gripper_position is None


def test_secondary_ik_request_is_explicitly_expressed_in_robot1_base():
    adapter, _, observation = _adapter()
    target = Pose([0.7, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], BASE_FRAME)

    transformed, state = adapter.ik_request(target, observation.robot_state, "secondary")

    assert transformed.frame == "robot1_base"
    assert state.base_frame == "robot1_base"
    assert state.arms == ("secondary",)
    np.testing.assert_allclose(transformed.position, [0.3, 0.0, 0.3])


def test_synchronized_trajectory_converges_and_grippers_persist():
    adapter, env, observation = _adapter()
    names = observation.robot_state.joint_names
    trajectory = SynchronizedTrajectory(
        joint_positions={
            "primary": np.array([[0.01] * 7, [0.02] * 7]),
            "secondary": np.array([[-0.01] * 7, [-0.02] * 7]),
        },
        dt_s=0.05,
        joint_names=names,
        planner="test",
        collision_aware=False,
        expected_start=observation.robot_state,
        gripper_positions={"primary": np.zeros(2), "secondary": np.ones(2)},
    )

    result = adapter.execute_trajectory(trajectory)

    assert result.ok and result.steps_executed == 2
    assert len(env.actions) == 2
    assert all(action[7] == 1.0 and action[15] == -1.0 for action in env.actions)
    np.testing.assert_allclose(
        result.final_observation.robot_state.joint_positions["primary"], 0.02
    )
    np.testing.assert_allclose(
        result.final_observation.robot_state.joint_positions["secondary"], -0.02
    )


def test_set_grippers_is_one_tick_and_preserves_cached_joint_targets():
    adapter, env, observation = _adapter()
    env.freeze = True
    adapter.step(
        RobotAction(
            arms={
                "primary": ArmCommand("joint_position", np.full(7, 0.2), 0.25),
                "secondary": ArmCommand("joint_position", np.full(7, -0.2), 0.75),
            }
        )
    )
    before = len(env.actions)

    result = adapter.set_grippers({"primary": 0.0, "secondary": 1.0})

    assert result.ok and result.steps_executed == 1
    assert len(env.actions) == before + 1
    native = env.actions[-1]
    np.testing.assert_allclose(native[:7], 1.0)
    np.testing.assert_allclose(native[8:15], -1.0)
    assert native[7] == pytest.approx(1.0)
    assert native[15] == pytest.approx(-1.0)


def test_primary_trajectory_preserves_inactive_arm_and_both_grippers():
    adapter, env, observation = _adapter()
    env.freeze = True
    adapter.step(
        RobotAction(
            arms={
                "primary": ArmCommand("joint_position", np.full(7, 0.1), 0.25),
                "secondary": ArmCommand("joint_position", np.full(7, -0.2), 0.75),
            }
        )
    )
    start = adapter.get_robot_state()
    before = len(env.actions)
    trajectory = Trajectory(
        joint_positions=np.full((1, 7), 0.2),
        dt_s=0.05,
        joint_names=start.joint_names["primary"],
        planner="individual",
        collision_aware=True,
        expected_start=start,
        arm="primary",
    )

    result = adapter.execute_trajectory(trajectory)

    assert not result.ok and result.error.code == ErrorCode.TIMEOUT
    assert len(env.actions) > before
    for native in env.actions[before:]:
        np.testing.assert_allclose(native[8:15], -1.0)
        assert native[7] == pytest.approx(0.5)
        assert native[15] == pytest.approx(-0.5)


def test_stale_trajectory_rejects_before_native_step():
    adapter, env, observation = _adapter()
    expected = observation.robot_state
    trajectory = SynchronizedTrajectory(
        joint_positions={
            "primary": np.full((1, 7), 0.1),
            "secondary": np.full((1, 7), -0.1),
        },
        dt_s=0.05,
        joint_names=expected.joint_names,
        planner="stale",
        collision_aware=True,
        expected_start=expected,
    )
    env.robots[1]._joint_positions[:] = 0.2

    result = adapter.execute_trajectory(trajectory)

    assert not result.ok and result.error.code == ErrorCode.STALE_PLAN
    assert result.steps_executed == 0
    assert env.actions == []


def test_synchronized_trajectory_requires_every_arm():
    adapter, _, observation = _adapter()
    trajectory = SynchronizedTrajectory(
        joint_positions={"primary": np.array([[0.01] * 7])},
        dt_s=0.05,
        joint_names={"primary": observation.robot_state.joint_names["primary"]},
        planner="incomplete-test",
        collision_aware=False,
        expected_start=RobotState(
            joint_positions={"primary": observation.robot_state.joint_positions["primary"]},
            joint_velocities={"primary": observation.robot_state.joint_velocities["primary"]},
            end_effector_poses={"primary": observation.robot_state.end_effector_poses["primary"]},
            gripper_positions={"primary": observation.robot_state.gripper_positions["primary"]},
            base_frame=observation.robot_state.base_frame,
            joint_names={"primary": observation.robot_state.joint_names["primary"]},
        ),
    )

    with pytest.raises(ValueError, match="every Robosuite arm"):
        adapter.execute_trajectory(trajectory)


def test_trajectory_joint_names_must_match_the_controlled_arm():
    adapter, _, observation = _adapter()
    state = observation.robot_state
    wrong_names = tuple(f"wrong_joint_{index}" for index in range(7))
    expected = RobotState(
        joint_positions=state.joint_positions,
        joint_velocities=state.joint_velocities,
        end_effector_poses=state.end_effector_poses,
        gripper_positions=state.gripper_positions,
        base_frame=state.base_frame,
        joint_names={"primary": wrong_names, "secondary": wrong_names},
    )
    trajectory = Trajectory(
        joint_positions=np.array([[0.01] * 7]),
        dt_s=0.05,
        joint_names=wrong_names,
        planner="wrong-order-test",
        collision_aware=False,
        expected_start=expected,
        arm="secondary",
    )

    result = adapter.execute_trajectory(trajectory)
    assert not result.ok and result.error.code == ErrorCode.STALE_PLAN


@pytest.mark.parametrize(
    ("success", "terminal_field"),
    ((True, "terminated"), (False, "truncated")),
)
def test_synchronized_trajectory_propagates_terminal_tick(success, terminal_field):
    adapter, env, observation = _adapter()
    env.done = True
    env.success = success
    trajectory = SynchronizedTrajectory(
        joint_positions={
            "primary": np.array([[0.01] * 7]),
            "secondary": np.array([[-0.01] * 7]),
        },
        dt_s=0.05,
        joint_names=observation.robot_state.joint_names,
        planner="terminal-test",
        collision_aware=False,
        expected_start=observation.robot_state,
    )

    result = adapter.execute_trajectory(trajectory)

    assert not result.ok and result.steps_executed == 1
    assert getattr(result, terminal_field)
    assert len(env.actions) == 1


def test_synchronized_trajectory_has_bounded_convergence_timeout():
    adapter, env, observation = _adapter()
    env.freeze = True
    trajectory = SynchronizedTrajectory(
        joint_positions={
            "primary": np.array([[0.02] * 7]),
            "secondary": np.array([[-0.02] * 7]),
        },
        dt_s=0.05,
        joint_names=observation.robot_state.joint_names,
        planner="timeout-test",
        collision_aware=False,
        expected_start=observation.robot_state,
    )

    result = adapter.execute_trajectory(trajectory)

    assert not result.ok
    assert result.steps_executed == 120
    assert result.error.code == ErrorCode.TIMEOUT


def test_trajectory_advances_intermediate_waypoints_at_fixed_rate() -> None:
    adapter, env, observation = _adapter()
    env.freeze = True
    trajectory = SynchronizedTrajectory(
        joint_positions={
            "primary": np.array([[0.01] * 7, [0.02] * 7]),
            "secondary": np.array([[-0.01] * 7, [-0.02] * 7]),
        },
        dt_s=0.05,
        joint_names=observation.robot_state.joint_names,
        planner="fixed-rate-test",
        collision_aware=False,
        expected_start=observation.robot_state,
    )

    result = adapter.execute_trajectory(trajectory)

    assert not result.ok and result.steps_executed == 121
    np.testing.assert_allclose(env.actions[0][0:7], 0.2)
    np.testing.assert_allclose(env.actions[1][0:7], 0.4)
    np.testing.assert_allclose(env.actions[-1][0:7], 0.4)


@pytest.fixture
def robot_state():
    from cap_harness.contracts import Pose, RobotState

    def build(*, arms):
        pose = Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), BASE_FRAME)
        return RobotState(
            joint_positions={arm: np.zeros(7) for arm in arms},
            joint_velocities={arm: np.zeros(7) for arm in arms},
            end_effector_poses=dict.fromkeys(arms, pose),
            gripper_positions=dict.fromkeys(arms, 0.5),
            base_frame=BASE_FRAME,
            joint_names={arm: tuple(f"j{i}" for i in range(7)) for arm in arms},
        )

    return build
