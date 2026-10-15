"""``EnvironmentAdapter`` for the real bimanual YAM.

Deliberately thin: normalize, delegate, never actuate. Every motion routes to
``env.execute_action_batch``, the single chokepoint the plant exposes, so this
surface and any other driving the same station produce comparable command
streams.

**The motion interlock.** Construction, observation and state reads always work.
Anything that can command the arms is refused unless ``allow_physical_motion``
was explicitly set, and refused *before* any RPC is issued -- the gate precedes
payload construction and transport, so an unauthorized call leaves no trace on
the wire. ``command_rpc_count`` counts commands actually submitted, never
attempted, which makes it usable as evidence rather than as a hint.

This is a client-side safeguard for a harness that runs model-generated code. It
is not a safety system: it does not replace the hardware E-stop, the arm
server's watchdog, or a human watching the robot. It stops this process from
commanding motion by accident; it cannot stop a robot that is already moving.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    ExecutionResult,
    Observation,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.yam_real import codec
from cap_harness.yam_real.action import YamActionBatch
from cap_harness.yam_real.registry import YamRealTaskMetadata, YamRealTaskRegistry

ARMS = codec.ARMS
ARM_DOF = codec.ARM_DOF

#: Held at the final waypoint after a trajectory is streamed, so a
#: position-controlled arm has time to close the following error it accumulated.
TRAJECTORY_SETTLE_S = 1.0

#: How close every joint must be to its last commanded waypoint for a motion to
#: count as executed. This is a following-error budget, not a precision claim: a
#: position-controlled joint settles a few hundredths of a radian short and that
#: is normal. The failure it exists to catch was an arm that stopped a quarter of
#: the way through its trajectory and reported success, so the threshold sits
#: well above servo error and well below a missed motion.
TRAJECTORY_JOINT_TOLERANCE_RAD = 0.10

#: Joint speed a homing move is scheduled at, rad/s. The interpolation is linear
#: with no acceleration limit, so this is both the average and the peak rate --
#: keep it gentle. 0.35 rad/s takes a full 1.4 rad sweep across the table in
#: about four seconds and a small correction in well under one.
HOMING_JOINT_SPEED_RAD_S = 0.35

#: Bounds on the scheduled window. The floor stops a tiny correction becoming a
#: step command; the ceiling stops a pathological start state parking the arm in
#: a minutes-long crawl.
MINIMUM_HOMING_S = 1.5
MAXIMUM_HOMING_S = 8.0
HOME_DIAGNOSTIC_TOLERANCE_RAD = 0.03


class YamRealAdapter:
    """Protocol-facing wrapper over a YAM station."""

    embodiment = codec.EMBODIMENT
    registry_prefix = "yam_real"

    def __init__(
        self,
        env: Any,
        *,
        registry: YamRealTaskRegistry | None = None,
        task_context: TaskContext | None = None,
        allow_physical_motion: bool = False,
        run_observer: object | None = None,
    ) -> None:
        self._env = env
        self._registry = registry or YamRealTaskRegistry()
        self._task_context = task_context
        self._allow_physical_motion = bool(allow_physical_motion)
        self._run_observer = run_observer
        self._command_rpc_count = 0
        self._closed = False
        self._episode_recorder: Any = None
        controller = env.config.controller
        self.control_frequency = controller.control_frequency_hz
        self.control_period_s = 1.0 / controller.control_frequency_hz

    # -- identity ----------------------------------------------------------

    @property
    def native_env(self) -> Any:
        """The plant, mirroring the simulator adapters' ``native_env``."""
        return self._env

    @property
    def arms(self) -> tuple[str, ...]:
        return ARMS

    def resolve_arm(self, arm: str) -> str:
        """This station's own name for the arm a caller asked for.

        ``left`` and ``right`` are the real arms and pass through untouched.
        The shared API's ``primary``/``secondary`` are resolved through the
        profile's ``arm_aliases``, which is the whole point of that field --
        until now it was validated on load, asserted in a test, and read by
        nothing, so every shared default failed on this embodiment.

        The mapping is deliberately data rather than a constant here. Which
        physical arm is "primary" depends on how a bench is rigged and what it
        faces; yam-example declares ``primary: right``, and a station that reaches its
        work with the left arm would say so in its own profile without a code
        change.

        An unknown name is returned unchanged so it fails in the caller's own
        terms -- a wrong arm key should read as "no arm 'wrist'", not as a
        resolver error.
        """
        name = str(arm)
        if name in ARMS:
            return name
        return str(self._env.config.arm_aliases.get(name, name))

    @property
    def dof(self) -> int:
        return ARM_DOF

    @property
    def command_rpc_count(self) -> int:
        """Command RPCs actually submitted through this adapter."""
        return self._command_rpc_count

    @property
    def physical_motion_authorized(self) -> bool:
        return self._allow_physical_motion

    # -- the interlock -----------------------------------------------------

    def _motion_denied(self) -> ApiError | None:
        """The single gate. Every command path calls this first.

        Returns the typed refusal, or None when motion is authorized. Keeping it
        in one place means a new motion method that forgets to call it is a
        visible omission rather than a silent hole.
        """
        if self._allow_physical_motion:
            return None
        return ApiError(
            code=ErrorCode.SAFETY_INTERLOCK,
            message=(
                "physical motion is not authorized for this YAM adapter; "
                "construct it with allow_physical_motion=True (cap-harness run --allow-motion)"
            ),
            recoverable=False,
            details={"command_rpc_count": self._command_rpc_count},
        )

    # -- recording ---------------------------------------------------------

    def _commanded_action(
        self,
        targets: Mapping[str, Any] | None = None,
        gripper: Mapping[str, float] | None = None,
    ) -> RobotAction:
        """A ``RobotAction`` describing what was just commanded, for the recorder.

        ``targets`` are the joint positions actually sent to each arm, and
        ``gripper`` the widths. Both fall back to the MEASURED state only for
        arms the caller did not command -- an arm holding still while the other
        moves genuinely was commanded to where it is.

        Passing them is the whole point. This used to read the measured pose for
        every arm and record that as the action, so ``episode/steps.jsonl`` said
        the policy commanded exactly where the arm already was. On a
        position-controlled arm that lags its target by centimetres, an action
        stream built from measurements is not a weaker record of the policy --
        it is a record of a different policy, one that never asks for anything
        it has not already got. Trained on, it teaches the arm to stand still.

        The recorder's contract is per-``step``, and this plant does not step:
        it streams a whole batch through one RPC. So each command that reaches
        the plant is reported as one action carrying its final target, which
        gives the run one recorded frame per commanded motion rather than one
        per control period. That is coarser than a simulator's record and it is
        the honest shape for this embodiment.
        """
        state = codec.robot_state(self._env)
        return RobotAction(
            arms={
                side: ArmCommand(
                    mode="joint_position",
                    target=(
                        state.joint_positions[side]
                        if targets is None or targets.get(side) is None
                        else np.asarray(targets[side], dtype=np.float64).reshape(ARM_DOF)
                    ),
                    gripper_position=(
                        state.gripper_positions[side]
                        if gripper is None or gripper.get(side) is None
                        else gripper[side]
                    ),
                    embodiment=self.embodiment,
                )
                for side in ARMS
            }
            # No `embodiment=` here: RobotAction takes only `arms` and derives
            # the embodiment from the commands, which must all agree.
        )

    def _before_command(self) -> None:
        """Let the recorder enforce the step limit before anything is commanded."""
        if self._run_observer is not None:
            self._run_observer.before_step(self._commanded_action())

    def _record(
        self,
        result: object,
        gripper: Mapping[str, float] | None = None,
        targets: Mapping[str, Any] | None = None,
    ) -> None:
        """Hand the recorder the action and its outcome, so a frame is captured."""
        if self._run_observer is None:
            return
        action = self._commanded_action(targets, gripper)
        step = (
            result
            if isinstance(result, StepResult)
            else StepResult(
                ok=bool(getattr(result, "ok", False)),
                observation=getattr(result, "final_observation", None),
                error=getattr(result, "error", None) if not getattr(result, "ok", False) else None,
            )
        )
        self._run_observer.after_step(action, step)

    def _denied_step(self, error: ApiError) -> StepResult:
        return StepResult(ok=False, error=error)

    def _denied_execution(self, error: ApiError) -> ExecutionResult:
        return ExecutionResult(ok=False, steps_executed=0, error=error)

    def _plant_failure(self, exc: Exception, operation: str) -> ApiError:
        """Turn a refusal or fault from the plant into a typed failure.

        The arm servers validate what they are asked to do and raise across the
        RPC boundary when they refuse -- a target outside the joint limits, for
        instance. Letting that surface as a bare ``RuntimeError`` would end a
        generated program's run with a traceback instead of a result it could
        respond to, so it is classified here.
        """
        message = str(exc)
        code = (
            ErrorCode.INVALID_REQUEST
            if "outside the configured limits" in message or "must contain" in message
            else ErrorCode.EXECUTION_FAILED
        )
        return ApiError(
            code=code,
            message=f"{operation} rejected by the station: {message}",
            details={"command_rpc_count": self._command_rpc_count},
        )

    def _plant_refusal(self, result: Mapping[str, Any], operation: str) -> ApiError | None:
        """Typed failure for a plant that REPORTS a refusal instead of raising.

        The sibling of :meth:`_plant_failure`, which only covers the case where
        the RPC raises. A station that answers ``{"success": False, "reason":
        ...}`` took the other path, and every command here used to build its
        result as ``ok=bool(result["success"])`` with no ``error``.

        ``_validate_result_status`` rejects a failed result carrying no
        ``ApiError``, so that combination did not degrade -- it raised
        ``ValueError: a failed result must carry an ApiError`` out of the
        adapter, ending the run with a traceback. Which is the exact opposite of
        what this class documents: a refusal from the station should reach the
        program as something it can respond to.
        """
        if bool(result.get("success", True)):
            return None
        reason = str(result.get("reason", "")).strip() or "no reason given"
        return ApiError(
            code=ErrorCode.EXECUTION_FAILED,
            message=f"{operation} refused by the station: {reason}",
            details={"command_rpc_count": self._command_rpc_count, "reason": reason},
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Reset what can be reset on hardware, and be explicit about the rest.

        A physical station cannot be restored to a seeded state, so this does not
        pretend to. It performs the *robot* reset -- homing both arms -- and
        records the seed as provenance only. Scene reset is a physical act and
        belongs to a human or a task procedure; conflating the two would let a run
        proceed against an un-reset scene while reporting that it had reset.

        Homing is motion, so it goes through the same gate as everything else. An
        unauthorized reset observes and returns without commanding anything.
        """
        if isinstance(task_ref, TaskContext):
            self._task_context = task_ref
        elif isinstance(task_ref, YamRealTaskMetadata):
            self._task_context = TaskContext(
                suite=task_ref.suite_name,
                task_id=task_ref.task_id,
                task_name=task_ref.task_name,
                language=task_ref.language,
                family=task_ref.family,
                metadata={
                    "scene_reset": task_ref.scene_reset,
                    "seed": int(seed),
                    "station": self._env.config.station,
                    "calibration_bundle": self._env.config.calibration.bundle_id,
                },
            )
        elif task_ref is not None:
            self._task_context = self._task_context_for(self._registry.resolve(task_ref), seed)

        if self._motion_denied() is None:
            # Home, the mechanical zero. It is the calibration reference, so a run
            # that starts here starts somewhere the profile actually defines,
            # rather than at a second pose that has to be kept correct alongside
            # it. Both plan; cuRobo solves IK from either.
            #
            # This was `go_ready` until it turned out that the ready pose parks
            # the forearm at z 1.23, inside the band where the camera sees arm
            # hardware the URDF does not model. Home keeps the highest link at
            # z 0.95, below that band entirely.
            self._env.go_home(duration_s=4.0, keep_grippers=False)
            self._command_rpc_count += 1
        observation = self.get_observation()
        # Opens the recorded episode: writes reset.json and captures frame 0 of
        # every camera. Without it a recorded run has no video at all, because
        # every later frame is written from `after_step`.
        if self._run_observer is not None:
            metadata = {} if self._task_context is None else dict(self._task_context.metadata or {})
            metadata.update(
                {
                    "embodiment": self.embodiment,
                    "station": self._env.config.station,
                    "physical_motion_authorized": self._allow_physical_motion,
                }
            )
            self._run_observer.on_reset(observation, metadata)
        self._start_episode_recording()
        return observation

    # -- episode recording -------------------------------------------------

    def _start_episode_recording(self) -> None:
        """Begin sampling the plant at 30 Hz into the run's episode directory.

        Separate from ``_run_observer``, and deliberately so. The harness
        recorder writes one frame per commanded action, which is the right
        granularity for a trace but produces a slideshow for video: a real run
        logged 9 frames over several minutes because a 2.8 s descent is one
        command. This samples on its own clock, so the trajectory between
        commands is captured too.

        Needs somewhere to write, so it is skipped when the adapter has no run
        observer -- an ad-hoc adapter built in a script records nothing rather
        than scattering episode directories.
        """
        root = getattr(self._run_observer, "root", None)
        if root is None:
            return
        self.finish_episode_recording()
        from .recorder import YamEpisodeRecorder

        task = "" if self._task_context is None else str(self._task_context.task_name or "")
        recorder = YamEpisodeRecorder(
            Path(root) / "episode" / "raw",
            # media/videos is where every embodiment's video lives; on this one
            # it is written here rather than from recorded steps.
            video_dir=Path(root) / "media" / "videos",
            fps=round(self.control_frequency),
            cameras=None if getattr(self._run_observer, "capture_videos", True) else (),
            task=task,
        )
        try:
            recorder.start(
                self._env,
                meta={
                    "station": self._env.config.station,
                    "calibration_bundle": self._env.config.calibration.bundle_id,
                    "physical_motion_authorized": self._allow_physical_motion,
                },
            )
        except Exception:
            return
        self._episode_recorder = recorder

    def finish_episode_recording(self) -> None:
        """Stop the sampler and write the episode. Safe when none is running.

        Public because the run teardown calls it BEFORE artifacts are sealed:
        the arrays are written here, and the manifest hashes what exists when it
        is built. Left to ``close`` -- which for this embodiment runs after
        recording, so the camera threads survive -- the episode would land
        unlisted next to a manifest that had already been written.
        """
        recorder = self._episode_recorder
        self._episode_recorder = None
        if recorder is None:
            return
        try:
            recorder.finalize()
        except Exception:
            pass

    def _task_context_for(self, metadata: YamRealTaskMetadata, seed: int) -> TaskContext:
        return TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={
                "scene_reset": metadata.scene_reset,
                "seed": int(seed),
                "station": self._env.config.station,
                "calibration_bundle": self._env.config.calibration.bundle_id,
            },
        )

    def close(self) -> None:
        """Close the client side. Never stops the arm servers."""
        if self._closed:
            return
        self._closed = True
        self.finish_episode_recording()
        self._env.close()

    # -- observation -------------------------------------------------------

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("no task context set; pass one to the adapter or to reset()")
        return self._task_context

    def get_observation(self) -> Observation:
        return codec.observation(self._env, task_context=self._task_context)

    def get_robot_state(self) -> RobotState:
        return codec.robot_state(self._env)

    def check_success(self) -> bool:
        """No task success predicate exists on hardware.

        Returning False is not a claim that the task failed; it is a statement
        that this embodiment cannot tell. A recorded run reports how it
        terminated, and success is judged from the video by whoever ran it.
        """
        return False

    # -- actuation ---------------------------------------------------------

    def _submit(
        self,
        joints: Mapping[str, np.ndarray],
        grippers: Mapping[str, float | None],
        *,
        duration_s: float,
        source: str,
    ) -> dict[str, Any]:
        """Build one action batch and push it through the plant's chokepoint."""
        state = codec.robot_state(self._env)
        # "Hold where you are" is built from the MEASURED pose, and a measured
        # pose can sit outside the commandable range -- an arm resting against a
        # limit sags a little past it. Commanding that verbatim is rejected by
        # the station, which would wedge the robot: an arm that drifted out of
        # range could never be commanded again, not even to move back in. Only
        # the derived hold target is clamped; a caller's own target is passed
        # through untouched and refused if it is genuinely out of range.
        config = self._env.config
        held = {
            side: np.clip(
                np.asarray(state.joint_positions[side], dtype=np.float64).reshape(ARM_DOF),
                config.joint_limits_lower,
                config.joint_limits_upper,
            )
            for side in ARMS
        }
        held_grip = {side: float(state.gripper_positions[side]) for side in ARMS}
        targets = {side: joints.get(side, held[side]) for side in ARMS}
        grips = {
            side: held_grip[side] if grippers.get(side) is None else float(grippers[side])
            for side in ARMS
        }
        batch = YamActionBatch.joint_abs(
            [0.0, max(1e-3, float(duration_s))],
            [held["left"], targets["left"]],
            [held["right"], targets["right"]],
            [[held_grip["left"]], [grips["left"]]],
            [[held_grip["right"]], [grips["right"]]],
            source=source,
        )
        result = self._env.execute_action_batch(batch)
        self._command_rpc_count += 1
        return result

    def step(self, action: RobotAction) -> StepResult:
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_step(denied)
        joints, grippers = codec.arm_command_targets(action)
        # ``RobotAction`` does not constrain its arm keys -- it has no start state
        # to check them against -- so a mistyped arm arrives here looking valid.
        # Without this, the unknown entry is dropped, both arms hold their current
        # position, and the step reports success having moved nothing.
        unknown = set(joints) - set(ARMS)
        if unknown:
            return self._denied_step(
                ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"action names unknown YAM arms: {sorted(unknown)}",
                    details={"known_arms": list(ARMS)},
                )
            )
        self._before_command()
        try:
            result = self._submit(
                joints, grippers, duration_s=self.control_period_s, source="harness_step"
            )
        except Exception as exc:
            denied = self._denied_step(self._plant_failure(exc, "step"))
            self._record(denied)
            return denied
        step = StepResult(
            ok=bool(result.get("success", True)),
            error=self._plant_refusal(result, "step"),
            observation=self.get_observation(),
            diagnostics={
                "reason": str(result.get("reason", "")),
                "command_rpc_count": self._command_rpc_count,
            },
        )
        self._record(step, gripper=grippers, targets=joints)
        return step

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        """Execute a joint trajectory through the level-1 controller.

        Two checks that ``yam_sim`` makes are deliberately absent here.

        **Embodiment** is not re-checked because it cannot fail. Both
        ``Trajectory`` and ``SynchronizedTrajectory`` already require
        ``expected_start.embodiment`` to equal their own, so a trajectory built
        against this adapter's state is a ``yam_real`` trajectory by
        construction. Repeating the test here would be unreachable code that
        reads like a live guard.

        **dt_s** is not required to equal the control period, which is where
        this genuinely differs from ``yam_sim`` rather than merely omitting
        something. Level 1 resamples waypoints against the wall clock, so a
        trajectory takes the duration it claims however many waypoints express
        it. Both working pick programs depend on that: they step at 1/10 s
        against a 1/30 s control period, because at 1/30 the descent ran at
        14 cm/s and the arm finished 2.7 cm short of its target.
        """
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)

        if isinstance(trajectory, Trajectory):
            per_arm = {trajectory.arm: np.asarray(trajectory.joint_positions, dtype=np.float64)}
            per_arm_grippers = (
                {}
                if trajectory.gripper_positions is None
                else {trajectory.arm: np.asarray(trajectory.gripper_positions, dtype=np.float64)}
            )
        else:
            per_arm = {
                side: np.asarray(value, dtype=np.float64)
                for side, value in trajectory.joint_positions.items()
            }
            per_arm_grippers = (
                {}
                if trajectory.gripper_positions is None
                else {
                    side: np.asarray(value, dtype=np.float64)
                    for side, value in trajectory.gripper_positions.items()
                }
            )
        # No arm-key check here, unlike ``step``: both trajectory contracts
        # validate their arms against ``expected_start``, so an unknown arm cannot
        # reach this point. ``RobotAction`` has no such start state, which is why
        # ``step`` must check.

        dt_s = float(trajectory.dt_s)
        state = codec.robot_state(self._env)
        steps = max(len(value) for value in per_arm.values())
        timestamps = [index * dt_s for index in range(steps)]

        def rows(side: str) -> np.ndarray:
            if side in per_arm:
                return per_arm[side][:, :ARM_DOF]
            held = np.asarray(state.joint_positions[side], dtype=np.float64).reshape(ARM_DOF)
            return np.tile(held, (steps, 1))

        held_grip = {side: float(state.gripper_positions[side]) for side in ARMS}

        def gripper_rows(side: str) -> list[list[float]]:
            """The trajectory's own gripper column, or the held width if it has none.

            The *command* has to survive, not the measurement. During a successful
            grasp the two deliberately differ: fingers stalled on an object read
            wider than they were told to close to -- commanding 0.12 on this
            station reads back around 0.36. Substituting the measurement here
            re-commands that wider value and releases the squeeze, which shows up
            as the object not rising and reads like a grasp or planning fault.
            """
            if side not in per_arm_grippers:
                return [[held_grip[side]]] * steps
            # Length is not re-checked: both trajectory contracts validate the
            # gripper column against the waypoint count in ``__post_init__``, and
            # a synchronized trajectory additionally requires every arm to be the
            # same length, so ``steps`` cannot disagree with it here.
            return [[float(value)] for value in per_arm_grippers[side].reshape(-1)]

        batch = YamActionBatch.joint_abs(
            timestamps,
            rows("left"),
            rows("right"),
            gripper_rows("left"),
            gripper_rows("right"),
            source="harness_trajectory",
        )
        self._before_command()
        try:
            result = self._env.execute_action_batch(batch, settle_s=TRAJECTORY_SETTLE_S)
        except Exception as exc:
            denied = self._denied_execution(self._plant_failure(exc, "trajectory"))
            self._record(denied)
            return denied
        self._command_rpc_count += 1
        ok = bool(result.get("success", True))

        # ``success`` from the batch means the commands were streamed, not that
        # the arm arrived. A position-controlled arm lags a fast trajectory, so a
        # batch can end with the arm still well behind its last target. Settling
        # holds the final waypoint; this checks that holding it was enough.
        reached = codec.robot_state(self._env)
        residuals = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(reached.joint_positions[side], dtype=np.float64).reshape(ARM_DOF)
                        - np.asarray(value[-1], dtype=np.float64).reshape(ARM_DOF)
                    )
                )
            )
            for side, value in per_arm.items()
        }
        worst_arm = max(residuals, key=residuals.get) if residuals else None
        worst = residuals.get(worst_arm, 0.0) if worst_arm else 0.0
        # A plant that REPORTED a refusal first: without this the result is
        # ok=False with error=None, because the tracking check below only fires
        # when ok is still True, and a failed result carrying no ApiError raises
        # out of the adapter instead of reaching the program.
        error = self._plant_refusal(result, "execute_trajectory")
        if ok and worst > TRAJECTORY_JOINT_TOLERANCE_RAD:
            ok = False
            error = ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=(
                    f"trajectory ended {worst:.3f} rad from its final waypoint on the "
                    f"{worst_arm} arm (tolerance {TRAJECTORY_JOINT_TOLERANCE_RAD:.3f} rad); "
                    "the arm did not track the commanded motion"
                ),
                details={"joint_residual_rad": residuals},
            )

        execution = ExecutionResult(
            ok=ok,
            steps_executed=int(result.get("command_count", steps)),
            final_observation=self.get_observation(),
            error=error,
            diagnostics={
                "reason": str(result.get("reason", "")),
                "joint_residual_rad": residuals,
                "command_rpc_count": self._command_rpc_count,
            },
        )
        # The trajectory's FINAL waypoint is what this batch asked the arms to
        # reach, so that is the action. `per_arm` holds only the arms the
        # trajectory named; anything else falls back to its measured pose in
        # `_commanded_action`, which is correct -- an arm this batch did not
        # address was commanded to hold where it is.
        final_targets = {side: value[-1] for side, value in per_arm.items()}
        commanded_gripper = {
            side: float(np.asarray(value, dtype=np.float64).reshape(-1)[-1])
            for side, value in per_arm_grippers.items()
            if np.asarray(value).size
        }
        self._record(execution, gripper=commanded_gripper, targets=final_targets)
        return execution

    def set_gripper(self, position: float, *, arm: str = "left") -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        self._before_command()
        try:
            result = self._env.set_gripper(str(arm), float(position))
        except Exception as exc:
            denied = self._denied_execution(self._plant_failure(exc, "set_gripper"))
            self._record(denied)
            return denied
        self._command_rpc_count += 1
        execution = ExecutionResult(
            ok=bool(result.get("success", True)),
            error=self._plant_refusal(result, "set_gripper"),
            steps_executed=1,
            final_observation=self.get_observation(),
            diagnostics={
                "reason": str(result.get("reason", "")),
                "command_rpc_count": self._command_rpc_count,
            },
        )
        # The width is the command. Joints were not addressed, so they fall
        # back to measured, which is exactly what holding means here.
        self._record(execution, gripper={self.resolve_arm(arm): float(position)})
        return execution

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        """Set both grippers, returning the first failure rather than continuing."""
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        steps = 0
        for arm, position in positions.items():
            result = self.set_gripper(float(position), arm=str(arm))
            steps += int(result.steps_executed)
            if not result.ok:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=result.final_observation,
                    error=result.error,
                )
        return ExecutionResult(
            ok=True, steps_executed=steps, final_observation=self.get_observation()
        )

    def _homing_duration_s(self) -> float:
        """Schedule the homing window from the largest joint the move must turn."""
        state = codec.robot_state(self._env)
        config = self._env.config
        worst = 0.0
        for side in ARMS:
            measured = np.asarray(state.joint_positions[side], dtype=np.float64).reshape(ARM_DOF)
            home = np.asarray(config.arms[side].home_joints, dtype=np.float64).reshape(ARM_DOF)
            worst = max(worst, float(np.max(np.abs(measured - home))))
        seconds = worst / HOMING_JOINT_SPEED_RAD_S
        return float(min(MAXIMUM_HOMING_S, max(MINIMUM_HOMING_S, seconds)))

    def go_home(self, *, arm: str = "both", duration: float | None = None) -> ExecutionResult:
        """Home through the plant's own primitive rather than a settling loop.

        Both arms move together because that is what the primitive does: home is
        a whole-robot pose here, and homing one arm while the other stays put is
        not a state the profile describes. ``arm`` is accepted for signature
        compatibility with the shared API and deliberately ignored.

        ``duration`` defaults to **scaled by how far the arms actually have to
        travel** rather than to a fixed number. A fixed 3 s made a 0.02 rad
        correction and a 1.5 rad sweep across the table take exactly as long,
        which meant the sweep ran at roughly seventy times the joint speed of the
        correction and looked and sounded like it. Distance-scaling keeps the
        peak joint rate roughly constant instead.

        The underlying interpolation is linear in joint space (``controller.py``
        ramps ``(1-alpha) * start + alpha * target`` over the window), so there is
        no acceleration limiting -- velocity steps from zero at both ends. Giving
        a long move a proportionally longer window is what keeps that step small.
        Pass an explicit ``duration`` to override.
        """
        denied = self._motion_denied()
        if denied is not None:
            return self._denied_execution(denied)
        seconds = self._homing_duration_s() if duration is None else float(duration)
        self._before_command()
        try:
            result = self._env.go_home(duration_s=seconds)
        except Exception as exc:
            denied = self._denied_execution(self._plant_failure(exc, "go_home"))
            self._record(denied)
            return denied
        self._command_rpc_count += 1
        observation = self.get_observation()
        residuals = {
            side: float(
                np.max(
                    np.abs(
                        observation.robot_state.joint_positions[side]
                        - self._env.config.arms[side].home_joints
                    )
                )
            )
            for side in ARMS
        }
        execution = ExecutionResult(
            ok=bool(result.get("success", True)),
            error=self._plant_refusal(result, "go_home"),
            steps_executed=int(result.get("command_count", 1)),
            final_observation=observation,
            diagnostics={
                "source": str(result.get("source", "cap_home")),
                "joint_residual_rad": residuals,
                "home_tolerance_rad": HOME_DIAGNOSTIC_TOLERANCE_RAD,
                "home_verified": max(residuals.values()) <= HOME_DIAGNOSTIC_TOLERANCE_RAD,
                "grippers_open": {
                    side: float(observation.robot_state.gripper_positions[side]) >= 0.95
                    for side in ARMS
                },
                # `seconds`, not `duration`: the argument is None on the default
                # path, so float(duration) raised TypeError *after* the arms had
                # already moved, outside the try above. Reporting the value
                # actually commanded is also the correct diagnostic -- with
                # distance-scaling the two differ by design.
                "duration_s": seconds,
                "requested_arm": str(arm),
                "command_rpc_count": self._command_rpc_count,
            },
        )
        # Homing commands the profile's home pose on both arms, so that is the
        # action -- not wherever the arms ended up, which is the same distinction
        # every other path here now makes.
        self._record(
            execution,
            targets={side: self._env.config.arms[side].home_joints for side in ARMS},
        )
        return execution

    # -- planning and metadata --------------------------------------------

    def get_planning_context(self) -> RobotPlanningContext:
        """Describe YAM to the planners instead of letting them guess.

        The shared fallback infers a model from arm count -- two arms means dual
        Panda -- which sends YAM requests to a 7-DOF model that rejects them
        before planning starts. YAM is 6-DOF with its own grasp frames, so the
        embodiment states that itself through the hook provided for it.
        """
        config = self._env.config
        return RobotPlanningContext(
            embodiment=self.embodiment,
            # Must match a model the cuRobo service registers. The YAM Sim
            # description is a single arm, so it cannot represent this station's
            # second arm -- and arm-to-arm collision is what a bimanual planner
            # is for. `yam_real` selects the dual-arm description.
            model="yam_real",
            joint_names={side: config.arms[side].joint_names for side in ARMS},
            # Identity, NOT the profile's measured base transforms. The cuRobo
            # model is bimanual and already places both arms 0.62 m apart, so
            # supplying the offset here applies it a second time and every goal
            # lands somewhere impossible. Measured symptom: IK refused the arm's
            # OWN current pose, which is reachable by definition.
            base_transforms={side: np.eye(4, dtype=np.float64) for side in ARMS},
            end_effector_links={side: f"{side}_grasp" for side in ARMS},
        )

    def planning_workspace(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame box the shared planning scene is cropped to.

        The station's own profile owns this, so a second bench with a different
        table is a data change. It bounds what the arms can reach, not what the
        task cares about: nothing reachable is ever cropped away, which is what
        keeps this an optimization rather than a way to hide obstacles.
        """
        config = self._env.config
        return config.planning_workspace_lower, config.planning_workspace_upper

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": self.embodiment,
            "station": self._env.config.station,
            "calibration_bundle": self._env.config.calibration.bundle_id,
            "arms": list(ARMS),
            "dof": ARM_DOF,
            "cameras": [self._env.config.camera.role],
            "language": "" if context is None else context.language,
        }

    def get_controller_metadata(self) -> dict[str, Any]:
        """Plant description, including the versioned control contract.

        This is where provenance belongs: a dataset that does not record which
        plant executed it is unattributable.
        """
        contract = self._env.control_contract()
        config = self._env.config
        lower = np.concatenate([config.joint_limits_lower, config.joint_limits_lower]).tolist()
        upper = np.concatenate([config.joint_limits_upper, config.joint_limits_upper]).tolist()
        return {
            "control_frequency_hz": self.control_frequency,
            "control_period_s": self.control_period_s,
            "arm_names": list(ARMS),
            "control_contract": contract,
            "control_contract_version": contract.get("version"),
            "action_lower_bounds": lower,
            "action_upper_bounds": upper,
            "physical_motion_authorized": self._allow_physical_motion,
            "command_rpc_count": self._command_rpc_count,
        }


__all__ = [
    "TRAJECTORY_JOINT_TOLERANCE_RAD",
    "TRAJECTORY_SETTLE_S",
    "YamRealAdapter",
]
