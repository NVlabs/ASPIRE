"""Orchestrate one recorded Code-as-Policy episode."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .api import CapApi
from .artifacts import RunRecorder, StepLimitReached, TracedProvider, redact
from .libero.adapter import LiberoAdapter
from .libero.registry import LiberoSuiteRegistry
from .providers.curobo.client import DEFAULT_CUROBO_URL, CuRoboProvider
from .providers.graspnet.client import ContactGraspNetProvider
from .providers.pyroki.client import PyRokiProvider
from .providers.sam3.client import DEFAULT_SAM3_URL, Sam3Provider
from .registry import ToolRegistry
from .runtime import ProgramExecutionResult, ProgramExecutor

DEFAULT_MAX_STEPS = 1000

_CLOSE_AFTER_RECORDING = frozenset({"behavior"})
"""Benchmarks whose simulator must outlive artifact finalization (Isaac Sim ends the process)."""
BEHAVIOR_VIDEO_FRAME_STRIDE = 3


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run_dir: Path
    program_ok: bool
    task_success: bool | None
    protocol_success: bool | None
    termination_reason: str


def run_program_with_api(
    *,
    api: CapApi,
    program_path: Path,
    extension_allowlist: frozenset[str] | None = None,
    recorder: RunRecorder | None = None,
) -> ProgramExecutionResult:
    """Execute a program against a caller-owned API without owning its environment.

    ``recorder`` is optional so a caller that owns its embodiment -- one that runs
    the program inside its own control loop -- can still get the run trace that
    `run_program` produces. Without it a failed program leaves nothing but stdout,
    and "failure_stage: perception" cannot be told apart from a provider outage.
    """
    path = Path(program_path).expanduser().resolve()
    source = path.read_text(encoding="utf-8")
    if not source.strip():
        raise ValueError("program file must contain non-empty Python source")
    registry = (
        ToolRegistry(public_extension_allowlist=extension_allowlist)
        if extension_allowlist is not None
        else ToolRegistry()
    )
    tools = api.register_tools(registry)
    if recorder is None:
        return ProgramExecutor(tools).execute_program(source)
    recorder.save_program(path)
    return ProgramExecutor(tools, recorder=recorder).execute_program(source)


def run_program(
    *,
    benchmark: str,
    suite: str,
    task_id: int,
    seed: int,
    program_path: Path,
    output_root: Path,
    max_steps: int | None = None,
    camera_width: int = 800,
    camera_height: int = 512,
    model_trace: Path | None = None,
    init_mode: str = "saved",
    flat_layout: bool = False,
    capture_videos: bool = True,
    station: str = "yam-example",
    station_config_root: Path | None = None,
    allow_motion: bool = False,
) -> RunOutcome:
    """Run an episode, using its task horizon when max_steps is not supplied."""
    if benchmark not in {"behavior", "libero-pro", "robosuite", "yam_real"}:
        raise ValueError("benchmark must be behavior, libero-pro, robosuite, or yam_real")
    if init_mode not in {"saved", "seeded"}:
        raise ValueError("init_mode must be 'saved' or 'seeded'")
    if init_mode != "saved" and benchmark != "libero-pro":
        raise ValueError("init_mode 'seeded' is only supported for the libero-pro benchmark")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if camera_width <= 0 or camera_height <= 0:
        raise ValueError("camera dimensions must be positive")
    source = program_path.read_text(encoding="utf-8")
    if not source.strip():
        raise ValueError("program file must contain non-empty Python source")

    if benchmark == "libero-pro":
        registry = LiberoSuiteRegistry()
        metadata = registry.resolve(suite, task_id)
        adapter_type = LiberoAdapter
        extension_allowlist = None
    elif benchmark == "behavior":
        from .behavior.adapter import BehaviorAdapter
        from .behavior.registry import BehaviorTaskRegistry
        from .registry import BEHAVIOR_PUBLIC_TOOL_NAMES

        registry = BehaviorTaskRegistry()
        metadata = registry.resolve(suite, task_id)
        adapter_type = BehaviorAdapter
        extension_allowlist = BEHAVIOR_PUBLIC_TOOL_NAMES
    elif benchmark == "yam_real":
        from .registry import YAM_REAL_PUBLIC_TOOL_NAMES
        from .yam_real.adapter import YamRealAdapter
        from .yam_real.registry import YamRealTaskRegistry

        registry = YamRealTaskRegistry()
        metadata = registry.resolve(suite, task_id)
        adapter_type = YamRealAdapter
        extension_allowlist = YAM_REAL_PUBLIC_TOOL_NAMES
    else:
        from cap_harness.registry import ROBOSUITE_PUBLIC_TOOL_NAMES
        from cap_harness.robosuite.adapter import RobosuiteAdapter
        from cap_harness.robosuite.registry import RobosuiteTaskRegistry

        registry = RobosuiteTaskRegistry()
        metadata = registry.resolve(suite, task_id)
        adapter_type = RobosuiteAdapter
        extension_allowlist = ROBOSUITE_PUBLIC_TOOL_NAMES
    if max_steps is None:
        stated = getattr(metadata, "horizon", None)
        max_steps = int(stated) if stated else DEFAULT_MAX_STEPS

    if benchmark == "behavior":
        # Motion is planned in-process on the simulator's own robot model and world geometry.
        provider_capabilities: dict[str, object] = {
            "segmentation": "sam3",
            "grasping": ("contact-graspnet",),
            "ik": ("curobo",),
            "trajectory_planning": ("interpolation", "curobo"),
            "pose_planning": ("curobo-integrated",),
        }
    else:
        provider_capabilities = {
            "segmentation": "sam3",
            "grasping": ("contact-graspnet",),
            "ik": ("mink", "curobo") if benchmark == "yam_real" else ("pyroki", "curobo"),
            "trajectory_planning": ("interpolation", "curobo"),
            "pose_planning": ("composed", "curobo-integrated"),
        }
    recorder = RunRecorder(
        output_root=output_root,
        benchmark=benchmark,
        suite=metadata.suite_name,
        task_id=metadata.task_id,
        task_name=metadata.task_name,
        seed=seed,
        max_steps=max_steps,
        camera_width=camera_width,
        camera_height=camera_height,
        init_mode=init_mode,
        flat_layout=flat_layout,
        capture_videos=capture_videos,
        providers=provider_capabilities,
        # Isaac renders at 30 Hz; every third frame keeps videos at 10 fps.
        video_frame_stride=BEHAVIOR_VIDEO_FRAME_STRIDE if benchmark == "behavior" else 1,
        step_video=benchmark != "yam_real",
    )
    adapter: object | None = None
    execution: ProgramExecutionResult | None = None
    task_success: bool | None = None
    protocol_success: bool | None = None
    protocol_evaluator: object | None = None
    protocol_saved = False
    termination_reason = "harness_error"
    adapter_closed = False
    cleanup_errors: list[str] = []

    def close_adapter() -> None:
        nonlocal adapter_closed
        if adapter is None or adapter_closed:
            return
        adapter_closed = True
        try:
            adapter.close()  # type: ignore[attr-defined]
        # Record teardown errors without discarding the episode evidence.
        except Exception as exc:
            cleanup_errors.append(f"adapter close: {type(exc).__name__}: {exc}")

    def close_before_recording() -> None:
        # Stop any episode sampler first. It writes its arrays on finalize, and
        # the run manifest hashes whatever exists when it is built -- so a
        # sampler still running here leaves files that appear after the manifest
        # is sealed, unlisted, next to an mp4 whose recorded sha256 is of a
        # partial file. Embodiments in _CLOSE_AFTER_RECORDING keep their simulator
        # alive until the artifacts are finalized: tearing Isaac down can end the
        # process before outcome.json is written.
        finish_episode = getattr(adapter, "finish_episode_recording", None)
        if callable(finish_episode):
            try:
                finish_episode()
            # Record teardown errors without discarding the episode evidence.
            except Exception as exc:
                cleanup_errors.append(f"episode recording: {type(exc).__name__}: {exc}")
        if benchmark not in _CLOSE_AFTER_RECORDING:
            close_adapter()

    def save_protocol_evidence(final_native_success: bool) -> None:
        nonlocal protocol_saved, protocol_success
        if protocol_evaluator is None or protocol_saved:
            return
        try:
            evidence = protocol_evaluator.evidence(  # type: ignore[attr-defined]
                final_native_success=final_native_success
            )
        # Record teardown errors without discarding the episode evidence.
        except Exception as exc:
            evidence = {
                "schema_version": 1,
                "task": metadata.task_name,
                "protocol_success": False,
                "checks": {},
                "witness_steps": {},
                "final_native_success": bool(final_native_success),
                "evaluator_errors": (type(exc).__name__,),
            }
        recorder.save_protocol_evidence(evidence)
        protocol_success = bool(evidence["protocol_success"])
        protocol_saved = True

    try:
        recorder.save_program(program_path)
        recorder.import_model_trace(model_trace)
        with recorder.timing.measure("setup"):
            adapter_kwargs: dict[str, object] = {
                "registry": registry,
                "run_observer": recorder,
                "camera_width": camera_width,
                "camera_height": camera_height,
                "horizon": max_steps,
            }
            if benchmark == "libero-pro":
                adapter_kwargs["init_mode"] = init_mode
            if benchmark == "yam_real":
                from .yam_real.station import build_real_station

                adapter = adapter_type(
                    build_real_station(station, config_root=station_config_root),
                    registry=registry,
                    run_observer=recorder,
                    allow_physical_motion=allow_motion,
                )
                recorder.frequency = adapter.control_frequency
            else:
                adapter = adapter_type(**adapter_kwargs)
        with recorder.timing.measure("reset"):
            adapter.reset(metadata, seed)
        # Take the embodiment's real control rate before anything is recorded:
        # RunRecorder defaults to 20 Hz, and an adapter that knows its own rate
        # exposes control_frequency. Probed after reset, not before, because an
        # adapter can only know its rate once its environment exists; frequency
        # is read only at finalization, so this ordering is safe.
        rate = getattr(adapter, "control_frequency", None)
        if rate:
            recorder.frequency = float(rate)
        if benchmark == "robosuite" and metadata.task_name in {
            "two_arm_lift",
            "two_arm_handover",
        }:
            from cap_harness.validation.evaluators.robosuite_bimanual import (
                RobosuiteBimanualProtocolEvaluator,
            )

            protocol_evaluator = RobosuiteBimanualProtocolEvaluator(
                metadata.task_name,
                adapter.native_env,
            )
            adapter.bind_protocol_evaluator(protocol_evaluator)
        elif benchmark == "behavior":
            from .validation.evaluators.behavior_pickup import BehaviorPickupWitness

            protocol_evaluator = BehaviorPickupWitness(
                metadata.task_name,
                adapter.native_env,
                target_scope=metadata.target_scope,
            )
            adapter.bind_protocol_evaluator(protocol_evaluator)
        # Service URLs are overridable so a run can point at a private, freshly
        # started service (CAP_HARNESS_SAM3_URL, CAP_HARNESS_CUROBO_URL) instead
        # of the shared stack.
        segmentation = TracedProvider(
            Sam3Provider(base_url=os.environ.get("CAP_HARNESS_SAM3_URL", DEFAULT_SAM3_URL)),
            recorder,
            "sam3",
        )
        contact_graspnet = TracedProvider(ContactGraspNetProvider(), recorder, "contact_graspnet")
        if benchmark == "behavior":
            from .contracts import MotionStrategy

            planner = TracedProvider(adapter.planner(), recorder, "curobo")  # type: ignore[attr-defined]
            api = CapApi(
                adapter,
                segmentation_provider=segmentation,
                grasp_providers={"contact-graspnet": contact_graspnet},
                ik_providers={"curobo": planner},
                trajectory_planning_providers={"curobo": planner},
                integrated_pose_planning_providers={"curobo-integrated": planner},
                default_camera="head",
                default_motion_strategy=MotionStrategy(
                    ik_solver="curobo",
                    trajectory_planner="curobo",
                    pose_planner="curobo-integrated",
                ),
                recorder=recorder,
            )
        else:
            pyroki = TracedProvider(PyRokiProvider(), recorder, "pyroki")
            curobo = TracedProvider(
                CuRoboProvider(
                    base_url=os.environ.get("CAP_HARNESS_CUROBO_URL", DEFAULT_CUROBO_URL),
                    **(
                        {"control_period_s": adapter.control_period_s}
                        if benchmark == "yam_real"
                        else {}
                    ),
                ),
                recorder,
                "curobo",
            )
            ik_providers = {"pyroki": pyroki, "curobo": curobo}
            if benchmark == "yam_real":
                from .providers.yam_kinematics import YamKinematicsIKProvider
                from .yam_real.kinematics import YamKinematics

                ik_providers = {
                    "mink": TracedProvider(
                        YamKinematicsIKProvider(YamKinematics(adapter.native_env.config.model_xml)),
                        recorder,
                        "mink",
                    ),
                    "curobo": curobo,
                }
            api = CapApi(
                adapter,
                segmentation_provider=segmentation,
                grasp_providers={"contact-graspnet": contact_graspnet},
                ik_providers=ik_providers,
                trajectory_planning_providers={"curobo": curobo},
                integrated_pose_planning_providers={"curobo-integrated": curobo},
                default_camera=adapter.native_env.config.camera.role
                if benchmark == "yam_real"
                else "agentview",
                recorder=recorder,
            )
        tools = api.register_tools(
            ToolRegistry(public_extension_allowlist=extension_allowlist)
            if extension_allowlist is not None
            else ToolRegistry()
        )
        execution = ProgramExecutor(tools, recorder=recorder).execute_program(source)
        sanitized_stdout = redact(execution.stdout)
        (recorder.root / "logs/program.stdout").write_text(str(sanitized_stdout), encoding="utf-8")
        (recorder.root / "logs/program.stderr").write_text("", encoding="utf-8")
        task_success = adapter.check_success()  # type: ignore[attr-defined]
        save_protocol_evidence(bool(task_success))
        if recorder.terminal_reason == "step_limit":
            termination_reason = "step_limit"
        elif task_success:
            termination_reason = "task_succeeded"
        elif recorder.terminated:
            termination_reason = "environment_terminated"
        elif recorder.truncated:
            termination_reason = "environment_truncated"
        elif not execution.ok:
            termination_reason = "program_error"
        else:
            termination_reason = "program_completed"
        program_ok = execution.ok and termination_reason != "step_limit"
        error = execution.error if not execution.ok else None
        close_before_recording()
        recorder.finalize(
            program_ok=program_ok,
            task_success=task_success,
            protocol_success=protocol_success,
            termination_reason=termination_reason,
            program_result=execution,
            error=error,
            cleanup_errors=tuple(cleanup_errors),
            success_observed_step=getattr(adapter, "success_observed_step", None),
        )
    except KeyboardInterrupt:
        if adapter is not None:
            try:
                task_success = adapter.check_success()  # type: ignore[attr-defined]
            # Record teardown errors without discarding the episode evidence.
            except Exception:
                task_success = None
        save_protocol_evidence(bool(task_success))
        close_before_recording()
        recorder.finalize(
            program_ok=False,
            task_success=task_success,
            protocol_success=protocol_success,
            termination_reason="user_interrupt",
            program_result=execution,
            error={"type": "KeyboardInterrupt", "message": "execution interrupted"},
            cleanup_errors=tuple(cleanup_errors),
        )
        raise
    except StepLimitReached as exc:
        # Hitting the step budget says nothing about the task: a program that
        # completed the task and kept stepping still succeeded. Ask the adapter,
        # as the interrupt branch does, instead of asserting failure.
        termination_reason = "step_limit"
        task_success = None
        if adapter is not None:
            try:
                task_success = adapter.check_success()  # type: ignore[attr-defined]
            except Exception:
                task_success = None
        save_protocol_evidence(bool(task_success))
        close_before_recording()
        recorder.finalize(
            program_ok=False,
            task_success=task_success,
            protocol_success=protocol_success,
            termination_reason=termination_reason,
            program_result=execution,
            error={"type": type(exc).__name__, "message": str(exc)},
            cleanup_errors=tuple(cleanup_errors),
            success_observed_step=getattr(adapter, "success_observed_step", None),
        )
    except Exception as exc:
        save_protocol_evidence(False)
        close_before_recording()
        recorder.close_incomplete(exc, cleanup_errors=tuple(cleanup_errors))
        raise
    finally:
        close_adapter()
    return RunOutcome(
        run_dir=recorder.root,
        program_ok=bool(execution and execution.ok and termination_reason != "step_limit"),
        task_success=task_success,
        protocol_success=protocol_success,
        termination_reason=termination_reason,
    )


__all__ = ["RunOutcome", "run_program", "run_program_with_api"]
