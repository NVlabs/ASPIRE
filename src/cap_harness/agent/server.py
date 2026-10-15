"""Standalone YAM System 2 server, also embeddable as an ASGI application."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import threading
from typing import Any

from .embedded import EmbeddedAgentRuntime
from .mcp import create_lifespan, create_mcp_transport


def create_app(session: Any) -> Any:
    """Serve MCP and operator steering without creating another environment."""
    from fastapi import FastAPI
    from starlette.routing import Mount

    manager, endpoint = create_mcp_transport(session)
    app = FastAPI(lifespan=create_lifespan(manager))
    app.router.routes.append(Mount("/mcp", app=endpoint))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "job": session.job()}

    @app.post("/steer")
    def steer(payload: dict[str, str]) -> dict[str, Any]:
        event = session.steer(payload["message"])
        return {"ok": True, "seq": event.seq}

    return app


def build_yam_runtime(
    env: Any,
    *,
    allow_motion: bool = False,
    episode_recorder: Any = None,
    event_sink: Any = None,
    tool_result_sink: Any = None,
    **api_options: Any,
) -> tuple[Any, EmbeddedAgentRuntime]:
    """Attach to a caller-owned station; construction performs no motion."""
    from cap_harness.api import CapApi
    from cap_harness.contracts import TaskContext
    from cap_harness.registry import YAM_REAL_PUBLIC_TOOL_NAMES, ToolRegistry
    from cap_harness.yam_real.adapter import YamRealAdapter
    from cap_harness.yam_real.control.runtime import RuntimeControl
    from cap_harness.yam_real.live_adapter import LiveYamAdapter

    raw = YamRealAdapter(
        env,
        allow_physical_motion=allow_motion,
        task_context=TaskContext(
            suite="system2",
            task_id=0,
            task_name="system2",
            language="Follow the operator goal.",
            family="yam_real",
        ),
    )
    control = RuntimeControl(raw)
    control.episode_recorder = episode_recorder
    adapter = LiveYamAdapter(raw, control)
    try:
        api = CapApi(
            adapter,
            default_camera=env.config.camera.role,
            **api_options,
        )
        registry = api.register_tools(
            ToolRegistry(public_extension_allowlist=YAM_REAL_PUBLIC_TOOL_NAMES)
        )
        runtime = EmbeddedAgentRuntime(
            api,
            registry,
            operation_controller=control,
            event_sink=event_sink,
            tool_result_sink=tool_result_sink,
        )
    except BaseException:
        control.close()
        raise
    return adapter, runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="yam-example")
    parser.add_argument("--station-config-root", type=Path)
    parser.add_argument("--sim", action="store_true", help="Use the off-robot station")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8222)
    parser.add_argument("--viser-port", type=int, default=8080, help="Camera viewer and Home button")
    parser.add_argument("--no-viser", action="store_true", help="Disable the browser visualizer")
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--sam3-url")
    parser.add_argument("--curobo-url")
    args = parser.parse_args(argv)
    from contextlib import ExitStack

    import uvicorn

    from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider
    from cap_harness.yam_real.kinematics import YamKinematics
    from cap_harness.yam_real.station import build_real_station, build_sim_station

    with ExitStack() as stack:
        factory = build_sim_station if args.sim else build_real_station
        env = factory(args.station, config_root=args.station_config_root)
        stack.callback(env.close)
        episode = None
        event_sink = tool_sink = None
        if args.record_dir:
            from cap_harness.yam_real.recorder import YamEpisodeRecorder

            args.record_dir.mkdir(parents=True, exist_ok=False)
            episode = YamEpisodeRecorder(
                args.record_dir / "episode",
                video_dir=args.record_dir / "videos",
                fps=round(env.config.controller.control_frequency_hz),
                cameras=() if args.no_video else None,
                task="system2",
            )
            episode.start(
                env, meta={"station": args.station, "physical_motion_authorized": args.allow_motion}
            )
            stack.callback(episode.finalize, terminal_event="session_closed")
            log_lock = threading.Lock()

            def record(path: str, value: Any) -> None:
                with log_lock, (args.record_dir / path).open("a") as stream:
                    stream.write(json.dumps(value, default=str) + "\n")

            event_sink = lambda event: record("events.jsonl", asdict(event))
            tool_sink = lambda **result: record("tools.jsonl", result)
        options: dict[str, Any] = {
            "ik_providers": {"mink": YamKinematicsIKProvider(YamKinematics(env.config.model_xml))}
        }
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            options["segmentation_provider"] = Sam3Provider(base_url=args.sam3_url)
        if args.curobo_url:
            from cap_harness.providers.curobo import CuRoboProvider

            planner = CuRoboProvider(
                base_url=args.curobo_url,
                control_period_s=1 / env.config.controller.control_frequency_hz,
            )
            options["ik_providers"]["curobo"] = planner
            options["trajectory_planning_providers"] = {"curobo": planner}
            options["integrated_pose_planning_providers"] = {"curobo-integrated": planner}
        adapter, runtime = build_yam_runtime(
            env,
            allow_motion=args.allow_motion,
            episode_recorder=episode,
            event_sink=event_sink,
            tool_result_sink=tool_sink,
            **options,
        )
        # Stop jobs before finalizing recordings, then release cameras and arms.
        stack.callback(runtime.close)
        if not args.no_viser:
            from .visualizer import System2Visualizer

            visualizer = System2Visualizer(
                runtime.session, allow_motion=args.allow_motion, host=args.host, port=args.viser_port
            )
            stack.callback(visualizer.close)
        uvicorn.run(create_app(runtime.session), host=args.host, port=args.port)
    return 0
