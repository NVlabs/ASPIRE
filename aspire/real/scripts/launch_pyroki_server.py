"""Pyroki IK / trajectory service — HTTP at ``host:port`` (default 127.0.0.1:9600).

Runs one FastAPI server that loads the PandaOmron (or any URDF you ask for)
once, then answers ``POST /ik`` and ``POST /plan`` from many CAP clients. Same
pattern as the cuRobo server at ``scripts/launch_curobo_server.py`` — 1:1,
one service per node.

Endpoints
---------
``POST /ik``    — nearest-neighbour IK anchored on ``prev_cfg``.
``POST /plan``  — linear interpolation in SE(3) + IK at each waypoint.

Run
---
Plain::

    uv run python scripts/launch_pyroki_server.py --port 9600

On OSMO (mirrors the cuRobo server pattern)::

    uv run python scripts/launch_pyroki_server.py \
        --robot panda_description --target-link panda_hand --port 9600

Ported from cap-x (MIT) at ``capx/serving/launch_pyroki_server.py``; sphere
decomposition is the one that ships with ``third_party/pyroki/examples/assets``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
from pathlib import Path

import numpy as np
import pyroki as pk  # type: ignore
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation, Slerp

from cap.integrations.motion import pyroki_snippets as pks  # type: ignore


# ---------------------------------------------------------------------------
# Planning helpers (linear-interp + per-waypoint IK)
# ---------------------------------------------------------------------------


def _slerp_quaternions(
    q_start: np.ndarray, q_end: np.ndarray, num_steps: int
) -> np.ndarray:
    """SLERP between two ``wxyz`` quaternions. Returns ``(num_steps, 4)`` wxyz."""
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])
    key_rots = Rotation.concatenate([r_start, r_end])
    slerp = Slerp([0, 1], key_rots)
    times = np.linspace(0, 1, num_steps)
    xyzw = slerp(times).as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]])


def _fk_link_positions(
    robot: pk.Robot,
    target_link_name: str,
    cfgs: np.ndarray,
) -> np.ndarray:
    """Return FK positions for ``target_link_name`` over a joint trajectory."""
    if target_link_name not in robot.links.names:
        raise ValueError(f"Unknown target link {target_link_name!r}")
    link_idx = robot.links.names.index(target_link_name)
    cfgs = np.asarray(cfgs, dtype=np.float64)
    if cfgs.ndim == 1:
        cfgs = cfgs[None, :]
    fk = np.asarray(robot.forward_kinematics(cfgs))
    return np.asarray(fk[:, link_idx, 4:7], dtype=np.float64)


def _fk_link_pose(
    robot: pk.Robot,
    target_link_name: str,
    cfg: np.ndarray,
) -> np.ndarray:
    """Return one ``wxyz_xyz`` FK pose for ``target_link_name``."""
    if target_link_name not in robot.links.names:
        raise ValueError(f"Unknown target link {target_link_name!r}")
    link_idx = robot.links.names.index(target_link_name)
    fk = np.asarray(robot.forward_kinematics(np.asarray(cfg, dtype=np.float64)[None, :]))
    return np.asarray(fk[0, link_idx], dtype=np.float64)


def _plan_trajectory_linear_ik(
    robot: pk.Robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    start_cfg: np.ndarray | None = None,
    num_waypoints: int = 25,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    """Straight-line SE(3) interpolation + per-waypoint IK with vel-cost anchoring.

    Returns ``(num_waypoints, num_joints)``. ``jump_threshold`` in radians is a
    sanity check — large steps between consecutive IK solutions are logged as
    warnings (usually a sign that the trajectory crosses an IK branch cut).
    """
    if target_link_name not in robot.links.names:
        raise ValueError(f"Unknown target link {target_link_name!r}")
    num_waypoints = max(int(num_waypoints), 3)
    positions = np.linspace(start_pos, end_pos, num_waypoints)
    orientations = _slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    trajectory: list[np.ndarray] = []
    prev_cfg: np.ndarray | None = None if start_cfg is None else np.asarray(start_cfg)
    jump_warnings: list[str] = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations)):
        if prev_cfg is not None:
            cfg = pks.solve_ik_vel_cost(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
                prev_cfg=prev_cfg,
                initial_cfg=prev_cfg,
            )
        else:
            cfg = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.asarray(cfg)

        if len(trajectory) > 0:
            joint_diff = np.abs(cfg - trajectory[-1])
            for joint_idx in np.where(joint_diff > jump_threshold)[0]:
                jump_warnings.append(
                    f"waypoint {i}: joint {joint_idx} jumped "
                    f"{np.degrees(joint_diff[joint_idx]):.1f}°"
                )

        trajectory.append(cfg)
        prev_cfg = cfg

    if jump_warnings:
        logger.warning(
            "Pyroki trajopt: %d large joint jump(s) detected (threshold %.1f°)",
            len(jump_warnings),
            np.degrees(jump_threshold),
        )
        for w in jump_warnings[:5]:
            logger.warning("  %s", w)

    return np.asarray(trajectory)


# ---------------------------------------------------------------------------
# FastAPI plumbing
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[pyroki-server] %(message)s")
logger = logging.getLogger("pyroki_server")

app = FastAPI()

_ROBOT: pk.Robot | None = None
_ROBOT_COLL = None
_TARGET_LINK: str | None = None


async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


class IkRequest(BaseModel):
    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None
    target_link_name: str | None = None


class IkResponse(BaseModel):
    joint_positions: list[float]
    joint_names: list[str] | None = None


class PlanRequest(BaseModel):
    start_pose_wxyz_xyz: list[float]
    end_pose_wxyz_xyz: list[float]
    target_link_name: str | None = None
    start_cfg: list[float] | None = None
    timesteps: int = 20
    dt: float = 0.02


class PlanResponse(BaseModel):
    waypoints: list[list[float]]
    dt: float
    joint_names: list[str] | None = None
    start_pos_error_m: float | None = None
    end_pos_error_m: float | None = None
    max_pos_error_m: float | None = None
    fk_cartesian_distance_m: float | None = None
    requested_cartesian_distance_m: float | None = None


class HealthResponse(BaseModel):
    ready: bool
    robot: str | None
    target_link: str | None
    joint_names: list[str] | None = None
    link_names: list[str] | None = None


# ---------------------------------------------------------------------------
# URDF loading
# ---------------------------------------------------------------------------


def _set_min_distance_from_limits(urdf, min_distance: float = 0.15):
    """Shrink joint limits inward so IK stays comfortably inside them."""
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance
                joint.limit.upper = joint.limit.upper - min_distance
    return urdf


def _sphere_decomposition_path(robot_urdf_name: str) -> Path | None:
    """Return the panda sphere decomposition that ships with pyroki examples."""
    if robot_urdf_name != "panda_description":
        return None
    forge_root = Path(__file__).resolve().parents[1]
    candidate = (
        forge_root
        / "third_party"
        / "pyroki"
        / "examples"
        / "assets"
        / "panda_spheres.json"
    )
    return candidate if candidate.exists() else None


def _load_urdf(robot_urdf_name: str, urdf_path: str | None):
    if urdf_path:
        import yourdfpy

        path = Path(urdf_path).expanduser().resolve()
        logger.info("Loading URDF from %s", path)
        return yourdfpy.URDF.load(str(path))

    if robot_urdf_name in ("yam", "yam_station"):
        import yourdfpy
        from robot.models.station.paths import get_station_urdf

        path = Path(get_station_urdf()).resolve()
        logger.info("Loading YAM station URDF from %s", path)
        return yourdfpy.URDF.load(str(path))

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    logger.info("Loading URDF %r via robot_descriptions", robot_urdf_name)
    return load_robot_description(robot_urdf_name)


def _init_pyroki(
    robot_urdf_name: str,
    target_link_name: str,
    urdf_path: str | None = None,
) -> None:
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK

    urdf = _load_urdf(robot_urdf_name, urdf_path)
    if robot_urdf_name not in ("yam", "yam_station") and urdf_path is None:
        urdf = _set_min_distance_from_limits(urdf)
    _ROBOT = pk.Robot.from_urdf(urdf)

    spheres_path = _sphere_decomposition_path(robot_urdf_name)
    if spheres_path is not None:
        logger.info("Loading sphere decomposition from %s", spheres_path)
        decomp = json.loads(spheres_path.read_text())
        _ROBOT_COLL = pk.collision.RobotCollision.from_sphere_decomposition(
            sphere_decomposition=decomp,
            urdf=urdf,
        )
    else:
        logger.info(
            "No sphere decomposition available for %r — /plan will still work "
            "for free-space motions, but collision-aware calls will degrade",
            robot_urdf_name,
        )

    _TARGET_LINK = target_link_name
    logger.info(
        "Pyroki server ready — robot=%r target_link=%r",
        robot_urdf_name,
        target_link_name,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _do_solve_ik(
    target_pose_wxyz_xyz: np.ndarray,
    prev_cfg: np.ndarray | None,
    target_link_name: str,
) -> list[float]:
    if prev_cfg is None:
        q = pks.solve_ik(
            robot=_ROBOT,
            target_link_name=target_link_name,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
        )
    else:
        q = pks.solve_ik_vel_cost(
            robot=_ROBOT,
            target_link_name=target_link_name,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
            prev_cfg=prev_cfg,
            initial_cfg=prev_cfg,
        )
    return [float(x) for x in q]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        ready=_ROBOT is not None,
        robot="loaded" if _ROBOT is not None else None,
        target_link=_TARGET_LINK,
        joint_names=list(_ROBOT.joints.actuated_names) if _ROBOT is not None else None,
        link_names=list(_ROBOT.links.names) if _ROBOT is not None else None,
    )


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest) -> IkResponse:
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")

    target = np.asarray(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = (
        np.asarray(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None
    )
    target_link_name = req.target_link_name or _TARGET_LINK

    try:
        joints = await _run_in_thread(_do_solve_ik, target, prev_cfg, target_link_name)
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")

    return IkResponse(
        joint_positions=joints,
        joint_names=list(_ROBOT.joints.actuated_names) if _ROBOT is not None else None,
    )


def _do_plan_motion(req: PlanRequest) -> PlanResponse:
    start_pose = np.asarray(req.start_pose_wxyz_xyz, dtype=np.float64)
    end_pose = np.asarray(req.end_pose_wxyz_xyz, dtype=np.float64)
    start_cfg = (
        np.asarray(req.start_cfg, dtype=np.float64) if req.start_cfg is not None else None
    )
    target_link_name = req.target_link_name or _TARGET_LINK
    if start_cfg is not None:
        # Anchor the path to PyRoki FK of the current joint state.  This avoids
        # an initial correction caused by small CAP-FK/PyRoki-FK differences.
        start_pose = _fk_link_pose(_ROBOT, target_link_name, start_cfg)
    traj = _plan_trajectory_linear_ik(
        robot=_ROBOT,
        target_link_name=target_link_name,
        start_pos=start_pose[4:],
        start_wxyz=start_pose[:4],
        end_pos=end_pose[4:],
        end_wxyz=end_pose[:4],
        start_cfg=start_cfg,
        num_waypoints=req.timesteps,
    )
    requested_positions = np.linspace(start_pose[4:], end_pose[4:], traj.shape[0])
    fk_positions = _fk_link_positions(_ROBOT, target_link_name, traj)
    pos_errors = np.linalg.norm(fk_positions - requested_positions, axis=1)
    return PlanResponse(
        waypoints=np.asarray(traj).tolist(),
        dt=float(req.dt),
        joint_names=list(_ROBOT.joints.actuated_names) if _ROBOT is not None else None,
        start_pos_error_m=float(pos_errors[0]),
        end_pos_error_m=float(pos_errors[-1]),
        max_pos_error_m=float(np.max(pos_errors)),
        fk_cartesian_distance_m=float(np.linalg.norm(fk_positions[-1] - fk_positions[0])),
        requested_cartesian_distance_m=float(
            np.linalg.norm(end_pose[4:] - start_pose[4:])
        ),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_motion(req: PlanRequest) -> PlanResponse:
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")
    try:
        return await _run_in_thread(_do_plan_motion, req)
    except Exception as e:
        logger.exception("Planning failure")
        raise HTTPException(500, f"Motion planning failed: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--robot",
        default="panda_description",
        help="robot_descriptions name (default: panda_description)",
    )
    parser.add_argument(
        "--target-link",
        default="panda_hand",
        help="End-effector link name (default: panda_hand)",
    )
    parser.add_argument(
        "--urdf-path",
        default=None,
        help="Direct URDF path. Overrides --robot when provided.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9600)
    args = parser.parse_args()

    _init_pyroki(
        robot_urdf_name=args.robot,
        target_link_name=args.target_link,
        urdf_path=args.urdf_path,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
