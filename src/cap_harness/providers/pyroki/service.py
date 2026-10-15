"""PyRoki IK service matching :mod:`cap_harness.providers.pyroki.client`."""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
from pydantic import BaseModel
import pyroki as pk
import uvicorn

from cap_harness.providers.service_runtime import install_service_runtime

from .wire import IK_PATH

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="cap-harness PyRoki provider")
# Bound concurrent GPU/inference requests so one instance safely serves
# multiple simulator clients (limit from CAP_HARNESS_SERVICE_CONCURRENCY).
install_service_runtime(app)
_ROBOT: pk.Robot | None = None
_TARGET_LINK_INDEX: int | None = None


class IkRequest(BaseModel):
    target_pose_wxyz_xyz: list[float]
    prev_cfg: list[float] | None = None


class IkResponse(BaseModel):
    joint_positions: list[float]
    position_error_m: float
    orientation_error_rad: float


@jdc.jit
def _solve_seeded_ik(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    seed: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(seed)]),
        )
    )
    return solution[joint_var]


def _solve(request: IkRequest) -> IkResponse:
    assert _ROBOT is not None and _TARGET_LINK_INDEX is not None
    target = np.asarray(request.target_pose_wxyz_xyz, dtype=np.float64)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise HTTPException(
            status_code=400, detail="target_pose_wxyz_xyz must contain seven finite values"
        )
    if request.prev_cfg is None:
        seed = np.zeros(_ROBOT.joints.num_actuated_joints, dtype=np.float64)
    else:
        seed = np.asarray(request.prev_cfg, dtype=np.float64)
    expected = (_ROBOT.joints.num_actuated_joints,)
    if seed.shape != expected or not np.all(np.isfinite(seed)):
        raise HTTPException(
            status_code=400, detail=f"prev_cfg must contain {expected[0]} finite values"
        )
    result = _solve_seeded_ik(
        _ROBOT,
        jnp.asarray(_TARGET_LINK_INDEX),
        jnp.asarray(target[:4]),
        jnp.asarray(target[4:]),
        jnp.asarray(seed),
    )
    solved_pose = np.asarray(_ROBOT.forward_kinematics(result)[_TARGET_LINK_INDEX])
    quaternion_dot = float(np.clip(abs(np.dot(solved_pose[:4], target[:4])), 0.0, 1.0))
    return IkResponse(
        joint_positions=np.asarray(result, dtype=float).tolist(),
        position_error_m=float(np.linalg.norm(solved_pose[4:] - target[4:])),
        orientation_error_rad=float(2.0 * np.arccos(quaternion_dot)),
    )


@app.post(IK_PATH, response_model=IkResponse)
async def solve_ik(request: IkRequest) -> IkResponse:
    if _ROBOT is None:
        raise HTTPException(status_code=503, detail="PyRoki not initialized")
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(_solve, request))
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("IK failed")
        raise HTTPException(status_code=500, detail=f"IK solve failed: {exc}") from exc


def _apply_joint_margin(urdf: Any, margin: float = 0.15) -> Any:
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower += margin
                joint.limit.upper -= margin
    return urdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cap-harness PyRoki IK service")
    parser.add_argument("--robot", default="panda_description")
    parser.add_argument("--target-link", default="panda_hand")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    args = parser.parse_args()

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    global _ROBOT, _TARGET_LINK_INDEX
    _ROBOT = pk.Robot.from_urdf(_apply_joint_margin(load_robot_description(args.robot)))
    _TARGET_LINK_INDEX = _ROBOT.links.names.index(args.target_link)
    LOGGER.info("PyRoki loaded for %s -> %s", args.robot, args.target_link)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
