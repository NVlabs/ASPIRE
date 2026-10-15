"""FastAPI service for cuRobo V2 IK and collision-aware motion planning."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import functools
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import uvicorn

from cap_harness.contracts import JOINT_DIMENSIONS
from cap_harness.providers.service_runtime import install_service_runtime

from ..wire import decode_numpy, encode_numpy
from .dual import build_dual_panda_config
from .wire import IK_PATH, PLAN_PATH, PLAN_SYNCHRONIZED_PATH

app = FastAPI(title="cap-harness cuRobo V2 provider")
# Bound concurrent GPU/inference requests so one instance safely serves
# multiple simulator clients (limit from CAP_HARNESS_SERVICE_CONCURRENCY).
install_service_runtime(app)
_PLANNERS: dict[tuple[str, float | None, float | None], Any] = {}
_IK_SOLVERS: dict[str, Any] = {}
_SEGMENTATION_KINEMATICS: dict[str, Any] = {}
_SEGMENTATION_TOLERANCE_M = 0.02
_LOGGER = logging.getLogger(__name__)


def _collision_disabled(model: str) -> bool:
    """Whether to drop every collision term from cuRobo.

    Collision checking is on for every model. ``CAP_HARNESS_CUROBO_DISABLE_COLLISION``
    disables it everywhere (any value but ``0``). It is read per request rather than
    at import, so a service restart is all it takes to change, and logged on every
    use so a blind plan cannot be mistaken for a checked one in a recorded run.
    """
    override = os.environ.get("CAP_HARNESS_CUROBO_DISABLE_COLLISION")
    return override is not None and override != "0"


_ROBOT_SEGMENTATION_DISTANCE_M = 0.05

#: Peak float64 elements per distance-matrix chunk in the robot carve. The dense
#: set is ~1250 spheres against ~46k points; chunking on the product rather than
#: a fixed point count keeps that under a few hundred MB whichever set is used.
_CARVE_CHUNK_ELEMENTS = 16_000_000


class Target(BaseModel):
    type: Literal["pose", "joints"]
    position: list[float] | None = None
    quaternion_wxyz: list[float] | None = None
    frame: str | None = None
    joint_positions: list[float] | None = None


class BaseRequest(BaseModel):
    base_frame: str
    model: Literal["panda", "dual_panda", "yam_real"]
    joint_positions: dict[str, list[float]]
    joint_names: dict[str, list[str]]
    base_transforms: dict[str, list[list[float]]]
    end_effector_links: dict[str, str]
    scene_points_base64: str | None = None
    voxel_size_m: float = 0.01
    interpolation_dt_s: float | None = None
    maximum_trajectory_dt_s: float | None = None


class IkRequest(BaseRequest):
    arm: str
    target: Target


class PlanRequest(BaseRequest):
    arm: str
    target: Target


class SynchronizedPlanRequest(BaseRequest):
    targets: dict[str, Target]


class PlanResponse(BaseModel):
    joint_positions_base64: str
    dt_s: float


class SynchronizedPlanResponse(BaseModel):
    joint_positions_base64: dict[str, str]
    dt_s: float


@functools.lru_cache(maxsize=1)
def _yam_real_robot_config() -> str:
    """CuRobo description of the physical bimanual station.

    Cached, and it must be. This writes a temp file whose name is random, and
    ``_planner`` keys its planner cache on that name. Uncached, every plan request
    produced a new name, so the key never repeated, so the cache never hit, so
    each request built a fresh ``MotionPlanner`` and called ``warmup`` -- which
    records CUDA graphs into private pools that are never released.

    The service then dies of an out-of-memory that looks like nothing to do with
    it: every plan is refused, and the arm reads as unable to reach anywhere.
    Measured on hardware before this decorator existed: 159 plan requests,
    158 solver initializations, 158 temp configs, and a bimanual planner leaked
    per request until 20 MiB could not be allocated. ``_yam_sim_robot_config``
    has always been cached, which is why only the real station showed it.

    Distinct from the YAM Sim config, which describes a **single** arm
    (``joint1..6`` with the fingers locked). The real station carries both arms
    in one model 0.62 m apart, and they can reach each other -- arm-to-arm
    self-collision is the thing a bimanual planner is for, and a single-arm
    description cannot express it at any setting.

    The config and the URDF are a matched pair and must stay one: this URDF
    declares the four finger joints ``fixed``, so they are absent from the
    kinematic tree, and its config correspondingly locks nothing. Pairing a
    config that locks the fingers with a URDF that articulates them (or the
    reverse) is rejected at load time. Its collision spheres are inline, because
    v2 will not load them from a path.

    The shipped config is in cuRobo's v1 schema, so it is converted here. Each
    step below is load-bearing; without them the v2 loader either refuses the
    config or accepts a subtly broken one.
    """
    configured = os.environ.get("CAP_HARNESS_CUROBO_YAM_REAL_CONFIG")
    station = Path(__file__).resolve().parents[2] / "yam_real" / "description" / "station"
    source = (
        Path(configured)
        if configured
        else station / "curobo" / "yam_dual_isaacsim_physics_fixed_fingers.yml"
    )
    if not source.is_file():
        raise FileNotFoundError(f"YAM real cuRobo config not found: {source}")

    import yaml

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]

    urdf = (station / kinematics["urdf_path"]).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"YAM real station URDF not found: {urdf}")
    # v1 keys the v2 loader rejects outright. The USD group describes an Isaac
    # asset v2 neither reads nor tolerates; the URDF is the real source.
    for legacy in (
        "use_usd_kinematics",
        "usd_path",
        "usd_robot_root",
        "isaac_usd_path",
        "usd_flip_joints",
        "usd_flip_joint_limits",
        "asset_root_path",
    ):
        kinematics.pop(legacy, None)

    # v1 named one ee_link plus a link_names list; v2 takes tool_frames. BOTH
    # grasp frames have to survive -- keeping only ee_link leaves the right arm
    # with no tool frame, and every right-arm goal becomes unsolvable.
    end_effector = kinematics.pop("ee_link", "left_grasp")
    link_names = kinematics.pop("link_names", None) or []
    kinematics["tool_frames"] = list(dict.fromkeys([end_effector, *link_names]))

    # The self-collision ignore map is left exactly as shipped. An earlier version
    # of this function stripped a pair from it, on the inherited claim that the
    # config "ignores every populated pair" and that cuRobo divides by zero on an
    # empty table. That claim is false for this config: 15 collision links give
    # 105 pairs and the ignore map removes about 6, so ~99 pairs are checked. The
    # mutation was also a no-op here, since neither link ignored the other.

    cspace = kinematics.get("cspace") or {}
    if "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")

    kinematics["format_version"] = 2.0
    kinematics["urdf_path"] = str(urdf)

    output = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="cap_yam_real_curobo_", delete=False
    )
    with output:
        yaml.safe_dump(config, output, sort_keys=False)
    return output.name


def _yam_real_segmentation_config() -> str:
    """The station model again, but wearing segmentation geometry.

    Same URDF, same kinematics, same joint order as the planning config -- only
    the sphere set differs. Building it as a second config rather than editing
    the first is the whole point of the fix: the planner keeps its sparse,
    fast self-collision skeleton, and the carve gets geometry that actually
    follows the arm.

    Self-collision is stripped rather than translated. This model never plans;
    it exists to answer "where is the robot right now" so those points can be
    subtracted from the depth cloud, and carrying a 20-link pair table for
    that would only be a second thing to keep consistent.
    """
    import yaml

    source = Path(_yam_real_robot_config())
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]

    spheres_path = (
        Path(__file__).resolve().parents[2]
        / "yam_real/description/station/curobo/yam_dual_segmentation_spheres.yml"
    )
    if not spheres_path.is_file():
        raise FileNotFoundError(
            f"YAM segmentation spheres not found: {spheres_path}; "
            "reinstall the packaged YAM descriptions"
        )
    document = yaml.safe_load(spheres_path.read_text(encoding="utf-8"))
    spheres = document["collision_spheres"]

    kinematics["collision_spheres"] = spheres
    kinematics["collision_link_names"] = sorted(spheres)
    kinematics["mesh_link_names"] = sorted(spheres)
    # The planning config adds 5 mm to every sphere. The segmentation set states
    # its own radii and its tolerance is applied at the carve, so no hidden
    # inflation here -- otherwise the effective carve is a number that appears
    # in neither file.
    kinematics["collision_sphere_buffer"] = 0.0
    kinematics["self_collision_ignore"] = {}
    kinematics["self_collision_buffer"] = dict.fromkeys(spheres, 0.0)

    output = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="cap_yam_real_segmentation_", delete=False
    )
    with output:
        yaml.safe_dump(config, output, sort_keys=False)
    return output.name


def _segmentation_kinematics(request: BaseRequest) -> Any | None:
    """Kinematics carrying the dense segmentation spheres, or None if unmodelled.

    Only real YAM has a segmentation set. Every other model falls back to its
    self-collision spheres, which is what shipped and is left alone here.
    """
    if request.model != "yam_real":
        return None
    cached = _SEGMENTATION_KINEMATICS.get(request.model)
    if cached is not None:
        return cached
    from curobo.kinematics import Kinematics, KinematicsCfg

    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(_yam_real_segmentation_config()))
    _SEGMENTATION_KINEMATICS[request.model] = kinematics
    return kinematics


def _robot_config(request: BaseRequest) -> str:
    if request.model == "panda":
        return os.environ.get("CAP_HARNESS_CUROBO_PANDA_CONFIG", "franka.yml")
    if request.model == "yam_real":
        return _yam_real_robot_config()
    value = os.environ.get("CAP_HARNESS_CUROBO_DUAL_CONFIG")
    if value and Path(value).is_file():
        return value
    try:
        import curobo

        curobo_root = Path(curobo.__file__).resolve().parents[1]
        secondary = np.asarray(request.base_transforms["secondary"], dtype=np.float64)
        return str(
            build_dual_panda_config(
                curobo_root,
                secondary,
                Path(os.environ.get("CAP_HARNESS_CUROBO_MODEL_CACHE", "/tmp/curobo-models")),
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"could not build dual-Panda model: {exc}"
        ) from exc


def _filter_robot_points(
    points: np.ndarray,
    spheres: np.ndarray,
    *,
    padding_m: float,
) -> np.ndarray:
    """Remove depth points belonging to the current collision-sphere model."""
    keep = np.ones(len(points), dtype=bool)
    stride = max(1024, _CARVE_CHUNK_ELEMENTS // max(len(spheres), 1))
    for start in range(0, len(points), stride):
        chunk = points[start : start + stride]
        distances = np.linalg.norm(chunk[:, None, :] - spheres[None, :, :3], axis=-1)
        keep[start : start + len(chunk)] = np.all(
            distances > spheres[None, :, 3] + padding_m,
            axis=1,
        )
    return points[keep]


def _mesh_from_pointcloud(
    pointcloud: np.ndarray,
    pitch: float,
    name: str,
    pose: list[float],
    mesh_type: Any,
) -> Any:
    """Create a mesh from a pointcloud via voxelized surface extraction.

    Verbatim copy of the pinned cuRobo implementation
    (third_party/curobo curobo/_src/geom/types.py::Mesh.from_pointcloud,
    commit a35a708) except for the single PATCH block below. Upstream winds
    the boundary triangles inward (a one-voxel mesh has volume -pitch**3),
    which inverts warp's signed-distance queries: free space near surfaces
    reads as inside the mesh and the planner refuses reachable goals.
    Delete this copy and call Mesh.from_pointcloud once upstream fixes the
    winding.
    """
    if len(pointcloud) == 0:
        return mesh_type(name, pose=pose, vertices=[[0, 0, 0]], faces=[0, 0, 0])

    pts = np.asarray(pointcloud, dtype=np.float64)
    origin = pts.min(axis=0) - pitch
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)

    grid_shape = ijk.max(axis=0) + 3  # +2 pad so boundary is always empty
    occupied = np.zeros(grid_shape, dtype=bool)
    occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    # For each axis direction, find faces between occupied and empty voxels
    # Face normals: +x, -x, +y, -y, +z, -z
    face_templates = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # -x face
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],  # +x face
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # -y face
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # +y face
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # -z face
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],  # +z face
        ],
        dtype=np.float64,
    )

    verts_list = []
    faces_list = []
    vi = 0  # running vertex index

    axis_shifts = [
        (0, 1, 0, 1),
        (0, -1, 0, 1),
        (1, 1, 2, 3),
        (1, -1, 2, 3),
        (2, 1, 4, 5),
        (2, -1, 4, 5),
    ]

    for axis, direction, neg_tpl, pos_tpl in axis_shifts:
        shifted = np.roll(occupied, -direction, axis=axis)
        if direction > 0:
            boundary = occupied & ~shifted
            tpl = face_templates[pos_tpl]
        else:
            boundary = occupied & ~shifted
            tpl = face_templates[neg_tpl]

        coords = np.argwhere(boundary)
        if len(coords) == 0:
            continue

        quad_verts = coords[:, None, :] + tpl[None, :, :]
        quad_verts = quad_verts.reshape(-1, 3).astype(np.float64) * pitch + origin
        n = len(coords)
        idx = np.arange(n) * 4 + vi
        # PATCH(cap-harness): upstream is `idx, idx + 1, idx + 2 / idx,
        # idx + 2, idx + 3`, winding the triangles inward. Swap two indices
        # per triangle so face normals point out of the occupied voxels.
        tri_faces = np.column_stack(
            [
                idx,
                idx + 2,
                idx + 1,
                idx,
                idx + 3,
                idx + 2,
            ]
        ).reshape(-1, 3)

        verts_list.append(quad_verts)
        faces_list.append(tri_faces)
        vi += n * 4

    if not verts_list:
        return mesh_type(name, pose=pose, vertices=[[0, 0, 0]], faces=[0, 0, 0])

    all_verts = np.concatenate(verts_list, axis=0)
    all_faces = np.concatenate(faces_list, axis=0)

    return mesh_type(
        name, pose=pose, vertices=all_verts.tolist(), faces=all_faces.flatten().tolist()
    )


def _voxelized_scene_mesh(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    mesh_type: Any | None = None,
) -> Any:
    """Build a cuRobo mesh without changing the declared scene resolution."""
    if mesh_type is None:
        from curobo._src.geom.types import Mesh

        mesh_type = Mesh
    return _mesh_from_pointcloud(
        points,
        pitch=float(voxel_size_m),
        name="public_rgbd_scene",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        mesh_type=mesh_type,
    )


_SCENE_MAX_RANGE_M = 5.0


def _drop_distant_points(
    points: np.ndarray, *, max_range_m: float = _SCENE_MAX_RANGE_M
) -> np.ndarray:
    """Drop depth outliers far outside the workspace.

    The voxel surface extraction allocates a dense occupancy grid over the
    cloud's bounding box; a handful of degenerate depth pixels hundreds of
    meters out explode it (observed: a 646 TiB allocation). Planning scenes
    are expressed in the robot base frame, so anything beyond a room radius
    is sensor noise, not reachable geometry.
    """
    return points[np.linalg.norm(points, axis=1) <= max_range_m]


def _carve_robot_points(request: BaseRequest, robot: Any, points: np.ndarray) -> np.ndarray:
    """Remove robot pixels using the embodiment's segmentation geometry."""
    segmentation = _segmentation_kinematics(request)
    source = segmentation if segmentation is not None else robot
    spheres = np.asarray(
        source.compute_kinematics(_state(request, source)).robot_spheres.detach().cpu(),
        dtype=np.float64,
    ).reshape(-1, 4)
    margin = (
        _SEGMENTATION_TOLERANCE_M
        if segmentation is not None
        else max(float(request.voxel_size_m), _ROBOT_SEGMENTATION_DISTANCE_M)
    )
    return _filter_robot_points(points, spheres, padding_m=margin)


def _scene(request: BaseRequest, robot: Any | None = None) -> Any | None:
    if _collision_disabled(request.model):
        _LOGGER.warning(
            "COLLISION DISABLED (CAP_HARNESS_CUROBO_DISABLE_COLLISION is set): the "
            "observed scene is being discarded. Any plan returned is unsafe to execute."
        )
        return None
    if request.scene_points_base64 is None:
        return None
    points = np.asarray(decode_numpy(request.scene_points_base64), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise HTTPException(status_code=400, detail="scene points must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise HTTPException(status_code=400, detail="scene points must be finite")
    points = _drop_distant_points(points)
    if len(points) == 0:
        raise HTTPException(status_code=422, detail="scene contains no points in range")
    if robot is not None:
        try:
            before = len(points)
            points = _carve_robot_points(request, robot, points)
            _LOGGER.info("robot carve (self-collision): %d -> %d points", before, len(points))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"robot point removal failed: {exc}"
            ) from exc
        if len(points) == 0:
            raise HTTPException(status_code=422, detail="scene contains only robot points")
    try:
        from curobo.scene import Scene

        # PointCloud.get_mesh() silently uses cuRobo's 2 cm default pitch.
        # Preserve the public scene resolution so filtered 1 cm voxels do not
        # expand back into the robot and invalidate otherwise feasible goals.
        obstacle = _voxelized_scene_mesh(
            points,
            voxel_size_m=request.voxel_size_m,
        )
        return Scene(mesh=[obstacle])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scene conversion failed: {exc}") from exc


def _purge_scene_mesh_cache(planner: Any) -> None:
    """Drop cached Warp BVHs so update_world loads the new scene geometry.

    cuRobo's mesh cache is keyed by name only (data_mesh.py
    _load_mesh_into_cache): loading a mesh whose name is already cached
    silently reuses the first geometry ever loaded under that name. Every
    request names its scene mesh "public_rgbd_scene", so without this purge
    a cached planner keeps checking collisions against the world from the
    first request it served.
    """
    cache = getattr(
        getattr(getattr(planner, "scene_collision_checker", None), "data", None),
        "meshes",
        None,
    )
    cache = getattr(cache, "wp_cache", None)
    if cache:
        cache.clear()


def _planner(request: BaseRequest) -> Any:
    robot = _robot_config(request)
    interpolation_dt = request.interpolation_dt_s
    maximum_dt = request.maximum_trajectory_dt_s
    for name, value in (
        ("interpolation_dt_s", interpolation_dt),
        ("maximum_trajectory_dt_s", maximum_dt),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise HTTPException(status_code=400, detail=f"{name} must be positive and finite")
    if interpolation_dt is not None and maximum_dt is not None and interpolation_dt > maximum_dt:
        raise HTTPException(
            status_code=400,
            detail="interpolation_dt_s must not exceed maximum_trajectory_dt_s",
        )
    key = (robot, interpolation_dt, maximum_dt)
    planner = _PLANNERS.get(key)
    try:
        if planner is None:
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

            config = MotionPlannerCfg.create(
                robot=robot,
                scene_model=None,
                collision_cache={"mesh": 1},
                # Kept on. If a specific pair ever turns out to be a false
                # positive, name it in ``self_collision_ignore`` in the robot
                # config rather than disabling the whole term.
                self_collision_check=not _collision_disabled(request.model),
            )
            if interpolation_dt is not None:
                config.trajopt_solver_config.interpolation_dt = float(interpolation_dt)
            if maximum_dt is not None:
                config.trajopt_solver_config.maximum_trajectory_dt = float(maximum_dt)
            planner = MotionPlanner(config)
            planner.warmup(enable_graph=True, num_warmup_iterations=2)
            _PLANNERS[key] = planner
        scene = _scene(request, planner)
        if scene is not None:
            _purge_scene_mesh_cache(planner)
            planner.update_world(scene)
        return planner
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"cuRobo initialization failed: {exc}") from exc


def _ik_solver(request: BaseRequest) -> Any:
    robot = _robot_config(request)
    solver = _IK_SOLVERS.get(robot)
    try:
        if solver is None:
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

            config = InverseKinematicsCfg.create(
                robot=robot,
                scene_model=None,
                collision_cache={"mesh": 1},
                num_seeds=32,
                # Kept in step with the planner above. If IK screened
                # self-collision while planning did not, IK would reject seeds
                # the planner is willing to use and the two would disagree about
                # the same pose.
                self_collision_check=not _collision_disabled(request.model),
            )
            solver = InverseKinematics(config)
            _IK_SOLVERS[robot] = solver
        scene = _scene(request, solver)
        if scene is not None:
            # Same name-keyed mesh cache as the planner above, and _IK_SOLVERS
            # keeps one solver per robot for the life of the service: without
            # this purge a cached solver screens every later request against the
            # first scene it ever loaded -- another step of the episode, or
            # another task entirely -- and rejects poses the planner accepts.
            _purge_scene_mesh_cache(solver)
            solver.update_world(scene)
        return solver
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"cuRobo IK initialization failed: {exc}"
        ) from exc


def _validate(request: BaseRequest) -> tuple[str, ...]:
    arms = tuple(request.joint_positions)
    joint_dimension = JOINT_DIMENSIONS["yam_real" if request.model == "yam_real" else "robosuite"]
    if not arms or set(request.joint_names) != set(arms):
        raise HTTPException(status_code=400, detail="joint mappings must share non-empty arms")
    if set(request.end_effector_links) != set(arms) or set(request.base_transforms) != set(arms):
        raise HTTPException(status_code=400, detail="robot context arm mappings do not match")
    for arm in arms:
        joints = np.asarray(request.joint_positions[arm], dtype=np.float64)
        transform = np.asarray(request.base_transforms[arm], dtype=np.float64)
        joints_finite = bool(np.all(np.isfinite(joints)))
        if joints.shape != (joint_dimension,) or not joints_finite:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"invalid joints for {arm}",
                    "arm": arm,
                    "actual_shape": list(joints.shape),
                    "expected_shape": [joint_dimension],
                    "all_finite": joints_finite,
                },
            )
        if len(request.joint_names[arm]) != joint_dimension or transform.shape != (4, 4):
            raise HTTPException(status_code=400, detail=f"invalid robot context for {arm}")
    return arms


def _pose(target: Target) -> Any:
    if target.type != "pose" or target.position is None or target.quaternion_wxyz is None:
        raise HTTPException(status_code=400, detail="cuRobo V2 service requires a pose target")
    position = np.asarray(target.position, dtype=np.float32)
    quaternion = np.asarray(target.quaternion_wxyz, dtype=np.float32)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise HTTPException(status_code=400, detail="pose target has invalid shape")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise HTTPException(status_code=400, detail="pose target must be finite")
    from curobo.types import Pose
    import torch

    return Pose(
        position=torch.as_tensor(position, device="cuda").reshape(1, 3),
        quaternion=torch.as_tensor(quaternion, device="cuda").reshape(1, 4),
    )


#: Pull encoder readings slightly inside modeled hard limits before planning.
_JOINT_LIMIT_MARGIN_RAD = 1e-3


def _clip_joint_positions(positions: np.ndarray, planner: Any) -> np.ndarray:
    """Clamp a planner-ordered joint vector inside its limits by a small margin."""
    clipped = np.asarray(positions, dtype=np.float64).reshape(-1).copy()
    kinematics = getattr(planner, "kinematics", None)
    if kinematics is None or not hasattr(kinematics, "get_joint_limits"):
        return clipped
    try:
        limits = kinematics.get_joint_limits()
        lower = np.asarray(limits.position_lower_limits.detach().cpu(), dtype=np.float64).reshape(
            -1
        )
        upper = np.asarray(limits.position_upper_limits.detach().cpu(), dtype=np.float64).reshape(
            -1
        )
    except Exception as exc:
        _LOGGER.warning("joint-limit clip skipped: %s", exc)
        return clipped
    if lower.shape != clipped.shape or upper.shape != clipped.shape:
        return clipped
    half = np.maximum((upper - lower) * 0.5 - 1e-9, 0.0)
    margin = np.minimum(_JOINT_LIMIT_MARGIN_RAD, half)
    before = clipped.copy()
    clipped = np.minimum(np.maximum(clipped, lower + margin), upper - margin)
    delta = clipped - before
    if np.any(np.abs(delta) > 0.0):
        _LOGGER.info(
            "clipped joint_positions into limits (margin=%.1e rad); max|delta|=%.3e",
            _JOINT_LIMIT_MARGIN_RAD,
            float(np.max(np.abs(delta))),
        )
    return clipped


def _state(request: BaseRequest, planner: Any) -> Any:
    from curobo.types import JointState
    import torch

    ordered_arms = tuple(request.joint_positions)
    positions = np.concatenate([request.joint_positions[arm] for arm in ordered_arms])
    names = [name for arm in ordered_arms for name in request.joint_names[arm]]
    expected = list(planner.joint_names)
    if set(names) != set(expected):
        if len(ordered_arms) == 1 and len(expected) == 7:
            names = expected
        else:
            raise HTTPException(
                status_code=400,
                detail=f"request joint names do not match cuRobo model: {names} != {expected}",
            )
    reorder = [names.index(name) for name in expected]
    positions = _clip_joint_positions(positions[reorder], planner)
    return JointState.from_position(
        torch.as_tensor(positions, device="cuda", dtype=torch.float32).reshape(1, -1),
        joint_names=expected,
    )


def _requested_indices(
    request: BaseRequest,
    arm: str,
    available_names: list[str],
) -> list[int]:
    requested = request.joint_names[arm]
    if set(requested) <= set(available_names):
        return [available_names.index(name) for name in requested]
    canonical_panda = [f"panda_joint{index}" for index in range(1, 8)]
    if (
        request.model == "panda"
        and len(request.joint_positions) == 1
        and len(requested) == 7
        and set(canonical_panda) <= set(available_names)
    ):
        return [available_names.index(name) for name in canonical_panda]
    raise HTTPException(
        status_code=400,
        detail=f"joint names for {arm} do not match the cuRobo model",
    )


def _joint_goal(request: PlanRequest, planner: Any) -> Any:
    if request.target.type != "joints" or request.target.joint_positions is None:
        raise HTTPException(status_code=400, detail="joint target is required")
    target = np.asarray(request.target.joint_positions, dtype=np.float64)
    joint_dimension = JOINT_DIMENSIONS["yam_real" if request.model == "yam_real" else "robosuite"]
    if target.shape != (joint_dimension,) or not np.all(np.isfinite(target)):
        raise HTTPException(
            status_code=400,
            detail=f"joint target must contain {joint_dimension} finite values",
        )

    from curobo.types import JointState
    import torch

    current = _state(request, planner)
    positions = current.position.clone()
    expected = list(planner.joint_names)
    indices = torch.as_tensor(
        _requested_indices(request, request.arm, expected),
        device=positions.device,
        dtype=torch.long,
    )
    positions[:, indices] = torch.as_tensor(target, device=positions.device, dtype=positions.dtype)
    clipped = _clip_joint_positions(
        positions.detach().cpu().numpy().reshape(-1),
        planner,
    )
    return JointState.from_position(
        torch.as_tensor(clipped, device=positions.device, dtype=positions.dtype).reshape(1, -1),
        joint_names=expected,
    )


def _configuration_feasibility(
    planner: Any,
    current: Any,
    goal: Any,
) -> tuple[bool, bool] | None:
    """Check c-space endpoints with the planner's configured constraints."""
    graph_planner = getattr(planner, "graph_planner", None)
    if graph_planner is None:
        return None
    import torch

    samples = torch.cat(
        (
            current.position.reshape(1, -1),
            goal.position.reshape(1, -1),
        ),
        dim=0,
    )
    mask = graph_planner.check_samples_feasibility(samples)
    values = np.asarray(mask.detach().cpu(), dtype=bool).reshape(-1)
    if values.shape != (2,):
        return None
    return bool(values[0]), bool(values[1])


def _configuration_constraint_detail(planner: Any, state: Any) -> dict[str, object]:
    """Expose which feasibility term rejects one configuration.

    The planner reports only that a state is infeasible, which collapses joint
    bounds, self-collision and scene collision into one 422 and makes a
    real-arm rejection impossible to diagnose. This evaluates the graph
    planner's own feasibility rollout and names each constraint.

    Ported from the reference branch, which added it for the same reason.
    """
    graph_planner = getattr(planner, "graph_planner", None)
    if graph_planner is None:
        return {}
    try:
        samples = state.position.reshape(1, -1)
        graph_planner.check_samples_feasibility(samples)
        buffer = graph_planner._max_act_buffer
        buffer[:1, :, :] = samples.unsqueeze(1)
        metrics = graph_planner.feasibility_rollout.compute_metrics_from_action(buffer)
        values: dict[str, object] = {}
        costs = metrics.costs_and_constraints
        for kind, collection in (
            ("constraint", costs.constraints),
            ("hybrid", costs.hybrid_costs_constraints),
        ):
            for name, value in zip(collection.names, collection.values):
                array = np.asarray(value[0].detach().cpu(), dtype=np.float64)
                values[f"{kind}:{name}"] = {
                    "maximum": float(np.max(array)),
                    "sum": float(np.sum(array)),
                }
        total = costs.get_sum_constraint(sum_horizon=True, include_all_hybrid=False)
        if total is not None:
            values["sum_constraint"] = float(total.reshape(-1)[0].detach().cpu())
        return values
    except Exception as exc:
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def _ordered_goal_poses(
    request: BaseRequest,
    targets: Mapping[str, Target],
    planner: Any,
    current: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Complete partial bimanual pose goals with the inactive live tool pose."""
    poses = {request.end_effector_links[arm]: _pose(target) for arm, target in targets.items()}
    ordered = list(getattr(planner, "tool_frames", ()))
    missing = [name for name in ordered if name not in poses]
    if missing:
        current_poses = planner.compute_kinematics(current).tool_poses.to_dict()
        unavailable = [name for name in missing if name not in current_poses]
        if unavailable:
            raise HTTPException(
                status_code=500,
                detail=f"cuRobo kinematics omitted tool frames: {unavailable}",
            )
        poses.update({name: current_poses[name] for name in missing})
    if not ordered:
        ordered = list(poses)
    return poses, ordered


def _goal(
    request: BaseRequest,
    targets: Mapping[str, Target],
    planner: Any,
    current: Any,
) -> Any:
    from curobo.types import GoalToolPose

    poses, ordered = _ordered_goal_poses(request, targets, planner, current)
    return GoalToolPose.from_poses(poses, ordered_tool_frames=ordered, num_goalset=1)


def _tensor_summary(value: Any) -> dict[str, object] | None:
    """Return small JSON-safe diagnostics for a cuRobo result tensor."""
    if value is None:
        return None
    try:
        array = np.asarray(value.detach().cpu())
    except (AttributeError, TypeError, ValueError):
        return None
    if array.size == 0:
        return {"shape": list(array.shape), "count": 0}
    finite = array[np.isfinite(array)]
    summary: dict[str, object] = {
        "shape": list(array.shape),
        "count": int(array.size),
    }
    if finite.size:
        summary.update(
            {
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
            }
        )
    if array.dtype == np.bool_:
        summary["true_count"] = int(np.count_nonzero(array))
    return summary


def _constraint_scores(constraints: Mapping[str, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name in ("cspace", "self_collision", "scene_collision"):
        entry = constraints.get(f"constraint:{name}")
        value = 0.0
        if isinstance(entry, Mapping):
            raw = entry.get("maximum", entry.get("sum", 0.0))
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
        scores[name] = value
    return scores


def _dominant_constraint(constraints: Mapping[str, object]) -> str | None:
    scores = _constraint_scores(constraints)
    dominant = max(scores, key=scores.get)
    return dominant if scores[dominant] > 0.0 else None


def _sample_feasible(planner: Any, state: Any) -> bool | None:
    graph_planner = getattr(planner, "graph_planner", None)
    if graph_planner is None:
        return None
    mask = graph_planner.check_samples_feasibility(state.position.reshape(1, -1))
    values = np.asarray(mask.detach().cpu(), dtype=bool).reshape(-1)
    return bool(values[0]) if values.size else None


def _planning_failure_detail(
    result: Any,
    *,
    planner: Any | None = None,
    current: Any | None = None,
    goal_state: Any | None = None,
) -> dict[str, object]:
    """Build a 422 detail that identifies infeasible endpoints when possible."""
    if result is None:
        detail: dict[str, object] = {"message": "cuRobo returned no planning result"}
    else:
        detail = {"message": "cuRobo could not find a feasible plan"}
        for name in (
            "success",
            "feasible",
            "cspace_error",
            "position_error",
            "rotation_error",
        ):
            summary = _tensor_summary(getattr(result, name, None))
            if summary is not None:
                detail[name] = summary

    constraints_start: dict[str, object] = {}
    constraints_goal: dict[str, object] = {}
    start_feasible: bool | None = None
    goal_feasible: bool | None = None
    if planner is not None and current is not None:
        constraints_start = _configuration_constraint_detail(planner, current)
        start_feasible = _sample_feasible(planner, current)
        detail["constraints_start"] = constraints_start
        detail["start_state_feasible"] = start_feasible
    if planner is not None and goal_state is not None:
        constraints_goal = _configuration_constraint_detail(planner, goal_state)
        goal_feasible = _sample_feasible(planner, goal_state)
        detail["constraints_goal"] = constraints_goal
        detail["goal_state_feasible"] = goal_feasible

    if start_feasible is False:
        kind = _dominant_constraint(constraints_start) or "unknown"
        detail["failure_kind"] = f"start_{kind}"
        detail["message"] = f"cuRobo start state infeasible ({kind})"
    elif goal_feasible is False:
        kind = _dominant_constraint(constraints_goal) or "unknown"
        detail["failure_kind"] = f"goal_{kind}"
        detail["message"] = f"cuRobo goal state infeasible ({kind})"
    elif result is not None:
        kind = _dominant_constraint(constraints_start)
        detail["failure_kind"] = kind or "plan_failed"
        if kind is not None:
            detail["message"] = f"cuRobo could not find a feasible plan ({kind})"
    else:
        kind = _dominant_constraint(constraints_start) or _dominant_constraint(constraints_goal)
        detail["failure_kind"] = kind or "no_result"
    return detail


def _trajectory(
    result: Any,
    *,
    planner: Any | None = None,
    current: Any | None = None,
    goal_state: Any | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    if result is None or result.success is None or not bool(result.success.any()):
        raise HTTPException(
            status_code=422,
            detail=_planning_failure_detail(
                result, planner=planner, current=current, goal_state=goal_state
            ),
        )
    plan = result.get_interpolated_plan()
    positions = np.asarray(plan.position.detach().cpu(), dtype=np.float64).reshape(
        -1, plan.position.shape[-1]
    )
    names = list(plan.joint_names)
    plan_dt = getattr(plan, "dt", None)
    dt = float(plan_dt.reshape(-1)[0].item()) if plan_dt is not None else 0.025
    return positions, dt, names


def _solve_ik(request: IkRequest) -> dict[str, object]:
    _validate(request)
    if request.arm not in request.joint_positions:
        raise HTTPException(status_code=400, detail="unknown arm")
    solver = _ik_solver(request)
    current = _state(request, solver)
    # A single-arm goal on a model that declares two tool frames is refused by
    # cuRobo ("Ordered link names [...] not a subset of [...]"): every declared
    # frame must appear in the goal. The planning path already completes partial
    # goals with the inactive arm's live pose; IK has to do the same, or one-arm
    # IK is unsolvable on the bimanual station. Single-tool-frame models are
    # unaffected, which is why this only surfaces on a bimanual model.
    goal = _goal(request, {request.arm: request.target}, solver, current)
    result = solver.solve_pose(goal, current_state=current)
    if result.success is None or not bool(result.success.any()) or result.js_solution is None:
        raise HTTPException(status_code=422, detail="cuRobo IK target is unreachable")
    solution = np.asarray(result.js_solution.position.detach().cpu()).reshape(-1)
    solution_names = list(result.js_solution.joint_names)
    joints = solution[_requested_indices(request, request.arm, solution_names)]
    return {"joint_positions": joints.tolist()}


def _plan(request: PlanRequest) -> PlanResponse:
    _validate(request)
    if request.arm not in request.joint_positions:
        raise HTTPException(status_code=400, detail="unknown arm")
    planner = _planner(request)
    current = _state(request, planner)
    goal_state: Any | None = None
    if request.target.type == "pose":
        result = planner.plan_pose(
            _goal(request, {request.arm: request.target}, planner, current),
            current,
        )
    else:
        joint_goal = _joint_goal(request, planner)
        goal_state = joint_goal
        feasibility = _configuration_feasibility(planner, current, joint_goal)
        if feasibility is not None and not all(feasibility):
            detail = _planning_failure_detail(
                None, planner=planner, current=current, goal_state=joint_goal
            )
            detail["start_state_feasible"] = feasibility[0]
            detail["goal_state_feasible"] = feasibility[1]
            raise HTTPException(status_code=422, detail=detail)
        result = planner.plan_cspace(joint_goal, current)
    positions, dt, names = _trajectory(
        result, planner=planner, current=current, goal_state=goal_state
    )
    indices = _requested_indices(request, request.arm, names)
    return PlanResponse(joint_positions_base64=encode_numpy(positions[:, indices]), dt_s=dt)


def _plan_synchronized(request: SynchronizedPlanRequest) -> SynchronizedPlanResponse:
    arms = _validate(request)
    if set(request.targets) != set(arms):
        raise HTTPException(status_code=400, detail="targets must name every arm")
    planner = _planner(request)
    current = _state(request, planner)
    result = planner.plan_pose(
        _goal(request, request.targets, planner, current),
        current,
    )
    positions, dt, names = _trajectory(result, planner=planner, current=current)
    output: dict[str, str] = {}
    for arm in arms:
        requested = request.joint_names[arm]
        if not set(requested) <= set(names):
            raise HTTPException(status_code=500, detail=f"planned trajectory omitted {arm}")
        output[arm] = encode_numpy(positions[:, [names.index(name) for name in requested]])
    return SynchronizedPlanResponse(joint_positions_base64=output, dt_s=dt)


async def _worker(function: Any, request: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(function, request))


async def _run_request(function: Any, request: Any) -> Any:
    try:
        return await _worker(function, request)
    except HTTPException as exc:
        _LOGGER.warning("cuRobo request rejected (%s): %s", exc.status_code, exc.detail)
        raise


@app.post(IK_PATH)
async def solve_ik(request: IkRequest) -> dict[str, object]:
    return await _run_request(_solve_ik, request)


@app.post(PLAN_PATH, response_model=PlanResponse)
async def plan(request: PlanRequest) -> PlanResponse:
    return await _run_request(_plan, request)


@app.post(PLAN_SYNCHRONIZED_PATH, response_model=SynchronizedPlanResponse)
async def plan_synchronized(request: SynchronizedPlanRequest) -> SynchronizedPlanResponse:
    return await _run_request(_plan_synchronized, request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cuRobo V2 provider")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8118)
    args = parser.parse_args()
    # uvicorn configures its own loggers and leaves the root at WARNING, which
    # silences this module. The carve counts are how you tell a scene that still
    # contains the robot from one that does not, so they default to visible.
    logging.basicConfig(
        level=os.environ.get("CAP_HARNESS_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
