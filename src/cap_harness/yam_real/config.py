"""Explicit, immutable YAM station profile and calibration-bundle loading.

One station is one YAML file plus one immutable calibration bundle, so adding a
bench is a data change rather than a code change. Everything the harness needs
to reach a physical station lives here: arm endpoints, joint limits, home pose,
level-1 gains, the camera role and serial, and the provider endpoints.

The bundle is content-addressed. ``manifest_sha256`` and ``calibration_sha256``
are verified on load, which is what makes a recorded episode attributable: an
episode names the bundle it ran under, and the bundle cannot be edited without
the digest changing. Calibration drifts as cameras are bumped and re-mounted, so
"which calibration produced this data" is not a bookkeeping question.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np
import yaml

#: The physical arms every YAM station has. Distinct from the shared API's
#: logical roles ("primary"/"secondary"), which a profile maps onto these.
ARM_NAMES = ("left", "right")

_STATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "yam_real"


def _readonly_array(value: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if result.shape != shape or not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(result, copy=True, order="C")
    result.setflags(write=False)
    return result


def _positive(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _port(value: object, name: str) -> int:
    if isinstance(value, bool):
        # Bad data, not a caller type error: a bad profile value is bad data, not a caller type error.
        raise ValueError(f"{name} must be a TCP port")
    result = int(value)
    if not 1 <= result <= 65535:
        raise ValueError(f"{name} must be a TCP port")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return result


def _homogeneous(value: object, name: str) -> np.ndarray:
    result = _readonly_array(value, name, (4, 4))
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must be homogeneous")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be proper")
    return result


@dataclass(frozen=True, slots=True)
class YamArmConfig:
    host: str
    port: int
    joint_names: tuple[str, ...]
    base_transform: np.ndarray
    home_joints: np.ndarray
    """Mechanical zero: the calibration reference, and what ``go_home`` means."""

    ready_joints: np.ndarray
    """Folded task-start pose ``reset`` drives to.

    Distinct from home on purpose: home is the mechanical zero, fully extended
    with joints 2 and 3 on their 0.0 bound. Both are plannable -- ready is
    preferred because it is folded and clear of the table, not because home is
    unreachable.
    """

    can_interface: str


@dataclass(frozen=True, slots=True)
class YamMotorConfig:
    """The motor bus an arm server drives. Not used by harness clients."""

    bustype: str
    arm_motor_ids: tuple[int, ...]
    arm_motor_types: tuple[str, ...]
    gripper_motor_id: int
    gripper_motor_type: str
    gripper_sign: int
    gripper_velocity_limit: float
    gripper_torque_limit_nm: float
    gravity_comp_factor: float


@dataclass(frozen=True, slots=True)
class YamControllerConfig:
    control_frequency_hz: float
    """Rate a caller is expected to issue discrete steps at (level 0)."""

    command_stream_hz: float
    """Rate the level-1 controller streams interpolated targets at.

    Distinct from both neighbours in the cascade: a caller may think in 30 Hz
    steps, this layer resamples to a smoother stream, and the arm server re-sends
    the latest target at its own hold rate. Collapsing any two of the three would
    silently retime trajectories.
    """

    follower_hold_frequency_hz: float
    """Rate the arm server re-sends its buffered target at (level 2)."""
    rpc_timeout_s: float
    close_timeout_s: float
    final_joint_tolerance_rad: float
    max_tracking_error_rad: float
    max_joint_velocity_rad_s: float
    kp: np.ndarray
    kd: np.ndarray
    gripper_velocity_limit: float
    gripper_torque_limit_nm: float
    gripper_full_travel_s: float


@dataclass(frozen=True, slots=True)
class YamCameraConfig:
    """The station's camera, opened directly by serial over USB.

    There is no camera service and no host/port: ``open_station_camera`` hands
    ``serial`` and ``resolution`` to ``RealSenseCamera``, which selects the
    device from ``pyrealsense2``'s enumeration. An earlier design reached a
    camera-owner Portal over TCP; the fields that addressed it are gone rather
    than left to imply a listener has to be running.
    """

    role: str
    serial: str
    resolution: tuple[int, int]
    max_acquisition_elapsed_s: float
    intrinsics: np.ndarray
    base_from_camera: np.ndarray


@dataclass(frozen=True, slots=True)
class YamAuxCameraConfig:
    """A camera opened for its images alone.

    It has no ``intrinsics`` and no ``base_from_camera``, and that absence is the
    point: these cameras are not in the calibration bundle, so there is no honest
    transform to publish. Filling the fields with placeholders would let a caller
    project pixels through them and get silently wrong metres.

    Missing the fields instead makes the misuse a ``AttributeError`` at the seam
    rather than a plausible number downstream. Use these to answer "is the object
    in view"; use :class:`YamCameraConfig` for anything spatial.

    ``open_station_camera`` reads only ``serial`` and ``resolution``, so this is
    everything a camera needs to be opened.
    """

    role: str
    serial: str
    resolution: tuple[int, int]
    max_acquisition_elapsed_s: float


@dataclass(frozen=True, slots=True)
class YamCalibrationBundle:
    bundle_id: str
    root: Path
    manifest_path: Path
    calibration_path: Path
    manifest_sha256: str
    calibration_sha256: str


@dataclass(frozen=True, slots=True)
class YamStationConfig:
    station: str
    base_frame: str
    default_arm: str
    arm_aliases: Mapping[str, str]
    arms: Mapping[str, YamArmConfig]
    joint_limits_lower: np.ndarray
    joint_limits_upper: np.ndarray
    planning_workspace_lower: np.ndarray
    planning_workspace_upper: np.ndarray
    controller: YamControllerConfig
    motors: YamMotorConfig
    camera: YamCameraConfig
    #: Cameras opened for their images only, keyed by role. They carry no
    #: intrinsics and no ``base_from_camera``, and deliberately are NOT checked
    #: against the calibration bundle -- the bundle solves for the top camera,
    #: and requiring these to appear in it would mean either faking entries or
    #: not having the cameras at all.
    #:
    #: The consequence is a hard boundary: anything that turns pixels into metres
    #: -- ``mask_to_point_cloud`` and everything downstream of it -- can only use
    #: ``camera``. These answer "is the object in view", not "where is it".
    aux_cameras: Mapping[str, YamAuxCameraConfig]
    calibration: YamCalibrationBundle
    model_xml: Path
    arm_model_xml: Path
    config_sha256: str
    source_path: Path

    @property
    def ready_joint_positions(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: arm.ready_joints for name, arm in self.arms.items()})

    @property
    def home_joint_positions(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType({name: arm.home_joints for name, arm in self.arms.items()})


def station_config_path(station: str, *, config_root: str | Path | None = None) -> Path:
    """Resolve an explicitly named station profile without hostname inference.

    One bench is one directory, holding everything that describes it: the
    profile, its calibration bundle, and its measured gripper travel. The
    directory is named for humans -- ``my-yam-station`` -- while the
    identifier callers pass is the ``station:`` field inside its
    ``station.yaml``. Those are allowed to differ so a directory can say where a
    bench physically is without changing every command that names it.

    Resolution therefore reads the profiles rather than guessing a filename. The
    flat ``<root>/<station>.yaml`` form is still accepted, so an existing
    checkout keeps working.
    """
    if not isinstance(station, str) or _STATION_PATTERN.fullmatch(station) is None:
        raise ValueError("station must be an explicit station identifier")
    root = _DEFAULT_CONFIG_ROOT if config_root is None else Path(config_root).expanduser().resolve()
    resolved_root = root.resolve()

    candidates = [root / f"{station}.yaml"]
    if root.is_dir():
        candidates.extend(sorted(root.glob("*/station.yaml")))

    matches = []
    for candidate in candidates:
        path = candidate.resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("station profile must remain within config_root") from exc
        if not path.is_file():
            continue
        if path.name == f"{station}.yaml":
            return path
        try:
            declared = yaml.safe_load(path.read_bytes())
        except yaml.YAMLError:
            continue
        if isinstance(declared, dict) and declared.get("station") == station:
            matches.append(path)

    if len(matches) > 1:
        names = ", ".join(str(match.parent.name) for match in matches)
        raise ValueError(f"station {station!r} is declared by more than one profile: {names}")
    if matches:
        return matches[0]
    # Return the flat path so the caller's FileNotFoundError names something
    # concrete rather than a glob.
    return (root / f"{station}.yaml").resolve()


def load_yam_station_config(
    station: str,
    *,
    config_root: str | Path | None = None,
) -> YamStationConfig:
    """Load and verify one explicit station profile and its immutable bundle."""
    path = station_config_path(station, config_root=config_root)
    try:
        raw_config = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"unknown YAM station {station!r}") from exc
    data = yaml.safe_load(raw_config)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("YAM station profile must use schema_version 1")
    if data.get("station") != station:
        raise ValueError("station profile identity does not match the requested station")
    # Check the invariant, not a literal. The previous form demanded exactly
    # {"primary": "right"}, which pinned one bench's rigging into the loader and
    # would have rejected a station that faces its work with the left arm.
    default_arm = data.get("default_arm")
    aliases = data.get("arm_aliases")
    if default_arm not in ARM_NAMES:
        raise ValueError(f"default_arm must be one of {ARM_NAMES}, got {default_arm!r}")
    if not isinstance(aliases, dict) or set(aliases) - {"primary", "secondary"}:
        raise ValueError("arm_aliases may only map 'primary' and 'secondary'")
    if set(aliases.values()) - set(ARM_NAMES):
        raise ValueError(f"arm_aliases must point at real arms {ARM_NAMES}")
    if aliases.get("primary") != default_arm:
        raise ValueError("arm_aliases['primary'] must be the default_arm")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("arm_aliases must not point two roles at the same arm")

    calibration_spec = data.get("calibration")
    if not isinstance(calibration_spec, dict):
        # Bad data, not a caller type error: a bad profile value is bad data, not a caller type error.
        raise ValueError("calibration must be a mapping")
    bundle_root = (path.parent / str(calibration_spec.get("bundle", ""))).resolve()
    try:
        bundle_root.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise ValueError("calibration bundle must remain within the station config tree") from exc
    manifest_path = bundle_root / "manifest.json"
    calibration_path = bundle_root / "calibration.json"
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != _digest(calibration_spec.get("manifest_sha256"), "manifest_sha256"):
        raise ValueError("calibration manifest SHA-256 does not match station profile")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("station") != station:
        raise ValueError("calibration manifest identity does not match station profile")
    calibration_sha = _sha256_file(calibration_path)
    if calibration_sha != _digest(manifest.get("calibration_sha256"), "calibration_sha256"):
        raise ValueError("calibration artifact SHA-256 does not match manifest")
    if calibration.get("schema_version") != 1 or not calibration.get("complete"):
        raise ValueError("calibration artifact must be complete schema_version 1")

    raw_arms = data.get("arms")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(ARM_NAMES):
        raise ValueError("YAM station must define exactly left and right arms")
    arms: dict[str, YamArmConfig] = {}
    for name in ARM_NAMES:
        raw = raw_arms[name]
        names = tuple(str(item) for item in raw.get("joint_names", ()))
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError(f"arms.{name}.joint_names must contain six unique names")
        arms[name] = YamArmConfig(
            host=str(raw["host"]),
            port=_port(raw["port"], f"arms.{name}.port"),
            joint_names=names,
            base_transform=_homogeneous(raw["base_transform"], f"arms.{name}.base_transform"),
            home_joints=_readonly_array(raw["home_joints"], f"arms.{name}.home_joints", (6,)),
            ready_joints=_readonly_array(raw["ready_joints"], f"arms.{name}.ready_joints", (6,)),
            can_interface=str(raw["can_interface"]),
        )

    raw_limits = data["joint_limits"]
    lower = _readonly_array(raw_limits["lower"], "joint_limits.lower", (6,))
    upper = _readonly_array(raw_limits["upper"], "joint_limits.upper", (6,))
    if np.any(lower >= upper):
        raise ValueError("joint limit lower bounds must be below upper bounds")
    for name, arm in arms.items():
        if np.any(arm.home_joints < lower) or np.any(arm.home_joints > upper):
            raise ValueError(f"arms.{name}.home_joints must lie within joint limits")
        if np.any(arm.ready_joints < lower) or np.any(arm.ready_joints > upper):
            raise ValueError(f"arms.{name}.ready_joints must lie within joint limits")

    raw_workspace = data["planning_workspace"]
    workspace_lower = _readonly_array(raw_workspace["lower"], "planning_workspace.lower", (3,))
    workspace_upper = _readonly_array(raw_workspace["upper"], "planning_workspace.upper", (3,))
    if np.any(workspace_lower >= workspace_upper):
        raise ValueError("planning workspace lower bounds must be below upper bounds")
    # The box has to contain the arms themselves, or the crop would discard the
    # robot's own surroundings and the planner would be reasoning about a volume
    # the robot is not in.
    for name, arm in arms.items():
        mount = np.asarray(arm.base_transform, dtype=np.float64)[:3, 3]
        if np.any(mount < workspace_lower) or np.any(mount > workspace_upper):
            raise ValueError(f"planning workspace does not contain the {name} arm mount")

    gripper = data.get("gripper")
    if not isinstance(gripper, dict) or gripper.get("closed") != 0.0 or gripper.get("open") != 1.0:
        raise ValueError("gripper must define normalized closed=0.0 and open=1.0 conventions")

    raw_controller = data["controller"]
    controller = YamControllerConfig(
        control_frequency_hz=_positive(raw_controller["control_frequency_hz"], "control frequency"),
        command_stream_hz=_positive(raw_controller["command_stream_hz"], "command stream rate"),
        follower_hold_frequency_hz=_positive(
            raw_controller["follower_hold_frequency_hz"], "follower hold frequency"
        ),
        rpc_timeout_s=_positive(raw_controller["rpc_timeout_s"], "RPC timeout"),
        close_timeout_s=_positive(raw_controller["close_timeout_s"], "close timeout"),
        final_joint_tolerance_rad=_positive(
            raw_controller["final_joint_tolerance_rad"], "final joint tolerance"
        ),
        max_tracking_error_rad=_positive(
            raw_controller["max_tracking_error_rad"], "maximum tracking error"
        ),
        max_joint_velocity_rad_s=_positive(
            raw_controller["max_joint_velocity_rad_s"], "maximum joint velocity"
        ),
        kp=_readonly_array(raw_controller["kp"], "controller.kp", (7,)),
        kd=_readonly_array(raw_controller["kd"], "controller.kd", (7,)),
        gripper_velocity_limit=_positive(
            raw_controller["gripper_velocity_limit"], "gripper velocity limit"
        ),
        gripper_torque_limit_nm=_positive(
            raw_controller["gripper_torque_limit_nm"], "gripper torque limit"
        ),
        gripper_full_travel_s=_positive(
            raw_controller["gripper_full_travel_s"], "gripper full travel time"
        ),
    )

    raw_motors = data["motors"]
    arm_ids = tuple(int(value) for value in raw_motors["arm_motor_ids"])
    arm_types = tuple(str(value) for value in raw_motors["arm_motor_types"])
    if len(arm_ids) != 6 or len(arm_types) != 6:
        raise ValueError("motors must describe exactly six arm joints")
    if int(raw_motors["gripper_sign"]) not in (-1, 1):
        raise ValueError("motors.gripper_sign must be -1 or 1")
    motors = YamMotorConfig(
        bustype=str(raw_motors["bustype"]),
        arm_motor_ids=arm_ids,
        arm_motor_types=arm_types,
        gripper_motor_id=int(raw_motors["gripper_motor_id"]),
        gripper_motor_type=str(raw_motors["gripper_motor_type"]),
        gripper_sign=int(raw_motors["gripper_sign"]),
        gripper_velocity_limit=_positive(
            raw_motors["gripper_velocity_limit"], "motors.gripper_velocity_limit"
        ),
        gripper_torque_limit_nm=_positive(
            raw_motors["gripper_torque_limit_nm"], "motors.gripper_torque_limit_nm"
        ),
        gravity_comp_factor=_positive(
            raw_motors["gravity_comp_factor"], "motors.gravity_comp_factor"
        ),
    )

    raw_camera = data["camera"]
    role = str(raw_camera["role"])
    top_manifest = manifest.get("cameras", {}).get(role)
    top_calibration = calibration.get("cameras", {}).get(role)
    if not isinstance(top_manifest, dict) or not isinstance(top_calibration, dict):
        # Bad data, not a caller type error: a bad bundle is bad data, not a caller type error.
        raise ValueError("calibration bundle does not contain the configured top camera")
    serial = str(raw_camera["serial"])
    resolution = tuple(int(item) for item in raw_camera["resolution"])
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("camera.resolution must be [width, height]")
    if serial != str(top_manifest.get("serial")) or serial != str(
        top_calibration.get("camera_serial")
    ):
        raise ValueError("top-camera serial does not match immutable calibration bundle")
    if list(resolution) != top_manifest.get("resolution"):
        raise ValueError("top-camera resolution does not match immutable calibration bundle")
    if top_calibration.get("parent_frame") != data.get("base_frame"):
        raise ValueError("top-camera calibration parent does not match base_frame")
    camera = YamCameraConfig(
        role=role,
        serial=serial,
        resolution=(resolution[0], resolution[1]),
        max_acquisition_elapsed_s=_positive(
            raw_camera["max_acquisition_elapsed_s"], "camera acquisition limit"
        ),
        intrinsics=_readonly_array(top_manifest.get("intrinsics"), "top intrinsics", (3, 3)),
        base_from_camera=_homogeneous(
            top_calibration.get("T_base_from_camera"), "top T_base_from_camera"
        ),
    )

    # Image-only cameras. Optional, so a bench that has not wired them still
    # loads. Their role must not collide with the calibrated camera's, because
    # role is the name programs pass to segment_text and render_rgb.
    aux_cameras: dict[str, YamAuxCameraConfig] = {}
    for raw_aux in data.get("aux_cameras", []) or []:
        aux_role = str(raw_aux["role"])
        if aux_role == role:
            raise ValueError(f"aux camera role {aux_role!r} collides with the calibrated camera")
        if aux_role in aux_cameras:
            raise ValueError(f"duplicate aux camera role {aux_role!r}")
        aux_resolution = tuple(int(item) for item in raw_aux["resolution"])
        if len(aux_resolution) != 2:
            raise ValueError(f"aux camera {aux_role} resolution must be [width, height]")
        aux_cameras[aux_role] = YamAuxCameraConfig(
            role=aux_role,
            serial=str(raw_aux["serial"]),
            resolution=(aux_resolution[0], aux_resolution[1]),
            max_acquisition_elapsed_s=_positive(
                raw_aux["max_acquisition_elapsed_s"], f"{aux_role} acquisition limit"
            ),
        )

    # The station names its own MuJoCo model. Resolving it here rather than
    # through an environment-variable lookup keeps one answer to "which model
    # describes this bench", and makes the model swap with the profile when a
    # second station arrives.
    raw_model = data.get("model")
    if not isinstance(raw_model, dict) or not str(raw_model.get("xml", "")).strip():
        raise ValueError("model.xml must name the station's MuJoCo model")
    package_root = Path(__file__).resolve().parents[1]
    model_xml = (package_root / str(raw_model["xml"])).resolve()
    try:
        model_xml.relative_to(package_root)
    except ValueError as exc:
        raise ValueError("model.xml must remain within the cap_harness package") from exc
    if not model_xml.is_file():
        raise ValueError(f"station model XML not found: {model_xml}")
    arm_model_xml = (package_root / str(raw_model["arm_xml"])).resolve()
    try:
        arm_model_xml.relative_to(package_root)
    except ValueError as exc:
        raise ValueError("model.arm_xml must remain within the cap_harness package") from exc
    if not arm_model_xml.is_file():
        raise ValueError(f"arm model XML not found: {arm_model_xml}")

    return YamStationConfig(
        station=station,
        base_frame=str(data["base_frame"]),
        # Read from the profile, not restated. These were hardcoded here while
        # the profile's own fields were validated above and then discarded --
        # so a bench that declared a different mapping was silently overridden
        # with this one, and the loader's validation guarded a value nothing
        # ever used.
        default_arm=str(default_arm),
        arm_aliases=MappingProxyType(dict(aliases)),
        arms=MappingProxyType(arms),
        joint_limits_lower=lower,
        joint_limits_upper=upper,
        planning_workspace_lower=workspace_lower,
        planning_workspace_upper=workspace_upper,
        controller=controller,
        motors=motors,
        camera=camera,
        aux_cameras=MappingProxyType(aux_cameras),
        calibration=YamCalibrationBundle(
            bundle_id=str(manifest["bundle_id"]),
            root=bundle_root,
            manifest_path=manifest_path,
            calibration_path=calibration_path,
            manifest_sha256=manifest_sha,
            calibration_sha256=calibration_sha,
        ),
        model_xml=model_xml,
        arm_model_xml=arm_model_xml,
        config_sha256=hashlib.sha256(raw_config).hexdigest(),
        source_path=path,
    )


__all__ = [
    "YamArmConfig",
    "YamAuxCameraConfig",
    "YamCalibrationBundle",
    "YamCameraConfig",
    "YamControllerConfig",
    "YamMotorConfig",
    "YamStationConfig",
    "load_yam_station_config",
    "station_config_path",
]
