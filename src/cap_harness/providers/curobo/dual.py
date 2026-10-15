"""Generate a dual-Panda cuRobo model from the pinned single-Panda assets."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

_SECONDARY_SUFFIX = "_2"


def _rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Convert a proper rotation matrix to fixed-axis XYZ roll/pitch/yaw."""
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    if abs(abs(pitch) - np.pi / 2.0) < 1e-7:
        roll = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        yaw = 0.0
    else:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def _suffix_tree(element: ET.Element, suffix: str) -> ET.Element:
    result = copy.deepcopy(element)
    for item in result.iter():
        if item.tag in {"link", "joint"} and "name" in item.attrib:
            item.attrib["name"] += suffix
        if item.tag in {"parent", "child"} and "link" in item.attrib:
            item.attrib["link"] += suffix
    return result


def _suffix_name(name: str, suffix: str = _SECONDARY_SUFFIX) -> str:
    return f"{name}{suffix}"


def _duplicate_mapping(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    for key, item in value.items():
        new_key = _suffix_name(key)
        if isinstance(item, list) and item and all(isinstance(entry, str) for entry in item):
            result[new_key] = [_suffix_name(entry) for entry in item]
        else:
            result[new_key] = copy.deepcopy(item)
    return result


def build_dual_panda_config(
    curobo_root: Path,
    secondary_base_transform: np.ndarray,
    output_root: Path,
) -> Path:
    """Build and cache a 14-DoF model for one calibrated secondary base pose."""
    transform = np.asarray(secondary_base_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("secondary_base_transform must be a finite 4x4 matrix")
    digest = hashlib.sha256(transform.tobytes()).hexdigest()[:16]
    directory = output_root / digest
    config_path = directory / "dual_panda.yml"
    if config_path.is_file():
        return config_path
    directory.mkdir(parents=True, exist_ok=True)

    content = curobo_root / "curobo/content"
    description = content / "assets/robot/franka_description"
    source_urdf = description / "franka_panda.urdf"
    source_config = content / "configs/robot/franka.yml"

    source_root = ET.parse(source_urdf).getroot()
    robot = ET.Element("robot", {"name": "dual_panda"})
    ET.SubElement(robot, "link", {"name": "world_base_link"})
    for child in source_root:
        robot.append(copy.deepcopy(child))
    for child in source_root:
        robot.append(_suffix_tree(child, _SECONDARY_SUFFIX))

    def fixed_joint(name: str, child: str, matrix: np.ndarray) -> None:
        joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
        roll, pitch, yaw = _rpy(matrix[:3, :3])
        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": " ".join(f"{value:.12g}" for value in matrix[:3, 3]),
                "rpy": f"{roll:.12g} {pitch:.12g} {yaw:.12g}",
            },
        )
        ET.SubElement(joint, "parent", {"link": "world_base_link"})
        ET.SubElement(joint, "child", {"link": child})

    fixed_joint("world_to_primary", "base_link", np.eye(4))
    fixed_joint("world_to_secondary", _suffix_name("base_link"), transform)
    urdf_path = directory / "dual_panda.urdf"
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)

    data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    source = data["robot_cfg"]["kinematics"]
    config = copy.deepcopy(source)
    config["asset_root_path"] = str(description)
    config["urdf_path"] = str(urdf_path)
    config["base_link"] = "world_base_link"
    for key in (
        "collision_link_names",
        "mesh_link_names",
        "tool_frames",
        "grasp_contact_link_names",
    ):
        values = [value for value in source.get(key, []) if value != "attached_object"]
        config[key] = values + [_suffix_name(value) for value in values]
    for key in ("collision_spheres", "self_collision_buffer"):
        values = {
            name: value
            for name, value in dict(source.get(key, {})).items()
            if name != "attached_object"
        }
        config[key] = _duplicate_mapping(values)
    ignore = {
        name: [entry for entry in values if entry != "attached_object"]
        for name, values in dict(source.get("self_collision_ignore", {})).items()
        if name != "attached_object"
    }
    config["self_collision_ignore"] = _duplicate_mapping(ignore)
    config["lock_joints"] = {
        **dict(source.get("lock_joints", {})),
        **{
            _suffix_name(name): value for name, value in dict(source.get("lock_joints", {})).items()
        },
    }
    config.pop("extra_links", None)
    config.pop("extra_collision_spheres", None)
    cspace = copy.deepcopy(source["cspace"])
    for key in ("joint_names",):
        values = list(cspace[key])
        cspace[key] = values + [_suffix_name(value) for value in values]
    for key in ("default_joint_position", "null_space_weight", "cspace_distance_weight"):
        cspace[key] = list(cspace[key]) * 2
    config["cspace"] = cspace
    config_path.write_text(
        yaml.safe_dump({"robot_cfg": {"kinematics": config}}, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


__all__ = ["build_dual_panda_config"]
