from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URDF_IN = ROOT / "station.urdf"
XML_IN = ROOT / "station.xml"
URDF_OUT = ROOT / "station_physics.urdf"


def parse_floats(text: str | None, default: list[float]) -> list[float]:
    if text is None:
        return default[:]
    return [float(x) for x in text.split()]


def quat_normalize(q: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in q))
    if n == 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / n for v in q]


def quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_to_matrix(q: list[float]) -> list[list[float]]:
    w, x, y, z = quat_normalize(q)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def rotate_vec(q: list[float], v: list[float]) -> list[float]:
    m = quat_to_matrix(q)
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def quat_to_rpy(q: list[float]) -> list[float]:
    w, x, y, z = quat_normalize(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def compose_tf(
    parent_xyz: list[float],
    parent_quat: list[float],
    child_xyz: list[float],
    child_quat: list[float],
) -> tuple[list[float], list[float]]:
    xyz = [a + b for a, b in zip(parent_xyz, rotate_vec(parent_quat, child_xyz))]
    quat = quat_multiply(parent_quat, child_quat)
    return xyz, quat_normalize(quat)


def fmt(vals: list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in vals)


def build_default_geom_attrs(xml_root: ET.Element) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    default_root = xml_root.find("default")
    if default_root is None:
        return out

    def walk(default_elem: ET.Element, inherited: dict[str, str]) -> None:
        current = inherited.copy()
        geom_elem = default_elem.find("geom")
        if geom_elem is not None:
            current.update(geom_elem.attrib)
        class_name = default_elem.get("class")
        if class_name is not None:
            out[class_name] = current.copy()
        for child in default_elem.findall("default"):
            walk(child, current)

    walk(default_root, {})
    return out


def resolved_geom_attr(
    geom: ET.Element, geom_defaults: dict[str, dict[str, str]], key: str
) -> str | None:
    if geom.get(key) is not None:
        return geom.get(key)
    geom_class = geom.get("class")
    if geom_class is None:
        return None
    return geom_defaults.get(geom_class, {}).get(key)


def resolve_geom_type(geom: ET.Element, geom_defaults: dict[str, dict[str, str]]) -> str | None:
    geom_type = resolved_geom_attr(geom, geom_defaults, "type")
    if geom_type is not None:
        return geom_type
    geom_class = geom.get("class")
    if geom_class == "collision":
        return "capsule"
    if geom_class == "sphere_collision":
        return "sphere"
    if geom_class == "grip_pad":
        return "box"
    if geom_class == "camera_collision":
        return "mesh"
    return None


def is_visual_only_geom(geom: ET.Element) -> bool:
    geom_class = geom.get("class")
    if geom_class in {"visual", "camera_visual"}:
        return True
    if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
        return True
    if geom.get("group") == "2" and geom.get("density") == "0":
        return True
    return False


def build_geom_element(
    geom: ET.Element,
    mesh_files: dict[str, str],
    geom_defaults: dict[str, dict[str, str]],
) -> ET.Element | None:
    geom_type = resolve_geom_type(geom, geom_defaults)
    if geom_type is None or is_visual_only_geom(geom):
        return None

    geometry = ET.Element("geometry")
    if geom_type == "box":
        size = parse_floats(resolved_geom_attr(geom, geom_defaults, "size"), [0.0, 0.0, 0.0])
        ET.SubElement(geometry, "box", size=fmt([2.0 * v for v in size]))
    elif geom_type == "sphere":
        size = parse_floats(resolved_geom_attr(geom, geom_defaults, "size"), [0.0, 0.0, 0.0])
        ET.SubElement(geometry, "sphere", radius=f"{size[0]:.9g}")
    elif geom_type == "capsule":
        size = parse_floats(resolved_geom_attr(geom, geom_defaults, "size"), [0.0, 0.0])
        radius = size[0]
        half_len = size[1] if len(size) > 1 else 0.0
        ET.SubElement(geometry, "cylinder", radius=f"{radius:.9g}", length=f"{2.0 * half_len:.9g}")
    elif geom_type == "mesh":
        mesh_name = resolved_geom_attr(geom, geom_defaults, "mesh")
        if mesh_name is None or mesh_name not in mesh_files:
            return None
        ET.SubElement(geometry, "mesh", filename=f"assets/{mesh_files[mesh_name]}")
    else:
        return None
    return geometry


def collect_collision_specs(
    node: ET.Element,
    link_name: str,
    link_names: set[str],
    mesh_files: dict[str, str],
    geom_defaults: dict[str, dict[str, str]],
    base_xyz: list[float] | None = None,
    base_quat: list[float] | None = None,
) -> list[tuple[list[float], list[float], ET.Element]]:
    if base_xyz is None:
        base_xyz = [0.0, 0.0, 0.0]
    if base_quat is None:
        base_quat = [1.0, 0.0, 0.0, 0.0]

    out: list[tuple[list[float], list[float], ET.Element]] = []

    for geom in node.findall("geom"):
        geometry = build_geom_element(geom, mesh_files, geom_defaults)
        if geometry is None:
            continue
        geom_xyz = parse_floats(geom.get("pos"), [0.0, 0.0, 0.0])
        geom_quat = parse_floats(geom.get("quat"), [1.0, 0.0, 0.0, 0.0])
        xyz, quat = compose_tf(base_xyz, base_quat, geom_xyz, geom_quat)
        out.append((xyz, quat_to_rpy(quat), geometry))

    for body in node.findall("body"):
        body_name = body.get("name")
        if body_name is not None and body_name in link_names and body_name != link_name:
            continue
        body_xyz = parse_floats(body.get("pos"), [0.0, 0.0, 0.0])
        body_quat = parse_floats(body.get("quat"), [1.0, 0.0, 0.0, 0.0])
        xyz, quat = compose_tf(base_xyz, base_quat, body_xyz, body_quat)
        out.extend(
            collect_collision_specs(
                body, link_name, link_names, mesh_files, geom_defaults, xyz, quat
            )
        )
    return out


def add_inertial(link_elem: ET.Element, body_elem: ET.Element | None) -> None:
    for child in list(link_elem):
        if child.tag == "inertial":
            link_elem.remove(child)

    if body_elem is None:
        return
    inertial = body_elem.find("inertial")
    if inertial is None:
        return

    inertial_elem = ET.Element("inertial")
    origin = ET.SubElement(
        inertial_elem,
        "origin",
        xyz=fmt(parse_floats(inertial.get("pos"), [0.0, 0.0, 0.0])),
        rpy=fmt(quat_to_rpy(parse_floats(inertial.get("quat"), [1.0, 0.0, 0.0, 0.0]))),
    )
    origin.tail = "\n    "
    mass_elem = ET.SubElement(inertial_elem, "mass", value=inertial.get("mass", "0"))
    mass_elem.tail = "\n    "
    ixx, iyy, izz = parse_floats(inertial.get("diaginertia"), [0.0, 0.0, 0.0])
    inertia_elem = ET.SubElement(
        inertial_elem,
        "inertia",
        ixx=f"{ixx:.9g}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.9g}",
        iyz="0",
        izz=f"{izz:.9g}",
    )
    inertia_elem.tail = "\n  "
    link_elem.insert(0, inertial_elem)


def add_collisions(
    link_elem: ET.Element,
    source_elem: ET.Element,
    link_name: str,
    link_names: set[str],
    mesh_files: dict[str, str],
    geom_defaults: dict[str, dict[str, str]],
) -> None:
    for child in list(link_elem):
        if child.tag == "collision":
            link_elem.remove(child)

    collisions = collect_collision_specs(
        source_elem, link_name, link_names, mesh_files, geom_defaults
    )
    for xyz, rpy, geometry in collisions:
        collision = ET.SubElement(link_elem, "collision")
        ET.SubElement(collision, "origin", xyz=fmt(xyz), rpy=fmt(rpy))
        collision.append(geometry)


def add_joint_dynamics(robot_root: ET.Element) -> None:
    for joint in robot_root.findall("joint"):
        if joint.get("type") in {"revolute", "prismatic"} and joint.find("dynamics") is None:
            joint.append(ET.Element("dynamics", friction="0.1", damping="0.1"))


def main() -> None:
    urdf_tree = ET.parse(URDF_IN)
    urdf_root = urdf_tree.getroot()

    xml_tree = ET.parse(XML_IN)
    xml_root = xml_tree.getroot()
    worldbody = xml_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("worldbody not found in station.xml")

    mesh_files = {
        mesh.get("name"): mesh.get("file")
        for mesh in xml_root.findall("./asset/mesh")
        if mesh.get("name") and mesh.get("file")
    }
    geom_defaults = build_default_geom_attrs(xml_root)

    link_names = {link.get("name") for link in urdf_root.findall("link") if link.get("name")}
    body_map = {
        body.get("name"): body
        for body in worldbody.iter("body")
        if body.get("name") in link_names
    }

    for link in urdf_root.findall("link"):
        link_name = link.get("name")
        if link_name is None:
            continue
        body_elem = body_map.get(link_name)
        add_inertial(link, body_elem)
        source = worldbody if link_name == "base_link" else body_elem
        if source is not None:
            add_collisions(link, source, link_name, link_names, mesh_files, geom_defaults)

    add_joint_dynamics(urdf_root)
    ET.indent(urdf_tree, space="  ")
    urdf_tree.write(URDF_OUT, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {URDF_OUT}")


if __name__ == "__main__":
    main()
