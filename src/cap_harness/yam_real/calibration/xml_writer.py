# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Write camera extrinsics into a copy of the supplied MuJoCo station XML."""

from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .core import transform


def write_calibrated_xml(base_xml: Path, output_xml: Path, record: dict) -> None:
    """Place fixed cameras in world, or wrist cameras relative to link 6.

    Reparenting matters: fixed-camera hand-eye returns a world transform, which
    must not be written as a local transform under an offset mounting bracket.
    """
    base_xml, output_xml = base_xml.resolve(), output_xml.resolve()
    if base_xml == output_xml or output_xml.exists():
        raise ValueError("choose a new output XML; the input model is never overwritten")
    tree = ET.parse(base_xml)
    root = tree.getroot()
    if root.find(".//include") is not None:
        raise ValueError("use a single-file station XML with no include elements")
    parent_name = record["parent_frame"]
    parent = (
        root.find("worldbody")
        if parent_name == "world"
        else root.find(f".//body[@name='{parent_name}']")
    )
    if parent is None:
        raise ValueError(f"camera parent not found in station XML: {parent_name}")
    body_name = record["camera_body"]
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        body = ET.Element("body", name=body_name)
    else:
        old_parent = next(node for node in root.iter() if body in list(node))
        old_parent.remove(body)
    parent.append(body)
    pose = transform(record["solution"]["T_parent_from_camera"])
    for key in ("euler", "axisangle", "xyaxes", "zaxis"):
        body.attrib.pop(key, None)
    body.set("pos", " ".join(f"{v:.10g}" for v in pose[:3, 3]))
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    body.set("quat", " ".join(f"{v:.10g}" for v in quat))
    for node in body.iter():
        for camera in list(node.findall("camera")):
            node.remove(camera)
    intrinsics = np.asarray(record["intrinsics"], dtype=float)
    width, height = record["resolution"]
    ET.SubElement(
        body,
        "camera",
        name=record["camera_name"],
        pos="0 0 0",
        # MuJoCo looks down -Z with +Y up; calibration uses optical +Z/+Y down.
        quat="0 1 0 0",
        resolution=f"{width} {height}",
        fovy=str(np.rad2deg(2 * np.arctan(height / (2 * intrinsics[1, 1])))),
    )
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    # Keep generated models loadable outside the repository's model directory.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    assets = root.find("asset")
    if assets is not None:
        for node in assets:
            filename = node.get("file")
            if not filename:
                continue
            directory = compiler.get("meshdir" if node.tag == "mesh" else "texturedir", "")
            source = (base_xml.parent / directory / filename).resolve()
            relative = Path("assets") / source.name
            destination = output_xml.parent / relative
            destination.parent.mkdir(exist_ok=True)
            if source != destination:
                if destination.exists() and destination.read_bytes() != source.read_bytes():
                    raise ValueError(f"different assets share a filename: {source.name}")
                shutil.copyfile(source, destination)
            node.set("file", relative.as_posix())
    compiler.attrib.pop("meshdir", None)
    compiler.attrib.pop("texturedir", None)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
