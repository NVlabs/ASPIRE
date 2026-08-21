# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "station.urdf"
DEFAULT_DST = ROOT / "station_fixed_fingers.urdf"

FINGER_JOINTS = {
    "left_left_finger_joint",
    "left_right_finger_joint",
    "right_left_finger_joint",
    "right_right_finger_joint",
}


def build_fixed_finger_urdf(src: Path, dst: Path) -> None:
    tree = ET.parse(src)
    root = tree.getroot()

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in FINGER_JOINTS:
            continue
        joint.set("type", "fixed")
        for child in list(joint):
            if child.tag in {"axis", "limit", "dynamics"}:
                joint.remove(child)

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--dst", default=str(DEFAULT_DST))
    args = parser.parse_args()
    build_fixed_finger_urdf(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
