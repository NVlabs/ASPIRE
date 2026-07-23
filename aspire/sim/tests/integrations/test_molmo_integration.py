# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from aspire.sim.cap.integrations.vision.molmo import init_molmo


@pytest.mark.integration
def test_molmo_table_points() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run real Molmo test")

    image_path = Path("scripts/images/table.jpg")
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    image = Image.open(image_path)
    molmo_det_fn = init_molmo()
    points = molmo_det_fn(
        image, objects=["handle of the square nut", "square nut center", "square block"]
    )
    assert isinstance(points, dict)
