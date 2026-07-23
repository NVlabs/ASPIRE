# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os

import numpy as np
import pytest
import requests
from PIL import Image


@pytest.mark.integration
def test_graspnet_real_depth_to_grasps() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run real GraspNet test")

    from aspire.sim.cap.integrations.vision.graspnet import init_contact_graspnet
    # The source-only Contact-GraspNet installation is external to ASPIRE.
    # Skip gracefully if its PointNet2 CUDA extension is unavailable.
    try:
        import pointnet2._ext  # type: ignore  # noqa: F401
    except Exception:
        pytest.skip(
            "external Contact-GraspNet pointnet2._ext is not installed for this environment"
        )

    # Init model (optional checkpoint path via env)
    ckpt = os.environ.get("ASPIRE_GRASPNET_CKPT", "") or None
    grasp_plan = init_contact_graspnet(device="cpu", checkpoint_path=ckpt)

    # Download an RGB image and synthesize a simple depth by grayscale
    url = os.environ.get(
        "ASPIRE_TEST_IMAGE_URL", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((160, 120))
    rgb = np.asarray(img, dtype=np.uint8)
    depth = np.mean(rgb, axis=2).astype(np.float32) / 255.0  # fake depth in meters ~ [0,1]

    # Simple pinhole intrinsics for backprojection; not used directly in graspnet baseline but required by our API
    K = np.array(
        [[120.0, 0, depth.shape[1] / 2], [0, 120.0, depth.shape[0] / 2], [0, 0, 1]],
        dtype=np.float32,
    )

    segmap = np.ones_like(depth, dtype=np.uint8)
    grasps, scores, contact_pts = grasp_plan(depth, K, segmap, 1)
    assert isinstance(grasps, (list, np.ndarray))
    assert isinstance(scores, (list, np.ndarray))
    assert isinstance(contact_pts, (list, np.ndarray))
