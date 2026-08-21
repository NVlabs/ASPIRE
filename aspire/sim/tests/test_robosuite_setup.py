# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest


@pytest.mark.integration
def test_robosuite_offscreen_smoke() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run Robosuite smoke test")

    suite = pytest.importorskip("robosuite")
    os.environ.setdefault("MUJOCO_GL", "egl")

    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )

    env.reset()
    for _ in range(5):
        action = np.random.randn(*env.action_spec[0].shape) * 0.1
        env.step(action)
    env.close()
