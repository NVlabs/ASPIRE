# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from aspire.sim.cap.integrations.motion import pyroki as pyroki_mod


def test_init_pyroki_returns_ik_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, payload: dict[str, Any], timeout_seconds: float = 15.0) -> dict:
        assert url == "http://planner.test/ik"
        assert "target_pose_wxyz_xyz" in payload
        assert timeout_seconds == 15.0
        return {"joint_positions": list(range(7))}

    monkeypatch.setattr(pyroki_mod, "post_with_retries", fake_post)

    ik_solve = pyroki_mod.init_pyroki("http://planner.test")
    q = ik_solve(np.eye(4, dtype=np.float64))

    assert q.shape == (7,)
    assert np.allclose(q, np.arange(7, dtype=np.float32))


def test_init_pyroki_trajopt_returns_plan_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, payload: dict[str, Any]) -> dict:
        assert url == "http://planner.test/plan"
        assert "start_pose_wxyz_xyz" in payload
        assert "end_pose_wxyz_xyz" in payload
        return {"waypoints": [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]}

    monkeypatch.setattr(pyroki_mod, "post_with_retries", fake_post)

    plan = pyroki_mod.init_pyroki_trajopt("http://planner.test")
    waypoints = plan(np.zeros(7), np.ones(7))

    assert waypoints.shape == (2, 3)
    assert np.allclose(waypoints[1], [0.3, 0.4, 0.5])


@pytest.mark.integration
def test_pyroki_real_service() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run real PyRoKI service test")

    ik_solve = pyroki_mod.init_pyroki(os.environ.get("PYROKI_SERVICE_URL", pyroki_mod.DEFAULT_URL))
    q = ik_solve(np.eye(4))
    assert q.shape[0] >= 6
