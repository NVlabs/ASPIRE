# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic preflight: enforce the locked NumPy/SciPy pair and catch
import-time ABI mismatches before services or task agents launch."""

import importlib
from pathlib import Path

import pytest

EXPECTED_NUMPY = "1.26.4"
EXPECTED_SCIPY = "1.15.3"

CRITICAL_IMPORTS = [
    "numpy",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.transform",
]


@pytest.mark.parametrize("module", CRITICAL_IMPORTS)
def test_critical_import(module):
    importlib.import_module(module)


def test_numpy_version_pinned():
    import numpy as np
    assert np.__version__ == EXPECTED_NUMPY, (
        f"Expected numpy=={EXPECTED_NUMPY}, got {np.__version__}"
    )


def test_scipy_version_pinned():
    import scipy
    assert scipy.__version__ == EXPECTED_SCIPY, (
        f"Expected scipy=={EXPECTED_SCIPY}, got {scipy.__version__}"
    )


def test_numpy_scipy_import_compat():
    """SciPy must import successfully against the installed NumPy — an ABI
    mismatch raises ImportError or AttributeError at import time."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("z", 45, degrees=True)
    m = r.as_matrix()
    assert m.shape == (3, 3)


def test_pyroki_imports():
    pyroki = pytest.importorskip("pyroki")
    assert hasattr(pyroki, "Robot")


def test_contact_graspnet_imports():
    pytest.importorskip("contact_graspnet_pytorch")
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    assert callable(GraspEstimator)


def _check_lock_versions(pkg: str, expected: str):
    """Assert every resolved version line for pkg in uv.lock matches expected."""
    import re
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    if not lock.exists():
        pytest.skip("uv.lock not found")
    pattern = re.compile(
        rf'\{{\s*name\s*=\s*"{re.escape(pkg)}"\s*,\s*version\s*=\s*"([^"]+)"'
    )
    for line in lock.read_text().splitlines():
        m = pattern.search(line)
        if m and m.group(1) != expected:
            pytest.fail(
                f"uv.lock resolves {pkg} {m.group(1)} (expected only {expected}): "
                f"{line.strip()}"
            )


def test_uv_lock_scipy_pinned():
    _check_lock_versions("scipy", EXPECTED_SCIPY)


def test_uv_lock_numpy_pinned():
    _check_lock_versions("numpy", EXPECTED_NUMPY)
