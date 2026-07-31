# SPDX-License-Identifier: Apache-2.0
"""Deterministic preflight: catch NumPy/SciPy ABI mismatches before services launch."""

import importlib
import subprocess
import sys

import pytest


CRITICAL_IMPORTS = [
    "numpy",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.transform",
]

NUMPY2_ONLY_PACKAGES = [
    ("scipy", "1.16"),
]


@pytest.mark.parametrize("module", CRITICAL_IMPORTS)
def test_critical_import(module):
    importlib.import_module(module)


def test_numpy_scipy_abi_compatible():
    import numpy as np
    import scipy

    np_major = int(np.__version__.split(".")[0])
    scipy_minor = tuple(int(x) for x in scipy.__version__.split(".")[:2])
    if scipy_minor >= (1, 16):
        assert np_major >= 2, (
            f"scipy {scipy.__version__} requires numpy>=2.0 but got {np.__version__}"
        )
    else:
        assert np_major < 2 or scipy_minor >= (1, 16), (
            f"numpy {np.__version__} / scipy {scipy.__version__} combination untested"
        )


def test_pyroki_imports():
    pyroki = pytest.importorskip("pyroki")
    importlib.import_module("pyroki.robot")


def test_contact_graspnet_imports():
    cgn = pytest.importorskip("contact_graspnet_pytorch")


def test_uv_lock_no_scipy_18():
    """Ensure uv.lock never resolves scipy>=1.16 (needs numpy 2.x)."""
    lock = __import__("pathlib").Path(__file__).resolve().parents[1] / "uv.lock"
    if not lock.exists():
        pytest.skip("uv.lock not found")
    content = lock.read_text()
    for line in content.splitlines():
        if "scipy" in line and "version" in line:
            for bad in ("1.16.", "1.17.", "1.18.", "1.19.", "2."):
                assert f'version = "{bad}' not in line, (
                    f"uv.lock contains scipy {bad}x which requires numpy>=2.0: {line.strip()}"
                )
