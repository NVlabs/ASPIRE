"""Optional NumPy compatibility shims for legacy third-party dependencies.

Python auto-imports ``sitecustomize`` for every process started from the repo
root, so keep this module inert by default. Call
``apply_numpy_compat_aliases()`` explicitly from legacy integration points (for
example the AnyGrasp server) or set ``CAP_ENABLE_NUMPY_LEGACY_ALIASES=1`` when a
whole process really needs the aliases.
"""

from __future__ import annotations

import os


def apply_numpy_compat_aliases() -> None:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return

    alias_map = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in alias_map.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    if not hasattr(np, "maximum_sctype"):
        def _maximum_sctype(t):
            dt = np.dtype(t)
            kind = dt.kind
            if kind == "b":
                return np.bool_
            if kind in {"i", "u"}:
                return np.int64 if kind == "i" else np.uint64
            if kind == "f":
                return np.float64
            if kind == "c":
                return np.complex128
            return dt.type

        np.maximum_sctype = _maximum_sctype


if os.environ.get("CAP_ENABLE_NUMPY_LEGACY_ALIASES") == "1":
    apply_numpy_compat_aliases()
