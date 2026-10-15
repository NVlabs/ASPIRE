"""Safe numpy/base64 helpers shared by provider wire protocols."""

from __future__ import annotations

import base64
import binascii
import io

import numpy as np


def encode_numpy(array: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_numpy(value: object) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError("numpy payload must be a base64 string")
    try:
        raw = base64.b64decode(value, validate=True)
        with io.BytesIO(raw) as buffer:
            result = np.load(buffer, allow_pickle=False)
    except (ValueError, OSError, binascii.Error) as exc:
        raise ValueError("invalid numpy/base64 payload") from exc
    return np.asarray(result)


__all__ = ["decode_numpy", "encode_numpy"]
