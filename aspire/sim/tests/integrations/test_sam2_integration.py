from __future__ import annotations

import io
import os
import base64

import numpy as np
import pytest
import requests
from PIL import Image

from aspire.sim.cap.integrations.vision import sam2 as sam2_mod


def test_sam2_registers_and_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    def fake_post(url: str, payload: dict) -> dict:
        assert url.endswith("/segment")
        assert "image_base64" in payload
        return {
            "masks": [
                {
                    "shape": list(mask.shape),
                    "mask_base64": base64.b64encode(mask.tobytes()).decode("utf-8"),
                    "score": 0.9,
                }
            ]
        }

    monkeypatch.setattr(sam2_mod, "post_with_retries", fake_post)

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    segment_anything = sam2_mod.init_sam2(device="cpu")
    masks = segment_anything(rgb)
    assert isinstance(masks, list) and len(masks) >= 1
    assert "mask" in masks[0] and "score" in masks[0]


@pytest.mark.integration
def test_sam2_real_init_and_segment() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run real SAM2 test")

    model_name = os.environ.get("ASPIRE_SAM2_MODEL", "facebook/sam2.1-hiera-large")

    segment_anything = sam2_mod.init_sam2(model_name, device="cpu")

    url = os.environ.get(
        "ASPIRE_TEST_IMAGE_URL", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((320, 320))
    rgb = np.asarray(img, dtype=np.uint8)

    masks = segment_anything(rgb)
    assert isinstance(masks, list)
