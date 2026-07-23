# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle
from PIL import Image

from aspire.sim.cap.integrations.vision.molmo import init_molmo
from aspire.sim.cap.integrations.vision.sam2 import init_sam2_point_prompt


def _to_numpy_mask(mask_like: Any) -> np.ndarray:
    """Convert various mask types to a boolean numpy array."""
    if hasattr(mask_like, "detach"):
        arr = mask_like.detach().cpu().numpy()
    elif hasattr(mask_like, "cpu"):
        arr = mask_like.cpu().numpy()
    elif hasattr(mask_like, "numpy"):
        arr = mask_like.numpy()
    else:
        arr = np.asarray(mask_like)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {arr.shape}")
    return arr > 0


def _iter_mask_candidates(masks: Any) -> list[Any]:
    """Collect mask-like objects from nested SAM2 outputs."""
    if masks is None:
        return []
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            return [masks]
        if masks.ndim >= 3:
            return [masks[idx] for idx in range(masks.shape[0])]
        raise ValueError(f"Unexpected mask array shape {masks.shape}")
    if isinstance(masks, Sequence) and not isinstance(masks, (bytes, str)):
        candidates: list[Any] = []
        for entry in masks:
            candidates.extend(_iter_mask_candidates(entry))
        return candidates
    return [masks]


def _visualize_masks(
    image: Image.Image,
    point_coords: tuple[float, float],
    scores: Sequence[float],
    masks: Sequence[Any],
    max_masks: int = 3,
    save_dir: str | Path | None = None,
) -> None:
    """Overlay SAM2 masks on the image, show point prompt, and optionally save JPGs."""
    mask_candidates = _iter_mask_candidates(masks)
    if not mask_candidates:
        raise ValueError("No masks found in SAM2 output; cannot visualize.")

    masks_to_show = [_to_numpy_mask(mask_like) for mask_like in mask_candidates[:max_masks]]
    output_dir = Path(save_dir) if save_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(masks_to_show) + 1, figsize=(4 * (len(masks_to_show) + 1), 4))

    axes[0].imshow(image)
    axes[0].add_patch(
        Circle(point_coords, radius=8, edgecolor="red", facecolor="none", linewidth=2)
    )
    axes[0].set_title("Original Image")

    image_np = np.array(image)
    for idx, mask in enumerate(masks_to_show, start=1):
        overlay = image_np.copy()
        overlay[mask] = [255, 0, 0]
        axes[idx].imshow(overlay)

        if output_dir:
            mask_path = output_dir / f"mask_{idx}_{scores[idx - 1]:.2f}.jpg"
            Image.fromarray(overlay).save(mask_path)

        score_txt = f"{scores[idx - 1]:.2f}" if idx - 1 < len(scores) else "n/a"
        axes[idx].set_title(f"Mask {idx} | score={score_txt}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "sam2_overlays.jpg")
        plt.close()
    else:
        plt.show()


@pytest.mark.integration
def test_sam2_point_prompt_visualization() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run real SAM2 visualization test")

    image_path = Path("first_frame.jpg")
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    image = Image.open(image_path)
    molmo_det_fn = init_molmo()
    points = molmo_det_fn(
        image, objects=["handle of the square nut", "square nut center", "square block"]
    )

    sam2_det_fn = init_sam2_point_prompt()
    for point_name, point_coords in points.items():
        scores, masks = sam2_det_fn(image, point_coords=point_coords)
        _visualize_masks(
            image=image,
            point_coords=point_coords,
            scores=scores,
            masks=masks,
            save_dir=f"outputs/sam2_overlays/{point_name}",
        )
