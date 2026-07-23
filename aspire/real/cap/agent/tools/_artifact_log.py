# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save visual artifacts (annotated images, masks) from tool calls to the run log.

Usage from any tool module::

    from cap.agent.tools._artifact_log import log_detection, set_artifact_dir

    set_artifact_dir(session.run_dir)
    log_detection(rgb, detections, tag="detect_object")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cap.agent.tools.base import Detection3D

logger = logging.getLogger(__name__)

_artifact_dir: Path | None = None


def set_artifact_dir(path: Path | None) -> None:
    """Set the directory for saving visual artifacts. Creates ``vis/`` subdir."""
    global _artifact_dir, _vlm_query_counter
    _vlm_query_counter = 0
    if path is not None:
        vis = Path(path) / "vis"
        vis.mkdir(parents=True, exist_ok=True)
        _artifact_dir = vis
        logger.info("Artifact log dir: %s", vis)
    else:
        _artifact_dir = None


def _stamp() -> str:
    return time.strftime("%H%M%S")


def _vis_subdir(name: str) -> Path | None:
    if _artifact_dir is None:
        return None
    path = _artifact_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_detection(
    rgb: np.ndarray,
    detections: list[Detection3D],
    *,
    tag: str = "detect",
) -> Path | None:
    """Save camera image annotated with detection bboxes and 3D positions."""
    if _artifact_dir is None or rgb is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(rgb.copy())
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
            )
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            label = det.label
            pos = det.position_3d
            score = det.score

            # Draw bbox if available
            if det.box_2d and len(det.box_2d) == 4:
                x1, y1, x2, y2 = [int(v) for v in det.box_2d]
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                text_pos = (x1, max(0, y1 - 16))
            else:
                text_pos = (10, 10)

            text = f"{label} [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] s={score:.2f}"
            draw.text(text_pos, text, fill="lime", font=font)

        safe_label = (
            detections[0].label.replace(" ", "_")[:20] if detections else "none"
        )
        name = f"{_stamp()}_{tag}_{safe_label}.png"
        out_dir = _vis_subdir("detect")
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        logger.debug("Saved detection artifact: %s", path)
        return path
    except Exception:
        logger.warning("Failed to save detection artifact", exc_info=True)
        return None


def log_grasp(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    grasp_candidates: list,
    *,
    query: str = "",
    tag: str = "grasp",
    alpha: float = 0.35,
) -> Path | None:
    """Save camera image with SAM3 mask overlay + grasp pose arrows.

    Each grasp candidate should have ``.position`` (world xyz),
    ``.rpy`` (display degrees), ``.score``, and ``.width``.
    The grasp positions are projected to 2D if ``_project_fn`` is set,
    otherwise drawn at fixed positions as a fallback legend.
    """
    if _artifact_dir is None or rgb is None:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        canvas = rgb.copy()
        if mask is not None:
            mask_bool = mask.astype(bool)
            canvas[mask_bool, 1] = np.clip(
                canvas[mask_bool, 1].astype(np.float32) + 80, 0, 255
            ).astype(np.uint8)
            canvas = (
                rgb.astype(np.float32) * (1 - alpha) + canvas.astype(np.float32) * alpha
            ).astype(np.uint8)

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
            )
        except Exception:
            font = ImageFont.load_default()

        colors = ["#FF4444", "#FF8800", "#FFCC00", "#44FF44", "#4488FF"]
        h = rgb.shape[0]
        for i, g in enumerate(grasp_candidates[:5]):
            color = colors[i % len(colors)]
            pos = g.position
            rpy = g.rpy
            score = g.score
            width = getattr(g, "width", 0.08)
            y_text = h - 18 * (len(grasp_candidates[:5]) - i) - 4
            text = (
                f"G{i}: [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
                f"rpy=[{rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f}] "
                f"s={score:.3f} w={width:.3f}"
            )
            draw.text((4, y_text), text, fill=color, font=font)

        if query:
            draw.text((4, 4), query, fill="lime", font=font)

        safe_q = query.replace(" ", "_")[:20] if query else "grasp"
        name = f"{_stamp()}_{tag}_{safe_q}.png"
        out_dir = _vis_subdir("grasp")
        if out_dir is None:
            return None
        path = out_dir / name
        img.save(path)
        logger.debug("Saved grasp artifact: %s", path)
        return path
    except Exception:
        logger.warning("Failed to save grasp artifact", exc_info=True)
        return None


_vlm_query_counter: int = 0


def log_vlm_query(
    prompt: str,
    response: str,
    *,
    images: list[np.ndarray] | None = None,
    media_labels: list[str] | None = None,
    backend: str = "",
    tag: str = "vlm_query",
) -> Path | None:
    """Save VLM query input and output as separate markdown files.

    Creates text transcripts under ``{run_dir}/vlm/{NNN}_{tag}/`` and copies
    browser-visible input images under ``{run_dir}/vis/vlm/{NNN}_{tag}/``.
    """
    if _artifact_dir is None:
        return None
    global _vlm_query_counter
    _vlm_query_counter += 1
    idx = _vlm_query_counter

    try:
        # vlm/ keeps text transcripts; vis/vlm/ keeps browser-visible images.
        vlm_dir = _artifact_dir.parent / "vlm" / f"{idx:03d}_{tag}"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        vlm_vis_root = _vis_subdir("vlm")
        if vlm_vis_root is None:
            return None
        vlm_vis_dir = vlm_vis_root / f"{idx:03d}_{tag}"
        vlm_vis_dir.mkdir(parents=True, exist_ok=True)

        # Input markdown
        input_lines = [f"# VLM Query #{idx}\n"]
        if backend:
            input_lines.append(f"**Backend:** {backend}\n")
        if media_labels:
            input_lines.append("**Images:** " + ", ".join(media_labels) + "\n")
        input_lines.append(f"\n## Prompt\n\n{prompt}\n")
        (vlm_dir / "input.md").write_text("\n".join(input_lines), encoding="utf-8")

        # Save input images
        if images:
            from PIL import Image

            for i, img_arr in enumerate(images):
                lbl = (
                    media_labels[i]
                    if media_labels and i < len(media_labels)
                    else f"image_{i}"
                )
                safe_lbl = (
                    lbl.replace(":", "_").replace("/", "_").replace(" ", "_")[:30]
                )
                img = Image.fromarray(img_arr)
                img.save(vlm_vis_dir / f"{safe_lbl}.png")

        # Output markdown
        output_lines = [f"# VLM Response #{idx}\n"]
        output_lines.append(f"\n## Response\n\n{response}\n")
        (vlm_dir / "output.md").write_text("\n".join(output_lines), encoding="utf-8")

        logger.debug("Saved VLM query #%d to %s", idx, vlm_dir)
        return vlm_dir
    except Exception:
        logger.warning("Failed to save VLM query artifact", exc_info=True)
        return None
