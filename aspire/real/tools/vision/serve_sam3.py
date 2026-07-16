#!/usr/bin/env python3
"""Lightweight SAM3-only segmentation server.

Accepts a base64-encoded RGB image + text prompt, runs SAM3 segmentation,
returns the binary mask. No BundleSdf, no tracking, no Portal RPC dependency.

Deployed on lecar server (GPU) — clients send images over HTTP.

Usage
-----
    python tools/vision/serve_sam3.py      # default port 6767
    python tools/vision/serve_sam3.py --port 8119
    python tools/vision/serve_sam3.py --preload   # warm-load SAM3 on startup
"""

import argparse
import base64
import gc
import io
import os
from pathlib import Path
import sys
import threading
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._bootstrap import maybe_reexec_with_uv

maybe_reexec_with_uv(__file__, REPO_ROOT, required_modules=["numpy", "uvicorn"])

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── SAM3 model state ───────────────────────────────────────────────────────────

_sam3_proc: Any = None
_sam3_model: Any = None
_sam3_lock = threading.Lock()
_sam3_attn_impl: str | None = None
_sam3_processor_ref: str | None = None
_sam3_model_ref: str | None = None


def _iter_hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    candidates = [
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        (
            str(Path(os.environ["HF_HOME"]) / "hub")
            if os.environ.get("HF_HOME")
            else None
        ),
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def _resolve_local_hf_snapshot(repo_id: str) -> Path | None:
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    for hub_root in _iter_hf_hub_roots():
        base = hub_root / repo_dir
        if not base.exists():
            continue
        ref_path = base / "refs" / "main"
        if ref_path.exists():
            ref = ref_path.read_text().strip()
            snapshot = base / "snapshots" / ref
            if snapshot.exists():
                return snapshot
        snapshots_dir = base / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(
                [p for p in snapshots_dir.iterdir() if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return snapshots[0]
    return None


def _resolve_sam3_refs() -> tuple[
    str | Path, dict[str, Any], str | Path, dict[str, Any]
]:
    processor_snapshot = _resolve_local_hf_snapshot("facebook/sam3")
    model_snapshot = processor_snapshot

    processor_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if processor_snapshot is None:
        processor_snapshot = "facebook/sam3"
        processor_kwargs["local_files_only"] = True
    if model_snapshot is None:
        model_snapshot = "facebook/sam3"
        model_kwargs["local_files_only"] = True
    return processor_snapshot, processor_kwargs, model_snapshot, model_kwargs


def _load_sam3(device: str):
    from transformers import Sam3Model, Sam3Processor

    global \
        _sam3_proc, \
        _sam3_model, \
        _sam3_attn_impl, \
        _sam3_processor_ref, \
        _sam3_model_ref

    processor_ref, processor_kwargs, model_ref, model_kwargs = _resolve_sam3_refs()
    _sam3_processor_ref = str(processor_ref)
    _sam3_model_ref = str(model_ref)

    print(
        "[text_to_mask] Loading SAM3 "
        f"(processor={_sam3_processor_ref}, model={_sam3_model_ref})..."
    )
    _sam3_proc = Sam3Processor.from_pretrained(processor_ref, **processor_kwargs)

    attn_candidates = ["sdpa", "eager"]
    last_error = None
    for attn_impl in attn_candidates:
        try:
            model = (
                Sam3Model.from_pretrained(
                    model_ref,
                    attn_implementation=attn_impl,
                    **model_kwargs,
                )
                .to(device)
                .eval()
            )
            _sam3_model = model
            _sam3_attn_impl = attn_impl
            print(f"[text_to_mask] SAM3 ready (attn={attn_impl}).")
            return
        except Exception as exc:
            last_error = exc
            print(f"[text_to_mask] SAM3 load failed with attn={attn_impl}: {exc}")

    raise RuntimeError(
        f"Failed to load SAM3 using attention backends: {attn_candidates}"
    ) from last_error


def text_to_masks(
    rgb: np.ndarray, text: str, score_threshold: float = 0.2
) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
    """SAM3 text-prompted segmentation on a single image. Thread-safe.

    Args:
        rgb:             uint8 HxWx3 numpy array (RGB).
        text:            Natural-language description, e.g. "yellow mustard bottle".
        score_threshold: Minimum detection confidence (default 0.2).

    Returns:
        List of (mask_01, bbox_xywh, score), sorted by descending score.
        mask_01   -- uint8 HxW, values 0=background / 1=object.
        bbox_xywh -- tight bounding box of the mask as (x, y, w, h).
        score     -- SAM3 detection confidence in [0, 1].

    Raises:
        RuntimeError if no object is found above score_threshold.
    """
    import torch

    global _sam3_proc, _sam3_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with _sam3_lock:
        if _sam3_model is None:
            _load_sam3(device)

        from PIL import Image as _PIL

        inputs = _sam3_proc(
            images=_PIL.fromarray(rgb),
            text=text,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = _sam3_model(**inputs)
            else:
                outputs = _sam3_model(**inputs)

        results = _sam3_proc.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        if len(masks) == 0:
            raise RuntimeError(
                f"SAM3 could not find '{text}' in frame (threshold={score_threshold}). "
                "Try a more descriptive prompt or move the object into view."
            )

        if hasattr(scores, "detach"):
            scores_np = scores.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            scores_np = np.asarray(scores, dtype=np.float32)

        detections: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
        for idx in np.argsort(-scores_np):
            raw_mask = masks[int(idx)]
            if hasattr(raw_mask, "detach"):
                mask = (
                    raw_mask.detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            else:
                mask = np.asarray(raw_mask, dtype=np.uint8)

            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bbox_xywh = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min()),
                int(ys.max() - ys.min()),
            )
            detections.append((mask, bbox_xywh, float(scores_np[int(idx)])))

        if not detections:
            raise RuntimeError(
                f"SAM3 found masks for '{text}', but all masks were empty."
            )
        return detections


def text_to_mask(
    rgb: np.ndarray, text: str, score_threshold: float = 0.2
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Return the highest-score SAM3 segmentation for backward compatibility."""
    return text_to_masks(rgb, text, score_threshold=score_threshold)[0]


def text_to_masks(
    rgb: np.ndarray, text: str, score_threshold: float = 0.1
) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
    """SAM3 text-prompted segmentation returning all detections above threshold.

    Returns a list of ``(mask_01, bbox_xywh, score)`` tuples sorted by
    descending score. Returns an empty list when nothing is found.
    """
    import torch

    global _sam3_proc, _sam3_model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with _sam3_lock:
        if _sam3_model is None:
            _load_sam3(device)

        from PIL import Image as _PIL

        inputs = _sam3_proc(
            images=_PIL.fromarray(rgb),
            text=text,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = _sam3_model(**inputs)
            else:
                outputs = _sam3_model(**inputs)

        results = _sam3_proc.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        if len(masks) == 0:
            return []

        if hasattr(scores, "detach"):
            scores_np = scores.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            scores_np = np.asarray(scores, dtype=np.float32)

        detections: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
        for i in np.argsort(-scores_np):
            m = masks[i]
            if hasattr(m, "detach"):
                mask = m.detach().to(dtype=torch.float32).cpu().numpy().astype(np.uint8)
            else:
                mask = np.asarray(m, dtype=np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            bbox_xywh = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min()),
                int(ys.max() - ys.min()),
            )
            detections.append((mask, bbox_xywh, float(scores_np[i])))

        return detections


def preload_sam3(device: str = "cuda") -> None:
    """Load SAM3 model into VRAM so the first text_to_mask() is instant."""
    global _sam3_proc, _sam3_model
    with _sam3_lock:
        if _sam3_model is None:
            _load_sam3(device)


class SegmentRequest(BaseModel):
    text: str
    image_b64: str  # base64-encoded uint8 HxWx3 RGB numpy array (np.save format)
    score_threshold: float = 0.2
    return_all: bool = False
    max_results: int | None = None


class SegmentResponse(BaseModel):
    mask_b64: str  # base64-encoded uint8 HxW mask (0/1, np.save format)
    bbox_xywh: list[int]
    score: float
    mask_area: int
    height: int
    width: int
    masks: list[dict[str, Any]] | None = None


class SegmentAllRequest(BaseModel):
    text: str
    image_b64: str
    score_threshold: float = 0.1


class SegmentAllDetection(BaseModel):
    mask_b64: str
    bbox_xywh: list[int]
    score: float
    mask_area: int


class SegmentAllResponse(BaseModel):
    detections: list[SegmentAllDetection]
    count: int
    height: int
    width: int


def create_app() -> FastAPI:
    app = FastAPI(title="SAM3 Segmentation Server")

    @app.post("/segment", response_model=SegmentResponse)
    def segment(req: SegmentRequest):
        """Run SAM3 text-prompted segmentation on the provided image."""
        try:
            img_bytes = base64.b64decode(req.image_b64)
            rgb = np.load(io.BytesIO(img_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

        try:
            detections = text_to_masks(
                rgb,
                req.text,
                score_threshold=float(req.score_threshold),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=404, detail=str(e))

        max_results = 1
        if req.return_all:
            max_results = None if req.max_results is None else max(1, int(req.max_results))
        if max_results is not None:
            detections = detections[:max_results]

        if not detections:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"SAM3 could not find {req.text!r} in frame "
                    f"(threshold={float(req.score_threshold):.3f})."
                ),
            )

        mask_01, bbox_xywh, score = detections[0]
        mask_uint8 = mask_01.astype(np.uint8)
        buf = io.BytesIO()
        np.save(buf, mask_uint8)
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        all_masks = None
        if req.return_all:
            all_masks = []
            for det_mask, det_bbox, det_score in detections:
                det_uint8 = det_mask.astype(np.uint8)
                det_buf = io.BytesIO()
                np.save(det_buf, det_uint8)
                all_masks.append(
                    {
                        "mask_b64": base64.b64encode(det_buf.getvalue()).decode(),
                        "bbox_xywh": [int(x) for x in det_bbox],
                        "score": round(float(det_score), 4),
                        "mask_area": int(det_uint8.sum()),
                    }
                )

        return SegmentResponse(
            mask_b64=mask_b64,
            bbox_xywh=[int(x) for x in bbox_xywh],
            score=round(float(score), 4),
            mask_area=int(mask_uint8.sum()),
            height=mask_uint8.shape[0],
            width=mask_uint8.shape[1],
            masks=all_masks,
        )

    @app.post("/segment_all", response_model=SegmentAllResponse)
    def segment_all(req: SegmentAllRequest):
        """Return all SAM3 detections above the score threshold."""
        try:
            img_bytes = base64.b64decode(req.image_b64)
            rgb = np.load(io.BytesIO(img_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

        all_dets = text_to_masks(
            rgb,
            req.text,
            score_threshold=float(req.score_threshold),
        )

        h, w = rgb.shape[:2]
        detections = []
        for mask_01, bbox_xywh, score in all_dets:
            mask_uint8 = mask_01.astype(np.uint8)
            buf = io.BytesIO()
            np.save(buf, mask_uint8)
            detections.append(
                SegmentAllDetection(
                    mask_b64=base64.b64encode(buf.getvalue()).decode(),
                    bbox_xywh=[int(x) for x in bbox_xywh],
                    score=round(float(score), 4),
                    mask_area=int(mask_uint8.sum()),
                )
            )

        return SegmentAllResponse(
            detections=detections,
            count=len(detections),
            height=h,
            width=w,
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": _sam3_model is not None,
            "attn_implementation": _sam3_attn_impl,
            "processor_ref": _sam3_processor_ref,
            "model_ref": _sam3_model_ref,
        }

    @app.post("/reset_state")
    def reset_state():
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "ok": True,
            "model_loaded": _sam3_model is not None,
            "attn_implementation": _sam3_attn_impl,
            "processor_ref": _sam3_processor_ref,
            "model_ref": _sam3_model_ref,
        }

    return app


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 segmentation server (lightweight)"
    )
    parser.add_argument(
        "--port", type=int, default=6767, help="HTTP port (default: 6767)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--preload", action="store_true", help="Preload SAM3 into VRAM on startup"
    )
    args = parser.parse_args()

    app = create_app()

    if args.preload:
        preload_sam3()

    print(f"[serve_sam3] Starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
