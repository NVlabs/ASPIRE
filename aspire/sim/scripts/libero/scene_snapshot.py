"""Capture dual-camera scene snapshots and optionally validate SAM3 prompts.

Set ``SNAPSHOT_DIR`` to choose the artifact directory. Set
``ASPIRE_SAM3_PROMPTS`` to a ``|``-separated list after inspecting the images.
The script prints the authoritative ``env.handle.task_language`` so remapped
LIBERO ``_task`` suites are explored correctly.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image


snapshot_dir = Path(os.environ.get("SNAPSHOT_DIR", "/tmp"))
snapshot_dir.mkdir(parents=True, exist_ok=True)

task_language = getattr(env.handle, "task_language", "")
print(f"TASK_LANGUAGE: {task_language}", flush=True)

obs = get_observation()
rgb = obs["agentview"]["images"]["rgb"]
rgb_wrist = obs["robot0_eye_in_hand"]["images"]["rgb"]

Image.fromarray(rgb).save(snapshot_dir / "scene_snapshot.jpg")
Image.fromarray(rgb_wrist).save(snapshot_dir / "scene_snapshot_wrist.jpg")
print(
    f"Snapshots saved: {snapshot_dir}/scene_snapshot.jpg, "
    f"{snapshot_dir}/scene_snapshot_wrist.jpg",
    flush=True,
)

prompts = [
    prompt.strip()
    for prompt in os.environ.get("ASPIRE_SAM3_PROMPTS", "").split("|")
    if prompt.strip()
]
if not prompts:
    print("SAM3 prompt validation skipped: ASPIRE_SAM3_PROMPTS is empty", flush=True)
else:
    print("SAM3 prompt validation:", flush=True)
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        scores = [round(float(mask["score"]), 3) for mask in masks]
        status = "OK" if any(score > 0.5 for score in scores) else ("WEAK" if masks else "FAIL")
        boxes = [mask.get("box") for mask in masks[:5]]
        print(
            f"  [{status}] {prompt!r}: {len(masks)} masks "
            f"scores={scores} boxes={boxes}",
            flush=True,
        )
