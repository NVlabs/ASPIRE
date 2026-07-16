---
name: localize
description: Object localization via SAM3 — prompting strategies, multi-prompt fallback pattern, 3D centroid extraction, and per-object prompt registry. Grows through experiment.
---

# Localize — SAM3 Prompting & Object Localization

> This skill tracks **what SAM3 prompts work for which objects** and the standard
> localization helper pattern. Add entries to the prompt registry as you discover
> working prompts through experiment.

---

## Standard Localization Helper

```python
import numpy as np

def localize_object(rgb, depth, K, E, prompts):
    """Try prompts in order, return (center, pts, mask) for first hit with ≥10 points."""
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(pts)["center"]
        return center, pts, mask
    return None, None, None
```

**Usage:**
```python
obs = get_observation()
cam = obs["agentview"]
center, pts, mask = localize_object(
    cam["images"]["rgb"], cam["images"]["depth"],
    cam["intrinsics"], cam["pose_mat"],
    ["<specific prompt>", "<fallback prompt>"]
)
if center is None:
    raise RuntimeError("Object not found")
```

**Why OBB center over mean:** `get_oriented_bounding_box_from_3d_points` gives a more
robust center estimate for elongated or partially occluded objects than `pts.mean(axis=0)`.

---

## Prompt Registry

Discovered working prompts, indexed by object. Add entries as you find them.
List prompts in priority order — first hit wins.

| Object | Working Prompts | Suite/Task | Notes |
|---|---|---|---|

---

## Disambiguation: Two Similar Objects in Scene

When a scene contains two visually similar objects, `max(score)` often returns the wrong one.
Use **bbox pixel area** to discriminate by size:

```python
def localize_smallest_bbox_mask(rgb, prompts, max_cy=None):
    """Select the mask with smallest bbox pixel area (most compact matching shape).
    max_cy: if set, only accept masks with bbox center-y below this row (upper image = farther away).
    Add min_w/max_w/min_h/max_h filters as you learn the object's image footprint.
    """
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best_mask = None
        best_area = float('inf')
        for m in masks[:10]:
            bbox = m.get('box', None)
            if bbox is None:
                continue
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            px_area = w * h
            cy = (bbox[1] + bbox[3]) / 2
            if (max_cy is None or cy < max_cy) and px_area < best_area:
                best_area = px_area
                best_mask = m
        if best_mask is not None:
            return best_mask["mask"]
    return None
```

**Key**: `area_pct` in trace dicts truncates to 99.00 for all masks. Compute area from `box` coordinates instead.

---

## Base-Object Anchored Localization

When object A sits **on top of** object B, and SAM3 for A is unreliable, find B first via SAM3
then depth-search above B's image region.

```python
def localize_above_object(rgb, depth, K, E, base_prompts, min_z=0.04, max_z=0.20):
    """Find object sitting on top of base_object by depth-searching above base image mask."""
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    for prompt in base_prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks: continue
        best = max(masks, key=lambda d: d["score"])
        base_mask = best["mask"]
        ys, xs = np.where(base_mask)
        if len(ys) == 0: continue
        extended_mask = base_mask.copy()
        for dy in range(1, 31):
            shifted = np.zeros_like(base_mask, dtype=bool)
            shifted[max(0, ys.min()-dy):max(0, ys.max()-dy+1), xs.min():xs.max()+1] = True
            extended_mask = extended_mask | shifted
        pts = mask_to_world_points(extended_mask.astype(np.uint8), depth_img, K, E)
        if pts is None: continue
        elevated = pts[(pts[:,2] > min_z) & (pts[:,2] < max_z)]
        if len(elevated) < 10: continue
        center = elevated.mean(axis=0)
        return center, elevated
    return None, None
```

**Why:** Finding a unique base object first gives a stable anchor when the target object has
unreliable SAM3 segmentation due to similar objects nearby.

---

## Key Signals

- `num_masks = 0` → prompt not recognized — try a more specific or different description
- `score < 0.5` → low confidence — try alternative prompt before accepting
- `len(pts) < 10` → mask too small or object occluded — try different prompt
- Usually pick `max(masks, key=lambda d: d["score"])` — highest confidence mask
- **Exception: size disambiguation** — when two similar objects exist, use bbox pixel area `(box[2]-box[0])*(box[3]-box[1])` to select by size, not score. The `box` field in each mask dict gives `[x1,y1,x2,y2]` in pixels.
- **Exception: geometry filtering** — when a taller/larger object gets a higher SAM3 score than the target, filter by 3D Z-height first, then pick highest score among geometry-matching candidates.

---

## Prompting Strategy

1. **Be specific first** — include color + shape + material: `"blue rectangular box"`
2. **Fall back to generic** — shorter, simpler descriptions
3. **For targets (bowls, plates, racks)** — material descriptor helps: `"silver bowl"` > `"bowl"`
4. **Always confirm with `env.handle.task_language`** — authoritative instruction regardless of suite
5. **Detect target BEFORE grasping** — post-lift re-observation is corrupted by the robot arm blocking the camera; detect both object and target while the arm is at home position and view is clean
6. **ARM OCCLUSION PATTERN** — If the target object is in the CENTER of the table, it may be hidden behind the robot arm at home position. Move arm to an observation position (e.g., `solve_ik([0.55, 0.30, 0.40], TOP_DOWN)`) before observing. Symptom: SAM3 finds cans/flat objects but not the target, scores are low (<0.2) for the intended object.

---

## Placement Z Notes

SAM3 mask centroid Z reflects the **camera-facing surface**, not the true top.
For targets with vertical extent (bowls, raised platforms), adjust placement Z from observed trace data.

| Target | Z formula | Notes |
|---|---|---|
