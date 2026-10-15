---
name: localize
description: Object localization via SAM3 — prompting strategies, multi-prompt fallback pattern, candidate disambiguation, and per-object prompt registry. Grows through experiment.
---

# Localize — SAM3 Prompting & Object Localization

> This skill tracks **what SAM3 prompts work for which objects** and the standard
> localization helper pattern. Prompt strings go in the registry table; any selection or
> filtering logic that needs code becomes its own subsection (trigger → code → evidence),
> modeled on *Disambiguation* below.
> Programs cannot import modules — all helpers below are pure Python over public contracts.

---

## Standard Localization Helper

`localize_object()` already composes text segmentation, RGB-D projection, and geometry
estimation. Wrap it with a prompt-fallback loop and a minimum-point guard:

```python
def locate(queries, camera_name="agentview"):
    """Try prompts in order, return the first LocalizationResult with >= 20 points."""
    base_frame = get_robot_state().base_frame
    for query in queries:
        found = localize_object(query, camera_name=camera_name, target_frame=base_frame)
        if found.ok and len(found.point_cloud.points) >= 20:
            return found
    return None

bottle = locate(("<specific prompt>", "<fallback prompt>"))
if bottle is None:
    raise ValueError("object not found")
center = bottle.geometry.pose.position   # framed center from estimate_geometry
```

**Why the geometry center over a raw mean:** `estimate_geometry` gives a more robust center
for elongated or partially occluded objects than averaging raw points.

---

## Prompt Registry

Discovered working prompts, indexed by object. Add entries as you find them.
List prompts in priority order — first hit wins. This table is for prompt strings and scores
only — if a finding needs procedure or code (candidate filtering, camera choice, occlusion
handling), write it as a subsection instead and keep at most a pointer row here.

| Object | Working Prompts | Benchmark/Suite/Task | Notes |
|---|---|---|---|

---

## Disambiguation: Two Similar Objects in Scene

When a scene contains two visually similar objects, the top-scored candidate is often the wrong
one. Inspect all candidates from `segment_text()` and discriminate by bbox pixel area, image
position, or projected 3D geometry:

```python
def smallest_box_candidate(camera_name, prompts, max_center_y=None):
    """Select the candidate with smallest bbox area (most compact matching shape).
    max_center_y: if set, only accept boxes whose center row is above it (upper image = farther).
    """
    for prompt in prompts:
        found = segment_text(camera_name, prompt)
        if not found.ok:
            continue
        best = None
        best_area = None
        for candidate in found.segmentations[:10]:
            if candidate.box_xyxy is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in candidate.box_xyxy)
            area = (x2 - x1) * (y2 - y1)
            center_y = (y1 + y2) / 2.0
            if max_center_y is not None and center_y >= max_center_y:
                continue
            if best_area is None or area < best_area:
                best_area = area
                best = candidate
        if best is not None:
            return best
    return None
```

Project the selected candidate with `mask_to_point_cloud(candidate, camera_name)` and compare
observable geometry with `estimate_geometry()` — filter by 3D Z-height first when a taller or
larger object outranks the target, then pick the highest score among geometry-matching
candidates.

Never disambiguate with native simulator objects, hidden IDs, known seed coordinates, or fixed
pixel locations. Re-localize after contact or a failed grasp — the object may have moved; do not
reuse a stale center.

---

## Key Signals

- `segment_text(...).ok` false or zero `segmentations` → prompt not recognized — try a more
  specific or different description (the trace span records every candidate's score and box)
- `score < 0.5` → low confidence — try an alternative prompt before accepting
- `len(point_cloud.points) < 20` → mask too small or object occluded — try a different prompt
- `media/overlays/` in the recorded run renders the retained top-1 mask — verify it covers the
  intended object

---

## Prompting Strategy

1. **Be specific first** — include color + shape + material: `"blue rectangular box"`
2. **Fall back to generic** — shorter, simpler descriptions; prefer category or shape prompts
   when a product-specific prompt is unreliable
3. **For targets (bowls, plates, racks)** — a material descriptor helps: `"silver bowl"` > `"bowl"`
4. **Always confirm with `get_task_context().language`** — the authoritative instruction
   regardless of suite or task name
5. **Detect target BEFORE grasping** — post-lift re-observation is corrupted by the robot arm
   blocking the camera; detect both object and target while the arm is at home position and the
   view is clean
6. **ARM OCCLUSION PATTERN** — if the target object is in the CENTER of the table, it may be
   hidden behind the robot arm at home position. Move the arm to an observation pose with
   `move_to_pose` before observing. Symptom: SAM3 finds other objects but not the target, and
   scores are low (<0.2) for the intended object.

---

## Placement Z Notes

A mask's projected Z reflects the **camera-facing surface**, not the true top.
For targets with vertical extent (bowls, raised platforms), adjust placement Z from observed
run evidence.

| Target | Z formula | Notes |
|---|---|---|
