### Bowl *(rim-edge offset grasp)* ⭐ **HERO**

#### Problem
Gripper closes on a wide concave object but the grasp is unstable: `gripper_width ≈ 0` yet the object Z doesn't change after lift. The grasp contacted the rim tangentially and slipped off.

#### Strategy
Maintain a curated **per-object prompt table**: canonical noun first, followed by color/texture-qualified alternates in priority order. Each entry is validated against ground truth on a fixed object set.

#### Code example
```python
SAM3_PROMPT_REGISTRY = {
    "bowl": ["white ceramic bowl", "bowl", "round dish"],
    "soda_can": ["red cylindrical can", "soda can", "aluminum can"],
    # … 24 objects total, validated 2026-03-06
}

def localize_object(name, rgb):
    for prompt in SAM3_PROMPT_REGISTRY[name]:
        masks = sam3(rgb, prompt)
        if len(masks) > 0 and masks[0].score > 0.25:
            return masks[0]
    return None
```

#### When to Apply
- Concave/wide objects (bowl, plate, moka pot base) with diameter wider than gripper span.
- Previous centroid-grasp attempt returned `success=True` with no Z change.

#### Strategy
Don't grasp at the centroid. Offset toward the **rim edge** so one jaw contacts the outer wall and the other hooks under the inner rim. Grasp at rim height (85th-percentile Z of the OBB), not at object centroid Z. This entry evolved from a smaller offset to the current one after trace analysis showed consistent rim-slip at the smaller value.

#### Code example
```python
# Offset from centroid toward rim edge (generally, not a magic number)
rim_offset = RIM_OFFSET_FOR_CLASS[object_class]   # tuned per shape class
grasp_xy   = centroid_xy + rim_offset * rim_direction

# Grasp at rim height, not centroid height
grasp_z    = percentile_z(object_points, 85)

goto_pose([grasp_xy[0], grasp_xy[1], grasp_z], quat=TOP_DOWN)
close_gripper()
close_gripper()  # double-close: seats the hook under the rim
```

#### YAM Lesson: Encoder Contact Is Not The Only Bowl Hold Evidence

For physical YAM bowl attempts, a staged close that reaches its commanded
normalized target with only a tiny final-target delta is not hard encoder
contact. It can still be usable hold evidence for a rim-straddle bowl grasp
when the pre-close axis/pose gates pass, the final gripper state is in the
plausible hold range, and the next action is only a small lift followed by
post-lift perception/video verification.

Validated examples:
`logs/place_orange_bowl_on_rack_one_shot_loop_20260530T151317` and
`logs/place_white_bowl_on_rack_one_shot_loop_20260530T161442`. Both used
right-arm top-down rim straddle, closed near target `0.04`, had
`contact_detected=false` from encoder delta, but had `plausible_for_lift=true`
and post-lift observation verifying object motion before returning the bowl to
the table.

Do not claim rack placement from target-reached alone. Use relaxed
contact-before-lift only for the bounded lift verification step, then require
fresh post-action evidence before any transport/place milestone.

#### YAM Lesson: White Bowl Geometry Needs Explicit Prompt And Radius Scaling

In the current white-bowl scene, generic `bowl` detections can refer to a
different bowl. Prefer explicit prompt rank (`white bowl` before `bowl`) and
reject rim-Z evidence whose XY disagrees with the selected top-camera source.
The validated white-bowl pickup/lift used a top-box-scaled radius estimate:
selected top bbox long axis `73 px` scaled from the orange-bowl reference
`56 px / 0.095 m`, giving radius `0.12384 m`.
The later after-analysis validation
`logs/place_white_bowl_on_rack_one_shot_loop_after_analysis_20260530T211151`
used `OPENFORGE_BOWL_TOP_DOWN_TCP_ABOVE_RIM_M=-0.040`, close target `0.04`,
and verified a lift before returning the bowl to the table. Treat this as a
white-bowl pickup/lift setting, not as rack-placement validation.

Keep bottom-camera artifacts image-only unless calibration/policy changes.
Bottom overlays can help human reasoning, but bottom detections/poses must not
drive YAM bowl motion, candidate ranking, close width, or success verification.

#### YAM Rejected Lesson: Display RPY Offset Broke Top-Down Assumption

For the orange-bowl rack attempt, the selected candidate used display RPY
`[-90, -180, -90]` from an `[-90, 0, 0]` RPY offset while the plan artifact
still described the model as "local Z points downward." Runtime robot state
showed the resulting EE local Z was effectively horizontal, not downward, so
the TCP-to-fingertip offset was applied under the wrong wrist geometry. The
visible fingers remained above/offset from the bowl wall and the close happened
in free space.

Future bowl planning should artifact the world-frame EE local axes after any
display-RPY offset and reject candidates whose approach/contact axis disagrees
with the named strategy. If attempting a true top-down rim straddle, avoid
ranking the `[-90,0,0]` display-RPY offset ahead of the base top-down pose
unless visual pre-close evidence proves the fingertips straddle the wall.

#### YAM Milestone: Top-Down Axis Validated, Rim Straddle Still Not Proven

`logs/place_orange_bowl_on_rack_one_shot_loop_20260530T000549` validates the
current right-arm top-down plumbing: selected `top_down_bowl_rim_straddle`,
display RPY `[0,-180,45]`, robot-state local Z nearly `[0,0,-1]`, opening axis
aligned across the rim, and pre-close EE within about `6 mm` of the commanded
grasp pose. This is no longer the old horizontal/side fallback failure.

The same run still rejected the pickup before lift because the gripper closed
from `0.683` to `0.252` for target `0.25`; the final-target delta was only
`0.002`, below the `0.010` contact gate. Right-camera video showed a blue
fingertip entering/overlapping the bowl interior, but that is not sufficient
evidence that one finger is inside and the other is outside the rim wall.

For the next physical retry, keep the top-down axis gate and contact-before-lift
gate. Improve the pre-close check so it verifies a true inside/outside rim-wall
straddle or run a bounded top-down gripper-gap/contact calibration for the exact
wrist pose. Do not treat "finger visible inside the bowl" as enough to lift.
