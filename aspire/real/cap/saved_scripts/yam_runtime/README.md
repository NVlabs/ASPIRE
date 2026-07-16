# YAM Runtime Helpers

Compact helper layer for `run_script.py` saved scripts.  It wraps the raw
injected APIs without importing `skill_library.*`.

Use this for new one-command YAM scripts:

```python
from cap.saved_scripts.yam_runtime import (
    capture_scene,
    generate_side_grasp_candidates,
    rank_motion_candidates,
    execute_grasp_lift_attempt,
    verify_lift,
)
```

Default flow:

1. `capture_scene(...)` for live top-camera detections, overlays, and robot state.
2. `generate_side_grasp_candidates(...)` from the selected live detection.
3. `rank_motion_candidates(...)` with `preview_only=True` planning.
4. `execute_grasp_lift_attempt(...)` only when the script's physical gate and
   task ticket permit motion.
5. `capture_scene(...)` again and `verify_lift(...)`.

Manual object XYZ overrides are for no-motion calibration/debug only.  Physical
scripts must plan from fresh live observations.
