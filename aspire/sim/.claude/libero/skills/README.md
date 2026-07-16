# LIBERO Shared Robot Skills

These files contain reusable, observation-only robot-control knowledge shared across LIBERO experiments. Pipeline orchestration belongs in experiment directories; measurement procedures belong in [../analysis/](../analysis/).

| Skill | Purpose |
|---|---|
| [grasp.md](grasp.md) | Grasp selection, orientations, gripper evidence, and pick/place skeletons |
| [localize.md](localize.md) | SAM3 prompting, disambiguation, and 3D localization helpers |
| [transport.md](transport.md) | Waypoints, collision-free transit, and placement approaches |
| [manipulation.md](manipulation.md) | Drawers, knobs, switches, pushing, and other contact tasks |

Promoted findings should include provenance: source suite/task, development seeds, held-out result when available, source code path, and date. Avoid universal claims from one scene or seed.
