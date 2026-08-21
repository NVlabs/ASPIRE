---
name: yam-full-demo
description: "Use for the current YAM full demo, yam_demo.sh, bottle rack, KitKat handover/trash, bowl/can commands, and canonical saved-script commands."
---

# YAM Full Demo

Run this skill from the Aspire real-robot workspace (`aspire/real`). Paths and
commands below are relative to that directory.

Canonical commands:

```bash
bash tools/yam_demo_preflight.sh
bash tmux/launch_yam_demo_services.sh --no-attach
bash tools/yam_demo_preflight.sh --services
bash cap/saved_scripts/yam_demo.sh full
bash cap/saved_scripts/yam_demo.sh bowls
bash cap/saved_scripts/yam_demo.sh bottle-rack
bash cap/saved_scripts/yam_demo.sh kitkat
bash cap/saved_scripts/yam_demo.sh drawer
bash cap/saved_scripts/yam_demo.sh drawer-close
bash cap/saved_scripts/yam_demo.sh drawer-candy
bash cap/saved_scripts/yam_demo.sh home
```

The demo launcher starts only both follower arms, the camera Portal, SAM3, and
BundleSDF. Do not require AnyGrasp, cuRobo, PyRoki, or provider credentials for
these saved-script commands.

Flow:

- `yam_demo.sh` is the preferred dispatcher for `full`, `bowls`,
  `white-dish`, `orange-on-white`, `can-trash`, `kitkat`, `bottle-rack`,
  `drawer`, `drawer-close`, `drawer-candy`, and `home`.
- `yam_demo.sh full` -> drawer-close/home, white-dish, orange-on-white,
  can-trash, KitKat, bottle-rack

Speed:

- Use `YAM_FULL_DEMO_SPEED_SCALE` as the operator-facing full-demo speed knob.
  `1.0` preserves validated defaults, values above `1.0` speed up planned
  moves and shorten direct step durations, values below `1.0` slow them down.
- The scale feeds home, drawer planning, rack dish/bowl planning and direct
  rack timing, can/trash planning, KitKat planning, and bottle-rack planning.
- Task-specific low-level `OPENFORGE_*_PLANNING_SPEED`, `*_STEP_S`,
  `*_PLAYBACK_SPEED`, and `OPENFORGE_OPEN_HOME_SPEED_RAD_S` env overrides are
  still available for focused debugging.

Top-level `cap/saved_scripts/yam_demo.sh` is the operator shell entrypoint.
Helper shells live under `cap/saved_scripts/shell_scripts/`; active Python task
implementations live at top level. Older probes, sweeps, alternates, and
retired shell flows live under `cap/saved_scripts/legacy_codes/`.
