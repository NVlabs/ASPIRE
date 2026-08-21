# YAM Robot Agent Contract

Role: robotics debugging engineer. This repo uses code as policy: inspect code,
run the robot when asked, inspect artifacts, patch, retry.

## Workspace

- This real-robot workspace lives at `aspire/real` in the Aspire repository.
- From the top-level checkout, run `cd aspire/real` before using commands in
  this file or its skills.
- Skill paths, `cap/**`, `robot/**`, `tools/**`, and `logs/**` are relative to
  `aspire/real`.
- The simulation workspace is the sibling `../sim`; borrow from it only through
  the `yam-simulation-transfer` guidance.

## Skills

- Skills are stored under `.agents/skills/` in this workspace.
- `yam-robot-debugging`: default real-robot run/inspect/fix loop.
- `yam-full-demo`: current `yam_demo.sh` flow and saved-script commands.
- `yam-server-setup`: arm/perception/planner server checks and launches.
- `yam-geometry`: DOF, frames, table/rack/bin/camera/gripper facts.
- `yam-runtime-artifacts`: `logs/<run>/` videos, overlays, plans, JSON.
- `yam-motion-planner`: cuRobo failures, waypoint previews, robustness sweeps.
- `yam-grasp-pickup`: detection, grasp candidates, staged close, lift checks.
- `yam-transport`: held-object approach/transfer/rack/bin motion patterns.
- `yam-retreat-recovery`: post-release retreat, home/open, held-object recovery.
- `yam-simulation-transfer`: LIBERO/robosuite ideas, not real coordinates.

## Rules

- For user-requested physical tasks, act autonomously: run bounded attempts,
  inspect evidence, patch, and retry until success, blocker, or operator stop.
- Do not add approval/ticket gate clutter.
- Real motion commands must set `OPENFORGE_ALLOW_PHYSICAL_MOTION=1`.
- Reset simple scene issues yourself; ask the human only when the physical scene
  is unsafe or impossible for the robot to reset, such as a fallen target.
- Prefer `cap/saved_scripts/**`; keep edits narrow.
- When a run teaches a reusable lesson, promote it into the smallest relevant
  skill with evidence path/date. Do not store one-off coordinates or long run
  narratives in skills.
- Keep active context compact. Put reusable geometry in `yam-geometry`
  references and long history in logs or archive only when useful.

## Evidence

Live runs should record videos/debug artifacts unless the failure is exactly
about recording. Success needs perception, robot state, video/overlay, or result
artifact evidence; a command returning is not enough.

## Checks

Use `rg`, `apply_patch`, `bash -n`, and `python3 -m py_compile`/`uv run python
-m py_compile` as appropriate. Do not revert user changes.
