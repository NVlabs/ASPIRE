---
name: yam-robot-debugging
description: "Use for YAM real-robot debugging: saved-script edits, physical runs, artifact inspection, autonomous patch-and-retry loops, and compact runtime context updates."
---

# YAM Robot Debugging

Work from the Aspire real-robot workspace (`aspire/real`). Read its `AGENTS.md`
contract first; paths below are relative to that directory.

Act as a robotics debugging engineer. For a user-requested physical task, run
the saved script, inspect evidence, patch, and retry bounded attempts until
success, blocker, or operator stop.

Hard rule: motion-capable commands must set
`OPENFORGE_ALLOW_PHYSICAL_MOTION=1`.

Loop:

1. Read the wrapper/script and recent logs.
2. Check needed servers (`yam-server-setup`).
3. Run syntax/no-motion checks when useful.
4. Run the physical script.
5. Inspect `logs/<run>/` (`yam-runtime-artifacts`).
6. Patch the smallest useful code/parameter change, or switch strategy if the
   evidence points to a better motion/perception logic.
7. Retry.

Reset the scene yourself when it is safe and simple. Pause and ask the human to
reset only when the task scene is physically broken in a way the robot cannot
fix safely, such as a target bottle falling down during a grasp attempt.

Promote reusable lessons after a verified success or clear rejection:

- Geometry, frames, camera/table/gripper facts -> `yam-geometry`.
- Demo wrappers and golden commands -> `yam-full-demo`.
- Log/artifact structure and failure signatures -> `yam-runtime-artifacts`.
- Server checks/restarts -> `yam-server-setup`.
- Pickup/grasp tactics -> `yam-grasp-pickup`.
- Held-object transport tactics -> `yam-transport`.
- Retreat, release, and recovery tactics -> `yam-retreat-recovery`.
- cuRobo sensitivity and waypoint sweeps -> `yam-motion-planner`.

Keep each lesson compact and cite the run path/date. Do not put one-off object
XYZs, long narratives, or unverified guesses in skills.

Avoid ticket/approval env clutter. Use evidence gates from the current run.
Prefer edits in `cap/saved_scripts/**`, `skill_library/**`, and
`yam_runtime/**`.
