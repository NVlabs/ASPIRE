# BEHAVIOR-1K Clean Task Slate

This is the general fix-loop checklist. The canonical campaign adds stricter
stage and seed isolation in
[`../aspire-protocol/clean-task-slate.md`](../aspire-protocol/clean-task-slate.md).

Before rerunning a BEHAVIOR config:

- Re-read `.claude/behavior/skills/system-pipeline.md` sections 3, 7, and 8
  for launch mode, supported config, and perception server expectations.
- Confirm the current policy file under `outputs/interactive/` contains only
  the intended appended blocks.
- Confirm the B1K virtual environment is active.
- Set `OMNI_KIT_ACCEPT_EULA=YES`.
- Set `OMNIGIBSON_HEADLESS=1` on headless machines.
- Set `ulimit -c 0` before starting Isaac Sim on eval nodes.
- Confirm no stale Isaac Sim process is still running.
- Confirm SAM3 and ContactGraspNet ports are free or intentionally reused.
- For interactive-policy runs, expect SAM3 on `8114` and ContactGraspNet on
  `8115`; if a YAML uses different ports, verify that this is intentional
  before launch.
- Confirm replay commands record video; use the bare `--record-video` flag.
- Use a new output directory for every attempt. Never move, rename, delete, or
  overwrite an earlier attempt to make a rerun appear clean.
- Keep videos and saved observations from failed trials until analysis is done.
- Keep `trace.json`, `keyframes/`, `differencing_feedback_*.txt`, and
  `prompts_and_responses/` until the failure has been classified.
- If disk is low after Isaac Sim crashes, remove only known crash dumps such as
  `core.*`; do not delete trial outputs before extracting diagnostics.
