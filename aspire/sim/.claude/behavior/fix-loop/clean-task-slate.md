# BEHAVIOR-1K Clean Task Slate

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
- Confirm replay commands include `--record-video True`.
- Move or rename prior output directories if you need a fresh run.
- Keep videos and saved observations from failed trials until analysis is done.
- Keep `trace.json`, `keyframes/`, `differencing_feedback_*.txt`, and
  `prompts_and_responses/` until the failure has been classified.
- If disk is low after Isaac Sim crashes, remove only known crash dumps such as
  `core.*`; do not delete trial outputs before extracting diagnostics.
