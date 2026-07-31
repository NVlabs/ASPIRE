# Canonical BEHAVIOR-1K Campaign Clean Slate

Apply this checklist before campaign creation and before every seed agent.

## Campaign Creation

- Confirm preflight was explicitly approved.
- Record the exact repository SHA and verify the checkout has no unrecorded
  protocol/config changes.
- Confirm the selected task/config and fixed budgets match the approval.
- Confirm the new campaign path does not exist. Never recycle a previous path.
- Copy repository skill templates into the campaign working library.
- Record environment, asset, perception-port, and GPU status.
- Set `OMNI_KIT_ACCEPT_EULA=YES`, `OMNIGIBSON_HEADLESS=1`, and `ulimit -c 0`.
- Activate `cap/third_party/b1k/.venv`; never run `uv sync` in it.
- Confirm no stale Isaac Sim process is alive and only one simulator will run.

## Before A Stage 1 Seed

- Confirm the seed is in 26-35 and no held-out artifact has been accessed.
- Use that seed's own policy and a new numbered attempt directory.
- Preserve all earlier attempts and observations.
- Allow updates only to `skill-library-working/`.

## Freeze Boundary

- Confirm seeds 26-35 are all terminal.
- Stop the Stage 1 agent and simulator.
- Copy and checksum the frozen skill library and experimental contract.
- Make the frozen copy read-only and verify its manifest.
- Confirm no Stage 1 policy is selected as an evaluation starting point.

## Before Every Stage 2 Seed

- Confirm the seed is the next incomplete member of 1-25.
- Confirm the frozen manifest still verifies.
- Confirm the prior seed agent/context has terminated.
- Confirm no Isaac Sim process remains from the prior seed.
- Create a new isolated seed directory and fresh empty `policy.py`.
- Launch a fresh non-resumed agent/context with no inherited experiment
  transcript, using the immutable seed prompt and fixed budget.
- Expose only the frozen library, fixed docs/config, and that seed's directory.
- Use a new numbered output directory for every replay.

If a boundary check fails, stop and record it in `campaign-state.md`. Never
repair apparent cleanliness by deleting, moving, renaming, or overwriting
evidence.
