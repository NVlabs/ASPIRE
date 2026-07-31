# BEHAVIOR-1K Fix-Loop Clean Slate

Before a campaign:

- Confirm approved task/config, commit, model, budgets, GPU, and fresh path.
- Activate `cap/third_party/b1k/.venv`; never run `uv sync` in it.
- Set `OMNI_KIT_ACCEPT_EULA=YES`, `OMNIGIBSON_HEADLESS=1`, and `ulimit -c 0`.
- Confirm perception ports and that no Isaac Sim process is alive.
- Copy repository skills to the campaign working library.

Before Stage 1 seeds:

- Confirm the seed is 26-35 and no held-out artifact was accessed.
- Use the seed's policy and a new attempt directory; preserve older attempts.
- Permit changes only to `skill-library-working/`.

Before Stage 2:

- Verify seeds 26-35 are terminal and frozen hashes match.
- Confirm no Stage 1 policy becomes an evaluation policy.
- For each seed 1-25, terminate the previous agent/simulator, create an empty
  policy, and start a non-resumed context with no inherited transcript.
- Expose only frozen inputs and that seed's directory.

If any check fails, stop and record it. Never create apparent cleanliness by
deleting, moving, renaming, or overwriting evidence.
