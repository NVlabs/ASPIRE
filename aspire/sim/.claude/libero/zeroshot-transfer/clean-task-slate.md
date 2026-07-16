# Clean Task Slate: Zero-Shot Transfer

Use this before rerunning a LIBERO-90 build task or chunk.

1. Confirm the chunk number, task list, and snapshot tag target.
2. Inspect current build outputs and decide whether they should be archived, reused, or superseded.
3. Check target GPU availability before dispatching build subagents.
4. Verify perception servers on ports 8114-8116 or let the coordinator preflight start them.
5. Keep build/debug seeds 51-80 separate from held-out eval seeds.
6. Only the coordinator updates [../skills/](../skills/) after reviewing subagent `findings.md`.
7. Do not move or retag existing snapshot tags unless the maintainer explicitly asks.
