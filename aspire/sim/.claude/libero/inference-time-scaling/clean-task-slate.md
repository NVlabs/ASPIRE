# Clean Task Slate: Inference-Time Scaling

Use this before rerunning a snapshot, task, or token-budget checkpoint.

1. Confirm snapshot, suite, task, and token-budget checkpoint.
2. Decide whether existing Stage 1 code versions, token traces, and Stage 2 outputs should be archived, reused, or superseded.
3. Check target GPU availability before dispatching debug subagents or seed runners.
4. Verify perception servers on ports 8114-8116 or let the coordinator preflight start them.
5. Keep Stage 1 debug seeds separate from Stage 2 held-out seeds.
6. Keep frozen snapshot skill libraries read-only during debug and eval.
