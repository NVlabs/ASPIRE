# Clean Task Slate: Library-Size Scaling

Use this before rerunning a snapshot, suite, or task.

1. Confirm the snapshot tag and worktree path.
2. Decide whether existing `outputs/scaling_eval/<snapshot>/` artifacts should be archived, reused, or superseded.
3. Check that Phase 1 `code.py` outputs and Phase 2 trial outputs agree with the intended rerun scope.
4. Verify perception servers on ports 8114-8116 or let the coordinator preflight start them.
5. Keep snapshot worktrees read-only except for expected eval outputs outside the worktree.
6. Do not mix generated code from one snapshot with another snapshot's skill library.
