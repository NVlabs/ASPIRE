#!/usr/bin/env bash
# Set up a detached git worktree for a frozen snapshot eval.
#
# Usage:
#   scripts/libero/eval_setup_worktree.sh --snapshot snapshot-N50
#
# Creates (idempotent):
#   outputs/worktrees/<snapshot>/   — full repository checkout at the snapshot tag
#
# After this runs, the coordinator dispatches eval subagents that point
# ASPIRE_ROOT_SNAPSHOT at outputs/worktrees/<snapshot>/aspire/sim. See
# .claude/libero/library-size-scaling/SKILL.md.
#
# NEVER pushes to remote. NEVER modifies the snapshot tag.

set -euo pipefail

SNAPSHOT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --snapshot) SNAPSHOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$SNAPSHOT" ]] && { echo "ERROR: --snapshot required" >&2; exit 1; }

ASPIRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_ROOT="$(cd "$ASPIRE_ROOT/../.." && pwd)"
WORKTREE_REPO="$ASPIRE_ROOT/outputs/worktrees/$SNAPSHOT"
WORKTREE="$WORKTREE_REPO/aspire/sim"

# Verify tag exists
if ! git -C "$GIT_ROOT" rev-parse "refs/tags/$SNAPSHOT" >/dev/null 2>&1; then
    echo "ERROR: tag '$SNAPSHOT' does not exist" >&2
    exit 1
fi

if [[ -d "$WORKTREE_REPO" ]]; then
    echo "[worktree] $WORKTREE_REPO already exists — reusing"
else
    echo "[worktree] Creating $WORKTREE_REPO at $SNAPSHOT"
    git -C "$GIT_ROOT" worktree add --detach "$WORKTREE_REPO" "$SNAPSHOT"
    echo "[worktree] Done: $WORKTREE_REPO"
fi

if [[ ! -d "$WORKTREE" ]]; then
    echo "ERROR: expected workspace not found: $WORKTREE" >&2
    exit 1
fi

echo "ASPIRE_ROOT_SNAPSHOT=$WORKTREE"
