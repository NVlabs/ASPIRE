# BEHAVIOR-1K Patches

Verified patches against the pinned `cap/third_party/b1k` submodule, for cases
where a fix cannot be applied from the ASPIRE side.

**This directory is currently empty, and that is deliberate.** Every deviation
needed to install BEHAVIOR-1K on a clean host is applied externally by
[`scripts/setup_behavior.sh`](../../scripts/setup_behavior.sh) — creating the
venv, pinning torch before the upstream installer resolves it, installing SAM3
from the path the upstream script fails to find, and building Contact-GraspNet
without build isolation. None of that requires editing B1K sources, so the
submodule stays clean and `git status` in it stays empty.

## When a patch is genuinely required

Prefer these options, in order:

1. Fix it in ASPIRE code or in `scripts/setup_behavior.sh`.
2. Move the submodule to an upstream commit that already contains the fix.
3. Only then add a patch here.

A patch that lives only as an uncommitted edit inside the nested repository is
not reproducible: it is invisible to `git status` at the top level, it is lost
on `git submodule update`, and nobody else can reproduce the environment from a
clean clone.

## Conventions

Follow [`scripts/common/apply_contact_graspnet_patch.sh`](../../scripts/common/apply_contact_graspnet_patch.sh),
which is the working example of the pattern:

- Pin the exact expected submodule revision and refuse to apply to any other.
- Verify the patched files byte-for-byte against the tested result.
- Be idempotent — re-check rather than re-apply, so setup can be safely re-run.
- Fail loudly on any unexpected local modification instead of stacking edits.

Name patches `<component>-<purpose>.patch` and add the apply/verify step to
`scripts/setup_behavior.sh` so a single setup command stays sufficient.
