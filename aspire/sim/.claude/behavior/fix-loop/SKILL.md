---
name: behavior/fix-loop
description: Complete BEHAVIOR-1K ASPIRE protocol: learn skills on seeds 26-35, freeze them, then run isolated fresh-agent adaptation on seeds 1-25.
---

# BEHAVIOR-1K ASPIRE Fix Loop

Use this skill for:

```text
Follow the protocol and run BEHAVIOR-1K Soda Can ASPIRE experiments.
```

```text
Follow the protocol and run BEHAVIOR-1K Radio ASPIRE experiments.
```

From `aspire/sim`, read:

1. `../../AGENTS.md`
2. `CLAUDE.md`
3. `.claude/behavior/CLAUDE.md`
4. `.claude/behavior/api-reference.md`
5. `.claude/behavior/skills/system-pipeline.md`
6. `.claude/behavior/fix-loop/INSTRUCTIONS.md`
7. `.claude/behavior/fix-loop/clean-task-slate.md`

Then follow `.claude/behavior/fix-loop/main-agent-prompt.md`.

The protocol is fixed:

1. Preflight and wait for approval.
2. Learn a campaign-owned skill library on seeds 26-35.
3. Freeze skills and the experimental contract—not a policy.
4. Run seeds 1-25 with a fresh context and empty policy per seed, allowing
   debugging only within that seed.
5. Preserve and report all outcomes.

Every trial uses replay mode, which calls `env.step` directly without an
internal model call. Do not delete the shared built-in multi-turn engine; other
experiments use it. Do not launch before preflight approval.
