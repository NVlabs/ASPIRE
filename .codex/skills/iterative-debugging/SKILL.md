---
name: iterative-debugging
description: "Run the baseline-free CaP iterative debugging (fix loop) pipeline on a user-supplied list of benchmark tasks: inspect one initial observed scene per task, generate task-level code, debug failures on development seeds using recorded run artifacts, validate on the held-out partition, grade the skill library against what actually worked, and promote only those patterns a verifier could reproduce the fix from."
---

# Iterative Debugging (Fix Loop)

**This file is a pointer, not a copy.** The pipeline lives in
`.claude/skills/iterative-debugging/SKILL.md`, and that file is canonical under every agent
harness including this one. Read it now and follow it as written: setup, run order, prompts,
scripts and seed partitions all apply here unchanged.

Nothing about the pipeline is restated below. An earlier version of this file summarised it, and
the summary drifted out of step with the canonical version. That drift is the failure this
pointer exists to prevent. If you want to add pipeline detail, add it there, not here.

## What lives where

| Path | What it is |
|---|---|
| `.claude/skills/iterative-debugging/SKILL.md` | the pipeline, canonical |
| `.claude/skills/iterative-debugging/main-agent-prompt.md` | the coordinator's guide, including how a campaign is scored |
| `.claude/skills/iterative-debugging/subagent-prompt.md` | the per-task worker template |
| `.claude/skills/iterative-debugging/skills/` | accumulated topic knowledge |
| `.claude/skills/iterative-debugging/scripts/` | campaign scripts, shared by both harnesses |
| `agents/openai.yaml` | this harness's own manifest, which has no counterpart under `.claude` |

## The one thing the canonical file cannot tell you from its title

BEHAVIOR-1K campaigns (R1 Pro on Isaac Sim) use the sibling folder
`.claude/skills/iterative-debugging/behavior/SKILL.md`. It is a **different protocol**, not
different prompts for the same one: each seed grows its own policy one code block at a time, the
campaign freezes a skill library rather than a program, and scoring is per final policy rather
than per attempt. Do not carry the tabletop seed partitions or the tabletop scoring across to it.
