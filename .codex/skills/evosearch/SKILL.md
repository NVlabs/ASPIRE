---
name: evosearch
description: "Run the CaP Fix Loop + Evolutionary Search experiment on a user-supplied list of low-performing benchmark tasks: iterative K=8 candidate search on development seeds 51–65, validation selection, and final held-out evaluation on seeds 1–50."
---

# Fix Loop + Evolutionary Search

**This file is a pointer, not a copy.** The pipeline lives in `.claude/skills/evosearch/SKILL.md`,
and that file is canonical under every agent harness including this one. Read it now and follow
it as written: setup, run order, prompts, scripts and seed partitions all apply here unchanged.

Nothing about the pipeline is restated below. An earlier version of this file summarised it, and
the summary drifted out of step with the canonical version. That drift is the failure this
pointer exists to prevent. If you want to add pipeline detail, add it there, not here.

## What lives where

| Path | What it is |
|---|---|
| `.claude/skills/evosearch/SKILL.md` | the pipeline, canonical |
| `.claude/skills/evosearch/main-agent-prompt.md` | the coordinator's guide |
| `.claude/skills/evosearch/subagent-prompt.md` | the per-task worker template |
| `.claude/skills/evosearch/skills/` | accumulated topic knowledge |
| `.claude/skills/evosearch/scripts/` | candidate evaluation and trace analysis |
| `agents/openai.yaml` | this harness's own manifest, which has no counterpart under `.claude` |

This skill builds on `iterative-debugging` and shares its held-out evaluation script at
`.claude/skills/iterative-debugging/scripts/run_validation.py`.
