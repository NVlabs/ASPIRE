# Claude Suite Registry

This directory is organized by simulator suite. Each suite owns its setup conventions, API reference, baseline collection guide, experiment runbooks, and suite-specific skill library.

## Index

| Suite | Experiment | Folder | What it measures |
|---|---|---|---|
| Robosuite | Fix Loop | [robosuite/fix-loop/](robosuite/fix-loop/) | Baseline -> iterative fix loop -> eval success rate on seeds 1-100 across seven tasks |
| Robosuite | Training Law | [robosuite/training-law/](robosuite/training-law/) | Cumulative tokens vs. success rate across fix-loop iterations |
| LIBERO | Fix Loop | [libero/fix-loop/](libero/fix-loop/) | LIBERO-Pro fix loop with debug and held-out eval seeds |
| LIBERO | Fix Loop + Evolutionary Search | [libero/evosearch/](libero/evosearch/) | Candidate search, validation selection, and final eval on LIBERO-Pro |
| LIBERO | Zero-Shot Transfer | [libero/zeroshot-transfer/](libero/zeroshot-transfer/) | LIBERO-90 skill-library build and LIBERO-Long-Pro transfer handoff |
| LIBERO | Library-Size Scaling | [libero/library-size-scaling/](libero/library-size-scaling/) | Frozen snapshot evals and scaling tables/plots |
| LIBERO | Inference-Time Scaling | [libero/inference-time-scaling/](libero/inference-time-scaling/) | Debug-compute/token-budget scaling on LIBERO-Long-Pro |
| BEHAVIOR-1K | ASPIRE Fix Loop | [behavior/fix-loop/](behavior/fix-loop/) | Learn skills on seeds 26-35, then run isolated per-seed adaptation on seeds 1-25 |

## Suite Entrypoints

| Suite | Entry point | Shared reference |
|---|---|---|
| Robosuite | [robosuite/CLAUDE.md](robosuite/CLAUDE.md) | [robosuite/api-reference.md](robosuite/api-reference.md) |
| LIBERO | [libero/CLAUDE.md](libero/CLAUDE.md) | [libero/api-reference.md](libero/api-reference.md) |
| BEHAVIOR-1K | [behavior/CLAUDE.md](behavior/CLAUDE.md) | [behavior/api-reference.md](behavior/api-reference.md) |

## Layout Convention

```text
.claude/
  README.md
  <suite>/
    CLAUDE.md
    api-reference.md
    run-baseline.md
    skills/
    <experiment>/
      INSTRUCTIONS.md
      SKILL.md
      main-agent-prompt.md
      <worker-or-stage>-prompt.md
      clean-task-slate.md
      skills/
```

All experiments under a suite share that suite's `CLAUDE.md`, `api-reference.md`, and `run-baseline.md`. On `master`, suite-level and experiment-level `skills/` files are clean templates/placeholders unless explicitly marked as API or pipeline reference material. Frozen learned skill snapshots live on the `learned-skills` branch under the same suite/experiment layout.

## Prerequisites

All experiments assume the base setup from the root [README.md](../README.md). Simulator environments are suite-specific and incompatible, so complete only the continuation setup for the suite you plan to run.
