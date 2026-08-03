# ASPIRE Agent Guide

This is the model-neutral entry point for repository-aware coding agents. Read it before installing dependencies, starting services, launching experiments, editing code, or accessing runtime artifacts.

## Request Routing

### Canonical LIBERO-Pro Quick Start

The canonical request is:

```text
Run the complete ASPIRE LIBERO-Pro Goal-Swap Quick Start for all ten tasks
in the libero_goal_swap suite.
```

For this request, read and follow:

1. [`aspire/sim/README.md`](aspire/sim/README.md)
2. [`aspire/sim/CLAUDE.md`](aspire/sim/CLAUDE.md)
3. [`aspire/sim/.claude/libero/CLAUDE.md`](aspire/sim/.claude/libero/CLAUDE.md)
4. [`aspire/sim/.claude/libero/fix-loop/QUICKSTART.md`](aspire/sim/.claude/libero/fix-loop/QUICKSTART.md)

Before any setup, service start, subagent dispatch, replay, or evaluation, provide a preflight report covering the host, GPU mapping, credentials and gated weights, required services, seed partitions, expected runtime, and output paths. Wait for explicit confirmation before continuing.

### Canonical BEHAVIOR-1K ASPIRE Protocol

Canonical requests are:

```text
Follow the protocol and run BEHAVIOR-1K Soda Can ASPIRE experiments.
```

```text
Follow the protocol and run BEHAVIOR-1K Radio ASPIRE experiments.
```

For either request, read and follow:

1. [`aspire/sim/README.md`](aspire/sim/README.md)
2. [`aspire/sim/CLAUDE.md`](aspire/sim/CLAUDE.md)
3. [`aspire/sim/.claude/behavior/CLAUDE.md`](aspire/sim/.claude/behavior/CLAUDE.md)
4. [`aspire/sim/.claude/behavior/fix-loop/SKILL.md`](aspire/sim/.claude/behavior/fix-loop/SKILL.md)
5. [`aspire/sim/.claude/behavior/fix-loop/INSTRUCTIONS.md`](aspire/sim/.claude/behavior/fix-loop/INSTRUCTIONS.md)

The named task resolves the task choice, but it does not waive preflight. Before
installing or changing dependencies, starting services, dispatching agents, or
running a trial, report the protocol commit, host and GPU, environment status,
model, fixed per-seed budgets, seed partitions, expected runtime, and fresh
campaign output path. Wait for explicit confirmation. After confirmation, the
coordinator may execute the complete protocol autonomously and resume it from
its campaign state file.

### Other simulation experiments

Use `aspire/sim` as the working root. Read [`aspire/sim/.claude/README.md`](aspire/sim/.claude/README.md), the selected suite constitution, and the experiment's `INSTRUCTIONS.md` and `SKILL.md`.

If the user has not named a suite and experiment, list the experiment choices from the registry and stop for selection. Never silently choose an experiment, task, model, seed range, or evaluation protocol. Before a paper-scale run, report its scope, expected trial count and runtime, compute, credentials, services, and outputs, then wait for confirmation.

### Real-robot work

Use `aspire/real` as the working root and read [`aspire/real/AGENTS.md`](aspire/real/AGENTS.md). A simulation or documentation request never authorizes starting robot services, opening cameras, contacting follower processes, enabling motion, or accessing physical hardware.

## Repository Rules

- Do not push to any remote unless the user explicitly requests it.
- Do not delete, overwrite, or mix existing experiment outputs without explicit confirmation and the applicable clean-slate procedure.
- Preserve development and held-out seed boundaries exactly.
- Do not substitute external baseline code or outputs when a runbook forbids them.
- Keep credentials in approved environment variables or protected files and out of generated-code processes, logs, prompts, YAML, Markdown, and committed files.
- Treat generated Python as untrusted. Trial isolation and watchdogs are reliability mechanisms, not a hardened security sandbox.
- Do not expose simulator ground truth or other forbidden APIs to generated programs. The selected suite constitution is authoritative.
- Report commands, outputs, artifact paths, blockers, and deviations precisely. Never claim an experiment completed when required manifests or trials are missing.

## Workspace Boundaries

- Project overview and navigation: repository root
- Simulation setup and execution: `aspire/sim`
- Real-robot setup and execution: `aspire/real`

Keep simulator dependencies, coordinates, APIs, and artifacts separate from physical-robot workflows.
