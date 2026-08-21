# Top-Level Agent Guide

> **Claude Code:** Read and follow [`AGENTS.md`](AGENTS.md) first. It is the
> repository-wide, model-neutral request-routing and safety contract. This file
> adds workspace guidance for Claude Code; `AGENTS.md` governs when the two
> overlap.

This repository is split into a lightweight project root, a simulation
workspace, and a real-robot workspace.

Use `aspire/sim` as the working root for simulation setup, experiments, configs, scripts, tests, and output inspection. The simulation package lives under `aspire/sim/cap` and imports as `aspire.sim.cap.*`.

For simulation work:

```bash
cd aspire/sim
export ASPIRE_ROOT="$(pwd)"
export PYTHON_ROOT="$(cd ../.. && pwd)"
```

Read `aspire/sim/CLAUDE.md` and `aspire/sim/.claude/README.md` before running suite-specific agents. Keep simulator setup details in `aspire/sim/README.md`; the top-level `README.md` should stay a project overview and pointer map.

For real-robot work:

```bash
cd aspire/real
```

Read `aspire/real/AGENTS.md` before running or editing YAM workflows. Its skills
live under `aspire/real/.agents/skills/`; paths and commands in those skills are
relative to `aspire/real`. Keep real-station procedures separate from the
simulation docs.

## Generated-Code Security Boundary

ASPIRE executes language-model-generated Python with full import access. Trial
processes, timeouts, and simulator watchdogs are not a hardened security
sandbox. Treat generated code as untrusted: keep credentials out of worker
environments, use an isolated host or container without sensitive mounts,
restrict network access, and never give generated code physical-hardware access
without review and the operator controls in `aspire/real/AGENTS.md`.
