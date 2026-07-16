# ASPIRE Project Memory

> ASPIRE
> Built on top of Cap-Gym. GEAR Lab collaboration.

---

## What We're Building

A **Self-Improving Coding Agent** for robot manipulation. The agent debugs its own failures, accumulates reusable skills, and improves its perception tools — the same way a human engineer would.

**Four layers:**
- **Foundation** — Rich debugging playground: per-trial traces, interactive REPL, failure diagnosis
- **Layer 1** — Auto-growing skill library: agent packages solutions as reusable skills, grows through experience
- **Layer 2** — Agent finetunes tools: when SAM3/GraspNet/Molmo systematically fails, agent finetunes from sim data
- **Layer 3** — Program search: Pass@K test-time scaling to find best program given current skills + tools

**Failure diagnosis loop:**
- Code logic wrong → refine skill library (Layer 1) or search more programs (Layer 3)
- Perception tool wrong → finetune the tool (Layer 2)

---

## Key Conventions (Robosuite)

- **Never use `uv run` or `uv sync`** on shared nodes — destroys manual pip installs in the shared venv. Use `.venv/bin/python3` directly.
- `replay_trial_robosuite.py` uses the **`--args.` prefix** (`--args.config`, `--args.trial`, `--args.replay-code`, `--args.output-dir`, `--args.interactive`)
- Trial seed = trial number: `env.reset(seed=5)` always gives the same initial state (deterministic). Seeds 101–125 = debug set; 1–100 = eval set.
- **Robosuite experiment prompts, skills, and run instructions live under `.claude/robosuite/`** — see [`.claude/README.md`](../README.md) for the index, and [`.claude/robosuite/CLAUDE.md`](../robosuite/CLAUDE.md) for the suite constitution.
- Logs go to `./docs/logs/YYYY-MM-DD.md` — append, don't overwrite
- **NEVER git commit or push unless the user explicitly asks**
- **NEVER read sim asset files** (`.xml`, `.urdf`, MuJoCo models) — diagnose from observations and traces only
- **Never use forbidden ground-truth APIs** (`sim.data.body_xpos`, `sim.data.set_joint_qpos`, `sim.forward()`, etc.) — full list in `.claude/robosuite/CLAUDE.md`
- **Always write reusable analysis as scripts** (not inline `python3 -c "..."`) and reference them in the relevant skill

---

## Robosuite Experiments

Organized by experiment under `.claude/robosuite/` (full index: [`.claude/README.md`](../README.md)):

| Experiment | Folder | What it is |
|---|---|---|
| Fix-loop eval | `.claude/robosuite/fix-loop/` | Baseline → iterative fix loop → eval success rate (seeds 1–100), 7 tasks |
| Training scaling law | `.claude/robosuite/training-law/` | Cumulative tokens vs. success rate across fix-loop iterations |

Each experiment folder holds its own `SKILL.md`, `main-agent-prompt.md`, `subagent-prompt.md`, `clean-task-slate.md`, and `skills/` (grasp, localize, transport, …). Suite-shared references: `.claude/robosuite/api-reference.md` (control API) and `.claude/robosuite/run-baseline.md`.
