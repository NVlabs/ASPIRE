# ASPIRE: Agentic /Skills Discovery for Robotics

[Project Page](https://research.nvidia.com/labs/gear/aspire/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2607.00272)

<img src="assets/media/mainvideoclip-2.gif" alt="ASPIRE robot demonstrations" width="100%">

ASPIRE is a continual-learning robotics system that autonomously writes, debugs, and distills robot-control programs into reusable skills across simulation and real-world settings.

## Quick Start

### Run with a coding agent

ASPIRE includes repository instructions for coding agents such as Codex and Claude Code. Clone the repository, open the agent at the repository root, and give it this request:

**Reference agent environments:** ASPIRE is coding-agent agnostic. Our simulation workflow is packaged for reproduction with Claude Code, while the real-robot agent experiments were conducted with Codex. Other coding agents can follow the model-neutral instructions in [`AGENTS.md`](AGENTS.md), although orchestration behavior may differ.

```text
Read AGENTS.md and run the ASPIRE LIBERO-Pro Goal-Swap Quick Start for
put_the_wine_bottle_on_top_of_the_cabinet.

Before executing, report the required GPUs, credentials, gated weights,
services, expected runtime, seed partitions, and output paths. Wait for
my confirmation before launching. Do not access real-robot code or push
repository changes.
```

This Quick Start runs one complete LIBERO-Pro Fix Loop task rather than a short demo:

- **Suite and task:** `libero_goal_swap/put_the_wine_bottle_on_top_of_the_cabinet`
- **Development:** seeds 51–65 for initial code generation and repair
- **Held-out evaluation:** seeds 1–50 using the selected fix
- **Reference GPU topology:** SAM3, GraspNet, and PyRoKi on GPUs 0–2; task execution on GPU 3
- **Runtime:** potentially several hours; failed trials can run for approximately 6–7 minutes each
- **Completion:** a selected `fix_code.py`, reusable findings, traces and videos, and an immutable 50-seed validation manifest and pass rate—not a guaranteed success threshold

The canonical procedure is [`aspire/sim/.claude/libero/fix-loop/QUICKSTART.md`](aspire/sim/.claude/libero/fix-loop/QUICKSTART.md). The agent must complete preflight and wait for confirmation before installing dependencies, starting services, or launching trials.

### Choose another paper experiment

For any experiment other than the canonical Quick Start, name the suite and experiment explicitly. If neither is named, the agent should present this table and stop for selection.

| Suite | Experiment | Runbook |
| ----- | ---------- | ------- |
| LIBERO-Pro | Fix Loop | [`libero/fix-loop/`](aspire/sim/.claude/libero/fix-loop/INSTRUCTIONS.md) |
| LIBERO-Pro | Fix Loop + Evolutionary Search | [`libero/evosearch/`](aspire/sim/.claude/libero/evosearch/INSTRUCTIONS.md) |
| LIBERO | Zero-Shot Transfer | [`libero/zeroshot-transfer/`](aspire/sim/.claude/libero/zeroshot-transfer/INSTRUCTIONS.md) |
| LIBERO-Long-Pro | Library-Size Scaling | [`libero/library-size-scaling/`](aspire/sim/.claude/libero/library-size-scaling/INSTRUCTIONS.md) |
| LIBERO-Long-Pro | Inference-Time Scaling | [`libero/inference-time-scaling/`](aspire/sim/.claude/libero/inference-time-scaling/INSTRUCTIONS.md) |
| Robosuite | Fix Loop | [`robosuite/fix-loop/`](aspire/sim/.claude/robosuite/fix-loop/INSTRUCTIONS.md) |
| Robosuite | Training Law | [`robosuite/training-law/`](aspire/sim/.claude/robosuite/training-law/INSTRUCTIONS.md) |
| BEHAVIOR-1K | Fix Loop | [`behavior/fix-loop/`](aspire/sim/.claude/behavior/fix-loop/INSTRUCTIONS.md) |

Before a paper-scale launch, the agent must report the selected tasks, seed schedule, expected trial count and runtime, GPU and credential requirements, services, and output paths, then wait for explicit confirmation.

### Manual setup

- Simulation installation, suite-specific environments, smoke tests, and troubleshooting: [`aspire/sim/README.md`](aspire/sim/README.md)
- Simulation experiment registry and runbooks: [`aspire/sim/.claude/README.md`](aspire/sim/.claude/README.md)
- Real-robot setup and operator-controlled workflows: [`aspire/real/README.md`](aspire/real/README.md)

> [!WARNING]
> ASPIRE executes language-model-generated Python with full import access. Trial processes and watchdogs are not a security sandbox. Run generated code on an isolated host without credentials or sensitive mounts, restrict network access, and never grant a simulation agent access to physical hardware. Real-robot work requires the controls in [`aspire/real/AGENTS.md`](aspire/real/AGENTS.md) and separate operator authorization.

## Repository Layout

| Area | Purpose |
| ---- | ------- |
| [`aspire/sim/`](aspire/sim/README.md) | Simulation workspace for LIBERO-PRO, Robosuite, and BEHAVIOR-1K setup, configs, scripts, tests, agent runbooks, and outputs. |
| [`aspire/sim/cap/`](aspire/sim/cap/) | Code-as-policy simulation package imported as `aspire.sim.cap.*`. |
| [`aspire/real/`](aspire/real/README.md) | Real-station deployment code, operator workflows, and YAM robot integrations. |

The simulation and real-robot workspaces intentionally keep separate commands, dependencies, runtime artifacts, and safety contracts. The `yam-simulation-transfer` skill links strategy knowledge across them without reusing simulator coordinates or APIs on the physical robot.

## Contribution Guidelines

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must include a Developer Certificate of Origin sign-off.

## License

ASPIRE material owned by NVIDIA or contributed under the project license is
available under the [Apache License 2.0](LICENSE). Third-party materials retain
their original terms. See [NOTICE](NOTICE) and the central
[licensing and compliance index](LICENSES/README.md) before using or
redistributing the full stack.

## Citation

If you find ASPIRE useful in your research, please cite:

```bibtex
@article{lu2026aspire,
  title   = {ASPIRE: Agentic /Skills Discovery for Robotics},
  author  = {Runyu Lu and Yubo Wu and Ethan Kou and Letian Fu and Wenli Xiao and
             Ajay Mandlekar and Yinzhen Xu and Guanya Shi and Ken Goldberg and
             Ang Chen and Mosharaf Chowdhury and Yuke Zhu and Linxi Fan and Guanzhi Wang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2607.00272},
  url     = {https://arxiv.org/abs/2607.00272}
}
```

ASPIRE was developed by researchers from NVIDIA, the University of Michigan, the University of Illinois Urbana-Champaign, UC Berkeley, and Carnegie Mellon University.
