![ASPIRE robot demonstrations](assets/media/mainvideoclip-2.gif)

# ASPIRE: Agentic /Skills Discovery for Robotics
[Project Page](https://research.nvidia.com/labs/gear/aspire/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2607.00272)

**Runyu Lu<sup>1,2,&#42;,&dagger;</sup>, Yubo Wu<sup>1,3,&#42;</sup>, Ethan Kou<sup>1,4,&#42;</sup>,
Letian Fu<sup>1,4</sup>, Wenli Xiao<sup>1,5</sup>, Ajay Mandlekar<sup>1</sup>, Yinzhen Xu<sup>1</sup>,
Guanya Shi<sup>5</sup>, Ken Goldberg<sup>4</sup>, Ang Chen<sup>2</sup>, Mosharaf Chowdhury<sup>2</sup>,
Yuke Zhu<sup>1,&dagger;</sup>, Linxi "Jim" Fan<sup>1,&dagger;</sup>, Guanzhi Wang<sup>1,&dagger;</sup>**

<sup>1</sup>NVIDIA &ensp; <sup>2</sup>University of Michigan &ensp; <sup>3</sup>University of Illinois Urbana-Champaign &ensp; <sup>4</sup>UC Berkeley &ensp; <sup>5</sup>Carnegie Mellon University

<sup>&#42;</sup>Equal contribution &ensp; <sup>&dagger;</sup>Project leads

---

## Repository Layout

ASPIRE is a continual-learning robotics system that autonomously writes, debugs, and distills robot control programs into reusable skills across simulation and real-world settings.

| Area | Purpose |
| ---- | ------- |
| [`aspire/sim/`](aspire/sim/README.md) | Simulation workspace for LIBERO-PRO, Robosuite, and BEHAVIOR-1K setup, configs, scripts, tests, agent runbooks, and outputs. |
| [`aspire/sim/cap/`](aspire/sim/cap/) | Code-as-policy simulation package imported as `aspire.sim.cap.*`. |
| [`aspire/real/`](aspire/real/README.md) | Real-station deployment code, operator workflows, and YAM robot integrations. |

## Setup

Simulation setup lives in [`aspire/sim/README.md`](aspire/sim/README.md). Start there for venv creation, submodule initialization, suite-specific setup, smoke tests, and experiment runbooks:

```bash
cd aspire/sim
```

Real-robot deployment and operator instructions live in [`aspire/real/README.md`](aspire/real/README.md), with fresh-machine requirements and the workstation recovery checklist in [`aspire/real/SETUP.md`](aspire/real/SETUP.md). Treat that directory as the working root for YAM commands:

```bash
cd aspire/real
bash tools/yam_demo_preflight.sh
bash tmux/launch_yam_demo_services.sh --no-attach
```

The focused launcher starts only the arm, camera, SAM3, and BundleSDF services
used by the canonical saved demo. See the real-workspace README before enabling
physical motion.

## Agent Guides

- Simulation agents: [`aspire/sim/CLAUDE.md`](aspire/sim/CLAUDE.md) and [`aspire/sim/.claude/README.md`](aspire/sim/.claude/README.md)
- Real-robot agent contract: [`aspire/real/AGENTS.md`](aspire/real/AGENTS.md)
- Real-robot skills: [`aspire/real/.agents/skills/`](aspire/real/.agents/skills/)

The simulation and real-robot workspaces intentionally keep separate commands,
dependencies, runtime artifacts, and agent instructions. The
`yam-simulation-transfer` skill links strategy knowledge across them without
reusing simulator coordinates or APIs on the physical robot.

## Contribution Guidelines

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must
include a Developer Certificate of Origin sign-off.

## License

ASPIRE material owned by NVIDIA or contributed under the project license is
available under the [Apache License 2.0](LICENSE). This repository also
contains inherited code, modified third-party code, dependency patches, Git
submodules, models, datasets, robot descriptions, and assets that retain their
original terms. See [NOTICE](NOTICE),
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSES/](LICENSES/)
before using or redistributing the full stack.

The root Apache-2.0 license does not override those component-specific terms.

## Citation

If you find ASPIRE useful in your research, please cite:

```bibtex
@article{lu2026aspire,
  title   = {ASPIRE: Agentic /Skills Discovery for Robotics},
  author  = {Runyu Lu and Yubo Wu and Ethan Kou and Max Fu and Wenli Xiao and
             Ajay Mandlekar and Yinzhen Xu and Guanya Shi and Ken Goldberg and
             Ang Chen and Mosharaf Chowdhury and Yuke Zhu and Linxi Fan and Guanzhi Wang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2607.00272},
  eprint  = {2607.00272},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2607.00272}
}
```
