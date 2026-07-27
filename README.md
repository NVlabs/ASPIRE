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

ASPIRE material owned by NVIDIA or contributed under the project license is available under the [Apache License 2.0](LICENSE). This repository also contains inherited code, modified third-party code, dependency patches, Git submodules, models, datasets, robot descriptions, and assets that retain their original terms. See [NOTICE](NOTICE), [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSES/](LICENSES/) before using or redistributing the full stack.

The root Apache-2.0 license does not override those component-specific terms.

### Retained third-party licenses

| Component | Repository path | License | Local license file | Upstream source |
| --- | --- | --- | --- | --- |
| Trossen ALOHA D405 model | `aspire/real/robot/models/station/assets/d405.stl` | BSD-3-Clause | [`BSD-3-Clause-ALOHA.txt`](LICENSES/BSD-3-Clause-ALOHA.txt) | [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie/tree/4a7015530bd7a4161103ae8f0905a96481e4cc1a/aloha) |
| CaP-X | `aspire/sim/cap/` and mapped simulation files | MIT | [`MIT-CaP-X.txt`](LICENSES/MIT-CaP-X.txt) | [`capgym/cap-x`](https://github.com/capgym/cap-x/tree/823fcc5dd3e565b45b414f5785668cf32cba13b4) |
| Hydra `_locate` utility | `aspire/sim/cap/envs/configs/instantiate.py` | MIT | [`MIT-Hydra.txt`](LICENSES/MIT-Hydra.txt) | [`facebookresearch/hydra`](https://github.com/facebookresearch/hydra/tree/57690d7c4e8b5e88dad07d67278f613a739e6d13) |
| i2rt YAM model | `aspire/real/robot/models/station/assets/model2*` | MIT | [`MIT-i2rt-YAM.txt`](LICENSES/MIT-i2rt-YAM.txt) | [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie/tree/4a7015530bd7a4161103ae8f0905a96481e4cc1a/i2rt_yam) |
| i2rt dependency patch | `aspire/real/patches/dependencies/i2rt/` | MIT upstream; NVIDIA modifications under Apache-2.0 | [`MIT-i2rt.txt`](LICENSES/MIT-i2rt.txt) | [`i2rt-robotics/i2rt`](https://github.com/i2rt-robotics/i2rt/tree/98d177bb511d545c80c0e8ec13ffaf227238a8d6) |
| PyRoKi snippets, assets, and patch | `aspire/**/pyroki_snippets/`, Panda spheres, and dependency patch | MIT | [`MIT-PyRoKi.txt`](LICENSES/MIT-PyRoKi.txt) | [`chungmin99/pyroki`](https://github.com/chungmin99/pyroki/tree/95afccc22658c461ab1042a048ae4e9c24bc2a47) |
| RoboCasa dependency patch | `aspire/real/patches/dependencies/robocasa/` | MIT, with DeepMind MuJoCo Apache-2.0 attribution | [`MIT-RoboCasa.txt`](LICENSES/MIT-RoboCasa.txt) | [`robocasa/robocasa`](https://github.com/robocasa/robocasa/tree/9a3a78680443734786c9784ab661413edb87067b) |
| Robosuite-derived YAM XML | `aspire/real/robot/models/station/*.xml` | MIT, with DeepMind MuJoCo Apache-2.0 attribution | [`MIT-Robosuite.txt`](LICENSES/MIT-Robosuite.txt) | [`uynitsuj/robosuite`](https://github.com/uynitsuj/robosuite/tree/97292732ed909ac3ae116579fb768607034a4dbd) |
| cuRobo v0.7.8 patch and dependency | `aspire/real/patches/dependencies/curobo/` and submodule | Custom NVIDIA license | [`NVIDIA-cuRobo-v0.7.8.txt`](LICENSES/NVIDIA-cuRobo-v0.7.8.txt) | [`NVlabs/curobo`](https://github.com/NVlabs/curobo/tree/d64c4b005459db10c5dd867d8b30a87d5bda9bdb) |

### Git submodules

The parent ASPIRE source artifact stores gitlinks only and does not include populated submodule contents. Recursive checkout fetches those repositories separately under their own terms.

| Submodule | Local path | Source URL | Exact gitlink SHA | License | Distribution status |
| --- | --- | --- | --- | --- | --- |
| LIBERO-PRO | `aspire/sim/cap/third_party/LIBERO-PRO` | [`uynitsuj/LIBERO-PRO`](https://github.com/uynitsuj/LIBERO-PRO) | `47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a` | Code MIT; datasets described upstream as CC BY 4.0 | Gitlink only; recursive asset coverage pending OSRB review |
| Robosuite (YAM fork) | `aspire/sim/cap/third_party/robosuite` | [`uynitsuj/robosuite`](https://github.com/uynitsuj/robosuite) | `97292732ed909ac3ae116579fb768607034a4dbd` | MIT with MuJoCo attribution | Gitlink only; bundled robot/CAD assets pending OSRB review |
| Robosuite (LIBERO dependency) | `aspire/sim/cap/third_party/libero_dependencies/robosuite` | [`Max-Fu/robosuite`](https://github.com/Max-Fu/robosuite) | `a498b087d4bc5a3981e3d27030d09bc537a537f3` | MIT with MuJoCo attribution | Gitlink only; bundled robot/CAD assets pending OSRB review |
| SAM 3 | `aspire/sim/cap/third_party/sam3` | [`Max-Fu/sam3`](https://github.com/Max-Fu/sam3) | `6fe87d64a5beb9084923d7a9e002741178635b09` | Custom SAM License | Gitlink only; custom-license approval required |
| cuRobo | `aspire/sim/cap/third_party/curobo` | [`NVlabs/curobo`](https://github.com/NVlabs/curobo) | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | Custom NVIDIA license plus asset terms | Gitlink only; custom-license and asset approval required |
| BEHAVIOR-1K bundle (`b1k`) | `aspire/sim/cap/third_party/b1k` | [`qingh097/b1k`](https://github.com/qingh097/b1k) | `272ec5ca9936453c4a8fd335c4dfba61245e33ca` | Mixed terms, including a no-redistribution Pixar asset | Gitlink only; recursive redistribution blocked pending OSRB/Legal confirmation |

See the populated-tree [`SUBMODULE_AUDIT.md`](SUBMODULE_AUDIT.md) and machine-readable [`SUBMODULE_AUDIT.tsv`](SUBMODULE_AUDIT.tsv) for license hashes, asset findings, Git LFS results, and open approval gates.

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
